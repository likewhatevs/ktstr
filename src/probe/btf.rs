//! BTF parsing and BPF program symbol resolution for probe attachment and output decoding.

use super::stack::StackFunction;

/// ELF sections the probe BTF loader reads from a vmlinux ELF
/// ([`parse_btf_functions`] and [`resolve_field_specs`]).
///
/// The cached-vmlinux strip pipeline
/// ([`crate::cache::strip_vmlinux_debug`]) preserves these bytes
/// verbatim via its keep-list predicate.
pub(crate) const VMLINUX_KEEP_SECTIONS: &[&[u8]] = &[
    b".BTF", // BPF Type Format — probe field resolution
];

/// Display hint derived from BTF type information.
///
/// Controls how auto-discovered field values are formatted in probe
/// output. Dedicated decoders in `decode.rs` fire by field-key name
/// (`dsq_id`, `enq_flags`, etc.) regardless of how the field was
/// discovered; this hint only applies to auto-discovered fields
/// that do not match a dedicated decoder and would otherwise
/// default to hex.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum RenderHint {
    /// Unsigned decimal (u32 sizes, counters).
    Decimal,
    /// Hexadecimal (u64 values, pointers, bitmasks).
    #[default]
    Hex,
    /// Boolean (1-byte int with is_bool set in BTF).
    Bool,
    /// Signed decimal (signed integers).
    Signed,
}

/// BTF-resolved parameter metadata for a probed function.
///
/// Each param maps to one register (fentry/kprobe). Struct pointer
/// params have their fields expanded via [`STRUCT_FIELDS`] or
/// auto-discovered from vmlinux or BPF program BTF.
#[derive(Debug, Clone, Default)]
pub struct BtfParam {
    /// Parameter name from BTF (argument name in the function signature).
    pub name: String,
    /// Known struct name (in STRUCT_FIELDS) for field key generation
    pub struct_name: Option<String>,
    /// True if the parameter is a pointer type.
    pub is_ptr: bool,
    /// True if this is a char * / const char * (string pointer).
    pub is_string_ptr: bool,
    /// Auto-discovered fields from vmlinux or BPF program BTF for
    /// struct pointer types not in STRUCT_FIELDS.
    /// Vec of (field_name, access_pattern, render_hint).
    pub auto_fields: Vec<(String, String, RenderHint)>,
    /// Type name for auto-discovered structs (used in output headers).
    pub type_name: Option<String>,
}

/// BTF-resolved function signature.
///
/// Produced by [`parse_btf_functions`] (kernel functions via vmlinux BTF)
/// or [`parse_bpf_btf_functions`] (BPF callbacks via program BTF).
/// Used by [`run_probe_skeleton`](super::process::run_probe_skeleton) to
/// populate field specs and by [`build_field_keys`](super::process) to
/// generate output labels.
#[derive(Debug, Clone, Default)]
pub struct BtfFunc {
    /// Fully qualified function name as it appears in BTF.
    pub name: String,
    /// Ordered parameter metadata, one entry per argument.
    pub params: Vec<BtfParam>,
    /// True if BTF FuncProto has a variadic sentinel parameter
    /// (name_off=0, type=0). Variadic functions should not have
    /// their displayed arg count capped.
    pub is_variadic: bool,
}

/// Known struct types and their fields for probe output decoding.
/// Maps struct name -> list of (field_access, output_key) pairs.
///
/// Field accesses use a mini-DSL: `->field` follows a pointer,
/// `.field` reads inline, `[N]` indexes an array.
pub const STRUCT_FIELDS: &[(&str, &[(&str, &str)])] = &[
    (
        "task_struct",
        &[
            ("->pid", "pid"),
            ("->cpus_ptr->bits[0]", "cpumask_0"),
            ("->cpus_ptr->bits[1]", "cpumask_1"),
            ("->cpus_ptr->bits[2]", "cpumask_2"),
            ("->cpus_ptr->bits[3]", "cpumask_3"),
            ("->scx.ddsp_dsq_id", "dsq_id"),
            ("->scx.ddsp_enq_flags", "enq_flags"),
            ("->scx.slice", "slice"),
            ("->scx.dsq_vtime", "vtime"),
            ("->scx.weight", "weight"),
            ("->scx.sticky_cpu", "sticky_cpu"),
            ("->scx.flags", "scx_flags"),
        ],
    ),
    ("rq", &[("->cpu", "cpu")]),
    ("scx_dispatch_q", &[("->id", "dsq_id")]),
    ("scx_init_task_args", &[("->fork", "fork")]),
    (
        "scx_exit_info",
        &[("->kind", "exit_kind"), ("->reason", "reason")],
    ),
    ("scx_cgroup_init_args", &[("->weight", "weight")]),
];

/// Runtime-resolved field dereference spec for the BPF skeleton.
/// Maps 1:1 to the C `struct field_spec` in intf.h.
#[derive(Debug, Clone)]
pub struct FieldSpec {
    /// Parameter index whose pointer value supplies the base address.
    pub param_idx: u32,
    /// Byte offset from the base pointer at which to read.
    pub offset: u32,
    /// Number of bytes to read (1/2/4/8 for scalars; larger for
    /// `Bytes` values).
    pub size: u32,
    /// Slot in the shared probe output map that receives this field.
    pub field_idx: u32,
    /// Byte offset to intermediate pointer for chained dereferences.
    /// 0 = single-level read. Nonzero = read ptr at base+ptr_offset,
    /// then read size bytes at ptr+offset.
    pub ptr_offset: u32,
}

/// One-shot wrapper that loads vmlinux BTF then delegates to
/// [`resolve_field_specs_with_btf`].
///
/// Reserved for callers that do NOT have a pre-parsed BTF handle in
/// scope; hot-path callers that resolve fields for many functions
/// must call [`resolve_field_specs_with_btf`] directly so the
/// multi-MB vmlinux BTF parse is paid once per call site rather
/// than once per function.
///
/// `vmlinux_path` accepts either a raw BTF blob (e.g.
/// `/sys/kernel/btf/vmlinux`) or an ELF vmlinux — both formats are
/// recognized by [`crate::monitor::btf_offsets::load_btf_from_path`].
/// `None` falls back to the host's `/sys/kernel/btf/vmlinux`, which
/// is the host kernel's BTF and does not match the guest kernel
/// under test; the wrapper does not warn in that case (the new
/// hot-path API forces the caller to be explicit about BTF source).
#[allow(dead_code)]
pub fn resolve_field_specs(btf_func: &BtfFunc, vmlinux_path: Option<&str>) -> Vec<FieldSpec> {
    let path = vmlinux_path.unwrap_or("/sys/kernel/btf/vmlinux");
    match crate::monitor::btf_offsets::load_btf_from_path(std::path::Path::new(path)) {
        Ok(btf) => resolve_field_specs_with_btf(btf_func, &btf),
        Err(_) => Vec::new(),
    }
}

pub fn resolve_field_specs_with_btf(btf_func: &BtfFunc, btf: &btf_rs::Btf) -> Vec<FieldSpec> {
    let mut specs = Vec::new();
    let mut field_idx: u32 = 0;

    let max_params = btf_func.params.len().min(6);
    for (param_idx, param) in btf_func.params[..max_params].iter().enumerate() {
        if let Some(ref struct_name) = param.struct_name {
            // Known struct in STRUCT_FIELDS — curated field list.
            let fields = match STRUCT_FIELDS
                .iter()
                .find(|(s, _)| *s == struct_name.as_str())
            {
                Some((_, f)) => *f,
                None => continue,
            };

            let struct_type = match resolve_struct_type(btf, struct_name) {
                Some(s) => s,
                None => {
                    // Skip field slots to stay aligned with build_field_keys.
                    field_idx += fields.len() as u32;
                    continue;
                }
            };

            for (access, _key) in fields {
                let access = access.trim_start_matches("->");

                if access.contains("->") {
                    let (ptr_member, target) = access.split_once("->").unwrap();
                    let ptr_off_result = resolve_member_offset(btf, &struct_type, ptr_member);
                    if let Some((ptr_off, _)) = ptr_off_result {
                        let pointed = resolve_pointed_struct(btf, &struct_type, ptr_member);
                        if let Some(pointed_struct) = pointed
                            && let Some((target_off, target_sz)) =
                                resolve_member_offset(btf, &pointed_struct, target)
                        {
                            specs.push(FieldSpec {
                                param_idx: param_idx as u32,
                                offset: target_off,
                                size: target_sz,
                                field_idx,
                                ptr_offset: ptr_off,
                            });
                        }
                    } else {
                        tracing::debug!(
                            member = ptr_member,
                            "chained deref: member offset not found",
                        );
                    }
                } else if let Some((offset, size)) =
                    resolve_member_offset(btf, &struct_type, access)
                {
                    specs.push(FieldSpec {
                        param_idx: param_idx as u32,
                        offset,
                        size,
                        field_idx,
                        ptr_offset: 0,
                    });
                }

                field_idx += 1;
                if field_idx >= 16 {
                    break;
                }
            }
        } else if !param.auto_fields.is_empty() {
            // Auto-discovered vmlinux struct fields.
            let sname = match param.type_name.as_deref() {
                Some(n) => n,
                None => {
                    field_idx += param.auto_fields.len() as u32;
                    continue;
                }
            };
            let struct_type = match resolve_struct_type(btf, sname) {
                Some(s) => s,
                None => {
                    field_idx += param.auto_fields.len() as u32;
                    continue;
                }
            };

            let remaining = (16 - field_idx) as usize;
            if param.auto_fields.len() > remaining {
                tracing::warn!(
                    func = %btf_func.name,
                    struct_name = sname,
                    total = param.auto_fields.len(),
                    budget = remaining,
                    "auto-discovered fields truncated to MAX_FIELDS budget",
                );
            }

            for (_fname, access, _hint) in &param.auto_fields {
                if field_idx >= 16 {
                    break;
                }
                let access = access.trim_start_matches("->");

                if access.contains("->") {
                    if let Some((ptr_member, target)) = access.split_once("->") {
                        let ptr_off_result = resolve_member_offset(btf, &struct_type, ptr_member);
                        if let Some((ptr_off, _)) = ptr_off_result {
                            let pointed = resolve_pointed_struct(btf, &struct_type, ptr_member);
                            if let Some(pointed_struct) = pointed
                                && let Some((target_off, target_sz)) =
                                    resolve_member_offset(btf, &pointed_struct, target)
                            {
                                specs.push(FieldSpec {
                                    param_idx: param_idx as u32,
                                    offset: target_off,
                                    size: target_sz,
                                    field_idx,
                                    ptr_offset: ptr_off,
                                });
                            }
                        }
                    }
                } else if let Some((offset, size)) =
                    resolve_member_offset(btf, &struct_type, access)
                {
                    specs.push(FieldSpec {
                        param_idx: param_idx as u32,
                        offset,
                        size,
                        field_idx,
                        ptr_offset: 0,
                    });
                }

                field_idx += 1;
            }
        } else if !param.is_ptr {
            // Scalar param takes one field slot (matched by build_field_keys).
            field_idx += 1;
        }
    }

    tracing::debug!(n = specs.len(), func = %btf_func.name, "resolve_field_specs");
    specs
}

/// Find a BTF struct type by name.
fn resolve_struct_type(btf: &btf_rs::Btf, name: &str) -> Option<btf_rs::Struct> {
    let types = btf.resolve_types_by_name(name).ok()?;
    for t in types {
        if let btf_rs::Type::Struct(s) = t {
            return Some(s);
        }
    }
    None
}

/// Resolve byte offset and read size for a possibly nested field access.
/// Access is dot-separated (e.g. "scx.ddsp_dsq_id").
fn resolve_member_offset(
    btf: &btf_rs::Btf,
    struct_type: &btf_rs::Struct,
    access: &str,
) -> Option<(u32, u32)> {
    let parts: Vec<&str> = access.split('.').collect();
    let mut current_struct = struct_type.clone();
    let mut total_offset: u32 = 0;

    for (i, part) in parts.iter().enumerate() {
        // Strip array index (e.g. "bits[0]" -> "bits", extract 0).
        let (member_name, array_idx) = if let Some(bracket) = part.find('[') {
            let name = &part[..bracket];
            let idx_str = &part[bracket + 1..part.len() - 1];
            let idx: u32 = idx_str.parse().unwrap_or(0);
            (name, Some(idx))
        } else {
            (*part, None)
        };

        // Find the member in the current struct.
        let member = current_struct.members.iter().find(|m| {
            btf.resolve_name(*m)
                .map(|n| n == member_name)
                .unwrap_or(false)
        })?;

        let bit_off = member.bit_offset();
        if bit_off % 8 != 0 {
            // Bitfield -- skip.
            return None;
        }
        total_offset += bit_off / 8;

        // Add array element offset if indexed.
        if let Some(idx) = array_idx {
            let elem_size = resolve_type_size(btf, member)?;
            total_offset += idx * elem_size;
        }

        let is_last = i == parts.len() - 1;
        if is_last {
            // Resolve the member's type to determine read size.
            let size = resolve_type_size(btf, member)?;
            return Some((total_offset, size));
        }

        // Not the last part: the member must be an embedded struct/union.
        // Follow the type chain to find the struct.
        let member_type = follow_to_struct_or_union(btf, member)?;
        current_struct = member_type;
    }

    None
}

/// Follow a member's type through qualifiers to find the underlying
/// type size. Returns 8 for pointers, the type's size for int/enum/struct.
fn resolve_type_size(btf: &btf_rs::Btf, member: &btf_rs::Member) -> Option<u32> {
    use btf_rs::{BtfType, Type};

    let tid = member.get_type_id()?;
    let mut t = btf.resolve_type_by_id(tid).ok()?;

    for _ in 0..20 {
        match t {
            Type::Ptr(_) => return Some(8),
            Type::Int(ref i) => return Some(i.size() as u32),
            Type::Enum(ref e) => return Some(e.size() as u32),
            Type::Enum64(_) => return Some(8),
            Type::Struct(ref s) => return Some(s.size() as u32),
            Type::Union(ref u) => return Some(u.size() as u32),
            Type::Array(ref a) => {
                // For array access like bits[0], return element size.
                let elem_tid = a.get_type_id()?;
                let elem = btf.resolve_type_by_id(elem_tid).ok()?;
                match elem {
                    Type::Int(ref i) => return Some(i.size() as u32),
                    Type::Ptr(_) => return Some(8),
                    _ => return Some(8),
                }
            }
            Type::Const(_)
            | Type::Volatile(_)
            | Type::Restrict(_)
            | Type::Typedef(_)
            | Type::TypeTag(_) => {
                t = btf.resolve_chained_type(t.as_btf_type()?).ok()?;
            }
            _ => return None,
        }
    }
    None
}

/// Follow a member's type through qualifiers to find an embedded
/// struct or union type.
fn follow_to_struct_or_union(btf: &btf_rs::Btf, member: &btf_rs::Member) -> Option<btf_rs::Struct> {
    use btf_rs::{BtfType, Type};

    let tid = member.get_type_id()?;
    let mut t = btf.resolve_type_by_id(tid).ok()?;

    for _ in 0..20 {
        match t {
            Type::Struct(s) | Type::Union(s) => return Some(s),
            Type::Const(_)
            | Type::Volatile(_)
            | Type::Restrict(_)
            | Type::Typedef(_)
            | Type::TypeTag(_) => {
                t = btf.resolve_chained_type(t.as_btf_type()?).ok()?;
            }
            _ => return None,
        }
    }
    None
}

/// Resolve a pointer member's pointed-to struct type.
/// Given a struct and a member name that is a pointer, follow the
/// pointer type to find the struct/union it points to.
fn resolve_pointed_struct(
    btf: &btf_rs::Btf,
    struct_type: &btf_rs::Struct,
    member_name: &str,
) -> Option<btf_rs::Struct> {
    use btf_rs::BtfType;

    // Strip array index (e.g. "bits[0]" -> "bits").
    let member_name = member_name.split('[').next().unwrap_or(member_name);

    let member = struct_type.members.iter().find(|m| {
        btf.resolve_name(*m)
            .map(|n| n == member_name)
            .unwrap_or(false)
    })?;

    let tid = member.get_type_id()?;
    crate::monitor::bpf_map::resolve_to_struct(btf, tid)
}

/// Parse BTF from vmlinux for kernel function signatures.
///
/// `vmlinux_path` accepts either a raw BTF blob (e.g.
/// `/sys/kernel/btf/vmlinux`) or an ELF vmlinux — both formats are
/// recognized by [`crate::monitor::btf_offsets::load_btf_from_path`].
/// `None` falls back to the host's `/sys/kernel/btf/vmlinux`, which
/// is the host kernel's BTF and does not match the guest kernel
/// under test; a warning is emitted in that case.
///
/// Resolves parameter types via btf-rs, following PTR/CONST/VOLATILE/TYPEDEF
/// chains to identify struct pointers. Records `struct_name` for types
/// listed in [`STRUCT_FIELDS`]; other struct pointers get auto-discovered
/// fields via [`discover_vmlinux_struct_fields`] and `type_name` set.
/// Detects `char *` parameters (`is_string_ptr`) by chasing the type chain
/// to an `Int` of size 1.
pub fn parse_btf_functions(func_names: &[&str], vmlinux_path: Option<&str>) -> Vec<BtfFunc> {
    use btf_rs::{BtfType, Type};

    let btf_path = match vmlinux_path {
        Some(p) => p,
        None => {
            tracing::warn!(
                "parse_btf_functions: no vmlinux_path; falling back to host \
                 /sys/kernel/btf/vmlinux — host/guest kernel mismatch will \
                 produce wrong function signatures"
            );
            "/sys/kernel/btf/vmlinux"
        }
    };
    let btf = match crate::monitor::btf_offsets::load_btf_from_path(std::path::Path::new(btf_path))
    {
        Ok(b) => b,
        Err(e) => {
            tracing::error!(
                %e,
                path = btf_path,
                requested = func_names.len(),
                "parse_btf_functions: BTF parse failed; returning empty Vec \
                 (caller will receive no function signatures — distinct from \
                 an empty result due to no func names matching)"
            );
            return Vec::new();
        }
    };

    // Resolve a parameter's type_id to its underlying struct name,
    // following PTR/CONST/VOLATILE/TYPEDEF chains.
    let resolve_struct_name = |type_id: u32| -> Option<String> {
        let mut t = match btf.resolve_type_by_id(type_id) {
            Ok(t) => t,
            Err(_) => return None,
        };
        for _ in 0..20 {
            match &t {
                Type::Ptr(_)
                | Type::Const(_)
                | Type::Volatile(_)
                | Type::Typedef(_)
                | Type::Restrict(_)
                | Type::TypeTag(_) => {
                    t = match btf.resolve_chained_type(t.as_btf_type().unwrap()) {
                        Ok(next) => next,
                        Err(_) => return None,
                    };
                }
                Type::Struct(s) => {
                    return btf.resolve_name(s).ok();
                }
                Type::Union(u) => {
                    return btf.resolve_name(u).ok();
                }
                _ => return None,
            }
        }
        None
    };

    let is_ptr =
        |type_id: u32| -> bool { matches!(btf.resolve_type_by_id(type_id), Ok(Type::Ptr(_))) };

    // Detect char * / const char * — Ptr → (Const/Volatile →) Int(size=1).
    let is_str_ptr = |type_id: u32| -> bool {
        let mut t = match btf.resolve_type_by_id(type_id) {
            Ok(t) => t,
            Err(_) => return false,
        };
        for _ in 0..20 {
            match &t {
                Type::Ptr(_)
                | Type::Const(_)
                | Type::Volatile(_)
                | Type::Restrict(_)
                | Type::Typedef(_)
                | Type::TypeTag(_) => {
                    t = match btf.resolve_chained_type(t.as_btf_type().unwrap()) {
                        Ok(next) => next,
                        Err(_) => return false,
                    };
                }
                Type::Int(i) => return i.size() == 1,
                _ => return false,
            }
        }
        false
    };

    let mut results = Vec::new();

    for func_name in func_names {
        let types = match btf.resolve_types_by_name(func_name) {
            Ok(t) => t,
            Err(_) => continue,
        };

        for t in &types {
            if let Type::Func(func) = t {
                // Resolve the FuncProto
                let proto = match btf.resolve_chained_type(func) {
                    Ok(Type::FuncProto(fp)) => fp,
                    _ => continue,
                };

                let variadic = proto
                    .parameters
                    .last()
                    .map(|p| p.is_variadic())
                    .unwrap_or(false);

                let mut params = Vec::new();
                for param in &proto.parameters {
                    // Skip the variadic sentinel (name_off=0, type=0).
                    if param.is_variadic() {
                        continue;
                    }
                    let name = btf.resolve_name(param).unwrap_or_default();
                    let tid = param.get_type_id().unwrap_or(0);
                    let all_struct_name = resolve_struct_name(tid);
                    let known_struct = all_struct_name
                        .as_ref()
                        .filter(|n| STRUCT_FIELDS.iter().any(|(s, _)| *s == n.as_str()))
                        .cloned();
                    let param_is_ptr = is_ptr(tid);

                    // Auto-discover fields for struct pointers not in STRUCT_FIELDS.
                    let (auto_fields, type_name) = if param_is_ptr && known_struct.is_none() {
                        if let Some(ref sname) = all_struct_name {
                            let fields = discover_vmlinux_struct_fields(&btf, tid);
                            (fields, Some(sname.clone()))
                        } else {
                            (Vec::new(), None)
                        }
                    } else {
                        (
                            Vec::new(),
                            all_struct_name.filter(|_| known_struct.is_none()),
                        )
                    };

                    params.push(BtfParam {
                        name,
                        struct_name: known_struct,
                        is_ptr: param_is_ptr,
                        is_string_ptr: is_str_ptr(tid),
                        auto_fields,
                        type_name,
                    });
                }

                results.push(BtfFunc {
                    name: func_name.to_string(),
                    params,
                    is_variadic: variadic,
                });
                break; // take first match
            }
        }
    }

    if !func_names.is_empty() && results.is_empty() {
        tracing::warn!(
            requested = func_names.len(),
            path = btf_path,
            "parse_btf_functions: none of the requested function names \
             resolved in BTF — distinct from a parse failure (which logs at \
             error level)"
        );
    }
    tracing::debug!(n = results.len(), "btf: parsed function signatures");
    results
}

/// Discover loaded sched_ext BPF programs via libbpf-rs `ProgInfoIter`.
///
/// Classifies each discovered program into one of two categories:
/// 1. All `StructOps` programs (scheduler callbacks).
/// 2. Any other loaded BPF program whose name matches a display name
///    in `stack_names` (e.g. `SEC("syscall")` programs like
///    `apply_cell_config` that appear in crash backtraces).
///
/// For non-StructOps matching, `bpf_prog_info.name` is truncated to
/// 15 characters (`BPF_OBJ_NAME_LEN - 1`). When a stack name is
/// longer than 15 chars, the truncated `info.name` is used as a
/// candidate prefix match, then the full name is confirmed via the
/// program's BTF.
///
/// Returns a [`StackFunction`] per discovered program with `is_bpf = true`
/// and the program's ID in `bpf_prog_id`. The `raw_name` is
/// `bpf_prog_{id}_{name}`, which reuses the kallsyms pattern shape
/// but substitutes the integer `prog_id` for the hex hash that real
/// kallsyms entries carry; treat it as an internal bookkeeping key,
/// not a string that will literally appear in `/proc/kallsyms`.
pub fn discover_bpf_symbols(stack_names: &[&str]) -> Vec<StackFunction> {
    use libbpf_rs::query::ProgInfoIter;

    let mut seen = std::collections::HashSet::new();
    let mut results = Vec::new();

    for info in ProgInfoIter::default() {
        let info_name = match info.name.to_str() {
            Ok(n) if !n.is_empty() => n.to_string(),
            _ => continue,
        };
        if info.ty == libbpf_rs::ProgramType::StructOps {
            // bpf_prog_info.name is truncated to 15 chars. Resolve
            // the full name from program BTF when truncated so that
            // set_attach_target can find the function.
            let full_name = if info_name.len() >= 15 {
                resolve_bpf_prog_full_name(info.id).unwrap_or(info_name.clone())
            } else {
                info_name.clone()
            };
            if !seen.insert(full_name.clone()) {
                continue;
            }
            results.push(StackFunction {
                raw_name: format!("bpf_prog_{}_{full_name}", info.id),
                display_name: full_name,
                is_bpf: true,
                bpf_prog_id: Some(info.id),
            });
        } else if !stack_names.is_empty() {
            // bpf_prog_info.name is truncated to 15 chars. For short
            // names, exact match works. For long names, check if any
            // stack name starts with the truncated info.name, then
            // confirm the full name via BTF.
            let matched_name = if stack_names.contains(&info_name.as_str()) {
                Some(info_name.clone())
            } else {
                // Candidate: a stack name whose prefix matches the
                // truncated info.name (only relevant when info.name
                // is at the 15-char limit).
                let candidate = stack_names
                    .iter()
                    .find(|sn| sn.len() > info_name.len() && sn.starts_with(&info_name));
                if let Some(target) = candidate {
                    resolve_bpf_prog_full_name(info.id).filter(|full| full == *target)
                } else {
                    None
                }
            };
            if let Some(func_name) = matched_name {
                if !seen.insert(func_name.clone()) {
                    continue;
                }
                tracing::debug!(
                    name = %func_name, id = info.id, ty = ?info.ty,
                    "discover_bpf_symbols: matched non-struct_ops program from stack",
                );
                results.push(StackFunction {
                    raw_name: format!("bpf_prog_{}_{func_name}", info.id),
                    display_name: func_name,
                    is_bpf: true,
                    bpf_prog_id: Some(info.id),
                });
            }
        }
    }

    tracing::debug!(n = results.len(), "discover_bpf_symbols");
    results
}

/// Resolve the full function name for a BPF program from its BTF.
///
/// `bpf_prog_info.name` is truncated to 15 characters. The full name
/// is resolved from `func_info[0].type_id` in the program's BTF.
/// The kernel populates the `func_info` array in `insn_off` order,
/// so the first entry corresponds to the program's entry point
/// (`insn_off == 0`); this is the convention the BPF loader relies
/// on, not a documented guarantee.
fn resolve_bpf_prog_full_name(prog_id: u32) -> Option<String> {
    use libbpf_rs::AsRawLibbpf;
    use libbpf_rs::libbpf_sys;

    let prog_btf = libbpf_rs::btf::Btf::from_prog_id(prog_id).ok()?;
    let btf_ptr = prog_btf.as_libbpf_object().as_ptr();

    let fd = unsafe { libbpf_sys::bpf_prog_get_fd_by_id(prog_id) };
    if fd < 0 {
        return None;
    }

    let mut info = libbpf_sys::bpf_prog_info::default();
    let mut info_len = std::mem::size_of::<libbpf_sys::bpf_prog_info>() as u32;
    let ret = unsafe {
        libbpf_sys::bpf_obj_get_info_by_fd(fd, &mut info as *mut _ as *mut _, &mut info_len)
    };
    if ret != 0 || info.nr_func_info == 0 {
        unsafe { libc::close(fd) };
        return None;
    }

    let fi_rec = info.func_info_rec_size as usize;
    let mut fi_buf = vec![0u8; info.nr_func_info as usize * fi_rec];

    let mut info2 = libbpf_sys::bpf_prog_info {
        nr_func_info: info.nr_func_info,
        func_info_rec_size: info.func_info_rec_size,
        func_info: fi_buf.as_mut_ptr() as u64,
        ..Default::default()
    };
    let mut info2_len = std::mem::size_of::<libbpf_sys::bpf_prog_info>() as u32;
    let ret = unsafe {
        libbpf_sys::bpf_obj_get_info_by_fd(fd, &mut info2 as *mut _ as *mut _, &mut info2_len)
    };
    unsafe { libc::close(fd) };
    if ret != 0 {
        return None;
    }

    // func_info[0] is the entry point (insn_off == 0).
    let fi = unsafe { &*(fi_buf.as_ptr() as *const libbpf_sys::bpf_func_info) };
    let t = unsafe { libbpf_sys::btf__type_by_id(btf_ptr, fi.type_id) };
    if t.is_null() {
        return None;
    }
    let name_ptr = unsafe { libbpf_sys::btf__name_by_offset(btf_ptr, (*t).name_off) };
    if name_ptr.is_null() {
        return None;
    }
    let name = unsafe { std::ffi::CStr::from_ptr(name_ptr) }
        .to_str()
        .ok()?
        .to_string();
    if name.is_empty() { None } else { Some(name) }
}

/// Resolve source locations for BPF functions from program BTF line_info.
///
/// Queries each program's `bpf_prog_info` via `bpf_obj_get_info_by_fd`
/// to get func_info and line_info buffers, then cross-references them
/// to find the first line_info entry at or after each function's insn_off.
/// Returns a map from function name to `"basename:line"`.
pub fn resolve_bpf_source_locs(prog_ids: &[u32]) -> std::collections::HashMap<String, String> {
    use libbpf_rs::AsRawLibbpf;
    use libbpf_rs::libbpf_sys;

    let mut locs = std::collections::HashMap::new();

    for prog_id in prog_ids {
        let prog_btf = match libbpf_rs::btf::Btf::from_prog_id(*prog_id) {
            Ok(b) => b,
            Err(_) => continue,
        };
        let btf_ptr = prog_btf.as_libbpf_object().as_ptr();

        let fd = unsafe { libbpf_sys::bpf_prog_get_fd_by_id(*prog_id) };
        if fd < 0 {
            continue;
        }

        // First query: get func_info/line_info counts.
        let mut info = libbpf_sys::bpf_prog_info::default();
        let mut info_len = std::mem::size_of::<libbpf_sys::bpf_prog_info>() as u32;
        let ret = unsafe {
            libbpf_sys::bpf_obj_get_info_by_fd(fd, &mut info as *mut _ as *mut _, &mut info_len)
        };
        if ret != 0 || info.nr_func_info == 0 || info.nr_line_info == 0 {
            unsafe { libc::close(fd) };
            continue;
        }

        let nr_fi = info.nr_func_info as usize;
        let nr_li = info.nr_line_info as usize;
        let fi_rec = info.func_info_rec_size as usize;
        let li_rec = info.line_info_rec_size as usize;
        let mut fi_buf = vec![0u8; nr_fi * fi_rec];
        let mut li_buf = vec![0u8; nr_li * li_rec];

        // Second query: populate func_info and line_info buffers.
        let mut info2 = libbpf_sys::bpf_prog_info {
            nr_func_info: nr_fi as u32,
            func_info_rec_size: fi_rec as u32,
            func_info: fi_buf.as_mut_ptr() as u64,
            nr_line_info: nr_li as u32,
            line_info_rec_size: li_rec as u32,
            line_info: li_buf.as_mut_ptr() as u64,
            ..Default::default()
        };
        let mut info2_len = std::mem::size_of::<libbpf_sys::bpf_prog_info>() as u32;
        let ret = unsafe {
            libbpf_sys::bpf_obj_get_info_by_fd(fd, &mut info2 as *mut _ as *mut _, &mut info2_len)
        };
        unsafe { libc::close(fd) };
        if ret != 0 {
            continue;
        }

        // Cross-reference func_info with line_info to resolve source
        // locations for each function.
        for i in 0..nr_fi {
            let fi =
                unsafe { &*(fi_buf.as_ptr().add(i * fi_rec) as *const libbpf_sys::bpf_func_info) };
            let t = unsafe { libbpf_sys::btf__type_by_id(btf_ptr, fi.type_id) };
            if t.is_null() {
                continue;
            }
            let name_ptr = unsafe { libbpf_sys::btf__name_by_offset(btf_ptr, (*t).name_off) };
            if name_ptr.is_null() {
                continue;
            }
            let fname = unsafe { std::ffi::CStr::from_ptr(name_ptr) }
                .to_str()
                .unwrap_or("")
                .to_string();
            if fname.is_empty() {
                continue;
            }

            // Find the first line_info entry at or after this function's
            // instruction offset.
            let mut best: Option<&libbpf_sys::bpf_line_info> = None;
            for j in 0..nr_li {
                let li = unsafe {
                    &*(li_buf.as_ptr().add(j * li_rec) as *const libbpf_sys::bpf_line_info)
                };
                if li.insn_off >= fi.insn_off && best.is_none_or(|b| li.insn_off < b.insn_off) {
                    best = Some(li);
                }
            }
            if let Some(li) = best {
                let file_ptr =
                    unsafe { libbpf_sys::btf__name_by_offset(btf_ptr, li.file_name_off) };
                if !file_ptr.is_null() {
                    let file = unsafe { std::ffi::CStr::from_ptr(file_ptr) }
                        .to_str()
                        .unwrap_or("");
                    if !file.is_empty() {
                        let basename = file.rsplit('/').next().unwrap_or(file);
                        let line = li.line_col >> 10;
                        locs.insert(fname, format!("{basename}:{line}"));
                    }
                }
            }
        }
    }

    tracing::debug!(n = locs.len(), "resolve_bpf_source_locs");
    locs
}

/// Resolve the struct/union name behind a BTF type id, peeling a
/// single pointer level and skipping mods/typedefs. Returns `None`
/// for non-struct/union types.
fn resolve_btf_struct_name(
    b: &libbpf_rs::btf::Btf<'_>,
    type_id: libbpf_rs::btf::TypeId,
) -> Option<String> {
    use libbpf_rs::btf;
    let t = b.type_by_id::<btf::BtfType<'_>>(type_id)?;
    let inner = t.skip_mods_and_typedefs();
    let deref = if inner.kind() == btf::BtfKind::Ptr {
        inner.next_type()?.skip_mods_and_typedefs()
    } else {
        inner
    };
    if deref.kind() == btf::BtfKind::Struct || deref.kind() == btf::BtfKind::Union {
        Some(deref.name()?.to_str()?.to_string())
    } else {
        None
    }
}

/// Parse BTF from loaded BPF programs for callback signatures.
///
/// For each `(display_name, prog_id)`, resolves the typed params by:
/// 1. Looking for `____name` (inner function with typed params) in program BTF.
/// 2. Falling back to [`resolve_ops_callback_proto`] from vmlinux `sched_ext_ops`.
/// 3. Last resort: wrapper function with `void *ctx` (no useful params).
///
/// For struct pointer params not in [`STRUCT_FIELDS`], auto-discovers
/// scalar and cpumask pointer fields from BPF program BTF via
/// `discover_bpf_struct_fields`.
pub fn parse_bpf_btf_functions(
    func_names: &[(&str, u32)], // (display_name, prog_id)
) -> Vec<BtfFunc> {
    use libbpf_rs::btf;

    let mut by_prog: std::collections::HashMap<u32, Vec<&str>> = std::collections::HashMap::new();
    for (name, pid) in func_names {
        by_prog.entry(*pid).or_default().push(name);
    }

    let vmlinux = match btf::Btf::from_vmlinux() {
        Ok(b) => b,
        Err(e) => {
            tracing::warn!(%e, "parse_bpf_btf: failed to load vmlinux BTF");
            return Vec::new();
        }
    };

    let mut results = Vec::new();

    for (prog_id, names) in &by_prog {
        let prog_btf = match btf::Btf::from_prog_id(*prog_id) {
            Ok(b) => b,
            Err(e) => {
                tracing::debug!(prog_id, %e, "parse_bpf_btf: failed to load prog BTF");
                continue;
            }
        };

        for func_name in names {
            // Resolve the real typed params for this struct_ops callback.
            // Strategy:
            // 1. Try ____name (inner function with typed params) — may not be in BTF
            // 2. Try vmlinux sched_ext_ops member for the callback signature
            // 3. Fall back to wrapper name (void *ctx — no useful params)
            let inner_name = format!("____{func_name}");
            let (proto, skip_first_param) =
                if let Some(f) = prog_btf.type_by_name::<btf::types::Func<'_>>(&inner_name) {
                    let bt: btf::BtfType<'_> = *f;
                    if let Some(pt) = bt
                        .next_type()
                        .filter(|t| t.kind() == btf::BtfKind::FuncProto)
                    {
                        if let Ok(p) = TryInto::<btf::types::FuncProto<'_>>::try_into(pt) {
                            (Some(p), true)
                        } else {
                            (None, false)
                        }
                    } else {
                        (None, false)
                    }
                } else {
                    (None, false)
                };

            // Fallback: resolve from vmlinux sched_ext_ops struct member.
            let ops_proto = if proto.is_none() {
                let p = resolve_ops_callback_proto(&vmlinux, func_name);
                tracing::debug!(
                    func = func_name,
                    ops_found = p.is_some(),
                    "bpf_btf: ____name not in BTF, trying ops fallback",
                );
                p
            } else {
                None
            };

            let (use_proto_from_ops, proto) = if let Some(ref p) = proto {
                (false, p)
            } else if let Some(ref p) = ops_proto {
                (true, p)
            } else {
                // Last resort: use the wrapper (void *ctx).
                let f = match prog_btf.type_by_name::<btf::types::Func<'_>>(func_name) {
                    Some(f) => f,
                    None => continue,
                };
                let bt: btf::BtfType<'_> = *f;
                let _pt = match bt
                    .next_type()
                    .filter(|t| t.kind() == btf::BtfKind::FuncProto)
                {
                    Some(t) => t,
                    None => continue,
                };
                // Can't hold the proto across the match — just push an empty BtfFunc.
                results.push(BtfFunc {
                    name: func_name.to_string(),
                    params: vec![],
                    is_variadic: false,
                });
                continue;
            };

            let btf_for_params: &btf::Btf<'_> = if use_proto_from_ops {
                &vmlinux
            } else {
                &prog_btf
            };

            let mut params = Vec::new();
            let param_iter: Vec<_> = if skip_first_param && !use_proto_from_ops {
                proto.iter().skip(1).collect()
            } else {
                proto.iter().collect()
            };
            for (param_pos, param) in param_iter.into_iter().enumerate() {
                let mut name = param
                    .name
                    .and_then(|n| n.to_str())
                    .unwrap_or("")
                    .to_string();
                // Infer param name from type when vmlinux FuncProto
                // has empty names (function pointer BTF often lacks them).
                if name.is_empty() {
                    let sname = resolve_btf_struct_name(btf_for_params, param.ty);
                    name = match sname.as_deref() {
                        Some("task_struct") => "p".into(),
                        Some("rq") => "rq".into(),
                        Some("scx_exit_info") => "ei".into(),
                        Some("scx_init_task_args") => "args".into(),
                        Some("scx_cgroup_init_args") => "args".into(),
                        Some("scx_dispatch_q") => "dsq".into(),
                        _ => {
                            // For scalars, infer from position in known callbacks.
                            infer_scalar_param_name(func_name, param_pos)
                        }
                    };
                }
                let all_struct_name = resolve_btf_struct_name(btf_for_params, param.ty);
                let known_struct = all_struct_name
                    .as_ref()
                    .filter(|n| STRUCT_FIELDS.iter().any(|(s, _)| *s == n.as_str()))
                    .cloned();
                let is_ptr = btf_for_params
                    .type_by_id::<btf::BtfType<'_>>(param.ty)
                    .map(|t| t.skip_mods_and_typedefs().kind() == btf::BtfKind::Ptr)
                    .unwrap_or(false);

                // For unknown struct pointers: auto-discover fields
                let (auto_fields, type_name) = if is_ptr && known_struct.is_none() {
                    if let Some(ref sname) = all_struct_name {
                        // Check if this is a vmlinux type (skip auto-discovery
                        // for those — they'd need vmlinux BTF offsets).
                        let is_vmlinux: Option<btf::types::Struct<'_>> =
                            vmlinux.type_by_name(sname);
                        if is_vmlinux.is_some() {
                            (Vec::new(), Some(sname.clone()))
                        } else {
                            let fields = discover_bpf_struct_fields(&prog_btf, param.ty);
                            (fields, Some(sname.clone()))
                        }
                    } else {
                        (Vec::new(), None)
                    }
                } else {
                    (
                        Vec::new(),
                        all_struct_name.filter(|_| known_struct.is_none()),
                    )
                };

                tracing::debug!(
                    func = func_name, param = %name, struct_name = ?known_struct,
                    is_ptr, auto_fields = auto_fields.len(),
                    "bpf_btf: resolved param",
                );
                params.push(BtfParam {
                    name,
                    struct_name: known_struct,
                    is_ptr,
                    auto_fields,
                    type_name,
                    ..Default::default()
                });
            }

            results.push(BtfFunc {
                name: func_name.to_string(),
                params,
                is_variadic: false,
            });
        }
    }

    tracing::debug!(
        n = results.len(),
        "parse_bpf_btf: parsed BPF function signatures"
    );
    results
}

/// Infer scalar param names for sched_ext_ops callbacks.
/// Used when vmlinux FuncProto has empty names.
fn infer_scalar_param_name(func_name: &str, param_pos: usize) -> String {
    // Common sched_ext callback param names.
    const OPS_SCALARS: &[(&str, &[&str])] = &[
        ("dispatch", &["cpu"]),
        ("select_cpu", &["", "prev_cpu", "wake_flags"]),
        ("set_weight", &["", "weight"]),
        ("update_idle", &["cpu", "idle"]),
        ("cpu_acquire", &["cpu"]),
        ("cpu_release", &["cpu"]),
        ("cpu_online", &["cpu"]),
        ("cpu_offline", &["cpu"]),
    ];
    for (op, names) in OPS_SCALARS {
        if func_name.ends_with(op)
            && let Some(name) = names.get(param_pos)
            && !name.is_empty()
        {
            return name.to_string();
        }
    }
    format!("arg{param_pos}")
}

/// Resolve callback signature from vmlinux BTF's `sched_ext_ops` struct.
///
/// Maps scheduler function names (e.g. `ktstr_enqueue`) to ops members
/// (e.g. `enqueue`) by suffix matching, then follows the member's type
/// through Ptr to reach the FuncProto with typed parameters.
pub(super) fn resolve_ops_callback_proto<'a>(
    vmlinux: &'a libbpf_rs::btf::Btf<'a>,
    func_name: &str,
) -> Option<libbpf_rs::btf::types::FuncProto<'a>> {
    use libbpf_rs::btf::{BtfKind, BtfType};

    // Map function name to ops member name by finding the suffix
    // that matches a sched_ext_ops member.
    let ops: libbpf_rs::btf::types::Struct<'_> = vmlinux.type_by_name("sched_ext_ops")?;

    for member in ops.iter() {
        let member_name = member.name.and_then(|n| n.to_str()).unwrap_or("");
        if member_name.is_empty() || !func_name.ends_with(member_name) {
            continue;
        }
        // Follow the member type to find FuncProto (through Ptr if needed).
        let mut t = vmlinux.type_by_id::<BtfType<'_>>(member.ty)?;
        for _ in 0..20 {
            let inner = t.skip_mods_and_typedefs();
            match inner.kind() {
                BtfKind::Ptr => {
                    t = inner.next_type()?;
                }
                BtfKind::FuncProto => {
                    return inner.try_into().ok();
                }
                _ => break,
            }
        }
    }
    None
}

/// Resolve field specs for a BPF function's auto-discovered fields.
///
/// Uses BPF program BTF (not vmlinux) for offset resolution. Handles
/// both single-level field access (`->field`) and chained pointer
/// dereferences (`->ptr->field`). Skips params that have `struct_name`
/// set (those are handled by [`resolve_field_specs_with_btf`] with vmlinux BTF).
pub fn resolve_bpf_field_specs(btf_func: &BtfFunc, prog_id: u32) -> Vec<FieldSpec> {
    use libbpf_rs::btf;

    let prog_btf = match btf::Btf::from_prog_id(prog_id) {
        Ok(b) => b,
        Err(_) => return Vec::new(),
    };

    let mut specs = Vec::new();
    let mut field_idx: u32 = 0;

    let max_params = btf_func.params.len().min(6);
    for (param_idx, param) in btf_func.params[..max_params].iter().enumerate() {
        if param.struct_name.is_some() {
            // Known struct — handled by resolve_field_specs with vmlinux BTF.
            if let Some((_, fields)) = STRUCT_FIELDS
                .iter()
                .find(|(s, _)| Some(*s) == param.struct_name.as_deref())
            {
                field_idx += fields.len() as u32;
            }
            continue;
        }
        if !param.auto_fields.is_empty() {
            // Resolve offsets from BPF program BTF.
            let sname = match param.type_name.as_deref() {
                Some(n) => n,
                None => {
                    field_idx += param.auto_fields.len() as u32;
                    continue;
                }
            };
            // Find the struct in BPF program BTF.
            let struct_type: Option<btf::types::Struct<'_>> = prog_btf.type_by_name(sname);
            let composite = match struct_type {
                Some(s) => s,
                None => {
                    field_idx += param.auto_fields.len() as u32;
                    continue;
                }
            };

            for (_fname, access, _hint) in &param.auto_fields {
                let access = access.trim_start_matches("->");
                // Simple single-level field access.
                if !access.contains("->") {
                    let member_name = access.split('[').next().unwrap_or(access);
                    if let Some(offset) = resolve_bpf_member_offset(&composite, member_name) {
                        let size = resolve_bpf_member_size(&prog_btf, &composite, member_name)
                            .unwrap_or(8);
                        specs.push(FieldSpec {
                            param_idx: param_idx as u32,
                            offset,
                            size,
                            field_idx,
                            ptr_offset: 0,
                        });
                    }
                } else if let Some((ptr_member, target)) = access.split_once("->") {
                    // Chained pointer dereference (e.g. cpumask->bits[0]).
                    // Read pointer at ptr_member offset, then read target
                    // field through it.
                    let ptr_member = ptr_member.split('[').next().unwrap_or(ptr_member);
                    if let Some(ptr_off) = resolve_bpf_member_offset(&composite, ptr_member) {
                        // Resolve target struct from the pointer member's type.
                        // Target may be dot-separated for nested embedded structs
                        // (e.g. "cpumask.bits[0]" for bpf_cpumask).
                        let target_stripped = target.split('[').next().unwrap_or(target);
                        let target_parts: Vec<&str> = target_stripped.split('.').collect();
                        let mut target_off = 0u32;
                        let mut target_sz = 8u32;
                        'resolve_target: for member in composite.iter() {
                            let name = member.name.and_then(|n| n.to_str()).unwrap_or("");
                            if name != ptr_member {
                                continue;
                            }
                            // Follow pointer to find the target struct.
                            if let Some(pointed) =
                                prog_btf.type_by_id::<libbpf_rs::btf::BtfType<'_>>(member.ty)
                            {
                                let deref = pointed.skip_mods_and_typedefs();
                                if deref.kind() == libbpf_rs::btf::BtfKind::Ptr
                                    && let Some(inner) = deref.next_type()
                                {
                                    let mut current = inner.skip_mods_and_typedefs();
                                    let mut accumulated_off = 0u32;
                                    // Walk dot-separated path (e.g. cpumask.bits).
                                    for (i, part) in target_parts.iter().enumerate() {
                                        if let Ok(cur_struct) =
                                            TryInto::<libbpf_rs::btf::types::Struct<'_>>::try_into(
                                                current,
                                            )
                                        {
                                            if let Some(off) =
                                                resolve_bpf_member_offset(&cur_struct, part)
                                            {
                                                accumulated_off += off;
                                            }
                                            if i == target_parts.len() - 1 {
                                                // Last part — resolve size.
                                                if let Some(sz) = resolve_bpf_member_size(
                                                    &prog_btf,
                                                    &cur_struct,
                                                    part,
                                                ) {
                                                    target_sz = sz;
                                                }
                                            } else {
                                                // Intermediate part — find member type and descend.
                                                for m in cur_struct.iter() {
                                                    let mn = m
                                                        .name
                                                        .and_then(|n| n.to_str())
                                                        .unwrap_or("");
                                                    if mn == *part {
                                                        if let Some(mt) = prog_btf
                                                            .type_by_id::<libbpf_rs::btf::BtfType<
                                                            '_,
                                                        >>(
                                                            m.ty
                                                        ) {
                                                            current = mt.skip_mods_and_typedefs();
                                                        }
                                                        break;
                                                    }
                                                }
                                            }
                                        }
                                    }
                                    target_off = accumulated_off;
                                }
                            }
                            break 'resolve_target;
                        }
                        specs.push(FieldSpec {
                            param_idx: param_idx as u32,
                            offset: target_off,
                            size: target_sz,
                            field_idx,
                            ptr_offset: ptr_off,
                        });
                    }
                }
                field_idx += 1;
                if field_idx >= 16 {
                    break;
                }
            }
        } else if !param.is_ptr {
            field_idx += 1;
        }
    }

    tracing::debug!(
        n = specs.len(),
        func = %btf_func.name,
        "resolve_bpf_field_specs",
    );
    specs
}

/// Resolve byte offset of a member within a BPF program BTF struct.
fn resolve_bpf_member_offset(
    composite: &libbpf_rs::btf::types::Struct<'_>,
    member_name: &str,
) -> Option<u32> {
    use libbpf_rs::btf::types::MemberAttr;
    for member in composite.iter() {
        let name = member.name.and_then(|n| n.to_str()).unwrap_or("");
        if name == member_name {
            let bit_off = match member.attr {
                MemberAttr::Normal { offset } => offset,
                MemberAttr::BitField { offset, .. } => offset,
            };
            if bit_off % 8 != 0 {
                return None; // bitfield
            }
            return Some(bit_off / 8);
        }
    }
    None
}

/// Resolve byte size of a member within a BPF program BTF struct.
fn resolve_bpf_member_size(
    btf: &libbpf_rs::btf::Btf<'_>,
    composite: &libbpf_rs::btf::types::Struct<'_>,
    member_name: &str,
) -> Option<u32> {
    use libbpf_rs::btf::{BtfKind, BtfType};

    for member in composite.iter() {
        let name = member.name.and_then(|n| n.to_str()).unwrap_or("");
        if name != member_name {
            continue;
        }
        let t = btf.type_by_id::<BtfType<'_>>(member.ty)?;
        let inner = t.skip_mods_and_typedefs();
        return match inner.kind() {
            BtfKind::Int => {
                let int_ty: Result<libbpf_rs::btf::types::Int<'_>, _> = inner.try_into();
                Some(int_ty.map(|i| (i.bits / 8) as u32).unwrap_or(8))
            }
            BtfKind::Enum => Some(4),
            BtfKind::Enum64 => Some(8),
            BtfKind::Ptr => Some(8),
            _ => Some(8),
        };
    }
    None
}

/// Classification of a struct/union member that auto-discovery supports.
///
/// Both backends (`btf_rs` for vmlinux, `libbpf_rs::btf` for BPF program
/// BTF) inspect each member type and reduce it to one of these
/// variants before [`emit_member_field`] appends the access-pattern
/// tuple. Anything the backend cannot classify is represented by `None`
/// (skip member) — keeping the shared emission logic library-agnostic
/// without forcing a deeper trait abstraction over the two incompatible
/// BTF APIs.
enum MemberClass {
    /// Scalar integer with a rendering hint derived from BTF encoding
    /// (bool / signed / otherwise hex).
    Int(RenderHint),
    /// Enum or 64-bit enum; rendered as hex.
    Enum,
    /// Pointer to `struct cpumask` / `cpumask_t`. Access pattern
    /// dereferences `->bits[0]`.
    CpumaskPtr,
    /// Pointer to `struct bpf_cpumask`. Access pattern dereferences
    /// `->cpumask.bits[0]` (inner cpumask field).
    BpfCpumaskPtr,
}

/// Append the access-pattern tuple for a classified member to `fields`.
///
/// Shared by [`discover_bpf_struct_fields`] and
/// [`discover_vmlinux_struct_fields`] so the access-pattern strings and
/// the render hints stay in one place; adding a new member class (or
/// changing the cpumask offset convention) flows to both backends
/// automatically.
fn emit_member_field(
    fields: &mut Vec<(String, String, RenderHint)>,
    fname: &str,
    class: MemberClass,
) {
    match class {
        MemberClass::Int(hint) => fields.push((fname.to_string(), format!("->{fname}"), hint)),
        MemberClass::Enum => {
            fields.push((fname.to_string(), format!("->{fname}"), RenderHint::Hex));
        }
        MemberClass::CpumaskPtr => fields.push((
            fname.to_string(),
            format!("->{fname}->bits[0]"),
            RenderHint::Hex,
        )),
        MemberClass::BpfCpumaskPtr => fields.push((
            fname.to_string(),
            format!("->{fname}->cpumask.bits[0]"),
            RenderHint::Hex,
        )),
    }
}

/// Auto-discover struct fields from BPF program BTF for types not in
/// STRUCT_FIELDS. Walks members one level deep, emitting access patterns
/// for scalar, enum, and cpumask pointer fields.
///
/// Per-backend classification in [`classify_bpf_member`] feeds the
/// shared [`emit_member_field`] so the output contract matches
/// [`discover_vmlinux_struct_fields`] byte-for-byte wherever the two
/// backends see the same member shape.
fn discover_bpf_struct_fields(
    btf: &libbpf_rs::btf::Btf<'_>,
    type_id: libbpf_rs::btf::TypeId,
) -> Vec<(String, String, RenderHint)> {
    use libbpf_rs::btf::{BtfKind, BtfType};

    let t = match btf.type_by_id::<BtfType<'_>>(type_id) {
        Some(t) => t.skip_mods_and_typedefs(),
        None => return Vec::new(),
    };
    let inner = if t.kind() == BtfKind::Ptr {
        match t.next_type() {
            Some(t) => t.skip_mods_and_typedefs(),
            None => return Vec::new(),
        }
    } else {
        t
    };

    if inner.kind() != BtfKind::Struct && inner.kind() != BtfKind::Union {
        return Vec::new();
    }

    let composite: libbpf_rs::btf::types::Struct<'_> = match inner.try_into() {
        Ok(s) => s,
        Err(_) => return Vec::new(),
    };

    let mut fields = Vec::new();
    for member in composite.iter() {
        let fname = match member.name.and_then(|n| n.to_str()) {
            Some(n) if !n.is_empty() => n.to_string(),
            _ => continue,
        };

        let member_type = match btf.type_by_id::<BtfType<'_>>(member.ty) {
            Some(t) => t.skip_mods_and_typedefs(),
            None => continue,
        };

        if let Some(class) = classify_bpf_member(member_type) {
            emit_member_field(&mut fields, &fname, class);
        }
    }
    fields
}

/// Classify one libbpf-rs member type. Returns `None` when the type
/// is not supported by auto-discovery (the caller skips it).
fn classify_bpf_member(member_type: libbpf_rs::btf::BtfType<'_>) -> Option<MemberClass> {
    use libbpf_rs::btf::{BtfKind, BtfType};

    match member_type.kind() {
        BtfKind::Int => {
            let hint = if let Ok(int_ty) =
                TryInto::<libbpf_rs::btf::types::Int<'_>>::try_into(member_type)
            {
                match int_ty.encoding {
                    libbpf_rs::btf::types::IntEncoding::Bool => RenderHint::Bool,
                    libbpf_rs::btf::types::IntEncoding::Signed => RenderHint::Signed,
                    _ => RenderHint::Hex,
                }
            } else {
                RenderHint::Hex
            };
            Some(MemberClass::Int(hint))
        }
        BtfKind::Enum | BtfKind::Enum64 => Some(MemberClass::Enum),
        BtfKind::Ptr => {
            let deref = member_type
                .next_type()
                .map(|t: BtfType<'_>| t.skip_mods_and_typedefs());
            let pointed_name = deref
                .as_ref()
                .and_then(|t| t.name())
                .and_then(|n| n.to_str());
            match pointed_name {
                Some("cpumask") => Some(MemberClass::CpumaskPtr),
                Some("bpf_cpumask") => Some(MemberClass::BpfCpumaskPtr),
                _ => None,
            }
        }
        _ => None,
    }
}

/// Auto-discover struct fields from vmlinux BTF for types not in
/// STRUCT_FIELDS. Walks members one level deep via btf_rs, emitting
/// access patterns for scalar, enum, and cpumask pointer fields.
///
/// Per-backend classification in [`classify_vmlinux_member`] feeds the
/// shared [`emit_member_field`] so the output matches the libbpf-rs
/// sibling [`discover_bpf_struct_fields`] byte-for-byte on the member
/// shapes both backends recognise.
fn discover_vmlinux_struct_fields(
    btf: &btf_rs::Btf,
    type_id: u32,
) -> Vec<(String, String, RenderHint)> {
    use btf_rs::{BtfType, Type};

    let t = match btf.resolve_type_by_id(type_id) {
        Ok(t) => t,
        Err(_) => return Vec::new(),
    };

    // Follow PTR/CONST/VOLATILE/TYPEDEF to find the struct.
    let mut current = t;
    let struct_type = loop {
        match current {
            Type::Ptr(_)
            | Type::Const(_)
            | Type::Volatile(_)
            | Type::Typedef(_)
            | Type::Restrict(_)
            | Type::TypeTag(_) => {
                current = match btf.resolve_chained_type(current.as_btf_type().unwrap()) {
                    Ok(next) => next,
                    Err(_) => return Vec::new(),
                };
            }
            Type::Struct(s) | Type::Union(s) => break s,
            _ => return Vec::new(),
        }
    };

    let mut fields = Vec::new();
    for member in &struct_type.members {
        let fname = match btf.resolve_name(member) {
            Ok(n) if !n.is_empty() => n,
            _ => continue,
        };

        let member_tid = match member.get_type_id() {
            Some(tid) => tid,
            None => continue,
        };

        if let Some(class) = classify_vmlinux_member(btf, member_tid) {
            emit_member_field(&mut fields, &fname, class);
        }
    }
    fields
}

/// Classify one btf-rs member. Returns `None` when the type is not
/// supported by auto-discovery. Resolves const/volatile/typedef/tag
/// chains on both the member and the pointer target; loops cap at 20
/// iterations each to terminate on pathological chains.
fn classify_vmlinux_member(btf: &btf_rs::Btf, member_tid: u32) -> Option<MemberClass> {
    use btf_rs::Type;

    let mut member_type = btf.resolve_type_by_id(member_tid).ok()?;
    for _ in 0..20 {
        match &member_type {
            Type::Const(_)
            | Type::Volatile(_)
            | Type::Restrict(_)
            | Type::Typedef(_)
            | Type::TypeTag(_) => {
                member_type = btf.resolve_chained_type(member_type.as_btf_type()?).ok()?;
            }
            _ => break,
        }
    }

    match &member_type {
        Type::Int(i) => {
            let hint = if i.is_bool() {
                RenderHint::Bool
            } else if i.is_signed() {
                RenderHint::Signed
            } else {
                RenderHint::Hex
            };
            Some(MemberClass::Int(hint))
        }
        Type::Enum(_) | Type::Enum64(_) => Some(MemberClass::Enum),
        Type::Ptr(_) => {
            // Chase qualifiers to find the pointed-to struct name.
            let mut inner = btf.resolve_chained_type(member_type.as_btf_type()?).ok()?;
            for _ in 0..20 {
                match &inner {
                    Type::Const(_)
                    | Type::Volatile(_)
                    | Type::Restrict(_)
                    | Type::Typedef(_)
                    | Type::TypeTag(_) => {
                        inner = btf.resolve_chained_type(inner.as_btf_type()?).ok()?;
                    }
                    _ => break,
                }
            }
            let pointed_name = match &inner {
                Type::Struct(s) | Type::Union(s) => btf.resolve_name(s).ok(),
                _ => None,
            };
            match pointed_name.as_deref() {
                Some("cpumask") | Some("cpumask_t") => Some(MemberClass::CpumaskPtr),
                _ => None,
            }
        }
        _ => None,
    }
}

#[cfg(test)]
#[path = "btf_tests.rs"]
mod tests;

// ===================================================================
// BPF-program-BTF side: synthetic libbpf-rs BTF parser tests.
//
// The sibling `tests` module (btf_tests.rs) covers the `btf_rs`
// (vmlinux) classifiers via `btf_rs::Btf::from_bytes`. This module
// covers the `libbpf_rs::btf` (BPF program BTF) classifiers
// ([`classify_bpf_member`], [`discover_bpf_struct_fields`],
// [`resolve_bpf_member_offset`], [`resolve_bpf_member_size`],
// [`resolve_ops_callback_proto`]) against hand-built raw-BTF blobs
// loaded through `libbpf_rs::btf::Btf::from_path`.
//
// libbpf's `btf__parse` (the `from_path` backend) runs
// `btf_sanity_check`, which validates string offsets and referenced
// type ids but tolerates the minimal blobs built here. The shared
// `crate::test_support::btf_blob::cast_build_btf` builder emits the
// same 24-byte-header raw-BTF wire format libbpf parses; it lacks a
// FuncProto kind, so the ops-callback test builds that one blob with
// a local raw encoder mirroring the same header layout.
//
// No kernel, no loaded BPF program, no VM — fully host-runnable.
// `parse_bpf_btf_functions`, `resolve_bpf_field_specs`,
// `resolve_bpf_prog_full_name`, `resolve_bpf_source_locs`, and
// `discover_bpf_symbols` are NOT covered here: each calls
// `Btf::from_vmlinux`/`from_prog_id` or raw `bpf_obj_get_info_by_fd`
// against a live kernel / loaded program, with no synthetic-blob seam.
// ===================================================================
#[cfg(test)]
mod bpf_btf_tests {
    use super::*;
    use crate::test_support::btf_blob::{CastSynMember, CastSynType, cast_build_btf};
    use libbpf_rs::btf::Btf;
    use libbpf_rs::btf::TypeId;
    use libbpf_rs::btf::types::{FuncProto, Struct};

    /// Append a NUL-terminated `name` to a BTF string section and
    /// return its byte offset. Mirrors the `push_str` helper the
    /// `kernel_op_dispatch` BTF-gated suite uses.
    fn push_str(strings: &mut Vec<u8>, name: &str) -> u32 {
        let off = strings.len() as u32;
        strings.extend_from_slice(name.as_bytes());
        strings.push(0);
        off
    }

    /// Write a raw-BTF `blob` to a fresh temp file and load it through
    /// libbpf's `btf__parse` (the `Btf::from_path` backend). The
    /// returned `Btf<'static>` does not borrow the file, so the
    /// `NamedTempFile` may drop immediately after the parse.
    fn load_btf(blob: &[u8]) -> Btf<'static> {
        use std::io::Write as _;
        let mut f = tempfile::NamedTempFile::new().expect("temp file");
        f.write_all(blob).expect("write blob");
        f.flush().expect("flush");
        Btf::from_path(f.path()).expect("synthetic BPF BTF must parse via libbpf btf__parse")
    }

    /// A plain-unsigned int BTF type. `encoding == 0` → the libbpf
    /// `IntEncoding::None` arm; `bits == size * 8` so
    /// `resolve_bpf_member_size` reads `bits / 8 == size`.
    fn int_ty(name_off: u32, size: u32, encoding: u32) -> CastSynType {
        CastSynType::Int {
            name_off,
            size,
            encoding,
            offset: 0,
            bits: size * 8,
        }
    }

    /// `cast_build_btf` member at a byte-aligned offset.
    fn member(name_off: u32, type_id: u32, byte_offset: u32) -> CastSynMember {
        CastSynMember {
            name_off,
            type_id,
            byte_offset,
        }
    }

    // ---- classify_bpf_member: Int encodings ----

    /// `classify_bpf_member` Int arm maps each libbpf `IntEncoding` to
    /// the documented `RenderHint`: Bool→Bool, Signed→Signed,
    /// unsigned(None)→Hex. The `btf_rs` sibling
    /// (`classify_vmlinux_member_variants`) is covered; this is the
    /// only host coverage for the libbpf Int classifier.
    ///
    /// libbpf decodes the encoding nibble as `0b1`→Signed, `0b100`→Bool
    /// (`IntEncoding::try_from`, registry libbpf-rs btf/types.rs:388),
    /// so the int-data word's encoding field is 1 (signed) / 4 (bool) /
    /// 0 (none) — matching `cast_build_btf`'s `encoding << 24`.
    #[test]
    fn classify_bpf_member_int_encodings() {
        let mut strings: Vec<u8> = vec![0];
        let n_bool = push_str(&mut strings, "boolt");
        let n_signed = push_str(&mut strings, "signedt");
        let n_unsigned = push_str(&mut strings, "unsignedt");
        // encoding 4 = libbpf Bool; 1 = Signed; 0 = None (hex).
        let types = vec![
            int_ty(n_bool, 1, 4),     // id=1 bool
            int_ty(n_signed, 4, 1),   // id=2 signed
            int_ty(n_unsigned, 8, 0), // id=3 unsigned
        ];
        let blob = cast_build_btf(&types, &strings);
        let btf = load_btf(&blob);

        let t_bool = btf
            .type_by_id::<libbpf_rs::btf::BtfType<'_>>(TypeId::from(1))
            .expect("bool int");
        let t_signed = btf
            .type_by_id::<libbpf_rs::btf::BtfType<'_>>(TypeId::from(2))
            .expect("signed int");
        let t_unsigned = btf
            .type_by_id::<libbpf_rs::btf::BtfType<'_>>(TypeId::from(3))
            .expect("unsigned int");

        assert!(matches!(
            classify_bpf_member(t_bool),
            Some(MemberClass::Int(RenderHint::Bool))
        ));
        assert!(matches!(
            classify_bpf_member(t_signed),
            Some(MemberClass::Int(RenderHint::Signed))
        ));
        assert!(matches!(
            classify_bpf_member(t_unsigned),
            Some(MemberClass::Int(RenderHint::Hex))
        ));

        // The full emitted tuple for a Bool int named "ok" pins the
        // emit_member_field contract end-to-end.
        let mut fields = Vec::new();
        emit_member_field(
            &mut fields,
            "ok",
            classify_bpf_member(t_bool).expect("bool classifies"),
        );
        assert_eq!(
            fields,
            vec![("ok".to_string(), "->ok".to_string(), RenderHint::Bool)]
        );
    }

    // ---- classify_bpf_member: Enum / Enum64 / Ptr / catch-all ----

    /// Enum and Enum64 → `MemberClass::Enum`; a `cpumask`-pointed Ptr
    /// → `CpumaskPtr`; a `bpf_cpumask`-pointed Ptr → `BpfCpumaskPtr`
    /// (the ONLY path that produces `BpfCpumaskPtr` — the vmlinux
    /// sibling never emits it); a non-cpumask Ptr and a bare Struct
    /// → `None`.
    #[test]
    fn classify_bpf_member_enum_enum64_and_ptr_kinds() {
        let mut strings: Vec<u8> = vec![0];
        let n_e = push_str(&mut strings, "color");
        let n_e64 = push_str(&mut strings, "bigcolor");
        let n_cpumask = push_str(&mut strings, "cpumask");
        let n_bpf_cpumask = push_str(&mut strings, "bpf_cpumask");
        let n_widget = push_str(&mut strings, "widget");
        let n_u64 = push_str(&mut strings, "u64");
        let n_bits = push_str(&mut strings, "bits");

        // ids: 1 enum, 2 enum64, 3 u64, 4 struct cpumask, 5 ptr->cpumask,
        // 6 struct bpf_cpumask, 7 ptr->bpf_cpumask, 8 struct widget,
        // 9 ptr->widget.
        let types = vec![
            CastSynType::Enum {
                name_off: n_e,
                size: 4,
                signed: false,
                members: vec![],
            }, // id=1
            CastSynType::Enum64 {
                name_off: n_e64,
                size: 8,
                signed: false,
                members: vec![],
            }, // id=2
            int_ty(n_u64, 8, 0), // id=3
            CastSynType::Struct {
                name_off: n_cpumask,
                size: 8,
                members: vec![member(n_bits, 3, 0)],
            }, // id=4
            CastSynType::Ptr { type_id: 4 }, // id=5
            CastSynType::Struct {
                name_off: n_bpf_cpumask,
                size: 8,
                members: vec![member(n_bits, 3, 0)],
            }, // id=6
            CastSynType::Ptr { type_id: 6 }, // id=7
            CastSynType::Struct {
                name_off: n_widget,
                size: 8,
                members: vec![member(n_bits, 3, 0)],
            }, // id=8
            CastSynType::Ptr { type_id: 8 }, // id=9
        ];
        let blob = cast_build_btf(&types, &strings);
        let btf = load_btf(&blob);

        let by_id = |id: u32| {
            btf.type_by_id::<libbpf_rs::btf::BtfType<'_>>(TypeId::from(id))
                .expect("type id present")
        };

        assert!(matches!(
            classify_bpf_member(by_id(1)),
            Some(MemberClass::Enum)
        ));
        assert!(matches!(
            classify_bpf_member(by_id(2)),
            Some(MemberClass::Enum)
        ));
        assert!(matches!(
            classify_bpf_member(by_id(5)),
            Some(MemberClass::CpumaskPtr)
        ));
        assert!(matches!(
            classify_bpf_member(by_id(7)),
            Some(MemberClass::BpfCpumaskPtr)
        ));
        // Pointer to a non-cpumask struct → None (skipped).
        assert!(classify_bpf_member(by_id(9)).is_none());
        // A bare struct (not int/enum/ptr) → None (catch-all arm).
        assert!(classify_bpf_member(by_id(4)).is_none());
    }

    // ---- discover_bpf_struct_fields ----

    /// `discover_bpf_struct_fields` emits the exact access-pattern Vec
    /// for a struct of mixed members, byte-for-byte the libbpf-side
    /// sibling of `discover_vmlinux_struct_fields_emits_expected_access`.
    /// The `bm`/BpfCpumaskPtr entry pins the BPF-only access string
    /// `->bm->cpumask.bits[0]` (unreachable on the vmlinux side).
    /// Anonymous and unclassified (non-cpumask pointer) members are
    /// dropped.
    #[test]
    fn discover_bpf_struct_fields_emits_expected_access() {
        let mut strings: Vec<u8> = vec![0];
        let n_u64 = push_str(&mut strings, "u64");
        let n_e = push_str(&mut strings, "ek");
        let n_cpumask = push_str(&mut strings, "cpumask");
        let n_bpf_cpumask = push_str(&mut strings, "bpf_cpumask");
        let n_widget = push_str(&mut strings, "widget");
        let n_bits = push_str(&mut strings, "bits");
        let n_foo = push_str(&mut strings, "foo");
        let n_sc = push_str(&mut strings, "sc");
        let n_ok = push_str(&mut strings, "ok");
        let n_em = push_str(&mut strings, "e");
        let n_cm = push_str(&mut strings, "cm");
        let n_bm = push_str(&mut strings, "bm");
        let n_other = push_str(&mut strings, "other");

        // Leaf / pointee types.
        let types = vec![
            int_ty(n_u64, 8, 0), // id=1 unsigned (hex)
            int_ty(0, 4, 1),     // id=2 signed (anonymous int type ok)
            int_ty(0, 1, 4),     // id=3 bool
            CastSynType::Enum {
                name_off: n_e,
                size: 4,
                signed: false,
                members: vec![],
            }, // id=4 enum
            CastSynType::Struct {
                name_off: n_cpumask,
                size: 8,
                members: vec![member(n_bits, 1, 0)],
            }, // id=5 cpumask
            CastSynType::Ptr { type_id: 5 }, // id=6 cpumask*
            CastSynType::Struct {
                name_off: n_bpf_cpumask,
                size: 8,
                members: vec![member(n_bits, 1, 0)],
            }, // id=7 bpf_cpumask
            CastSynType::Ptr { type_id: 7 }, // id=8 bpf_cpumask*
            CastSynType::Struct {
                name_off: n_widget,
                size: 8,
                members: vec![member(n_bits, 1, 0)],
            }, // id=9 widget
            CastSynType::Ptr { type_id: 9 }, // id=10 widget*
            // id=11 foo: signed sc, bool ok, enum e, cpumask* cm,
            // bpf_cpumask* bm, widget* (non-cpumask, dropped),
            // anonymous member (dropped).
            CastSynType::Struct {
                name_off: n_foo,
                size: 48,
                members: vec![
                    member(n_sc, 2, 0),
                    member(n_ok, 3, 8),
                    member(n_em, 4, 16),
                    member(n_cm, 6, 24),
                    member(n_bm, 8, 32),
                    member(n_other, 10, 40),
                    member(0, 1, 44), // anonymous → skipped
                ],
            }, // id=11
            CastSynType::Ptr { type_id: 11 }, // id=12 foo*
        ];
        let blob = cast_build_btf(&types, &strings);
        let btf = load_btf(&blob);

        let expected = vec![
            ("sc".to_string(), "->sc".to_string(), RenderHint::Signed),
            ("ok".to_string(), "->ok".to_string(), RenderHint::Bool),
            ("e".to_string(), "->e".to_string(), RenderHint::Hex),
            (
                "cm".to_string(),
                "->cm->bits[0]".to_string(),
                RenderHint::Hex,
            ),
            (
                "bm".to_string(),
                "->bm->cpumask.bits[0]".to_string(),
                RenderHint::Hex,
            ),
            // `other` (non-cpumask ptr) and the anonymous member are
            // both absent.
        ];

        // Direct struct id.
        let fields = discover_bpf_struct_fields(&btf, TypeId::from(11));
        assert_eq!(fields, expected);

        // Ptr-to-foo id exercises the Ptr-peel branch and yields the
        // same Vec.
        let via_ptr = discover_bpf_struct_fields(&btf, TypeId::from(12));
        assert_eq!(via_ptr, expected);
    }

    // ---- resolve_bpf_member_offset ----

    /// `resolve_bpf_member_offset`: a byte-aligned `Normal` member
    /// returns `Some(byte_offset)`; a bitfield at a non-byte-aligned
    /// bit offset returns `None`; an unknown member returns `None`.
    #[test]
    fn resolve_bpf_member_offset_normal_and_bitfield_reject() {
        // (a) Byte-aligned normal members.
        let mut strings: Vec<u8> = vec![0];
        let n_u64 = push_str(&mut strings, "u64");
        let n_s = push_str(&mut strings, "s");
        let n_cpu = push_str(&mut strings, "cpu");
        let n_second = push_str(&mut strings, "second");
        let normal = vec![
            int_ty(n_u64, 8, 0), // id=1
            CastSynType::Struct {
                name_off: n_s,
                size: 16,
                members: vec![member(n_cpu, 1, 0), member(n_second, 1, 8)],
            }, // id=2
        ];
        let normal_blob = cast_build_btf(&normal, &strings);
        let btf = load_btf(&normal_blob);
        let composite: Struct<'_> = btf.type_by_name("s").expect("struct s");
        assert_eq!(resolve_bpf_member_offset(&composite, "cpu"), Some(0));
        assert_eq!(resolve_bpf_member_offset(&composite, "second"), Some(8));
        assert_eq!(resolve_bpf_member_offset(&composite, "nonexistent"), None);

        // (b) Bitfield member at bit offset 4 (size 1) — kind_flag set,
        // so libbpf decodes MemberAttr::BitField { offset: 4 }; 4 % 8
        // != 0 triggers the bitfield reject.
        let mut bstrings: Vec<u8> = vec![0];
        let bn_u32 = push_str(&mut bstrings, "u32");
        let bn_bf = push_str(&mut bstrings, "bf");
        let bn_flag = push_str(&mut bstrings, "flag");
        let bf_types = vec![
            int_ty(bn_u32, 4, 0), // id=1
            CastSynType::BitfieldStruct {
                name_off: bn_bf,
                size: 8,
                members: vec![crate::test_support::btf_blob::CastSynBitMember {
                    name_off: bn_flag,
                    type_id: 1,
                    bit_offset: 4,
                    bitfield_size: 1,
                }],
            }, // id=2
        ];
        let bf_blob = cast_build_btf(&bf_types, &bstrings);
        let bf_btf = load_btf(&bf_blob);
        let bf: Struct<'_> = bf_btf.type_by_name("bf").expect("struct bf");
        assert_eq!(
            resolve_bpf_member_offset(&bf, "flag"),
            None,
            "non-byte-aligned bitfield member must be rejected"
        );
    }

    // ---- resolve_bpf_member_size ----

    /// `resolve_bpf_member_size` per kind: Int → bits/8; Enum →
    /// hardcoded 4 (even when the declared enum size is 8 — the
    /// divergence from a declared-size read); Enum64 → 8; Ptr → 8;
    /// any other kind (embedded struct) → 8 default; missing → None.
    #[test]
    fn resolve_bpf_member_size_per_kind() {
        let mut strings: Vec<u8> = vec![0];
        let n_u32 = push_str(&mut strings, "u32t");
        let n_u64 = push_str(&mut strings, "u64t");
        let n_e = push_str(&mut strings, "ek");
        let n_inner = push_str(&mut strings, "inner");
        let n_x = push_str(&mut strings, "x");
        let n_s = push_str(&mut strings, "s");
        let n_u32m = push_str(&mut strings, "u32m");
        let n_u64m = push_str(&mut strings, "u64m");
        let n_em = push_str(&mut strings, "e");
        let n_e64m = push_str(&mut strings, "e64");
        let n_ptrm = push_str(&mut strings, "p");
        let n_structm = push_str(&mut strings, "embed");
        let n_e64 = push_str(&mut strings, "be");

        let types = vec![
            int_ty(n_u32, 4, 0), // id=1 u32
            int_ty(n_u64, 8, 0), // id=2 u64
            CastSynType::Enum {
                name_off: n_e,
                size: 8, // declared size 8 — must NOT be reflected (hardcoded 4).
                signed: false,
                members: vec![],
            }, // id=3 enum
            CastSynType::Enum64 {
                name_off: n_e64,
                size: 8,
                signed: false,
                members: vec![],
            }, // id=4 enum64
            CastSynType::Ptr { type_id: 2 }, // id=5 u64*
            CastSynType::Struct {
                name_off: n_inner,
                size: 4,
                members: vec![member(n_x, 1, 0)],
            }, // id=6 embedded struct
            // id=7 s with one member of each kind.
            CastSynType::Struct {
                name_off: n_s,
                size: 64,
                members: vec![
                    member(n_u32m, 1, 0),
                    member(n_u64m, 2, 8),
                    member(n_em, 3, 16),
                    member(n_e64m, 4, 24),
                    member(n_ptrm, 5, 32),
                    member(n_structm, 6, 40),
                ],
            }, // id=7
        ];
        let blob = cast_build_btf(&types, &strings);
        let btf = load_btf(&blob);
        let s: Struct<'_> = btf.type_by_name("s").expect("struct s");

        assert_eq!(resolve_bpf_member_size(&btf, &s, "u32m"), Some(4));
        assert_eq!(resolve_bpf_member_size(&btf, &s, "u64m"), Some(8));
        // Enum is hardcoded to 4 regardless of the declared size_type (8).
        assert_eq!(resolve_bpf_member_size(&btf, &s, "e"), Some(4));
        assert_eq!(resolve_bpf_member_size(&btf, &s, "e64"), Some(8));
        assert_eq!(resolve_bpf_member_size(&btf, &s, "p"), Some(8));
        // Embedded struct member → default arm (8).
        assert_eq!(resolve_bpf_member_size(&btf, &s, "embed"), Some(8));
        // Missing member name → None fall-through.
        assert_eq!(resolve_bpf_member_size(&btf, &s, "nope"), None);
    }

    // ---- resolve_ops_callback_proto ----

    /// Build a raw-BTF blob containing one Int, a 2-param FuncProto, a
    /// Ptr to it, and a `sched_ext_ops` struct with an `enqueue`
    /// member typed as that function pointer. `cast_build_btf` has no
    /// FuncProto kind, so this is hand-encoded with the same 24-byte
    /// header layout. `with_ops` controls whether the `sched_ext_ops`
    /// struct is emitted.
    fn build_ops_btf(with_ops: bool) -> Vec<u8> {
        const KIND_INT: u32 = 1;
        const KIND_PTR: u32 = 2;
        const KIND_STRUCT: u32 = 4;
        const KIND_FUNC_PROTO: u32 = 13;

        let mut strings: Vec<u8> = vec![0];
        let n_int = push_str(&mut strings, "int");
        let n_a = push_str(&mut strings, "a");
        let n_b = push_str(&mut strings, "b");
        let n_ops = push_str(&mut strings, "sched_ext_ops");
        let n_enqueue = push_str(&mut strings, "enqueue");

        // 12-byte `btf_type` header: name_off, info, size_type.
        let push_hdr = |sec: &mut Vec<u8>, name_off: u32, kind: u32, vlen: u32, size_type: u32| {
            sec.extend_from_slice(&name_off.to_le_bytes());
            let info = ((kind << 24) & 0x1f00_0000) | (vlen & 0xffff);
            sec.extend_from_slice(&info.to_le_bytes());
            sec.extend_from_slice(&size_type.to_le_bytes());
        };

        let mut sec: Vec<u8> = Vec::new();
        // id=1: int (12-byte hdr + 4-byte int data; encoding 0, bits 32).
        push_hdr(&mut sec, n_int, KIND_INT, 0, 4);
        sec.extend_from_slice(&32u32.to_le_bytes());
        // id=2: FuncProto, ret=int(1), 2 params (a:int, b:int).
        push_hdr(&mut sec, 0, KIND_FUNC_PROTO, 2, 1);
        sec.extend_from_slice(&n_a.to_le_bytes());
        sec.extend_from_slice(&1u32.to_le_bytes());
        sec.extend_from_slice(&n_b.to_le_bytes());
        sec.extend_from_slice(&1u32.to_le_bytes());
        // id=3: Ptr -> FuncProto(2).
        push_hdr(&mut sec, 0, KIND_PTR, 0, 2);
        if with_ops {
            // id=4: struct sched_ext_ops { enqueue @0 : Ptr(3) }.
            push_hdr(&mut sec, n_ops, KIND_STRUCT, 1, 8);
            sec.extend_from_slice(&n_enqueue.to_le_bytes());
            sec.extend_from_slice(&3u32.to_le_bytes()); // member type = Ptr(3)
            sec.extend_from_slice(&0u32.to_le_bytes()); // member offset (bits) = 0
        }

        let type_len = sec.len() as u32;
        let str_len = strings.len() as u32;
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
        blob.extend_from_slice(&strings);
        blob
    }

    /// `resolve_ops_callback_proto`: a func name whose suffix matches a
    /// `sched_ext_ops` member resolves through the member's Ptr to the
    /// FuncProto (param count pinned); a non-matching func name and a
    /// BTF lacking `sched_ext_ops` both return `None`.
    #[test]
    fn resolve_ops_callback_proto_suffix_match_and_ptr_chase() {
        let btf = load_btf(&build_ops_btf(true));
        let proto: FuncProto<'_> =
            resolve_ops_callback_proto(&btf, "ktstr_enqueue").expect("enqueue resolves");
        assert_eq!(
            proto.iter().count(),
            2,
            "the enqueue FuncProto has exactly 2 params"
        );

        // func_name matching no ops member → None.
        assert!(resolve_ops_callback_proto(&btf, "ktstr_no_such_op").is_none());

        // BTF without a sched_ext_ops struct → type_by_name None early
        // return.
        let btf_no_ops = load_btf(&build_ops_btf(false));
        assert!(resolve_ops_callback_proto(&btf_no_ops, "ktstr_enqueue").is_none());
    }
}
