//! End-to-end tests for the BPF cast analysis pipeline.
//!
//! Boots scx-ktstr with `--stall-after=1`, lets the SCX watchdog
//! latch `SCX_EXIT_ERROR_STALL`, and inspects the freeze
//! coordinator's `FailureDumpReport` to verify that the host-side
//! cast analysis (`src/monitor/cast_analysis/mod.rs`) actually rewrote
//! plain `u64` fields into typed-pointer renders that the dump
//! pipeline chased through to the target struct's contents.
//!
//! # Pipeline under test
//!
//! 1. The VM builder constructs `LazyCastMap::new(scheduler_binary)`
//!    (no I/O at builder time).
//! 2. At first failure-dump, `LazyCastMap::get_full()` runs
//!    `cached_cast_analysis_for_scheduler`, which reads the binary,
//!    hashes it, parses each embedded `.bpf.objs` ELF, runs
//!    `analyze_casts`, and caches the resulting
//!    `CastAnalysisOutput` (cast map + cross-BTF index).
//! 3. The freeze coordinator threads
//!    `CastAnalysisOutput.cast_maps.first()` into
//!    `DumpContext::cast_map` and `CastAnalysisOutput.fwd_index`
//!    plus `btfs` into `DumpContext::cross_btf_fwd_index` for the
//!    dump pass.
//! 4. The BPF program in `scx-ktstr/src/bpf/main.bpf.c` is
//!    constructed so its bytecode contains the patterns the
//!    analyzer detects on two distinct cross-domain paths:
//!      - `ktstr_stash_task_kptr(taskc, p)` is a static BPF-to-BPF
//!        helper whose `.BTF.ext` `func_info` entry seeds R1 with
//!        `Pointer{ktstr_arena_ctx}` and R2 with
//!        `Pointer{task_struct}` at function entry. The helper's
//!        body stores R2 into `*(R1 + 16)` — a plain DW STX through
//!        two typed registers, which `Analyzer::handle_stx` records
//!        as `(ktstr_arena_ctx, 16) → task_struct,
//!        AddrSpace::Kernel`. Source domain: arena (the per-task
//!        ktstr_arena_ctx page); target domain: kernel slab.
//!      - `ktstr_train_bss_to_arena(holder)` is a static BPF-to-BPF
//!        helper whose `.BTF.ext` `func_info` entry seeds R1 with
//!        `Pointer{ktstr_bss_arena_holder}`. The body loads
//!        `holder->arena_target` (LDX through R1, offset 0) into a
//!        register that becomes `LoadedU64Field`, casts to
//!        `struct ktstr_arena_ctx __arena *` (BPF_ADDR_SPACE_CAST
//!        marks `arena_confirmed`), then dereferences three fields
//!        of the target struct so `Analyzer`'s shape-intersection
//!        step uniquely resolves the target as `ktstr_arena_ctx`.
//!        The resulting CastMap entry is
//!        `(ktstr_bss_arena_holder, 0) → ktstr_arena_ctx,
//!        AddrSpace::Arena`. Source domain: .bss (a global struct
//!        in the scheduler's data section); target domain: arena
//!        (the captured per-task page).
//! 5. At freeze time, the dump pipeline walks every map. For
//!    scx_task_map (TASK_STORAGE), `chase_sdt_data_payload`
//!    renders `meta.target_type_id` (== `ktstr_arena_ctx`)
//!    against each per-task arena page. For the scheduler's
//!    `.bss` map, the BTF Datasec walker surfaces every global
//!    variable, including `ktstr_bss_arena_holder`, as a struct
//!    render.
//! 6. Inside each per-member render, the cast intercept in
//!    `render_member` consults
//!    `MemReader::cast_lookup(parent=*_btf_id, off=*_offset)`. On
//!    a hit, `render_cast_pointer` chases via `read_kva`
//!    (kernel-tagged hits) or `read_arena` (arena-tagged hits)
//!    and emits `Ptr{value, deref: Some(Struct{...})}` instead of
//!    a raw u64 counter.
//!
//! # Assertion strategy
//!
//! The user-facing bar is "the rendered dump shows chased struct
//! contents, NOT raw u64 integers." The tests below enforce that
//! end-to-end, against the actual JSON the freeze coordinator
//! writes, not against synthetic BTF or stub readers (the unit
//! tests in `src/monitor/btf_render/tests/` already cover those
//! shapes). Each assertion reports the failing structural gate and
//! points at the persisted dump rather than copying captured arena
//! pages and nested payloads into nextest output.

mod common;

use anyhow::Result;
use common::failure_dump::{failure_dump_artifact, read_failure_dump};
use ktstr::assert::AssertResult;
use ktstr::prelude::VmResult;
use ktstr::scenario::ops::{HoldSpec, Step, await_accessor_ready, execute_steps};
use ktstr::test_support::{Scheduler, SchedulerSpec};

const KTSTR_SCHED: Scheduler =
    Scheduler::named("ktstr_sched").binary(SchedulerSpec::Discover("scx-ktstr"));

/// Locate scx-ktstr's TASK_STORAGE map (`scx_task_map`) inside the
/// dump JSON's `maps` array. Used by every cast E2E scenario as the
/// entry point — the rendered ktstr_arena_ctx that triggers the
/// cast intercept lives under `entries[].payload`. Returns the JSON
/// value for the map; bails with a diagnostic if it cannot be found.
fn find_task_storage_map<'a>(
    dump: &'a serde_json::Value,
    dump_artifact: &str,
) -> Result<&'a serde_json::Value> {
    // BPF_MAP_TYPE_TASK_STORAGE = 29 in `enum bpf_map_type`
    // (include/uapi/linux/bpf.h; 23 is BPF_MAP_TYPE_STACK). The
    // failure-dump JSON exposes the map_type integer verbatim from
    // libbpf, so we filter on that rather than the libbpf-name
    // (`scx_task_map` is the BPF-side var name; the kernel-side
    // info.name may carry a BTF section prefix on some libbpf
    // versions).
    const BPF_MAP_TYPE_TASK_STORAGE: u64 = 29;
    let maps = dump
        .get("maps")
        .and_then(|m| m.as_array())
        .ok_or_else(|| anyhow::anyhow!("dump missing top-level `maps` array"))?;
    maps.iter()
        .find(|m| {
            m.get("map_type")
                .and_then(|t| t.as_u64())
                .is_some_and(|t| t == BPF_MAP_TYPE_TASK_STORAGE)
        })
        .ok_or_else(|| {
            anyhow::anyhow!(
                "dump has no BPF_MAP_TYPE_TASK_STORAGE map (looked across {} maps). \
                 scx-ktstr declares `scx_task_map` via lib/sdt_task.bpf.c so the \
                 map MUST appear; absence means the walker filtered it, \
                 sdt_alloc was disabled, or the scheduler aborted before \
                 task_storage allocation; {dump_artifact}",
                maps.len()
            )
        })
}

/// Look up a member by name inside a `Struct`-shaped `RenderedValue`.
/// Returns the member's `value` JSON. Bails with a clear error if
/// the parent isn't a struct or the member is missing.
fn struct_member<'a>(
    parent: &'a serde_json::Value,
    member_name: &str,
    dump_artifact: &str,
) -> Result<&'a serde_json::Value> {
    let kind = parent
        .get("kind")
        .and_then(|k| k.as_str())
        .unwrap_or("<no-kind>");
    if kind != "struct" {
        anyhow::bail!(
            "expected a `struct`-kind RenderedValue but got kind={kind:?}; \
             {dump_artifact}"
        );
    }
    let members = parent
        .get("members")
        .and_then(|m| m.as_array())
        .ok_or_else(|| anyhow::anyhow!("struct has no `members` array; {dump_artifact}"))?;
    let member = members
        .iter()
        .find(|m| m.get("name").and_then(|n| n.as_str()) == Some(member_name))
        .ok_or_else(|| {
            let names: Vec<&str> = members
                .iter()
                .filter_map(|m| m.get("name").and_then(|n| n.as_str()))
                .collect();
            anyhow::anyhow!(
                "struct member `{member_name}` not found; got names: {names:?}; \
                 {dump_artifact}"
            )
        })?;
    member.get("value").ok_or_else(|| {
        anyhow::anyhow!(
            "member `{member_name}` has no `value` field; \
             {dump_artifact}"
        )
    })
}

/// Peel every truncation wrapper around a rendered value.
///
/// A clipped pointer chase legitimately produces two wrappers: the
/// chase records that its read was shorter than the pointee's BTF
/// size, then the inner struct renderer independently records that
/// the same byte slice was shorter than the struct. Keep walking
/// until the underlying value is reached rather than baking in one
/// particular number of clipping layers.
fn peel_truncated<'a>(
    mut value: &'a serde_json::Value,
    dump_artifact: &str,
) -> Result<(&'a serde_json::Value, usize)> {
    let mut depth = 0usize;
    while value.get("kind").and_then(|kind| kind.as_str()) == Some("truncated") {
        value = value.get("partial").ok_or_else(|| {
            anyhow::anyhow!(
                "Truncated value at depth {depth} has no `partial`; \
                 {dump_artifact}"
            )
        })?;
        depth += 1;
    }
    Ok((value, depth))
}

#[test]
fn peel_truncated_reaches_struct_through_multiple_clipping_layers() {
    let rendered = serde_json::json!({
        "kind": "truncated",
        "needed": 8240,
        "had": 4096,
        "partial": {
            "kind": "truncated",
            "needed": 8240,
            "had": 4096,
            "partial": {
                "kind": "struct",
                "type_name": "task_struct",
                "members": [],
            },
        },
    });

    let (inner, depth) = peel_truncated(&rendered, "synthetic dump")
        .expect("nested truncation wrappers must be transparent");
    assert_eq!(depth, 2);
    assert_eq!(
        inner.get("kind").and_then(|kind| kind.as_str()),
        Some("struct")
    );
    assert_eq!(
        inner.get("type_name").and_then(|name| name.as_str()),
        Some("task_struct")
    );
}

/// Asserts that scx-ktstr's per-task arena context (`struct
/// ktstr_arena_ctx`) renders with its `task_kptr` u64 field
/// rewritten by the cast analysis pipeline into a typed Ptr that
/// chases through to the target `task_struct` and surfaces its
/// kernel-side fields (pid, comm).
///
/// The chase verifies the entire cross-domain pointer pipeline:
///   - cast analyzer (host) detected `(ktstr_arena_ctx, off=16) →
///     (task_struct, AddrSpace::Kernel)` from the BPF bytecode
///   - the freeze coordinator threaded the resulting CastMap into
///     the dump pipeline
///   - the renderer's `MemReader::cast_lookup` returned the hit on
///     the right `(parent_btf_id, member_offset)`
///   - `render_cast_pointer`'s kernel-arm read `task_struct` bytes
///     via `read_kva` and recursed
///   - the recursive render walked task_struct's BTF and surfaced
///     identifying fields (pid, comm) the test asserts on.
///
/// Negative assertions on the same struct prove the gate fires
/// only on flagged fields — `magic` (u64 at offset 0) and
/// `counter` (u32 at offset 8) MUST render as plain integers,
/// never as Ptr.
fn scenario_cast_analysis_chases_kernel_kptr(ctx: &ktstr::scenario::Ctx) -> Result<AssertResult> {
    await_accessor_ready();
    let steps = vec![Step {
        setup: vec![ctx.cgroup_def("cg_0")].into(),
        ops: vec![],
        hold: HoldSpec::FULL,
    }];
    execute_steps(ctx, steps)
}

/// Host-side post_vm assertion for `cast_analysis_chases_kernel_kptr`.
/// The guest is a separate process and cannot read the host-written
/// dump, so the chase assertions run here; the callback's Err is a hard
/// FAIL (`PostVmAssertionFailure`) that `expect_err` does not invert.
fn check_cast_analysis_chases_kernel_kptr(result: &VmResult) -> Result<()> {
    let dump = read_failure_dump(result)?;
    let dump_artifact = failure_dump_artifact(result);

    let task_storage = find_task_storage_map(&dump, &dump_artifact)?;
    let entries = task_storage
        .get("entries")
        .and_then(|e| e.as_array())
        .ok_or_else(|| {
            anyhow::anyhow!(
                "scx_task_map has no `entries` array — TASK_STORAGE walker did \
                 not populate the map. With workers_per_cgroup>0 driving load, \
                 at least one task must have a per-task ktstr_arena_ctx. \
                 {dump_artifact}"
            )
        })?;
    if entries.is_empty() {
        anyhow::bail!(
            "scx_task_map.entries is empty — `bpf_task_storage_get` was never \
             called (no task ran ktstr_init_task before freeze) or the local-\
             storage walker found no live owners; {dump_artifact}"
        );
    }

    // Scan entries: each one's `payload` is the BTF-rendered
    // `ktstr_arena_ctx` (chased through `sdt_data.payload[]`). The
    // cast intercept fires on the `task_kptr` member when the cast
    // map produced an entry. Some entries may be from kthreads
    // (init_task ran but `p` was the swapper / a kthread the
    // scheduler treats specially); collect every payload so the
    // assertions can find one that proves the chase fired.
    let payloads: Vec<&serde_json::Value> = entries
        .iter()
        .filter_map(|e| e.get("payload"))
        .filter(|p| !p.is_null())
        .collect();
    if payloads.is_empty() {
        anyhow::bail!(
            "no scx_task_map entry has a non-null `payload` — \
             chase_sdt_data_payload returned None for every entry. The \
             allocator metadata may be unresolved (no target_type_id \
             discovery), the per-task `sdt_data __arena *` field offset \
             was not found, or every captured arena pointer fell outside \
             the kern_vm window. entry count: {}; {dump_artifact}",
            entries.len(),
        );
    }

    // Each payload should be a Struct{type_name: Some("ktstr_arena_ctx"),
    // members: [...]}. A task_struct is larger than the pointer-chase cap,
    // and kernel reads stop at the current page edge. Consequently an entry
    // whose task_struct begins near the page tail may not include the later
    // `pid` and `comm` members even though the cast and dereference both
    // succeeded. Scan all typed payloads and pick one whose page window
    // contains those identity fields, as the comment above promises, rather
    // than making correctness depend on TASK_STORAGE iteration order.
    let typed_payloads: Vec<&serde_json::Value> = payloads
        .iter()
        .copied()
        .filter(|payload| {
            payload.get("kind").and_then(|kind| kind.as_str()) == Some("struct")
                && payload.get("type_name").and_then(|name| name.as_str())
                    == Some("ktstr_arena_ctx")
        })
        .collect();
    if typed_payloads.is_empty() {
        let kinds: Vec<String> = payloads
            .iter()
            .take(12)
            .map(|payload| {
                let kind = payload
                    .get("kind")
                    .and_then(|kind| kind.as_str())
                    .unwrap_or("<no-kind>");
                let name = payload
                    .get("type_name")
                    .and_then(|name| name.as_str())
                    .unwrap_or("<no-name>");
                format!("{kind}/{name}")
            })
            .collect();
        anyhow::bail!(
            "no payload rendered as Struct(type_name=\"ktstr_arena_ctx\"); \
             non_null_payloads={}; first_kinds/type_names={kinds:?}; \
             {dump_artifact}",
            payloads.len(),
        );
    }

    let mut ptr_count = 0usize;
    let mut nonzero_ptr_count = 0usize;
    let mut deref_count = 0usize;
    let mut task_struct_count = 0usize;
    let mut min_had: Option<u64> = None;
    let mut max_had: Option<u64> = None;
    let mut min_truncation_depth: Option<usize> = None;
    let mut max_truncation_depth: Option<usize> = None;
    let mut selected_payload = None;
    for payload in &typed_payloads {
        let Ok(task_kptr) = struct_member(payload, "task_kptr", &dump_artifact) else {
            continue;
        };
        if task_kptr.get("kind").and_then(|kind| kind.as_str()) != Some("ptr") {
            continue;
        }
        ptr_count += 1;
        if !matches!(
            task_kptr.get("value").and_then(|value| value.as_u64()),
            Some(value) if value != 0
        ) || task_kptr
            .get("deref_skipped_reason")
            .and_then(|reason| reason.as_str())
            .is_some()
        {
            continue;
        }
        nonzero_ptr_count += 1;
        let Some(deref) = task_kptr.get("deref") else {
            continue;
        };
        deref_count += 1;
        if let Some(had) = deref.get("had").and_then(|had| had.as_u64()) {
            min_had = Some(min_had.map_or(had, |current| current.min(had)));
            max_had = Some(max_had.map_or(had, |current| current.max(had)));
        }
        let Ok((deref_struct, truncation_depth)) = peel_truncated(deref, &dump_artifact) else {
            continue;
        };
        min_truncation_depth = Some(
            min_truncation_depth.map_or(truncation_depth, |current| current.min(truncation_depth)),
        );
        max_truncation_depth = Some(
            max_truncation_depth.map_or(truncation_depth, |current| current.max(truncation_depth)),
        );
        if deref_struct.get("kind").and_then(|kind| kind.as_str()) != Some("struct")
            || deref_struct.get("type_name").and_then(|name| name.as_str()) != Some("task_struct")
        {
            continue;
        }
        task_struct_count += 1;
        let Some(members) = deref_struct
            .get("members")
            .and_then(|members| members.as_array())
        else {
            continue;
        };
        let has_pid = members
            .iter()
            .any(|member| member.get("name").and_then(|name| name.as_str()) == Some("pid"));
        let has_comm = members
            .iter()
            .any(|member| member.get("name").and_then(|name| name.as_str()) == Some("comm"));
        if has_pid && has_comm {
            selected_payload = Some(*payload);
            break;
        }
    }
    let payload = selected_payload.ok_or_else(|| {
        anyhow::anyhow!(
            "no ktstr_arena_ctx payload exposed a cast-chased task_struct \
             containing both `pid` and `comm`; typed_payloads={}; ptrs={ptr_count}; \
             nonzero_ptrs_without_skip={nonzero_ptr_count}; derefs={deref_count}; \
             task_structs={task_struct_count}; outer_read_had={min_had:?}..={max_had:?}; \
             truncation_depth={min_truncation_depth:?}..={max_truncation_depth:?}; \
             {dump_artifact}",
            typed_payloads.len(),
        )
    })?;

    // Negative assertion #1: `magic` must render as plain Uint, NOT
    // as a Ptr. The cast analyzer must NOT have flagged offset 0 of
    // ktstr_arena_ctx — the BPF code only loads magic for printing,
    // never as a pointer base, and never stores a typed pointer
    // into it.
    let magic = struct_member(payload, "magic", &dump_artifact)?;
    let magic_kind = magic
        .get("kind")
        .and_then(|k| k.as_str())
        .unwrap_or("<no-kind>");
    if magic_kind != "uint" {
        anyhow::bail!(
            "NEGATIVE ASSERTION FAILED: `magic` must render as a plain Uint \
             (BPF code only stores the immediate sentinel into it, never a \
             typed pointer; the analyzer must not flag this field), but got \
             kind={magic_kind:?}. cast intercept fired falsely on offset 0. \
             {dump_artifact}"
        );
    }
    // The magic value comes from BPF code that writes
    // `KTSTR_ARENA_MAGIC = 0xDEADBEEFCAFEBABE`. A correct render is
    // a u64 with that value; anything else means the bytes either
    // weren't captured or got rewritten. Mirror the same constant
    // the existing tests/failure_dump_e2e.rs check uses so a future
    // change to the BPF magic only needs one update site.
    const KTSTR_ARENA_MAGIC: u64 = 0xDEADBEEFCAFEBABE;
    let magic_value = magic
        .get("value")
        .and_then(|v| v.as_u64())
        .ok_or_else(|| anyhow::anyhow!("`magic` value not a u64; {dump_artifact}"))?;
    if magic_value != KTSTR_ARENA_MAGIC {
        anyhow::bail!(
            "`magic` value mismatch: got 0x{magic_value:016x}, expected \
             0x{KTSTR_ARENA_MAGIC:016x}; {dump_artifact}"
        );
    }

    // Negative assertion #2: `counter` is a u32 — the cast intercept's
    // `int.size() != 8` gate must reject it. The render kind is
    // `uint` with bits=32.
    let counter = struct_member(payload, "counter", &dump_artifact)?;
    let counter_kind = counter
        .get("kind")
        .and_then(|k| k.as_str())
        .unwrap_or("<no-kind>");
    if counter_kind != "uint" {
        anyhow::bail!(
            "NEGATIVE ASSERTION FAILED: `counter` (u32) must render as Uint; \
             the cast intercept's size==8 gate must reject sub-u64 fields. \
             Got kind={counter_kind:?}; {dump_artifact}"
        );
    }
    let counter_bits = counter.get("bits").and_then(|b| b.as_u64()).unwrap_or(0);
    if counter_bits != 32 {
        anyhow::bail!(
            "`counter` bits mismatch: got {counter_bits}, expected 32; \
             {dump_artifact}"
        );
    }
    let counter_value = counter
        .get("value")
        .and_then(|v| v.as_u64())
        .ok_or_else(|| anyhow::anyhow!("`counter` value not numeric; {dump_artifact}"))?;
    // `KTSTR_TASK_COUNTER = 42` is what `ktstr_init_task` stamps.
    const KTSTR_TASK_COUNTER: u64 = 42;
    if counter_value != KTSTR_TASK_COUNTER {
        anyhow::bail!(
            "`counter` value mismatch: got {counter_value}, expected \
             {KTSTR_TASK_COUNTER}; the BPF code's `taskc->counter = \
             KTSTR_TASK_COUNTER` write did not land or the captured page is \
             stale; {dump_artifact}"
        );
    }

    // Positive assertion: `task_kptr` MUST render as a Ptr. This is
    // the user-facing bar — the cast analysis pipeline turned a u64
    // field into a typed pointer that the renderer chased to its
    // target struct.
    let task_kptr = struct_member(payload, "task_kptr", &dump_artifact)?;
    let task_kptr_kind = task_kptr
        .get("kind")
        .and_then(|k| k.as_str())
        .unwrap_or("<no-kind>");
    if task_kptr_kind != "ptr" {
        anyhow::bail!(
            "PRIMARY POSITIVE ASSERTION FAILED: `task_kptr` (u64 holding a \
             kernel task_struct *) must render as a Ptr after cast analysis \
             rewrites it. Got kind={task_kptr_kind:?}. \
             Failure modes: \
             (a) cast_analysis_load did not produce a CastMap entry for \
             (ktstr_arena_ctx, off=16) — the analyzer's STX detection did \
             not fire on `ktstr_stash_task_kptr`'s body. \
             (b) the freeze coordinator did not thread the CastMap into \
             the dump's MemReader. \
             (c) `MemReader::cast_lookup` did not return Some for the \
             (parent, offset) the renderer asked. \
             (d) `render_cast_pointer` bailed before emitting Ptr. \
             {dump_artifact}"
        );
    }

    // The pointer value MUST be non-zero — `ktstr_init_task` writes
    // the live `task_struct *p` parameter, which is non-null on
    // every entry.  A zero value here would mean the helper never
    // wrote, the page was captured before the write landed, or the
    // wrong arena slot got rendered. Surface that failure mode
    // distinctly from the chase-failure path below.
    let task_kptr_value = task_kptr
        .get("value")
        .and_then(|v| v.as_u64())
        .ok_or_else(|| anyhow::anyhow!("`task_kptr` Ptr has no `value` field; {dump_artifact}"))?;
    if task_kptr_value == 0 {
        anyhow::bail!(
            "`task_kptr` value is 0x0 — `ktstr_stash_task_kptr` never wrote \
             a live task_struct pointer for this entry, or the captured \
             arena page predates the write. The render correctly identified \
             the field as a Ptr (cast analysis pipeline OK), but the source \
             data is zero; {dump_artifact}"
        );
    }

    // The chase MUST have succeeded — `deref` is `Some(...)`. A
    // null `deref` with `deref_skipped_reason` populated would mean
    // the kernel kva read failed (unmapped page, plausibility gate
    // tripped). Surface the reason so the failure mode is
    // identifiable.
    if let Some(reason) = task_kptr
        .get("deref_skipped_reason")
        .and_then(|r| r.as_str())
    {
        anyhow::bail!(
            "`task_kptr` chase was attempted but did not complete: \
             deref_skipped_reason={reason:?}. The cast analysis flagged the \
             field correctly, but the renderer could not read the target \
             struct. Likely causes: read_kva failed (target page unmapped), \
             plausibility gate rejected the first qword as a freelist \
             pointer, or the BTF size of task_struct exceeded \
             POINTER_CHASE_CAP. task_kptr value: 0x{task_kptr_value:x}"
        );
    }
    let deref = task_kptr.get("deref").ok_or_else(|| {
        anyhow::anyhow!(
            "`task_kptr` Ptr has no `deref` AND no `deref_skipped_reason` — \
             the chase was either not attempted (depth cap, cycle, null \
             value), or the JSON shape changed. task_kptr value: \
             0x{task_kptr_value:x}; {dump_artifact}"
        )
    })?;

    // The dereffed value must be a Struct whose type_name is the
    // kernel `task_struct`. `render_cast_pointer`'s kernel arm
    // calls `render_value_inner(target_type_id=task_struct)`, which
    // walks the struct and surfaces its members. The exact set of
    // members visible depends on POINTER_CHASE_CAP truncating the
    // read; modern task_struct is far larger than 4 KiB, so we
    // expect one or more Truncated{partial: ...} wrappers around a
    // Struct, or a Struct directly. The outer pointer chase and the
    // inner struct renderer each record their own clipping boundary.
    let (deref_struct, truncation_depth) = peel_truncated(deref, &dump_artifact)?;
    let was_truncated = truncation_depth > 0;
    let deref_kind = deref_struct
        .get("kind")
        .and_then(|k| k.as_str())
        .unwrap_or("<no-kind>");
    if deref_kind != "struct" {
        anyhow::bail!(
            "task_kptr deref's inner kind must be struct (post-truncation), \
             got {deref_kind:?}; {dump_artifact}"
        );
    }
    let deref_type_name = deref_struct
        .get("type_name")
        .and_then(|n| n.as_str())
        .ok_or_else(|| {
            anyhow::anyhow!(
                "deref struct has no type_name (anonymous struct?); \
                 {dump_artifact}"
            )
        })?;
    if deref_type_name != "task_struct" {
        anyhow::bail!(
            "deref type_name mismatch: got {deref_type_name:?}, expected \
             \"task_struct\"; the cast analyzer flagged the wrong target. \
             {dump_artifact}"
        );
    }

    // Strong content assertion: the rendered task_struct MUST
    // contain identifying fields any kernel observer expects.
    // `pid` (i32) and `comm` (char[16]) are stable members defined
    // in include/linux/sched.h that the BTF in any debug-info
    // kernel surfaces. Their presence proves the BTF Datasec walk
    // descended into task_struct and rendered real members — not a
    // garbage byte slice.
    let members = deref_struct
        .get("members")
        .and_then(|m| m.as_array())
        .ok_or_else(|| anyhow::anyhow!("deref task_struct has no `members`; {dump_artifact}"))?;
    let names: std::collections::HashSet<&str> = members
        .iter()
        .filter_map(|m| m.get("name").and_then(|n| n.as_str()))
        .collect();
    let required: &[&str] = &["pid", "comm"];
    for r in required {
        if !names.contains(r) {
            anyhow::bail!(
                "task_struct deref missing required member `{r}` — the \
                 cast chase produced a struct render but the BTF Datasec \
                 walk did not surface real task_struct fields, or the \
                 read returned bytes shorter than the field offset. \
                 Got members: {names:?}; {dump_artifact}"
            );
        }
    }

    // Pid must be a non-negative integer that proves we read a real
    // task. The captured task is the running task at freeze, so pid
    // is whatever was scheduled — but it MUST be parseable.
    let pid = struct_member(deref_struct, "pid", &dump_artifact)?;
    let pid_kind = pid
        .get("kind")
        .and_then(|k| k.as_str())
        .unwrap_or("<no-kind>");
    if pid_kind != "int" && pid_kind != "uint" {
        anyhow::bail!(
            "task_struct.pid must render as int/uint, got kind={pid_kind:?}; \
             {dump_artifact}"
        );
    }
    let pid_value = pid
        .get("value")
        .ok_or_else(|| anyhow::anyhow!("pid has no `value`; {dump_artifact}"))?;
    let pid_int = pid_value
        .as_i64()
        .or_else(|| pid_value.as_u64().map(|u| u as i64));
    if pid_int.is_none() {
        anyhow::bail!("pid value not numeric; {dump_artifact}");
    }

    // comm should be either a Bytes hex (the renderer's char[]
    // path) or a Struct-like rendering. Just confirm it exists with
    // a value field — the structure shape varies by BTF rendering
    // mode and that's fine for this assertion.
    let comm = struct_member(deref_struct, "comm", &dump_artifact)?;
    let comm_kind = comm
        .get("kind")
        .and_then(|k| k.as_str())
        .unwrap_or("<no-kind>");
    if comm_kind == "unsupported" {
        anyhow::bail!(
            "task_struct.comm rendered as Unsupported — the BTF Datasec walk \
             could not handle the field type; {dump_artifact}"
        );
    }

    eprintln!(
        "cast analysis pipeline E2E: dump carries scx_task_map \
         with {} entries, {} non-null payloads. Located ktstr_arena_ctx \
         render with cast-chased task_kptr=0x{task_kptr_value:x} → \
         {}{deref_type_name}{{pid={pid_int:?}, comm.kind={comm_kind:?}, \
         member count={}}}; magic=0x{magic_value:016x} (Uint, not chased), \
         counter={counter_value} (Uint, not chased)",
        entries.len(),
        payloads.len(),
        if was_truncated { "truncated " } else { "" },
        members.len(),
    );

    Ok(())
}

#[ktstr::ktstr_test_entry]
static __KTSTR_ENTRY_CAST_ANALYSIS_KERNEL_KPTR: ktstr::test_support::KtstrTestEntry =
    ktstr::test_support::KtstrTestEntry {
        name: "cast_analysis_chases_kernel_kptr",
        func: scenario_cast_analysis_chases_kernel_kptr,
        scheduler: &KTSTR_SCHED,
        // `--stall-after=1` triggers SCX_EXIT_ERROR_STALL via the
        // kernel watchdog, which fires the freeze coordinator's
        // dump_state path. Same mechanism the existing
        // `failure_dump_e2e.rs` scenarios use.
        extra_sched_args: &["--stall-after=1"],
        watchdog_timeout: std::time::Duration::from_secs(3),
        duration: std::time::Duration::from_secs(10),
        // The scheduler intentionally dies (SCX_EXIT_ERROR_STALL);
        // `expect_err: true` inverts that to PASS. The chase assertions
        // run in check_cast_analysis_chases_kernel_kptr, a
        // post_vm_unconditional callback whose Err is a hard FAIL via
        // PostVmAssertionFailure — `expect_err` does NOT invert it, so a
        // real chase regression fails the test.
        expect_err: true,
        post_vm_unconditional: Some(check_cast_analysis_chases_kernel_kptr),
        ..ktstr::test_support::KtstrTestEntry::DEFAULT
    };

/// Find the `sdt_allocations` entry whose slot range contains a chased
/// arena pointer, returning its rendered `payload`. Used by the
/// bss-to-arena check when the renderer's dedup ("already rendered in
/// sdt_allocations") skips the inline chase: the typed payload is then in
/// `dump.sdt_allocations` instead. `user_addr` (sdt_alloc/snapshot.rs) is
/// the slot start windowed to the low 32 bits (`data_ptr & 0xFFFF_FFFF`);
/// the chased `arena_value` is the payload pointer (slot_start + header),
/// so this RANGE-matches `[user_addr, user_addr + elem_size)` rather than
/// testing equality. `elem_size` is the slot size; slots do not overlap,
/// so the match is unique.
fn find_sdt_allocation_by_addr(
    dump: &serde_json::Value,
    arena_value: u64,
) -> Option<&serde_json::Value> {
    let needle = arena_value & 0xFFFF_FFFF;
    let allocations = dump.get("sdt_allocations")?.as_array()?;
    for snapshot in allocations {
        let Some(elem_size) = snapshot.get("elem_size").and_then(|s| s.as_u64()) else {
            continue;
        };
        let Some(entries) = snapshot.get("entries").and_then(|e| e.as_array()) else {
            continue;
        };
        for entry in entries {
            let Some(user_addr) = entry.get("user_addr").and_then(|a| a.as_u64()) else {
                continue;
            };
            if user_addr <= needle && needle < user_addr.saturating_add(elem_size) {
                return entry.get("payload");
            }
        }
    }
    None
}

/// Locate the scheduler's `.bss` map inside the dump. Same name-suffix
/// rule the existing `failure_dump_e2e.rs::scenario_failure_dump_renders_bss_fields`
/// uses: libbpf composes `<obj>.bss` for the global-section map, scx-ktstr
/// builds with object name `bpf` (per `scx-ktstr/build.rs::enable_skel`), so
/// the map's `name` ends with `.bss` and is NOT one of the framework probe
/// maps (filtered with the `probe_bp.` / `fentry_p.` prefix exclusions).
fn find_scheduler_bss_map<'a>(
    dump: &'a serde_json::Value,
    dump_artifact: &str,
) -> Result<&'a serde_json::Value> {
    let maps = dump
        .get("maps")
        .and_then(|m| m.as_array())
        .ok_or_else(|| anyhow::anyhow!("dump missing top-level `maps` array"))?;
    maps.iter()
        .find(|m| {
            m.get("name")
                .and_then(|n| n.as_str())
                .map(|n| {
                    n.ends_with(".bss")
                        && !n.starts_with("probe_bp.")
                        && !n.starts_with("fentry_p.")
                })
                .unwrap_or(false)
        })
        .ok_or_else(|| {
            anyhow::anyhow!(
                "dump has no scheduler `.bss` map (looked across {} maps); the \
                 scx-ktstr BPF program must surface a `.bss` global section. \
                 {dump_artifact}",
                maps.len()
            )
        })
}

/// Asserts that scx-ktstr's `.bss`-resident `ktstr_bss_arena_holder`
/// struct renders with its `arena_target` u64 field rewritten by the
/// cast analysis pipeline into a typed Ptr that chases through to the
/// target `ktstr_arena_ctx` arena allocation.
///
/// Path under test (BSS source -> arena target):
///   - `ktstr_train_bss_to_arena` (in scx-ktstr/src/bpf/main.bpf.c) is
///     a `__noinline` BPF-to-BPF helper whose `.BTF.ext` `func_info`
///     entry seeds R1 with `Pointer{ktstr_bss_arena_holder}` at
///     function entry.
///   - The helper body loads `holder->arena_target` (LDX from R1+0)
///     into R2, marking R2 as `LoadedU64Field { source: ktstr_bss_arena_holder,
///     offset: 0 }`. The subsequent addr_space_cast (lowered from the
///     `(struct ktstr_arena_ctx __arena *)(unsigned long)raw` idiom)
///     marks the field arena_confirmed and propagates the
///     LoadedU64Field state.
///   - Three subsequent LDX accesses through the cast result record
///     access pattern entries `{(0, 8), (8, 4), (16, 8)}` under the
///     source key `(ktstr_bss_arena_holder, 0)`. After the forward
///     walk, `Analyzer::finalize` intersects these against the program
///     BTF; only `ktstr_arena_ctx` matches all three offsets with the
///     declared widths, so the resulting cast finding is
///     `(ktstr_bss_arena_holder, 0) -> (ktstr_arena_ctx, AddrSpace::Arena)`.
///   - `ktstr_init_task` writes the freshly-allocated `taskc` user-side
///     arena VA into `ktstr_bss_arena_holder.arena_target` so the
///     captured `.bss` page carries a non-zero pointer at dump time.
///   - The dump renderer walks the `.bss` Datasec, descends into the
///     `ktstr_bss_arena_holder` struct, and for the `arena_target`
///     member calls `MemReader::cast_lookup`. The hit fires
///     `render_cast_pointer`'s arena arm, which reads the captured
///     arena bytes via `MemReader::read_arena` and recursively renders
///     `ktstr_arena_ctx` against them. The chased struct's members
///     surface `magic = KTSTR_ARENA_MAGIC` and `counter = KTSTR_TASK_COUNTER`
///     so the assertions below confirm both the chase mechanics and the
///     captured byte content.
///
/// Negative assertion: the sibling `bss_plain_counter` u64 must NOT
/// render as a Ptr -- the BPF code only `__sync_fetch_and_add`s into
/// it, never dereferences it as a pointer base. The cast analyzer must
/// not flag the offset, mirroring the `magic`/`counter` negative-control
/// pattern from the existing kernel-kptr scenario.
fn scenario_cast_analysis_chases_bss_to_arena(ctx: &ktstr::scenario::Ctx) -> Result<AssertResult> {
    await_accessor_ready();
    let steps = vec![Step {
        setup: vec![ctx.cgroup_def("cg_0")].into(),
        ops: vec![],
        hold: HoldSpec::FULL,
    }];
    execute_steps(ctx, steps)
}

/// Host-side post_vm assertion for `cast_analysis_chases_bss_to_arena`.
/// The guest cannot read the host-written dump, so the BSS→arena chase
/// assertions run here; the callback's Err is a hard FAIL
/// (`PostVmAssertionFailure`) that `expect_err` does not invert.
fn check_cast_analysis_chases_bss_to_arena(result: &VmResult) -> Result<()> {
    let dump = read_failure_dump(result)?;
    let dump_artifact = failure_dump_artifact(result);

    let bss_map = find_scheduler_bss_map(&dump, &dump_artifact)?;
    // The .bss map's `value` is the BTF-rendered Datasec (libbpf
    // exposes the entire .bss as a single Datasec type). Its members
    // enumerate every global declared in main.bpf.c.
    let bss_value = bss_map
        .get("value")
        .ok_or_else(|| anyhow::anyhow!(".bss map has no `value` field; {dump_artifact}"))?;
    let bss_kind = bss_value
        .get("kind")
        .and_then(|k| k.as_str())
        .unwrap_or("<no-kind>");
    if bss_kind != "struct" {
        anyhow::bail!(
            ".bss value must render as a Struct (the renderer maps Datasec to Struct \
             with type_name set to the section name), got kind={bss_kind:?}; \
             {dump_artifact}"
        );
    }

    // Locate the `ktstr_bss_arena_holder` member inside the .bss
    // Datasec. The Datasec member's `name` is the global variable
    // name; the inner `value` carries the BTF-rendered struct.
    let holder_outer =
        struct_member(bss_value, "ktstr_bss_arena_holder", &dump_artifact).map_err(|e| {
            anyhow::anyhow!(
                "{e}\n\nNo `ktstr_bss_arena_holder` Var in .bss -- either the \
                 BSS test fixture in scx-ktstr/src/bpf/main.bpf.c was renamed, \
                 the BTF Datasec walker filtered it, or the global was elided \
                 by the BPF compiler because no in-program writer kept it live; \
                 {dump_artifact}"
            )
        })?;
    let holder_kind = holder_outer
        .get("kind")
        .and_then(|k| k.as_str())
        .unwrap_or("<no-kind>");
    if holder_kind != "struct" {
        anyhow::bail!(
            "`ktstr_bss_arena_holder` must render as a Struct (it's declared as \
             `struct ktstr_bss_arena_holder` in BPF source), got kind={holder_kind:?}: \
             {dump_artifact}"
        );
    }
    // Sanity-check the type_name lines up so a BTF rename surfaces
    // here rather than silently passing through. A missing type_name
    // is allowed (anonymous structs can occur), but a wrong name
    // means the renderer descended into the wrong Var.
    if let Some(name) = holder_outer.get("type_name").and_then(|n| n.as_str())
        && name != "ktstr_bss_arena_holder"
    {
        anyhow::bail!(
            "ktstr_bss_arena_holder rendered with unexpected type_name={name:?}; \
             {dump_artifact}"
        );
    }

    // PRIMARY POSITIVE ASSERTION: `arena_target` MUST render as Ptr.
    // The cast analysis pipeline turned the u64 field into a typed
    // pointer that the renderer chased to its arena target.
    let arena_target = struct_member(holder_outer, "arena_target", &dump_artifact)?;
    let arena_kind = arena_target
        .get("kind")
        .and_then(|k| k.as_str())
        .unwrap_or("<no-kind>");
    if arena_kind != "ptr" {
        anyhow::bail!(
            "PRIMARY POSITIVE ASSERTION FAILED: `arena_target` (BSS u64 holding \
             an arena VA) must render as a Ptr after cast analysis rewrites it. \
             Got kind={arena_kind:?}. \
             Failure modes: \
             (a) cast_analysis_load did not produce a CastMap entry for \
             (ktstr_bss_arena_holder, off=0) -- the analyzer's LDX-side detection \
             did not fire on `ktstr_train_bss_to_arena`'s body (FuncProto seeding \
             missing, addr_space_cast not recognized, or the access pattern \
             intersected non-uniquely against the program BTF and dropped). \
             (b) the freeze coordinator did not thread the CastMap into \
             the dump's MemReader. \
             (c) `MemReader::cast_lookup` did not return Some for \
             (ktstr_bss_arena_holder, 0). \
             (d) `render_cast_pointer` bailed before emitting Ptr. \
             {dump_artifact}"
        );
    }

    // The pointer value MUST be non-zero -- `ktstr_init_task` writes
    // the live arena VA every time it runs, so by the time the
    // freeze fires there must be at least one task that ran through
    // init_task and stamped the global. A zero value would mean the
    // write never happened OR the captured page predates every
    // init_task invocation.
    let arena_value = arena_target
        .get("value")
        .and_then(|v| v.as_u64())
        .ok_or_else(|| anyhow::anyhow!("`arena_target` Ptr has no `value`; {dump_artifact}"))?;
    if arena_value == 0 {
        anyhow::bail!(
            "`arena_target` value is 0x0 -- `ktstr_init_task`'s write to \
             `ktstr_bss_arena_holder.arena_target` never landed, or the captured \
             .bss page predates every init_task invocation. The render correctly \
             flagged the field (cast pipeline OK), but the source data is zero. \
             {dump_artifact}"
        );
    }

    // Branch on the chase outcome (three cases):
    //   (a) `deref` present, no skip reason -> the chase inlined the
    //       target struct; verify it directly.
    //   (b) skip reason == "already rendered in sdt_allocations" -> the
    //       renderer's intentional dedup fired (chase_arena_pointer in
    //       btf_render/mod.rs): the chased slot is also a typed entry in
    //       `dump.sdt_allocations`, so the renderer declined to render it
    //       twice. This is correct production behavior, not a failure --
    //       the same typed payload is in the dump; verify it THERE.
    //       Whether the dedup fires is real arena state: it turns on
    //       whether the task whose taskc sits in `arena_target` is still
    //       live (its sdt_alloc slot enumerated by the freeze-time
    //       pre-pass) at the dump moment, so the test must accept both
    //       outcomes to be deterministic.
    //   (c) any other skip reason -> a real chase failure; FAIL.
    let skipped_reason = arena_target
        .get("deref_skipped_reason")
        .and_then(|r| r.as_str());
    let (deref_struct, was_truncated, verify_source): (&serde_json::Value, bool, &str) =
        match skipped_reason {
            None => {
                let deref = arena_target.get("deref").ok_or_else(|| {
                    anyhow::anyhow!(
                        "`arena_target` Ptr has no `deref` AND no \
                         `deref_skipped_reason` -- the chase was either not \
                         attempted (depth cap, cycle, null value), or the JSON \
                         shape changed. arena_target value: 0x{arena_value:x}; \
                         {dump_artifact}"
                    )
                })?;
                // ktstr_arena_ctx is 24 bytes so no Truncated wrap is
                // expected, but accept it for forward compatibility if the
                // struct ever exceeds POINTER_CHASE_CAP.
                match deref.get("kind").and_then(|k| k.as_str()) {
                    Some("struct") => (deref, false, "inline deref"),
                    Some("truncated") => (
                        deref.get("partial").ok_or_else(|| {
                            anyhow::anyhow!("Truncated has no `partial`; {dump_artifact}")
                        })?,
                        true,
                        "inline deref (truncated)",
                    ),
                    Some(other) => anyhow::bail!(
                        "`arena_target` deref must be Struct or \
                         Truncated{{partial: Struct}}, got kind={other:?}; \
                         {dump_artifact}"
                    ),
                    None => {
                        anyhow::bail!("`arena_target` deref has no `kind` field; {dump_artifact}")
                    }
                }
            }
            // Must match the dedup reason string emitted at
            // btf_render/mod.rs (chase_arena_pointer); a drift falls to
            // the `Some(other)` arm and fails loudly.
            Some("already rendered in sdt_allocations") => {
                // The typed payload is in `dump.sdt_allocations`. Match the
                // chased pointer to its slot by the LOW 32 BITS (sdt_alloc
                // `user_addr` is `data_ptr & 0xFFFF_FFFF`); range-match
                // because `arena_target` holds the payload pointer
                // (slot_start + header) while `user_addr` is the slot start.
                let payload = find_sdt_allocation_by_addr(&dump, arena_value).ok_or_else(|| {
                    anyhow::anyhow!(
                        "`arena_target` chase was dedup-skipped (\"already rendered \
                         in sdt_allocations\") but no sdt_allocations slot's \
                         [user_addr, user_addr+elem_size) range contains the low 32 \
                         bits of arena_value=0x{arena_value:x} (0x{:x}). The \
                         renderer's dedup index claims the slot is rendered there, \
                         but the dump's sdt_allocations disagree.",
                        arena_value & 0xFFFF_FFFF
                    )
                })?;
                (payload, false, "sdt_allocations (dedup)")
            }
            Some(other) => anyhow::bail!(
                "`arena_target` chase did not complete: deref_skipped_reason=\
                 {other:?}. The cast analysis flagged the field correctly, but the \
                 renderer could not read the target struct. Likely causes: \
                 read_arena returned None (page outside captured snapshot), \
                 `is_arena_addr` rejected the value (the BSS write put a non-arena \
                 address into the field), or the BTF size of ktstr_arena_ctx was \
                 unresolvable. arena_target value: 0x{arena_value:x}"
            ),
        };
    let deref_kind_inner = deref_struct
        .get("kind")
        .and_then(|k| k.as_str())
        .unwrap_or("<no-kind>");
    if deref_kind_inner != "struct" {
        anyhow::bail!(
            "arena_target deref's inner kind must be struct (post-truncation), \
             got {deref_kind_inner:?}; {dump_artifact}"
        );
    }
    let deref_type_name = deref_struct
        .get("type_name")
        .and_then(|n| n.as_str())
        .ok_or_else(|| {
            anyhow::anyhow!(
                "deref struct has no type_name (anonymous struct?); \
                 {dump_artifact}"
            )
        })?;
    if deref_type_name != "ktstr_arena_ctx" {
        anyhow::bail!(
            "deref type_name mismatch: got {deref_type_name:?}, expected \
             \"ktstr_arena_ctx\"; the cast analyzer flagged the wrong target. \
             This is the correctness bar -- a wrong target struct means the \
             access-pattern intersection picked a same-shape decoy out of the \
             program BTF; {dump_artifact}"
        );
    }

    // STRONG CONTENT ASSERTION: the chased `ktstr_arena_ctx`'s `magic`
    // member must equal `KTSTR_ARENA_MAGIC`. This proves the chase
    // landed on a real ktstr_arena_ctx allocation (not a same-shape
    // garbage page) AND that the renderer descended into the chased
    // struct's bytes correctly. Mirrors the `KTSTR_ARENA_MAGIC` check
    // the existing kernel-kptr scenario uses on the per-task arena
    // payload, so a future change to the BPF magic only needs one
    // update site (this constant block).
    const KTSTR_ARENA_MAGIC: u64 = 0xDEADBEEFCAFEBABE;
    const KTSTR_TASK_COUNTER: u64 = 42;
    let magic = struct_member(deref_struct, "magic", &dump_artifact)?;
    let magic_kind = magic
        .get("kind")
        .and_then(|k| k.as_str())
        .unwrap_or("<no-kind>");
    if magic_kind != "uint" {
        anyhow::bail!(
            "chased `ktstr_arena_ctx.magic` must render as Uint (the analyzer \
             must NOT recurse into magic -- it's only loaded as a sentinel, \
             never as a pointer base), got kind={magic_kind:?}; {dump_artifact}"
        );
    }
    let magic_value = magic
        .get("value")
        .and_then(|v| v.as_u64())
        .ok_or_else(|| anyhow::anyhow!("magic value not a u64; {dump_artifact}"))?;
    if magic_value != KTSTR_ARENA_MAGIC {
        anyhow::bail!(
            "chased `ktstr_arena_ctx.magic` mismatch: got 0x{magic_value:016x}, \
             expected 0x{KTSTR_ARENA_MAGIC:016x}. The cast chase completed but \
             landed on bytes whose first qword is not the alloc-time sentinel. \
             Either the captured arena page is stale, the user_addr in \
             `arena_target` does not point at a current allocation, or a \
             same-shape decoy struct in the program BTF won the access-pattern \
             intersection; {dump_artifact}"
        );
    }

    let counter = struct_member(deref_struct, "counter", &dump_artifact)?;
    let counter_value = counter
        .get("value")
        .and_then(|v| v.as_u64())
        .ok_or_else(|| anyhow::anyhow!("counter value not numeric; {dump_artifact}"))?;
    if counter_value != KTSTR_TASK_COUNTER {
        anyhow::bail!(
            "chased `ktstr_arena_ctx.counter` mismatch: got {counter_value}, \
             expected {KTSTR_TASK_COUNTER}. The cast chase landed on the right \
             struct shape but the captured bytes do not carry the alloc-time \
             value, indicating a stale arena page; {dump_artifact}"
        );
    }

    // NEGATIVE ASSERTION: `bss_plain_counter` (the sibling u64 in the
    // same .bss struct) must NOT render as a Ptr. The BPF code only
    // increments it via `__sync_fetch_and_add`; the analyzer must not
    // flag offset 8 of ktstr_bss_arena_holder as a typed pointer.
    let plain_counter = struct_member(holder_outer, "bss_plain_counter", &dump_artifact)?;
    let plain_kind = plain_counter
        .get("kind")
        .and_then(|k| k.as_str())
        .unwrap_or("<no-kind>");
    if plain_kind != "uint" {
        anyhow::bail!(
            "NEGATIVE ASSERTION FAILED: `bss_plain_counter` (a u64 counter \
             never used as a pointer base) must render as a plain Uint. The \
             cast intercept fired falsely on offset 8 of \
             ktstr_bss_arena_holder. Got kind={plain_kind:?}. \
             {dump_artifact}"
        );
    }
    let plain_value = plain_counter
        .get("value")
        .and_then(|v| v.as_u64())
        .ok_or_else(|| anyhow::anyhow!("plain counter value not numeric; {dump_artifact}"))?;
    if plain_value == 0 {
        anyhow::bail!(
            "`bss_plain_counter` is 0 -- `ktstr_init_task` never executed the \
             increment, which means the test fixture in main.bpf.c did not \
             run before the freeze; {dump_artifact}"
        );
    }

    eprintln!(
        "BSS->arena cast pipeline E2E ({verify_source}): dump carries `.bss` map \
         with ktstr_bss_arena_holder render where arena_target=0x{arena_value:x} -> \
         {}{deref_type_name}{{magic=0x{magic_value:016x}, counter={counter_value}}}; \
         bss_plain_counter={plain_value} (Uint, not chased -- negative control)",
        if was_truncated { "truncated " } else { "" },
    );

    Ok(())
}

#[ktstr::ktstr_test_entry]
static __KTSTR_ENTRY_CAST_ANALYSIS_BSS_TO_ARENA: ktstr::test_support::KtstrTestEntry =
    ktstr::test_support::KtstrTestEntry {
        name: "cast_analysis_chases_bss_to_arena",
        func: scenario_cast_analysis_chases_bss_to_arena,
        scheduler: &KTSTR_SCHED,
        // Same trigger as the kernel-kptr scenario: SCX_EXIT_ERROR_STALL
        // fires the freeze coordinator's dump_state path. The .bss-side
        // fixture is exercised inside ktstr_init_task on every task
        // ktstr scheduler initializes, so by the time the watchdog
        // fires the .bss global has been written and the trainer has
        // been called.
        extra_sched_args: &["--stall-after=1"],
        watchdog_timeout: std::time::Duration::from_secs(3),
        duration: std::time::Duration::from_secs(10),
        // Matches the kernel-kptr scenario: the scheduler intentionally
        // dies via SCX_EXIT_ERROR_STALL; `expect_err: true` inverts that
        // to PASS. The BSS→arena chase assertions run in
        // check_cast_analysis_chases_bss_to_arena, a post_vm_unconditional
        // callback whose Err is a hard FAIL via PostVmAssertionFailure —
        // `expect_err` does NOT invert it.
        expect_err: true,
        post_vm_unconditional: Some(check_cast_analysis_chases_bss_to_arena),
        ..ktstr::test_support::KtstrTestEntry::DEFAULT
    };

/// Asserts the sdt_alloc bridge resolves the Fwd-target chase on
/// `scx_task_map_val.data` end-to-end. The dump pipeline produces
/// per-entry `value` (the BTF-rendered surface struct) and
/// `payload` (the chased per-task allocator payload). Both depend
/// on the bridge:
///
///   - `scx_task_map_val.data` is `struct sdt_data __arena *` whose
///     pointee is a `BTF_KIND_FWD` in scx-ktstr's program BTF
///     (`struct sdt_data`'s body lives in the sdt_alloc library
///     BTF, not the program BTF). Without the bridge, the renderer
///     would skip the chase with
///     `deref_skipped_reason="… forward declaration; body not in
///     this BTF"` and never recover the per-task struct content.
///     With the bridge, [`MemReader::resolve_arena_type`] returns
///     the real payload BTF type id (the scheduler's
///     `ktstr_arena_ctx`), the chase succeeds, and the resulting
///     `Ptr` carries `cast_annotation: Some("sdt_alloc")`.
///
///   - `payload` is rendered via `chase_sdt_data_payload` against
///     the discovered allocator's `target_type_id`. The
///     bridge does not directly fire on this path, but the dump's
///     `sdt_alloc_meta.target_type_id` (which the bridge
///     consumes to populate its index) MUST be the same id, so a
///     correctly-rendered `payload` proves the upstream allocator
///     metadata is wired.
///
/// This scenario catches the regression where the Fwd chase
/// previously surfaced "forward declaration" instead of recovered
/// fields — that defect was historically only visible via manual
/// dump inspection.
fn scenario_cast_analysis_sdt_alloc_bridge_resolves_fwd(
    ctx: &ktstr::scenario::Ctx,
) -> Result<AssertResult> {
    await_accessor_ready();
    let steps = vec![Step {
        setup: vec![ctx.cgroup_def("cg_0")].into(),
        ops: vec![],
        hold: HoldSpec::FULL,
    }];
    execute_steps(ctx, steps)
}

/// Host-side post_vm assertion for
/// `cast_analysis_sdt_alloc_bridge_resolves_fwd`. The guest cannot read
/// the host-written dump, so the sdt_alloc bridge / Fwd-chase assertions
/// run here; the callback's Err is a hard FAIL (`PostVmAssertionFailure`)
/// that `expect_err` does not invert.
fn check_cast_analysis_sdt_alloc_bridge_resolves_fwd(result: &VmResult) -> Result<()> {
    let dump = read_failure_dump(result)?;
    let dump_artifact = failure_dump_artifact(result);

    let task_storage = find_task_storage_map(&dump, &dump_artifact)?;
    let entries = task_storage
        .get("entries")
        .and_then(|e| e.as_array())
        .ok_or_else(|| anyhow::anyhow!("scx_task_map has no `entries` array; {dump_artifact}"))?;
    if entries.is_empty() {
        anyhow::bail!(
            "scx_task_map.entries is empty -- ktstr_init_task never registered \
             a per-task arena context for any task, so neither the surface-struct \
             chase nor the payload chase has anything to operate on. \
             {dump_artifact}"
        );
    }

    // ASSERTION 1: every entry whose `value` exists must NOT carry
    // a `data` member with `deref_skipped_reason` containing
    // "forward declaration". Iterate every entry rather than picking
    // one — a single passing entry with the bulk failing would still
    // be a regression, and the existing dump's structure makes the
    // per-entry walk cheap.
    let mut data_members_seen: usize = 0;
    let mut any_bridge_fired: bool = false;
    let mut any_data_with_chase: bool = false;
    for (idx, entry) in entries.iter().enumerate() {
        let Some(value) = entry.get("value") else {
            continue;
        };
        if value.is_null() {
            continue;
        }
        // The `value` is a BTF-rendered Struct of `scx_task_map_val`.
        let kind = value
            .get("kind")
            .and_then(|k| k.as_str())
            .unwrap_or("<no-kind>");
        if kind != "struct" {
            // The entry is hex-only or rendered into a non-struct
            // shape (e.g. Truncated wrapping a Bytes). Either way,
            // there's no `data` member to assert on; skip.
            continue;
        }
        let Some(members) = value.get("members").and_then(|m| m.as_array()) else {
            continue;
        };
        let Some(data) = members
            .iter()
            .find(|m| m.get("name").and_then(|n| n.as_str()) == Some("data"))
        else {
            // The renderer truncated before reaching `data`, or
            // the value type doesn't carry that member name. The
            // bridge is moot here — skip without flagging.
            continue;
        };
        data_members_seen += 1;
        let data_value = data.get("value").ok_or_else(|| {
            anyhow::anyhow!("`data` member has no `value`; entry={idx}; {dump_artifact}")
        })?;
        let data_kind = data_value
            .get("kind")
            .and_then(|k| k.as_str())
            .unwrap_or("<no-kind>");
        // `data` is `struct sdt_data __arena *` -- the renderer
        // emits Ptr (BTF Type::Ptr arm). Anything else is a
        // regression in the surface-struct render itself.
        if data_kind != "ptr" {
            anyhow::bail!(
                "entry[{idx}].value.data must render as Ptr (BTF Type::Ptr arm \
                 for `struct sdt_data __arena *`); got kind={data_kind:?}. \
                 {dump_artifact}"
            );
        }
        let data_value_u64 = data_value
            .get("value")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| {
                anyhow::anyhow!("`data` Ptr has no `value`; entry={idx}; {dump_artifact}")
            })?;
        if data_value_u64 == 0 {
            // Null `data` cannot exercise the bridge -- skip without
            // flagging. ktstr_init_task writes a non-null pointer on
            // every alloc, but a freshly-stored entry can race the
            // bridge index population in pathological cases.
            continue;
        }

        any_data_with_chase = true;

        // PRIMARY ASSERTION: no entry's `data` may surface a Fwd
        // skip reason. The bridge's job is exactly to prevent that.
        if let Some(reason) = data_value
            .get("deref_skipped_reason")
            .and_then(|r| r.as_str())
            && reason.contains("forward declaration")
        {
            anyhow::bail!(
                "REGRESSION: entry[{idx}].value.data surfaced a 'forward \
                 declaration' skip reason -- the sdt_alloc bridge did NOT \
                 fire. The chased pointer 0x{data_value_u64:x} fell outside \
                 every known allocator slot's payload-start index, the dump \
                 pre-pass failed to populate the index, or \
                 [`MemReader::resolve_arena_type`] returned None. Without \
                 the bridge the per-task struct content is unrenderable on \
                 the surface-struct path. Skip reason: {reason:?}; \
                 {dump_artifact}"
            );
        }

        // When the chase succeeded AND the bridge fired, the
        // `cast_annotation` MUST be exactly "sdt_alloc". (BTF
        // Type::Ptr arm -- not the cast-analyzer arm, so no
        // "cast→arena" prefix.)
        if let Some(ann) = data_value.get("cast_annotation").and_then(|a| a.as_str()) {
            if ann == "sdt_alloc" {
                any_bridge_fired = true;
            } else {
                anyhow::bail!(
                    "entry[{idx}].value.data carried unexpected \
                     cast_annotation={ann:?}; the BTF Type::Ptr arm only \
                     emits 'sdt_alloc' (no cast→ prefix) when the bridge \
                     fires on a Fwd target; {dump_artifact}"
                );
            }
        }
    }

    if data_members_seen == 0 {
        anyhow::bail!(
            "no scx_task_map entry exposed a `data` member in its rendered \
             value -- either every value-side render dropped to hex, the \
             BTF was missing, or the value type did not include the \
             `struct sdt_data __arena *` field. Without surfacing `data`, \
             the bridge has no chase to gate. Total entries: {}",
            entries.len()
        );
    }

    if !any_data_with_chase {
        anyhow::bail!(
            "every scx_task_map entry's `value.data` was 0x0 -- ktstr_init_task \
             never wrote a non-null `mval->data`, or every captured map slot \
             was snapshotted between the create-zeroed-entry phase and the \
             populate-fields phase of `scx_task_alloc`. The bridge has \
             no chase to validate. data_members_seen={data_members_seen}, \
             entries={}",
            entries.len()
        );
    }

    if !any_bridge_fired {
        anyhow::bail!(
            "REGRESSION: scx_task_map entries carried non-null `value.data` \
             pointers with no `deref_skipped_reason`, but NONE surfaced \
             cast_annotation='sdt_alloc'. That means the chase succeeded \
             via the BTF-only path -- which would only happen if the \
             program BTF carried a complete `struct sdt_data` body, \
             contradicting scx-ktstr's compiled BTF where `sdt_data` is \
             emitted as a Fwd. Either the bridge ran but failed to set \
             the annotation, or the test fixture's BTF shape changed in \
             a way that bypassed the Fwd path. \
             data_members_seen={data_members_seen}"
        );
    }

    // ASSERTION 2: per-entry `payload` (rendered via
    // `chase_sdt_data_payload`) must show ktstr_arena_ctx fields
    // populated by ktstr_init_task. Confirms the upstream
    // allocator metadata (which the bridge index keys into) is
    // wired. Reuse the same constants the existing scenarios pin.
    const KTSTR_ARENA_MAGIC: u64 = 0xDEADBEEFCAFEBABE;
    const KTSTR_TASK_COUNTER: u64 = 42;
    let mut payloads_inspected: usize = 0;
    for (idx, entry) in entries.iter().enumerate() {
        let Some(payload) = entry.get("payload") else {
            continue;
        };
        if payload.is_null() {
            continue;
        }
        let kind = payload
            .get("kind")
            .and_then(|k| k.as_str())
            .unwrap_or("<no-kind>");
        // Some entries can render `payload` as Truncated when the
        // payload struct exceeds POINTER_CHASE_CAP, but
        // ktstr_arena_ctx is 24 bytes; only Struct is expected.
        if kind != "struct" {
            continue;
        }
        let type_name = payload
            .get("type_name")
            .and_then(|n| n.as_str())
            .unwrap_or("<no-type-name>");
        if type_name != "ktstr_arena_ctx" {
            continue;
        }
        payloads_inspected += 1;

        // magic must read the alloc-time sentinel.
        let magic = struct_member(payload, "magic", &dump_artifact)?;
        let magic_value = magic.get("value").and_then(|v| v.as_u64()).ok_or_else(|| {
            anyhow::anyhow!("magic value not a u64; entry={idx}; {dump_artifact}")
        })?;
        if magic_value != KTSTR_ARENA_MAGIC {
            anyhow::bail!(
                "entry[{idx}].payload.magic mismatch: got 0x{magic_value:016x}, \
                 expected 0x{KTSTR_ARENA_MAGIC:016x}. The chase landed on a \
                 ktstr_arena_ctx-shaped struct but the bytes are not the \
                 alloc-time sentinel -- either the upstream allocator \
                 metadata pointed at a stale slot, or the .data chase / \
                 payload chase landed on different allocator state. \
                 {dump_artifact}"
            );
        }

        let counter = struct_member(payload, "counter", &dump_artifact)?;
        let counter_value = counter
            .get("value")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| anyhow::anyhow!("counter not numeric; entry={idx}; {dump_artifact}"))?;
        if counter_value != KTSTR_TASK_COUNTER {
            anyhow::bail!(
                "entry[{idx}].payload.counter mismatch: got {counter_value}, \
                 expected {KTSTR_TASK_COUNTER}; {dump_artifact}"
            );
        }
    }

    if payloads_inspected == 0 {
        anyhow::bail!(
            "no scx_task_map entry surfaced a Struct(type_name=\"ktstr_arena_ctx\") \
             payload -- `chase_sdt_data_payload` returned None for every \
             entry, sdt_alloc_meta.target_type_id was unresolved, or \
             every captured arena pointer fell outside the kern_vm window. \
             Without a rendered payload the bridge index has no payload \
             type id to publish, so the surface-struct bridge would also \
             fail. entries: {}",
            entries.len()
        );
    }

    eprintln!(
        "sdt_alloc bridge E2E: dump carries scx_task_map with \
         {} entries; data_members_seen={data_members_seen}, \
         any_bridge_fired={any_bridge_fired}, payloads_inspected={payloads_inspected}. \
         No entry's `data` showed a 'forward declaration' skip reason; \
         every chased ktstr_arena_ctx payload carries the alloc-time \
         sentinel and counter.",
        entries.len(),
    );

    Ok(())
}

/// Asserts that the cross-subprog arena pointer propagation
/// (fixpoint iteration) correctly tags `ktstr_arena_ctx.stashed_arena_ptr`
/// as an arena pointer in the rendered failure dump.
///
/// Path under test:
///   1. `ktstr_cross_btf_publish` (subprog) calls `scx_static_alloc`
///      → R0 is `ArenaU64FromAlloc`. STXs into hash map value's
///      `cached_ptr` field → `arena_stx_findings` records
///      `(ktstr_cross_btf_value, 0) → Arena`.
///   2. `ktstr_cross_btf_chase` (subprog, defined AFTER publish in
///      source) loads from hash map value's `cached_ptr` →
///      `handle_ldx` checks `arena_stx_findings` and promotes the
///      register to `ArenaU64FromAlloc`. STXs into
///      `taskc->stashed_arena_ptr` → cast map records
///      `(ktstr_arena_ctx, 24) → Arena`.
///   3. Because the callee (`scx_static_alloc_internal`) appears
///      before the caller in the ELF, the fixpoint iteration must
///      propagate the allocator-return typing across passes.
///
/// The assertion: `stashed_arena_ptr` renders as `ptr` (not `uint`)
/// with a `cast_annotation` containing "arena". If the fixpoint
/// fails to propagate, the field renders as a plain u64.
fn scenario_cast_analysis_cross_subprog_arena_chase(
    ctx: &ktstr::scenario::Ctx,
) -> Result<AssertResult> {
    await_accessor_ready();
    let steps = vec![Step {
        setup: vec![ctx.cgroup_def("cg_0")].into(),
        ops: vec![],
        hold: HoldSpec::FULL,
    }];
    execute_steps(ctx, steps)
}

/// Host-side post_vm assertion for
/// `cast_analysis_cross_subprog_arena_chase`. The guest cannot read the
/// host-written dump, so the cross-subprog fixpoint assertions run here;
/// the callback's Err is a hard FAIL (`PostVmAssertionFailure`) that
/// `expect_err` does not invert.
fn check_cast_analysis_cross_subprog_arena_chase(result: &VmResult) -> Result<()> {
    let dump = read_failure_dump(result)?;
    let dump_artifact = failure_dump_artifact(result);

    let task_storage = find_task_storage_map(&dump, &dump_artifact)?;
    let entries = task_storage
        .get("entries")
        .and_then(|e| e.as_array())
        .ok_or_else(|| anyhow::anyhow!("scx_task_map has no `entries` array; {dump_artifact}"))?;
    if entries.is_empty() {
        anyhow::bail!(
            "scx_task_map.entries is empty -- no per-task arena context was \
             allocated before freeze; {dump_artifact}"
        );
    }

    let payloads: Vec<&serde_json::Value> = entries
        .iter()
        .filter_map(|e| e.get("payload"))
        .filter(|p| !p.is_null())
        .filter(|p| {
            p.get("kind").and_then(|k| k.as_str()) == Some("struct")
                && p.get("type_name").and_then(|n| n.as_str()) == Some("ktstr_arena_ctx")
        })
        .collect();
    if payloads.is_empty() {
        anyhow::bail!(
            "no scx_task_map entry has a Struct(type_name=\"ktstr_arena_ctx\") \
             payload -- chase_sdt_data_payload did not resolve any per-task \
             arena context. entries: {}",
            entries.len()
        );
    }

    let mut any_arena_chase = false;
    for payload in &payloads {
        let stashed = match struct_member(payload, "stashed_arena_ptr", &dump_artifact) {
            Ok(v) => v,
            Err(_) => continue,
        };
        let kind = stashed
            .get("kind")
            .and_then(|k| k.as_str())
            .unwrap_or("<no-kind>");
        if kind == "ptr" {
            any_arena_chase = true;
            if let Some(ann) = stashed.get("cast_annotation").and_then(|a| a.as_str())
                && !ann.contains("arena")
            {
                anyhow::bail!(
                    "stashed_arena_ptr rendered as Ptr but cast_annotation \
                     does not contain 'arena': got {ann:?}. The cast \
                     analyzer tagged the field but with the wrong domain. \
                     {dump_artifact}"
                );
            }
            break;
        }
        if kind == "uint" {
            let value = stashed.get("value").and_then(|v| v.as_u64()).unwrap_or(0);
            if value != 0 {
                anyhow::bail!(
                    "FIXPOINT REGRESSION: stashed_arena_ptr is a non-zero u64 \
                     (0x{value:x}) that rendered as plain Uint instead of Ptr. \
                     The cross-subprog arena typing did NOT propagate through \
                     the fixpoint -- the publish helper's STX into the hash \
                     map value's cached_ptr was not carried to the chase \
                     helper's LDX site across passes; {dump_artifact}"
                );
            }
        }
    }

    if !any_arena_chase {
        anyhow::bail!(
            "no ktstr_arena_ctx payload rendered stashed_arena_ptr as Ptr -- \
             the cross-subprog fixpoint did not produce a cast finding for \
             (ktstr_arena_ctx, 24). Either the publish helper's allocator \
             return was not tagged as ArenaU64FromAlloc, or the chase \
             helper's LDX through the hash map value did not inherit the \
             tag, or the final STX into the per-task field was not recorded. \
             Checked {} payloads.",
            payloads.len()
        );
    }

    eprintln!(
        "cross-subprog arena chase E2E: dump carries scx_task_map \
         with {} entries, {} ktstr_arena_ctx payloads. Located \
         stashed_arena_ptr rendered as Ptr with arena annotation -- \
         fixpoint propagation across publish→map→chase subprog boundary \
         is working.",
        entries.len(),
        payloads.len(),
    );

    Ok(())
}

#[ktstr::ktstr_test_entry]
static __KTSTR_ENTRY_CAST_ANALYSIS_CROSS_SUBPROG: ktstr::test_support::KtstrTestEntry =
    ktstr::test_support::KtstrTestEntry {
        name: "cast_analysis_cross_subprog_arena_chase",
        func: scenario_cast_analysis_cross_subprog_arena_chase,
        scheduler: &KTSTR_SCHED,
        extra_sched_args: &["--stall-after=1"],
        watchdog_timeout: std::time::Duration::from_secs(3),
        duration: std::time::Duration::from_secs(10),
        // SCX_EXIT_ERROR_STALL is the intentional kill; `expect_err: true`
        // inverts that to PASS. The cross-subprog fixpoint assertions run
        // in check_cast_analysis_cross_subprog_arena_chase, a
        // post_vm_unconditional callback whose Err is a hard FAIL via
        // PostVmAssertionFailure — `expect_err` does NOT invert it.
        expect_err: true,
        post_vm_unconditional: Some(check_cast_analysis_cross_subprog_arena_chase),
        ..ktstr::test_support::KtstrTestEntry::DEFAULT
    };

#[ktstr::ktstr_test_entry]
static __KTSTR_ENTRY_CAST_ANALYSIS_SDT_ALLOC_BRIDGE: ktstr::test_support::KtstrTestEntry =
    ktstr::test_support::KtstrTestEntry {
        name: "cast_analysis_sdt_alloc_bridge_resolves_fwd",
        func: scenario_cast_analysis_sdt_alloc_bridge_resolves_fwd,
        scheduler: &KTSTR_SCHED,
        // Same trigger as the sibling scenarios: SCX_EXIT_ERROR_STALL
        // latches the freeze-and-dump path so the bridge index is
        // populated AND the surface-struct render sees a Fwd-pointee
        // chase value. ktstr_init_task wires `scx_task_map_val.data`
        // on every task it processes, so by the time the watchdog
        // fires multiple per-task arena allocations exist for the
        // bridge to resolve.
        extra_sched_args: &["--stall-after=1"],
        watchdog_timeout: std::time::Duration::from_secs(3),
        duration: std::time::Duration::from_secs(10),
        // SCX_EXIT_ERROR_STALL is the intentional kill; `expect_err: true`
        // inverts that to PASS. The bridge / Fwd-chase assertions run in
        // check_cast_analysis_sdt_alloc_bridge_resolves_fwd, a
        // post_vm_unconditional callback whose Err is a hard FAIL via
        // PostVmAssertionFailure — `expect_err` does NOT invert it.
        expect_err: true,
        post_vm_unconditional: Some(check_cast_analysis_sdt_alloc_bridge_resolves_fwd),
        ..ktstr::test_support::KtstrTestEntry::DEFAULT
    };
