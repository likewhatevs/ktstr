use super::super::*;
use super::common::{
    DEFAULT_START_TIME, EXT_KVA, PAGE_OFFSET, build_test_kernel, paint_valid_task, synth_task,
};
use crate::test_support::btf_blob::{CastSynMember, CastSynType, cast_build_btf};
use btf_rs::Btf;

// ---- BTF-gated dispatch-arm tests (PerCpuField + TaskField) ----
//
// `resolve_per_cpu_field_pa` (kernel_op_dispatch.rs:595) and
// `resolve_and_validate_task_field` (:1402) are the only two
// dispatcher paths that need BTF: both resolve a field byte offset
// via `nested_member_byte_offset` against a kernel struct. These
// tests build a SYNTHETIC BTF blob (via `cast_build_btf`, hoisted
// to `crate::test_support::btf_blob`) whose struct layouts match the
// `synth_task` offsets in `super::common`, then drive the resolver
// fns over the shared synthetic `GuestKernel` — no VM boot, no
// vmlinux. The walker / validation-chain error arms are already
// covered by `dispatch_arms.rs` + `task_field.rs`; these focus on
// the BTF-resolution layer (struct/field lookup, the per-cpu offset
// math, and the happy-path round trip through the real
// read/write dispatchers).

/// Append a NUL-terminated `name` to a BTF string section and return
/// its byte offset. Mirrors the `push` closure the `cast_*` renderer
/// suites use; factored out here because every blob below needs it.
fn push_str(strings: &mut Vec<u8>, name: &str) -> u32 {
    let off = strings.len() as u32;
    strings.extend_from_slice(name.as_bytes());
    strings.push(0);
    off
}

/// A plain-unsigned `u64` BTF Int (`type_id == 1` in every blob
/// below). Used as the leaf type for every scalar struct member —
/// `nested_member_byte_offset` only descends through NON-leaf
/// segments (via the member's `type_id` → struct), so the leaf
/// member's concrete int type is immaterial to offset resolution.
fn u64_int(name_off: u32) -> CastSynType {
    CastSynType::Int {
        name_off,
        size: 8,
        encoding: 0,
        offset: 0,
        bits: 64,
    }
}

// ===============================================================
// resolve_per_cpu_field_pa
// ===============================================================

/// Synthetic BTF for the only per-CPU symbol these tests exercise:
/// `runqueues`→`rq` (the `struct_name_for_per_cpu_symbol` map at
/// kernel_op_dispatch.rs:555). The `rq` struct carries a single
/// `nr_running@0x0` member.
fn rq_btf() -> Vec<u8> {
    let mut strings: Vec<u8> = vec![0];
    let n_u64 = push_str(&mut strings, "u64");
    let n_rq = push_str(&mut strings, "rq");
    let n_nr = push_str(&mut strings, "nr_running");
    let types = vec![
        u64_int(n_u64), // id=1
        CastSynType::Struct {
            name_off: n_rq,
            size: 0x40,
            members: vec![CastSynMember {
                name_off: n_nr,
                type_id: 1,
                byte_offset: 0x0,
            }],
        }, // id=2
    ];
    cast_build_btf(&types, &strings)
}

/// PA the `runqueues` per-CPU template direct-maps to. The symbol KVA
/// is `PAGE_OFFSET + BASE_PA` (direct-map fast path: pa = kva -
/// page_offset). With `__per_cpu_offset[0] == 0` and `kaslr == 0`,
/// `per_cpu_kva == template_kva`, so the resolved PA is exactly
/// `BASE_PA`.
const BASE_PA: u64 = 0x800;
/// PA holding the synthetic `__per_cpu_offset[]` array. Slot `cpu` is
/// the u64 at `ARR_PA + cpu*8`. `__per_cpu_offset` is a text/data
/// symbol, so its KVA is `START_KERNEL_MAP + ARR_PA`.
const ARR_PA: u64 = 0x100;

/// Build the symbol map + paint `__per_cpu_offset` for the per-CPU
/// happy path: `runqueues` direct-maps to `BASE_PA`, the offset array
/// lives at `ARR_PA`, and `__per_cpu_offset[cpu]` is set to
/// `per_cpu_off`.
fn percpu_symbols_and_paint(
    buf: &mut [u8],
    cpu: u32,
    per_cpu_off: u64,
) -> std::collections::HashMap<String, u64> {
    let arr_slot_pa = ARR_PA + (cpu as u64) * 8;
    buf[arr_slot_pa as usize..arr_slot_pa as usize + 8].copy_from_slice(&per_cpu_off.to_le_bytes());
    let mut m = std::collections::HashMap::new();
    m.insert("runqueues".to_string(), PAGE_OFFSET + BASE_PA);
    m.insert(
        "__per_cpu_offset".to_string(),
        crate::monitor::symbols::START_KERNEL_MAP + ARR_PA,
    );
    m
}

/// Happy path: `runqueues.nr_running[cpu=0]` resolves to `BASE_PA`
/// (per_cpu_off[0]=0, kaslr=0, nr_running@0x0). Pins the exact PA.
#[test]
fn resolve_per_cpu_field_pa_happy_path_returns_base_pa() {
    let blob = rq_btf();
    let btf = Btf::from_bytes(&blob).expect("synthetic rq BTF parses");
    let mut buf = vec![0u8; 0x4000];
    let symbols = percpu_symbols_and_paint(&mut buf, 0, 0);
    let kernel = build_test_kernel(&mut buf, symbols);
    let pa = resolve_per_cpu_field_pa(&kernel, Some(&btf), 0, "runqueues", "nr_running", 0)
        .expect("per-cpu field PA must resolve");
    assert_eq!(
        pa as u64, BASE_PA,
        "nr_running@0x0 + per_cpu_off=0 → BASE_PA"
    );
}

/// Non-zero per-cpu offset shifts the resolved PA by exactly that
/// offset: `per_cpu_off=0x40`, `runqueues`@`BASE_PA` → PA
/// `BASE_PA + 0x40`. Pins the additive per-CPU math.
#[test]
fn resolve_per_cpu_field_pa_applies_per_cpu_offset() {
    let blob = rq_btf();
    let btf = Btf::from_bytes(&blob).expect("synthetic rq BTF parses");
    let mut buf = vec![0u8; 0x4000];
    const OFF: u64 = 0x40;
    let symbols = percpu_symbols_and_paint(&mut buf, 1, OFF);
    let kernel = build_test_kernel(&mut buf, symbols);
    let pa = resolve_per_cpu_field_pa(&kernel, Some(&btf), 0, "runqueues", "nr_running", 1)
        .expect("per-cpu field PA must resolve");
    assert_eq!(
        pa as u64,
        BASE_PA + OFF,
        "per_cpu_off=0x40 must shift the resolved PA additively"
    );
}

/// End-to-end PerCpuField write through the production dispatcher:
/// the LE word lands at `BASE_PA + nr_running_off`.
#[test]
fn per_cpu_field_write_u32_lands_exact_bytes() {
    const VAL: u32 = 0x0BAD_CAFE;
    let blob = rq_btf();
    let btf = Btf::from_bytes(&blob).expect("synthetic rq BTF parses");
    let mut buf = vec![0u8; 0x4000];
    let symbols = percpu_symbols_and_paint(&mut buf, 0, 0);
    let kernel = build_test_kernel(&mut buf, symbols);
    dispatch_one_write(
        &kernel,
        Some(&btf),
        0,
        &KernelOpTarget::PerCpuField {
            symbol: "runqueues".into(),
            field: "nr_running".into(),
            cpu: 0,
        },
        &KernelOpValue::U32(VAL),
    )
    .expect("per-cpu field u32 write must succeed");
    assert_eq!(
        &buf[BASE_PA as usize..BASE_PA as usize + 4],
        &VAL.to_le_bytes(),
        "PerCpuField/Write/U32 must store the LE word at BASE_PA"
    );
}

/// End-to-end PerCpuField read through the production dispatcher
/// returns the exact U64 painted at `BASE_PA`.
#[test]
fn per_cpu_field_read_u64_returns_exact_value() {
    const VAL: u64 = 0x1122_3344_5566_7788;
    let blob = rq_btf();
    let btf = Btf::from_bytes(&blob).expect("synthetic rq BTF parses");
    let mut buf = vec![0u8; 0x4000];
    let symbols = percpu_symbols_and_paint(&mut buf, 0, 0);
    buf[BASE_PA as usize..BASE_PA as usize + 8].copy_from_slice(&VAL.to_le_bytes());
    let kernel = build_test_kernel(&mut buf, symbols);
    let got = dispatch_one_read(
        &kernel,
        Some(&btf),
        0,
        &KernelOpTarget::PerCpuField {
            symbol: "runqueues".into(),
            field: "nr_running".into(),
            cpu: 0,
        },
        &KernelOpValue::U64(0),
    )
    .expect("per-cpu field u64 read must succeed");
    assert_eq!(got, KernelOpValue::U64(VAL));
}

/// PerCpuField Bytes write is rejected (per-CPU scheduler fields are
/// scalars) — and the rejection fires AFTER PA resolution, in the
/// width match. Pins the typed reason.
#[test]
fn per_cpu_field_bytes_write_rejects() {
    let blob = rq_btf();
    let btf = Btf::from_bytes(&blob).expect("synthetic rq BTF parses");
    let mut buf = vec![0u8; 0x4000];
    let symbols = percpu_symbols_and_paint(&mut buf, 0, 0);
    let kernel = build_test_kernel(&mut buf, symbols);
    let err = dispatch_one_write(
        &kernel,
        Some(&btf),
        0,
        &KernelOpTarget::PerCpuField {
            symbol: "runqueues".into(),
            field: "nr_running".into(),
            cpu: 0,
        },
        &KernelOpValue::Bytes(vec![1, 2, 3, 4]),
    )
    .expect_err("Bytes write to a per-cpu scalar must reject");
    assert!(
        err.contains("Bytes write not supported"),
        "must name the scalar-only restriction: {err}"
    );
}

/// PerCpuField OrU32 as a READ width-hint is wire misuse — the
/// read-side width match rejects it with the "no read semantic"
/// reason.
#[test]
fn per_cpu_field_oru32_read_rejects() {
    let blob = rq_btf();
    let btf = Btf::from_bytes(&blob).expect("synthetic rq BTF parses");
    let mut buf = vec![0u8; 0x4000];
    let symbols = percpu_symbols_and_paint(&mut buf, 0, 0);
    let kernel = build_test_kernel(&mut buf, symbols);
    let err = dispatch_one_read(
        &kernel,
        Some(&btf),
        0,
        &KernelOpTarget::PerCpuField {
            symbol: "runqueues".into(),
            field: "nr_running".into(),
            cpu: 0,
        },
        &KernelOpValue::OrU32(0xFF),
    )
    .expect_err("OrU32 read must reject");
    assert!(
        err.contains("OrU32 has no read semantic"),
        "must surface the OrU32-no-read reason: {err}"
    );
}

// ---- resolve_per_cpu_field_pa error arms ----

/// btf=None → "BTF not loaded": the resolver bails before any symbol
/// or memory touch.
#[test]
fn resolve_per_cpu_field_pa_btf_none_rejects() {
    let mut buf = vec![0u8; 0x4000];
    let symbols = percpu_symbols_and_paint(&mut buf, 0, 0);
    let kernel = build_test_kernel(&mut buf, symbols);
    let err = resolve_per_cpu_field_pa(&kernel, None, 0, "runqueues", "nr_running", 0)
        .expect_err("btf=None must reject");
    assert!(err.contains("BTF not loaded"), "{err}");
}

/// Unknown per-CPU symbol → struct_name_for_per_cpu_symbol rejects
/// with the v1-supported-set message.
#[test]
fn resolve_per_cpu_field_pa_unknown_symbol_rejects() {
    let blob = rq_btf();
    let btf = Btf::from_bytes(&blob).expect("synthetic rq BTF parses");
    let mut buf = vec![0u8; 0x4000];
    let symbols = percpu_symbols_and_paint(&mut buf, 0, 0);
    let kernel = build_test_kernel(&mut buf, symbols);
    let err = resolve_per_cpu_field_pa(&kernel, Some(&btf), 0, "bogus_symbol", "x", 0)
        .expect_err("unknown per-cpu symbol must reject");
    assert!(
        err.contains("unknown per-CPU symbol 'bogus_symbol'"),
        "{err}"
    );
}

/// Known symbol but absent from the vmlinux symtab → "symbol absent".
/// `runqueues` is in the struct map but not painted into the symbol
/// HashMap.
#[test]
fn resolve_per_cpu_field_pa_symbol_absent_rejects() {
    let blob = rq_btf();
    let btf = Btf::from_bytes(&blob).expect("synthetic rq BTF parses");
    let mut buf = vec![0u8; 0x4000];
    // Only paint __per_cpu_offset, NOT runqueues.
    let arr_slot_pa = ARR_PA;
    buf[arr_slot_pa as usize..arr_slot_pa as usize + 8].copy_from_slice(&0u64.to_le_bytes());
    let mut symbols = std::collections::HashMap::new();
    symbols.insert(
        "__per_cpu_offset".to_string(),
        crate::monitor::symbols::START_KERNEL_MAP + ARR_PA,
    );
    let kernel = build_test_kernel(&mut buf, symbols);
    let err = resolve_per_cpu_field_pa(&kernel, Some(&btf), 0, "runqueues", "nr_running", 0)
        .expect_err("absent runqueues symbol must reject");
    assert!(
        err.contains("'runqueues' symbol absent"),
        "must name the missing template symbol: {err}"
    );
}

/// `__per_cpu_offset` absent → "kernel built without SMP".
#[test]
fn resolve_per_cpu_field_pa_no_per_cpu_offset_rejects() {
    let blob = rq_btf();
    let btf = Btf::from_bytes(&blob).expect("synthetic rq BTF parses");
    let mut buf = vec![0u8; 0x4000];
    let mut symbols = std::collections::HashMap::new();
    symbols.insert("runqueues".to_string(), PAGE_OFFSET + BASE_PA);
    let kernel = build_test_kernel(&mut buf, symbols);
    let err = resolve_per_cpu_field_pa(&kernel, Some(&btf), 0, "runqueues", "nr_running", 0)
        .expect_err("absent __per_cpu_offset must reject");
    assert!(err.contains("kernel built without SMP"), "{err}");
}

/// `__per_cpu_offset[cpu]==0 && cpu>0` → "cpu beyond nr_cpu_ids".
/// Slot 1 painted 0; request cpu=1.
#[test]
fn resolve_per_cpu_field_pa_zero_offset_high_cpu_rejects() {
    let blob = rq_btf();
    let btf = Btf::from_bytes(&blob).expect("synthetic rq BTF parses");
    let mut buf = vec![0u8; 0x4000];
    // cpu=1 offset slot left 0.
    let symbols = percpu_symbols_and_paint(&mut buf, 1, 0);
    let kernel = build_test_kernel(&mut buf, symbols);
    let err = resolve_per_cpu_field_pa(&kernel, Some(&btf), 0, "runqueues", "nr_running", 1)
        .expect_err("zero per-cpu offset for cpu>0 must reject");
    assert!(err.contains("cpu beyond nr_cpu_ids"), "{err}");
}

/// per_cpu_kva below page_offset → "below kernel page_offset" wrap
/// guard. The template KVA is set below page_offset (so the slide
/// keeps it low) and per_cpu_off=0, forcing the floor check to fire.
#[test]
fn resolve_per_cpu_field_pa_below_page_offset_rejects() {
    let blob = rq_btf();
    let btf = Btf::from_bytes(&blob).expect("synthetic rq BTF parses");
    let mut buf = vec![0u8; 0x4000];
    // Paint __per_cpu_offset[0]=0.
    buf[ARR_PA as usize..ARR_PA as usize + 8].copy_from_slice(&0u64.to_le_bytes());
    let mut symbols = std::collections::HashMap::new();
    // template below page_offset → per_cpu_kva (kaslr=0, off=0) stays
    // below the floor. 0x1000 is well under PAGE_OFFSET and under the
    // 1<<48 slide threshold, so slid_kernel_kva is identity.
    symbols.insert("runqueues".to_string(), 0x1000);
    symbols.insert(
        "__per_cpu_offset".to_string(),
        crate::monitor::symbols::START_KERNEL_MAP + ARR_PA,
    );
    let kernel = build_test_kernel(&mut buf, symbols);
    let err = resolve_per_cpu_field_pa(&kernel, Some(&btf), 0, "runqueues", "nr_running", 0)
        .expect_err("per_cpu_kva below page_offset must reject");
    assert!(err.contains("below kernel page_offset"), "{err}");
}

/// Struct present but field missing from BTF → "BTF nested-offset"
/// error naming the absent field. `nr_running` exists; query a field
/// that does not.
#[test]
fn resolve_per_cpu_field_pa_field_btf_miss_rejects() {
    let blob = rq_btf();
    let btf = Btf::from_bytes(&blob).expect("synthetic rq BTF parses");
    let mut buf = vec![0u8; 0x4000];
    let symbols = percpu_symbols_and_paint(&mut buf, 0, 0);
    let kernel = build_test_kernel(&mut buf, symbols);
    let err = resolve_per_cpu_field_pa(&kernel, Some(&btf), 0, "runqueues", "no_such_field", 0)
        .expect_err("missing field must reject");
    assert!(
        err.contains("BTF nested-offset for 'no_such_field'"),
        "{err}"
    );
}

/// Struct absent from BTF entirely → "'struct rq' BTF lookup" error.
/// Use a blob that has NO `rq` struct (only the leaf Int).
#[test]
fn resolve_per_cpu_field_pa_struct_btf_miss_rejects() {
    let mut strings: Vec<u8> = vec![0];
    let n_u64 = push_str(&mut strings, "u64");
    let blob = cast_build_btf(&[u64_int(n_u64)], &strings);
    let btf = Btf::from_bytes(&blob).expect("synthetic int-only BTF parses");
    let mut buf = vec![0u8; 0x4000];
    let symbols = percpu_symbols_and_paint(&mut buf, 0, 0);
    let kernel = build_test_kernel(&mut buf, symbols);
    let err = resolve_per_cpu_field_pa(&kernel, Some(&btf), 0, "runqueues", "nr_running", 0)
        .expect_err("absent rq struct must reject");
    assert!(err.contains("'struct rq' BTF"), "{err}");
}

/// per_cpu_kva that translates to a PA beyond the guest memory size →
/// "unmapped (translate_any_kva returned None)". Place the template
/// just past the buffer end so the direct-map fast path rejects.
#[test]
fn resolve_per_cpu_field_pa_unmapped_rejects() {
    let blob = rq_btf();
    let btf = Btf::from_bytes(&blob).expect("synthetic rq BTF parses");
    let mut buf = vec![0u8; 0x4000];
    buf[ARR_PA as usize..ARR_PA as usize + 8].copy_from_slice(&0u64.to_le_bytes());
    let dram_size = buf.len() as u64;
    let mut symbols = std::collections::HashMap::new();
    // template direct-maps to PA == dram_size (one past the last
    // valid byte) → translate_any_kva bounds-rejects.
    symbols.insert("runqueues".to_string(), PAGE_OFFSET + dram_size);
    symbols.insert(
        "__per_cpu_offset".to_string(),
        crate::monitor::symbols::START_KERNEL_MAP + ARR_PA,
    );
    let kernel = build_test_kernel(&mut buf, symbols);
    let err = resolve_per_cpu_field_pa(&kernel, Some(&btf), 0, "runqueues", "nr_running", 0)
        .expect_err("per_cpu_kva past DRAM must reject");
    assert!(err.contains("unmapped"), "{err}");
}

// ===============================================================
// resolve_and_validate_task_field
// ===============================================================

// Layout for the task-field happy path: a single leader task at
// LEADER_PA carries the target pid; init_task is the text-mapped
// list head; the ring closes head → leader → head. Mirrors the
// dispatch_arms.rs walker fixtures.
const TF_INIT_TASK_PA: u64 = 0x100;
const TF_LEADER_PA: u64 = 0x800;
const TF_PID: u32 = 4242;

/// Build a `task_struct` + `sched_ext_entity` + `signal_struct` BTF
/// whose member byte offsets match the `synth_task` consts in
/// `super::common`. `scx` is a NESTED member (`type_id` → the
/// `sched_ext_entity` struct), so `nested_member_byte_offset`
/// descends into it for `scx.dsq` / `scx.runnable_node`.
///
/// Type ids: 1=u64 Int (leaf), 2=sched_ext_entity, 3=task_struct,
/// 4=signal_struct.
fn task_struct_btf() -> Vec<u8> {
    let mut strings: Vec<u8> = vec![0];
    let n_u64 = push_str(&mut strings, "u64");
    let n_sched_ext_entity = push_str(&mut strings, "sched_ext_entity");
    let n_dsq = push_str(&mut strings, "dsq");
    let n_runnable_node = push_str(&mut strings, "runnable_node");
    let n_task_struct = push_str(&mut strings, "task_struct");
    let n_pid = push_str(&mut strings, "pid");
    let n_start_time = push_str(&mut strings, "start_time");
    let n_state = push_str(&mut strings, "__state");
    let n_on_rq = push_str(&mut strings, "on_rq");
    let n_sched_class = push_str(&mut strings, "sched_class");
    let n_start_boottime = push_str(&mut strings, "start_boottime");
    let n_scx = push_str(&mut strings, "scx");
    let n_tasks = push_str(&mut strings, "tasks");
    let n_signal = push_str(&mut strings, "signal");
    let n_thread_node = push_str(&mut strings, "thread_node");
    let n_signal_struct = push_str(&mut strings, "signal_struct");
    let n_thread_head = push_str(&mut strings, "thread_head");

    let member = |name_off: u32, type_id: u32, byte_offset: u32| CastSynMember {
        name_off,
        type_id,
        byte_offset,
    };

    let types = vec![
        u64_int(n_u64), // id=1
        // id=2: sched_ext_entity { dsq@0x0, runnable_node@0x8 }.
        // scx.dsq → 0x48 + 0x0 = SCX_DSQ_OFF; scx.runnable_node →
        // 0x48 + 0x8 = SCX_RUNNABLE_NODE_OFF (see super::common).
        CastSynType::Struct {
            name_off: n_sched_ext_entity,
            size: 0x10,
            members: vec![member(n_dsq, 1, 0x0), member(n_runnable_node, 1, 0x8)],
        },
        // id=3: task_struct. scalar members → id=1; scx → id=2 so
        // nested descent reaches sched_ext_entity. Offsets are the
        // synth_task consts.
        CastSynType::Struct {
            name_off: n_task_struct,
            size: 0x80,
            members: vec![
                member(n_pid, 1, synth_task::PID_OFF as u32),
                member(n_start_time, 1, synth_task::START_TIME_OFF as u32),
                member(n_state, 1, synth_task::STATE_OFF as u32),
                member(n_on_rq, 1, synth_task::ON_RQ_OFF as u32),
                member(n_sched_class, 1, synth_task::SCHED_CLASS_OFF as u32),
                member(n_start_boottime, 1, synth_task::START_BOOTTIME_OFF as u32),
                member(n_scx, 2, synth_task::SCX_DSQ_OFF as u32),
                member(n_tasks, 1, synth_task::TASKS_OFF as u32),
                member(n_signal, 1, synth_task::SIGNAL_OFF as u32),
                member(n_thread_node, 1, synth_task::THREAD_NODE_OFF as u32),
            ],
        },
        // id=4: signal_struct { thread_head@SIGNAL_THREAD_HEAD_OFF }.
        CastSynType::Struct {
            name_off: n_signal_struct,
            size: 0x20,
            members: vec![member(
                n_thread_head,
                1,
                synth_task::SIGNAL_THREAD_HEAD_OFF as u32,
            )],
        },
    ];
    cast_build_btf(&types, &strings)
}

/// init_task KVA (text/data symbol path: START_KERNEL_MAP + PA).
fn tf_init_task_kva() -> u64 {
    crate::monitor::symbols::START_KERNEL_MAP + TF_INIT_TASK_PA
}

/// Paint a single-leader ring (head → leader → head), a valid task at
/// `TF_LEADER_PA` carrying `TF_PID`, and return the symbol map with
/// `init_task` + `ext_sched_class` wired. With `kaslr=0`,
/// `ext_sched_class_kva == EXT_KVA` (the value paint_valid_task
/// stores in `sched_class`), so the L6 identity check passes.
fn paint_task_field_fixture(buf: &mut [u8]) -> std::collections::HashMap<String, u64> {
    paint_valid_task(buf, TF_LEADER_PA as usize, TF_PID);
    // init_task.tasks.next → leader node.
    let head_link_pa = TF_INIT_TASK_PA + synth_task::TASKS_OFF as u64;
    let leader_node_kva = PAGE_OFFSET + TF_LEADER_PA + synth_task::TASKS_OFF as u64;
    buf[head_link_pa as usize..head_link_pa as usize + 8]
        .copy_from_slice(&leader_node_kva.to_le_bytes());
    // leader.tasks.next → head (closes ring).
    let leader_tasks_pa = TF_LEADER_PA + synth_task::TASKS_OFF as u64;
    let head_kva = tf_init_task_kva() + synth_task::TASKS_OFF as u64;
    buf[leader_tasks_pa as usize..leader_tasks_pa as usize + 8]
        .copy_from_slice(&head_kva.to_le_bytes());

    let mut symbols = std::collections::HashMap::new();
    symbols.insert("init_task".to_string(), tf_init_task_kva());
    symbols.insert("ext_sched_class".to_string(), EXT_KVA);
    symbols
}

/// Happy path: resolve_and_validate_task_field walks to the valid
/// leader, passes all 8 validation layers, and returns the leader's
/// task_pa (== TF_LEADER_PA, slab direct-mapped) + the task_struct
/// BTF type.
#[test]
fn resolve_and_validate_task_field_happy_path_returns_task_pa() {
    let blob = task_struct_btf();
    let btf = Btf::from_bytes(&blob).expect("synthetic task_struct BTF parses");
    let mut buf = vec![0u8; 0x4000];
    let symbols = paint_task_field_fixture(&mut buf);
    let kernel = build_test_kernel(&mut buf, symbols);
    let (task_pa, _struct_t) =
        resolve_and_validate_task_field(&kernel, Some(&btf), 0, TF_PID, DEFAULT_START_TIME)
            .expect("task field must resolve + validate");
    assert_eq!(
        task_pa, TF_LEADER_PA,
        "must return the leader's direct-map PA"
    );
}

/// End-to-end TaskField write through the production dispatcher: the
/// LE word lands at `task_pa + nested_member_byte_offset(field)`.
/// `start_boottime` is a direct task_struct member; write a u64 and
/// assert the bytes at `TF_LEADER_PA + START_BOOTTIME_OFF`.
#[test]
fn task_field_write_u64_lands_at_nested_offset() {
    const VAL: u64 = 0xFEED_FACE_DEAD_BEEF;
    let blob = task_struct_btf();
    let btf = Btf::from_bytes(&blob).expect("synthetic task_struct BTF parses");
    let mut buf = vec![0u8; 0x4000];
    let symbols = paint_task_field_fixture(&mut buf);
    let kernel = build_test_kernel(&mut buf, symbols);
    dispatch_one_write(
        &kernel,
        Some(&btf),
        0,
        &KernelOpTarget::TaskField {
            pid: TF_PID,
            expected_start_time_ns: DEFAULT_START_TIME,
            field: "start_boottime".into(),
        },
        &KernelOpValue::U64(VAL),
    )
    .expect("task field u64 write must succeed");
    let off = synth_task::START_BOOTTIME_OFF;
    assert_eq!(
        &buf[TF_LEADER_PA as usize + off..TF_LEADER_PA as usize + off + 8],
        &VAL.to_le_bytes(),
        "TaskField/Write/U64 must store the LE word at task_pa + field offset"
    );
}

/// End-to-end TaskField write to a NESTED field (`scx.dsq`) lands at
/// `task_pa + SCX_DSQ_OFF` — pins the nested-path descent through the
/// `scx` member into `sched_ext_entity`.
#[test]
fn task_field_write_nested_scx_dsq_lands_at_descended_offset() {
    const VAL: u64 = 0x0123_4567_89AB_CDEF;
    let blob = task_struct_btf();
    let btf = Btf::from_bytes(&blob).expect("synthetic task_struct BTF parses");
    let mut buf = vec![0u8; 0x4000];
    let symbols = paint_task_field_fixture(&mut buf);
    let kernel = build_test_kernel(&mut buf, symbols);
    dispatch_one_write(
        &kernel,
        Some(&btf),
        0,
        &KernelOpTarget::TaskField {
            pid: TF_PID,
            expected_start_time_ns: DEFAULT_START_TIME,
            field: "scx.dsq".into(),
        },
        &KernelOpValue::U64(VAL),
    )
    .expect("nested scx.dsq u64 write must succeed");
    let off = synth_task::SCX_DSQ_OFF;
    assert_eq!(
        &buf[TF_LEADER_PA as usize + off..TF_LEADER_PA as usize + off + 8],
        &VAL.to_le_bytes(),
        "scx.dsq must descend to 0x48 + 0x0"
    );
}

/// End-to-end TaskField read returns the exact U32 painted at
/// `task_pa + pid_off`. Reads `pid` itself (the painter set it to
/// TF_PID) rather than a field a mutated value would have to defeat
/// the validation chain to observe.
#[test]
fn task_field_read_u32_returns_exact_value() {
    let blob = task_struct_btf();
    let btf = Btf::from_bytes(&blob).expect("synthetic task_struct BTF parses");
    let mut buf = vec![0u8; 0x4000];
    let symbols = paint_task_field_fixture(&mut buf);
    let kernel = build_test_kernel(&mut buf, symbols);
    let got = dispatch_one_read(
        &kernel,
        Some(&btf),
        0,
        &KernelOpTarget::TaskField {
            pid: TF_PID,
            expected_start_time_ns: DEFAULT_START_TIME,
            field: "pid".into(),
        },
        &KernelOpValue::U32(0),
    )
    .expect("task field u32 read must succeed");
    assert_eq!(got, KernelOpValue::U32(TF_PID));
}

/// TaskField read of a NON-key field round-trips a distinct sentinel:
/// write `start_boottime` to a value unrelated to the walk key, then
/// read it back. This proves the read returns the bytes at the
/// resolved field offset rather than echoing the `pid` the walker
/// matched on — a gap `task_field_read_u32` (which reads `pid` itself)
/// can't close.
#[test]
fn task_field_read_nonkey_roundtrips_distinct_value() {
    // Non-zero so the `start_boottime != 0` validation (L7) still
    // passes after the write, and distinct from TF_PID so a read that
    // echoed the walk key would fail this assertion.
    const BOOTTIME_SENTINEL: u64 = 0xDEAD_BEEF_0000_1234;
    let blob = task_struct_btf();
    let btf = Btf::from_bytes(&blob).expect("synthetic task_struct BTF parses");
    let mut buf = vec![0u8; 0x4000];
    let symbols = paint_task_field_fixture(&mut buf);
    let kernel = build_test_kernel(&mut buf, symbols);
    let target = KernelOpTarget::TaskField {
        pid: TF_PID,
        expected_start_time_ns: DEFAULT_START_TIME,
        field: "start_boottime".into(),
    };
    dispatch_one_write(
        &kernel,
        Some(&btf),
        0,
        &target,
        &KernelOpValue::U64(BOOTTIME_SENTINEL),
    )
    .expect("task field u64 write must succeed");
    let got = dispatch_one_read(&kernel, Some(&btf), 0, &target, &KernelOpValue::U64(0))
        .expect("task field u64 read must succeed");
    assert_eq!(got, KernelOpValue::U64(BOOTTIME_SENTINEL));
}

/// TaskField Bytes write is rejected (per-task scheduler fields are
/// scalars) AFTER walk + validation succeed.
#[test]
fn task_field_bytes_write_rejects() {
    let blob = task_struct_btf();
    let btf = Btf::from_bytes(&blob).expect("synthetic task_struct BTF parses");
    let mut buf = vec![0u8; 0x4000];
    let symbols = paint_task_field_fixture(&mut buf);
    let kernel = build_test_kernel(&mut buf, symbols);
    let err = dispatch_one_write(
        &kernel,
        Some(&btf),
        0,
        &KernelOpTarget::TaskField {
            pid: TF_PID,
            expected_start_time_ns: DEFAULT_START_TIME,
            field: "start_boottime".into(),
        },
        &KernelOpValue::Bytes(vec![1, 2, 3, 4]),
    )
    .expect_err("Bytes write to a per-task scalar must reject");
    assert!(err.contains("Bytes write not supported in v1"), "{err}");
}

/// TaskField OrU32 RMW is rejected (no use case; per-task scheduler
/// fields are scalars not flags).
#[test]
fn task_field_oru32_write_rejects() {
    let blob = task_struct_btf();
    let btf = Btf::from_bytes(&blob).expect("synthetic task_struct BTF parses");
    let mut buf = vec![0u8; 0x4000];
    let symbols = paint_task_field_fixture(&mut buf);
    let kernel = build_test_kernel(&mut buf, symbols);
    let err = dispatch_one_write(
        &kernel,
        Some(&btf),
        0,
        &KernelOpTarget::TaskField {
            pid: TF_PID,
            expected_start_time_ns: DEFAULT_START_TIME,
            field: "start_boottime".into(),
        },
        &KernelOpValue::OrU32(0x1),
    )
    .expect_err("OrU32 RMW on a per-task field must reject");
    assert!(err.contains("OrU32 RMW not supported"), "{err}");
}

// ---- resolve_and_validate_task_field error arms ----

/// btf=None → "BTF not loaded": bails before symbol resolution.
#[test]
fn resolve_and_validate_task_field_btf_none_rejects() {
    let mut buf = vec![0u8; 0x4000];
    let symbols = paint_task_field_fixture(&mut buf);
    let kernel = build_test_kernel(&mut buf, symbols);
    let err = resolve_and_validate_task_field(&kernel, None, 0, TF_PID, DEFAULT_START_TIME)
        .expect_err("btf=None must reject");
    assert!(err.contains("BTF not loaded"), "{err}");
}

/// init_task symbol absent → "init_task symbol absent" (cannot anchor
/// the walker).
#[test]
fn resolve_and_validate_task_field_no_init_task_rejects() {
    let blob = task_struct_btf();
    let btf = Btf::from_bytes(&blob).expect("synthetic task_struct BTF parses");
    let mut buf = vec![0u8; 0x4000];
    let mut symbols = paint_task_field_fixture(&mut buf);
    symbols.remove("init_task");
    let kernel = build_test_kernel(&mut buf, symbols);
    let err = resolve_and_validate_task_field(&kernel, Some(&btf), 0, TF_PID, DEFAULT_START_TIME)
        .expect_err("absent init_task must reject");
    assert!(err.contains("init_task symbol absent"), "{err}");
}

/// ext_sched_class absent → "kernel built without CONFIG_SCHED_CLASS_EXT"
/// (TaskField is SCX-only).
#[test]
fn resolve_and_validate_task_field_no_ext_sched_class_rejects() {
    let blob = task_struct_btf();
    let btf = Btf::from_bytes(&blob).expect("synthetic task_struct BTF parses");
    let mut buf = vec![0u8; 0x4000];
    let mut symbols = paint_task_field_fixture(&mut buf);
    symbols.remove("ext_sched_class");
    let kernel = build_test_kernel(&mut buf, symbols);
    let err = resolve_and_validate_task_field(&kernel, Some(&btf), 0, TF_PID, DEFAULT_START_TIME)
        .expect_err("absent ext_sched_class must reject");
    assert!(err.contains("without CONFIG_SCHED_CLASS_EXT"), "{err}");
}

/// task_struct absent from BTF → TaskValidationOffsets::resolve_from_btf
/// bails with "'struct task_struct' lookup". Int-only blob.
#[test]
fn resolve_and_validate_task_field_task_struct_btf_miss_rejects() {
    let mut strings: Vec<u8> = vec![0];
    let n_u64 = push_str(&mut strings, "u64");
    let blob = cast_build_btf(&[u64_int(n_u64)], &strings);
    let btf = Btf::from_bytes(&blob).expect("synthetic int-only BTF parses");
    let mut buf = vec![0u8; 0x4000];
    let symbols = paint_task_field_fixture(&mut buf);
    let kernel = build_test_kernel(&mut buf, symbols);
    let err = resolve_and_validate_task_field(&kernel, Some(&btf), 0, TF_PID, DEFAULT_START_TIME)
        .expect_err("absent task_struct BTF must reject");
    assert!(err.contains("'struct task_struct' lookup"), "{err}");
}
