use super::super::*;
use super::common::{
    PAGE_OFFSET, build_test_kernel, paint_valid_task, synth_task, synth_validation_offsets,
};

// ---- Symbol / Direct / Kva dispatch-arm + batch + walker tests ----
//
// Drive `dispatch_one_write` / `dispatch_one_read` / the batch
// entry points / `find_task_by_pid` over the shared synthetic
// `GuestKernel` (no VM boot, no BTF). BTF-gated arms (PerCpuField,
// TaskField) are out of scope — they need a synthetic BTF fixture.

/// Install a single Symbol whose KVA resolves to physical address
/// `pa` in the synthetic buffer. The Symbol path translates via
/// `START_KERNEL_MAP` (phys_base=0 in the test ctor), so
/// `kva = START_KERNEL_MAP + pa` lands the read/write at byte `pa`.
fn symbol_at(name: &str, pa: u64) -> std::collections::HashMap<String, u64> {
    let mut m = std::collections::HashMap::new();
    m.insert(
        name.to_string(),
        crate::monitor::symbols::START_KERNEL_MAP + pa,
    );
    m
}

// ---------------------------------------------------------------
// Symbol write arms (btf=None) — assert the exact bytes at the PA.
// ---------------------------------------------------------------

/// Symbol/Write/U32 lands the little-endian word at the resolved PA.
#[test]
fn symbol_write_u32_lands_exact_bytes() {
    const PA: u64 = 0x40;
    const VAL: u32 = 0x1234_5678;
    let mut buf = vec![0u8; 4096];
    let kernel = build_test_kernel(&mut buf, symbol_at("s", PA));
    dispatch_one_write(
        &kernel,
        None,
        0,
        &KernelOpTarget::Symbol("s".into()),
        &KernelOpValue::U32(VAL),
    )
    .expect("symbol u32 write must succeed");
    assert_eq!(
        &buf[PA as usize..PA as usize + 4],
        &VAL.to_le_bytes(),
        "Symbol/Write/U32 must store the LE word at the resolved PA"
    );
}

/// Symbol/Write/U64 lands the little-endian doubleword at the PA.
#[test]
fn symbol_write_u64_lands_exact_bytes() {
    const PA: u64 = 0x80;
    const VAL: u64 = 0xDEAD_BEEF_CAFE_F00D;
    let mut buf = vec![0u8; 4096];
    let kernel = build_test_kernel(&mut buf, symbol_at("s", PA));
    dispatch_one_write(
        &kernel,
        None,
        0,
        &KernelOpTarget::Symbol("s".into()),
        &KernelOpValue::U64(VAL),
    )
    .expect("symbol u64 write must succeed");
    assert_eq!(&buf[PA as usize..PA as usize + 8], &VAL.to_le_bytes());
}

/// Symbol/Write/Bytes copies the payload verbatim at the PA.
#[test]
fn symbol_write_bytes_lands_exact_payload() {
    const PA: u64 = 0x100;
    let payload: [u8; 6] = [0x11, 0x22, 0x33, 0x44, 0x55, 0x66];
    let mut buf = vec![0u8; 4096];
    let kernel = build_test_kernel(&mut buf, symbol_at("s", PA));
    dispatch_one_write(
        &kernel,
        None,
        0,
        &KernelOpTarget::Symbol("s".into()),
        &KernelOpValue::Bytes(payload.to_vec()),
    )
    .expect("symbol bytes write must succeed");
    assert_eq!(&buf[PA as usize..PA as usize + payload.len()], &payload);
}

/// Symbol/Write/OrU32 is a read-modify-write: prepaint 0x01, OR 0x10,
/// the byte becomes 0x11 (union, not overwrite).
#[test]
fn symbol_write_oru32_is_read_modify_write() {
    const PA: u64 = 0x140;
    let mut buf = vec![0u8; 4096];
    buf[PA as usize..PA as usize + 4].copy_from_slice(&0x0000_0001u32.to_le_bytes());
    let kernel = build_test_kernel(&mut buf, symbol_at("s", PA));
    dispatch_one_write(
        &kernel,
        None,
        0,
        &KernelOpTarget::Symbol("s".into()),
        &KernelOpValue::OrU32(0x0000_0010),
    )
    .expect("symbol OrU32 RMW must succeed");
    assert_eq!(
        u32::from_le_bytes(buf[PA as usize..PA as usize + 4].try_into().unwrap()),
        0x0000_0011,
        "OrU32 must union the mask into the live word (0x01 | 0x10 = 0x11)"
    );
}

// ---------------------------------------------------------------
// Symbol read arms (btf=None) — assert the exact KernelOpValue.
// ---------------------------------------------------------------

/// Symbol/Read/U32 returns the exact word painted at the PA. The
/// width hint variant (U32) picks the read family; its payload is
/// ignored.
#[test]
fn symbol_read_u32_returns_exact_value() {
    const PA: u64 = 0x40;
    const VAL: u32 = 0x0BAD_F00D;
    let mut buf = vec![0u8; 4096];
    buf[PA as usize..PA as usize + 4].copy_from_slice(&VAL.to_le_bytes());
    let kernel = build_test_kernel(&mut buf, symbol_at("s", PA));
    let got = dispatch_one_read(
        &kernel,
        None,
        0,
        &KernelOpTarget::Symbol("s".into()),
        &KernelOpValue::U32(0), // width hint; payload ignored
    )
    .expect("symbol u32 read must succeed");
    assert_eq!(got, KernelOpValue::U32(VAL));
}

/// Symbol/Read/U64 returns the exact doubleword.
#[test]
fn symbol_read_u64_returns_exact_value() {
    const PA: u64 = 0x80;
    const VAL: u64 = 0x0102_0304_0506_0708;
    let mut buf = vec![0u8; 4096];
    buf[PA as usize..PA as usize + 8].copy_from_slice(&VAL.to_le_bytes());
    let kernel = build_test_kernel(&mut buf, symbol_at("s", PA));
    let got = dispatch_one_read(
        &kernel,
        None,
        0,
        &KernelOpTarget::Symbol("s".into()),
        &KernelOpValue::U64(0),
    )
    .expect("symbol u64 read must succeed");
    assert_eq!(got, KernelOpValue::U64(VAL));
}

/// Symbol/Read/Bytes returns exactly the painted bytes; the length
/// comes from the placeholder's `len()`.
#[test]
fn symbol_read_bytes_returns_exact_payload() {
    const PA: u64 = 0x100;
    let painted: [u8; 5] = [0xAA, 0xBB, 0xCC, 0xDD, 0xEE];
    let mut buf = vec![0u8; 4096];
    buf[PA as usize..PA as usize + painted.len()].copy_from_slice(&painted);
    let kernel = build_test_kernel(&mut buf, symbol_at("s", PA));
    let got = dispatch_one_read(
        &kernel,
        None,
        0,
        &KernelOpTarget::Symbol("s".into()),
        &KernelOpValue::Bytes(vec![0u8; painted.len()]), // placeholder sets read len
    )
    .expect("symbol bytes read must succeed");
    assert_eq!(got, KernelOpValue::Bytes(painted.to_vec()));
}

/// Symbol read of an absent symbol surfaces a typed error naming the
/// missing symbol. `require_symbol` rejects before any memory touch.
#[test]
fn symbol_read_absent_symbol_rejects() {
    let mut buf = vec![0u8; 4096];
    let kernel = build_test_kernel(&mut buf, Default::default());
    let err = dispatch_one_read(
        &kernel,
        None,
        0,
        &KernelOpTarget::Symbol("nonexistent".into()),
        &KernelOpValue::U32(0),
    )
    .expect_err("absent symbol must reject");
    assert!(
        err.contains("read_symbol_u32('nonexistent')"),
        "error must name the read helper + symbol: {err}"
    );
}

// ---------------------------------------------------------------
// Direct write/read arms (btf=None) — pa = kva - page_offset.
// ---------------------------------------------------------------

/// Direct/Write/U32 then Direct/Read/U32 at the same KVA round-trips
/// the exact word. KVA = PAGE_OFFSET + pa lands at byte `pa`.
#[test]
fn direct_write_then_read_u32_round_trips() {
    const PA: u64 = 0x200;
    const VAL: u32 = 0xFEED_FACE;
    let kva = PAGE_OFFSET + PA;
    let mut buf = vec![0u8; 4096];
    let kernel = build_test_kernel(&mut buf, Default::default());
    dispatch_one_write(
        &kernel,
        None,
        0,
        &KernelOpTarget::Direct(kva),
        &KernelOpValue::U32(VAL),
    )
    .expect("direct u32 write must succeed");
    // Bytes landed at pa.
    assert_eq!(
        u32::from_le_bytes(buf[PA as usize..PA as usize + 4].try_into().unwrap()),
        VAL
    );
    let got = dispatch_one_read(
        &kernel,
        None,
        0,
        &KernelOpTarget::Direct(kva),
        &KernelOpValue::U32(0),
    )
    .expect("direct u32 read must succeed");
    assert_eq!(got, KernelOpValue::U32(VAL));
}

/// Direct/Write/U64 lands the exact doubleword at pa = kva -
/// page_offset.
#[test]
fn direct_write_u64_lands_exact_bytes() {
    const PA: u64 = 0x280;
    const VAL: u64 = 0x1122_3344_5566_7788;
    let kva = PAGE_OFFSET + PA;
    let mut buf = vec![0u8; 4096];
    let kernel = build_test_kernel(&mut buf, Default::default());
    dispatch_one_write(
        &kernel,
        None,
        0,
        &KernelOpTarget::Direct(kva),
        &KernelOpValue::U64(VAL),
    )
    .expect("direct u64 write must succeed");
    assert_eq!(&buf[PA as usize..PA as usize + 8], &VAL.to_le_bytes());
}

/// Direct/Read/Bytes returns exactly the painted payload.
#[test]
fn direct_read_bytes_returns_exact_payload() {
    const PA: u64 = 0x300;
    let painted: [u8; 7] = [1, 2, 3, 4, 5, 6, 7];
    let kva = PAGE_OFFSET + PA;
    let mut buf = vec![0u8; 4096];
    buf[PA as usize..PA as usize + painted.len()].copy_from_slice(&painted);
    let kernel = build_test_kernel(&mut buf, Default::default());
    let got = dispatch_one_read(
        &kernel,
        None,
        0,
        &KernelOpTarget::Direct(kva),
        &KernelOpValue::Bytes(vec![0u8; painted.len()]),
    )
    .expect("direct bytes read must succeed");
    assert_eq!(got, KernelOpValue::Bytes(painted.to_vec()));
}

/// Direct/Write/OrU32 RMW: prepaint 0x01, OR 0x10, byte becomes 0x11.
#[test]
fn direct_write_oru32_is_read_modify_write() {
    const PA: u64 = 0x340;
    let kva = PAGE_OFFSET + PA;
    let mut buf = vec![0u8; 4096];
    buf[PA as usize..PA as usize + 4].copy_from_slice(&0x0000_0001u32.to_le_bytes());
    let kernel = build_test_kernel(&mut buf, Default::default());
    dispatch_one_write(
        &kernel,
        None,
        0,
        &KernelOpTarget::Direct(kva),
        &KernelOpValue::OrU32(0x0000_0010),
    )
    .expect("direct OrU32 RMW must succeed");
    assert_eq!(
        u32::from_le_bytes(buf[PA as usize..PA as usize + 4].try_into().unwrap()),
        0x0000_0011
    );
}

/// Direct/Write below page_offset rejects with the "would wrap"
/// reason BEFORE any store (validate_direct_target gate).
#[test]
fn direct_write_below_page_offset_rejects() {
    let kva = PAGE_OFFSET - 1; // underflows kva_to_pa
    let mut buf = vec![0u8; 4096];
    let kernel = build_test_kernel(&mut buf, Default::default());
    let err = dispatch_one_write(
        &kernel,
        None,
        0,
        &KernelOpTarget::Direct(kva),
        &KernelOpValue::U32(0xFFFF_FFFF),
    )
    .expect_err("kva below page_offset must reject");
    assert!(
        err.contains("below page_offset"),
        "must name the band: {err}"
    );
    assert!(err.contains("would wrap"));
}

/// Direct/Write past the direct-map end rejects with the "overruns
/// direct-map end" reason. dram_size = buf.len(); last valid U32 is
/// at PAGE_OFFSET + dram_size - 4, so + dram_size - 3 overruns.
#[test]
fn direct_write_past_end_rejects() {
    let mut buf = vec![0u8; 4096];
    let dram_size = buf.len() as u64;
    let kva = PAGE_OFFSET + dram_size - 3; // 4-byte write runs 1 past the end
    let kernel = build_test_kernel(&mut buf, Default::default());
    let err = dispatch_one_write(
        &kernel,
        None,
        0,
        &KernelOpTarget::Direct(kva),
        &KernelOpValue::U32(0xFFFF_FFFF),
    )
    .expect_err("kva+len past direct-map end must reject");
    assert!(err.contains("overruns direct-map end"));
}

// ---------------------------------------------------------------
// Kva arms (btf=None) — page-walked; cr3_pa=0 ⇒ unmapped; user-half
// rejected before the walk.
// ---------------------------------------------------------------

/// Kva/Write above the kernel-half threshold with cr3_pa=0 page-walks
/// against the zeroed page-table region and returns "page unmapped".
/// The KVA is high enough that the direct-map fast path
/// (kva - page_offset) lands far past dram_size, forcing the walk.
#[test]
fn kva_write_unmapped_with_zero_cr3() {
    // Typical vmalloc KVA: passes validate_kva_target (>= 5-level
    // conservative threshold) and is well above page_offset, so
    // translate_any_kva's fast path misses and the cr3=0 walk fails.
    const KVA: u64 = 0xFFFF_C900_0000_0000;
    let mut buf = vec![0u8; 4096];
    let kernel = build_test_kernel(&mut buf, Default::default());
    let err = dispatch_one_write(
        &kernel,
        None,
        0,
        &KernelOpTarget::Kva(KVA),
        &KernelOpValue::U32(0xABCD_1234),
    )
    .expect_err("unmapped Kva write must reject");
    assert!(
        err.contains("write_kva_u32") && err.contains("page unmapped"),
        "must surface the page-unmapped failure: {err}"
    );
}

/// Kva/Read above the threshold with cr3_pa=0 returns "page
/// unmapped".
#[test]
fn kva_read_unmapped_with_zero_cr3() {
    const KVA: u64 = 0xFFFF_C900_0000_0000;
    let mut buf = vec![0u8; 4096];
    let kernel = build_test_kernel(&mut buf, Default::default());
    let err = dispatch_one_read(
        &kernel,
        None,
        0,
        &KernelOpTarget::Kva(KVA),
        &KernelOpValue::U64(0),
    )
    .expect_err("unmapped Kva read must reject");
    assert!(
        err.contains("read_kva_u64") && err.contains("page unmapped"),
        "must surface the page-unmapped failure: {err}"
    );
}

/// Kva/Write below the kernel-half conservative threshold is rejected
/// by validate_kva_target BEFORE the page-walk, with the user-half
/// rejection reason.
#[test]
fn kva_write_user_half_rejects_before_walk() {
    const KVA: u64 = 0x0000_4000_0000_0000; // user-half
    let mut buf = vec![0u8; 4096];
    let kernel = build_test_kernel(&mut buf, Default::default());
    let err = dispatch_one_write(
        &kernel,
        None,
        0,
        &KernelOpTarget::Kva(KVA),
        &KernelOpValue::U32(0),
    )
    .expect_err("user-half Kva must reject");
    // dispatch routes through validate_kva_target →
    // user_half_kva_rejection_reason (pinned verbatim in
    // target_validation.rs); assert the band wording here.
    assert!(
        err.contains("below kernel-half"),
        "must name the band: {err}"
    );
    assert!(err.contains(&format!("{KVA:#x}")));
}

// ---------------------------------------------------------------
// OrU32 read-direction misuse (catch-all arm).
// ---------------------------------------------------------------

/// OrU32 as a read width-hint is wire-format misuse on ANY target —
/// the read catch-all rejects it. Pins that the catch-all fires for
/// a Symbol target (it has no per-target OrU32 read arm).
#[test]
fn read_direction_oru32_rejects_via_catch_all() {
    const MASK: u32 = 0x0000_00FF;
    let mut buf = vec![0u8; 4096];
    let kernel = build_test_kernel(&mut buf, symbol_at("s", 0x40));
    let err = dispatch_one_read(
        &kernel,
        None,
        0,
        &KernelOpTarget::Symbol("s".into()),
        &KernelOpValue::OrU32(MASK),
    )
    .expect_err("OrU32 read must reject");
    assert_eq!(err, oru32_read_rejection_reason(MASK));
}

// ---------------------------------------------------------------
// Batch dispatch (dispatch_kernel_op_batch / write_batch /
// read_batch).
// ---------------------------------------------------------------

/// All-ok write batch: every entry succeeds, reply is success with
/// empty reason + no read_values; each entry's bytes land at its PA.
#[test]
fn write_batch_all_ok() {
    let mut buf = vec![0u8; 4096];
    let mut symbols = std::collections::HashMap::new();
    symbols.insert(
        "a".to_string(),
        crate::monitor::symbols::START_KERNEL_MAP + 0x40,
    );
    symbols.insert(
        "b".to_string(),
        crate::monitor::symbols::START_KERNEL_MAP + 0x80,
    );
    let kernel = build_test_kernel(&mut buf, symbols);
    let req = crate::vmm::wire::KernelOpRequestPayload {
        request_id: 7,
        mode: crate::vmm::wire::KernelOpMode::Cold,
        direction: KernelOpDirection::Write,
        tag: String::new(),
        entries: vec![
            KernelOpEntry {
                target: KernelOpTarget::Symbol("a".into()),
                value: KernelOpValue::U32(0x1111_1111),
            },
            KernelOpEntry {
                target: KernelOpTarget::Symbol("b".into()),
                value: KernelOpValue::U64(0x2222_2222_2222_2222),
            },
        ],
    };
    let reply = dispatch_kernel_op_batch(&kernel, None, 0, &req);
    assert!(reply.success);
    assert_eq!(reply.request_id, 7);
    assert_eq!(reply.reason, "");
    assert!(reply.read_values.is_empty());
    assert_eq!(
        u32::from_le_bytes(buf[0x40..0x44].try_into().unwrap()),
        0x1111_1111
    );
    assert_eq!(
        u64::from_le_bytes(buf[0x80..0x88].try_into().unwrap()),
        0x2222_2222_2222_2222
    );
}

/// Write batch with a bad entry at index 1: reply is failure, reason
/// is prefixed `entry[1]:`. The index-0 write still landed (no
/// rollback per the module's partial-prefix contract).
#[test]
fn write_batch_failure_at_index_1_prefixes_entry_index() {
    let mut buf = vec![0u8; 4096];
    let kernel = build_test_kernel(&mut buf, symbol_at("ok", 0x40));
    let req = crate::vmm::wire::KernelOpRequestPayload {
        request_id: 9,
        mode: crate::vmm::wire::KernelOpMode::Cold,
        direction: KernelOpDirection::Write,
        tag: String::new(),
        entries: vec![
            KernelOpEntry {
                target: KernelOpTarget::Symbol("ok".into()),
                value: KernelOpValue::U32(0x3333_3333),
            },
            KernelOpEntry {
                // Absent symbol — fails dispatch.
                target: KernelOpTarget::Symbol("missing".into()),
                value: KernelOpValue::U32(0),
            },
        ],
    };
    let reply = dispatch_kernel_op_batch(&kernel, None, 0, &req);
    assert!(!reply.success);
    assert_eq!(reply.request_id, 9);
    assert!(
        reply.reason.starts_with("entry[1]:"),
        "batch must name the failing index: {}",
        reply.reason
    );
    // Index-0 write landed before the index-1 failure (partial prefix).
    assert_eq!(
        u32::from_le_bytes(buf[0x40..0x44].try_into().unwrap()),
        0x3333_3333,
        "earlier-index write is applied; no rollback"
    );
}

/// Read batch preserves read-value ordering: read_values[i]
/// corresponds to entries[i].
#[test]
fn read_batch_preserves_value_ordering() {
    let mut buf = vec![0u8; 4096];
    // a@0x40 = u32 0xAAAA_0001, b@0x80 = u64 0xBBBB_0000_0000_0002.
    buf[0x40..0x44].copy_from_slice(&0xAAAA_0001u32.to_le_bytes());
    buf[0x80..0x88].copy_from_slice(&0xBBBB_0000_0000_0002u64.to_le_bytes());
    let mut symbols = std::collections::HashMap::new();
    symbols.insert(
        "a".to_string(),
        crate::monitor::symbols::START_KERNEL_MAP + 0x40,
    );
    symbols.insert(
        "b".to_string(),
        crate::monitor::symbols::START_KERNEL_MAP + 0x80,
    );
    let kernel = build_test_kernel(&mut buf, symbols);
    let req = crate::vmm::wire::KernelOpRequestPayload {
        request_id: 5,
        mode: crate::vmm::wire::KernelOpMode::Cold,
        direction: KernelOpDirection::Read,
        tag: String::new(),
        entries: vec![
            KernelOpEntry {
                target: KernelOpTarget::Symbol("a".into()),
                value: KernelOpValue::U32(0),
            },
            KernelOpEntry {
                target: KernelOpTarget::Symbol("b".into()),
                value: KernelOpValue::U64(0),
            },
        ],
    };
    let reply = dispatch_kernel_op_batch(&kernel, None, 0, &req);
    assert!(reply.success);
    assert_eq!(reply.reason, "");
    assert_eq!(
        reply.read_values,
        vec![
            KernelOpValue::U32(0xAAAA_0001),
            KernelOpValue::U64(0xBBBB_0000_0000_0002),
        ],
        "read_values must be index-aligned with entries"
    );
}

/// Read batch with a bad entry at index 1: failure, reason prefixed
/// `entry[1]:`, no read_values returned (short-circuit).
#[test]
fn read_batch_failure_at_index_1_prefixes_entry_index() {
    let mut buf = vec![0u8; 4096];
    let kernel = build_test_kernel(&mut buf, symbol_at("ok", 0x40));
    let req = crate::vmm::wire::KernelOpRequestPayload {
        request_id: 6,
        mode: crate::vmm::wire::KernelOpMode::Cold,
        direction: KernelOpDirection::Read,
        tag: String::new(),
        entries: vec![
            KernelOpEntry {
                target: KernelOpTarget::Symbol("ok".into()),
                value: KernelOpValue::U32(0),
            },
            KernelOpEntry {
                target: KernelOpTarget::Symbol("missing".into()),
                value: KernelOpValue::U32(0),
            },
        ],
    };
    let reply = dispatch_kernel_op_batch(&kernel, None, 0, &req);
    assert!(!reply.success);
    assert!(reply.reason.starts_with("entry[1]:"), "{}", reply.reason);
    assert!(
        reply.read_values.is_empty(),
        "failed read batch returns no values"
    );
}

// ---------------------------------------------------------------
// find_task_by_pid walker (btf not needed — offsets passed directly).
// ---------------------------------------------------------------

// Layout shared by the walker tests. init_task lives in the text/data
// mapping (Symbol path); leaders are slab tasks in the direct map.
const INIT_TASK_PA: u64 = 0x100;
const LEADER1_PA: u64 = 0x800;
const LEADER2_PA: u64 = 0x1000;

/// init_task KVA (text/data symbol): translated via START_KERNEL_MAP.
fn init_task_kva() -> u64 {
    crate::monitor::symbols::START_KERNEL_MAP + INIT_TASK_PA
}

/// Slab list-node KVA for a leader at PA `leader_pa`: the KVA of its
/// `tasks` list_head member. `leader_kva = node_kva - TASKS_OFF` and
/// `leader_kva = PAGE_OFFSET + leader_pa`.
fn leader_node_kva(leader_pa: u64) -> u64 {
    PAGE_OFFSET + leader_pa + synth_task::TASKS_OFF as u64
}

/// Write a u64 at PA `pa` in the buffer (list_head.next painter).
fn put_u64(buf: &mut [u8], pa: u64, val: u64) {
    buf[pa as usize..pa as usize + 8].copy_from_slice(&val.to_le_bytes());
}

/// HIT in a 2-leader ring: head → leader1 → leader2 → head. Target
/// pid is leader2's; the walker returns leader2's KVA
/// (PAGE_OFFSET + LEADER2_PA).
#[test]
fn find_task_by_pid_hit_returns_leader_kva() {
    let mut buf = vec![0u8; 0x4000];
    paint_valid_task(&mut buf, LEADER1_PA as usize, 111);
    paint_valid_task(&mut buf, LEADER2_PA as usize, 222);
    // init_task.tasks.next (PA = INIT_TASK_PA + TASKS_OFF) → leader1 node.
    let head_link_pa = INIT_TASK_PA + synth_task::TASKS_OFF as u64;
    put_u64(&mut buf, head_link_pa, leader_node_kva(LEADER1_PA));
    // leader1.tasks.next → leader2 node.
    put_u64(
        &mut buf,
        LEADER1_PA + synth_task::TASKS_OFF as u64,
        leader_node_kva(LEADER2_PA),
    );
    // leader2.tasks.next → head (closes ring; head_kva = init_task_kva + TASKS_OFF).
    put_u64(
        &mut buf,
        LEADER2_PA + synth_task::TASKS_OFF as u64,
        init_task_kva() + synth_task::TASKS_OFF as u64,
    );
    let kernel = build_test_kernel(&mut buf, Default::default());
    let offs = synth_validation_offsets();
    let got =
        find_task_by_pid(&kernel, init_task_kva(), 0, &offs, 222).expect("pid 222 must be found");
    assert_eq!(got, PAGE_OFFSET + LEADER2_PA, "must return leader2's KVA");
}

/// HIT on the FIRST leader: the walker matches before advancing.
#[test]
fn find_task_by_pid_hit_first_leader() {
    let mut buf = vec![0u8; 0x4000];
    paint_valid_task(&mut buf, LEADER1_PA as usize, 111);
    let head_link_pa = INIT_TASK_PA + synth_task::TASKS_OFF as u64;
    put_u64(&mut buf, head_link_pa, leader_node_kva(LEADER1_PA));
    // single-leader ring: leader1.tasks.next → head.
    put_u64(
        &mut buf,
        LEADER1_PA + synth_task::TASKS_OFF as u64,
        init_task_kva() + synth_task::TASKS_OFF as u64,
    );
    let kernel = build_test_kernel(&mut buf, Default::default());
    let offs = synth_validation_offsets();
    let got =
        find_task_by_pid(&kernel, init_task_kva(), 0, &offs, 111).expect("pid 111 must be found");
    assert_eq!(got, PAGE_OFFSET + LEADER1_PA);
}

/// MISS: ring closes back to head with no matching pid → "pid not
/// found".
#[test]
fn find_task_by_pid_miss_returns_not_found() {
    let mut buf = vec![0u8; 0x4000];
    paint_valid_task(&mut buf, LEADER1_PA as usize, 111);
    paint_valid_task(&mut buf, LEADER2_PA as usize, 222);
    let head_link_pa = INIT_TASK_PA + synth_task::TASKS_OFF as u64;
    put_u64(&mut buf, head_link_pa, leader_node_kva(LEADER1_PA));
    put_u64(
        &mut buf,
        LEADER1_PA + synth_task::TASKS_OFF as u64,
        leader_node_kva(LEADER2_PA),
    );
    put_u64(
        &mut buf,
        LEADER2_PA + synth_task::TASKS_OFF as u64,
        init_task_kva() + synth_task::TASKS_OFF as u64,
    );
    let kernel = build_test_kernel(&mut buf, Default::default());
    let offs = synth_validation_offsets();
    let err = find_task_by_pid(&kernel, init_task_kva(), 0, &offs, 999)
        .expect_err("absent pid must reject");
    assert!(err.contains("pid=999 not found"), "{err}");
}

/// Empty ring (head.next == head): no user tasks → "init_task.tasks
/// is empty".
#[test]
fn find_task_by_pid_empty_ring_rejects() {
    let mut buf = vec![0u8; 0x4000];
    let head_link_pa = INIT_TASK_PA + synth_task::TASKS_OFF as u64;
    // head.next = head_kva (= init_task_kva + TASKS_OFF).
    put_u64(
        &mut buf,
        head_link_pa,
        init_task_kva() + synth_task::TASKS_OFF as u64,
    );
    let kernel = build_test_kernel(&mut buf, Default::default());
    let offs = synth_validation_offsets();
    let err = find_task_by_pid(&kernel, init_task_kva(), 0, &offs, 1)
        .expect_err("empty ring must reject");
    assert!(err.contains("init_task.tasks is empty"), "{err}");
}

/// head.next read as 0 (zeroed/unmapped head bytes) → torn-read
/// rejection.
#[test]
fn find_task_by_pid_head_next_zero_rejects() {
    // head_link_pa left zeroed.
    let mut buf = vec![0u8; 0x4000];
    let kernel = build_test_kernel(&mut buf, Default::default());
    let offs = synth_validation_offsets();
    let err = find_task_by_pid(&kernel, init_task_kva(), 0, &offs, 1)
        .expect_err("zero head.next must reject");
    assert!(
        err.contains("read as 0") && err.contains("unmapped or torn read"),
        "{err}"
    );
}

/// Corrupt leader ring that revisits a node without closing through
/// head → "leader ring revisited". leader1.tasks.next points back at
/// leader1's own node (self-loop, never reaches head).
#[test]
fn find_task_by_pid_leader_cycle_rejects() {
    let mut buf = vec![0u8; 0x4000];
    paint_valid_task(&mut buf, LEADER1_PA as usize, 111);
    let head_link_pa = INIT_TASK_PA + synth_task::TASKS_OFF as u64;
    put_u64(&mut buf, head_link_pa, leader_node_kva(LEADER1_PA));
    // leader1.tasks.next → its OWN node (cycle that never reaches head).
    put_u64(
        &mut buf,
        LEADER1_PA + synth_task::TASKS_OFF as u64,
        leader_node_kva(LEADER1_PA),
    );
    let kernel = build_test_kernel(&mut buf, Default::default());
    let offs = synth_validation_offsets();
    // Target a pid NOT on the ring so the walk advances into the cycle.
    let err = find_task_by_pid(&kernel, init_task_kva(), 0, &offs, 999)
        .expect_err("leader cycle must reject");
    assert!(err.contains("leader ring revisited"), "{err}");
}

/// Broken chain: leader1.tasks.next = 0 → "list_head.next read as 0
/// ... chain broken". Distinct from the empty-ring case (that is
/// head.next == head).
#[test]
fn find_task_by_pid_leader_next_zero_rejects() {
    let mut buf = vec![0u8; 0x4000];
    paint_valid_task(&mut buf, LEADER1_PA as usize, 111);
    let head_link_pa = INIT_TASK_PA + synth_task::TASKS_OFF as u64;
    put_u64(&mut buf, head_link_pa, leader_node_kva(LEADER1_PA));
    // leader1.tasks.next left 0 → chain broken before finding pid.
    put_u64(&mut buf, LEADER1_PA + synth_task::TASKS_OFF as u64, 0);
    let kernel = build_test_kernel(&mut buf, Default::default());
    let offs = synth_validation_offsets();
    // pid not on leader1 so the walk tries to advance and hits next=0.
    let err = find_task_by_pid(&kernel, init_task_kva(), 0, &offs, 999)
        .expect_err("zero leader next must reject");
    assert!(
        err.contains("list_head.next read as 0") && err.contains("chain broken"),
        "{err}"
    );
}

/// HIT on a non-leader THREAD via leader's signal->thread_head walk.
/// leader1 (pid 111) has signal != 0; the thread ring head →
/// thread2_node → head holds a thread with the target pid.
#[test]
fn find_task_by_pid_hit_non_leader_thread() {
    let mut buf = vec![0u8; 0x4000];
    paint_valid_task(&mut buf, LEADER1_PA as usize, 111);
    // A second task_struct acting as a non-leader thread of leader1.
    const THREAD_PA: u64 = 0x1800;
    paint_valid_task(&mut buf, THREAD_PA as usize, 333);

    // Leader ring: head → leader1 → head.
    let head_link_pa = INIT_TASK_PA + synth_task::TASKS_OFF as u64;
    put_u64(&mut buf, head_link_pa, leader_node_kva(LEADER1_PA));
    put_u64(
        &mut buf,
        LEADER1_PA + synth_task::TASKS_OFF as u64,
        init_task_kva() + synth_task::TASKS_OFF as u64,
    );

    // leader1.signal → a signal_struct laid out at PA SIGNAL_STRUCT_PA.
    const SIGNAL_STRUCT_PA: u64 = 0x2000;
    let signal_kva = PAGE_OFFSET + SIGNAL_STRUCT_PA;
    put_u64(
        &mut buf,
        LEADER1_PA + synth_task::SIGNAL_OFF as u64,
        signal_kva,
    );

    // signal->thread_head (PA = SIGNAL_STRUCT_PA + SIGNAL_THREAD_HEAD_OFF)
    // is the list anchor; thread linkage is task.thread_node.
    let thread_head_kva = signal_kva + synth_task::SIGNAL_THREAD_HEAD_OFF as u64;
    let thread_head_pa = SIGNAL_STRUCT_PA + synth_task::SIGNAL_THREAD_HEAD_OFF as u64;
    // thread_node KVA of a task at PA `pa`.
    let thread_node_kva = |pa: u64| PAGE_OFFSET + pa + synth_task::THREAD_NODE_OFF as u64;

    // Thread ring: thread_head → leader1.thread_node → thread2.thread_node
    // → thread_head. The leader's own thread_node is on the list and is
    // skipped (already checked as leader); thread2 carries pid 333.
    put_u64(&mut buf, thread_head_pa, thread_node_kva(LEADER1_PA));
    put_u64(
        &mut buf,
        LEADER1_PA + synth_task::THREAD_NODE_OFF as u64,
        thread_node_kva(THREAD_PA),
    );
    put_u64(
        &mut buf,
        THREAD_PA + synth_task::THREAD_NODE_OFF as u64,
        thread_head_kva,
    );

    let kernel = build_test_kernel(&mut buf, Default::default());
    let offs = synth_validation_offsets();
    let got = find_task_by_pid(&kernel, init_task_kva(), 0, &offs, 333)
        .expect("non-leader thread pid 333 must be found");
    assert_eq!(got, PAGE_OFFSET + THREAD_PA, "must return the thread's KVA");
}
