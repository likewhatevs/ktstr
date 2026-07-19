use super::super::*;

// ---- Shared synthetic-guest-memory fixtures ----
//
// These build a host-side `GuestKernel` over a caller-owned buffer so the
// dispatcher's production read/write paths run with no VM boot and no
// BTF. Reused by `task_field` (validation chain) and `dispatch_arms`
// (Symbol/Direct/Kva arms + batch + walker).

/// Direct-map base wired into [`build_test_kernel`]: PA 0 maps to KVA
/// `0xFFFF_8880_0000_0000` (the real x86_64 4-level direct-map base
/// for slab-allocated `task_struct`s). Slab/Direct KVAs are
/// `PAGE_OFFSET + pa`; the Symbol path translates via
/// [`crate::monitor::symbols::START_KERNEL_MAP`] instead.
pub(super) const PAGE_OFFSET: u64 = 0xFFFF_8880_0000_0000;

/// `ext_sched_class` KVA the validator's L6 pointer-identity check
/// compares against. Arbitrary kernel-half value; tests that paint a
/// valid task store this in the `sched_class` field.
pub(super) const EXT_KVA: u64 = 0xFFFF_FFFF_8200_0100;

/// Default `task->start_time` the valid-task painter stamps. Tests
/// drive the L2 identity window around this value.
pub(super) const DEFAULT_START_TIME: u64 = 1_700_000_000_000;

/// Synthetic `task_struct` field offsets used by every fixture below.
/// Real kernel offsets are BTF-derived at runtime; these synthetic
/// values just need to be (a) distinct, (b) non-overlapping for the
/// 32/64-bit reads, and (c) within the test buffer.
pub(super) mod synth_task {
    // Visible across the `tests` module's sibling test files
    // (`task_field`, `dispatch_arms`) via the explicit ancestor path,
    // matching the repo's `pub(in crate::...)` idiom.
    pub(in crate::vmm::freeze_coord::kernel_op_dispatch::tests) const PID_OFF: usize = 0x10;
    pub(in crate::vmm::freeze_coord::kernel_op_dispatch::tests) const START_TIME_OFF: usize = 0x18;
    pub(in crate::vmm::freeze_coord::kernel_op_dispatch::tests) const STATE_OFF: usize = 0x20;
    pub(in crate::vmm::freeze_coord::kernel_op_dispatch::tests) const ON_RQ_OFF: usize = 0x28;
    pub(in crate::vmm::freeze_coord::kernel_op_dispatch::tests) const SCHED_CLASS_OFF: usize = 0x30;
    pub(in crate::vmm::freeze_coord::kernel_op_dispatch::tests) const START_BOOTTIME_OFF: usize =
        0x40;
    pub(in crate::vmm::freeze_coord::kernel_op_dispatch::tests) const SCX_DSQ_OFF: usize = 0x48;
    pub(in crate::vmm::freeze_coord::kernel_op_dispatch::tests) const SCX_RUNNABLE_NODE_OFF: usize =
        0x50;
    pub(in crate::vmm::freeze_coord::kernel_op_dispatch::tests) const TASKS_OFF: usize = 0x60;
    pub(in crate::vmm::freeze_coord::kernel_op_dispatch::tests) const SIGNAL_OFF: usize = 0x70;
    pub(in crate::vmm::freeze_coord::kernel_op_dispatch::tests) const SIGNAL_THREAD_HEAD_OFF:
        usize = 0x10;
    pub(in crate::vmm::freeze_coord::kernel_op_dispatch::tests) const THREAD_NODE_OFF: usize = 0x78;
}

/// [`TaskValidationOffsets`] populated from [`synth_task`]. Passed to
/// `validate_task_for_field_op` and `find_task_by_pid` in place of the
/// BTF-resolved offsets.
pub(super) fn synth_validation_offsets() -> TaskValidationOffsets {
    TaskValidationOffsets {
        pid: IntegerField {
            offset: synth_task::PID_OFF,
            width: 4,
        },
        start_time: IntegerField {
            offset: synth_task::START_TIME_OFF,
            width: 8,
        },
        state: IntegerField {
            offset: synth_task::STATE_OFF,
            width: 4,
        },
        on_rq: IntegerField {
            offset: synth_task::ON_RQ_OFF,
            width: 4,
        },
        sched_class: synth_task::SCHED_CLASS_OFF,
        start_boottime: IntegerField {
            offset: synth_task::START_BOOTTIME_OFF,
            width: 8,
        },
        scx_dsq: synth_task::SCX_DSQ_OFF,
        scx_runnable_node: synth_task::SCX_RUNNABLE_NODE_OFF,
        tasks: synth_task::TASKS_OFF,
        signal: synth_task::SIGNAL_OFF,
        signal_thread_head: synth_task::SIGNAL_THREAD_HEAD_OFF,
        thread_node: synth_task::THREAD_NODE_OFF,
    }
}

/// Paint a complete "live, SCX-managed, sleepable" task_struct at PA
/// `pa` into `buf`. After painting, the task passes all 8 validation
/// layers. Caller mutates individual fields to trigger specific
/// rejections. `signal` (offset [`synth_task::SIGNAL_OFF`]) is left 0
/// so [`find_task_by_pid`]'s thread-group walk is skipped.
pub(super) fn paint_valid_task(buf: &mut [u8], pa: usize, pid: u32) {
    // pid (u32 LE)
    buf[pa + synth_task::PID_OFF..pa + synth_task::PID_OFF + 4].copy_from_slice(&pid.to_le_bytes());
    // start_time = arbitrary
    buf[pa + synth_task::START_TIME_OFF..pa + synth_task::START_TIME_OFF + 8]
        .copy_from_slice(&DEFAULT_START_TIME.to_le_bytes());
    // state = TASK_RUNNING (0) — not TASK_DEAD
    buf[pa + synth_task::STATE_OFF..pa + synth_task::STATE_OFF + 4]
        .copy_from_slice(&0u32.to_le_bytes());
    // on_rq = 0 (sleeping)
    buf[pa + synth_task::ON_RQ_OFF..pa + synth_task::ON_RQ_OFF + 4]
        .copy_from_slice(&0u32.to_le_bytes());
    // sched_class = EXT_KVA
    buf[pa + synth_task::SCHED_CLASS_OFF..pa + synth_task::SCHED_CLASS_OFF + 8]
        .copy_from_slice(&EXT_KVA.to_le_bytes());
    // start_boottime = arbitrary non-zero (1 hour in ns)
    buf[pa + synth_task::START_BOOTTIME_OFF..pa + synth_task::START_BOOTTIME_OFF + 8]
        .copy_from_slice(&3_600_000_000_000u64.to_le_bytes());
    // scx.dsq = NULL (not queued)
    buf[pa + synth_task::SCX_DSQ_OFF..pa + synth_task::SCX_DSQ_OFF + 8]
        .copy_from_slice(&0u64.to_le_bytes());
    // scx.runnable_node = list-empty (next points at self)
    // task_kva = task_pa + PAGE_OFFSET; runnable_node KVA =
    // task_kva + SCX_RUNNABLE_NODE_OFF.
    let task_kva = (pa as u64).wrapping_add(PAGE_OFFSET);
    let self_kva = task_kva.wrapping_add(synth_task::SCX_RUNNABLE_NODE_OFF as u64);
    buf[pa + synth_task::SCX_RUNNABLE_NODE_OFF..pa + synth_task::SCX_RUNNABLE_NODE_OFF + 8]
        .copy_from_slice(&self_kva.to_le_bytes());
}

/// Build a synthetic `GuestKernel`; PAGE_OFFSET set so PA 0 → KVA
/// [`PAGE_OFFSET`] (matches the real direct-map layout for
/// slab-allocated task_structs). `cr3_pa = 0` and `l5 = false`: the
/// direct-map fast path resolves Symbol/Direct/slab KVAs by
/// subtraction, while Kva targets fall through to the page-table walk
/// (which returns `None` against the zeroed page-table region).
pub(super) fn build_test_kernel(
    buf: &mut [u8],
    symbols: std::collections::HashMap<String, u64>,
) -> crate::monitor::guest::GuestKernel {
    // SAFETY: buf outlives the kernel because the caller's owning
    // `Vec` is dropped after the returned `GuestKernel`; GuestMem::new
    // requires the backing buffer remain valid for the GuestMem's
    // lifetime.
    let mem = unsafe {
        std::sync::Arc::new(crate::monitor::reader::GuestMem::new(
            buf.as_mut_ptr(),
            buf.len() as u64,
        ))
    };
    crate::monitor::guest::GuestKernel::new_for_test(
        mem,
        symbols,
        PAGE_OFFSET,
        0,     // cr3_pa unused for direct-map translation
        false, // l5 = false (4-level)
    )
}
