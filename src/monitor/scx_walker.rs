//! Host-side rq->scx + DSQ + task enumeration walkers for the
//! failure dump.
//!
//! Entry points:
//!
//! 1. [`walk_rq_scx`] — for one CPU's `struct rq.scx`, captures the
//!    scalar fields the kernel's own `scx_dump_state` reads
//!    (nr_running, flags, cpu_released, ops_qseq, kick_sync) plus
//!    nr_immed, clock, and the curr task pid+comm. Walks
//!    `rq->scx.runnable_list` and emits a list of
//!    [`super::dump::TaskWalkerEntry`] tuples for each runnable task —
//!    these feed directly into the per-task enrichment capture
//!    pipeline.
//!
//! 2. [`walk_local_dsqs`] — per-CPU local DSQs at
//!    `rq->scx.local_dsq`. Runs unconditionally — local DSQs are
//!    initialized at boot (`init_dsq` at `kernel/sched/ext.c:7772`
//!    for every possible CPU) and exist whether or not a scheduler
//!    is attached, so this surfaces local-DSQ state even when
//!    `*scx_root == 0`.
//!
//! 3. [`walk_dsqs`] — sched-rooted DSQs reachable from `*scx_root`
//!    (excluding per-CPU local DSQs, which [`walk_local_dsqs`]
//!    handles separately):
//!    - per-CPU bypass DSQs via `scx_sched_pcpu.bypass_dsq`
//!    - per-node global DSQs via `scx_sched.pnode[node]->global_dsq`
//!    - user-allocated DSQs via the `scx_sched.dsq_hash` rhashtable
//!
//!    For each DSQ captures the scalar state (id, nr, seq) and walks
//!    its `list_head` to enumerate queued tasks. The kernel's own
//!    `scx_dump_state` does NOT enumerate per-DSQ depths — this
//!    walker surfaces queue depth and per-task ordering that the
//!    in-tree dump path does not.
//!
//! 4. [`walk_scx_tasks_global`] — walks the kernel's global
//!    `scx_tasks` LIST_HEAD via each task's `scx.tasks_node`.
//!    Surfaces every task owned by an scx_sched, surviving the
//!    per-rq runnable_list drain that scheduler teardown
//!    (`scx_bypass`, `kernel/sched/ext.c:5304-5404`) triggers.
//!
//! All walkers are best-effort: any address that fails to translate
//! (slab page race, PA out of bounds) yields a partial result rather
//! than aborting. Cycle protection is per-list (MAX_NODES_PER_LIST);
//! the rhashtable walk caps total bucket-table chain length at
//! MAX_RHT_NODES.
//!
//! # Lock-free reads
//!
//! These walkers run from the freeze coordinator after the vCPU
//! rendezvous. All vCPUs are parked at a known KVM exit; the host
//! reads guest memory directly with no in-guest synchronization. The
//! kernel-side locks (`scx_dispatch_q.lock` raw_spinlock,
//! `rhashtable.mutex`) are not honored — the freeze rendezvous IS
//! the synchronization. A torn read can still happen if a vCPU was
//! mid-write at the freeze instant; the walker treats torn results
//! as best-effort partial output.

use serde::{Deserialize, Serialize};

use super::btf_offsets::{RHT_PTR_LOCK_BIT, SCX_DSQ_LNODE_ITER_CURSOR, ScxWalkerOffsets};
use super::dump::TaskWalkerEntry;
use super::guest::GuestKernel;
use super::idr::translate_any_kva;
use super::reader::{GuestMem, WalkContext};

/// Maximum entries any single list_head walk visits before bailing
/// with what's been collected. Bounds CPU + memory cost on a
/// corrupt-pointer chain that loops back on itself or runs into
/// arbitrary slab. 4096 is generous: real per-CPU runnable_list has
/// at most ~num_threads entries on a given CPU, capped well below
/// this; user DSQs can in principle hold millions of tasks but the
/// per-DSQ walker still bails at this cap so a million-entry DSQ
/// surfaces 4096 task entries plus the `nr` count (truncation
/// surfaces via `truncated: true` on [`DsqState`]).
const MAX_NODES_PER_LIST: u32 = 4096;

/// Maximum total node visits across all rhashtable buckets in the
/// `dsq_hash` walk. Bounds the cost of a runaway bucket chain.
/// Mainline ScxLib creates at most a few hundred user DSQs.
const MAX_RHT_NODES: u32 = 8192;

/// Maximum number of buckets the rhashtable walker enumerates.
/// `bucket_table.size` is normally a small power of two
/// (rhashtable starts at 16, grows by 2x); a pathological torn
/// read could surface a huge value. Caps the bucket walk at 64K to
/// protect freeze-path latency.
const MAX_RHT_BUCKETS: u32 = 65_536;

/// Maximum nodes any single rhashtable bucket chain visits before
/// bailing. A healthy rhashtable holds ~1 element per bucket on
/// average; a pathological chain of 1024 entries in one bucket is
/// orders of magnitude beyond legitimate use and almost certainly
/// indicates a corrupted `next` chain or torn read. Bounded
/// independently of [`MAX_RHT_NODES`] so a single runaway bucket
/// cannot starve the walk's per-bucket budget on the way to the
/// global cap. The condition `chain_visited < PER_BUCKET_CHAIN_CAP`
/// admits exactly 1024 body executions: chain_visited starts at 0
/// and increments inside the loop body, so the comparison reads
/// 0,1,...,1023 across the 1024 iterations and exits on the next
/// check (1024 < 1024 is false).
const PER_BUCKET_CHAIN_CAP: u32 = 1024;

/// Snapshot of one CPU's `struct rq.scx` state at freeze time.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[non_exhaustive]
pub struct RqScxState {
    /// CPU index (0-based) this state describes.
    pub cpu: u32,
    /// `rq->scx.nr_running`.
    pub nr_running: u32,
    /// `rq->scx.flags`.
    pub flags: u32,
    /// `rq->scx.cpu_released` — `true` when the kernel released
    /// the CPU back to the BPF scheduler (see `scx_pre_release_cpu`
    /// in kernel/sched/ext.c).
    pub cpu_released: bool,
    /// `rq->scx.ops_qseq`.
    pub ops_qseq: u64,
    /// `rq->scx.kick_sync` — present on post-v7.0-rc5 kernels. None
    /// when the BTF lookup of the field returns absent (v6.14 and
    /// v7.0 release-line layouts predate the `kick_sync` member).
    /// Skipped on serde when None so older dumps stay tight.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kick_sync: Option<u64>,
    /// `rq->scx.nr_immed` — count of ENQ_IMMED tasks on local_dsq.
    /// Same kernel-version provenance as [`Self::kick_sync`]: the
    /// field is post-v7.0-rc5 and absent on the v6.14/v7.0 CI matrix.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub nr_immed: Option<u32>,
    /// `rq->scx.clock` — per-CPU scx_rq clock (the value
    /// `scx_bpf_now()` returns) at the freeze instant. Optional
    /// because the field was added by the `scx_bpf_now()` series in
    /// v6.14 (commit 3a9910b5904d); v6.12 and v6.13 release kernels
    /// have no equivalent member on `struct scx_rq`. None when the
    /// BTF lookup of `rq->scx.clock` resolves absent — consumers
    /// that need the value gate on Some.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rq_clock: Option<u64>,
    /// `rq->curr->pid` — the currently-running task. `None` when
    /// the curr pointer didn't translate (idle or torn read).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub curr_pid: Option<i32>,
    /// `rq->curr->comm`. Mirrors `curr_pid`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub curr_comm: Option<String>,
    /// `task_struct` KVAs of every entry walked off
    /// `rq->scx.runnable_list`. The freeze coordinator passes this
    /// vec into the per-task enrichment capture so the same
    /// task list drives both rq->scx state AND per-task records.
    pub runnable_task_kvas: Vec<u64>,
    /// True when the runnable_list walk hit the
    /// `MAX_NODES_PER_LIST` safety cap before reaching the head
    /// — typical only on a corrupted chain.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub runnable_truncated: bool,
}

/// Snapshot of one DSQ's state — built-in (per-CPU local, per-CPU
/// bypass, per-node global) or user-allocated.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[non_exhaustive]
pub struct DsqState {
    /// `scx_dispatch_q.id` — built-in DSQs use synthetic ids
    /// (`SCX_DSQ_LOCAL`, `SCX_DSQ_GLOBAL` per-node); user DSQs
    /// carry the BPF-allocated id.
    pub id: u64,
    /// Operator-facing tag describing where the DSQ came from:
    /// `"local cpu N"`, `"bypass cpu N"`, `"global node N"`, or
    /// `"user"`. Aligned with the kernel's own `scx_dump_state`
    /// terminology where comparable.
    pub origin: String,
    /// `scx_dispatch_q.nr` — number of tasks currently queued.
    pub nr: u32,
    /// `scx_dispatch_q.seq` — BPF-iter seq counter, used by the
    /// dual-snapshot delta to distinguish dead vs busy DSQs:
    /// `Δnr=0 + Δseq=0` is a dead DSQ; `Δseq>>Δ(seq-nr)` indicates
    /// unbounded queue growth.
    pub seq: u32,
    /// `task_struct` KVAs walked off the DSQ's `list_head`. Same
    /// shape as [`RqScxState::runnable_task_kvas`] — feeds into
    /// the same per-task enrichment pipeline.
    pub task_kvas: Vec<u64>,
    /// True when the DSQ list walk hit the
    /// `MAX_NODES_PER_LIST` cap before reaching the head.
    /// Distinct from `nr`: the kernel may report `nr` larger than
    /// our walk cap on legitimately-deep DSQs.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub truncated: bool,
}

/// Top-level scheduler state captured from `*scx_root`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ScxSchedState {
    /// `scx_sched.aborting`. `true` when the scheduler is in the
    /// abort path; `bypass_depth` typically rises here.
    pub aborting: bool,
    /// `scx_sched.bypass_depth`. Nesting depth of the bypass-mode
    /// stack; non-zero means the kernel is dispatching tasks
    /// without consulting the BPF scheduler.
    pub bypass_depth: i32,
    /// `scx_sched.exit_kind` — the SCX_EXIT_* enum value latched
    /// at `scx_error()` time. 0 means no exit yet; non-zero values
    /// match `enum scx_exit_kind` in
    /// `include/linux/sched/ext.h`.
    pub exit_kind: u32,
    /// `scx_sched.watchdog_timeout` (jiffies) at the snapshot
    /// instant. `None` when the field was not captured — either
    /// because the live `read_scx_sched_state` path was taken on a
    /// kernel that still exposes `watchdog_timeout` only via the
    /// monitor's `WatchdogOverride` plumbing (not as a BTF field on
    /// every release), or because the BPF .bss fallback was used
    /// without the snapshot var set. Some when populated via the
    /// probe BPF .bss snapshot
    /// (`ktstr_exit_watchdog_timeout`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub watchdog_timeout: Option<u64>,
    /// Provenance tag identifying which path produced this state.
    /// `None` for the default-built / serde-deserialized case where
    /// the source isn't recorded; `Some("live")` when populated by
    /// `read_scx_sched_state` reading `*scx_root` directly;
    /// `Some("bss_snapshot")` when populated from the probe BPF
    /// .bss snapshot fallback (the `ktstr_exit_*` vars). Lets the
    /// dump consumer distinguish "scheduler was alive at freeze
    /// time" from "scheduler had already torn down and we read the
    /// pre-teardown snapshot the BPF probe latched".
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
    /// Kernel virtual address of the `scx_sched` instance these
    /// values describe. `None` when not captured. Same provenance
    /// rule as [`Self::source`]: live path stamps the resolved
    /// `*scx_root` value; the BPF .bss snapshot stamps the
    /// `ktstr_exit_sched_kva` field. Lets a consumer correlate
    /// dumps across reloads (a different scx_sched instance has a
    /// different KVA).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sched_kva: Option<u64>,
}

/// Walk one CPU's `rq->scx` state. Reads the scalar fields and
/// the runnable_list, returning [`RqScxState`] plus a vec of
/// [`TaskWalkerEntry`] entries for the per-task enrichment pipeline.
///
/// `cpu` is the 0-based CPU index; `rq_kva` and `rq_pa` address that
/// CPU's `struct rq`. Caller resolves both via
/// `runqueues + per_cpu_offset[cpu]` (KVA) plus the corresponding PA
/// via `compute_rq_pas`.
///
/// Cap on visited nodes: `MAX_NODES_PER_LIST`. A truncated walk
/// surfaces via [`RqScxState::runnable_truncated`].
///
/// Returns `None` when any of the offset sub-groups required for
/// scalar reads (`rq`, `scx_rq`, `task`) is absent — the walker
/// cannot synthesize partial scalars meaningfully without the rq /
/// scx_rq base offsets. Per-CPU runnable_list walking additionally
/// requires `see` (sched_ext_entity); when only `see` is missing the
/// scalar capture still lands but the runnable_list walk yields
/// nothing.
#[allow(dead_code)]
pub fn walk_rq_scx(
    kernel: &GuestKernel,
    cpu: u32,
    rq_kva: u64,
    rq_pa: u64,
    offsets: &ScxWalkerOffsets,
) -> Option<(RqScxState, Vec<TaskWalkerEntry>)> {
    let rq_offs = offsets.rq.as_ref()?;
    let scx_rq_offs = offsets.scx_rq.as_ref()?;
    let task_offs = offsets.task.as_ref()?;

    let mem = kernel.mem();
    let walk = kernel.walk_context();

    let scx_off = rq_offs.scx;

    // Scalar reads off rq + scx_rq.
    let nr_running = mem.read_u32(rq_pa, scx_off + scx_rq_offs.nr_running);
    let flags = mem.read_u32(rq_pa, scx_off + scx_rq_offs.flags);
    let cpu_released = mem.read_u8(rq_pa, scx_off + scx_rq_offs.cpu_released) != 0;
    let ops_qseq = mem.read_u64(rq_pa, scx_off + scx_rq_offs.ops_qseq);
    // kick_sync / nr_immed are post-v7.0-rc5 fields; offsets resolve
    // as None on v6.14 and v7.0 release-line BTFs. Gate the read on
    // Some so we don't fabricate a u64/u32 from rq_pa+0 (which would
    // alias the local_dsq head pointer — a non-zero garbage read
    // that the dump would render as legitimate kernel state).
    let kick_sync = scx_rq_offs
        .kick_sync
        .map(|off| mem.read_u64(rq_pa, scx_off + off));
    let nr_immed = scx_rq_offs
        .nr_immed
        .map(|off| mem.read_u32(rq_pa, scx_off + off));
    // rq->scx.clock added in v6.14 (commit 3a9910b5904d). Gate the
    // read on Some(off): on v6.12/v6.13 the offset is None and the
    // walker must NOT fall back to rq_pa+0 (would alias local_dsq's
    // raw_spinlock — non-zero junk rendered as a legitimate clock
    // reading). The downstream RqScxState carries an Option<u64> so
    // the JSON elides scx_rq_clock on unsupported kernels.
    let rq_clock = scx_rq_offs
        .clock
        .map(|off| mem.read_u64(rq_pa, scx_off + off));

    // curr task — pointer follow.
    let curr_kva = mem.read_u64(rq_pa, rq_offs.curr);
    let (curr_pid, curr_comm) =
        read_task_pid_comm(mem, walk, curr_kva, task_offs.pid, task_offs.comm);

    // Walk runnable_list when sched_ext_entity offsets are available.
    // Without `see` we can still report scalar state but cannot
    // container_of a runnable_node back to its task_struct.
    let (runnable_task_kvas, runnable_truncated) = if let Some(see_offs) = offsets.see.as_ref() {
        let list_head_off = scx_off + scx_rq_offs.runnable_list;
        let head_kva = rq_kva.wrapping_add(list_head_off as u64);
        let head_pa = rq_pa.wrapping_add(list_head_off as u64);

        // container_of offset within task_struct: each runnable_node
        // is at task + task_struct.scx + see.runnable_node.
        let runnable_node_off_in_task = task_offs.scx + see_offs.runnable_node;

        walk_list_head_for_task_kvas(mem, walk, head_kva, head_pa, runnable_node_off_in_task)
    } else {
        (Vec::new(), false)
    };

    let walker_entries: Vec<TaskWalkerEntry> = runnable_task_kvas
        .iter()
        .map(|&task_kva| TaskWalkerEntry {
            task_kva,
            // Runnable on this CPU's scx — eligible for the
            // pi_boosted_out_of_scx flag.
            is_runnable_in_scx: true,
            // running_pc only known for the curr task; the
            // freeze coordinator can fill that via
            // VcpuRegSnapshot.instruction_pointer at a higher
            // level. The walker leaves it None.
            running_pc: None,
        })
        .collect();

    let state = RqScxState {
        cpu,
        nr_running,
        flags,
        cpu_released,
        ops_qseq,
        kick_sync,
        nr_immed,
        rq_clock,
        curr_pid,
        curr_comm,
        runnable_task_kvas,
        runnable_truncated,
    };

    Some((state, walker_entries))
}

/// Read scalar `scx_sched` fields off `*scx_root`.
///
/// `scx_root` is a kernel-text-mapped pointer at the resolved KVA;
/// `*scx_root` points at the active `struct scx_sched`. Returns
/// `None` when scx_root is unset (no scheduler attached), the read
/// fails, or the `scx_sched` offset sub-group is missing from BTF.
///
/// Emits `tracing::debug!` at each gate that returns `None` so an
/// operator parsing the failure-dump trace can pinpoint exactly
/// where the read aborted: BTF sub-group missing, scx_root_kva
/// zero, dereferenced sched_kva zero, or sched_kva translate
/// failure.
#[allow(dead_code)]
pub fn read_scx_sched_state(
    kernel: &GuestKernel,
    scx_root_kva: u64,
    offsets: &ScxWalkerOffsets,
) -> Option<(u64, ScxSchedState)> {
    let Some(sched_offs) = offsets.sched.as_ref() else {
        tracing::debug!(
            "read_scx_sched_state: ScxSchedOffsets BTF sub-group missing — \
             vmlinux lacks `struct scx_sched` (kernel without sched_ext or stripped vmlinux)",
        );
        return None;
    };

    let mem = kernel.mem();
    let walk = kernel.walk_context();

    if scx_root_kva == 0 {
        tracing::debug!(
            "read_scx_sched_state: scx_root_kva is 0 — vmlinux had no \
             `scx_root` symbol (pre-6.16 kernel or stripped vmlinux)",
        );
        return None;
    }

    let root_pa = kernel.text_kva_to_pa(scx_root_kva);
    let sched_kva = mem.read_u64(root_pa, 0);
    if sched_kva == 0 {
        tracing::debug!(
            scx_root_kva = format_args!("{:#x}", scx_root_kva),
            root_pa = format_args!("{:#x}", root_pa),
            "read_scx_sched_state: *scx_root == 0 — no scheduler attached at the freeze instant",
        );
        return None;
    }
    let Some(sched_pa) = translate_any_kva(
        mem,
        walk.cr3_pa,
        walk.page_offset,
        sched_kva,
        walk.l5,
        walk.tcr_el1,
    ) else {
        tracing::debug!(
            sched_kva = format_args!("{:#x}", sched_kva),
            "read_scx_sched_state: translate_any_kva failed for sched_kva — \
             page-table walk yielded no PA (slab page race or torn read)",
        );
        return None;
    };

    // `aborting` and `bypass_depth` are dev-only fields (absent on
    // every release tag in our supported range). Gate each read on
    // its offset being Some — falling back to 0 / false matches the
    // semantics of reading from a kernel that never had the field
    // (no in-flight abort, no bypass nesting). The downstream
    // ScxSchedState carries plain bool/i32 because those defaults
    // are meaningful and serializable; an Option wrapper would just
    // complicate every consumer for no extra signal on release
    // kernels.
    let aborting = sched_offs
        .aborting
        .map(|off| mem.read_u8(sched_pa, off) != 0)
        .unwrap_or(false);
    let bypass_depth = sched_offs
        .bypass_depth
        .map(|off| mem.read_u32(sched_pa, off) as i32)
        .unwrap_or(0);
    // `exit_kind` is `atomic_t`; the value lives in the `counter`
    // field at offset 0 of atomic_t. We're already at the
    // outer-struct offset of `exit_kind`, so a u32 read at that
    // offset reads the `counter` directly. Mandatory on every
    // kernel that has `scx_sched`.
    let exit_kind = mem.read_u32(sched_pa, sched_offs.exit_kind);

    Some((
        sched_kva,
        ScxSchedState {
            aborting,
            bypass_depth,
            exit_kind,
            // Live read from `*scx_root` doesn't capture
            // `watchdog_timeout`. The BTF sub-group does not carry
            // an offset for the field today (it would need to be
            // added to `ScxSchedOffsets`), and the host tracks the
            // configured timeout via the `WatchdogOverride` plumbing
            // anyway. Leave None; the BPF .bss snapshot's
            // `ktstr_exit_watchdog_timeout` path populates this when
            // the live read is unavailable.
            watchdog_timeout: None,
            source: Some(SCX_SCHED_STATE_SOURCE_LIVE.to_string()),
            sched_kva: Some(sched_kva),
        },
    ))
}

/// Provenance tag for [`ScxSchedState::source`] when the state was
/// read directly from `*scx_root` via `read_scx_sched_state`. The
/// scheduler was alive at freeze time and the host walked its slab
/// page directly. Pinned as a constant so the dump's display layer
/// and tests reference the same string without drift.
pub const SCX_SCHED_STATE_SOURCE_LIVE: &str = "live";

/// Provenance tag for [`ScxSchedState::source`] when the state was
/// reconstructed from the probe BPF program's `.bss` snapshot
/// (`ktstr_exit_*` vars). The scheduler had already torn down by
/// freeze time (`*scx_root == 0`), so the live walker returned None
/// and the host fell back to the snapshot the BPF tp_btf handler
/// captured at err-exit time.
pub const SCX_SCHED_STATE_SOURCE_BSS: &str = "bss_snapshot";

/// `SCX_TASK_CURSOR` flag value (`1 << 31`) on `sched_ext_entity.flags`.
/// Cursor entries are stack-allocated `sched_ext_entity` placeholders
/// that `scx_task_iter_start` (`kernel/sched/ext.c:843-846`) inserts
/// into `scx_tasks` to mark the iterator's progress; they are NOT
/// embedded in any `task_struct` so the global walker must skip them
/// to avoid container_of producing a bogus task KVA. Pinned per
/// `include/linux/sched/ext.h:142::SCX_TASK_CURSOR`.
const SCX_TASK_CURSOR: u32 = 1 << 31;
/// Walk the kernel's global `scx_tasks` LIST_HEAD and recover every
/// task linked into it via `task_struct.scx.tasks_node`.
///
/// `scx_tasks` is `static LIST_HEAD(scx_tasks)` at
/// `kernel/sched/ext.c:47`. Tasks are added on
/// `scx_init_task` (`kernel/sched/ext.c:3742` —
/// `list_add_tail(&p->scx.tasks_node, &scx_tasks)`) and removed on
/// `sched_ext_dead` (`kernel/sched/ext.c:3803` —
/// `list_del_init(&p->scx.tasks_node)`). The list outlives the
/// per-rq `runnable_list` because `scx_bypass`
/// (`kernel/sched/ext.c:5304-5404`) drains runnable_list during
/// scheduler teardown without touching `scx_tasks` — making this
/// the durable task source for failure-dump enrichment.
///
/// Cursor entries (`scx_task_iter_start` inserts a stack-allocated
/// `sched_ext_entity` with `flags = SCX_TASK_CURSOR` into
/// `scx_tasks` while iterating) are skipped via the
/// `tasks_node_off_in_see` parameter — the walker reads
/// `sched_ext_entity.flags` for each list entry and skips entries
/// whose flag is set.
///
/// `scx_tasks_kva` is the symbol KVA of the global LIST_HEAD;
/// `tasks_node_off_in_task` is the byte offset of `tasks_node`
/// within `task_struct` (`task.scx + see.tasks_node`);
/// `tasks_node_off_in_see` is the byte offset of `tasks_node`
/// within `sched_ext_entity` (`see.tasks_node` alone — used to
/// recover the see base for cursor-flag testing on entries that
/// are not embedded in a `task_struct`); `flags_off_in_see` is the
/// byte offset of `flags` within `sched_ext_entity`.
///
/// Returns an empty vec when `scx_tasks_kva` is 0 (symbol absent —
/// stripped vmlinux or kernel without sched_ext) or when the list
/// head reads as empty (tasks_node points at itself).
///
/// Bounded by `MAX_NODES_PER_LIST` to protect against a corrupt
/// chain.
#[allow(dead_code)]
pub fn walk_scx_tasks_global(
    kernel: &GuestKernel,
    scx_tasks_kva: u64,
    tasks_node_off_in_task: usize,
    tasks_node_off_in_see: usize,
    flags_off_in_see: usize,
) -> Vec<u64> {
    if scx_tasks_kva == 0 {
        tracing::debug!(
            "walk_scx_tasks_global: scx_tasks_kva is 0 — vmlinux had no \
             `scx_tasks` symbol (kernel without sched_ext or stripped vmlinux)",
        );
        return Vec::new();
    }
    let mem = kernel.mem();
    let walk = kernel.walk_context();

    // The LIST_HEAD lives in the kernel text/.data mapping; convert
    // KVA → PA via the GuestKernel's runtime kernel image base. The
    // first u64 at that PA is list_head.next (the LIST_HEAD struct's
    // first field).
    let head_kva = scx_tasks_kva;
    let head_pa = kernel.text_kva_to_pa(scx_tasks_kva);

    let mut task_kvas: Vec<u64> = Vec::new();
    let mut node_kva = mem.read_u64(head_pa, 0);
    if node_kva == 0 {
        tracing::debug!(
            scx_tasks_kva = format_args!("{:#x}", scx_tasks_kva),
            head_pa = format_args!("{:#x}", head_pa),
            "walk_scx_tasks_global: head.next read as 0 — list-head bytes \
             unmapped or torn read; no tasks harvested",
        );
        return task_kvas;
    }

    let mut visited: u32 = 0;
    while node_kva != head_kva {
        if visited >= MAX_NODES_PER_LIST {
            return task_kvas;
        }
        visited += 1;

        // Recover the sched_ext_entity base for this list entry so we
        // can read its `flags`. For task-embedded entries this base is
        // inside a task_struct (`task_kva + task.scx`); for cursor
        // entries this base is a stack-allocated sched_ext_entity.
        // Either way, `see_kva = node_kva - see.tasks_node`.
        let see_kva = node_kva.wrapping_sub(tasks_node_off_in_see as u64);
        let cursor = match translate_any_kva(
            mem,
            walk.cr3_pa,
            walk.page_offset,
            see_kva,
            walk.l5,
            walk.tcr_el1,
        ) {
            Some(see_pa) => {
                let flags = mem.read_u32(see_pa, flags_off_in_see);
                flags & SCX_TASK_CURSOR != 0
            }
            // Translate failure on the see base — be conservative and
            // treat as not-cursor so the entry surfaces; downstream
            // walk_task_enrichment will revalidate via translate and
            // drop it cleanly if the address is bogus.
            None => false,
        };

        if !cursor {
            // container_of: task_kva = node_kva - tasks_node_off_in_task.
            let task_kva = node_kva.wrapping_sub(tasks_node_off_in_task as u64);
            task_kvas.push(task_kva);
        }

        // Advance to the next node via the list_head.next pointer
        // at offset 0 of the tasks_node list_head.
        let Some(node_pa) = translate_any_kva(
            mem,
            walk.cr3_pa,
            walk.page_offset,
            node_kva,
            walk.l5,
            walk.tcr_el1,
        ) else {
            return task_kvas;
        };
        let next_kva = mem.read_u64(node_pa, 0);
        if next_kva == 0 {
            return task_kvas;
        }
        node_kva = next_kva;
    }
    task_kvas
}

/// Walk every per-CPU local DSQ — the DSQs embedded in `rq->scx.local_dsq`.
///
/// This is a strict subset of [`walk_dsqs`]'s pass 1, extracted so
/// the dump path can call it INDEPENDENTLY of `*scx_root`. Per-CPU
/// local DSQs are kernel-initialized at boot (`init_dsq` from
/// `kernel/sched/ext.c:7772`, called for every possible CPU in the
/// `__init` path), so they exist even when no scheduler is attached
/// (`*scx_root == NULL`) and survive scheduler teardown's bypass
/// drain.
///
/// Returns one `DsqState` per CPU whose translate succeeds, plus a
/// flat vec of [`TaskWalkerEntry`] for the per-task enrichment
/// pipeline (these entries carry `is_runnable_in_scx: false` —
/// tasks queued on a DSQ are staged for dispatch, not yet runnable
/// in the rq->scx sense).
///
/// `rq_kvas`, `rq_pas`, and `per_cpu_offsets` index by CPU id
/// (parallel arrays, same shape `walk_rq_scx` consumes). The walker
/// skips BSS-zero-tail CPUs by checking the per-CPU offset directly
/// (`per_cpu_offsets[cpu] == 0 && cpu > 0`) — those entries fall out
/// of un-written `__per_cpu_offset[]` slots past `nr_cpu_ids` and
/// would otherwise surface a phantom DSQ row at the bare `runqueues`
/// symbol KVA. Comparing the resolved `rq_kva` against `rq_kvas[0]`
/// would miss the alias on x86_64 SMP: `setup_per_cpu_areas`
/// (`arch/x86/kernel/setup_percpu.c`) writes
/// `__per_cpu_offset[cpu] = delta + pcpu_unit_offsets[cpu]` with
/// `delta = pcpu_base_addr - __per_cpu_start` non-zero, so CPU 0's
/// `rq_kva` is `runqueues + delta` while a BSS-zero-tail CPU's is
/// `runqueues + 0` — the two differ and `rq_kva == rq_kvas[0]` would
/// let the phantom row through. Mirrors the canonical `cpu_off == 0
/// && cpu_index > 0` guard
/// [`super::bpf_map::read_percpu_array_value`] applies for percpu
/// reads, expressed against the same `__per_cpu_offset[]` array.
///
/// Empty arrays mean "no CPUs walked successfully"; the caller's
/// freeze-path retry guard normally rejects empty inputs before
/// reaching this pass.
///
/// `None` return when any required offset sub-group is missing
/// (`rq`, `scx_rq`, `dsq`, `dsq_lnode`, `task`, `see` — the same
/// leaf set [`walk_dsqs`]'s pass 1 needs). A partial offset set
/// is the same gating condition that blinds every other DSQ pass.
#[allow(dead_code)]
pub fn walk_local_dsqs(
    kernel: &GuestKernel,
    rq_kvas: &[u64],
    rq_pas: &[u64],
    per_cpu_offsets: &[u64],
    offsets: &ScxWalkerOffsets,
) -> Option<(Vec<DsqState>, Vec<TaskWalkerEntry>)> {
    let Some(rq_offs) = offsets.rq.as_ref() else {
        tracing::debug!(
            "walk_local_dsqs: ScxWalkerOffsets.rq sub-group missing — \
             local DSQ pass blinded",
        );
        return None;
    };
    let Some(scx_rq_offs) = offsets.scx_rq.as_ref() else {
        tracing::debug!(
            "walk_local_dsqs: ScxWalkerOffsets.scx_rq sub-group missing — \
             local DSQ pass blinded",
        );
        return None;
    };
    let Some(dsq_offs) = offsets.dsq.as_ref() else {
        tracing::debug!(
            "walk_local_dsqs: ScxWalkerOffsets.dsq sub-group missing — \
             local DSQ pass blinded",
        );
        return None;
    };
    let Some(dsq_lnode_offs) = offsets.dsq_lnode.as_ref() else {
        tracing::debug!(
            "walk_local_dsqs: ScxWalkerOffsets.dsq_lnode sub-group missing — \
             local DSQ pass blinded",
        );
        return None;
    };
    let Some(task_offs) = offsets.task.as_ref() else {
        tracing::debug!(
            "walk_local_dsqs: ScxWalkerOffsets.task sub-group missing — \
             local DSQ pass blinded",
        );
        return None;
    };
    let Some(see_offs) = offsets.see.as_ref() else {
        tracing::debug!(
            "walk_local_dsqs: ScxWalkerOffsets.see sub-group missing — \
             local DSQ pass blinded",
        );
        return None;
    };

    let mem = kernel.mem();
    let walk = kernel.walk_context();

    let mut states: Vec<DsqState> = Vec::new();
    let mut entries: Vec<TaskWalkerEntry> = Vec::new();

    for (cpu, (&rq_kva, &rq_pa)) in rq_kvas.iter().zip(rq_pas.iter()).enumerate() {
        // BSS-zero-tail guard: kernel `setup_per_cpu_areas`
        // only writes `__per_cpu_offset[cpu]` for CPUs in
        // `for_each_possible_cpu`, leaving slots beyond
        // `nr_cpu_ids` at the BSS-initialized 0. The caller
        // builds rq_kvas via `runqueues + per_cpu_offset[cpu]`,
        // so a BSS-zero-tail entry produces an rq_kva of
        // `runqueues + 0` instead of CPU 0's
        // `runqueues + __per_cpu_offset[0]`. The two are NOT
        // equal on x86_64 SMP because
        // `__per_cpu_offset[0] = pcpu_base_addr - __per_cpu_start`
        // is non-zero (`arch/x86/kernel/setup_percpu.c`); a
        // resolved-rq_kva comparison would let the phantom
        // BSS-zero entry through. Check the per-CPU offset
        // directly instead — `cpu_off == 0 && cpu > 0` is the
        // canonical guard
        // [`super::bpf_map::read_percpu_array_value`] uses for
        // percpu reads and the matching guard at the per-CPU
        // bypass DSQ pass below. A `per_cpu_offsets` slice
        // shorter than `rq_kvas` (length-mismatched caller)
        // is treated conservatively: an absent offset for
        // `cpu > 0` skips the slot, since the walker can't
        // distinguish a real CPU from a BSS-zero tail without
        // the offset.
        let cpu_off = per_cpu_offsets.get(cpu).copied();
        match cpu_off {
            Some(off) if off == 0 && cpu > 0 => continue,
            None if cpu > 0 => continue,
            _ => {}
        }
        let local_dsq_off = rq_offs.scx + scx_rq_offs.local_dsq;
        let dsq_kva = rq_kva.wrapping_add(local_dsq_off as u64);
        let dsq_pa = rq_pa.wrapping_add(local_dsq_off as u64);
        if let Some((state, e)) = walk_one_dsq(
            mem,
            walk,
            dsq_kva,
            dsq_pa,
            || format!("local cpu {cpu}"),
            dsq_offs,
            dsq_lnode_offs,
            task_offs,
            see_offs,
        ) {
            entries.extend(e);
            states.push(state);
        }
    }

    Some((states, entries))
}

/// Walk every DSQ reachable from a `scx_sched` (the bypass / global
/// / user-hash passes — NOT per-CPU local DSQs) and produce one
/// `DsqState` per DSQ plus a flat vec of `TaskWalkerEntry` rows for
/// the per-task enrichment pipeline.
///
/// Per-CPU local DSQs (`rq->scx.local_dsq`) are NOT walked here —
/// they live in each rq independently of `*scx_root`, so callers
/// invoke [`walk_local_dsqs`] separately and unconditionally for
/// the local pass. This split lets the dump path surface local DSQ
/// state even when no scheduler is attached
/// (`*scx_root == NULL`) — the local_dsq struct is initialized at
/// boot per `init_dsq` (`kernel/sched/ext.c:7772`) for every
/// possible CPU, so it has well-defined contents long before any
/// scheduler attaches.
///
/// Walks (in this order, gated on the relevant sub-group offsets
/// being present):
///   1. Per-CPU bypass DSQs at `scx_sched_pcpu.bypass_dsq` for
///      every CPU (needs `sched`, `sched_pcpu`, plus the leaf set).
///   2. Per-node global DSQs at `scx_sched.pnode[node]->global_dsq`
///      for every NUMA node (needs `sched`, `sched_pnode`, plus
///      leaf set).
///   3. User-allocated DSQs walked through `scx_sched.dsq_hash`
///      (needs `sched`, `rht`, plus leaf set).
///
/// Each pass is independent: missing offsets for one pass blind
/// only that pass. A translate failure on one DSQ leaves it out of
/// the result without affecting the others.
#[allow(dead_code)]
pub fn walk_dsqs(
    kernel: &GuestKernel,
    sched_pa: u64,
    per_cpu_offsets: &[u64],
    nr_nodes: u32,
    offsets: &ScxWalkerOffsets,
) -> (Vec<DsqState>, Vec<TaskWalkerEntry>) {
    let mem = kernel.mem();
    let walk = kernel.walk_context();

    let mut dsq_states: Vec<DsqState> = Vec::new();
    let mut all_entries: Vec<TaskWalkerEntry> = Vec::new();

    // Leaf offsets common to every pass — all three DSQ-walking
    // passes feed `walk_one_dsq` which needs these. If any leaf
    // group is missing, no pass can run.
    let (Some(dsq_offs), Some(dsq_lnode_offs), Some(task_offs), Some(see_offs)) = (
        offsets.dsq.as_ref(),
        offsets.dsq_lnode.as_ref(),
        offsets.task.as_ref(),
        offsets.see.as_ref(),
    ) else {
        return (dsq_states, all_entries);
    };

    // Pass 1: per-CPU bypass DSQs. The percpu base lives at
    // sched->pcpu, dereferenced as a __percpu pointer; each CPU's
    // address is `pcpu_base + per_cpu_offset[cpu] +
    // scx_sched_pcpu.bypass_dsq`.
    //
    // Both `sched_offs.pcpu` (v6.18+) and `pcpu_offs.bypass_dsq`
    // (dev-only) are kernel-version-gated. Skip the entire pass
    // unless both offsets resolved — partial state would compute
    // a bogus DSQ KVA from `sched_pa + 0` (aliasing dsq_hash) and
    // surface phantom DSQ entries.
    if let (Some(sched_offs), Some(pcpu_offs)) =
        (offsets.sched.as_ref(), offsets.sched_pcpu.as_ref())
        && let (Some(sched_pcpu_off), Some(bypass_dsq_off)) =
            (sched_offs.pcpu, pcpu_offs.bypass_dsq)
    {
        let pcpu_kva = mem.read_u64(sched_pa, sched_pcpu_off);
        if pcpu_kva != 0 {
            for (cpu, &cpu_off) in per_cpu_offsets.iter().enumerate() {
                // Skip out-of-range CPUs — same heuristic as
                // read_percpu_array_value (cpu_off==0 && cpu_index>0
                // means BSS-zero tail).
                if cpu_off == 0 && cpu > 0 {
                    continue;
                }
                let dsq_kva = pcpu_kva
                    .wrapping_add(cpu_off)
                    .wrapping_add(bypass_dsq_off as u64);
                if let Some(dsq_pa) = translate_any_kva(
                    mem,
                    walk.cr3_pa,
                    walk.page_offset,
                    dsq_kva,
                    walk.l5,
                    walk.tcr_el1,
                ) && let Some((state, entries)) = walk_one_dsq(
                    mem,
                    walk,
                    dsq_kva,
                    dsq_pa,
                    || format!("bypass cpu {cpu}"),
                    dsq_offs,
                    dsq_lnode_offs,
                    task_offs,
                    see_offs,
                ) {
                    all_entries.extend(entries);
                    dsq_states.push(state);
                }
            }
        }
    }

    // Pass 2: per-node global DSQs. `sched->pnode` is a pointer
    // to an array of `struct scx_sched_pnode *` of length nr_nodes.
    // Both `sched_offs.pnode` and `pnode_offs.global_dsq` are
    // dev-only — skip the pass unless both resolved.
    if let (Some(sched_offs), Some(pnode_offs)) =
        (offsets.sched.as_ref(), offsets.sched_pnode.as_ref())
        && let (Some(sched_pnode_off), Some(global_dsq_off)) =
            (sched_offs.pnode, pnode_offs.global_dsq)
    {
        let pnode_kva = mem.read_u64(sched_pa, sched_pnode_off);
        if pnode_kva != 0
            && let Some(pnode_arr_pa) = translate_any_kva(
                mem,
                walk.cr3_pa,
                walk.page_offset,
                pnode_kva,
                walk.l5,
                walk.tcr_el1,
            )
        {
            for node in 0..nr_nodes as u64 {
                let pnode_ptr_kva = mem.read_u64(pnode_arr_pa, (node * 8) as usize);
                if pnode_ptr_kva == 0 {
                    continue;
                }
                let Some(pnode_pa) = translate_any_kva(
                    mem,
                    walk.cr3_pa,
                    walk.page_offset,
                    pnode_ptr_kva,
                    walk.l5,
                    walk.tcr_el1,
                ) else {
                    continue;
                };
                let dsq_kva = pnode_ptr_kva.wrapping_add(global_dsq_off as u64);
                let dsq_pa = pnode_pa.wrapping_add(global_dsq_off as u64);
                if let Some((state, entries)) = walk_one_dsq(
                    mem,
                    walk,
                    dsq_kva,
                    dsq_pa,
                    || format!("global node {node}"),
                    dsq_offs,
                    dsq_lnode_offs,
                    task_offs,
                    see_offs,
                ) {
                    all_entries.extend(entries);
                    dsq_states.push(state);
                }
            }
        }
    }

    // Pass 3: user-allocated DSQs via the scx_sched.dsq_hash
    // rhashtable. Walks at most MAX_RHT_NODES nodes total across
    // all buckets.
    if let (Some(sched_offs), Some(rht_offs)) = (offsets.sched.as_ref(), offsets.rht.as_ref()) {
        // dsq_hash is embedded in scx_sched (not a pointer), so its
        // PA is the sched_pa with the field offset added directly —
        // same pattern Pass 2 uses for pnode->global_dsq. Computing a
        // KVA here would require sched_kva which the caller already
        // discarded; translating sched_pa as a KVA would underflow
        // page_offset and silently empty the user-DSQ list.
        let rht_pa = sched_pa.wrapping_add(sched_offs.dsq_hash as u64);
        let (user_dsqs, user_dsqs_truncated) =
            walk_user_dsq_hash(mem, walk, rht_pa, rht_offs, dsq_offs);
        if user_dsqs_truncated {
            // Surface the cap-hit so an operator parsing the
            // failure dump trace sees that the user-DSQ list is
            // partial. Without this log the dump silently
            // omits the tail of the dsq_hash bucket table or
            // the tail of one bucket's chain.
            tracing::warn!(
                visited = user_dsqs.len(),
                cap_buckets = MAX_RHT_BUCKETS,
                cap_nodes = MAX_RHT_NODES,
                "walk_user_dsq_hash: truncated — bucket-table or node cap fired; \
                 dsq_kvas list is incomplete",
            );
        }
        for dsq_kva in user_dsqs {
            let Some(dsq_pa) = translate_any_kva(
                mem,
                walk.cr3_pa,
                walk.page_offset,
                dsq_kva,
                walk.l5,
                walk.tcr_el1,
            ) else {
                continue;
            };
            if let Some((state, entries)) = walk_one_dsq(
                mem,
                walk,
                dsq_kva,
                dsq_pa,
                || "user".to_string(),
                dsq_offs,
                dsq_lnode_offs,
                task_offs,
                see_offs,
            ) {
                all_entries.extend(entries);
                dsq_states.push(state);
            }
        }
    }

    (dsq_states, all_entries)
}

/// Walk one `scx_dispatch_q`. Returns the DSQ scalar state plus the
/// task entries on its `list`.
///
/// `origin` is taken as a `FnOnce` closure so the per-call
/// `format!("local cpu {cpu}")` / `format!("bypass cpu {cpu}")` /
/// `format!("global node {node}")` heap allocation only fires
/// after the `dsq_pa == 0` early-out has been cleared. Eagerly
/// formatting at every caller wasted one short-string allocation
/// per skipped DSQ on every freeze.
///
/// Returns `None` when `dsq_pa == 0`. Reading at PA 0 would
/// surface the boot-page contents as DSQ scalars (`id`, `nr`,
/// `seq`) and an all-zero list-head as an apparently-empty queue
/// — indistinguishable from a real empty DSQ. The early check
/// rejects that case so the caller does not push a phantom
/// DsqState row built from PA-0 garbage.
#[allow(clippy::too_many_arguments)]
fn walk_one_dsq(
    mem: &GuestMem,
    walk: WalkContext,
    dsq_kva: u64,
    dsq_pa: u64,
    origin: impl FnOnce() -> String,
    dsq_offs: &super::btf_offsets::ScxDispatchQOffsets,
    dsq_lnode_offs: &super::btf_offsets::ScxDsqListNodeOffsets,
    task_offs: &super::btf_offsets::TaskStructCoreOffsets,
    see_offs: &super::btf_offsets::SchedExtEntityOffsets,
) -> Option<(DsqState, Vec<TaskWalkerEntry>)> {
    if dsq_pa == 0 {
        tracing::debug!(
            dsq_kva = format_args!("{:#x}", dsq_kva),
            "walk_one_dsq: dsq_pa == 0 — would alias the boot page; \
             skipping to avoid surfacing phantom all-zero DSQ state",
        );
        return None;
    }
    let origin = origin();
    let id = mem.read_u64(dsq_pa, dsq_offs.id);
    let nr = mem.read_u32(dsq_pa, dsq_offs.nr);
    let seq = mem.read_u32(dsq_pa, dsq_offs.seq);

    // List head at dsq + list.
    let head_kva = dsq_kva.wrapping_add(dsq_offs.list as u64);
    let head_pa = dsq_pa.wrapping_add(dsq_offs.list as u64);

    // The DSQ list links sched_ext_entity.dsq_list.node fields
    // (struct list_head inside scx_dsq_list_node inside
    // sched_ext_entity inside task_struct). container_of computes:
    //   task_kva = node_kva
    //            - task.scx
    //            - see.dsq_list
    //            - dsq_lnode.node
    let dsq_node_off_in_task = task_offs.scx + see_offs.dsq_list + dsq_lnode_offs.node;

    let (task_kvas, truncated) = walk_list_head_for_dsq_task_kvas(
        mem,
        walk,
        head_kva,
        head_pa,
        dsq_node_off_in_task,
        dsq_lnode_offs,
    );

    let entries: Vec<TaskWalkerEntry> = task_kvas
        .iter()
        .map(|&task_kva| TaskWalkerEntry {
            task_kva,
            // Tasks queued on a DSQ are NOT on the per-CPU
            // runnable_list — they're staged for dispatch but not
            // yet runnable in the rq->scx sense. The
            // pi_boosted_out_of_scx flag only fires for
            // runnable_list tasks (the scenario it diagnoses is a
            // task that should have left the runnable_list when
            // its sched_class changed but didn't).
            is_runnable_in_scx: false,
            running_pc: None,
        })
        .collect();

    Some((
        DsqState {
            id,
            origin,
            nr,
            seq,
            task_kvas,
            truncated,
        },
        entries,
    ))
}

/// Walk a generic `list_head` chain starting at `head_kva`/`head_pa`,
/// recovering each task_struct KVA via container_of with
/// `runnable_node_off_in_task` as the field offset within
/// task_struct.
///
/// Returns (task_kvas, truncated). `truncated` is true when the
/// MAX_NODES_PER_LIST cap kicked in before the walk closed back to
/// the head.
fn walk_list_head_for_task_kvas(
    mem: &GuestMem,
    walk: WalkContext,
    head_kva: u64,
    head_pa: u64,
    runnable_node_off_in_task: usize,
) -> (Vec<u64>, bool) {
    let mut task_kvas = Vec::new();
    let mut node_kva = mem.read_u64(head_pa, 0);
    if node_kva == 0 {
        return (task_kvas, false);
    }

    let mut visited: u32 = 0;
    while node_kva != head_kva {
        if visited >= MAX_NODES_PER_LIST {
            return (task_kvas, true);
        }
        visited += 1;

        // container_of: task_kva = node_kva - runnable_node_off_in_task
        let task_kva = node_kva.wrapping_sub(runnable_node_off_in_task as u64);
        task_kvas.push(task_kva);

        // Step to next node — translate node_kva, read .next at offset 0.
        let Some(node_pa) = translate_any_kva(
            mem,
            walk.cr3_pa,
            walk.page_offset,
            node_kva,
            walk.l5,
            walk.tcr_el1,
        ) else {
            return (task_kvas, false);
        };
        let next_kva = mem.read_u64(node_pa, 0);
        if next_kva == 0 {
            return (task_kvas, false);
        }
        node_kva = next_kva;
    }
    (task_kvas, false)
}

/// Walk a DSQ's `list` chain (a list of `scx_dsq_list_node.node`
/// entries embedded in `sched_ext_entity.dsq_list`). Skips iterator
/// cursor entries marked with `SCX_DSQ_LNODE_ITER_CURSOR`.
fn walk_list_head_for_dsq_task_kvas(
    mem: &GuestMem,
    walk: WalkContext,
    head_kva: u64,
    head_pa: u64,
    dsq_node_off_in_task: usize,
    dsq_lnode_offs: &super::btf_offsets::ScxDsqListNodeOffsets,
) -> (Vec<u64>, bool) {
    let mut task_kvas = Vec::new();
    let mut node_kva = mem.read_u64(head_pa, 0);
    if node_kva == 0 {
        return (task_kvas, false);
    }

    let mut visited: u32 = 0;
    while node_kva != head_kva {
        if visited >= MAX_NODES_PER_LIST {
            return (task_kvas, true);
        }
        visited += 1;

        // The list_head we're walking is `scx_dsq_list_node.node`
        // (the inner `struct list_head`). Recover the parent
        // scx_dsq_list_node start by subtracting `dsq_lnode.node`
        // — fixed at 0 in current kernels but we read it from the
        // offsets struct for forward-compatibility.
        let lnode_kva = node_kva.wrapping_sub(dsq_lnode_offs.node as u64);

        // Read the lnode's flags to skip iterator-cursor entries.
        // is_cursor: Some(true) → cursor (skip), Some(false) → real
        // task entry (push), None → translate failed and we cannot
        // distinguish. When the cursor flag cannot be read, treat
        // the entry as a cursor and skip it: pushing it would
        // record a phantom `task_kva = node_kva - dsq_node_off_in_task`
        // built from a node whose enclosing sched_ext_entity isn't
        // mappable, which downstream task enrichment would surface
        // as bogus pid/comm reads at an arbitrary address.
        let is_cursor = match translate_any_kva(
            mem,
            walk.cr3_pa,
            walk.page_offset,
            lnode_kva,
            walk.l5,
            walk.tcr_el1,
        ) {
            Some(lnode_pa) => {
                let lnode_flags = mem.read_u32(lnode_pa, dsq_lnode_offs.flags);
                Some(lnode_flags & SCX_DSQ_LNODE_ITER_CURSOR != 0)
            }
            None => None,
        };

        let skip_entry = match is_cursor {
            Some(true) => true,   // cursor entry — advance without recording
            Some(false) => false, // real task entry — push and advance
            None => true,         // cursor-detection unreliable — skip rather than push bogus
        };

        if !skip_entry {
            // Real task entry: container_of from the inner list_head's
            // node_kva back to task_struct. The full offset within
            // task_struct is task.scx + see.dsq_list + dsq_lnode.node.
            let task_kva = node_kva.wrapping_sub(dsq_node_off_in_task as u64);
            task_kvas.push(task_kva);
        }

        // Advance to the next node. The list_head.next pointer
        // lives at offset 0 of the inner list_head we landed on,
        // which is `node_kva` itself.
        let Some(node_pa) = translate_any_kva(
            mem,
            walk.cr3_pa,
            walk.page_offset,
            node_kva,
            walk.l5,
            walk.tcr_el1,
        ) else {
            return (task_kvas, false);
        };
        let next_kva = mem.read_u64(node_pa, 0);
        if next_kva == 0 {
            return (task_kvas, false);
        }
        node_kva = next_kva;
    }
    (task_kvas, false)
}

/// Walk the user-allocated DSQ rhashtable rooted at `rht_pa`.
///
/// `rht_pa` is the DRAM-relative offset of the embedded
/// `struct rhashtable` inside `scx_sched.dsq_hash`. The caller
/// computes it as `sched_pa + sched.dsq_hash` — the field is embedded
/// (not a pointer), so its PA is just the containing struct's PA plus
/// the field offset. The walker reads `tbl` (bucket_table pointer),
/// then for each of `tbl.buckets[i]` it strips the LSB tag
/// (`RHT_PTR_LOCK_BIT`) and chases the `rhash_head.next` chain. For
/// each node the walker computes
/// `dsq_kva = node_kva - scx_dispatch_q.hash_node` (container_of).
///
/// Caps:
/// - bucket count at [`MAX_RHT_BUCKETS`]
/// - total nodes visited at [`MAX_RHT_NODES`]
/// - per-bucket chain length at [`PER_BUCKET_CHAIN_CAP`]
///
/// Returns `(dsq_kvas, truncated)`. `truncated` is `true` when any
/// cap fired before the walk could reach the natural end of every
/// bucket chain — either `bucket_table.size > MAX_RHT_BUCKETS`,
/// `total_nodes >= MAX_RHT_NODES` mid-walk, or a per-bucket chain
/// reached its `PER_BUCKET_CHAIN_CAP` cap. Without this signal,
/// callers cannot distinguish "small DSQ count" from "cap silently
/// dropped tail entries" — see DsqState.truncated for the same
/// pattern on per-DSQ task lists.
fn walk_user_dsq_hash(
    mem: &GuestMem,
    walk: WalkContext,
    rht_pa: u64,
    rht_offs: &super::btf_offsets::RhashtableOffsets,
    dsq_offs: &super::btf_offsets::ScxDispatchQOffsets,
) -> (Vec<u64>, bool) {
    let mut dsq_kvas = Vec::new();

    let tbl_kva = mem.read_u64(rht_pa, rht_offs.tbl);
    if tbl_kva == 0 {
        return (dsq_kvas, false);
    }
    let Some(tbl_pa) = translate_any_kva(
        mem,
        walk.cr3_pa,
        walk.page_offset,
        tbl_kva,
        walk.l5,
        walk.tcr_el1,
    ) else {
        return (dsq_kvas, false);
    };

    let size = mem.read_u32(tbl_pa, rht_offs.bucket_table_size);
    let bucket_count = size.min(MAX_RHT_BUCKETS) as u64;
    // A bucket_table.size larger than the bucket cap means we'll
    // walk only the first MAX_RHT_BUCKETS buckets and the tail is
    // silently dropped. Surface that as truncation up front.
    let mut truncated = size as u64 > bucket_count;
    let buckets_off = rht_offs.bucket_table_buckets;

    let mut total_nodes: u32 = 0;
    for i in 0..bucket_count {
        if total_nodes >= MAX_RHT_NODES {
            // Hit the global node cap — remaining buckets unwalked.
            return (dsq_kvas, true);
        }
        let entry_off = buckets_off + (i as usize) * 8;
        let raw_ptr = mem.read_u64(tbl_pa, entry_off);
        // Strip the LSB lock-bit tag. NULL or pure tag (0 with
        // bit 0 unset) means empty bucket.
        let head_kva = raw_ptr & !RHT_PTR_LOCK_BIT;
        if head_kva == 0 {
            continue;
        }
        // Chase the `rhash_head.next` chain. Each node is a
        // `rhash_head` embedded in scx_dispatch_q at
        // `hash_node`; container_of yields the dsq KVA.
        let mut node_kva = head_kva;
        let mut chain_visited: u32 = 0;
        let mut chain_terminated_naturally = false;
        while node_kva != 0 && total_nodes < MAX_RHT_NODES && chain_visited < PER_BUCKET_CHAIN_CAP {
            chain_visited += 1;
            total_nodes += 1;
            let dsq_kva = node_kva.wrapping_sub(dsq_offs.hash_node as u64);
            dsq_kvas.push(dsq_kva);
            let Some(node_pa) = translate_any_kva(
                mem,
                walk.cr3_pa,
                walk.page_offset,
                node_kva,
                walk.l5,
                walk.tcr_el1,
            ) else {
                // Translate failure — chain ended for this bucket
                // without hitting a cap. Not a truncation signal.
                chain_terminated_naturally = true;
                break;
            };
            let next_raw = mem.read_u64(node_pa, rht_offs.rhash_head_next);
            // The chain terminator is a "nulls" pointer with bit 0
            // set encoding the bucket index; treat any LSB-tagged
            // pointer as terminator.
            if next_raw & RHT_PTR_LOCK_BIT != 0 || next_raw == 0 {
                chain_terminated_naturally = true;
                break;
            }
            node_kva = next_raw;
        }
        // The loop exited; if it wasn't via a natural terminator
        // (LSB-tagged pointer / NULL / translate failure), one of
        // the two caps fired (chain_visited >= PER_BUCKET_CHAIN_CAP
        // or total_nodes >= MAX_RHT_NODES) and we silently dropped
        // the rest of this bucket's chain.
        if !chain_terminated_naturally {
            truncated = true;
        }
    }

    (dsq_kvas, truncated)
}

/// Read `(pid, comm)` for a `task_struct *` after a NULL-check and
/// translate. Returns `(None, None)` on NULL or untranslatable.
fn read_task_pid_comm(
    mem: &GuestMem,
    walk: WalkContext,
    task_kva: u64,
    pid_off: usize,
    comm_off: usize,
) -> (Option<i32>, Option<String>) {
    if task_kva == 0 {
        return (None, None);
    }
    let Some(task_pa) = translate_any_kva(
        mem,
        walk.cr3_pa,
        walk.page_offset,
        task_kva,
        walk.l5,
        walk.tcr_el1,
    ) else {
        return (None, None);
    };
    let pid = mem.read_u32(task_pa, pid_off) as i32;
    let mut buf = [0u8; 16];
    mem.read_bytes(task_pa + comm_off as u64, &mut buf);
    let n = buf.iter().position(|&b| b == 0).unwrap_or(16);
    let comm = String::from_utf8_lossy(&buf[..n]).to_string();
    (Some(pid), Some(comm))
}

#[cfg(test)]
#[path = "scx_walker_tests.rs"]
mod tests;
