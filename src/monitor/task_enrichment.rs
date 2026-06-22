//! Per-task failure-dump enrichment: read every Tier-1 task field the
//! failure-dump renderer surfaces in one pass.
//!
//! Given a task_struct KVA from a freeze-time walker (rq->scx walker
//! for runnable tasks, DSQ walker for queued tasks, init_task→tasks
//! walker for thread-group enumeration), this module reads:
//!
//! - Identity: pid, tgid, comm, group_leader_pid, real_parent_pid+comm
//! - Process tree: pgid, sid, nr_threads (via signal_struct)
//! - Scheduling: prio, static_prio, normal_prio, rt_priority,
//!   sched_class decoded to a name (CFS / RT / DL / IDLE / STOP / EXT),
//!   scx.weight, core_cookie (CONFIG_SCHED_CORE-gated)
//! - Watchdog disambiguation: `pi_boosted_out_of_scx` flag set when
//!   the runnable task's `sched_class` is not `ext_sched_class` (the
//!   PI-boost path moved it out and the failure isn't the BPF
//!   scheduler's fault — see scx core.c rt_mutex_setprio interactions)
//! - Context-switch counters: per-task nvcsw/nivcsw + per-thread-group
//!   signal->nvcsw/nivcsw
//! - Lock-contention hints: stack-trace pattern match against the
//!   sched_class symbol KVAs of `queued_spin_lock_slowpath`,
//!   `__mutex_lock_slowpath`, `rwsem_down_read_slowpath`,
//!   `rwsem_down_write_slowpath`. A PC inside any slowpath function
//!   on a runnable ('R') task indicates lock contention rather than
//!   scheduler fault. Stack-trace walking is only attempted when the
//!   caller supplies a non-empty stack-PC slice (typically harvested
//!   from the freeze coordinator's `VcpuRegSnapshot.instruction_pointer`
//!   for currently-running tasks; runnable-but-not-current tasks have
//!   no stack PCs without a kernel-side unwinder, which ktstr does
//!   not implement).
//!
//! The walker is best-effort: any pointer follow that fails to
//! translate yields a `None` for that derived field rather than
//! aborting the whole enrichment.

use serde::{Deserialize, Serialize};

use super::btf_offsets::{TaskEnrichmentOffsets, pid_type};
use super::guest::GuestKernel;
use super::idr::translate_any_kva;

/// Maximum bytes of `comm` to read.
///
/// Kernel-pinned at 16 by `include/linux/sched.h::TASK_COMM_LEN`. The
/// walker reads exactly this many bytes; trailing nuls are stripped
/// when forming the `String`.
const TASK_COMM_LEN: usize = 16;

/// Sched-class symbol KVAs cached for decode + watchdog
/// disambiguation. All six are resolved via vmlinux ELF symbol table
/// at coordinator start; missing symbols (kernel built without the
/// corresponding scheduling class — typically `dl_sched_class` on
/// CONFIG_SCHED_DEADLINE=n) leave the slot as `None`, and the
/// decoder returns `None` for that class.
///
/// All addresses are KVAs of the per-class `sched_class` static
/// (`fair_sched_class`, `rt_sched_class`, `dl_sched_class`,
/// `idle_sched_class`, `stop_sched_class`, `ext_sched_class`). On a
/// running guest, `task_struct.sched_class` points at exactly one of
/// these — comparing the read pointer to the cached set yields the
/// class name without needing a kallsyms parse.
#[derive(Debug, Clone, Default)]
pub struct SchedClassRegistry {
    pub fair: Option<u64>,
    pub rt: Option<u64>,
    pub dl: Option<u64>,
    pub idle: Option<u64>,
    pub stop: Option<u64>,
    pub ext: Option<u64>,
}

#[allow(dead_code)] // wired through DumpContext::TaskEnrichmentCapture;
// freeze coordinator passes None until the rq->scx
// walker lands a walker producer.
impl SchedClassRegistry {
    /// Resolve all six class symbols via the GuestKernel's vmlinux
    /// symbol table. Each lookup is independent — a missing symbol
    /// for one class doesn't fail the others.
    pub fn from_guest_kernel(kernel: &GuestKernel) -> Self {
        Self {
            fair: kernel.symbol_kva("fair_sched_class"),
            rt: kernel.symbol_kva("rt_sched_class"),
            dl: kernel.symbol_kva("dl_sched_class"),
            idle: kernel.symbol_kva("idle_sched_class"),
            stop: kernel.symbol_kva("stop_sched_class"),
            ext: kernel.symbol_kva("ext_sched_class"),
        }
    }

    /// Decode a `task_struct.sched_class` pointer to a class name.
    /// Returns `None` when the pointer matches no known class
    /// (stripped vmlinux, an out-of-tree class the kernel added,
    /// or a torn read landing on garbage).
    pub fn decode(&self, sched_class_kva: u64) -> Option<&'static str> {
        if sched_class_kva == 0 {
            return None;
        }
        if Some(sched_class_kva) == self.fair {
            return Some("fair");
        }
        if Some(sched_class_kva) == self.rt {
            return Some("rt");
        }
        if Some(sched_class_kva) == self.dl {
            return Some("dl");
        }
        if Some(sched_class_kva) == self.idle {
            return Some("idle");
        }
        if Some(sched_class_kva) == self.stop {
            return Some("stop");
        }
        if Some(sched_class_kva) == self.ext {
            return Some("ext");
        }
        None
    }
}

/// Lock-slowpath function KVAs. Used by the stack-trace lock detector
/// to flag runnable tasks whose stack contains a slowpath PC —
/// indicating the apparent scheduler stall is actually lock
/// contention, not BPF scheduler fault.
///
/// Each address is the function entry; the detector flags any PC in
/// `[start, start + LOCK_SLOWPATH_FN_MAX_SIZE)`. Without ELF symbol
/// size info we can't bound this exactly, so we use a conservative
/// 4 KiB window. False positives on adjacent functions are acceptable
/// for a diagnostic flag; false negatives only matter on slowpaths
/// longer than 4 KiB, none of which occur in mainline.
#[derive(Debug, Clone, Default)]
pub struct LockSlowpathRegistry {
    pub queued_spin_lock_slowpath: Option<u64>,
    pub mutex_lock_slowpath: Option<u64>,
    pub rwsem_down_read_slowpath: Option<u64>,
    pub rwsem_down_write_slowpath: Option<u64>,
}

/// Conservative max function size for stack-PC matching against
/// lock-slowpath entry symbols. See `LockSlowpathRegistry` doc.
const LOCK_SLOWPATH_FN_MAX_SIZE: u64 = 4096;

#[allow(dead_code)] // same wiring rationale as SchedClassRegistry above.
impl LockSlowpathRegistry {
    /// Resolve the four lock-slowpath symbols from the GuestKernel's
    /// vmlinux. Each lookup is independent; absent symbols leave the
    /// corresponding slot None and the matcher silently skips that
    /// pattern.
    pub fn from_guest_kernel(kernel: &GuestKernel) -> Self {
        Self {
            queued_spin_lock_slowpath: kernel.symbol_kva("queued_spin_lock_slowpath"),
            // `__mutex_lock_slowpath` is the historical name; modern
            // kernels (~4.15+) inline the slowpath into
            // `__mutex_lock`, but a leftover symbol remains in many
            // configs. Fall through to `__mutex_lock` if the
            // slowpath symbol is absent — both indicate the same
            // contention pattern at PC granularity.
            mutex_lock_slowpath: kernel
                .symbol_kva("__mutex_lock_slowpath")
                .or_else(|| kernel.symbol_kva("__mutex_lock")),
            rwsem_down_read_slowpath: kernel.symbol_kva("rwsem_down_read_slowpath"),
            rwsem_down_write_slowpath: kernel.symbol_kva("rwsem_down_write_slowpath"),
        }
    }

    /// Match a single PC against the four slowpath windows. Returns
    /// the pattern name when any window contains the PC, or `None`
    /// otherwise.
    pub fn match_pc(&self, pc: u64) -> Option<&'static str> {
        let probe = |start: Option<u64>, name: &'static str| -> Option<&'static str> {
            let s = start?;
            // Symbol KVAs come from the guest's vmlinux. A corrupt
            // ELF could place a slowpath symbol near u64::MAX; the
            // window upper bound `s + 4096` would wrap, and `pc <
            // wrapped` would falsely match every PC. checked_add
            // returning None means "this symbol can't define a
            // valid window" — treat as no match for that pattern.
            let end = s.checked_add(LOCK_SLOWPATH_FN_MAX_SIZE)?;
            if pc >= s && pc < end {
                Some(name)
            } else {
                None
            }
        };
        probe(self.queued_spin_lock_slowpath, "queued_spin_lock_slowpath")
            .or_else(|| probe(self.mutex_lock_slowpath, "mutex_lock_slowpath"))
            .or_else(|| probe(self.rwsem_down_read_slowpath, "rwsem_down_read_slowpath"))
            .or_else(|| probe(self.rwsem_down_write_slowpath, "rwsem_down_write_slowpath"))
    }
}

/// Per-task enrichment captured at freeze time.
///
/// Every field is best-effort: read failures (untranslatable RCU
/// pointer, slab-page eviction race, missing BTF field) yield `None`
/// rather than failing the whole capture. Optional fields cover both
/// "absent on this kernel build" (e.g. `core_cookie` without
/// CONFIG_SCHED_CORE) and "unreadable at this freeze instant" (e.g.
/// `real_parent_pid` when the parent task_struct's slab page didn't
/// translate).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[non_exhaustive]
pub struct TaskEnrichment {
    /// `task_struct.pid`. The kernel's per-task identifier.
    pub pid: i32,
    /// `task_struct.tgid`. Thread-group identifier (POSIX `getpid()`).
    pub tgid: i32,
    /// `task_struct.comm` truncated at the first nul byte.
    pub comm: String,
    /// `task_struct.group_leader->pid`. Pointer-followed; `None` on
    /// translate failure or NULL group_leader (init_task case).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub group_leader_pid: Option<i32>,
    /// `task_struct.real_parent->pid`. RCU pointer-followed;
    /// `None` on translate failure or NULL real_parent (init_task).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub real_parent_pid: Option<i32>,
    /// `task_struct.real_parent->comm` truncated at the first nul.
    /// `None` if real_parent unreadable.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub real_parent_comm: Option<String>,
    /// `signal->pids[PIDTYPE_PGID]->numbers[0].nr`. Process group id.
    /// `None` on signal_struct translate failure or NULL pids slot.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pgid: Option<i32>,
    /// `signal->pids[PIDTYPE_SID]->numbers[0].nr`. Session id.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sid: Option<i32>,
    /// `signal->nr_threads`. Live thread count for the thread group.
    /// `None` on signal_struct translate failure.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub nr_threads: Option<i32>,
    /// `task_struct.scx.weight` (u32). scx-domain CFS-equivalent
    /// weight; 100 default.
    pub weight: u32,
    /// `task_struct.prio`. Effective scheduling priority
    /// (PI-boost-aware).
    pub prio: i32,
    /// `task_struct.static_prio`. User-set priority before PI boost.
    pub static_prio: i32,
    /// `task_struct.normal_prio`. Normal priority for the class.
    pub normal_prio: i32,
    /// `task_struct.rt_priority`. RT priority (1-99) for SCHED_FIFO/RR.
    pub rt_priority: u32,
    /// Decoded sched_class name: "fair", "rt", "dl", "idle", "stop",
    /// or "ext". `None` when the pointer matches no cached class
    /// (stripped vmlinux or out-of-tree class).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sched_class: Option<String>,
    /// `task_struct.core_cookie` (unsigned long).
    /// CONFIG_SCHED_CORE-gated; `None` on kernels built without it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub core_cookie: Option<u64>,
    /// True iff the task was on the rq->scx.runnable_list at freeze
    /// time AND `sched_class != ext_sched_class`. Indicates the PI
    /// boost path moved it out of SCX (rt_mutex_setprio) — failure
    /// is not the BPF scheduler's fault. Set only by the runnable
    /// walker; the queued-DSQ walker leaves this `false`.
    pub pi_boosted_out_of_scx: bool,
    /// `task_struct.nvcsw` (unsigned long). Voluntary context
    /// switches; the live thread count.
    pub nvcsw: u64,
    /// `task_struct.nivcsw` (unsigned long). Involuntary context
    /// switches.
    pub nivcsw: u64,
    /// `signal->nvcsw` (unsigned long). Thread-group accumulator
    /// for dead threads. `None` on signal_struct translate failure.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub signal_nvcsw: Option<u64>,
    /// `signal->nivcsw` (unsigned long). Mirror of `signal_nvcsw`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub signal_nivcsw: Option<u64>,
    /// `task_struct.utime` (u64, nanoseconds). This live thread's
    /// cumulative user-mode CPU time, task-lifetime monotonic. The raw
    /// kernel accumulator (NOT the cputime_adjust-scaled /proc value),
    /// equal to taskstats `ac_utime` modulo ns→us truncation. Counter
    /// semantics: per-phase user time = end-start delta of the
    /// summed-across-tasks reading.
    pub utime: u64,
    /// `task_struct.stime` (u64, nanoseconds). This live thread's
    /// cumulative system-mode (in-kernel) CPU time — the DSQ-spinlock
    /// regression's direct symptom. Same raw-accumulator / Counter
    /// semantics as [`Self::utime`].
    pub stime: u64,
    /// `signal->utime` (u64, ns). Thread-group accumulator of EXITED
    /// threads' user-mode CPU time (`__exit_signal` folds a dying
    /// thread's utime here). `None` on signal_struct translate failure.
    /// Shared across a thread group, so a per-group sum must add it
    /// exactly once; combined with the live-thread [`Self::utime`] sum
    /// it keeps the per-phase total from dipping when a worker exits.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub signal_utime: Option<u64>,
    /// `signal->stime` (u64, ns). Exited threads' system-mode CPU time
    /// accumulator. Mirror of [`Self::signal_utime`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub signal_stime: Option<u64>,
    /// Lock-slowpath pattern matched on a PC supplied by the caller
    /// (typically `VcpuRegSnapshot.instruction_pointer` for the task
    /// running on a vCPU at freeze time). One of
    /// "queued_spin_lock_slowpath", "mutex_lock_slowpath",
    /// "rwsem_down_read_slowpath", "rwsem_down_write_slowpath", or
    /// `None` when the supplied PC matched nothing OR the caller
    /// supplied no PCs.
    ///
    /// Set only when `walk_task_enrichment_with_pcs` is used; the
    /// no-PC entry point `walk_task_enrichment` always leaves this
    /// `None`. A stack walker that produces multiple PCs (a future
    /// kernel-side unwinder) would surface them as a `Vec<String>`
    /// in a non_exhaustive struct extension.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lock_slowpath_match: Option<String>,
}

/// Walk one task_struct and populate every Tier-1 enrichment field.
/// `task_kva` must point at a valid `struct task_struct` reachable
/// via `translate_any_kva`. `is_runnable_in_scx` is set by the rq->scx
/// walker for tasks read off `rq->scx.runnable_list` (used for the
/// PI-boost-out-of-SCX flag); the queued-DSQ walker passes `false`.
///
/// `pc` (`Option<u64>`) is the task's instruction pointer for the
/// lock-slowpath stack matcher. Pass the corresponding vCPU's
/// `instruction_pointer` when this task was running on that vCPU at
/// freeze time; pass `None` for tasks not actively running (the
/// matcher needs an unwinder we don't have).
#[allow(dead_code)]
pub fn walk_task_enrichment(
    kernel: &GuestKernel,
    task_kva: u64,
    offsets: &TaskEnrichmentOffsets,
    classes: &SchedClassRegistry,
    locks: &LockSlowpathRegistry,
    is_runnable_in_scx: bool,
    pc: Option<u64>,
) -> Option<TaskEnrichment> {
    let mem = kernel.mem();
    let walk = kernel.walk_context();

    let task_pa = translate_any_kva(
        mem,
        walk.cr3_pa,
        walk.page_offset,
        task_kva,
        walk.l5,
        walk.tcr_el1,
    )?;

    // Identity.
    let pid = mem.read_u32(task_pa, offsets.task_struct_pid) as i32;
    let tgid = mem.read_u32(task_pa, offsets.task_struct_tgid) as i32;
    let comm = read_comm(mem, task_pa, offsets.task_struct_comm);

    // Scheduling fields.
    let prio = mem.read_u32(task_pa, offsets.task_struct_prio) as i32;
    let static_prio = mem.read_u32(task_pa, offsets.task_struct_static_prio) as i32;
    let normal_prio = mem.read_u32(task_pa, offsets.task_struct_normal_prio) as i32;
    let rt_priority = mem.read_u32(task_pa, offsets.task_struct_rt_priority);
    let sched_class_kva = mem.read_u64(task_pa, offsets.task_struct_sched_class);
    let sched_class = classes.decode(sched_class_kva).map(str::to_string);
    let weight = mem.read_u32(task_pa, offsets.task_struct_scx + offsets.see_weight);
    let core_cookie = offsets
        .task_struct_core_cookie
        .map(|off| mem.read_u64(task_pa, off));

    // PI-boost-out-of-SCX flag: set only when the task was reached
    // via the rq->scx.runnable_list AND its current sched_class is
    // not ext_sched_class. This catches the case where rt_mutex_setprio
    // moved the task to a higher-prio class while it remained on the
    // SCX runnable list.
    let pi_boosted_out_of_scx =
        is_runnable_in_scx && classes.ext.is_some() && Some(sched_class_kva) != classes.ext;

    // Per-task context-switch counters.
    let nvcsw = mem.read_u64(task_pa, offsets.task_struct_nvcsw);
    let nivcsw = mem.read_u64(task_pa, offsets.task_struct_nivcsw);
    // Per-task cumulative CPU time (nanoseconds), task-lifetime
    // monotonic. Read raw from frozen guest memory — zero guest work
    // (vs the taskstats genetlink query). utime already includes gtime
    // (guest time is double-counted into utime); do not sum the two.
    let utime = mem.read_u64(task_pa, offsets.task_struct_utime);
    let stime = mem.read_u64(task_pa, offsets.task_struct_stime);

    // Pointer follows: group_leader, real_parent, signal.
    let group_leader_kva = mem.read_u64(task_pa, offsets.task_struct_group_leader);
    let group_leader_pid =
        follow_task_for_pid(mem, walk, group_leader_kva, offsets.task_struct_pid);

    let real_parent_kva = mem.read_u64(task_pa, offsets.task_struct_real_parent);
    let (real_parent_pid, real_parent_comm) = follow_task_for_pid_and_comm(
        mem,
        walk,
        real_parent_kva,
        offsets.task_struct_pid,
        offsets.task_struct_comm,
    );

    let signal_kva = mem.read_u64(task_pa, offsets.task_struct_signal);
    let (nr_threads, signal_nvcsw, signal_nivcsw, signal_utime, signal_stime, pgid, sid) =
        if signal_kva == 0 {
            (None, None, None, None, None, None, None)
        } else {
            match translate_any_kva(
                mem,
                walk.cr3_pa,
                walk.page_offset,
                signal_kva,
                walk.l5,
                walk.tcr_el1,
            ) {
                None => (None, None, None, None, None, None, None),
                Some(signal_pa) => {
                    let nr_threads_v =
                        mem.read_u32(signal_pa, offsets.signal_struct_nr_threads) as i32;
                    let signal_nvcsw_v = mem.read_u64(signal_pa, offsets.signal_struct_nvcsw);
                    let signal_nivcsw_v = mem.read_u64(signal_pa, offsets.signal_struct_nivcsw);
                    // Exited-thread CPU-time accumulators (ns): added to
                    // the live-thread sum so a mid-phase exit does not
                    // undercount per-phase CPU time.
                    let signal_utime_v = mem.read_u64(signal_pa, offsets.signal_struct_utime);
                    let signal_stime_v = mem.read_u64(signal_pa, offsets.signal_struct_stime);
                    // pids[PIDTYPE_PGID] / pids[PIDTYPE_SID] traversal.
                    // Each slot is `struct pid *` (8 bytes); the
                    // numbers[0].nr deref reads the canonical root-ns
                    // pid number.
                    let pgid_v = read_pid_nr_at_index(
                        mem,
                        walk,
                        signal_pa,
                        offsets.signal_struct_pids,
                        pid_type::PGID,
                        offsets.pid_numbers,
                        offsets.upid_size,
                        offsets.upid_nr,
                    );
                    let sid_v = read_pid_nr_at_index(
                        mem,
                        walk,
                        signal_pa,
                        offsets.signal_struct_pids,
                        pid_type::SID,
                        offsets.pid_numbers,
                        offsets.upid_size,
                        offsets.upid_nr,
                    );
                    (
                        Some(nr_threads_v),
                        Some(signal_nvcsw_v),
                        Some(signal_nivcsw_v),
                        Some(signal_utime_v),
                        Some(signal_stime_v),
                        pgid_v,
                        sid_v,
                    )
                }
            }
        };

    // Lock-slowpath PC match, if a PC was supplied.
    let lock_slowpath_match = pc.and_then(|p| locks.match_pc(p)).map(str::to_string);

    Some(TaskEnrichment {
        pid,
        tgid,
        comm,
        group_leader_pid,
        real_parent_pid,
        real_parent_comm,
        pgid,
        sid,
        nr_threads,
        weight,
        prio,
        static_prio,
        normal_prio,
        rt_priority,
        sched_class,
        core_cookie,
        pi_boosted_out_of_scx,
        nvcsw,
        nivcsw,
        signal_nvcsw,
        signal_nivcsw,
        utime,
        stime,
        signal_utime,
        signal_stime,
        lock_slowpath_match,
    })
}

/// Read `comm` as a `String` truncated at the first nul.
fn read_comm(mem: &super::reader::GuestMem, task_pa: u64, comm_off: usize) -> String {
    let mut buf = [0u8; TASK_COMM_LEN];
    mem.read_bytes(task_pa + comm_off as u64, &mut buf);
    let n = buf.iter().position(|&b| b == 0).unwrap_or(TASK_COMM_LEN);
    String::from_utf8_lossy(&buf[..n]).to_string()
}

/// Translate a `task_struct *` to its physical address and return
/// `(pid, comm)`. Returns `(None, None)` on any failure.
fn follow_task_for_pid_and_comm(
    mem: &super::reader::GuestMem,
    walk: super::reader::WalkContext,
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
    let comm = read_comm(mem, task_pa, comm_off);
    (Some(pid), Some(comm))
}

/// Translate a `task_struct *` and read just the pid.
fn follow_task_for_pid(
    mem: &super::reader::GuestMem,
    walk: super::reader::WalkContext,
    task_kva: u64,
    pid_off: usize,
) -> Option<i32> {
    if task_kva == 0 {
        return None;
    }
    let task_pa = translate_any_kva(
        mem,
        walk.cr3_pa,
        walk.page_offset,
        task_kva,
        walk.l5,
        walk.tcr_el1,
    )?;
    Some(mem.read_u32(task_pa, pid_off) as i32)
}

/// Read `signal->pids[idx]->numbers[0].nr`.
///
/// Three pointer hops:
///   1. `signal_pa + pids_off + idx * 8` reads the `struct pid *`.
///   2. Translate the pid pointer; the `numbers[0]` element starts at
///      `pid_pa + numbers_off`.
///   3. Read the `nr` field at `numbers[0] + nr_off`.
///
/// Returns `None` on any translate failure or when the pid pointer is
/// NULL (typical for `pids[PIDTYPE_PGID/SID]` on threads that aren't
/// session/process group leaders).
#[allow(clippy::too_many_arguments)]
fn read_pid_nr_at_index(
    mem: &super::reader::GuestMem,
    walk: super::reader::WalkContext,
    signal_pa: u64,
    pids_off: usize,
    idx: usize,
    numbers_off: usize,
    upid_size: usize,
    nr_off: usize,
) -> Option<i32> {
    let pid_kva = mem.read_u64(signal_pa, pids_off + idx * 8);
    if pid_kva == 0 {
        return None;
    }
    let pid_pa = translate_any_kva(
        mem,
        walk.cr3_pa,
        walk.page_offset,
        pid_kva,
        walk.l5,
        walk.tcr_el1,
    )?;
    // numbers[0] is at offset `numbers_off`; subsequent levels are at
    // `numbers_off + level * upid_size`. We always read level 0
    // (root pid namespace) per the kernel's `pid_nr` contract.
    let _ = upid_size; // captured in signature for level>0 callers
    Some(mem.read_u32(pid_pa, numbers_off + nr_off) as i32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sched_class_registry_decode_known_class() {
        let r = SchedClassRegistry {
            fair: Some(0xffff_ffff_8000_1000),
            rt: Some(0xffff_ffff_8000_1100),
            dl: None,
            idle: None,
            stop: None,
            ext: Some(0xffff_ffff_8000_1300),
        };
        assert_eq!(r.decode(0xffff_ffff_8000_1000), Some("fair"));
        assert_eq!(r.decode(0xffff_ffff_8000_1100), Some("rt"));
        assert_eq!(r.decode(0xffff_ffff_8000_1300), Some("ext"));
    }

    #[test]
    fn sched_class_registry_decode_unknown_returns_none() {
        let r = SchedClassRegistry {
            fair: Some(0xffff_ffff_8000_1000),
            rt: None,
            dl: None,
            idle: None,
            stop: None,
            ext: None,
        };
        assert_eq!(r.decode(0xffff_ffff_8000_2000), None);
        // Zero pointer never decodes (would-be-NULL sched_class).
        assert_eq!(r.decode(0), None);
    }

    #[test]
    fn lock_slowpath_match_within_window() {
        let r = LockSlowpathRegistry {
            queued_spin_lock_slowpath: Some(0xffff_ffff_8001_0000),
            mutex_lock_slowpath: Some(0xffff_ffff_8002_0000),
            rwsem_down_read_slowpath: None,
            rwsem_down_write_slowpath: None,
        };
        // Inside the qsl window.
        assert_eq!(
            r.match_pc(0xffff_ffff_8001_0010),
            Some("queued_spin_lock_slowpath")
        );
        // Inside the mutex window.
        assert_eq!(
            r.match_pc(0xffff_ffff_8002_0fff),
            Some("mutex_lock_slowpath")
        );
        // Past the qsl window (4 KiB cap).
        assert!(r.match_pc(0xffff_ffff_8001_2000).is_none());
        // Before the qsl window.
        assert!(r.match_pc(0xffff_ffff_8000_ffff).is_none());
    }

    #[test]
    fn lock_slowpath_no_match_when_all_none() {
        let r = LockSlowpathRegistry::default();
        assert_eq!(r.match_pc(0xdeadbeef), None);
    }

    /// Pin the wire shape of `TaskEnrichment` — every optional field
    /// should skip on `None` so a populated capture renders cleanly
    /// without a wall of nulls in the JSON.
    #[test]
    fn task_enrichment_serde_skip_none_fields() {
        let e = TaskEnrichment {
            pid: 42,
            tgid: 42,
            comm: "ktstr_worker".to_string(),
            group_leader_pid: None,
            real_parent_pid: None,
            real_parent_comm: None,
            pgid: None,
            sid: None,
            nr_threads: None,
            weight: 100,
            prio: 120,
            static_prio: 120,
            normal_prio: 120,
            rt_priority: 0,
            sched_class: Some("fair".to_string()),
            core_cookie: None,
            pi_boosted_out_of_scx: false,
            nvcsw: 0,
            nivcsw: 0,
            signal_nvcsw: None,
            signal_nivcsw: None,
            utime: 0,
            stime: 0,
            signal_utime: None,
            signal_stime: None,
            lock_slowpath_match: None,
        };
        let json = serde_json::to_string(&e).unwrap();
        // Skipped fields must not appear in the JSON.
        assert!(!json.contains("group_leader_pid"));
        assert!(!json.contains("real_parent_pid"));
        assert!(!json.contains("pgid"));
        assert!(!json.contains("nr_threads"));
        assert!(!json.contains("core_cookie"));
        assert!(!json.contains("signal_nvcsw"));
        // signal_utime/signal_stime skip on None (exited-thread
        // accumulators absent when signal_struct didn't translate).
        assert!(!json.contains("signal_utime"));
        assert!(!json.contains("signal_stime"));
        assert!(!json.contains("lock_slowpath_match"));
        // Required fields must appear.
        assert!(json.contains("\"pid\":42"));
        assert!(json.contains("\"comm\":\"ktstr_worker\""));
        assert!(json.contains("\"weight\":100"));
        assert!(json.contains("\"sched_class\":\"fair\""));
        // utime/stime are unconditional (always serialized, even 0).
        assert!(json.contains("\"utime\":0"));
        assert!(json.contains("\"stime\":0"));
    }

    #[test]
    fn task_enrichment_serde_roundtrip_populated() {
        let e = TaskEnrichment {
            pid: 1234,
            tgid: 1230,
            comm: "stress-ng".to_string(),
            group_leader_pid: Some(1230),
            real_parent_pid: Some(1),
            real_parent_comm: Some("systemd".to_string()),
            pgid: Some(1230),
            sid: Some(1),
            nr_threads: Some(8),
            weight: 200,
            prio: 100,
            static_prio: 120,
            normal_prio: 100,
            rt_priority: 50,
            sched_class: Some("rt".to_string()),
            core_cookie: Some(0xc0c01e),
            pi_boosted_out_of_scx: true,
            nvcsw: 12345,
            nivcsw: 678,
            signal_nvcsw: Some(50_000),
            signal_nivcsw: Some(1_234),
            utime: 99_000,
            stime: 88_000,
            signal_utime: Some(7_000),
            signal_stime: Some(6_000),
            lock_slowpath_match: Some("queued_spin_lock_slowpath".to_string()),
        };
        let json = serde_json::to_string(&e).unwrap();
        let parsed: TaskEnrichment = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.pid, 1234);
        assert_eq!(parsed.comm, "stress-ng");
        assert_eq!(parsed.real_parent_comm.as_deref(), Some("systemd"));
        assert_eq!(parsed.nr_threads, Some(8));
        assert_eq!(parsed.core_cookie, Some(0xc0c01e));
        assert!(parsed.pi_boosted_out_of_scx);
        assert_eq!(parsed.utime, 99_000);
        assert_eq!(parsed.stime, 88_000);
        assert_eq!(parsed.signal_utime, Some(7_000));
        assert_eq!(parsed.signal_stime, Some(6_000));
        assert_eq!(
            parsed.lock_slowpath_match.as_deref(),
            Some("queued_spin_lock_slowpath"),
        );
    }

    // -- walk_task_enrichment memory-walk coverage --
    //
    // All fixtures use page_offset = 0 so a planted KVA equals its
    // DRAM offset (kva_to_pa is `kva.wrapping_sub(0)`), and
    // translate_any_kva returns the KVA directly when it is
    // `< mem.size()`. cr3_pa = 0 / l5 = false means an out-of-bounds
    // KVA falls through to the page-table walk, which fails with a
    // zero root => None. OOB scalar reads return 0 (GuestMem's
    // read_scalar), so every buffer is sized past the highest
    // touched offset.

    /// Synthetic struct layout shared by the walk fixtures. Offsets
    /// are chosen so no two read ranges overlap. `scx` base is 0x30
    /// and `see_weight` is 0x04 within it, so the weight read lands
    /// at task+0x34.
    fn fixture_offsets() -> TaskEnrichmentOffsets {
        TaskEnrichmentOffsets {
            // task_struct fields
            task_struct_pid: 0x00,
            task_struct_tgid: 0x04,
            task_struct_prio: 0x08,
            task_struct_static_prio: 0x0c,
            task_struct_normal_prio: 0x10,
            task_struct_rt_priority: 0x14,
            task_struct_comm: 0x18,
            task_struct_sched_class: 0x28,
            task_struct_scx: 0x30,
            task_struct_core_cookie: Some(0x38),
            task_struct_nvcsw: 0x40,
            task_struct_nivcsw: 0x48,
            task_struct_utime: 0x50,
            task_struct_stime: 0x58,
            task_struct_group_leader: 0x60,
            task_struct_real_parent: 0x68,
            task_struct_signal: 0x70,
            task_struct_stack: 0x78,
            // sched_ext_entity field (relative to scx base)
            see_weight: 0x04,
            // signal_struct fields
            signal_struct_nr_threads: 0x00,
            signal_struct_nvcsw: 0x08,
            signal_struct_nivcsw: 0x10,
            signal_struct_utime: 0x18,
            signal_struct_stime: 0x20,
            signal_struct_pids: 0x30,
            // struct pid / struct upid fields
            pid_numbers: 0x00,
            pid_size: 0x00,
            upid_nr: 0x00,
            upid_size: 16,
        }
    }

    // In-buffer addresses (== KVA == PA under page_offset = 0).
    const TASK_ADDR: u64 = 0x100;
    const SIGNAL_ADDR: u64 = 0x400;
    const PGID_PID_ADDR: u64 = 0x600;
    const SID_PID_ADDR: u64 = 0x680;
    const PARENT_TASK_ADDR: u64 = 0x800;
    // sched_class comparison sentinels. Never dereferenced — only
    // compared against the SchedClassRegistry slots.
    const EXT_CLASS_KVA: u64 = 0x9000;
    const FAIR_CLASS_KVA: u64 = 0x9100;

    fn put_u32(buf: &mut [u8], at: u64, off: usize, val: u32) {
        let a = at as usize + off;
        buf[a..a + 4].copy_from_slice(&val.to_le_bytes());
    }

    fn put_u64(buf: &mut [u8], at: u64, off: usize, val: u64) {
        let a = at as usize + off;
        buf[a..a + 8].copy_from_slice(&val.to_le_bytes());
    }

    fn put_comm(buf: &mut [u8], at: u64, off: usize, bytes: &[u8]) {
        let a = at as usize + off;
        buf[a..a + bytes.len()].copy_from_slice(bytes);
    }

    /// Plant a complete, well-formed task_struct + signal_struct +
    /// PGID/SID struct pids into `buf` using [`fixture_offsets`].
    /// sched_class is planted as [`EXT_CLASS_KVA`]. The caller may
    /// then mutate individual fields before walking to exercise a
    /// specific branch.
    fn plant_happy_task(buf: &mut [u8]) {
        let o = fixture_offsets();
        // Identity.
        put_u32(buf, TASK_ADDR, o.task_struct_pid, 4321);
        put_u32(buf, TASK_ADDR, o.task_struct_tgid, 4300);
        put_comm(buf, TASK_ADDR, o.task_struct_comm, b"worker\0");
        // Scheduling scalars.
        put_u32(buf, TASK_ADDR, o.task_struct_prio, 100);
        put_u32(buf, TASK_ADDR, o.task_struct_static_prio, 120);
        put_u32(buf, TASK_ADDR, o.task_struct_normal_prio, 100);
        put_u32(buf, TASK_ADDR, o.task_struct_rt_priority, 50);
        put_u64(buf, TASK_ADDR, o.task_struct_sched_class, EXT_CLASS_KVA);
        // weight at scx + see_weight.
        put_u32(buf, TASK_ADDR, o.task_struct_scx + o.see_weight, 200);
        // core_cookie.
        put_u64(buf, TASK_ADDR, o.task_struct_core_cookie.unwrap(), 0xABCD);
        // Per-task context-switch + cputime counters.
        put_u64(buf, TASK_ADDR, o.task_struct_nvcsw, 11);
        put_u64(buf, TASK_ADDR, o.task_struct_nivcsw, 22);
        put_u64(buf, TASK_ADDR, o.task_struct_utime, 999_000);
        put_u64(buf, TASK_ADDR, o.task_struct_stime, 888_000);
        // Pointer fields: parent + group_leader point at the mini
        // task; signal points at the signal_struct.
        put_u64(buf, TASK_ADDR, o.task_struct_group_leader, PARENT_TASK_ADDR);
        put_u64(buf, TASK_ADDR, o.task_struct_real_parent, PARENT_TASK_ADDR);
        put_u64(buf, TASK_ADDR, o.task_struct_signal, SIGNAL_ADDR);

        // Mini parent/group-leader task: only pid + comm are read by
        // follow_task_for_pid (group_leader) and
        // follow_task_for_pid_and_comm (real_parent), both at the
        // task_struct pid/comm offsets.
        put_u32(buf, PARENT_TASK_ADDR, o.task_struct_pid, 1);
        put_comm(buf, PARENT_TASK_ADDR, o.task_struct_comm, b"init\0");

        // signal_struct.
        put_u32(buf, SIGNAL_ADDR, o.signal_struct_nr_threads, 8);
        put_u64(buf, SIGNAL_ADDR, o.signal_struct_nvcsw, 70_000);
        put_u64(buf, SIGNAL_ADDR, o.signal_struct_nivcsw, 3);
        put_u64(buf, SIGNAL_ADDR, o.signal_struct_utime, 7_000);
        put_u64(buf, SIGNAL_ADDR, o.signal_struct_stime, 6_000);
        // pids[PGID] / pids[SID] slots -> struct pid KVAs.
        put_u64(
            buf,
            SIGNAL_ADDR,
            o.signal_struct_pids + pid_type::PGID * 8,
            PGID_PID_ADDR,
        );
        put_u64(
            buf,
            SIGNAL_ADDR,
            o.signal_struct_pids + pid_type::SID * 8,
            SID_PID_ADDR,
        );
        // struct pid numbers[0].nr for PGID (4300) and SID (1).
        put_u32(buf, PGID_PID_ADDR, o.pid_numbers + o.upid_nr, 4300);
        put_u32(buf, SID_PID_ADDR, o.pid_numbers + o.upid_nr, 1);
    }

    /// Build a direct-mapped test kernel over `buf` (page_offset = 0,
    /// cr3_pa = 0, l5 = false). `buf` must outlive the returned
    /// kernel — the GuestMem holds a raw pointer into it.
    fn build_kernel(buf: &mut [u8]) -> GuestKernel {
        // SAFETY: `buf` is a live caller-owned slice that outlives the
        // GuestKernel (and thus the GuestMem) in every test below.
        let mem =
            unsafe { crate::monitor::reader::GuestMem::new(buf.as_mut_ptr(), buf.len() as u64) };
        GuestKernel::new_for_test(
            std::sync::Arc::new(mem),
            std::collections::HashMap::new(),
            0,
            0,
            false,
        )
    }

    fn ext_registry() -> SchedClassRegistry {
        SchedClassRegistry {
            fair: Some(FAIR_CLASS_KVA),
            ext: Some(EXT_CLASS_KVA),
            ..Default::default()
        }
    }

    #[test]
    fn walk_task_enrichment_root_translate_fail_returns_none() {
        // task_kva == mem.size() => direct_pa == size (NOT < size) so
        // translate_any_kva skips the direct map, then the cr3_pa = 0
        // page-table walk fails => the whole fn returns None.
        let mut buf = vec![0u8; 0x2000];
        let kernel = build_kernel(&mut buf);
        let offsets = fixture_offsets();
        let oob_task_kva = kernel.mem().size(); // == 0x2000
        let result = walk_task_enrichment(
            &kernel,
            oob_task_kva,
            &offsets,
            &ext_registry(),
            &LockSlowpathRegistry::default(),
            false,
            None,
        );
        assert!(
            result.is_none(),
            "unreadable task_struct must abort the whole enrichment"
        );
    }

    #[test]
    fn walk_task_enrichment_full_happy_path_exact_fields() {
        let mut buf = vec![0u8; 0x2000];
        plant_happy_task(&mut buf);
        let kernel = build_kernel(&mut buf);
        let offsets = fixture_offsets();
        let e = walk_task_enrichment(
            &kernel,
            TASK_ADDR,
            &offsets,
            &ext_registry(),
            &LockSlowpathRegistry::default(),
            false,
            None,
        )
        .expect("task built");

        assert_eq!(e.pid, 4321);
        assert_eq!(e.tgid, 4300);
        assert_eq!(e.comm, "worker");
        assert_eq!(e.prio, 100);
        assert_eq!(e.static_prio, 120);
        assert_eq!(e.normal_prio, 100);
        assert_eq!(e.rt_priority, 50);
        assert_eq!(e.sched_class.as_deref(), Some("ext"));
        assert_eq!(e.weight, 200);
        assert_eq!(e.core_cookie, Some(0xABCD));
        assert_eq!(e.nvcsw, 11);
        assert_eq!(e.nivcsw, 22);
        assert_eq!(e.utime, 999_000);
        assert_eq!(e.stime, 888_000);
        assert_eq!(e.nr_threads, Some(8));
        assert_eq!(e.signal_nvcsw, Some(70_000));
        assert_eq!(e.signal_nivcsw, Some(3));
        assert_eq!(e.signal_utime, Some(7_000));
        assert_eq!(e.signal_stime, Some(6_000));
        assert_eq!(e.pgid, Some(4300));
        assert_eq!(e.sid, Some(1));
        assert_eq!(e.group_leader_pid, Some(1));
        assert_eq!(e.real_parent_pid, Some(1));
        assert_eq!(e.real_parent_comm.as_deref(), Some("init"));
        // pi_boosted_out_of_scx is false because is_runnable_in_scx
        // is false on this call.
        assert!(!e.pi_boosted_out_of_scx);
        // pc = None => no lock-slowpath match attempted.
        assert_eq!(e.lock_slowpath_match, None);
    }

    /// Helper for the pi_boosted truth table: plant a task whose
    /// sched_class is `sched_class_kva`, build a registry with the
    /// given `ext` slot, and return the resulting flag.
    fn pi_boosted(sched_class_kva: u64, ext: Option<u64>, is_runnable_in_scx: bool) -> bool {
        let mut buf = vec![0u8; 0x2000];
        plant_happy_task(&mut buf);
        let offsets = fixture_offsets();
        put_u64(&mut buf, TASK_ADDR, offsets.task_struct_sched_class, sched_class_kva);
        let kernel = build_kernel(&mut buf);
        let classes = SchedClassRegistry {
            fair: Some(FAIR_CLASS_KVA),
            ext,
            ..Default::default()
        };
        walk_task_enrichment(
            &kernel,
            TASK_ADDR,
            &offsets,
            &classes,
            &LockSlowpathRegistry::default(),
            is_runnable_in_scx,
            None,
        )
        .expect("task built")
        .pi_boosted_out_of_scx
    }

    #[test]
    fn walk_task_enrichment_pi_boosted_out_of_scx_truth_table() {
        // runnable + sched_class == ext => not boosted out.
        assert!(!pi_boosted(EXT_CLASS_KVA, Some(EXT_CLASS_KVA), true));
        // runnable + sched_class != ext + ext.is_some() => boosted out.
        assert!(pi_boosted(FAIR_CLASS_KVA, Some(EXT_CLASS_KVA), true));
        // runnable + ext == None => short-circuits on is_some() => false.
        assert!(!pi_boosted(FAIR_CLASS_KVA, None, true));
        // not runnable + sched_class != ext => false (gated on runnable).
        assert!(!pi_boosted(FAIR_CLASS_KVA, Some(EXT_CLASS_KVA), false));
    }

    #[test]
    fn walk_task_enrichment_core_cookie_absent_offset_yields_none() {
        // Even though a non-zero u64 sits at the would-be core_cookie
        // offset, task_struct_core_cookie = None skips the read.
        let mut buf = vec![0u8; 0x2000];
        plant_happy_task(&mut buf);
        let kernel = build_kernel(&mut buf);
        let mut offsets = fixture_offsets();
        offsets.task_struct_core_cookie = None;
        let e = walk_task_enrichment(
            &kernel,
            TASK_ADDR,
            &offsets,
            &ext_registry(),
            &LockSlowpathRegistry::default(),
            false,
            None,
        )
        .expect("task built");
        assert_eq!(e.core_cookie, None);
    }

    #[test]
    fn walk_task_enrichment_signal_null_pointer_all_signal_fields_none() {
        // signal_kva == 0 short-circuits before translate; the whole
        // signal-derived septuple is None.
        let mut buf = vec![0u8; 0x2000];
        plant_happy_task(&mut buf);
        let offsets = fixture_offsets();
        put_u64(&mut buf, TASK_ADDR, offsets.task_struct_signal, 0);
        let kernel = build_kernel(&mut buf);
        let e = walk_task_enrichment(
            &kernel,
            TASK_ADDR,
            &offsets,
            &ext_registry(),
            &LockSlowpathRegistry::default(),
            false,
            None,
        )
        .expect("task built");
        assert_eq!(e.nr_threads, None);
        assert_eq!(e.signal_nvcsw, None);
        assert_eq!(e.signal_nivcsw, None);
        assert_eq!(e.signal_utime, None);
        assert_eq!(e.signal_stime, None);
        assert_eq!(e.pgid, None);
        assert_eq!(e.sid, None);
        // The rest of the task still populated.
        assert_eq!(e.pid, 4321);
    }

    #[test]
    fn walk_task_enrichment_signal_translate_fail_all_signal_fields_none() {
        // signal_kva != 0 but out of bounds => translate_any_kva None
        // (distinct path from the signal_kva == 0 short-circuit).
        let mut buf = vec![0u8; 0x2000];
        plant_happy_task(&mut buf);
        let offsets = fixture_offsets();
        // 0xF000_0000 is far above mem.size() (0x2000); cr3_pa = 0 walk fails.
        put_u64(&mut buf, TASK_ADDR, offsets.task_struct_signal, 0xF000_0000);
        let kernel = build_kernel(&mut buf);
        let e = walk_task_enrichment(
            &kernel,
            TASK_ADDR,
            &offsets,
            &ext_registry(),
            &LockSlowpathRegistry::default(),
            false,
            None,
        )
        .expect("task built");
        assert_eq!(
            (
                e.nr_threads,
                e.signal_nvcsw,
                e.signal_nivcsw,
                e.signal_utime,
                e.signal_stime,
                e.pgid,
                e.sid
            ),
            (None, None, None, None, None, None, None)
        );
        assert_eq!(e.tgid, 4300);
    }

    #[test]
    fn read_pid_nr_at_index_null_slot_returns_none_via_pgid() {
        // PGID slot zeroed => pgid None (the common non-leader case);
        // SID slot populated => sid Some. Pins the pids_off + idx*8
        // index math discriminating PGID (idx 2) from SID (idx 3).
        let mut buf = vec![0u8; 0x2000];
        plant_happy_task(&mut buf);
        let offsets = fixture_offsets();
        put_u64(
            &mut buf,
            SIGNAL_ADDR,
            offsets.signal_struct_pids + pid_type::PGID * 8,
            0,
        );
        let kernel = build_kernel(&mut buf);
        let e = walk_task_enrichment(
            &kernel,
            TASK_ADDR,
            &offsets,
            &ext_registry(),
            &LockSlowpathRegistry::default(),
            false,
            None,
        )
        .expect("task built");
        assert_eq!(e.pgid, None);
        assert_eq!(e.sid, Some(1));
    }

    #[test]
    fn follow_task_for_pid_and_comm_null_and_translate_fail() {
        let offsets = fixture_offsets();

        // Sub-case A: both pointer fields NULL (init_task case).
        let mut buf_a = vec![0u8; 0x2000];
        plant_happy_task(&mut buf_a);
        put_u64(&mut buf_a, TASK_ADDR, offsets.task_struct_group_leader, 0);
        put_u64(&mut buf_a, TASK_ADDR, offsets.task_struct_real_parent, 0);
        let kernel_a = build_kernel(&mut buf_a);
        let ea = walk_task_enrichment(
            &kernel_a,
            TASK_ADDR,
            &offsets,
            &ext_registry(),
            &LockSlowpathRegistry::default(),
            false,
            None,
        )
        .expect("task built");
        assert_eq!(ea.group_leader_pid, None);
        assert_eq!(ea.real_parent_pid, None);
        assert_eq!(ea.real_parent_comm, None);

        // Sub-case B: non-null but out-of-bounds => translate fail.
        let mut buf_b = vec![0u8; 0x2000];
        plant_happy_task(&mut buf_b);
        put_u64(
            &mut buf_b,
            TASK_ADDR,
            offsets.task_struct_group_leader,
            0xF000_0000,
        );
        put_u64(
            &mut buf_b,
            TASK_ADDR,
            offsets.task_struct_real_parent,
            0xF000_0000,
        );
        let kernel_b = build_kernel(&mut buf_b);
        let eb = walk_task_enrichment(
            &kernel_b,
            TASK_ADDR,
            &offsets,
            &ext_registry(),
            &LockSlowpathRegistry::default(),
            false,
            None,
        )
        .expect("task built");
        assert_eq!(eb.group_leader_pid, None);
        assert_eq!(eb.real_parent_pid, None);
        assert_eq!(eb.real_parent_comm, None);

        // Sub-case C: valid parent (positive control) — the
        // happy-path layout already points both at the mini task.
        let mut buf_c = vec![0u8; 0x2000];
        plant_happy_task(&mut buf_c);
        let kernel_c = build_kernel(&mut buf_c);
        let ec = walk_task_enrichment(
            &kernel_c,
            TASK_ADDR,
            &offsets,
            &ext_registry(),
            &LockSlowpathRegistry::default(),
            false,
            None,
        )
        .expect("task built");
        assert_eq!(ec.group_leader_pid, Some(1));
        assert_eq!(ec.real_parent_pid, Some(1));
        assert_eq!(ec.real_parent_comm.as_deref(), Some("init"));
    }

    #[test]
    fn read_comm_no_nul_reads_full_16_bytes_and_nul_truncates() {
        let offsets = fixture_offsets();

        // Nul-terminated: truncates at the nul.
        let mut buf_t = vec![0u8; 0x2000];
        plant_happy_task(&mut buf_t);
        put_comm(&mut buf_t, TASK_ADDR, offsets.task_struct_comm, b"short\0");
        let kernel_t = build_kernel(&mut buf_t);
        let et = walk_task_enrichment(
            &kernel_t,
            TASK_ADDR,
            &offsets,
            &ext_registry(),
            &LockSlowpathRegistry::default(),
            false,
            None,
        )
        .expect("task built");
        assert_eq!(et.comm, "short");

        // No nul within the 16-byte window: reads the full 16 bytes.
        let mut buf_n = vec![0u8; 0x2000];
        plant_happy_task(&mut buf_n);
        put_comm(
            &mut buf_n,
            TASK_ADDR,
            offsets.task_struct_comm,
            b"sixteencharcomm!",
        );
        let kernel_n = build_kernel(&mut buf_n);
        let en = walk_task_enrichment(
            &kernel_n,
            TASK_ADDR,
            &offsets,
            &ext_registry(),
            &LockSlowpathRegistry::default(),
            false,
            None,
        )
        .expect("task built");
        assert_eq!(en.comm, "sixteencharcomm!");
        assert_eq!(en.comm.len(), 16);

        // Invalid UTF-8 before the nul: from_utf8_lossy substitutes
        // the replacement char rather than panicking.
        let mut buf_l = vec![0u8; 0x2000];
        plant_happy_task(&mut buf_l);
        put_comm(&mut buf_l, TASK_ADDR, offsets.task_struct_comm, &[0xFF, 0x00]);
        let kernel_l = build_kernel(&mut buf_l);
        let el = walk_task_enrichment(
            &kernel_l,
            TASK_ADDR,
            &offsets,
            &ext_registry(),
            &LockSlowpathRegistry::default(),
            false,
            None,
        )
        .expect("task built");
        assert_eq!(el.comm, "\u{FFFD}");
    }

    #[test]
    fn lock_slowpath_match_pc_supplied_sets_field_and_nonmatch_clears() {
        let offsets = fixture_offsets();
        let locks = LockSlowpathRegistry {
            queued_spin_lock_slowpath: Some(0x5_0000),
            ..Default::default()
        };

        // Matching PC inside the qsl window => field set.
        let mut buf_m = vec![0u8; 0x2000];
        plant_happy_task(&mut buf_m);
        let kernel_m = build_kernel(&mut buf_m);
        let em = walk_task_enrichment(
            &kernel_m,
            TASK_ADDR,
            &offsets,
            &ext_registry(),
            &locks,
            false,
            Some(0x5_0010),
        )
        .expect("task built");
        assert_eq!(
            em.lock_slowpath_match.as_deref(),
            Some("queued_spin_lock_slowpath")
        );

        // Non-matching PC => field stays None.
        let mut buf_x = vec![0u8; 0x2000];
        plant_happy_task(&mut buf_x);
        let kernel_x = build_kernel(&mut buf_x);
        let ex = walk_task_enrichment(
            &kernel_x,
            TASK_ADDR,
            &offsets,
            &ext_registry(),
            &locks,
            false,
            Some(0x9_9999),
        )
        .expect("task built");
        assert_eq!(ex.lock_slowpath_match, None);
    }

    #[test]
    fn lock_slowpath_match_pc_window_overflow_returns_none() {
        // A symbol KVA near u64::MAX whose `s + 4096` window wraps
        // must not falsely match a PC that would be inside the
        // non-wrapping window. checked_add returning None => no match.
        let r = LockSlowpathRegistry {
            queued_spin_lock_slowpath: Some(u64::MAX - 100),
            ..Default::default()
        };
        assert_eq!(r.match_pc(u64::MAX - 50), None);
        assert_eq!(r.match_pc(u64::MAX), None);
    }
}
