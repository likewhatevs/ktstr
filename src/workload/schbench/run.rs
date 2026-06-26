//! schbench's run engine, ported faithfully from `schbench.c`: the
//! message-thread / worker-thread topology, the lockless wait-list, the
//! handshake-driven wakeup-latency loop, and the matrix work under the
//! per-CPU lock. This is the default (non-RPS) mode -- the wakeup-latency +
//! request-latency benchmark, with per-phase schedstat (run-delay) capture. The
//! RPS injector, request queue, and auto-rps are a later phase.
//!
//! # Topology (`schbench.c` `message_thread` :1540, `worker_thread` :1419)
//!
//! `message_threads` message threads each spawn `worker_threads` worker
//! threads in one process (schbench is single-process pthreads; schbench_rs
//! re-expresses that with [`std::thread`], so the
//! [`handshake`](super::handshake) futex is PRIVATE and the per-CPU locks +
//! matrices live in one address space). A worker loops: block until its
//! message thread wakes it (measuring wakeup latency), then think-sleep +
//! matrix work under the per-CPU lock (measuring request latency). The message
//! thread batch-wakes all waiting workers (`run_msg_thread` :1166 /
//! `xlist_wake_all` :969).
//!
//! # Fidelity
//!
//! - Clock: `CLOCK_MONOTONIC` (ruling), not schbench's `gettimeofday`
//!   wall-clock. Monotonic is freeze-robust (a host-side VM pause cannot make
//!   a delta go negative) and is the correct source for a latency delta. The
//!   measured quantity -- elapsed nanoseconds between two reads -- is identical
//!   to what schbench measures; only the clock id differs.
//! - The lockless wait-list is a Treiber stack ([`TreiberStack`]) over an
//!   intrusive next-pointer, matching schbench's `xlist_add`/`xlist_splice`
//!   cmpxchg list (`schbench.c:866-896`): userspace CAS only, no lock/syscall
//!   on the hot path, so the syscall profile stays futex-dominated like
//!   schbench's. (schbench hand-duplicates this for its thread list and its
//!   request list; schbench_rs shares one generic implementation.)

use core::cell::UnsafeCell;
use core::ptr;
use core::sync::atomic::{AtomicBool, AtomicPtr, AtomicU32, AtomicU64, Ordering};

use super::percpu_lock::PerCpuLocks;
use super::plat::{Percentiles, PlatStats};

/// Read `CLOCK_MONOTONIC` as nanoseconds (ruling: monotonic, not wall-clock).
/// Self-contained so the schbench modules build into the standalone validation
/// binary without reaching into `worker`'s `pub(super)` clock wrapper.
fn monotonic_nanos() -> u64 {
    // SAFETY: `clock_gettime` writes a `timespec` through the out-pointer and
    // reads nothing else; CLOCK_MONOTONIC is always available on Linux.
    let mut ts: libc::timespec = unsafe { core::mem::zeroed() };
    let rc = unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut ts) };
    assert_eq!(rc, 0, "clock_gettime(CLOCK_MONOTONIC) failed");
    (ts.tv_sec as u64) * 1_000_000_000 + ts.tv_nsec as u64
}

/// A node that can be linked into a [`TreiberStack`] via an intrusive
/// next-pointer it owns.
trait Linked: Sized {
    fn next_link(&self) -> &AtomicPtr<Self>;
}

/// Lockless LIFO stack via an intrusive next-pointer, the shared form of
/// schbench's `xlist_add`/`xlist_splice` (`schbench.c:866-896`). Nodes are
/// referenced by raw pointer and must outlive every concurrent operation; the
/// caller owns their storage and lifetimes.
struct TreiberStack<T: Linked> {
    head: AtomicPtr<T>,
}

impl<T: Linked> TreiberStack<T> {
    fn new() -> Self {
        Self {
            head: AtomicPtr::new(ptr::null_mut()),
        }
    }

    /// Push `node` onto the stack. Faithful port of `xlist_add`
    /// (`schbench.c:866`): set the node's link to the current head, then
    /// CAS-publish it, retrying on contention. The caller must not push a node
    /// that is already on the stack (its link would be overwritten while
    /// reachable) -- schbench upholds this (a worker re-queues only after being
    /// spliced off and woken).
    fn add(&self, node: *mut T) {
        // SAFETY: `node` is a valid pointer to a `T` that outlives all list
        // operations (caller contract).
        let link = unsafe { (*node).next_link() };
        loop {
            let old = self.head.load(Ordering::Acquire);
            link.store(old, Ordering::Relaxed);
            if self
                .head
                .compare_exchange(old, node, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                return;
            }
        }
    }

    /// Atomically take the whole stack, leaving it empty, returning the head
    /// (null if empty). Faithful port of `xlist_splice` (`schbench.c:884`): an
    /// atomic swap to null (the single-op equivalent of schbench's CAS loop).
    /// The returned chain is walked via [`Linked::next_link`] in LIFO order
    /// (schbench's thread-list splice does not reverse; wake order is
    /// irrelevant).
    fn splice(&self) -> *mut T {
        self.head.swap(ptr::null_mut(), Ordering::AcqRel)
    }
}

/// Declarative config for the [`Schbench`](crate::workload::WorkType::Schbench)
/// workload. Construct via [`SchbenchConfig::default`] (schbench's own
/// defaults) plus the chainable setters, e.g.
/// `SchbenchConfig::default().message_threads(2).worker_threads(4)`. Derives
/// Clone/Debug/PartialEq/Eq/Hash/serde; the builder shape follows
/// [`WorkloadConfig`](crate::workload::WorkloadConfig), but `Eq`+`Hash` (which
/// `WorkloadConfig` and `WorkSpec` omit because of their transitive `f64`) are
/// available here since every field is integer/bool -- the ktstr f64-free
/// leaf-config convention.
#[derive(Clone, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct SchbenchConfig {
    /// Number of message threads (`schbench.c` `-m`, default 1).
    pub message_threads: usize,
    /// Worker threads per message thread (`schbench.c` `-t`). 0 means "one per
    /// CPU in the allocated cpuset" (schbench's `get_nprocs` default, scoped to
    /// the guest cpuset per ruling).
    pub worker_threads: usize,
    /// Per-worker matrix cache footprint in KiB (`schbench.c` `-F`, default
    /// 256); sets the matrix dimension.
    pub cache_footprint_kib: usize,
    /// Matrix multiplications per work cycle (`schbench.c` `-n`, default 5).
    pub operations: usize,
    /// Think-time sleep before the matrix work, microseconds (`schbench.c`
    /// `-s`, default 100); simulates networking. 0 disables.
    pub sleep_usec: u64,
    /// Skip the per-CPU lock around the matrix work (`schbench.c` `-L`,
    /// default false: locking on).
    pub skip_locking: bool,
}

impl Default for SchbenchConfig {
    fn default() -> Self {
        // schbench defaults (schbench.c option table + globals).
        Self {
            message_threads: 1,
            worker_threads: 0,
            cache_footprint_kib: 256,
            operations: 5,
            sleep_usec: 100,
            skip_locking: false,
        }
    }
}

impl SchbenchConfig {
    /// Set the number of message threads (schbench `-m`).
    #[must_use = "builder methods consume self; bind the result"]
    pub fn message_threads(mut self, n: usize) -> Self {
        self.message_threads = n;
        self
    }
    /// Set worker threads per message thread (schbench `-t`); 0 = one per
    /// allocated CPU.
    #[must_use = "builder methods consume self; bind the result"]
    pub fn worker_threads(mut self, n: usize) -> Self {
        self.worker_threads = n;
        self
    }
    /// Set the per-worker matrix cache footprint in KiB (schbench `-F`).
    #[must_use = "builder methods consume self; bind the result"]
    pub fn cache_footprint_kib(mut self, kib: usize) -> Self {
        self.cache_footprint_kib = kib;
        self
    }
    /// Set the matrix multiplications per work cycle (schbench `-n`).
    #[must_use = "builder methods consume self; bind the result"]
    pub fn operations(mut self, n: usize) -> Self {
        self.operations = n;
        self
    }
    /// Set the think-time sleep in microseconds (schbench `-s`); 0 disables.
    #[must_use = "builder methods consume self; bind the result"]
    pub fn sleep_usec(mut self, usec: u64) -> Self {
        self.sleep_usec = usec;
        self
    }
    /// Skip the per-CPU lock around the matrix work (schbench `-L`).
    #[must_use = "builder methods consume self; bind the result"]
    pub fn skip_locking(mut self, skip: bool) -> Self {
        self.skip_locking = skip;
        self
    }

    /// Matrix dimension from the cache footprint, identical to schbench
    /// (`schbench.c:1880`: `sqrt(cache_footprint_kb * 1024 / 3 /
    /// sizeof(unsigned long))`) and to ktstr's `FanOutCompute` precompute. Zero
    /// `operations` or `cache_footprint_kib` yields a 0 dimension (no matrix
    /// work).
    pub(crate) fn matrix_size(&self) -> usize {
        if self.operations > 0 && self.cache_footprint_kib > 0 {
            ((self.cache_footprint_kib * 1024 / 3 / core::mem::size_of::<u64>()) as f64).sqrt()
                as usize
        } else {
            0
        }
    }
}

/// Per-thread state, the Rust counterpart of schbench's `struct thread_data`
/// (`schbench.c:766`). Shared across threads by raw pointer for the lockless
/// wait-list, so cross-thread access is restricted to the atomic fields
/// (`next`, `futex`, `wake_time`); the histogram fields are owned solely by the
/// worker thread the `ThreadData` belongs to.
pub(crate) struct ThreadData {
    /// Treiber-stack link for the message thread's wait-list
    /// (schbench `thread_data->next`, `:776`). Null when not queued.
    next: AtomicPtr<ThreadData>,
    /// Wake handshake futex (`schbench.c` `thread_data->futex`, `:791`).
    futex: super::handshake::Handshake,
    /// Monotonic-ns timestamp the waker stamps just before posting, so the
    /// woken thread can measure scheduler wakeup latency (schbench `wake_time`,
    /// `:788`, stamped in `xlist_wake_all` `:984`).
    wake_time: AtomicU64,
    /// Wakeup-latency histogram (`schbench.c` `wakeup_stats`, `:794`).
    /// Owner-thread-only; see the `Sync` impl SAFETY note.
    wakeup_stats: UnsafeCell<PlatStats>,
    /// Request (work-cycle) latency histogram (`schbench.c` `request_stats`,
    /// `:795`). Owner-thread-only.
    request_stats: UnsafeCell<PlatStats>,
    /// Mean per-schedule run-queue wait (ns), read from `/proc/<tid>/schedstat`
    /// at thread exit (schbench's `read_sched_delay`, `:1118`). Owner-only.
    /// Feeds the WHOLE-RUN `SchbenchResult` (mean-of-means, matching real
    /// schbench for validation); the per-phase delay comes from
    /// [`Self::phase_snapshots`] instead.
    sched_delay_ns: UnsafeCell<u64>,
    /// Per-phase snapshots this thread accumulated via drain-on-change against
    /// the shared `phase_epoch` (one [`PhaseSnapshot`] per phase the thread did
    /// work in, plus a final in-flight phase at exit). Owner-thread-only,
    /// drained by the main thread after join — same happens-before as the
    /// histogram cells. Empty when run non-phasic (`phase_epoch == None`) until
    /// the single end-of-run drain.
    phase_snapshots: UnsafeCell<Vec<PhaseSnapshot>>,
}

/// One thread's latency + run-queue-delay accumulation over a single phase
/// (the window between two `phase_epoch` transitions). Built on the owning
/// thread at each drain-on-change boundary; the histograms are `take`n from the
/// live cells (snapshot-and-reset) and the run-delay is a RAW
/// `/proc/<tid>/schedstat` `(run_delay, pcount)` DELTA over the phase, so the
/// host re-derives the per-phase mean as `Σrun_delay / Σpcount` (the
/// sample-weighted pooled mean — a deliberate divergence from schbench's
/// whole-run mean-of-means, correct for a heterogeneous per-phase thread set).
/// Message threads leave `wakeup`/`request` empty and `loop_count` 0 (they
/// carry only run-delay); workers fill all fields.
struct PhaseSnapshot {
    /// The `phase_epoch` value active while this snapshot's samples were
    /// recorded (NOT the value at drain time). `0` = BASELINE, `u32::MAX` =
    /// inter-step gap; both are emitted as-is and discarded host-side.
    epoch: u32,
    wakeup: PlatStats,
    request: PlatStats,
    /// `/proc/<tid>/schedstat` field 2 (run_delay ns) delta over this phase.
    run_delay_ns: u64,
    /// `/proc/<tid>/schedstat` field 3 (pcount = timeslices) delta over this
    /// phase. The host guards `pcount == 0` → metric absent (never a div-by-zero
    /// 0), matching [`mean_sched_delay`].
    pcount: u64,
    /// Completed work cycles this worker ran in this phase (0 for the message
    /// thread).
    loop_count: u64,
}

/// Per-phase, cross-thread aggregate for one `phase_epoch`: the wakeup + request
/// histograms merged across every worker that ran in the phase, plus the
/// run-delay raw pairs split by thread class (message vs worker). The per-phase
/// wire carrier — rides inside [`crate::workload::PhaseSlice`] guest→host. All
/// fields are integer, so the host re-pools across workers/cgroups by
/// [`PlatStats::combine`] (histogram add) + integer sums; percentiles are
/// re-derived from the merged histogram, NEVER averaged.
///
/// ESTIMATOR NOTE: the per-phase run-delay mean the host derives from
/// `*_run_delay_ns / *_pcount` is SAMPLE-WEIGHTED (`Σrun_delay / Σpcount`), a
/// DIFFERENT estimator from the whole-run [`SchbenchResult`]'s `sched_delay_*`,
/// which keeps schbench's mean-of-per-thread-means (`collect_sched_delay`) for
/// parity. They measure different things by design — never cross-compare a
/// per-phase value against a whole-run threshold (or vice-versa). `pcount == 0`
/// for a class means that class was never scheduled in the phase → the host
/// emits the metric as ABSENT, not `0`.
#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub(crate) struct SchbenchPhaseStats {
    /// Wakeup-latency histogram merged across the phase's workers.
    pub(crate) wakeup: PlatStats,
    /// Request-latency histogram merged across the phase's workers.
    pub(crate) request: PlatStats,
    /// Σ message-thread `run_delay` (ns) over the phase.
    pub(crate) msg_run_delay_ns: u64,
    /// Σ message-thread `pcount` over the phase (the mean's denominator).
    pub(crate) msg_pcount: u64,
    /// Σ worker `run_delay` (ns) over the phase.
    pub(crate) worker_run_delay_ns: u64,
    /// Σ worker `pcount` over the phase (the mean's denominator).
    pub(crate) worker_pcount: u64,
    /// Σ completed work cycles across the phase's workers.
    pub(crate) loop_count: u64,
}

impl SchbenchPhaseStats {
    /// Merge `other` into `self`: combine the wakeup + request histograms
    /// (bucket-count addition, [`PlatStats::combine`]) and integer-add the
    /// run-delay raw pairs + loop_count. Associative AND commutative (combine +
    /// `saturating_add` both are), so pooling is order-independent — the SAME
    /// operation whether pooling per-engine across message threads (in [`run`])
    /// or per-cgroup across workers host-side
    /// (`crate::assert::PhaseCgroupStats::merge`). Percentiles are NEVER
    /// merged; the merged histogram is re-derived to percentiles by the reader.
    pub(crate) fn merge(&mut self, other: &SchbenchPhaseStats) {
        self.wakeup.combine(&other.wakeup);
        self.request.combine(&other.request);
        self.msg_run_delay_ns = self.msg_run_delay_ns.saturating_add(other.msg_run_delay_ns);
        self.msg_pcount = self.msg_pcount.saturating_add(other.msg_pcount);
        self.worker_run_delay_ns = self
            .worker_run_delay_ns
            .saturating_add(other.worker_run_delay_ns);
        self.worker_pcount = self.worker_pcount.saturating_add(other.worker_pcount);
        self.loop_count = self.loop_count.saturating_add(other.loop_count);
    }
}

impl Linked for ThreadData {
    fn next_link(&self) -> &AtomicPtr<Self> {
        &self.next
    }
}

// SAFETY: `ThreadData` is shared across threads only via the lockless
// wait-list, whose operations touch exclusively the atomic fields (`next`,
// `futex`, `wake_time`) -- all internally synchronized. The `UnsafeCell`
// fields (`wakeup_stats`, `request_stats`, `sched_delay_ns`,
// `phase_snapshots`) are written and read ONLY by the single worker thread
// that owns this `ThreadData`, and only the main thread reads/drains them
// after all workers have joined (a happens-before via the join). The
// per-phase drain (`worker_loop` / `run_msg_thread`) `take`s and pushes into
// the owning thread's own cells, never another's. No two threads ever touch a
// cell concurrently, so sharing `&ThreadData` across threads is sound.
unsafe impl Sync for ThreadData {}

impl ThreadData {
    fn new() -> Self {
        Self {
            next: AtomicPtr::new(ptr::null_mut()),
            futex: super::handshake::Handshake::new(),
            wake_time: AtomicU64::new(0),
            wakeup_stats: UnsafeCell::new(PlatStats::default()),
            request_stats: UnsafeCell::new(PlatStats::default()),
            sched_delay_ns: UnsafeCell::new(0),
            phase_snapshots: UnsafeCell::new(Vec::new()),
        }
    }
}

/// schbench's per-request think-time sleep ("simulated networking",
/// `schbench.c:1461`). This is the workload's defined behavior, not a
/// synchronization wait: `usleep` maps to `clock_nanosleep`, matching
/// schbench's syscall profile.
fn think_sleep(usec: u64) {
    std::thread::sleep(std::time::Duration::from_micros(usec));
}

/// One work cycle's matrix multiplications under the per-CPU lock. Faithful to
/// schbench's `do_work` (`schbench.c:1379`): take the current CPU's mutex
/// (unless `skip_locking`), run `operations` matrix multiplies on the worker's
/// buffer, unlock. The guard drops at the end of this function, holding the
/// lock across all operations exactly as schbench does (`:1387`/`:1411`).
fn do_work(
    matrix_buf: &mut [u64],
    matrix_size: usize,
    operations: usize,
    locks: Option<&PerCpuLocks>,
    work_units: &mut u64,
) {
    let _guard = locks.map(|l| l.lock_this_cpu());
    for _ in 0..operations {
        if matrix_size > 0 {
            crate::workload::worker::matrix_multiply(matrix_buf, matrix_size, work_units);
        }
    }
}

/// Worker side of one wakeup cycle. Faithful to schbench's `msg_and_wait`
/// (`schbench.c:997`, default branch): stamp our `wake_time`, push ourselves
/// onto the message thread's wait-list, wake the message thread, then block
/// until it wakes us back; record the wakeup (scheduler) latency. The `!stop`
/// guard mirrors schbench's `if (!stopping)` (`schbench.c:1030`) so we do not
/// block during shutdown.
fn msg_and_wait(
    td: &ThreadData,
    msg_td: &ThreadData,
    wait_list: &TreiberStack<ThreadData>,
    stop: &AtomicBool,
) {
    // Our futex is BLOCKED here (consumed by the prior wait, or fresh).
    td.wake_time.store(monotonic_nanos(), Ordering::Release);
    wait_list.add(td as *const ThreadData as *mut ThreadData);
    msg_td.futex.post();
    if !stop.load(Ordering::Acquire) {
        td.futex.wait_forever();
    }
    let now = monotonic_nanos();
    let wake = td.wake_time.load(Ordering::Acquire);
    // schbench buckets in microseconds (gettimeofday resolution); the monotonic
    // clock gives ns, so divide. `if delta > 0` matches schbench (`:1036`).
    let delta_us = now.saturating_sub(wake) / 1000;
    if delta_us > 0 {
        // SAFETY: only this worker thread accesses its own wakeup_stats cell.
        unsafe { (*td.wakeup_stats.get()).add_lat(delta_us.min(u32::MAX as u64) as u32) };
    }
}

/// Wake every worker on the wait-list, stamping each one's `wake_time` so it
/// can measure scheduler latency. Faithful to schbench's `xlist_wake_all`
/// (`schbench.c:969`): splice the whole list, read the clock ONCE, stamp every
/// worker with that single time, then post each. The single clock read is what
/// detects the scheduler preempting the waker mid-batch (`schbench.c:961-964`).
fn wake_all(wait_list: &TreiberStack<ThreadData>) {
    let mut cur = wait_list.splice();
    let now = monotonic_nanos();
    while !cur.is_null() {
        // SAFETY: `cur` is a worker `ThreadData` alive for the whole run; only
        // its atomic fields (next / wake_time / futex) are touched here.
        let td = unsafe { &*cur };
        let next = td.next.load(Ordering::Acquire);
        td.next.store(ptr::null_mut(), Ordering::Relaxed);
        td.wake_time.store(now, Ordering::Release);
        td.futex.post();
        cur = next;
    }
}

/// The message thread's loop. Faithful to schbench's `run_msg_thread`
/// (`schbench.c:1166`): batch-wake all waiting workers, then block until a
/// worker posts us back. On stop, drain once more (to wake any worker that
/// queued during the final wake) and exit.
///
/// `phase_epoch` drives the per-phase drain exactly as in [`worker_loop`], but
/// the message thread records only per-phase RUN-DELAY (its `wakeup`/`request`
/// histograms stay empty, taken empty by [`drain_phase`]). It polls the epoch
/// after each wake cycle; a phase in which no worker ever posts it leaves that
/// phase's run-delay drained late at the next wake (bounded boundary fuzz,
/// matching the worker drain).
fn run_msg_thread(
    msg_td: &ThreadData,
    wait_list: &TreiberStack<ThreadData>,
    stop: &AtomicBool,
    phase_epoch: Option<&AtomicU32>,
) {
    let tid = gettid_self();
    let mut cur_epoch = phase_epoch.map_or(0, |e| e.load(Ordering::Relaxed));
    let mut phase_ss_start = read_schedstat_raw(tid);
    loop {
        // msg_td.futex is BLOCKED here (consumed by the prior wait, or fresh).
        wake_all(wait_list);
        if stop.load(Ordering::Acquire) {
            wake_all(wait_list);
            break;
        }
        msg_td.futex.wait_forever();
        if let Some(pe) = phase_epoch {
            let new_epoch = pe.load(Ordering::Relaxed);
            if new_epoch != cur_epoch {
                let ss_end = read_schedstat_raw(tid);
                // SAFETY: this is the thread that owns msg_td (the coordinator
                // thread of run_one_message_thread).
                unsafe { drain_phase(msg_td, cur_epoch, phase_ss_start, ss_end, 0) };
                cur_epoch = new_epoch;
                phase_ss_start = ss_end;
            }
        }
    }
    // Final drain: close the still-open phase (the sole snapshot when non-phasic).
    let ss_end = read_schedstat_raw(tid);
    // SAFETY: this is the thread that owns msg_td.
    unsafe { drain_phase(msg_td, cur_epoch, phase_ss_start, ss_end, 0) };
    // Record the message thread's whole-run mean run-queue wait (`schbench.c:1664`),
    // reusing the cumulative `ss_end` pair just read — one /proc read, so the mean
    // and the final-phase delta derive from one consistent snapshot.
    // SAFETY: owner-only access to msg_td's sched_delay_ns cell.
    unsafe { *msg_td.sched_delay_ns.get() = mean_sched_delay(ss_end) };
}

/// The calling thread's kernel tid (`gettid`), for reading its own schedstat.
fn gettid_self() -> libc::pid_t {
    // SAFETY: gettid takes no arguments and only returns the caller's tid.
    unsafe { libc::syscall(libc::SYS_gettid) as libc::pid_t }
}

/// Raw `(run_delay_ns, pcount)` pair from `/proc/<tid>/schedstat` (fields 2, 3),
/// UNDIVIDED — schbench's `read_sched_delay` source (`schbench.c:1118`) without
/// the division. Both consumers want the raw pair, not a pre-divided mean: the
/// per-phase path re-pools `Σrun_delay / Σpcount` host-side, and the whole-run
/// per-thread mean is [`mean_sched_delay`] of the thread's final cumulative
/// pair. An absent file (the thread exited) yields `(0, 0)`, like schbench's
/// fopen-failure path.
///
/// `CONFIG_SCHED_INFO` (selected by `CONFIG_SCHEDSTATS`, and also by
/// `CONFIG_TASK_DELAY_ACCT` -- ktstr.kconfig enables both) populates these
/// fields even under sched_ext and regardless of the `kernel.sched_schedstats`
/// sysctl (that sysctl gates the separate per-rq/domain counters, not
/// `sched_info`).
fn read_schedstat_raw(tid: libc::pid_t) -> (u64, u64) {
    match std::fs::read_to_string(format!("/proc/{tid}/schedstat")) {
        Ok(s) => parse_schedstat_raw(&s),
        // The thread may have exited; schbench's fopen-failure path also -> 0.
        Err(_) => (0, 0),
    }
}

/// Parse a `/proc/<tid>/schedstat` line into the raw `(run_delay_ns, pcount)`
/// pair (field 2 = run_delay ns, field 3 = pcount = timeslices). The kernel's
/// `proc_pid_schedstat` always emits three integer fields, so a malformed
/// *present* line is a kernel/parse bug: panic rather than silently report 0
/// (matching the handshake's fail-loud stance + the no-silent-wrong-answer
/// rule). An absent file is the caller's concern, handled as `(0, 0)`.
fn parse_schedstat_raw(s: &str) -> (u64, u64) {
    let mut fields = s.split_whitespace();
    let _run = fields.next(); // field 1: sum_exec_runtime (on-CPU ns), unused
    let run_delay: u64 = fields
        .next()
        .and_then(|f| f.parse().ok())
        .expect("schedstat field 2 (run_delay) must be a present integer");
    let pcount: u64 = fields
        .next()
        .and_then(|f| f.parse().ok())
        .expect("schedstat field 3 (pcount) must be a present integer");
    (run_delay, pcount)
}

/// Mean per-schedule run-queue wait (ns) from a raw `(run_delay_ns, pcount)`
/// pair: `run_delay / pcount`, schbench's `read_sched_delay` arithmetic
/// (`schbench.c:1146`). Guards `pcount == 0` (a never-scheduled thread, or the
/// `!sched_info_on()` "0 0 0" line) to 0, where schbench divides by zero. Used
/// for the WHOLE-RUN per-thread mean (the mean-of-means component); the
/// per-phase path instead keeps the raw pair and re-pools `Σrd/Σpc` host-side.
fn mean_sched_delay((run_delay, pcount): (u64, u64)) -> u64 {
    if pcount == 0 { 0 } else { run_delay / pcount }
}

/// Finalize the just-ended phase on the OWNING thread: `take` the live
/// histograms (snapshot-and-reset, so the next phase starts clean) and push a
/// [`PhaseSnapshot`] tagged `epoch`, carrying the schedstat run-delay/pcount
/// delta (`ss_end − ss_start`) and this thread's `loop_count` for the phase.
/// Mirrors the worker/mod.rs:3327-3374 backdrop drain-on-change. For the
/// message thread the histograms are unwritten (taken empty) and `loop_count`
/// is 0; an empty histogram folds harmlessly host-side ([`PlatStats::combine`]).
///
/// # Safety
/// Must be called by the thread that owns `td` — it is the sole writer of
/// `td`'s `wakeup_stats` / `request_stats` / `phase_snapshots` cells (the
/// [`ThreadData`] `Sync` contract).
unsafe fn drain_phase(
    td: &ThreadData,
    epoch: u32,
    ss_start: (u64, u64),
    ss_end: (u64, u64),
    loop_count: u64,
) {
    // SAFETY: owner-only cell access, per this function's contract.
    unsafe {
        let wakeup = (*td.wakeup_stats.get()).take();
        let request = (*td.request_stats.get()).take();
        (*td.phase_snapshots.get()).push(PhaseSnapshot {
            epoch,
            wakeup,
            request,
            run_delay_ns: ss_end.0.saturating_sub(ss_start.0),
            pcount: ss_end.1.saturating_sub(ss_start.1),
            loop_count,
        });
    }
}

/// The per-message-thread shared context every worker borrows — bundled so
/// [`worker_loop`]'s signature stays within clippy's argument budget. All
/// fields are `Copy` shared references; every worker spawned by one
/// [`run_one_message_thread`] reads the same instance.
struct WorkerCtx<'a> {
    msg_td: &'a ThreadData,
    wait_list: &'a TreiberStack<ThreadData>,
    locks: Option<&'a PerCpuLocks>,
    config: &'a SchbenchConfig,
    stop: &'a AtomicBool,
    progress: &'a AtomicU64,
    phase_epoch: Option<&'a AtomicU32>,
}

/// One worker thread's loop. Faithful to schbench's `worker_thread`
/// (`schbench.c:1419`, default branch): block until woken (recording wakeup
/// latency), then think-sleep + matrix work under the per-CPU lock (recording
/// request latency), until stop. `ctx.progress` counts completed work cycles
/// across all workers (the live cycle counter / achieved request rate).
///
/// `ctx.phase_epoch` (when `Some`) is the shared per-phase generation the
/// scenario engine bumps at each step boundary. The worker polls it once per
/// cycle and, on change, drains the just-ended phase ([`drain_phase`]) —
/// `take`ing its own histograms + a schedstat raw delta — so per-phase
/// percentiles are isolated across e.g. an scx→detached-EEVDF transition. A
/// final drain at exit closes the last phase (and is the SOLE snapshot when
/// non-phasic, `cur_epoch` 0).
fn worker_loop(td: &ThreadData, ctx: &WorkerCtx) {
    // Destructure the all-`Copy` context into the same bindings the body used
    // before the bundle, so the loop logic below reads unchanged.
    let WorkerCtx {
        msg_td,
        wait_list,
        locks,
        config,
        stop,
        progress,
        phase_epoch,
    } = *ctx;
    let tid = gettid_self();
    let matrix_size = config.matrix_size();
    let mut matrix_buf = if matrix_size > 0 {
        vec![0u64; 3 * matrix_size * matrix_size]
    } else {
        Vec::new()
    };
    let mut work_units = 0u64;
    // Per-phase drain-on-change state. `cur_epoch` is the epoch the current
    // phase's samples belong to; `phase_ss_start` baselines the schedstat raw
    // pair so each phase reports its own delta.
    let mut cur_epoch = phase_epoch.map_or(0, |e| e.load(Ordering::Relaxed));
    let mut phase_ss_start = read_schedstat_raw(tid);
    let mut phase_loop_count = 0u64;
    while !stop.load(Ordering::Acquire) {
        msg_and_wait(td, msg_td, wait_list, stop);
        if stop.load(Ordering::Acquire) {
            break;
        }
        // `work_start` is stamped before the think-sleep so the request latency
        // covers think-time + matrix work (`schbench.c:1464-1481`).
        let work_start = monotonic_nanos();
        if config.sleep_usec > 0 {
            think_sleep(config.sleep_usec);
        }
        do_work(
            &mut matrix_buf,
            matrix_size,
            config.operations,
            locks,
            &mut work_units,
        );
        let now = monotonic_nanos();
        let delta_us = now.saturating_sub(work_start) / 1000;
        if delta_us > 0 {
            // SAFETY: only this worker thread accesses its own request_stats cell.
            unsafe { (*td.request_stats.get()).add_lat(delta_us.min(u32::MAX as u64) as u32) };
        }
        progress.fetch_add(1, Ordering::Relaxed);
        phase_loop_count += 1;
        // Drain-on-change: when the parent advances `phase_epoch`, finalize the
        // phase just ended (tagged with the OLD epoch the samples were recorded
        // under) and re-baseline. A worker blocked across a whole phase simply
        // contributes an empty histogram for it (no cycle ran), which folds
        // harmlessly.
        if let Some(pe) = phase_epoch {
            let new_epoch = pe.load(Ordering::Relaxed);
            if new_epoch != cur_epoch {
                let ss_end = read_schedstat_raw(tid);
                // SAFETY: this is the thread that owns `td`.
                unsafe { drain_phase(td, cur_epoch, phase_ss_start, ss_end, phase_loop_count) };
                cur_epoch = new_epoch;
                phase_ss_start = ss_end;
                phase_loop_count = 0;
            }
        }
    }
    // Shutdown wake: the loop exits on `stop` without a final `msg_and_wait`, so
    // the message thread may still be parked in `wait_forever` (its only waker is
    // a worker post). schbench's MAIN thread handles this -- it fposts each
    // message thread after `stopping = 1` (`schbench.c:1832`, `:1933`) -- but our
    // `run()` joins passively and `msg_td` is private to `run_one_message_thread`,
    // so the exiting worker wakes its own message thread instead. Without this, a
    // worker that exits mid-`do_work` leaves the message thread parked forever and
    // `run()` deadlocks. Idempotent across workers: the first post the message
    // thread observes makes it re-check `stop` and break; later posts are no-ops.
    msg_td.futex.post();
    // Final drain: close the still-open phase. When non-phasic this is the only
    // snapshot (`cur_epoch` 0), so run() builds the whole-run result uniformly
    // from snapshots.
    let ss_end = read_schedstat_raw(tid);
    // SAFETY: this is the thread that owns `td`.
    unsafe { drain_phase(td, cur_epoch, phase_ss_start, ss_end, phase_loop_count) };
    // Record this worker's whole-run mean run-queue wait at exit — the
    // mean-of-means component for the whole-run SchbenchResult (schbench reads
    // each thread's schedstat for the final aggregate, `schbench.c:1664-1670`).
    // Reuses the cumulative `ss_end` pair just read (one /proc read; mean and
    // final-phase delta share one consistent snapshot).
    // SAFETY: owner-only access to this thread's sched_delay_ns cell.
    unsafe { *td.sched_delay_ns.get() = mean_sched_delay(ss_end) };
}

/// Resolve the worker-thread default and the per-CPU lock-array size from the
/// calling thread's CPU affinity (the allocated cpuset, per ruling). Returns
/// `(allowed_cpu_count, lock_array_size)` where the array size is the highest
/// allowed CPU id + 1, so `sched_getcpu` indexes it without clamping even on a
/// sparse cpuset.
fn resolve_cpu_topology() -> (usize, usize) {
    // SAFETY: a zeroed cpu_set_t filled by sched_getaffinity for the calling
    // thread (pid 0); CPU_ISSET only reads it.
    unsafe {
        let mut set: libc::cpu_set_t = core::mem::zeroed();
        let rc = libc::sched_getaffinity(0, core::mem::size_of::<libc::cpu_set_t>(), &mut set);
        if rc != 0 {
            return (1, 1);
        }
        let mut count = 0usize;
        let mut max_id = 0usize;
        for cpu in 0..libc::CPU_SETSIZE as usize {
            if libc::CPU_ISSET(cpu, &set) {
                count += 1;
                max_id = cpu;
            }
        }
        (count.max(1), (max_id + 1).max(1))
    }
}

/// Combined WHOLE-RUN results of a schbench run: the merged wakeup + request
/// latency percentiles and the achieved request rate (completed work
/// cycles/second). The histograms are the union of every phase; `sched_delay_*`
/// keep schbench's mean-of-means (`collect_sched_delay`) so the side-by-side
/// validation matches real schbench's reported number.
pub(crate) struct SchbenchResult {
    pub(crate) wakeup: Percentiles,
    pub(crate) request: Percentiles,
    pub(crate) loop_count: u64,
    pub(crate) achieved_rps: f64,
    /// Mean message-thread run-queue wait (ns), averaged across message threads
    /// (schbench's `message_thread_delay`, `schbench.c:1673`).
    pub(crate) sched_delay_msg_ns: u64,
    /// Mean worker-thread run-queue wait (ns), averaged across all workers
    /// (schbench's `worker_thread_delay`, `schbench.c:1674`).
    pub(crate) sched_delay_worker_ns: u64,
}

/// The full result of [`run`]: the whole-run [`SchbenchResult`] plus the
/// per-phase aggregates keyed by `phase_epoch` (each the cross-thread merge for
/// one scenario step's HOLD window). When non-phasic (`phase_epoch == None`)
/// `phases` holds the single `(0, ..)` baseline entry the host discards; tests
/// read `whole_run`. The worker dispatch turns each `(epoch, SchbenchPhaseStats)`
/// into a `PhaseSlice`.
pub(crate) struct SchbenchOutcome {
    pub(crate) whole_run: SchbenchResult,
    pub(crate) phases: Vec<(u32, SchbenchPhaseStats)>,
}

/// One message thread's pooled results, returned from [`run_one_message_thread`]
/// to [`run`]: the whole-run histograms (Σ over its workers' phase snapshots),
/// the run-delay components for the whole-run mean-of-means, and the per-epoch
/// aggregate keyed by `phase_epoch`.
struct MessageThreadResult {
    whole_wakeup: PlatStats,
    whole_request: PlatStats,
    msg_sched_delay_ns: u64,
    workers_sched_delay_sum: u64,
    phases: std::collections::BTreeMap<u32, SchbenchPhaseStats>,
}

/// Run one message thread plus its workers, returning the whole-run histograms
/// AND the per-epoch aggregate. The message thread runs on the calling thread;
/// workers are scoped so they are joined before their owner-only phase snapshots
/// are drained.
fn run_one_message_thread(
    worker_threads: usize,
    locks: Option<&PerCpuLocks>,
    config: &SchbenchConfig,
    stop: &AtomicBool,
    progress: &AtomicU64,
    phase_epoch: Option<&AtomicU32>,
) -> MessageThreadResult {
    let workers: Vec<ThreadData> = (0..worker_threads).map(|_| ThreadData::new()).collect();
    let msg_td = ThreadData::new();
    let wait_list = TreiberStack::new();
    let ctx = WorkerCtx {
        msg_td: &msg_td,
        wait_list: &wait_list,
        locks,
        config,
        stop,
        progress,
        phase_epoch,
    };

    std::thread::scope(|inner| {
        for w in &workers {
            inner.spawn(|| worker_loop(w, &ctx));
        }
        run_msg_thread(&msg_td, &wait_list, stop, phase_epoch);
        // Stop is set: wake every worker so a blocked one observes stop and
        // exits (schbench fposts each worker before joining, `:1599-1602`).
        for w in &workers {
            w.futex.post();
        }
        // The inner scope joins the workers here.
    });

    // After join (happens-before): drain each thread's owner-only phase
    // snapshots into the whole-run histograms + the per-epoch aggregate.
    let mut whole_wakeup = PlatStats::default();
    let mut whole_request = PlatStats::default();
    let mut workers_sched_delay_sum = 0u64;
    let mut phases: std::collections::BTreeMap<u32, SchbenchPhaseStats> =
        std::collections::BTreeMap::new();
    for w in &workers {
        // SAFETY: every worker has joined (inner scope ended), so this is the
        // sole access to their cells.
        unsafe {
            workers_sched_delay_sum =
                workers_sched_delay_sum.saturating_add(*w.sched_delay_ns.get());
            for snap in (*w.phase_snapshots.get()).drain(..) {
                whole_wakeup.combine(&snap.wakeup);
                whole_request.combine(&snap.request);
                let e = phases.entry(snap.epoch).or_default();
                e.wakeup.combine(&snap.wakeup);
                e.request.combine(&snap.request);
                e.worker_run_delay_ns = e.worker_run_delay_ns.saturating_add(snap.run_delay_ns);
                e.worker_pcount = e.worker_pcount.saturating_add(snap.pcount);
                e.loop_count = e.loop_count.saturating_add(snap.loop_count);
            }
        }
    }
    // SAFETY: run_msg_thread ran on this thread inside the scope above (before
    // the join), so msg_td's cells are settled and this is sole access.
    let msg_sched_delay = unsafe { *msg_td.sched_delay_ns.get() };
    // SAFETY: same — sole access to msg_td's snapshot cell after the scope.
    unsafe {
        for snap in (*msg_td.phase_snapshots.get()).drain(..) {
            let e = phases.entry(snap.epoch).or_default();
            e.msg_run_delay_ns = e.msg_run_delay_ns.saturating_add(snap.run_delay_ns);
            e.msg_pcount = e.msg_pcount.saturating_add(snap.pcount);
        }
    }
    MessageThreadResult {
        whole_wakeup,
        whole_request,
        msg_sched_delay_ns: msg_sched_delay,
        workers_sched_delay_sum,
        phases,
    }
}

/// Run the schbench workload until `stop` is set, returning the whole-run
/// percentiles + achieved rate AND the per-phase aggregates. `progress` is the
/// live count of completed work cycles across all workers. `phase_epoch` (when
/// `Some`) is the scenario engine's shared per-phase generation; the engine
/// splits its histograms at each transition (see [`worker_loop`]).
pub(crate) fn run(
    config: &SchbenchConfig,
    stop: &AtomicBool,
    progress: &AtomicU64,
    phase_epoch: Option<&AtomicU32>,
) -> SchbenchOutcome {
    let (allowed_count, lock_array_size) = resolve_cpu_topology();
    let worker_threads = if config.worker_threads == 0 {
        allowed_count
    } else {
        config.worker_threads
    };
    let locks = if config.skip_locking {
        None
    } else {
        Some(PerCpuLocks::new(lock_array_size))
    };

    let start = monotonic_nanos();
    let mut all_wakeup = PlatStats::default();
    let mut all_request = PlatStats::default();
    let mut total_msg_sched_delay = 0u64;
    let mut total_worker_sched_delay = 0u64;
    let mut all_phases: std::collections::BTreeMap<u32, SchbenchPhaseStats> =
        std::collections::BTreeMap::new();

    std::thread::scope(|outer| {
        let handles: Vec<_> = (0..config.message_threads)
            .map(|_| {
                let locks = locks.as_ref();
                outer.spawn(move || {
                    run_one_message_thread(worker_threads, locks, config, stop, progress, phase_epoch)
                })
            })
            .collect();
        for h in handles {
            let mtr = h.join().expect("schbench message thread panicked");
            all_wakeup.combine(&mtr.whole_wakeup);
            all_request.combine(&mtr.whole_request);
            total_msg_sched_delay = total_msg_sched_delay.saturating_add(mtr.msg_sched_delay_ns);
            total_worker_sched_delay =
                total_worker_sched_delay.saturating_add(mtr.workers_sched_delay_sum);
            for (epoch, sps) in mtr.phases {
                all_phases.entry(epoch).or_default().merge(&sps);
            }
        }
    });

    let loop_count = progress.load(Ordering::Relaxed);
    let elapsed_ns = monotonic_nanos().saturating_sub(start);
    let achieved_rps = if elapsed_ns > 0 {
        loop_count as f64 / (elapsed_ns as f64 / 1e9)
    } else {
        0.0
    };
    // Average the per-thread run-queue waits, matching schbench's
    // collect_sched_delay (`schbench.c:1673-1674`): message delay over
    // message_threads, worker delay over all workers. (Whole-run mean-of-means;
    // the per-phase path uses sample-weighted Σrd/Σpc — see SchbenchPhaseStats.)
    let sched_delay_msg_ns = total_msg_sched_delay / (config.message_threads.max(1) as u64);
    let total_workers = (config.message_threads * worker_threads).max(1) as u64;
    let sched_delay_worker_ns = total_worker_sched_delay / total_workers;

    SchbenchOutcome {
        whole_run: SchbenchResult {
            wakeup: all_wakeup.percentiles(),
            request: all_request.percentiles(),
            loop_count,
            achieved_rps,
            sched_delay_msg_ns,
            sched_delay_worker_ns,
        },
        phases: all_phases.into_iter().collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Lightweight node for stack stress tests (ThreadData is ~38 KiB, too
    /// heavy to allocate by the thousand).
    struct TestNode {
        next: AtomicPtr<TestNode>,
    }
    impl Linked for TestNode {
        fn next_link(&self) -> &AtomicPtr<Self> {
            &self.next
        }
    }
    impl TestNode {
        fn new() -> Self {
            Self {
                next: AtomicPtr::new(ptr::null_mut()),
            }
        }
    }

    #[test]
    fn matrix_size_matches_schbench_formula() {
        // sqrt(256*1024/3/8) = sqrt(10922) = 104.
        assert_eq!(
            SchbenchConfig {
                cache_footprint_kib: 256,
                operations: 5,
                ..Default::default()
            }
            .matrix_size(),
            104
        );
        // Zero operations -> no matrix work.
        assert_eq!(
            SchbenchConfig {
                operations: 0,
                ..Default::default()
            }
            .matrix_size(),
            0
        );
    }

    #[test]
    fn stack_add_splice_is_lifo() {
        let a = TestNode::new();
        let b = TestNode::new();
        let stack: TreiberStack<TestNode> = TreiberStack::new();
        assert!(stack.splice().is_null(), "empty stack splices to null");
        stack.add(&a as *const _ as *mut _);
        stack.add(&b as *const _ as *mut _);
        // LIFO: b (pushed last) is the head; a follows.
        let head = stack.splice();
        assert_eq!(head.cast_const(), &b as *const TestNode);
        // SAFETY: head -> b (alive on stack); its link -> a.
        let second = unsafe { (*head).next.load(Ordering::Acquire) };
        assert_eq!(second.cast_const(), &a as *const TestNode);
        // SAFETY: second -> a; its link is null.
        assert!(unsafe { (*second).next.load(Ordering::Acquire) }.is_null());
        assert!(stack.splice().is_null(), "splice emptied the stack");
    }

    #[test]
    fn stack_concurrent_add_loses_no_nodes() {
        // N threads each concurrently push K distinct nodes; after all join, a
        // single splice must return exactly N*K nodes. A broken CAS (lost
        // update / ABA) would drop nodes and the count would be short. Push is
        // deterministic (each node pushed once), so this never hangs.
        const THREADS: usize = 8;
        const PER_THREAD: usize = 2000;
        let nodes: Vec<TestNode> = (0..THREADS * PER_THREAD).map(|_| TestNode::new()).collect();
        let stack = TreiberStack::new();

        std::thread::scope(|s| {
            for chunk in nodes.chunks(PER_THREAD) {
                let stack = &stack;
                s.spawn(move || {
                    for n in chunk {
                        stack.add(n as *const TestNode as *mut TestNode);
                    }
                });
            }
        });

        // All pushes complete (scope joined). Drain and count distinct nodes.
        let mut seen = std::collections::HashSet::new();
        let mut cur = stack.splice();
        while !cur.is_null() {
            assert!(seen.insert(cur), "node observed twice");
            // SAFETY: cur is one of the live `nodes`; next is its link.
            cur = unsafe { (*cur).next.load(Ordering::Acquire) };
        }
        assert_eq!(seen.len(), THREADS * PER_THREAD, "no node lost under contention");
    }

    #[test]
    fn engine_runs_and_produces_latency_samples() {
        // Small topology; run until ~50 work cycles complete, then stop. The
        // test completing proves the shutdown does not hang (a scope-join
        // shutdown bug would block here until the nextest timeout). Stop is
        // event-driven: spin on the shared progress counter, not a sleep.
        let config = SchbenchConfig {
            message_threads: 1,
            worker_threads: 2,
            cache_footprint_kib: 16,
            operations: 1,
            sleep_usec: 0,
            skip_locking: false,
        };
        let stop = AtomicBool::new(false);
        let progress = AtomicU64::new(0);
        let outcome = std::thread::scope(|s| {
            let runner = s.spawn(|| run(&config, &stop, &progress, None));
            while progress.load(Ordering::Relaxed) < 50 {
                core::hint::spin_loop();
            }
            stop.store(true, Ordering::Release);
            runner.join().expect("run panicked")
        });
        let result = &outcome.whole_run;
        assert!(result.loop_count >= 50, "engine did work: {}", result.loop_count);
        assert!(result.wakeup.nr_samples > 0, "wakeup samples recorded");
        assert!(result.request.nr_samples > 0, "request samples recorded");
        assert!(result.achieved_rps > 0.0, "positive achieved rps");
        // Non-phasic (phase_epoch None): the only snapshot is the baseline
        // epoch 0, which the host discards. No per-phase metrics are emitted.
        assert_eq!(outcome.phases.len(), 1, "non-phasic => single baseline phase");
        assert_eq!(outcome.phases[0].0, 0, "the lone phase is BASELINE epoch 0");
    }

    #[test]
    fn engine_terminates_when_lone_worker_stops() {
        // Regression for the shutdown deadlock: with a SINGLE worker there is no
        // second worker to post the message thread, so a worker that exits on
        // `stop` while the message thread is parked in `wait_forever` must wake it
        // (the unconditional post at worker_loop's exit), or run() hangs forever.
        // Reaching the assertion at all proves run() returned. `sleep_usec` 0
        // keeps the lone worker almost always mid-`do_work` when stop fires (the
        // deadlock-prone window). engine_runs / engine_splits use 2 workers and
        // missed this -- a second worker posting the message thread masks the bug;
        // a hang here is the nextest timeout, exactly how the bug first surfaced.
        let config = SchbenchConfig {
            message_threads: 1,
            worker_threads: 1,
            cache_footprint_kib: 256,
            operations: 5,
            sleep_usec: 0,
            skip_locking: false,
        };
        let stop = AtomicBool::new(false);
        let progress = AtomicU64::new(0);
        let outcome = std::thread::scope(|s| {
            let runner = s.spawn(|| run(&config, &stop, &progress, None));
            while progress.load(Ordering::Relaxed) < 10 {
                core::hint::spin_loop();
            }
            stop.store(true, Ordering::Release);
            // Deadlocks here on regression: the lone worker exits without waking
            // the parked message thread, so this join never returns.
            runner.join().expect("run panicked")
        });
        assert!(
            outcome.whole_run.loop_count >= 10,
            "engine did work and returned: {}",
            outcome.whole_run.loop_count
        );
    }

    #[test]
    fn engine_splits_stats_across_phase_epochs() {
        // Drive the shared phase_epoch 1 -> 2 mid-run and confirm the engine
        // partitions its histograms + loop_count into per-epoch snapshots
        // (the scx-phase vs detached-EEVDF-phase mechanism). Event-driven via
        // the progress counter, no sleeps.
        let config = SchbenchConfig {
            message_threads: 1,
            worker_threads: 2,
            cache_footprint_kib: 16,
            operations: 1,
            sleep_usec: 0,
            skip_locking: false,
        };
        let stop = AtomicBool::new(false);
        let progress = AtomicU64::new(0);
        let epoch = AtomicU32::new(1); // start in a real step epoch (phase 1)
        let outcome = std::thread::scope(|s| {
            let runner = s.spawn(|| run(&config, &stop, &progress, Some(&epoch)));
            while progress.load(Ordering::Relaxed) < 50 {
                core::hint::spin_loop();
            }
            let after_phase1 = progress.load(Ordering::Relaxed);
            epoch.store(2, Ordering::Release); // transition to phase 2
            while progress.load(Ordering::Relaxed) < after_phase1 + 50 {
                core::hint::spin_loop();
            }
            stop.store(true, Ordering::Release);
            runner.join().expect("run panicked")
        });

        let by_epoch: std::collections::BTreeMap<u32, &SchbenchPhaseStats> =
            outcome.phases.iter().map(|(e, s)| (*e, s)).collect();
        let p1 = by_epoch.get(&1).expect("phase 1 present");
        let p2 = by_epoch.get(&2).expect("phase 2 present");
        // Both phases ran work cycles (we waited for >=50 cycles in each
        // window). loop_count is the robust split proof: latency samples gate
        // on delta_us>0, which a sub-µs cycle can miss, but a cycle always
        // increments loop_count.
        assert!(p1.loop_count > 0, "phase 1 ran cycles: {}", p1.loop_count);
        assert!(p2.loop_count > 0, "phase 2 ran cycles: {}", p2.loop_count);
        // Per-phase run-delay split is populated per thread class. This guards a
        // DISTINCT path the loop_count/histogram asserts don't touch: the
        // schedstat raw-pair baseline/re-baseline (phase_ss_start) + the
        // per-class Σ fold (worker vs msg). pcount is the deterministic
        // denominator — any thread scheduled in the phase accrues pcount>=1
        // (sched_info_arrive), and both phases ran cycles (workers dispatched)
        // while the msg thread is scheduled on every wake. run_delay_ns itself
        // is NOT asserted: it can be 0 on an uncontended host, so a value check
        // would flake; pcount>0 pins that the class was measured and folded.
        assert!(p1.worker_pcount > 0, "phase 1 worker run-delay split populated");
        assert!(p2.worker_pcount > 0, "phase 2 worker run-delay split populated");
        assert!(p1.msg_pcount > 0, "phase 1 msg run-delay split populated");
        assert!(p2.msg_pcount > 0, "phase 2 msg run-delay split populated");
        // loop_count partitions across phases, summing to the global progress —
        // no cycle is lost or double-counted at the boundary.
        let loop_sum: u64 = outcome.phases.iter().map(|(_, s)| s.loop_count).sum();
        assert_eq!(
            loop_sum, outcome.whole_run.loop_count,
            "per-phase loop_count partitions the whole-run count"
        );
        // The whole-run histogram is EXACTLY the union of the per-phase ones
        // (the drained snapshots, recombined), so the per-phase split loses no
        // samples vs the whole-run accounting.
        let phase_request_sum: u64 = outcome
            .phases
            .iter()
            .map(|(_, s)| s.request.percentiles().nr_samples)
            .sum();
        assert_eq!(
            outcome.whole_run.request.nr_samples, phase_request_sum,
            "whole-run request count == Σ per-phase counts"
        );
        let phase_wakeup_sum: u64 = outcome
            .phases
            .iter()
            .map(|(_, s)| s.wakeup.percentiles().nr_samples)
            .sum();
        assert_eq!(
            outcome.whole_run.wakeup.nr_samples, phase_wakeup_sum,
            "whole-run wakeup count == Σ per-phase counts"
        );
    }

    #[test]
    fn schbench_config_serde_roundtrips() {
        // The new serialized type roundtrips unchanged.
        let cfg = SchbenchConfig::default()
            .message_threads(3)
            .worker_threads(7)
            .cache_footprint_kib(512)
            .operations(9)
            .sleep_usec(250)
            .skip_locking(true);
        let json = serde_json::to_string(&cfg).expect("SchbenchConfig must serialize");
        let back: SchbenchConfig =
            serde_json::from_str(&json).expect("SchbenchConfig must deserialize");
        assert_eq!(cfg, back, "config roundtrips unchanged");
    }

    #[test]
    fn worktype_schbench_registration_and_serde() {
        use crate::workload::WorkType;
        let wt = WorkType::schbench(
            SchbenchConfig::default().message_threads(2).worker_threads(4),
        );
        assert_eq!(wt.name(), "Schbench");
        // from_name yields the default-config variant.
        assert_eq!(
            WorkType::from_name("Schbench"),
            Some(WorkType::Schbench {
                config: SchbenchConfig::default()
            })
        );
        // The variant serde-roundtrips, carrying its config.
        let json = serde_json::to_string(&wt).expect("WorkType::Schbench must serialize");
        let back: WorkType = serde_json::from_str(&json).expect("WorkType::Schbench must deserialize");
        assert_eq!(wt, back);
    }

    #[test]
    fn schbench_config_reachable_via_prelude() {
        // Regression-pin the prelude placement: test authors construct the
        // config via `use ktstr::prelude::*`. Dropping SchbenchConfig from the
        // prelude would fail this compile. Also exercises the Eq derive.
        let cfg: crate::prelude::SchbenchConfig = crate::prelude::SchbenchConfig::default();
        assert_eq!(cfg, SchbenchConfig::default());
    }

    #[test]
    fn read_schedstat_raw_parses_own_and_handles_missing() {
        // The current thread has been scheduled, so its /proc/<tid>/schedstat
        // parses without panic into a (run_delay, pcount) pair.
        let _own = read_schedstat_raw(gettid_self());
        // A non-existent tid -> (0,0) via the file-read-failure path (thread
        // exited), matching schbench's fopen-failure handling. The parse
        // boundaries (incl. the pcount==0 mean guard) are covered below.
        assert_eq!(
            read_schedstat_raw(-1),
            (0, 0),
            "absent schedstat yields (0,0), no panic"
        );
    }

    #[test]
    fn schedstat_raw_parse_and_mean_pcount_guard() {
        // Raw parse keeps run_delay + pcount undivided (the re-poolable pair).
        assert_eq!(parse_schedstat_raw("123456 50 5"), (50, 5));
        // mean_sched_delay divides with the pcount==0 guard (no div-by-zero).
        assert_eq!(mean_sched_delay((50, 5)), 10); // run_delay/pcount
        assert_eq!(mean_sched_delay((50, 0)), 0); // pcount==0 guard
        assert_eq!(mean_sched_delay(parse_schedstat_raw("0 0 0")), 0); // !sched_info_on()
        assert_eq!(mean_sched_delay((0, 5)), 0); // 0 run_delay -> 0 mean
    }

    #[test]
    #[should_panic(expected = "schedstat field 3")]
    fn parse_schedstat_raw_short_line_panics() {
        // A present-but-short line is a kernel/parse bug: fail loud, not a
        // silent 0 (the fail-loud ruling).
        parse_schedstat_raw("100 50");
    }

    #[test]
    #[should_panic(expected = "schedstat field 2")]
    fn parse_schedstat_raw_nonnumeric_panics() {
        parse_schedstat_raw("alpha beta gamma");
    }
}
