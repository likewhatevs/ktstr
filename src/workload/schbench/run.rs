//! schbench's run engine, ported faithfully from `schbench.c`: the
//! message-thread / worker-thread topology, the lockless wait-list, the
//! handshake-driven wakeup-latency loop, and the matrix work under the
//! per-CPU lock. This is the default (non-RPS) mode -- the wakeup-latency +
//! request-latency benchmark. The RPS injector, request queue, auto-rps, and
//! schedstat capture are a later phase.
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
use core::sync::atomic::{AtomicBool, AtomicPtr, AtomicU64, Ordering};

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
}

impl Linked for ThreadData {
    fn next_link(&self) -> &AtomicPtr<Self> {
        &self.next
    }
}

// SAFETY: `ThreadData` is shared across threads only via the lockless
// wait-list, whose operations touch exclusively the atomic fields (`next`,
// `futex`, `wake_time`) -- all internally synchronized. The `UnsafeCell`
// fields (`wakeup_stats`, `request_stats`) are written and read ONLY by the
// single worker thread that owns this `ThreadData`, and only the main thread
// reads them after all workers have joined (a happens-before via the join). No
// two threads ever touch a cell concurrently, so sharing `&ThreadData` across
// threads is sound.
unsafe impl Sync for ThreadData {}

impl ThreadData {
    fn new() -> Self {
        Self {
            next: AtomicPtr::new(ptr::null_mut()),
            futex: super::handshake::Handshake::new(),
            wake_time: AtomicU64::new(0),
            wakeup_stats: UnsafeCell::new(PlatStats::default()),
            request_stats: UnsafeCell::new(PlatStats::default()),
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
fn run_msg_thread(msg_td: &ThreadData, wait_list: &TreiberStack<ThreadData>, stop: &AtomicBool) {
    loop {
        // msg_td.futex is BLOCKED here (consumed by the prior wait, or fresh).
        wake_all(wait_list);
        if stop.load(Ordering::Acquire) {
            wake_all(wait_list);
            break;
        }
        msg_td.futex.wait_forever();
    }
}

/// One worker thread's loop. Faithful to schbench's `worker_thread`
/// (`schbench.c:1419`, default branch): block until woken (recording wakeup
/// latency), then think-sleep + matrix work under the per-CPU lock (recording
/// request latency), until stop. `progress` counts completed work cycles across
/// all workers (the live cycle counter / achieved request rate).
fn worker_loop(
    td: &ThreadData,
    msg_td: &ThreadData,
    wait_list: &TreiberStack<ThreadData>,
    locks: Option<&PerCpuLocks>,
    config: &SchbenchConfig,
    stop: &AtomicBool,
    progress: &AtomicU64,
) {
    let matrix_size = config.matrix_size();
    let mut matrix_buf = if matrix_size > 0 {
        vec![0u64; 3 * matrix_size * matrix_size]
    } else {
        Vec::new()
    };
    let mut work_units = 0u64;
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
    }
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

/// Combined results of a schbench run: the merged wakeup + request latency
/// percentiles and the achieved request rate (completed work cycles/second).
pub(crate) struct SchbenchResult {
    pub(crate) wakeup: Percentiles,
    pub(crate) request: Percentiles,
    pub(crate) loop_count: u64,
    pub(crate) achieved_rps: f64,
}

/// Run one message thread plus its workers, returning the combined per-worker
/// wakeup + request histograms. The message thread runs on the calling thread;
/// workers are scoped so they are joined before the histograms are read.
fn run_one_message_thread(
    worker_threads: usize,
    locks: Option<&PerCpuLocks>,
    config: &SchbenchConfig,
    stop: &AtomicBool,
    progress: &AtomicU64,
) -> (PlatStats, PlatStats) {
    let workers: Vec<ThreadData> = (0..worker_threads).map(|_| ThreadData::new()).collect();
    let msg_td = ThreadData::new();
    let wait_list = TreiberStack::new();

    std::thread::scope(|inner| {
        for w in &workers {
            inner.spawn(|| worker_loop(w, &msg_td, &wait_list, locks, config, stop, progress));
        }
        run_msg_thread(&msg_td, &wait_list, stop);
        // Stop is set: wake every worker so a blocked one observes stop and
        // exits (schbench fposts each worker before joining, `:1599-1602`).
        for w in &workers {
            w.futex.post();
        }
        // The inner scope joins the workers here.
    });

    let mut wakeup = PlatStats::default();
    let mut request = PlatStats::default();
    for w in &workers {
        // SAFETY: every worker has joined (inner scope ended), so this is the
        // sole access to their histogram cells.
        unsafe {
            wakeup.combine(&*w.wakeup_stats.get());
            request.combine(&*w.request_stats.get());
        }
    }
    (wakeup, request)
}

/// Run the schbench workload until `stop` is set, returning the combined wakeup
/// and request latency percentiles and the achieved request rate. `progress`
/// is the live count of completed work cycles across all workers.
pub(crate) fn run(config: &SchbenchConfig, stop: &AtomicBool, progress: &AtomicU64) -> SchbenchResult {
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

    std::thread::scope(|outer| {
        let handles: Vec<_> = (0..config.message_threads)
            .map(|_| {
                let locks = locks.as_ref();
                outer.spawn(move || {
                    run_one_message_thread(worker_threads, locks, config, stop, progress)
                })
            })
            .collect();
        for h in handles {
            let (w, r) = h.join().expect("schbench message thread panicked");
            all_wakeup.combine(&w);
            all_request.combine(&r);
        }
    });

    let loop_count = progress.load(Ordering::Relaxed);
    let elapsed_ns = monotonic_nanos().saturating_sub(start);
    let achieved_rps = if elapsed_ns > 0 {
        loop_count as f64 / (elapsed_ns as f64 / 1e9)
    } else {
        0.0
    };

    SchbenchResult {
        wakeup: all_wakeup.percentiles(),
        request: all_request.percentiles(),
        loop_count,
        achieved_rps,
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
        let result = std::thread::scope(|s| {
            let runner = s.spawn(|| run(&config, &stop, &progress));
            while progress.load(Ordering::Relaxed) < 50 {
                core::hint::spin_loop();
            }
            stop.store(true, Ordering::Release);
            runner.join().expect("run panicked")
        });
        assert!(result.loop_count >= 50, "engine did work: {}", result.loop_count);
        assert!(result.wakeup.nr_samples > 0, "wakeup samples recorded");
        assert!(result.request.nr_samples > 0, "request samples recorded");
        assert!(result.achieved_rps > 0.0, "positive achieved rps");
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
}
