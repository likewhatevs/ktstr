//! schbench's run engine, ported faithfully from `schbench.c`: the
//! message-thread / worker-thread topology, the lockless wait-list, the
//! handshake-driven wakeup-latency loop, and the matrix work under the
//! per-CPU lock. The default (non-RPS) mode is the wakeup-latency +
//! request-latency benchmark, with per-phase schedstat (run-delay) capture; the
//! RPS-injector mode (`-R`) and its auto-RPS closed-loop rate control (`-A`)
//! layer on the request queue + once-per-second control thread.
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
//! - do_work timing: the matrix multiply compiles to the SAME scalar inner loop
//!   as schbench's `-O2` `do_some_math` -- a `u64` multiply-accumulate
//!   (`imul` with a fused memory operand, then `add`), NOT SIMD. It does not
//!   auto-vectorize: there is no vector `u64` multiply on the build target
//!   (`x86-64-v3`; `vpmullq` is AVX-512) and the serial accumulator reduction
//!   blocks SLP -- exactly as for schbench's `-O2` build (its Makefile
//!   `CFLAGS = -Wall -O2`, no `-march`, likewise scalar; verified by disassembly
//!   of both). A debug build (opt-level 0) emits far more overhead (no register
//!   allocation, unrolling, or memory-operand fusion), so request latency runs
//!   higher than reference schbench's. ABSOLUTE side-by-side fidelity vs the
//!   reference (the `ktstr-schbench-validate` comparison) therefore needs a
//!   release build. The in-ktstr A/B comparison (perf-delta / per-phase claims)
//!   is build-invariant -- both sides use the same build -- so it is unaffected.

use core::cell::UnsafeCell;
use core::ptr;
use core::sync::atomic::{AtomicBool, AtomicPtr, AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};

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

    /// Atomically take the whole stack and return the chain in FIFO (insertion)
    /// order -- the reverse of [`splice`](Self::splice)'s LIFO. Faithful port of
    /// `request_splice` (`schbench.c:920-940`): swap to null, then reverse the
    /// spliced chain so the consumer services the oldest queued request first
    /// (request order matters, unlike the thread wait-list). Returns null if
    /// empty. Single-consumer: only the owning worker drains its own queue.
    fn splice_reversed(&self) -> *mut T {
        let mut cur = self.head.swap(ptr::null_mut(), Ordering::AcqRel);
        let mut rev: *mut T = ptr::null_mut();
        while !cur.is_null() {
            // SAFETY: `cur` is a node that was published on the stack and whose
            // storage the caller owns; the single consumer reverses the chain it
            // just took exclusive ownership of, so no other thread touches these
            // links concurrently.
            let link = unsafe { (*cur).next_link() };
            let next = link.load(Ordering::Acquire);
            link.store(rev, Ordering::Relaxed);
            rev = cur;
            cur = next;
        }
        rev
    }
}

/// A queued work request in RPS-injector mode (`schbench.c` `struct request`,
/// `:757`). The RPS thread heap-allocates one (`Box`) per request and links it
/// onto the target worker's queue; the worker drains its queue and frees each --
/// matching schbench's per-request malloc/free (part of the syscall profile the
/// standalone validation compares). schbench's `start_time` field is set
/// (`:951`) but never read, so it is omitted: the wakeup latency is measured
/// from the worker's `wake_time` (stamped by the RPS thread at enqueue, and by
/// the worker itself before it blocks -- see [`rps_wait`]) and the request
/// latency from a local work-start -- neither reads `start_time`.
/// Omitting it also drops schbench's per-request `gettimeofday` for that dead
/// field (`:951`) -- one fewer clock syscall per request, consistent with the
/// engine's `CLOCK_MONOTONIC`-vs-`gettimeofday` clock divergence.
struct Request {
    /// Intrusive link for the per-worker request [`TreiberStack`].
    next: AtomicPtr<Request>,
}

impl Request {
    fn new() -> Self {
        Self {
            next: AtomicPtr::new(ptr::null_mut()),
        }
    }
}

impl Linked for Request {
    fn next_link(&self) -> &AtomicPtr<Self> {
        &self.next
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
///
/// # schbench(8) CLI parity
///
/// This port re-expresses schbench's default (matrix-work) mode natively, so
/// its tunables are config fields and topology rather than CLI flags. The
/// mapping to schbench's option table (`schbench.c:138-187`):
///
/// | schbench flag | ktstr |
/// |---|---|
/// | `-m` message-threads | `message_threads` |
/// | `-t` threads | `worker_threads` (workers per message thread; `0` = `ceil(cpuset_cpus / message_threads)`, see below) |
/// | `-F` cache_footprint | `cache_footprint_kib` |
/// | `-n` operations | `operations` |
/// | `-s` sleep_usec | `sleep_usec` |
/// | `-L` no-locking | `skip_locking` |
/// | `-R` rps | `requests_per_sec` |
/// | `-A` auto-rps | `auto_rps` |
/// | `--split` (long-only) | `split_percent` (`None` = no split, all-private) |
/// | `-r` runtime | in-VM: the scenario engine's run window (the engine runs until `stop`); host-side: the `run_secs` argument to [`run_standalone`](crate::workload::run_standalone) |
///
/// ## Set by ktstr topology, not a flag
///
/// - `-t` default: with `worker_threads = 0`, ktstr matches schbench's `-t`
///   0-default -- it divides the CPU count across the message threads,
///   `ceil(cpus / message_threads)` per thread (`schbench.c:1849-1852`), so the
///   total worker count stays near the CPU count. ktstr scopes "cpus" to the
///   allocated guest cpuset (the worker's `sched_getaffinity` mask, set by the
///   scenario's topology / `CgroupDef`) rather than schbench's `get_nprocs`, so
///   the total is ≈ the cpuset's CPU count. An explicit non-zero `worker_threads`
///   is workers-per-message-thread in both.
/// - `-M` (message-cpus) / `-W` (worker-cpus) thread pinning: ktstr places
///   threads through its affinity / cpuset layer, so there is deliberately no
///   per-thread-pin knob.
///
/// ## Observability flags -> the metric API
///
/// schbench's `-w` (warmuptime), `-i` (intervaltime), `-z` (zerotime), `-j`
/// (json), and `-J` (jobname) shape its streaming stderr/JSON report. ktstr's
/// numbers flow through the metric API instead -- per-phase attribution and the
/// sidecar -- so these have no flag equivalent. `ktstr-schbench-validate`
/// reproduces schbench's stderr-table shape for a side-by-side comparison.
///
/// ## Split mode (`--split`)
///
/// `Some(p)` partitions `cache_footprint_kib` into a per-thread private matrix
/// (`p`%) and ONE process-global shared matrix (`100-p`%) that every worker
/// multiplies into concurrently, reproducing schbench's cross-core
/// shared-working-set cache contention (`schbench.c:1390-1404`, `:1858-1863`).
/// ktstr models the shared matrix with `AtomicU64` `Relaxed` accesses. Like
/// schbench's emitted code, the shared kernel keeps the running sum in a register
/// and STORES it to each shared C cell on every inner (`k`) iteration -- C is
/// write-only in the loop (A and B are loaded each `k`, C is never reloaded), and
/// that per-k store is what generates the contention. Both gcc and clang keep the
/// per-k store: `do_some_math` reads `m1`/`m2`/`m3` as offsets into one base
/// pointer, so neither can prove the `m3` store doesn't alias the next `k`'s
/// `m1`/`m2` loads. On x86-64 a `Relaxed` load/store lowers to a plain `MOV` (no
/// `LOCK`), so the contention is identical to schbench's plain shared-memory race
/// -- but sound (atomics, no data race), with zero unsafe. `None` (default) is
/// the legacy all-private single matrix.
///
/// ## Modes not ported
///
/// - `-p` (pipe): schbench's pipe-transfer test -- a separate workload from the
///   matrix-work default this port re-expresses (`schbench.c:177`). Not
///   available.
/// - `-C` (calibrate): a tuning aid that times schbench's own work loop and
///   forces `-L` (`schbench.c:166`, `:389`). Intentionally out of scope -- ktstr
///   measures through the metric path.
#[derive(Clone, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct SchbenchConfig {
    /// Number of message threads (`schbench.c` `-m`, default 1).
    pub message_threads: usize,
    /// Worker threads per message thread (`schbench.c` `-t`). 0 resolves to
    /// `ceil(cpuset_cpus / message_threads)` -- the CPU count of the allocated
    /// guest cpuset (the worker's `sched_getaffinity` mask, per ruling) divided
    /// across the message threads, matching schbench's 0-default
    /// (`schbench.c:1849-1852`) scoped to the cpuset rather than `get_nprocs`.
    /// See `resolve_worker_count` and the CLI-parity section above.
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
    /// Fixed request rate, requests/second (`schbench.c` `-R`, default 0 = off).
    /// 0 selects the default message-handshake mode (each worker is woken by its
    /// message thread); non-zero switches to the RPS-injector mode, where a
    /// dedicated thread enqueues `requests_per_sec` requests/second round-robin
    /// across the workers (`schbench.c` `run_rps_thread`, `:1258`).
    pub requests_per_sec: usize,
    /// Auto-RPS target CPU-busy percentage (`schbench.c` `-A`, default 0 = off).
    /// Non-zero turns on closed-loop rate control: a once-per-second control
    /// thread grows/shrinks the live request rate toward this host-busy% target
    /// (`schbench.c` `auto_scale_rps`, `:1180`). Setting it seeds the rate to 10
    /// when `requests_per_sec` is 0 (`schbench.c:286`), so auto-RPS starts low
    /// and climbs.
    pub auto_rps: usize,
    /// Percent of the cache footprint that is PRIVATE per worker thread
    /// (`schbench.c` `--split`, long-only, 0-100). `None` = no split:
    /// schbench's legacy all-private single matrix (`schbench.c:1405-1408`,
    /// `:1879-1880`). `Some(p)` partitions `cache_footprint_kib` into a
    /// per-thread private matrix (`p`%) and ONE process-global shared matrix
    /// (`100-p`%) that every worker multiplies into concurrently, reproducing
    /// schbench's cross-core shared-working-set cache contention
    /// (`schbench.c:1390-1404`, `:1858-1863`). `Some(0)` = all shared,
    /// `Some(100)` = all private (same matrix sizes as `None`, but routed
    /// through the split branch, matching schbench's `split_specified` path).
    /// Out-of-range `Some(p > 100)` panics when the engine consumes it
    /// (`schbench.c:362-365` exits on the same); the builder also debug-asserts
    /// the bound.
    pub split_percent: Option<usize>,
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
            requests_per_sec: 0,
            auto_rps: 0,
            split_percent: None,
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
    /// Set the fixed request rate in requests/second (schbench `-R`); 0 selects
    /// the default message-handshake mode, non-zero the RPS-injector mode.
    #[must_use = "builder methods consume self; bind the result"]
    pub fn requests_per_sec(mut self, rps: usize) -> Self {
        self.requests_per_sec = rps;
        self
    }
    /// Set the auto-RPS target host-busy percentage (schbench `-A`); 0 disables
    /// auto-scaling. Non-zero seeds the rate to 10 when `requests_per_sec` is 0.
    #[must_use = "builder methods consume self; bind the result"]
    pub fn auto_rps(mut self, target_pct: usize) -> Self {
        self.auto_rps = target_pct;
        self
    }
    /// Set the private/shared cache-footprint split percentage (schbench
    /// `--split`); `None` (default) = no split, all-private single matrix.
    /// `Some(p)` requires `p <= 100`: the builder debug-asserts it for early
    /// feedback, and the engine hard-panics on an out-of-range value at the
    /// consumption boundary in `run` -- the analog of schbench exiting on a bad
    /// `--split` (`schbench.c:362-365`).
    #[must_use = "builder methods consume self; bind the result"]
    pub fn split_percent(mut self, percent: Option<usize>) -> Self {
        debug_assert!(
            percent.is_none_or(|p| p <= 100),
            "split_percent must be 0..=100"
        );
        self.split_percent = percent;
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

    /// Shared-matrix dimension for `--split` (`schbench.c:1859,1862`:
    /// `sqrt(cache_footprint_kb*(100-split)/100 * 1024/3/sizeof(ulong))`).
    /// `None` split (or 0 `operations`/`cache_footprint_kib`) => 0 (no shared
    /// matrix); `Some(100)` => 0 (all private, no shared work).
    pub(crate) fn shared_matrix_size(&self) -> usize {
        match self.split_percent {
            Some(p) if self.operations > 0 && self.cache_footprint_kib > 0 => {
                let shared_kb = self.cache_footprint_kib * (100 - p) / 100;
                ((shared_kb * 1024 / 3 / core::mem::size_of::<u64>()) as f64).sqrt() as usize
            }
            _ => 0,
        }
    }

    /// Private (per-worker) matrix dimension. `Some(p)` (`schbench.c:1860,1863`):
    /// `sqrt(cache_footprint_kb*split/100 * 1024/3/sizeof(ulong))`. `None`
    /// (legacy): the full-footprint [`matrix_size`](Self::matrix_size)
    /// (`schbench.c:1880`, the all-private single matrix).
    pub(crate) fn private_matrix_size(&self) -> usize {
        match self.split_percent {
            Some(p) if self.operations > 0 && self.cache_footprint_kib > 0 => {
                let private_kb = self.cache_footprint_kib * p / 100;
                ((private_kb * 1024 / 3 / core::mem::size_of::<u64>()) as f64).sqrt() as usize
            }
            _ => self.matrix_size(),
        }
    }

    /// The effective TOTAL request rate after schbench's startup seed: the user's
    /// `requests_per_sec`, or 10 when auto-RPS is on and no fixed rate was given
    /// (`schbench.c:286`: auto-RPS starts at 10 and climbs toward its target).
    /// Both the mode gate ([`rps_per_message_thread`](Self::rps_per_message_thread))
    /// and the live auto-scaled injection rate seed from this.
    pub(crate) fn normalized_total_rps(&self) -> usize {
        if self.auto_rps != 0 && self.requests_per_sec == 0 {
            10
        } else {
            self.requests_per_sec
        }
    }

    /// Per-message-thread request rate driving the RPS-vs-default MODE GATE: the
    /// normalized total ([`normalized_total_rps`](Self::normalized_total_rps))
    /// divided across the message threads, each of which runs its own RPS thread.
    /// schbench divides once at startup (`requests_per_sec /= message_threads`,
    /// `schbench.c:1899`) BEFORE the per-thread mode gate (`if (requests_per_sec)`,
    /// `schbench.c:1594`), so the gate sees the divided value: a total below the
    /// message-thread count rounds to 0 and the thread runs the DEFAULT
    /// message-handshake mode (not a zero-rate RPS). The gate is decided ONCE from
    /// this seeded value; the actual injection rate is the live auto-scaled total
    /// (which may climb above the seed). Integer division drops the remainder,
    /// matching schbench -- so auto-RPS with more than 10 message threads (seed
    /// 10 / m = 0) degenerates to default mode, exactly as schbench does.
    pub(crate) fn rps_per_message_thread(&self) -> usize {
        self.normalized_total_rps() / self.message_threads.max(1)
    }

    /// Effective per-message-thread worker count for [`run`]. A non-zero
    /// `worker_threads` is workers-per-message-thread as-is; the `0` default
    /// mirrors schbench's `-t` 0-default (`schbench.c:1849-1852`:
    /// `worker_threads = (num_cpus + message_threads - 1) / message_threads`)
    /// but over the allocated guest cpuset (`allowed_count` from
    /// [`resolve_cpu_topology`], per the ktstr cpuset ruling) instead of
    /// `get_nprocs`: it divides the cpuset CPU count across the message threads
    /// (ceil), so the TOTAL worker count (`message_threads * this`) stays ≈ the
    /// cpuset CPU count -- matching schbench's total ≈ `num_cpus`.
    /// `message_threads` floors at 1 to avoid a zero divisor; the divide is a
    /// no-op at the default `message_threads = 1`.
    pub(crate) fn resolve_worker_count(&self, allowed_count: usize) -> usize {
        if self.worker_threads != 0 {
            return self.worker_threads;
        }
        allowed_count.div_ceil(self.message_threads.max(1))
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
    /// schbench for the side-by-side validation); the per-phase delay comes from
    /// [`Self::phase_snapshots`] instead.
    sched_delay_ns: UnsafeCell<u64>,
    /// Per-phase snapshots this thread accumulated via drain-on-change against
    /// the shared `phase_epoch` (one [`PhaseSnapshot`] per phase the thread did
    /// work in, plus a final in-flight phase at exit). Owner-thread-only,
    /// drained by the main thread after join — same happens-before as the
    /// histogram cells. Empty when run non-phasic (`phase_epoch == None`) until
    /// the single end-of-run drain.
    phase_snapshots: UnsafeCell<Vec<PhaseSnapshot>>,
    /// RPS-injector mode: this worker's pending request queue. The RPS thread
    /// pushes heap-allocated [`Request`]s (round-robin across workers); the
    /// owning worker drains via `splice_reversed` (FIFO) and frees each. Unused
    /// in the default message-handshake mode (`requests_per_sec == 0`).
    requests: TreiberStack<Request>,
    /// RPS-injector backpressure counter (`schbench.c` `thread_data->pending`,
    /// `:779`): incremented by the RPS thread per enqueue, reset to 0 by the
    /// owning worker at each splice. The RPS thread stops enqueuing to a worker
    /// whose `pending` exceeds the batch cap (`:1284`).
    pending: AtomicU64,
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
/// schbench parity. They measure different things by design — never cross-compare a
/// per-phase value against a whole-run threshold (or vice-versa). `pcount == 0`
/// for a class means that class was never scheduled in the phase → the host
/// emits the metric as ABSENT, not `0`.
#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub(crate) struct SchbenchPhaseStats {
    /// Wakeup-latency histogram merged across the phase's workers.
    pub(crate) wakeup: PlatStats,
    /// Request-latency histogram merged across the phase's workers.
    pub(crate) request: PlatStats,
    /// Per-phase achieved-RPS distribution: the control thread's per-second
    /// samples attributed to this phase's epoch (the whole-run rps is unchanged;
    /// this is purely additive). A 1s sample straddling an epoch boundary lands
    /// wholly in the sample-time epoch (bounded fuzz, <=1 sample/transition, like
    /// the msg-thread late-drain). Empty for a phase shorter than the ~1s control
    /// cadence -- the host gates on `sample_count() > 0` so a sub-second phase
    /// reads ABSENT, never rps=0. A rate; re-derived from the merged histogram.
    pub(crate) rps: PlatStats,
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
        self.rps.combine(&other.rps);
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

// SAFETY: `ThreadData` is shared across threads only via the lockless wait-list
// and (in RPS mode) the per-worker request queue, whose operations touch
// exclusively the atomic fields (`next`, `futex`, `wake_time`, `requests`,
// `pending`) -- all internally synchronized. `requests` is single-producer /
// single-consumer: the RPS thread is the only pusher to a given worker's queue
// (round-robin assignment) and the owning worker is the only drainer, with the
// push's Release CAS / drain's Acquire swap establishing happens-before;
// `pending` is plain atomic counting. The `UnsafeCell` fields (`wakeup_stats`,
// `request_stats`, `sched_delay_ns`, `phase_snapshots`) are written and read
// ONLY by the single worker thread that owns this `ThreadData`, and only the
// main thread reads/drains them after all workers have joined (a happens-before
// via the join). The per-phase drain (`worker_loop` / `run_msg_thread`) `take`s
// and pushes into the owning thread's own cells, never another's. No two threads
// ever touch a cell concurrently, so sharing `&ThreadData` across threads is
// sound.
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
            requests: TreiberStack::new(),
            pending: AtomicU64::new(0),
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

/// schbench's `--split` op partition (`schbench.c:1391-1392`): of `operations`
/// total, `p`% run on the private matrix and the rest on the shared matrix.
/// Returns `(ops_shared, ops_private)`. Extracted so the integer-division split
/// is pinned by a unit test rather than re-implemented inline in `do_work`.
fn ops_split(operations: usize, p: usize) -> (usize, usize) {
    let ops_private = (operations * p) / 100;
    let ops_shared = operations - ops_private;
    (ops_shared, ops_private)
}

/// One work cycle's matrix multiplications under the per-CPU lock. Faithful to
/// schbench's `do_work` (`schbench.c:1379-1413`): take the current CPU's mutex
/// (unless `skip_locking`), run `operations` matrix multiplies, unlock. With
/// `split_percent` set, the ops are partitioned (`schbench.c:1390-1404`):
/// `ops_shared` run on the ONE process-global shared matrix (every worker
/// contends), then `ops_private` on this worker's private buffer; `None` is the
/// legacy all-private single matrix (`:1406-1408`). The guard drops at the end,
/// holding the lock across all operations exactly as schbench does (`:1387`/
/// `:1411`).
fn do_work(
    private_buf: &mut [u64],
    private_matrix_size: usize,
    shared_buf: &[core::sync::atomic::AtomicU64],
    shared_matrix_size: usize,
    split_percent: Option<usize>,
    operations: usize,
    locks: Option<&PerCpuLocks>,
    work_units: &mut u64,
) {
    let _guard = locks.map(|l| l.lock_this_cpu());
    match split_percent {
        Some(p) => {
            let (ops_shared, ops_private) = ops_split(operations, p);
            // schbench.c:1395-1398: shared ops first, on the global matrix.
            if shared_matrix_size > 0 && ops_shared > 0 {
                for _ in 0..ops_shared {
                    crate::workload::worker::matrix_multiply_shared(
                        shared_buf,
                        shared_matrix_size,
                        work_units,
                    );
                }
            }
            // schbench.c:1401-1404: private ops second, on this worker's buffer.
            if private_matrix_size > 0 && ops_private > 0 {
                for _ in 0..ops_private {
                    crate::workload::worker::matrix_multiply(
                        private_buf,
                        private_matrix_size,
                        work_units,
                    );
                }
            }
        }
        None => {
            // schbench.c:1406-1408: legacy all-private single matrix.
            for _ in 0..operations {
                if private_matrix_size > 0 {
                    crate::workload::worker::matrix_multiply(
                        private_buf,
                        private_matrix_size,
                        work_units,
                    );
                }
            }
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

/// Worker side of one RPS-injector cycle. Faithful to schbench's `msg_and_wait`
/// RPS branch (`schbench.c:1007-1037`): stamp our own `wake_time`, reset
/// `pending`, then splice this worker's request queue (FIFO). If non-empty,
/// return the chain immediately (fast path, no wait, no wakeup sample --
/// `:1014-1017`). If empty, block on this worker's own futex until the RPS thread
/// posts it, record the wakeup (scheduler) latency, then splice whatever it
/// queued (null on a spurious wake or the shutdown wake-all). Unlike
/// [`msg_and_wait`], the worker does NOT enqueue on the message thread's
/// wait-list or post a message thread -- the RPS thread is the sole waker.
///
/// `wake_time` is stamped HERE before the splice (schbench `:1008`) AND by the
/// RPS thread at enqueue (`:1294`). On a TRUE block the RPS thread's stamp wins
/// (it overwrites ours before posting), so `now - wake_time` is the enqueue->run
/// scheduler latency. But on a FAST wake -- a leftover RUNNING futex token from a
/// request the worker already drained via splice (the word is consumed by splice,
/// not by `wait`, so it stays RUNNING) -- the RPS thread does NOT re-stamp, so our
/// own stamp makes the delta measure THIS block decision (near-zero), not the
/// stale enqueue time of the already-served request. Omitting it inflates the
/// fast-wake samples to the full inter-request gap (the wakeup-p50 bug: 543ms
/// vs schbench's 7µs).
fn rps_wait(td: &ThreadData, stop: &AtomicBool) -> *mut Request {
    // Stamp our own wake_time before splicing (schbench `:1008`); see the doc for
    // why this is what keeps fast-wake samples honest.
    td.wake_time.store(monotonic_nanos(), Ordering::Release);
    // Reset the backpressure counter before draining (schbench `td->pending = 0`,
    // `:1012`): the RPS thread stops enqueuing while pending exceeds the batch
    // cap, so clearing it re-opens this worker for the next batch.
    td.pending.store(0, Ordering::Release);
    let chain = td.requests.splice_reversed();
    if !chain.is_null() {
        return chain; // work already queued: serve it without blocking
    }
    if stop.load(Ordering::Acquire) {
        return ptr::null_mut();
    }
    // Empty queue: block until the RPS thread posts our futex, then `now -
    // wake_time` is the scheduler wakeup latency (schbench `fwait` NULL then
    // add_lat, `:1032`/`:1034-1037`).
    td.futex.wait_forever();
    // Shutdown wake: if stop is set, the wake came from run_rps_thread's stop-path
    // wake-all (`:1308-1312`), which posts every worker WITHOUT restamping
    // wake_time. A worker blocked between bursts when stop fires would otherwise
    // record `now - (its own stale block-time stamp)` -- the whole inter-burst
    // gap -- as a phantom wakeup, one garbage tail sample per worker. That is not
    // a request-driven wakeup, so skip the sample (schbench's `if (!stopping)
    // fwait` guard, `:1030`, encodes the same intent: shutdown is not a latency
    // event). Return any leftover requests for the caller to free.
    if stop.load(Ordering::Acquire) {
        return td.requests.splice_reversed();
    }
    let now = monotonic_nanos();
    let wake = td.wake_time.load(Ordering::Acquire);
    let delta_us = now.saturating_sub(wake) / 1000;
    if delta_us > 0 {
        // SAFETY: only this worker thread accesses its own wakeup_stats cell.
        unsafe { (*td.wakeup_stats.get()).add_lat(delta_us.min(u32::MAX as u64) as u32) };
    }
    // Splice whatever the RPS thread enqueued before posting us (null if the post
    // was a spurious wake with nothing queued).
    td.requests.splice_reversed()
}

/// Free a chain of [`Request`]s spliced off a worker's queue but not processed
/// (the shutdown path, and the post-join cleanup of any requests the RPS thread
/// enqueued after a worker's last splice). Each node was `Box::into_raw`'d by the
/// RPS thread; the draining/cleaning thread owns them exclusively after the
/// splice, so each is freed exactly once.
fn free_request_chain(mut req: *mut Request) {
    while !req.is_null() {
        // SAFETY: `req` is a node taken via `splice_reversed` (exclusive
        // ownership); freed exactly once.
        let next = unsafe { (*req).next.load(Ordering::Acquire) };
        drop(unsafe { Box::from_raw(req) });
        req = next;
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

/// The RPS-injector dispatcher loop, replacing [`run_msg_thread`] when
/// `requests_per_sec != 0`. Each second it enqueues `requests_per_sec` requests
/// round-robin across this group's `workers`, skipping a worker whose `pending`
/// exceeds the batch cap (backpressure, `schbench.c:1284-1290`); on stop, it
/// wakes every worker and exits. Runs on the message thread's calling thread, so
/// — like [`run_msg_thread`] — it records the dispatcher thread's per-phase +
/// whole-run run-queue wait into `msg_td` (the "message thread delay" in RPS
/// mode, `schbench.c:1664-1670`).
///
/// DIVERGENCE from schbench's `run_rps_thread` (`schbench.c:1258-1314`),
/// intentional: schbench FRONT-LOADS the second (bursts all N enqueues, then
/// `usleep`s to fill the rest of the second) and relies on its workers
/// interleaving with the burst to keep each queue under the cap. That interleave
/// is scheduling-dependent; ktstr's burst instead front-ran the workers (the
/// first splice grabbed a large batch the worker drained to completion before
/// re-splicing), so `pending` stayed pegged and the injector dropped the
/// remainder every second -- a ~BATCH/worker supply cap that under-delivered the
/// requested rate (e.g. ~257/s at a 400/s target). ktstr instead PACES the
/// enqueues evenly across the second (one every `1s/N`), which forces the
/// injector to interleave with the workers deterministically: `pending` drains
/// between enqueues and the full rate is delivered up to worker capacity. Above
/// capacity the pacing collapses to a flat-out loop and the `BATCH` backpressure
/// bounds the queue, matching schbench's bounded-backlog overload behavior. The
/// pacing sleep and the bounded backpressure usleep are workload-defined `-R`
/// cadence (like the per-request think-sleep), not synchronization waits. Uses
/// `CLOCK_MONOTONIC` (ruling), not schbench's `gettimeofday`.
///
/// Verified by `ktstr-schbench-validate` (release, standalone), not a nextest
/// unit test: the under-delivery signal only appears when worker capacity exceeds
/// the front-load cap (~2*BATCH/s), which requires a release build (debug matrix
/// work caps capacity well below that) on dedicated CPU (nextest's parallel
/// execution starves the workers, collapsing both the paced and a hypothetical
/// reverted injector to the same capacity-limited rate). The fixed `-R` axis in
/// the schbench validation comparison is the regression evidence.
fn run_rps_thread(
    workers: &[ThreadData],
    msg_td: &ThreadData,
    stop: &AtomicBool,
    phase_epoch: Option<&AtomicU32>,
    live_rate: &AtomicUsize,
) {
    let tid = gettid_self();
    let mut cur_epoch = phase_epoch.map_or(0, |e| e.load(Ordering::Relaxed));
    let mut phase_ss_start = read_schedstat_raw(tid);
    // schbench's per-worker queue-depth cap before the injector backs off
    // (`schbench.c:1267`).
    const BATCH: u64 = 128;
    const ONE_SEC_NS: u64 = 1_000_000_000;
    let worker_count = workers.len().max(1);
    let mut cur_tid: usize = 0;
    while !stop.load(Ordering::Acquire) {
        // Re-read the live per-message-thread rate each second. `live_rate` is
        // already the per-thread value (schbench's post-`/= message_threads`
        // global `requests_per_sec`, `:1899`), injected directly with no further
        // division (`schbench.c:1290`). Auto-RPS retargets it from the control
        // thread; a fixed `-R` leaves it constant.
        let requests_per_sec = live_rate.load(Ordering::Relaxed);
        let start = monotonic_nanos();
        // PACE the enqueues evenly across the second rather than front-loading
        // them. schbench bursts all N then fills the rest of the second
        // (`schbench.c:1300-1306`), relying on its workers interleaving with the
        // burst to keep each worker's queue under the `BATCH` cap. ktstr's burst
        // front-ran the workers: the first splice grabbed a large batch the worker
        // processed to completion before re-splicing, so `pending` stayed pegged
        // and the injector dropped the remainder every second (a steady-state
        // ~BATCH/worker supply cap -> only ~257/s at a 400/s target). Pacing one
        // enqueue every `1s/N` forces the injector to interleave with the workers,
        // so `pending` drains between enqueues and the full rate is delivered up
        // to worker capacity. Above capacity `due` falls behind real time, the
        // pacing sleep collapses to zero, and the `BATCH` backpressure bounds the
        // queue -- matching schbench's overload behavior (a bounded backlog).
        let interval_ns = if requests_per_sec > 0 {
            ONE_SEC_NS / requests_per_sec as u64
        } else {
            ONE_SEC_NS
        };
        let mut due = start;
        for _ in 0..requests_per_sec {
            if stop.load(Ordering::Acquire) {
                break;
            }
            // Wait until this enqueue's scheduled time. A no-op once `due` falls
            // behind real time (rate above what the workers can drain), so the
            // loop then runs flat-out and the backpressure below bounds the queue.
            // `due` is re-based to `start` each second and accrues at most
            // N*(ONE_SEC_NS/N) ~= 1s within a pass, so saturating_add never wraps
            // (the saturating form documents that the per-pass re-base is the
            // load-bearing invariant).
            due = due.saturating_add(interval_ns);
            let now = monotonic_nanos();
            if now < due {
                std::thread::sleep(std::time::Duration::from_nanos(due - now));
            }
            let worker = &workers[cur_tid % worker_count];
            cur_tid += 1;
            // Backpressure: don't queue more to a worker already `BATCH` deep
            // (`schbench.c:1284-1290`). The bounded usleep is schbench's injector
            // throttle (workload behavior), not a sync poll. The round-robin
            // cursor already advanced, so the skipped slot moves to the next
            // worker, matching schbench (the request for this slot is dropped).
            if worker.pending.load(Ordering::Acquire) > BATCH {
                std::thread::sleep(std::time::Duration::from_micros(100));
                continue;
            }
            worker.pending.fetch_add(1, Ordering::AcqRel);
            // schbench mallocs one request per enqueue; the worker frees it after
            // processing (`schbench.c:1292`/`:1476`). Box matches that profile.
            let req = Box::into_raw(Box::new(Request::new()));
            worker.requests.add(req);
            // Stamp the target's wake_time so it measures wakeup latency as
            // now - enqueue (`schbench.c:1294`), then post it.
            worker.wake_time.store(monotonic_nanos(), Ordering::Release);
            worker.futex.post();
        }
        // The pacing loop already spans ~1 second; top up only if it finished
        // early (rate 0, or `stop`), preserving schbench's per-second cadence.
        let elapsed = monotonic_nanos().saturating_sub(start);
        if !stop.load(Ordering::Acquire) && elapsed < ONE_SEC_NS {
            std::thread::sleep(std::time::Duration::from_nanos(ONE_SEC_NS - elapsed));
        }
        // Dispatcher per-phase run-delay drain (mirrors run_msg_thread): on epoch
        // change, finalize this thread's run-delay for the ended phase. Empty
        // histograms (the dispatcher records only run-delay).
        if let Some(pe) = phase_epoch {
            let new_epoch = pe.load(Ordering::Relaxed);
            if new_epoch != cur_epoch {
                let ss_end = read_schedstat_raw(tid);
                // SAFETY: this thread owns msg_td (the dispatcher's ThreadData).
                unsafe { drain_phase(msg_td, cur_epoch, phase_ss_start, ss_end, 0) };
                cur_epoch = new_epoch;
                phase_ss_start = ss_end;
            }
        }
        if stop.load(Ordering::Acquire) {
            // Wake every worker so a blocked one observes stop and exits
            // (`schbench.c:1308-1312`).
            for w in workers {
                w.futex.post();
            }
            break;
        }
    }
    // Final drain + whole-run mean run-delay for the dispatcher thread, matching
    // run_msg_thread's exit (`schbench.c:1664`).
    let ss_end = read_schedstat_raw(tid);
    // SAFETY: this thread owns msg_td.
    unsafe { drain_phase(msg_td, cur_epoch, phase_ss_start, ss_end, 0) };
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
    /// The ONE process-global shared matrix (`--split` mode), shared across
    /// every worker of every message thread; empty slice when not splitting.
    shared: &'a [core::sync::atomic::AtomicU64],
    shared_matrix_size: usize,
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
        shared,
        shared_matrix_size,
    } = *ctx;
    let tid = gettid_self();
    // Per-worker PRIVATE matrix (schbench.c:1568-1574: private_matrix_size when
    // splitting, else the full matrix_size). The shared matrix (split mode) is
    // the process-global `shared` allocated once in run().
    let private_matrix_size = config.private_matrix_size();
    let mut private_buf = if private_matrix_size > 0 {
        vec![0u64; 3 * private_matrix_size * private_matrix_size]
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
    // RPS mode iff the per-thread rate is non-zero (the `-R` total covers at
    // least one request/sec per message thread). Below that it rounds to 0 and
    // the worker runs the default message-handshake mode -- matching schbench's
    // startup-divide-before-gate ([`SchbenchConfig::rps_per_message_thread`]).
    let rps_mode = config.rps_per_message_thread() != 0;
    while !stop.load(Ordering::Acquire) {
        // Acquire work. Default mode: the message-thread handshake (records
        // wakeup latency; no request chain -> a single work cycle). RPS mode:
        // splice this worker's request queue, blocking on its own futex until
        // the RPS thread posts; returns the spliced chain (FIFO) to drain.
        let mut req: *mut Request = if rps_mode {
            let chain = rps_wait(td, stop);
            if stop.load(Ordering::Acquire) {
                // Free any spliced-but-unprocessed requests, then exit.
                free_request_chain(chain);
                break;
            }
            if chain.is_null() {
                continue; // spurious wake with an empty queue
            }
            chain
        } else {
            msg_and_wait(td, msg_td, wait_list, stop);
            if stop.load(Ordering::Acquire) {
                break;
            }
            ptr::null_mut()
        };
        // Process the work: one cycle in default mode (`req` null), or one cycle
        // per queued request in RPS mode (freeing each). `work_start` is stamped
        // before the think-sleep so the request latency covers think-time +
        // matrix work (`schbench.c:1464-1481`).
        loop {
            let work_start = monotonic_nanos();
            if config.sleep_usec > 0 {
                think_sleep(config.sleep_usec);
            }
            do_work(
                &mut private_buf,
                private_matrix_size,
                shared,
                shared_matrix_size,
                config.split_percent,
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
            if req.is_null() {
                break; // default mode: a single work cycle
            }
            // RPS mode: advance to the next queued request and free this one.
            // SAFETY: `req` was `Box::into_raw`'d by the RPS thread and handed to
            // this worker exclusively via the queue splice; we own it and free it
            // exactly once here.
            let next = unsafe { (*req).next.load(Ordering::Acquire) };
            drop(unsafe { Box::from_raw(req) });
            req = next;
            if req.is_null() {
                break;
            }
        }
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
    // In RPS mode this is itself a harmless no-op: the dispatcher runs
    // `run_rps_thread` (which never parks on `msg_td`) and observes `stop` itself.
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
    /// Per-second achieved-RPS distribution (schbench's `rps_stats`), sampled
    /// once per second by the control thread. For auto-RPS, samples are gated
    /// until the target is hit (`schbench.c:1781`); for fixed/default mode every
    /// second is sampled.
    pub(crate) rps: Percentiles,
    pub(crate) loop_count: u64,
    pub(crate) achieved_rps: f64,
    /// Auto-RPS final TOTAL target rate at run exit: the live per-message-thread
    /// rate * message_threads (schbench's `requests_per_sec * message_threads`,
    /// `schbench.c:1995`). For fixed `-R` and default mode this equals the seeded
    /// total; only auto-RPS makes it diverge as the control loop retargets.
    pub(crate) final_rps_goal: usize,
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
    live_rate: &AtomicUsize,
    shared: &[core::sync::atomic::AtomicU64],
    shared_matrix_size: usize,
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
        shared,
        shared_matrix_size,
    };

    std::thread::scope(|inner| {
        for w in &workers {
            inner.spawn(|| worker_loop(w, &ctx));
        }
        // Dispatcher: RPS-injector mode runs the rate-driven request thread;
        // default mode runs the message-thread handshake. Both run on this
        // calling thread and record the dispatcher's run-delay into `msg_td`
        // (schbench `message_thread`, `:1594`).
        if config.rps_per_message_thread() != 0 {
            run_rps_thread(&workers, &msg_td, stop, phase_epoch, live_rate);
        } else {
            run_msg_thread(&msg_td, &wait_list, stop, phase_epoch);
        }
        // Stop is set: wake every worker so a blocked one observes stop and
        // exits (schbench fposts each worker before joining, `:1599-1602`).
        for w in &workers {
            w.futex.post();
        }
        // The inner scope joins the workers here.
    });

    // RPS mode: free any requests the RPS thread enqueued after a worker's last
    // splice (the worker exited on stop without draining them). After the join,
    // this is the sole access to each worker's queue, so no request leaks across
    // run() calls.
    if config.rps_per_message_thread() != 0 {
        for w in &workers {
            free_request_chain(w.requests.splice_reversed());
        }
    }

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

/// Persistent /proc/stat reader state for auto-RPS, carrying the previous
/// cumulative totals so each call computes a delta over the interval (schbench
/// keeps one fd + the prior total/idle, `schbench.c` `read_busy`, `:1046`).
#[derive(Default)]
struct ReadBusyState {
    fd: Option<File>,
    prev_total: u64,
    prev_idle: u64,
}

/// Host-busy percentage over the interval since the last call, from the
/// aggregate `cpu` line of /proc/stat (schbench `read_busy`,
/// `schbench.c:1046-1112`). `None` means "no usable reading this tick": the first
/// call (baseline seed, schbench's `first_run`) OR a zero jiffy-delta (two reads
/// in the same tick -- schbench has no guard and divides by zero into a NaN that
/// spuriously trips its target-hit `else` branch; we skip the tick instead). A
/// real reading is `Some(0.0..=100.0)`, including a genuine fully-idle `Some(0.0)`
/// -- distinguished from the seed via the `prev_total` state, NOT via the value,
/// so an idle interval still drives a grow (the bug a `busy == 0.0` test had). In
/// the guest VM /proc/stat covers exactly the allocated cpuset (the guest has
/// only `cores=N`), so this system-wide read is correctly scoped to the
/// workload's CPUs. f32 to match schbench's `float` arithmetic.
fn read_busy(s: &mut ReadBusyState) -> Option<f32> {
    if s.fd.is_none() {
        s.fd = File::open("/proc/stat").ok();
    }
    let f = s.fd.as_mut()?;
    f.seek(SeekFrom::Start(0)).ok()?;
    let mut buf = [0u8; 512];
    let n = f.read(&mut buf).ok()?;
    let text = core::str::from_utf8(&buf[..n]).ok()?;
    // First line only; first token must be "cpu" (the aggregate line). Sum every
    // numeric field -> total; field index 3 (after "cpu") is idle -- kernel
    // show_stat order is user/nice/system/IDLE/iowait/irq/softirq/steal/guest/
    // guest_nice. `split_whitespace` collapses the "cpu  " double space, matching
    // schbench's `strtok_r` (`schbench.c:1082-1096`).
    let line = text.lines().next().unwrap_or("");
    let mut fields = line.split_whitespace();
    if fields.next() != Some("cpu") {
        return None;
    }
    let mut total = 0u64;
    let mut idle = 0u64;
    for (i, tok) in fields.enumerate() {
        let v: u64 = tok.parse().unwrap_or(0);
        if i == 3 {
            idle = v;
        }
        total += v;
    }
    // First call: seed the baseline, no busy% yet (`schbench.c:1098-1101`).
    if s.prev_total == 0 {
        s.prev_total = total;
        s.prev_idle = idle;
        return None;
    }
    let dt = total.saturating_sub(s.prev_total);
    let di = idle.saturating_sub(s.prev_idle);
    s.prev_total = total;
    s.prev_idle = idle;
    if dt == 0 {
        // Zero jiffy-delta: skip rather than NaN (see the doc; schbench diverges).
        return None;
    }
    Some(100.0 - (di as f32 / dt as f32) * 100.0)
}

/// Grow or shrink the live per-message-thread request rate toward `target_pct`
/// host-busy, given an already-read `busy` percentage (schbench `auto_scale_rps`'s
/// scaling body, `schbench.c:1203-1250`). Pure: no I/O, so the scaling decision is
/// unit-tested with injected `busy` values. Stores the new per-thread rate into
/// `live_rate` -- schbench scales its global `requests_per_sec` (already per-thread
/// after `:1899`) and writes it back at `:1250`. Returns true on the transition
/// INTO the near-target band (when `target_hit` flips false->true), so the caller
/// resets the rps histogram (schbench memsets `rps_stats` on that transition). The
/// near-target test reads the delta AFTER its damping reassignment, exactly as
/// schbench does (`:1210`,`:1233`). f32 math + `ceil` (grow) / `floor` (shrink)
/// match schbench's float deltas + rounding.
fn scale_rps_for_busy(
    busy: f32,
    live_rate: &AtomicUsize,
    target_hit: &AtomicBool,
    target_pct: usize,
) -> bool {
    let target = target_pct as f32;
    let rps = live_rate.load(Ordering::Relaxed) as f32;
    let already_hit = target_hit.load(Ordering::Relaxed);
    let mut just_hit = false;
    let new_rate: usize = if busy < target {
        // Under target: grow (`schbench.c:1203-1224`).
        let mut delta = target / busy;
        if delta > 3.0 {
            delta = 3.0;
        } else if delta < 1.2 {
            delta = 1.0 + (delta - 1.0) / 8.0;
            // Threshold check AFTER the damping reassignment (`schbench.c:1209-1213`).
            if delta < 1.05 && !already_hit {
                just_hit = true;
            }
        } else if delta < 1.5 {
            delta = 1.0 + (delta - 1.0) / 4.0;
        }
        let t = (rps * delta).ceil();
        // Not enough capacity to reach the target -> hold (`schbench.c:1219-1224`).
        if t >= (1u64 << 31) as f32 {
            rps as usize
        } else {
            t as usize
        }
    } else if busy > target {
        // Over target: shrink (`schbench.c:1226-1242`).
        let mut delta = target / busy;
        if delta < 0.3 {
            delta = 0.3;
        } else if delta > 0.9 {
            delta += (1.0 - delta) / 8.0;
            // Threshold check AFTER the damping reassignment (`schbench.c:1232-1236`).
            if delta > 0.95 && !already_hit {
                just_hit = true;
            }
        } else if delta > 0.8 {
            delta += (1.0 - delta) / 4.0;
        }
        (rps * delta).floor().max(0.0) as usize
    } else {
        // Exactly on target (`schbench.c:1243-1248`).
        if !already_hit {
            just_hit = true;
        }
        rps as usize
    };
    live_rate.store(new_rate, Ordering::Relaxed);
    if just_hit {
        target_hit.store(true, Ordering::Relaxed);
    }
    just_hit
}

/// Read host-busy from /proc/stat and scale the live rate toward the target
/// (schbench `auto_scale_rps`, `schbench.c:1180-1251`). The first read only seeds
/// the baseline (schbench's `first_run` early return, `:1201`); a zero-delta tick
/// is skipped (see [`read_busy`]). Returns [`scale_rps_for_busy`]'s just-hit flag.
fn auto_scale_rps(
    busy_state: &mut ReadBusyState,
    live_rate: &AtomicUsize,
    target_hit: &AtomicBool,
    target_pct: usize,
) -> bool {
    let Some(busy) = read_busy(busy_state) else {
        return false;
    };
    scale_rps_for_busy(busy, live_rate, target_hit, target_pct)
}

/// Reset BOTH per-phase rps accumulators in LOCKSTEP at an auto-RPS target-hit.
/// schbench memsets `rps_stats` on the target-hit transition to discard the
/// pre-target ramp; the per-epoch map MUST reset with it, or the per-phase view
/// keeps ramp samples the whole-run view dropped and `Σ per-epoch == whole-run`
/// breaks (asymmetric-counter-bookkeeping).
fn reset_rps_accumulators(
    whole: &mut PlatStats,
    per_epoch: &mut std::collections::BTreeMap<u32, PlatStats>,
) {
    *whole = PlatStats::default();
    per_epoch.clear();
}

/// The once-per-second control thread, schbench's main control loop
/// (`schbench.c:1756-1827`): sample the achieved per-second RPS into `rps_stats`
/// and -- when auto-RPS is on -- adjust the live rate toward the target. Returns
/// BOTH the whole-run `rps_stats` AND a per-epoch RPS map (the same samples
/// bucketed by the phase epoch observed at sample time), moved out on join. The
/// whole-run `rps_stats` is byte-identical to the pre-per-phase behavior -- the
/// per-epoch map is PURELY ADDITIVE (each sample lands in the whole-run histogram
/// AND exactly one epoch bucket), so `Σ per-epoch == whole-run`. `phase_epoch` is
/// read Relaxed (telemetry, gates no coupled memory); a 1s sample straddling an
/// epoch change is attributed wholly to the sample-time epoch (bounded fuzz,
/// mirroring the worker drain-on-change). The 1-second cadence is the workload's
/// defined sampling/scaling interval, not a synchronization wait; it observes
/// `stop` after each interval.
fn control_loop(
    progress: &AtomicU64,
    stop: &AtomicBool,
    config: &SchbenchConfig,
    live_rate: &AtomicUsize,
    target_hit: &AtomicBool,
    phase_epoch: Option<&AtomicU32>,
) -> (PlatStats, std::collections::BTreeMap<u32, PlatStats>) {
    let mut rps_stats = PlatStats::default();
    let mut per_epoch_rps: std::collections::BTreeMap<u32, PlatStats> =
        std::collections::BTreeMap::new();
    let mut last_loop = 0u64;
    let mut last_t = monotonic_nanos();
    let mut busy_state = ReadBusyState::default();
    while !stop.load(Ordering::Acquire) {
        std::thread::sleep(std::time::Duration::from_secs(1));
        // Sample the just-elapsed interval BEFORE re-checking stop at the loop
        // top: it is a complete second of data even if stop fired during the
        // sleep (the common integer-window case), so an early break here would
        // silently drop a full valid sample. schbench likewise samples the
        // boundary interval in which its run expires (`schbench.c:1756,1781`).
        // The interval that races stop is understated (workers halt partway) --
        // the same property schbench's final boundary sample has.
        let now = monotonic_nanos();
        let lc = progress.load(Ordering::Relaxed);
        let dt = now.saturating_sub(last_t);
        // The `dt > 0` guard drops a zero-Δt tick rather than recording
        // schbench's `isfinite(rps) ? rps : 0` substitution (`schbench.c:1782`);
        // combined with the sleep-FIRST loop shape, it also omits schbench's
        // first ~µs-window sample (schbench reads `now` immediately at loop start,
        // `schbench.c:1757`, before any work). Both are unreachable on the
        // monotonic clock (dt is ~1e9 ns/tick, never 0) -- a deliberate, cleaner
        // divergence like read_busy's dt==0 NaN-skip. A short fixed-`-R` rps
        // side-by-side can still differ by schbench's one garbage first sample.
        if dt > 0 {
            // Achieved per-second rate = Δcompleted-cycles / Δt (schbench
            // `combine_message_thread_rps` over the shared loop count, `:1777`).
            let rps = lc.saturating_sub(last_loop) as f64 * 1e9 / dt as f64;
            // Gate: for auto-RPS, sample only once the target is hit
            // (`schbench.c:1781`); fixed/default mode samples every second.
            if config.auto_rps == 0 || target_hit.load(Ordering::Relaxed) {
                let sample = (rps as u64).min(u32::MAX as u64) as u32;
                rps_stats.add_lat(sample);
                // Per-phase parity: also bucket the sample by the epoch observed
                // NOW (the sample-time epoch; a straddling 1s sample lands wholly
                // here, bounded fuzz). 0/u32::MAX sentinel epochs route into the
                // same all_phases entries the host discards (the run() merge).
                let epoch = phase_epoch.map_or(0, |e| e.load(Ordering::Relaxed));
                per_epoch_rps.entry(epoch).or_default().add_lat(sample);
            }
        }
        last_loop = lc;
        last_t = now;
        if config.auto_rps != 0 {
            let just_hit = auto_scale_rps(&mut busy_state, live_rate, target_hit, config.auto_rps);
            if just_hit {
                // Discard ALL pre-target ramp samples (schbench memsets rps_stats
                // on the target-hit transition -- grow/shrink/on-target bands all
                // funnel through just_hit, `schbench.c:1212`/`:1235`/`:1247`). The
                // per-epoch map resets in LOCKSTEP so the per-phase view never
                // includes ramp samples the whole-run view discards (the
                // Σ-per-epoch == whole-run invariant survives the reset).
                reset_rps_accumulators(&mut rps_stats, &mut per_epoch_rps);
            }
        }
    }
    (rps_stats, per_epoch_rps)
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
    let worker_threads = config.resolve_worker_count(allowed_count);
    let locks = if config.skip_locking {
        None
    } else {
        Some(PerCpuLocks::new(lock_array_size))
    };
    // schbench rejects a `--split` outside 0..=100 at parse and exits
    // (schbench.c:362-365). ktstr's analog is a loud panic here at the
    // consumption boundary -- before any matrix sizing or the shared Arc
    // allocation below -- so an out-of-range config (a `pub split_percent` set
    // via struct literal bypasses the builder's debug_assert) can never
    // underflow `100 - p` (a release `usize` wrap that would size a garbage
    // shared matrix and OOM the allocation) or silently mis-split.
    if let Some(p) = config.split_percent {
        assert!(
            p <= 100,
            "schbench split_percent must be in 0..=100, got {p} (schbench.c:362-365 rejects out of range)"
        );
    }
    // schbench.c:1871-1872: ONE process-global shared matrix for `--split`,
    // allocated once and shared (interior-mutable AtomicU64) by every worker
    // across all message threads. Empty when not splitting (`None`) or when the
    // shared portion rounds to 0 (split=100): the `shared_matrix_size > 0`
    // guard in do_work then skips the shared branch, so the empty slice is
    // never indexed.
    let shared_matrix_size = config.shared_matrix_size();
    let shared: std::sync::Arc<[core::sync::atomic::AtomicU64]> = if shared_matrix_size > 0 {
        (0..3 * shared_matrix_size * shared_matrix_size)
            .map(|_| core::sync::atomic::AtomicU64::new(0))
            .collect()
    } else {
        std::sync::Arc::from([] as [core::sync::atomic::AtomicU64; 0])
    };

    let start = monotonic_nanos();
    let mut all_wakeup = PlatStats::default();
    let mut all_request = PlatStats::default();
    let mut total_msg_sched_delay = 0u64;
    let mut total_worker_sched_delay = 0u64;
    let mut all_phases: std::collections::BTreeMap<u32, SchbenchPhaseStats> =
        std::collections::BTreeMap::new();

    // The live PER-MESSAGE-THREAD request rate the RPS injectors read each second
    // -- schbench's global `requests_per_sec` AFTER its startup `/= message_threads`
    // (`schbench.c:1899`), which is exactly the value `auto_scale_rps` scales
    // (`schbench.c:1250`) and each `run_rps_thread` injects directly with no
    // further division (`schbench.c:1290`). Seeded from `rps_per_message_thread()`
    // (the -R total / m, or the auto-RPS seed 10 / m). The control thread
    // auto-scales it when auto-RPS is on; otherwise it stays constant -- a fixed
    // -R then injects that seed rate unchanged. `target_hit` gates the rps_stats sampling
    // for auto-RPS (`schbench.c:1781`); `rps_stats` is moved out of the control
    // thread on join. `live_rate` is read back after the scope as the auto-RPS
    // "final rps goal" (`schbench.c:1995`, per-thread * m).
    //
    // Ordering: both atomics are lone scalars that gate NO coupled memory --
    // `live_rate` is a plain count the RPS thread loops on (the requests it
    // enqueues publish via the Treiber stack's own AcqRel), and `target_hit` is
    // touched only by the control thread. So Relaxed is correct (a 1-second-stale
    // read is benign within the auto-RPS cadence; schbench reads its global with
    // no sync at all). The final read after the scope is synchronized by the join.
    let live_rate = AtomicUsize::new(config.rps_per_message_thread());
    let target_hit = AtomicBool::new(false);
    let mut rps_stats = PlatStats::default();

    std::thread::scope(|outer| {
        let handles: Vec<_> = (0..config.message_threads)
            .map(|_| {
                let locks = locks.as_ref();
                let live_rate = &live_rate;
                let shared = &shared;
                outer.spawn(move || {
                    run_one_message_thread(
                        worker_threads,
                        locks,
                        config,
                        stop,
                        progress,
                        phase_epoch,
                        live_rate,
                        &shared[..],
                        shared_matrix_size,
                    )
                })
            })
            .collect();
        // Control thread: schbench's main control loop -- once/sec, sample the
        // achieved RPS into rps_stats and (for auto-RPS) auto-scale `live_rate`.
        let control = outer
            .spawn(|| control_loop(progress, stop, config, &live_rate, &target_hit, phase_epoch));
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
        let (whole_rps, control_per_epoch_rps) =
            control.join().expect("schbench control thread panicked");
        rps_stats = whole_rps;
        // Merge the control thread's per-epoch RPS into the per-phase aggregate
        // EXACTLY ONCE, post-join, OUTSIDE the per-message-thread loop above (else
        // it would double-count by message_threads). The control thread is the
        // SOLE rps source; the workers contribute empty rps via mtr.phases.
        for (epoch, hist) in control_per_epoch_rps {
            all_phases.entry(epoch).or_default().rps.combine(&hist);
        }
    });

    // The auto-RPS "final rps goal" (`schbench.c:1995`): the live per-thread rate
    // at exit * message_threads. Read after the scope (all threads joined, so the
    // control loop's last store is visible).
    let final_rps_goal = live_rate.load(Ordering::Relaxed) * config.message_threads;
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
            rps: rps_stats.percentiles(),
            loop_count,
            achieved_rps,
            final_rps_goal,
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
    fn resolve_worker_count_divides_cpuset_across_message_threads() {
        // Explicit non-zero worker_threads is honored as-is (per message thread).
        let c = SchbenchConfig::default().worker_threads(3).message_threads(4);
        assert_eq!(c.resolve_worker_count(8), 3);

        // 0-default mirrors schbench's ceil(cpuset_cpus / message_threads)
        // (schbench.c:1849-1852), scoped to the cpuset count:
        //   m=1: 8/1 = 8 per thread, total 8 (== the cpuset count; no divide).
        let c = SchbenchConfig::default().message_threads(1);
        assert_eq!(c.resolve_worker_count(8), 8);
        //   m=2: 8/2 = 4 per thread, total 8.
        let c = SchbenchConfig::default().message_threads(2);
        assert_eq!(c.resolve_worker_count(8), 4);
        //   m=3: ceil(8/3) = 3 per thread, total 9 (≈ 8; schbench rounds up too).
        let c = SchbenchConfig::default().message_threads(3);
        assert_eq!(c.resolve_worker_count(8), 3);
        //   message_threads floored at 1 — no div-by-zero.
        let c = SchbenchConfig::default().message_threads(0);
        assert_eq!(c.resolve_worker_count(8), 8);
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
    fn split_matrix_sizes_match_schbench_formula() {
        // schbench.c:1858-1863: with --split=p the footprint splits into a
        // private matrix (p%) and a shared matrix (100-p%); each dimension is
        // sqrt(kib*frac/100 * 1024/3/sizeof(ulong)). None => the legacy single
        // all-private matrix (full footprint) and no shared matrix.
        let cfg = |split| SchbenchConfig {
            cache_footprint_kib: 256,
            operations: 5,
            split_percent: split,
            ..Default::default()
        };
        // None: private == the full-footprint matrix_size (104), no shared.
        assert_eq!(cfg(None).private_matrix_size(), 104);
        assert_eq!(cfg(None).shared_matrix_size(), 0);
        // 0%: nothing private; the whole footprint is the shared matrix.
        assert_eq!(cfg(Some(0)).private_matrix_size(), 0);
        assert_eq!(cfg(Some(0)).shared_matrix_size(), 104);
        // 25%: 64 KiB private (sqrt(64*1024/3/8)=52), 192 KiB shared (=90).
        assert_eq!(cfg(Some(25)).private_matrix_size(), 52);
        assert_eq!(cfg(Some(25)).shared_matrix_size(), 90);
        // 100%: all private (full footprint), no shared work.
        assert_eq!(cfg(Some(100)).private_matrix_size(), 104);
        assert_eq!(cfg(Some(100)).shared_matrix_size(), 0);
        // Zero operations -> no matrix work either way.
        let none_ops = SchbenchConfig {
            operations: 0,
            split_percent: Some(50),
            ..Default::default()
        };
        assert_eq!(none_ops.private_matrix_size(), 0);
        assert_eq!(none_ops.shared_matrix_size(), 0);
    }

    #[test]
    fn ops_split_matches_schbench_integer_division() {
        // schbench.c:1391-1392: ops_private = operations*split/100 (integer
        // truncation), ops_shared = operations - ops_private. Returns
        // (ops_shared, ops_private).
        assert_eq!(ops_split(5, 0), (5, 0)); // all shared
        assert_eq!(ops_split(5, 25), (4, 1)); // 5*25/100 = 1 private
        assert_eq!(ops_split(5, 50), (3, 2)); // 5*50/100 = 2 private
        assert_eq!(ops_split(5, 100), (0, 5)); // all private
    }

    #[test]
    fn engine_split_runs_shared_matrix_concurrently() {
        // --split with two workers: both concurrently multiply into the ONE
        // process-global shared AtomicU64 matrix (per-k stores to the shared C
        // cells -- the deliberate cache contention schbench models). Reaching the
        // assertion proves the shared-matrix path
        // neither deadlocks nor trips a data-race trap -- the atomics make the
        // shared race sound (vs schbench's plain shared-memory race). operations=8
        // @ split=50 => 4 shared + 4 private ops/cycle, so both paths run. Stop is
        // event-driven (spin on the progress counter, not a sleep).
        let config = SchbenchConfig {
            message_threads: 1,
            worker_threads: 2,
            cache_footprint_kib: 16,
            operations: 8,
            sleep_usec: 0,
            skip_locking: false,
            requests_per_sec: 0,
            auto_rps: 0,
            split_percent: Some(50),
        };
        // Sanity: both split matrices are non-empty at this footprint, so the
        // test exercises shared + private work, not a degenerate 0-dim path.
        assert!(config.shared_matrix_size() > 0, "shared matrix present");
        assert!(config.private_matrix_size() > 0, "private matrix present");
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
        assert!(
            outcome.whole_run.loop_count >= 50,
            "engine did split work: {}",
            outcome.whole_run.loop_count
        );
    }

    #[test]
    #[should_panic(expected = "split_percent must be in 0..=100")]
    fn engine_panics_on_out_of_range_split() {
        // Regression pin for the out-of-range finding: a `pub split_percent` set
        // past 100 via STRUCT LITERAL (the path that bypasses the builder's
        // debug_assert) must fail LOUDLY at the run() consumption boundary in
        // every build profile -- the analog of schbench exiting on a bad
        // `--split` -- never silently underflow `100 - p` into a garbage-huge
        // shared matrix. The assert fires before any thread spawn or allocation,
        // so this is cheap (`stop` is pre-set just in case).
        let config = SchbenchConfig {
            message_threads: 1,
            worker_threads: 1,
            cache_footprint_kib: 16,
            operations: 1,
            sleep_usec: 0,
            skip_locking: false,
            requests_per_sec: 0,
            auto_rps: 0,
            split_percent: Some(101),
        };
        let stop = AtomicBool::new(true);
        let progress = AtomicU64::new(0);
        let _ = run(&config, &stop, &progress, None);
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
        assert_eq!(
            seen.len(),
            THREADS * PER_THREAD,
            "no node lost under contention"
        );
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
            requests_per_sec: 0,
            auto_rps: 0,
            split_percent: None,
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
        assert!(
            result.loop_count >= 50,
            "engine did work: {}",
            result.loop_count
        );
        assert!(result.wakeup.nr_samples > 0, "wakeup samples recorded");
        assert!(result.request.nr_samples > 0, "request samples recorded");
        assert!(result.achieved_rps > 0.0, "positive achieved rps");
        // Non-phasic (phase_epoch None): the only snapshot is the baseline
        // epoch 0, which the host discards. No per-phase metrics are emitted.
        assert_eq!(
            outcome.phases.len(),
            1,
            "non-phasic => single baseline phase"
        );
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
            requests_per_sec: 0,
            auto_rps: 0,
            split_percent: None,
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
    fn engine_rps_mode_injects_drains_and_terminates() {
        // RPS-injector mode (requests_per_sec != 0): a dedicated thread enqueues
        // requests round-robin across the workers, which splice + drain their
        // queues instead of the message-thread handshake. Verify the RPS path
        // runs (loop_count + request samples) and terminates cleanly -- run()
        // returns (no deadlock: the RPS thread is the waker, the worker the
        // waitee, and shutdown wakes the workers via run_one_message_thread's
        // worker-wake + the RPS thread's stop-time fpost-all; a hang is the
        // nextest timeout) with no request leak (the post-join free).
        let config = SchbenchConfig {
            message_threads: 1,
            worker_threads: 2,
            cache_footprint_kib: 16,
            operations: 1,
            sleep_usec: 0,
            skip_locking: false,
            requests_per_sec: 10_000,
            auto_rps: 0,
            split_percent: None,
        };
        let stop = AtomicBool::new(false);
        let progress = AtomicU64::new(0);
        let outcome = std::thread::scope(|s| {
            let runner = s.spawn(|| run(&config, &stop, &progress, None));
            while progress.load(Ordering::Relaxed) < 50 {
                core::hint::spin_loop();
            }
            stop.store(true, Ordering::Release);
            // Deadlocks here on regression: a worker parked in rps_wait that the
            // shutdown path fails to wake, or the RPS thread failing to observe
            // stop, would hang this join.
            runner.join().expect("run panicked")
        });
        let result = &outcome.whole_run;
        assert!(
            result.loop_count >= 50,
            "RPS workers serviced requests: {}",
            result.loop_count
        );
        assert!(
            result.request.nr_samples > 0,
            "RPS request-latency samples recorded"
        );
    }

    #[test]
    fn engine_rps_below_message_threads_falls_to_default() {
        // Regression: when the -R total is below the message-thread count,
        // the per-thread rate (rps_per_message_thread) rounds to 0, and the
        // engine must run the DEFAULT message-handshake mode -- workers do real
        // wake-cycle work -- matching schbench's startup-divide-before-gate. NOT
        // a zero-rate RPS mode (which would leave the workers parked with no
        // requests, never advancing progress -- the spin below would then hang,
        // caught by the nextest timeout).
        let config = SchbenchConfig {
            message_threads: 2,
            worker_threads: 1,
            cache_footprint_kib: 16,
            operations: 1,
            sleep_usec: 0,
            skip_locking: false,
            requests_per_sec: 1, // 1 / 2 message threads = 0 per thread -> default
            auto_rps: 0,
            split_percent: None,
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
        assert!(
            outcome.whole_run.loop_count >= 50,
            "sub-1-per-thread RPS ran default-mode work: {}",
            outcome.whole_run.loop_count
        );
    }

    #[test]
    fn auto_scale_grows_when_idle_and_shrinks_when_busy() {
        // Pure scaling math (no /proc/stat I/O) at injected busy levels, matching
        // schbench's auto_scale_rps body (`schbench.c:1203-1250`).

        // Far under target: delta = 80/10 = 8, capped to 3 -> ceil(100*3) = 300.
        // Not a near-target transition, so just_hit is false (the >3 cap branch
        // never trips target_hit).
        let rate = AtomicUsize::new(100);
        let hit = AtomicBool::new(false);
        assert!(!scale_rps_for_busy(10.0, &rate, &hit, 80));
        assert_eq!(rate.load(Ordering::Acquire), 300);
        assert!(!hit.load(Ordering::Acquire));

        // Far over target: delta = 20/100 = 0.2, clamped to 0.3 -> floor(300*0.3)
        // = 90. Also not a near-target transition.
        let rate = AtomicUsize::new(300);
        let hit = AtomicBool::new(false);
        assert!(!scale_rps_for_busy(100.0, &rate, &hit, 20));
        assert_eq!(rate.load(Ordering::Acquire), 90);
        assert!(!hit.load(Ordering::Acquire));

        // Exactly on target: rate held, target_hit set on the first arrival
        // (`schbench.c:1243-1248`).
        let rate = AtomicUsize::new(123);
        let hit = AtomicBool::new(false);
        assert!(scale_rps_for_busy(50.0, &rate, &hit, 50));
        assert_eq!(rate.load(Ordering::Acquire), 123);
        assert!(hit.load(Ordering::Acquire));
    }

    #[test]
    fn auto_scale_target_hit_uses_post_damping_delta() {
        // The near-target test reads the delta AFTER its damping reassignment
        // (`schbench.c:1209-1213` grow, `:1232-1236` shrink) -- the bug the
        // pre-reassignment check had. These cases trip target_hit ONLY under the
        // correct post-damping order.

        // Grow band: busy 90, target 100 -> delta 1.111 (NOT < 1.05), damped to
        // 1 + 0.111/8 = 1.0139 (< 1.05) -> target_hit. A pre-damping check on
        // 1.111 would miss it.
        let rate = AtomicUsize::new(1000);
        let hit = AtomicBool::new(false);
        assert!(scale_rps_for_busy(90.0, &rate, &hit, 100));
        assert!(hit.load(Ordering::Acquire));

        // Shrink band: busy 100, target 95 -> delta 0.95 (NOT > 0.95), damped to
        // 0.95 + 0.05/8 = 0.95625 (> 0.95) -> target_hit. A pre-damping check on
        // 0.95 would miss it.
        let rate = AtomicUsize::new(1000);
        let hit = AtomicBool::new(false);
        assert!(scale_rps_for_busy(100.0, &rate, &hit, 95));
        assert!(hit.load(Ordering::Acquire));

        // Already hit: no second transition is reported (just_hit guards on
        // !already_hit), even on an exact-target tick.
        let rate = AtomicUsize::new(1000);
        let hit = AtomicBool::new(true);
        assert!(!scale_rps_for_busy(50.0, &rate, &hit, 50));
    }

    #[test]
    fn reset_rps_accumulators_clears_both_in_lockstep() {
        // The auto-RPS target-hit reset MUST clear the whole-run AND per-epoch rps
        // accumulators together; if a future edit drops one, Σ-per-epoch != whole-run
        // (asymmetric-counter-bookkeeping). Pins the lockstep deterministically — the
        // engine just_hit path is timing-dependent (the controller must reach target).
        let mut whole = PlatStats::default();
        whole.add_lat(500);
        let mut per_epoch: std::collections::BTreeMap<u32, PlatStats> =
            std::collections::BTreeMap::new();
        per_epoch.entry(1).or_default().add_lat(500);
        reset_rps_accumulators(&mut whole, &mut per_epoch);
        assert_eq!(whole.sample_count(), 0, "whole-run rps reset");
        assert!(per_epoch.is_empty(), "per-epoch rps reset in lockstep");
    }

    #[test]
    fn engine_auto_rps_mode_runs_and_terminates() {
        // auto_rps != 0 turns on the once-per-second control thread that
        // auto-scales the live rate toward the busy target. Seeded with an
        // explicit -R so the injector services requests immediately (progress
        // hits 50 well before the first 1s control tick), this proves the
        // auto-RPS path boots, injects, and run() terminates cleanly (the control
        // thread observes stop; a hang is the nextest timeout). The scaling MATH
        // is pinned by the scale_rps_for_busy unit tests; host-busy + target-hit
        // timing is environment-dependent, so there is no sample-count assertion.
        let config = SchbenchConfig {
            message_threads: 1,
            worker_threads: 2,
            cache_footprint_kib: 16,
            operations: 1,
            sleep_usec: 0,
            skip_locking: false,
            requests_per_sec: 10_000,
            auto_rps: 50,
            split_percent: None,
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
        assert!(
            outcome.whole_run.loop_count >= 50,
            "auto-RPS injector serviced requests: {}",
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
            requests_per_sec: 0,
            auto_rps: 0,
            split_percent: None,
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
        assert!(
            p1.worker_pcount > 0,
            "phase 1 worker run-delay split populated"
        );
        assert!(
            p2.worker_pcount > 0,
            "phase 2 worker run-delay split populated"
        );
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
        // Per-phase RPS: the additive invariant (deterministic) + a timing-tolerant
        // bound on the brief first phase. These phases run far under the 1s control
        // cadence, so phase 1 normally gets ZERO rps ticks (the ~1s tick lands in
        // phase 2, epoch already advanced); on a slow/contended host phase 1 may catch
        // a single straddle tick, so we tolerate <= 1 rather than a wall-clock-fragile
        // == 0. The deterministic empty-rps -> ABSENT contract is pinned by
        // derive_rps_distribution_values_and_absent_guard; this bound still catches
        // gross mis-attribution of the bulk of rps to the brief first phase.
        assert!(
            p1.rps.sample_count() <= 1,
            "brief phase 1 gets at most one straddle rps tick, got {}",
            p1.rps.sample_count()
        );
        let phase_rps_sum: u64 = outcome
            .phases
            .iter()
            .map(|(_, s)| s.rps.sample_count())
            .sum();
        assert_eq!(
            outcome.whole_run.rps.nr_samples, phase_rps_sum,
            "whole-run rps count == Σ per-phase rps counts"
        );
    }

    #[test]
    fn engine_per_phase_rps_populates_and_sums_to_whole_run() {
        // POPULATED per-phase RPS: the control thread samples once/sec, so a phase
        // window >= ~2s gets multiple per-phase rps samples. Time-gated (the 1s
        // control cadence is the granularity floor) -- the progress-gated
        // engine_splits phases are too short to tick. Drive epoch 1 for ~2.2s then
        // epoch 2 for ~2.2s and confirm BOTH epochs carry rps samples AND the
        // per-epoch samples sum to the whole-run count (the additive invariant:
        // every sample lands in one epoch bucket AND the whole-run histogram).
        // The ~4.4s wall-clock is the irreducible cost of exercising the real 1s control
        // cadence end-to-end (2 ticks/epoch for robustness against a delayed tick on a
        // loaded host); the deterministic per-epoch invariants — the lockstep reset and
        // Σ-per-epoch==whole-run — are pinned sleep-free by
        // reset_rps_accumulators_clears_both_in_lockstep + engine_splits_stats_across_phase_epochs.
        // This test uniquely confirms the LIVE control thread POPULATES both epochs.
        let config = SchbenchConfig {
            message_threads: 1,
            worker_threads: 2,
            cache_footprint_kib: 16,
            operations: 1,
            sleep_usec: 0,
            skip_locking: false,
            requests_per_sec: 0,
            auto_rps: 0,
            split_percent: None,
        };
        let stop = AtomicBool::new(false);
        let progress = AtomicU64::new(0);
        let epoch = AtomicU32::new(1);
        let outcome = std::thread::scope(|s| {
            let runner = s.spawn(|| run(&config, &stop, &progress, Some(&epoch)));
            // ~2.2s per epoch => >= 2 control ticks each (first tick at ~1s).
            std::thread::sleep(std::time::Duration::from_millis(2200));
            epoch.store(2, Ordering::Release);
            std::thread::sleep(std::time::Duration::from_millis(2200));
            stop.store(true, Ordering::Release);
            runner.join().expect("run panicked")
        });
        let by_epoch: std::collections::BTreeMap<u32, &SchbenchPhaseStats> =
            outcome.phases.iter().map(|(e, s)| (*e, s)).collect();
        let p1 = by_epoch.get(&1).expect("phase 1 present");
        let p2 = by_epoch.get(&2).expect("phase 2 present");
        assert!(
            p1.rps.sample_count() > 0,
            "phase 1 per-phase rps populated: {}",
            p1.rps.sample_count()
        );
        assert!(
            p2.rps.sample_count() > 0,
            "phase 2 per-phase rps populated: {}",
            p2.rps.sample_count()
        );
        // Additive invariant (no auto-RPS reset in default mode, so EXACT): Σ
        // per-epoch rps samples == whole-run rps samples.
        let phase_rps_sum: u64 = outcome
            .phases
            .iter()
            .map(|(_, s)| s.rps.sample_count())
            .sum();
        assert_eq!(
            phase_rps_sum, outcome.whole_run.rps.nr_samples,
            "Σ per-epoch rps samples == whole-run rps samples"
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
            .skip_locking(true)
            // Exercise the new field's Some(p) serde surface, not just the None
            // default (the API-review roundtrip requirement).
            .split_percent(Some(33));
        let json = serde_json::to_string(&cfg).expect("SchbenchConfig must serialize");
        let back: SchbenchConfig =
            serde_json::from_str(&json).expect("SchbenchConfig must deserialize");
        assert_eq!(cfg, back, "config roundtrips unchanged");
    }

    #[test]
    fn worktype_schbench_registration_and_serde() {
        use crate::workload::WorkType;
        let wt = WorkType::schbench(
            SchbenchConfig::default()
                .message_threads(2)
                .worker_threads(4),
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
        let back: WorkType =
            serde_json::from_str(&json).expect("WorkType::Schbench must deserialize");
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
