//! The taobench engine: a bounded, sharded, evicting key-value cache served by a
//! client population (closed-loop by default, or open-loop fixed-rate arrival),
//! with a fast in-cache path (hit) and a slow backing-store-miss path (a
//! worker-defined sleep on a dispatcher thread). It is
//! entirely in-process — no sockets, no TLS, no subprocess — and re-expresses the
//! taobench ACCESS PATTERN in ktstr primitives so its qps / hit-ratio flow
//! through the metric API. See `validation.md` for the per-aspect real-vs-port
//! fidelity comparison.
//!
//! ## Model
//!
//! - A `Cache` of `SHARDS` independently-locked shards, each a FIFO-evicting
//!   map bounded to a per-shard object cap. Total capacity is `cache_capacity_mib`
//!   worth of objects; the key range is sized `capacity / target_hit` so a
//!   uniform-random key stream hits the resident set with probability
//!   ≈ `target_hit` at equilibrium. Eviction is the load-bearing mechanism: with
//!   no eviction a self-healing cache drifts to a 1.0 hit ratio; a bounded cache
//!   whose key range exceeds its capacity holds a steady-state miss stream.
//! - `client_threads` CLIENT threads pick a key, look it up, and on a HIT touch
//!   the stored value bytes (the cache read-bandwidth cost) and count a fast op;
//!   on a MISS hand the key to a slow dispatcher and block until it is filled,
//!   then count a slow op.
//! - Arrival: `arrival_rate == 0` (default) is CLOSED loop — each client issues
//!   its next request as soon as the prior completes. `arrival_rate > 0` is OPEN
//!   loop — each client has a fixed intended-arrival SCHEDULE
//!   (`arrival_rate / client_threads` per client) independent of completion, and
//!   serve latency is measured from that intended time. A client still holds at
//!   most one outstanding request (it blocks on a miss before issuing the next),
//!   so a slow completion delays the next issue; that backlog is folded into the
//!   late requests' serve latency (coordinated-omission correction) instead of
//!   being omitted. The serve-latency histogram is per-phase data (empty in
//!   closed loop).
//! - `slow_threads` DISPATCHER threads serve misses: sleep `slow_path_sleep_us`
//!   (the simulated backing-store fetch — a worker-defined cost, the same model
//!   schbench's think-sleep uses, not a synchronization wait), insert a freshly
//!   sized+touched value, and wake the waiting client.
//!
//! ## Counters (request-time vs response-time)
//!
//! `get_cmds` / `get_misses` are counted at LOOKUP (request) time; `fast_ops` /
//! `slow_ops` at COMPLETION (response) time. In a closed loop they are equal once
//! every in-flight request drains, but a request that straddles a phase boundary
//! lands its lookup in one phase and its completion in the next, so a per-phase
//! command-time hit_rate (`1 - get_misses/get_cmds`) differs slightly from the
//! response-time hit_ratio (`fast_ops/(fast_ops+slow_ops)`) — the same skew the
//! real reports between its interval hit_rate and its final hit_ratio.

use crate::workload::schbench::plat::PlatStats;
use core::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::collections::HashMap;
use std::collections::{BTreeMap, VecDeque};
use std::sync::{Condvar, Mutex};

/// Number of independently-locked cache shards (power of two). Sized to keep
/// per-shard lock contention low for typical guest thread counts while bounding
/// the lock-array footprint; the cache capacity is split evenly across shards.
const SHARDS: usize = 256;

/// Representative value-size distribution: a small-object-heavy, long-tailed
/// histogram (bytes → relative weight) approximating the empirical object-size
/// profile of the real workload (mean ≈ 332 B, tail to 64 KiB). This is ktstr's
/// own approximation, not a copy of any external size table.
const VALUE_SIZES: [usize; 8] = [64, 128, 256, 512, 1024, 4096, 16384, 65536];
const VALUE_WEIGHTS: [u32; 8] = [450, 300, 130, 70, 30, 12, 3, 1];

/// Cap on a single open-loop pacing sleep (ns). A client waiting for its next
/// scheduled arrival re-checks `stop` at least this often, so a shutdown is
/// observed within this bound rather than after a full inter-arrival interval —
/// which at a degenerate low `arrival_rate` (`interval_ns = 1e9 * n_clients /
/// arrival_rate`) can exceed the scenario cleanup budget and risk a watchdog kill.
/// At realistic open-loop rates the per-arrival interval is well under this, so the
/// pacing sleep is a single uncapped wait — the cap only engages at pathological
/// low rates.
const STOP_POLL_QUANTUM_NS: u64 = 50_000_000;

/// Read `CLOCK_MONOTONIC` as nanoseconds (monotonic, not wall-clock), matching
/// the schbench engine's clock source.
fn monotonic_nanos() -> u64 {
    // SAFETY: `clock_gettime` writes a `timespec` through the out-pointer and
    // reads nothing else; CLOCK_MONOTONIC is always available on Linux.
    let mut ts: libc::timespec = unsafe { core::mem::zeroed() };
    let rc = unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut ts) };
    assert_eq!(rc, 0, "clock_gettime(CLOCK_MONOTONIC) failed");
    (ts.tv_sec as u64) * 1_000_000_000 + ts.tv_nsec as u64
}

/// The simulated backing-store fetch on the slow path. Worker-defined cost (like
/// schbench's think-sleep), not a synchronization wait: `std::thread::sleep` maps
/// to `clock_nanosleep` on Linux. `0` is a no-op (the dispatcher still does the
/// fill + wakeup, so the slow path remains a distinct thread hop).
fn backing_store_fetch(usec: u64) {
    if usec > 0 {
        std::thread::sleep(std::time::Duration::from_micros(usec));
    }
}

/// Touch every byte of a served/filled value so the read cannot be elided — the
/// cache memory-bandwidth cost that makes this a cache workload rather than a
/// control-flow micro-benchmark (schbench's `black_box`-guarded memset is the
/// precedent). Returns the (black-boxed) checksum.
fn touch(bytes: &[u8]) -> u64 {
    let mut acc = 0u64;
    for &b in bytes {
        acc = acc.wrapping_add(b as u64);
    }
    std::hint::black_box(acc)
}

/// A small, fast, per-thread xorshift64 PRNG (workload key/size sampling only —
/// not cryptographic). Seeded distinctly per thread so threads do not march in
/// lockstep.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        // Avoid the xorshift fixed point at 0.
        Rng(seed ^ 0x9E37_79B9_7F4A_7C15)
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    /// Uniform in `[0, n)` (`n > 0`); a modulo reduction — the tiny bias is
    /// irrelevant for a workload key stream.
    fn below(&mut self, n: u64) -> u64 {
        self.next_u64() % n
    }
}

/// Sample a value size from [`VALUE_SIZES`] weighted by [`VALUE_WEIGHTS`].
fn sample_value_size(rng: &mut Rng) -> usize {
    let total: u32 = VALUE_WEIGHTS.iter().sum();
    let mut pick = (rng.below(total as u64)) as u32;
    for (i, &w) in VALUE_WEIGHTS.iter().enumerate() {
        if pick < w {
            return VALUE_SIZES[i];
        }
        pick -= w;
    }
    VALUE_SIZES[VALUE_SIZES.len() - 1]
}

/// Build a freshly-allocated, byte-filled value of `size` (the fill touches every
/// byte, the write-side bandwidth cost).
fn make_value(size: usize) -> Box<[u8]> {
    vec![0xABu8; size].into_boxed_slice()
}

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// User-facing config for the [`Taobench`](crate::workload::WorkType::Taobench)
/// workload — a bounded, evicting key-value cache with a fast hit path and a slow
/// miss path, driven to a steady-state hit ratio.
///
/// All fields are integer/scalar so the type keeps `Eq + Hash` (fractional knobs
/// are expressed as integer percents). Every field has a chainable builder
/// setter; [`Default`] is a useful working config.
#[derive(Clone, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct TaobenchConfig {
    /// CLIENT threads that issue lookups and serve hits. `0` resolves
    /// to the allocated guest cpuset CPU count (one client per CPU).
    pub client_threads: usize,
    /// SLOW dispatcher threads that serve misses (sleep + fill + wake). `0`
    /// resolves to `max(1, client_threads / 3)`, the real's fast:slow staffing
    /// ratio.
    pub slow_threads: usize,
    /// Resident cache budget in MiB. The cache FIFO-evicts to stay near this many
    /// bytes' worth of objects; larger than the guest LLC makes the value touch a
    /// real memory-bandwidth cost.
    pub cache_capacity_mib: usize,
    /// Target steady-state hit ratio, in percent (`1..=99`). The key range is
    /// sized `capacity / target_hit` so a uniform key stream hits at this rate at
    /// equilibrium. Clamped into range at consumption.
    pub target_hit_pct: usize,
    /// Simulated backing-store fetch latency on a miss, in microseconds (the slow
    /// dispatcher sleeps this long before filling). `0` keeps the slow path as a
    /// pure thread hop with no sleep.
    pub slow_path_sleep_us: u64,
    /// Open-loop arrival rate, AGGREGATE ops/sec across all client threads (the
    /// taobench analog of schbench's `-R`). `0` (default) = CLOSED loop: each
    /// client blocks until its request completes before issuing the next (the
    /// legacy behavior, byte-identical). Non-zero = OPEN loop: each client has a
    /// fixed intended-arrival SCHEDULE (`arrival_rate / client_threads` per client)
    /// independent of completion, and serve latency is measured from that intended
    /// time. A client still holds at most one outstanding request, so a slow
    /// completion delays the next issue; that backlog is folded into the late
    /// requests' serve latency (coordinated-omission correction) rather than
    /// omitted. Divided evenly across resolved clients. (Named for the standard
    /// queueing-theory term; the schbench analog is `requests_per_sec`, which is a
    /// verbatim mirror of schbench's own `-R` CLI flag — a constraint this
    /// ktstr-added field, with no reference `-R` to mirror, does not share.)
    pub arrival_rate: usize,
}

impl Default for TaobenchConfig {
    fn default() -> Self {
        Self {
            client_threads: 0,
            slow_threads: 0,
            cache_capacity_mib: 64,
            target_hit_pct: 90,
            slow_path_sleep_us: 100,
            arrival_rate: 0,
        }
    }
}

impl TaobenchConfig {
    /// Set the client thread count (`0` = one per allocated CPU).
    #[must_use = "builder methods consume self; bind the result"]
    pub fn client_threads(mut self, n: usize) -> Self {
        self.client_threads = n;
        self
    }
    /// Set the slow dispatcher thread count (`0` = `max(1, client_threads/3)`).
    #[must_use = "builder methods consume self; bind the result"]
    pub fn slow_threads(mut self, n: usize) -> Self {
        self.slow_threads = n;
        self
    }
    /// Set the resident cache budget in MiB.
    #[must_use = "builder methods consume self; bind the result"]
    pub fn cache_capacity_mib(mut self, mib: usize) -> Self {
        self.cache_capacity_mib = mib;
        self
    }
    /// Set the target steady-state hit ratio in percent (`1..=99`).
    #[must_use = "builder methods consume self; bind the result"]
    pub fn target_hit_pct(mut self, pct: usize) -> Self {
        self.target_hit_pct = pct;
        self
    }
    /// Set the simulated backing-store fetch latency on a miss, microseconds.
    #[must_use = "builder methods consume self; bind the result"]
    pub fn slow_path_sleep_us(mut self, us: u64) -> Self {
        self.slow_path_sleep_us = us;
        self
    }
    /// Set the open-loop AGGREGATE arrival rate in ops/sec across all clients
    /// (`0` = closed loop). Divided evenly across resolved client threads; serve
    /// latency is then measured from the intended arrival (coordinated-omission).
    #[must_use = "builder methods consume self; bind the result"]
    pub fn arrival_rate(mut self, ops_per_sec: usize) -> Self {
        self.arrival_rate = ops_per_sec;
        self
    }

    /// Resolve the client thread count: the configured value, or the allocated
    /// CPU count when `0`.
    fn resolve_client_threads(&self, allowed_cpus: usize) -> usize {
        if self.client_threads == 0 {
            allowed_cpus.max(1)
        } else {
            self.client_threads
        }
    }

    /// Resolve the slow dispatcher count: the configured value, or
    /// `max(1, clients/3)` when `0`.
    fn resolve_slow_threads(&self, clients: usize) -> usize {
        if self.slow_threads == 0 {
            (clients / 3).max(1)
        } else {
            self.slow_threads
        }
    }

    /// Clamp the target hit ratio to a usable open interval (a 0% or ≥100% target
    /// has no finite key range / no miss stream).
    fn target_hit_fraction(&self) -> f64 {
        (self.target_hit_pct.clamp(1, 99) as f64) / 100.0
    }
}

// ---------------------------------------------------------------------------
// Cache
// ---------------------------------------------------------------------------

/// One cache shard: a FIFO-evicting map bounded to `cap` objects. FIFO eviction
/// over a uniform key stream yields the same equilibrium hit ratio as LRU
/// (resident_fraction = cap / key_range) while being O(1) and lock-cheap.
struct Shard {
    map: HashMap<u64, Box<[u8]>>,
    fifo: VecDeque<u64>,
    cap: usize,
}

impl Shard {
    fn with_cap(cap: usize) -> Self {
        Shard {
            map: HashMap::new(),
            fifo: VecDeque::new(),
            cap,
        }
    }
    /// Look up `k`; on a hit, touch its bytes and return `true`.
    fn get_touch(&self, k: u64) -> bool {
        match self.map.get(&k) {
            Some(v) => {
                touch(v);
                true
            }
            None => false,
        }
    }
    /// Insert a freshly-built value for `k` (only called on a miss, so `k` is
    /// absent), FIFO-evicting down to `cap`.
    fn insert(&mut self, k: u64, v: Box<[u8]>) {
        if self.map.insert(k, v).is_none() {
            self.fifo.push_back(k);
            while self.map.len() > self.cap {
                match self.fifo.pop_front() {
                    Some(old) => {
                        self.map.remove(&old);
                    }
                    None => break,
                }
            }
        }
    }
}

/// A bounded, sharded, FIFO-evicting cache. Keys map to shards by the low bits of
/// the key (`SHARDS` is a power of two).
struct Cache {
    shards: Vec<Mutex<Shard>>,
}

impl Cache {
    /// Build a cache holding ≈ `total_objects` across `SHARDS` shards.
    fn new(total_objects: usize) -> Self {
        let per_shard = (total_objects / SHARDS).max(1);
        let shards = (0..SHARDS)
            .map(|_| Mutex::new(Shard::with_cap(per_shard)))
            .collect();
        Cache { shards }
    }
    fn shard(&self, k: u64) -> &Mutex<Shard> {
        &self.shards[(k as usize) & (SHARDS - 1)]
    }
    /// Look up + touch `k`; `true` on a hit.
    fn get_touch(&self, k: u64) -> bool {
        self.shard(k)
            .lock()
            .expect("cache shard poisoned")
            .get_touch(k)
    }
    /// Fill `k` with a freshly sized + built value (the value bytes are touched
    /// by `make_value`); FIFO-evicts to keep the shard bounded.
    fn fill(&self, k: u64, size: usize) {
        let v = make_value(size);
        self.shard(k)
            .lock()
            .expect("cache shard poisoned")
            .insert(k, v);
    }
}

// ---------------------------------------------------------------------------
// Slow-path handoff
// ---------------------------------------------------------------------------

/// A miss handed from a client to a slow dispatcher. The client measures serve
/// latency from its own local intended-arrival timestamp after `wait_filled`, so
/// the request itself carries no timing — only the key + the client to wake.
struct SlowReq {
    key: u64,
    client: usize,
}

/// The slow-request queue: dispatchers block here when idle; clients push misses
/// and wake one dispatcher.
struct SlowQueue {
    q: Mutex<VecDeque<SlowReq>>,
    cv: Condvar,
}

impl SlowQueue {
    fn new() -> Self {
        SlowQueue {
            q: Mutex::new(VecDeque::new()),
            cv: Condvar::new(),
        }
    }
    fn push(&self, req: SlowReq) {
        self.q.lock().expect("slow queue poisoned").push_back(req);
        self.cv.notify_one();
    }
}

/// A per-client response slot: the dispatcher sets `done` + notifies after the
/// fill; the client blocks here until its outstanding miss is served. At most one
/// outstanding request per client in BOTH arrival modes (a client blocks on
/// `wait_filled` for a miss before issuing its next request), so the slot is
/// reused. This single-outstanding-request serialization is exactly why open-loop
/// serve latency must be measured from the intended arrival (coordinated-omission):
/// a slow completion delays the next issue, and that backlog is folded into the
/// late requests' latency instead of omitted.
struct Slot {
    done: Mutex<bool>,
    cv: Condvar,
}

impl Slot {
    fn new() -> Self {
        Slot {
            done: Mutex::new(false),
            cv: Condvar::new(),
        }
    }
    /// Block until the dispatcher marks this slot done, then reset it.
    fn wait_filled(&self) {
        let mut done = self.done.lock().expect("slot poisoned");
        while !*done {
            done = self.cv.wait(done).expect("slot poisoned");
        }
        *done = false;
    }
    /// Mark this slot filled and wake the waiting client.
    fn signal(&self) {
        *self.done.lock().expect("slot poisoned") = true;
        self.cv.notify_one();
    }
}

// ---------------------------------------------------------------------------
// Stats
// ---------------------------------------------------------------------------

/// Taobench engine counters for one accounting window — a single phase
/// epoch (the per-phase `crate::workload::PhaseSlice::taobench` carrier) or a
/// whole worker run (the [`crate::workload::WorkerReport::taobench_whole`] /
/// [`crate::assert::CgroupStats::taobench_whole`] aggregate). Integer-only so the
/// enclosing `PhaseSlice` keeps `Eq`. `get_cmds` / `get_misses` are request-time;
/// `fast_ops` / `slow_ops` are response-time (see the module docs). `Self::merge`
/// pools two windows (Σ ops, MAX wall) and [`Self::total_ops`] is the throughput
/// numerator; the host derives the run-level `taobench_*` Rate metrics from the
/// pooled aggregate.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct TaobenchStats {
    /// Lookups issued (request time).
    pub get_cmds: u64,
    /// Lookups that missed (request time).
    pub get_misses: u64,
    /// Hits served (response time).
    pub fast_ops: u64,
    /// Misses served via the slow path (response time).
    pub slow_ops: u64,
    /// Wall-clock window this stat covers, ns — the qps denominator. Per-phase:
    /// the phase segment; whole-run: the run window. Merged as MAX (the window is
    /// shared by the concurrent threads/workers being pooled, not summed).
    pub elapsed_ns: u64,
}

impl TaobenchStats {
    pub(crate) fn merge(&mut self, o: &TaobenchStats) {
        // saturating_add: these are guest-runtime monotonic op counters pooled
        // across concurrent workers/cgroups. Overflow is unreachable for honest
        // data (real counts ~1e7..1e14 << u64::MAX), but a corrupt/hostile guest
        // value would otherwise debug-panic or release-wrap to a silently-wrong
        // qps/hit Rate; saturating is exact for every in-range value.
        self.get_cmds = self.get_cmds.saturating_add(o.get_cmds);
        self.get_misses = self.get_misses.saturating_add(o.get_misses);
        self.fast_ops = self.fast_ops.saturating_add(o.fast_ops);
        self.slow_ops = self.slow_ops.saturating_add(o.slow_ops);
        // The wall window is shared across the pooled concurrent threads/workers
        // (they run the same phase at the same time), so MAX, not sum.
        self.elapsed_ns = self.elapsed_ns.max(o.elapsed_ns);
    }
    /// Completed ops (fast + slow) — the throughput numerator.
    pub fn total_ops(&self) -> u64 {
        self.fast_ops.saturating_add(self.slow_ops)
    }
}

/// One PER-PHASE accounting window's taobench carrier: the integer counters
/// ([`TaobenchStats`]) plus the open-loop serve-latency histogram. Internal
/// (`pub(crate)`) — the taobench analog of
/// [`crate::workload::schbench::run::SchbenchPhaseStats`] (counters + `PlatStats`
/// histograms together), and like it, never on a public field. It rides only the
/// per-phase carriers (`PhaseSlice::taobench` / `PhaseCgroupStats::taobench`); the
/// WHOLE-RUN carrier is the counters-only [`TaobenchStats`] (the serve histogram
/// is per-phase data — the whole-run serve distribution is the union of these
/// per-phase histograms, computed where needed). `serve_lat` is EMPTY in closed
/// loop: serve latency is measured only under open-loop arrival, so its
/// percentiles read absent there.
#[derive(Clone, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub(crate) struct TaobenchPhaseStats {
    /// Throughput / hit counters (request- and response-time).
    pub(crate) counters: TaobenchStats,
    /// Coordinated-omission serve latency (µs), measured from intended arrival.
    /// Exposed to test authors only via the derived `taobench_serve_*_us` metric
    /// keys (the schbench-histogram model), never as a direct field.
    pub(crate) serve_lat: PlatStats,
}

impl TaobenchPhaseStats {
    /// Pool another window: Σ counters ([`TaobenchStats::merge`]) and union the
    /// serve histograms ([`PlatStats::combine`]) — the per-cgroup / cross-thread
    /// fold, mirroring the schbench carrier merge.
    pub(crate) fn merge(&mut self, other: &TaobenchPhaseStats) {
        self.counters.merge(&other.counters);
        self.serve_lat.combine(&other.serve_lat);
    }
}

/// The engine's return: the whole-run merged COUNTERS ([`TaobenchStats`], the
/// qps/hit aggregate) and the per-phase-epoch carriers (each counters + the
/// serve-latency histogram). The whole-run carrier is counters-only — mirroring
/// the wire carriers (`WorkerReport::taobench_whole` / `CgroupStats::taobench_whole`);
/// a whole-run serve distribution, when needed (the standalone driver), is the
/// union of the per-phase `serve_lat` histograms.
pub(crate) struct TaobenchOutcome {
    pub whole_run: TaobenchStats,
    pub phases: Vec<(u32, TaobenchPhaseStats)>,
}

/// Per-thread accumulation with phase-epoch bucketing: the current epoch's
/// counters roll into `phases` when the epoch changes (the schbench drain-on-epoch
/// pattern), and `whole` accumulates the un-bucketed run total. Wall windows are
/// stamped from `CLOCK_MONOTONIC` at each roll (per-phase segment) and at finalize
/// (whole-run).
struct ThreadAccum {
    cur_epoch: u32,
    cur: TaobenchPhaseStats,
    phases: BTreeMap<u32, TaobenchPhaseStats>,
    /// Whole-run counters only (the qps/hit aggregate). The serve histogram is
    /// per-phase data (in `cur`/`phases`); the whole-run serve distribution, when
    /// needed, is the union of the per-phase histograms.
    whole: TaobenchStats,
    /// When the current phase segment started (ns).
    phase_start_ns: u64,
    /// When this thread started (ns) — the whole-run window start.
    thread_start_ns: u64,
}

impl ThreadAccum {
    fn new(epoch: u32) -> Self {
        let now = monotonic_nanos();
        ThreadAccum {
            cur_epoch: epoch,
            cur: TaobenchPhaseStats::default(),
            phases: BTreeMap::new(),
            whole: TaobenchStats::default(),
            phase_start_ns: now,
            thread_start_ns: now,
        }
    }
    /// Roll the current bucket into `phases` (stamping its segment wall) and start
    /// a fresh one for `epoch`.
    fn roll_to(&mut self, epoch: u32) {
        if epoch != self.cur_epoch {
            let now = monotonic_nanos();
            let mut cur = std::mem::take(&mut self.cur);
            cur.counters.elapsed_ns = now.saturating_sub(self.phase_start_ns);
            self.phases.entry(self.cur_epoch).or_default().merge(&cur);
            self.cur_epoch = epoch;
            self.phase_start_ns = now;
        }
    }
    /// Record a lookup at request time in `epoch`.
    fn record_cmd(&mut self, epoch: u32, hit: bool) {
        self.roll_to(epoch);
        self.cur.counters.get_cmds += 1;
        self.whole.get_cmds += 1;
        if !hit {
            self.cur.counters.get_misses += 1;
            self.whole.get_misses += 1;
        }
    }
    /// Record a completion at response time in `epoch`.
    fn record_complete(&mut self, epoch: u32, hit: bool) {
        self.roll_to(epoch);
        if hit {
            self.cur.counters.fast_ops += 1;
            self.whole.fast_ops += 1;
        } else {
            self.cur.counters.slow_ops += 1;
            self.whole.slow_ops += 1;
        }
    }
    /// Record an open-loop serve latency (`ns` from the intended arrival) in
    /// `epoch`, into the current phase segment's histogram (µs buckets). Closed
    /// loop never calls this, so the histograms stay empty and the serve-latency
    /// keys read absent. The whole-run serve distribution is the union of the
    /// per-phase histograms, so only the per-phase bucket is recorded here.
    fn record_serve_lat(&mut self, epoch: u32, ns: u64) {
        self.roll_to(epoch);
        let us = (ns / 1000).min(u32::MAX as u64) as u32;
        self.cur.serve_lat.add_lat(us);
    }
    /// Flush the last open bucket (stamping its segment wall) and stamp the
    /// whole-run window.
    fn finalize(mut self) -> Self {
        let now = monotonic_nanos();
        let mut cur = std::mem::take(&mut self.cur);
        cur.counters.elapsed_ns = now.saturating_sub(self.phase_start_ns);
        self.phases.entry(self.cur_epoch).or_default().merge(&cur);
        self.whole.elapsed_ns = now.saturating_sub(self.thread_start_ns);
        self
    }
}

/// Read the current phase epoch (`0` when phases are not tracked).
fn read_epoch(phase_epoch: Option<&AtomicU32>) -> u32 {
    phase_epoch.map(|e| e.load(Ordering::Relaxed)).unwrap_or(0)
}

// ---------------------------------------------------------------------------
// Engine
// ---------------------------------------------------------------------------

/// Run the taobench engine until `stop` is set. `progress` counts completed ops
/// (the live work-unit counter); `phase_epoch` buckets per-phase stats.
pub(crate) fn run(
    config: &TaobenchConfig,
    stop: &AtomicBool,
    progress: &AtomicU64,
    phase_epoch: Option<&AtomicU32>,
) -> TaobenchOutcome {
    let allowed_cpus = resolve_allowed_cpus();
    let n_clients = config.resolve_client_threads(allowed_cpus);
    let n_slow = config.resolve_slow_threads(n_clients);
    // Open-loop per-client inter-arrival = 1s / (arrival_rate / n_clients) =
    // 1e9 * n_clients / arrival_rate. `0` arrival_rate ⇒ `0` interval ⇒ closed
    // loop (issue as fast as completions allow — the legacy behavior). `.max(1)`
    // keeps a non-zero rate strictly open-loop even at an absurd rate (so the
    // `interval_ns != 0` mode gate in client_loop never misfires to closed loop).
    // saturating_mul: an absurd client count (physically unreachable — it would
    // OOM on thread spawn first) cannot wrap the numerator to a tiny interval, the
    // same overflow-safe discipline as `TaobenchStats::merge`.
    let interval_ns: u64 = if config.arrival_rate == 0 {
        0
    } else {
        (1_000_000_000u64.saturating_mul(n_clients as u64) / config.arrival_rate as u64).max(1)
    };

    // Size the resident object count and the key range. Mean value size pins the
    // object count for a byte budget; key_range = capacity / target_hit makes a
    // uniform key stream hit the resident set at ≈ target_hit at equilibrium.
    let mean_value = mean_value_size();
    let capacity_bytes = config.cache_capacity_mib.max(1) * 1024 * 1024;
    let total_objects = (capacity_bytes / mean_value).max(SHARDS);
    let key_range = ((total_objects as f64) / config.target_hit_fraction()).ceil() as u64;

    let cache = Cache::new(total_objects);
    // Warm the cache to capacity so the hit ratio starts at the target rather
    // than climbing from empty (the real warms before measuring). Keys
    // [0, total_objects) are the initial resident set; the client key stream over
    // [0, key_range) then hits them at ≈ target_hit and the eviction<->refill
    // equilibrium holds it there.
    {
        let mut warm = Rng::new(0xC0FFEE);
        for k in 0..total_objects as u64 {
            cache.fill(k, sample_value_size(&mut warm));
        }
    }

    let slow_q = SlowQueue::new();
    let slots: Vec<Slot> = (0..n_clients).map(|_| Slot::new()).collect();
    let disp_stop = AtomicBool::new(false);

    let client_accums: Vec<ThreadAccum> = std::thread::scope(|s| {
        // Dispatchers first so a client miss is served immediately.
        let dispatchers: Vec<_> = (0..n_slow)
            .map(|i| {
                let cache = &cache;
                let slow_q = &slow_q;
                let slots = &slots;
                let disp_stop = &disp_stop;
                s.spawn(move || dispatcher_loop(i, config, cache, slow_q, slots, disp_stop))
            })
            .collect();

        let clients: Vec<_> = (0..n_clients)
            .map(|i| {
                let cache = &cache;
                let slow_q = &slow_q;
                let slot = &slots[i];
                s.spawn(move || {
                    client_loop(
                        i,
                        stop,
                        progress,
                        phase_epoch,
                        cache,
                        slow_q,
                        slot,
                        key_range,
                        interval_ns,
                    )
                })
            })
            .collect();

        // Join the clients first: once every client has returned (each only
        // exits at the top of its loop, after its outstanding request — if any —
        // has been served), NO further misses can be enqueued. Only then signal
        // the dispatchers, so a dispatcher can never exit while a client is still
        // blocked on an unserved miss.
        let accums: Vec<ThreadAccum> = clients
            .into_iter()
            .map(|c| c.join().expect("taobench client panicked"))
            .collect();
        disp_stop.store(true, Ordering::Release);
        slow_q.cv.notify_all();
        for d in dispatchers {
            d.join().expect("taobench dispatcher panicked");
        }
        accums
    });

    // Reduce: merge per-epoch and whole-run across clients. Each ThreadAccum
    // stamped its per-phase segment walls + its whole-run window in finalize().
    let mut all_phases: BTreeMap<u32, TaobenchPhaseStats> = BTreeMap::new();
    let mut whole = TaobenchStats::default();
    for accum in client_accums {
        for (e, s) in accum.phases {
            all_phases.entry(e).or_default().merge(&s);
        }
        whole.merge(&accum.whole);
    }

    TaobenchOutcome {
        whole_run: whole,
        phases: all_phases.into_iter().collect(),
    }
}

/// One client: pick a key, look it up, serve a hit inline (touch + fast op) or
/// hand a miss to a dispatcher and block until filled (slow op). `interval_ns ==
/// 0` is CLOSED loop (issue the next request as soon as the prior completes — the
/// legacy behavior). `interval_ns > 0` is OPEN loop: requests are due on a fixed
/// `interval_ns` cadence independent of completion (the first one interval after
/// the loop start `t0`), and serve latency is measured from the INTENDED arrival
/// (coordinated-omission) so a backlog from slow service is folded into the late
/// requests' latency instead of omitted.
#[allow(clippy::too_many_arguments)]
fn client_loop(
    id: usize,
    stop: &AtomicBool,
    progress: &AtomicU64,
    phase_epoch: Option<&AtomicU32>,
    cache: &Cache,
    slow_q: &SlowQueue,
    slot: &Slot,
    key_range: u64,
    interval_ns: u64,
) -> ThreadAccum {
    let mut rng = Rng::new(0x5EED_0000 ^ (id as u64).wrapping_mul(0x9E37_79B1));
    let mut accum = ThreadAccum::new(read_epoch(phase_epoch));
    let open_loop = interval_ns != 0;
    // Open-loop schedule cursor, advanced one `interval` before each request: the
    // first request is due at `t0 + interval`, then every `interval` thereafter.
    let mut intended = monotonic_nanos();

    'client: while !stop.load(Ordering::Acquire) {
        if open_loop {
            // Pace to the next scheduled arrival; if behind (overload), the wait
            // collapses to zero and the gap is captured as serve latency below.
            // Each sleep is capped at STOP_POLL_QUANTUM_NS so a shutdown set during
            // a long inter-arrival gap is observed within the quantum (not after a
            // full interval) — exiting promptly via `break 'client` to `finalize`,
            // bounding client-join teardown regardless of `arrival_rate`.
            intended = intended.saturating_add(interval_ns);
            loop {
                if stop.load(Ordering::Acquire) {
                    break 'client;
                }
                let now = monotonic_nanos();
                if now >= intended {
                    break;
                }
                let nap = (intended - now).min(STOP_POLL_QUANTUM_NS);
                std::thread::sleep(std::time::Duration::from_nanos(nap));
            }
        }
        let epoch_cmd = read_epoch(phase_epoch);
        let key = rng.below(key_range);
        let hit = cache.get_touch(key);
        accum.record_cmd(epoch_cmd, hit);

        if hit {
            // Hit completes inline; response epoch == request epoch.
            accum.record_complete(epoch_cmd, true);
            if open_loop {
                let lat = monotonic_nanos().saturating_sub(intended);
                accum.record_serve_lat(epoch_cmd, lat);
            }
        } else {
            // Miss: hand to a dispatcher and block until it fills + wakes us.
            // The dispatcher always serves us before `disp_stop` is set (which
            // happens only after every client has joined), so this never blocks
            // past shutdown.
            slow_q.push(SlowReq { key, client: id });
            slot.wait_filled();
            let resp_epoch = read_epoch(phase_epoch);
            accum.record_complete(resp_epoch, false);
            if open_loop {
                // Completion timestamp taken by the CLIENT after wait_filled, so
                // serve latency covers the full slow_q queueing + dispatcher
                // service + wakeup, all measured from the intended arrival.
                let lat = monotonic_nanos().saturating_sub(intended);
                accum.record_serve_lat(resp_epoch, lat);
            }
        }
        progress.fetch_add(1, Ordering::Relaxed);
    }
    accum.finalize()
}

/// One slow dispatcher: pull a miss, perform the simulated backing-store fetch,
/// fill the cache, and wake the waiting client. Drains any queued misses before
/// exiting on `disp_stop`.
fn dispatcher_loop(
    id: usize,
    config: &TaobenchConfig,
    cache: &Cache,
    slow_q: &SlowQueue,
    slots: &[Slot],
    disp_stop: &AtomicBool,
) {
    let mut rng = Rng::new(0xD15A_0000 ^ (id as u64).wrapping_mul(0x9E37_79B1));
    loop {
        let req = {
            let mut q = slow_q.q.lock().expect("slow queue poisoned");
            loop {
                if let Some(r) = q.pop_front() {
                    break Some(r);
                }
                if disp_stop.load(Ordering::Acquire) {
                    break None;
                }
                q = slow_q.cv.wait(q).expect("slow queue poisoned");
            }
        };
        let Some(req) = req else { break };

        backing_store_fetch(config.slow_path_sleep_us);
        cache.fill(req.key, sample_value_size(&mut rng));
        slots[req.client].signal();
    }
}

// ---------------------------------------------------------------------------
// Topology + size helpers
// ---------------------------------------------------------------------------

/// The mean of the value-size distribution (for sizing the object count).
fn mean_value_size() -> usize {
    let total_w: u64 = VALUE_WEIGHTS.iter().map(|&w| w as u64).sum();
    let weighted: u64 = VALUE_SIZES
        .iter()
        .zip(VALUE_WEIGHTS.iter())
        .map(|(&s, &w)| s as u64 * w as u64)
        .sum();
    (weighted / total_w).max(1) as usize
}

/// Resolve the number of CPUs the worker is allowed to run on (its affinity
/// mask), falling back to the online CPU count. Mirrors the schbench engine's
/// cpuset-scoped resolution.
fn resolve_allowed_cpus() -> usize {
    // SAFETY: `sched_getaffinity` writes the calling thread's CPU mask into the
    // provided cpu_set; reads nothing else.
    unsafe {
        let mut set: libc::cpu_set_t = core::mem::zeroed();
        if libc::sched_getaffinity(0, core::mem::size_of::<libc::cpu_set_t>(), &mut set) == 0 {
            let n = libc::CPU_COUNT(&set);
            if n > 0 {
                return n as usize;
            }
        }
    }
    let n = unsafe { libc::sysconf(libc::_SC_NPROCESSORS_ONLN) };
    if n > 0 { n as usize } else { 1 }
}

// ---------------------------------------------------------------------------
// Host-side standalone runner (validation driver entry)
// ---------------------------------------------------------------------------

/// Host-side standalone run of the taobench engine for `run_secs`, returning a
/// summary report — the analog of schbench's `run_standalone`, backing the
/// `ktstr-taobench-validate` driver for the side-by-side comparison against the
/// reference taobench. NOT used in-VM (the scenario engine drives `run` there).
///
/// The engine warms the cache synchronously before the client threads start; the
/// report's `elapsed_secs` is the engine-measured CLIENT window (post-warmup), so
/// the qps figures are steady-state, not diluted by warmup.
pub fn run_standalone(config: &TaobenchConfig, run_secs: u64) -> TaobenchStandaloneReport {
    let stop = AtomicBool::new(false);
    let progress = AtomicU64::new(0);
    let allowed_cpus = resolve_allowed_cpus();
    let nr_client_threads = config.resolve_client_threads(allowed_cpus);
    let nr_slow_threads = config.resolve_slow_threads(nr_client_threads);
    let outcome = std::thread::scope(|s| {
        let h = s.spawn(|| run(config, &stop, &progress, None));
        // The benchmark run window — a workload-defined duration (the same model
        // as the scenario engine's hold step), not a synchronization wait.
        std::thread::sleep(std::time::Duration::from_secs(run_secs));
        stop.store(true, Ordering::Release);
        h.join().expect("taobench standalone run panicked")
    });
    // Whole-run serve-latency distribution = union of the per-phase histograms
    // (the whole-run carrier is counters-only). Empty in closed loop, so the
    // report's serve percentiles read absent there.
    let mut serve = PlatStats::default();
    for (_epoch, p) in &outcome.phases {
        serve.combine(&p.serve_lat);
    }
    TaobenchStandaloneReport::from_run(
        &outcome.whole_run,
        &serve,
        nr_client_threads,
        nr_slow_threads,
    )
}

/// Summary of a [`run_standalone`] run — the headline taobench metrics in the
/// shape the reference taobench server reports (`fast_qps` / `hit_rate` /
/// `slow_qps`) plus the derived `total_qps` (= fast + slow) and `hit_ratio`
/// (= fast / total). Under open-loop arrival (`arrival_rate > 0`) it also carries
/// the coordinated-omission serve-latency percentiles; these are `None` in closed
/// loop (no intended-arrival schedule, so no serve latency is measured).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TaobenchStandaloneReport {
    /// (fast + slow) ops per second over the measured window.
    pub total_qps: f64,
    /// Hits served per second.
    pub fast_qps: f64,
    /// Misses served (slow path) per second.
    pub slow_qps: f64,
    /// Response-time hit ratio: fast / (fast + slow).
    pub hit_ratio: f64,
    /// Command-time hit rate: 1 - get_misses / get_cmds.
    pub hit_rate: f64,
    /// Completed ops (fast + slow).
    pub total_ops: u64,
    /// Hits served.
    pub fast_ops: u64,
    /// Misses served via the slow path.
    pub slow_ops: u64,
    /// The engine-measured client window, seconds.
    pub elapsed_secs: f64,
    /// Resolved client thread count.
    pub nr_client_threads: usize,
    /// Resolved slow dispatcher count.
    pub nr_slow_threads: usize,
    /// Open-loop coordinated-omission serve-latency percentiles (µs), measured
    /// from intended arrival. `None` in closed loop / when no serve samples were
    /// recorded (matching the metric-key ABSENT discipline).
    pub serve_p50_us: Option<u32>,
    /// Open-loop serve-latency p99 (µs) — the headline coordinated-omission tail.
    pub serve_p99_us: Option<u32>,
    /// Open-loop serve-latency p99.9 (µs).
    pub serve_p999_us: Option<u32>,
    /// Open-loop serve-latency minimum sample (µs).
    pub serve_min_us: Option<u32>,
    /// Open-loop serve-latency maximum sample (µs).
    pub serve_max_us: Option<u32>,
}

impl TaobenchStandaloneReport {
    fn from_run(
        w: &TaobenchStats,
        serve: &PlatStats,
        nr_client_threads: usize,
        nr_slow_threads: usize,
    ) -> Self {
        use crate::workload::schbench::plat::Pct;
        let secs = (w.elapsed_ns as f64 / 1e9).max(f64::MIN_POSITIVE);
        let total = w.total_ops();
        // Serve-latency percentiles are present iff open-loop arrival recorded
        // samples; closed loop leaves the histogram empty, so all fields read
        // None (the metric-key ABSENT discipline).
        let serve_q = (serve.sample_count() > 0).then(|| serve.percentiles());
        TaobenchStandaloneReport {
            total_qps: total as f64 / secs,
            fast_qps: w.fast_ops as f64 / secs,
            slow_qps: w.slow_ops as f64 / secs,
            hit_ratio: if total > 0 {
                w.fast_ops as f64 / total as f64
            } else {
                0.0
            },
            hit_rate: if w.get_cmds > 0 {
                1.0 - (w.get_misses as f64 / w.get_cmds as f64)
            } else {
                0.0
            },
            total_ops: total,
            fast_ops: w.fast_ops,
            slow_ops: w.slow_ops,
            elapsed_secs: secs,
            nr_client_threads,
            nr_slow_threads,
            serve_p50_us: serve_q.as_ref().map(|q| q.value_at(Pct::P50)),
            serve_p99_us: serve_q.as_ref().map(|q| q.value_at(Pct::P99)),
            serve_p999_us: serve_q.as_ref().map(|q| q.value_at(Pct::P999)),
            serve_min_us: serve_q.as_ref().map(|q| q.min),
            serve_max_us: serve_q.as_ref().map(|q| q.max),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn value_size_distribution_mean_is_small_object_heavy() {
        // The representative distribution is small-object-heavy (mean well under
        // 1 KiB) with a tail to 64 KiB.
        let m = mean_value_size();
        assert!(
            (200..=500).contains(&m),
            "mean value size {m} B is small-object-heavy"
        );
        assert_eq!(*VALUE_SIZES.last().unwrap(), 65536, "tail reaches 64 KiB");
    }

    #[test]
    fn taobench_stats_merge_and_total_ops_saturate_on_overflow() {
        // Pooling guest-runtime op counters must saturate, not wrap: a corrupt
        // u64::MAX component must never produce a silently-wrong (wrapped-small)
        // qps/hit Rate. Saturating is exact for all in-range data.
        let mut a = TaobenchStats {
            get_cmds: u64::MAX,
            get_misses: u64::MAX,
            fast_ops: u64::MAX,
            slow_ops: 1,
            elapsed_ns: 1000,
        };
        let b = TaobenchStats {
            get_cmds: 5,
            get_misses: 5,
            fast_ops: 5,
            slow_ops: 5,
            elapsed_ns: 9000,
        };
        a.merge(&b);
        assert_eq!(a.get_cmds, u64::MAX, "saturates, not wraps to 4");
        assert_eq!(a.get_misses, u64::MAX);
        assert_eq!(a.fast_ops, u64::MAX);
        assert_eq!(a.slow_ops, 6);
        assert_eq!(a.elapsed_ns, 9000, "wall window is MAX, not summed");
        // total_ops = fast_ops + slow_ops saturates (u64::MAX + 6 -> u64::MAX).
        assert_eq!(a.total_ops(), u64::MAX, "total_ops saturates");
    }

    #[test]
    fn engine_serves_ops_and_hit_ratio_settles_near_target_not_one() {
        // A short run on a small cache: the hit ratio must settle near the target
        // (the eviction<->refill equilibrium), NOT drift to 1.0 (which a
        // self-healing cache with no eviction would do).
        let cfg = TaobenchConfig::default()
            .client_threads(4)
            .slow_threads(2)
            .cache_capacity_mib(8)
            .target_hit_pct(90)
            .slow_path_sleep_us(10);
        let stop = AtomicBool::new(false);
        let progress = AtomicU64::new(0);

        std::thread::scope(|s| {
            let h = s.spawn(|| run(&cfg, &stop, &progress, None));
            // Spin until the engine has served enough ops to reach equilibrium.
            while progress.load(Ordering::Relaxed) < 100_000 {
                std::hint::spin_loop();
            }
            stop.store(true, Ordering::Release);
            let out = h.join().expect("engine panicked");

            let total = out.whole_run.total_ops();
            assert!(total > 0, "engine served ops");
            assert!(
                out.whole_run.elapsed_ns > 0,
                "the run window was measured"
            );

            let hit = out.whole_run.fast_ops as f64 / total as f64;
            assert!(
                (0.80..=0.97).contains(&hit),
                "hit ratio {hit} settles near target 0.90 (eviction equilibrium), not 1.0"
            );
            assert!(
                out.whole_run.slow_ops > 0,
                "the slow/miss path is exercised"
            );
        });
    }

    #[test]
    fn taobench_config_serde_roundtrips() {
        // The new serialized type roundtrips unchanged.
        // Every field set to a non-default value to exercise the full serde surface.
        let cfg = TaobenchConfig::default()
            .client_threads(8)
            .slow_threads(3)
            .cache_capacity_mib(128)
            .target_hit_pct(85)
            .slow_path_sleep_us(250)
            .arrival_rate(50_000);
        let json = serde_json::to_string(&cfg).expect("TaobenchConfig must serialize");
        let back: TaobenchConfig =
            serde_json::from_str(&json).expect("TaobenchConfig must deserialize");
        assert_eq!(cfg, back, "config roundtrips unchanged");
    }

    #[test]
    fn worktype_taobench_registration_and_serde() {
        use crate::workload::WorkType;
        let wt = WorkType::taobench(
            TaobenchConfig::default()
                .client_threads(4)
                .target_hit_pct(95),
        );
        assert_eq!(wt.name(), "Taobench");
        // from_name yields the default-config variant.
        assert_eq!(
            WorkType::from_name("Taobench"),
            Some(WorkType::Taobench {
                config: TaobenchConfig::default()
            })
        );
        // The variant serde-roundtrips, carrying its config.
        let json = serde_json::to_string(&wt).expect("WorkType::Taobench must serialize");
        let back: WorkType =
            serde_json::from_str(&json).expect("WorkType::Taobench must deserialize");
        assert_eq!(wt, back);
    }

    #[test]
    fn taobench_config_reachable_via_prelude() {
        // Regression-pin the prelude placement: test authors construct the config
        // via `use ktstr::prelude::*`. Dropping TaobenchConfig from the prelude
        // would fail this compile. Also exercises the Eq derive.
        let cfg: crate::prelude::TaobenchConfig = crate::prelude::TaobenchConfig::default();
        assert_eq!(cfg, TaobenchConfig::default());
    }

    #[test]
    fn taobench_stats_serde_roundtrips() {
        // TaobenchStats is a pub serialized type that rides real wires — a field
        // on WorkerReport (the worker→host postcard payload) and CgroupStats (the
        // sidecar JSON) — so a field-rename or serde-attr drift is a silent
        // data-loss risk this pins. Every field distinct +
        // non-zero to exercise the full serde surface.
        let s = TaobenchStats {
            get_cmds: 1000,
            get_misses: 150,
            fast_ops: 850,
            slow_ops: 150,
            elapsed_ns: 9_000_000_000,
        };
        let json = serde_json::to_string(&s).expect("TaobenchStats must serialize");
        let back: TaobenchStats =
            serde_json::from_str(&json).expect("TaobenchStats must deserialize");
        assert_eq!(s, back, "TaobenchStats roundtrips unchanged");
    }

    #[test]
    fn taobench_stats_reachable_via_prelude() {
        // Regression-pin the prelude placement: TaobenchStats is read off the
        // preluded WorkerReport/CgroupStats `taobench_whole` fields, so it must be
        // nameable via `use ktstr::prelude::*`. Dropping it from the prelude would
        // fail this compile. Also exercises Default + Eq.
        let s: crate::prelude::TaobenchStats = crate::prelude::TaobenchStats::default();
        assert_eq!(s, TaobenchStats::default());
    }

    #[test]
    fn taobench_config_arrival_rate_default_and_setter() {
        // Default is closed loop (no open-loop schedule); the setter sets the
        // aggregate ops/sec field.
        assert_eq!(
            TaobenchConfig::default().arrival_rate,
            0,
            "default arrival_rate is 0 (closed loop)"
        );
        let cfg = TaobenchConfig::default().arrival_rate(100_000);
        assert_eq!(cfg.arrival_rate, 100_000, "setter sets arrival_rate");
    }

    #[test]
    fn taobench_phase_stats_merge_sums_counters_and_unions_serve_lat() {
        // The carrier merge pools counters (Σ ops, MAX wall — TaobenchStats::merge)
        // AND unions the serve histograms (PlatStats::combine) — the per-cgroup /
        // cross-thread fold.
        let mut a = TaobenchPhaseStats {
            counters: TaobenchStats {
                get_cmds: 10,
                get_misses: 2,
                fast_ops: 8,
                slow_ops: 2,
                elapsed_ns: 1000,
            },
            ..Default::default()
        };
        a.serve_lat.add_lat(5);
        a.serve_lat.add_lat(50);
        let mut b = TaobenchPhaseStats {
            counters: TaobenchStats {
                get_cmds: 6,
                get_misses: 1,
                fast_ops: 5,
                slow_ops: 1,
                elapsed_ns: 4000,
            },
            ..Default::default()
        };
        b.serve_lat.add_lat(500);
        a.merge(&b);
        assert_eq!(a.counters.get_cmds, 16, "Σ get_cmds");
        assert_eq!(a.counters.fast_ops, 13, "Σ fast_ops");
        assert_eq!(a.counters.slow_ops, 3, "Σ slow_ops");
        assert_eq!(a.counters.elapsed_ns, 4000, "wall is MAX, not summed");
        assert_eq!(
            a.serve_lat.sample_count(),
            3,
            "serve histograms union (2 + 1 samples)"
        );
    }

    #[test]
    fn taobench_phase_stats_serde_roundtrips() {
        // TaobenchPhaseStats rides the WorkerReport postcard + CgroupStats sidecar
        // JSON wires; both the counters and the serve histogram must roundtrip.
        let mut s = TaobenchPhaseStats {
            counters: TaobenchStats {
                get_cmds: 1000,
                get_misses: 150,
                fast_ops: 850,
                slow_ops: 150,
                elapsed_ns: 9_000_000_000,
            },
            ..Default::default()
        };
        s.serve_lat.add_lat(12);
        s.serve_lat.add_lat(340);
        s.serve_lat.add_lat(99_999);
        let json = serde_json::to_string(&s).expect("TaobenchPhaseStats must serialize");
        let back: TaobenchPhaseStats =
            serde_json::from_str(&json).expect("TaobenchPhaseStats must deserialize");
        assert_eq!(
            s, back,
            "TaobenchPhaseStats roundtrips unchanged (counters + serve_lat)"
        );
    }

    #[test]
    fn open_loop_stamps_serve_latency_on_every_completion() {
        // Open loop (non-zero arrival_rate) measures coordinated-omission serve
        // latency: every completion — a hit (served inline) AND a miss (served via
        // the slow path) — records exactly one sample from its intended arrival, so
        // the histogram count equals total_ops. An absurdly high rate keeps the
        // pacing in permanent overload (the per-arrival sleep collapses to zero), so
        // the test runs at full speed while still exercising stamping on both arms.
        let cfg = TaobenchConfig::default()
            .client_threads(4)
            .slow_threads(2)
            .cache_capacity_mib(8)
            .target_hit_pct(90)
            .slow_path_sleep_us(10)
            .arrival_rate(1_000_000_000);
        let stop = AtomicBool::new(false);
        let progress = AtomicU64::new(0);
        let out = std::thread::scope(|s| {
            let h = s.spawn(|| run(&cfg, &stop, &progress, None));
            while progress.load(Ordering::Relaxed) < 50_000 {
                std::hint::spin_loop();
            }
            stop.store(true, Ordering::Release);
            h.join().expect("engine panicked")
        });
        let total = out.whole_run.total_ops();
        assert!(total > 0, "engine served ops");
        assert!(out.whole_run.slow_ops > 0, "the miss arm is exercised");
        // The whole-run serve distribution is the union of the per-phase
        // histograms (the whole-run carrier is counters-only). Its count equals
        // total_ops: every completion (hit inline + miss after wait_filled)
        // records exactly one serve-latency sample.
        let mut serve = PlatStats::default();
        for (_epoch, p) in &out.phases {
            serve.combine(&p.serve_lat);
        }
        assert_eq!(
            serve.sample_count(),
            total,
            "every completion (hit + miss) records one serve-latency sample"
        );
    }

    #[test]
    fn closed_loop_records_no_serve_latency() {
        // arrival_rate == 0 is closed loop: there is no intended-arrival schedule,
        // so serve latency is undefined and the histogram stays EMPTY (the
        // taobench_serve_*_us keys then read absent, distinct from a measured 0).
        let cfg = TaobenchConfig::default()
            .client_threads(4)
            .slow_threads(2)
            .cache_capacity_mib(8)
            .target_hit_pct(90)
            .slow_path_sleep_us(10); // arrival_rate defaults to 0 (closed loop)
        let stop = AtomicBool::new(false);
        let progress = AtomicU64::new(0);
        let out = std::thread::scope(|s| {
            let h = s.spawn(|| run(&cfg, &stop, &progress, None));
            while progress.load(Ordering::Relaxed) < 20_000 {
                std::hint::spin_loop();
            }
            stop.store(true, Ordering::Release);
            h.join().expect("engine panicked")
        });
        assert!(out.whole_run.total_ops() > 0, "ops completed");
        for (epoch, p) in &out.phases {
            assert_eq!(
                p.serve_lat.sample_count(),
                0,
                "closed loop records no per-phase serve latency (epoch {epoch})"
            );
        }
    }

    #[test]
    fn open_loop_per_phase_carrier_holds_serve_latency() {
        // The per-phase carrier (PhaseSlice.taobench) carries the serve histogram
        // (the whole-run carrier is counters-only). With a single static epoch
        // every completion's sample lands in that epoch's bucket, so the per-phase
        // serve count equals the whole-run completion count (total_ops).
        let cfg = TaobenchConfig::default()
            .client_threads(4)
            .slow_threads(2)
            .cache_capacity_mib(8)
            .target_hit_pct(90)
            .slow_path_sleep_us(10)
            .arrival_rate(1_000_000_000);
        let stop = AtomicBool::new(false);
        let progress = AtomicU64::new(0);
        let epoch = AtomicU32::new(7);
        let out = std::thread::scope(|s| {
            let h = s.spawn(|| run(&cfg, &stop, &progress, Some(&epoch)));
            while progress.load(Ordering::Relaxed) < 30_000 {
                std::hint::spin_loop();
            }
            stop.store(true, Ordering::Release);
            h.join().expect("engine panicked")
        });
        let phase7 = out
            .phases
            .iter()
            .find(|(e, _)| *e == 7)
            .map(|(_, p)| p)
            .expect("epoch-7 phase carrier present");
        assert!(
            phase7.serve_lat.sample_count() > 0,
            "per-phase serve latency recorded"
        );
        assert_eq!(
            phase7.serve_lat.sample_count(),
            out.whole_run.total_ops(),
            "single-epoch run: every completion stamps one per-phase serve sample"
        );
    }

    #[test]
    fn open_loop_pacing_sleep_is_bounded_for_prompt_shutdown() {
        // A degenerate low arrival_rate makes the inter-arrival interval ~1s, but a
        // client waiting for its next scheduled arrival must observe `stop` within
        // STOP_POLL_QUANTUM_NS (50 ms) and exit, so the join completes well under
        // one interval. An unbounded pacing sleep would block the join for the full
        // ~1s. The 500 ms bound is 10x the quantum (robust to scheduling jitter)
        // and half the interval (so it cleanly distinguishes bounded from
        // unbounded).
        let cfg = TaobenchConfig::default()
            .client_threads(1)
            .slow_threads(1)
            .cache_capacity_mib(4)
            .target_hit_pct(90)
            .slow_path_sleep_us(0)
            .arrival_rate(1); // interval_ns = 1e9 * 1 / 1 = 1s
        let stop = AtomicBool::new(false);
        let progress = AtomicU64::new(0);
        let start = std::time::Instant::now();
        std::thread::scope(|s| {
            let h = s.spawn(|| run(&cfg, &stop, &progress, None));
            // Let the client reach its first (long) pacing sleep, then shut down.
            std::thread::sleep(std::time::Duration::from_millis(20));
            stop.store(true, Ordering::Release);
            let _ = h.join().expect("engine panicked");
        });
        let elapsed = start.elapsed();
        assert!(
            elapsed < std::time::Duration::from_millis(500),
            "open-loop client joined in {elapsed:?}; the pacing sleep must be \
             bounded (interval is ~1s — an unbounded sleep would block the join)"
        );
    }
}
