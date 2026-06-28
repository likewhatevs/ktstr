//! The taobench engine: a bounded, sharded, evicting key-value cache served by a
//! closed-loop client population, with a fast in-cache path (hit) and a slow
//! backing-store-miss path (a worker-defined sleep on a dispatcher thread). It is
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
//! - `client_threads` CLIENT threads run a closed loop: pick a key, look it up,
//!   and on a HIT touch the stored value bytes (the cache read-bandwidth cost) and
//!   count a fast op; on a MISS hand the key to a slow dispatcher and block until
//!   it is filled, then count a slow op.
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
    /// Closed-loop CLIENT threads that issue lookups and serve hits. `0` resolves
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
}

impl Default for TaobenchConfig {
    fn default() -> Self {
        Self {
            client_threads: 0,
            slow_threads: 0,
            cache_capacity_mib: 64,
            target_hit_pct: 90,
            slow_path_sleep_us: 100,
        }
    }
}

impl TaobenchConfig {
    /// Set the closed-loop client thread count (`0` = one per allocated CPU).
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

/// A miss handed from a client to a slow dispatcher.
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
/// fill; the client blocks here until its outstanding miss is served. One
/// outstanding request per client (closed loop), so the slot is reused.
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

/// Per-phase taobench counters. Integer-only so the enclosing
/// `PhaseSlice` keeps `Eq`. `get_cmds` /
/// `get_misses` are request-time; `fast_ops` / `slow_ops` are response-time (see
/// the module docs).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub(crate) struct TaobenchPhaseStats {
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

impl TaobenchPhaseStats {
    pub(crate) fn merge(&mut self, o: &TaobenchPhaseStats) {
        self.get_cmds += o.get_cmds;
        self.get_misses += o.get_misses;
        self.fast_ops += o.fast_ops;
        self.slow_ops += o.slow_ops;
        // The wall window is shared across the pooled concurrent threads/workers
        // (they run the same phase at the same time), so MAX, not sum.
        self.elapsed_ns = self.elapsed_ns.max(o.elapsed_ns);
    }
    /// Completed ops (fast + slow) — the throughput numerator.
    pub fn total_ops(&self) -> u64 {
        self.fast_ops + self.slow_ops
    }
}

/// The engine's return: the whole-run merged stats and per-phase-epoch stats.
pub(crate) struct TaobenchOutcome {
    pub whole_run: TaobenchPhaseStats,
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
    whole: TaobenchPhaseStats,
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
            whole: TaobenchPhaseStats::default(),
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
            cur.elapsed_ns = now.saturating_sub(self.phase_start_ns);
            self.phases.entry(self.cur_epoch).or_default().merge(&cur);
            self.cur_epoch = epoch;
            self.phase_start_ns = now;
        }
    }
    /// Record a lookup at request time in `epoch`.
    fn record_cmd(&mut self, epoch: u32, hit: bool) {
        self.roll_to(epoch);
        self.cur.get_cmds += 1;
        self.whole.get_cmds += 1;
        if !hit {
            self.cur.get_misses += 1;
            self.whole.get_misses += 1;
        }
    }
    /// Record a completion at response time in `epoch`.
    fn record_complete(&mut self, epoch: u32, hit: bool) {
        self.roll_to(epoch);
        if hit {
            self.cur.fast_ops += 1;
            self.whole.fast_ops += 1;
        } else {
            self.cur.slow_ops += 1;
            self.whole.slow_ops += 1;
        }
    }
    /// Flush the last open bucket (stamping its segment wall) and stamp the
    /// whole-run window.
    fn finalize(mut self) -> Self {
        let now = monotonic_nanos();
        let mut cur = std::mem::take(&mut self.cur);
        cur.elapsed_ns = now.saturating_sub(self.phase_start_ns);
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
    let mut whole = TaobenchPhaseStats::default();
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

/// One closed-loop client: pick a key, look it up, serve a hit inline (touch +
/// fast op) or hand a miss to a dispatcher and block until filled (slow op).
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
) -> ThreadAccum {
    let mut rng = Rng::new(0x5EED_0000 ^ (id as u64).wrapping_mul(0x9E37_79B1));
    let mut accum = ThreadAccum::new(read_epoch(phase_epoch));

    while !stop.load(Ordering::Acquire) {
        let epoch_cmd = read_epoch(phase_epoch);
        let key = rng.below(key_range);
        let hit = cache.get_touch(key);
        accum.record_cmd(epoch_cmd, hit);

        if hit {
            // Hit completes instantly; response epoch == request epoch.
            accum.record_complete(epoch_cmd, true);
        } else {
            // Miss: hand to a dispatcher and block until it fills + wakes us.
            // The dispatcher always serves us before `disp_stop` is set (which
            // happens only after every client has joined), so this never blocks
            // past shutdown.
            slow_q.push(SlowReq { key, client: id });
            slot.wait_filled();
            accum.record_complete(read_epoch(phase_epoch), false);
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
    TaobenchStandaloneReport::from_run(&outcome.whole_run, nr_client_threads, nr_slow_threads)
}

/// Summary of a [`run_standalone`] run — the headline taobench metrics in the
/// shape the reference taobench server reports (`fast_qps` / `hit_rate` /
/// `slow_qps`) plus the derived `total_qps` (= fast + slow) and `hit_ratio`
/// (= fast / total).
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
}

impl TaobenchStandaloneReport {
    fn from_run(w: &TaobenchPhaseStats, nr_client_threads: usize, nr_slow_threads: usize) -> Self {
        let secs = (w.elapsed_ns as f64 / 1e9).max(f64::MIN_POSITIVE);
        let total = w.total_ops();
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
            assert!(out.whole_run.elapsed_ns > 0, "the run window was measured");

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
            .slow_path_sleep_us(250);
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
}
