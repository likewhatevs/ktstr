# taobench port fidelity validation

This file is committed evidence that ktstr's native taobench port
(`WorkType::Taobench` / the `ktstr-taobench-validate` driver) reproduces the
ACCESS PATTERN of the reference taobench — a GET-dominated key-value object cache
with a fast in-cache hit path and a slow backing-store-miss path, driven to a
steady-state hit ratio.

Unlike the schbench validation (a per-flag-axis numeric envelope vs gcc/clang
builds of the same source), this is a **structural** comparison. The reference
taobench's published throughput is a many-core, many-instance figure (its
documented hit-ratio band is ~0.88–0.90), not head-to-head with a single host-side
engine run or an in-VM ktstr guest. The port is faithful when it reproduces the **shape** — GET-dominated
access, self-healing SET-on-miss, a fast≫slow split, a steady-state hit ratio
engineered by key-range-vs-capacity, and small-heavy long-tail value sizes — and
computes the same headline metrics (`total_qps = fast + slow`,
`hit_ratio = fast/(fast+slow)`) by the same formulas. Magnitude (absolute qps) is
environment-bound and is reported as context, not as a pass/fail target.

## Method

Two implementations of the same access pattern are run single-host and compared
for shape:

| implementation | how |
|---|---|
| reference taobench | built from the reference's upstream source (its cache server + load-generator client); server + client run on disjoint cores over loopback with TLS disabled |
| ktstr port | `ktstr-taobench-validate` (the native engine via `run_standalone`), run host-side |

The reference server prints `fast_qps` / `hit_rate` / `slow_qps` every
stats-interval; the parser derives `total_qps = fast + slow` and
`hit_ratio = fast/total`. The port's driver prints the same fields computed by the
same formulas. Both are sized for a ~0.9 target hit ratio.

## Per-aspect comparison

Each aspect: the reference taobench behavior, the port's behavior, and whether the
port matches, approximates, or intentionally diverges.

| aspect | reference taobench | ktstr port | verdict |
|---|---|---|---|
| request mix | GET-dominated; SET only as the miss-fill the server tells the client to write (self-heal), plus warmup populate | GET-only lookups; a miss inserts (fills) the key on the slow path — the same self-heal | match |
| key distribution | ~uniform random over a key id range (not zipfian); hit ratio set by RANGE size vs cache capacity | uniform random over `[0, key_range)`; `key_range = capacity_objects / target_hit` | match |
| hit ratio | ~0.88–0.90, an emergent property of a CAPACITY-BOUNDED LRU cache whose key range exceeds capacity (eviction ↔ refill equilibrium) | settles to `target_hit_pct` via a bounded FIFO-evicting sharded cache with `key_range > capacity` (eviction ↔ refill equilibrium); pinned by the in-process test (`0.80..=0.97`, not drifting to 1.0) | match (mechanism; FIFO ≈ LRU for uniform access) |
| value sizes | the reference's empirical value-size histogram (mean ≈ 341 B, long tail to ~205 KB); bytes are stored + served | representative small-heavy long-tail distribution (mean ≈ 332 B, tail to 64 KiB); bytes are allocated per the distribution and TOUCHED on serve (the cache memory-bandwidth cost) | approximate (own distribution, same shape + same touch) |
| fast / slow tiers | fast worker thread serves a hit (`fast_qps`); a miss is enqueued to a slow dispatcher pool that simulates a backing-store fetch (sleep) then SET-fills (`slow_qps`) | a client thread serves its own hit inline (fast path); a miss is handed to a slow dispatcher thread that sleeps `slow_path_sleep_us` then fills + wakes the client | match (slow tier present; client+fast merged — see divergences) |
| headline metrics | `fast_qps`, `slow_qps`, `total_qps = fast+slow`, `hit_ratio = fast/total`; per-interval `hit_rate = 1 - get_misses/get_cmds` | same five, same formulas (`write_taobench_scalars`); request-time `get_cmds`/`get_misses` vs response-time `fast_ops`/`slow_ops` split preserved | match |

## Measured

Both sized for a ~0.9 target hit ratio. The comparison is **shape** (steady-state
hit ratio + fast≫slow composition), not absolute magnitude.

**Reference taobench** — 1 GiB cache, 16-core single-host, steady state (the
server's own fast/hit/slow stats):

| metric | value |
|---|---|
| hit_rate (1 − misses/cmds) | **0.889** |
| fast_qps | ~485,000 |
| slow_qps | ~60,000 |
| total_qps (fast + slow) | ~545,000 |
| fast : slow split | ~8 : 1 |
| client-side throughput / tail | 320k GET/s, p50 7.5 ms, p99 54 ms, p99.9 113 ms |

The reference's hit_rate (0.889) lands squarely in its documented 0.88–0.90 band —
a faithful real run.

**ktstr port** — `ktstr-taobench-validate` host-side, 128 MiB cache,
`target_hit_pct 90`, 16 client + 5 slow threads, 14.6 s window:

| metric | value |
|---|---|
| hit_ratio (fast/total) | **0.8999** |
| hit_rate (1 − misses/cmds) | 0.8999 |
| fast_qps | ~290,700 |
| slow_qps | ~32,300 |
| total_qps (fast + slow) | ~323,000 |
| fast : slow split | ~9 : 1 |

The port settles to **0.8999** — exactly its 0.90 target — by the same
eviction↔refill equilibrium the reference reaches at 0.889. (hit_ratio == hit_rate
here because a non-phased whole-run run drains all in-flight requests, so
response-time and command-time counts coincide; the two diverge only across a
phase boundary, the intended non-tautology.) Absolute qps is a single in-process
host run.

**Shape match:** the reference settles to hit_rate 0.889 (≈ its 0.9 target) with a
~8:1 fast:slow split; the port settles to 0.8999 (its 0.90 target) with a ~9:1
split (fast:slow = hit:miss at the target ratio). Both reach the steady-state hit ratio
as an emergent eviction↔refill property, and both compute `total_qps`/`hit_ratio`
by the same formulas — the structural fidelity claim. Absolute qps differs by
environment (16-core multi-process reference vs a single in-process engine) and is
not a comparison axis.

## Documented divergences

The port models the access pattern and the CPU / memory / scheduling
characteristic, not the wire protocol. Deliberate divergences from the reference:

- **No sockets / TLS.** The reference is a cache server + a load-generator
  client over a real loopback socket; the port is entirely in-process (client →
  fast → slow via condvar/queue handoffs). The scheduler-relevant load is the
  thread wakeups + the slow-path off-CPU + the cache lock contention, which the
  in-process model preserves; the socket I/O layer is dropped (as is TLS).
- **Client + fast tier merged.** The reference has separate client connection
  threads and server fast worker threads; the port's client thread serves its own
  hit inline. The slow dispatcher tier is kept separate (the miss handoff + wakeup
  is preserved).
- **No OOM path.** The reference reports `slow_qps_oom` and OOM lines under memory
  pressure; the port's bounded-evicting cache models steady state only, so
  `slow_qps_oom` is always 0 (not ported).
- **FIFO vs LRU eviction.** The reference evicts LRU; the port evicts FIFO. For a
  uniform key stream the equilibrium hit ratio is `cap/key_range` under either
  policy, so the steady-state hit ratio matches.
- **Value-size distribution is an approximation,** not a copy of the reference's
  value-size table — the port uses its own small-heavy long-tail histogram of the
  same shape (mean ≈ 332 B vs ≈ 341 B), so no external data table is carried.
- **Serve-path latency not measured.** The reference reports per-request p99; the
  port reports qps + hit ratio only (per-request latency on the serve path is a
  follow-up — schbench's histogram machinery exists to feed it).
- **Per-phase metric surface.** The port's qps/hit_ratio are `MetricKind::PerPhase`
  (surfaced via the per-phase `PhaseBucket` / `VmResult::phase_metric` path, like
  schbench's latency metrics), not the run-level cross-run scalar fold.
- **Magnitude is not comparable.** The reference's numbers are a many-core,
  multi-process run; the port is a single in-process engine. Only the shape
  (composition + hit ratio) is compared.
