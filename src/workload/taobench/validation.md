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

The same access pattern is realized two ways and compared for shape — the ktstr
port measured host-side (the numbers in ## Measured), the reference taobench against
its documented behavior (its harness's reported metrics + the documented 0.88–0.90
hit-ratio band; a single-host server + client build exercises the same access pattern):

| implementation | how |
|---|---|
| reference taobench | built from the reference's upstream source (its cache server + load-generator client); server + client run on disjoint cores over loopback with TLS disabled |
| ktstr port | `ktstr-taobench-validate` (the native engine via `run_standalone`), run host-side |

The reference server prints `fast_qps` / `hit_rate` / `slow_qps` every
stats-interval; the parser derives `total_qps = fast + slow` and
`hit_ratio = fast/total`. The port's driver prints the same fields computed by the
same formulas. Both are sized for a ~0.9 target hit ratio.

## Per-aspect comparison

Each row: the aspect, the reference taobench behavior, the port's behavior, and the
verdict (match, approximate, or intentional divergence).

| aspect | reference taobench | ktstr port | verdict |
|---|---|---|---|
| request mix | GET-dominated; SET only as the miss-fill the server tells the client to write (self-heal) — the GET-only warmup fills the cache via the same self-heal path | GET-only lookups; a miss inserts (fills) the key on the slow path — the same self-heal | match |
| key distribution | ~uniform random over a key id range (not zipfian); hit ratio set by RANGE size vs cache capacity | uniform random over `[0, key_range)`; `key_range = capacity_objects / target_hit` | match |
| hit ratio | ~0.88–0.90, an emergent property of a CAPACITY-BOUNDED LRU cache whose working set exceeds the cache's EFFECTIVE object capacity (eviction ↔ refill equilibrium; the reference tunes its key range with a correction factor so the steady state lands on the target) | settles to `target_hit_pct` via a bounded FIFO-evicting sharded cache with `key_range > capacity` (eviction ↔ refill equilibrium); pinned by the in-process test (`0.80..=0.97`, not drifting to 1.0) | match (mechanism; FIFO ≈ LRU for uniform access) |
| value sizes | small-object-heavy size profile (the reference's documented average item size is ≈ 434 B); the server clips every served item to a 64 KiB maximum, so served bytes cap at 64 KiB | representative small-heavy long-tail distribution (mean ≈ 332 B, tail to 64 KiB); bytes are allocated per the distribution and TOUCHED on serve (the cache memory-bandwidth cost) | approximate (own distribution, same small-object-heavy shape + 64 KiB ceiling + same touch) |
| fast / slow tiers | fast worker thread serves a hit (`fast_qps`); a miss is enqueued to a slow dispatcher pool that simulates a backing-store fetch (sleep) then SET-fills (`slow_qps`) | a client thread serves its own hit inline (fast path); a miss is handed to a slow dispatcher thread that sleeps the backing-store fetch then fills + wakes the client | match (slow tier present; client+fast merged — see divergences) |
| slow-path service time | a FIXED per-request `usleep(tao_slow_path_sleep_us)` — no jitter, no tail. (The `[target/2, 2·target]` uniform jitter + idle-poll backoff of "smart" mode is the reference's dispatcher/slow-thread/worker POLL & idle-wait sleep, a separate knob, not the per-request fetch.) | fixed `slow_path_sleep_us` by default; `slow_path_p99_us > slow_path_sleep_us` switches to a per-fetch heavy-tailed Pareto with median `slow_path_sleep_us` and 99th percentile `slow_path_p99_us` (clamped at a 2s max) | enhancement (the port adds a true heavy-tailed per-request service-time distribution; the reference's per-request service time is fixed) |
| headline metrics | `fast_qps`, `slow_qps`, `total_qps = fast+slow`, `hit_ratio = fast/total`; per-interval `hit_rate = 1 - get_misses/get_cmds` | same five, same formulas (`write_taobench_scalars`); request-time `get_cmds`/`get_misses` vs response-time `fast_ops`/`slow_ops` split preserved | match |
| arrival model | closed-loop, fixed-connection-count load generator (fixed `--clients` / `--threads`, no rate target; each connection issues its next request on the prior's completion); its harness reports throughput + hit ratio, not request-latency percentiles | closed-loop by default (`arrival_rate == 0`); `arrival_rate > 0` switches to OPEN-loop fixed-rate arrival and measures serve latency from the intended arrival (coordinated-omission) | match (both closed-loop); port adds an open-loop CO mode (enhancement) |
| serve latency | the reference's harness surfaces throughput + hit ratio only, no request-latency percentiles | open-loop coordinated-omission serve latency, per-phase `taobench_serve_{p50,p90,p99,p999,min,max}_us` + whole-run `taobench_serve_*_us_whole`, measured from intended arrival via schbench's `PlatStats` histogram | enhancement (the port adds an open-loop CO serve-latency the reference harness does not report) |

## Measured

Both sized for a ~0.9 target hit ratio. The comparison is **shape** (steady-state
hit ratio + fast≫slow composition), not absolute magnitude.

**Reference taobench.** The reference's headline result is a many-core,
multi-process aggregate: its harness scales to multiple server instances across the
host's cores and reports per-interval `fast_qps` / `slow_qps` / `total_qps` and a
`hit_rate`, with a documented steady-state hit-ratio band of **0.88–0.90**. It does
not surface request-latency percentiles, and its absolute qps is environment-bound
(a many-core, multi-process aggregate) — so magnitude is not a head-to-head target.
Only the shape is compared: a steady-state hit ratio in the 0.88–0.90 band reached
by eviction↔refill, a fast≫slow composition, and the same `total_qps` / `hit_ratio`
formulas.

**ktstr port.** `ktstr-taobench-validate` host-side — a single in-process engine.
Reproduce with:

```
cargo run --release --bin ktstr-taobench-validate \
  --features cli-bins,integration -- -m 128 -c 16 -H 90 -r 15
```

(128 MiB cache, `target_hit_pct 90`, 16 client + 5 slow threads, closed loop; one
representative run, 14.91 s measured window):

| metric | value |
|---|---|
| hit_ratio (fast/total) | **0.8999** |
| hit_rate (1 − misses/cmds) | 0.8999 |
| fast_qps | ~292,100 |
| slow_qps | ~32,500 |
| total_qps (fast + slow) | ~324,600 |
| fast : slow split | ~9 : 1 |

The port settles to **0.8999** — its 0.90 target — by the same eviction↔refill
equilibrium that holds the reference inside its 0.88–0.90 band. (hit_ratio ==
hit_rate here because a non-phased whole-run run drains all in-flight requests, so
response-time and command-time counts coincide; the two diverge only across a
phase boundary, the intended non-tautology.) Absolute qps is a single in-process
host run on one machine — environment-bound, not comparable to the reference's
many-core aggregate, and not a pass/fail axis.

**Shape match:** the port settles to 0.8999 (its 0.90 target) with a ~9:1
fast:slow split (fast:slow = hit:miss at the target ratio), inside the reference's
documented 0.88–0.90 band. Both reach the steady-state hit ratio as an emergent
eviction↔refill property and compute `total_qps` / `hit_ratio` by the same
formulas — the structural fidelity claim. Absolute qps differs by environment
(many-core multi-process reference vs a single in-process engine) and is not a
comparison axis.

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
  value-size table — the port uses its own small-heavy long-tail histogram (mean
  ≈ 332 B, 64 KiB ceiling) approximating the reference's small-object-heavy profile
  (documented average ≈ 434 B, served items clipped at 64 KiB), so no external data
  table is carried.
- **Serve-path latency is coordinated-omission-corrected, open-loop only.** Under
  open-loop arrival (`arrival_rate > 0`) the port measures serve latency from the
  INTENDED arrival time (so a backlog from slow service inflates the late requests'
  latency instead of being omitted) into schbench's `PlatStats` histogram, exposed
  as the per-phase `taobench_serve_*_us` and whole-run `taobench_serve_*_us_whole`
  percentiles. It is CO-corrected (measured from intended arrival, not issue time)
  and recorded only in open loop — the default closed loop has no intended-arrival
  schedule, so the serve-latency keys read absent there. The reference harness
  surfaces no request-latency percentile, so this is a port-only metric, not a
  head-to-head latency axis.
- **Heavy-tailed slow-path service time (port enhancement).** The reference's
  per-request slow-path service time is a FIXED `usleep(tao_slow_path_sleep_us)` —
  no jitter, no tail. (Its `[target/2, 2×target]` per-thread-cached uniform jitter +
  idle-poll exponential backoff, "smart" mode's `my_nanosleep`, governs the
  reference's dispatcher/slow-thread/worker POLL & idle-wait sleeps — a separate
  knob, not the per-request fetch.) The port adds a true per-fetch heavy-tailed
  service time: with `slow_path_p99_us > slow_path_sleep_us`, each miss's
  backing-store fetch is a Pareto draw (median `slow_path_sleep_us`, 99th percentile
  `slow_path_p99_us`, clamped at a 2 s maximum), so the open-loop serve-latency tail
  reflects realistic backing-store variance the reference's fixed service time
  cannot. `slow_path_p99_us == 0` (default), `<= slow_path_sleep_us`, or
  `slow_path_sleep_us == 0` keeps the fixed-latency legacy behavior.
- **Per-phase metric surface.** The port's qps/hit_ratio are `MetricKind::PerPhase`
  (surfaced via the per-phase `PhaseBucket` / `VmResult::phase_metric` path, like
  schbench's latency metrics), not the run-level cross-run scalar fold.
- **Magnitude is not comparable.** The reference's numbers are a many-core,
  multi-process run; the port is a single in-process engine. Only the shape
  (composition + hit ratio) is compared.
