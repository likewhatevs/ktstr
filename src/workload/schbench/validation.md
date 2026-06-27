# schbench port fidelity validation

This file is committed empirical evidence that ktstr's native schbench port
(`WorkType::Schbench` / the `ktstr-schbench-validate` driver) reproduces the
behavior of the reference [schbench](https://github.com/masoncl/schbench) across
its full flag surface. The numbers below are a captured run. To reproduce: build
the reference (`gcc`/`clang -O2 -march=x86-64-v3` on upstream `schbench.c`) and
the ktstr driver (`cargo build --release --features integration`), run each with
the per-axis flags shown three times, and average. There is no committed
one-command harness; the deliverable is this captured comparison, not a runner.

## Method

Three implementations of the **same** workload are compared, all built for the
same ISA (`x86-64-v3`) so the comparison isolates the port (Rust re-expression
vs the C original) from compiler-target codegen:

| implementation | build |
|---|---|
| `schbench-gcc` | `gcc -O2 -march=x86-64-v3` (upstream `schbench.c`) |
| `schbench-clang` | `clang -O2 -march=x86-64-v3` (same `schbench.c`) |
| `ktstr` | `ktstr-schbench-validate`, `cargo build --release` (`.cargo/config.toml` pins `target-cpu=x86-64-v3`) |

For each axis, every implementation is run **3 times** with identical flags; we
report each metric's **avg-of-3** and the run-to-run **spread** `[min–max]`. The
metrics are the wakeup/request latency percentiles and the RPS/throughput that
schbench's `show_latencies` prints, which `ktstr-schbench-validate` mirrors
byte-for-byte in shape.

### Acceptance: within the gcc↔clang envelope

The pass criterion is **ktstr's avg-of-3 falls inside the gcc↔clang envelope**
— the range the *reference workload itself* spans when only the compiler
changes. This is the honest bar: the same `schbench.c` compiled by gcc vs clang
differs by ~3–6% on most axes (e.g. default avg-rps gcc ~467 vs clang ~494), so
requiring ktstr to match a *single* compiler more tightly than schbench matches
itself across compilers would over-constrain. ktstr is faithful when it behaves
like a third compilation of the same workload.

### Optimization-defeat check: disassembly, not perf

The intended way to confirm the matrix work is real (not optimized away) is
`perf stat` IPC. **perf is unreliable on this host** — repeated runs gave
implausible, run-to-run-unstable instruction/cycle counts (and several events
report `<not supported>`), so perf IPC is *not* used as evidence here. The
optimization-defeat check is instead grounded in **disassembly** (objdump of the
three release binaries' inner loops), which is static and exact. The observations
below are from that disassembly run (the dumps themselves are not committed):

- All three implementations' matrix multiply compiles to a **scalar `imul`
  multiply-accumulate** inner loop. There is **no SIMD `u64` multiply** (none
  exists below AVX-512 `vpmullq`, which `x86-64-v3` excludes), and the serial
  accumulator reduction blocks SLP — confirmed in ktstr, gcc-schbench, and
  clang-schbench. ktstr's `matrix_multiply` keeps the work live under `-O2`/v3
  with one entry `black_box` + a `write_volatile` C store (a per-load barrier
  would instead block the fused memory-operand `imul` and unrolling, the ~1.3x
  de-opt found earlier — not SIMD).
- For `--split`, schbench's shared `do_some_math` keeps the running sum in a
  register but **stores it to the shared C cell every k** (`do_some_math` reads
  `m1`/`m2`/`m3` as offsets into one base pointer, so neither gcc nor clang can
  prove the `m3` store doesn't alias the next iteration's `m1`/`m2` loads — one
  `do_some_math`, three call sites). That per-k store *is* `--split`'s cross-core
  cache contention. ktstr's `matrix_multiply_shared` reproduces it exactly
  (register accumulator + per-k atomic store to the shared C cell, sound via
  `AtomicU64` `Relaxed`, which lowers to plain `MOV` on x86-64).

## Results

One table per axis. Each cell is the avg-of-3 with the `[min–max]` run spread; the
verdict is `yes` if ktstr's avg-of-3 lies within `[min(gcc,clang), max(gcc,clang)]`,
else `NO (±x%)` where x is the signed distance from the nearer gcc/clang avg edge
(`(ktstr − nearest)/nearest`). `wakeup p99` is reported as `noise` — see [1].

### Default (no flags beyond topology)

`-m 1 -t 2 -r 10` (n=3 per impl; lower=better except rps)

| metric | schbench-gcc | schbench-clang | ktstr | ktstr in gcc↔clang envelope |
|---|---|---|---|---|
| wakeup p99 (us) | 8.0 [7–9] | 8.0 [7–10] | 7.0 [7–7] | noise [1] |
| request p50 (us) | 4274.7 [4264–4280] | 4025.3 [4020–4028] | 4157.3 [4120–4184] | yes |
| request p99 (us) | 4584.0 [4584–4584] | 4189.3 [4168–4216] | 4456.0 [4408–4488] | yes |
| rps p50 | 465.0 [465–465] | 493.3 [492–494] | 477.7 [475–481] | yes |
| avg rps | 466.6 [466.2–466.9] | 494.3 [493.3–494.8] | 479.3 [475.7–483.3] | yes |

### `-m` message threads = 2

`-m 2 -t 2 -r 10` (n=3 per impl; lower=better except rps)

| metric | schbench-gcc | schbench-clang | ktstr | ktstr in gcc↔clang envelope |
|---|---|---|---|---|
| wakeup p99 (us) | 8.3 [8–9] | 8.0 [8–8] | 7.7 [7–8] | noise [1] |
| request p50 (us) | 4264.0 [4264–4264] | 4028.0 [4012–4036] | 4157.3 [4152–4168] | yes |
| request p99 (us) | 4589.3 [4584–4600] | 4200.0 [4152–4232] | 4498.7 [4472–4520] | yes |
| rps p50 | 935.0 [933–937] | 986.3 [983–991] | 959.7 [957–961] | yes |
| avg rps | 934.9 [933.3–936.6] | 987.1 [984.4–992.1] | 958.9 [957.2–961.1] | yes |

### `-m` message threads = 4

`-m 4 -t 2 -r 10` (n=3 per impl; lower=better except rps)

| metric | schbench-gcc | schbench-clang | ktstr | ktstr in gcc↔clang envelope |
|---|---|---|---|---|
| wakeup p99 (us) | 8.3 [7–9] | 7.3 [7–8] | 7.3 [7–8] | noise [1] |
| request p50 (us) | 4274.7 [4264–4280] | 4014.7 [4012–4020] | 4157.3 [4152–4168] | yes |
| request p99 (us) | 4610.7 [4600–4616] | 4162.7 [4136–4216] | 4456.0 [4456–4456] | yes |
| rps p50 | 1867.3 [1866–1870] | 1983.3 [1978–1986] | 1920.7 [1918–1922] | yes |
| avg rps | 1866.0 [1863.2–1868.8] | 1983.0 [1978.8–1985.2] | 1921.5 [1918.2–1923.4] | yes |
