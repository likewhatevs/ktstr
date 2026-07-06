# CI

ktstr boots KVM microVMs and builds Linux kernels, so a CI job has two
unusual needs: runners that expose `/dev/kvm`, and aggressive caching —
the first run on a fresh runner downloads and compiles a full Linux
kernel, by far the slowest step in any workflow. Once the kernel cache
is warm, the kernel resolves in under a second and wall-clock is
dominated by the tests themselves:

<div class="kt-doc-grid">
<div class="kt-doc-card"><strong>Runner</strong><p>Use self-hosted machines with <code>/dev/kvm</code>, enough CPUs, and stable cgroup setup.</p></div>
<div class="kt-doc-card"><strong>Cache</strong><p>Persist kernels, BTF anchors, and build artifacts so CI spends time on tests.</p></div>
<div class="kt-doc-card"><strong>Gate</strong><p>Keep sidecars, run affected suites, and compare metrics with <code>perf-delta</code>.</p></div>
</div>

Everything below is a variation on: get KVM, cache the kernel, run the
tests, keep the stats. This repo's own CI is the living reference:
[`.github/workflows/ci.yml`](https://github.com/likewhatevs/ktstr/blob/main/.github/workflows/ci.yml).

## Runner requirements

GitHub-hosted `ubuntu-latest` runners do **not** expose `/dev/kvm`.
Use self-hosted runners with project-specific labels (this repo uses
`ktstr-x64` and `ktstr-arm64`; substitute your own pool's labels):

```yaml
runs-on: [ktstr-x64]    # x86_64 self-hosted KVM runner
runs-on: [ktstr-arm64]  # aarch64 self-hosted KVM runner
```

See [Troubleshooting: /dev/kvm not accessible](troubleshooting.md#devkvm-not-accessible)
for diagnosing KVM on runners, including cloud-VM nested
virtualization setup (GCP, AWS, Azure). Runners also need the build
dependencies from [Getting Started](getting-started.md) and at least
5 GB of free disk for kernel sources, build artifacts, and cached
images. Gauntlet topology presets go up to 252 vCPUs; tests whose
preset exceeds the runner's capacity skip cleanly (or fail under
`--no-skip-mode`), so small runners run a subset rather than
breaking.

## A minimal workflow

Builds a kernel, caches it, runs the tests:

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:

jobs:
  test:
    runs-on: [ktstr-x64]
    env:
      KTSTR_GHA_CACHE: "1"
    steps:
      - uses: actions/checkout@v5
      - uses: dtolnay/rust-toolchain@stable
      - uses: taiki-e/install-action@v2
        with:
          tool: cargo-nextest
      - name: Install ktstr
        run: cargo install --path . --locked --features remote-cache
      - name: Cache kernel images
        uses: actions/cache@v4
        with:
          path: ~/.cache/ktstr/kernels
          key: ktstr-kernels-x64-${{ hashFiles('ktstr.kconfig') }}
          restore-keys: ktstr-kernels-x64-
      - name: Build test kernel
        run: cargo ktstr kernel build
      - run: cargo ktstr test -- --profile ci --features integration
```

The load-bearing lines: `KTSTR_GHA_CACHE: "1"` enables a remote
kernel-cache layer on top of the local one ([Caching](#caching)); the
`actions/cache` key hashes `ktstr.kconfig`, so a kconfig change
invalidates cached kernels; `--profile ci` selects the nextest
profile tuned for contended runners
([Nextest CI profile](#nextest-ci-profile)); `--features integration`
enables ktstr's full end-to-end suite when testing ktstr itself — in
a scheduler repo, pass your own crate's feature flags or drop it.
The test harness auto-discovers the built kernel; to pin versions,
use the matrix below.

## Kernel pinning

Pin kernel versions via the matrix strategy (this repo's CI tests
`6.14` and `7.1` this way):

```yaml
strategy:
  fail-fast: false
  matrix:
    kernel-version: ['6.14', '7.1']
# then, in steps:
  - run: cargo ktstr kernel build --kernel ${{ matrix.kernel-version }}
  - run: cargo ktstr test --kernel ${{ matrix.kernel-version }} -- --profile ci --features integration
```

`--kernel` tells `cargo ktstr test` which cached kernel to use at
runtime. A major.minor prefix (e.g. `6.14`) resolves to the highest
patch release in that series; see
[cargo ktstr kernel](running-tests/cargo-ktstr.md) for the full
resolution chain.

The cache-key footgun: when testing multiple kernel versions, add
`${{ matrix.kernel-version }}` to the cache key and restore-keys —
the minimal workflow's version-less key would make matrix cells
evict each other's kernels.

## Caching

`actions/cache` persists `~/.cache/ktstr/kernels` across runs.
`KTSTR_GHA_CACHE=1` adds a remote layer that shares kernels across
jobs and workflow runs; remote failures are non-fatal and the local
cache is authoritative. The remote layer is compiled in only with
`--features remote-cache` (off by default) — without it the variable
is a no-op, which is why the install steps above pass the feature.

If you set a global `RUSTC_WRAPPER: sccache` for compile caching (as
this repo's CI does), `sccache` must be on `$PATH` on every targeted
runner — x64 and arm64 alike — or the first cargo invocation fails.

## Dynamic matrix: `cargo ktstr affected`

On a fleet repo with many schedulers, `cargo ktstr affected` emits
the scheduler packages a `base..HEAD` diff touches, as a flat JSON
array for a GitHub Actions dynamic matrix — one job per affected
scheduler instead of building and testing everything on every push:

```sh
cargo ktstr affected                    # vs merge-base(HEAD, main)
# -> e.g. ["scx_lavd","scx_rusty"]
```

Attribution is the union of the cargo dependency closure (shared
Rust library changes) and per-scheduler dep-info parsing of the
compiled BPF sources (shared `.bpf.c` / header includes). The design
is **fail-safe**: a false negative — silently skipping an affected
scheduler — is the worst outcome, so every uncertainty (unresolvable
base, diff failure, build-graph or `Cargo.lock` change, unattributable
non-docs path) widens to the full testable set, never to a skip.
Only a strictly docs-only change (or `base == HEAD`) emits `[]`.

Only `Discover` (cargo-package) schedulers appear in the array —
package-less schedulers (EEVDF, kernel-builtin) have no package to
key a matrix cell on and need a separate unconditional CI leg. On a
`pull_request` event the baseline defaults to
`merge-base(HEAD, origin/$GITHUB_BASE_REF)`; check out with full
history so the merge-base exists.

```yaml
jobs:
  matrix:
    runs-on: [ktstr-x64]
    outputs:
      schedulers: ${{ steps.affected.outputs.schedulers }}
    steps:
      - uses: actions/checkout@v5
        with:
          fetch-depth: 0          # merge-base needs history
      - name: Install ktstr
        run: cargo install ktstr --locked
      - id: affected
        run: echo "schedulers=$(cargo ktstr affected)" >> "$GITHUB_OUTPUT"

  test:
    needs: matrix
    if: needs.matrix.outputs.schedulers != '[]'
    runs-on: [ktstr-x64]
    strategy:
      fail-fast: false
      matrix:
        scheduler: ${{ fromJSON(needs.matrix.outputs.schedulers) }}
    steps:
      - uses: actions/checkout@v5
      # ... install ktstr + nextest, restore the kernel cache, and
      #     `cargo ktstr kernel build` as in the minimal workflow ...
      # Adjust the filter to how your repo organizes per-scheduler tests.
      - run: cargo ktstr test -- --profile ci -E 'package(${{ matrix.scheduler }})'
```

The local counterpart is `cargo ktstr test --relevant`, which runs
the same attribution against your working tree — see
[cargo-ktstr](running-tests/cargo-ktstr.md).

## Perf gate on pull requests

`cargo ktstr perf-delta --noise-adjust` runs the `performance_mode`
tests at HEAD and at the PR's merge-base, then exits non-zero when
metrics regress with statistical confidence — a performance gate in
one step. On `pull_request` events the baseline resolves from
`$GITHUB_BASE_REF` automatically:

```yaml
perf-gate:
  if: github.event_name == 'pull_request'
  runs-on: [ktstr-x64]
  steps:
    - uses: actions/checkout@v5
      with:
        fetch-depth: 0          # merge-base needs history
    # ... install ktstr + nextest as in the minimal workflow ...
    - run: cargo ktstr perf-delta --noise-adjust 5 --kernel 7.0
```

Budget for it: `--noise-adjust 5` runs every `performance_mode` test
ten times (five per side). Narrow with `-E` or `--relevant`, and add
`--must-fail <metric>` for metrics that must never regress. See
[Runs and Regression Gates](running-tests/runs.md) for how the
verdict is computed and
[A/B Compare Branches](recipes/ab-compare.md) for the local
equivalent.

## Budget-based test selection

Set `KTSTR_BUDGET_SECS` (e.g. `"300"`) on the test step to bound a
smoke-test job: the selector greedily picks the tests that maximize
feature coverage within the time budget. See
[Running Tests](running-tests.md) for the selection model.

## Coverage

Same job shape as the minimal workflow, with the
`llvm-tools-preview` rustup component and `cargo-llvm-cov` added, and
the test step swapped for:

```yaml
- run: cargo ktstr coverage -- --profile ci --lcov --output-path lcov.info --features integration --exclude-from-report scx-ktstr
```

`--exclude-from-report <crate>` keeps scheduler crates out of the
coverage report — the example excludes `scx-ktstr`, ktstr's own
fixture scheduler.

## Test statistics

```yaml
- name: Test statistics
  if: ${{ !cancelled() }}
  run: cargo ktstr stats
```

`stats` reads the sidecar JSON files under `target/ktstr/` and prints
gauntlet analysis, BPF verifier stats, callback profile, and KVM
stats ([Runs and Regression Gates](running-tests/runs.md)).
`if: !cancelled()` collects stats even when the test step failed —
which is exactly when you want them.

## aarch64

aarch64 runners use the same workflows with two substitutions: runner
labels (`[ktstr-arm64]` or your pool's) and the cache-key prefix
(`arm64` instead of `x64`). The guest image name differs (`Image`
instead of `bzImage`) but ktstr handles that internally.

## Performance mode

CI runners often lack `CAP_SYS_NICE`, rtprio limits, or enough host
CPUs for exclusive LLC reservation. Set `KTSTR_NO_PERF_MODE: "1"` on
the test step to disable performance mode; tests with
`performance_mode=true` are then skipped entirely. See
[Performance Mode](concepts/performance-mode.md), and
[Tests pass locally but fail in CI](troubleshooting.md#tests-pass-locally-but-fail-in-ci)
for the wider skip/fail triage.

## Nextest CI profile

The workspace ships a `ci` profile in `.config/nextest.toml`. VM
boots on a contended runner run slower and flake differently than on
a dev box, so the CI profile trades latency for stability — longer
slow-timeouts, one more retry, deferred failure output, and no
fail-fast:

```toml
[profile.ci]
slow-timeout = { period = "90s", terminate-after = 3 }
retries = { backoff = "exponential", count = 6, delay = "1s", jitter = true, max-delay = "3s" }
failure-output = "final"
fail-fast = false

# Heavier test classes get their own budgets, e.g.:
[[profile.ci.overrides]]
filter = "test(verifier_)"
slow-timeout = { period = "180s", terminate-after = 3 }
```

Use it with `--profile ci`. If a test in your repo drives an
unusually slow boot (huge topology, nested VM), give it its own
override rather than raising the profile-wide timeout.

The CI-relevant environment variables are `KTSTR_GHA_CACHE`,
`KTSTR_BUDGET_SECS`, `KTSTR_NO_PERF_MODE`, `KTSTR_KERNEL`, `KTSTR_CI`
(tags sidecars as CI-produced), and `KTSTR_CACHE_DIR` — see the
[full reference](reference/environment-variables.md).
