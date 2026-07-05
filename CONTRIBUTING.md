# Contributing to ktstr

Notes for contributors modifying the workspace or its build
configuration. Day-to-day test authoring does not need any of
this.

## Dev workflow

Install [just](https://github.com/casey/just) (`cargo install just`).
All dev and CI commands are defined in the `justfile` — run
`just --list` to see every recipe. CI runs the same recipes, so a
green local run of a recipe is the same check CI applies.

### Pre-PR checks (no kernel required)

Run these locally before opening a PR — they need no kernel or KVM and
mirror the `lint`, `compile-fail`, `docs-link-check`, and
`devdep-isolation` GitHub Actions jobs that run on every pull request
(and on pushes to `main`):

```
just lint              # fmt --check, cargo check + clippy (both feature
                       #   sets), rustdoc-warnings-as-errors
just compile-fail      # trybuild diagnostic-snapshot fixtures
just link-check        # guide build + mdbook test + lychee link walk
just devdep-isolation  # keep ktstr out of a downstream release binary
```

`just compile-fail` shells `cargo nextest run` and `just
devdep-isolation` shells `rust-script`, so install both first:

```
cargo install --locked cargo-nextest rust-script
```

### Running the test suite (needs KVM)

The `test-x64`, `test-arm64`, `coverage-x64`, and `coverage-arm64` jobs
boot the integration suite in VMs on self-hosted KVM runners, so they
are not part of the pre-PR loop. To run them locally you need KVM access
and `cargo-nextest`. Build a test kernel, then run the suite against it
— the same two steps CI takes:

```
just kernel-build 6.14   # CI's matrix pins 6.14 and 7.1
just test 6.14           # boot the integration suite in VMs
```

`just test <kernel>` wraps `cargo run --bin cargo-ktstr -- ktstr test
--kernel <kernel>`, running the suite under nextest's `ci` profile with
the `integration` feature. A trailing feature (e.g. `just test 6.14
wprof`) is passed to both the `cargo-ktstr` build and the inner test
feature list. `just coverage <kernel>` runs the same suite under
`cargo-llvm-cov` and writes `lcov.info`.

### Git hooks

Optional local hooks live in `.githooks/`; enable them with:

```
git config core.hooksPath .githooks
```

`pre-commit` runs `cargo fmt` + clippy; `pre-push` runs `cargo build
--tests` against the worktree, catching a commit that compiles the
library but breaks the test build before it reaches shared history.
Skip either with `--no-verify` when you have a deliberate WIP need.

### Doc tooling

Needed only when you change `doc/guide/src/` and want to validate
locally: `mdbook` for `just docs` (runs `mdbook build && mdbook test`);
`mdbook-linkcheck2` + `lychee` for `just link-check` (runs `mdbook
build`, then `mdbook test`, then `lychee` against the rendered HTML).
`just book-serve` renders the guide and opens it in a browser for a live
preview.

```
cargo install mdbook mdbook-linkcheck2 lychee --locked
```

CI installs these automatically; install them locally only if you plan
to validate guide changes before pushing.

## Compile-fail tests (trybuild)

Fixtures under `tests/compile_fail/` pin two kinds of compile-time
diagnostics: (1) the proc-macro diagnostics that `#[ktstr_test]`,
`declare_scheduler!`, and `#[derive(Payload)]` emit (the
`ktstr_test_*`, `declare_scheduler_*`, and `derive_payload_*`
fixtures), and (2) the trait-bound errors from the sealed metric-type
traits `Summable` / `Maxable` / `Modeable` / `Rangeable` in
`src/metric_types.rs` — e.g. a generic site bound on `T: Maxable`
refusing `Bytes` (the `metric_types_*` fixtures). Without them an
upstream `syn` / `proc-macro2` bump, or a change to a trait's
`#[diagnostic::on_unimplemented]` message, can silently degrade an
error message that a test author would otherwise see at compile time.

The fixtures live in their own `[[test]] name = "compile_fail"`
target. The test driver function in `tests/compile_fail.rs` carries
`#[ignore]` so `cargo nextest run` skips it by default; `just
compile-fail` runs it via `--run-ignored all`. Each fixture is its
own `cargo build` invocation; trybuild iterates them sequentially
inside the driver. The `compile-fail` nextest test-group in
`.config/nextest.toml` pins `max-threads = 1` for the matched
filter so the driver doesn't share a runner slot with neighbour
tests that also spawn cargo invocations (or otherwise mutate
`target/`) — concurrent cargo runs across tests can leave stale
intermediate artifacts that let a fixture compile cleanly when it
should fail. The test-group addresses that cross-test contention;
within the driver, trybuild's fixture loop is already serial. CI
runs `just compile-fail` as a dedicated job on every pull-request
(and on pushes to `main`), so a new fixture is picked up automatically.

When you change a diagnostic intentionally, regenerate every
fixture's `.stderr` snapshot with:

```
TRYBUILD=overwrite just compile-fail
```

Only run this when the diagnostic change is intentional. If a
fixture fails unexpectedly, the test is telling you a recent
change degraded the error message — revert the change rather than
overwriting the snapshot. Inspect the regenerated `.stderr` files
before committing; the snapshot is what tells the test author what
message they will see, so it should read cleanly.

## Doc link validation

`just link-check` runs `mdbook build doc/guide`, then `mdbook
test doc/guide`, then `lychee --offline doc/guide/book/html` to
walk every rendered HTML file and verify each internal link +
`#fragment` resolves. The `--offline` flag skips external HTTP
fetches so the check is deterministic and
not subject to network flakes. CI runs the same recipe via the
`docs-link-check` job on every pull-request (and on pushes to `main`).

When lychee fails on a broken link, the report cites the path of
the rendered HTML file (`doc/guide/book/html/<page>.html`). The
source for that page is `doc/guide/src/<page>.md`. Locate the
broken link target in the source markdown and either correct the
link or rename the target heading. Run `just link-check` locally
to verify before pushing.

`mdbook-linkcheck2` (an output backend configured in
`doc/guide/book.toml` via `[output.linkcheck2]`) catches
pre-render link errors at `mdbook build` time. `lychee` runs
against the rendered HTML and
catches the post-render class — typo'd `#fragment` refs against
heading IDs that mdbook's slug-generation pipeline produces, slug
collisions, and other anchors that the source-level check
cannot see.

## Release profile — `panic = "abort"`

The release profile sets `panic = "abort"` (`Cargo.toml`,
`[profile.release]`). Any panic on any thread tears down the
entire process without unwinding: `Drop` impls do not run,
`std::panic::catch_unwind` cannot observe the failure, and
`libc::abort` delivers SIGABRT before the kernel returns
control.

Write panic-free code on every thread that runs in the release
profile — especially the monitor loop, KVM vCPU threads, and
anything spawned from `WorkloadHandle`. Relying on
`catch_unwind` as a soft failure boundary is a bug; introduce
explicit `Result` plumbing instead. The only escape hatch is
the vCPU panic-hook shim (`install_once` in
`src/vmm/vcpu_panic.rs`), which runs synchronously on the
panicking thread before `libc::abort` to
flip kill/exited signalling atomics; it does not recover, only
classifies.

Tests run under the default `panic = "unwind"` profile, so
`catch_unwind` works as expected inside `#[test]` bodies — but
code paths that only execute under the release profile cannot
be tested for unwind-safety directly.

## liblzma build configuration

ktstr depends on the `xz2` crate with the `static` feature,
which builds `liblzma` from bundled C source during `cargo
build`. The C compiler and autotools listed in the README (see
the "Ubuntu/Debian" / "Fedora" install blocks) are sufficient
for the static build — no separate `liblzma-dev` / `xz-devel`
package is required, and the resulting binary has no runtime
dependency on the host's `liblzma`.

### Switching to the dynamic path

If you modify the workspace to drop the `static` feature on
`xz2`:

1. Install your distro's liblzma development package:
   - Debian / Ubuntu: `liblzma-dev`
   - Fedora: `xz-devel`
2. Ensure `pkg-config` can find it (the package manager's
   install should handle this; if not, inspect
   `PKG_CONFIG_PATH`).

### Why the default is static

The static build keeps CI builds reproducible across host
distros: a `liblzma` ABI bump on one runner no longer silently
shifts tarball-decompression behaviour on another, and the
resulting binary is self-contained enough to copy across
machines without tracking an extra shared-library dependency.
The `ldd` pin test (`tests/ldd_pin.rs`) guards against an
accidental flip away from static by counting dynamic-library
entries — a bump there on any PR needs an explicit
acknowledgement in the commit message.
