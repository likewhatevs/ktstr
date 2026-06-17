# Contributing to ktstr

Notes for contributors modifying the workspace or its build
configuration. Day-to-day test authoring does not need any of
this.

## Dev workflow

Install [just](https://github.com/casey/just) (`cargo install just`).
All dev and CI commands are defined in the `justfile` — run
`just --list` to see available recipes. CI uses the same recipes.

Pre-PR sanity check: run `just lint && just compile-fail && just
link-check` locally before opening a PR. These mirror the `lint`,
`compile-fail`, and `docs-link-check` GitHub Actions jobs that run
on every pull-request (and on pushes to `main`); the heavier `test-x64`,
`test-arm64`, `coverage-x64`, and `coverage-arm64` jobs run on
self-hosted KVM runners and don't need to be run locally.
`just compile-fail` shells `cargo nextest run`, so install
nextest locally (`cargo install --locked cargo-nextest`) before
running it.

Doc tooling (needed only when you change `doc/guide/src/` and
want to validate locally): `mdbook` for `just docs` (runs
`mdbook build && mdbook test`); `mdbook-linkcheck2` + `lychee`
for `just link-check` (runs `mdbook build`, then `mdbook test`,
then `lychee` against the rendered HTML).

```
cargo install mdbook mdbook-linkcheck2 lychee --locked
```

CI installs these automatically; install them locally only if you
plan to validate guide changes before pushing.

## Compile-fail tests (trybuild)

Fixtures under `tests/compile_fail/` pin the proc-macro diagnostics
that `#[ktstr_test]` and `declare_scheduler!` emit. Without them an
upstream `syn` / `proc-macro2` bump can silently degrade an error
message that a test author would otherwise see at compile time.

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
