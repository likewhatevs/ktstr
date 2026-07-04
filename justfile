# All commands live here so they are runnable locally and on CI
# identically.  CI YAML handles only checkout, toolchain, cache, and
# tool installation; every `run:` step calls a justfile recipe.

# rust-script recipes (the devdep-isolation check) live in scripts.just.
mod scripts

default:
    @just --list

# Format code
fmt:
    cargo fmt --all

# Run all lints: format check, type check (x2 feature sets), clippy
# (x2), and rustdoc-warnings-as-errors. Every leg runs even when an
# earlier one fails; the recipe then reports the full set of failing
# legs and exits non-zero if any failed, so one CI run yields a
# complete lint-failure inventory instead of stopping at the first.
lint:
    #!/usr/bin/env bash
    set -uo pipefail
    failed=()
    run() {
        local label="$1"; shift
        echo "=== lint leg: ${label} ==="
        if ! "$@"; then failed+=("${label}"); fi
    }
    run "fmt --check"              cargo fmt -- --check
    run "check"                    cargo check --workspace --all-targets
    run "check wprof,integration"  cargo check --workspace --all-targets --features wprof,integration
    run "clippy"                   cargo clippy --workspace --all-targets
    run "clippy wprof,integration" cargo clippy --workspace --all-targets --features wprof,integration
    run "doc-strict"               just doc-strict
    if [ ${#failed[@]} -ne 0 ]; then
        echo
        echo "lint: ${#failed[@]} leg(s) FAILED: ${failed[*]}"
        exit 1
    fi
    echo "lint: all legs passed"

# Promote every rustdoc warning to an error. RUSTDOCFLAGS reaches every
# crate in the workspace (including ktstr-macros), where `cargo doc -- -D
# warnings` would only forward the flag to the top-level invocation.
# `--document-private-items` gates the private-item intra-doc link
# and broken-html-tag warnings the project cleared in batch — without
# it a regression would silently slip past CI on a private-symbol
# rename that broke its intra-doc references.
doc-strict:
    RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps --all-features --document-private-items

# Build a test kernel
kernel-build version="":
    cargo run --bin cargo-ktstr -- ktstr kernel build --skip-sha256 {{ if version != "" { "--kernel " + version } else { "" } }}

# Run tests against a kernel version. `extra-features` is passed BOTH to
# the cargo-ktstr build AND appended to the inner test feature list. The
# build pass matters for a blob-embedding feature like `wprof`: cargo-ktstr
# only embeds WPROF_BYTES — exported as KTSTR_WPROF_PATH at startup, which
# the #[ktstr_test(wprof)] tests require — when itself built with the
# feature. E.g. `just test 6.14 wprof` → cargo-ktstr built `--features
# wprof`; tests run `--features integration,wprof`.
test kernel extra-features="":
    cargo run --bin cargo-ktstr {{ if extra-features != "" { "--features " + extra-features } else { "" } }} -- ktstr test --kernel {{kernel}} -- --profile ci --features integration{{ if extra-features != "" { "," + extra-features } else { "" } }} --no-fail-fast -j $(nproc)

# Run trybuild compile_fail fixtures.
#
# Each trybuild fixture spawns its own `cargo build`, all of which
# share target/'s build lock; trybuild iterates them sequentially.
# The stale-target race surfaces only when compile_fail runs
# concurrently with OTHER tests that also touch target/ — the
# `compile-fail` nextest test-group (see .config/nextest.toml)
# reserves a single slot for the matched filter so no neighbour
# test runs alongside the compile_fail driver. The filter
# `binary(compile_fail) & test(=compile_fail)` is anchored to the
# exact driver fn so a future test whose name happens to contain
# `compile_fail` is not accidentally swept into the serial slot.
#
# Regenerate snapshots after intentional diagnostic changes:
#   TRYBUILD=overwrite just compile-fail
# Re-bless with the pinned minimal toolchain (no rust-src — see
# rust-toolchain.toml); rust-src makes rustc emit std-source snippets
# that won't match CI's minimal toolchain.
#
# KTSTR_SKIP_WPROF_BUILD stubs the wprof blob (0-byte $OUT_DIR/wprof):
# trybuild is compile-only and needs the `wprof` feature (so the
# wprof-gated compile_fail fixtures emit their diagnostics) but not a
# real blob, and the wprof/blazesym clone-build fails on the
# GitHub-hosted compile-fail runner (libblazesym_c.a not produced).
compile-fail:
    KTSTR_SKIP_WPROF_BUILD=1 cargo nextest run --profile ci --features wprof -E 'binary(compile_fail) & test(=compile_fail)' --run-ignored all --no-fail-fast

# Run the ktstr-macros proc-macro crate's host unit tests (attr parsing,
# codegen, cross-attr validation). cargo-ktstr's VM test runner only
# discovers the main ktstr crate's #[ktstr_test] binaries, NOT this
# proc-macro crate, so without an explicit run these never execute in CI.
# `--features wprof` (ktstr-macros's own feature — a pure cfg, no blob
# build) so the wprof-gated expect_auto_repro parse tests run too; the
# feature-agnostic tests run regardless.
test-macros:
    cargo nextest run -p ktstr-macros --features wprof --no-fail-fast

# Run the crate's rustdoc doctests. cargo-ktstr's VM test runner and
# cargo-nextest both SKIP `///` examples (`--doc` is not a nextest concept),
# so a broken doc example would never execute in CI — it did, undetected,
# until this gate was added. `cargo test --doc` is the only runner that
# compiles and runs doc examples; it keeps the rustdoc examples in lockstep
# with the live API. Default feature set (the pub-API examples); the
# wprof/integration-gated paths have no doctests today.
test-doc:
    cargo test --doc

# Verify ktstr stays out of a downstream consumer's release binary.
# Thin wrapper over the `scripts::devdep-isolation` rust-script recipe
# (see scripts.just): it builds the dev-dep fixture and asserts
#   (1) `cargo build --release` of the fixture compiles ZERO ktstr code,
#   (2) the resulting binary contains no `ktstr::` symbols.
# Pins Cargo's dev-dep isolation contract so a future regression that
# accidentally widens ktstr's reach into prod builds fails CI loudly.
devdep-isolation:
    @just scripts::devdep-isolation

# Run coverage against a kernel version. `kernel` is pinned like `test`
# (`--kernel`), NOT auto-discovered: CI's coverage job must run the SAME kernel
# as the test matrix, otherwise `just kernel-build` (no version) picks the latest
# upstream kernel, which can diverge on config-gated layout (e.g. the psi enum
# offsets) and scheduler behavior from the pinned test kernels. `extra-features`
# is passed to the cargo-ktstr build (so a blob-embedding feature like `wprof` is
# provisioned — see `test`) AND appended to the inner coverage feature list.
coverage kernel extra-features="":
    cargo run --bin cargo-ktstr {{ if extra-features != "" { "--features " + extra-features } else { "" } }} -- ktstr coverage --kernel {{kernel}} -- --profile ci --lcov --output-path lcov.info --features integration{{ if extra-features != "" { "," + extra-features } else { "" } }} --exclude-from-report scx-ktstr

# Show sccache statistics
sccache-stats:
    sccache --show-stats

# Show test statistics
stats:
    cargo run --bin cargo-ktstr -- ktstr stats

# Compare performance_mode metrics: HEAD vs a baseline commit (noise-adjusted; runs per side defaults to 5)
perf-delta kernel base="" runs="5":
    cargo run --bin cargo-ktstr -- ktstr perf-delta --noise-adjust {{runs}} --kernel {{kernel}}{{ if base != "" { " --base " + base } else { "" } }}

# Build and link-check the guide book
docs:
    mdbook build doc/guide
    mdbook test doc/guide

# Build the guide book and validate every internal link / anchor in the
# rendered HTML via lychee. `--offline` skips external HTTP fetches so
# the check is deterministic and CI-friendly (no network flakes). Run
# locally before opening a PR that touches doc/guide/src to catch
# broken cross-page anchors that mdbook-linkcheck2's pre-render check
# can miss (e.g. typo'd #fragment refs against post-render heading IDs).
# `mdbook test` also runs every doctest inside the guide so a code
# block that drifts away from the live API surfaces at PR time
# rather than after a release goes out.
link-check:
    mdbook build doc/guide
    mdbook test doc/guide
    lychee --offline --no-progress --exclude-path 'doc/guide/book/html/404.html' doc/guide/book/html

# Build API reference
api-docs:
    cargo doc --workspace --no-deps --all-features

# Build and serve the guide locally
book-serve:
    mdbook serve doc/guide --open

# Assemble the full documentation site (guide + API docs)
site: docs api-docs
    #!/usr/bin/env bash
    set -euo pipefail
    mkdir -p _site/guide _site/rustdoc
    cp -r doc/guide/book/html/* _site/guide/
    cp -r target/doc/* _site/rustdoc/
    cat > _site/index.html <<'HTML'
    <!DOCTYPE html>
    <meta http-equiv="refresh" content="0; url=guide/">
    HTML
