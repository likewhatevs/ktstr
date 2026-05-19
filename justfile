# All commands live here so they are runnable locally and on CI
# identically.  CI YAML handles only checkout, toolchain, cache, and
# tool installation; every `run:` step calls a justfile recipe.

default:
    @just --list

# Format code
fmt:
    cargo fmt --all

# Run all lints (format check + type check + clippy + rustdoc warnings)
lint:
    cargo fmt -- --check
    cargo check --workspace --all-targets
    cargo clippy --workspace --all-targets
    just doc-strict

# Promote every rustdoc warning to an error. RUSTDOCFLAGS reaches every
# crate in the workspace (including ktstr-macros), where `cargo doc -- -D
# warnings` would only forward the flag to the top-level invocation.
doc-strict:
    RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps --all-features

# Build a test kernel
kernel-build version="":
    cargo run --bin cargo-ktstr -- ktstr kernel build --skip-sha256 {{version}}

# Run tests against a kernel version
test kernel:
    cargo run --bin cargo-ktstr -- ktstr test --kernel {{kernel}} -- --profile ci --features integration -j $(( $(nproc) * 5 ))

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
compile-fail:
    cargo nextest run --profile ci -E 'binary(compile_fail) & test(=compile_fail)' --run-ignored all

# Run coverage
coverage:
    cargo run --bin cargo-ktstr -- ktstr coverage -- --profile ci --lcov --output-path lcov.info --features integration --exclude-from-report scx-ktstr

# Show sccache statistics
sccache-stats:
    sccache --show-stats

# Show test statistics
stats:
    cargo run --bin cargo-ktstr -- ktstr stats

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
    lychee --offline --no-progress doc/guide/book/html

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
    mkdir -p _site/guide _site/api
    cp -r doc/guide/book/html/* _site/guide/
    cp -r target/doc/* _site/api/
    cat > _site/index.html <<'HTML'
    <!DOCTYPE html>
    <meta http-equiv="refresh" content="0; url=guide/">
    HTML
