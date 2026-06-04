#!/usr/bin/env bash
# Verify that a downstream consumer adding ktstr as `[dev-dependencies]`
# does NOT get ktstr compiled into their release binary.
#
# The contract being pinned:
#   1. `cargo build --release` of a crate with ktstr as a dev-dep
#      compiles ZERO ktstr code. Cargo's dev-dep isolation is
#      structural — ktstr's source, build.rs, and transitive crates
#      not shared with the prod tree are skipped.
#   2. The resulting release binary contains no `ktstr::` symbols.
#
# A regression that accidentally moves ktstr into a non-dev dep section
# (e.g. someone shifts an import out of `#[cfg(test)]` and then
# "fixes" the unresolved-import error by moving ktstr to
# `[dependencies]`) trips one of these assertions and fails CI.
#
# Driven by the `devdep-isolation` justfile recipe and the matching
# CI job. Run locally with: `just devdep-isolation`.

set -euo pipefail

cd "$(dirname "$0")"

echo "==> Running clean release build of devdep-fixture"
# Full clean so the "Compiling" output check sees this build's
# actual work, not stale cached artifacts. Targets the fixture's
# isolated target dir, not the parent workspace's, so the clean
# doesn't disturb other CI artifacts.
cargo clean --quiet

# Capture both stdout and stderr — cargo emits "Compiling …" on
# stderr. `tee` keeps the output visible for the CI log while we
# also grep it.
build_log=$(mktemp)
trap 'rm -f "$build_log"' EXIT
cargo build --release 2>&1 | tee "$build_log"

# ---- Assertion 1: cargo did not compile ktstr for the release build.
#
# Match "Compiling ktstr " (trailing space — pins the bare crate name,
# avoids matching e.g. "ktstr-macros" or "ktstr_sys" if either is ever
# added). Both the lib and bin compilations would emit a line through
# this matcher, so a single hit is sufficient signal.
if grep -E '^\s*Compiling ktstr [v0-9]' "$build_log"; then
    echo
    echo "FAIL: \`cargo build --release\` compiled ktstr — ktstr is leaking" \
         "into the prod build path of a dev-dep consumer." >&2
    exit 1
fi
echo "PASS: cargo did not compile ktstr for the release build"

# ---- Assertion 2: release binary contains no ktstr symbols.
#
# Demangle so we can match the `ktstr::` namespace path rather than the
# raw mangled prefix — substring match on `ktstr` alone would
# false-positive on the fixture's own `ktstr_devdep_fixture::` symbols.
# `--defined-only` filters out undefined symbols (would otherwise
# include linker-emitted stubs for libc / runtime intrinsics).
bin=target/release/devdep-fixture
if [ ! -x "$bin" ]; then
    echo "FAIL: expected release binary at $bin — cargo build did not produce it." >&2
    exit 1
fi
# `nm` is in binutils on every standard Linux distro and on macOS
# (LLVM nm is the default on macOS). The output format differs slightly
# between GNU nm and LLVM nm, but both honor `-C` for demangling and
# `--defined-only` for filtering, so the grep below works on both.
hit_count=$(nm -C --defined-only "$bin" 2>/dev/null | grep -c 'ktstr::' || true)
if [ "$hit_count" -ne 0 ]; then
    echo
    echo "FAIL: release binary $bin contains $hit_count ktstr:: symbols:" >&2
    nm -C --defined-only "$bin" | grep 'ktstr::' >&2 | head -20
    exit 1
fi
echo "PASS: release binary $bin contains no ktstr:: symbols"

echo
echo "OK: ktstr stays out of the downstream consumer's release binary."
