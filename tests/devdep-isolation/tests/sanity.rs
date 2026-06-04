//! Sanity integration test: proves the dev-dep wiring works (test-time
//! compilation DOES see ktstr) so the verify script's negative
//! assertion (release build does NOT see ktstr) is contrasted with a
//! positive that ktstr is reachable when expected. Without this, the
//! verify script could pass simply because the dev-dep wiring is
//! broken — the "ktstr not in release" check is only meaningful when
//! we know ktstr IS in tests.
//!
//! This test is NOT run by the verify script (`cargo test --no-run`
//! suffices to confirm test-time compilation); a developer can run
//! `cargo test --manifest-path tests/devdep-isolation/Cargo.toml` to
//! exercise it locally.

use ktstr::prelude::*;

#[test]
fn ktstr_prelude_is_reachable_from_dev_deps() {
    // Constructing an AssertResult via the public prelude exercises
    // both the type and its constructor — a regression that hid the
    // type or removed the constructor would fail this compile.
    let r = AssertResult::pass();
    assert!(r.passed);
}
