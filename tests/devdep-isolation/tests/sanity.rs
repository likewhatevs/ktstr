//! Sanity integration test: proves the dev-dep wiring works (test-time
//! compilation DOES see ktstr) so the `devdep-isolation` recipe's
//! negative assertion (release build does NOT see ktstr) is contrasted
//! with a positive that ktstr is reachable when expected. Without this,
//! the `devdep-isolation` recipe could pass simply because the dev-dep
//! wiring is broken — the "ktstr not in release" check is only
//! meaningful when we know ktstr IS in tests.
//!
//! This test is NOT run by the `devdep-isolation` recipe (scripts.just /
//! justfile), which only runs `cargo build --release` of the fixture and
//! asserts the release build compiles no ktstr and the binary has no
//! `ktstr::` symbols — it does not compile the test target. A developer
//! can run this sanity test manually via
//! `cargo test --manifest-path tests/devdep-isolation/Cargo.toml`.

use ktstr::prelude::*;

#[test]
fn ktstr_prelude_is_reachable_from_dev_deps() {
    // Constructing an AssertResult via the public prelude exercises
    // both the type and its constructor — a regression that hid the
    // type or removed the constructor would fail this compile. The
    // verdict is read via is_pass() rather than a struct field.
    let r = AssertResult::pass();
    assert!(r.is_pass());
}
