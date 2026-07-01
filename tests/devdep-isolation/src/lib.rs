//! Minimal prod-side library surface for the devdep-isolation fixture.
//! Exists so the fixture has both a `[lib]` and `[bin]` target — same
//! shape as a typical downstream crate that exposes a library and
//! ships a CLI on top of it. The contents are deliberately trivial;
//! the verify script asserts on dev-dep isolation, not on this code's
//! behavior.

/// Sample prod function — an unused lib-crate stub. The bin
/// (`src/main.rs`) does not use the lib crate, so this function is dead
/// code and its symbol is not linked into `target/release/devdep-fixture`,
/// the binary the verify script scans (`nm -C --defined-only`, filtering
/// for `ktstr::`). It therefore does not currently exercise the search's
/// specificity.
pub fn fixture_greeting() -> &'static str {
    "devdep-fixture prod surface"
}
