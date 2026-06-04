//! Minimal prod-side library surface for the devdep-isolation fixture.
//! Exists so the fixture has both a `[lib]` and `[bin]` target — same
//! shape as a typical downstream crate that exposes a library and
//! ships a CLI on top of it. The contents are deliberately trivial;
//! the verify script asserts on dev-dep isolation, not on this code's
//! behavior.

/// Sample prod function. The verify script greps `nm`'s output for
/// `ktstr` symbols, so this function's mangled name (containing the
/// crate name `ktstr_devdep_fixture`) tests the grep's specificity —
/// a regression that searched for the substring "ktstr" instead of a
/// `ktstr::` namespace path would false-positive on this symbol.
pub fn fixture_greeting() -> &'static str {
    "devdep-fixture prod surface"
}
