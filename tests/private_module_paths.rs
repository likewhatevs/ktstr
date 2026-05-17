//! External-context check for `ktstr::__private::{ctor, serde_json}`.
//!
//! The `#[ktstr_test]` proc macro emits a
//! `#[::ktstr::__private::linkme::distributed_slice(::ktstr::test_support::KTSTR_TESTS)]`
//! static of type `KtstrTestEntry` that registers the test in the
//! `KTSTR_TESTS` distributed slice at link time. The `__private::ctor`
//! and `__private::serde_json` re-exports are part of the surface
//! contract for downstream test-author code: `__private::serde_json`
//! lets downstream tests parse sidecar output without listing
//! `serde_json` as a direct dep, and `__private::ctor` re-exports both
//! ctor forms — the proc-macro attribute `__private::ctor::ctor` and
//! the declarative `__private::ctor::declarative::ctor!{}` macro — so
//! downstream authors can register pre-`main()` setup using either
//! form. ktstr's own in-tree sites use the declarative form (see
//! `src/test_support/dispatch.rs`, `profraw.rs`, `entry.rs`) because
//! it avoids the TT-muncher recursion-limit cost the proc-macro form
//! would impose on the `#[ktstr_test]` expansion path.
//!
//! If any of these re-exports change path or disappear, downstream
//! crates that depend on the surface fail to compile. This file
//! exercises both ctor forms + serde_json directly from external test
//! code — i.e. treating `ktstr` as a dev-dependency — so a silent
//! regression in the private re-export surface would fail this
//! binary's build before the broader integration suite runs.
//!
//! The assertions live inside plain `#[test]` fns because this file
//! holds no `#[ktstr_test]` entries. Each confirms a path resolves,
//! can be invoked, and produces the same behavior that the macro
//! expansion relies on.

use ktstr::__private;

/// `serde_json::to_string` must be reachable through the re-export
/// and must serialize a simple structure the same way the top-level
/// `serde_json` crate would.
#[test]
fn private_serde_json_to_string_roundtrip() {
    let v: Vec<(&str, u32)> = vec![("llc", 0), ("borrow", 1)];
    let json = __private::serde_json::to_string(&v).expect("serialize via __private path");
    // serde_json formats tuple structs as JSON arrays; the expected
    // output is stable and equality-testable.
    assert_eq!(json, r#"[["llc",0],["borrow",1]]"#);
}

/// `serde_json::from_str` is used by downstream consumers reading
/// sidecar output. Roundtrip a value through `__private::serde_json`
/// both directions to prove the re-export exposes the full crate,
/// not just a subset.
#[test]
fn private_serde_json_from_str_roundtrip() {
    let v: Vec<(String, u32)> =
        __private::serde_json::from_str(r#"[["llc",0]]"#).expect("parse via __private path");
    assert_eq!(v, vec![("llc".to_string(), 0)]);
}

/// `__private::ctor` must expose the `#[ctor]` attribute macro used by
/// the test-flag registration path. Attach it here via the fully
/// qualified re-export path (matching the macro's emission style) and
/// observe its side effect — the ctor fires before `#[test]` runs,
/// so by the time the test body executes, the static has been
/// initialized.
static INIT_FIRED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

#[::ktstr::__private::ctor::ctor(unsafe, crate_path = ::ktstr::__private::ctor)]
fn mark_ctor_fired() {
    INIT_FIRED.store(true, std::sync::atomic::Ordering::Release);
}

#[test]
fn private_ctor_attribute_fires_before_tests() {
    assert!(
        INIT_FIRED.load(std::sync::atomic::Ordering::Acquire),
        "`#[::ktstr::__private::ctor::ctor(...)]` must run before the test harness dispatches"
    );
}

/// `__private::ctor::declarative::ctor!{}` must also be reachable —
/// it's the form ktstr's own in-tree sites use (see
/// `src/test_support/dispatch.rs`, `profraw.rs`, `entry.rs`). Pin
/// the declarative re-export here so a future refactor that drops it
/// (e.g. removing the `ctor::declarative` re-export) fails this
/// binary's build before downstream test-authors who depend on the
/// form notice.
static DECLARATIVE_FIRED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

::ktstr::__private::ctor::declarative::ctor! {
#[ctor(unsafe)]
fn mark_declarative_ctor_fired() {
    DECLARATIVE_FIRED.store(true, std::sync::atomic::Ordering::Release);
}
}

#[test]
fn private_ctor_declarative_form_fires_before_tests() {
    assert!(
        DECLARATIVE_FIRED.load(std::sync::atomic::Ordering::Acquire),
        "`::ktstr::__private::ctor::declarative::ctor!` must run before the test harness dispatches"
    );
}
