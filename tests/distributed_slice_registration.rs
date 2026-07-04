//! External-context check for the `KTSTR_TESTS` distributed
//! slice.
//!
//! The `#[ktstr_test]` macro has been covered by every integration
//! test that uses it — if the macro emission broke, the whole suite
//! would fail to compile or link. This file covers the complementary
//! surface: the *manual* `#[distributed_slice]` path documented at
//! `test_support::KTSTR_TESTS`, for downstream crates that
//! programmatically populate entries without going through the
//! macro.
//!
//! Manual registration depends on `ktstr::distributed_slice` and
//! `ktstr::linkme` being re-exported at the crate root so consumers
//! can spell the `#[distributed_slice]` attribute and the companion
//! `#[linkme(crate = ktstr::linkme)]` annotation. If either re-export
//! disappears or the path changes, this file fails to compile. At
//! runtime, the framework's `--list` protocol must surface the
//! manually-registered entry under its declared name; nextest
//! discovery proves that by listing the entry below as a runnable
//! test.
//!
//! No standalone `#[test]` assertions are needed: the manually-
//! registered entry surfaces in nextest's `--list` as a runnable
//! test (`ktstr/external_context_marker`), and its presence there is
//! the registration proof. Plain `#[test]` functions in such a
//! binary stay visible too — `test_support::ktstr_main` intercepts
//! `--list` but re-lists them via `list_plain_tests`.

use anyhow::Result;
use ktstr::assert::AssertResult;
use ktstr::scenario::Ctx;
use ktstr::test_support::KtstrTestEntry;

fn external_context_test_fn(_ctx: &Ctx) -> Result<AssertResult> {
    Ok(AssertResult::pass())
}

/// Manual `#[distributed_slice]` registration reachable through
/// `ktstr::distributed_slice` and `ktstr::linkme`. If nextest
/// `--list` for this binary does not emit
/// `ktstr::distributed_slice_registration ktstr/external_context_marker`,
/// the manual-registration surface regressed — the macro expansion
/// still works but programmatic test generation has silently broken.
#[ktstr::distributed_slice(ktstr::test_support::KTSTR_TESTS)]
#[linkme(crate = ktstr::linkme)]
static EXTERNAL_CONTEXT_MARKER: KtstrTestEntry = KtstrTestEntry {
    name: "external_context_marker",
    func: external_context_test_fn,
    auto_repro: false,
    ..KtstrTestEntry::DEFAULT
};
