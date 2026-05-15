//! Pins the cross-crate visibility of the
//! `KtstrTestEntry::extra_include_files` aggregation seam: building an
//! entry from outside the crate, with a runtime-constructed
//! `&'static str` path, and calling `all_include_files()` must
//! round-trip the entry into the returned vec.
//!
//! The in-crate unit tests at
//! `src/test_support/entry.rs` already pin the aggregation order /
//! interaction with payload + workloads using static string literals.
//! This integration test adds two distinct guarantees those tests
//! cannot cover from inside the crate:
//!
//! 1. The `extra_include_files` field and the `all_include_files`
//!    method are reachable through the public surface
//!    (`ktstr::test_support::*`) — a future refactor that makes
//!    either non-`pub` (or shifts them out of the re-export tree)
//!    fails this test rather than silently reducing the test author's
//!    construction options.
//!
//! 2. Runtime-allocated `&'static str` values work in the field,
//!    not just compile-time string literals — confirming the
//!    `Box::leak` pattern (the only way to construct a `'static`
//!    string from a runtime path) round-trips correctly. This is
//!    the realistic shape for any test author who needs to compute
//!    a fixture path at runtime rather than hard-code it.

#[test]
fn extra_include_files_round_trips_through_all_include_files() {
    // `KtstrTestEntry::extra_include_files` is typed as
    // `&'static [&'static str]` because the macro emits each entry
    // as a `static`. To construct one at runtime, leak both the
    // path-string allocation and the surrounding slice — single-test
    // memory growth, reclaimed by process exit.
    let fixture_path_str: &'static str = Box::leak(
        "/tmp/ktstr-test-fixture-extra-include.json"
            .to_owned()
            .into_boxed_str(),
    );

    let entry = ktstr::test_support::KtstrTestEntry {
        name: "extra_include_test",
        extra_include_files: Box::leak(Box::new([fixture_path_str])),
        ..ktstr::test_support::KtstrTestEntry::DEFAULT
    };

    // With DEFAULT (payload=None, workloads=&[]), the only source
    // contributing to all_include_files is extra_include_files. Pin
    // the exact returned vec so a future regression that drops the
    // entry, reorders it relative to other sources, or adds phantom
    // entries surfaces here.
    let all = entry.all_include_files();
    assert_eq!(
        all,
        vec![fixture_path_str],
        "all_include_files must round-trip extra_include_files entries unchanged \
         when payload+workloads are default-empty; got {all:?}",
    );
}
