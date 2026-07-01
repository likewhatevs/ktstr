#[test]
#[ignore]
fn compile_fail() {
    let t = trybuild::TestCases::new();
    t.compile_fail("tests/compile_fail/*.rs");
}

/// Collect stems of `.rs` and `.stderr` files in `tests/compile_fail/`.
fn fixture_pairs() -> (Vec<String>, Vec<String>) {
    let manifest_dir =
        std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR set by cargo");
    let fixture_dir = std::path::Path::new(&manifest_dir).join("tests/compile_fail");
    let mut rs_stems = Vec::new();
    let mut stderr_stems = Vec::new();
    for entry in
        std::fs::read_dir(&fixture_dir).unwrap_or_else(|e| panic!("read_dir({fixture_dir:?}): {e}"))
    {
        let entry = entry.expect("read_dir entry");
        let path = entry.path();
        let stem = match path.file_stem().and_then(|s| s.to_str()) {
            Some(s) => s.to_string(),
            None => continue,
        };
        match path.extension().and_then(|e| e.to_str()) {
            Some("rs") => rs_stems.push(stem),
            Some("stderr") => stderr_stems.push(stem),
            _ => {}
        }
    }
    (rs_stems, stderr_stems)
}

/// Walk `tests/compile_fail/` and assert every `*.rs` fixture has a
/// paired `*.stderr` snapshot (and vice versa), plus a non-zero
/// fixture count. Runs by default (no `#[ignore]`) because the
/// walk is a cheap filesystem scan — no `cargo build` is invoked,
/// so the test-group serialization the `compile_fail` driver needs
/// does NOT apply here.
///
/// Catches three regression classes the `compile_fail` driver
/// itself cannot:
///
/// 1. An `.rs` fixture added without its `.stderr` snapshot — a
///    contributor wrote a new compile-fail case but forgot to run
///    `TRYBUILD=overwrite just compile-fail` to land the snapshot.
///    This inventory test catches that cheaply and by default,
///    whereas the `compile_fail` driver is `#[ignore]` and so does
///    not run in a normal test pass. When the driver IS run under
///    trybuild's default (Wip) mode, a missing `.stderr` is written
///    to a `wip/` directory (not the pinned path) with a printed
///    NOTE, and the driver then fails rather than landing the
///    pinned snapshot; only `TRYBUILD=overwrite` writes the real
///    `.stderr`.
///
/// 2. An `.stderr` snapshot left over after its `.rs` was deleted
///    — typically a rename without removing the old snapshot.
///
/// 3. The fixture directory accidentally cleared (e.g. a botched
///    `git clean` or a mis-targeted rebase) — the driver would
///    pass trivially because trybuild treats a zero-glob match as
///    "nothing to verify".
#[test]
fn compile_fail_fixture_inventory() {
    let (rs_stems, stderr_stems) = fixture_pairs();
    assert!(
        !rs_stems.is_empty(),
        "tests/compile_fail/ has zero `.rs` fixtures — directory may have \
         been accidentally cleared. trybuild's compile_fail test would \
         silently pass with no fixtures to verify."
    );

    let rs_set: std::collections::BTreeSet<&str> = rs_stems.iter().map(String::as_str).collect();
    let stderr_set: std::collections::BTreeSet<&str> =
        stderr_stems.iter().map(String::as_str).collect();

    let missing_stderr: Vec<&&str> = rs_set.difference(&stderr_set).collect();
    let orphan_stderr: Vec<&&str> = stderr_set.difference(&rs_set).collect();

    assert!(
        missing_stderr.is_empty(),
        ".rs fixtures without paired .stderr snapshot: {missing_stderr:?}. \
         Run `TRYBUILD=overwrite just compile-fail` to generate the missing \
         snapshots, then inspect each `.stderr` for clean diagnostic text \
         before committing."
    );
    assert!(
        orphan_stderr.is_empty(),
        ".stderr snapshots without paired .rs fixture: {orphan_stderr:?}. \
         The fixture was likely renamed or deleted — remove the stale \
         snapshot file."
    );
}
