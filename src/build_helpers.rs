// Build-script helpers that are also lib-testable.
//
// Following kernel_path.rs's pattern: this file is `include!`d
// into build.rs (so the helper is available during the build
// script) AND `#[cfg(test)] mod`d into lib.rs (so the `#[cfg(test)]`
// blocks run via `cargo nextest`/`cargo ktstr test`). The
// `#[cfg(test)]` items below are elided from the build.rs include
// because `cargo build` doesn't define `cfg(test)`.

/// Retry `attempt` up to `max_attempts` times with exponential
/// backoff (2s, 4s, 8s before the 2nd / 3rd / 4th tries; no sleep
/// after the final attempt). Emits `cargo:warning=...` log lines
/// at each attempt start and on each failure so build output stays
/// diagnosable.
///
/// `attempt` receives the 1-indexed attempt number; the closure
/// owns its own per-attempt cleanup if the work leaves partial
/// state behind that would block the next try (the helper does no
/// inter-attempt mutation — fully transactional work doesn't need
/// per-attempt cleanup; partial-state work like git clone into a
/// non-empty dir does, and handles it inside the closure conditional
/// on `i > 1`).
///
/// Retry sites in build.rs (busybox tarball download, and wprof
/// git clone when the `wprof` feature is enabled) route through
/// this helper so backoff timing, attempt
/// counting, and log wording stay in lockstep; a change to the
/// retry strategy (e.g. jittered backoff, max-attempts bump) only
/// edits this one function.
///
/// # Panics
///
/// Panics when `max_attempts == 0` — caller bug; the helper has
/// no work to retry. Call sites use `const MAX_*_ATTEMPTS: u32 = 4`
/// literals; the assert guards future callers.
fn retry_with_backoff<F, T>(label: &str, max_attempts: u32, mut attempt: F) -> Result<T, String>
where
    F: FnMut(u32) -> Result<T, String>,
{
    assert!(
        max_attempts > 0,
        "retry_with_backoff requires max_attempts >= 1; got 0 for label {label:?}",
    );
    let mut last_err: Option<String> = None;
    for i in 1..=max_attempts {
        println!("cargo:warning={label}: attempt {i}/{max_attempts}");
        match attempt(i) {
            Ok(v) => return Ok(v),
            Err(e) => {
                println!("cargo:warning={label}: attempt {i} failed: {e}");
                last_err = Some(e);
                if i < max_attempts {
                    // Exponential backoff: 2s, 4s, 8s before the next try.
                    let backoff = 1u64 << i;
                    std::thread::sleep(std::time::Duration::from_secs(backoff));
                }
            }
        }
    }
    Err(last_err.expect(
        "max_attempts > 0 guarded above; loop ran at least once; last_err set on every Err arm",
    ))
}

/// Reuse a pre-built blob binary that `cargo-ktstr` already embedded
/// and extracted, instead of fetching + compiling our own. `cargo-ktstr`
/// bakes the compiled busybox / wprof into its binary
/// (`bin/cargo_ktstr/blobs.rs` `BUSYBOX_BYTES` / `WPROF_BYTES`),
/// extracts them to disk at startup, and hands the child build their
/// paths via `KTSTR_BUSYBOX_BIN` / `KTSTR_WPROF_BIN`; copying that
/// binary into `$OUT_DIR/{blob_name}` is byte-equivalent to a local
/// build's output (both originate from the same pinned source) and
/// skips the network fetch entirely.
///
/// Returns `true` only when `dest` was populated from `src`. Returns
/// `false` — so the caller falls through to its normal download/compile
/// path — when `src` is `None`, an empty string, or points at a path
/// that is not an existing, non-empty regular file (missing, a
/// directory, or 0-byte). The is-file + non-empty guard is load-bearing:
/// a `cargo-ktstr` built with `KTSTR_SKIP_{BUSYBOX,WPROF}_BUILD=1`
/// carries an empty placeholder, and copying that would bake an
/// un-exec'able blob into the guest initramfs. `install_env` empty-gates
/// both blobs (so the path var stays unset for an empty blob and the
/// caller normally never passes such a path — defense in depth); this
/// guard is the last line. A set-but-unusable path warns; an unset path
/// is silent (the normal not-under-cargo-ktstr / first-build case).
fn copy_prebuilt_blob(src: Option<&str>, dest: &std::path::Path, blob_name: &str) -> bool {
    let src = match src {
        Some(s) if !s.is_empty() => s,
        _ => return false,
    };
    let src_path = std::path::Path::new(src);
    match std::fs::metadata(src_path) {
        Ok(m) if m.is_file() && m.len() > 0 => {}
        _ => {
            println!(
                "cargo:warning=prebuilt {blob_name} at {src} is missing, not a \
                 regular file, or empty — building {blob_name} from source instead"
            );
            return false;
        }
    }
    std::fs::copy(src_path, dest).unwrap_or_else(|e| {
        panic!(
            "copy prebuilt {blob_name} from {src} to {}: {e}",
            dest.display()
        )
    });
    println!("cargo:warning=using embedded {blob_name} from cargo-ktstr (skipped fetch + compile)");
    true
}

/// Does `wprof_src` hold a complete recursive git clone? Requires
/// both `.git/HEAD` (init reached) AND `src/Makefile` (working tree
/// populated) — catches "init succeeded, checkout failed" partial
/// clones that the prior Makefile-only check missed.
#[cfg(feature = "wprof")]
fn is_wprof_clone_complete(wprof_src: &std::path::Path) -> bool {
    wprof_src.join(".git").join("HEAD").exists() && wprof_src.join("src").join("Makefile").exists()
}

/// Append an empty `[workspace]` sentinel to every wprof `src/`
/// sub-crate manifest that lacks one, so cargo stops its upward
/// workspace walk at the sub-crate instead of reaching ktstr's
/// workspace via OUT_DIR (the clone lives under `target/`). The wprof
/// Makefile runs a standalone `cargo build` for each sub-crate
/// (`demangle`, `wpb`, `wrust` → `lib*_c.a`); without the sentinel
/// each fails with "current package believes it's in a workspace when
/// it's not." This generalizes the former demangle-only patch so a
/// wprof rev that ships additional sub-crates needs no build.rs edit.
///
/// Scope: immediate child dirs of `src/`. A child whose manifest
/// already declares `[workspace]` is skipped — that covers both the
/// idempotent re-run case and a hypothetical sub-workspace ROOT under
/// `src/` (patching it would be wrong; its members are never immediate
/// `src/` children, so they are never visited). The blazesym submodule
/// lives outside `src/` and is its own workspace, so it is untouched.
///
/// The exact-line check (`l.trim() == "[workspace]"`, not substring)
/// avoids matching `[workspace.lints]` or a commented `# [workspace]`,
/// either of which would trick a substring check into skipping a
/// manifest that lacks the real sentinel table.
#[cfg(feature = "wprof")]
fn isolate_wprof_subcrate_workspaces(wprof_src: &std::path::Path) {
    let src = wprof_src.join("src");
    let entries = match std::fs::read_dir(&src) {
        Ok(e) => e,
        // No `src/` dir means the clone layout changed out from under
        // us; the wprof build (make in `src/`) will fail loudly next,
        // so there is nothing to isolate here.
        Err(_) => return,
    };
    for entry in entries.flatten() {
        if !entry.file_type().map(|t| t.is_dir()).unwrap_or(false) {
            continue;
        }
        let manifest = entry.path().join("Cargo.toml");
        if !manifest.exists() {
            continue;
        }
        let existing = std::fs::read_to_string(&manifest)
            .unwrap_or_else(|e| panic!("read {}: {e}", manifest.display()));
        let is_package = existing.lines().any(|l| l.trim() == "[package]");
        let has_workspace = existing.lines().any(|l| l.trim() == "[workspace]");
        if is_package && !has_workspace {
            use std::io::Write;
            let mut f = std::fs::OpenOptions::new()
                .append(true)
                .open(&manifest)
                .unwrap_or_else(|e| panic!("open {} for append: {e}", manifest.display()));
            f.write_all(b"\n[workspace]\n")
                .unwrap_or_else(|e| panic!("append [workspace] to {}: {e}", manifest.display()));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;
    use std::time::Instant;

    #[test]
    fn succeeds_on_first_try_no_sleep() {
        let calls = Cell::new(0u32);
        let started = Instant::now();
        let r: Result<u32, String> = retry_with_backoff("succeeds-first", 4, |_| {
            calls.set(calls.get() + 1);
            Ok(42)
        });
        assert_eq!(r.unwrap(), 42);
        assert_eq!(calls.get(), 1, "must not retry on success");
        assert!(
            started.elapsed().as_secs() < 1,
            "no sleep on first-try success",
        );
    }

    #[test]
    fn returns_last_err_after_max_attempts() {
        let calls = Cell::new(0u32);
        let r: Result<(), String> = retry_with_backoff("returns-last", 2, |i| {
            calls.set(calls.get() + 1);
            Err(format!("attempt {i} failed"))
        });
        assert_eq!(calls.get(), 2);
        assert!(
            r.unwrap_err().contains("attempt 2 failed"),
            "returns the LAST err, not the first",
        );
    }

    #[test]
    #[should_panic(expected = "retry_with_backoff requires max_attempts >= 1")]
    fn max_zero_panics_with_actionable_message() {
        let _: Result<(), String> = retry_with_backoff("max-zero", 0, |_| Ok(()));
    }

    #[test]
    fn copy_prebuilt_blob_copies_nonempty_source() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let src = tmp.path().join("src-busybox");
        std::fs::write(&src, b"BUSYBOX-BINARY-BYTES").expect("write src");
        let dest = tmp.path().join("out-busybox");
        assert!(
            copy_prebuilt_blob(Some(src.to_str().unwrap()), &dest, "busybox"),
            "a non-empty source must be copied",
        );
        assert_eq!(
            std::fs::read(&dest).expect("read dest"),
            b"BUSYBOX-BINARY-BYTES",
            "dest must hold the source bytes verbatim",
        );
    }

    #[test]
    fn copy_prebuilt_blob_rejects_none_and_empty_string() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let dest = tmp.path().join("out");
        assert!(
            !copy_prebuilt_blob(None, &dest, "busybox"),
            "None → no copy"
        );
        assert!(
            !copy_prebuilt_blob(Some(""), &dest, "busybox"),
            "empty path string → no copy",
        );
        assert!(
            !dest.exists(),
            "dest must not be created when there is no usable source",
        );
    }

    #[test]
    fn copy_prebuilt_blob_rejects_missing_source() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let missing = tmp.path().join("does-not-exist");
        let dest = tmp.path().join("out");
        assert!(
            !copy_prebuilt_blob(Some(missing.to_str().unwrap()), &dest, "busybox"),
            "a missing source file → no copy (fall through to build)",
        );
        assert!(!dest.exists());
    }

    #[test]
    fn copy_prebuilt_blob_rejects_zero_byte_source() {
        // A KTSTR_SKIP_BUSYBOX_BUILD-built cargo-ktstr embeds a 0-byte
        // placeholder; copying it would bake a broken busybox into the
        // guest initramfs. Must reject and fall through to a real build.
        let tmp = tempfile::tempdir().expect("tempdir");
        let empty = tmp.path().join("empty-busybox");
        std::fs::write(&empty, b"").expect("write empty");
        let dest = tmp.path().join("out");
        assert!(
            !copy_prebuilt_blob(Some(empty.to_str().unwrap()), &dest, "busybox"),
            "a 0-byte source must be rejected, not copied",
        );
        assert!(!dest.exists(), "must not copy a 0-byte placeholder");
    }

    #[test]
    fn copy_prebuilt_blob_rejects_directory_source() {
        // A directory reports metadata().len() > 0 on common
        // filesystems; without the is_file() guard it would pass the
        // non-empty check and then panic in fs::copy. Must be rejected
        // (return false) cleanly so the caller falls through to build.
        let tmp = tempfile::tempdir().expect("tempdir");
        let dir = tmp.path().join("a-directory");
        std::fs::create_dir(&dir).expect("create dir");
        let dest = tmp.path().join("out");
        assert!(
            !copy_prebuilt_blob(Some(dir.to_str().unwrap()), &dest, "busybox"),
            "a directory source must be rejected, not copied",
        );
        assert!(!dest.exists());
    }

    #[cfg(feature = "wprof")]
    #[test]
    fn is_wprof_clone_complete_rejects_missing_git_head() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let src = tmp.path();
        std::fs::create_dir_all(src.join("src")).expect("create src/");
        std::fs::write(src.join("src/Makefile"), "").expect("write Makefile");
        // .git/HEAD missing → clone interrupted before init reached:
        // init writes .git/HEAD as its very first step, so absence
        // here is the strongest "no clone happened" signal.
        assert!(
            !is_wprof_clone_complete(src),
            "Makefile alone is not enough; .git/HEAD must also exist",
        );
    }

    #[cfg(feature = "wprof")]
    #[test]
    fn is_wprof_clone_complete_rejects_missing_src_makefile() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let src = tmp.path();
        std::fs::create_dir_all(src.join(".git")).expect("create .git/");
        std::fs::write(src.join(".git/HEAD"), "ref: refs/heads/main\n").expect("write .git/HEAD");
        // .git/HEAD present + Makefile missing → init reached but
        // working tree wasn't populated (fetch / checkout / submodule
        // step failed mid-clone).
        assert!(
            !is_wprof_clone_complete(src),
            ".git/HEAD alone is not enough; src/Makefile must also exist",
        );
    }

    #[cfg(feature = "wprof")]
    #[test]
    fn is_wprof_clone_complete_accepts_both_present() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let src = tmp.path();
        std::fs::create_dir_all(src.join(".git")).expect("create .git/");
        std::fs::write(src.join(".git/HEAD"), "ref: refs/heads/main\n").expect("write .git/HEAD");
        std::fs::create_dir_all(src.join("src")).expect("create src/");
        std::fs::write(src.join("src/Makefile"), "").expect("write Makefile");
        assert!(
            is_wprof_clone_complete(src),
            "both files present → clone considered complete",
        );
    }

    #[cfg(feature = "wprof")]
    #[test]
    fn isolate_patches_subcrate_lacking_workspace() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let ws = tmp.path();
        let crate_dir = ws.join("src").join("wpb");
        std::fs::create_dir_all(&crate_dir).expect("create src/wpb");
        let manifest = crate_dir.join("Cargo.toml");
        std::fs::write(&manifest, "[package]\nname = \"wpb\"\n").expect("write manifest");
        isolate_wprof_subcrate_workspaces(ws);
        let patched = std::fs::read_to_string(&manifest).expect("read manifest");
        assert!(
            patched.lines().any(|l| l.trim() == "[workspace]"),
            "a [package] sub-crate with no [workspace] must get the sentinel",
        );
    }

    #[cfg(feature = "wprof")]
    #[test]
    fn isolate_skips_manifest_already_carrying_workspace() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let ws = tmp.path();
        let crate_dir = ws.join("src").join("demangle");
        std::fs::create_dir_all(&crate_dir).expect("create src/demangle");
        let manifest = crate_dir.join("Cargo.toml");
        std::fs::write(&manifest, "[package]\nname = \"demangle\"\n\n[workspace]\n")
            .expect("write manifest");
        isolate_wprof_subcrate_workspaces(ws);
        let after = std::fs::read_to_string(&manifest).expect("read manifest");
        assert_eq!(
            after.lines().filter(|l| l.trim() == "[workspace]").count(),
            1,
            "an already-patched manifest must not gain a second [workspace] (idempotent)",
        );
    }

    #[cfg(feature = "wprof")]
    #[test]
    fn isolate_patches_every_src_subcrate() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let ws = tmp.path();
        for name in ["demangle", "wpb", "wrust"] {
            let d = ws.join("src").join(name);
            std::fs::create_dir_all(&d).expect("create src/<name>");
            std::fs::write(
                d.join("Cargo.toml"),
                format!("[package]\nname = \"{name}\"\n"),
            )
            .expect("write manifest");
        }
        isolate_wprof_subcrate_workspaces(ws);
        for name in ["demangle", "wpb", "wrust"] {
            let m = std::fs::read_to_string(ws.join("src").join(name).join("Cargo.toml"))
                .expect("read manifest");
            assert!(
                m.lines().any(|l| l.trim() == "[workspace]"),
                "src/{name} must be isolated",
            );
        }
    }

    #[cfg(feature = "wprof")]
    #[test]
    fn isolate_leaves_crates_outside_src_untouched() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let ws = tmp.path();
        // A src/ sub-crate so the isolation loop actually runs.
        let inner = ws.join("src").join("wpb");
        std::fs::create_dir_all(&inner).expect("create src/wpb");
        std::fs::write(inner.join("Cargo.toml"), "[package]\nname = \"wpb\"\n").expect("write");
        // blazesym is a submodule at <wprof>/blazesym (sibling of src/)
        // and is its OWN workspace — patching it would break that
        // workspace, so it must stay untouched.
        let blaze = ws.join("blazesym");
        std::fs::create_dir_all(&blaze).expect("create blazesym");
        let blaze_manifest = blaze.join("Cargo.toml");
        std::fs::write(&blaze_manifest, "[package]\nname = \"blazesym\"\n").expect("write");
        isolate_wprof_subcrate_workspaces(ws);
        assert!(
            std::fs::read_to_string(&blaze_manifest)
                .expect("read")
                .lines()
                .all(|l| l.trim() != "[workspace]"),
            "a crate outside src/ (blazesym submodule) must not be patched",
        );
        assert!(
            std::fs::read_to_string(inner.join("Cargo.toml"))
                .expect("read")
                .lines()
                .any(|l| l.trim() == "[workspace]"),
            "sanity: the src/ sub-crate WAS isolated (loop ran)",
        );
    }
}
