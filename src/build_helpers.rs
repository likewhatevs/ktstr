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
/// Both retry sites in build.rs (busybox tarball download + wprof
/// git clone) route through this helper so backoff timing, attempt
/// counting, and log wording stay in lockstep; a change to the
/// retry strategy (e.g. jittered backoff, max-attempts bump) only
/// edits this one function.
///
/// # Panics
///
/// Panics when `max_attempts == 0` — caller bug; the helper has
/// no work to retry. Both current call sites use `const
/// MAX_*_ATTEMPTS: u32 = 4` literals, so the assert guards future
/// callers.
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

/// Predicate: does `wprof_src` hold a complete recursive git
/// clone? A complete clone has both `.git/HEAD` (git init reached)
/// AND `src/Makefile` (working tree populated). Returns `false`
/// if either is missing — the caller wipes wprof_src and re-clones.
///
/// Pulling this out as a pure-fs predicate makes it unit-testable
/// without spawning a real `git clone`. The two-file conjunction
/// catches the "init succeeded, checkout failed" partial state
/// that the prior Makefile-only check missed.
fn is_wprof_clone_complete(wprof_src: &std::path::Path) -> bool {
    wprof_src.join(".git").join("HEAD").exists() && wprof_src.join("src").join("Makefile").exists()
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
}
