//! Blob extraction / install / load helpers shared by the `ktstr`
//! and `cargo-ktstr` binaries.
//!
//! The `ktstr` library itself embeds no binary blob bytes (busybox,
//! wprof, etc.): those live inside each *binary*'s executable via
//! `include_bytes!`, kept out of `ktstr.rlib` so library consumers
//! (test binaries, scheduler-author crates depending on `ktstr` as a
//! dev-dep) do not pay the size cost for blobs they never invoke. The
//! extraction *mechanism* lives here (no bytes) so both binaries share
//! one implementation.
//!
//! At startup each binary calls [`install_blob_env_if_unset`] (or, for
//! `cargo-ktstr`, its own always-set `install_env`) to extract its
//! embedded blob to a content-addressed tempfile and export the path
//! via an env var. Child processes inherit the var; the load helpers
//! here ([`load_busybox_bytes`]) read it and load the bytes on demand
//! at the point the library packs the blob into an initramfs.
//!
//! The standalone `ktstr` binary embeds busybox and installs it this
//! way, so `ktstr shell` works with no env var. Direct `cargo nextest
//! run` invocations run bare test binaries that install nothing and
//! leave the env vars unset — the load helpers return an error in that
//! case, never silently no-op. The canonical test invocation is
//! `cargo ktstr test`.

use anyhow::{Context, Result};

/// Load the busybox blob bytes from the path exported at binary
/// startup via the `KTSTR_BUSYBOX_PATH` env var.
///
/// Returns an error if the env var is unset (running a bare test
/// binary, or a binary built without the busybox blob) or if the file
/// cannot be read (tempfile gone, permissions wrong, etc.). Never
/// returns `None` silently — busybox is a load-bearing dependency for
/// shell-mode VMs and disk-template builds; a missing blob must fail
/// loudly.
pub fn load_busybox_bytes() -> Result<Vec<u8>> {
    load_blob_from_env(crate::KTSTR_BUSYBOX_PATH_ENV, "busybox")
}

/// Return the on-disk path to the wprof binary extracted by
/// `cargo-ktstr` at startup. Returns an error if `KTSTR_WPROF_PATH`
/// is unset.
#[cfg(feature = "wprof")]
pub fn load_wprof_path() -> Result<std::path::PathBuf> {
    let env_var = crate::KTSTR_WPROF_PATH_ENV;
    let path = std::env::var(env_var).map_err(|_| {
        anyhow::anyhow!(
            "{env_var} env var unset — wprof blob is provided \
             by `cargo-ktstr` at startup. Run tests via \
             `cargo ktstr test`, not direct `cargo nextest run`."
        )
    })?;
    Ok(std::path::PathBuf::from(path))
}

fn load_blob_from_env(env_var: &str, blob_name: &str) -> Result<Vec<u8>> {
    let path = std::env::var(env_var).map_err(|_| {
        anyhow::anyhow!(
            "{env_var} env var unset — the `ktstr` and `cargo-ktstr` \
             binaries embed the {blob_name} blob and export this path \
             at startup. An unset var means either the binary was \
             built without the {blob_name} blob (a skip-build \
             override) or you are running a bare test binary directly \
             — run tests via `cargo ktstr test`, not direct \
             `cargo nextest run`."
        )
    })?;
    std::fs::read(&path).with_context(|| format!("read {blob_name} blob from {env_var}={path}"))
}

/// Extract `bytes` to `$TMPDIR/ktstr-blob-{name_hint}-{sha256:16}`
/// and return the path.
///
/// Content-addressed naming makes extraction idempotent across
/// invocations: identical bytes always resolve to the same path, so a
/// fresh invocation that finds a sized-matching file at the target
/// reuses it instead of rewriting. A different blob (different bytes →
/// different hash) writes a new file alongside; sweeping stale files
/// is delegated to the OS `$TMPDIR` lifecycle (tmpfiles.d cleanup,
/// container teardown, CI workspace wipe).
///
/// The hash is truncated to 16 hex chars (64 bits) — collision-
/// resistant for the handful of blobs ktstr ships; the birthday bound
/// against a 64-bit hash keeps the collision probability far below any
/// rate that matters here.
///
/// Concurrency: two processes racing on the same blob both write a
/// sibling staging tempfile and `rename(2)` it onto the target. rename
/// is atomic on a single filesystem, and last-writer-wins is safe
/// because both writers produce bit-identical content.
pub fn extract_blob(bytes: &[u8], name_hint: &str) -> std::io::Result<std::path::PathBuf> {
    use sha2::{Digest, Sha256};
    use std::io::Write;
    use std::os::unix::fs::PermissionsExt;

    let mut hasher = Sha256::new();
    hasher.update(bytes);
    let digest = hasher.finalize();
    let hash_hex = hex::encode(&digest[..8]);

    let target = std::env::temp_dir().join(format!("ktstr-blob-{name_hint}-{hash_hex}"));

    // Reuse if a sized-matching file already exists. A
    // (name, truncated-sha256) match implies the bytes are
    // identical — see fn-level doc on the truncation bound.
    if let Ok(meta) = std::fs::metadata(&target)
        && meta.len() == bytes.len() as u64
    {
        return Ok(target);
    }

    // Write to a sibling staging tempfile so the rename below is
    // same-filesystem (atomic). keep() disables the tempfile's
    // drop-cleanup and returns the persisted staging path; the
    // subsequent rename(staging_path, target) removes the
    // random-suffix name and installs the content-addressed target.
    let dir = target.parent().unwrap_or_else(|| std::path::Path::new("."));
    let mut staging = tempfile::Builder::new()
        .prefix(&format!("ktstr-blob-{name_hint}-staging-"))
        .tempfile_in(dir)?;
    staging.write_all(bytes)?;
    staging.flush()?;
    std::fs::set_permissions(staging.path(), std::fs::Permissions::from_mode(0o755))?;
    let (_file, staging_path) = staging
        .keep()
        .map_err(|e| std::io::Error::other(format!("persist staging tempfile: {e}")))?;
    std::fs::rename(&staging_path, &target)?;
    Ok(target)
}

/// Extract `bytes` via [`extract_blob`] and export its path through
/// `env_var`, unless the var is already set or `bytes` is empty.
///
/// An already-set `env_var` wins (explicit override — used by tests
/// and callers pointing ktstr at a pre-extracted blob). Empty `bytes`
/// mean the blob's build was skipped (e.g. `KTSTR_SKIP_BUSYBOX_BUILD`
/// wrote a 0-byte placeholder): nothing is exported, so consumers get
/// the clean "unset" error from the load helpers at use time rather
/// than an opaque "exec format error" on a 0-byte blob.
///
/// Mirrors `cargo-ktstr`'s startup blob install for the standalone
/// `ktstr` binary, which embeds its own copy of busybox and calls this
/// from `main`.
///
/// # Safety
///
/// The caller MUST invoke this before spawning any thread or running
/// any code that reads environment variables concurrently:
/// `std::env::set_var` is unsound with a concurrent reader.
pub fn install_blob_env_if_unset(
    env_var: &str,
    name_hint: &str,
    bytes: &[u8],
) -> std::io::Result<()> {
    if std::env::var_os(env_var).is_some() || bytes.is_empty() {
        return Ok(());
    }
    let path = extract_blob(bytes, name_hint)?;
    // SAFETY: per the fn-level precondition, the caller runs this
    // before any thread spawn, so no concurrent env reader exists.
    unsafe {
        std::env::set_var(env_var, &path);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Same blob bytes resolve to the same content-addressed path
    /// across calls — the second call must not write a new file.
    /// Pins the idempotency contract that motivates content
    /// addressing (no per-invocation `$TMPDIR` accumulation).
    #[test]
    fn extract_blob_is_idempotent_for_same_bytes() {
        let bytes = b"ktstr-blob-test-idempotent-payload";
        let name = "idempotent-test";

        let first = extract_blob(bytes, name).expect("first extract");
        let first_mtime = std::fs::metadata(&first)
            .expect("first metadata")
            .modified()
            .expect("mtime");

        let second = extract_blob(bytes, name).expect("second extract");
        let second_mtime = std::fs::metadata(&second)
            .expect("second metadata")
            .modified()
            .expect("mtime");

        assert_eq!(
            first, second,
            "same bytes must resolve to the same content-addressed path",
        );
        assert_eq!(
            first_mtime, second_mtime,
            "second call must reuse the existing file, not rewrite it (would bump mtime)",
        );

        // Hygiene — keep `$TMPDIR` clean for parallel test runs.
        let _ = std::fs::remove_file(&first);
    }

    /// Different blob bytes resolve to different paths — pins that
    /// the hash is actually content-derived, not derived from the
    /// name_hint alone.
    #[test]
    fn extract_blob_distinguishes_distinct_bytes() {
        let name = "distinguish-test";
        let alpha = extract_blob(b"alpha-payload-bytes", name).expect("alpha extract");
        let beta = extract_blob(b"beta-payload-bytes", name).expect("beta extract");

        assert_ne!(
            alpha, beta,
            "distinct bytes under the same name_hint must produce distinct paths",
        );

        let _ = std::fs::remove_file(&alpha);
        let _ = std::fs::remove_file(&beta);
    }

    /// The extracted file must be executable (mode 0o755) — pins the
    /// executable-mode contract set at the
    /// `set_permissions(.., from_mode(0o755))` call. The extracted
    /// host tempfile is never exec'd: the library reads its bytes to
    /// pack them into the guest initramfs, where the guest exec bit is
    /// set independently by the cpio `write_entry` mode (`0o100755`).
    #[test]
    fn extract_blob_sets_executable_mode() {
        use std::os::unix::fs::PermissionsExt;
        let path = extract_blob(b"exec-mode-test-bytes", "exec-mode-test").expect("extract");
        let mode = std::fs::metadata(&path)
            .expect("metadata")
            .permissions()
            .mode()
            & 0o777;
        assert_eq!(
            mode, 0o755,
            "extracted blob must be rwx for owner / rx for everyone — got {mode:o}",
        );
        let _ = std::fs::remove_file(&path);
    }

    /// Empty bytes (blob build skipped) must leave the env var unset —
    /// consumers then get the clean "unset" error, never a 0-byte blob.
    #[test]
    fn install_blob_env_if_unset_skips_empty_bytes() {
        use crate::test_support::test_helpers::{EnvVarGuard, lock_env};
        let _lock = lock_env();
        let env_var = "KTSTR_TEST_BLOB_EMPTY_PATH";
        let _guard = EnvVarGuard::remove(env_var);
        install_blob_env_if_unset(env_var, "empty-blob-test", b"").expect("install empty");
        assert!(
            std::env::var_os(env_var).is_none(),
            "empty bytes must not export a blob path",
        );
    }

    /// A preset env var wins — the embedded blob never overrides an
    /// explicit override (the tests/override precedence contract).
    #[test]
    fn install_blob_env_if_unset_preserves_preset_var() {
        use crate::test_support::test_helpers::{EnvVarGuard, lock_env};
        let _lock = lock_env();
        let env_var = "KTSTR_TEST_BLOB_PRESET_PATH";
        let _guard = EnvVarGuard::set(env_var, "/preset/override/path");
        install_blob_env_if_unset(env_var, "preset-blob-test", b"nonempty-bytes")
            .expect("install with preset");
        assert_eq!(
            std::env::var(env_var).ok().as_deref(),
            Some("/preset/override/path"),
            "a preset env var must win over the embedded blob",
        );
    }

    /// Unset var + non-empty bytes: the extracted, content-addressed
    /// path is exported and points at a readable file.
    #[test]
    fn install_blob_env_if_unset_exports_extracted_path() {
        use crate::test_support::test_helpers::{EnvVarGuard, lock_env};
        let _lock = lock_env();
        let env_var = "KTSTR_TEST_BLOB_EXPORT_PATH";
        let _guard = EnvVarGuard::remove(env_var);
        let bytes = b"install-export-payload-bytes";
        install_blob_env_if_unset(env_var, "export-blob-test", bytes).expect("install export");
        let exported = std::env::var(env_var).expect("env var must be set after install");
        let read_back = std::fs::read(&exported).expect("exported path must be readable");
        assert_eq!(read_back, bytes, "exported blob bytes must round-trip");
        let _ = std::fs::remove_file(&exported);
    }
}
