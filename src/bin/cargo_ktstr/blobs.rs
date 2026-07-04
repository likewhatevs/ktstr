//! Binary blobs embedded into the `cargo-ktstr` binary and propagated
//! to subcommand handlers + spawned test binaries via env vars.
//!
//! The `ktstr` library itself contains no embedded blobs — keeping
//! these bytes out of `ktstr.rlib` means library consumers (test
//! binaries, scheduler-author crates depending on `ktstr` as a
//! dev-dep) do not pay the binary-size cost for blobs they never
//! invoke. Only the `cargo-ktstr` binary carries the bytes.
//!
//! Runtime propagation: at startup, [`install_env`] extracts each
//! embedded blob to a **content-addressed** file under `$TMPDIR`
//! (`ktstr-blob-{name}-{sha256:16}`) and exports its path through
//! an env var (e.g. `KTSTR_BUSYBOX_PATH`). Child processes spawned
//! by cargo-ktstr (e.g. nextest, which fans out to per-test-binary
//! processes) inherit these env vars; the `ktstr` library's blob-
//! loading helpers (e.g. `ktstr::vmm::blobs::load_busybox_bytes`) read
//! the env vars and load the bytes from disk on demand. Subsequent
//! cargo-ktstr invocations with the same blob bytes hit the same
//! path and skip the write — only a fresh cargo-ktstr binary (new
//! blob content → new hash) creates a new file. Stale files from
//! earlier blob versions accumulate in `$TMPDIR` and are not swept
//! by this module; rely on the OS's `$TMPDIR` lifecycle (tmpfiles.d
//! `/tmp` cleanup, container teardown, CI workspace wipe) to
//! reclaim them.
//!
//! This mechanism requires that the user always invokes ktstr
//! through `cargo ktstr <SUB>` rather than `cargo nextest run`
//! directly. The `cargo ktstr test` entry point is the canonical
//! test invocation precisely so this propagation works.

use std::io::Write;

/// Statically-linked busybox binary, compiled by `build.rs`.
/// Embedded as bytes in the `cargo-ktstr` binary only.
pub const BUSYBOX_BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/busybox"));

#[cfg(feature = "wprof")]
pub const WPROF_BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/wprof"));

/// Extract a blob to `$TMPDIR/ktstr-blob-{name_hint}-{sha256:16}`.
///
/// Content-addressed naming makes the extraction idempotent across
/// cargo-ktstr invocations: the same blob bytes always produce the
/// same path, so a fresh invocation that finds a sized-matching
/// file at the target reuses it instead of writing again. A new
/// cargo-ktstr binary (different blob content → different hash)
/// writes a new file alongside; sweep of stale files is delegated
/// to the OS's `$TMPDIR` lifecycle (see the module-level note).
///
/// The previous random-suffix variant of this helper leaked one
/// file per cargo-ktstr invocation. On CI runners the leaked
/// `ktstr-blob-*` files accumulated until the workspace tear-down
/// reaped `$TMPDIR`. The hash truncation to 16 hex chars (64 bits)
/// is collision-resistant for the two known blobs we ship; even
/// allowing for many future blob versions the birthday bound
/// against a 64-bit hash leaves the collision probability below
/// any rate that matters here.
///
/// Concurrency: two cargo-ktstr processes racing on the same blob
/// both write to a sibling staging tempfile and rename(2) to the
/// content-addressed target. rename is atomic on a single
/// filesystem, and last-writer-wins is safe because both writers
/// produce bit-identical content.
fn extract_to_content_addressed_file(
    bytes: &[u8],
    name_hint: &str,
) -> std::io::Result<std::path::PathBuf> {
    use sha2::{Digest, Sha256};
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

/// Extract every embedded blob to its content-addressed `$TMPDIR`
/// path and export the resolved path via the matching env var.
/// Called from `cargo-ktstr`'s `main` **before any thread is
/// spawned**. The `unsafe` on `std::env::set_var` is sound only
/// under that precondition (no concurrent reader); see
/// std::env::set_var docs.
///
/// # Safety
///
/// Caller MUST invoke this before spawning any thread or calling
/// any code that reads environment variables concurrently.
pub fn install_env() -> std::io::Result<()> {
    // Skip the busybox extract+set_var pair when the embedded blob
    // is empty — that's the `KTSTR_SKIP_BUSYBOX_BUILD=1` path
    // (build.rs writes a 0-byte placeholder at $OUT_DIR/busybox so
    // the `include_bytes!` site compiles, but there is no real
    // binary to extract). Leaving `KTSTR_BUSYBOX_PATH` unset is the
    // right contract: any consumer that tries to use shell mode
    // gets a clean "env var not set" / "shell mode unavailable"
    // signal rather than an opaque "exec format error" the kernel
    // would emit on an attempted 0-byte exec.
    if !BUSYBOX_BYTES.is_empty() {
        let busybox_path = extract_to_content_addressed_file(BUSYBOX_BYTES, "busybox")?;
        // SAFETY: per fn-level precondition, the caller (cargo-ktstr's
        // main) calls this before any thread spawn, so no concurrent
        // env reader exists.
        unsafe {
            std::env::set_var(ktstr::KTSTR_BUSYBOX_PATH_ENV, &busybox_path);
        }
    }
    // Skip the wprof extract+set_var when the embedded blob is empty —
    // the `KTSTR_SKIP_WPROF_BUILD=1` path (build.rs writes a 0-byte
    // $OUT_DIR/wprof placeholder). Mirrors the busybox empty-gate above:
    // leaving `KTSTR_WPROF_PATH` unset gives consumers a clean "unset"
    // error and keeps a 0-byte path out of the child build's
    // `KTSTR_WPROF_BIN`, rather than handing over an empty blob.
    #[cfg(feature = "wprof")]
    if !WPROF_BYTES.is_empty() {
        let wprof_path = extract_to_content_addressed_file(WPROF_BYTES, "wprof")?;
        unsafe {
            std::env::set_var(ktstr::KTSTR_WPROF_PATH_ENV, &wprof_path);
        }
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
    fn extract_to_content_addressed_file_is_idempotent_for_same_bytes() {
        let bytes = b"ktstr-blob-test-idempotent-payload";
        let name = "idempotent-test";

        let first = extract_to_content_addressed_file(bytes, name).expect("first extract");
        let first_mtime = std::fs::metadata(&first)
            .expect("first metadata")
            .modified()
            .expect("mtime");

        let second = extract_to_content_addressed_file(bytes, name).expect("second extract");
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
    fn extract_to_content_addressed_file_distinguishes_distinct_bytes() {
        let name = "distinguish-test";
        let alpha =
            extract_to_content_addressed_file(b"alpha-payload-bytes", name).expect("alpha extract");
        let beta =
            extract_to_content_addressed_file(b"beta-payload-bytes", name).expect("beta extract");

        assert_ne!(
            alpha, beta,
            "distinct bytes under the same name_hint must produce distinct paths",
        );

        let _ = std::fs::remove_file(&alpha);
        let _ = std::fs::remove_file(&beta);
    }

    /// The extracted file must be executable (mode 0o755) — this pins
    /// the executable-mode contract of
    /// `extract_to_content_addressed_file` itself, set at the
    /// `set_permissions(.., from_mode(0o755))` call. The extracted
    /// host tempfile is never exec'd: the library reads its bytes to
    /// pack them into the guest initramfs, where the guest exec bit is
    /// set independently by the cpio `write_entry` mode (`0o100755`).
    #[test]
    fn extract_to_content_addressed_file_sets_executable_mode() {
        use std::os::unix::fs::PermissionsExt;
        let path = extract_to_content_addressed_file(b"exec-mode-test-bytes", "exec-mode-test")
            .expect("extract");
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
}
