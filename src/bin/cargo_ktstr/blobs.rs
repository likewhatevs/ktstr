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
//! embedded blob to a tempfile that lives for the remainder of the
//! process and exports its path through an env var (e.g.
//! `KTSTR_BUSYBOX_PATH`). Child processes spawned by cargo-ktstr
//! (e.g. nextest, which fans out to per-test-binary processes)
//! inherit these env vars; the `ktstr` library's blob-loading
//! helpers (e.g. `ktstr::vmm::load_busybox_bytes`) read the
//! env vars and load the bytes from disk on demand.
//!
//! This mechanism requires that the user always invokes ktstr
//! through `cargo ktstr <SUB>` rather than `cargo nextest run`
//! directly. The `cargo ktstr test` entry point is the canonical
//! test invocation precisely so this propagation works.

use std::io::Write;

/// Statically-linked busybox binary, compiled by `build.rs`.
/// Embedded as bytes in the `cargo-ktstr` binary only.
pub const BUSYBOX_BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/busybox"));

/// wprof binary (BPF-based system-wide tracer/profiler, BSD-3-Clause,
/// github.com/anakryiko/wprof), built by `build.rs`. Embedded as
/// bytes in the `cargo-ktstr` binary only. Loaded into auto-repro
/// VMs by `ktstr::vmm::blobs::load_wprof_bytes` when the test entry
/// requests wprof capture.
pub const WPROF_BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/wprof"));

// Env var names live as pub consts on the library — see
// [`ktstr::KTSTR_BUSYBOX_PATH_ENV`] and [`ktstr::KTSTR_WPROF_PATH_ENV`].
// Use those rather than re-spelling the strings here, so the inventory
// at the top of `lib.rs` stays the single source of truth.

/// Extract a blob to a tempfile that persists for the process
/// lifetime (no cleanup on drop — the kernel reclaims tempfiles on
/// process exit anyway, and we want child processes to be able to
/// read the file long after cargo-ktstr returns from its setup
/// path).
///
/// Returns the absolute path of the persisted tempfile.
fn extract_to_persistent_tempfile(
    bytes: &[u8],
    name_hint: &str,
) -> std::io::Result<std::path::PathBuf> {
    use std::os::unix::fs::PermissionsExt;
    let mut tempfile = tempfile::Builder::new()
        .prefix(&format!("ktstr-blob-{name_hint}-"))
        .tempfile()?;
    tempfile.write_all(bytes)?;
    tempfile.flush()?;
    std::fs::set_permissions(tempfile.path(), std::fs::Permissions::from_mode(0o755))?;
    // Convert into a persistent (path, file) pair — Drop becomes a
    // no-op, the file outlives this function.
    let (_file, path) = tempfile
        .keep()
        .map_err(|e| std::io::Error::other(format!("persist tempfile: {e}")))?;
    Ok(path)
}

/// Extract every embedded blob to a tempfile and export its path
/// via the matching env var. Called from `cargo-ktstr`'s `main`
/// **before any thread is spawned**. The `unsafe` on
/// `std::env::set_var` is sound only under that precondition (no
/// concurrent reader); see std::env::set_var docs.
///
/// # Safety
///
/// Caller MUST invoke this before spawning any thread or calling
/// any code that reads environment variables concurrently.
pub fn install_env() -> std::io::Result<()> {
    let busybox_path = extract_to_persistent_tempfile(BUSYBOX_BYTES, "busybox")?;
    let wprof_path = extract_to_persistent_tempfile(WPROF_BYTES, "wprof")?;
    // SAFETY: per fn-level precondition, the caller (cargo-ktstr's
    // main) calls this before any thread spawn, so no concurrent
    // env reader exists.
    unsafe {
        std::env::set_var(ktstr::KTSTR_BUSYBOX_PATH_ENV, &busybox_path);
        std::env::set_var(ktstr::KTSTR_WPROF_PATH_ENV, &wprof_path);
    }
    Ok(())
}
