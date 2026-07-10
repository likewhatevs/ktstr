//! Binary blobs embedded into the `cargo-ktstr` binary and propagated
//! to subcommand handlers + spawned test binaries via env vars.
//!
//! The `ktstr` library itself contains no embedded blobs — keeping
//! these bytes out of `ktstr.rlib` means library consumers (test
//! binaries, scheduler-author crates depending on `ktstr` as a
//! dev-dep) do not pay the binary-size cost for blobs they never
//! invoke. Only the `cargo-ktstr` binary (and, for busybox, the
//! standalone `ktstr` binary) carries the bytes.
//!
//! Runtime propagation: at startup, [`install_env`] extracts each
//! embedded blob to a **content-addressed** file under `$TMPDIR`
//! (`ktstr-blob-{name}-{sha256:16}`, via [`ktstr::extract_blob`]) and
//! exports its path through an env var (e.g. `KTSTR_BUSYBOX_PATH`).
//! Child processes spawned by cargo-ktstr (e.g. nextest, which fans
//! out to per-test-binary processes) inherit these env vars; the
//! `ktstr` library's blob-loading helpers (e.g.
//! `ktstr::vmm::blobs::load_busybox_bytes`) read the env vars and load
//! the bytes from disk on demand.
//!
//! This mechanism requires that the user always invokes ktstr
//! through `cargo ktstr <SUB>` rather than `cargo nextest run`
//! directly. The `cargo ktstr test` entry point is the canonical
//! test invocation precisely so this propagation works.

/// Statically-linked busybox binary, compiled by `build.rs`.
/// Embedded as bytes in the `cargo-ktstr` binary.
pub const BUSYBOX_BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/busybox"));

#[cfg(feature = "wprof")]
pub const WPROF_BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/wprof"));

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
        let busybox_path = ktstr::extract_blob(BUSYBOX_BYTES, "busybox")?;
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
        let wprof_path = ktstr::extract_blob(WPROF_BYTES, "wprof")?;
        unsafe {
            std::env::set_var(ktstr::KTSTR_WPROF_PATH_ENV, &wprof_path);
        }
    }
    Ok(())
}
