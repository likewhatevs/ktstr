//! Runtime blob loaders for binaries that `cargo-ktstr` extracts
//! and exports via env vars.
//!
//! The `ktstr` library carries no embedded binary blobs (busybox,
//! wprof, etc.) — those bytes only live inside the `cargo-ktstr`
//! binary's executable, kept out of `ktstr.rlib` to avoid bloating
//! every library consumer (each test binary, scheduler-author
//! crates depending on `ktstr` as a dev-dep). At cargo-ktstr
//! startup, [`crate::vmm::guest_comms`] inherits an env var pointing
//! at a tempfile holding the blob bytes; helpers in this module
//! read the env var and load the file on demand at the point the
//! library actually needs to pack the blob into an initramfs.
//!
//! Required entry point: `cargo ktstr <SUB>`. Direct `cargo nextest
//! run` invocations bypass the cargo-ktstr setup and leave the env
//! vars unset — every helper here returns an error in that case,
//! never silently no-ops. The canonical test invocation is
//! `cargo ktstr test`.

use anyhow::{Context, Result};

/// Load the busybox blob bytes from the path exported by
/// `cargo-ktstr` at startup via the `KTSTR_BUSYBOX_PATH` env var.
///
/// Returns an error if the env var is unset (caller bypassed
/// `cargo ktstr <SUB>`) or if the file cannot be read (tempfile
/// gone, permissions wrong, etc.). Never returns `None` silently —
/// busybox is a load-bearing dependency for shell-mode VMs and
/// disk-template builds; a missing blob must fail loudly.
pub fn load_busybox_bytes() -> Result<Vec<u8>> {
    load_blob_from_env(crate::KTSTR_BUSYBOX_PATH_ENV, "busybox")
}

/// Return the on-disk path to the wprof binary that `cargo-ktstr`
/// extracted at startup. The path lives for the lifetime of the
/// `cargo-ktstr` process (and inherited child processes). Returns
/// an error if `KTSTR_WPROF_PATH` is unset.
///
/// Preferred over `load_wprof_bytes` when the caller intends to
/// hand the file off to the existing `include_files` initramfs
/// pipeline, which performs `DT_NEEDED` shared-library resolution
/// against the binary's ELF — wprof is dynamically linked
/// (libelf, libz, blazesym C ABI, etc.) and will fail to run
/// inside the guest without those libs packed alongside.
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
            "{env_var} env var unset — {blob_name} blob is provided \
             by `cargo-ktstr` at startup. Run tests via \
             `cargo ktstr test`, not direct `cargo nextest run`."
        )
    })?;
    std::fs::read(&path).with_context(|| format!("read {blob_name} blob from {env_var}={path}"))
}
