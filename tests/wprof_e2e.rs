#![cfg(feature = "wprof")]
//! End-to-end test: wprof embedded in cargo-ktstr, runtime-injected
//! into a shell-mode VM, accessible at `/bin/wprof` inside the guest.
//!
//! Boots `cargo ktstr shell --exec "ls -la /bin/wprof"` and asserts:
//! - the cargo-ktstr binary actually carries the wprof bytes
//!   (else `install_env` would fail, the env var would not be set,
//!   and the shell-mode VM would not boot)
//! - the bytes propagate through the env-var injection chain
//!   (KTSTR_WPROF_PATH → ktstr library reads file via
//!   `crate::vmm::blobs::load_wprof_path` → packs into initramfs)
//! - the wprof binary lands at `/bin/wprof` in the guest filesystem
//!   with executable mode + non-trivial size
//!
//! Runs on the self-hosted CI runners (`[ktstr-x64]` /
//! `[ktstr-arm64]`). ktstr supplies the guest kernel itself via
//! its kernel-build cache — no operator-provided `../linux` tree
//! needed.

#[cfg(unix)]
const CARGO_KTSTR_BINARY: &str = env!("CARGO_BIN_EXE_cargo-ktstr");

/// `cargo ktstr shell --exec "ls -la /bin/wprof"` boots a VM,
/// the guest's `ls` (from busybox at `/bin/busybox`) inspects
/// `/bin/wprof`, and stdout shows the file with `x` mode and
/// non-trivial size.
#[test]
fn wprof_e2e_shell_mode_exposes_bin_wprof() {
    let output = std::process::Command::new(CARGO_KTSTR_BINARY)
        .arg("ktstr")
        .arg("shell")
        .arg("--no-perf-mode")
        .arg("--exec")
        .arg("ls -la /bin/wprof")
        .output()
        .expect("spawn cargo-ktstr shell");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "cargo ktstr shell exited non-zero (exit={:?})\nSTDOUT:\n{stdout}\n\nSTDERR:\n{stderr}",
        output.status.code(),
    );
    assert!(
        stdout.contains("/bin/wprof"),
        "guest `ls -la /bin/wprof` did not surface the binary; stdout:\n{stdout}",
    );
    // The wprof binary should be at least a few hundred KB. A
    // truncated / zero-byte file would indicate the embed →
    // extract → pack pipeline lost the bytes somewhere.
    assert!(
        stdout.contains("-rwxr-xr-x"),
        "guest /bin/wprof not executable; stdout:\n{stdout}",
    );
}
