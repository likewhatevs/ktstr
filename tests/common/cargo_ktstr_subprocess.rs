#![allow(dead_code)]

//! Subprocess helpers for integration tests that exercise
//! `cargo ktstr` CLI subcommands by spawning the built
//! `cargo-ktstr` binary as a child process.
//!
//! The integration-test pattern is: build the CLI via
//! `env!("CARGO_BIN_EXE_cargo-ktstr")`, spawn it with the desired
//! args, capture stdout + stderr via `std::process::Output`,
//! assert on the captured bytes. Used by shell-mode lifecycle
//! and topology-inheritance tests where the assertion target is
//! the CLI's externally-visible behavior (banner, exit code,
//! routed console writes) rather than ktstr's in-process API.

#[cfg(unix)]
pub const CARGO_KTSTR_BINARY: &str = env!("CARGO_BIN_EXE_cargo-ktstr");

/// Spawn `cargo ktstr shell --no-perf-mode --test <fixture>
/// --exec <payload>` and return the captured Output. Lets
/// cargo-ktstr resolve the guest kernel via its own kernel-build
/// cache rather than passing an operator-style `--kernel` path —
/// ktstr supplies the kernel itself.
///
/// The `--no-perf-mode` flag is added unconditionally — integration
/// tests run in shared CI environments where the
/// performance-mode LLC-EX lock causes nextest contention; the
/// non-perf path is sufficient for shell-mode lifecycle pins.
#[cfg(unix)]
pub fn run_cargo_ktstr_shell(fixture: &str, exec: &str) -> std::process::Output {
    run_cargo_ktstr_shell_inner(fixture, exec, None)
}

/// Like [`run_cargo_ktstr_shell`] but with an explicit
/// `--exec-timeout` (humantime, e.g. `"3s"`). Used by the
/// panic-less-hang test: a `--exec` payload that never exits
/// must be force-killed at the deadline rather than blocking the BSP
/// run loop forever.
#[cfg(unix)]
pub fn run_cargo_ktstr_shell_with_timeout(
    fixture: &str,
    exec: &str,
    exec_timeout: &str,
) -> std::process::Output {
    run_cargo_ktstr_shell_inner(fixture, exec, Some(exec_timeout))
}

#[cfg(unix)]
fn run_cargo_ktstr_shell_inner(
    fixture: &str,
    exec: &str,
    exec_timeout: Option<&str>,
) -> std::process::Output {
    let mut cmd = std::process::Command::new(CARGO_KTSTR_BINARY);
    cmd.arg("ktstr")
        .arg("shell")
        .arg("--no-perf-mode")
        .arg("--test")
        .arg(fixture)
        .arg("--exec")
        .arg(exec);
    if let Some(t) = exec_timeout {
        cmd.arg("--exec-timeout").arg(t);
    }
    // Reuse THIS already-built test binary for `--test` fixture
    // resolution instead of forcing cargo-ktstr to re-run
    // `cargo build --tests` — which, spawned from nextest (TEST
    // profile), re-compiles the whole test set and blows past the
    // slow-timeout (see build_test_binaries). current_exe() is the
    // nextest-built test binary that carries the fixture's
    // `#[ktstr_test]` registration.
    cmd.env(
        "KTSTR_TEST_BINARY",
        std::env::current_exe().expect("current_exe for KTSTR_TEST_BINARY"),
    )
    .output()
    .expect("spawn cargo-ktstr shell --test")
}

/// Combined stdout + stderr from a subprocess Output, with a
/// visible STDERR delimiter so failure diagnostics make the
/// stream split unambiguous. The shell-mode lifecycle marker
/// pattern routes both init-side writes (via `/proc/1/fd/1`)
/// and the payload's stdout through the same init-fd-1 console
/// pipeline (virtio-console `/dev/hvc0` when available, COM2
/// `/dev/ttyS1` otherwise), so most assertions search the
/// combined view; the
/// delimiter survives in panic messages so a reader can tell
/// which stream the marker landed in.
pub fn combined_output(out: &std::process::Output) -> String {
    format!(
        "{}\n--- STDERR ---\n{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    )
}
