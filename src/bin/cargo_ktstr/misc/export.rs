//! `cargo ktstr export` — package a registered test as a self-
//! extracting `.run` reproducer.
//!
//! The exporter cannot run inside cargo-ktstr because the test
//! registry it needs (the `#[ktstr_test]` distributed-slice) lives
//! in user-crate test binaries, not here. [`run_export`] therefore
//! builds every workspace test binary via
//! [`build_test_binaries`] and exec's each in turn with
//! `--ktstr-export-test=NAME`, surfacing the first binary that
//! exits 0 as the winner. All-fail surfaces the most informative
//! per-binary stderr (exit-2 "rejected" preferred over exit-1
//! "not registered"), so the operator sees the actual rejection
//! reason rather than N×"missing" lines.

use std::path::PathBuf;
use std::process::Command;

use super::probe::{ProbeError, probe_first};

/// Route `cargo ktstr export <NAME>` to the test binary that owns
/// the named `#[ktstr_test]` registration. cargo-ktstr cannot embed
/// itself into the .run file because it has no `#[ktstr_test]`
/// entries from the user's crate — only the test binary that
/// links against the user's code carries the registry. The router
/// builds every workspace test binary, then exec's each in turn
/// with `--ktstr-export-test=NAME`. The first binary that exits 0
/// wins; all-fail surfaces the per-binary stderrs so the operator
/// can see why the lookup missed (typically: typoed test name).
///
/// `package` (when `Some`) restricts the build via
/// `cargo build --tests --package <NAME>` — necessary in
/// multi-package workspaces where a test name might exist in
/// multiple packages and the operator wants a deterministic
/// resolution.
///
/// `release: true` builds with `--release` so the embedded test
/// binary matches the profile the operator is running. Mismatched
/// profiles can produce a `.run` whose embedded binary's threshold
/// behavior differs from the test runs the operator is reproducing
/// from.
pub(crate) fn run_export(
    test: String,
    output: Option<PathBuf>,
    package: Option<String>,
    release: bool,
) -> Result<(), String> {
    let test_flag = format!("--ktstr-export-test={test}");
    // Resolve relative paths against cwd BEFORE the probe loop so
    // the test binary writes to the operator's pwd, not its own
    // (the binary lives under target/debug/deps/...). Done once
    // outside the loop so a transient cwd change between binaries
    // can't desync the per-binary output target.
    let output_flag = match output.as_deref() {
        Some(o) => {
            let abs = if o.is_absolute() {
                o.to_path_buf()
            } else {
                std::env::current_dir()
                    .map_err(|e| format!("resolve cwd for --output: {e}"))?
                    .join(o)
            };
            Some(format!("--ktstr-export-output={}", abs.display()))
        }
        None => None,
    };

    let configure_cmd = |bin: &std::path::Path| {
        let mut cmd = Command::new(bin);
        cmd.arg(&test_flag);
        if let Some(o) = &output_flag {
            cmd.arg(o);
        }
        cmd.stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::inherit())
            // Capture stderr so per-candidate "no registered test
            // named X" diagnostics don't spam the operator's
            // terminal for every binary we try. Winner's stderr
            // is forwarded in on_success below; on full miss the
            // probe helper surfaces the exit-2 rejection stderr
            // (or last exit-1 stderr) via ProbeMiss.
            .stderr(std::process::Stdio::piped());
        cmd
    };

    let on_success = |_bin: &std::path::Path, out: &std::process::Output| -> Result<(), String> {
        // Forward the winner's stderr so the "wrote ..." line (and
        // any operator-visible diagnostics) reach the user's
        // terminal.
        std::io::Write::write_all(&mut std::io::stderr(), &out.stderr)
            .map_err(|e| format!("forward winner stderr: {e}"))?;
        Ok(())
    };

    match probe_first(package.as_deref(), release, configure_cmd, on_success) {
        Ok(()) => Ok(()),
        Err(ProbeError::Setup(msg)) => Err(msg),
        Err(ProbeError::Miss(miss)) => Err(miss.render(&test, "cannot be exported")),
    }
}

/// Assemble the `cargo build --tests --message-format=json` argv,
/// forwarding the cargo features the running cargo-ktstr was built with.
///
/// `cargo build --tests` rebuilds every test target AND its
/// `CARGO_BIN_EXE` dependencies — cargo-ktstr itself included — over the
/// shared `target/<profile>/cargo-ktstr` slot. The ktstr crate has no
/// default features and `wprof` is opt-in (default-off), so omitting the
/// feature flags rebuilds at `{}`, stripping the embedded wprof blob from
/// the very binary that sibling integration tests spawn via
/// `env!("CARGO_BIN_EXE_cargo-ktstr")` — surfacing downstream as a
/// spurious `/bin/wprof: No such file` inside the guest (the no-wprof
/// binary's shell-mode path compiles the wprof include out, no error).
/// Forwarding the running binary's exact feature set makes the rebuild a
/// fingerprint hit against the already-built artifacts — no downgrade.
fn build_test_binaries_argv(package: Option<&str>, release: bool) -> Vec<String> {
    let mut argv = vec![
        "build".to_string(),
        "--tests".to_string(),
        "--message-format=json".to_string(),
    ];
    let mut forwarded: Vec<&str> = Vec::new();
    if cfg!(feature = "wprof") {
        forwarded.push("wprof");
    }
    if cfg!(feature = "integration") {
        forwarded.push("integration");
    }
    if !forwarded.is_empty() {
        argv.push("--features".to_string());
        argv.push(forwarded.join(","));
    }
    if let Some(p) = package {
        argv.push("--package".to_string());
        argv.push(p.to_string());
    }
    if release {
        argv.push("--release".to_string());
    }
    argv
}

/// Compile the workspace's test binaries via
/// `cargo build --tests --message-format=json` and collect the
/// resulting executable paths.
///
/// Filters to artifacts where `executable != null` AND either
/// `target.kind` contains `"test"` (integration tests under
/// `tests/`) or `profile.test == true` (unit-test binaries built
/// from `[lib]` / `[bin]` targets). Both shapes carry the
/// `#[ktstr_test]` distributed-slice registry that the export
/// dispatcher reads, so both are valid candidates.
pub(crate) fn build_test_binaries(
    package: Option<&str>,
    release: bool,
) -> Result<Vec<PathBuf>, String> {
    let mut cmd = Command::new("cargo");
    cmd.args(build_test_binaries_argv(package, release));
    cmd.stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::inherit());

    let out = cmd
        .output()
        .map_err(|e| format!("spawn cargo build --tests: {e}"))?;
    if !out.status.success() {
        return Err(format!(
            "cargo build --tests failed (exit {})",
            out.status.code().unwrap_or(-1),
        ));
    }

    let stdout = String::from_utf8_lossy(&out.stdout);
    let mut bins: Vec<PathBuf> = Vec::new();
    for line in stdout.lines() {
        let Ok(msg) = serde_json::from_str::<serde_json::Value>(line) else {
            continue;
        };
        if msg.get("reason").and_then(|r| r.as_str()) != Some("compiler-artifact") {
            continue;
        }
        let Some(exe) = msg.get("executable").and_then(|e| e.as_str()) else {
            continue;
        };
        let kinds: Vec<&str> = msg
            .get("target")
            .and_then(|t| t.get("kind"))
            .and_then(|k| k.as_array())
            .map(|arr| arr.iter().filter_map(|v| v.as_str()).collect())
            .unwrap_or_default();
        let is_test_target = kinds.contains(&"test");
        let is_unit_test = msg
            .get("profile")
            .and_then(|p| p.get("test"))
            .and_then(|t| t.as_bool())
            == Some(true);
        if is_test_target || is_unit_test {
            bins.push(PathBuf::from(exe));
        }
    }
    bins.sort();
    bins.dedup();
    Ok(bins)
}

#[cfg(test)]
mod tests {
    use super::build_test_binaries_argv;

    /// Regression pin for the wprof-clobber bug: the test-binary rebuild
    /// must forward the features the running cargo-ktstr carries, so it
    /// cannot downgrade the shared `target/<profile>/cargo-ktstr` slot (a
    /// `CARGO_BIN_EXE` dependency of every test) to a wprof-less binary
    /// that sibling tests would then spawn. Exercised by both CI legs: the
    /// no-wprof leg pins the absent-wprof path, the wprof leg pins
    /// forwarding.
    #[test]
    fn argv_forwards_running_binary_features() {
        let argv = build_test_binaries_argv(None, false);
        assert_eq!(argv[0], "build");
        assert_eq!(argv[1], "--tests");
        assert_eq!(argv[2], "--message-format=json");
        let features = argv
            .iter()
            .position(|a| a == "--features")
            .map(|i| argv[i + 1].clone());
        match (cfg!(feature = "wprof"), cfg!(feature = "integration")) {
            (false, false) => assert!(
                features.is_none(),
                "a no-feature build must not pass --features (got {features:?})"
            ),
            (want_wprof, want_integration) => {
                let f = features.expect("a feature build must forward --features");
                let set: Vec<&str> = f.split(',').collect();
                assert_eq!(
                    set.contains(&"wprof"),
                    want_wprof,
                    "wprof forwarding (got {f:?})"
                );
                assert_eq!(
                    set.contains(&"integration"),
                    want_integration,
                    "integration forwarding (got {f:?})"
                );
            }
        }
    }

    /// `--package` and `--release` are threaded through verbatim.
    #[test]
    fn argv_threads_package_and_release() {
        let argv = build_test_binaries_argv(Some("scx-ktstr"), true);
        assert!(
            argv.windows(2)
                .any(|w| w[0] == "--package" && w[1] == "scx-ktstr"),
            "argv: {argv:?}"
        );
        assert!(argv.iter().any(|a| a == "--release"), "argv: {argv:?}");
    }
}
