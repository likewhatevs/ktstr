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

/// Assemble the workspace-wide (or explicitly package-scoped)
/// `cargo build --tests --message-format=json` argv.
///
/// `cargo build --tests` rebuilds every test target AND its
/// `CARGO_BIN_EXE` dependencies — cargo-ktstr itself included — over the
/// shared `target/<profile>/cargo-ktstr` slot. `wprof` is opt-in
/// (default-off), so omitting that capability can strip the embedded wprof
/// blob from the very binary that sibling integration tests spawn via
/// `env!("CARGO_BIN_EXE_cargo-ktstr")` — surfacing downstream as a
/// spurious `/bin/wprof: No such file` inside the guest (the no-wprof
/// binary's shell-mode path compiles the wprof include out, no error).
/// [`build_test_binaries`] preserves those capabilities only after metadata
/// proves that ktstr itself is a selected workspace member, and then uses
/// package-qualified selectors. They must never be forwarded here as
/// unqualified features into a consumer workspace.
fn build_test_binaries_argv(package: Option<&str>, release: bool) -> Vec<String> {
    let mut argv = vec![
        "build".to_string(),
        "--tests".to_string(),
        "--message-format=json".to_string(),
    ];
    if let Some(p) = package {
        argv.push("--package".to_string());
        argv.push(p.to_string());
    } else {
        // Probes promise to inspect every workspace registry. Cargo's bare
        // build selection is only the workspace's default members, which can
        // omit scheduler/test packages explicitly kept out of default-members.
        argv.push("--workspace".to_string());
    }
    if release {
        argv.push("--release".to_string());
    }
    argv
}

/// Assemble a registry-discovery build from an already-prepared
/// nextest/cargo-llvm-cov-nextest argv.
///
/// Unlike the workspace-wide export/shell probe above, `--relevant` must
/// inspect exactly the registries the eventual run selected. `cargo test
/// --no-run` preserves Cargo's default-member behavior and makes explicit
/// target selectors (`--lib`, `--test`, ...) select test harnesses rather
/// than ordinary non-test artifacts. The caller's native `--release` is
/// threaded separately because cargo-ktstr consumes it before constructing
/// the nextest passthrough.
fn contextual_test_binaries_argv(args: &[String], release: bool) -> Vec<String> {
    let mut argv = vec![
        "test".to_string(),
        "--no-run".to_string(),
        "--message-format=json".to_string(),
    ];
    if release {
        argv.push("--release".to_string());
    }
    argv.extend(crate::feature_discovery::test_registry_build_options(args));
    argv
}

fn test_binary_override() -> Option<Vec<PathBuf>> {
    std::env::var("KTSTR_TEST_BINARY")
        .ok()
        .filter(|binary| !binary.is_empty())
        .map(|binary| vec![PathBuf::from(binary)])
}

fn execute_test_binary_build(argv: Vec<String>, description: &str) -> Result<Vec<PathBuf>, String> {
    let mut cmd = Command::new("cargo");
    cmd.args(argv);
    cmd.stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::inherit());

    let out = cmd
        .output()
        .map_err(|e| format!("spawn {description}: {e}"))?;
    if !out.status.success() {
        return Err(format!(
            "{description} failed (exit {})",
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
    // Reuse an already-built test binary instead of re-running
    // `cargo build --tests` when the caller points us at one via
    // KTSTR_TEST_BINARY (e.g. an integration test already running inside a
    // freshly-built test binary under nextest, which sets it to its own
    // `current_exe`). Without this, resolving a `cargo ktstr shell --test`
    // fixture from inside a nextest run triggers a cold `cargo build
    // --tests` (DEV profile) that re-compiles the whole test set — the
    // running test binary was built TEST profile, so the fingerprints miss
    // — over the shared target slot, blowing past nextest's slow-timeout
    // (the test is SIGTERM'd mid-`Compiling`). The named binary carries the
    // same `#[ktstr_test]` distributed-slice registry the probe reads, so
    // resolution is identical, just build-free.
    if let Some(binaries) = test_binary_override() {
        return Ok(binaries);
    }
    let mut self_features = Vec::new();
    if cfg!(feature = "wprof") {
        self_features.push("wprof");
    }
    if cfg!(feature = "integration") {
        self_features.push("integration");
    }
    let argv = crate::feature_discovery::augment_test_features_with_workspace_package_features(
        build_test_binaries_argv(package, release),
        env!("CARGO_PKG_NAME"),
        &self_features,
    )
    .map_err(|error| format!("discover ktstr test features: {error}"))?;
    execute_test_binary_build(argv, "cargo build --tests")
}

/// Compile exactly the test registries selected by an already-prepared
/// nextest/cargo-llvm-cov-nextest argv.
///
/// This is the registry source for `--relevant`. The same metadata-driven
/// feature preparation that shapes the final test run has already augmented
/// `args`, so this path must not infer a second, potentially different feature
/// set. Package/default-member selection, explicit exclusions, target
/// selectors/triple, Cargo profile, and every explicit feature mode are
/// normalized onto `cargo test --no-run` by
/// [`crate::feature_discovery::test_registry_build_options`].
pub(crate) fn build_contextual_test_binaries(
    args: &[String],
    release: bool,
) -> Result<Vec<PathBuf>, String> {
    if let Some(binaries) = test_binary_override() {
        return Ok(binaries);
    }
    execute_test_binary_build(
        contextual_test_binaries_argv(args, release),
        "cargo test --no-run for --relevant registry discovery",
    )
}

#[cfg(test)]
mod tests {
    use super::{build_test_binaries_argv, contextual_test_binaries_argv};

    /// An unscoped probe promises every workspace test registry, including
    /// packages excluded from Cargo's default-members. Compile-time features
    /// of the running cargo-ktstr are deliberately absent here: metadata may
    /// add them later only as safe `ktstr/FEATURE` selectors when ktstr itself
    /// is a selected workspace member.
    #[test]
    fn unscoped_argv_selects_workspace_without_unqualified_self_features() {
        let argv = build_test_binaries_argv(None, false);
        assert_eq!(
            argv,
            ["build", "--tests", "--message-format=json", "--workspace"].map(ToString::to_string),
        );
        assert!(
            !argv.iter().any(|argument| argument == "--features"),
            "consumer workspaces must never receive unqualified cargo-ktstr features: {argv:?}",
        );
    }

    /// `--package` remains narrower than the unscoped workspace build and
    /// `--release` is threaded through verbatim.
    #[test]
    fn argv_threads_package_and_release() {
        let argv = build_test_binaries_argv(Some("scx-ktstr"), true);
        assert!(
            argv.windows(2)
                .any(|w| w[0] == "--package" && w[1] == "scx-ktstr"),
            "argv: {argv:?}"
        );
        assert!(argv.iter().any(|a| a == "--release"), "argv: {argv:?}");
        assert!(
            !argv.iter().any(|argument| argument == "--workspace"),
            "an explicit package probe must not widen to the workspace: {argv:?}",
        );
        assert!(
            !argv.iter().any(|argument| argument == "--features"),
            "self features are added only after metadata proves package scope: {argv:?}",
        );
    }

    #[test]
    fn contextual_argv_preserves_composite_features_and_no_default_mode() {
        let argv = contextual_test_binaries_argv(
            &[
                "--features",
                "alpha,beta gamma",
                "-Fdelta,epsilon",
                "--no-default-features",
                "-E",
                "test(irrelevant_runtime_filter)",
                "--",
                "--ignored",
            ]
            .map(ToString::to_string),
            false,
        );
        assert_eq!(
            argv,
            [
                "test",
                "--no-run",
                "--message-format=json",
                "--features",
                "alpha,beta gamma",
                "-Fdelta,epsilon",
                "--no-default-features",
            ]
            .map(ToString::to_string),
            "composite expressions remain one Cargo value and the binary suffix is opaque",
        );
    }

    #[test]
    fn contextual_argv_keeps_package_exclusions_and_effective_build_context() {
        let argv = contextual_test_binaries_argv(
            &[
                "-p",
                "selected",
                "--workspace",
                "--exclude",
                "ordinary",
                "--exclude-from-test=coverage-only",
                "--target",
                "aarch64-unknown-linux-gnu",
                "--cargo-profile",
                "ci",
                "--test=scheduler_registry",
            ]
            .map(ToString::to_string),
            true,
        );
        assert_eq!(
            argv,
            [
                "test",
                "--no-run",
                "--message-format=json",
                "--release",
                "-p",
                "selected",
                "--workspace",
                "--exclude",
                "ordinary",
                "--exclude=coverage-only",
                "--target",
                "aarch64-unknown-linux-gnu",
                "--profile",
                "ci",
                "--test=scheduler_registry",
            ]
            .map(ToString::to_string),
            "nextest and llvm-cov-nextest controls must select the same registry build",
        );
    }

    #[test]
    fn contextual_argv_preserves_explicit_all_features_without_inference() {
        let argv = contextual_test_binaries_argv(
            &["--all-features", "--exclude-from-report", "report-only"].map(ToString::to_string),
            false,
        );
        assert_eq!(
            argv,
            [
                "test",
                "--no-run",
                "--message-format=json",
                "--all-features",
            ]
            .map(ToString::to_string),
            "explicit --all-features remains authoritative and report-only coverage flags do not \
             shape the test registry",
        );
    }
}
