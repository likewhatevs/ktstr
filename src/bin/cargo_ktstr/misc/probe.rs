//! Shared probe-loop scaffolding for cargo-ktstr subcommands that
//! locate a `#[ktstr_test]` registration by exec'ing every
//! workspace test binary with a discriminator flag.
//!
//! Both [`super::export::run_export`] and
//! `super::shell::resolve_shell_from_test_entry` walk the same
//! [`super::export::build_test_binaries`] output; they differ only
//! in (a) stdio shape, (b) per-success disposition (forward stderr
//! vs parse JSON + push), and (c) stop-on-first vs walk-all
//! semantics. Two dedicated helpers ([`probe_first`] and
//! [`probe_collect`]) encode the stop-vs-walk choice in the type
//! system — caller's match drops the otherwise-unreachable arm
//! entirely. The caller provides closures for (a) + (b); the
//! shared exit-code categorisation (0 = win, 2 = "registered but
//! rejected", other = "not registered here") and miss-stderr
//! bookkeeping live in [`process_bin`].
//!
//! All items are `pub(crate)` — the dispatch protocol
//! (`--ktstr-export-test` / `--ktstr-shell-test` exit-code
//! contract) is a private agreement between cargo-ktstr (router)
//! and `crate::test_support::dispatch` (dispatcher), not a
//! ktstr-library general capability.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use super::export::build_test_binaries;

/// Aggregate of all-binaries-tried-and-missed. The caller renders
/// a subject-specific message via [`ProbeMiss::render`].
#[derive(Debug)]
pub(crate) struct ProbeMiss {
    pub bins_tried: usize,
    pub rejection_stderr: Option<String>,
    pub last_miss_stderr: String,
}

impl ProbeMiss {
    /// Render the miss diagnostic. `rejection_subject` slots into
    /// the "registered but {rejection_subject}" template (e.g.
    /// `"cannot be exported"`, `"cannot be used for shell mode"`);
    /// the trailing "not found in any workspace test binary" form
    /// is identical across callers.
    pub(crate) fn render(&self, test: &str, rejection_subject: &str) -> String {
        if let Some(reason) = &self.rejection_stderr {
            return format!(
                "test '{test}' is registered but {rejection_subject}:\n{}",
                reason.trim_end(),
            );
        }
        format!(
            "test '{test}' not found in any workspace test binary ({} candidates tried). \
             Last stderr from a candidate:\n{}",
            self.bins_tried,
            self.last_miss_stderr.trim_end(),
        )
    }
}

/// Helper-level error: either setup failed
/// ([`ProbeError::Setup`] — `build_test_binaries` errored or
/// returned empty, exec hit an I/O error, or on_success bubbled
/// up a String) or every binary missed
/// ([`ProbeError::Miss`]).
#[derive(Debug)]
pub(crate) enum ProbeError {
    Setup(String),
    Miss(ProbeMiss),
}

/// Per-binary categorised outcome — only the success arm carries
/// payload. exit-1 / exit-2 / other-nonzero outcomes are bucketed
/// into the caller's `&mut rejection_stderr` / `&mut
/// last_miss_stderr` accumulators by [`process_bin`] before
/// returning [`BinOutcome::Continue`].
enum BinOutcome<T> {
    Success(T),
    Continue,
}

/// Exec one bin via `configure_cmd`, bucket its outcome:
/// exit 0 → call `on_success`, return [`BinOutcome::Success`].
/// exit 2 → set `rejection_stderr` IFF currently None (first-
/// rejection-wins), return [`BinOutcome::Continue`].
/// other non-zero → overwrite `last_miss_stderr` (last-write-
/// wins for miss diagnostics), return [`BinOutcome::Continue`].
/// I/O error spawning the child → [`ProbeError::Setup`].
fn process_bin<T>(
    bin: &Path,
    configure_cmd: &impl Fn(&Path) -> Command,
    on_success: &impl Fn(&Path, &Output) -> Result<T, String>,
    rejection_stderr: &mut Option<String>,
    last_miss_stderr: &mut String,
) -> Result<BinOutcome<T>, ProbeError> {
    let mut cmd = configure_cmd(bin);
    let out = cmd
        .output()
        .map_err(|e| ProbeError::Setup(format!("exec {}: {e}", bin.display())))?;
    if out.status.success() {
        let value = on_success(bin, &out).map_err(ProbeError::Setup)?;
        return Ok(BinOutcome::Success(value));
    }
    let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
    // Exit 2 = "registered but rejected here" (host_only,
    // bpf_map_write, KernelBuiltin, etc.). ALWAYS the most
    // informative outcome — save the FIRST and keep probing;
    // other candidates might still admit the test. Exit 1 (and
    // other non-2 non-zero) = "not registered here"; overwrite
    // each pass so the operator sees the most-recent miss
    // diagnostic.
    if out.status.code() == Some(2) {
        if rejection_stderr.is_none() {
            *rejection_stderr = Some(stderr);
        }
    } else {
        *last_miss_stderr = stderr;
    }
    Ok(BinOutcome::Continue)
}

/// Empty-bins diagnostic shared by [`probe_first`] /
/// [`probe_collect`] + their `_with_bins_*` cores. The message
/// exactly matches the pre-refactor `run_export` /
/// `resolve_shell_from_test_entry` text so failure shape is
/// preserved across the refactor.
const EMPTY_BINS_SETUP_ERROR: &str = "cargo build --tests produced no executable artifacts; \
     ensure the workspace has at least one [[test]] target or \
     a [lib]/[bin] with #[cfg(test)] tests";

/// Walk every workspace test binary, stop at the FIRST that
/// returns Ok via `on_success`, and return its value. Used by
/// `run_export` (first-match-wins semantics).
///
/// `package` + `release` flow straight to
/// [`build_test_binaries`]; the `configure_cmd` closure receives
/// each candidate's path and returns a fully-configured `Command`
/// (with the discriminator flag + stdio shape set by the caller).
///
/// Errors: [`ProbeError::Setup`] for build failure / empty
/// executables / exec I/O failure / on_success error.
/// [`ProbeError::Miss`] when every binary returned non-zero.
pub(crate) fn probe_first<T>(
    package: Option<&str>,
    release: bool,
    configure_cmd: impl Fn(&Path) -> Command,
    on_success: impl Fn(&Path, &Output) -> Result<T, String>,
) -> Result<T, ProbeError> {
    let bins = build_test_binaries(package, release).map_err(ProbeError::Setup)?;
    probe_first_with_bins(&bins, configure_cmd, on_success)
}

/// Walk every workspace test binary, call `on_success` on each
/// hit, return the accumulated Vec. Used by
/// `resolve_shell_from_test_entry` (walk-all + ambiguity-bail
/// semantics — caller checks `Vec::len()` after the walk).
///
/// See [`probe_first`] for arg + error semantics.
pub(crate) fn probe_collect<T>(
    package: Option<&str>,
    release: bool,
    configure_cmd: impl Fn(&Path) -> Command,
    on_success: impl Fn(&Path, &Output) -> Result<T, String>,
) -> Result<Vec<T>, ProbeError> {
    let bins = build_test_binaries(package, release).map_err(ProbeError::Setup)?;
    probe_collect_with_bins(&bins, configure_cmd, on_success)
}

/// Walk an exact caller-supplied set of already-built test executables.
///
/// The verifier dispatcher obtains this set directly from the reserved
/// `cargo nextest run --no-run` Cargo JSON stream. Reusing those paths is
/// load-bearing: rebuilding via [`build_test_binaries`] would be unscoped
/// from nextest's package/feature selection and could both miss declarations
/// and trigger a second workspace build. Other probe consumers should keep
/// using [`probe_collect`].
pub(crate) fn probe_collect_from_bins<T>(
    bins: &[PathBuf],
    configure_cmd: impl Fn(&Path) -> Command,
    on_success: impl Fn(&Path, &Output) -> Result<T, String>,
) -> Result<Vec<T>, ProbeError> {
    probe_collect_with_bins(bins, configure_cmd, on_success)
}

/// Unit-testable core of [`probe_first`]: bins pre-built by
/// caller. Separated so the loop + bookkeeping + Setup/Miss
/// dispatch can be exercised without spawning a real
/// `cargo build --tests`.
///
/// PRIVATE on purpose: callers with an authoritative prebuilt set use
/// [`probe_collect_from_bins`]; ordinary subcommands use [`probe_first`] or
/// [`probe_collect`] so they cannot accidentally bypass the canonical build.
/// Tests below reach this fn via Rust's same-module sibling visibility.
fn probe_first_with_bins<T>(
    bins: &[PathBuf],
    configure_cmd: impl Fn(&Path) -> Command,
    on_success: impl Fn(&Path, &Output) -> Result<T, String>,
) -> Result<T, ProbeError> {
    if bins.is_empty() {
        return Err(ProbeError::Setup(EMPTY_BINS_SETUP_ERROR.to_string()));
    }
    let mut rejection_stderr: Option<String> = None;
    let mut last_miss_stderr = String::new();
    for bin in bins {
        match process_bin(
            bin,
            &configure_cmd,
            &on_success,
            &mut rejection_stderr,
            &mut last_miss_stderr,
        )? {
            BinOutcome::Success(value) => return Ok(value),
            BinOutcome::Continue => continue,
        }
    }
    Err(ProbeError::Miss(ProbeMiss {
        bins_tried: bins.len(),
        rejection_stderr,
        last_miss_stderr,
    }))
}

/// Unit-testable core of [`probe_collect`]: bins pre-built by
/// caller. See [`probe_first_with_bins`] for the
/// PRIVATE-on-purpose rationale.
fn probe_collect_with_bins<T>(
    bins: &[PathBuf],
    configure_cmd: impl Fn(&Path) -> Command,
    on_success: impl Fn(&Path, &Output) -> Result<T, String>,
) -> Result<Vec<T>, ProbeError> {
    if bins.is_empty() {
        return Err(ProbeError::Setup(EMPTY_BINS_SETUP_ERROR.to_string()));
    }
    let mut collected: Vec<T> = Vec::new();
    let mut rejection_stderr: Option<String> = None;
    let mut last_miss_stderr = String::new();
    for bin in bins {
        match process_bin(
            bin,
            &configure_cmd,
            &on_success,
            &mut rejection_stderr,
            &mut last_miss_stderr,
        )? {
            BinOutcome::Success(value) => collected.push(value),
            BinOutcome::Continue => continue,
        }
    }
    if !collected.is_empty() {
        return Ok(collected);
    }
    Err(ProbeError::Miss(ProbeMiss {
        bins_tried: bins.len(),
        rejection_stderr,
        last_miss_stderr,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fake_bin(idx: usize) -> PathBuf {
        PathBuf::from(format!("/fake/bin{idx}"))
    }

    /// Walking all bins, every Ok(T) is appended in iteration
    /// order. Pins the Vec append contract (caller relies on it
    /// for the shell-mode ambiguity-bail path which checks
    /// `matches.len() > 1`).
    #[test]
    fn probe_collect_with_bins_appends_in_order() {
        let bins = vec![fake_bin(0), fake_bin(1)];
        let configure_cmd = |_bin: &Path| Command::new("true");
        let on_success =
            |bin: &Path, _out: &Output| -> Result<PathBuf, String> { Ok(bin.to_path_buf()) };
        let result = probe_collect_with_bins(&bins, configure_cmd, on_success)
            .expect("two successes should collect");
        assert_eq!(result, vec![fake_bin(0), fake_bin(1)]);
    }

    /// First success short-circuits the remaining bins. Pins
    /// the export-style "first-match-wins + skip the rest"
    /// contract — without this, an operator with multiple
    /// matching test binaries would observe surprising spawn
    /// counts.
    #[test]
    fn probe_first_with_bins_short_circuits_after_first_success() {
        let bins = vec![fake_bin(0), fake_bin(1), fake_bin(2)];
        let spawn_count = std::cell::Cell::new(0usize);
        let configure_cmd = |_bin: &Path| {
            spawn_count.set(spawn_count.get() + 1);
            Command::new("true")
        };
        let on_success = |_bin: &Path, _out: &Output| -> Result<(), String> { Ok(()) };
        probe_first_with_bins(&bins, configure_cmd, on_success).expect("first success");
        assert_eq!(spawn_count.get(), 1, "must not spawn after first success");
    }

    /// Empty bin list surfaces [`ProbeError::Setup`] with the
    /// shared empty-bins diagnostic. Pins the "no executable
    /// artifacts" wording that the operator may grep for.
    #[test]
    fn probe_first_with_bins_empty_returns_setup_error() {
        let bins: Vec<PathBuf> = vec![];
        let configure_cmd = |_bin: &Path| Command::new("true");
        let on_success = |_bin: &Path, _out: &Output| -> Result<(), String> { Ok(()) };
        match probe_first_with_bins(&bins, configure_cmd, on_success) {
            Err(ProbeError::Setup(msg)) => assert!(
                msg.contains("no executable artifacts"),
                "expected empty-bins diagnostic, got {msg:?}",
            ),
            _ => panic!("expected Setup error"),
        }
    }

    /// `probe_collect_with_bins` empty-bins error mirrors
    /// `probe_first_with_bins` — both helpers share
    /// [`EMPTY_BINS_SETUP_ERROR`].
    #[test]
    fn probe_collect_with_bins_empty_returns_setup_error() {
        let bins: Vec<PathBuf> = vec![];
        let configure_cmd = |_bin: &Path| Command::new("true");
        let on_success = |_bin: &Path, _out: &Output| -> Result<(), String> { Ok(()) };
        match probe_collect_with_bins(&bins, configure_cmd, on_success) {
            Err(ProbeError::Setup(msg)) => assert!(msg.contains("no executable artifacts")),
            _ => panic!("expected Setup error"),
        }
    }

    /// Exit-2 rejection from the FIRST exit-2 bin wins; later
    /// exit-2 bins do NOT overwrite. Exit-1 (and other
    /// non-2-non-zero) miss diagnostics overwrite per iteration
    /// (last-write-wins). Pins both contracts on a single
    /// fixture so a regression in either flips the assertion
    /// loudly.
    #[test]
    fn probe_collect_with_bins_exit_2_first_wins_exit_1_overwrites() {
        let bins = vec![fake_bin(0), fake_bin(1), fake_bin(2), fake_bin(3)];
        let configure_cmd = |bin: &Path| {
            let suffix = bin.file_name().unwrap().to_str().unwrap().to_string();
            let (code, stderr) = match suffix.as_str() {
                "bin0" => (2, "REJECTED_A"),
                "bin1" => (2, "REJECTED_B"),
                "bin2" => (1, "MISS_C"),
                "bin3" => (1, "MISS_D"),
                _ => unreachable!(),
            };
            let mut cmd = Command::new("sh");
            cmd.args(["-c", &format!("printf '%s' '{stderr}' >&2; exit {code}")]);
            cmd.stderr(std::process::Stdio::piped());
            cmd
        };
        let on_success = |_bin: &Path, _out: &Output| -> Result<(), String> { Ok(()) };
        match probe_collect_with_bins(&bins, configure_cmd, on_success) {
            Err(ProbeError::Miss(miss)) => {
                assert_eq!(miss.bins_tried, 4);
                assert_eq!(
                    miss.rejection_stderr.as_deref(),
                    Some("REJECTED_A"),
                    "first rejection must win — REJECTED_B should be ignored",
                );
                assert_eq!(
                    miss.last_miss_stderr, "MISS_D",
                    "last miss must overwrite — MISS_C should be replaced",
                );
            }
            _ => panic!("expected Miss with rejection + last-miss"),
        }
    }

    /// All-miss with no exit-2 produces ProbeMiss with the
    /// correct bin count and `rejection_stderr = None`. Pins
    /// the "no rejection_stderr → operator sees the last-miss
    /// diagnostic" contract.
    #[test]
    fn probe_collect_with_bins_all_miss_no_rejection_returns_probe_miss() {
        let bins = vec![fake_bin(0), fake_bin(1), fake_bin(2)];
        let configure_cmd = |_bin: &Path| Command::new("false");
        let on_success = |_bin: &Path, _out: &Output| -> Result<(), String> { Ok(()) };
        match probe_collect_with_bins(&bins, configure_cmd, on_success) {
            Err(ProbeError::Miss(miss)) => {
                assert_eq!(miss.bins_tried, 3);
                assert!(
                    miss.rejection_stderr.is_none(),
                    "no exit-2 → rejection_stderr must stay None",
                );
            }
            _ => panic!("expected Miss"),
        }
    }
}
