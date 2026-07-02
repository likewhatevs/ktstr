//! `cargo ktstr verifier` subcommand: thin wrapper around
//! `cargo nextest run` filtered to the `verifier/` test-name prefix.
//!
//! Each test binary that links the `ktstr` crate's `test_support`
//! module and has at least one `declare_scheduler!` declaration
//! emits one nextest test per
//! (declared scheduler × kernel-list entry × accepted gauntlet
//! preset) cell. The lister + cell handler live in
//! `src/test_support/dispatch.rs::list_verifier_cells_all` and
//! `run_verifier_cell`. Nextest provides per-cell parallelism,
//! retries, and failure isolation; this dispatcher resolves the
//! `--kernel` argument into the `KTSTR_KERNEL_LIST` env-var matrix
//! dimension the test binary's lister walks, plumbs `--raw` via
//! `KTSTR_VERIFIER_RAW`, and spawns nextest with a verifier-prefix
//! filter expression. The trailing `args` are forwarded verbatim to that
//! `cargo nextest run` — feature selection (`cargo ktstr verifier
//! --features integration,wprof`, no `--` separator) reaches the
//! integration-gated `declare_scheduler!` cell build; the
//! scheduler-under-test builds release by default.
//!
//! `KTSTR_KERNEL_LIST` is ALWAYS populated by this dispatcher — even
//! with no `--kernel` flag the dispatcher auto-discovers one kernel
//! and synthesizes a single-entry list with a path-derived label.
//! That keeps the test-binary cell handler's lookup path unified
//! (always look up by label in the list, never fall through to a
//! resolve_test_kernel single-kernel fallback that would silently
//! run a cell against an unrelated kernel).

use std::path::PathBuf;
use std::process::Command;

use crate::kernel::{
    encode_kernel_list, path_kernel_label, resolve_kernel_image, resolve_kernel_set,
};

/// Dispatch the `cargo ktstr verifier` subcommand.
///
/// The trailing `args` are forwarded verbatim to the inner
/// `cargo nextest run` — the path for feature selection
/// (`cargo ktstr verifier --features integration,wprof`), which the
/// integration-gated `declare_scheduler!` cells require to compile.
/// Without the feature passthrough the verifier-prefix filter matches
/// only the module's unit tests and the command collects no verifier
/// statistics.
///
/// `profile` is the scheduler-under-test's cargo BUILD profile
/// (`--profile <NAME>`): set as `KTSTR_SCHEDULER_PROFILE` so
/// [`ktstr::build_and_find_binary`] passes `cargo build -p <scheduler>
/// --profile <name>`. Omitted, the scheduler builds `release` (that
/// default lives in `build_and_find_binary`). `nextest_profile` is the
/// NEXTEST test profile (`--nextest-profile <NAME>`), emitted as
/// nextest's own `--profile <NAME>` before the user's trailing args.
pub(crate) fn run_verifier(
    kernel: Vec<String>,
    raw: bool,
    profile: Option<String>,
    nextest_profile: Option<String>,
    args: Vec<String>,
) -> Result<(), String> {
    let mut cmd = Command::new("cargo");
    // `--run-ignored all` is load-bearing: verifier cells are emitted
    // IGNORE-GATED (like gauntlet variants). `list_verifier_cells_all`
    // emits every `verifier/<sched>/<kernel>/<preset>` line
    // unconditionally — including on nextest's `--list --ignored` pass —
    // so nextest classifies each cell as ignored. Without opting in,
    // `cargo nextest run` skips every cell and this command runs only the
    // non-ignored `verifier::tests::*` unit tests (0 cells collected).
    // `all` (not `only`) keeps those unit tests running alongside the
    // cells.
    cmd.args([
        "nextest",
        "run",
        "--run-ignored",
        "all",
        "-E",
        "test(/^verifier/)",
    ]);
    // `--nextest-profile <NAME>` selects the NEXTEST test profile;
    // nextest's own flag for it is `--profile`. Emitted before the user's
    // trailing args so a passthrough token can't shadow it.
    if let Some(np) = &nextest_profile {
        cmd.args(["--profile", np]);
    }
    // Forward the user's cargo/nextest flags (features, `--cargo-profile`,
    // ...) verbatim; no `--` separator is needed — clap captures them as
    // the trailing_var_arg group.
    cmd.args(&args);

    if raw {
        cmd.env(ktstr::KTSTR_VERIFIER_RAW_ENV, "1");
    }

    // `--profile <NAME>` sets the scheduler-under-test's cargo BUILD
    // profile via `KTSTR_SCHEDULER_PROFILE`; absent, `build_and_find_binary`
    // defaults it to `release`.
    if let Some(p) = &profile {
        cmd.env(ktstr::KTSTR_SCHEDULER_PROFILE_ENV, p);
    }

    // Always produce a non-empty kernel list. When --kernel is
    // omitted, auto-discover one kernel and synthesize a single
    // entry with a path-basename label. The test-binary cell
    // handler keys on this list as its single source of truth.
    let resolved: Vec<(String, PathBuf)> = if !kernel.is_empty() {
        let r = resolve_kernel_set(&kernel)?;
        if r.is_empty() {
            return Err(
                "--kernel: every supplied value parsed to empty / whitespace; \
                 omit the flag for auto-discovery, or supply a kernel \
                 identifier"
                    .to_string(),
            );
        }
        r
    } else {
        let path = resolve_kernel_image(None)?;
        let label = path_kernel_label(&path);
        vec![(label, path)]
    };

    cmd.env(ktstr::KTSTR_KERNEL_ENV, &resolved[0].1);
    let encoded = encode_kernel_list(&resolved)?;
    cmd.env(ktstr::KTSTR_KERNEL_LIST_ENV, encoded);
    // Mark this test invocation as cargo-ktstr-orchestrated so
    // VM-boot tests can skip when run under raw nextest. Mirrors
    // the `cargo ktstr test` dispatcher in run_cargo.rs.
    cmd.env(ktstr::KTSTR_ORCHESTRATED_ENV, "1");
    let kernel_count = resolved.len();

    eprintln!(
        "cargo ktstr verifier: dispatching to nextest with filter test(/^verifier/) \
         on {kernel_count} resolved kernel(s){raw}{fwd}",
        raw = if raw { " (raw output)" } else { "" },
        fwd = if args.is_empty() {
            String::new()
        } else {
            format!(" forwarding to nextest: {}", args.join(" "))
        },
    );

    let status = cmd
        .status()
        .map_err(|e| format!("spawn cargo nextest run: {e}"))?;
    if status.success() {
        Ok(())
    } else {
        Err(format!(
            "cargo nextest run exited with {}",
            status
                .code()
                .map_or("signal".to_string(), |c| c.to_string()),
        ))
    }
}
