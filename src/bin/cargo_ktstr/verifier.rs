//! `cargo ktstr verifier` subcommand: thin wrapper around
//! `cargo nextest run` filtered to the `verifier/` test-name prefix.
//!
//! Each test binary that links the `ktstr` crate's `test_support`
//! module and has at least one `declare_scheduler!` declaration
//! emits one nextest test per
//! (declared scheduler × kernel-list entry) cell — one cell per
//! (scheduler, kernel), on the smallest topology the scheduler's
//! constraints accept (`verified_insns` is topology-independent). A
//! cell PASSes only when its scheduler both verifies (BPF loads) AND
//! turns on — the guest attach gate confirms sched_ext `enabled`. The
//! lister + cell handler live in
//! `src/test_support/dispatch.rs::list_verifier_cells_all` and
//! `run_verifier_cell`. Nextest provides per-cell parallelism,
//! retries, and failure isolation; this dispatcher resolves the
//! `--kernel` argument into the `KTSTR_KERNEL_LIST` env-var matrix
//! dimension the test binary's lister walks, plumbs `--raw` via
//! `KTSTR_VERIFIER_RAW`, and spawns nextest filtered to the CELL names
//! only (`test(/^verifier/) & !test(/^verifier::/)` — the `verifier/...`
//! cells, NOT the verifier module's own `verifier::tests::*` unit tests,
//! which also start with "verifier"). The trailing `args` are forwarded
//! verbatim to that `cargo nextest run` — feature selection (`cargo ktstr
//! verifier --features integration,wprof`, no `--` separator) reaches the
//! integration-gated `declare_scheduler!` cell build; the
//! scheduler-under-test builds release by default, and each cell boots
//! with performance mode disabled (its `verified_insns` count is
//! perf-mode-independent, so cells take only a shared LLC reservation and
//! no longer starve each other on the LLC lock — see
//! `collect_verifier_output`). After nextest returns, the dispatcher
//! reads each cell's PASS/FAIL record (written under
//! `KTSTR_VERIFIER_RESULT_DIR`) and prints a scheduler × kernel
//! summary table.
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
/// Without the feature passthrough the integration-gated cells never
/// compile, so the cell-only filter matches nothing and the command
/// collects no verifier statistics.
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
    // Two load-bearing pieces:
    //   * `--run-ignored all`: verifier cells are emitted IGNORE-GATED
    //     (like gauntlet variants) — `list_verifier_cells_all` emits every
    //     `verifier/<sched>/<kernel>` line unconditionally,
    //     including on nextest's `--list --ignored` pass, so nextest marks
    //     each cell ignored. Without opting in, `cargo nextest run` skips
    //     every cell.
    //   * `test(/^verifier/) & !test(/^verifier::/)`: match the CELLS
    //     (named `verifier/...`, with a slash) but NOT the verifier
    //     module's own unit tests (`verifier::tests::...`, colons), which
    //     also start with "verifier" and would otherwise run under a bare
    //     `^verifier` prefix. `cargo ktstr verifier` collects BPF verifier
    //     stats from VM-boot cells; the module unit tests belong to
    //     `cargo ktstr test`, so they are excluded here.
    cmd.args([
        "nextest",
        "run",
        "--run-ignored",
        "all",
        "-E",
        "test(/^verifier/) & !test(/^verifier::/)",
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

    // Per-cell result dir: each verifier cell writes its PASS/FAIL record
    // here (via KTSTR_VERIFIER_RESULT_DIR), and after nextest returns we
    // read them back to render the summary table. Unique per dispatcher pid
    // so concurrent `cargo ktstr verifier` runs don't cross-read; wiped
    // first so a stale dir from a crashed prior run can't leak old records.
    let result_dir =
        std::env::temp_dir().join(format!("ktstr-verifier-results-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&result_dir);
    if let Err(e) = std::fs::create_dir_all(&result_dir) {
        return Err(format!(
            "create verifier result dir {}: {e}",
            result_dir.display()
        ));
    }
    cmd.env(ktstr::KTSTR_VERIFIER_RESULT_DIR_ENV, &result_dir);

    let kernel_count = resolved.len();

    eprintln!(
        "cargo ktstr verifier: dispatching to nextest (verifier/ cells only) \
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

    // Render the per-cell PASS/FAIL summary table from the records each
    // cell wrote into `result_dir`. Printed on BOTH success and failure so
    // the operator sees the full scheduler × kernel matrix
    // even when some cells failed. Best-effort: no records (e.g. 0 cells
    // ran) -> render_result_table returns None and nothing prints.
    let records = ktstr::verifier::read_cell_records(&result_dir);
    if let Some(table) = ktstr::verifier::render_result_table(&records) {
        print!("{table}");
    }
    let _ = std::fs::remove_dir_all(&result_dir);

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
