//! `cargo ktstr verifier` subcommand: thin wrapper around
//! `cargo nextest run` filtered to the `verifier/` test-name prefix.
//!
//! Each test binary that links the `ktstr` crate's `test_support`
//! module and has at least one `declare_scheduler!` declaration emits
//! one nextest test per (declared scheduler × kernel-list entry ×
//! accepted topology preset) cell — the verifier sweeps each declared
//! scheduler ACROSS topologies, because whether a scheduler attaches
//! and dispatches is topology-DEPENDENT (a scheduler can attach on one
//! topology and wedge on another). Every cell boots in no_perf_mode, so
//! a preset is emitted only when the scheduler's constraints accept it
//! under `accepts_no_perf_mode`. A cell PASSes only when its scheduler
//! (1) verifies (BPF loads — `verified_insns`), (2) attaches (the guest
//! attach gate confirms sched_ext `enabled`), AND (3) dispatches an
//! injected SpinWait workload (the guest emits a `WorkloadDispatched`
//! frame when a worker makes forward progress after attach). The cell
//! lister and handler live in
//! `src/test_support/dispatch.rs::list_verifier_cells_all` and
//! `run_verifier_cell`. Nextest provides per-cell parallelism, retries,
//! and failure isolation; this dispatcher resolves the `--kernel`
//! argument into the `KTSTR_KERNEL_LIST` env-var matrix dimension the
//! test binary's lister walks, plumbs `--raw` via `KTSTR_VERIFIER_RAW`,
//! restricts the sweep to one declared scheduler via `--scheduler
//! <NAME>` (plumbed through `KTSTR_VERIFIER_SCHEDULER`), and spawns
//! nextest filtered to the CELL names only (`test(/^verifier/) &
//! !test(/^verifier::/)` — the `verifier/...` cells, NOT the verifier
//! module's own `verifier::tests::*` unit tests, which also start with
//! "verifier"). The trailing `args` are forwarded verbatim to that
//! `cargo nextest run` (a nextest filterset, `--cargo-profile`, ...);
//! native flags may appear in any order relative to them and no `--`
//! separator is needed (see the bin's `argsplit` module). The
//! `declare_scheduler!` verifier cells carry no
//! `required-features`, so they build without a feature flag — no
//! `--features` passthrough is needed to collect verifier statistics.
//! The scheduler-under-test builds release by default, and each cell boots
//! with performance mode disabled (its `verified_insns` count is
//! perf-mode-independent, so cells take only a shared LLC reservation
//! and no longer starve each other on the LLC lock — see
//! `collect_verifier_output`). After nextest returns, the dispatcher
//! reads each cell's PASS/FAIL record (written under
//! `KTSTR_VERIFIER_RESULT_DIR`) and prints one `verified_insns` table
//! per declared scheduler followed by a topology × scheduler PASS/FAIL
//! grid.
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

/// Sweep verifier result dirs orphaned by interrupted prior runs.
///
/// The result dir is keyed on the dispatcher pid, so a run killed before
/// its post-run `remove_dir_all` (Ctrl-C / SIGKILL / crash) orphans its
/// dir — and nothing reclaims it later, since the per-run wipe only
/// targets the CURRENT pid. This runs at startup and removes every
/// `ktstr-verifier-results-<pid>` dir whose owning pid is no longer alive
/// (`kill(pid, 0)` -> ESRCH), mirroring `cleanup_stale_shm`'s next-run
/// reclamation. A dir owned by a LIVE pid is a concurrent verifier run and
/// is left untouched.
fn sweep_stale_result_dirs(temp_root: &std::path::Path) {
    let Ok(entries) = std::fs::read_dir(temp_root) else {
        return;
    };
    let self_pid = std::process::id();
    for entry in entries.flatten() {
        let name = entry.file_name();
        let Some(pid) = name
            .to_str()
            .and_then(|n| n.strip_prefix("ktstr-verifier-results-"))
            .and_then(|p| p.parse::<i32>().ok())
        else {
            continue;
        };
        // Skip non-positive pids (kill(0)/kill(-N) probe process GROUPS,
        // not a single process) and our own dir (owned by the run below).
        if pid <= 0 || pid == self_pid as i32 {
            continue;
        }
        // Only a definitively-dead owner (ESRCH) is an orphan. A live pid
        // (Ok) or a permission error (EPERM) is left alone.
        let dead = matches!(
            nix::sys::signal::kill(nix::unistd::Pid::from_raw(pid), None),
            Err(nix::errno::Errno::ESRCH),
        );
        if dead {
            let _ = std::fs::remove_dir_all(entry.path());
        }
    }
}

/// Dispatch the `cargo ktstr verifier` subcommand.
///
/// The trailing `args` are forwarded verbatim to the inner
/// `cargo nextest run` (a nextest filterset, `--cargo-profile`, ...).
/// The `declare_scheduler!` verifier cells carry no `required-features`,
/// so they build without a feature flag — no `--features` passthrough is
/// needed for the cell-only filter to match and collect verifier
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
    scheduler: Option<String>,
    include_eol: bool,
    args: Vec<String>,
) -> Result<(), String> {
    let mut cmd = Command::new("cargo");
    // The nextest argument vector — base flags (`--run-ignored all`, the
    // load-bearing `--no-tests pass`, and the `verifier/...`-cell filter),
    // then the optional `--nextest-profile` as nextest's `--profile`, then
    // the user's forwarded trailing args verbatim — is built by
    // `ktstr::verifier::build_nextest_args`, which documents each flag
    // and is unit-tested so the reachability-critical ones cannot be
    // silently dropped. The profile is emitted before the forwarded args
    // so a passthrough token cannot shadow it; no `--` separator is needed
    // (the bin's argsplit rewrite routes native flags to ktstr and the
    // passthrough to the `last = true` `args` field before clap parses).
    cmd.args(ktstr::verifier::build_nextest_args(
        nextest_profile.as_deref(),
        &args,
    ));

    if raw {
        cmd.env(ktstr::KTSTR_VERIFIER_RAW_ENV, "1");
    }

    // `--profile <NAME>` sets the scheduler-under-test's cargo BUILD
    // profile via `KTSTR_SCHEDULER_PROFILE`; absent, `build_and_find_binary`
    // defaults it to `release`.
    if let Some(p) = &profile {
        cmd.env(ktstr::KTSTR_SCHEDULER_PROFILE_ENV, p);
    }

    // `--scheduler <NAME>` restricts the sweep to a single declared
    // scheduler: forwarded via KTSTR_VERIFIER_SCHEDULER so the cell
    // emission (`list_verifier_cells_all`, which runs in the test binary
    // where the `declare_scheduler!` registry is linked) skips every
    // other declared scheduler. Validation is emission-side: this CLI
    // bin does not link that registry, so a name typo surfaces as an
    // empty record set, reported after nextest returns.
    if let Some(s) = &scheduler {
        cmd.env(ktstr::KTSTR_VERIFIER_SCHEDULER_ENV, s);
    }

    // Always produce a non-empty kernel list. When --kernel is
    // omitted, auto-discover one kernel and synthesize a single
    // entry with a path-basename label. The test-binary cell
    // handler keys on this list as its single source of truth.
    let resolved: Vec<(String, PathBuf)> = if !kernel.is_empty() {
        let r = resolve_kernel_set(&kernel, include_eol)?;
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
    // so concurrent `cargo ktstr verifier` runs don't cross-read. First
    // sweep dirs orphaned by dead-pid prior runs (an interrupted run skips
    // the post-run wipe below), then wipe our own pid's dir in case a prior
    // run reused this pid.
    let temp_root = std::env::temp_dir();
    sweep_stale_result_dirs(&temp_root);
    let result_dir = temp_root.join(format!("ktstr-verifier-results-{}", std::process::id()));
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

    // Survive Ctrl-C / SIGTERM so the result-dir cleanup below runs; nextest
    // tears down its own test children (see crate::interrupt).
    let interrupt_guard = crate::interrupt::InterruptGuard::install();
    let status = cmd
        .status()
        .map_err(|e| format!("spawn cargo nextest run: {e}"))?;

    // From the records each cell wrote into `result_dir`: print the
    // per-scheduler verified_insns tables first, then the topology ×
    // scheduler PASS/FAIL grid LAST so the operator's final view is the
    // pass/fail matrix. Both print on success AND failure so failing cells
    // stay visible. Best-effort: no records (e.g. 0 cells ran) -> the
    // renderers return None and nothing prints.
    let records = ktstr::verifier::read_cell_records(&result_dir);
    if let Some(tables) = ktstr::verifier::render_instruction_count_tables(&records) {
        print!("{tables}");
    }
    if let Some(table) = ktstr::verifier::render_result_table(&records) {
        print!("{table}");
    }
    let _ = std::fs::remove_dir_all(&result_dir);
    // Cleanup done; if interrupted, propagate as 128+signal now.
    let caught = interrupt_guard.interrupted();
    drop(interrupt_guard);
    if let Some(sig) = caught {
        crate::interrupt::reraise(sig);
    }

    // Decide the outcome from nextest's exit + the records. With
    // `--no-tests pass` a zero-cell selection exits 0, so an empty record
    // set on success is diagnosed here (a `--scheduler` typo, no scheduler
    // declared, or no topology preset fits this host) rather than
    // surfacing nextest's generic no-tests error. A real build/exec
    // failure still exits non-zero and is surfaced verbatim.
    ktstr::verifier::classify_run_outcome(
        status.success(),
        records.is_empty(),
        scheduler.as_deref(),
        status.code(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sweep_removes_dead_pid_result_dirs_keeps_live() {
        // A result dir owned by a dead pid is reclaimed; one owned by our
        // own (live) pid is kept. pid 2147483647 (i32::MAX) is above
        // pid_max, so kill() returns ESRCH. Each nextest test is its own
        // process, so the shared temp dir is test-isolated by pid.
        let base = std::env::temp_dir().join(format!("ktstr-vsweep-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        std::fs::create_dir_all(&base).expect("mk temp root");
        let dead = base.join("ktstr-verifier-results-2147483647");
        let live = base.join(format!("ktstr-verifier-results-{}", std::process::id()));
        std::fs::create_dir(&dead).expect("mk dead dir");
        std::fs::create_dir(&live).expect("mk live dir");

        sweep_stale_result_dirs(&base);

        assert!(!dead.exists(), "a dead-owner result dir is reclaimed");
        assert!(live.exists(), "our own (live) result dir is kept");

        let _ = std::fs::remove_dir_all(&base);
    }
}
