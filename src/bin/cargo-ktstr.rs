//! Cargo-integrated `cargo ktstr <SUB>` binary entry point.
//!
//! This file is the bin target itself: the global jemalloc
//! allocator, tracing init, SIGPIPE restore, top-level [`clap::Parser`]
//! dispatch,
//! and the `KtstrCommand` match arm that fans out to each subcommand
//! handler. The handlers themselves live in submodules under
//! `src/bin/cargo_ktstr/`:
//!
//! - `cli`    — clap-derived `Cargo` / `CargoSub` / `Ktstr` /
//!   `KtstrCommand` / `StatsCommand`
//!   types that drive argument parsing and shell
//!   completion generation.
//! - `feature_discovery` — Cargo-metadata inspection that finds narrow
//!   optional-ktstr feature gates and package-qualifies them for every
//!   supported nextest or workspace test-registry build/probe command.
//! - `kernel` — `--kernel <SPEC>` resolution shared by the `shell`,
//!   `verifier`, and gauntlet-expansion code paths, plus
//!   the `kernel build` subcommand dispatcher. Pure
//!   wire-format helpers (label emission, `KTSTR_KERNEL_LIST`
//!   encoding, dedup, collision detection) live in the
//!   inner `kernel::wire_format` submodule.
//! - `run_cargo` — `test`, `coverage`, `llvm-cov` dispatchers that
//!   wrap `cargo nextest` with the kernel/topology
//!   gauntlet wire format.
//! - `perf_delta` — `perf-delta` dispatcher that resolves the
//!   baseline commit a perf run is compared against and
//!   surfaces the A/B commit pair.
//! - `replay` — `replay` dispatcher that re-runs the failing
//!   subset of a prior sidecar pool.
//! - `stats`  — single-run inspection subcommands (list /
//!   list-values / list-metrics / show-host / explain-sidecar)
//!   over the `target/ktstr/` sidecar pool.
//! - `verifier` — `verifier` subcommand that runs a scheduler
//!   binary under the BPF-stats verifier and renders
//!   per-program verified-instruction counts.
//! - `misc`   — smaller subcommand dispatchers, one submodule per
//!   CLI verb: `shell`, `completions`, `export`.
//! - `parse_tests` (test-only) — clap parse-shape coverage: every
//!   `KtstrCommand` variant gets at least one test that
//!   pins flag wiring + conflict/requires constraints.
//!
//! Each `mod` declaration uses `#[path = "cargo_ktstr/<file>.rs"]`
//! because rustc derives module file names from the bin's *file*
//! name (`cargo-ktstr`), not the *crate* name. Without `#[path]` it
//! would look for `src/bin/cargo-ktstr/<mod>.rs`, an underscore-vs-hyphen
//! mismatch with the actual `src/bin/cargo_ktstr/` directory.

#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

#[path = "cargo_ktstr/affected/mod.rs"]
mod affected;
#[path = "cargo_ktstr/cli.rs"]
mod cli;
#[path = "cargo_ktstr/feature_discovery.rs"]
mod feature_discovery;
#[path = "cargo_ktstr/interrupt.rs"]
mod interrupt;
#[path = "cargo_ktstr/kernel/mod.rs"]
mod kernel;
#[path = "cargo_ktstr/misc/mod.rs"]
mod misc;
#[path = "cargo_ktstr/nextest_config.rs"]
mod nextest_config;
#[path = "cargo_ktstr/perf_delta.rs"]
mod perf_delta;
#[path = "cargo_ktstr/replay.rs"]
mod replay;
#[path = "cargo_ktstr/reserved_build_progress.rs"]
mod reserved_build_progress;
#[path = "cargo_ktstr/run_cargo.rs"]
mod run_cargo;
#[path = "cargo_ktstr/stats.rs"]
mod stats;
#[path = "cargo_ktstr/verifier.rs"]
mod verifier;

#[path = "cargo_ktstr/btf_catalog.rs"]
mod btf_catalog;

#[path = "cargo_ktstr/blobs.rs"]
mod blobs;

#[path = "cargo_ktstr/argsplit.rs"]
mod argsplit;

#[cfg(test)]
#[path = "cargo_ktstr/parse_tests.rs"]
mod parse_tests;

use clap::{CommandFactory, Parser};
use ktstr::cli::KernelCommand;

use crate::cli::{Cargo, CargoSub, KtstrCommand};

/// Decide whether startup must install a freshly detected project commit.
///
/// A non-empty inherited value is authoritative (notably perf-delta's
/// baseline and HEAD labels) and suppresses detection. Missing or empty values
/// trigger exactly one probe. Split from [`install_project_commit_env`] so the
/// precedence and one-probe contract are testable without mutating the
/// process-wide environment from a parallel unit test.
fn project_commit_to_install_with(
    existing: Option<std::ffi::OsString>,
    detect: impl FnOnce() -> Option<String>,
) -> Option<String> {
    if existing.as_ref().is_some_and(|value| !value.is_empty()) {
        None
    } else {
        detect()
    }
}

/// Resolve the invoking project's commit once and export it to every
/// descendant of cargo-ktstr.
///
/// This is intentionally top-level rather than attached to individual
/// subcommands: verifier, replay, shell, test, coverage, and raw
/// `llvm-cov nextest` all spawn processes that may eventually write a
/// sidecar. A single inherited value gives every path the same project
/// snapshot and prevents process-per-test commit/status rediscovery.
///
/// SAFETY: `main` calls this before tracing initialization, signal-handler
/// installation, or any other persistent thread spawn. `repo_is_dirty`
/// bounds gix's tracked-file work to one worker and exhausts its status
/// iterator, joining gix's temporary producer before detection returns.
fn install_project_commit_env() {
    let install = project_commit_to_install_with(
        std::env::var_os(ktstr::KTSTR_PROJECT_COMMIT_ENV),
        ktstr::test_support::detect_project_commit,
    );
    if let Some(commit) = install {
        // SAFETY: see the function doc — startup is single-threaded, and the
        // detector joins its temporary gix producer before returning.
        unsafe {
            std::env::set_var(ktstr::KTSTR_PROJECT_COMMIT_ENV, commit);
        }
    }
}

fn main() {
    // Process-group anchors re-exec this binary with a private marker and
    // control pipes. Handle that mode before blob extraction, Cargo metadata,
    // tracing, argument parsing, or any other ordinary CLI initialization.
    if interrupt::run_anchor_mode_if_requested() {
        return;
    }
    interrupt::run_startup_supervision();

    ktstr::host_heap::mark_jemalloc_global_allocator();
    // Restore SIGPIPE so piping `cargo ktstr ... | head` doesn't
    // panic inside `print!`. See `ktstr::cli::restore_sigpipe_default`
    // for the full rationale; shared across all three ktstr bins so
    // the rationale + SAFETY text lives in one place.
    ktstr::cli::restore_sigpipe_default();
    // Extract embedded binary blobs to tempfiles and export their
    // paths via env vars. Done BEFORE tracing subscriber init or
    // anything else that might spawn a thread — `std::env::set_var`
    // requires no concurrent reader (see `blobs::install_env` safety
    // doc). Child processes spawned later (e.g. nextest fanning out
    // to test bins) inherit these env vars; the `ktstr` library's
    // blob-loading helpers read from them on demand. A failure here
    // aborts before any side-effects so the operator gets a clean
    // error.
    if let Err(e) = blobs::install_env() {
        eprintln!("error: extract embedded blobs: {e}");
        interrupt::commit_startup_worker_exit(1);
        std::process::exit(1);
    }
    // Resolve project identity once for every descendant-producing command.
    // This must stay before tracing/thread initialization; see the installer's
    // environment-safety contract.
    install_project_commit_env();
    // Pin KTSTR_RUNS_ROOT to the absolute cargo target dir's ktstr
    // subdir so this orchestrator's footer / stats / replay reads and
    // the child test processes' sidecar writes resolve the SAME dir
    // regardless of CWD (CWD-relative runs_root() otherwise splits them
    // across a Cargo workspace). MUST run here — before tracing init
    // or anything that spawns a thread — for the same `set_var` safety
    // reason as `blobs::install_env` above; child processes inherit it.
    run_cargo::install_runs_root_env();
    // Mirror `ktstr`'s tracing init (src/bin/ktstr.rs main()) so
    // `tracing::warn!` calls inside `cli::` / `test_support::` surface
    // on stderr instead of being silently dropped. Default to `warn`
    // so normal CLI invocations (kernel build, shell, etc.) stay
    // quiet; users who want finer detail set `RUST_LOG=info,debug,...`.
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
        )
        .with_writer(std::io::stderr)
        .init();

    // Position-independent ktstr flags: rewrite argv so a passthrough
    // subcommand's ktstr-owned flags parse regardless of their position
    // relative to nextest passthrough args (see the `argsplit` module).
    // Non-passthrough subcommands pass through unchanged.
    let raw: Vec<std::ffi::OsString> = std::env::args_os().collect();
    let rewritten = argsplit::rewrite(&Cargo::command(), &raw);
    let Cargo {
        command: CargoSub::Ktstr(ktstr),
    } = match Cargo::try_parse_from(&rewritten) {
        Ok(c) => c,
        Err(e) => {
            let code = u8::try_from(e.exit_code()).unwrap_or(1);
            let _ = e.print();
            interrupt::commit_startup_worker_exit(code);
            std::process::exit(i32::from(code));
        }
    };

    // One handler installation spans the dispatched CLI lifetime. It starts
    // in EARLY mode, where SIGINT/SIGTERM retain terminate-immediately
    // semantics through command-specific metadata, network, and kernel
    // preflight. Run-producing dispatchers cross into cleanup ownership
    // immediately before their first reservation/result-dir/checkout, then
    // this top-level owner restores and re-raises only after every
    // dispatcher-local RAII cleanup has completed. Installing after parsing
    // also leaves Clap's direct error exit under the ordinary dispositions.
    let interrupt_guard = interrupt::InterruptGuard::install();
    let result = dispatch_command(ktstr.command);

    let caught = interrupt::restore_and_caught(interrupt_guard);
    if let Some(signal) = caught {
        // A signal exit is intentionally not a clean-release commit. The
        // startup subreaper drains anything that survived worker cleanup
        // before relaying the exact signal status.
        interrupt::reraise(signal);
    }
    if let Some(code) = interrupt::take_deferred_exit_code() {
        let code = code as u8;
        interrupt::commit_startup_worker_exit(code);
        std::process::exit(i32::from(code));
    }
    if let Err(e) = result {
        eprintln!("error: {e:#}");
        interrupt::commit_startup_worker_exit(1);
        std::process::exit(1);
    }
    interrupt::commit_startup_worker_exit(0);
}

/// Fan out a parsed [`KtstrCommand`] to its subcommand handler.
///
/// Split into [`dispatch_run_command`] (test/coverage/llvm-cov/stats/
/// replay/perf-delta) and [`dispatch_admin_command`] (kernel/
/// verifier/completions/host/thresholds/export/locks/shell)
/// purely to keep each function under the source-function size guard;
/// the run-group helper matches its variants and forwards every other
/// variant to the admin-group helper, so the two together cover the
/// enum exhaustively with the same arm bodies main used to inline.
fn dispatch_command(command: KtstrCommand) -> Result<(), String> {
    dispatch_run_command(command)
}

/// Dispatch the run-producing subcommands; forward the rest to
/// [`dispatch_admin_command`].
///
/// Match-arm order mirrors the `KtstrCommand` enum declaration in
/// `cli.rs`. Keeping the two orderings in lockstep lets a reviewer
/// eyeball "every variant is dispatched" in one linear scan instead
/// of cross-referencing two different orders; a future variant
/// addition then lands in the matching enum position and here
/// without requiring the reader to rebuild the mapping.
fn dispatch_run_command(command: KtstrCommand) -> Result<(), String> {
    match command {
        KtstrCommand::Test {
            kernel,
            no_perf_mode,
            no_skip_mode,
            release,
            profile,
            nextest_profile,
            include_eol,
            relevant,
            base,
            base_ref,
            default_branch,
            args,
        } => run_cargo::run_test(
            kernel,
            no_perf_mode,
            no_skip_mode,
            release,
            profile,
            nextest_profile,
            include_eol,
            relevant,
            base,
            base_ref,
            default_branch,
            args,
        ),
        KtstrCommand::Coverage {
            kernel,
            no_perf_mode,
            no_skip_mode,
            release,
            profile,
            nextest_profile,
            include_eol,
            relevant,
            base,
            base_ref,
            default_branch,
            args,
        } => run_cargo::run_coverage(
            kernel,
            no_perf_mode,
            no_skip_mode,
            release,
            profile,
            nextest_profile,
            include_eol,
            relevant,
            base,
            base_ref,
            default_branch,
            args,
        ),
        KtstrCommand::LlvmCov {
            kernel,
            no_perf_mode,
            no_skip_mode,
            include_eol,
            args,
        } => run_cargo::run_llvm_cov(kernel, no_perf_mode, no_skip_mode, include_eol, args),
        KtstrCommand::Stats { ref command } => stats::run_stats(command),
        KtstrCommand::Replay {
            dir,
            filter,
            exec,
            profile,
            nextest_profile,
            args,
        } => match replay::run_replay(
            dir.as_deref(),
            filter.as_deref(),
            exec,
            profile.as_deref(),
            nextest_profile.as_deref(),
            &args,
        ) {
            Ok(0) => Ok(()),
            Ok(code) => {
                interrupt::defer_exit_code(code);
                Ok(())
            }
            Err(e) => Err(format!("{e:#}")),
        },
        KtstrCommand::PerfDelta {
            base,
            base_ref,
            filter,
            relevant,
            default_branch,
            kernel,
            threshold,
            policy,
            noise_adjust,
            noise_spread_threshold,
            no_phases,
            phases_only,
            steps_only,
            phase,
            phase_threshold,
            profile,
            nextest_profile,
            all_metrics,
            fail_threshold,
            must_fail,
            args: passthrough,
        } => {
            let args = perf_delta::PerfDeltaArgs {
                passthrough: &passthrough,
                base: base.as_deref(),
                base_ref: base_ref.as_deref(),
                filter: filter.as_deref(),
                relevant,
                default_branch: &default_branch,
                kernel: kernel.as_deref(),
                threshold,
                policy: policy.as_deref(),
                noise_adjust,
                noise_spread_threshold,
                profile: profile.as_deref(),
                nextest_profile: nextest_profile.as_deref(),
                all_metrics,
                fail_threshold,
                must_fail: must_fail.as_deref(),
                phase_display: ktstr::cli::PhaseDisplayOptions {
                    no_phases,
                    phases_only,
                    steps_only,
                    phase,
                    phase_threshold,
                },
            };
            match perf_delta::run(&args) {
                Ok(0) => Ok(()),
                Ok(code) => {
                    interrupt::defer_exit_code(code);
                    Ok(())
                }
                Err(e) => Err(format!("{e:#}")),
            }
        }
        // Forward the admin/introspection group verbatim. Listing the
        // variants explicitly (rather than a `_` wildcard) keeps the
        // match exhaustive over the full enum: a future `KtstrCommand`
        // variant fails to compile here until it is routed, preserving
        // the single-match compile-time exhaustiveness guarantee.
        cmd @ (KtstrCommand::Kernel { .. }
        | KtstrCommand::Verifier { .. }
        | KtstrCommand::Completions { .. }
        | KtstrCommand::ShowHost
        | KtstrCommand::ShowThresholds { .. }
        | KtstrCommand::Affected { .. }
        | KtstrCommand::Export { .. }
        | KtstrCommand::Locks { .. }
        | KtstrCommand::Shell { .. }) => dispatch_admin_command(cmd),
    }
}

/// Dispatch the cache/admin/introspection subcommands. Reached only
/// for the variants [`dispatch_run_command`] forwards; its match
/// covers exactly the remaining `KtstrCommand` variants in enum order.
fn dispatch_admin_command(command: KtstrCommand) -> Result<(), String> {
    match command {
        KtstrCommand::Kernel { command } => match command {
            KernelCommand::List {
                json,
                kernel,
                include_eol,
            } => match kernel {
                Some(k) => ktstr::cli::kernel_list_range_preview(json, &k, include_eol)
                    .map_err(|e| format!("{e:#}")),
                None => ktstr::cli::kernel_list(json).map_err(|e| format!("{e:#}")),
            },
            KernelCommand::Build {
                kernel,
                force,
                clean,
                cpu_cap,
                extra_kconfig,
                skip_sha256,
                include_eol,
            } => kernel::kernel_build(
                kernel,
                force,
                clean,
                cpu_cap,
                extra_kconfig,
                skip_sha256,
                include_eol,
            ),
            KernelCommand::Clean {
                keep,
                force,
                corrupt_only,
            } => ktstr::cli::kernel_clean(keep, force, corrupt_only).map_err(|e| format!("{e:#}")),
        },
        KtstrCommand::Verifier {
            kernel,
            raw,
            profile,
            nextest_profile,
            scheduler,
            include_eol,
            args,
        } => verifier::run_verifier(
            kernel,
            raw,
            profile,
            nextest_profile,
            scheduler,
            include_eol,
            args,
        ),
        KtstrCommand::Completions { shell, binary } => {
            misc::run_completions(shell, &binary);
            Ok(())
        }
        KtstrCommand::Affected {
            base,
            base_ref,
            default_branch,
        } => affected::run(base.as_deref(), base_ref.as_deref(), &default_branch)
            .map_err(|e| format!("{e:#}")),
        KtstrCommand::ShowHost => {
            print!("{}", ktstr::cli::show_host());
            Ok(())
        }
        KtstrCommand::ShowThresholds { test } => match ktstr::cli::show_thresholds(&test) {
            Ok(s) => {
                print!("{s}");
                Ok(())
            }
            Err(e) => Err(format!("{e:#}")),
        },
        KtstrCommand::Export {
            test,
            output,
            package,
            release,
        } => misc::run_export(test, output, package, release),
        KtstrCommand::Locks { json, watch } => {
            ktstr::cli::list_locks(json, watch).map_err(|e| format!("{e:#}"))
        }
        KtstrCommand::Shell {
            kernel,
            test,
            topology,
            include_files,
            memory_mib,
            dmesg,
            exec,
            exec_timeout,
            no_perf_mode,
            cpu_cap,
            disk,
        } => match misc::run_shell(
            kernel,
            test,
            topology,
            include_files,
            memory_mib,
            dmesg,
            exec,
            exec_timeout,
            no_perf_mode,
            cpu_cap,
            disk,
        ) {
            // Shell mode exits with the guest payload's own exit code
            // (recovered from the ExecExit bulk frame); interactive mode
            // (None) exits 0. Defer a non-zero code until the top-level
            // signal owner has restored dispositions; Err routes to the
            // shared error handler.
            Ok(Some(code)) if code != 0 => {
                interrupt::defer_exit_code(code);
                Ok(())
            }
            Ok(_) => Ok(()),
            Err(e) => Err(e),
        },
        // Reached only for variants `dispatch_run_command` handles;
        // it forwards everything else here, so those variants never
        // arrive. The arm exists to satisfy exhaustiveness without
        // restating the run-group patterns.
        KtstrCommand::Test { .. }
        | KtstrCommand::Coverage { .. }
        | KtstrCommand::LlvmCov { .. }
        | KtstrCommand::Stats { .. }
        | KtstrCommand::Replay { .. }
        | KtstrCommand::PerfDelta { .. } => unreachable!(
            "run-group variants are handled by dispatch_run_command and never forwarded here"
        ),
    }
}

#[cfg(test)]
mod startup_tests {
    use super::*;

    #[test]
    fn project_commit_install_preserves_nonempty_override_without_probing() {
        let calls = std::cell::Cell::new(0);
        let install =
            project_commit_to_install_with(Some(std::ffi::OsString::from("baseline1")), || {
                calls.set(calls.get() + 1);
                Some("wrong".to_string())
            });
        assert_eq!(install, None);
        assert_eq!(
            calls.get(),
            0,
            "an inherited perf-delta label must remain authoritative",
        );
    }

    #[test]
    fn project_commit_install_probes_once_for_missing_or_empty_value() {
        for existing in [None, Some(std::ffi::OsString::new())] {
            let calls = std::cell::Cell::new(0);
            let install = project_commit_to_install_with(existing, || {
                calls.set(calls.get() + 1);
                Some("deadbee-dirty".to_string())
            });
            assert_eq!(install.as_deref(), Some("deadbee-dirty"));
            assert_eq!(calls.get(), 1, "startup must resolve exactly once");
        }
    }

    #[test]
    fn project_commit_install_leaves_env_unset_when_detection_fails() {
        let calls = std::cell::Cell::new(0);
        let install = project_commit_to_install_with(None, || {
            calls.set(calls.get() + 1);
            None
        });
        assert_eq!(install, None);
        assert_eq!(calls.get(), 1);
    }
}
