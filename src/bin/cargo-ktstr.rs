//! Cargo-integrated `cargo ktstr <SUB>` binary entry point.
//!
//! This file is the bin target itself: the global jemalloc allocator,
//! tracing init, SIGPIPE restore, top-level [`clap::Parser`] dispatch,
//! and the `KtstrCommand` match arm that fans out to each subcommand
//! handler. The handlers themselves live in submodules under
//! `src/bin/cargo_ktstr/`:
//!
//! - `cli`    — clap-derived `Cargo` / `CargoSub` / `Ktstr` /
//!   `KtstrCommand` / `ModelCommand` / `StatsCommand`
//!   types that drive argument parsing and shell
//!   completion generation.
//! - `kernel` — `--kernel <SPEC>` resolution shared by the `shell`,
//!   `verifier`, and gauntlet-expansion code paths, plus
//!   the `kernel build` subcommand dispatcher. Pure
//!   wire-format helpers (label emission, `KTSTR_KERNEL_LIST`
//!   encoding, dedup, collision detection) live in the
//!   inner `kernel::wire_format` submodule.
//! - `run_cargo` — `test`, `coverage`, `llvm-cov` dispatchers that
//!   wrap `cargo nextest` with the kernel/topology
//!   gauntlet wire format.
//! - `stats`  — `stats compare` subcommand that diffs
//!   `target/stats/` JSON across two kernel/scheduler
//!   build commits.
//! - `verifier` — `verifier` subcommand that runs a scheduler
//!   binary under the BPF-stats verifier and renders
//!   per-program verified-instruction counts.
//! - `misc`   — smaller subcommand dispatchers, one submodule per
//!   CLI verb: `shell`, `completions`, `funify`,
//!   `model {fetch,status,clean}`, `export`.
//! - `parse_tests` (test-only) — clap parse-shape coverage: every
//!   `KtstrCommand` variant gets at least one test that
//!   pins flag wiring + conflict/requires constraints.
//!
//! Each `mod` declaration uses `#[path = "cargo_ktstr/<file>.rs"]`
//! because rustc derives module file names from the bin's *file*
//! name (`cargo-ktstr`), not the *crate* name. Without `#[path]` it
//! would look for `src/bin/cargo-ktstr/<mod>.rs`, an underscore-vs-hyphen
//! mismatch with the actual `src/bin/cargo_ktstr/` directory.

// Global allocator (jemalloc) is provided centrally by the ktstr
// library crate (src/lib.rs) and inherited by this bin.
#[path = "cargo_ktstr/cli.rs"]
mod cli;
#[path = "cargo_ktstr/kernel/mod.rs"]
mod kernel;
#[path = "cargo_ktstr/misc/mod.rs"]
mod misc;
#[path = "cargo_ktstr/perf_delta.rs"]
mod perf_delta;
#[path = "cargo_ktstr/replay.rs"]
mod replay;
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

#[cfg(test)]
#[path = "cargo_ktstr/parse_tests.rs"]
mod parse_tests;

use clap::Parser;
use ktstr::cli::KernelCommand;

#[cfg(feature = "llm")]
use crate::cli::ModelCommand;
use crate::cli::{Cargo, CargoSub, KtstrCommand};

fn main() {
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
        std::process::exit(1);
    }
    // Mirror `ktstr`'s tracing init (src/bin/ktstr.rs main()) so
    // `tracing::warn!` calls inside `cli::` / `test_support::` surface
    // on stderr instead of being silently dropped. Default to `warn`
    // so normal CLI invocations (kernel build, model fetch, etc.) stay
    // quiet; users who want finer detail set `RUST_LOG=info,debug,...`.
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
        )
        .with_writer(std::io::stderr)
        .init();

    let Cargo {
        command: CargoSub::Ktstr(ktstr),
    } = Cargo::parse();

    // Match-arm order mirrors the `KtstrCommand` enum declaration in
    // `cli.rs`. Keeping the two orderings in lockstep lets a reviewer
    // eyeball "every variant is dispatched" in one linear scan instead
    // of cross-referencing two different orders; a future variant
    // addition then lands in the matching enum position and here
    // without requiring the reader to rebuild the mapping.
    let result = match ktstr.command {
        KtstrCommand::Test {
            kernel,
            no_perf_mode,
            no_skip_mode,
            release,
            release_scheduler,
            args,
        } => run_cargo::run_test(
            kernel,
            no_perf_mode,
            no_skip_mode,
            release,
            release_scheduler,
            args,
        ),
        KtstrCommand::Coverage {
            kernel,
            no_perf_mode,
            no_skip_mode,
            release,
            release_scheduler,
            args,
        } => run_cargo::run_coverage(
            kernel,
            no_perf_mode,
            no_skip_mode,
            release,
            release_scheduler,
            args,
        ),
        KtstrCommand::LlvmCov {
            kernel,
            no_perf_mode,
            no_skip_mode,
            args,
        } => run_cargo::run_llvm_cov(kernel, no_perf_mode, no_skip_mode, args),
        KtstrCommand::Stats { ref command } => stats::run_stats(command),
        KtstrCommand::Replay { dir, filter, exec } => {
            match replay::run_replay(dir.as_deref(), filter.as_deref(), exec) {
                Ok(0) => Ok(()),
                Ok(code) => std::process::exit(code),
                Err(e) => Err(format!("{e:#}")),
            }
        }
        KtstrCommand::PerfDelta {
            base,
            base_ref,
            filter,
            default_branch,
            kernel,
            dual_run,
            threshold,
            policy,
            no_phases,
            phases_only,
            steps_only,
            phase,
            phase_threshold,
        } => {
            let args = perf_delta::PerfDeltaArgs {
                base: base.as_deref(),
                base_ref: base_ref.as_deref(),
                filter: filter.as_deref(),
                default_branch: &default_branch,
                kernel: kernel.as_deref(),
                dual_run,
                threshold,
                policy: policy.as_deref(),
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
                Ok(code) => std::process::exit(code),
                Err(e) => Err(format!("{e:#}")),
            }
        }
        KtstrCommand::Kernel { command } => match command {
            KernelCommand::List { json, range } => match range {
                Some(r) => {
                    ktstr::cli::kernel_list_range_preview(json, &r).map_err(|e| format!("{e:#}"))
                }
                None => ktstr::cli::kernel_list(json).map_err(|e| format!("{e:#}")),
            },
            KernelCommand::Build {
                version,
                source,
                git,
                git_ref,
                force,
                clean,
                cpu_cap,
                extra_kconfig,
                skip_sha256,
            } => kernel::kernel_build(
                version,
                source,
                git,
                git_ref,
                force,
                clean,
                cpu_cap,
                extra_kconfig,
                skip_sha256,
            ),
            KernelCommand::Clean {
                keep,
                force,
                corrupt_only,
            } => ktstr::cli::kernel_clean(keep, force, corrupt_only).map_err(|e| format!("{e:#}")),
        },
        #[cfg(feature = "llm")]
        KtstrCommand::Model { command } => match command {
            ModelCommand::Fetch => misc::run_model_fetch(),
            ModelCommand::Status => misc::run_model_status(),
            ModelCommand::Clean => misc::run_model_clean(),
        },
        KtstrCommand::Verifier { kernel, raw } => verifier::run_verifier(kernel, raw),
        KtstrCommand::Funify {
            input,
            seed,
            pretty,
        } => misc::run_funify(input, seed, pretty),
        KtstrCommand::Completions { shell, binary } => {
            misc::run_completions(shell, &binary);
            Ok(())
        }
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
            // (None) exits 0. Diverge here so it does not fall through to
            // the uniform exit-0 path below; Err routes to the shared
            // error handler.
            Ok(opt) => std::process::exit(opt.unwrap_or(0)),
            Err(e) => Err(e),
        },
    };

    if let Err(e) = result {
        eprintln!("error: {e:#}");
        std::process::exit(1);
    }
}
