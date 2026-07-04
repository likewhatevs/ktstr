//! Clap parse tests for the cargo-ktstr CLI surface.
//!
//! Lives outside the bin entry file so the entry stays focused on
//! dispatch, and mirrors the shape used elsewhere in the workspace
//! where parse-only test fixtures cluster in their own module. The
//! tests assert the user-visible spelling of every `--flag` and
//! the round-trip of clap-parsed values into the variant fields,
//! so a derive-rename or attribute drop surfaces here at compile-
//! time / test-time rather than in production.
//!
//! # Coverage shape
//!
//! Every [`KtstrCommand`] variant has at least one positive test
//! that round-trips an argument through clap into the variant's
//! fields, plus negative tests for `requires` / `conflicts_with` /
//! `value_parser` constraints clap enforces at parse time.
//!
//! Sections are separated by `// -- <theme> --` banners.
//!
//! # External pins
//!
//! The `kconfig_status_*` and `format_entry_row_*` fixtures
//! co-pin the [`ktstr::cache::CacheEntry`] /
//! [`ktstr::cache::KernelMetadata`] shape; if the cache types'
//! constructors, methods, or variants change, those tests must
//! be updated in lockstep with the production types in
//! [`ktstr::cache`].
//!
//! # Why parse-only
//!
//! These tests deliberately do not invoke any subcommand body —
//! they verify that clap parses what we expect into the type-system
//! shape the body matches against. Behaviour-level coverage of each
//! handler lives next to its production code (e.g. `kernel.rs`'s
//! `tests` module exercises label collision detection;
//! `verifier.rs` exercises profile expansion).

#![cfg(test)]

use clap::{CommandFactory, Parser};
use ktstr::cache::{CacheArtifacts, CacheDir, CacheEntry, KernelMetadata};
use ktstr::cli;
use ktstr::cli::KernelCommand;

use crate::cli::{Cargo, CargoSub, KtstrCommand, StatsCommand};

// -- DRY helpers for the parse-only test surface --
//
// These helpers collapse the ~30-line boilerplate that every
// destructuring parse test repeats — build args, `try_parse_from`,
// destructure the `Cargo { CargoSub::Ktstr } -> KtstrCommand`
// outer chrome, and panic with a recognisable "expected X"
// message on a wrong-variant parse.
//
// Helpers panic on the wrong variant rather than returning
// `Result` because every call site is already in a `#[test]`
// where the panic IS the failure mode. Returning the variant
// by value (not by reference) lets the call site bind owned
// fields with a let-else destructure rather than yet another
// indirection.

/// Parse a full `cargo ktstr <sub> …` argv through the SAME pre-parse
/// split `main` applies (the `argsplit` module) and then clap, returning
/// the [`KtstrCommand`]. Use this — NOT a bare `Cargo::try_parse_from` —
/// for the passthrough subcommands (test / coverage / llvm-cov /
/// verifier / replay / perf-delta) so a no-`--` invocation with ktstr
/// flags interleaved among nextest passthrough args parses exactly as it
/// does in production. Panics if clap rejects the rewritten argv.
fn parse_via_split(argv: &[&str]) -> KtstrCommand {
    let raw: Vec<std::ffi::OsString> = argv.iter().map(std::ffi::OsString::from).collect();
    let rewritten = crate::argsplit::rewrite(&Cargo::command(), &raw);
    let Cargo {
        command: CargoSub::Ktstr(k),
    } = Cargo::try_parse_from(&rewritten).unwrap_or_else(|e| panic!("{e}"));
    k.command
}

#[test]
fn parse_perf_delta_flags_and_defaults() {
    // Explicit flags round-trip (kebab-case subcommand `perf-delta`).
    let Cargo {
        command: CargoSub::Ktstr(k),
    } = Cargo::try_parse_from([
        "cargo",
        "ktstr",
        "perf-delta",
        "--base",
        "abc123",
        "--base-ref",
        "release",
        "-E",
        "perf::",
        "--kernel",
        "6.14",
        "--threshold",
        "12.5",
    ])
    .unwrap_or_else(|e| panic!("{e}"));
    match k.command {
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
            args,
        } => {
            assert!(args.is_empty(), "no cargo passthrough on this invocation");
            assert!(
                !all_metrics && fail_threshold.is_none() && must_fail.is_none(),
                "gating / render flags default off when unset",
            );
            assert_eq!(base.as_deref(), Some("abc123"));
            assert_eq!(base_ref.as_deref(), Some("release"));
            assert_eq!(filter.as_deref(), Some("perf::"));
            assert!(!relevant, "--relevant defaults off");
            assert_eq!(default_branch, "main", "default branch defaults to main");
            assert_eq!(kernel.as_deref(), Some("6.14"));
            assert_eq!(threshold, Some(12.5));
            assert!(policy.is_none());
            assert!(
                !no_phases
                    && !phases_only
                    && !steps_only
                    && phase.is_none()
                    && phase_threshold.is_none(),
                "phase flags require --noise-adjust and are absent on the scalar path",
            );
            assert!(
                noise_adjust.is_none() && noise_spread_threshold.is_none(),
                "no --noise-adjust on this invocation",
            );
            assert!(
                profile.is_none() && nextest_profile.is_none(),
                "no --profile / --nextest-profile on this invocation",
            );
        }
        _ => panic!("expected PerfDelta"),
    }
    // Bare invocation: overrides None, default branch = main, no run production.
    let Cargo {
        command: CargoSub::Ktstr(k),
    } = Cargo::try_parse_from(["cargo", "ktstr", "perf-delta"]).unwrap_or_else(|e| panic!("{e}"));
    match k.command {
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
            args,
        } => {
            assert!(
                args.is_empty(),
                "bare perf-delta parses no cargo passthrough"
            );
            assert!(
                !all_metrics && fail_threshold.is_none() && must_fail.is_none(),
                "bare perf-delta defaults gating / render flags off",
            );
            assert!(base.is_none() && base_ref.is_none() && filter.is_none());
            assert!(!relevant, "bare perf-delta defaults --relevant off");
            assert_eq!(default_branch, "main");
            assert!(kernel.is_none());
            assert!(threshold.is_none() && policy.is_none());
            assert!(
                !no_phases
                    && !phases_only
                    && !steps_only
                    && phase.is_none()
                    && phase_threshold.is_none(),
                "phase flags default off (meaningful rows shown by default)",
            );
            assert!(noise_adjust.is_none() && noise_spread_threshold.is_none());
            assert!(
                profile.is_none() && nextest_profile.is_none(),
                "bare perf-delta defaults --profile / --nextest-profile to None",
            );
        }
        _ => panic!("expected PerfDelta"),
    }
    // Noise axis: --noise-adjust N + --noise-spread-threshold round-trip.
    let Cargo {
        command: CargoSub::Ktstr(k),
    } = Cargo::try_parse_from([
        "cargo",
        "ktstr",
        "perf-delta",
        "--noise-adjust",
        "3",
        "--noise-spread-threshold",
        "1.5",
        "--phases-only",
        "--steps-only",
        "--phase-threshold",
        "5",
        "--all-metrics",
        "--fail-threshold",
        "3",
        "--must-fail",
        "worst_spread",
    ])
    .unwrap_or_else(|e| panic!("{e}"));
    match k.command {
        KtstrCommand::PerfDelta {
            noise_adjust,
            noise_spread_threshold,
            phases_only,
            steps_only,
            phase_threshold,
            all_metrics,
            fail_threshold,
            must_fail,
            ..
        } => {
            assert_eq!(noise_adjust, Some(3));
            assert_eq!(noise_spread_threshold, Some(1.5));
            assert!(
                phases_only,
                "--phases-only round-trips under --noise-adjust"
            );
            assert!(steps_only, "--steps-only round-trips under --noise-adjust");
            assert_eq!(
                phase_threshold,
                Some(5.0),
                "--phase-threshold round-trips under --noise-adjust",
            );
            assert!(all_metrics, "--all-metrics round-trips");
            assert_eq!(fail_threshold, Some(3), "--fail-threshold N round-trips");
            assert_eq!(
                must_fail.as_deref(),
                Some("worst_spread"),
                "--must-fail round-trips the raw csv (validated in run())",
            );
        }
        _ => panic!("expected PerfDelta"),
    }
    // --noise-adjust must be >= 2: a single run per side has no spread to
    // observe, so N=1 would emit a confident verdict on pure noise.
    assert!(
        Cargo::try_parse_from(["cargo", "ktstr", "perf-delta", "--noise-adjust", "1"]).is_err(),
        "--noise-adjust 1 must be rejected at parse time (needs >= 2 runs)",
    );
    // --noise-spread-threshold requires --noise-adjust.
    assert!(
        Cargo::try_parse_from([
            "cargo",
            "ktstr",
            "perf-delta",
            "--noise-spread-threshold",
            "1.0"
        ])
        .is_err(),
        "--noise-spread-threshold alone must fail (requires --noise-adjust)",
    );
    // Noise axis conflicts with --threshold / --policy — the noise branch returns
    // before either is used, so a clap conflict prevents them being silently
    // ignored.
    assert!(
        Cargo::try_parse_from([
            "cargo",
            "ktstr",
            "perf-delta",
            "--noise-adjust",
            "3",
            "--threshold",
            "10",
        ])
        .is_err(),
        "--noise-adjust must conflict with --threshold",
    );
    // --threshold and --policy are mutually exclusive.
    assert!(
        Cargo::try_parse_from([
            "cargo",
            "ktstr",
            "perf-delta",
            "--threshold",
            "10",
            "--policy",
            "/tmp/p.json",
        ])
        .is_err(),
        "--threshold and --policy must conflict at parse time",
    );
    // --no-phases conflicts with every other phase flag (all under
    // --noise-adjust, which the phase flags require).
    assert!(
        Cargo::try_parse_from([
            "cargo",
            "ktstr",
            "perf-delta",
            "--noise-adjust",
            "3",
            "--no-phases",
            "--phases-only"
        ])
        .is_err(),
        "--no-phases must conflict with --phases-only",
    );
    // --steps-only conflicts with the single --phase filter.
    assert!(
        Cargo::try_parse_from([
            "cargo",
            "ktstr",
            "perf-delta",
            "--noise-adjust",
            "3",
            "--steps-only",
            "--phase",
            "1"
        ])
        .is_err(),
        "--steps-only must conflict with --phase",
    );
    // Each phase flag REQUIRES --noise-adjust (per-phase output exists only
    // under the noise-adjusted path). A phase flag alone must be rejected at
    // parse time, not silently accepted as an inert no-op on the scalar path.
    assert!(
        Cargo::try_parse_from(["cargo", "ktstr", "perf-delta", "--no-phases"]).is_err(),
        "--no-phases without --noise-adjust must be rejected (requires = noise_adjust)",
    );
    assert!(
        Cargo::try_parse_from(["cargo", "ktstr", "perf-delta", "--phases-only"]).is_err(),
        "--phases-only without --noise-adjust must be rejected (requires = noise_adjust)",
    );
    assert!(
        Cargo::try_parse_from(["cargo", "ktstr", "perf-delta", "--steps-only"]).is_err(),
        "--steps-only without --noise-adjust must be rejected (requires = noise_adjust)",
    );
    assert!(
        Cargo::try_parse_from(["cargo", "ktstr", "perf-delta", "--phase", "1"]).is_err(),
        "--phase without --noise-adjust must be rejected (requires = noise_adjust)",
    );
    assert!(
        Cargo::try_parse_from(["cargo", "ktstr", "perf-delta", "--phase-threshold", "5"]).is_err(),
        "--phase-threshold without --noise-adjust must be rejected (requires = noise_adjust)",
    );
}

/// `--profile <NAME>` (scheduler BUILD profile) and `--nextest-profile
/// <NAME>` (nextest test profile) round-trip on `perf-delta`. `run`
/// forwards both to BOTH sides' `cargo ktstr test` on the run-producing
/// paths; this pins the clap binding so a dropped/renamed derive arg
/// surfaces at parse time.
#[test]
fn parse_perf_delta_with_profiles() {
    let Cargo {
        command: CargoSub::Ktstr(k),
    } = Cargo::try_parse_from([
        "cargo",
        "ktstr",
        "perf-delta",
        "--noise-adjust",
        "3",
        "--kernel",
        "6.14",
        "--profile",
        "dev",
        "--nextest-profile",
        "ci",
    ])
    .unwrap_or_else(|e| panic!("{e}"));
    let KtstrCommand::PerfDelta {
        profile,
        nextest_profile,
        ..
    } = k.command
    else {
        panic!("expected PerfDelta");
    };
    assert_eq!(profile.as_deref(), Some("dev"), "--profile round-trips");
    assert_eq!(
        nextest_profile.as_deref(),
        Some("ci"),
        "--nextest-profile round-trips"
    );
}

/// Build a `cargo ktstr <subcommand> -- <passthrough...>` invocation,
/// parse it, and assert that the trailing args round-trip verbatim
/// into the variant's `args` Vec without spuriously populating any
/// of the named flags (`--kernel`, `--no-perf-mode`, `--no-skip-mode`,
/// `--release`, `--profile`, `--nextest-profile`).
///
/// `subcommand` must be one of the passthrough-bearing subcommands:
/// `test`, `nextest` (alias), `coverage`, `llvm-cov`. Other
/// subcommands panic with an actionable error.
fn assert_passthrough_args(subcommand: &str, passthrough: &[&str]) {
    let mut argv: Vec<&str> = vec!["cargo", "ktstr", subcommand, "--"];
    argv.extend(passthrough.iter().copied());
    let Cargo {
        command: CargoSub::Ktstr(k),
    } = Cargo::try_parse_from(argv).unwrap_or_else(|e| panic!("{e}"));

    let expected: Vec<String> = passthrough.iter().map(|s| s.to_string()).collect();
    match k.command {
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
        } => {
            assert!(
                kernel.is_empty(),
                "bare `--` passthrough must not spuriously populate --kernel",
            );
            assert!(
                !no_perf_mode,
                "bare `--` passthrough must not spuriously set --no-perf-mode",
            );
            assert!(
                !no_skip_mode,
                "bare `--` passthrough must not spuriously set --no-skip-mode",
            );
            assert!(
                !release,
                "bare `--` passthrough must not spuriously set --release",
            );
            assert!(
                profile.is_none(),
                "bare `--` passthrough must not spuriously set --profile",
            );
            assert!(
                nextest_profile.is_none(),
                "bare `--` passthrough must not spuriously set --nextest-profile",
            );
            assert!(
                !include_eol,
                "bare `--` passthrough must not spuriously set --include-eol",
            );
            assert!(
                !relevant && base.is_none() && base_ref.is_none() && default_branch == "main",
                "bare `--` passthrough must not spuriously set --relevant / base flags",
            );
            assert_eq!(args, expected);
        }
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
        } => {
            assert!(
                kernel.is_empty(),
                "bare `--` passthrough must not spuriously populate --kernel",
            );
            assert!(
                !no_perf_mode,
                "bare `--` passthrough must not spuriously set --no-perf-mode",
            );
            assert!(
                !no_skip_mode,
                "bare `--` passthrough must not spuriously set --no-skip-mode",
            );
            assert!(
                !release,
                "bare `--` passthrough must not spuriously set --release",
            );
            assert!(
                profile.is_none(),
                "bare `--` passthrough must not spuriously set --profile",
            );
            assert!(
                nextest_profile.is_none(),
                "bare `--` passthrough must not spuriously set --nextest-profile",
            );
            assert!(
                !include_eol,
                "bare `--` passthrough must not spuriously set --include-eol",
            );
            assert!(
                !relevant && base.is_none() && base_ref.is_none() && default_branch == "main",
                "bare `--` passthrough must not spuriously set --relevant / base flags",
            );
            assert_eq!(args, expected);
        }
        KtstrCommand::LlvmCov {
            kernel,
            no_perf_mode,
            no_skip_mode,
            include_eol,
            args,
        } => {
            assert!(
                kernel.is_empty(),
                "bare `--` passthrough must not spuriously populate --kernel",
            );
            assert!(
                !no_perf_mode,
                "bare `--` passthrough must not spuriously set --no-perf-mode",
            );
            assert!(
                !no_skip_mode,
                "bare `--` passthrough must not spuriously set --no-skip-mode",
            );
            assert!(
                !include_eol,
                "bare `--` passthrough must not spuriously set --include-eol",
            );
            assert_eq!(args, expected);
        }
        _ => panic!("expected passthrough-bearing variant for `{subcommand}`"),
    }
}

// -- structural validation --

/// Run clap's structural self-check on the entire [`Cargo`] derive tree.
///
/// `clap::Command::debug_assert` walks every subcommand, every
/// arg, every group, and every relationship (`conflicts_with`,
/// `requires`, `default_value_if`, `value_parser`, …) and panics
/// at test time on issues that would otherwise surface as cryptic
/// runtime parse errors or silent UX bugs:
///
///   - duplicate arg / subcommand IDs
///   - dangling references in `conflicts_with` / `requires`
///   - default values that fail the arg's `value_parser`
///   - help/version conflicts with user-defined args
///   - misordered positionals (greedy followed by non-greedy)
///
/// Upstream clap recommends running this helper in a unit test for
/// every derive root; we put it FIRST in the parse-tests file so
/// any structural break stops the rest of the suite immediately
/// rather than producing a wall of less-informative downstream
/// failures from individual `try_parse_from` calls.
#[test]
fn cli_debug_assert() {
    Cargo::command().debug_assert();
}

// -- try_get_matches_from: test subcommand --

#[test]
fn parse_test_minimal() {
    let m = Cargo::try_parse_from(["cargo", "ktstr", "test"]);
    assert!(m.is_ok(), "{}", m.err().unwrap());
}

#[test]
fn parse_test_with_kernel() {
    let m = Cargo::try_parse_from(["cargo", "ktstr", "test", "--kernel", "6.14.2"]);
    assert!(m.is_ok(), "{}", m.err().unwrap());
}

/// `--release` on `test` parses to `KtstrCommand::Test { release:
/// true, .. }` so `run_test` prepends `--cargo-profile release`
/// to the cargo nextest invocation. A clap regression that
/// dropped the flag would turn the user-visible `--release` into
/// either a silent no-op (default false) or a passthrough-arg
/// typo — this test pins the clap-level wiring.
#[test]
fn parse_test_with_release_flag() {
    let Cargo {
        command: CargoSub::Ktstr(k),
    } = Cargo::try_parse_from(["cargo", "ktstr", "test", "--release"])
        .unwrap_or_else(|e| panic!("{e}"));
    let KtstrCommand::Test { release, .. } = k.command else {
        panic!("expected Test");
    };
    assert!(release, "`--release` must set `release=true`");
}

/// `--profile <NAME>` on `test` parses to `KtstrCommand::Test {
/// profile: Some(NAME), .. }` so `run_test` exports
/// `KTSTR_SCHEDULER_PROFILE=<NAME>` (the scheduler-under-test's cargo
/// BUILD profile). It is INDEPENDENT of `--release` — passing only
/// `--profile dev` must leave `release=false` so the harness/test binary
/// stays on its default profile. Omitting `--profile` leaves
/// `profile=None` (the scheduler build then defaults to release inside
/// `build_and_find_binary`).
#[test]
fn parse_test_with_profile_flag() {
    let Cargo {
        command: CargoSub::Ktstr(k),
    } = Cargo::try_parse_from(["cargo", "ktstr", "test", "--profile", "dev"])
        .unwrap_or_else(|e| panic!("{e}"));
    let KtstrCommand::Test {
        release, profile, ..
    } = k.command
    else {
        panic!("expected Test");
    };
    assert_eq!(
        profile.as_deref(),
        Some("dev"),
        "`--profile dev` must set profile=Some(\"dev\")"
    );
    assert!(
        !release,
        "`--profile` alone must NOT set --release (the harness stays on its default)"
    );
}

/// `--nextest-profile <NAME>` on `test` parses to `KtstrCommand::Test {
/// nextest_profile: Some(NAME), .. }` so `run_cargo_sub` forwards nextest
/// `--profile <NAME>` (the nextest test profile). Independent of both
/// `--profile` (the scheduler BUILD profile) and `--release`.
#[test]
fn parse_test_with_nextest_profile_flag() {
    let Cargo {
        command: CargoSub::Ktstr(k),
    } = Cargo::try_parse_from(["cargo", "ktstr", "test", "--nextest-profile", "ci"])
        .unwrap_or_else(|e| panic!("{e}"));
    let KtstrCommand::Test {
        release,
        profile,
        nextest_profile,
        ..
    } = k.command
    else {
        panic!("expected Test");
    };
    assert_eq!(
        nextest_profile.as_deref(),
        Some("ci"),
        "`--nextest-profile ci` must set nextest_profile=Some(\"ci\")"
    );
    assert!(
        profile.is_none(),
        "`--nextest-profile` must NOT set --profile (the scheduler build profile)"
    );
    assert!(!release, "`--nextest-profile` alone must NOT set --release");
}

/// `--include-eol` on `test` round-trips to `KtstrCommand::Test {
/// include_eol: true, .. }` so `run_cargo_sub` forwards it into
/// `resolve_kernel_set` → `expand_kernel_range`. Pins the clap
/// binding for the new flag alongside a range `--kernel`.
#[test]
fn parse_test_include_eol_flag() {
    let Cargo {
        command: CargoSub::Ktstr(k),
    } = Cargo::try_parse_from([
        "cargo",
        "ktstr",
        "test",
        "--kernel",
        "6.11..6.14",
        "--include-eol",
    ])
    .unwrap_or_else(|e| panic!("{e}"));
    let KtstrCommand::Test {
        kernel,
        include_eol,
        ..
    } = k.command
    else {
        panic!("expected Test");
    };
    assert_eq!(kernel, vec!["6.11..6.14".to_string()]);
    assert!(
        include_eol,
        "`--include-eol` must round-trip as true so the range expands EOL series"
    );
}

/// `--relevant` + its base flags round-trip on `test`, and a user `-E`
/// stays in the passthrough `args` (it is NOT a ktstr-owned flag on
/// `test`, so `argsplit` leaves it for `compose_relevant_filter` to fold
/// downstream). Goes through `parse_via_split` because a bare `-E` is
/// position-dependent to raw clap; `argsplit::rewrite` routes it into the
/// passthrough exactly as the real `main` does.
#[test]
fn parse_test_relevant_flags() {
    let command = parse_via_split(&[
        "cargo",
        "ktstr",
        "test",
        "--relevant",
        "--base",
        "abc123",
        "--base-ref",
        "release",
        "--default-branch",
        "dev",
        "-E",
        "test(foo)",
    ]);
    let KtstrCommand::Test {
        relevant,
        base,
        base_ref,
        default_branch,
        args,
        ..
    } = command
    else {
        panic!("expected Test");
    };
    assert!(relevant, "--relevant round-trips");
    assert_eq!(base.as_deref(), Some("abc123"));
    assert_eq!(base_ref.as_deref(), Some("release"));
    assert_eq!(default_branch, "dev");
    assert_eq!(
        args,
        vec!["-E".to_string(), "test(foo)".to_string()],
        "a user -E is not ktstr-owned on `test`; it stays in the passthrough",
    );
}

/// Pin passthrough args (the `last = true` field) forwarded verbatim after `--`.
#[test]
fn parse_test_with_passthrough_args() {
    assert_passthrough_args("test", &["-p", "ktstr", "--no-capture"]);
}

// -- try_get_matches_from: `test` visible alias `nextest` --

/// `cargo ktstr nextest` resolves to the canonical `Test`
/// variant. `visible_alias = "nextest"` on the variant makes
/// the alias user-facing (shows in --help) and dispatch-
/// transparent (the existing `KtstrCommand::Test` arm handles
/// both spellings). A regression that dropped the attribute
/// would fail this test at runtime.
#[test]
fn parse_nextest_alias_dispatches_to_test() {
    let Cargo {
        command: CargoSub::Ktstr(k),
    } = Cargo::try_parse_from(["cargo", "ktstr", "nextest"]).unwrap_or_else(|e| panic!("{e}"));
    assert!(
        matches!(k.command, KtstrCommand::Test { .. }),
        "`nextest` alias must dispatch to the Test variant",
    );
}

/// `nextest` alias carries trailing args through the same
/// passthrough (`last = true`) pipeline as `test`. Pins the alias's
/// passthrough behaviour byte-exactly so a clap regression
/// that treated the alias as a distinct parse tree surfaces
/// here rather than in runtime dispatch.
#[test]
fn parse_nextest_alias_with_passthrough_args() {
    assert_passthrough_args("nextest", &["-p", "ktstr", "--no-capture"]);
}

/// Verify the `nextest` alias preserves all Test fields in a
/// single invocation: `--kernel`, `--no-perf-mode`, and empty
/// trailing `args`. A clap regression that silently dropped a
/// field on the alias path (e.g. a derive bug that re-generated
/// the subcommand without inheriting the Test variant's args)
/// would surface here.
#[test]
fn parse_nextest_alias_with_kernel_and_no_perf_mode() {
    let Cargo {
        command: CargoSub::Ktstr(k),
    } = Cargo::try_parse_from([
        "cargo",
        "ktstr",
        "nextest",
        "--kernel",
        "6.14.2",
        "--no-perf-mode",
    ])
    .unwrap_or_else(|e| panic!("{e}"));
    let KtstrCommand::Test {
        kernel,
        no_perf_mode,
        no_skip_mode,
        release,
        profile,
        nextest_profile,
        include_eol,
        args,
        ..
    } = k.command
    else {
        panic!("expected Test (via `nextest` alias)");
    };
    assert_eq!(kernel, vec!["6.14.2".to_string()]);
    assert!(no_perf_mode);
    assert!(!no_skip_mode);
    assert!(!release, "bare invocation must default --release to false");
    assert!(
        profile.is_none(),
        "bare invocation must default --profile to None"
    );
    assert!(
        nextest_profile.is_none(),
        "bare invocation must default --nextest-profile to None"
    );
    assert!(
        !include_eol,
        "bare invocation must default --include-eol to false"
    );
    assert!(args.is_empty());
}

// -- try_get_matches_from: coverage subcommand --

#[test]
fn parse_coverage_minimal() {
    let m = Cargo::try_parse_from(["cargo", "ktstr", "coverage"]);
    assert!(m.is_ok(), "{}", m.err().unwrap());
}

#[test]
fn parse_coverage_with_kernel() {
    let m = Cargo::try_parse_from(["cargo", "ktstr", "coverage", "--kernel", "6.14.2"]);
    assert!(m.is_ok(), "{}", m.err().unwrap());
}

/// `coverage --kernel R --include-eol` round-trips `include_eol=true`
/// so `run_coverage` forwards it into the range expansion.
#[test]
fn parse_coverage_include_eol_flag() {
    let Cargo {
        command: CargoSub::Ktstr(k),
    } = Cargo::try_parse_from([
        "cargo",
        "ktstr",
        "coverage",
        "--kernel",
        "6.11..6.14",
        "--include-eol",
    ])
    .unwrap_or_else(|e| panic!("{e}"));
    let KtstrCommand::Coverage {
        kernel,
        include_eol,
        ..
    } = k.command
    else {
        panic!("expected Coverage");
    };
    assert_eq!(kernel, vec!["6.11..6.14".to_string()]);
    assert!(
        include_eol,
        "`--include-eol` must round-trip as true on coverage"
    );
}

/// `--release` on `coverage` parses to `KtstrCommand::Coverage
/// { release: true, .. }` so `run_coverage` prepends
/// `--cargo-profile release` to the cargo llvm-cov nextest
/// invocation. Same rationale as the sibling
/// `parse_test_with_release_flag` — pins the clap-level wiring
/// against a regression that turns the flag into a no-op.
#[test]
fn parse_coverage_with_release_flag() {
    let Cargo {
        command: CargoSub::Ktstr(k),
    } = Cargo::try_parse_from(["cargo", "ktstr", "coverage", "--release"])
        .unwrap_or_else(|e| panic!("{e}"));
    let KtstrCommand::Coverage { release, .. } = k.command else {
        panic!("expected Coverage");
    };
    assert!(release, "`--release` must set `release=true`");
}

/// `--profile` / `--nextest-profile` on `coverage` parse independently
/// of `--release` (mirrors `parse_test_with_profile_flag` /
/// `parse_test_with_nextest_profile_flag` for the Coverage variant —
/// all three flags are settable on both subcommands).
#[test]
fn parse_coverage_with_profile_flags() {
    let Cargo {
        command: CargoSub::Ktstr(k),
    } = Cargo::try_parse_from([
        "cargo",
        "ktstr",
        "coverage",
        "--profile",
        "dev",
        "--nextest-profile",
        "ci",
    ])
    .unwrap_or_else(|e| panic!("{e}"));
    let KtstrCommand::Coverage {
        release,
        profile,
        nextest_profile,
        ..
    } = k.command
    else {
        panic!("expected Coverage");
    };
    assert_eq!(
        profile.as_deref(),
        Some("dev"),
        "`--profile dev` must set profile=Some(\"dev\") on coverage"
    );
    assert_eq!(
        nextest_profile.as_deref(),
        Some("ci"),
        "`--nextest-profile ci` must set nextest_profile=Some(\"ci\") on coverage"
    );
    assert!(
        !release,
        "`--profile`/`--nextest-profile` alone must NOT set --release"
    );
}

/// Pin passthrough args (the `last = true` field) forwarded verbatim after `--`.
#[test]
fn parse_coverage_with_passthrough_args() {
    assert_passthrough_args(
        "coverage",
        &["--workspace", "--lcov", "--output-path", "lcov.info"],
    );
}

/// Combined round-trip for Coverage: `--kernel`, `--no-perf-mode`,
/// AND trailing args all populate on a single invocation. Mirrors
/// `parse_llvm_cov_with_kernel_and_no_perf_mode` — a clap
/// regression that dropped one field on the multi-flag path (or
/// mis-ordered `--` with flags) would surface here for the
/// Coverage variant.
#[test]
fn parse_coverage_with_kernel_and_no_perf_mode() {
    let Cargo {
        command: CargoSub::Ktstr(k),
    } = Cargo::try_parse_from([
        "cargo",
        "ktstr",
        "coverage",
        "--kernel",
        "6.14.2",
        "--no-perf-mode",
        "--",
        "--workspace",
    ])
    .unwrap_or_else(|e| panic!("{e}"));
    let KtstrCommand::Coverage {
        kernel,
        no_perf_mode,
        no_skip_mode,
        release,
        profile,
        nextest_profile,
        include_eol,
        args,
        ..
    } = k.command
    else {
        panic!("expected Coverage");
    };
    assert_eq!(kernel, vec!["6.14.2".to_string()]);
    assert!(no_perf_mode);
    assert!(!no_skip_mode);
    assert!(!release, "bare invocation must default --release to false");
    assert!(
        profile.is_none(),
        "bare invocation must default --profile to None"
    );
    assert!(
        nextest_profile.is_none(),
        "bare invocation must default --nextest-profile to None"
    );
    assert!(
        !include_eol,
        "bare invocation must default --include-eol to false"
    );
    assert_eq!(args, vec!["--workspace"]);
}

// -- try_get_matches_from: llvm-cov raw passthrough subcommand --

#[test]
fn parse_llvm_cov_minimal() {
    let m = Cargo::try_parse_from(["cargo", "ktstr", "llvm-cov"]);
    assert!(m.is_ok(), "{}", m.err().unwrap());
}

#[test]
fn parse_llvm_cov_with_kernel() {
    let Cargo {
        command: CargoSub::Ktstr(k),
    } = Cargo::try_parse_from(["cargo", "ktstr", "llvm-cov", "--kernel", "6.14.2"])
        .unwrap_or_else(|e| panic!("{e}"));
    let KtstrCommand::LlvmCov { kernel, .. } = k.command else {
        panic!("expected LlvmCov");
    };
    assert_eq!(kernel, vec!["6.14.2".to_string()]);
}

/// `llvm-cov --kernel R --include-eol` round-trips `include_eol=true`
/// so `run_llvm_cov` forwards it into the range expansion.
#[test]
fn parse_llvm_cov_include_eol_flag() {
    let Cargo {
        command: CargoSub::Ktstr(k),
    } = Cargo::try_parse_from([
        "cargo",
        "ktstr",
        "llvm-cov",
        "--kernel",
        "6.11..6.14",
        "--include-eol",
    ])
    .unwrap_or_else(|e| panic!("{e}"));
    let KtstrCommand::LlvmCov {
        kernel,
        include_eol,
        ..
    } = k.command
    else {
        panic!("expected LlvmCov");
    };
    assert_eq!(kernel, vec!["6.11..6.14".to_string()]);
    assert!(
        include_eol,
        "`--include-eol` must round-trip as true on llvm-cov"
    );
}

/// Pin passthrough args (the `last = true` field) forwarded verbatim after `--`.
#[test]
fn parse_llvm_cov_with_passthrough_args() {
    assert_passthrough_args(
        "llvm-cov",
        &["report", "--lcov", "--output-path", "lcov.info"],
    );
}

/// Combined round-trip: `--kernel`, `--no-perf-mode`, AND
/// trailing args all populate on a single LlvmCov invocation.
/// A clap regression that dropped one field on the multi-flag
/// path (or mis-ordered `--` with flags) would surface here.
#[test]
fn parse_llvm_cov_with_kernel_and_no_perf_mode() {
    let Cargo {
        command: CargoSub::Ktstr(k),
    } = Cargo::try_parse_from([
        "cargo",
        "ktstr",
        "llvm-cov",
        "--kernel",
        "6.14.2",
        "--no-perf-mode",
        "--",
        "report",
        "--lcov",
    ])
    .unwrap_or_else(|e| panic!("{e}"));
    let KtstrCommand::LlvmCov {
        kernel,
        no_perf_mode,
        no_skip_mode,
        include_eol,
        args,
    } = k.command
    else {
        panic!("expected LlvmCov");
    };
    assert_eq!(kernel, vec!["6.14.2".to_string()]);
    assert!(no_perf_mode);
    assert!(!no_skip_mode);
    assert!(
        !include_eol,
        "bare invocation must default --include-eol to false"
    );
    assert_eq!(args, vec!["report", "--lcov"]);
}

/// Negative pin: the variant is `LlvmCov`, and clap derive's
/// default casing is kebab-case (see clap_derive
/// `DEFAULT_CASING`), so the subcommand name is `llvm-cov`,
/// NOT `llvm_cov`. A regression that switched the derive's
/// rename_all default (or silently aliased the underscore
/// form) would turn this negative pin positive. The parent-
/// level `aliases` slot is empty, so clap rejects the
/// underscore form with an unknown-subcommand error.
#[test]
fn parse_llvm_cov_underscore_rejected() {
    let rejected = Cargo::try_parse_from(["cargo", "ktstr", "llvm_cov"]);
    assert!(
        rejected.is_err(),
        "`llvm_cov` (underscore) must be rejected — the \
         canonical name is `llvm-cov` (kebab-case)",
    );
}

/// Positive companion to [`parse_llvm_cov_underscore_rejected`]:
/// the kebab-case form `llvm-cov` MUST resolve to
/// [`KtstrCommand::LlvmCov`] without alias indirection. The
/// existing `parse_llvm_cov_minimal` exercises the spelling but
/// only asserts `is_ok()` — this test pins the variant binding
/// so that a future rename of the derive variant or the
/// subcommand attribute (e.g. `command(name = "llvm-coverage")`)
/// surfaces here as a variant-mismatch panic instead of silently
/// breaking under a renamed-but-still-parseable form.
#[test]
fn parse_llvm_cov_kebab_accepted() {
    let Cargo {
        command: CargoSub::Ktstr(k),
    } = Cargo::try_parse_from(["cargo", "ktstr", "llvm-cov"]).unwrap_or_else(|e| panic!("{e}"));
    assert!(
        matches!(k.command, KtstrCommand::LlvmCov { .. }),
        "kebab `llvm-cov` must bind to KtstrCommand::LlvmCov",
    );
}

// -- try_get_matches_from: shell subcommand --

#[test]
fn parse_shell_minimal() {
    let m = Cargo::try_parse_from(["cargo", "ktstr", "shell"]);
    assert!(m.is_ok(), "{}", m.err().unwrap());
}

#[test]
fn parse_shell_with_topology() {
    let Cargo {
        command: CargoSub::Ktstr(k),
    } = Cargo::try_parse_from(["cargo", "ktstr", "shell", "--topology", "1,2,4,1"])
        .unwrap_or_else(|e| panic!("{e}"));
    let KtstrCommand::Shell { topology, .. } = k.command else {
        panic!("expected Shell");
    };
    assert_eq!(topology, "1,2,4,1");
}

#[test]
fn parse_shell_default_topology() {
    let Cargo {
        command: CargoSub::Ktstr(k),
    } = Cargo::try_parse_from(["cargo", "ktstr", "shell"]).unwrap_or_else(|e| panic!("{e}"));
    let KtstrCommand::Shell { topology, .. } = k.command else {
        panic!("expected Shell");
    };
    assert_eq!(topology, "1,1,1,1");
}

/// Pin `-i` / `--include-files` `ArgAction::Append` round-trip with ordering.
#[test]
fn parse_shell_include_files() {
    let Cargo {
        command: CargoSub::Ktstr(k),
    } = Cargo::try_parse_from(["cargo", "ktstr", "shell", "-i", "/tmp/a", "-i", "/tmp/b"])
        .unwrap_or_else(|e| panic!("{e}"));
    let KtstrCommand::Shell { include_files, .. } = k.command else {
        panic!("expected Shell");
    };
    assert_eq!(
        include_files,
        vec![
            std::path::PathBuf::from("/tmp/a"),
            std::path::PathBuf::from("/tmp/b"),
        ],
        "-i flag must accumulate paths in order via ArgAction::Append",
    );
}

/// `cargo ktstr shell --disk 256mib` parses; the disk arg lands as
/// `Some("256mib")` on the `Shell` variant. The string is parsed
/// into a `DiskConfig` later in `run_shell` via
/// [`ktstr::cli::parse_disk_size_mib`]; the clap stage stores the
/// raw string so a malformed input surfaces with the consistent
/// disk-size diagnostic instead of a generic clap parse error.
#[test]
fn parse_shell_disk_arg() {
    let Cargo {
        command: CargoSub::Ktstr(k),
    } = Cargo::try_parse_from(["cargo", "ktstr", "shell", "--disk", "256mib"])
        .unwrap_or_else(|e| panic!("{e}"));
    let KtstrCommand::Shell { disk, .. } = k.command else {
        panic!("expected Shell");
    };
    assert_eq!(disk.as_deref(), Some("256mib"));
}

/// Omitting `--disk` produces `None`, matching the no-disk default
/// in `run_shell` and `KtstrVm::builder`.
#[test]
fn parse_shell_disk_arg_omitted() {
    let Cargo {
        command: CargoSub::Ktstr(k),
    } = Cargo::try_parse_from(["cargo", "ktstr", "shell"]).unwrap_or_else(|e| panic!("{e}"));
    let KtstrCommand::Shell { disk, .. } = k.command else {
        panic!("expected Shell");
    };
    assert!(disk.is_none(), "no --disk must produce None");
}

// -- try_get_matches_from: stats subcommand --

#[test]
fn parse_stats_bare() {
    let m = Cargo::try_parse_from(["cargo", "ktstr", "stats"]);
    assert!(m.is_ok(), "{}", m.err().unwrap());
}

#[test]
fn parse_stats_list() {
    let m = Cargo::try_parse_from(["cargo", "ktstr", "stats", "list"]);
    assert!(m.is_ok(), "{}", m.err().unwrap());
}

/// `cargo ktstr stats list-metrics` parses (no flags required)
/// and dispatches to the `ListMetrics` variant with `json=false`.
#[test]
fn parse_stats_list_metrics_bare() {
    let Cargo {
        command: CargoSub::Ktstr(k),
    } = Cargo::try_parse_from(["cargo", "ktstr", "stats", "list-metrics"])
        .unwrap_or_else(|e| panic!("{e}"));
    let KtstrCommand::Stats {
        command: Some(StatsCommand::ListMetrics { json }),
        ..
    } = k.command
    else {
        panic!("expected Stats ListMetrics");
    };
    assert!(
        !json,
        "bare `list-metrics` must default to text mode (json=false)",
    );
}

/// `cargo ktstr stats list-metrics --json` sets `json=true`.
/// Pins the flag name so a clap-derive-default rename
/// (kebab-case) cannot drift — `--json` is the same flag name
/// other list-style subcommands use (e.g. `kernel list --json`).
#[test]
fn parse_stats_list_metrics_json() {
    let Cargo {
        command: CargoSub::Ktstr(k),
    } = Cargo::try_parse_from(["cargo", "ktstr", "stats", "list-metrics", "--json"])
        .unwrap_or_else(|e| panic!("{e}"));
    let KtstrCommand::Stats {
        command: Some(StatsCommand::ListMetrics { json }),
        ..
    } = k.command
    else {
        panic!("expected Stats ListMetrics");
    };
    assert!(json, "--json must set the flag true");
}

/// `list-metrics` takes no positional args — a stray positional
/// must be rejected by clap so a typo like `list-metrics
/// worst_spread` doesn't silently look like success.
#[test]
fn parse_stats_list_metrics_rejects_positional() {
    let rejected =
        Cargo::try_parse_from(["cargo", "ktstr", "stats", "list-metrics", "worst_spread"]);
    assert!(
        rejected.is_err(),
        "list-metrics must reject positional arguments",
    );
}

/// `cargo ktstr stats list-values` parses with no flags and
/// dispatches to the `ListValues` variant with `json=false` and
/// `dir=None`. Pins the bare-call defaults.
#[test]
fn parse_stats_list_values_bare() {
    let Cargo {
        command: CargoSub::Ktstr(k),
    } = Cargo::try_parse_from(["cargo", "ktstr", "stats", "list-values"])
        .unwrap_or_else(|e| panic!("{e}"));
    let KtstrCommand::Stats {
        command: Some(StatsCommand::ListValues { json, dir }),
        ..
    } = k.command
    else {
        panic!("expected Stats ListValues");
    };
    assert!(!json, "bare `list-values` must default to text mode");
    assert!(
        dir.is_none(),
        "bare `list-values` must default to no --dir override"
    );
}

/// `cargo ktstr stats list-values --json` sets `json=true`.
/// Pins the flag name so the same `--json` convention used by
/// `list-metrics` and `kernel list` carries here too.
#[test]
fn parse_stats_list_values_json() {
    let Cargo {
        command: CargoSub::Ktstr(k),
    } = Cargo::try_parse_from(["cargo", "ktstr", "stats", "list-values", "--json"])
        .unwrap_or_else(|e| panic!("{e}"));
    let KtstrCommand::Stats {
        command: Some(StatsCommand::ListValues { json, .. }),
        ..
    } = k.command
    else {
        panic!("expected Stats ListValues");
    };
    assert!(json, "--json must set the flag true");
}

/// `cargo ktstr stats list-values --dir PATH` round-trips the
/// path through clap to the dispatch site. Same `--dir`
/// convention as `show-host --dir` and `list-values --dir`.
#[test]
fn parse_stats_list_values_with_dir() {
    let Cargo {
        command: CargoSub::Ktstr(k),
    } = Cargo::try_parse_from([
        "cargo",
        "ktstr",
        "stats",
        "list-values",
        "--dir",
        "/tmp/archived-runs",
    ])
    .unwrap_or_else(|e| panic!("{e}"));
    let KtstrCommand::Stats {
        command: Some(StatsCommand::ListValues { dir, json }),
        ..
    } = k.command
    else {
        panic!("expected Stats ListValues");
    };
    assert_eq!(
        dir.as_deref(),
        Some(std::path::Path::new("/tmp/archived-runs")),
        "--dir must round-trip to Some(PathBuf)",
    );
    assert!(!json, "bare --dir must not spuriously set --json");
}

/// `list-values` takes no positional args — clap must reject
/// strays so a typo like `list-values kernel` (intending a
/// per-dim filter) fails loudly rather than getting silently
/// dropped.
#[test]
fn parse_stats_list_values_rejects_positional() {
    let rejected = Cargo::try_parse_from(["cargo", "ktstr", "stats", "list-values", "kernel"]);
    assert!(
        rejected.is_err(),
        "list-values must reject positional arguments",
    );
}

/// `cargo ktstr stats show-host --run X` parses to
/// `StatsCommand::ShowHost { run: X, dir: None }`.
#[test]
fn parse_stats_show_host_with_run() {
    let Cargo {
        command: CargoSub::Ktstr(k),
    } = Cargo::try_parse_from(["cargo", "ktstr", "stats", "show-host", "--run", "my-run-id"])
        .unwrap_or_else(|e| panic!("{e}"));
    let KtstrCommand::Stats {
        command: Some(StatsCommand::ShowHost { run, dir }),
        ..
    } = k.command
    else {
        panic!("expected Stats ShowHost");
    };
    assert_eq!(run, "my-run-id");
    assert!(dir.is_none(), "bare --run must not populate --dir");
}

/// `cargo ktstr stats show-host --run X --dir PATH` carries
/// both flags through. Same --dir threading contract as
/// `compare` — parse layer preserves the PathBuf; resolution
/// against `runs_root()` is `cli::show_run_host`'s job.
#[test]
fn parse_stats_show_host_with_dir() {
    let Cargo {
        command: CargoSub::Ktstr(k),
    } = Cargo::try_parse_from([
        "cargo",
        "ktstr",
        "stats",
        "show-host",
        "--run",
        "archive-2024-01-15",
        "--dir",
        "/tmp/archived-runs",
    ])
    .unwrap_or_else(|e| panic!("{e}"));
    let KtstrCommand::Stats {
        command: Some(StatsCommand::ShowHost { run, dir }),
        ..
    } = k.command
    else {
        panic!("expected Stats ShowHost");
    };
    assert_eq!(run, "archive-2024-01-15");
    assert_eq!(
        dir.as_deref(),
        Some(std::path::Path::new("/tmp/archived-runs")),
    );
}

/// `cargo ktstr stats show-host` WITHOUT `--run` must fail at
/// parse time — the flag is required and clap's default shape
/// says so. A regression that accidentally made `--run`
/// optional would silently let operators invoke the command
/// with no target, producing a no-op failure.
#[test]
fn parse_stats_show_host_missing_run_rejected() {
    let rejected = Cargo::try_parse_from(["cargo", "ktstr", "stats", "show-host"]);
    assert!(rejected.is_err(), "stats show-host must require --run",);
}

/// `cargo ktstr stats explain-sidecar --run X` parses to
/// `StatsCommand::ExplainSidecar { run: X, dir: None,
/// json: false }`. Mirrors `parse_stats_show_host_with_run`
/// for the explain-sidecar shape.
#[test]
fn parse_stats_explain_sidecar_with_run() {
    let Cargo {
        command: CargoSub::Ktstr(k),
    } = Cargo::try_parse_from([
        "cargo",
        "ktstr",
        "stats",
        "explain-sidecar",
        "--run",
        "my-run-id",
    ])
    .unwrap_or_else(|e| panic!("{e}"));
    let KtstrCommand::Stats {
        command: Some(StatsCommand::ExplainSidecar { run, dir, json }),
        ..
    } = k.command
    else {
        panic!("expected Stats ExplainSidecar");
    };
    assert_eq!(run, "my-run-id");
    assert!(dir.is_none(), "bare --run must not populate --dir");
    assert!(!json, "default output is text, not json");
}

/// `cargo ktstr stats explain-sidecar --run X --dir PATH
/// --json` carries all three flags. Same --dir threading
/// contract as `show-host`; the `--json` flag toggles the
/// aggregate-by-field output shape.
#[test]
fn parse_stats_explain_sidecar_with_dir_and_json() {
    let Cargo {
        command: CargoSub::Ktstr(k),
    } = Cargo::try_parse_from([
        "cargo",
        "ktstr",
        "stats",
        "explain-sidecar",
        "--run",
        "archive-2024-01-15",
        "--dir",
        "/tmp/archived-runs",
        "--json",
    ])
    .unwrap_or_else(|e| panic!("{e}"));
    let KtstrCommand::Stats {
        command: Some(StatsCommand::ExplainSidecar { run, dir, json }),
        ..
    } = k.command
    else {
        panic!("expected Stats ExplainSidecar");
    };
    assert_eq!(run, "archive-2024-01-15");
    assert_eq!(
        dir.as_deref(),
        Some(std::path::Path::new("/tmp/archived-runs")),
    );
    assert!(json, "--json must toggle aggregate JSON output");
}

/// `cargo ktstr stats explain-sidecar` WITHOUT `--run` must
/// fail at parse time. Same required-flag contract as
/// `show-host`; without it, an operator could invoke the
/// command with no target.
#[test]
fn parse_stats_explain_sidecar_missing_run_rejected() {
    let rejected = Cargo::try_parse_from(["cargo", "ktstr", "stats", "explain-sidecar"]);
    assert!(
        rejected.is_err(),
        "stats explain-sidecar must require --run",
    );
}

// -- try_get_matches_from: kernel list --

#[test]
fn parse_kernel_list() {
    let m = Cargo::try_parse_from(["cargo", "ktstr", "kernel", "list"]);
    assert!(m.is_ok(), "{}", m.err().unwrap());
}

#[test]
fn parse_kernel_list_json() {
    let m = Cargo::try_parse_from(["cargo", "ktstr", "kernel", "list", "--json"]);
    assert!(m.is_ok(), "{}", m.err().unwrap());
}

/// `affected` base flags round-trip; bare `affected` defaults base/base_ref to
/// None and default_branch to "main" (mirrors the perf-delta base block).
#[test]
fn parse_affected_flags_and_defaults() {
    let Cargo {
        command: CargoSub::Ktstr(k),
    } = Cargo::try_parse_from(["cargo", "ktstr", "affected"]).unwrap_or_else(|e| panic!("{e}"));
    match k.command {
        KtstrCommand::Affected {
            base,
            base_ref,
            default_branch,
        } => {
            assert!(base.is_none() && base_ref.is_none());
            assert_eq!(default_branch, "main");
        }
        _ => panic!("expected Affected"),
    }
    let Cargo {
        command: CargoSub::Ktstr(k),
    } = Cargo::try_parse_from([
        "cargo",
        "ktstr",
        "affected",
        "--base",
        "abc123",
        "--base-ref",
        "release",
        "--default-branch",
        "dev",
    ])
    .unwrap_or_else(|e| panic!("{e}"));
    match k.command {
        KtstrCommand::Affected {
            base,
            base_ref,
            default_branch,
        } => {
            assert_eq!(base.as_deref(), Some("abc123"));
            assert_eq!(base_ref.as_deref(), Some("release"));
            assert_eq!(default_branch, "dev");
        }
        _ => panic!("expected Affected"),
    }
}

/// `affected` takes no positional/passthrough argument.
#[test]
fn parse_affected_rejects_positional() {
    assert!(
        Cargo::try_parse_from(["cargo", "ktstr", "affected", "stray-positional"]).is_err(),
        "affected must reject a bare positional argument",
    );
}

/// `kernel list --kernel R` round-trips to
/// `KernelCommand::List { kernel: Some(R), .. }` so the
/// dispatch site routes through `kernel_list_range_preview`
/// rather than the cache-walk path. Pins the clap binding
/// for the `--kernel` flag — a regression that dropped the
/// `kernel` field from the Subcommand enum would surface
/// here as a parse rejection.
#[test]
fn parse_kernel_list_range() {
    let Cargo {
        command: CargoSub::Ktstr(k),
    } = Cargo::try_parse_from(["cargo", "ktstr", "kernel", "list", "--kernel", "6.12..6.14"])
        .unwrap_or_else(|e| panic!("{e}"));
    let KtstrCommand::Kernel { command } = k.command else {
        panic!("expected Kernel");
    };
    let KernelCommand::List {
        json,
        kernel,
        include_eol,
    } = command
    else {
        panic!("expected KernelCommand::List, got {command:?}");
    };
    assert!(!json, "bare --kernel must not enable --json");
    assert_eq!(
        kernel.as_deref(),
        Some("6.12..6.14"),
        "--kernel must round-trip the literal spec for \
         dispatch to pass to `expand_kernel_range`",
    );
    assert!(
        !include_eol,
        "bare --kernel range must default --include-eol to false"
    );
}

/// `kernel list --kernel R --json` round-trips both flags.
/// Pins the JSON-output mode is reachable on the range-preview
/// path (a regression that wired `--kernel` only on the text
/// path would surface here).
#[test]
fn parse_kernel_list_range_with_json() {
    let Cargo {
        command: CargoSub::Ktstr(k),
    } = Cargo::try_parse_from([
        "cargo",
        "ktstr",
        "kernel",
        "list",
        "--kernel",
        "6.12..6.14",
        "--json",
    ])
    .unwrap_or_else(|e| panic!("{e}"));
    let KtstrCommand::Kernel { command } = k.command else {
        panic!("expected Kernel");
    };
    let KernelCommand::List {
        json,
        kernel,
        include_eol,
    } = command
    else {
        panic!("expected KernelCommand::List, got {command:?}");
    };
    assert!(json, "--json must round-trip alongside --kernel");
    assert_eq!(kernel.as_deref(), Some("6.12..6.14"));
    assert!(
        !include_eol,
        "--kernel --json without --include-eol must default it to false"
    );
}

/// `kernel list --kernel R --include-eol` round-trips
/// `include_eol=true` so the preview enumerates EOL series.
#[test]
fn parse_kernel_list_range_include_eol() {
    let Cargo {
        command: CargoSub::Ktstr(k),
    } = Cargo::try_parse_from([
        "cargo",
        "ktstr",
        "kernel",
        "list",
        "--kernel",
        "6.11..6.14",
        "--include-eol",
    ])
    .unwrap_or_else(|e| panic!("{e}"));
    let KtstrCommand::Kernel { command } = k.command else {
        panic!("expected Kernel");
    };
    let KernelCommand::List {
        kernel,
        include_eol,
        ..
    } = command
    else {
        panic!("expected KernelCommand::List, got {command:?}");
    };
    assert_eq!(kernel.as_deref(), Some("6.11..6.14"));
    assert!(
        include_eol,
        "`--include-eol` must round-trip as true on kernel list --kernel"
    );
}

// -- try_get_matches_from: kernel build --

#[test]
fn parse_kernel_build_version() {
    let m = Cargo::try_parse_from(["cargo", "ktstr", "kernel", "build", "--kernel", "6.14.2"]);
    assert!(m.is_ok(), "{}", m.err().unwrap());
}

/// `kernel build RANGE --include-eol` round-trips `include_eol=true`
/// so the range-build path expands EOL series.
#[test]
fn parse_kernel_build_range_include_eol() {
    let Cargo {
        command: CargoSub::Ktstr(k),
    } = Cargo::try_parse_from([
        "cargo",
        "ktstr",
        "kernel",
        "build",
        "--kernel",
        "6.11..6.14",
        "--include-eol",
    ])
    .unwrap_or_else(|e| panic!("{e}"));
    let KtstrCommand::Kernel { command } = k.command else {
        panic!("expected Kernel");
    };
    let KernelCommand::Build {
        kernel,
        include_eol,
        ..
    } = command
    else {
        panic!("expected KernelCommand::Build, got {command:?}");
    };
    assert_eq!(kernel.as_deref(), Some("6.11..6.14"));
    assert!(
        include_eol,
        "`--include-eol` must round-trip as true on kernel build --kernel RANGE"
    );
}

#[test]
fn parse_kernel_build_path() {
    let m = Cargo::try_parse_from(["cargo", "ktstr", "kernel", "build", "--kernel", "../linux"]);
    assert!(m.is_ok(), "{}", m.err().unwrap());
}

/// `kernel build --kernel git+URL#tag=NAME` round-trips a git source
/// to `KernelCommand::Build { kernel: Some(git+…), .. }` so the
/// dispatch routes through `resolve_git_kernel`. The old `--git` /
/// `--ref` flags are retired; the explicit git grammar on `--kernel`
/// is the one git surface (and additionally supports `#branch=` /
/// `#sha=`).
#[test]
fn parse_kernel_build_git() {
    let Cargo {
        command: CargoSub::Ktstr(k),
    } = Cargo::try_parse_from([
        "cargo",
        "ktstr",
        "kernel",
        "build",
        "--kernel",
        "git+https://example.com/linux.git#tag=v6.14",
    ])
    .unwrap_or_else(|e| panic!("{e}"));
    let KtstrCommand::Kernel {
        command: KernelCommand::Build { kernel, .. },
    } = k.command
    else {
        panic!("expected KernelCommand::Build");
    };
    assert_eq!(
        kernel.as_deref(),
        Some("git+https://example.com/linux.git#tag=v6.14"),
        "a git source must round-trip verbatim through --kernel",
    );
}

/// `kernel build --kernel VERSION --extra-kconfig PATH` round-trips to
/// `KernelCommand::Build { kernel: Some(..), extra_kconfig:
/// Some(..), .. }` so the dispatch site forwards the path
/// through `kernel_build` → `kernel_build_one` →
/// `cli::kernel_build_pipeline` with `Some(content)`. Pins the
/// clap binding for the new flag — a regression that dropped
/// the field would surface here as a parse rejection or a None
/// `extra_kconfig`.
#[test]
fn parse_kernel_build_with_extra_kconfig() {
    let Cargo {
        command: CargoSub::Ktstr(k),
    } = Cargo::try_parse_from([
        "cargo",
        "ktstr",
        "kernel",
        "build",
        "--kernel",
        "6.14.2",
        "--extra-kconfig",
        "/tmp/extra.kconfig",
    ])
    .unwrap_or_else(|e| panic!("{e}"));
    let KtstrCommand::Kernel {
        command:
            KernelCommand::Build {
                kernel,
                extra_kconfig,
                ..
            },
    } = k.command
    else {
        panic!("expected KernelCommand::Build");
    };
    assert_eq!(kernel.as_deref(), Some("6.14.2"));
    assert_eq!(
        extra_kconfig,
        Some(std::path::PathBuf::from("/tmp/extra.kconfig")),
        "--extra-kconfig must round-trip the literal path",
    );
}

/// Bare `kernel build VERSION` (no `--extra-kconfig`) parses to
/// `extra_kconfig: None`. Pins that the flag is OPTIONAL — a
/// regression that made it required would fail this test.
#[test]
fn parse_kernel_build_without_extra_kconfig_is_none() {
    let Cargo {
        command: CargoSub::Ktstr(k),
    } = Cargo::try_parse_from(["cargo", "ktstr", "kernel", "build", "--kernel", "6.14.2"])
        .unwrap_or_else(|e| panic!("{e}"));
    let KtstrCommand::Kernel {
        command: KernelCommand::Build { extra_kconfig, .. },
    } = k.command
    else {
        panic!("expected KernelCommand::Build");
    };
    assert!(
        extra_kconfig.is_none(),
        "no --extra-kconfig must produce None, got {extra_kconfig:?}",
    );
}

/// `kernel build VERSION --skip-sha256` round-trips to
/// `KernelCommand::Build { skip_sha256: true, .. }` so the
/// dispatch site forwards the boolean through `kernel_build` →
/// `kernel_build_one` → `fetch::download_tarball` →
/// `download_stable_tarball`. Pins the clap binding for the
/// security-sensitive bypass flag — a regression that dropped
/// the field or flipped the default would surface here.
#[test]
fn parse_kernel_build_with_skip_sha256() {
    let Cargo {
        command: CargoSub::Ktstr(k),
    } = Cargo::try_parse_from([
        "cargo",
        "ktstr",
        "kernel",
        "build",
        "--kernel",
        "6.14.2",
        "--skip-sha256",
    ])
    .unwrap_or_else(|e| panic!("{e}"));
    let KtstrCommand::Kernel {
        command:
            KernelCommand::Build {
                kernel,
                skip_sha256,
                ..
            },
    } = k.command
    else {
        panic!("expected KernelCommand::Build");
    };
    assert_eq!(kernel.as_deref(), Some("6.14.2"));
    assert!(
        skip_sha256,
        "--skip-sha256 must round-trip as true; without this the \
         download path would still verify against sha256sums.asc"
    );
}

/// Bare `kernel build VERSION` (no `--skip-sha256`) parses to
/// `skip_sha256: false`. Pins the safe default — a regression
/// that flipped the default to true would silently disable
/// checksum verification on every download.
#[test]
fn parse_kernel_build_without_skip_sha256_is_false() {
    let Cargo {
        command: CargoSub::Ktstr(k),
    } = Cargo::try_parse_from(["cargo", "ktstr", "kernel", "build", "--kernel", "6.14.2"])
        .unwrap_or_else(|e| panic!("{e}"));
    let KtstrCommand::Kernel {
        command: KernelCommand::Build { skip_sha256, .. },
    } = k.command
    else {
        panic!("expected KernelCommand::Build");
    };
    assert!(
        !skip_sha256,
        "absent --skip-sha256 must produce skip_sha256: false — \
         the default must keep checksum verification enabled, \
         got skip_sha256={skip_sha256}"
    );
}

/// `--skip-sha256` works alongside a `--kernel <path>` source tree.
/// Pins that the flag is not mutually exclusive with a path source —
/// skip-sha256 is documented as a no-op there (a path or git source
/// downloads no tarball), but clap must still ACCEPT the combination
/// so the help-text-promised orthogonality holds at parse time.
#[test]
fn parse_kernel_build_skip_sha256_with_path() {
    let Cargo {
        command: CargoSub::Ktstr(k),
    } = Cargo::try_parse_from([
        "cargo",
        "ktstr",
        "kernel",
        "build",
        "--kernel",
        "/tmp/src",
        "--skip-sha256",
    ])
    .unwrap_or_else(|e| panic!("{e}"));
    let KtstrCommand::Kernel {
        command:
            KernelCommand::Build {
                kernel,
                skip_sha256,
                ..
            },
    } = k.command
    else {
        panic!("expected KernelCommand::Build");
    };
    assert_eq!(kernel.as_deref(), Some("/tmp/src"));
    assert!(
        skip_sha256,
        "--skip-sha256 must round-trip when combined with a --kernel \
         <path> source (the help text promises the flag is a no-op \
         there, but clap must still accept the combination)"
    );
}

/// Underscore form `--skip_sha256` MUST be rejected by clap. The
/// canonical name is `--skip-sha256` (kebab-case). A regression
/// that added an alias for the underscore form (or changed the
/// arg-name parser to accept either) would turn this negative
/// pin positive. Mirrors `parse_llvm_cov_underscore_rejected`.
#[test]
fn parse_kernel_build_skip_sha256_underscore_rejected() {
    let rejected = Cargo::try_parse_from([
        "cargo",
        "ktstr",
        "kernel",
        "build",
        "--kernel",
        "6.14.2",
        "--skip_sha256",
    ]);
    assert!(
        rejected.is_err(),
        "`--skip_sha256` (underscore) must be rejected — the \
         canonical name is `--skip-sha256` (kebab-case)",
    );
}

/// Range expansion + --skip-sha256 composes at the parse layer.
/// A range version + the bypass flag both round-trip to their
/// fields on `KernelCommand::Build`. The dispatch then fans out
/// per version inside `kernel_build`, threading the same
/// `skip_sha256` boolean to every `kernel_build_one` call — so
/// every version in a range observes the same bypass setting.
/// Pin the parse-level composition.
#[test]
fn parse_kernel_build_skip_sha256_range_compose() {
    let Cargo {
        command: CargoSub::Ktstr(k),
    } = Cargo::try_parse_from([
        "cargo",
        "ktstr",
        "kernel",
        "build",
        "--kernel",
        "6.14.2..6.14.4",
        "--skip-sha256",
    ])
    .unwrap_or_else(|e| panic!("{e}"));
    let KtstrCommand::Kernel {
        command:
            KernelCommand::Build {
                kernel,
                skip_sha256,
                ..
            },
    } = k.command
    else {
        panic!("expected KernelCommand::Build");
    };
    assert_eq!(kernel.as_deref(), Some("6.14.2..6.14.4"));
    assert!(
        skip_sha256,
        "--skip-sha256 must round-trip on a range version so every \
         per-version `kernel_build_one` invocation sees the bypass \
         flag"
    );
}

/// Four-flag orthogonality: --skip-sha256 + --extra-kconfig +
/// --force + --clean must all coexist on a single `kernel build`
/// invocation. None pair conflicts. A regression that introduced
/// a clap `conflicts_with` between any pair (e.g. wrongly tying
/// --skip-sha256 to --force "for safety") would surface here.
/// Mirrors `parse_kernel_build_force_clean_and_extra_kconfig_compose`.
#[test]
fn parse_kernel_build_skip_sha256_with_extra_kconfig_and_force_clean() {
    let Cargo {
        command: CargoSub::Ktstr(k),
    } = Cargo::try_parse_from([
        "cargo",
        "ktstr",
        "kernel",
        "build",
        "--kernel",
        "6.14.2",
        "--skip-sha256",
        "--extra-kconfig",
        "/tmp/k",
        "--force",
        "--clean",
    ])
    .unwrap_or_else(|e| panic!("{e}"));
    let KtstrCommand::Kernel {
        command:
            KernelCommand::Build {
                force,
                clean,
                extra_kconfig,
                skip_sha256,
                ..
            },
    } = k.command
    else {
        panic!("expected KernelCommand::Build");
    };
    assert!(
        force,
        "--force must round-trip alongside --skip-sha256 + --clean + --extra-kconfig"
    );
    assert!(
        clean,
        "--clean must round-trip alongside --skip-sha256 + --force + --extra-kconfig"
    );
    assert_eq!(
        extra_kconfig,
        Some(std::path::PathBuf::from("/tmp/k")),
        "--extra-kconfig must round-trip alongside --skip-sha256 + --force + --clean"
    );
    assert!(
        skip_sha256,
        "--skip-sha256 must round-trip alongside --force + --clean + --extra-kconfig"
    );
}

/// Range expansion + --extra-kconfig composes at the parse
/// layer. A range version + an extra-kconfig path both round-
/// trip to their fields on `KernelCommand::Build`. The dispatch
/// then fans out per version inside `kernel_build`, and the
/// `extra_content` String is read ONCE up front and threaded as
/// `Option<&str>` to every `kernel_build_one` call — so every
/// version in a range observes byte-identical extras. Pin the
/// parse-level composition; the per-version threading is a
/// code-structure invariant of `kernel_build`'s shared read.
#[test]
fn parse_kernel_build_range_with_extra_kconfig() {
    let Cargo {
        command: CargoSub::Ktstr(k),
    } = Cargo::try_parse_from([
        "cargo",
        "ktstr",
        "kernel",
        "build",
        "--kernel",
        "6.14.2..6.14.4",
        "--extra-kconfig",
        "/tmp/range-extra.kconfig",
    ])
    .unwrap_or_else(|e| panic!("{e}"));
    let KtstrCommand::Kernel {
        command:
            KernelCommand::Build {
                kernel,
                extra_kconfig,
                ..
            },
    } = k.command
    else {
        panic!("expected KernelCommand::Build");
    };
    assert_eq!(kernel.as_deref(), Some("6.14.2..6.14.4"));
    assert_eq!(
        extra_kconfig,
        Some(std::path::PathBuf::from("/tmp/range-extra.kconfig")),
    );
}

/// --force + --clean + --extra-kconfig orthogonality. None of
/// these flags conflict with each other; pin that all three
/// can co-exist on a single invocation. A regression that
/// introduced a clap `conflicts_with` between any pair would
/// surface here.
#[test]
fn parse_kernel_build_force_clean_and_extra_kconfig_compose() {
    let Cargo {
        command: CargoSub::Ktstr(k),
    } = Cargo::try_parse_from([
        "cargo",
        "ktstr",
        "kernel",
        "build",
        "--kernel",
        "../linux",
        "--force",
        "--clean",
        "--extra-kconfig",
        "/tmp/extra.kconfig",
    ])
    .unwrap_or_else(|e| panic!("{e}"));
    let KtstrCommand::Kernel {
        command:
            KernelCommand::Build {
                force,
                clean,
                extra_kconfig,
                ..
            },
    } = k.command
    else {
        panic!("expected KernelCommand::Build");
    };
    assert!(
        force,
        "--force must round-trip alongside --clean and --extra-kconfig"
    );
    assert!(
        clean,
        "--clean must round-trip alongside --force and --extra-kconfig"
    );
    assert_eq!(
        extra_kconfig,
        Some(std::path::PathBuf::from("/tmp/extra.kconfig")),
        "--extra-kconfig must round-trip when combined with --force + --clean",
    );
}

/// Non-build subcommands that accept `--extra-kconfig` would
/// silently produce wrong cache lookups. The flag is `kernel
/// build`-only at the configuration layer; this test pins that a
/// passthrough subcommand (verifier) forwards it downstream, while
/// the sibling `shell` test pins the parse-level reject for a
/// non-passthrough subcommand.
///
/// Subcommands and their behavior:
/// - `shell`: REJECTS at parse time — not a passthrough subcommand,
///   so `argsplit::rewrite` leaves its argv unchanged and clap has
///   no `args` field to absorb the unknown flag. Pin via
///   `try_parse_from` returning `Err`.
/// - `verifier` / `test` / `coverage` / `llvm-cov`: PASSTHROUGH via
///   the `args: Vec<String>` (`last = true`) field. `argsplit::rewrite`
///   routes an unknown `--extra-kconfig ...` into the passthrough, so
///   it reaches `cargo nextest run` (or `cargo llvm-cov`), which then
///   rejects it as an unknown cargo flag — at the cargo subprocess
///   layer, NOT at parse time. We pin these as args-capture, NOT
///   parse errors, because that is the actual shape.
#[test]
fn parse_extra_kconfig_passes_through_verifier_subcommand_to_args_vec() {
    let KtstrCommand::Verifier { args, .. } = parse_via_split(&[
        "cargo",
        "ktstr",
        "verifier",
        "--kernel",
        "../linux",
        "--extra-kconfig",
        "/tmp/x.kconfig",
    ]) else {
        panic!("expected KtstrCommand::Verifier");
    };
    assert_eq!(
        args,
        vec!["--extra-kconfig", "/tmp/x.kconfig"],
        "verifier has no native --extra-kconfig, so the argv split routes \
         it into `args` (the inner cargo nextest run rejects it downstream) — \
         verifier is a passthrough subcommand like test/coverage, NOT a \
         clean-surface reject like shell",
    );
}

#[test]
fn parse_extra_kconfig_rejected_on_shell_subcommand() {
    let m = Cargo::try_parse_from([
        "cargo",
        "ktstr",
        "shell",
        "--extra-kconfig",
        "/tmp/x.kconfig",
    ]);
    assert!(
        m.is_err(),
        "--extra-kconfig must be rejected on `cargo ktstr shell` \
         (shell is not a passthrough subcommand — no `last = true` args \
         field and argsplit leaves its argv unchanged — so unknown flags \
         fail at parse time)",
    );
}

/// Documents the passthrough behavior on `test` / `coverage` /
/// `llvm-cov`: `argsplit::rewrite` routes an unknown `--extra-kconfig`
/// into the `args: Vec<String>` (`last = true`) field, forwarded to
/// `cargo nextest run` / `cargo llvm-cov`. The rejection happens later,
/// at the cargo subprocess layer, not at parse time. Pin the shape so a
/// future change to the passthrough routing surfaces here.
#[test]
fn parse_extra_kconfig_passes_through_test_subcommand_to_args_vec() {
    let KtstrCommand::Test { args, .. } = parse_via_split(&[
        "cargo",
        "ktstr",
        "test",
        "--extra-kconfig",
        "/tmp/x.kconfig",
    ]) else {
        panic!("expected KtstrCommand::Test");
    };
    assert_eq!(
        args,
        vec!["--extra-kconfig", "/tmp/x.kconfig"],
        "--extra-kconfig must passthrough into `args` Vec on test \
         subcommand (the argv split routes it there). The cargo nextest \
         subprocess will reject it as an unknown flag downstream."
    );
}

/// `--extra-kconfig` works alongside a `--kernel <path>` source
/// tree. Pins that the flag is orthogonal to where the kernel
/// SOURCE comes from.
#[test]
fn parse_kernel_build_extra_kconfig_with_path() {
    let Cargo {
        command: CargoSub::Ktstr(k),
    } = Cargo::try_parse_from([
        "cargo",
        "ktstr",
        "kernel",
        "build",
        "--kernel",
        "../linux",
        "--extra-kconfig",
        "/tmp/extra.kconfig",
    ])
    .unwrap_or_else(|e| panic!("{e}"));
    let KtstrCommand::Kernel {
        command:
            KernelCommand::Build {
                kernel,
                extra_kconfig,
                ..
            },
    } = k.command
    else {
        panic!("expected KernelCommand::Build");
    };
    assert_eq!(kernel.as_deref(), Some("../linux"));
    assert_eq!(
        extra_kconfig,
        Some(std::path::PathBuf::from("/tmp/extra.kconfig")),
    );
}

// -- try_get_matches_from: kernel clean --

#[test]
fn parse_kernel_clean() {
    let m = Cargo::try_parse_from(["cargo", "ktstr", "kernel", "clean"]);
    assert!(m.is_ok(), "{}", m.err().unwrap());
}

#[test]
fn parse_kernel_clean_keep() {
    let Cargo {
        command: CargoSub::Ktstr(k),
    } = Cargo::try_parse_from(["cargo", "ktstr", "kernel", "clean", "--keep", "3"])
        .unwrap_or_else(|e| panic!("{e}"));
    let KtstrCommand::Kernel {
        command: KernelCommand::Clean { keep, .. },
    } = k.command
    else {
        panic!("expected Kernel Clean");
    };
    assert_eq!(keep, Some(3));
}

// -- try_get_matches_from: verifier --
//
// The verifier subcommand's native flags are --kernel (repeatable),
// --raw, --profile, --nextest-profile, and --scheduler. The declared
// scheduler set is discovered from `declare_scheduler!` registrations
// in linked test binaries (the matrix is driven by the test binary's
// `KTSTR_SCHEDULERS` distributed slice); --scheduler narrows that
// sweep to a single declared scheduler by name.

#[test]
fn parse_verifier_bare() {
    let Cargo {
        command: CargoSub::Ktstr(k),
    } = Cargo::try_parse_from(["cargo", "ktstr", "verifier"]).unwrap_or_else(|e| panic!("{e}"));
    let KtstrCommand::Verifier {
        kernel,
        raw,
        include_eol,
        ..
    } = k.command
    else {
        panic!("expected Verifier");
    };
    assert!(
        kernel.is_empty(),
        "bare verifier must default --kernel to empty Vec"
    );
    assert!(!raw, "bare verifier must default --raw to false");
    assert!(
        !include_eol,
        "bare verifier must default --include-eol to false"
    );
}

/// `verifier --kernel R --include-eol` round-trips `include_eol=true`
/// so the range expands EOL series in the verifier sweep too.
#[test]
fn parse_verifier_include_eol_flag() {
    let Cargo {
        command: CargoSub::Ktstr(k),
    } = Cargo::try_parse_from([
        "cargo",
        "ktstr",
        "verifier",
        "--kernel",
        "6.11..6.14",
        "--include-eol",
    ])
    .unwrap_or_else(|e| panic!("{e}"));
    let KtstrCommand::Verifier {
        kernel,
        include_eol,
        ..
    } = k.command
    else {
        panic!("expected Verifier");
    };
    assert_eq!(kernel, vec!["6.11..6.14".to_string()]);
    assert!(
        include_eol,
        "`--include-eol` must round-trip as true on verifier"
    );
}

#[test]
fn parse_verifier_with_kernel_single() {
    let Cargo {
        command: CargoSub::Ktstr(k),
    } = Cargo::try_parse_from(["cargo", "ktstr", "verifier", "--kernel", "6.14.2"])
        .unwrap_or_else(|e| panic!("{e}"));
    let KtstrCommand::Verifier { kernel, raw, .. } = k.command else {
        panic!("expected Verifier");
    };
    assert_eq!(kernel, vec!["6.14.2"]);
    assert!(!raw);
}

#[test]
fn parse_verifier_with_kernel_repeatable() {
    let Cargo {
        command: CargoSub::Ktstr(k),
    } = Cargo::try_parse_from([
        "cargo", "ktstr", "verifier", "--kernel", "6.14.2", "--kernel", "6.15.0",
    ])
    .unwrap_or_else(|e| panic!("{e}"));
    let KtstrCommand::Verifier { kernel, raw, .. } = k.command else {
        panic!("expected Verifier");
    };
    assert_eq!(kernel, vec!["6.14.2", "6.15.0"]);
    assert!(!raw);
}

#[test]
fn parse_verifier_with_raw() {
    let Cargo {
        command: CargoSub::Ktstr(k),
    } = Cargo::try_parse_from(["cargo", "ktstr", "verifier", "--raw"])
        .unwrap_or_else(|e| panic!("{e}"));
    let KtstrCommand::Verifier { kernel, raw, .. } = k.command else {
        panic!("expected Verifier");
    };
    assert!(kernel.is_empty());
    assert!(raw, "--raw must lift the flag to true");
}

#[test]
fn parse_verifier_scheduler_defaults_none() {
    let Cargo {
        command: CargoSub::Ktstr(k),
    } = Cargo::try_parse_from(["cargo", "ktstr", "verifier"]).unwrap_or_else(|e| panic!("{e}"));
    let KtstrCommand::Verifier { scheduler, .. } = k.command else {
        panic!("expected Verifier");
    };
    assert!(
        scheduler.is_none(),
        "bare verifier must default --scheduler to None (full declared-scheduler sweep)",
    );
}

#[test]
fn parse_verifier_with_scheduler() {
    let Cargo {
        command: CargoSub::Ktstr(k),
    } = Cargo::try_parse_from(["cargo", "ktstr", "verifier", "--scheduler", "scx-ktstr"])
        .unwrap_or_else(|e| panic!("{e}"));
    let KtstrCommand::Verifier { scheduler, .. } = k.command else {
        panic!("expected Verifier");
    };
    assert_eq!(
        scheduler.as_deref(),
        Some("scx-ktstr"),
        "--scheduler lifts the single-scheduler sweep filter",
    );
}

/// The trailing `args` forward cargo/nextest flags to the inner
/// `cargo nextest run` with NO `--` separator. The argv split (the bin's
/// `argsplit` module) routes `--kernel` to the native flag and
/// `--features integration` to the passthrough `args` (`last = true`) by
/// name, regardless of their order.
#[test]
fn parse_verifier_forwards_flags_without_separator() {
    let KtstrCommand::Verifier {
        kernel, raw, args, ..
    } = parse_via_split(&[
        "cargo",
        "ktstr",
        "verifier",
        "--kernel",
        "../linux",
        "--features",
        "integration",
    ])
    else {
        panic!("expected Verifier");
    };
    assert_eq!(kernel, vec!["../linux"], "--kernel parsed as a native flag");
    assert!(!raw);
    assert_eq!(
        args,
        vec!["--features".to_string(), "integration".to_string()],
        "cargo/nextest flags forward into `args` with no `--` separator",
    );
}

/// Position-independence (the exact ordering-bug repro): a native flag
/// placed AFTER a passthrough flag still parses as native — the argv
/// split (`argsplit`) routes each token by name — and only the genuine
/// passthrough lands in `args`. Previously `--kernel` here was swallowed
/// into `args` and rejected by nextest.
#[test]
fn parse_verifier_flag_after_passthrough_is_native() {
    let KtstrCommand::Verifier { kernel, args, .. } = parse_via_split(&[
        "cargo",
        "ktstr",
        "verifier",
        "--features",
        "integration",
        "--kernel",
        "../linux",
    ]) else {
        panic!("expected Verifier");
    };
    assert_eq!(
        kernel,
        vec!["../linux"],
        "--kernel placed AFTER a passthrough flag parses as a native flag, not swallowed",
    );
    assert_eq!(
        args,
        vec!["--features".to_string(), "integration".to_string()],
        "only the genuine passthrough lands in args",
    );
}

/// `--profile <NAME>` (scheduler BUILD profile) and `--nextest-profile
/// <NAME>` (nextest test profile) round-trip as native Verifier flags.
/// Placed BEFORE the trailing capture, they parse as `Some(_)` while a
/// following passthrough flag (`--features`) still lands in `args`.
#[test]
fn parse_verifier_with_profiles() {
    let KtstrCommand::Verifier {
        kernel,
        profile,
        nextest_profile,
        args,
        ..
    } = parse_via_split(&[
        "cargo",
        "ktstr",
        "verifier",
        "--kernel",
        "../linux",
        "--profile",
        "dev",
        "--nextest-profile",
        "ci",
        "--features",
        "integration",
    ])
    else {
        panic!("expected Verifier");
    };
    assert_eq!(kernel, vec!["../linux"]);
    assert_eq!(profile.as_deref(), Some("dev"), "--profile round-trips");
    assert_eq!(
        nextest_profile.as_deref(),
        Some("ci"),
        "--nextest-profile round-trips"
    );
    assert_eq!(
        args,
        vec!["--features".to_string(), "integration".to_string()],
        "a passthrough flag AFTER the native profiles still lands in args",
    );
}

/// `--all-profiles` was removed alongside the flag-profile sweep;
/// verifier has no native flag for it. As a passthrough subcommand
/// verifier FORWARDS it into `args` (nextest rejects it downstream)
/// rather than clap-rejecting. Pinning the args-capture still trips
/// a future native re-add (which would not land in `args`).
#[test]
fn parse_verifier_all_profiles_forwarded_not_native() {
    let KtstrCommand::Verifier { args, .. } =
        parse_via_split(&["cargo", "ktstr", "verifier", "--all-profiles"])
    else {
        panic!("expected Verifier");
    };
    assert_eq!(
        args,
        vec!["--all-profiles"],
        "--all-profiles is not a native verifier flag; the argv split \
         routes it into `args`. A future native re-add would empty `args`.",
    );
}

/// `--profiles` was removed alongside the flag-profile sweep;
/// verifier has no native flag for it. Passthrough verifier
/// FORWARDS it (with its value) into `args`; nextest rejects it
/// downstream. Pinning the args-capture still trips a future native
/// re-add.
#[test]
fn parse_verifier_profiles_filter_forwarded_not_native() {
    let KtstrCommand::Verifier { args, .. } = parse_via_split(&[
        "cargo",
        "ktstr",
        "verifier",
        "--profiles",
        "default,llc,llc+steal",
    ]) else {
        panic!("expected Verifier");
    };
    assert_eq!(
        args,
        vec!["--profiles", "default,llc,llc+steal"],
        "--profiles is not a native verifier flag; the argv split routes it \
         and its value into `args`. A future native re-add would consume the \
         value and empty `args`.",
    );
}

// -- argv split: position-independent ktstr flags --

/// `test`: a ktstr flag (`--include-eol`) placed AFTER a passthrough
/// flag parses as native, and only the passthrough lands in `args`.
#[test]
fn parse_test_flag_after_passthrough_is_native() {
    let KtstrCommand::Test {
        include_eol,
        kernel,
        args,
        ..
    } = parse_via_split(&[
        "cargo",
        "ktstr",
        "test",
        "--kernel",
        "6.11..6.14",
        "--features",
        "integration",
        "--include-eol",
    ])
    else {
        panic!("expected Test");
    };
    assert!(
        include_eol,
        "--include-eol after a passthrough flag is consumed by ktstr"
    );
    assert_eq!(kernel, vec!["6.11..6.14"]);
    assert_eq!(
        args,
        vec!["--features".to_string(), "integration".to_string()]
    );
}

/// `test`: the same flags in the other order parse identically —
/// position does not matter (regression pin for the reported bug).
#[test]
fn parse_test_flag_before_passthrough_is_native() {
    let KtstrCommand::Test {
        include_eol,
        kernel,
        args,
        ..
    } = parse_via_split(&[
        "cargo",
        "ktstr",
        "test",
        "--kernel",
        "6.11..6.14",
        "--include-eol",
        "--features",
        "integration",
    ])
    else {
        panic!("expected Test");
    };
    assert!(include_eol);
    assert_eq!(kernel, vec!["6.11..6.14"]);
    assert_eq!(
        args,
        vec!["--features".to_string(), "integration".to_string()]
    );
}

/// A ktstr flag interleaved BETWEEN two passthrough tokens is still
/// routed to ktstr; the passthrough tokens land in `args` in order.
#[test]
fn parse_test_interleaved_ktstr_flag() {
    let KtstrCommand::Test { kernel, args, .. } = parse_via_split(&[
        "cargo",
        "ktstr",
        "test",
        "--features",
        "integration",
        "--kernel",
        "6.14",
        "--no-capture",
    ]) else {
        panic!("expected Test");
    };
    assert_eq!(kernel, vec!["6.14"]);
    assert_eq!(
        args,
        vec![
            "--features".to_string(),
            "integration".to_string(),
            "--no-capture".to_string()
        ]
    );
}

/// The `--flag=value` form of a value-taking ktstr flag is routed whole,
/// even after a passthrough flag.
#[test]
fn parse_test_kernel_eq_value_after_passthrough() {
    let KtstrCommand::Test { kernel, args, .. } = parse_via_split(&[
        "cargo",
        "ktstr",
        "test",
        "--features",
        "integration",
        "--kernel=6.14",
    ]) else {
        panic!("expected Test");
    };
    assert_eq!(kernel, vec!["6.14"]);
    assert_eq!(
        args,
        vec!["--features".to_string(), "integration".to_string()]
    );
}

/// Repeated `--kernel` (Append) after a passthrough flag collects both
/// values.
#[test]
fn parse_test_repeated_kernel_after_passthrough() {
    let KtstrCommand::Test { kernel, args, .. } = parse_via_split(&[
        "cargo",
        "ktstr",
        "test",
        "--features",
        "x",
        "--kernel",
        "6.14",
        "--kernel",
        "6.15",
    ]) else {
        panic!("expected Test");
    };
    assert_eq!(kernel, vec!["6.14", "6.15"]);
    assert_eq!(args, vec!["--features".to_string(), "x".to_string()]);
}

/// `coverage`: `--include-eol` after a passthrough flag is native.
#[test]
fn parse_coverage_flag_after_passthrough_is_native() {
    let KtstrCommand::Coverage {
        include_eol, args, ..
    } = parse_via_split(&[
        "cargo",
        "ktstr",
        "coverage",
        "--features",
        "integration",
        "--include-eol",
    ])
    else {
        panic!("expected Coverage");
    };
    assert!(include_eol);
    assert_eq!(
        args,
        vec!["--features".to_string(), "integration".to_string()]
    );
}

/// `llvm-cov`: `--include-eol` after a passthrough flag is native.
#[test]
fn parse_llvm_cov_flag_after_passthrough_is_native() {
    let KtstrCommand::LlvmCov {
        include_eol, args, ..
    } = parse_via_split(&[
        "cargo",
        "ktstr",
        "llvm-cov",
        "--features",
        "integration",
        "--include-eol",
    ])
    else {
        panic!("expected LlvmCov");
    };
    assert!(include_eol);
    assert_eq!(
        args,
        vec!["--features".to_string(), "integration".to_string()]
    );
}

/// An explicit `--` forces EVERYTHING after it to passthrough, even a
/// token spelled like a ktstr flag — the escape hatch for a passthrough
/// token that collides with a native flag name.
#[test]
fn parse_test_double_dash_forces_passthrough() {
    let KtstrCommand::Test {
        include_eol,
        kernel,
        args,
        ..
    } = parse_via_split(&[
        "cargo",
        "ktstr",
        "test",
        "--kernel",
        "6.14",
        "--",
        "--include-eol",
    ])
    else {
        panic!("expected Test");
    };
    assert!(
        !include_eol,
        "--include-eol after -- is passthrough, not the native flag"
    );
    assert_eq!(kernel, vec!["6.14"]);
    assert_eq!(args, vec!["--include-eol".to_string()]);
}

/// Name-collision case: `--profile` is a ktstr flag (scheduler build
/// profile). Before `--` it is native; after `--` it forwards to the
/// inner tool — the documented way to pass nextest's own `--profile`.
#[test]
fn parse_test_profile_native_vs_passthrough_by_double_dash() {
    let KtstrCommand::Test { profile, args, .. } = parse_via_split(&[
        "cargo",
        "ktstr",
        "test",
        "--profile",
        "dev",
        "--features",
        "x",
    ]) else {
        panic!("expected Test");
    };
    assert_eq!(
        profile.as_deref(),
        Some("dev"),
        "--profile before -- is the native ktstr flag"
    );
    assert_eq!(args, vec!["--features".to_string(), "x".to_string()]);

    let KtstrCommand::Test { profile, args, .. } = parse_via_split(&[
        "cargo",
        "ktstr",
        "test",
        "--kernel",
        "6.14",
        "--",
        "--profile",
        "nextest-ci",
    ]) else {
        panic!("expected Test");
    };
    assert_eq!(
        profile, None,
        "--profile after -- forwards to the inner tool"
    );
    assert_eq!(
        args,
        vec!["--profile".to_string(), "nextest-ci".to_string()]
    );
}

/// `replay`: the short `-E` (its native filter) is routed to ktstr even
/// after a passthrough flag, in both `-E value` and glued `-Evalue`
/// forms; the passthrough lands in `args`.
#[test]
fn parse_replay_short_e_after_passthrough() {
    let KtstrCommand::Replay { filter, args, .. } = parse_via_split(&[
        "cargo",
        "ktstr",
        "replay",
        "--exec",
        "--features",
        "x",
        "-E",
        "scheduler_",
    ]) else {
        panic!("expected Replay");
    };
    assert_eq!(
        filter.as_deref(),
        Some("scheduler_"),
        "-E value after a passthrough flag is native"
    );
    assert_eq!(args, vec!["--features".to_string(), "x".to_string()]);

    let KtstrCommand::Replay { filter, args, .. } = parse_via_split(&[
        "cargo",
        "ktstr",
        "replay",
        "--exec",
        "--features",
        "x",
        "-Escheduler_",
    ]) else {
        panic!("expected Replay");
    };
    assert_eq!(
        filter.as_deref(),
        Some("scheduler_"),
        "glued -Evalue after a passthrough flag is native"
    );
    assert_eq!(args, vec!["--features".to_string(), "x".to_string()]);
}

/// A passthrough-only invocation (no ktstr flag among the tail) forwards
/// every token and steals nothing.
#[test]
fn parse_test_passthrough_only_steals_nothing() {
    let KtstrCommand::Test { kernel, args, .. } = parse_via_split(&[
        "cargo",
        "ktstr",
        "test",
        "--kernel",
        "6.14",
        "--no-capture",
        "-j4",
    ]) else {
        panic!("expected Test");
    };
    assert_eq!(kernel, vec!["6.14"]);
    assert_eq!(args, vec!["--no-capture".to_string(), "-j4".to_string()]);
}

/// `--help` stays a native (ktstr) flag — clap shows help rather than
/// forwarding `--help` to the inner tool. `rewrite` routes it to the
/// ktstr side, so `try_parse_from` surfaces clap's `DisplayHelp`.
#[test]
fn parse_test_help_stays_native() {
    let raw: Vec<std::ffi::OsString> = ["cargo", "ktstr", "test", "--help"]
        .iter()
        .map(std::ffi::OsString::from)
        .collect();
    let rewritten = crate::argsplit::rewrite(&Cargo::command(), &raw);
    // `.err()` drops the Ok(Cargo) value (Cargo is not Debug, so
    // `expect_err` would not compile) and keeps the clap error.
    let err = Cargo::try_parse_from(&rewritten)
        .err()
        .expect("--help must not parse as a command");
    assert_eq!(
        err.kind(),
        clap::error::ErrorKind::DisplayHelp,
        "--help stays native (clap help), not forwarded to args",
    );
}

/// A non-passthrough subcommand (e.g. `shell`) is returned unchanged by
/// `rewrite` — no `--` is injected, so its own args parse normally.
#[test]
fn rewrite_leaves_non_passthrough_subcommand_unchanged() {
    let raw: Vec<std::ffi::OsString> = ["cargo", "ktstr", "shell", "--topology", "1,1,2,1"]
        .iter()
        .map(std::ffi::OsString::from)
        .collect();
    let rewritten = crate::argsplit::rewrite(&Cargo::command(), &raw);
    assert_eq!(
        rewritten, raw,
        "shell is not a passthrough subcommand; argv is untouched"
    );
}

// -- try_get_matches_from: replay --

/// Bare `cargo ktstr replay` parses to the Replay variant with every
/// field at its default: no `--dir` / `--filter`, `--exec` off, no
/// `--profile` / `--nextest-profile`, empty passthrough `args`.
#[test]
fn parse_replay_defaults() {
    let Cargo {
        command: CargoSub::Ktstr(k),
    } = Cargo::try_parse_from(["cargo", "ktstr", "replay"]).unwrap_or_else(|e| panic!("{e}"));
    let KtstrCommand::Replay {
        dir,
        filter,
        exec,
        profile,
        nextest_profile,
        args,
    } = k.command
    else {
        panic!("expected Replay");
    };
    assert!(dir.is_none(), "bare replay defaults --dir to None");
    assert!(filter.is_none(), "bare replay defaults --filter to None");
    assert!(!exec, "bare replay defaults --exec off (dry-run)");
    assert!(profile.is_none(), "bare replay defaults --profile to None");
    assert!(
        nextest_profile.is_none(),
        "bare replay defaults --nextest-profile to None"
    );
    assert!(args.is_empty(), "bare replay parses no passthrough");
}

/// Full round-trip: `--dir`, `-E` filter, `--exec`, `--profile`,
/// `--nextest-profile`, and a trailing passthrough flag all populate on
/// a single invocation. The native flags precede the trailing capture,
/// so `--features` (unrecognized) lands in `args` while the profiles
/// parse as `Some(_)`.
#[test]
fn parse_replay_all_flags_and_passthrough() {
    let KtstrCommand::Replay {
        dir,
        filter,
        exec,
        profile,
        nextest_profile,
        args,
    } = parse_via_split(&[
        "cargo",
        "ktstr",
        "replay",
        "--dir",
        "/tmp/archived-runs",
        "-E",
        "scheduler_",
        "--exec",
        "--profile",
        "dev",
        "--nextest-profile",
        "ci",
        "--features",
        "integration",
    ])
    else {
        panic!("expected Replay");
    };
    assert_eq!(
        dir.as_deref(),
        Some(std::path::Path::new("/tmp/archived-runs")),
        "--dir round-trips to Some(PathBuf)"
    );
    assert_eq!(
        filter.as_deref(),
        Some("scheduler_"),
        "-E filter round-trips"
    );
    assert!(exec, "--exec lifts the flag to true");
    assert_eq!(profile.as_deref(), Some("dev"), "--profile round-trips");
    assert_eq!(
        nextest_profile.as_deref(),
        Some("ci"),
        "--nextest-profile round-trips"
    );
    assert_eq!(
        args,
        vec!["--features".to_string(), "integration".to_string()],
        "a passthrough flag AFTER the native flags lands in args",
    );
}

// -- try_get_matches_from: completions --

#[test]
fn parse_completions_bash() {
    let m = Cargo::try_parse_from(["cargo", "ktstr", "completions", "bash"]);
    assert!(m.is_ok(), "{}", m.err().unwrap());
}

#[test]
fn parse_completions_invalid_shell() {
    let m = Cargo::try_parse_from(["cargo", "ktstr", "completions", "noshell"]);
    assert!(m.is_err());
}

// -- error cases --

#[test]
fn parse_missing_subcommand() {
    let m = Cargo::try_parse_from(["cargo", "ktstr"]);
    assert!(m.is_err());
}

#[test]
fn parse_unknown_subcommand() {
    let m = Cargo::try_parse_from(["cargo", "ktstr", "nonexistent"]);
    assert!(m.is_err());
}

// -- completions --

#[test]
fn completions_bash_non_empty() {
    let mut buf = Vec::new();
    let mut cmd = Cargo::command();
    clap_complete::generate(clap_complete::Shell::Bash, &mut cmd, "cargo", &mut buf);
    assert!(!buf.is_empty());
}

#[test]
fn completions_zsh_contains_subcommands() {
    let mut buf = Vec::new();
    let mut cmd = Cargo::command();
    clap_complete::generate(clap_complete::Shell::Zsh, &mut cmd, "cargo", &mut buf);
    let output = String::from_utf8(buf).expect("completions should be valid UTF-8");
    // clap_complete's zsh generator emits each subcommand as a
    // `'NAME:HELP'` describe-list entry (see `add_subcommands`
    // in clap_complete-4.6.5/src/aot/shells/zsh.rs:163). The
    // `'<name>:` prefix pin identifies an actual subcommand
    // completion, not an incidental substring match inside
    // rendered doc text.
    assert!(
        output.contains("'test:"),
        "zsh completions missing 'test:' describe-list entry"
    );
    assert!(
        output.contains("'coverage:"),
        "zsh completions missing 'coverage:' describe-list entry"
    );
    assert!(
        output.contains("'shell:"),
        "zsh completions missing 'shell:' describe-list entry"
    );
    assert!(
        output.contains("'kernel:"),
        "zsh completions missing 'kernel:' describe-list entry"
    );
    // `visible_alias = "nextest"` on the Test variant makes the
    // alias user-facing — clap_complete's zsh generator iterates
    // `get_visible_aliases` (zsh.rs:177) and emits a dedicated
    // describe entry per alias. A regression that dropped the
    // attribute (or silently switched to `alias` which is
    // NON-visible) would drop the entry and fail this assertion.
    assert!(
        output.contains("'nextest:"),
        "zsh completions missing 'nextest:' describe-list \
         entry (visible alias of `test`)"
    );
    // `LlvmCov` variant renders as the kebab-case `llvm-cov`
    // subcommand (clap derive default rename — see
    // clap_derive-4.6.1/src/item.rs:27 `DEFAULT_CASING =
    // CasingStyle::Kebab`). Pinned with the same `'name:`
    // prefix so an accidental doc-text match doesn't mask a
    // missing registration.
    assert!(
        output.contains("'llvm-cov:"),
        "zsh completions missing 'llvm-cov:' describe-list entry"
    );
}

// -- format_entry_row helpers --

fn test_metadata() -> KernelMetadata {
    KernelMetadata::new(
        ktstr::cache::KernelSource::Tarball,
        "x86_64",
        "bzImage",
        "2026-04-12T10:00:00Z",
    )
    .with_version("6.14.2")
}

/// Store a fake kernel image and return the CacheEntry.
fn store_test_entry(cache: &CacheDir, key: &str, meta: &KernelMetadata) -> CacheEntry {
    let src = tempfile::TempDir::new().unwrap();
    let image = src.path().join(&meta.image_name);
    std::fs::write(&image, b"fake kernel").unwrap();
    cache
        .store(key, &CacheArtifacts::new(&image), meta)
        .unwrap()
}

// -- format_entry_row --
//
// The (Matches / Stale / Untracked) × (not-EOL / EOL) outcome
// matrix plus the `version == None` → "-" dash-render branch are
// pinned by `format_entry_row_renders_eol_kconfig_matrix` in
// `src/cli/kernel_list.rs` — see that test for the full case
// list. The test below covers a distinct corner the matrix does
// not: `KernelSource::Local` rendering through format_entry_row,
// since the matrix uses `Tarball` exclusively for determinism.

#[test]
fn format_entry_row_no_version() {
    let tmp = tempfile::TempDir::new().unwrap();
    let cache = CacheDir::with_root(tmp.path().join("cache"));
    let meta = KernelMetadata::new(
        ktstr::cache::KernelSource::Local {
            source_tree_path: None,
            git_hash: None,
        },
        "x86_64",
        "bzImage",
        "2026-04-12T10:00:00Z",
    );
    let entry = store_test_entry(&cache, "local-key", &meta);
    let row = cli::format_entry_row(&entry, "hash", &[]);
    // Anchor the dash to the version COLUMN. The row format is
    // `"  {key:<48} {version:<12} {source:<8} {arch:<7} {built_at}{tags}"`
    // (see `format_entry_row` in src/cli/kernel_list.rs). A bare
    // `row.contains("-")` would also match the `-` in the timestamp
    // `2026-04-12T10:00:00Z` even if the version dash were missing.
    // Splitting on whitespace and inspecting the second token isolates
    // the version slot — token 0 is the key, token 1 is the version.
    let tokens: Vec<&str> = row.split_whitespace().collect();
    assert!(
        tokens.len() >= 2,
        "row must have at least key + version columns: {row:?}",
    );
    assert_eq!(
        tokens[1], "-",
        "missing version must render as `-` in the version column: {row:?}",
    );
}

// Corrupt-entry formatting moved inline into the caller iteration
// in cli::kernel_list, so no test on format_entry_row covers it;
// the helper itself now takes only the valid CacheEntry shape.

// -- kconfig_status (via CacheEntry method) --

/// Companion to the stale-kconfig case in
/// `format_entry_row_renders_eol_kconfig_matrix` (in
/// `src/cli/kernel_list.rs`): that test pins the `(stale kconfig)`
/// tag emitted by `cli::format_entry_row` for a hash-mismatch entry;
/// this test pins the enum variant
/// (`KconfigStatus::Stale { cached, current }`) returned by
/// `CacheEntry::kconfig_status` that drives the tag.
#[test]
fn kconfig_status_reports_stale_on_hash_mismatch() {
    let tmp = tempfile::TempDir::new().unwrap();
    let cache = CacheDir::with_root(tmp.path().join("cache"));
    let meta = test_metadata().with_ktstr_kconfig_hash("old");
    let entry = store_test_entry(&cache, "stale", &meta);
    assert_eq!(
        entry.kconfig_status("new"),
        ktstr::cache::KconfigStatus::Stale {
            cached: "old".to_string(),
            current: "new".to_string(),
        }
    );
}

/// Companion to the matching-kconfig case in
/// `format_entry_row_renders_eol_kconfig_matrix` (in
/// `src/cli/kernel_list.rs`): that test pins the no-tag contract
/// emitted by `cli::format_entry_row` when the hashes agree; this
/// test pins the `KconfigStatus::Matches` variant returned by
/// `CacheEntry::kconfig_status` that drives the no-tag branch.
#[test]
fn kconfig_status_reports_matches_on_hash_equality() {
    let tmp = tempfile::TempDir::new().unwrap();
    let cache = CacheDir::with_root(tmp.path().join("cache"));
    let meta = test_metadata().with_ktstr_kconfig_hash("same");
    let entry = store_test_entry(&cache, "fresh", &meta);
    assert_eq!(
        entry.kconfig_status("same"),
        ktstr::cache::KconfigStatus::Matches
    );
}

/// Companion to the untracked-kconfig case in
/// `format_entry_row_renders_eol_kconfig_matrix` (in
/// `src/cli/kernel_list.rs`): that test pins the
/// `(untracked kconfig)` tag emitted by `cli::format_entry_row`
/// when an entry has no recorded hash; this test pins the
/// `KconfigStatus::Untracked` variant returned by
/// `CacheEntry::kconfig_status` that drives the tag.
#[test]
fn kconfig_status_reports_untracked_when_entry_has_no_hash() {
    let tmp = tempfile::TempDir::new().unwrap();
    let cache = CacheDir::with_root(tmp.path().join("cache"));
    let meta = test_metadata();
    let entry = store_test_entry(&cache, "no-hash", &meta);
    assert_eq!(
        entry.kconfig_status("anything"),
        ktstr::cache::KconfigStatus::Untracked
    );
}

// Corrupt entries no longer surface as CacheEntry — they are
// ListedEntry::Corrupt with no metadata-bearing struct — so
// kconfig_status isn't reachable from that state.

/// Differential pin on the three `KconfigStatus` strings that flow
/// into the `kconfig_status` field of `cargo ktstr kernel list
/// --json`. `cli::kernel_list` emits the JSON field via
/// `entry.kconfig_status(&kconfig_hash).to_string()`, so CI scripts
/// that key off the stringified variant break if any of these
/// three words changes. This test exercises the full
/// `CacheEntry::kconfig_status(..).to_string()` chain (not just
/// `KconfigStatus::<variant>.to_string()` in isolation) to pin the
/// end-to-end JSON contract in a single test covering all three
/// variants.
#[test]
fn kconfig_status_json_string_pins_all_three_variants() {
    use ktstr::cache::KconfigStatus;
    let tmp = tempfile::TempDir::new().unwrap();
    let cache = CacheDir::with_root(tmp.path().join("cache"));

    let matches_meta = test_metadata().with_ktstr_kconfig_hash("h");
    let matches_entry = store_test_entry(&cache, "matches-key", &matches_meta);
    let matches_status = matches_entry.kconfig_status("h");
    assert!(
        matches!(matches_status, KconfigStatus::Matches),
        "hash equality must yield KconfigStatus::Matches"
    );
    assert_eq!(matches_status.to_string(), "matches");

    let stale_meta = test_metadata().with_ktstr_kconfig_hash("old");
    let stale_entry = store_test_entry(&cache, "stale-key", &stale_meta);
    let stale_status = stale_entry.kconfig_status("new");
    assert!(
        matches!(stale_status, KconfigStatus::Stale { .. }),
        "hash mismatch must yield KconfigStatus::Stale"
    );
    assert_eq!(stale_status.to_string(), "stale");

    let untracked_meta = test_metadata();
    let untracked_entry = store_test_entry(&cache, "untracked-key", &untracked_meta);
    let untracked_status = untracked_entry.kconfig_status("anything");
    assert!(
        matches!(untracked_status, KconfigStatus::Untracked),
        "entry without hash must yield KconfigStatus::Untracked"
    );
    assert_eq!(untracked_status.to_string(), "untracked");
}

// -- embedded_kconfig_hash --

#[test]
fn embedded_kconfig_hash_deterministic() {
    let h1 = cli::embedded_kconfig_hash();
    let h2 = cli::embedded_kconfig_hash();
    assert_eq!(h1, h2);
}

#[test]
fn embedded_kconfig_hash_is_hex() {
    let h = cli::embedded_kconfig_hash();
    assert_eq!(h.len(), 8, "CRC32 hex should be 8 chars");
    assert!(
        h.chars().all(|c| c.is_ascii_hexdigit()),
        "should be hex digits: {h}"
    );
}

#[test]
fn embedded_kconfig_hash_matches_manual_crc32() {
    let expected = format!("{:08x}", crc32fast::hash(cli::EMBEDDED_KCONFIG.as_bytes()));
    assert_eq!(cli::embedded_kconfig_hash(), expected);
}

// -- show-host --

/// `cargo ktstr show-host` parses with no arguments and maps to
/// the `ShowHost` variant.
#[test]
fn parse_show_host_minimal() {
    let Cargo {
        command: CargoSub::Ktstr(k),
    } = Cargo::try_parse_from(["cargo", "ktstr", "show-host"]).unwrap_or_else(|e| panic!("{e}"));
    assert!(matches!(k.command, KtstrCommand::ShowHost));
}

/// A stray positional argument on `show-host` must be rejected at
/// parse time (clap default) so a typo like
/// `cargo ktstr show-host host_context` fails loudly instead of
/// silently looking like success.
#[test]
fn parse_show_host_rejects_positional() {
    let rejected = Cargo::try_parse_from(["cargo", "ktstr", "show-host", "stray"]);
    assert!(
        rejected.is_err(),
        "show-host must reject positional arguments",
    );
}

/// `cargo ktstr show-thresholds <test>` parses with exactly one
/// positional argument and maps to the `ShowThresholds` variant
/// carrying the test name. Missing argument rejected at parse
/// time; extra argument rejected too. Pins the arg count so a
/// future variadic refactor surfaces here.
#[test]
fn parse_show_thresholds_with_test_arg() {
    let Cargo {
        command: CargoSub::Ktstr(k),
    } = Cargo::try_parse_from(["cargo", "ktstr", "show-thresholds", "my_test_fn"])
        .unwrap_or_else(|e| panic!("{e}"));
    let KtstrCommand::ShowThresholds { test } = k.command else {
        panic!("expected ShowThresholds");
    };
    assert_eq!(test, "my_test_fn");
}

/// `show-thresholds` without the test-name argument must fail
/// at parse time — the positional is required.
#[test]
fn parse_show_thresholds_without_arg_rejected() {
    let rejected = Cargo::try_parse_from(["cargo", "ktstr", "show-thresholds"]);
    assert!(
        rejected.is_err(),
        "show-thresholds requires a test-name argument",
    );
}

/// `show-thresholds <a> <b>` is rejected — variadic inputs would
/// silently drop the second arg or reinterpret it as a flag.
#[test]
fn parse_show_thresholds_extra_arg_rejected() {
    let rejected = Cargo::try_parse_from(["cargo", "ktstr", "show-thresholds", "a", "b"]);
    assert!(
        rejected.is_err(),
        "show-thresholds must accept exactly one positional arg",
    );
}

/// `cli::show_host` produces a non-empty report under normal
/// Linux CI conditions. Catches a regression in the underlying
/// `HostContext::format_human` (e.g. a panic in the
/// destructuring bind that surfaces every field) before the
/// ShowHost dispatch arm reaches it. Named without a
/// `dispatch_` prefix because this exercises the leaf helper
/// directly; true dispatch-path coverage lives in the parse
/// tests above + the binary's `main` call.
#[test]
fn show_host_helper_produces_non_empty_output() {
    let out = cli::show_host();
    assert!(
        !out.is_empty(),
        "show_host must return a non-empty report under normal Linux CI",
    );
    // Stronger pin: `HostContext::format_human` always includes
    // `kernel_release` even when most other fields are `None`
    // (uname is a syscall, filesystem-independent). Asserting
    // the stable field name catches a regression that returned
    // a non-empty but garbage report (e.g. only comments).
    assert!(
        out.contains("kernel_release"),
        "show_host output must include the stable `kernel_release` row: {out}",
    );
}

/// `cli::show_thresholds` returns `Err` with the actionable
/// "no registered ktstr test named" diagnostic when called with
/// an unknown test name. Named without a `dispatch_` prefix for
/// the same reason as `show_host_helper_produces_non_empty_output`
/// — this exercises the leaf helper, not the dispatch path
/// wrapping it.
#[test]
fn show_thresholds_helper_unknown_test_returns_error() {
    let err = cli::show_thresholds("definitely_not_a_registered_test_xyz").unwrap_err();
    let msg = format!("{err:#}");
    assert!(
        msg.contains("no registered ktstr test named"),
        "error path must preserve the actionable diagnostic: {msg}",
    );
}

// -- clap argument-parse pins: Shell --cpu-cap requires --no-perf-mode
//
// `#[arg(long, requires = "no_perf_mode", ...)]` on the
// Shell subcommand's `cpu_cap` field enforces the constraint
// that --cpu-cap is only meaningful in no-perf-mode (perf-mode
// already holds every LLC exclusively, so capping under
// perf-mode would double-reserve). These tests pin the
// invariant so a future refactor that drops or renames the
// `requires` attribute trips a unit-test regression instead of
// surfacing as a runtime double-reservation conflict.

/// `cargo ktstr shell --cpu-cap 4 --no-perf-mode` parses
/// successfully with both flags set. Pins the positive path of
/// the `requires = "no_perf_mode"` constraint — the happy-path
/// invocation an operator would type.
#[test]
fn parse_shell_cpu_cap_with_no_perf_mode_succeeds() {
    let Cargo {
        command: CargoSub::Ktstr(k),
    } = Cargo::try_parse_from([
        "cargo",
        "ktstr",
        "shell",
        "--cpu-cap",
        "4",
        "--no-perf-mode",
    ])
    .unwrap_or_else(|e| panic!("{e}"));
    let KtstrCommand::Shell {
        cpu_cap,
        no_perf_mode,
        ..
    } = k.command
    else {
        panic!("expected Shell");
    };
    assert_eq!(cpu_cap, Some(4));
    assert!(no_perf_mode, "--no-perf-mode must be set");
}

/// `cargo ktstr shell --cpu-cap 4` without `--no-perf-mode`
/// must FAIL at parse time because of the `requires =
/// "no_perf_mode"` constraint. Pins the negative path: if
/// the constraint is ever dropped, this test fails so the
/// regression can't reach production where it would cause a
/// silent double-reservation under perf-mode.
#[test]
fn parse_shell_cpu_cap_without_no_perf_mode_fails() {
    // `Cargo` intentionally has no Debug derive, so unwrap
    // helpers that format the Ok variant are unavailable.
    // Match on Err directly to extract the clap error.
    let msg = match Cargo::try_parse_from(["cargo", "ktstr", "shell", "--cpu-cap", "4"]) {
        Err(e) => e.to_string(),
        Ok(_) => panic!("--cpu-cap without --no-perf-mode must fail the parse"),
    };
    // clap renders "the following required arguments were not provided"
    // or similar; lowercase + substring-match is lenient against
    // clap version-to-version message tweaks while still proving
    // the constraint fired.
    assert!(
        msg.to_ascii_lowercase().contains("no-perf-mode")
            || msg.to_ascii_lowercase().contains("no_perf_mode"),
        "clap error must name the missing --no-perf-mode flag, got: {msg}",
    );
}

/// `cargo ktstr shell --no-perf-mode` without `--cpu-cap`
/// parses successfully with `cpu_cap: None`. Pins the shape of
/// the unset sentinel (expanded to the 30%-of-allowed default by
/// the planner) — a user who wants --no-perf-mode with the
/// implicit default must still be able to invoke the shell. A
/// regression that tied --cpu-cap to --no-perf-mode
/// bidirectionally would fail here.
#[test]
fn parse_shell_no_perf_mode_without_cpu_cap_succeeds() {
    let Cargo {
        command: CargoSub::Ktstr(k),
    } = Cargo::try_parse_from(["cargo", "ktstr", "shell", "--no-perf-mode"])
        .unwrap_or_else(|e| panic!("{e}"));
    let KtstrCommand::Shell {
        cpu_cap,
        no_perf_mode,
        ..
    } = k.command
    else {
        panic!("expected Shell");
    };
    assert_eq!(cpu_cap, None, "no --cpu-cap must produce None");
    assert!(no_perf_mode);
}

// ---------------------------------------------------------------
// KERNEL_LIST_LONG_ABOUT — range-mode JSON schema discoverability
// ---------------------------------------------------------------
//
// `cargo ktstr kernel list --kernel R --json` emits a
// structurally-different JSON shape from the cache-walk mode:
// four top-level fields (`range`, `start`, `end`, `versions`)
// with no cache metadata. The help copy is the
// discoverability contract for scripted consumers — without a
// unit-test pin, a JSON emitter that adds, renames, or removes
// a range-mode field could ship without a matching help update
// and silently break dispatch-on-key consumers. The sibling
// `kernel_list_long_about_exposes_json_schema` test in
// `src/cli/kernel_cmd.rs` covers cache-walk mode; this companion
// fills the range-mode gap from the cargo-ktstr binary's
// perspective and exercises the same `pub const` re-exported
// through `ktstr::cli::KERNEL_LIST_LONG_ABOUT`.

/// Pins that every range-mode JSON top-level field name appears
/// in the help copy by its column-aligned row. Range-mode emits
/// `{ range, start, end, versions }` per the schema block in
/// `KERNEL_LIST_LONG_ABOUT` (`src/cli/kernel_cmd.rs`). Each field
/// is pinned against its column-aligned row prefix (e.g.
/// `  range     literal`) rather than the bare word, since
/// `start` / `end` / `range` appear elsewhere in the help copy
/// (e.g. "parsed start endpoint", "the inclusive range") and a
/// bare-word substring would match the prose, masking a regression
/// that dropped the actual schema row.
///
/// Co-update contract: when the JSON schema changes (field
/// added, renamed, removed, or its emission site moves), three
/// updates land in the same commit:
///   1. the JSON emitter — `cli::kernel_list` /
///      `kernel_list_range_preview` in `src/cli/kernel_list.rs`,
///   2. the help-copy schema block — `KERNEL_LIST_LONG_ABOUT`
///      in `src/cli/kernel_cmd.rs` (the column-aligned table
///      this test reads), and
///   3. this test's column-aligned assertions.
///
/// Updating any one without the others either silently breaks
/// scripted consumers (1 without 2) or surfaces a misleading
/// stale assertion (2 without 3).
#[test]
fn kernel_list_long_about_exposes_range_mode_json_keys() {
    let about = ktstr::cli::KERNEL_LIST_LONG_ABOUT;
    // Column-aligned rows from kernel_cmd.rs's range-mode schema
    // block — each begins with two spaces, the field name, and
    // padding to the description column. Pinning against this
    // exact prefix shape rejects matches inside surrounding prose.
    assert!(
        about.contains("  range     literal"),
        "KERNEL_LIST_LONG_ABOUT must carry the `range` row from the \
         range-mode schema block: got: {about:?}",
    );
    assert!(
        about.contains("  start     parsed start endpoint"),
        "KERNEL_LIST_LONG_ABOUT must carry the `start` row from the \
         range-mode schema block: got: {about:?}",
    );
    assert!(
        about.contains("  end       parsed end endpoint"),
        "KERNEL_LIST_LONG_ABOUT must carry the `end` row from the \
         range-mode schema block: got: {about:?}",
    );
    assert!(
        about.contains("  versions  array of resolved version strings"),
        "KERNEL_LIST_LONG_ABOUT must carry the `versions` row from the \
         range-mode schema block: got: {about:?}",
    );
    // The help copy must explicitly distinguish range-mode from
    // cache-walk-mode by mentioning that the range-mode shape
    // "never carries cache metadata" (the dispatch-on-key contract).
    assert!(
        about.contains("Range-mode output never carries cache metadata"),
        "KERNEL_LIST_LONG_ABOUT must call out the `Range-mode output \
         never carries cache metadata` contract so scripted consumers \
         know to dispatch on the presence of the `range` key versus \
         the `entries` key: got: {about:?}",
    );
    assert!(
        about.contains("--kernel"),
        "KERNEL_LIST_LONG_ABOUT must reference the `--kernel` flag \
         so a `kernel list --help` reader sees the range-mode \
         entry point: got: {about:?}",
    );
    // The exact phrase from KERNEL_LIST_LONG_ABOUT's
    // `SWITCHES to range-preview\nmode` line splits across a
    // line break (`...range-preview\nmode...`), so pin the
    // unambiguous hyphenated token directly. Plain "range mode"
    // also appears in surrounding prose (e.g. help text — see
    // KERNEL_LIST_LONG_ABOUT's
    // `the `range` key (range mode) versus `entries` key (list mode)`
    // line) so a disjunction would re-introduce
    // false-positive risk.
    assert!(
        about.contains("range-preview"),
        "KERNEL_LIST_LONG_ABOUT must use the `range-preview` term so \
         scripted consumers know to dispatch on the presence of the \
         `range` key: got: {about:?}",
    );
}

// -- try_get_matches_from: export subcommand --
//
// `cargo ktstr export <test>` produces a self-extracting `.run`
// reproducer for a registered test. Tests pin the positional
// test name plus the `--output`, `--package`, and `--release`
// flags.

/// `cargo ktstr export <test>` round-trips the bare positional
/// test name with all option fields defaulting to None/false.
#[test]
fn parse_export_with_test_arg() {
    let Cargo {
        command: CargoSub::Ktstr(k),
    } = Cargo::try_parse_from([
        "cargo",
        "ktstr",
        "export",
        "preempt_regression_fault_under_load",
    ])
    .unwrap_or_else(|e| panic!("{e}"));
    let KtstrCommand::Export {
        test,
        output,
        package,
        release,
    } = k.command
    else {
        panic!("expected Export");
    };
    assert_eq!(test, "preempt_regression_fault_under_load");
    assert!(
        output.is_none(),
        "bare export must default --output to None"
    );
    assert!(
        package.is_none(),
        "bare export must default --package to None"
    );
    assert!(!release, "bare export must default --release to false");
}

/// `cargo ktstr export <test> -o PATH --package P --release`
/// round-trips every flag plus the positional argument.
#[test]
fn parse_export_with_all_flags() {
    let Cargo {
        command: CargoSub::Ktstr(k),
    } = Cargo::try_parse_from([
        "cargo",
        "ktstr",
        "export",
        "my_test_fn",
        "-o",
        "/tmp/out.run",
        "--package",
        "scx_rusty",
        "--release",
    ])
    .unwrap_or_else(|e| panic!("{e}"));
    let KtstrCommand::Export {
        test,
        output,
        package,
        release,
    } = k.command
    else {
        panic!("expected Export");
    };
    assert_eq!(test, "my_test_fn");
    assert_eq!(
        output,
        Some(std::path::PathBuf::from("/tmp/out.run")),
        "-o must round-trip to Some(PathBuf)",
    );
    assert_eq!(package.as_deref(), Some("scx_rusty"));
    assert!(release, "--release must lift the flag to true");
}

/// `cargo ktstr export --output PATH ...` (long form of `-o`)
/// must work identically. Pins the long-form spelling so a
/// regression that dropped the long-form attribute surfaces here.
#[test]
fn parse_export_with_output_long_form() {
    let Cargo {
        command: CargoSub::Ktstr(k),
    } = Cargo::try_parse_from([
        "cargo",
        "ktstr",
        "export",
        "test_fn",
        "--output",
        "/tmp/long.run",
    ])
    .unwrap_or_else(|e| panic!("{e}"));
    let KtstrCommand::Export { output, .. } = k.command else {
        panic!("expected Export");
    };
    assert_eq!(output, Some(std::path::PathBuf::from("/tmp/long.run")));
}

/// `cargo ktstr export -p PKG ...` (short form of `--package`)
/// must work identically.
#[test]
fn parse_export_with_package_short_form() {
    let Cargo {
        command: CargoSub::Ktstr(k),
    } = Cargo::try_parse_from(["cargo", "ktstr", "export", "test_fn", "-p", "ktstr"])
        .unwrap_or_else(|e| panic!("{e}"));
    let KtstrCommand::Export { package, .. } = k.command else {
        panic!("expected Export");
    };
    assert_eq!(package.as_deref(), Some("ktstr"));
}

/// `cargo ktstr export` without a positional test name must fail
/// at parse time — the test name is required.
#[test]
fn parse_export_missing_test_arg_rejected() {
    let rejected = Cargo::try_parse_from(["cargo", "ktstr", "export"]);
    assert!(
        rejected.is_err(),
        "export must require a positional test-name argument",
    );
}

/// `cargo ktstr export <a> <b>` is rejected — `export` accepts
/// exactly one positional test name. A variadic regression would
/// silently drop the second arg (or reinterpret it as a flag value
/// like `--package b`), masking the operator's typo. Mirrors
/// `parse_show_thresholds_extra_arg_rejected` for the export
/// subcommand.
#[test]
fn parse_export_extra_arg_rejected() {
    let rejected = Cargo::try_parse_from(["cargo", "ktstr", "export", "a", "b"]);
    assert!(
        rejected.is_err(),
        "export must accept exactly one positional arg",
    );
}

// -- try_get_matches_from: locks subcommand --
//
// `cargo ktstr locks` snapshots ktstr flock state under
// `/tmp/ktstr-*.lock` and `{cache_root}/.locks/*.lock` for
// `--cpu-cap` contention diagnosis. Tests pin the `--json` and
// `--watch` flags plus the `humantime` value parser on `--watch`.

/// `cargo ktstr locks` parses bare with both fields default.
#[test]
fn parse_locks_bare() {
    let Cargo {
        command: CargoSub::Ktstr(k),
    } = Cargo::try_parse_from(["cargo", "ktstr", "locks"]).unwrap_or_else(|e| panic!("{e}"));
    let KtstrCommand::Locks { json, watch } = k.command else {
        panic!("expected Locks");
    };
    assert!(!json, "bare locks must default --json to false");
    assert!(watch.is_none(), "bare locks must default --watch to None");
}

/// `cargo ktstr locks --json` lifts the json field to true.
#[test]
fn parse_locks_with_json() {
    let Cargo {
        command: CargoSub::Ktstr(k),
    } = Cargo::try_parse_from(["cargo", "ktstr", "locks", "--json"])
        .unwrap_or_else(|e| panic!("{e}"));
    let KtstrCommand::Locks { json, watch } = k.command else {
        panic!("expected Locks");
    };
    assert!(json, "--json must lift the flag to true");
    assert!(watch.is_none(), "bare --json must not populate --watch");
}

/// `cargo ktstr locks --watch <DURATION>` round-trips a humantime
/// duration through the `value_parser =
/// humantime::parse_duration` attribute.
#[test]
fn parse_locks_with_watch_duration() {
    let Cargo {
        command: CargoSub::Ktstr(k),
    } = Cargo::try_parse_from(["cargo", "ktstr", "locks", "--watch", "500ms"])
        .unwrap_or_else(|e| panic!("{e}"));
    let KtstrCommand::Locks { json, watch } = k.command else {
        panic!("expected Locks");
    };
    assert!(!json, "bare --watch must not populate --json");
    assert_eq!(
        watch,
        Some(std::time::Duration::from_millis(500)),
        "--watch 500ms must round-trip to Duration::from_millis(500)",
    );
}

/// `cargo ktstr locks --watch 5s --json` round-trips both flags
/// in combination — the `--watch` redraw mode emits ndjson when
/// `--json` is also set.
#[test]
fn parse_locks_with_watch_and_json() {
    let Cargo {
        command: CargoSub::Ktstr(k),
    } = Cargo::try_parse_from(["cargo", "ktstr", "locks", "--watch", "5s", "--json"])
        .unwrap_or_else(|e| panic!("{e}"));
    let KtstrCommand::Locks { json, watch } = k.command else {
        panic!("expected Locks");
    };
    assert!(json, "--json must lift the flag to true alongside --watch");
    assert_eq!(
        watch,
        Some(std::time::Duration::from_secs(5)),
        "--watch 5s must round-trip to Duration::from_secs(5)",
    );
}

/// `cargo ktstr locks --watch <BAD>` rejects malformed duration
/// strings at parse time via humantime's `parse_duration`. Catches
/// a regression that dropped the `value_parser` attribute and
/// turned the field into a raw String / unbounded text input.
#[test]
fn parse_locks_watch_rejects_malformed_duration() {
    let rejected = Cargo::try_parse_from(["cargo", "ktstr", "locks", "--watch", "not-a-duration"]);
    assert!(
        rejected.is_err(),
        "--watch must reject malformed humantime input via the \
         value_parser = humantime::parse_duration attribute",
    );
}

// -- try_get_matches_from: shell --memory-mib / --exec / --dmesg --

/// `cargo ktstr shell --memory-mib 256` round-trips the value
/// through clap's `value_parser!(u32).range(128..)` attribute.
#[test]
fn parse_shell_memory_mib_valid() {
    let Cargo {
        command: CargoSub::Ktstr(k),
    } = Cargo::try_parse_from(["cargo", "ktstr", "shell", "--memory-mib", "256"])
        .unwrap_or_else(|e| panic!("{e}"));
    let KtstrCommand::Shell { memory_mib, .. } = k.command else {
        panic!("expected Shell");
    };
    assert_eq!(memory_mib, Some(256), "--memory-mib 256 must round-trip");
}

/// `cargo ktstr shell --memory-mib 128` accepts the range floor —
/// the clap range is `128..` (inclusive).
#[test]
fn parse_shell_memory_mib_at_range_floor() {
    let Cargo {
        command: CargoSub::Ktstr(k),
    } = Cargo::try_parse_from(["cargo", "ktstr", "shell", "--memory-mib", "128"])
        .unwrap_or_else(|e| panic!("{e}"));
    let KtstrCommand::Shell { memory_mib, .. } = k.command else {
        panic!("expected Shell");
    };
    assert_eq!(
        memory_mib,
        Some(128),
        "--memory-mib 128 must succeed at the inclusive range floor",
    );
}

/// `cargo ktstr shell --memory-mib 64` is rejected — below the
/// `value_parser!(u32).range(128..)` floor. Pins the constraint:
/// a regression that dropped the range or relaxed the lower
/// bound surfaces here.
#[test]
fn parse_shell_memory_mib_below_range_rejected() {
    let rejected = Cargo::try_parse_from(["cargo", "ktstr", "shell", "--memory-mib", "64"]);
    assert!(
        rejected.is_err(),
        "--memory-mib 64 must be rejected — value_parser range floor is 128",
    );
}

/// `cargo ktstr shell --memory-mib -1` is rejected at parse time —
/// the field is `u32`, so a signed value cannot satisfy the
/// type-level value parser. Pins the unsigned-integer
/// constraint.
#[test]
fn parse_shell_memory_mib_negative_rejected() {
    let rejected = Cargo::try_parse_from(["cargo", "ktstr", "shell", "--memory-mib", "-1"]);
    assert!(
        rejected.is_err(),
        "--memory-mib -1 must be rejected — the field is u32",
    );
}

/// `cargo ktstr shell --memory-mb 256` is rejected because the
/// canonical flag was renamed to `--memory-mib`. Pre-1.0
/// break-cleanly forbids a compat alias for the old form. A
/// regression that re-added `alias = "memory-mb"` to the Shell
/// variant's `memory_mib` field would silently accept the old
/// spelling — this test pins the reject.
#[test]
fn parse_shell_memory_mb_rejected() {
    let rejected = Cargo::try_parse_from(["cargo", "ktstr", "shell", "--memory-mb", "256"]);
    assert!(
        rejected.is_err(),
        "`--memory-mb` (old flag name) must be rejected — the \
         canonical name is `--memory-mib`. A regression that \
         re-added an alias for the old form would surface here.",
    );
}

/// `cargo ktstr shell --exec "uname -a"` round-trips the command
/// string through clap into `Shell { exec: Some(..), .. }`. The
/// dispatch site forwards the string into the VM's init line.
#[test]
fn parse_shell_with_exec() {
    let Cargo {
        command: CargoSub::Ktstr(k),
    } = Cargo::try_parse_from(["cargo", "ktstr", "shell", "--exec", "uname -a"])
        .unwrap_or_else(|e| panic!("{e}"));
    let KtstrCommand::Shell { exec, .. } = k.command else {
        panic!("expected Shell");
    };
    assert_eq!(exec.as_deref(), Some("uname -a"));
}

/// `cargo ktstr shell --dmesg` lifts the dmesg field to true.
#[test]
fn parse_shell_with_dmesg() {
    let Cargo {
        command: CargoSub::Ktstr(k),
    } = Cargo::try_parse_from(["cargo", "ktstr", "shell", "--dmesg"])
        .unwrap_or_else(|e| panic!("{e}"));
    let KtstrCommand::Shell { dmesg, .. } = k.command else {
        panic!("expected Shell");
    };
    assert!(dmesg, "--dmesg must lift the flag to true");
}

/// Bare `cargo ktstr shell` defaults `--dmesg` to false and
/// `--exec` to None. Pins that neither flag is implicitly set.
#[test]
fn parse_shell_dmesg_and_exec_default_unset() {
    let Cargo {
        command: CargoSub::Ktstr(k),
    } = Cargo::try_parse_from(["cargo", "ktstr", "shell"]).unwrap_or_else(|e| panic!("{e}"));
    let KtstrCommand::Shell { dmesg, exec, .. } = k.command else {
        panic!("expected Shell");
    };
    assert!(!dmesg, "bare shell must default --dmesg to false");
    assert!(exec.is_none(), "bare shell must default --exec to None");
}

// -- try_get_matches_from: --kernel ArgAction::Append on every
// `Vec<String> kernel` subcommand. Repeats fan out the gauntlet
// across kernels at the dispatch layer; a regression that lost
// `ArgAction::Append` would either reject the second occurrence
// outright (`Vec<String>` derive without the action would fail
// "the argument was supplied more than once") or silently keep
// only the last value.

/// `cargo ktstr test --kernel A --kernel B` accumulates both
/// values into the `kernel` Vec via `ArgAction::Append`.
#[test]
fn parse_test_kernel_repeatable() {
    let Cargo {
        command: CargoSub::Ktstr(k),
    } = Cargo::try_parse_from([
        "cargo", "ktstr", "test", "--kernel", "6.14.2", "--kernel", "6.15-rc3",
    ])
    .unwrap_or_else(|e| panic!("{e}"));
    let KtstrCommand::Test { kernel, .. } = k.command else {
        panic!("expected Test");
    };
    assert_eq!(
        kernel,
        vec!["6.14.2".to_string(), "6.15-rc3".to_string()],
        "test --kernel must accumulate via ArgAction::Append",
    );
}

/// `cargo ktstr coverage --kernel A --kernel B` accumulates both
/// values via `ArgAction::Append`.
#[test]
fn parse_coverage_kernel_repeatable() {
    let Cargo {
        command: CargoSub::Ktstr(k),
    } = Cargo::try_parse_from([
        "cargo", "ktstr", "coverage", "--kernel", "6.14.2", "--kernel", "6.15-rc3",
    ])
    .unwrap_or_else(|e| panic!("{e}"));
    let KtstrCommand::Coverage { kernel, .. } = k.command else {
        panic!("expected Coverage");
    };
    assert_eq!(
        kernel,
        vec!["6.14.2".to_string(), "6.15-rc3".to_string()],
        "coverage --kernel must accumulate via ArgAction::Append",
    );
}

/// `cargo ktstr llvm-cov --kernel A --kernel B` accumulates both
/// values via `ArgAction::Append`.
#[test]
fn parse_llvm_cov_kernel_repeatable() {
    let Cargo {
        command: CargoSub::Ktstr(k),
    } = Cargo::try_parse_from([
        "cargo", "ktstr", "llvm-cov", "--kernel", "6.14.2", "--kernel", "6.15-rc3",
    ])
    .unwrap_or_else(|e| panic!("{e}"));
    let KtstrCommand::LlvmCov { kernel, .. } = k.command else {
        panic!("expected LlvmCov");
    };
    assert_eq!(
        kernel,
        vec!["6.14.2".to_string(), "6.15-rc3".to_string()],
        "llvm-cov --kernel must accumulate via ArgAction::Append",
    );
}

/// `cargo ktstr verifier --kernel A --kernel B` accumulates both
/// values via `ArgAction::Append`. Mirrors
/// `parse_test_kernel_repeatable` for the verifier subcommand —
/// the verifier surface is `--kernel` + `--raw` only (no
/// `--scheduler`).
#[test]
fn parse_verifier_kernel_repeatable() {
    let Cargo {
        command: CargoSub::Ktstr(k),
    } = Cargo::try_parse_from([
        "cargo", "ktstr", "verifier", "--kernel", "6.14.2", "--kernel", "6.15-rc3",
    ])
    .unwrap_or_else(|e| panic!("{e}"));
    let KtstrCommand::Verifier { kernel, .. } = k.command else {
        panic!("expected Verifier");
    };
    assert_eq!(
        kernel,
        vec!["6.14.2".to_string(), "6.15-rc3".to_string()],
        "verifier --kernel must accumulate via ArgAction::Append",
    );
}

// -- try_get_matches_from: completions --binary default --

/// `cargo ktstr completions bash` defaults `--binary` to `cargo`
/// per the `default_value = "cargo"` attribute on the field.
/// A regression that dropped or changed the default surfaces
/// here.
#[test]
fn parse_completions_binary_default_is_cargo() {
    let Cargo {
        command: CargoSub::Ktstr(k),
    } = Cargo::try_parse_from(["cargo", "ktstr", "completions", "bash"])
        .unwrap_or_else(|e| panic!("{e}"));
    let KtstrCommand::Completions { binary, .. } = k.command else {
        panic!("expected Completions");
    };
    assert_eq!(binary, "cargo", "default --binary must be `cargo`",);
}

/// `cargo ktstr completions bash --binary X` overrides the
/// default. Pins the override path.
#[test]
fn parse_completions_binary_override() {
    let Cargo {
        command: CargoSub::Ktstr(k),
    } = Cargo::try_parse_from(["cargo", "ktstr", "completions", "bash", "--binary", "ktstr"])
        .unwrap_or_else(|e| panic!("{e}"));
    let KtstrCommand::Completions { binary, .. } = k.command else {
        panic!("expected Completions");
    };
    assert_eq!(binary, "ktstr");
}

/// A perf-delta flag REMOVED from the native surface (the per-side A/B axis,
/// moved in-test to the Verdict DSL `better_across_phases`) must surface as a
/// clean top-level clap "unexpected argument" error, NOT be forwarded verbatim
/// into the nextest passthrough (which produced a confusing nested error).
/// `argsplit::rewrite` routes the rejected flag into the ktstr bucket (before
/// the emitted `--`) so clap rejects it by name.
#[test]
fn parse_perf_delta_removed_ab_flag_is_clean_unknown_flag_error() {
    for flag in [
        "--a-scheduler",
        "--b-scheduler",
        "--a-topology",
        "--b-topology",
    ] {
        // Both the space-separated (`--a-scheduler scx_foo`) and the =value
        // (`--a-scheduler=scx_foo`) forms take distinct partition code paths
        // (the latter splits the token at `=`); both must route the flag into
        // the ktstr bucket for a clean clap error rather than the passthrough.
        let eq_form = format!("{flag}=scx_foo");
        let forms: [Vec<&str>; 2] = [
            vec!["cargo", "ktstr", "perf-delta", flag, "scx_foo"],
            vec!["cargo", "ktstr", "perf-delta", eq_form.as_str()],
        ];
        for argv in forms {
            let raw: Vec<std::ffi::OsString> = argv.iter().map(std::ffi::OsString::from).collect();
            let rewritten = crate::argsplit::rewrite(&Cargo::command(), &raw);
            let err = Cargo::try_parse_from(&rewritten).err().unwrap_or_else(|| {
                panic!("removed flag {flag} ({argv:?}) must be rejected, not forwarded to nextest")
            });
            assert_eq!(
                err.kind(),
                clap::error::ErrorKind::UnknownArgument,
                "removed flag {flag} ({argv:?}) must yield a clean UnknownArgument error; got {:?}",
                err.kind(),
            );
            assert!(
                err.to_string().contains(flag),
                "the error must name the offending flag {flag} ({argv:?}); got: {err}",
            );
        }
    }
}

/// `--dual-run` was removed (single-run production could not distinguish a real
/// regression from run-to-run noise; use `--noise-adjust N` for fresh runs). It
/// must now be a clean UnknownArgument error, not silently forwarded to the
/// nextest passthrough.
#[test]
fn parse_perf_delta_dual_run_removed_is_unknown_flag_error() {
    let raw: Vec<std::ffi::OsString> = [
        "cargo",
        "ktstr",
        "perf-delta",
        "--dual-run",
        "--kernel",
        "6.14",
    ]
    .iter()
    .map(std::ffi::OsString::from)
    .collect();
    let rewritten = crate::argsplit::rewrite(&Cargo::command(), &raw);
    let err = Cargo::try_parse_from(&rewritten)
        .err()
        .unwrap_or_else(|| panic!("--dual-run must be rejected, not forwarded to nextest"));
    assert_eq!(
        err.kind(),
        clap::error::ErrorKind::UnknownArgument,
        "--dual-run must yield a clean UnknownArgument error; got {:?}",
        err.kind(),
    );
    assert!(
        err.to_string().contains("--dual-run"),
        "the error must name --dual-run; got: {err}",
    );
}
