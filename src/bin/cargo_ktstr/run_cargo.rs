//! Dispatch helpers for the `test`, `coverage`, and `llvm-cov`
//! subcommands.
//!
//! All three subcommands share the `cargo nextest`/`cargo
//! llvm-cov` execve plumbing, the `--no-perf-mode` /
//! `--no-skip-mode` env-var pass-throughs, and the multi-kernel
//! [`ktstr::KTSTR_KERNEL_LIST_ENV`] export. The differences live
//! in the leading `cargo` subcommand argv (`{nextest run}` vs
//! `{llvm-cov nextest}` vs `{llvm-cov}`) and the optional
//! `--cargo-profile release` injection on the test/coverage
//! paths. [`run_cargo_sub`] folds the shared shape; thin
//! per-subcommand wrappers fix the argv constants.
//!
//! test/coverage additionally accept `--profile <NAME>` (the
//! scheduler-under-test's cargo BUILD profile, forwarded via the
//! [`ktstr::KTSTR_SCHEDULER_PROFILE_ENV`] env) and `--nextest-profile
//! <NAME>` (the nextest test profile, emitted as nextest's own
//! `--profile`); the raw `llvm-cov` passthrough sets neither. `--profile`
//! is INDEPENDENT of `--release` (`--cargo-profile release`, the harness
//! build profile).

use std::collections::HashMap;
use std::ffi::{OsStr, OsString};
use std::os::unix::ffi::OsStrExt as _;
use std::os::unix::fs::PermissionsExt as _;
use std::path::{Path, PathBuf};
use std::process::Command;

use cargo_metadata::semver::Version;

use crate::feature_discovery::{
    MetadataMode, augment_test_features_from_metadata_for_context_at, effective_target_context,
    query_metadata_for_target, query_resolved_metadata_for_invocation,
    selected_workspace_packages_for_invocation,
};
use crate::kernel::{encode_kernel_list, resolve_kernel_set};

/// Cargo sub-argv that `run_test` passes to `run_cargo_sub`. Named
/// constant so the dispatch wiring is pinnable from a test — see
/// `cargo_sub_argv_constants_are_pinned`.
pub(crate) const TEST_SUB_ARGV: &[&str] = &["nextest", "run"];
/// Cargo sub-argv for the `coverage` subcommand (cargo llvm-cov
/// nextest).
pub(crate) const COVERAGE_SUB_ARGV: &[&str] = &["llvm-cov", "nextest"];
/// Cargo sub-argv for the `llvm-cov` raw-passthrough subcommand.
/// Single element — the user's trailing args supply the llvm-cov
/// subcommand (`report`, `clean`, `show-env`, ...).
pub(crate) const LLVM_COV_SUB_ARGV: &[&str] = &["llvm-cov"];

/// cargo-llvm-cov global options whose space-separated spelling consumes the
/// following token. Keep this table aligned with cargo-llvm-cov's CLI; the
/// detector tests exercise every entry with a value literally named `nextest`
/// so no option value can be mistaken for the subcommand.
const LLVM_COV_GLOBAL_VALUE_OPTIONS: &[&str] = &[
    "--output-path",
    "--output-dir",
    "--failure-mode",
    "--ignore-filename-regex",
    "--fail-under-functions",
    "--fail-under-lines",
    "--fail-under-file-lines",
    "--fail-under-regions",
    "--fail-uncovered-lines",
    "--fail-uncovered-regions",
    "--fail-uncovered-functions",
    "--dep-coverage",
    "--bin",
    "--example",
    "--test",
    "--bench",
    "-p",
    "--package",
    "--exclude",
    "--exclude-from-test",
    "--exclude-from-report",
    "-j",
    "--jobs",
    "--profile",
    "-F",
    "--features",
    "--target",
    "--color",
    "--manifest-path",
    "--cargo-profile",
    "--archive-file",
    "--nextest-archive-file",
    "-Z",
];

/// Additional value-taking spellings accepted after cargo-llvm-cov's
/// `nextest` subcommand.
///
/// The lifecycle scanner must not mistake an option value literally named
/// `--no-clean`, `--no-report`, or `--no-run` for a cargo-llvm-cov retention
/// flag. This table covers the nextest/Cargo controls that are not already in
/// [`LLVM_COV_GLOBAL_VALUE_OPTIONS`].
const LLVM_COV_NEXTEST_VALUE_OPTIONS: &[&str] = &[
    "--cargo-profile",
    "--exclude",
    "--exclude-from-test",
    "--exclude-from-report",
    "--build-jobs",
    "--test-threads",
    "--retries",
    "--max-fail",
    "--cargo-message-format",
    "--config",
    "--target-dir",
    "--archive-file",
    "--archive-format",
    "--zstd-level",
    "--nextest-archive-file",
    "--config-file",
    "--user-config-file",
    "--tool-config-file",
    "-P",
    "-E",
    "--filterset",
    "--filter-expr",
];

/// Whether the raw `cargo ktstr llvm-cov` passthrough explicitly selects
/// cargo-llvm-cov's nextest subcommand.
///
/// Bare `llvm-cov` and its `test` subcommand use Cargo's standard test
/// harness, not ktstr's nextest listing/dispatch protocol. Report, clean,
/// show-env, and run do not enumerate ktstr test binaries either. Keep this
/// deliberately narrow: the documented test-producing form is
/// `cargo ktstr llvm-cov nextest ...`.
fn llvm_cov_nextest_index(args: &[String]) -> Option<usize> {
    let mut index = 0;
    while index < args.len() {
        let argument = &args[index];
        if argument == "--" {
            return None;
        }
        if !argument.starts_with('-') {
            return (argument == "nextest").then_some(index);
        }

        // cargo-llvm-cov accepts its global options before the subcommand.
        // Skip the value of every current value-taking global option so a
        // value literally named `nextest` is not mistaken for the subcommand.
        let takes_separate_value = LLVM_COV_GLOBAL_VALUE_OPTIONS.contains(&argument.as_str());
        index += if takes_separate_value { 2 } else { 1 };
    }
    None
}

fn llvm_cov_uses_nextest(args: &[String]) -> bool {
    llvm_cov_nextest_index(args).is_some()
}

/// Apply the ordinary nextest metadata preflight to the raw llvm-cov
/// passthrough only when it explicitly selects llvm-cov's `nextest`
/// subcommand.
///
/// Keeping the preparer injectable makes the routing contract independently
/// testable without running Cargo metadata. Production passes
/// [`prepare_nextest_args`], so `cargo ktstr llvm-cov nextest` cannot drift from
/// `test` / `coverage` in package selection, target filtering, inferred
/// features, or version checks. Non-test llvm-cov modes never call the
/// preparer and retain their exact argv.
#[cfg(test)]
fn prepare_llvm_cov_args_with(
    args: Vec<String>,
    prepare: impl FnOnce(Vec<String>) -> Result<Vec<String>, String>,
) -> Result<Vec<String>, String> {
    if llvm_cov_uses_nextest(&args)
        && !llvm_cov_reuses_archive(LLVM_COV_SUB_ARGV, &args)
        && !llvm_cov_has_lifecycle_flag(&args, "--no-run")
    {
        prepare(args)
    } else {
        Ok(args)
    }
}

/// Whether this cargo sub-invocation ultimately runs tests through nextest.
///
/// `test` and `coverage` have fixed nextest command shapes. The raw
/// `llvm-cov` passthrough is conditional: only its explicit `nextest`
/// subcommand accepts nextest's tool-config flag. Keeping this decision next
/// to [`llvm_cov_uses_nextest`] prevents report/clean/show-env passthroughs
/// from being mutated.
pub(crate) fn cargo_sub_uses_nextest(sub_argv: &[&str], args: &[String]) -> bool {
    let selects_nextest = sub_argv == TEST_SUB_ARGV
        || sub_argv == COVERAGE_SUB_ARGV
        || (sub_argv == LLVM_COV_SUB_ARGV && llvm_cov_uses_nextest(args));
    selects_nextest && !(sub_argv != TEST_SUB_ARGV && llvm_cov_has_lifecycle_flag(args, "--no-run"))
}

/// Whether the final cargo-llvm-cov invocation deliberately retains existing
/// coverage artifacts.
///
/// cargo-llvm-cov makes `--no-report` and `--no-run` imply `--no-clean`.
/// Those modes are user-owned merge/build-only workflows: ktstr must not
/// pre-clean them or add a second lifecycle policy. Value-taking options are
/// skipped so a value which happens to equal one of these flags stays opaque.
fn llvm_cov_has_lifecycle_flag(args: &[String], wanted: &str) -> bool {
    let mut index = 0;
    while index < args.len() {
        let argument = &args[index];
        if argument == "--" {
            break;
        }
        if argument == wanted {
            return true;
        }
        let takes_value = llvm_cov_or_nextest_option_takes_value(argument);
        index += if takes_value { 2 } else { 1 };
    }
    false
}

fn llvm_cov_retains_artifacts(args: &[String]) -> bool {
    ["--no-clean", "--no-report", "--no-run"]
        .iter()
        .any(|flag| llvm_cov_has_lifecycle_flag(args, flag))
}

/// Whether a nextest-backed llvm-cov invocation requires cargo-llvm-cov's
/// ordinary mutable target rather than ktstr's reusable COW closure.
///
/// `--no-run` is report-only and `--no-clean` explicitly asks to accumulate in
/// the caller-owned target. A lone `--no-report` is different: ktstr can retain
/// the exact private closure as one bounded persistent bundle and provide a
/// replay script, so it remains eligible for cached artifact reuse.
fn llvm_cov_requires_ordinary_retained_target(args: &[String]) -> bool {
    ["--no-clean", "--no-run"]
        .iter()
        .any(|flag| llvm_cov_has_lifecycle_flag(args, flag))
}

/// Whether ktstr must own cargo-llvm-cov's clean/warm/run lifecycle.
fn llvm_cov_managed_lifecycle(sub_argv: &[&str], args: &[String]) -> bool {
    (sub_argv == COVERAGE_SUB_ARGV
        || (sub_argv == LLVM_COV_SUB_ARGV && llvm_cov_uses_nextest(args)))
        && !llvm_cov_retains_artifacts(args)
        && !llvm_cov_reuses_archive(sub_argv, args)
}

fn nextest_region_start(sub_argv: &[&str], args: &[String]) -> Option<usize> {
    if sub_argv == TEST_SUB_ARGV || sub_argv == COVERAGE_SUB_ARGV {
        Some(0)
    } else if sub_argv == LLVM_COV_SUB_ARGV {
        llvm_cov_nextest_index(args).map(|index| index + 1)
    } else {
        None
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct NextestArchiveOption {
    option_index: usize,
    separate_value: bool,
    value: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct NextestArchiveReuse {
    file: NextestArchiveOption,
    format: Option<NextestArchiveOption>,
}

fn nextest_archive_reuse(
    sub_argv: &[&str],
    args: &[String],
) -> Result<Option<NextestArchiveReuse>, String> {
    if sub_argv == LLVM_COV_SUB_ARGV {
        let mut index = 0;
        while index < args.len() {
            let argument = &args[index];
            if argument == "--archive-format"
                && args[index + 1..].iter().any(|value| value == "nextest")
            {
                return Err("cargo ktstr: split --archive-format is not valid before \
                     cargo-llvm-cov's nextest subcommand; use \
                     --archive-format=tar-zst or place it after nextest"
                    .to_string());
            }
            if !argument.starts_with('-') || argument == "--" {
                break;
            }
            index += if LLVM_COV_GLOBAL_VALUE_OPTIONS.contains(&argument.as_str()) {
                2
            } else {
                1
            };
        }
    }
    let raw_nextest_index = (sub_argv == LLVM_COV_SUB_ARGV)
        .then(|| llvm_cov_nextest_index(args))
        .flatten();
    let Some(mut index) = (if sub_argv == LLVM_COV_SUB_ARGV {
        raw_nextest_index.map(|_| 0)
    } else {
        nextest_region_start(sub_argv, args)
    }) else {
        return Ok(None);
    };
    let mut file = None;
    let mut format = None;
    while index < args.len() {
        let argument = &args[index];
        if argument == "--" {
            break;
        }
        if Some(index) == raw_nextest_index {
            index += 1;
            continue;
        }
        let parsed = if argument == "--archive-file" || argument == "--archive-format" {
            let archive_file = argument == "--archive-file";
            let value = args.get(index + 1).filter(|value| {
                value.as_str() != "--" && (archive_file || !value.starts_with('-'))
            });
            let Some(value) = value else {
                return Err(format!("cargo ktstr: {argument} requires a nonempty value"));
            };
            let parsed = NextestArchiveOption {
                option_index: index,
                separate_value: true,
                value: value.clone(),
            };
            index += 2;
            Some((argument.as_str(), parsed))
        } else if let Some(value) = argument.strip_prefix("--archive-file=") {
            if value.is_empty() {
                return Err("cargo ktstr: --archive-file requires a nonempty value".to_string());
            }
            let parsed = NextestArchiveOption {
                option_index: index,
                separate_value: false,
                value: value.to_string(),
            };
            index += 1;
            Some(("--archive-file", parsed))
        } else if let Some(value) = argument.strip_prefix("--archive-format=") {
            if value.is_empty() {
                return Err("cargo ktstr: --archive-format requires a nonempty value".to_string());
            }
            let parsed = NextestArchiveOption {
                option_index: index,
                separate_value: false,
                value: value.to_string(),
            };
            index += 1;
            Some(("--archive-format", parsed))
        } else {
            None
        };
        if let Some((option, parsed)) = parsed {
            let slot = if option == "--archive-file" {
                &mut file
            } else {
                &mut format
            };
            if slot.replace(parsed).is_some() {
                return Err(format!("cargo ktstr: duplicate {option} is ambiguous"));
            }
            continue;
        }
        index += if llvm_cov_or_nextest_option_takes_value(argument) {
            2
        } else {
            1
        };
    }
    let Some(file) = file else {
        if format.is_some() {
            return Err("cargo ktstr: --archive-format requires --archive-file".to_string());
        }
        return Ok(None);
    };
    let supported = format.as_ref().map_or_else(
        || file.value.ends_with(".tar.zst"),
        |format| {
            matches!(format.value.as_str(), "tar-zst" | "tar-zstd")
                || (format.value == "auto" && file.value.ends_with(".tar.zst"))
        },
    );
    if !supported {
        return Err(format!(
            "cargo ktstr: archive format {} does not resolve to tar-zst \
             for --archive-file {:?}",
            format
                .as_ref()
                .map_or("auto", |format| format.value.as_str()),
            file.value,
        ));
    }
    Ok(Some(NextestArchiveReuse { file, format }))
}

fn llvm_cov_reuses_archive(sub_argv: &[&str], args: &[String]) -> bool {
    nextest_archive_reuse(sub_argv, args)
        .ok()
        .flatten()
        .is_some()
}

fn llvm_cov_archive_path(
    sub_argv: &[&str],
    args: &[String],
    invocation_dir: &std::path::Path,
) -> Result<Option<PathBuf>, String> {
    let Some(reuse) = nextest_archive_reuse(sub_argv, args)? else {
        return Ok(None);
    };
    let path = PathBuf::from(reuse.file.value);
    Ok(Some(if path.is_absolute() {
        path
    } else {
        invocation_dir.join(path)
    }))
}

fn rewrite_nextest_archive_reuse(
    sub_argv: &[&str],
    mut args: Vec<String>,
    archive_path: &std::path::Path,
) -> Result<Vec<String>, String> {
    let reuse = nextest_archive_reuse(sub_argv, &args)?.ok_or_else(|| {
        "cargo ktstr: cannot rewrite a nextest invocation without --archive-file".to_string()
    })?;
    let path = archive_path.to_string_lossy().into_owned();
    if reuse.file.separate_value {
        args[reuse.file.option_index + 1] = path;
    } else {
        args[reuse.file.option_index] = format!("--archive-file={path}");
    }
    if let Some(format) = reuse.format {
        if format.separate_value {
            args[format.option_index + 1] = "tar-zst".to_string();
        } else {
            args[format.option_index] = "--archive-format=tar-zst".to_string();
        }
    } else {
        let region_start =
            nextest_region_start(sub_argv, &args).expect("validated archive reuse selects nextest");
        let insertion = args[region_start..]
            .iter()
            .position(|argument| argument == "--")
            .map_or(args.len(), |offset| region_start + offset);
        args.insert(insertion, "--archive-format=tar-zst".to_string());
    }
    Ok(args)
}

fn cargo_sub_needs_reserved_prebuild(sub_argv: &[&str], args: &[String]) -> bool {
    if !cargo_sub_uses_nextest(sub_argv, args) {
        return false;
    }
    !llvm_cov_has_lifecycle_flag(args, "--no-run") && !llvm_cov_reuses_archive(sub_argv, args)
}

const ORCHESTRATED_NEXTEST_TEST_THREADS: usize = 1_000_000;

fn valid_nextest_test_threads(value: &str) -> bool {
    value == "num-cpus" || value.parse::<isize>().is_ok_and(|threads| threads != 0)
}

fn normalize_nextest_region(args: Vec<String>, region_start: usize) -> Vec<String> {
    let separator = args[region_start..]
        .iter()
        .position(|argument| argument == "--")
        .map_or(args.len(), |offset| region_start + offset);
    let mut out = Vec::with_capacity(args.len() + 1);
    out.extend(args[..region_start].iter().cloned());
    let mut index = region_start;
    while index < separator {
        let argument = &args[index];
        if matches!(argument.as_str(), "-j" | "--test-threads")
            && args
                .get(index + 1)
                .filter(|_| index + 1 < separator)
                .is_some_and(|value| valid_nextest_test_threads(value))
        {
            index += 2;
            continue;
        }
        let valid_joined = argument
            .strip_prefix("-j")
            .filter(|value| !value.is_empty())
            .map(|value| value.strip_prefix('=').unwrap_or(value))
            .is_some_and(valid_nextest_test_threads)
            || argument
                .strip_prefix("--test-threads=")
                .is_some_and(valid_nextest_test_threads);
        if valid_joined {
            index += 1;
            continue;
        }
        out.push(argument.clone());
        index += 1;
    }
    out.push(format!(
        "--test-threads={ORCHESTRATED_NEXTEST_TEST_THREADS}"
    ));
    out.extend(args[separator..].iter().cloned());
    out
}

fn normalize_raw_llvm_cov_build_jobs(args: Vec<String>) -> Vec<String> {
    let Some(nextest_index) = llvm_cov_nextest_index(&args) else {
        return args;
    };
    let mut out = Vec::with_capacity(args.len() + 2);
    let mut build_jobs = None;
    let mut index = 0;
    while index < nextest_index {
        let argument = &args[index];
        if matches!(argument.as_str(), "-j" | "--jobs") {
            if let Some(value) = args.get(index + 1) {
                build_jobs = Some(value.clone());
                index += 2;
            } else {
                out.push(argument.clone());
                index += 1;
            }
            continue;
        }
        if let Some(value) = argument.strip_prefix("--jobs=").or_else(|| {
            argument
                .strip_prefix("-j")
                .filter(|value| !value.is_empty())
                .map(|value| value.strip_prefix('=').unwrap_or(value))
        }) {
            build_jobs = Some(value.to_string());
            index += 1;
            continue;
        }
        out.push(argument.clone());
        index += 1;
        if LLVM_COV_GLOBAL_VALUE_OPTIONS.contains(&argument.as_str()) && index < nextest_index {
            out.push(args[index].clone());
            index += 1;
        }
    }
    out.push("nextest".to_string());
    index = nextest_index + 1;
    let separator = args[index..]
        .iter()
        .position(|argument| argument == "--")
        .map_or(args.len(), |offset| index + offset);
    while index < separator {
        let argument = &args[index];
        if argument == "--build-jobs" {
            if let Some(value) = args.get(index + 1).filter(|_| index + 1 < separator) {
                build_jobs = Some(value.clone());
                index += 2;
            } else {
                out.push(argument.clone());
                index += 1;
            }
            continue;
        }
        if let Some(value) = argument.strip_prefix("--build-jobs=") {
            build_jobs = Some(value.to_string());
            index += 1;
            continue;
        }
        out.push(argument.clone());
        index += 1;
        if llvm_cov_or_nextest_option_takes_value(argument) && index < separator {
            out.push(args[index].clone());
            index += 1;
        }
    }
    if let Some(build_jobs) = build_jobs {
        out.extend(["--build-jobs".to_string(), build_jobs]);
    }
    out.extend_from_slice(&args[separator..]);
    out
}

/// Enforce one admission scheduler for every cargo-ktstr nextest run.
///
/// User/repository nextest run-slot limits would otherwise strand ktstr cells
/// before they reach the machine-wide topology queue. Replace every
/// syntactically valid CLI spelling with the orchestrator's effectively
/// unbounded admission count. Malformed spellings remain intact so nextest,
/// rather than this wrapper, reports the user's argument error.
/// Raw llvm-cov build-job flags before its `nextest` subcommand remain Cargo
/// controls, and an inner `--` keeps the test-binary suffix opaque.
pub(crate) fn normalize_nextest_admission(sub_argv: &[&str], args: Vec<String>) -> Vec<String> {
    let args = if sub_argv == LLVM_COV_SUB_ARGV {
        normalize_raw_llvm_cov_build_jobs(args)
    } else {
        args
    };
    let region_start = nextest_region_start(sub_argv, &args);
    region_start.map_or(args.clone(), |start| normalize_nextest_region(args, start))
}

/// Variant for a complete `cargo nextest run ...` argv, used by verifier.
pub(crate) fn normalize_nextest_command_admission(args: Vec<String>) -> Vec<String> {
    if args.first().map(String::as_str) == Some("nextest")
        && args.get(1).map(String::as_str) == Some("run")
    {
        normalize_nextest_region(args, 2)
    } else {
        args
    }
}

/// Add ktstr's low-priority nextest defaults to every nextest-backed route.
///
/// The injector is parameterized so command-shape tests can pin the routing
/// contract without materializing the embedded config. Production passes the
/// content-addressed [`crate::nextest_config::inject`] implementation.
fn inject_nextest_tool_config_with(
    sub_argv: &[&str],
    args: Vec<String>,
    inject: impl FnOnce(Vec<String>) -> Result<Vec<String>, String>,
) -> Result<Vec<String>, String> {
    if cargo_sub_uses_nextest(sub_argv, &args) {
        inject(args)
    } else {
        Ok(args)
    }
}

/// Decide whether to inject `LLVM_PROFILE_FILE` for a given cargo
/// sub-invocation, returning the pattern to set or `None` to leave
/// the env untouched.
///
/// When the user invokes `cargo ktstr test` from inside a kernel
/// source tree, every link in the spawn chain (cargo-ktstr ->
/// cargo nextest -> test binary) inherits the shell's cwd. A
/// coverage-instrumented test binary would then drop
/// `default.profraw` directly in the kernel tree at exit because
/// the LLVM runtime defaults to writing in cwd when
/// `LLVM_PROFILE_FILE` is unset. Injecting a workspace-local
/// pattern here keeps the host's profraw next to the build output
/// regardless of cwd. `%p` (process id) and `%m` (binary hash) are
/// LLVM runtime expansions that keep parallel-test output files
/// distinct.
///
/// Returns `Some(pattern)` only when both:
///   - `sub_argv` selects the bare `nextest` path (the `test`
///     subcommand). The `coverage` path execs `cargo llvm-cov
///     nextest`, which manages `LLVM_PROFILE_FILE` itself for its
///     profraw collection pipeline; pre-setting the env here would
///     race that pipeline. The `llvm-cov` raw-passthrough path is
///     user-controlled by contract and must not be touched.
///   - `existing_env` is `None`. An operator who has already
///     exported `LLVM_PROFILE_FILE` keeps that value — we only set
///     when the env is currently absent, so an explicit override
///     stays authoritative. Operators who want a different
///     workspace-local target without touching `LLVM_PROFILE_FILE`
///     can set `LLVM_COV_TARGET_DIR` instead, which
///     [`ktstr::test_support::profraw_target_dir`] honors as the
///     highest-precedence entry in its cascade.
///
/// Pure with respect to its arguments — does no env read of its
/// own — so callers can drive the gate from a unit test by
/// supplying the env probe explicitly.
pub(crate) fn profraw_inject_for(
    sub_argv: &[&str],
    existing_env: Option<std::ffi::OsString>,
) -> Option<PathBuf> {
    if sub_argv != TEST_SUB_ARGV || existing_env.is_some() {
        return None;
    }
    let dir = ktstr::test_support::profraw_target_dir();
    Some(dir.join("default-%p-%m.profraw"))
}

/// Whether the exact resolved Cargo graph needs the `ktstr` `wprof` feature.
///
/// `None` metadata, or metadata without a resolve graph, is inconclusive and
/// therefore conservatively returns true. Once Cargo has supplied a resolve
/// graph, however, the node feature lists are authoritative: `wprof` is needed
/// iff at least one resolved package named `ktstr` has that feature active.
/// Looking at package manifest feature declarations would be wrong here -- it
/// says what *can* be enabled, not what this selected invocation built.
fn resolved_metadata_needs_wprof(metadata: Option<&cargo_metadata::Metadata>) -> bool {
    let Some(metadata) = metadata else {
        return true;
    };
    let Some(resolve) = metadata.resolve.as_ref() else {
        return true;
    };
    resolve.nodes.iter().any(|node| {
        node.features.iter().any(|feature| feature == "wprof")
            && metadata
                .packages
                .iter()
                .any(|package| package.id == node.id && package.name == "ktstr")
    })
}

/// Build-time env vars handing `cargo-ktstr`'s already-extracted
/// busybox / required wprof binaries to the child build, so the downstream
/// `ktstr` `build.rs` copies them into `$OUT_DIR` instead of
/// re-fetching + recompiling (see `install_prebuilt_blob` in
/// `build_helpers.rs`). `cargo-ktstr` exported `KTSTR_BUSYBOX_PATH` /
/// `KTSTR_WPROF_PATH` at startup (`bin/cargo_ktstr/blobs.rs`
/// `install_env`) pointing at the extracted blobs; this re-exports each
/// present path under the build-time `KTSTR_*_BIN` name `build.rs`
/// reads. A path var is present only when the embedded blob was
/// non-empty, so an absent var (cargo-ktstr built without that blob)
/// yields no pair and the child build falls back to its fetch path.
/// Busybox is unconditional. Wprof is handed over only when the exact resolved
/// metadata says a `ktstr` node has feature `wprof`; unavailable metadata keeps
/// the conservative historical handoff. That prevents a disabled optional
/// tool path from splitting the reusable Cargo artifact identity while keeping
/// metadata-failure behavior safe. Pure with respect to its args so a unit
/// test can drive every present/absent/metadata combination. `pub(crate)`
/// because the verifier
/// dispatcher's reserved warm-up (`verifier.rs`) applies the same
/// pairs so its pre-build and combined run share one build
/// fingerprint (`build.rs` watches `KTSTR_BUSYBOX_BIN` /
/// `KTSTR_WPROF_BIN` via `rerun-if-env-changed`).
pub(crate) fn prebuilt_blob_bin_envs(
    busybox_path: Option<std::ffi::OsString>,
    wprof_path: Option<std::ffi::OsString>,
    metadata: Option<&cargo_metadata::Metadata>,
) -> Vec<(&'static str, std::ffi::OsString)> {
    let mut pairs = Vec::new();
    if let Some(p) = busybox_path {
        pairs.push(("KTSTR_BUSYBOX_BIN", p));
    }
    if resolved_metadata_needs_wprof(metadata)
        && let Some(p) = wprof_path
    {
        pairs.push(("KTSTR_WPROF_BIN", p));
    }
    pairs
}

/// Shared runner for `cargo ktstr test`, `cargo ktstr coverage`, and
/// `cargo ktstr llvm-cov`.
///
/// All three subcommands share the same plumbing: resolve `--kernel`
/// to a flat `(label, kernel_dir)` set, propagate `--no-perf-mode`
/// via an env var, optionally prepend `--cargo-profile release`,
/// append the user's trailing args, and `cmd.status()` once. The
/// cargo subcommand name (`["nextest","run"]` vs `["llvm-cov",
/// "nextest"]` vs `["llvm-cov"]`) and the log / error-message
/// prefix are the only static differences.
///
/// Multi-kernel fan-out lives entirely in the test binary's
/// gauntlet expansion (`src/test_support/dispatch.rs`): when the
/// resolved set has more than one entry, the test binary's
/// `--list` handler prints `gauntlet/{name}/{preset}/
/// {kernel_label}` for every kernel and the `--exact` handler
/// strips the kernel suffix and re-exports `KTSTR_KERNEL` to that
/// kernel's directory before booting the VM. `cargo nextest`
/// already handles parallelism, retries, and `-E` filtering;
/// cargo-ktstr never spawns its own loop.
///
/// Empty `--kernel` (the default): no `KTSTR_KERNEL` /
/// `KTSTR_KERNEL_LIST` export — the test binary resolves its own
/// kernel via the existing `find_kernel` chain.
///
/// Single-entry `--kernel` (one Path / Version / CacheKey / Git, OR a
/// Range that expanded to exactly one release): export
/// `KTSTR_KERNEL` only. Test names stay backward-compatible — no
/// kernel suffix is appended in `--list` output.
///
/// Multi-entry `--kernel` (≥ 2 entries after expansion): export
/// `KTSTR_KERNEL_LIST` AND set `KTSTR_KERNEL` to the first entry so
/// downstream code that reads `KTSTR_KERNEL` directly (e.g. budget
/// listing in dispatch.rs that needs ANY kernel for vmlinux probe)
/// still gets a valid path. The test binary's `--list` / `--exact`
/// handlers prefer `KTSTR_KERNEL_LIST` when set.
///
/// Assemble the cargo `Command` argv + the flag-gated env vars that are
/// driven purely by the CLI flags — `--cargo-profile release` injection
/// (`--release`, the HARNESS profile), the nextest test profile
/// (`--nextest-profile` -> nextest `--profile`), the scheduler build
/// profile (`--profile` -> `KTSTR_SCHEDULER_PROFILE`), and the
/// `--no-perf-mode` / `--no-skip-mode` env passthroughs. Split out of
/// [`run_cargo_sub`] as a pure `Command` factory (no `std::env` reads, no
/// fs, no exec) so the argv ordering and the flag->argv/env coupling are
/// unit-testable via the stable [`Command::get_args`] /
/// [`Command::get_envs`] APIs; `run_cargo_sub` itself execs cargo and
/// can't be inspected directly.
///
/// `--cargo-profile release` is prepended BEFORE the user's trailing
/// nextest args so the profile selection applies to the whole invocation.
/// nextest reads `--cargo-profile` directly; `cargo llvm-cov nextest`
/// forwards it to its inner nextest invocation. Nextest-backed llvm-cov
/// routes additionally receive one global `--no-clean` immediately before
/// the FINAL `nextest`: the reserved `nextest-archive` warm command has
/// already run cargo-llvm-cov's exact clean_partial and built the identical
/// instrumented artifacts. Explicit user retention modes (`--no-clean`,
/// `--no-report`, and report-only `--no-run`) remain authoritative and receive
/// no wrapper lifecycle flag. For `cargo llvm-cov <non-nextest-sub>` (the raw
/// passthrough binding) the caller passes `release == false` and `profile` /
/// `nextest_profile` `None`, so nothing is injected and the argv remains
/// byte-for-byte user-controlled.
///
/// The `std::env`-reading envs (prebuilt-blob paths, `LLVM_PROFILE_FILE`
/// profraw injection) and the kernel-resolution envs are layered on by
/// `run_cargo_sub` AFTER this returns — they read process env / probe
/// the kernel cache and are not part of the pure flag->argv/env shape.
fn build_cargo_command(
    sub_argv: &[&str],
    release: bool,
    profile: Option<&str>,
    nextest_profile: Option<&str>,
    no_perf_mode: bool,
    no_skip_mode: bool,
    args: &[String],
) -> Command {
    let mut cmd = Command::new("cargo");
    let managed_llvm_cov = llvm_cov_managed_lifecycle(sub_argv, args);
    let command_args = if sub_argv == COVERAGE_SUB_ARGV {
        cmd.arg("llvm-cov");
        if managed_llvm_cov {
            cmd.arg("--no-clean");
        }
        cmd.arg("nextest");
        args
    } else if sub_argv == LLVM_COV_SUB_ARGV {
        if let Some(nextest_index) = llvm_cov_nextest_index(args) {
            cmd.arg("llvm-cov");
            cmd.args(&args[..nextest_index]);
            if managed_llvm_cov {
                cmd.arg("--no-clean");
            }
            cmd.arg("nextest");
            &args[nextest_index + 1..]
        } else {
            cmd.args(sub_argv);
            args
        }
    } else {
        cmd.args(sub_argv);
        args
    };
    if release {
        cmd.args(["--cargo-profile", "release"]);
    }
    // `--nextest-profile <NAME>` selects the NEXTEST test profile
    // (`.config/nextest.toml`); nextest's own flag for it is `--profile`,
    // and `cargo llvm-cov nextest` forwards it to its inner nextest.
    if let Some(np) = nextest_profile {
        cmd.args(["--profile", np]);
    }
    cmd.args(command_args);
    if no_perf_mode {
        cmd.env(ktstr::KTSTR_NO_PERF_MODE_ENV, "1");
    }
    if no_skip_mode {
        cmd.env(ktstr::KTSTR_NO_SKIP_MODE_ENV, "1");
    }
    // `--profile <NAME>` selects the scheduler-under-test's cargo BUILD
    // profile: `build_and_find_binary` reads `KTSTR_SCHEDULER_PROFILE` and
    // passes `cargo build -p <scheduler> --profile <name>`. Absent -> that
    // build defaults the scheduler to `release`. This is independent of
    // the harness `--release` (`--cargo-profile release`) above.
    if let Some(p) = profile {
        cmd.env(ktstr::KTSTR_SCHEDULER_PROFILE_ENV, p);
    }
    cmd
}

/// Execute cargo-llvm-cov's deprecated report-only `--no-run` mode without
/// entering ktstr's test orchestration lifecycle.
///
/// cargo-llvm-cov rewrites `--no-run` to its `report` subcommand before it
/// considers the originally selected test runner. It therefore does not read a
/// nextest archive, enumerate tests, build schedulers, or boot a VM. Running
/// this command directly preserves that contract: no KVM/tool preflight,
/// Cargo metadata query, archive snapshot, kernel resolution, BTF anchor,
/// shared-memory cleanup guard, or result-artifact scan can precede the
/// upstream report operation. The signal phase still crosses immediately
/// before spawn so the child is group-owned and reaped under interruption.
#[allow(clippy::too_many_arguments)]
fn run_llvm_cov_report_only(
    sub_argv: &[&str],
    release: bool,
    profile: Option<&str>,
    nextest_profile: Option<&str>,
    no_perf_mode: bool,
    no_skip_mode: bool,
    args: &[String],
) -> Result<(), String> {
    let command = build_cargo_command(
        sub_argv,
        release,
        profile,
        nextest_profile,
        no_perf_mode,
        no_skip_mode,
        args,
    );
    tracing::debug!(
        program = ?command.get_program(),
        argv = ?command.get_args().collect::<Vec<_>>(),
        "running report-only cargo-llvm-cov passthrough",
    );
    crate::interrupt::enter_cleanup_phase()
        .map_err(|error| format!("cargo ktstr: enter cleanup phase: {error}"))?;
    let status = crate::interrupt::run_status(command)
        .map_err(|error| format!("spawn cargo {}: {error}", sub_argv.join(" ")))?;
    if status.success() {
        Ok(())
    } else {
        Err(format!(
            "cargo {} exited with {}",
            sub_argv.join(" "),
            status
                .code()
                .map_or("signal".to_string(), |code| code.to_string()),
        ))
    }
}

/// Consume the resolved `--kernel` set, bailing if a non-empty `kernel`
/// vec resolved to nothing.
///
/// `resolve_kernel_set` skips arguments that trim to empty, so `--kernel
/// ""` or `--kernel "  "` produce `Ok(vec![])` without ever entering the
/// per-spec resolve branch. An empty input `kernel` (flag omitted)
/// likewise yields `Ok(vec![])` — but that is the auto-discovery path,
/// not an error. This helper distinguishes the two: only an
/// all-whitespace `--kernel` (non-empty input, empty resolution) is an
/// operator error worth surfacing; an omitted flag returns `Ok(vec![])`
/// so the caller falls through to the `find_kernel` chain.
///
/// Split out of [`run_cargo_sub`] so the bail diagnostic is unit-testable
/// without the exec/fs tail — all-whitespace specs reach `resolve_kernel_set`
/// but `filter_map` drops them before any `KernelId::parse`, so this path
/// does no network/build I/O.
fn kernel_set_or_bail(
    kernel: &[String],
    include_eol: bool,
) -> Result<Vec<(String, PathBuf)>, String> {
    if kernel.is_empty() {
        return Ok(Vec::new());
    }
    let resolved = resolve_kernel_set(kernel, include_eol)?;
    if resolved.is_empty() {
        // `resolve_kernel_set` skips arguments that trim to
        // empty, so `--kernel ""` or `--kernel "  "` reach
        // here without ever entering the per-spec resolve
        // branch. Bail with an actionable error rather than
        // letting the child reach for `find_kernel` as if
        // `--kernel` had never been passed (which would mask
        // the operator's intent).
        return Err(
            "--kernel: every supplied value parsed to empty / whitespace; \
             omit the flag for auto-discovery, or supply a kernel \
             identifier"
                .to_string(),
        );
    }
    Ok(resolved)
}

const NEXTEST_ARCHIVE_DROP_VALUE_OPTIONS: &[&str] = &[
    "-j",
    "--test-threads",
    "--retries",
    "--flaky-result",
    "--max-fail",
    "-R",
    "--rerun",
    "--run-id",
    "--debugger",
    "--tracer",
    "--stress-count",
    "--stress-duration",
    "--status-level",
    "--final-status-level",
    "--success-output",
    "--failure-output",
    "--show-progress",
    "--max-progress-running",
    "--message-format",
    "--message-format-version",
    "--run-ignored",
    "--partition",
    "--platform-filter",
    "--no-tests",
    "-E",
    "--filterset",
    "--filter-expr",
    "--cargo-message-format",
    "--archive-file",
    "--archive-format",
    "--zstd-level",
    "--extract-to",
    "--cargo-metadata",
    "--workspace-remap",
    "--binaries-metadata",
    "--target-dir-remap",
    "--build-dir-remap",
];

const NEXTEST_ARCHIVE_DROP_FLAGS: &[&str] = &[
    "--fail-fast",
    "--ff",
    "--no-fail-fast",
    "--nff",
    "--no-capture",
    "--capture",
    "--hide-progress-bar",
    "--no-output-indent",
    "--no-input-handler",
    "--ignore-default-filter",
    "--extract-overwrite",
    "--persist-extract-tempdir",
    "--ignore-run-fail",
    "--no-run",
];

const LLVM_COV_ARCHIVE_DROP_REPORT_VALUE_OPTIONS: &[&str] = &[
    "--output-path",
    "--output-dir",
    "--failure-mode",
    "--ignore-filename-regex",
    "--fail-under-functions",
    "--fail-under-lines",
    "--fail-under-file-lines",
    "--fail-under-regions",
    "--fail-uncovered-lines",
    "--fail-uncovered-regions",
    "--fail-uncovered-functions",
    "--nextest-archive-file",
];

const LLVM_COV_ARCHIVE_DROP_REPORT_FLAGS: &[&str] = &[
    "--json",
    "--lcov",
    "--cobertura",
    "--codecov",
    "--text",
    "--html",
    "--open",
    "--summary-only",
    "--no-default-ignore-filename-regex",
    "--disable-default-ignore-filename-regex",
    "--show-instantiations",
    "--hide-instantiations",
    "--show-missing-lines",
    "--include-build-script",
    "--skip-functions",
];

const LLVM_COV_ARCHIVE_PRESERVE_FLAGS: &[&str] = &[
    "--all",
    "--workspace",
    "--lib",
    "--bins",
    "--examples",
    "--tests",
    "--benches",
    "--all-targets",
    "--all-features",
    "--no-default-features",
    "--release",
    "--frozen",
    "--locked",
    "--offline",
    "--ignore-rust-version",
    "--keep-going",
    "--verbose",
    "-v",
    "--quiet",
    "-q",
    "--cargo-quiet",
    "--cargo-verbose",
    "--future-incompat-report",
    "--override-version-check",
    "--no-pager",
    "--unit-graph",
    "--timings",
    "--no-clean",
    "--branch",
    "--mcdc",
    "--doctests",
    "--coverage-target-only",
    "--coverage-host-only",
    "--include-ffi",
    "--no-rustc-wrapper",
    "--no-cfg-coverage",
    "--no-cfg-coverage-nightly",
    "--remap-path-prefix",
];

fn option_matches_bool_flag(argument: &str, option: &str) -> bool {
    argument == option
        || argument
            .strip_prefix(option)
            .is_some_and(|suffix| suffix.starts_with('='))
}

fn nextest_archive_drops_flag(argument: &str) -> bool {
    NEXTEST_ARCHIVE_DROP_FLAGS
        .iter()
        .any(|option| option_matches_bool_flag(argument, option))
}

fn nextest_archive_drops_joined_option(argument: &str) -> bool {
    NEXTEST_ARCHIVE_DROP_VALUE_OPTIONS.iter().any(|option| {
        let short = option.starts_with('-') && !option.starts_with("--");
        argument
            .strip_prefix(option)
            .is_some_and(|suffix| suffix.starts_with('=') || (short && !suffix.is_empty()))
    })
}

fn llvm_cov_archive_drops_joined_report_option(argument: &str) -> bool {
    LLVM_COV_ARCHIVE_DROP_REPORT_VALUE_OPTIONS
        .iter()
        .any(|option| {
            argument
                .strip_prefix(option)
                .is_some_and(|suffix| suffix.starts_with('='))
        })
}

fn llvm_cov_or_nextest_option_takes_value(argument: &str) -> bool {
    LLVM_COV_GLOBAL_VALUE_OPTIONS.contains(&argument)
        || LLVM_COV_NEXTEST_VALUE_OPTIONS.contains(&argument)
        || NEXTEST_ARCHIVE_DROP_VALUE_OPTIONS.contains(&argument)
}

fn llvm_cov_or_nextest_joined_value_option(argument: &str) -> bool {
    if argument
        .strip_prefix("--timings=")
        .is_some_and(|value| !value.is_empty())
    {
        return true;
    }
    LLVM_COV_GLOBAL_VALUE_OPTIONS
        .iter()
        .chain(LLVM_COV_NEXTEST_VALUE_OPTIONS.iter())
        .any(|option| {
            let short = option.starts_with('-') && !option.starts_with("--");
            argument
                .strip_prefix(option)
                .is_some_and(|suffix| suffix.starts_with('=') || (short && !suffix.is_empty()))
        })
}

/// Project one llvm-cov nextest run argv onto nextest-archive's build surface.
///
/// Archive filtering happens only after nextest builds the complete
/// Cargo-selected binary list. Removing run filters/positional test filters and
/// adding `none()` therefore preserves the exact build while keeping the
/// archive itself metadata-sized. cargo-llvm-cov owns the clean_partial call
/// for this supported subcommand, so no private cleanup behavior is duplicated
/// here.
fn nextest_build_surface_args(
    sub_argv: &[&str],
    args: &[String],
) -> Result<(Vec<String>, Vec<String>), String> {
    let raw_nextest_index = (sub_argv == LLVM_COV_SUB_ARGV)
        .then(|| llvm_cov_nextest_index(args))
        .flatten();
    let separator = args
        .iter()
        .position(|argument| argument == "--")
        .unwrap_or(args.len());
    let mut globals = Vec::new();
    let mut projected = Vec::with_capacity(separator + 8);
    let mut translated_no_report = false;
    let mut raw_build_jobs = None;
    let mut index = 0;
    if let Some(nextest_index) = raw_nextest_index {
        while index < nextest_index {
            let argument = &args[index];
            if argument == "--no-report" {
                translated_no_report = true;
                index += 1;
                continue;
            }
            if nextest_archive_drops_flag(argument) {
                index += 1;
                continue;
            }
            if LLVM_COV_ARCHIVE_DROP_REPORT_VALUE_OPTIONS.contains(&argument.as_str()) {
                index += 2;
                continue;
            }
            if LLVM_COV_ARCHIVE_DROP_REPORT_FLAGS.contains(&argument.as_str())
                || llvm_cov_archive_drops_joined_report_option(argument)
            {
                index += 1;
                continue;
            }
            if matches!(argument.as_str(), "-j" | "--jobs") {
                if let Some(value) = args.get(index + 1) {
                    raw_build_jobs = Some(value.clone());
                    index += 2;
                } else {
                    globals.push(argument.clone());
                    index += 1;
                }
                continue;
            }
            if let Some(value) = argument.strip_prefix("--jobs=").or_else(|| {
                argument
                    .strip_prefix("-j")
                    .filter(|value| !value.is_empty())
                    .map(|value| value.strip_prefix('=').unwrap_or(value))
            }) {
                raw_build_jobs = Some(value.to_string());
                index += 1;
                continue;
            }
            globals.push(argument.clone());
            index += 1;
            if LLVM_COV_GLOBAL_VALUE_OPTIONS.contains(&argument.as_str()) && index < nextest_index {
                globals.push(args[index].clone());
                index += 1;
            }
        }
        index += 1; // replace the raw `nextest` subcommand
    }
    while index < separator {
        let argument = &args[index];
        if argument == "--no-report" {
            // `nextest-archive` always performs a build and cargo-llvm-cov
            // warns that --no-report has no effect there. With
            // CARGO_LLVM_COV_DENY_WARNINGS that warning is fatal. Preserve
            // the user's retain/merge intent with the underlying lifecycle
            // flag that --no-report implies.
            translated_no_report = true;
            index += 1;
            continue;
        }
        if NEXTEST_ARCHIVE_DROP_VALUE_OPTIONS.contains(&argument.as_str())
            || LLVM_COV_ARCHIVE_DROP_REPORT_VALUE_OPTIONS.contains(&argument.as_str())
        {
            index += 2;
            continue;
        }
        if nextest_archive_drops_flag(argument)
            || nextest_archive_drops_joined_option(argument)
            || LLVM_COV_ARCHIVE_DROP_REPORT_FLAGS.contains(&argument.as_str())
            || llvm_cov_archive_drops_joined_report_option(argument)
        {
            index += 1;
            continue;
        }
        if !argument.starts_with('-') {
            // A nextest run positional name filter cannot change the Cargo
            // build and is not accepted by `nextest archive`.
            index += 1;
            continue;
        }
        if !LLVM_COV_ARCHIVE_PRESERVE_FLAGS.contains(&argument.as_str())
            && !llvm_cov_or_nextest_option_takes_value(argument)
            && !llvm_cov_or_nextest_joined_value_option(argument)
        {
            return Err(format!(
                "cargo ktstr: cannot safely project unknown post-nextest option \
                 {argument:?} onto cargo-llvm-cov nextest-archive"
            ));
        }
        projected.push(argument.clone());
        index += 1;
        if llvm_cov_or_nextest_option_takes_value(argument)
            && index < separator
            && Some(index) != raw_nextest_index
        {
            projected.push(args[index].clone());
            index += 1;
        }
    }
    if translated_no_report
        && !globals.iter().any(|argument| argument == "--no-clean")
        && !projected.iter().any(|argument| argument == "--no-clean")
    {
        globals.push("--no-clean".to_string());
    }
    if let Some(build_jobs) = raw_build_jobs {
        projected.extend(["--build-jobs".to_string(), build_jobs]);
    }
    Ok((globals, projected))
}

/// Project a complete `cargo nextest run ...` argv onto the generic cached
/// build surface. Verifier recursion uses this entry point so it cannot drift
/// from ordinary `cargo ktstr test` filtering of run-only arguments.
pub(crate) fn nextest_command_build_surface(args: &[String]) -> Result<Vec<String>, String> {
    if args.first().map(String::as_str) != Some("nextest")
        || args.get(1).map(String::as_str) != Some("run")
    {
        return Err("cached nextest command must begin with `nextest run`".to_string());
    }
    direct_nextest_build_surface_args(TEST_SUB_ARGV, &args[2..])
}

/// Project a complete `cargo nextest run ...` argv onto the run-only surface
/// accepted beside nextest reuse-build metadata.
pub(crate) fn nextest_command_run_surface(args: &[String]) -> Result<Vec<String>, String> {
    if args.first().map(String::as_str) != Some("nextest")
        || args.get(1).map(String::as_str) != Some("run")
    {
        return Err("cached nextest command must begin with `nextest run`".to_string());
    }
    let mut out = vec!["nextest".to_string(), "run".to_string()];
    out.extend(cached_nextest_run_args(TEST_SUB_ARGV, &args[2..]));
    Ok(out)
}

/// Add nextest reuse-build metadata to a complete run command, ahead of the
/// opaque test-binary suffix.
pub(crate) fn inject_nextest_command_reuse_args(
    args: Vec<String>,
    reuse: &[String],
) -> Result<Vec<String>, String> {
    let mut args = nextest_command_run_surface(&args)?;
    let insertion = args
        .iter()
        .position(|argument| argument == "--")
        .unwrap_or(args.len());
    args.splice(insertion..insertion, reuse.iter().cloned());
    Ok(args)
}

fn llvm_cov_nextest_archive_args(
    sub_argv: &[&str],
    args: &[String],
    archive_path: &std::path::Path,
) -> Result<(Vec<String>, Vec<String>), String> {
    let (globals, mut projected) = nextest_build_surface_args(sub_argv, args)?;
    projected.extend([
        "--archive-file".to_string(),
        archive_path.to_string_lossy().into_owned(),
        "--zstd-level=-7".to_string(),
        "-E".to_string(),
        "none()".to_string(),
        "--cargo-message-format=json-render-diagnostics".to_string(),
    ]);
    Ok((globals, projected))
}

#[allow(clippy::too_many_arguments)]
fn build_llvm_cov_archive_warm_command(
    sub_argv: &[&str],
    release: bool,
    profile: Option<&str>,
    nextest_profile: Option<&str>,
    no_perf_mode: bool,
    no_skip_mode: bool,
    args: &[String],
    archive_path: &std::path::Path,
) -> Result<Command, String> {
    let (globals, archive_args) = llvm_cov_nextest_archive_args(sub_argv, args, archive_path)?;
    let mut command = Command::new("cargo");
    command.arg("llvm-cov");
    command.args(globals);
    command.arg("nextest-archive");
    if release {
        command.args(["--cargo-profile", "release"]);
    }
    if let Some(nextest_profile) = nextest_profile {
        command.args(["--profile", nextest_profile]);
    }
    command.args(archive_args);
    if no_perf_mode {
        command.env(ktstr::KTSTR_NO_PERF_MODE_ENV, "1");
    }
    if no_skip_mode {
        command.env(ktstr::KTSTR_NO_SKIP_MODE_ENV, "1");
    }
    if let Some(profile) = profile {
        command.env(ktstr::KTSTR_SCHEDULER_PROFILE_ENV, profile);
    }
    Ok(command)
}

/// Build flavor stored in the generic nextest artifact cache.
///
/// Verifier recursion is a plain nextest consumer, exactly like `test`.
/// Coverage differs only in the instrumentation environment used by the cold
/// producer and in the final cargo-llvm-cov report wrapper; both flavors use
/// the same binary-only listing, artifact-tree capture, and reuse-build args.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CachedNextestMode {
    Plain,
    Coverage,
}

impl CachedNextestMode {
    pub(crate) fn identity_label(self) -> &'static str {
        match self {
            Self::Plain => "nextest",
            Self::Coverage => "llvm-cov-nextest",
        }
    }
}

const KTSTR_BUILD_BTF_ENV: &str = "KTSTR_BUILD_BTF";

fn apply_command_envs(command: &mut Command, environment: &[(OsString, OsString)]) {
    for (name, value) in environment {
        command.env(name, value);
    }
}

/// Remove test/runtime coordinates from one immutable Cargo producer.
///
/// Inspect inherited and explicitly overlaid values. The latter matters when
/// the parent assembled the command before selecting the stable producer;
/// `env_remove` must win over both sources. Coverage's LLVM_PROFILE_FILE is
/// deliberately not part of this generic ktstr-runtime classification.
fn sanitize_cached_cargo_build_child_environment(command: &mut Command) {
    let mut names = std::env::vars_os()
        .map(|(name, _)| name)
        .chain(command.get_envs().map(|(name, _)| name.to_os_string()))
        .collect::<Vec<_>>();
    names.sort();
    names.dedup();
    for name in names {
        if crate::verifier::cached_cargo_build_environment_is_runtime(&name) {
            command.env_remove(name);
        }
    }
}

/// Decode one shell-escaped value emitted by `cargo llvm-cov show-env`'s
/// default `KEY=value` format.
///
/// cargo-llvm-cov uses `shell_escape::escape`, which emits a single POSIX shell
/// word (possibly assembled from adjacent quoted and unquoted spans). Parsing
/// that word directly avoids executing a shell just to transfer environment
/// variables into the binary-only nextest listing.
fn decode_shell_word(value: &str) -> Result<String, String> {
    #[derive(Clone, Copy)]
    enum Quote {
        Unquoted,
        Single,
        Double,
    }

    let mut quote = Quote::Unquoted;
    let mut escaped = false;
    let mut decoded = String::with_capacity(value.len());
    for character in value.chars() {
        if escaped {
            decoded.push(character);
            escaped = false;
            continue;
        }
        match quote {
            Quote::Unquoted => match character {
                '\'' => quote = Quote::Single,
                '"' => quote = Quote::Double,
                '\\' => escaped = true,
                character if character.is_whitespace() => {
                    return Err("unquoted whitespace in escaped shell word".to_string());
                }
                _ => decoded.push(character),
            },
            Quote::Single => {
                if character == '\'' {
                    quote = Quote::Unquoted;
                } else {
                    decoded.push(character);
                }
            }
            Quote::Double => match character {
                '"' => quote = Quote::Unquoted,
                '\\' => escaped = true,
                _ => decoded.push(character),
            },
        }
    }
    if escaped || !matches!(quote, Quote::Unquoted) {
        return Err("unterminated escape or quote in escaped shell word".to_string());
    }
    Ok(decoded)
}

fn parse_llvm_cov_show_env(stdout: &[u8]) -> Result<Vec<(OsString, OsString)>, String> {
    let text = std::str::from_utf8(stdout)
        .map_err(|error| format!("cargo llvm-cov show-env emitted non-UTF-8 output: {error}"))?;
    let mut environment = Vec::new();
    for (line_index, line) in text.lines().enumerate() {
        if line.is_empty() {
            continue;
        }
        let (name, value) = line.split_once('=').ok_or_else(|| {
            format!(
                "cargo llvm-cov show-env line {} is not KEY=value: {line:?}",
                line_index + 1,
            )
        })?;
        if name.is_empty()
            || !name.bytes().enumerate().all(|(index, byte)| {
                byte == b'_'
                    || byte.is_ascii_alphanumeric() && (index > 0 || !byte.is_ascii_digit())
            })
        {
            return Err(format!(
                "cargo llvm-cov show-env line {} has an invalid environment name: {name:?}",
                line_index + 1,
            ));
        }
        environment.push((
            OsString::from(name),
            OsString::from(decode_shell_word(value).map_err(|error| {
                format!(
                    "decode cargo llvm-cov show-env line {} ({name}): {error}",
                    line_index + 1,
                )
            })?),
        ));
    }
    if environment.is_empty() {
        return Err("cargo llvm-cov show-env emitted no environment variables".to_string());
    }
    Ok(environment)
}

const LLVM_COV_SHOW_ENV_VALUE_OPTIONS: &[&str] = &["--target", "--manifest-path", "--dep-coverage"];

const LLVM_COV_SHOW_ENV_FLAGS: &[&str] = &[
    "--branch",
    "--mcdc",
    "--doctests",
    "--coverage-target-only",
    "--coverage-host-only",
    "--include-ffi",
    "--no-rustc-wrapper",
    "--no-cfg-coverage",
    "--no-cfg-coverage-nightly",
    "--remap-path-prefix",
];

fn llvm_cov_show_env_args(sub_argv: &[&str], args: &[String]) -> Vec<String> {
    let raw_nextest_index = (sub_argv == LLVM_COV_SUB_ARGV)
        .then(|| llvm_cov_nextest_index(args))
        .flatten();
    let separator = args
        .iter()
        .position(|argument| argument == "--")
        .unwrap_or(args.len());
    let mut out = Vec::new();
    let mut index = 0;
    while index < separator {
        if Some(index) == raw_nextest_index {
            index += 1;
            continue;
        }
        let argument = &args[index];
        if LLVM_COV_SHOW_ENV_VALUE_OPTIONS.contains(&argument.as_str()) {
            out.push(argument.clone());
            index += 1;
            if index < separator {
                out.push(args[index].clone());
            }
            index += 1;
            continue;
        }
        let joined_value = LLVM_COV_SHOW_ENV_VALUE_OPTIONS.iter().any(|option| {
            let short = option.starts_with('-') && !option.starts_with("--");
            argument
                .strip_prefix(option)
                .is_some_and(|suffix| suffix.starts_with('=') || short && !suffix.is_empty())
        });
        if joined_value || LLVM_COV_SHOW_ENV_FLAGS.contains(&argument.as_str()) {
            out.push(argument.clone());
        }
        index += 1;
    }
    out
}

fn set_or_replace_environment(
    environment: &mut Vec<(OsString, OsString)>,
    name: &str,
    value: impl Into<OsString>,
) {
    let value = value.into();
    if let Some((_, current)) = environment
        .iter_mut()
        .find(|(candidate, _)| candidate == OsStr::new(name))
    {
        *current = value;
    } else {
        environment.push((OsString::from(name), value));
    }
}

fn stable_cargo_producer_environment(
    environment: &[(OsString, OsString)],
) -> Vec<(OsString, OsString)> {
    let mut environment = environment
        .iter()
        .filter(|(name, _)| !crate::verifier::cached_cargo_build_environment_is_runtime(name))
        .cloned()
        .collect::<Vec<_>>();
    set_or_replace_environment(&mut environment, "CARGO_INCREMENTAL", "0");
    environment
}

/// Retarget cargo-llvm-cov's complete output environment as one unit.
///
/// cargo-llvm-cov discovers raw profiles by scanning the immediate
/// `CARGO_LLVM_COV_TARGET_DIR` for `*.profraw`. `LLVM_PROFILE_FILE` therefore
/// has to name a file directly below that same directory: pointing it at a
/// private child silently lets every test write valid profiles that the later
/// `report` subcommand cannot see.
fn retarget_llvm_cov_environment(
    environment: &mut Vec<(OsString, OsString)>,
    output_target: &Path,
    output_build: &Path,
) {
    let profile_name = environment
        .iter()
        .find(|(name, _)| name == OsStr::new("LLVM_PROFILE_FILE"))
        .and_then(|(_, value)| Path::new(value).file_name())
        .map_or_else(
            || OsString::from("ktstr-%p-%m.profraw"),
            OsStr::to_os_string,
        );
    set_or_replace_environment(
        environment,
        "CARGO_LLVM_COV_TARGET_DIR",
        output_target.as_os_str(),
    );
    set_or_replace_environment(
        environment,
        "CARGO_LLVM_COV_BUILD_DIR",
        output_build.as_os_str(),
    );
    set_or_replace_environment(
        environment,
        "LLVM_PROFILE_FILE",
        output_target.join(profile_name).into_os_string(),
    );
}

fn llvm_cov_build_environment(
    stable_workspace: &Path,
    output_target: &Path,
    output_build: &Path,
    show_env_args: &[String],
    producer_environment: &[(OsString, OsString)],
) -> Result<Vec<(OsString, OsString)>, String> {
    let mut command = Command::new("cargo");
    command
        .args(["llvm-cov", "show-env"])
        .args(show_env_args)
        .current_dir(stable_workspace)
        .env("CARGO_LLVM_COV_TARGET_DIR", output_target)
        .env("CARGO_LLVM_COV_BUILD_DIR", output_build);
    apply_command_envs(&mut command, producer_environment);
    sanitize_cached_cargo_build_child_environment(&mut command);
    let output = crate::interrupt::run_output(command)
        .map_err(|error| format!("run cargo llvm-cov show-env: {error}"))?;
    if !output.status.success() {
        return Err(format!(
            "cargo llvm-cov show-env failed with {}: {}",
            output.status,
            String::from_utf8_lossy(&output.stderr).trim(),
        ));
    }
    let mut environment = parse_llvm_cov_show_env(&output.stdout)?;
    retarget_llvm_cov_environment(&mut environment, output_target, output_build);
    Ok(environment)
}

fn force_nextest_output(args: &[String], output_target: &Path) -> Vec<String> {
    let mut out = Vec::with_capacity(args.len() + 2);
    let mut index = 0;
    while index < args.len() {
        let argument = &args[index];
        if argument == "--target-dir" {
            index += 2;
            continue;
        }
        if argument.starts_with("--target-dir=") {
            index += 1;
            continue;
        }
        out.push(argument.clone());
        index += 1;
    }
    out.extend([
        "--target-dir".to_string(),
        output_target.display().to_string(),
    ]);
    out
}

/// Move nextest's independent run store alongside the writable Cargo output.
///
/// `--target-dir` only controls Cargo. Nextest resolves `[store].dir` against
/// the workspace root and eagerly creates it whenever the selected profile has
/// JUnit enabled, including for `nextest list`. Cached producers execute from
/// ktstr's immutable stable source, so leaving the default `target/nextest`
/// there turns a read-only source snapshot into an output directory and fails
/// before Cargo can build anything.
///
/// Materialize the exact selected repository config with only `store.dir`
/// replaced by an absolute path under this producer/materialization's private
/// target directory. An explicit `--config-file` keeps all of its settings;
/// otherwise the workspace default is copied when present. Supplying the
/// generated file through nextest's highest-priority `--config-file` surface
/// also relocates an explicitly configured relative store rather than relying
/// on lower-priority tool config.
pub(crate) fn remap_nextest_store_output(
    args: &[String],
    workspace_root: &Path,
    invocation_dir: &Path,
    output_target: &Path,
) -> Result<Vec<String>, String> {
    let separator = args
        .iter()
        .position(|argument| argument == "--")
        .unwrap_or(args.len());
    let mut explicit = None;
    let mut explicit_range = None;
    let mut index = 0;
    while index < separator {
        let argument = &args[index];
        if argument == "--config-file" {
            let Some(value) = args.get(index + 1).filter(|_| index + 1 < separator) else {
                // Preserve malformed argv for nextest's own diagnostic.
                return Ok(args.to_vec());
            };
            if explicit.is_some() {
                // Clap owns duplicate-option semantics; do not accidentally
                // turn malformed input into a successful invocation.
                return Ok(args.to_vec());
            }
            explicit = Some(value.as_str());
            explicit_range = Some(index..index + 2);
            index += 2;
            continue;
        }
        if let Some(value) = argument.strip_prefix("--config-file=") {
            if value.is_empty() || explicit.is_some() {
                return Ok(args.to_vec());
            }
            explicit = Some(value);
            explicit_range = Some(index..index + 1);
        }
        index += 1;
    }

    let source = explicit.map_or_else(
        || workspace_root.join(".config/nextest.toml"),
        |value| {
            let path = PathBuf::from(value);
            if path.is_absolute() {
                path
            } else {
                invocation_dir.join(path)
            }
        },
    );
    let mut config = if source.is_file() {
        let text = std::fs::read_to_string(&source).map_err(|error| {
            format!(
                "cargo ktstr: read nextest config {} for output remapping: {error}",
                source.display(),
            )
        })?;
        text.parse::<toml::Table>().map_err(|error| {
            format!(
                "cargo ktstr: parse nextest config {} for output remapping: {error}",
                source.display(),
            )
        })?
    } else if explicit.is_some() {
        // Preserve nextest's native missing-file error and exact argv.
        return Ok(args.to_vec());
    } else {
        toml::Table::new()
    };

    let store = config
        .entry("store")
        .or_insert_with(|| toml::Value::Table(toml::Table::new()));
    let store = store.as_table_mut().ok_or_else(|| {
        format!(
            "cargo ktstr: nextest config {} has non-table `store`",
            source.display(),
        )
    })?;
    let output_target = if output_target.is_absolute() {
        output_target.to_path_buf()
    } else {
        invocation_dir.join(output_target)
    };
    let store_dir = output_target.join("nextest");
    let store_dir = store_dir.to_str().ok_or_else(|| {
        format!(
            "cargo ktstr: nextest store path is not valid UTF-8: {}",
            store_dir.display(),
        )
    })?;
    store.insert(
        "dir".to_string(),
        toml::Value::String(store_dir.to_string()),
    );

    let overlay_dir = output_target.join(".ktstr-nextest");
    std::fs::create_dir_all(&overlay_dir).map_err(|error| {
        format!(
            "cargo ktstr: create nextest output-config directory {}: {error}",
            overlay_dir.display(),
        )
    })?;
    let rendered = toml::to_string(&config)
        .map_err(|error| format!("cargo ktstr: encode remapped nextest config: {error}"))?;
    let overlay = publish_nextest_output_config(&overlay_dir, rendered.as_bytes())?;

    let replacement = format!("--config-file={}", overlay.display());
    let mut out = args.to_vec();
    if let Some(range) = explicit_range {
        out.splice(range, [replacement]);
    } else {
        out.insert(separator, replacement);
    }
    Ok(out)
}

fn publish_nextest_output_config(directory: &Path, bytes: &[u8]) -> Result<PathBuf, String> {
    use std::hash::{BuildHasher as _, Hasher as _};
    use std::io::Write as _;

    let mut hasher = ahash::RandomState::with_seeds(0, 0, 0, 0).build_hasher();
    hasher.write_u64(b"ktstr-nextest-output-config".len() as u64);
    hasher.write(b"ktstr-nextest-output-config");
    hasher.write_u64(bytes.len() as u64);
    hasher.write(bytes);
    let target = directory.join(format!("config-{:016x}.toml", hasher.finish()));
    if std::fs::read(&target).is_ok_and(|current| current == bytes) {
        return Ok(target);
    }

    let mut staging = tempfile::Builder::new()
        .prefix(".config-staging-")
        .tempfile_in(directory)
        .map_err(|error| {
            format!(
                "cargo ktstr: create nextest output-config staging file in {}: {error}",
                directory.display(),
            )
        })?;
    staging
        .write_all(bytes)
        .and_then(|()| staging.flush())
        .map_err(|error| {
            format!(
                "cargo ktstr: write nextest output-config staging file in {}: {error}",
                directory.display(),
            )
        })?;
    match staging.persist_noclobber(&target) {
        Ok(_) => Ok(target),
        Err(_error) if std::fs::read(&target).is_ok_and(|current| current == bytes) => Ok(target),
        Err(error) => Err(format!(
            "cargo ktstr: publish nextest output config {}: {}",
            target.display(),
            error.error,
        )),
    }
}

fn remap_workspace_argument_path(
    value: &str,
    original_workspace: &Path,
    invocation_dir: &Path,
    stable_workspace: &Path,
) -> String {
    let path = Path::new(value);
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        invocation_dir.join(path)
    };
    absolute
        .strip_prefix(original_workspace)
        .map_or(absolute.clone(), |relative| stable_workspace.join(relative))
        .display()
        .to_string()
}

fn remap_cached_build_paths(
    args: &[String],
    original_workspace: &Path,
    invocation_dir: &Path,
    stable_workspace: &Path,
) -> Vec<String> {
    let mut out = Vec::with_capacity(args.len());
    let mut index = 0;
    while index < args.len() {
        let argument = &args[index];
        if matches!(argument.as_str(), "--manifest-path" | "--target") {
            out.push(argument.clone());
            index += 1;
            if let Some(value) = args.get(index) {
                let path_like_target = argument != "--target"
                    || value.ends_with(".json")
                    || value.contains('/')
                    || value.contains('\\');
                out.push(if path_like_target {
                    remap_workspace_argument_path(
                        value,
                        original_workspace,
                        invocation_dir,
                        stable_workspace,
                    )
                } else {
                    value.clone()
                });
            }
            index += 1;
            continue;
        }
        if let Some(value) = argument.strip_prefix("--manifest-path=") {
            out.push(format!(
                "--manifest-path={}",
                remap_workspace_argument_path(
                    value,
                    original_workspace,
                    invocation_dir,
                    stable_workspace,
                )
            ));
        } else if let Some(value) = argument.strip_prefix("--target=") {
            let path_like = value.ends_with(".json") || value.contains('/') || value.contains('\\');
            if path_like {
                out.push(format!(
                    "--target={}",
                    remap_workspace_argument_path(
                        value,
                        original_workspace,
                        invocation_dir,
                        stable_workspace,
                    )
                ));
            } else {
                out.push(argument.clone());
            }
        } else {
            out.push(argument.clone());
        }
        index += 1;
    }
    out
}

fn inject_nextest_reuse_args(
    sub_argv: &[&str],
    mut args: Vec<String>,
    reuse: &[String],
) -> Vec<String> {
    let Some(region_start) = nextest_region_start(sub_argv, &args) else {
        return args;
    };
    let insertion = args[region_start..]
        .iter()
        .position(|argument| argument == "--")
        .map_or(args.len(), |offset| region_start + offset);
    args.splice(insertion..insertion, reuse.iter().cloned());
    args
}

const LLVM_COV_DIRECT_ONLY_VALUE_OPTIONS: &[&str] = &[
    "--dep-coverage",
    "--exclude-from-report",
    "--output-path",
    "--output-dir",
    "--failure-mode",
    "--ignore-filename-regex",
    "--fail-under-functions",
    "--fail-under-lines",
    "--fail-under-file-lines",
    "--fail-under-regions",
    "--fail-uncovered-lines",
    "--fail-uncovered-regions",
    "--fail-uncovered-functions",
    "--nextest-archive-file",
];

const LLVM_COV_DIRECT_ONLY_FLAGS: &[&str] = &[
    "--no-clean",
    "--no-report",
    "--ignore-run-fail",
    "--json",
    "--lcov",
    "--cobertura",
    "--codecov",
    "--text",
    "--html",
    "--open",
    "--summary-only",
    "--no-default-ignore-filename-regex",
    "--disable-default-ignore-filename-regex",
    "--show-instantiations",
    "--hide-instantiations",
    "--show-missing-lines",
    "--include-build-script",
    "--skip-functions",
    "--branch",
    "--mcdc",
    "--doctests",
    "--coverage-target-only",
    "--coverage-host-only",
    "--include-ffi",
    "--no-rustc-wrapper",
    "--no-cfg-coverage",
    "--no-cfg-coverage-nightly",
    "--remap-path-prefix",
];

const NEXTEST_REUSE_CARGO_VALUE_OPTIONS: &[&str] = &[
    "-p",
    "--package",
    "--exclude",
    "--bin",
    "--example",
    "--test",
    "--bench",
    "-F",
    "--features",
    "--build-jobs",
    "--cargo-profile",
    "--target",
    "--target-dir",
    "--cargo-message-format",
    "--manifest-path",
    "-Z",
];

const NEXTEST_REUSE_CARGO_FLAGS: &[&str] = &[
    "--workspace",
    "--all",
    "--lib",
    "--bins",
    "--examples",
    "--tests",
    "--benches",
    "--all-targets",
    "--all-features",
    "--no-default-features",
    "-r",
    "--release",
    "--frozen",
    "--locked",
    "--offline",
    "--unit-graph",
    "--timings",
    "--cargo-quiet",
    "--cargo-verbose",
    "--ignore-rust-version",
    "--future-incompat-report",
];

fn option_has_joined_value(argument: &str, options: &[&str]) -> bool {
    options.iter().any(|option| {
        let short = option.starts_with('-') && !option.starts_with("--");
        argument
            .strip_prefix(option)
            .is_some_and(|suffix| suffix.starts_with('=') || short && !suffix.is_empty())
    })
}

/// Convert cargo-llvm-cov's build surface to the argv accepted by a direct
/// `cargo nextest list`. cargo-llvm-cov-only instrumentation and report flags
/// live in the show-env/report commands; `--exclude-from-test` is the one
/// build selector whose spelling must be translated for nextest itself.
fn direct_nextest_build_surface_args(
    sub_argv: &[&str],
    args: &[String],
) -> Result<Vec<String>, String> {
    let (globals, projected) = nextest_build_surface_args(sub_argv, args)?;
    let combined = globals.into_iter().chain(projected).collect::<Vec<_>>();
    let mut out = Vec::with_capacity(combined.len());
    let mut index = 0;
    while index < combined.len() {
        let argument = &combined[index];
        if argument == "--exclude-from-test" {
            out.push("--exclude".to_string());
            index += 1;
            if let Some(value) = combined.get(index) {
                out.push(value.clone());
            }
            index += 1;
            continue;
        }
        if let Some(value) = argument.strip_prefix("--exclude-from-test=") {
            out.push(format!("--exclude={value}"));
            index += 1;
            continue;
        }
        if LLVM_COV_DIRECT_ONLY_VALUE_OPTIONS.contains(&argument.as_str()) {
            index += 2;
            continue;
        }
        if option_has_joined_value(argument, LLVM_COV_DIRECT_ONLY_VALUE_OPTIONS)
            || LLVM_COV_DIRECT_ONLY_FLAGS.contains(&argument.as_str())
        {
            index += 1;
            continue;
        }
        out.push(argument.clone());
        index += 1;
    }
    Ok(out)
}

/// Project a cached invocation onto nextest's run-only surface. Reuse-build
/// explicitly conflicts with Cargo package/target/feature/build options; the
/// cached binary metadata already embodies those choices, while filters,
/// retries, profiles, output controls, and test-binary arguments remain live.
fn cached_nextest_run_args(sub_argv: &[&str], args: &[String]) -> Vec<String> {
    let raw_nextest = (sub_argv == LLVM_COV_SUB_ARGV)
        .then(|| llvm_cov_nextest_index(args))
        .flatten();
    let mut out = Vec::with_capacity(args.len());
    let mut index = 0;
    while index < args.len() {
        if Some(index) == raw_nextest {
            index += 1;
            continue;
        }
        let argument = &args[index];
        if argument == "--" {
            out.extend_from_slice(&args[index..]);
            break;
        }
        if NEXTEST_REUSE_CARGO_VALUE_OPTIONS.contains(&argument.as_str())
            || LLVM_COV_DIRECT_ONLY_VALUE_OPTIONS.contains(&argument.as_str())
            || matches!(
                argument.as_str(),
                "--exclude-from-test" | "--exclude-from-report" | "--dep-coverage"
            )
        {
            index += 2;
            continue;
        }
        if option_has_joined_value(argument, NEXTEST_REUSE_CARGO_VALUE_OPTIONS)
            || option_has_joined_value(argument, LLVM_COV_DIRECT_ONLY_VALUE_OPTIONS)
            || argument.starts_with("--exclude-from-test=")
            || argument.starts_with("--exclude-from-report=")
            || argument.starts_with("--dep-coverage=")
            || argument.starts_with("--timings=")
            || NEXTEST_REUSE_CARGO_FLAGS.contains(&argument.as_str())
            || LLVM_COV_DIRECT_ONLY_FLAGS.contains(&argument.as_str())
        {
            index += 1;
            continue;
        }
        out.push(argument.clone());
        index += 1;
    }
    out
}

/// cargo-llvm-cov's `--ignore-run-fail` runs the complete test set before it
/// reports. Once ktstr decomposes that command into a direct nextest run plus a
/// report command, reproduce that behavior explicitly and keep the test-binary
/// suffix opaque.
fn force_nextest_no_fail_fast(args: Vec<String>) -> Vec<String> {
    let separator = args
        .iter()
        .position(|argument| argument == "--")
        .unwrap_or(args.len());
    let mut out = Vec::with_capacity(args.len() + 1);
    let mut index = 0;
    while index < separator {
        match args[index].as_str() {
            "--fail-fast" | "--ff" | "--no-fail-fast" | "--nff" => {
                index += 1;
            }
            "--max-fail" => {
                index += 2;
            }
            argument if argument.starts_with("--max-fail=") => {
                index += 1;
            }
            _ => {
                out.push(args[index].clone());
                index += 1;
            }
        }
    }
    out.push("--no-fail-fast".to_string());
    out.extend_from_slice(&args[separator..]);
    out
}

const LLVM_COV_REPORT_VALUE_OPTIONS: &[&str] = &[
    "--output-path",
    "--output-dir",
    "--failure-mode",
    "--ignore-filename-regex",
    "--fail-under-functions",
    "--fail-under-lines",
    "--fail-under-file-lines",
    "--fail-under-regions",
    "--fail-uncovered-lines",
    "--fail-uncovered-regions",
    "--fail-uncovered-functions",
    "--dep-coverage",
    "--target",
    "--color",
    "--manifest-path",
    "-Z",
];

const LLVM_COV_REPORT_FLAGS: &[&str] = &[
    "--json",
    "--lcov",
    "--cobertura",
    "--codecov",
    "--text",
    "--html",
    "--open",
    "--summary-only",
    "--no-default-ignore-filename-regex",
    "--disable-default-ignore-filename-regex",
    "--show-instantiations",
    "--hide-instantiations",
    "--show-missing-lines",
    "--include-build-script",
    "--skip-functions",
    "-r",
    "--release",
    "--frozen",
    "--locked",
    "--offline",
    "--doctests",
    "--coverage-target-only",
    "--coverage-host-only",
    "--remap-path-prefix",
    "--include-ffi",
    "--branch",
    "--mcdc",
    "-v",
    "--verbose",
    "-q",
    "--quiet",
];

/// Options accepted by a build/test invocation but rejected by
/// cargo-llvm-cov's `report` subcommand. Keep these explicit even though the
/// report projection is allow-listed: consuming a separate value prevents an
/// option-looking value from being reinterpreted as a report flag.
const LLVM_COV_BUILD_OR_RUN_ONLY_VALUE_OPTIONS: &[&str] = &[
    "-F",
    "--features",
    "-j",
    "--jobs",
    "--build-jobs",
    "-p",
    "--package",
    "--exclude",
    "--exclude-from-test",
    "--exclude-from-report",
    "--bin",
    "--example",
    "--test",
    "--bench",
    "--target-dir",
    "--cargo-message-format",
    "--profile",
    "--test-threads",
    "--retries",
    "--max-fail",
    "--config",
    "--archive-file",
    "--archive-format",
    "--zstd-level",
    "--config-file",
    "--user-config-file",
    "--tool-config-file",
    "-P",
    "-E",
    "--filterset",
    "--filter-expr",
];

const LLVM_COV_BUILD_OR_RUN_ONLY_FLAGS: &[&str] = &[
    "--workspace",
    "--all",
    "--lib",
    "--bins",
    "--examples",
    "--tests",
    "--benches",
    "--all-targets",
    "--all-features",
    "--no-default-features",
    "--fail-fast",
    "--ff",
    "--no-fail-fast",
    "--nff",
    "--unit-graph",
    "--timings",
    "--cargo-quiet",
    "--cargo-verbose",
    "--ignore-rust-version",
    "--future-incompat-report",
];

fn llvm_cov_report_selection_args(args: &[String]) -> Vec<String> {
    let mut out = Vec::with_capacity(args.len());
    let mut index = 0;
    while index < args.len() {
        let argument = &args[index];
        if argument == "--" {
            out.extend_from_slice(&args[index..]);
            break;
        }
        if argument == "--exclude-from-test" {
            index += 2;
            continue;
        }
        if argument == "--exclude-from-report" {
            out.push("--exclude".to_string());
            index += 1;
            if let Some(value) = args.get(index) {
                out.push(value.clone());
            }
            index += 1;
            continue;
        }
        if argument.starts_with("--exclude-from-test=") {
            index += 1;
            continue;
        }
        if let Some(value) = argument.strip_prefix("--exclude-from-report=") {
            out.push(format!("--exclude={value}"));
            index += 1;
            continue;
        }
        out.push(argument.clone());
        index += 1;
    }
    out
}

fn cached_llvm_cov_report_args(
    sub_argv: &[&str],
    args: &[String],
    release: bool,
    metadata: &cargo_metadata::Metadata,
    invocation_dir: &Path,
) -> Vec<String> {
    fn remap_report_path(value: &str, invocation_dir: &Path) -> String {
        let path = Path::new(value);
        let absolute = if path.is_absolute() {
            path.to_path_buf()
        } else {
            invocation_dir.join(path)
        };
        absolute.display().to_string()
    }

    let raw_nextest = (sub_argv == LLVM_COV_SUB_ARGV)
        .then(|| llvm_cov_nextest_index(args))
        .flatten();
    let mut out = vec!["report".to_string()];
    if release {
        out.push("--release".to_string());
    }
    let mut index = 0;
    while index < args.len() {
        if Some(index) == raw_nextest {
            index += 1;
            continue;
        }
        let argument = &args[index];
        if argument == "--" {
            break;
        }
        if argument == "--cargo-profile" {
            index += 1;
            if let Some(value) = args.get(index) {
                out.extend(["--profile".to_string(), value.clone()]);
            }
            index += 1;
            continue;
        }
        if let Some(value) = argument.strip_prefix("--cargo-profile=") {
            out.push(format!("--profile={value}"));
            index += 1;
            continue;
        }
        if LLVM_COV_BUILD_OR_RUN_ONLY_VALUE_OPTIONS.contains(&argument.as_str()) {
            index += 2;
            continue;
        }
        if option_has_joined_value(argument, LLVM_COV_BUILD_OR_RUN_ONLY_VALUE_OPTIONS)
            || LLVM_COV_BUILD_OR_RUN_ONLY_FLAGS.contains(&argument.as_str())
        {
            index += 1;
            continue;
        }
        if LLVM_COV_REPORT_VALUE_OPTIONS.contains(&argument.as_str()) {
            out.push(argument.clone());
            index += 1;
            if let Some(value) = args.get(index) {
                let value = if matches!(
                    argument.as_str(),
                    "--output-path" | "--output-dir" | "--manifest-path"
                ) {
                    remap_report_path(value, invocation_dir)
                } else {
                    value.clone()
                };
                out.push(value);
            }
            index += 1;
            continue;
        }
        if let Some((option, value)) = argument.split_once('=').filter(|(option, _)| {
            matches!(
                *option,
                "--output-path" | "--output-dir" | "--manifest-path"
            )
        }) {
            out.push(format!(
                "{option}={}",
                remap_report_path(value, invocation_dir)
            ));
            index += 1;
            continue;
        }
        if option_has_joined_value(argument, LLVM_COV_REPORT_VALUE_OPTIONS)
            || LLVM_COV_REPORT_FLAGS.contains(&argument.as_str())
        {
            out.push(argument.clone());
        }
        index += 1;
    }
    let wants_directory_report =
        llvm_cov_has_lifecycle_flag(args, "--html") || llvm_cov_has_lifecycle_flag(args, "--open");
    let has_output_directory = args
        .iter()
        .take_while(|argument| *argument != "--")
        .any(|argument| argument == "--output-dir" || argument.starts_with("--output-dir="));
    if wants_directory_report && !has_output_directory {
        out.extend([
            "--output-dir".to_string(),
            metadata
                .target_directory
                .as_std_path()
                .join("llvm-cov")
                .display()
                .to_string(),
        ]);
    }
    let report_selection = llvm_cov_report_selection_args(args);
    if let Some(packages) =
        selected_workspace_packages_for_invocation(metadata, &report_selection, invocation_dir)
    {
        for package in packages {
            out.extend(["--package".to_string(), package.name.to_string()]);
        }
    }
    out
}

const PROFRAW_CLEANUP_DIAGNOSTIC_LIMIT: usize = 4;
const PROFRAW_CLEANUP_DIAGNOSTIC_CHARS: usize = 320;
const COVERAGE_RECOVERY_PREFIX: &str = "report-";
const COVERAGE_RECOVERY_LIVE_LOCK: &str = ".ktstr-live.lock";
const COVERAGE_RECOVERY_GC_LOCK: &str = ".ktstr-gc.lock";
const COVERAGE_RECOVERY_SIZE_FILE: &str = ".ktstr-logical-bytes";
const COVERAGE_RECOVERY_MAX_AGE: std::time::Duration = std::time::Duration::from_secs(24 * 60 * 60);
const COVERAGE_RECOVERY_MAX_BUNDLES: usize = 4;
const COVERAGE_RECOVERY_MAX_BYTES: u64 = 8 << 30;

/// Post-run state needed to finish a cached cargo-llvm-cov invocation.
///
/// The test process writes raw counters into its private materialized target,
/// while the report itself runs from the caller's writable checkout. Keeping
/// the owned raw directory here makes report/merge and reclamation one ordered
/// lifecycle instead of leaving failed test runs to accumulate shards.
struct CachedCoverageReport {
    report_args: Vec<String>,
    environment: Vec<(OsString, OsString)>,
    report_dir: PathBuf,
    llvm_cov_flags: OsString,
    profraw_directory: PathBuf,
    producer_profdata: Option<PathBuf>,
    merged_profdata: PathBuf,
    no_report: bool,
    ignore_run_fail: bool,
}

/// The final failure footer must name nextest only when nextest remains the
/// authoritative failure. A successful (or explicitly ignored) test run
/// followed by a failed coverage report is a post-run failure instead.
fn should_render_nextest_failure_footer(
    final_success: bool,
    nextest_success: bool,
    ignore_run_fail: bool,
) -> bool {
    !final_success && !nextest_success && !ignore_run_fail
}

#[derive(Debug, Default, Eq, PartialEq)]
struct ProfrawCleanup {
    removed: usize,
    failures: usize,
    diagnostics: Vec<String>,
}

/// Raw coverage shards temporarily removed from cargo-llvm-cov's scan root.
///
/// cargo-llvm-cov overwrites an existing profdata whenever any `*.profraw`
/// remains beside it. ktstr therefore merges the cached producer seed and the
/// live shards itself, then holds the shards here until the report outcome is
/// known. Success drops the directory; failure detaches it for recovery.
struct StagedProfraw {
    directory: Option<CoverageRecoveryDirectory>,
    recovery_parent: PathBuf,
    count: usize,
}

/// One live coverage recovery bundle.
///
/// The lock prevents a concurrent opportunistic collector from reclaiming a
/// bundle while raw profiles, a merged profile, or a retained artifact tree
/// are still being installed. Persisting the bundle disarms TempDir cleanup;
/// dropping it on the green report path removes everything immediately.
struct CoverageRecoveryDirectory {
    directory: tempfile::TempDir,
    _live: std::fs::File,
}

impl CoverageRecoveryDirectory {
    fn path(&self) -> &Path {
        self.directory.path()
    }

    fn close(self) -> std::io::Result<()> {
        self.directory.close()
    }

    fn keep(self, recovery_parent: &Path) -> PathBuf {
        let Self { directory, _live } = self;
        let protected = directory.path().to_path_buf();
        // Persist a constant-time size summary before detaching ownership.
        // Recovery bundles are immutable until an operator explicitly uses
        // them, so later green coverage runs do not restat an entire retained
        // Cargo artifact closure merely to enforce the byte budget.
        let _ = coverage_recovery_bundle_bytes(&protected);
        if let Err(error) = gc_coverage_recovery_bundles_in(
            recovery_parent,
            std::time::SystemTime::now(),
            CoverageRecoveryLimits::DEFAULT,
            Some(&protected),
        ) {
            tracing::warn!(
                error = %error,
                parent = %recovery_parent.display(),
                "could not prune retained coverage recovery bundles",
            );
        }
        let path = directory.keep();
        drop(_live);
        // Run once more after publishing/unlocking. Concurrent persistors may
        // all have been live during the protected pass; serializing this
        // unlocked pass through the global GC gate guarantees the last
        // finisher observes and bounds every earlier completed bundle.
        if let Err(error) = gc_coverage_recovery_bundles_in(
            recovery_parent,
            std::time::SystemTime::now(),
            CoverageRecoveryLimits::DEFAULT,
            None,
        ) {
            tracing::warn!(
                error = %error,
                parent = %recovery_parent.display(),
                "could not finalize coverage recovery bundle pruning",
            );
        }
        path
    }
}

#[derive(Clone, Copy)]
struct CoverageRecoveryLimits {
    max_age: std::time::Duration,
    max_bundles: usize,
    max_bytes: u64,
}

impl CoverageRecoveryLimits {
    const DEFAULT: Self = Self {
        max_age: COVERAGE_RECOVERY_MAX_AGE,
        max_bundles: COVERAGE_RECOVERY_MAX_BUNDLES,
        max_bytes: COVERAGE_RECOVERY_MAX_BYTES,
    };
}

impl StagedProfraw {
    fn discard(mut self) -> Result<(), String> {
        let Some(directory) = self.directory.take() else {
            return Ok(());
        };
        let path = directory.path().to_path_buf();
        directory.close().map_err(|error| {
            format!(
                "remove successfully reported coverage raw shards under {}: {error}",
                path.display(),
            )
        })
    }

    fn persist(self, merged_profdata: &Path) -> Result<Option<PathBuf>, String> {
        self.persist_with(merged_profdata, false, |_| Ok(()))
    }

    fn persist_with<F>(
        mut self,
        merged_profdata: &Path,
        require_directory: bool,
        populate: F,
    ) -> Result<Option<PathBuf>, String>
    where
        F: FnOnce(&Path) -> Result<(), String>,
    {
        if self.directory.is_none() && (require_directory || merged_profdata.is_file()) {
            self.directory = Some(create_coverage_recovery_dir(&self.recovery_parent)?);
        }
        let Some(directory) = self.directory.take() else {
            return Ok(None);
        };
        if merged_profdata.is_file() {
            let destination = directory.path().join("merged.profdata");
            if let Err(error) = ktstr::cache::reflink_file_required(merged_profdata, &destination) {
                let retained = directory.keep(&self.recovery_parent);
                return Err(format!(
                    "preserve merged coverage profile as a strict COW clone {} -> {}: {error:#}; raw shards retained at {}",
                    merged_profdata.display(),
                    destination.display(),
                    retained.display(),
                ));
            }
        }
        if let Err(error) = populate(directory.path()) {
            let retained = directory.keep(&self.recovery_parent);
            return Err(format!(
                "populate retained coverage bundle: {error}; partial bundle retained at {}",
                retained.display(),
            ));
        }
        Ok(Some(directory.keep(&self.recovery_parent)))
    }
}

/// Render the operator-facing suffix describing what a failed coverage report
/// managed to retain. `retained` names the profile in the success wording; the
/// exact text is diagnostic-only and has no test asserting it, so keep the
/// wording stable when editing.
fn coverage_recovery_suffix(recovery: Result<Option<PathBuf>, String>, retained: &str) -> String {
    match recovery {
        Ok(Some(path)) => format!("; raw shards and {retained} retained at {}", path.display()),
        Ok(None) => String::new(),
        Err(error) => format!("; additionally failed to retain coverage inputs: {error}"),
    }
}

fn coverage_recovery_parent() -> Result<PathBuf, String> {
    let cache_root = ktstr::cache::cargo_artifact_tree_cache_root()
        .map_err(|error| format!("resolve coverage recovery cache root: {error:#}"))?;
    Ok(cache_root.join("coverage-profraw-recovery-v1"))
}

fn coverage_recovery_bundle_bytes(path: &Path) -> Result<u64, String> {
    let size_path = path.join(COVERAGE_RECOVERY_SIZE_FILE);
    if let Ok(size) = std::fs::read_to_string(&size_path)
        && let Ok(size) = size.trim().parse::<u64>()
    {
        return Ok(size);
    }
    let mut bytes = 0u64;
    for entry in walkdir::WalkDir::new(path).follow_links(false) {
        let entry = entry.map_err(|error| {
            format!("walk coverage recovery bundle {}: {error}", path.display(),)
        })?;
        if entry.file_type().is_file()
            && entry.path().file_name() != Some(OsStr::new(COVERAGE_RECOVERY_SIZE_FILE))
        {
            bytes = bytes.saturating_add(
                entry
                    .metadata()
                    .map_err(|error| {
                        format!(
                            "inspect coverage recovery entry {}: {error}",
                            entry.path().display(),
                        )
                    })?
                    .len(),
            );
        }
    }
    // This marker is only an acceleration hint. A read-only or manually
    // damaged bundle remains reclaimable through the measured value above.
    let _ = std::fs::write(&size_path, format!("{bytes}\n"));
    Ok(bytes)
}

struct CoverageRecoveryCandidate {
    path: PathBuf,
    modified: std::time::SystemTime,
    bytes: u64,
    _live: std::fs::File,
}

/// Opportunistically bound persisted report-failure and `--no-report`
/// bundles.
///
/// Active writers are excluded by their per-bundle liveness flock. Expired
/// bundles are always removed. The remaining unlocked bundles are oldest-first
/// reclaimed until both count and logical-byte bounds hold. One newest bundle
/// is retained even when it alone exceeds the byte cap, so the most recent
/// failure remains diagnosable without allowing repeated jobs to accumulate
/// without bound.
fn gc_coverage_recovery_bundles_in(
    recovery_parent: &Path,
    now: std::time::SystemTime,
    limits: CoverageRecoveryLimits,
    protected: Option<&Path>,
) -> Result<(), String> {
    std::fs::create_dir_all(recovery_parent).map_err(|error| {
        format!(
            "create coverage recovery directory {}: {error}",
            recovery_parent.display(),
        )
    })?;
    let Some(_collector) = ktstr::flock::try_flock(
        recovery_parent.join(COVERAGE_RECOVERY_GC_LOCK),
        ktstr::flock::FlockMode::Exclusive,
    )
    .map_err(|error| {
        format!(
            "acquire coverage recovery GC lock in {}: {error:#}",
            recovery_parent.display(),
        )
    })?
    else {
        return Ok(());
    };

    let mut candidates = Vec::new();
    for entry in std::fs::read_dir(recovery_parent).map_err(|error| {
        format!(
            "scan coverage recovery directory {}: {error}",
            recovery_parent.display(),
        )
    })? {
        let entry = match entry {
            Ok(entry) => entry,
            Err(error) => {
                tracing::debug!(error = %error, "could not read coverage recovery entry");
                continue;
            }
        };
        if !entry
            .file_name()
            .as_bytes()
            .starts_with(COVERAGE_RECOVERY_PREFIX.as_bytes())
        {
            continue;
        }
        let path = entry.path();
        if protected.is_some_and(|protected| protected == path) {
            continue;
        }
        let metadata = match std::fs::symlink_metadata(&path) {
            Ok(metadata) if metadata.file_type().is_dir() => metadata,
            Ok(_) => continue,
            Err(error) => {
                tracing::debug!(path = %path.display(), error = %error, "could not inspect coverage recovery bundle");
                continue;
            }
        };
        let modified = metadata.modified().unwrap_or(now);
        let live_path = path.join(COVERAGE_RECOVERY_LIVE_LOCK);
        let live = match ktstr::flock::try_flock(&live_path, ktstr::flock::FlockMode::Exclusive) {
            Ok(Some(live)) => live,
            Ok(None) => continue,
            Err(error) => {
                tracing::debug!(path = %path.display(), error = %error, "could not probe coverage recovery liveness");
                continue;
            }
        };
        let bytes = match coverage_recovery_bundle_bytes(&path) {
            Ok(bytes) => bytes,
            Err(error) => {
                tracing::debug!(path = %path.display(), error = %error, "could not size coverage recovery bundle");
                continue;
            }
        };
        candidates.push(CoverageRecoveryCandidate {
            path,
            modified,
            bytes,
            _live: live.into(),
        });
    }
    candidates.sort_by(|left, right| {
        left.modified.cmp(&right.modified).then_with(|| {
            left.path
                .as_os_str()
                .as_bytes()
                .cmp(right.path.as_os_str().as_bytes())
        })
    });

    let mut retained = Vec::with_capacity(candidates.len());
    for candidate in candidates {
        let expired = now
            .duration_since(candidate.modified)
            .unwrap_or(std::time::Duration::ZERO)
            >= limits.max_age;
        if expired {
            match std::fs::remove_dir_all(&candidate.path) {
                Ok(()) => continue,
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => continue,
                Err(error) => tracing::debug!(
                    path = %candidate.path.display(),
                    error = %error,
                    "could not remove expired coverage recovery bundle",
                ),
            }
        }
        retained.push(candidate);
    }

    let protected_count = if protected.is_some_and(|path| path.exists()) {
        1
    } else {
        0
    };
    let protected_bytes = protected
        .filter(|path| path.exists())
        .and_then(|path| coverage_recovery_bundle_bytes(path).ok())
        .unwrap_or(0);
    let mut total_count = retained.len().saturating_add(protected_count);
    let mut total_bytes = retained.iter().fold(protected_bytes, |total, candidate| {
        total.saturating_add(candidate.bytes)
    });
    // With no protected in-progress bundle, preserve the newest completed
    // bundle even if its own profile happens to exceed the nominal byte cap.
    let minimum_unprotected = if protected_count == 0 && !retained.is_empty() {
        1
    } else {
        0
    };
    let mut remaining_unprotected = retained.len();
    for candidate in retained {
        if total_count <= limits.max_bundles && total_bytes <= limits.max_bytes {
            break;
        }
        if remaining_unprotected <= minimum_unprotected {
            break;
        }
        remaining_unprotected = remaining_unprotected.saturating_sub(1);
        match std::fs::remove_dir_all(&candidate.path) {
            Ok(()) => {
                total_count = total_count.saturating_sub(1);
                total_bytes = total_bytes.saturating_sub(candidate.bytes);
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                total_count = total_count.saturating_sub(1);
                total_bytes = total_bytes.saturating_sub(candidate.bytes);
            }
            Err(error) => tracing::debug!(
                path = %candidate.path.display(),
                error = %error,
                "could not prune coverage recovery bundle",
            ),
        }
    }
    Ok(())
}

fn create_coverage_recovery_dir(
    recovery_parent: &Path,
) -> Result<CoverageRecoveryDirectory, String> {
    std::fs::create_dir_all(recovery_parent).map_err(|error| {
        format!(
            "create coverage recovery directory {}: {error}",
            recovery_parent.display(),
        )
    })?;
    if let Err(error) = gc_coverage_recovery_bundles_in(
        recovery_parent,
        std::time::SystemTime::now(),
        CoverageRecoveryLimits::DEFAULT,
        None,
    ) {
        tracing::warn!(
            error = %error,
            parent = %recovery_parent.display(),
            "could not prune coverage recovery bundles before staging",
        );
    }
    let directory = tempfile::Builder::new()
        .prefix(COVERAGE_RECOVERY_PREFIX)
        .tempdir_in(recovery_parent)
        .map_err(|error| {
            format!(
                "create coverage recovery staging directory in {}: {error}",
                recovery_parent.display(),
            )
        })?;
    let live_path = directory.path().join(COVERAGE_RECOVERY_LIVE_LOCK);
    let live = ktstr::flock::try_flock(&live_path, ktstr::flock::FlockMode::Exclusive)
        .map_err(|error| {
            format!(
                "acquire coverage recovery liveness lock {}: {error:#}",
                live_path.display(),
            )
        })?
        .ok_or_else(|| {
            format!(
                "new coverage recovery liveness lock is unexpectedly held: {}",
                live_path.display(),
            )
        })?;
    Ok(CoverageRecoveryDirectory {
        directory,
        _live: live.into(),
    })
}

fn environment_value<'a>(environment: &'a [(OsString, OsString)], name: &str) -> Option<&'a OsStr> {
    environment
        .iter()
        .rev()
        .find(|(candidate, _)| candidate == OsStr::new(name))
        .map(|(_, value)| value.as_os_str())
}

fn resolve_llvm_profdata(
    current_dir: &Path,
    environment: &[(OsString, OsString)],
) -> Result<PathBuf, String> {
    if let Some(tool) = environment_value(environment, "LLVM_PROFDATA")
        .filter(|tool| !tool.is_empty())
        .map(PathBuf::from)
        .or_else(|| {
            std::env::var_os("LLVM_PROFDATA")
                .filter(|tool| !tool.is_empty())
                .map(PathBuf::from)
        })
    {
        return Ok(tool);
    }

    let rustc = environment_value(environment, "RUSTC")
        .filter(|tool| !tool.is_empty())
        .map(OsStr::to_os_string)
        .or_else(|| std::env::var_os("RUSTC").filter(|tool| !tool.is_empty()))
        .or_else(|| {
            environment_value(environment, "CARGO_BUILD_RUSTC")
                .filter(|tool| !tool.is_empty())
                .map(OsStr::to_os_string)
        })
        .or_else(|| std::env::var_os("CARGO_BUILD_RUSTC").filter(|tool| !tool.is_empty()))
        .unwrap_or_else(|| OsString::from("rustc"));
    let mut command = Command::new(&rustc);
    command
        .args(["--print", "target-libdir"])
        .current_dir(current_dir);
    apply_command_envs(&mut command, environment);
    let output = crate::interrupt::run_output(command)
        .map_err(|error| format!("resolve llvm-profdata via rustc target-libdir: {error}"))?;
    if !output.status.success() {
        return Err(format!(
            "resolve llvm-profdata via {} --print target-libdir failed with {}: {}",
            rustc.to_string_lossy(),
            output.status,
            String::from_utf8_lossy(&output.stderr).trim(),
        ));
    }
    let target_libdir = std::str::from_utf8(&output.stdout)
        .map_err(|error| format!("rustc target-libdir was not UTF-8: {error}"))?
        .trim();
    if target_libdir.is_empty() {
        return Err("rustc returned an empty target-libdir".to_string());
    }
    let tool = llvm_profdata_from_target_libdir(Path::new(target_libdir))?;
    if !tool.is_file() {
        return Err(format!(
            "llvm-profdata is missing at {}; install rustup component llvm-tools-preview or set LLVM_PROFDATA",
            tool.display(),
        ));
    }
    Ok(tool)
}

fn llvm_profdata_from_target_libdir(target_libdir: &Path) -> Result<PathBuf, String> {
    let mut tool = target_libdir.to_path_buf();
    if !tool.pop() {
        return Err(format!(
            "rustc target-libdir has no parent: {}",
            target_libdir.display(),
        ));
    }
    tool.push("bin");
    tool.push("llvm-profdata");
    Ok(tool)
}

fn profraw_files(directory: &Path) -> Result<Vec<PathBuf>, String> {
    let entries = match std::fs::read_dir(directory) {
        Ok(entries) => entries,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(error) => {
            return Err(format!(
                "read coverage profile directory {}: {error}",
                directory.display(),
            ));
        }
    };
    let mut profiles = Vec::new();
    for entry in entries {
        let entry = match entry {
            Ok(entry) => entry,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => continue,
            Err(error) => return Err(format!("read coverage profile entry: {error}")),
        };
        let path = entry.path();
        if path.extension() != Some(OsStr::new("profraw")) {
            continue;
        }
        let file_type = match entry.file_type() {
            Ok(file_type) => file_type,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => continue,
            Err(error) => {
                return Err(format!(
                    "inspect coverage profile {}: {error}",
                    path.display(),
                ));
            }
        };
        if !file_type.is_file() {
            return Err(format!(
                "coverage profile is not a regular file: {}",
                path.display(),
            ));
        }
        profiles.push(path);
    }
    profiles.sort();
    Ok(profiles)
}

fn llvm_profdata_flags(environment: &[(OsString, OsString)]) -> Result<Vec<OsString>, String> {
    let command_value = |name| {
        environment_value(environment, name)
            .map(OsStr::to_os_string)
            .or_else(|| std::env::var_os(name))
    };
    let flags =
        command_value("LLVM_PROFDATA_FLAGS").or_else(|| command_value("CARGO_LLVM_PROFDATA_FLAGS"));
    let Some(flags) = flags else {
        return Ok(Vec::new());
    };
    let flags = flags
        .into_string()
        .map_err(|_| "LLVM_PROFDATA_FLAGS is not valid UTF-8".to_string())?;
    Ok(flags
        .split(' ')
        .filter(|flag| !flag.trim_start().is_empty())
        .map(OsString::from)
        .collect())
}

fn configure_llvm_profdata_merge(
    command: &mut Command,
    input_list: &Path,
    output: &Path,
    failure_mode: Option<&str>,
    flags: impl IntoIterator<Item = OsString>,
) {
    command
        .args(["merge", "-sparse"])
        .arg("-f")
        .arg(input_list)
        .arg("-o")
        .arg(output);
    if let Some(failure_mode) = failure_mode {
        command.arg(format!("-failure-mode={failure_mode}"));
    }
    command.args(flags);
}

fn llvm_cov_failure_mode(report_args: &[String]) -> Option<&str> {
    let mut index = 0;
    while index < report_args.len() {
        let argument = &report_args[index];
        if argument == "--failure-mode" {
            return report_args.get(index + 1).map(String::as_str);
        }
        if let Some(value) = argument.strip_prefix("--failure-mode=") {
            return Some(value);
        }
        index += 1;
    }
    None
}

/// Match cargo-llvm-cov's `Workspace::new` profile lookup exactly: its merged
/// profile is `<CARGO_LLVM_COV_TARGET_DIR>/<workspace-root-basename>.profdata`.
fn cargo_llvm_cov_profdata_path(target_directory: &Path, workspace_root: &Path) -> PathBuf {
    let mut name = workspace_root
        .file_name()
        .unwrap_or_else(|| OsStr::new("default"))
        .to_os_string();
    name.push(".profdata");
    target_directory.join(name)
}

fn merge_profdata(
    current_dir: &Path,
    environment: &[(OsString, OsString)],
    inputs: &[PathBuf],
    output: &Path,
    failure_mode: Option<&str>,
) -> Result<(), String> {
    if inputs.is_empty() {
        return Err("no coverage profile inputs were produced".to_string());
    }
    let parent = output.parent().ok_or_else(|| {
        format!(
            "merged coverage profile has no parent directory: {}",
            output.display(),
        )
    })?;
    std::fs::create_dir_all(parent).map_err(|error| {
        format!(
            "create merged coverage profile directory {}: {error}",
            parent.display(),
        )
    })?;
    let mut input_list = tempfile::Builder::new()
        .prefix(".ktstr-profdata-inputs-")
        .tempfile_in(parent)
        .map_err(|error| {
            format!(
                "create llvm-profdata input list in {}: {error}",
                parent.display()
            )
        })?;
    for input in inputs {
        use std::io::Write as _;
        let input = input.to_str().ok_or_else(|| {
            format!(
                "coverage profile path is not valid UTF-8: {}",
                input.display()
            )
        })?;
        if input.contains(['\n', '\r']) {
            return Err(format!(
                "coverage profile path contains a newline: {input:?}"
            ));
        }
        writeln!(input_list.as_file_mut(), "{input}")
            .map_err(|error| format!("write llvm-profdata input list: {error}"))?;
    }
    input_list
        .as_file_mut()
        .sync_all()
        .map_err(|error| format!("flush llvm-profdata input list: {error}"))?;
    let staged_dir = tempfile::Builder::new()
        .prefix(".ktstr-profdata-output-")
        .tempdir_in(parent)
        .map_err(|error| {
            format!(
                "create llvm-profdata output directory in {}: {error}",
                parent.display()
            )
        })?;
    let staged = staged_dir.path().join("merged.profdata");
    let tool = resolve_llvm_profdata(current_dir, environment)?;
    let mut command = Command::new(&tool);
    // Default to `-failure-mode=all`: llvm-profdata fails only when EVERY
    // input is unreadable, so one torn shard (a watchdog SIGKILL caught a
    // test process mid-write) costs just that process's coverage instead of
    // aborting the whole merge. The pinned llvm-profdata (LLVM 21.1) documents
    // `any` (its default) as "fail if any profile is invalid" and `all` as
    // "fail only if all profiles are invalid"; verified empirically that
    // `all` drops a corrupt shard beside a valid one and still fails loudly
    // on an all-corrupt set. An explicit caller mode (from the report's
    // `--failure-mode`) still wins.
    let failure_mode = failure_mode.unwrap_or("all");
    configure_llvm_profdata_merge(
        &mut command,
        input_list.path(),
        &staged,
        Some(failure_mode),
        llvm_profdata_flags(environment)?,
    );
    command.current_dir(current_dir);
    apply_command_envs(&mut command, environment);
    let merge_output = crate::interrupt::run_output(command)
        .map_err(|error| format!("spawn {}: {error}", tool.display()))?;
    // Forward llvm-profdata's own per-shard warnings so a dropped shard is
    // never invisible, on both the success and all-corrupt-failure paths.
    let stderr = String::from_utf8_lossy(&merge_output.stderr);
    if !stderr.is_empty() {
        eprint!("{stderr}");
    }
    if !merge_output.status.success() {
        return Err(format!(
            "llvm-profdata merge exited with {}",
            merge_output
                .status
                .code()
                .map_or("signal".to_string(), |code| code.to_string()),
        ));
    }
    // On a tolerant success, name the shards llvm-profdata dropped so the
    // coverage loss is a visible, quantified line rather than a warning buried
    // among thousands.
    let dropped = dropped_shards_from_stderr(&stderr, inputs);
    if !dropped.is_empty() {
        let names = dropped
            .iter()
            .map(|path| path.display().to_string())
            .collect::<Vec<_>>()
            .join(", ");
        ktstr::ktstr_status!(
            "cargo ktstr: dropped {} of {} coverage shard(s) as unreadable; their coverage is lost: {names}",
            dropped.len(),
            inputs.len(),
        );
    }
    std::fs::rename(&staged, output).map_err(|error| {
        format!(
            "publish merged coverage profile {} -> {}: {error}",
            staged.display(),
            output.display(),
        )
    })?;
    Ok(())
}

/// Identify which merge inputs llvm-profdata rejected, from its stderr.
///
/// Under `-failure-mode=all` each unreadable input yields a
/// `warning: <path>: <reason>` line while the merge still succeeds on the
/// survivors. Match by input path (not by the reason text, which differs per
/// corruption kind) so unrelated warnings are never miscounted as drops.
fn dropped_shards_from_stderr<'a>(stderr: &str, inputs: &'a [PathBuf]) -> Vec<&'a Path> {
    inputs
        .iter()
        .filter(|input| {
            let prefix = format!("warning: {}: ", input.display());
            stderr.lines().any(|line| line.starts_with(&prefix))
        })
        .map(PathBuf::as_path)
        .collect()
}

fn compact_coverage_producer_profiles(
    current_dir: &Path,
    environment: &[(OsString, OsString)],
    profile_directory: &Path,
) -> Result<Option<PathBuf>, String> {
    let profiles = profraw_files(profile_directory)?;
    if profiles.is_empty() {
        return Ok(None);
    }
    let merged = profile_directory.join(".ktstr-coverage-producer.profdata");
    merge_profdata(current_dir, environment, &profiles, &merged, None)?;
    Ok(Some(merged))
}

fn stage_profraw_for_report(directory: &Path) -> Result<StagedProfraw, String> {
    stage_profraw_for_report_in(directory, coverage_recovery_parent()?)
}

fn stage_profraw_for_report_in(
    directory: &Path,
    recovery_parent: PathBuf,
) -> Result<StagedProfraw, String> {
    let profiles = profraw_files(directory)?;
    if profiles.is_empty() {
        return Ok(StagedProfraw {
            directory: None,
            recovery_parent,
            count: 0,
        });
    }
    let recovery = create_coverage_recovery_dir(&recovery_parent)?;
    for profile in &profiles {
        let name = profile
            .file_name()
            .ok_or_else(|| format!("coverage profile has no file name: {}", profile.display()))?;
        let destination = recovery.path().join(name);
        if let Err(error) = std::fs::rename(profile, &destination) {
            let retained = recovery.keep(&recovery_parent);
            return Err(format!(
                "stage coverage profile {} -> {}: {error}; already-staged shards retained at {}",
                profile.display(),
                destination.display(),
                retained.display(),
            ));
        }
    }
    Ok(StagedProfraw {
        directory: Some(recovery),
        recovery_parent,
        count: profiles.len(),
    })
}

impl ProfrawCleanup {
    fn record_failure(&mut self, diagnostic: impl AsRef<str>) {
        self.failures = self.failures.saturating_add(1);
        if self.diagnostics.len() >= PROFRAW_CLEANUP_DIAGNOSTIC_LIMIT {
            return;
        }
        let diagnostic = diagnostic.as_ref();
        let mut characters = diagnostic.chars();
        let mut bounded = characters
            .by_ref()
            .take(PROFRAW_CLEANUP_DIAGNOSTIC_CHARS)
            .map(|character| match character {
                '\n' | '\r' | '\t' => ' ',
                character => character,
            })
            .collect::<String>();
        if characters.next().is_some() {
            bounded.push('…');
        }
        self.diagnostics.push(bounded);
    }

    fn failure_message(&self, directory: &Path) -> Option<String> {
        if self.failures == 0 {
            return None;
        }
        let omitted = self.failures.saturating_sub(self.diagnostics.len());
        let detail = if self.diagnostics.is_empty() {
            String::new()
        } else {
            format!(": {}", self.diagnostics.join("; "))
        };
        let omitted = if omitted == 0 {
            String::new()
        } else {
            format!("; {omitted} additional failure(s) omitted")
        };
        Some(format!(
            "remove cached coverage .profraw shards under {}: {} failure(s){detail}{omitted}",
            directory.display(),
            self.failures,
        ))
    }
}

/// Remove only raw coverage shards after cargo-llvm-cov has successfully
/// merged them. Missing directories and files are normal: cargo-llvm-cov may
/// consume them itself, and concurrent cleanup may win the unlink race.
/// Symlinks, directories, merged profiles, and report output are preserved.
fn remove_cached_profraw_shards(directory: &Path) -> ProfrawCleanup {
    let mut cleanup = ProfrawCleanup::default();
    let entries = match std::fs::read_dir(directory) {
        Ok(entries) => entries,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return cleanup,
        Err(error) => {
            cleanup.record_failure(format!("read {}: {error}", directory.display()));
            return cleanup;
        }
    };
    for entry in entries {
        let entry = match entry {
            Ok(entry) => entry,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => continue,
            Err(error) => {
                cleanup.record_failure(format!("read directory entry: {error}"));
                continue;
            }
        };
        let path = entry.path();
        if path.extension() != Some(OsStr::new("profraw")) {
            continue;
        }
        let file_type = match entry.file_type() {
            Ok(file_type) => file_type,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => continue,
            Err(error) => {
                cleanup.record_failure(format!("inspect {}: {error}", path.display()));
                continue;
            }
        };
        if !file_type.is_file() {
            continue;
        }
        match std::fs::remove_file(&path) {
            Ok(()) => cleanup.removed = cleanup.removed.saturating_add(1),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => {
                cleanup.record_failure(format!("remove {}: {error}", path.display()));
            }
        }
    }
    cleanup
}

fn append_shell_word(script: &mut Vec<u8>, value: &OsStr) {
    script.push(b'\'');
    for byte in value.as_bytes() {
        if *byte == b'\'' {
            script.extend_from_slice(b"'\"'\"'");
        } else {
            script.push(*byte);
        }
    }
    script.push(b'\'');
}

fn write_retained_coverage_report_script(
    bundle: &Path,
    report_dir: &Path,
    environment: &[(OsString, OsString)],
    llvm_cov_flags: &OsStr,
    report_args: &[String],
    target_directory: &Path,
    build_directory: &Path,
) -> Result<PathBuf, String> {
    let profile_name = environment_value(environment, "LLVM_PROFILE_FILE")
        .and_then(|value| Path::new(value).file_name())
        .map_or_else(
            || OsString::from("ktstr-%p-%m.profraw"),
            OsStr::to_os_string,
        );
    let mut environment = environment
        .iter()
        .cloned()
        .collect::<std::collections::BTreeMap<_, _>>();
    environment.insert(
        OsString::from("CARGO_LLVM_COV_TARGET_DIR"),
        target_directory.as_os_str().to_os_string(),
    );
    environment.insert(
        OsString::from("CARGO_LLVM_COV_BUILD_DIR"),
        build_directory.as_os_str().to_os_string(),
    );
    environment.insert(
        OsString::from("LLVM_PROFILE_FILE"),
        target_directory.join(profile_name).into_os_string(),
    );
    environment.insert(
        OsString::from("LLVM_COV_FLAGS"),
        llvm_cov_flags.to_os_string(),
    );

    let mut script = b"#!/bin/sh\nset -eu\nexec 9>".to_vec();
    append_shell_word(
        &mut script,
        bundle.join(COVERAGE_RECOVERY_LIVE_LOCK).as_os_str(),
    );
    script.extend_from_slice(b"\nflock --exclusive 9\nrm -f ");
    append_shell_word(
        &mut script,
        bundle.join(COVERAGE_RECOVERY_SIZE_FILE).as_os_str(),
    );
    script.extend_from_slice(b"\ncd ");
    append_shell_word(&mut script, report_dir.as_os_str());
    script.push(b'\n');
    for (name, value) in &environment {
        let name_bytes = name.as_bytes();
        let valid_name = !name_bytes.is_empty()
            && (name_bytes[0].is_ascii_alphabetic() || name_bytes[0] == b'_')
            && name_bytes[1..]
                .iter()
                .all(|byte| byte.is_ascii_alphanumeric() || *byte == b'_');
        if !valid_name {
            return Err(format!(
                "coverage report environment has invalid shell variable name: {:?}",
                name,
            ));
        }
        script.extend_from_slice(b"export ");
        script.extend_from_slice(name_bytes);
        script.push(b'=');
        append_shell_word(&mut script, value);
        script.push(b'\n');
    }
    script.extend_from_slice(b"exec cargo llvm-cov");
    for argument in report_args {
        script.push(b' ');
        append_shell_word(&mut script, OsStr::new(argument));
    }
    script.push(b'\n');

    let path = bundle.join("report.sh");
    std::fs::write(&path, script)
        .map_err(|error| format!("write retained coverage report script: {error}"))?;
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o700)).map_err(|error| {
        format!(
            "make retained coverage report script executable {}: {error}",
            path.display(),
        )
    })?;
    Ok(path)
}

/// Retain a complete cached `--no-report` run for a later report.
///
/// Raw shards are merged with the cached producer profile first, then moved
/// out of cargo-llvm-cov's scan directory so a later report consumes the exact
/// merged profile instead of overwriting it. The complete instrumented
/// artifact materialization is installed beside those raw recovery inputs and
/// an executable report script records the rebased environment and report
/// argv.
fn retain_cached_coverage_for_later_report(
    coverage: &CachedCoverageReport,
    cached: crate::nextest_artifact_cache::MaterializedNextestArtifacts,
) -> Result<PathBuf, String> {
    let runtime_profraw = profraw_files(&coverage.profraw_directory)?;
    let mut inputs = Vec::with_capacity(runtime_profraw.len() + 1);
    if let Some(producer_profdata) = &coverage.producer_profdata {
        inputs.push(producer_profdata.clone());
    }
    inputs.extend(runtime_profraw);
    let merge_failure = if inputs.is_empty() {
        None
    } else {
        merge_profdata(
            &coverage.report_dir,
            &coverage.environment,
            &inputs,
            &coverage.merged_profdata,
            llvm_cov_failure_mode(&coverage.report_args),
        )
        .err()
    };
    // On a successful merge, keep raw shards beside the bundle's recovery
    // metadata so the replay target contains only the authoritative merged
    // profile. If merging failed, leave the raws inside the artifact tree;
    // preserving every input is more important than manufacturing a partial
    // bundle which cannot be retried.
    let staged = if merge_failure.is_none() {
        stage_profraw_for_report(&coverage.profraw_directory)?
    } else {
        StagedProfraw {
            directory: None,
            recovery_parent: coverage_recovery_parent()?,
            count: 0,
        }
    };
    let retained = staged
        .persist_with(&coverage.merged_profdata, true, |bundle| {
            let (target_directory, build_directory) = cached.persist_for_coverage(bundle)?;
            write_retained_coverage_report_script(
                bundle,
                &coverage.report_dir,
                &coverage.environment,
                &coverage.llvm_cov_flags,
                &coverage.report_args,
                &target_directory,
                &build_directory,
            )?;
            Ok(())
        })?
        .ok_or_else(|| "--no-report produced no persistent coverage bundle".to_string())?;
    if let Some(error) = merge_failure {
        return Err(format!(
            "pre-merge retained --no-report coverage: {error}; complete raw inputs and artifact closure retained at {}",
            retained.display(),
        ));
    }
    Ok(retained)
}

fn llvm_cov_flags_with_path_equivalence(
    existing: Option<OsString>,
    stable_workspace: &Path,
    original_workspace: &Path,
) -> Result<OsString, String> {
    let stable = stable_workspace.to_string_lossy();
    let original = original_workspace.to_string_lossy();
    if stable.contains(' ')
        || stable.contains(',')
        || original.contains(' ')
        || original.contains(',')
    {
        return Err(format!(
            "cargo ktstr: llvm-cov path equivalence cannot encode workspace paths containing spaces or commas: {} -> {}",
            stable_workspace.display(),
            original_workspace.display(),
        ));
    }
    let mut flags = existing.unwrap_or_default();
    if !flags.is_empty() {
        flags.push(" ");
    }
    flags.push(format!("--path-equivalence={stable},{original}"));
    Ok(flags)
}

fn nextest_binary_list_command(
    stable_workspace: &Path,
    output_target: &Path,
    build_args: &[String],
    release: bool,
    producer_environment: &[(OsString, OsString)],
) -> Command {
    let mut command = Command::new("cargo");
    command
        .args(["nextest", "list"])
        .current_dir(stable_workspace);
    if release {
        command.args(["--cargo-profile", "release"]);
    }
    command.args(force_nextest_output(build_args, output_target));
    command.args(["--list-type=binaries-only", "--message-format=json"]);
    apply_command_envs(&mut command, producer_environment);
    sanitize_cached_cargo_build_child_environment(&mut command);
    command
}

fn cargo_metadata_command(
    stable_workspace: &Path,
    output_target: &Path,
    build_args: &[String],
    producer_environment: &[(OsString, OsString)],
) -> Command {
    let mut options = crate::feature_discovery::metadata_passthrough_options(build_args);
    options.extend(crate::feature_discovery::metadata_resolution_options(
        build_args,
    ));
    let mut command = Command::new("cargo");
    command
        .args(["metadata", "--format-version=1"])
        .args(options)
        .current_dir(stable_workspace)
        .env("CARGO_TARGET_DIR", output_target);
    apply_command_envs(&mut command, producer_environment);
    sanitize_cached_cargo_build_child_environment(&mut command);
    command
}

fn cargo_metadata_json_from_output(output: &std::process::Output) -> Result<Vec<u8>, String> {
    if !output.status.success() {
        return Err(format!(
            "Cargo metadata for nextest reuse failed with {}: {}",
            output.status,
            String::from_utf8_lossy(&output.stderr).trim(),
        ));
    }
    Ok(output.stdout.clone())
}

fn validate_nextest_producer_pair(
    output: &crate::interrupt::CommandOutputPair,
    cli_label: &str,
) -> Result<Vec<u8>, String> {
    let build_failure = || {
        format!(
            "{cli_label}: nextest binary-only build failed ({}) — see Cargo output above",
            output
                .primary
                .status
                .code()
                .map_or("signal".to_string(), |code| code.to_string()),
        )
    };
    if output.failed_first == Some(crate::interrupt::CommandPairSide::Secondary) {
        let metadata = cargo_metadata_json_from_output(&output.secondary)?;
        if !output.primary.status.success() {
            return Err(build_failure());
        }
        return Ok(metadata);
    }
    if !output.primary.status.success() {
        return Err(build_failure());
    }
    cargo_metadata_json_from_output(&output.secondary)
}

/// Select one compile-time BTF for all guest-kernel lanes on this host/arch.
///
/// ktstr's BPF objects are CO-RE: the selected guest BTF is consumed when the
/// object is loaded, not when this harness is compiled. Prefer the running
/// host's stable sysfs BTF, then the ordinary host/local resolver, retaining
/// the selected guest only as a last-resort source on hosts without either.
fn cached_cargo_build_btf(kernel_dir: Option<&Path>) -> Option<PathBuf> {
    let host = Path::new("/sys/kernel/btf/vmlinux");
    if host.is_file() {
        return Some(host.to_path_buf());
    }
    ktstr::kernel_path::resolve_btf(None)
        .or_else(|| ktstr::kernel_path::resolve_btf(kernel_dir.and_then(Path::to_str)))
}

fn producer_environment_identity(
    environment: &[(OsString, OsString)],
) -> Result<Vec<String>, String> {
    let mut identity = Vec::with_capacity(environment.len());
    for (name, value) in environment {
        let name_text = name.to_string_lossy();
        if crate::verifier::cached_cargo_build_environment_is_runtime(name) {
            // These values shape runtime scheduling/listing, not the Cargo
            // artifact closure. Nextest coordinates describe only the current
            // test attempt. The canonical compile BTF is installed below as
            // an explicit content-digested build input.
            continue;
        }
        if name_text == "BPF_EXTRA_CFLAGS_PRE_INCL" {
            let text = value.to_string_lossy();
            let mut words = text.split_whitespace();
            let mut normalized = Vec::new();
            while let Some(word) = words.next() {
                normalized.push(word.to_string());
                if word == "-include"
                    && let Some(path) = words.next()
                {
                    let digest = ktstr::cache::content_file_digest(Path::new(path)).map_err(
                        |error| {
                            format!(
                                "cargo ktstr: digest BTF anchor {path} for harness cache identity: {error:#}",
                            )
                        },
                    )?;
                    normalized.push(format!("content:{digest:016x}"));
                }
            }
            identity.push(format!("producer-env:{name_text}={}", normalized.join(" ")));
            continue;
        }
        let value_path = Path::new(value);
        let semantic_value = if value_path.is_file() {
            let digest = ktstr::cache::content_file_digest(value_path).map_err(|error| {
                format!(
                    "cargo ktstr: digest producer input {} for {}: {error:#}",
                    value_path.display(),
                    name.to_string_lossy(),
                )
            })?;
            format!("file:{digest:016x}")
        } else {
            value.to_string_lossy().into_owned()
        };
        identity.push(format!("producer-env:{}={semantic_value}", name_text));
    }
    identity.sort();
    Ok(identity)
}

/// Build or reuse the exact binary closure for one nextest invocation.
///
/// This is the single producer used by ordinary tests, recursive verifier
/// dispatch, and llvm-cov nextest. On a miss it builds from the immutable
/// source snapshot into a private output directory, asks nextest for binary
/// metadata without executing any harness, and pins the complete closure while
/// the output lease is still held. Hits never invoke Cargo and materialize a
/// private reflink tree directly.
#[allow(clippy::too_many_arguments)]
pub(crate) fn load_or_build_nextest_artifacts(
    metadata: &cargo_metadata::Metadata,
    mode: CachedNextestMode,
    build_args: &[String],
    llvm_cov_environment_args: &[String],
    release: bool,
    output_roots: &[PathBuf],
    producer_environment: &[(OsString, OsString)],
    kernel_dir: Option<&Path>,
    cli_label: &str,
) -> Result<crate::nextest_artifact_cache::MaterializedNextestArtifacts, String> {
    // Stable producers publish complete reusable closures, never an
    // incrementally resumed compiler workspace. Incremental dep graphs are
    // both dead weight in the captured output and large enough to exhaust a
    // busy CI host. Normalize before computing identity and use the same
    // effective environment for every producer-side Cargo command.
    let mut producer_environment = stable_cargo_producer_environment(producer_environment);
    if let Some(btf) = cached_cargo_build_btf(kernel_dir) {
        set_or_replace_environment(
            &mut producer_environment,
            KTSTR_BUILD_BTF_ENV,
            btf.into_os_string(),
        );
    }
    let invocation_dir = std::env::current_dir()
        .map_err(|error| format!("read nextest producer invocation directory: {error}"))?;
    let mut identity_surface = Vec::new();
    identity_surface.push(format!("release={release}"));
    identity_surface.extend(
        llvm_cov_environment_args
            .iter()
            .map(|argument| format!("llvm-cov-env:{argument}")),
    );
    identity_surface.extend(
        build_args
            .iter()
            .map(|argument| format!("nextest:{argument}")),
    );
    identity_surface.extend(producer_environment_identity(&producer_environment)?);
    let plan = crate::nextest_artifact_cache::identity_plan(
        metadata,
        mode.identity_label(),
        &identity_surface,
        output_roots,
    )?;
    let shared_build_dir =
        crate::nextest_artifact_cache::shared_build_scratch_dir(plan.build_bucket)?;
    gc_stale_shared_build_scratch(std::time::SystemTime::now(), plan.build_bucket);
    // Named rather than inline so the retry loop below can run it twice.
    let produce =
        |stable: &crate::nextest_artifact_cache::StableCargoSource,
         stable_build: &crate::nextest_artifact_cache::StableCargoBuild| {
            let stable_invocation_dir = stable.invocation_root.clone();
            let build_args = stable.remap_cargo_args(build_args);
            let build_args = remap_cached_build_paths(
                &build_args,
                metadata.workspace_root.as_std_path(),
                &invocation_dir,
                &stable.workspace_root,
            );
            let llvm_cov_environment_args = stable.remap_cargo_args(llvm_cov_environment_args);
            let llvm_cov_environment_args = remap_cached_build_paths(
                &llvm_cov_environment_args,
                metadata.workspace_root.as_std_path(),
                &invocation_dir,
                &stable.workspace_root,
            );
            let output_target = &stable_build.target_directory;
            // Cargo builds dependency objects, fingerprints, and build-script
            // OUT_DIRs into a persistent directory shared across source digests of
            // the same non-source bucket, so unchanged dependencies are not
            // recompiled and OUT_DIR stays stable for sccache. Final artifacts still
            // land in the per-source, sealed target directory. `capture_source`
            // snapshots this build directory into the per-source materialized tree
            // (under the build-dir lease held below), so nextest's build-dir-remap
            // still resolves each test binary's OUT_DIR at execution time.
            let output_build = shared_build_dir.clone();
            let build_args = remap_nextest_store_output(
                &build_args,
                &stable.workspace_root,
                &stable_invocation_dir,
                output_target,
            )?;
            let mut environment = producer_environment.clone();
            set_or_replace_environment(
                &mut environment,
                "CARGO_BUILD_BUILD_DIR",
                output_build.as_os_str(),
            );
            if mode == CachedNextestMode::Coverage {
                environment.extend(llvm_cov_build_environment(
                    &stable_invocation_dir,
                    output_target,
                    &output_build,
                    &llvm_cov_environment_args,
                    &producer_environment,
                )?);
            }
            let metadata_command = cargo_metadata_command(
                &stable_invocation_dir,
                output_target,
                &build_args,
                &environment,
            );
            let command = nextest_binary_list_command(
                &stable_invocation_dir,
                output_target,
                &build_args,
                release,
                &environment,
            );
            // Serialize every same-bucket producer on the shared build directory,
            // then purge workspace members so this digest recompiles its own members
            // (never stale-reusing another digest's), and hold the lease through the
            // capture below which snapshots the shared directory. Acquisition order
            // is always build-dir lease then target-dir lease (inside the call), so
            // it composes with Cargo's own build-dir lock without deadlock.
            let _shared_build_lease =
                acquire_cargo_build_output_lease(&shared_build_dir, cli_label)?;
            stamp_shared_build_scratch_use(&shared_build_dir);
            purge_shared_build_dir_workspace_members(&shared_build_dir, metadata)?;
            run_reserved_build_output_pair_under_lease(
                command,
                metadata_command,
                cli_label,
                "nextest binary-only build with reusable artifact capture",
                crate::reserved_build_progress::ReservedBuildOutputKind::Opaque,
                output_target,
                |output| {
                    let cargo_metadata = validate_nextest_producer_pair(output, cli_label)?;
                    let build = &output.primary;
                    if mode == CachedNextestMode::Coverage {
                        let producer_profdata = compact_coverage_producer_profiles(
                            &stable_invocation_dir,
                            &environment,
                            output_target,
                        )?;
                        let source =
                            crate::nextest_artifact_cache::capture_source_with_producer_profdata(
                                &build.stdout,
                                &cargo_metadata,
                                producer_profdata.as_deref(),
                            )?;
                        let cleanup = remove_cached_profraw_shards(output_target);
                        if let Some(error) = cleanup.failure_message(output_target) {
                            return Err(error);
                        }
                        Ok(source)
                    } else {
                        crate::nextest_artifact_cache::capture_source(
                            &build.stdout,
                            &cargo_metadata,
                        )
                    }
                },
            )
        };
    // The test binaries resolve build-script `OUT_DIR` artifacts (notably the
    // compiled BPF `probe.o`) from this scratch bucket AT RUNTIME through
    // `env!("OUT_DIR")` absolute paths baked in at compile time — nextest's
    // build-dir remap redirects only runtime `OUT_DIR` env lookups, not baked
    // literals, so the materialized tree does not cover them. A closure whose
    // bucket was reclaimed is therefore unusable, which is why the cache
    // rejects such a record and rebuilds (see `IdentityPlan::load_or_build`).
    //
    // That check runs under the record lease, before this run holds the bucket,
    // so re-check with the runtime lease in hand and rebuild once if the bucket
    // was reclaimed in between. One retry is the bound: the rebuild repopulates
    // the bucket under the exclusive build lease, and this run's own sweeps
    // already exempt it.
    let mut attempts = 0;
    loop {
        attempts += 1;
        let mut materialized = plan.load_or_build(cli_label, produce)?;
        // The build's EXCLUSIVE lease was released when its closure returned;
        // take a SHARED lease now and hand it to the artifacts so it is held for
        // the entire nextest run. The pressure sweep and aged GC take a
        // non-blocking EXCLUSIVE lease, which fails against this SHARED hold, so
        // neither reclaims the bucket mid-run.
        let runtime_lease = acquire_cargo_build_output_lease_shared(&shared_build_dir, cli_label)?;
        // Contents, not existence: acquiring the lease just created the
        // directory if a sweep had removed it.
        if shared_build_scratch_has_build_output(&shared_build_dir)
            || attempts == BUCKET_REBUILD_ATTEMPTS
        {
            stamp_shared_build_scratch_use(&shared_build_dir);
            materialized.set_runtime_bucket_lease(runtime_lease);
            return Ok(materialized);
        }
        drop(runtime_lease);
        tracing::info!(
            bucket = %shared_build_dir.display(),
            "rebuilding cached nextest artifacts whose shared build-scratch bucket was reclaimed",
        );
    }
}

/// How many times [`load_or_build_nextest_artifacts`] may produce a closure
/// before accepting whatever shared build-scratch bucket it ends up with: one
/// ordinary attempt plus one rebuild for a bucket reclaimed under it.
const BUCKET_REBUILD_ATTEMPTS: u32 = 2;

const NEXTEST_ARCHIVE_BINARIES_METADATA: &str = "target/nextest/binaries-metadata.json";
const NEXTEST_ARCHIVE_METADATA_MAX_BYTES: u64 = 64 << 20;

#[derive(serde::Deserialize)]
#[serde(rename_all = "kebab-case")]
struct NextestArchiveBinaryMetadata {
    rust_build_meta: NextestArchiveBuildMetadata,
    rust_binaries: std::collections::BTreeMap<String, NextestArchiveBinary>,
}

#[derive(serde::Deserialize)]
#[serde(rename_all = "kebab-case")]
struct NextestArchiveBuildMetadata {
    target_directory: PathBuf,
    #[serde(default)]
    build_directory: Option<PathBuf>,
}

#[derive(serde::Deserialize)]
#[serde(rename_all = "kebab-case")]
struct NextestArchiveBinary {
    binary_path: PathBuf,
}

struct ExtractedNextestArchive {
    _owner: tempfile::TempDir,
    test_binaries: Vec<PathBuf>,
}

fn validated_archive_target_path(relative: &std::path::Path) -> Result<PathBuf, String> {
    use std::path::Component;

    if relative.as_os_str().is_empty()
        || !relative
            .components()
            .all(|component| matches!(component, Component::Normal(_)))
    {
        return Err(format!(
            "nextest archive has unsafe target-relative path: {}",
            relative.display()
        ));
    }
    Ok(PathBuf::from("target").join(relative))
}

fn nextest_archive_binary_entry_paths(
    metadata: &NextestArchiveBinaryMetadata,
) -> Result<std::collections::BTreeSet<PathBuf>, String> {
    if !metadata.rust_build_meta.target_directory.is_absolute() {
        return Err(format!(
            "nextest archive target-directory is not absolute: {}",
            metadata.rust_build_meta.target_directory.display()
        ));
    }
    let build_directory = metadata
        .rust_build_meta
        .build_directory
        .as_ref()
        .unwrap_or(&metadata.rust_build_meta.target_directory);
    if !build_directory.is_absolute() {
        return Err(format!(
            "nextest archive build-directory is not absolute: {}",
            build_directory.display()
        ));
    }
    let mut paths = std::collections::BTreeSet::new();
    for binary in metadata.rust_binaries.values() {
        let relative = binary
            .binary_path
            .strip_prefix(build_directory)
            .map_err(|_| {
                format!(
                    "nextest archive test binary {} is outside build-directory {}",
                    binary.binary_path.display(),
                    build_directory.display(),
                )
            })?;
        paths.insert(validated_archive_target_path(relative)?);
    }
    Ok(paths)
}

fn read_nextest_archive_binary_metadata(
    archive_path: &std::path::Path,
) -> Result<NextestArchiveBinaryMetadata, String> {
    use std::io::Read as _;

    let file = std::fs::File::open(archive_path).map_err(|error| {
        format!(
            "cargo ktstr: open nextest archive {}: {error}",
            archive_path.display()
        )
    })?;
    let decoder = zstd::Decoder::new(std::io::BufReader::new(file)).map_err(|error| {
        format!(
            "cargo ktstr: decode nextest archive {} as tar.zst: {error}",
            archive_path.display()
        )
    })?;
    let mut archive = tar::Archive::new(decoder);
    for entry in archive.entries().map_err(|error| {
        format!(
            "cargo ktstr: read nextest archive {}: {error}",
            archive_path.display()
        )
    })? {
        let entry = entry.map_err(|error| {
            format!(
                "cargo ktstr: read nextest archive entry from {}: {error}",
                archive_path.display()
            )
        })?;
        let path = entry.path().map_err(|error| {
            format!(
                "cargo ktstr: read nextest archive entry path from {}: {error}",
                archive_path.display()
            )
        })?;
        if path.as_ref() != std::path::Path::new(NEXTEST_ARCHIVE_BINARIES_METADATA) {
            continue;
        }
        if !entry.header().entry_type().is_file() {
            return Err(format!(
                "cargo ktstr: nextest archive metadata is not a regular file: \
                 {NEXTEST_ARCHIVE_BINARIES_METADATA}"
            ));
        }
        let declared_len = entry.size();
        if declared_len > NEXTEST_ARCHIVE_METADATA_MAX_BYTES {
            return Err(format!(
                "cargo ktstr: nextest archive binaries metadata is too large \
                 ({declared_len} bytes)"
            ));
        }
        let mut bytes = Vec::with_capacity(usize::try_from(declared_len).unwrap_or(0));
        entry
            .take(NEXTEST_ARCHIVE_METADATA_MAX_BYTES + 1)
            .read_to_end(&mut bytes)
            .map_err(|error| {
                format!(
                    "cargo ktstr: read nextest archive binaries metadata from {}: {error}",
                    archive_path.display()
                )
            })?;
        if bytes.len() as u64 > NEXTEST_ARCHIVE_METADATA_MAX_BYTES {
            return Err(
                "cargo ktstr: nextest archive binaries metadata exceeded size limit".to_string(),
            );
        }
        return serde_json::from_slice(&bytes).map_err(|error| {
            format!(
                "cargo ktstr: parse {NEXTEST_ARCHIVE_BINARIES_METADATA} \
                 from {}: {error}",
                archive_path.display()
            )
        });
    }
    Err(format!(
        "cargo ktstr: nextest archive {} has no {NEXTEST_ARCHIVE_BINARIES_METADATA}",
        archive_path.display()
    ))
}

fn validate_archive_loader_link(
    entry_path: &std::path::Path,
    target: &std::path::Path,
    hard_link: bool,
) -> Result<(), String> {
    use std::path::Component;

    let mut resolved = if hard_link {
        Vec::new()
    } else {
        entry_path
            .parent()
            .into_iter()
            .flat_map(std::path::Path::components)
            .filter_map(|component| match component {
                Component::Normal(value) => Some(value.to_os_string()),
                _ => None,
            })
            .collect::<Vec<_>>()
    };
    for component in target.components() {
        match component {
            Component::Normal(value) => resolved.push(value.to_os_string()),
            Component::CurDir => {}
            Component::ParentDir if resolved.len() > 1 => {
                resolved.pop();
            }
            Component::ParentDir | Component::RootDir | Component::Prefix(_) => {
                return Err(format!(
                    "cargo ktstr: nextest archive loader link {} escapes target tree via {}",
                    entry_path.display(),
                    target.display(),
                ));
            }
        }
    }
    if resolved
        .first()
        .is_none_or(|component| component != std::ffi::OsStr::new("target"))
    {
        return Err(format!(
            "cargo ktstr: nextest archive loader link {} targets outside target tree: {}",
            entry_path.display(),
            target.display(),
        ));
    }
    Ok(())
}

fn extract_nextest_archive_test_binaries(
    archive_path: &std::path::Path,
) -> Result<ExtractedNextestArchive, String> {
    use std::os::unix::fs::PermissionsExt as _;

    let metadata = read_nextest_archive_binary_metadata(archive_path)?;
    let wanted = nextest_archive_binary_entry_paths(&metadata)?;
    let owner = tempfile::Builder::new()
        .prefix("ktstr-nextest-archive-probe-")
        .tempdir()
        .map_err(|error| format!("cargo ktstr: create nextest archive probe directory: {error}"))?;
    let file = std::fs::File::open(archive_path).map_err(|error| {
        format!(
            "cargo ktstr: reopen nextest archive {}: {error}",
            archive_path.display()
        )
    })?;
    let decoder = zstd::Decoder::new(std::io::BufReader::new(file)).map_err(|error| {
        format!(
            "cargo ktstr: decode nextest archive {} as tar.zst: {error}",
            archive_path.display()
        )
    })?;
    let mut archive = tar::Archive::new(decoder);
    let mut extracted = std::collections::BTreeSet::new();
    let mut extracted_binaries = std::collections::BTreeSet::new();
    for entry in archive.entries().map_err(|error| {
        format!(
            "cargo ktstr: read nextest archive {}: {error}",
            archive_path.display()
        )
    })? {
        let mut entry = entry.map_err(|error| {
            format!(
                "cargo ktstr: read nextest archive entry from {}: {error}",
                archive_path.display()
            )
        })?;
        let path = entry
            .path()
            .map_err(|error| {
                format!(
                    "cargo ktstr: read nextest archive entry path from {}: {error}",
                    archive_path.display()
                )
            })?
            .into_owned();
        let is_test_binary = wanted.contains(&path);
        let under_target = path.components().next().is_some_and(|component| {
            component == std::path::Component::Normal(std::ffi::OsStr::new("target"))
        }) && path
            .components()
            .all(|component| matches!(component, std::path::Component::Normal(_)));
        if !under_target {
            continue;
        }
        let entry_type = entry.header().entry_type();
        if is_test_binary && !entry_type.is_file() {
            return Err(format!(
                "cargo ktstr: nextest archive test binary is not a regular file: {}",
                path.display()
            ));
        }
        if !entry_type.is_file() && !entry_type.is_dir() {
            let hard_link = entry_type.is_hard_link();
            if !entry_type.is_symlink() && !hard_link {
                return Err(format!(
                    "cargo ktstr: nextest archive loader entry has unsupported type: {}",
                    path.display()
                ));
            }
            let target = entry
                .link_name()
                .map_err(|error| {
                    format!(
                        "cargo ktstr: read nextest archive loader link {}: {error}",
                        path.display()
                    )
                })?
                .ok_or_else(|| {
                    format!(
                        "cargo ktstr: nextest archive loader link has no target: {}",
                        path.display()
                    )
                })?;
            validate_archive_loader_link(&path, target.as_ref(), hard_link)?;
        }
        if !extracted.insert(path.clone()) {
            let duplicate = if path == std::path::Path::new(NEXTEST_ARCHIVE_BINARIES_METADATA) {
                "binaries metadata"
            } else {
                "target"
            };
            return Err(format!(
                "cargo ktstr: nextest archive contains duplicate {duplicate} entry: {}",
                path.display()
            ));
        }
        if is_test_binary {
            extracted_binaries.insert(path.clone());
        }
        let unpacked = entry.unpack_in(owner.path()).map_err(|error| {
            format!(
                "cargo ktstr: extract nextest archive probe entry {}: {error}",
                path.display()
            )
        })?;
        if !unpacked {
            return Err(format!(
                "cargo ktstr: nextest archive probe entry escaped extraction root: {}",
                path.display()
            ));
        }
    }
    if extracted_binaries != wanted {
        let missing = wanted
            .difference(&extracted_binaries)
            .map(|path| path.display().to_string())
            .collect::<Vec<_>>()
            .join(", ");
        return Err(format!(
            "cargo ktstr: nextest archive {} is missing test binaries declared \
             by its metadata: {missing}",
            archive_path.display()
        ));
    }
    let mut test_binaries = Vec::with_capacity(wanted.len());
    for path in wanted {
        let extracted = owner.path().join(path);
        let metadata = std::fs::metadata(&extracted).map_err(|error| {
            format!(
                "cargo ktstr: stat extracted nextest test binary {}: {error}",
                extracted.display()
            )
        })?;
        if !metadata.is_file() || metadata.permissions().mode() & 0o111 == 0 {
            return Err(format!(
                "cargo ktstr: extracted nextest test binary is not executable: {}",
                extracted.display()
            ));
        }
        test_binaries.push(std::fs::canonicalize(&extracted).map_err(|error| {
            format!(
                "cargo ktstr: canonicalize extracted nextest test binary {}: {error}",
                extracted.display()
            )
        })?);
    }
    Ok(ExtractedNextestArchive {
        _owner: owner,
        test_binaries,
    })
}

// Private internal dispatch helper with a cohesive run-config arg list
// (sub-command identity + the four CLI flags + passthrough args);
// bundling into a struct would not improve clarity for a fn called from
// exactly the three sibling wrappers above.
#[allow(clippy::too_many_arguments)]
fn run_cargo_sub(
    sub_argv: &[&str],
    label: &str,
    kernel: Vec<String>,
    no_perf_mode: bool,
    no_skip_mode: bool,
    release: bool,
    profile: Option<String>,
    nextest_profile: Option<String>,
    include_eol: bool,
    nextest_metadata: Option<cargo_metadata::Metadata>,
    args: Vec<String>,
) -> Result<(), String> {
    let invocation_dir = std::env::current_dir()
        .map_err(|error| format!("cargo ktstr: read invocation directory: {error}"))?;
    // Archive reuse must be one immutable revision across metadata parsing,
    // scheduler-declaration probing, and nextest's eventual extraction. Pin
    // and publish the validated source before constructing either command,
    // rewrite the user's exact archive option to the leased CAS pathname, and
    // retain the owner through the final child.
    let report_only_llvm_cov =
        sub_argv != TEST_SUB_ARGV && llvm_cov_has_lifecycle_flag(&args, "--no-run");
    let archive_source_path = if report_only_llvm_cov {
        None
    } else {
        llvm_cov_archive_path(sub_argv, &args, &invocation_dir)?
    };
    let archive_snapshot = archive_source_path
        .as_deref()
        .map(|path| {
            ktstr::cache::snapshot_content_file(path).map_err(|error| {
                format!(
                    "cargo ktstr: snapshot nextest archive {}: {error:#}",
                    path.display()
                )
            })
        })
        .transpose()?;
    let args = if let Some(snapshot) = &archive_snapshot {
        rewrite_nextest_archive_reuse(sub_argv, args, snapshot.path())?
    } else {
        args
    };
    // Materialize and inject once for the whole invocation family. The final
    // run and the reserved no-run warm-up below are both derived from this
    // exact argv, so they cannot drift onto different nextest admission
    // policies. Raw llvm-cov modes that do not select nextest remain byte-for-
    // byte passthroughs.
    let args = inject_nextest_tool_config_with(sub_argv, args, crate::nextest_config::inject)?;
    let args = if cargo_sub_uses_nextest(sub_argv, &args) {
        normalize_nextest_admission(sub_argv, args)
    } else {
        args
    };
    let mut cmd = build_cargo_command(
        sub_argv,
        release,
        profile.as_deref(),
        nextest_profile.as_deref(),
        no_perf_mode,
        no_skip_mode,
        &args,
    );

    // Hand the child build cargo-ktstr's embedded busybox / wprof so its
    // build.rs copies them instead of re-downloading (see
    // prebuilt_blob_bin_envs). KTSTR_WPROF_PATH uses the literal name —
    // ktstr::KTSTR_WPROF_PATH_ENV is `#[cfg(feature = "wprof")]`, and
    // this propagation is a harmless no-op when wprof was not embedded.
    //
    // Kept in `blob_envs` (not applied-and-dropped) because the reserved
    // warm-up pre-build below must carry the IDENTICAL build-affecting env:
    // a differing build env would cache-miss those crates and rebuild them
    // UNRESERVED in the combined run, defeating the reservation.
    let blob_envs = prebuilt_blob_bin_envs(
        std::env::var_os(ktstr::KTSTR_BUSYBOX_PATH_ENV),
        std::env::var_os("KTSTR_WPROF_PATH"),
        nextest_metadata.as_ref(),
    );
    for (var, val) in &blob_envs {
        cmd.env(var, val);
    }

    let inherited_profraw = std::env::var_os("LLVM_PROFILE_FILE");
    let profraw_inject = profraw_inject_for(sub_argv, inherited_profraw.clone());
    if let Some(pattern) = &profraw_inject {
        cmd.env("LLVM_PROFILE_FILE", pattern);
    }

    // Empty `kernel` (flag omitted) -> empty set, auto-discovery path.
    // Non-empty but all-whitespace -> actionable bail (see
    // `kernel_set_or_bail`). Otherwise the resolved (label, dir) set.
    let resolved = kernel_set_or_bail(&kernel, include_eol)?;
    let mut kernel_commit_map = None;
    let mut encoded_kernel_list = None;
    if !resolved.is_empty() {
        // `KTSTR_KERNEL` always points at the first resolved entry
        // so downstream code that inspects the env directly (e.g.
        // budget listing's vmlinux probe in `dispatch.rs`) sees a
        // valid kernel even when running under multi-kernel.
        let first_dir = &resolved[0].1;
        tracing::debug!("cargo ktstr: using kernel {}", first_dir.display());
        cmd.env(ktstr::KTSTR_KERNEL_ENV, first_dir);
        // Mark this test invocation as cargo-ktstr-orchestrated so
        // VM-boot integration tests can distinguish "running via
        // cargo ktstr test" (resource budgets honored) from raw
        // `cargo nextest run --lib` (no concurrency cap → VM-boot
        // tests starve and fail loud with an unrelated "kill set
        // by AP" shape). See KTSTR_ORCHESTRATED_ENV doc for the
        // detection-vs-KTSTR_KERNEL discrimination rationale.
        cmd.env(ktstr::KTSTR_ORCHESTRATED_ENV, "1");

        // Resolve each kernel's build commit ONCE here, in the orchestrator,
        // and pass a `dir=commit;...` map down via
        // KTSTR_KERNEL_COMMIT so the sidecar writer skips a redundant gix
        // HEAD + dirty-walk in every per-test nextest process (that walk
        // is memoized per process but not across processes — N tests
        // would re-pay it). Keyed by the same dir string exported as
        // KTSTR_KERNEL / KTSTR_KERNEL_LIST so each sidecar can look
        // itself up. `;` joins entries, `=` splits dir from commit;
        // neither appears in a short hash (hex + optional `-dirty`), and
        // a kernel path containing either would already have broken
        // KTSTR_KERNEL_LIST's own encoding. Normal resolved cache entries use
        // the commit already captured in metadata when the kernel was built;
        // only raw source paths and legacy Local entries need a gix dirty
        // walk. The sidecar calls the same helper on a map miss, so kernels
        // with no recoverable commit consistently remain omitted.
        let commit_map = resolved
            .iter()
            .filter_map(|(_, dir)| {
                let raw = dir.display().to_string();
                let commit = ktstr::test_support::kernel_commit_for_resolved(&raw)?;
                Some(format!("{raw}={commit}"))
            })
            .collect::<Vec<_>>()
            .join(";");
        if !commit_map.is_empty() {
            cmd.env(ktstr::KTSTR_KERNEL_COMMIT_ENV, &commit_map);
            kernel_commit_map = Some(commit_map);
        }

        if resolved.len() > 1 {
            let encoded = encode_kernel_list(&resolved)?;
            ktstr::ktstr_status!(
                "cargo ktstr: fanning gauntlet across {n} kernels",
                n = resolved.len(),
            );
            cmd.env(ktstr::KTSTR_KERNEL_LIST_ENV, &encoded);
            encoded_kernel_list = Some(encoded);
        }
    }

    let target_dir_path = resolve_cargo_target_dir_for_args(&args, &invocation_dir)?;

    // BTF type anchor: if a prior build left .bpf.o files, extract
    // struct definitions from the BPF source tree and generate a
    // -include header with weak global anchors that force clang to
    // retain struct types that inlining + DCE would eliminate. The
    // anchor is cached in target/ktstr_btf_anchor.h. First build
    // has no anchor (no prior .bpf.o files); second build onward
    // always uses it. Delete the header to regenerate.
    //
    // Computed into `btf_anchor_inject` (not applied-and-dropped) so the
    // reserved warm-up below carries the same `BPF_EXTRA_CFLAGS_PRE_INCL`
    // — same build-cache-parity reason as `blob_envs` above.
    let btf_anchor_inject: Option<String> =
        generate_btf_anchor(&target_dir_path, release).map(|anchor_path| {
            let existing = std::env::var("BPF_EXTRA_CFLAGS_PRE_INCL").unwrap_or_default();
            ktstr::ktstr_status!("cargo ktstr: BTF type anchor at {}", anchor_path.display());
            format!("-include {} {existing}", anchor_path.display())
                .trim()
                .to_string()
        });
    if let Some(inject) = &btf_anchor_inject {
        cmd.env("BPF_EXTRA_CFLAGS_PRE_INCL", inject);
    }

    // Everything above is preflight: a signal should terminate immediately
    // without parking around metadata, kernel resolution, or anchor discovery.
    // From this point onward the command may acquire a compile reservation or
    // spawn the final run, so establish cleanup ownership first and expose a
    // signal caught during the transition before acquiring either resource.
    crate::interrupt::enter_cleanup_phase()
        .map_err(|error| format!("cargo ktstr: enter cleanup phase: {error}"))?;
    let _shm_cleanup = ShmCleanupGuard;

    // Every environment value the final Cargo build can observe is replayed
    // on the cache producer. File-valued inputs are content-digested by the
    // generic cache helper, so embedded blob and BTF-anchor paths do not turn
    // a checkout/cache location into identity while their bytes still do.
    let mut producer_environment = blob_envs
        .iter()
        .map(|(name, value)| (OsString::from(*name), value.clone()))
        .collect::<Vec<_>>();
    producer_environment.push((OsString::from("GIT_OPTIONAL_LOCKS"), OsString::from("0")));
    if sub_argv != TEST_SUB_ARGV {
        for name in [
            "LLVM_PROFDATA",
            "LLVM_PROFDATA_FLAGS",
            "CARGO_LLVM_PROFDATA_FLAGS",
        ] {
            if let Some(value) = std::env::var_os(name) {
                producer_environment.push((OsString::from(name), value));
            }
        }
    }
    cmd.env("GIT_OPTIONAL_LOCKS", "0");
    if sub_argv == TEST_SUB_ARGV
        && let Some(pattern) = profraw_inject
            .as_deref()
            .map(Path::as_os_str)
            .or(inherited_profraw.as_deref())
    {
        producer_environment.push((OsString::from("LLVM_PROFILE_FILE"), pattern.to_os_string()));
    }
    if let Some(inject) = &btf_anchor_inject {
        producer_environment.push((
            OsString::from("BPF_EXTRA_CFLAGS_PRE_INCL"),
            OsString::from(inject),
        ));
    }
    if no_perf_mode {
        producer_environment.push((
            OsString::from(ktstr::KTSTR_NO_PERF_MODE_ENV),
            OsString::from("1"),
        ));
    }
    if no_skip_mode {
        producer_environment.push((
            OsString::from(ktstr::KTSTR_NO_SKIP_MODE_ENV),
            OsString::from("1"),
        ));
    }
    if let Some(profile) = &profile {
        producer_environment.push((
            OsString::from(ktstr::KTSTR_SCHEDULER_PROFILE_ENV),
            OsString::from(profile),
        ));
    }
    if let Some((_, first_dir)) = resolved.first() {
        producer_environment.push((
            OsString::from(ktstr::KTSTR_KERNEL_ENV),
            first_dir.as_os_str().to_os_string(),
        ));
        producer_environment.push((
            OsString::from(ktstr::KTSTR_ORCHESTRATED_ENV),
            OsString::from("1"),
        ));
    }
    if let Some(commit_map) = &kernel_commit_map {
        producer_environment.push((
            OsString::from(ktstr::KTSTR_KERNEL_COMMIT_ENV),
            OsString::from(commit_map),
        ));
    }
    if let Some(kernel_list) = &encoded_kernel_list {
        producer_environment.push((
            OsString::from(ktstr::KTSTR_KERNEL_LIST_ENV),
            OsString::from(kernel_list),
        ));
    }

    // Reserve + cgroup-confine the harness COMPILE phase only — NOT the
    // test-running phase (each VM cell takes its own LLC reservation; a
    // reservation still held here would deadlock/starve them). Plain nextest
    // uses its supported `--no-run` compile path. cargo-llvm-cov's `--no-run`
    // is report-only, so coverage instead uses its supported
    // `nextest-archive` path: cargo-llvm-cov performs its exact clean_partial,
    // nextest builds every selected binary, then an empty-set archive filter
    // keeps the temporary archive metadata-sized. The final coverage command
    // receives `--no-clean` only for this default lifecycle and reuses the
    // identical instrumented build. User-owned retain/merge modes keep their
    // own no-clean semantics.
    //
    // acquire_build_reservation consolidates the LLC footprint (leaving whole
    // cache domains free for exclusive perf-mode reservations), while choosing
    // least-held CPUs within it and sizing elastic parallelism from unshared
    // capacity first.
    let needs_prebuild = cargo_sub_needs_reserved_prebuild(sub_argv, &args);
    let archive_reuse_path = if cargo_sub_uses_nextest(sub_argv, &args)
        && !llvm_cov_has_lifecycle_flag(&args, "--no-run")
    {
        llvm_cov_archive_path(sub_argv, &args, &invocation_dir)?
    } else {
        None
    };
    let mut cached_nextest = if needs_prebuild
        && archive_reuse_path.is_none()
        && (sub_argv == TEST_SUB_ARGV || !llvm_cov_requires_ordinary_retained_target(&args))
        && let Some(metadata) = nextest_metadata.as_ref()
    {
        let build_args = direct_nextest_build_surface_args(sub_argv, &args)?;
        let coverage_environment_args = if sub_argv == TEST_SUB_ARGV {
            Vec::new()
        } else {
            llvm_cov_show_env_args(sub_argv, &args)
        };
        let mode = if sub_argv == TEST_SUB_ARGV {
            CachedNextestMode::Plain
        } else {
            CachedNextestMode::Coverage
        };
        let materialized = load_or_build_nextest_artifacts(
            metadata,
            mode,
            &build_args,
            &coverage_environment_args,
            release,
            std::slice::from_ref(&target_dir_path),
            &producer_environment,
            resolved.first().map(|(_, directory)| directory.as_path()),
            "cargo ktstr nextest artifact cache",
        )?;
        ktstr::ktstr_status!(
            "cargo ktstr: {} reusable nextest build",
            if materialized.cache_hit() {
                "reused"
            } else {
                "published"
            },
        );
        Some(materialized)
    } else {
        None
    };
    let archive_dir =
        if needs_prebuild && cached_nextest.is_none() && sub_argv != TEST_SUB_ARGV {
            Some(tempfile::tempdir().map_err(|error| {
                format!("cargo ktstr: create coverage warm archive dir: {error}")
            })?)
        } else {
            None
        };
    let archive_probe = archive_reuse_path
        .as_deref()
        .map(|archive_path| {
            ktstr::ktstr_status!(
                "cargo ktstr: extracting test binaries for scheduler discovery from {}",
                archive_path.display()
            );
            extract_nextest_archive_test_binaries(archive_path)
        })
        .transpose()?;
    let (test_bins, pinned_test_bins) = if let Some(cached) = &cached_nextest {
        (cached.test_binaries.clone(), None)
    } else if needs_prebuild {
        let mut warm = if let Some(archive_dir) = &archive_dir {
            build_llvm_cov_archive_warm_command(
                sub_argv,
                release,
                profile.as_deref(),
                nextest_profile.as_deref(),
                no_perf_mode,
                no_skip_mode,
                &args,
                &archive_dir.path().join("warm-build.tar.zst"),
            )?
        } else {
            build_cargo_command(
                sub_argv,
                release,
                profile.as_deref(),
                nextest_profile.as_deref(),
                no_perf_mode,
                no_skip_mode,
                &prebuild_no_run_json_args(&args),
            )
        };
        for (var, val) in &blob_envs {
            warm.env(var, val);
        }
        if let Some(inject) = &btf_anchor_inject {
            warm.env("BPF_EXTRA_CFLAGS_PRE_INCL", inject);
        }
        // build.rs carries `rerun-if-env-changed=KTSTR_KERNEL` and bakes
        // `vmlinux.h` from that kernel's BTF, so the warm-up MUST see the
        // SAME KTSTR_KERNEL the combined run below sets — else build.rs
        // regenerates vmlinux.h under the run, cache-missing the whole
        // ktstr crate and recompiling it UNRESERVED (defeating the
        // warm-up). Mirrors the `cmd.env(KTSTR_KERNEL_ENV, first_dir)`
        // above; empty `resolved` (auto-discovery) sets neither, so both
        // inherit the identical process env.
        if let Some((_, first_dir)) = resolved.first() {
            warm.env(ktstr::KTSTR_KERNEL_ENV, first_dir);
        }
        let pinned =
            run_reserved_prebuild_collect_test_bins(warm, "cargo ktstr", &target_dir_path)?;
        (pinned.probe_paths(), Some(pinned))
    } else {
        (
            archive_probe
                .as_ref()
                .map_or_else(Vec::new, |archive| archive.test_binaries.clone()),
            None,
        )
    };
    let cached_scheduler_manifests = cached_nextest.as_ref().map(|cached| {
        cached
            .scheduler_stamps
            .iter()
            .map(|stamp| stamp.manifest.clone())
            .collect::<Vec<_>>()
    });
    let prepared_scheduler_artifacts = if let Some(manifests) = &cached_scheduler_manifests {
        Some(
            crate::verifier::prepare_scheduler_artifacts_from_cached_manifests(
                manifests,
                cached_nextest
                    .as_ref()
                    .expect("cached manifests came from cached nextest artifacts")
                    .stable_source(),
                profile.as_deref(),
                &args,
                &invocation_dir,
            )?,
        )
    } else if needs_prebuild || archive_probe.is_some() {
        Some(crate::verifier::prepare_scheduler_artifacts(
            &test_bins,
            profile.as_deref(),
            &args,
            &invocation_dir,
        )?)
    } else {
        None
    };
    // Descriptor-backed paths are needed only while probing declarations.
    // Release test-binary fds before the potentially long nextest run; the
    // prepared scheduler manifest owns all durable artifacts it needs.
    drop(pinned_test_bins);

    let mut cached_coverage_report = None;
    if let Some(cached) = &cached_nextest {
        let ignore_run_fail =
            sub_argv != TEST_SUB_ARGV && llvm_cov_has_lifecycle_flag(&args, "--ignore-run-fail");
        let run_args = cached_nextest_run_args(sub_argv, &args);
        let run_args = if ignore_run_fail {
            force_nextest_no_fail_fast(run_args)
        } else {
            run_args
        };
        let stable_run_args = cached.remap_cargo_args(&run_args);
        let run_args = remap_nextest_store_output(
            &stable_run_args,
            &cached.stable_workspace_root,
            &cached.stable_invocation_root,
            &target_dir_path,
        )?;
        let cached_args =
            inject_nextest_reuse_args(TEST_SUB_ARGV, run_args, &cached.reuse_build_args());
        cmd = build_cargo_command(
            TEST_SUB_ARGV,
            false,
            profile.as_deref(),
            nextest_profile.as_deref(),
            no_perf_mode,
            no_skip_mode,
            &cached_args,
        );
        let metadata = nextest_metadata
            .as_ref()
            .expect("cached nextest artifacts require resolved metadata");
        apply_command_envs(&mut cmd, &producer_environment);
        let final_run_dir = cached.writable_invocation_root.clone();
        if sub_argv == TEST_SUB_ARGV {
            if let Some(pattern) = &profraw_inject {
                cmd.env("LLVM_PROFILE_FILE", pattern);
            }
        } else {
            // cargo-llvm-cov's report path scans this directory itself rather
            // than following LLVM_PROFILE_FILE into a child directory. Keep
            // execution, report discovery, and post-merge cleanup on the
            // exact same private materialized target root.
            let profraw_directory = cached.target_directory.clone();
            std::fs::create_dir_all(&profraw_directory).map_err(|error| {
                format!(
                    "cargo ktstr: create cached coverage profile directory {}: {error}",
                    profraw_directory.display(),
                )
            })?;
            let coverage_environment_args = llvm_cov_show_env_args(sub_argv, &args);
            let coverage_environment_args = cached.remap_cargo_args(&coverage_environment_args);
            let coverage_environment_args = remap_cached_build_paths(
                &coverage_environment_args,
                metadata.workspace_root.as_std_path(),
                &invocation_dir,
                &cached.stable_workspace_root,
            );
            let coverage_environment = llvm_cov_build_environment(
                &final_run_dir,
                &cached.target_directory,
                &cached.build_directory,
                &coverage_environment_args,
                &producer_environment,
            )?;
            apply_command_envs(&mut cmd, &coverage_environment);
            cached_coverage_report = Some(CachedCoverageReport {
                report_args: cached_llvm_cov_report_args(
                    sub_argv,
                    &args,
                    release,
                    metadata,
                    &invocation_dir,
                ),
                environment: coverage_environment,
                report_dir: invocation_dir.clone(),
                llvm_cov_flags: llvm_cov_flags_with_path_equivalence(
                    std::env::var_os("LLVM_COV_FLAGS")
                        .or_else(|| std::env::var_os("CARGO_LLVM_COV_FLAGS")),
                    &cached.stable_workspace_root,
                    metadata.workspace_root.as_std_path(),
                )?,
                profraw_directory,
                producer_profdata: cached.producer_profdata.clone(),
                merged_profdata: cargo_llvm_cov_profdata_path(
                    &cached.target_directory,
                    metadata.workspace_root.as_std_path(),
                ),
                no_report: llvm_cov_has_lifecycle_flag(&args, "--no-report"),
                ignore_run_fail,
            });
        }
        cached.apply_execution_context(&mut cmd)?;
    }

    // Analyze only requirement-derived, parent-snapshotted scheduler
    // artifacts. Export the same strict manifest to ordinary tests, coverage,
    // and raw llvm-cov nextest; the owner remains alive through run_status.
    if let Some(prepared) = &prepared_scheduler_artifacts {
        cmd.env(ktstr::KTSTR_SCHEDULER_MANIFEST_ENV, &prepared.manifest_path);
    }
    // This joins all work before nextest starts, so child processes hit
    // complete atomic CAS entries instead of racing background warmers.
    precompute_cast_cache(
        prepared_scheduler_artifacts
            .as_ref()
            .map_or(&[][..], |prepared| prepared.binaries.as_slice()),
    )?;

    tracing::debug!("cargo ktstr: running {label}");
    // Capture the run-start instant BEFORE the nextest build+run so
    // the footer's mtime gate (`format_run_artifact_footer`) can
    // distinguish this run's artifacts from stale ones left in a
    // reused `{kernel}-{project_commit}` run directory. The build
    // runs first, so genuine artifacts are written well after this
    // instant.
    let run_start = std::time::SystemTime::now();
    // Stamp a per-invocation SESSION TOKEN so every child test
    // process's `pre_clear_run_dir_once` spares sidecars written THIS
    // run by peer processes. nextest is process-per-test and all
    // tests sharing one {kernel}-{project_commit} dir would otherwise
    // have a later process's pre-clear delete an earlier peer's fresh
    // .ktstr.json — silent stats loss across the suite. The value is
    // opaque (only per-invocation uniqueness matters); `run_start`
    // nanos serve, and double as the footer's mtime boundary below.
    if let Ok(d) = run_start.duration_since(std::time::UNIX_EPOCH) {
        cmd.env(ktstr::KTSTR_RUN_EPOCH_ENV, d.as_nanos().to_string());
    }
    // The shared runner creates a dedicated process group and forwards a
    // caught SIGINT/SIGTERM to it. The outer wrapper keeps the parent alive
    // through cleanup and re-raises afterward.
    let status = if cargo_sub_uses_nextest(sub_argv, &args) {
        crate::nextest_process::run_status(cmd)
    } else {
        crate::interrupt::run_status(cmd)
    }
    .map_err(|e| format!("spawn cargo {}: {e}", sub_argv.join(" ")))?;
    let mut final_success = status.success();
    let mut final_failure = (!status.success()).then(|| {
        format!(
            "cargo nextest run exited with {}",
            status
                .code()
                .map_or("signal".to_string(), |code| code.to_string()),
        )
    });
    if let Some(coverage) = &cached_coverage_report {
        if !status.success() && coverage.ignore_run_fail {
            ktstr::ktstr_status!(
                "cargo ktstr: coverage test run failed under --ignore-run-fail; {}",
                if coverage.no_report {
                    "retaining coverage inputs"
                } else {
                    "generating report"
                },
            );
            final_success = true;
            final_failure = None;
        }
        // cargo-llvm-cov can merge every shard emitted before a failed test
        // run stopped. A cached --no-report invocation retains the complete
        // COW artifact closure and a replay script in the bounded recovery
        // namespace; ordinary invocations merge/report immediately. A
        // post-run failure is secondary to an existing nextest failure and
        // must not replace its authoritative signal.
        if coverage.no_report {
            let retention = cached_nextest
                .take()
                .ok_or_else(|| {
                    "cached --no-report invocation lost its artifact closure".to_string()
                })
                .and_then(|cached| retain_cached_coverage_for_later_report(coverage, cached));
            match retention {
                Ok(path) => ktstr::ktstr_status!(
                    "cargo ktstr: retained --no-report coverage bundle at {}; run {}/report.sh to generate the report",
                    path.display(),
                    path.display(),
                ),
                Err(error) => {
                    final_success = false;
                    let retention_failure =
                        format!("retain cached --no-report coverage inputs: {error}");
                    if final_failure.is_some() {
                        ktstr::ktstr_status!(
                            "cargo ktstr: coverage retention also failed; preserving the original test failure: {retention_failure}"
                        );
                    } else {
                        final_failure = Some(retention_failure);
                    }
                }
            }
        } else {
            let runtime_profraw = profraw_files(&coverage.profraw_directory);
            let prepared = runtime_profraw.and_then(|runtime_profraw| {
                let mut inputs = Vec::with_capacity(runtime_profraw.len() + 1);
                if let Some(producer_profdata) = &coverage.producer_profdata {
                    inputs.push(producer_profdata.clone());
                }
                inputs.extend(runtime_profraw);
                merge_profdata(
                    &coverage.report_dir,
                    &coverage.environment,
                    &inputs,
                    &coverage.merged_profdata,
                    llvm_cov_failure_mode(&coverage.report_args),
                )?;
                stage_profraw_for_report(&coverage.profraw_directory)
            });
            let report_failure = match prepared {
                Ok(staged) => {
                    tracing::debug!(
                        staged = staged.count,
                        directory = %coverage.profraw_directory.display(),
                        "merged and staged cached coverage raw profiles",
                    );
                    let mut report = Command::new("cargo");
                    report
                        .arg("llvm-cov")
                        .args(&coverage.report_args)
                        .current_dir(&coverage.report_dir);
                    apply_command_envs(&mut report, &coverage.environment);
                    report.env("LLVM_COV_FLAGS", &coverage.llvm_cov_flags);
                    match crate::interrupt::run_status(report) {
                        Ok(report_status) if report_status.success() => staged.discard().err(),
                        Ok(report_status) => {
                            let recovery = coverage_recovery_suffix(
                                staged.persist(&coverage.merged_profdata),
                                "merged profile",
                            );
                            Some(format!(
                                "cargo llvm-cov report exited with {}{recovery}",
                                report_status
                                    .code()
                                    .map_or("signal".to_string(), |code| code.to_string()),
                            ))
                        }
                        Err(error) => {
                            let recovery = coverage_recovery_suffix(
                                staged.persist(&coverage.merged_profdata),
                                "merged profile",
                            );
                            Some(format!("spawn cargo llvm-cov report: {error}{recovery}"))
                        }
                    }
                }
                Err(error) => {
                    let recovery =
                        stage_profraw_for_report(&coverage.profraw_directory).and_then(|staged| {
                            let retained_profile = coverage
                                .producer_profdata
                                .as_deref()
                                .unwrap_or(&coverage.merged_profdata);
                            staged.persist(retained_profile)
                        });
                    let recovery = coverage_recovery_suffix(recovery, "available merged profile");
                    Some(format!("prepare cargo llvm-cov report: {error}{recovery}"))
                }
            };
            if let Some(report_failure) = report_failure {
                final_success = false;
                if final_failure.is_some() {
                    ktstr::ktstr_status!(
                        "cargo ktstr: coverage post-run also failed; preserving the original test failure: {report_failure}"
                    );
                } else {
                    final_failure = Some(report_failure);
                }
            }
        }
    }
    drop(archive_snapshot);
    // Surface per-test debugging artefacts: name each test that
    // FAILED this run and the concrete path to each of its artifacts
    // (failure dump, auto-repro dump, stats sidecar, wprof trace), so
    // an operator does not have to guess which `*.failure-dump.json`
    // in a reused run directory belongs to the test that just failed.
    // `format_run_artifact_footer` scans every run dir under
    // `runs_root()` and keeps only files written at/after `run_start`
    // (mtime gate) — this excludes stale artifacts from a prior run
    // at the same `{kernel}-{project_commit}` key, and captures the
    // real output dir(s) for single-kernel AND gauntlet runs without
    // re-deriving the leaf name from the orchestrator's env (which
    // carries no `KTSTR_KERNEL`, unlike the child test processes).
    let runs_root = ktstr::test_support::runs_root();
    let footer = ktstr::test_support::format_run_artifact_footer(&runs_root, run_start);
    if !footer.is_empty() {
        eprint!("{footer}");
    }
    let ignore_run_fail = cached_coverage_report
        .as_ref()
        .is_some_and(|coverage| coverage.ignore_run_fail);
    if should_render_nextest_failure_footer(final_success, status.success(), ignore_run_fail) {
        // nextest is the authoritative pass/fail signal. The footer
        // above lists per-test artifacts for failures that produced
        // them; a failure that left NO artifact — a build / vm.run
        // error, a pre-build host error (kvm probe, kernel/scheduler
        // resolve, validation), a host panic, or an unparseable guest
        // result — never reaches the dump / sidecar write sites, so it
        // has no entry above. Defer to the nextest summary for the
        // authoritative failed-test set rather than implying the
        // artifact list is exhaustive.
        eprintln!();
        ktstr::ktstr_status!(
            "cargo ktstr: nextest reported failures (see its summary above); \
             per-test artifacts for failures that produced them are listed above. \
             Artifacts under {}.",
            runs_root.display(),
        );
    }
    if final_success {
        Ok(())
    } else {
        Err(final_failure.unwrap_or_else(|| "cached nextest invocation failed".to_string()))
    }
}

/// Append `--no-run` to a `cargo nextest run` passthrough argv so the
/// reserved warm-up COMPILES every test binary but RUNS nothing, and
/// strip the run-phase flags nextest HARD-REJECTS beside `--no-run`
/// (`--fail-fast` / `--no-fail-fast` / `--max-fail <N>` — probed on
/// nextest 0.9: these three error out; the other run-phase flags,
/// `--test-threads`/`-j`/`--retries`, only warn and are left alone to
/// keep the argv-parity delta minimal). Run-phase flags never enter the
/// build fingerprint, so stripping them cannot cache-miss the combined
/// run.
///
/// User filtersets (`-E`) and positional filters are left intact:
/// `--no-run` ignores the run-selection dimension and builds the whole
/// set, so the subsequent (filtered) combined run finds every artifact
/// cached regardless of which subset it selects. `pub(crate)`: the
/// verifier dispatcher's warm-up (`verifier.rs`) builds its argv the
/// same way. An inner `--` ends nextest option parsing: `--no-run` is inserted
/// before it and the entire test-binary suffix remains opaque.
pub(crate) fn prebuild_no_run_args(args: &[String]) -> Vec<String> {
    let mut v: Vec<String> = Vec::with_capacity(args.len() + 1);
    let mut skip_value = false;
    let separator = args
        .iter()
        .position(|argument| argument == "--")
        .unwrap_or(args.len());
    for a in &args[..separator] {
        if skip_value {
            skip_value = false;
            continue;
        }
        match a.as_str() {
            "--fail-fast" | "--no-fail-fast" => continue,
            "--max-fail" => {
                // Value-taking form `--max-fail N`: drop the value too.
                skip_value = true;
                continue;
            }
            _ if a.starts_with("--max-fail=") => continue,
            _ => v.push(a.clone()),
        }
    }
    v.push("--no-run".to_string());
    v.extend_from_slice(&args[separator..]);
    v
}

/// Build the verifier warm-up argv and force Cargo's machine-readable stream.
///
/// Nextest forwards Cargo JSON to stdout only when
/// `--cargo-message-format` requests it. The verifier parent needs that stream
/// to capture the exact test executables selected by the already-scoped warm
/// build; probing through a second `cargo build --tests` would lose package and
/// feature scoping. Any user-supplied cargo message format is replaced for the
/// warm-up only (the real run keeps the user's output choice), and the option
/// is inserted before an inner `--`.
pub(crate) fn prebuild_no_run_json_args(args: &[String]) -> Vec<String> {
    let warmed = prebuild_no_run_args(args);
    let separator = warmed
        .iter()
        .position(|argument| argument == "--")
        .unwrap_or(warmed.len());
    let mut out = Vec::with_capacity(warmed.len() + 1);
    let mut index = 0;
    while index < separator {
        let argument = &warmed[index];
        if argument == "--cargo-message-format" {
            // Drop its separate value as well. A missing value remains Cargo's
            // error on the real command; the warm-up still gets a valid
            // forced format so artifact discovery is deterministic.
            index += 2;
            continue;
        }
        if argument.starts_with("--cargo-message-format=") {
            index += 1;
            continue;
        }
        out.push(argument.clone());
        index += 1;
    }
    out.push("--cargo-message-format=json-render-diagnostics".to_string());
    out.extend_from_slice(&warmed[separator..]);
    out
}

#[derive(Default)]
struct CargoJsonDiscovery {
    test_executables: Vec<PathBuf>,
    saw_build_finished: bool,
}

/// Parse Cargo JSON emitted by one reserved warm-up stream.
///
/// Integration tests identify themselves with target kind `test`; unit-test
/// harnesses built from lib/bin targets carry `profile.test = true`. Both can
/// link distributed scheduler declarations and therefore must be probed.
fn observe_cargo_json_stream(bytes: &[u8], discovery: &mut CargoJsonDiscovery) {
    for line in bytes.split(|byte| *byte == b'\n') {
        let Ok(message) = serde_json::from_slice::<serde_json::Value>(line) else {
            continue;
        };
        if message.get("reason").and_then(|value| value.as_str()) == Some("build-finished")
            && message
                .get("success")
                .is_some_and(serde_json::Value::is_boolean)
        {
            discovery.saw_build_finished = true;
            continue;
        }
        if message.get("reason").and_then(|value| value.as_str()) != Some("compiler-artifact") {
            continue;
        }
        let Some(executable) = message.get("executable").and_then(|value| value.as_str()) else {
            continue;
        };
        let target_is_test = message
            .get("target")
            .and_then(|target| target.get("kind"))
            .and_then(|kind| kind.as_array())
            .is_some_and(|kinds| kinds.iter().any(|kind| kind.as_str() == Some("test")));
        let profile_is_test = message
            .get("profile")
            .and_then(|profile| profile.get("test"))
            .and_then(|test| test.as_bool())
            == Some(true);
        if target_is_test || profile_is_test {
            discovery.test_executables.push(PathBuf::from(executable));
        }
    }
}

fn cargo_json_discovery(stdout: &[u8], stderr: &[u8]) -> CargoJsonDiscovery {
    let mut discovery = CargoJsonDiscovery::default();
    observe_cargo_json_stream(stdout, &mut discovery);
    observe_cargo_json_stream(stderr, &mut discovery);
    discovery.test_executables.sort();
    discovery.test_executables.dedup();
    discovery
}

fn validated_test_executables_from_cargo_output(
    stdout: &[u8],
    stderr: &[u8],
) -> Result<Vec<PathBuf>, ()> {
    let discovery = cargo_json_discovery(stdout, stderr);
    discovery
        .saw_build_finished
        .then_some(discovery.test_executables)
        .ok_or(())
}

#[cfg(test)]
fn test_executables_from_cargo_json(stdout: &[u8]) -> Vec<PathBuf> {
    cargo_json_discovery(stdout, &[]).test_executables
}

/// Exact warmed test executable revisions pinned while the wrapper still owns
/// their Cargo target directory.
pub(crate) struct PinnedTestExecutables {
    artifacts: Vec<ktstr::cache::PinnedContentFile>,
}

impl PinnedTestExecutables {
    /// Executable pathnames that resolve through the retained descriptors.
    ///
    /// `execve("/proc/self/fd/N", ...)` resolves the descriptor before its
    /// CLOEXEC close, so declaration probes consume the exact warmed inode
    /// even after Cargo atomically replaces the ordinary target pathname.
    pub(crate) fn probe_paths(&self) -> Vec<PathBuf> {
        self.artifacts
            .iter()
            .map(ktstr::cache::PinnedContentFile::proc_fd_path)
            .collect()
    }

    /// Map each descriptor-backed probe path to Cargo's canonical emitted
    /// pathname. Verifier ownership remains keyed to the path nextest will
    /// execute, while declaration bytes come from the pinned inode.
    pub(crate) fn probe_provenance(&self) -> HashMap<PathBuf, PathBuf> {
        self.probe_paths()
            .into_iter()
            .zip(
                self.artifacts
                    .iter()
                    .map(|artifact| artifact.source_path().to_path_buf()),
            )
            .collect()
    }
}

/// Acquire and apply the shared compile reservation to one warm command.
fn prepare_reserved_prebuild(
    warm_cmd: &mut Command,
    cli_label: &str,
) -> Result<ktstr::cli::BuildReservation, String> {
    let cpu_cap = ktstr::cli::CpuCap::resolve(None)
        .map_err(|e| format!("{cli_label}: resolve harness-build CPU cap: {e:#}"))?;
    let wait_progress = std::rc::Rc::new(std::cell::RefCell::new(
        crate::reserved_build_progress::ReservationWaitProgress::start(cli_label),
    ));
    let wait_tick = std::rc::Rc::clone(&wait_progress);
    let reservation = ktstr::cli::acquire_build_reservation_waiting_interruptible_with_progress(
        cli_label,
        cpu_cap,
        &crate::interrupt::INTERRUPTED,
        move || wait_tick.borrow_mut().tick(),
    );
    match &reservation {
        Ok(_) => wait_progress.borrow_mut().acquired(),
        Err(error) => wait_progress.borrow_mut().failed(error),
    }
    let reservation = reservation
        .map_err(|e| format!("{cli_label}: acquire harness-build reservation: {e:#}"))?;
    if crate::interrupt::caught().is_some() {
        return Err(format!(
            "{cli_label}: harness-build reservation interrupted before pre-build"
        ));
    }
    if let Some(jobs) = reservation.make_jobs() {
        warm_cmd.env("CARGO_BUILD_JOBS", jobs.to_string());
    }
    Ok(reservation)
}

/// Exclusive wrapper ownership of one canonical Cargo target directory.
///
/// Cargo already admits one writer to a target directory at a time, but its
/// internal lock ends before a parent can parse machine output and pin the
/// emitted executable pathnames. New ktstr writers extend that same
/// single-writer lifetime through descriptor pinning with this per-target-dir
/// lock. There is no legacy lock lookup or fallback namespace.
pub(crate) struct CargoBuildOutputLease {
    _lock: std::os::fd::OwnedFd,
    canonical_target_dir: PathBuf,
}

impl CargoBuildOutputLease {
    pub(crate) fn target_dir(&self) -> &std::path::Path {
        &self.canonical_target_dir
    }
}

fn cargo_build_output_lock_path(
    root: &std::path::Path,
    canonical_target_dir: &std::path::Path,
) -> PathBuf {
    use std::hash::{BuildHasher as _, Hasher as _};
    use std::os::unix::ffi::OsStrExt as _;

    let mut hasher = ahash::RandomState::with_seeds(0, 0, 0, 0).build_hasher();
    hasher.write(canonical_target_dir.as_os_str().as_bytes());
    root.join(format!("{:016x}.lock", hasher.finish()))
}

fn canonical_cargo_target_dir(target_dir: &std::path::Path) -> Result<PathBuf, String> {
    let target_dir = if target_dir.is_absolute() {
        target_dir.to_path_buf()
    } else {
        std::env::current_dir()
            .map_err(|error| format!("cargo ktstr: read invocation directory: {error}"))?
            .join(target_dir)
    };
    std::fs::create_dir_all(&target_dir).map_err(|error| {
        format!(
            "cargo ktstr: create Cargo target directory {} before output locking: {error}",
            target_dir.display()
        )
    })?;
    std::fs::canonicalize(&target_dir).map_err(|error| {
        format!(
            "cargo ktstr: canonicalize Cargo target directory {}: {error}",
            target_dir.display()
        )
    })
}

/// Remove every workspace-member and path-dependency product from the shared,
/// non-source-keyed Cargo build directory before a build.
///
/// Exactness is load-bearing. The shared build directory is reused across source
/// digests that differ only in workspace-member source content, and Cargo's
/// per-member metadata hash is source-path- and content-independent while its
/// freshness is mtime-based. If any member's fingerprint or output survives
/// here, a later digest whose materialized source mtime does not strictly exceed
/// the surviving fingerprint stale-reuses the earlier digest's member and
/// publishes the wrong bytes under the new identity. Registry dependencies
/// (`source != None`) are intentionally retained: they are byte-identical within
/// a bucket and are exactly the reuse this mechanism exists to keep. Over-purging
/// (e.g. a registry crate that happens to share a member's stem) only costs a
/// recompile and stays correct; under-purging is a silent stale-reuse bug, so the
/// match anchors on the Cargo `<stem>-<metadata-hash>` separator and never on a
/// bare prefix.
///
/// Must run while the caller holds the build-directory lease, immediately before
/// the Cargo invocation, so no concurrent same-bucket builder observes a
/// half-purged directory.
pub(crate) fn purge_shared_build_dir_workspace_members(
    build_dir: &std::path::Path,
    metadata: &cargo_metadata::Metadata,
) -> Result<(), String> {
    // Cargo does not spell a package's file stem consistently: `deps/` and
    // `incremental/` use the underscored crate name (`libktstr_macros-<hash>`),
    // while `.fingerprint/` and `build/` keep the original dashed package name
    // (`ktstr-macros-<hash>`). Matching only one form silently leaves the other
    // alive — the dashed `.fingerprint`/`build` entries hold the build-script
    // run fingerprint and its `OUT_DIR`, so missing them lets a later digest
    // keep an earlier digest's generated `OUT_DIR` (e.g. a stale BPF skeleton).
    // Purge both forms.
    let variants: std::collections::BTreeSet<String> = metadata
        .packages
        .iter()
        .filter(|package| package.source.is_none())
        .flat_map(|package| [package.name.clone(), package.name.replace('-', "_")])
        .collect();
    if variants.is_empty() {
        return Ok(());
    }
    // Cargo lays intermediates out per profile: <build_dir>/<profile>/{deps,
    // .fingerprint,build,incremental}. Custom profiles use their own name, so
    // scan every profile subdirectory rather than assuming release/debug.
    let profiles = match std::fs::read_dir(build_dir) {
        Ok(entries) => entries,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(error) => {
            return Err(format!(
                "scan shared build directory {} for member purge: {error}",
                build_dir.display()
            ));
        }
    };
    for profile in profiles {
        let profile = profile.map_err(|error| {
            format!(
                "read shared build directory entry under {}: {error}",
                build_dir.display()
            )
        })?;
        if !profile
            .file_type()
            .map(|kind| kind.is_dir())
            .unwrap_or(false)
        {
            continue;
        }
        let profile_dir = profile.path();
        for subdir in ["deps", ".fingerprint", "build", "incremental"] {
            purge_member_products_in(&profile_dir.join(subdir), &variants)?;
        }
    }
    Ok(())
}

/// Whether a Cargo intermediate filename is a product of a workspace member.
///
/// Cargo names products `<stem>-<metadata-hash>` (with an optional `lib` prefix
/// and file extension in `deps/`). `variants` holds every member's dashed and
/// underscored stem. A name matches when, after optionally dropping a `lib`
/// prefix, it begins `<variant>-` and the following hash segment (up to the
/// first `.`) is a non-empty run of hex digits. Anchoring on the `-<hash>`
/// boundary and requiring a hex hash means a member stem can never partial-match
/// a longer dependency stem that merely shares its prefix (e.g. member
/// `ktstr` never matches `ktstr-macros-<hash>`, whose residual `macros-<hash>`
/// is not hex).
fn name_is_member_product(name: &str, variants: &std::collections::BTreeSet<String>) -> bool {
    for core in [name, name.strip_prefix("lib").unwrap_or(name)] {
        for variant in variants {
            let Some(rest) = core.strip_prefix(variant) else {
                continue;
            };
            let Some(hash) = rest.strip_prefix('-') else {
                continue;
            };
            let hash = hash.split('.').next().unwrap_or(hash);
            if !hash.is_empty() && hash.bytes().all(|byte| byte.is_ascii_hexdigit()) {
                return true;
            }
        }
    }
    false
}

/// Unlink every entry in one Cargo intermediate subdirectory that belongs to a
/// workspace member (see [`name_is_member_product`]).
fn purge_member_products_in(
    directory: &std::path::Path,
    variants: &std::collections::BTreeSet<String>,
) -> Result<(), String> {
    let entries = match std::fs::read_dir(directory) {
        Ok(entries) => entries,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(error) => {
            return Err(format!(
                "scan Cargo intermediate directory {}: {error}",
                directory.display()
            ));
        }
    };
    for entry in entries {
        let entry = entry.map_err(|error| {
            format!(
                "read Cargo intermediate entry under {}: {error}",
                directory.display()
            )
        })?;
        let name = entry.file_name();
        let Some(name) = name.to_str() else {
            continue;
        };
        if !name_is_member_product(name, variants) {
            continue;
        }
        let path = entry.path();
        let is_dir = entry.file_type().map(|kind| kind.is_dir()).unwrap_or(false);
        let removal = if is_dir {
            std::fs::remove_dir_all(&path)
        } else {
            std::fs::remove_file(&path)
        };
        if let Err(error) = removal
            && error.kind() != std::io::ErrorKind::NotFound
        {
            return Err(format!(
                "purge stale workspace-member product {}: {error}",
                path.display()
            ));
        }
    }
    Ok(())
}

/// Idle age after which an unused shared build-scratch bucket may be reclaimed.
const SHARED_BUILD_SCRATCH_MAX_IDLE: std::time::Duration =
    std::time::Duration::from_secs(14 * 24 * 60 * 60);

/// Directory of per-bucket last-use stamps, a sibling of the buckets themselves.
///
/// Deliberately outside the buckets: a bucket is handed to Cargo as its
/// `build-dir`, so anything ktstr writes inside it is indistinguishable from
/// build output — including to the mtime fallback below, which would then read
/// ktstr's own stamp as evidence that a build ran. The name is not 16 hex
/// digits, so neither sweep mistakes it for a bucket.
const SHARED_BUILD_SCRATCH_STAMP_DIR: &str = ".ktstr-last-used";

/// Bucket id encoded by a shared build-scratch directory name, if it is one.
fn shared_build_scratch_bucket_id(name: &std::ffi::OsStr) -> Option<u64> {
    let name = name.to_str()?;
    if name.len() != 16 || !name.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return None;
    }
    u64::from_str_radix(name, 16).ok()
}

fn shared_build_scratch_stamp_path(bucket: &std::path::Path) -> Option<PathBuf> {
    let name = bucket.file_name()?;
    Some(
        bucket
            .parent()?
            .join(SHARED_BUILD_SCRATCH_STAMP_DIR)
            .join(name),
    )
}

/// Record that this run is using `bucket`.
///
/// Called under the bucket's build-output lease by every producer and by every
/// cache hit that reuses the bucket at runtime. Cargo's own writes cannot serve
/// as the liveness signal: on cargo 1.94.1 a rebuild moves only
/// `<bucket>/<profile>/deps` and a no-op build moves no directory mtime at all,
/// so a continuously reused bucket can still present a months-old root and
/// per-profile mtime. Best-effort: a failed stamp only leaves the bucket judged
/// by those mtimes, as it was before any stamp existed.
pub(crate) fn stamp_shared_build_scratch_use(bucket: &std::path::Path) {
    if let Err(error) = write_shared_build_scratch_stamp(bucket) {
        tracing::debug!(
            bucket = %bucket.display(),
            error = %error,
            "could not stamp shared build-scratch bucket use",
        );
    }
}

fn write_shared_build_scratch_stamp(bucket: &std::path::Path) -> std::io::Result<()> {
    let stamp = shared_build_scratch_stamp_path(bucket).ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!(
                "shared build-scratch bucket has no parent directory: {}",
                bucket.display()
            ),
        )
    })?;
    if let Some(parent) = stamp.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::File::options()
        .write(true)
        .create(true)
        .truncate(false)
        .open(&stamp)?
        .set_modified(std::time::SystemTime::now())
}

/// Whether a shared build-scratch bucket still holds the build output the
/// binaries compiled against it resolve at runtime.
///
/// Existence is not the question: taking the bucket's build-output lease
/// creates the directory, so a lease holder always finds one. Both reclaimers
/// remove the bucket whole, which leaves exactly this signature — an empty
/// directory, or none at all.
pub(crate) fn shared_build_scratch_has_build_output(bucket: &std::path::Path) -> bool {
    std::fs::read_dir(bucket).is_ok_and(|mut entries| entries.next().is_some())
}

/// Remove a reclaimed bucket together with its last-use stamp.
fn remove_shared_build_scratch_bucket(bucket: &std::path::Path) -> std::io::Result<()> {
    std::fs::remove_dir_all(bucket)?;
    if let Some(stamp) = shared_build_scratch_stamp_path(bucket)
        && let Err(error) = std::fs::remove_file(&stamp)
        && error.kind() != std::io::ErrorKind::NotFound
    {
        tracing::debug!(
            stamp = %stamp.display(),
            error = %error,
            "could not remove last-use stamp of a reclaimed shared build-scratch bucket",
        );
    }
    Ok(())
}

/// Reclaim shared, non-source-keyed Cargo build-scratch buckets that no builder
/// is using and that have been idle past [`SHARED_BUILD_SCRATCH_MAX_IDLE`].
///
/// Buckets are keyed by build configuration (features, profile, resolved
/// dependency set, toolchain, environment), not by source revision, so they are
/// few and long-lived; this reclaims only buckets left behind by a dependency,
/// toolchain, or feature change. It is lease-safe: each candidate is removed
/// only while this process holds that bucket's exclusive build-output lease
/// (the same lock builders take), so it never deletes a directory under an
/// active builder. A builder that races in afterwards simply re-creates the
/// bucket and does one cold build.
///
/// Best-effort: any per-bucket error is logged and skipped so cleanup never
/// fails a build.
///
/// `exempt_bucket` is the bucket the calling run is about to lease, exempted
/// exactly as in [`reclaim_shared_build_scratch_under_pressure`]: this sweep
/// runs before that lease is taken, so the non-blocking lock below would
/// otherwise succeed against the caller's own bucket and delete the very
/// directory it is about to build in or reuse at runtime.
/// Roots both shared build-scratch reclamation sweeps scan: the parent holding
/// every bucket, and the build-output lock root whose leases gate removal.
struct SweepRoots {
    parent: std::path::PathBuf,
    lock_root: std::path::PathBuf,
}

fn shared_build_scratch_sweep_roots() -> Option<SweepRoots> {
    let bucket_zero = crate::nextest_artifact_cache::shared_build_scratch_dir(0).ok()?;
    let parent = bucket_zero.parent()?.to_path_buf();
    let lock_root = ktstr::cache::cargo_build_output_lock_root().ok()?;
    Some(SweepRoots { parent, lock_root })
}

/// Buckets under `parent` a sweep may consider: real directories with a
/// parseable bucket id, minus the bucket the calling run is about to lease.
fn reclaimable_bucket_candidates(
    parent: &std::path::Path,
    exempt_bucket: Option<u64>,
) -> Vec<(u64, std::path::PathBuf)> {
    let Ok(entries) = std::fs::read_dir(parent) else {
        return Vec::new();
    };
    entries
        .flatten()
        .filter(|entry| entry.file_type().map(|kind| kind.is_dir()).unwrap_or(false))
        .filter_map(|entry| {
            let bucket_id = shared_build_scratch_bucket_id(&entry.file_name())?;
            (exempt_bucket != Some(bucket_id)).then(|| (bucket_id, entry.path()))
        })
        .collect()
}

pub(crate) fn gc_stale_shared_build_scratch(now: std::time::SystemTime, exempt_bucket: u64) {
    let Some(roots) = shared_build_scratch_sweep_roots() else {
        return;
    };
    gc_stale_shared_build_scratch_in(&roots.parent, &roots.lock_root, now, Some(exempt_bucket));
}

fn gc_stale_shared_build_scratch_in(
    parent: &std::path::Path,
    lock_root: &std::path::Path,
    now: std::time::SystemTime,
    exempt_bucket: Option<u64>,
) {
    for (_bucket_id, bucket) in reclaimable_bucket_candidates(parent, exempt_bucket) {
        if !shared_build_scratch_bucket_is_idle(&bucket, now) {
            continue;
        }
        let Ok(canonical) = std::fs::canonicalize(&bucket) else {
            continue;
        };
        let lock_path = cargo_build_output_lock_path(lock_root, &canonical);
        // Non-blocking: a live builder holding this lease keeps its bucket.
        let Ok(Some(_lease)) =
            ktstr::flock::try_flock(&lock_path, ktstr::flock::FlockMode::Exclusive)
        else {
            continue;
        };
        // Re-check idleness under the lease: a builder may have run between the
        // first stat and acquiring the lock.
        if !shared_build_scratch_bucket_is_idle(&bucket, now) {
            continue;
        }
        if let Err(error) = remove_shared_build_scratch_bucket(&bucket)
            && error.kind() != std::io::ErrorKind::NotFound
        {
            tracing::warn!(
                bucket = %bucket.display(),
                error = %error,
                "could not reclaim idle shared build-scratch bucket",
            );
        }
    }
}

/// When a ktstr run last used a bucket, as far as this host can tell.
///
/// The last-use stamp is authoritative when present: every producer and every
/// reusing hit writes it under the bucket's lease (see
/// [`stamp_shared_build_scratch_use`]). Buckets left by ktstr versions that
/// predate the stamp carry none and fall back to the mtimes of the bucket root
/// and its immediate children — a weak proxy for use, but the only one such a
/// bucket offers.
fn bucket_last_used(bucket: &std::path::Path) -> Option<std::time::SystemTime> {
    if let Some(stamp) = shared_build_scratch_stamp_path(bucket)
        && let Ok(modified) =
            std::fs::symlink_metadata(&stamp).and_then(|metadata| metadata.modified())
    {
        return Some(modified);
    }
    let mut newest = std::fs::symlink_metadata(bucket)
        .ok()
        .and_then(|metadata| metadata.modified().ok());
    if let Ok(children) = std::fs::read_dir(bucket) {
        for child in children.flatten() {
            if let Ok(modified) = child.metadata().and_then(|metadata| metadata.modified()) {
                newest = Some(newest.map_or(modified, |current| current.max(modified)));
            }
        }
    }
    newest
}

/// Whether a bucket's last recorded use is older than the idle threshold, used
/// by the no-pressure [`SHARED_BUILD_SCRATCH_MAX_IDLE`] sweep.
fn shared_build_scratch_bucket_is_idle(
    bucket: &std::path::Path,
    now: std::time::SystemTime,
) -> bool {
    match bucket_last_used(bucket) {
        Some(newest) => now
            .duration_since(newest)
            .map(|idle| idle >= SHARED_BUILD_SCRATCH_MAX_IDLE)
            .unwrap_or(false),
        None => false,
    }
}

/// Total bytes of the regular files under a bucket (symlinks are not followed),
/// used to estimate how much a pressure sweep reclaimed by removing it.
fn bucket_size_bytes(bucket: &std::path::Path) -> u64 {
    let mut total = 0u64;
    let mut stack = vec![bucket.to_path_buf()];
    while let Some(current) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&current) else {
            continue;
        };
        for entry in entries.flatten() {
            let Ok(kind) = entry.file_type() else {
                continue;
            };
            if kind.is_dir() {
                stack.push(entry.path());
            } else if kind.is_file()
                && let Ok(metadata) = entry.metadata()
            {
                total = total.saturating_add(metadata.len());
            }
        }
    }
    total
}

/// Reclaim lease-free shared build-scratch buckets under cold-build disk
/// pressure, oldest-idle first, until `shortfall_bytes` is covered or no
/// unleased bucket remains — regardless of [`SHARED_BUILD_SCRATCH_MAX_IDLE`].
///
/// Installed as the [`ktstr::cache::artifact_tree::BuildSpaceReclaimer`] on the
/// nextest cold-build cache: when `reserve_cold_build_space` finds measured free
/// space below the filesystem's own floor after lifecycle collection, this frees
/// the pure-cache Cargo scratch dirs that lifecycle GC never sees (a stale
/// GITHUB_SHA-era bucket, a retired toolchain/feature bucket). Worst case is one
/// cold rebuild of a reclaimed configuration. The 14-day
/// [`gc_stale_shared_build_scratch`] sweep still covers the no-pressure case.
///
/// The concurrent builds' own reservations are deliberately excluded from that
/// trigger: they forecast space those builds are about to consume, and these
/// buckets are what keep them incremental, so reacting to the forecast deletes
/// the very state that would have kept the forecast small.
///
/// `exempt_bucket` is the bucket the pending build is about to lease; it is
/// never a candidate (the build has not taken its lease yet when the reservation
/// runs, so a non-blocking lock would otherwise succeed and delete it).
pub(crate) fn reclaim_shared_build_scratch_under_pressure(
    shortfall_bytes: u64,
    exempt_bucket: u64,
) -> u64 {
    let Some(roots) = shared_build_scratch_sweep_roots() else {
        return 0;
    };
    reclaim_shared_build_scratch_under_pressure_in(
        &roots.parent,
        &roots.lock_root,
        shortfall_bytes,
        Some(exempt_bucket),
    )
}

fn reclaim_shared_build_scratch_under_pressure_in(
    parent: &std::path::Path,
    lock_root: &std::path::Path,
    shortfall_bytes: u64,
    exempt_bucket: Option<u64>,
) -> u64 {
    if shortfall_bytes == 0 {
        return 0;
    }
    // (last_used, path, bucket_id) for every reclaimable bucket except the one
    // the pending build will use. Ordered oldest-idle first so the
    // least-recently-built configurations go before hotter ones.
    let mut candidates: Vec<(std::time::SystemTime, std::path::PathBuf, u64)> =
        reclaimable_bucket_candidates(parent, exempt_bucket)
            .into_iter()
            .map(|(id, path)| {
                let last_used =
                    bucket_last_used(&path).unwrap_or(std::time::SystemTime::UNIX_EPOCH);
                (last_used, path, id)
            })
            .collect();
    candidates.sort_by(|left, right| left.0.cmp(&right.0).then_with(|| left.2.cmp(&right.2)));

    let mut reclaimed = 0u64;
    for (_, bucket, bucket_id) in candidates {
        if reclaimed >= shortfall_bytes {
            break;
        }
        let Ok(canonical) = std::fs::canonicalize(&bucket) else {
            continue;
        };
        let lock_path = cargo_build_output_lock_path(lock_root, &canonical);
        // Non-blocking: a live builder holding this lease keeps its bucket.
        let Ok(Some(_lease)) =
            ktstr::flock::try_flock(&lock_path, ktstr::flock::FlockMode::Exclusive)
        else {
            continue;
        };
        // Re-check under the lease: a builder may have taken and released this
        // bucket between the directory scan and the lock.
        if !bucket.is_dir() {
            continue;
        }
        let size = bucket_size_bytes(&bucket);
        match remove_shared_build_scratch_bucket(&bucket) {
            Ok(()) => {
                reclaimed = reclaimed.saturating_add(size);
                tracing::info!(
                    bucket = %format!("{bucket_id:016x}"),
                    reclaimed_bytes = size,
                    cumulative_reclaimed_bytes = reclaimed,
                    shortfall_bytes,
                    "reclaimed lease-free shared build-scratch bucket under cold-build disk pressure",
                );
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => {
                tracing::warn!(
                    bucket = %format!("{bucket_id:016x}"),
                    error = %error,
                    "could not reclaim shared build-scratch bucket under disk pressure",
                );
            }
        }
    }
    reclaimed
}

pub(crate) fn acquire_cargo_build_output_lease(
    target_dir: &std::path::Path,
    cli_label: &str,
) -> Result<CargoBuildOutputLease, String> {
    let root = ktstr::cache::cargo_build_output_lock_root()
        .map_err(|error| format!("{cli_label}: resolve Cargo output lock root: {error:#}"))?;
    acquire_cargo_build_output_lease_at_root(
        target_dir,
        &root,
        ktstr::flock::FlockMode::Exclusive,
        cli_label,
        &crate::interrupt::INTERRUPTED,
    )
}

/// Acquire a SHARED build-output lease on a scratch bucket, held for the whole
/// run as a runtime dependency guard (see
/// [`crate::nextest_artifact_cache::MaterializedNextestArtifacts`]'s
/// `_runtime_bucket_lease`). Multiple runs across colocated lanes coexist under
/// SHARED; a concurrent builder's EXCLUSIVE lease still parks this until the
/// build finishes, and the reclamation paths' non-blocking EXCLUSIVE attempts
/// fail against this SHARED hold, so the bucket survives while its baked
/// `env!("OUT_DIR")` artifacts are in use.
pub(crate) fn acquire_cargo_build_output_lease_shared(
    target_dir: &std::path::Path,
    cli_label: &str,
) -> Result<CargoBuildOutputLease, String> {
    let root = ktstr::cache::cargo_build_output_lock_root()
        .map_err(|error| format!("{cli_label}: resolve Cargo output lock root: {error:#}"))?;
    acquire_cargo_build_output_lease_at_root(
        target_dir,
        &root,
        ktstr::flock::FlockMode::Shared,
        cli_label,
        &crate::interrupt::INTERRUPTED,
    )
}

fn acquire_cargo_build_output_lease_at_root(
    target_dir: &std::path::Path,
    root: &std::path::Path,
    mode: ktstr::flock::FlockMode,
    cli_label: &str,
    interrupted: &std::sync::atomic::AtomicBool,
) -> Result<CargoBuildOutputLease, String> {
    let canonical_target_dir = canonical_cargo_target_dir(target_dir)?;
    std::fs::create_dir_all(root).map_err(|error| {
        format!(
            "{cli_label}: create Cargo output lock root {}: {error}",
            root.display()
        )
    })?;
    let lock_path = cargo_build_output_lock_path(root, &canonical_target_dir);
    let started = std::time::Instant::now();
    let mut next_heartbeat = started + std::time::Duration::from_secs(10);
    loop {
        if interrupted.load(std::sync::atomic::Ordering::Acquire) {
            return Err(format!(
                "{cli_label}: interrupted while waiting to own Cargo output directory {}",
                canonical_target_dir.display()
            ));
        }
        match ktstr::flock::try_flock(&lock_path, mode).map_err(|error| {
            format!(
                "{cli_label}: lock Cargo output directory {} via {}: {error:#}",
                canonical_target_dir.display(),
                lock_path.display(),
            )
        })? {
            Some(lock) => {
                return Ok(CargoBuildOutputLease {
                    _lock: lock,
                    canonical_target_dir,
                });
            }
            None => {
                let now = std::time::Instant::now();
                if now >= next_heartbeat {
                    ktstr::ktstr_status!(
                        "{cli_label}: waiting to pin Cargo artifacts from {}; elapsed={:.1}s",
                        canonical_target_dir.display(),
                        now.saturating_duration_since(started).as_secs_f64(),
                    );
                    next_heartbeat = now + std::time::Duration::from_secs(10);
                }
                std::thread::sleep(std::time::Duration::from_millis(100));
            }
        }
    }
}

/// Spawn and observe a build whose command already carries its reservation's
/// CPU placement and job count.
fn run_prepared_reserved_build_output(
    command: Command,
    cli_label: &str,
    description: &str,
    output_kind: crate::reserved_build_progress::ReservedBuildOutputKind,
) -> Result<std::process::Output, String> {
    tracing::debug!("{cli_label}: reserved {description}");
    let progress = crate::reserved_build_progress::ReservedBuildProgress::start(
        cli_label,
        description,
        output_kind,
    );
    let output = crate::interrupt::run_output_observed(command, progress)
        .map_err(|error| format!("{cli_label}: spawn {description}: {error}"))?;
    if let Err(error) = persist_reserved_build_diagnostics(&output, cli_label, description) {
        ktstr::ktstr_status!("{cli_label}: could not preserve reserved-build diagnostics: {error}");
    }
    Ok(output)
}

fn run_prepared_reserved_build_output_pair(
    primary: Command,
    secondary: Command,
    cli_label: &str,
    description: &str,
    output_kind: crate::reserved_build_progress::ReservedBuildOutputKind,
) -> Result<crate::interrupt::CommandOutputPair, String> {
    tracing::debug!("{cli_label}: reserved {description} with concurrent Cargo metadata");
    let progress = crate::reserved_build_progress::ReservedBuildProgress::start(
        cli_label,
        description,
        output_kind,
    );
    let output = crate::interrupt::run_output_pair_observed(primary, secondary, progress)
        .map_err(|error| format!("{cli_label}: spawn {description}: {error}"))?;
    if let Err(error) = persist_reserved_build_diagnostics(&output.primary, cli_label, description)
    {
        ktstr::ktstr_status!("{cli_label}: could not preserve reserved-build diagnostics: {error}");
    }
    if let Err(error) = persist_reserved_build_diagnostics(
        &output.secondary,
        cli_label,
        "concurrent Cargo metadata",
    ) {
        ktstr::ktstr_status!("{cli_label}: could not preserve Cargo-metadata diagnostics: {error}");
    }
    Ok(output)
}

const BUILD_DIAGNOSTICS_DIR_ENV: &str = "KTSTR_BUILD_DIAGNOSTICS_DIR";
const BUILD_DIAGNOSTIC_STREAM_LIMIT: usize = 16 * 1024 * 1024;
static BUILD_DIAGNOSTIC_SEQUENCE: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// Preserve the exact captured streams when CI (or an operator) requests a
/// diagnostics directory. Live stderr remains concise, while the raw Cargo
/// machine stream and wrapper diagnostics stay downloadable after the run.
fn persist_reserved_build_diagnostics(
    output: &std::process::Output,
    cli_label: &str,
    description: &str,
) -> Result<Option<PathBuf>, String> {
    let Some(root) = std::env::var_os(BUILD_DIAGNOSTICS_DIR_ENV)
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
    else {
        return Ok(None);
    };
    persist_reserved_build_diagnostics_to(&root, output, cli_label, description).map(Some)
}

fn persist_reserved_build_diagnostics_to(
    root: &std::path::Path,
    output: &std::process::Output,
    cli_label: &str,
    description: &str,
) -> Result<PathBuf, String> {
    std::fs::create_dir_all(root)
        .map_err(|error| format!("create diagnostics directory {}: {error}", root.display()))?;
    let sequence = BUILD_DIAGNOSTIC_SEQUENCE.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let stem = format!(
        "{}-{}-{}-{sequence}",
        diagnostic_filename_component(cli_label),
        diagnostic_filename_component(description),
        std::process::id(),
    );
    let stdout_path = root.join(format!("{stem}.stdout.log"));
    let stderr_path = root.join(format!("{stem}.stderr.log"));
    write_bounded_diagnostic_stream(&stdout_path, &output.stdout)?;
    write_bounded_diagnostic_stream(&stderr_path, &output.stderr)?;
    Ok(root.to_path_buf())
}

fn diagnostic_filename_component(value: &str) -> String {
    let mut component = String::with_capacity(value.len().min(64));
    let mut separator = false;
    for character in value.chars() {
        if component.len() >= 64 {
            break;
        }
        if character.is_ascii_alphanumeric() {
            component.push(character.to_ascii_lowercase());
            separator = false;
        } else if !separator && !component.is_empty() {
            component.push('-');
            separator = true;
        }
    }
    while component.ends_with('-') {
        component.pop();
    }
    if component.is_empty() {
        component.push_str("build");
    }
    component
}

fn write_bounded_diagnostic_stream(path: &std::path::Path, bytes: &[u8]) -> Result<(), String> {
    write_bounded_diagnostic_stream_with_limit(path, bytes, BUILD_DIAGNOSTIC_STREAM_LIMIT)
}

fn write_bounded_diagnostic_stream_with_limit(
    path: &std::path::Path,
    bytes: &[u8],
    limit: usize,
) -> Result<(), String> {
    use std::io::Write as _;

    debug_assert!(limit >= 2, "diagnostic stream limit must retain both ends");
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|error| format!("create {}: {error}", path.display()))?;
    if bytes.len() <= limit {
        file.write_all(bytes)
            .map_err(|error| format!("write {}: {error}", path.display()))?;
        return Ok(());
    }

    let half = limit / 2;
    let omitted = bytes.len() - (half * 2);
    file.write_all(&bytes[..half])
        .and_then(|()| {
            writeln!(
                file,
                "\n--- ktstr omitted {omitted} diagnostic bytes to keep this artifact bounded ---"
            )
        })
        .and_then(|()| file.write_all(&bytes[bytes.len() - half..]))
        .map_err(|error| format!("write bounded {}: {error}", path.display()))
}

/// Run a reserved Cargo build while extending target-dir writer ownership
/// through an exact output postprocess.
///
/// The closure runs after Cargo exits but before the per-target-dir lease is
/// released. It should parse successful Cargo JSON and open every emitted
/// artifact it needs; hashing and CAS publication can happen after this
/// returns because those file descriptors pin their inodes.
pub(crate) fn run_reserved_build_output_under_lease<T>(
    mut command: Command,
    cli_label: &str,
    description: &str,
    output_kind: crate::reserved_build_progress::ReservedBuildOutputKind,
    target_dir: &std::path::Path,
    postprocess: impl FnOnce(&std::process::Output) -> Result<T, String>,
) -> Result<T, String> {
    // A target directory is already a single-writer resource inside Cargo.
    // Own that exact directory before claiming scarce build permits so every
    // same-target waiter remains outside admission instead of hoarding CPU
    // capacity while Cargo serializes it on the artifact-directory lock.
    // Every leased build follows this one ordering.
    let (lease, reservation) = acquire_cargo_build_resources(
        || acquire_cargo_build_output_lease(target_dir, cli_label),
        || prepare_reserved_prebuild(&mut command, cli_label),
    )?;
    tracing::debug!(
        target_dir = %lease.target_dir().display(),
        "{cli_label}: acquired Cargo build-output ownership",
    );
    let output = run_prepared_reserved_build_output(command, cli_label, description, output_kind)?;
    let processed = postprocess(&output);
    drop(reservation);
    drop(lease);
    processed
}

fn run_reserved_build_output_pair_under_lease<T>(
    mut primary: Command,
    secondary: Command,
    cli_label: &str,
    description: &str,
    output_kind: crate::reserved_build_progress::ReservedBuildOutputKind,
    target_dir: &std::path::Path,
    postprocess: impl FnOnce(&crate::interrupt::CommandOutputPair) -> Result<T, String>,
) -> Result<T, String> {
    let (lease, reservation) = acquire_cargo_build_resources(
        || acquire_cargo_build_output_lease(target_dir, cli_label),
        || prepare_reserved_prebuild(&mut primary, cli_label),
    )?;
    tracing::debug!(
        target_dir = %lease.target_dir().display(),
        "{cli_label}: acquired Cargo build-output ownership",
    );
    let output = run_prepared_reserved_build_output_pair(
        primary,
        secondary,
        cli_label,
        description,
        output_kind,
    )?;
    let processed = postprocess(&output);
    drop(reservation);
    drop(lease);
    processed
}

fn acquire_cargo_build_resources<L, R>(
    acquire_lease: impl FnOnce() -> Result<L, String>,
    acquire_reservation: impl FnOnce() -> Result<R, String>,
) -> Result<(L, R), String> {
    let lease = acquire_lease()?;
    let reservation = acquire_reservation()?;
    Ok((lease, reservation))
}

/// Reserved warm-up variant used by scheduler-declaration discovery.
///
/// Captures Cargo JSON from both streams while teeing stderr live, then returns
/// the exact executable paths selected by nextest's scoped build. The
/// dual-stream parse is required because cargo-llvm-cov redirects inner Cargo
/// stdout to stderr. The compile reservation is released before the caller
/// probes those binaries or starts verifier cells.
pub(crate) fn run_reserved_prebuild_collect_test_bins(
    warm_cmd: Command,
    cli_label: &str,
    target_dir: &std::path::Path,
) -> Result<PinnedTestExecutables, String> {
    run_reserved_build_output_under_lease(
        warm_cmd,
        cli_label,
        "selected test-binary compile with Cargo artifact capture",
        crate::reserved_build_progress::ReservedBuildOutputKind::CargoJson,
        target_dir,
        |output| {
            if !output.status.success() {
                return Err(format!(
                    "{cli_label}: reserved pre-build failed ({}) — see cargo output above",
                    output
                        .status
                        .code()
                        .map_or("signal".to_string(), |code| code.to_string()),
                ));
            }
            let paths =
                validated_test_executables_from_cargo_output(&output.stdout, &output.stderr)
                    .map_err(|()| {
                        format!(
                            "{cli_label}: successful reserved pre-build emitted no Cargo \
                             build-finished message on stdout or stderr; cannot trust scheduler \
                             declaration discovery"
                        )
                    })?;
            let artifacts = paths
                .into_iter()
                .map(|path| {
                    let canonical = std::fs::canonicalize(&path).map_err(|error| {
                        format!(
                            "{cli_label}: canonicalize warmed Cargo test executable {} \
                             while target output is exclusively owned: {error}",
                            path.display(),
                        )
                    })?;
                    ktstr::cache::pin_content_file(&canonical).map_err(|error| {
                        format!(
                            "{cli_label}: pin warmed Cargo test executable {} \
                             while target output is exclusively owned: {error:#}",
                            canonical.display(),
                        )
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
            Ok(PinnedTestExecutables { artifacts })
        },
    )
}

const ITEM_PROGRESS_HEARTBEAT: std::time::Duration = std::time::Duration::from_secs(10);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ItemProgressSnapshot {
    completed: usize,
    failed: usize,
}

struct ItemProgressState {
    completed: std::sync::atomic::AtomicUsize,
    failed: std::sync::atomic::AtomicUsize,
    finished: std::sync::Mutex<bool>,
    finish_wake: std::sync::Condvar,
}

impl ItemProgressState {
    fn new() -> Self {
        Self {
            completed: std::sync::atomic::AtomicUsize::new(0),
            failed: std::sync::atomic::AtomicUsize::new(0),
            finished: std::sync::Mutex::new(false),
            finish_wake: std::sync::Condvar::new(),
        }
    }

    fn record(&self, success: bool) -> ItemProgressSnapshot {
        use std::sync::atomic::Ordering;

        if !success {
            self.failed.fetch_add(1, Ordering::Relaxed);
        }
        self.completed.fetch_add(1, Ordering::Release);
        self.snapshot()
    }

    fn snapshot(&self) -> ItemProgressSnapshot {
        use std::sync::atomic::Ordering;

        ItemProgressSnapshot {
            completed: self.completed.load(Ordering::Acquire),
            failed: self.failed.load(Ordering::Relaxed),
        }
    }

    fn wait_until_finished(&self, deadline: std::time::Instant) -> bool {
        let finished = self
            .finished
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let wait = deadline.saturating_duration_since(std::time::Instant::now());
        let (finished, _) = self
            .finish_wake
            .wait_timeout_while(finished, wait, |finished| !*finished)
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        *finished
    }

    fn mark_finished(&self) {
        let mut finished = self
            .finished
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        *finished = true;
        self.finish_wake.notify_all();
    }
}

/// Shared progress for bounded item sets whose workers may all be silent.
///
/// TTY callers receive a live item bar. Non-TTY callers receive an immediate
/// start line, ten-second heartbeats from one sleeping reporter thread, and an
/// exact terminal line. Item completion itself is only two atomic increments
/// and never serializes the Rayon workers or sequential probe loop.
pub(crate) struct ItemProgress {
    label: String,
    total: usize,
    started: std::time::Instant,
    state: std::sync::Arc<ItemProgressState>,
    tty: Option<indicatif::ProgressBar>,
    reporter: Option<std::thread::JoinHandle<()>>,
    terminal: bool,
}

impl ItemProgress {
    pub(crate) fn start(label: &str, total: usize) -> Self {
        use std::io::IsTerminal as _;
        use std::sync::Arc;

        let label = crate::reserved_build_progress::escape_free(label);
        let started = std::time::Instant::now();
        let state = Arc::new(ItemProgressState::new());
        if std::io::stderr().is_terminal() {
            let bar = indicatif::ProgressBar::new(total as u64);
            bar.set_draw_target(indicatif::ProgressDrawTarget::stderr());
            bar.set_style(
                indicatif::ProgressStyle::with_template(
                    "{spinner:.cyan} {msg} [{bar:32.cyan/blue}] {pos}/{len} \
                     [{elapsed_precise}]",
                )
                .expect("item progress template is valid"),
            );
            if !bar.is_hidden() {
                bar.set_message(label.clone());
                bar.tick();
                bar.enable_steady_tick(std::time::Duration::from_millis(80));
                return Self {
                    label,
                    total,
                    started,
                    state,
                    tty: Some(bar),
                    reporter: None,
                    terminal: false,
                };
            }
        }

        ktstr::ktstr_status!("{label}: starting; total={total}");
        let reporter_state = Arc::clone(&state);
        let reporter_label = label.clone();
        let reporter = std::thread::Builder::new()
            .name("ktstr-item-progress".to_string())
            .spawn(move || {
                let mut next_heartbeat = started + ITEM_PROGRESS_HEARTBEAT;
                loop {
                    if reporter_state.wait_until_finished(next_heartbeat) {
                        break;
                    }
                    let now = std::time::Instant::now();
                    let snapshot = reporter_state.snapshot();
                    ktstr::cli::print_status_line(&item_progress_line(
                        &reporter_label,
                        total,
                        snapshot,
                        now.saturating_duration_since(started),
                        "working",
                        None,
                    ));
                    next_heartbeat = now + ITEM_PROGRESS_HEARTBEAT;
                }
            })
            .map_err(|error| {
                ktstr::ktstr_status!(
                    "{label}: could not start progress heartbeat thread: {}",
                    crate::reserved_build_progress::escape_free(&error.to_string()),
                );
            })
            .ok();
        Self {
            label,
            total,
            started,
            state,
            tty: None,
            reporter,
            terminal: false,
        }
    }

    pub(crate) fn item_finished(&self, success: bool) {
        let snapshot = self.state.record(success);
        if let Some(bar) = &self.tty {
            bar.set_position(snapshot.completed.min(self.total) as u64);
            bar.set_message(format!("{} — {} failed", self.label, snapshot.failed,));
        }
    }

    pub(crate) fn finish_success(&mut self) {
        self.finish("completed", None);
    }

    pub(crate) fn finish_error(&mut self, error: &str) {
        self.finish("failed", Some(error));
    }

    fn finish(&mut self, phase: &str, detail: Option<&str>) {
        if self.terminal {
            return;
        }
        self.state.mark_finished();
        if let Some(reporter) = self.reporter.take() {
            let _ = reporter.join();
        }
        let message = item_progress_line(
            &self.label,
            self.total,
            self.state.snapshot(),
            self.started.elapsed(),
            phase,
            detail,
        );
        if let Some(bar) = self.tty.take() {
            if phase == "completed" {
                bar.finish_with_message(message);
            } else {
                bar.abandon_with_message(message);
            }
        } else {
            ktstr::cli::print_status_line(&message);
        }
        self.terminal = true;
    }
}

impl Drop for ItemProgress {
    fn drop(&mut self) {
        if !self.terminal {
            self.finish(
                "stopped",
                Some("progress owner dropped before terminal outcome"),
            );
        }
    }
}

fn item_progress_line(
    label: &str,
    total: usize,
    snapshot: ItemProgressSnapshot,
    elapsed: std::time::Duration,
    phase: &str,
    detail: Option<&str>,
) -> String {
    let mut line = format!(
        "{label}: {phase}; completed={}/{}; failed={}; elapsed={}",
        snapshot.completed,
        total,
        snapshot.failed,
        crate::reserved_build_progress::format_elapsed(elapsed),
    );
    if let Some(detail) = detail {
        line.push_str("; error=");
        line.push_str(&crate::reserved_build_progress::escape_free(detail));
    }
    line
}

/// Precompute cast analysis for exact declaration-derived scheduler
/// artifacts so the first test needing one does not pay the analysis cost.
pub(crate) fn precompute_cast_cache(binaries: &[std::path::PathBuf]) -> Result<(), String> {
    if binaries.is_empty() {
        return Ok(());
    }
    let mut progress = ItemProgress::start(
        "cargo ktstr: precomputing cast analysis for scheduler binaries",
        binaries.len(),
    );
    let configured_limit = std::thread::available_parallelism()
        .map(|count| count.get())
        .unwrap_or(1);
    let result = precompute_cast_paths_with_pool_builder(
        binaries,
        configured_limit,
        |threads| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .map_err(|error| error.to_string())
        },
        |path| {
            let result = ktstr::precompute_cast_analysis(path).map_err(|error| {
                format!("precompute cast analysis for {}: {error:#}", path.display())
            });
            progress.item_finished(result.is_ok());
            result
        },
    );
    match &result {
        Ok(()) => progress.finish_success(),
        Err(error) => progress.finish_error(error),
    }
    result
}

/// Analyze a prepared scheduler set under a work-sized private pool and wait
/// for every item before returning.
///
/// The join is intentional: the test processes spawned after this function
/// must see complete atomic cache entries, not race detached warmers. Zero and
/// one item stay on the caller; larger sets use
/// `min(configured_limit, binaries.len())` workers. A private-pool allocation
/// failure remains sequential and never initializes Rayon's global pool.
fn precompute_cast_paths_with_pool_builder<B, F>(
    binaries: &[std::path::PathBuf],
    configured_limit: usize,
    build_pool: B,
    precompute: F,
) -> Result<(), String>
where
    B: FnOnce(usize) -> Result<rayon::ThreadPool, String>,
    F: Fn(&std::path::Path) -> Result<(), String> + Sync,
{
    use rayon::prelude::*;

    let width = configured_limit.max(1).min(binaries.len());
    if width <= 1 {
        let mut first_error = None;
        for binary in binaries {
            if let Err(error) = precompute(binary)
                && first_error.is_none()
            {
                first_error = Some(error);
            }
        }
        return first_error.map_or(Ok(()), Err);
    }

    let results = match build_pool(width) {
        Ok(pool) => pool.install(|| {
            binaries
                .par_iter()
                .map(|binary| precompute(binary.as_path()))
                .collect::<Vec<_>>()
        }),
        Err(error) => {
            tracing::warn!(
                %error,
                width,
                work_items = binaries.len(),
                "rayon ThreadPoolBuilder failed; falling back to sequential cast precompute"
            );
            binaries
                .iter()
                .map(|binary| precompute(binary))
                .collect::<Vec<_>>()
        }
    };
    for result in results {
        result?;
    }
    Ok(())
}

fn generate_btf_anchor(target_dir: &std::path::Path, release: bool) -> Option<std::path::PathBuf> {
    let anchor_path = target_dir.join("ktstr_btf_anchor.h");
    let profile = if release { "release" } else { "debug" };
    let build_root = target_dir.join(profile).join("build");

    let mut bpf_object_dirs: Vec<PathBuf> = Vec::new();
    if let Ok(entries) = std::fs::read_dir(&build_root) {
        for entry in entries.flatten() {
            let out = entry.path().join("out");
            if out.join("bpf.bpf.o").is_file() {
                bpf_object_dirs.push(out);
            }
        }
    }
    if bpf_object_dirs.is_empty() {
        return None;
    }
    bpf_object_dirs.sort_by_key(|d| {
        std::cmp::Reverse(
            std::fs::read_dir(d)
                .map(|r| {
                    r.flatten()
                        .filter(|e| {
                            e.file_name()
                                .to_str()
                                .is_some_and(|n| n.ends_with(".bpf.o"))
                        })
                        .count()
                })
                .unwrap_or(0),
        )
    });
    let bpf_object_dir = &bpf_object_dirs[0];

    // Collect cflags and compute struct set for cache invalidation.
    let mut cflags: Vec<String> = Vec::new();
    if let Ok(base) = std::env::var("BPF_BASE_CFLAGS") {
        cflags.extend(base.split_whitespace().map(String::from));
    } else {
        cflags.extend(["-g", "-O2"].iter().map(|s| s.to_string()));
    }
    if let Ok(pre) = std::env::var("BPF_EXTRA_CFLAGS_PRE_INCL") {
        cflags.extend(pre.split_whitespace().map(String::from));
    }
    if let Ok(entries) = std::fs::read_dir(&build_root) {
        for entry in entries.flatten() {
            let bpf_h = entry.path().join("out/scx_utils-bpf_h");
            if bpf_h.is_dir() {
                cflags.push(format!("-I{}", bpf_h.display()));
            }
        }
    }
    if let Ok(post) = std::env::var("BPF_EXTRA_CFLAGS_POST_INCL") {
        cflags.extend(post.split_whitespace().map(String::from));
    }

    let clang = std::env::var("BPF_CLANG").unwrap_or_else(|_| "clang".to_string());
    crate::btf_catalog::generate_btf_anchor(bpf_object_dir, &clang, &cflags, &anchor_path)
}

fn explicit_cargo_target_dir(args: &[String], invocation_dir: &std::path::Path) -> Option<PathBuf> {
    let mut target_dir = None;
    let mut index = 0;
    while index < args.len() {
        let argument = &args[index];
        if argument == "--" {
            break;
        }
        if argument == "--target-dir" {
            if let Some(value) = args.get(index + 1) {
                target_dir = Some(PathBuf::from(value));
            }
            index += 2;
            continue;
        }
        if let Some(value) = argument.strip_prefix("--target-dir=") {
            target_dir = Some(PathBuf::from(value));
            index += 1;
            continue;
        }
        index += if llvm_cov_or_nextest_option_takes_value(argument) {
            2
        } else {
            1
        };
    }
    target_dir.map(|path| {
        if path.is_absolute() {
            path
        } else {
            invocation_dir.join(path)
        }
    })
}

pub(crate) fn resolve_cargo_target_dir_for_args(
    args: &[String],
    invocation_dir: &std::path::Path,
) -> Result<PathBuf, String> {
    if let Some(target_dir) = explicit_cargo_target_dir(args, invocation_dir) {
        return Ok(target_dir);
    }
    let mut command = cargo_metadata::MetadataCommand::new();
    command
        .cargo_path("cargo")
        .current_dir(invocation_dir)
        .other_options(crate::feature_discovery::metadata_passthrough_options(args))
        .no_deps();
    command
        .exec()
        .map(|metadata| metadata.target_directory.into_std_path_buf())
        .map_err(|error| {
            format!(
                "cargo ktstr: resolve effective Cargo target directory from {}: {error}",
                invocation_dir.display(),
            )
        })
}

fn resolve_target_dir() -> std::path::PathBuf {
    if let Ok(d) = std::env::var("CARGO_TARGET_DIR") {
        return std::path::PathBuf::from(d);
    }
    let mut metadata = Command::new("cargo");
    metadata.args(["metadata", "--format-version=1", "--no-deps"]);
    if let Ok(output) = crate::interrupt::run_output(metadata)
        && output.status.success()
        && let Ok(v) = serde_json::from_slice::<serde_json::Value>(&output.stdout)
        && let Some(dir) = v["target_directory"].as_str()
    {
        return std::path::PathBuf::from(dir);
    }
    std::path::PathBuf::from("target")
}

/// Pin [`ktstr::KTSTR_RUNS_ROOT_ENV`] to the absolute cargo target
/// dir's `ktstr` subdir so the orchestrator's footer / `stats` /
/// `replay` reads AND the child test processes' sidecar writes resolve
/// the SAME directory regardless of CWD.
///
/// Without this, [`ktstr::test_support::runs_root`] is CWD-relative
/// (`{CARGO_TARGET_DIR or "target"}/ktstr`): in a Cargo workspace the
/// test binaries run with CWD = the package dir (nextest), writing
/// sidecars to `{package}/target/ktstr`, while this orchestrator —
/// invoked from a different CWD (e.g. the workspace root) — scans
/// elsewhere, so the post-run footer finds nothing.
///
/// Resolved ONCE here (a single `cargo metadata` via
/// [`resolve_target_dir`]) and exported so child test processes
/// inherit it; they never re-run `cargo metadata` (it would be one
/// subprocess spawn per test process on the hot path). A relative
/// target dir (a relative `CARGO_TARGET_DIR`, or the `"target"`
/// fallback when `cargo metadata` is unavailable) is anchored to this
/// process's cwd so the exported value is always absolute — cargo
/// resolves a relative `CARGO_TARGET_DIR` against the cargo-invocation
/// cwd, which is this orchestrator's cwd. No-ops only when already set
/// non-empty (operator / test override) or when the cwd cannot be read.
///
/// SAFETY: called from `cargo-ktstr` `main` before any thread is
/// spawned (alongside `blobs::install_env`), so the `set_var` has no
/// concurrent env reader — see `blobs::install_env`'s safety doc.
pub(crate) fn install_runs_root_env() {
    if std::env::var_os(ktstr::KTSTR_RUNS_ROOT_ENV)
        .filter(|v| !v.is_empty())
        .is_some()
    {
        return;
    }
    let runs_root = resolve_target_dir().join("ktstr");
    let runs_root = if runs_root.is_absolute() {
        runs_root
    } else {
        // A relative root would leave the orchestrator and the child
        // test processes resolving it against DIFFERENT cwds in a
        // workspace (nextest runs each test with cwd = its package dir),
        // reintroducing the empty-footer split. Anchor it to this
        // process's cwd: cargo resolves a relative CARGO_TARGET_DIR
        // against the cargo-invocation cwd, which is this orchestrator's
        // cwd (build_cargo_command spawns cargo without overriding
        // current_dir), so this matches where the artifacts land.
        let Ok(cwd) = std::env::current_dir() else {
            return;
        };
        cwd.join(runs_root)
    };
    // SAFETY: see the function doc — startup, before any threads.
    unsafe {
        std::env::set_var(ktstr::KTSTR_RUNS_ROOT_ENV, &runs_root);
    }
}

/// A prior CI run's sidecar directory is kept for at least this long past its
/// last write before a later run may reclaim it. The `run_id`/`run_attempt`
/// ordering already excludes every newer or concurrent run; this idle window
/// is the secondary guard for an *older*-numbered run that overlapped and is
/// still executing on this persistent runner (jobs queue and run
/// concurrently). It comfortably exceeds a single ktstr CI job's wall time
/// (kernel build + scheduler + gauntlet) while bounding accumulation to a few
/// hours of small forensic directories.
const PRIOR_RUN_SIDECAR_GRACE: std::time::Duration = std::time::Duration::from_secs(6 * 60 * 60);

/// Parse the `<run_id>-<run_attempt>` prefix a CI sidecar directory leaf embeds
/// (`.github/workflows/ci.yml` sets `KTSTR_SIDECAR_DIR` to
/// `<run_id>-<run_attempt>-<lane>`). Returns `None` for any leaf that does not
/// begin with two integer, dash-separated fields, so a non-CI operator override
/// never parses into an orderable run coordinate.
fn parse_sidecar_run_coordinate(leaf: &str) -> Option<(u64, u64)> {
    let mut fields = leaf.splitn(3, '-');
    let run_id = fields.next()?.parse::<u64>().ok()?;
    let attempt = fields.next()?.parse::<u64>().ok()?;
    // A bare `<run_id>-<attempt>` with no lane suffix is not the CI shape.
    fields.next()?;
    Some((run_id, attempt))
}

/// Best-effort removal of *prior* CI runs' sidecar directories that share this
/// run's `KTSTR_SIDECAR_DIR` parent on a persistent runner.
///
/// CI stamps `KTSTR_SIDECAR_DIR` = `<workspace>/target/ktstr-ci-artifacts/`
/// `<run_id>-<run_attempt>-<lane>`; each run's forensics are uploaded as
/// artifacts by that run's own `upload-artifact` step, so once a run is over
/// its directory is pure accumulated noise. Nothing else deletes these, so a
/// busy runner grows one tree per (run, attempt, lane) without bound.
///
/// This is deliberately CI-scoped: it runs only under GitHub Actions
/// (`GITHUB_ACTIONS`) against directory leaves that parse as the
/// `<run_id>-<attempt>-<lane>` layout. Local/default runs never produce per-run
/// sidecar directories — they write to the `{kernel}-{project_commit}`
/// last-writer-wins run directory under `runs_root()`, which is bounded by
/// construction — and an operator who points `KTSTR_SIDECAR_DIR` at a
/// hand-chosen path owns its contents (mirroring the write path, which skips
/// pre-clear on an override for the same reason). Neither case is touched here.
///
/// Safety against concurrent lanes and overlapping runs:
/// - A sibling sharing this run's exact `(run_id, run_attempt)` is a
///   concurrent lane of the SAME run and is never removed.
/// - Only strictly older `(run_id, run_attempt)` siblings are candidates, so a
///   newer or concurrent *different* run (higher `run_id`) is never touched.
/// - A candidate is removed only after it has been idle past
///   [`PRIOR_RUN_SIDECAR_GRACE`], guarding an older-numbered run that queued
///   behind this one and is still executing.
///
/// Best-effort throughout: a missing parent, unparsable leaf, or per-entry
/// error is skipped so startup never fails on cleanup.
pub(crate) fn prune_prior_ci_sidecar_dirs() {
    if std::env::var_os("GITHUB_ACTIONS").is_none() {
        return;
    }
    let Some(sidecar_dir) =
        std::env::var_os(ktstr::KTSTR_SIDECAR_DIR_ENV).filter(|value| !value.is_empty())
    else {
        return;
    };
    let sidecar_dir = PathBuf::from(sidecar_dir);
    let (Some(parent), Some(current_leaf)) = (
        sidecar_dir.parent(),
        sidecar_dir.file_name().and_then(|leaf| leaf.to_str()),
    ) else {
        return;
    };
    prune_prior_ci_sidecar_dirs_in(parent, current_leaf, std::time::SystemTime::now());
}

fn prune_prior_ci_sidecar_dirs_in(parent: &Path, current_leaf: &str, now: std::time::SystemTime) {
    let Some(current) = parse_sidecar_run_coordinate(current_leaf) else {
        return;
    };
    let entries = match std::fs::read_dir(parent) {
        Ok(entries) => entries,
        Err(_) => return,
    };
    for entry in entries.flatten() {
        if !entry.file_type().map(|kind| kind.is_dir()).unwrap_or(false) {
            continue;
        }
        let Some(leaf) = entry.file_name().to_str().map(str::to_owned) else {
            continue;
        };
        let Some(coordinate) = parse_sidecar_run_coordinate(&leaf) else {
            continue;
        };
        // Never a same-run concurrent lane, and never a newer/concurrent run.
        if coordinate >= current {
            continue;
        }
        let path = entry.path();
        if !sidecar_dir_is_idle(&path, now) {
            continue;
        }
        if let Err(error) = std::fs::remove_dir_all(&path)
            && error.kind() != std::io::ErrorKind::NotFound
        {
            tracing::debug!(
                dir = %path.display(),
                error = %error,
                "could not reclaim prior CI run sidecar directory",
            );
        }
    }
}

/// Whether a sidecar directory's newest recorded write (its own mtime and its
/// immediate children's) is older than [`PRIOR_RUN_SIDECAR_GRACE`].
fn sidecar_dir_is_idle(dir: &Path, now: std::time::SystemTime) -> bool {
    let mut newest = std::fs::symlink_metadata(dir)
        .ok()
        .and_then(|metadata| metadata.modified().ok());
    if let Ok(children) = std::fs::read_dir(dir) {
        for child in children.flatten() {
            if let Ok(modified) = child.metadata().and_then(|metadata| metadata.modified()) {
                newest = Some(newest.map_or(modified, |current| current.max(modified)));
            }
        }
    }
    match newest {
        Some(newest) => now
            .duration_since(newest)
            .map(|idle| idle >= PRIOR_RUN_SIDECAR_GRACE)
            .unwrap_or(false),
        None => false,
    }
}

struct ShmCleanupGuard;

impl Drop for ShmCleanupGuard {
    fn drop(&mut self) {
        cleanup_shm();
    }
}

fn cleanup_shm() {
    let Ok(dir) = std::fs::read_dir("/dev/shm") else {
        return;
    };
    for entry in dir.flatten() {
        let name = entry.file_name();
        let Some(name_str) = name.to_str() else {
            continue;
        };
        if !name_str.starts_with("ktstr-base-")
            && !name_str.starts_with("ktstr-lz4-")
            && !name_str.starts_with("ktstr-gz-")
        {
            continue;
        }
        let shm_name = format!("/{name_str}");
        let Ok(fd) = rustix::shm::open(
            shm_name.as_str(),
            rustix::shm::OFlags::RDONLY,
            rustix::fs::Mode::empty(),
        ) else {
            continue;
        };
        if rustix::fs::flock(&fd, rustix::fs::FlockOperation::NonBlockingLockExclusive).is_err() {
            continue;
        }
        let _ = rustix::shm::unlink(shm_name.as_str());
        let _ = rustix::fs::flock(&fd, rustix::fs::FlockOperation::Unlock);
    }
}

/// Outcome of comparing the cargo-ktstr CLI's own ktstr version with the
/// `ktstr` dependency version the test project was built against.
#[derive(Debug, PartialEq, Eq)]
enum VersionGuard {
    /// Versions match — proceed silently.
    Ok,
    /// Versions differ but the test's ktstr is OLDER than the CLI —
    /// usually drivable, but the skew is worth surfacing.
    Warn(String),
    /// The test's ktstr is NEWER than the CLI — the CLI predates the API
    /// the test was built against and cannot drive it; abort.
    Error(String),
}

/// Pure CLI-vs-test ktstr-version comparison — the testable core of the
/// version guard. Compares by semver PRECEDENCE (major.minor.patch +
/// pre-release, IGNORING build metadata): ktstr is pre-1.0, where a minor
/// bump is breaking, so any test > cli aborts.
fn version_guard(cli: &Version, test: &Version) -> VersionGuard {
    use std::cmp::Ordering;
    // `cmp_precedence`, NOT the derived `Version::cmp`: the derived `Ord`
    // includes build metadata as a final tie-breaker (and
    // `BuildMetadata::EMPTY < non-empty`), so a `+build`-tagged path/git
    // ktstr dep — the `cargo install --path .` flow this guard recommends —
    // would compare unequal to a plain-version CLI and spuriously
    // abort/warn two identical releases. Precedence ignores build metadata.
    match test.cmp_precedence(cli) {
        Ordering::Equal => VersionGuard::Ok,
        Ordering::Less => VersionGuard::Warn(format!(
            "the test was built against ktstr {test} but the cargo-ktstr CLI \
             is {cli} — version skew. Align them (bump the test's ktstr \
             dependency, or use a matching CLI) to avoid surprises."
        )),
        Ordering::Greater => VersionGuard::Error(format!(
            "the test was built against ktstr {test} but the cargo-ktstr CLI \
             is {cli} — the CLI is older than the ktstr the test depends on \
             and cannot drive it. Upgrade the CLI: `cargo install ktstr` \
             (or `cargo install --path .` in the ktstr repo)."
        )),
    }
}

/// The `ktstr` version the about-to-run test/bin targets actually LINK,
/// resolved from the `cargo metadata` graph — NOT a `.max()` over all
/// `ktstr` packages. ktstr is pre-1.0, so two `0.x` are
/// semver-incompatible and cargo keeps BOTH in a dual-ktstr graph; a
/// `.max()` would pick a higher TRANSITIVE ktstr the run's targets do not
/// link and false-abort a compatible run — violating the contract below
/// ("never block a run it cannot assess").
///
/// `None` (→ guard skips) when there is no resolve graph (`--no-deps`), or
/// the root is not `ktstr` and has no linked `ktstr` dep. The root package
/// itself being `ktstr` (the in-repo workspace, whose own bins/tests link
/// itself) yields its own version. cargo metadata's resolve graph is
/// per-PACKAGE, not per-target, so "which ktstr the test targets link"
/// reduces to the root package's `ktstr` lib edge: `deps` (rename-aware),
/// Normal/Development kind (a pure build-dep is not linked by tests).
///
#[cfg(test)]
fn linked_ktstr_versions<'metadata>(
    meta: &'metadata cargo_metadata::Metadata,
    root_id: &cargo_metadata::PackageId,
) -> Vec<&'metadata Version> {
    linked_ktstr_versions_for_context(meta, root_id, None)
}

fn linked_ktstr_versions_for_context<'metadata>(
    meta: &'metadata cargo_metadata::Metadata,
    root_id: &cargo_metadata::PackageId,
    target: Option<&crate::feature_discovery::TargetContext>,
) -> Vec<&'metadata Version> {
    let Some(resolve) = meta.resolve.as_ref() else {
        return Vec::new();
    };
    let Some(root_pkg) = meta.packages.iter().find(|package| &package.id == root_id) else {
        return Vec::new();
    };
    // The root package may BE ktstr (the in-repo workspace).
    if root_pkg.name == "ktstr" {
        return vec![&root_pkg.version];
    }
    // Else: the ktstr the root's bins/tests link is its `ktstr` dep edge.
    let Some(node) = resolve.nodes.iter().find(|node| &node.id == root_id) else {
        return Vec::new();
    };
    let mut versions = Vec::new();
    for dep in &node.deps {
        let Some(dep_pkg) = meta.packages.iter().find(|package| package.id == dep.pkg) else {
            continue;
        };
        if dep_pkg.name != "ktstr" {
            continue;
        }
        // A pure build-dependency edge is not linked by the test/bin
        // targets; require a Normal/Development kind. Empty dep_kinds
        // (older cargo metadata) → treat as a normal link.
        let linked = dep.dep_kinds.is_empty()
            || dep.dep_kinds.iter().any(|kind| {
                matches!(
                    kind.kind,
                    cargo_metadata::DependencyKind::Normal
                        | cargo_metadata::DependencyKind::Development
                ) && kind.target.as_ref().is_none_or(|platform| {
                    target.is_none_or(|target| target.matches_platform(platform))
                })
            });
        if linked {
            versions.push(&dep_pkg.version);
        }
    }
    versions
}

#[cfg(test)]
fn linked_ktstr_version<'metadata>(
    meta: &'metadata cargo_metadata::Metadata,
    root_id: &cargo_metadata::PackageId,
) -> Option<&'metadata Version> {
    linked_ktstr_versions(meta, root_id).into_iter().next()
}

#[cfg(test)]
fn resolved_ktstr_version(meta: &cargo_metadata::Metadata) -> Option<&Version> {
    let resolve = meta.resolve.as_ref()?;
    // Preserve the historical root lookup for the focused graph tests below.
    // The command guard uses `selected_resolved_ktstr_versions`, which follows
    // the actual Cargo package selection instead of returning the first member
    // of a virtual workspace.
    match &resolve.root {
        Some(root) => linked_ktstr_version(meta, root),
        None => meta
            .workspace_members
            .iter()
            .find_map(|root| linked_ktstr_version(meta, root)),
    }
}

/// Every distinct ktstr version linked by Cargo's selected workspace members.
#[cfg(test)]
fn selected_resolved_ktstr_versions<'metadata>(
    meta: &'metadata cargo_metadata::Metadata,
    args: &[String],
) -> Vec<&'metadata Version> {
    selected_resolved_ktstr_versions_for_context(meta, args, None)
}

#[cfg(test)]
fn selected_resolved_ktstr_versions_for_context<'metadata>(
    meta: &'metadata cargo_metadata::Metadata,
    args: &[String],
    target: Option<&crate::feature_discovery::TargetContext>,
) -> Vec<&'metadata Version> {
    let Some(packages) = crate::feature_discovery::selected_workspace_packages(meta, args) else {
        // Malformed/unsupported selection is left for Cargo to diagnose.
        return Vec::new();
    };
    selected_resolved_ktstr_versions_from_packages(meta, packages, target)
}

fn selected_resolved_ktstr_versions_for_context_at<'metadata>(
    meta: &'metadata cargo_metadata::Metadata,
    args: &[String],
    target: Option<&crate::feature_discovery::TargetContext>,
    invocation_dir: &Path,
) -> Vec<&'metadata Version> {
    let Some(packages) = selected_workspace_packages_for_invocation(meta, args, invocation_dir)
    else {
        // Malformed/unsupported selection is left for Cargo to diagnose.
        return Vec::new();
    };
    selected_resolved_ktstr_versions_from_packages(meta, packages, target)
}

fn selected_resolved_ktstr_versions_from_packages<'metadata>(
    meta: &'metadata cargo_metadata::Metadata,
    packages: Vec<&'metadata cargo_metadata::Package>,
    target: Option<&crate::feature_discovery::TargetContext>,
) -> Vec<&'metadata Version> {
    let mut versions = packages
        .into_iter()
        .flat_map(|package| linked_ktstr_versions_for_context(meta, &package.id, target))
        .collect::<Vec<_>>();
    versions.sort_by(|left, right| left.cmp_precedence(right));
    versions.dedup_by(|left, right| left.cmp_precedence(right).is_eq());
    versions
}

/// Guard CLI↔test ktstr-version skew before running the suite.
///
/// Reads every selected test package's RESOLVED `ktstr` dependency version
/// (what its test binaries link) from `cargo metadata` and compares it with
/// the CLI's own compiled-in version. Warns on older versions; errors —
/// aborting the run — when any selected ktstr is newer than the CLI.
///
/// A selection with no resolved `ktstr` dependency skips the guard.
fn check_ktstr_version_compat(
    meta: &cargo_metadata::Metadata,
    args: &[String],
    target: &crate::feature_discovery::TargetContext,
    invocation_dir: &Path,
) -> Result<(), String> {
    let cli = Version::parse(env!("CARGO_PKG_VERSION"))
        .expect("cargo-ktstr's own CARGO_PKG_VERSION is valid semver");
    let tests =
        selected_resolved_ktstr_versions_for_context_at(meta, args, Some(target), invocation_dir);
    if tests.is_empty() {
        // No linked `ktstr` (or no resolve graph) — running outside a
        // ktstr-dependent project, or cannot assess; nothing to guard.
        return Ok(());
    }
    let outcomes = tests
        .into_iter()
        .map(|test| version_guard(&cli, test))
        .collect::<Vec<_>>();
    if let Some(message) = outcomes.iter().find_map(|outcome| match outcome {
        VersionGuard::Error(message) => Some(message),
        VersionGuard::Ok | VersionGuard::Warn(_) => None,
    }) {
        return Err(message.clone());
    }
    for outcome in outcomes {
        if let VersionGuard::Warn(message) = outcome {
            ktstr::ktstr_status!("cargo ktstr: warning: {message}");
        }
    }
    Ok(())
}

/// Shared preflight for every cargo-ktstr frontend that directly invokes
/// `cargo nextest run` (including cargo-llvm-cov's nextest delegation).
///
/// Reuse one manifest-only metadata result for targeted optional ktstr feature
/// inference, then resolve the exact selected/augmented feature set for the
/// version guard. Keeping this callable by replay prevents a second nextest
/// route from drifting in package selection, target cfg evaluation, version
/// compatibility, or metadata-failure behavior.
///
/// Metadata inspection has historically been best-effort on these general test
/// paths. Preserve that behavior: if it fails, warn and let the underlying
/// Cargo command diagnose (or successfully handle) its own arguments.
struct PreparedNextestArgs {
    args: Vec<String>,
    metadata: Option<cargo_metadata::Metadata>,
}

fn prepare_nextest_invocation(args: Vec<String>) -> Result<PreparedNextestArgs, String> {
    let invocation_dir = std::env::current_dir()
        .map_err(|error| format!("read Cargo invocation directory: {error}"))?;
    let target = match effective_target_context(&args) {
        Ok(target) => target,
        Err(error) => {
            tracing::warn!(
                error,
                "ktstr target discovery/version guard failed; forwarding original Cargo args"
            );
            return Ok(PreparedNextestArgs {
                args,
                metadata: None,
            });
        }
    };
    let manifests = match query_metadata_for_target(&args, MetadataMode::NoDeps, &target) {
        Ok(metadata) => metadata,
        Err(error) => {
            tracing::warn!(
                error,
                "ktstr feature discovery/version guard failed; forwarding original Cargo args"
            );
            return Ok(PreparedNextestArgs {
                args,
                metadata: None,
            });
        }
    };
    let augmented = augment_test_features_from_metadata_for_context_at(
        args.clone(),
        &manifests,
        Some(&target),
        &invocation_dir,
    );
    let resolved = match query_resolved_metadata_for_invocation(
        &augmented,
        &augmented,
        &manifests,
        &target,
        &invocation_dir,
    ) {
        Ok(metadata) => metadata,
        Err(error) => {
            tracing::warn!(
                error,
                "ktstr version guard could not resolve inferred features; forwarding Cargo args"
            );
            return Ok(PreparedNextestArgs {
                args: augmented,
                metadata: None,
            });
        }
    };
    check_ktstr_version_compat(&resolved, &augmented, &target, &invocation_dir)?;
    Ok(PreparedNextestArgs {
        args: augmented,
        metadata: Some(resolved),
    })
}

pub(crate) fn prepare_nextest_args(args: Vec<String>) -> Result<Vec<String>, String> {
    prepare_nextest_invocation(args).map(|prepared| prepared.args)
}

/// Split nextest FILTERSET tokens (`-E` / `--filterset` / the legacy
/// `--filter-expr`; space-, `=`-, and glued-short forms) out of a passthrough
/// argv, returning (filterset expressions, remaining args). A bare trailing
/// `-E` with no following value is dropped (nextest would reject it anyway).
///
/// Only FILTERSETS are extracted — positional test-name filters are left in
/// `rest` deliberately: nextest ANDs the name-filter dimension with the
/// filterset dimension (verified in nextest-runner 0.118 `test_filter.rs`
/// `filter_match_base`: a test must match BOTH), so a positional filter already
/// intersects the injected relevant filterset correctly and needs no folding.
/// Multiple filtersets, by contrast, UNION among themselves (`exprs.iter().any`
/// in `matches_expression`), so they MUST be folded into one `&`-composed
/// expression to narrow rather than widen.
/// An inner `--` and every following test-binary argument remain opaque.
pub(crate) fn extract_nextest_filtersets(args: Vec<String>) -> (Vec<String>, Vec<String>) {
    let mut filters = Vec::new();
    let mut rest = Vec::new();
    let mut it = args.into_iter();
    while let Some(tok) = it.next() {
        if tok == "--" {
            rest.push(tok);
            rest.extend(it);
            break;
        }
        if tok == "-E" || tok == "--filterset" || tok == "--filter-expr" {
            if let Some(val) = it.next() {
                if val == "--" {
                    rest.push(val);
                    rest.extend(it);
                    break;
                }
                filters.push(val);
            }
        } else if let Some(val) = tok
            .strip_prefix("--filterset=")
            .or_else(|| tok.strip_prefix("--filter-expr="))
        {
            filters.push(val.to_string());
        } else if let Some(rest_short) = tok.strip_prefix("-E") {
            // Glued short form: `-E=EXPR` or `-EEXPR`. Any `-E`-prefixed token is
            // treated as the nextest filterset flag: `-E` is nextest's only
            // `-E*` short flag, and no legitimate cargo/nextest passthrough value
            // begins with `-E` (feature lists, profiles, paths, and test-name
            // filters do not), so this cannot swallow a non-filterset token.
            filters.push(
                rest_short
                    .strip_prefix('=')
                    .unwrap_or(rest_short)
                    .to_string(),
            );
        } else {
            rest.push(tok);
        }
    }
    (filters, rest)
}

/// Fold the change-relevant nextest filterset into `args`, intersecting it with
/// any user filtersets already present so the net effect NARROWS the run
/// (`(relevant) & (userA | userB | ...)`), never widens it (nextest UNIONs
/// multiple `-E`). User filterset tokens are removed and replaced by one
/// composed `-E`; every other token (positional name filters, `--features`,
/// …) is preserved and passes through untouched. The replacement filter is
/// inserted before an inner `--`.
fn compose_relevant_filter(args: Vec<String>, relevant: &str) -> Vec<String> {
    let (user_filtersets, mut rest) = extract_nextest_filtersets(args);
    let combined = if user_filtersets.is_empty() {
        relevant.to_string()
    } else {
        let union = user_filtersets
            .iter()
            .map(|f| format!("({f})"))
            .collect::<Vec<_>>()
            .join(" | ");
        format!("({relevant}) & ({union})")
    };
    let insertion = rest
        .iter()
        .position(|argument| argument == "--")
        .unwrap_or(rest.len());
    rest.splice(insertion..insertion, ["-E".to_string(), combined]);
    rest
}

/// Resolve and apply `--relevant` narrowing to a test/coverage passthrough
/// argv. A no-op when `relevant` is false. Otherwise builds + introspects the
/// declared schedulers to map the `base..worktree` change onto a nextest
/// filterset ([`crate::affected::relevant_test_filter`]) and folds it into
/// `args`. `Ok(None)` from the resolver (a broad / unattributable change, or a
/// mapping that could not be built) means "do not narrow" — run the user's
/// unmodified selection (the fail-safe).
fn apply_relevant_narrowing(
    args: Vec<String>,
    relevant: bool,
    base: Option<String>,
    base_ref: Option<String>,
    default_branch: String,
    release: bool,
) -> Result<Vec<String>, String> {
    if !relevant {
        return Ok(args);
    }
    match crate::affected::relevant_test_filter(
        base.as_deref(),
        base_ref.as_deref(),
        &default_branch,
        &args,
        release,
    ) {
        Ok(Some(expr)) => Ok(compose_relevant_filter(args, &expr)),
        Ok(None) => Ok(args),
        Err(e) => Err(format!("compute --relevant test set: {e:#}")),
    }
}

/// Validate and export an operator `--cpu-cap` for a test-family run.
///
/// Mirrors the `shell --cpu-cap` propagation (`misc/shell.rs::run_shell`):
/// validate eagerly, then export `KTSTR_CPU_CAP` so the harness-prebuild
/// LLC reservation, every nextest child test process (env inheritance),
/// and each per-test no-perf VM budget re-resolve the same cap. The
/// clap-level `requires = "no_perf_mode"` rule enforces the mode contract
/// before this runs; the on-miss auto kernel build stays uncapped by
/// design (`cli::resolve` passes a typed `None`).
fn apply_cpu_cap(cpu_cap: Option<usize>) -> Result<(), String> {
    let Some(cap) = cpu_cap else {
        return Ok(());
    };
    if ktstr::bypass_llc_locks_active() {
        return Err(
            "--cpu-cap conflicts with KTSTR_BYPASS_LLC_LOCKS=1; unset one of them. \
             --cpu-cap is a resource contract; bypass disables the contract entirely."
                .to_string(),
        );
    }
    // Validate early so a bad cap surfaces at CLI-parse time.
    ktstr::cli::CpuCap::new(cap).map_err(|e| format!("{e:#}"))?;
    // SAFETY: reached from `dispatch_run_command` before any helper on
    // this chain spawns a thread — the same single-threaded position the
    // Shell arm's `set_var` relies on (see misc/shell.rs::run_shell).
    unsafe { std::env::set_var(ktstr::KTSTR_CPU_CAP_ENV, cap.to_string()) };
    Ok(())
}

/// Minimum cargo-nextest version the orchestrated routes support.
///
/// Binding constraints, oldest first: the injected `--tool-config-file`
/// flag needs 0.9.27; the embedded tool config's `[test-groups]` +
/// `@tool:` namespace + `max-threads = "num-cpus"` need 0.9.48 — and
/// BELOW that, nextest warn-and-ignores the unknown keys, silently
/// deleting the ordinary-host concurrency cap while the deliberately
/// unbounded `test-threads` admission budget stays in force
/// (run-everything-at-once oversubscription, not an error); the
/// config-side `nextest-version` self-enforcement key is honored from
/// tool configs since 0.9.55; the verifier route's `--no-tests=pass`
/// needs 0.9.75. The wrapper preflight below is the primary gate
/// because versions predating 0.9.55 cannot self-enforce a floor;
/// `nextest-tool.toml` carries the same floor as defense in depth
/// (pinned equal by `embedded_config_pins_the_wrapper_version_floor`).
pub(crate) const MIN_NEXTEST_VERSION: Version = Version::new(0, 9, 75);

/// Parse the version out of `cargo nextest --version` stdout, shaped
/// `cargo-nextest 0.9.98 (hash date)`.
fn parse_nextest_version(stdout: &str) -> Result<Version, String> {
    let first_line = stdout.lines().next().unwrap_or_default().trim();
    let token = first_line.split_whitespace().nth(1).ok_or_else(|| {
        format!("cargo ktstr: unrecognized `cargo nextest --version` output {first_line:?}")
    })?;
    Version::parse(token)
        .map_err(|error| format!("cargo ktstr: parse cargo-nextest version {token:?}: {error}"))
}

fn nextest_version_floor_error(found: &Version) -> Option<String> {
    (*found < MIN_NEXTEST_VERSION).then(|| {
        format!(
            "cargo ktstr: cargo-nextest {found} is older than the minimum supported \
             {MIN_NEXTEST_VERSION}. ktstr's generated nextest tool config and CLI shape \
             rely on features this version lacks (an older nextest silently ignores the \
             tool config's concurrency caps and oversubscribes the host). Update with \
             `cargo install --locked cargo-nextest` or `cargo nextest self update`."
        )
    })
}

/// Enforce [`MIN_NEXTEST_VERSION`] against the installed cargo-nextest.
/// Runs on every nextest-backed route (test, coverage, `llvm-cov
/// nextest`, verifier, replay `--exec`) after that route's
/// cargo-nextest PATH-presence check, before any build work.
pub(crate) fn check_nextest_version() -> Result<(), String> {
    let output = Command::new("cargo")
        .args(["nextest", "--version"])
        .output()
        .map_err(|error| format!("cargo ktstr: run `cargo nextest --version`: {error}"))?;
    if !output.status.success() {
        return Err(format!(
            "cargo ktstr: `cargo nextest --version` failed with {}: {}",
            output.status,
            String::from_utf8_lossy(&output.stderr).trim(),
        ));
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    let version = parse_nextest_version(&stdout)?;
    match nextest_version_floor_error(&version) {
        Some(error) => Err(error),
        None => Ok(()),
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn run_test(
    kernel: Vec<String>,
    no_perf_mode: bool,
    no_skip_mode: bool,
    cpu_cap: Option<usize>,
    release: bool,
    profile: Option<String>,
    nextest_profile: Option<String>,
    include_eol: bool,
    relevant: bool,
    base: Option<String>,
    base_ref: Option<String>,
    default_branch: String,
    args: Vec<String>,
) -> Result<(), String> {
    apply_cpu_cap(cpu_cap)?;
    nextest_archive_reuse(TEST_SUB_ARGV, &args)?;
    ktstr::cli::check_kvm().map_err(|e| format!("{e:#}"))?;
    ktstr::cli::check_tools(&["cargo-nextest"]).map_err(|e| format!("{e:#}"))?;
    check_nextest_version()?;
    // Smart feature inference must precede registry discovery: `--relevant`
    // probes the exact feature/package/target/profile selection the eventual
    // nextest run will build, rather than a second workspace-wide default.
    let prepared = if llvm_cov_reuses_archive(TEST_SUB_ARGV, &args) {
        PreparedNextestArgs {
            args,
            metadata: None,
        }
    } else {
        prepare_nextest_invocation(args)?
    };
    let args = apply_relevant_narrowing(
        prepared.args,
        relevant,
        base,
        base_ref,
        default_branch,
        release,
    )?;
    run_cargo_sub(
        TEST_SUB_ARGV,
        "tests",
        kernel,
        no_perf_mode,
        no_skip_mode,
        release,
        profile,
        nextest_profile,
        include_eol,
        prepared.metadata,
        args,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn run_coverage(
    kernel: Vec<String>,
    no_perf_mode: bool,
    no_skip_mode: bool,
    cpu_cap: Option<usize>,
    release: bool,
    profile: Option<String>,
    nextest_profile: Option<String>,
    include_eol: bool,
    relevant: bool,
    base: Option<String>,
    base_ref: Option<String>,
    default_branch: String,
    args: Vec<String>,
) -> Result<(), String> {
    apply_cpu_cap(cpu_cap)?;
    if llvm_cov_has_lifecycle_flag(&args, "--no-run") {
        return run_llvm_cov_report_only(
            COVERAGE_SUB_ARGV,
            release,
            profile.as_deref(),
            nextest_profile.as_deref(),
            no_perf_mode,
            no_skip_mode,
            &args,
        );
    }
    nextest_archive_reuse(COVERAGE_SUB_ARGV, &args)?;
    ktstr::cli::check_kvm().map_err(|e| format!("{e:#}"))?;
    ktstr::cli::check_tools(&["cargo-nextest", "cargo-llvm-cov"]).map_err(|e| format!("{e:#}"))?;
    check_nextest_version()?;
    // `coverage` runs the same suite through `cargo llvm-cov nextest`, so use
    // the same version guard and targeted feature inference as `test`.
    let prepared = if llvm_cov_reuses_archive(COVERAGE_SUB_ARGV, &args) {
        PreparedNextestArgs {
            args,
            metadata: None,
        }
    } else {
        prepare_nextest_invocation(args)?
    };
    let args = apply_relevant_narrowing(
        prepared.args,
        relevant,
        base,
        base_ref,
        default_branch,
        release,
    )?;
    run_cargo_sub(
        COVERAGE_SUB_ARGV,
        "coverage",
        kernel,
        no_perf_mode,
        no_skip_mode,
        release,
        profile,
        nextest_profile,
        include_eol,
        prepared.metadata,
        args,
    )
}

pub(crate) fn run_llvm_cov(
    kernel: Vec<String>,
    no_perf_mode: bool,
    no_skip_mode: bool,
    cpu_cap: Option<usize>,
    include_eol: bool,
    args: Vec<String>,
) -> Result<(), String> {
    apply_cpu_cap(cpu_cap)?;
    if llvm_cov_has_lifecycle_flag(&args, "--no-run") {
        return run_llvm_cov_report_only(
            LLVM_COV_SUB_ARGV,
            false,
            None,
            None,
            no_perf_mode,
            no_skip_mode,
            &args,
        );
    }
    nextest_archive_reuse(LLVM_COV_SUB_ARGV, &args)?;
    // `llvm-cov nextest` is the same nextest-backed suite as `coverage`
    // (identical tool-config injection and forced admission), so it needs
    // the same preflights; report/clean/show-env passthrough modes invoke
    // no nextest and keep their exact pre-existing behavior.
    if llvm_cov_uses_nextest(&args) {
        ktstr::cli::check_tools(&["cargo-nextest", "cargo-llvm-cov"])
            .map_err(|e| format!("{e:#}"))?;
        check_nextest_version()?;
    }
    // `llvm-cov` is raw passthrough — the user supplies every
    // argument after the subcommand name, including any profile
    // selection. `release: false` / `profile: None` / `nextest_profile:
    // None` here mean "don't inject any profile ourselves"; the user
    // decides via the raw args.
    //
    // Report/clean/show-env and other non-test modes do not build or enumerate
    // ktstr tests, so they retain exact passthrough semantics. Explicit
    // `llvm-cov nextest` is the same test suite as `coverage`, and therefore
    // goes through the identical target-aware feature and version preflight.
    let prepared =
        if llvm_cov_uses_nextest(&args) && !llvm_cov_reuses_archive(LLVM_COV_SUB_ARGV, &args) {
            prepare_nextest_invocation(args)
                .map_err(|error| format!("cargo ktstr llvm-cov nextest: {error}"))?
        } else {
            PreparedNextestArgs {
                args,
                metadata: None,
            }
        };
    run_cargo_sub(
        LLVM_COV_SUB_ARGV,
        "llvm-cov",
        kernel,
        no_perf_mode,
        no_skip_mode,
        false,
        None,
        None,
        include_eol,
        prepared.metadata,
        prepared.args,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::{BTreeMap, BTreeSet};
    use std::sync::Arc;

    use crate::test_env::{ChildEnv, is_reexec_case, reexec_current_test};

    fn v(s: &str) -> Version {
        Version::parse(s).expect("test version literal is valid semver")
    }

    /// Guards the whole shared-build-dir mechanism: a workspace member that
    /// survives the purge is a silent stale-reuse of another source digest's
    /// member. Every member product must be removed across `deps/` (underscored,
    /// `lib`-prefixed) AND `.fingerprint/`/`build/` (dashed) — missing the dashed
    /// build-script/`OUT_DIR` entries would keep a stale generated skeleton.
    /// Registry dependencies must be kept (they are the reuse we want), and a
    /// member stem must never partial-match a longer dependency stem.
    #[test]
    fn purge_removes_every_member_product_but_keeps_dependencies() {
        let profile = tempfile::tempdir().expect("profile tempdir");
        let profile = profile.path();
        let write_file = |sub: &str, name: &str| {
            let dir = profile.join(sub);
            std::fs::create_dir_all(&dir).unwrap();
            std::fs::write(dir.join(name), b"x").unwrap();
        };
        let write_dir = |sub: &str, name: &str| {
            let dir = profile.join(sub).join(name);
            std::fs::create_dir_all(&dir).unwrap();
            std::fs::write(dir.join("marker"), b"x").unwrap();
        };
        // Member `scx-ktstr`: underscored in deps/incremental, dashed in
        // .fingerprint/build (mirrors real Cargo naming).
        write_file("deps", "libscx_ktstr-0123456789abcdef.rlib");
        write_file("deps", "libscx_ktstr-0123456789abcdef.rmeta");
        write_file("deps", "scx_ktstr-0123456789abcdef.d");
        write_dir("incremental", "scx_ktstr-0123456789abcdef");
        write_dir(".fingerprint", "scx-ktstr-0123456789abcdef");
        write_dir(".fingerprint", "scx-ktstr-fedcba9876543210"); // build-script run
        write_dir("build", "scx-ktstr-0123456789abcdef");
        // A registry dependency that must be retained.
        write_file("deps", "libserde-fedcba9876543210.rlib");
        write_dir(".fingerprint", "serde-fedcba9876543210");
        write_dir("build", "serde-fedcba9876543210");
        // Dependencies whose stems share a member's prefix — anchoring on the
        // `<stem>-<hex-hash>` boundary must keep them.
        write_file("deps", "libscx_ktstr_helper-aaaaaaaaaaaaaaaa.rlib");
        write_dir(".fingerprint", "scx-ktstr-helper-aaaaaaaaaaaaaaaa");

        // Real member set for this workspace includes another dashed member and
        // a prefix member to exercise disambiguation.
        let variants = BTreeSet::from([
            "scx-ktstr".to_string(),
            "scx_ktstr".to_string(),
            "ktstr".to_string(),
        ]);
        for sub in ["deps", ".fingerprint", "build", "incremental"] {
            purge_member_products_in(&profile.join(sub), &variants).expect("purge subdir");
        }

        // Every member product gone, including the dashed build/fingerprint ones.
        assert!(
            !profile
                .join("deps/libscx_ktstr-0123456789abcdef.rlib")
                .exists()
        );
        assert!(
            !profile
                .join("deps/libscx_ktstr-0123456789abcdef.rmeta")
                .exists()
        );
        assert!(!profile.join("deps/scx_ktstr-0123456789abcdef.d").exists());
        assert!(
            !profile
                .join("incremental/scx_ktstr-0123456789abcdef")
                .exists()
        );
        assert!(
            !profile
                .join(".fingerprint/scx-ktstr-0123456789abcdef")
                .exists(),
            "dashed member fingerprint must be purged",
        );
        assert!(
            !profile
                .join(".fingerprint/scx-ktstr-fedcba9876543210")
                .exists()
        );
        assert!(
            !profile.join("build/scx-ktstr-0123456789abcdef").exists(),
            "dashed member build-script OUT_DIR must be purged",
        );
        // Dependencies retained.
        assert!(profile.join("deps/libserde-fedcba9876543210.rlib").exists());
        assert!(profile.join(".fingerprint/serde-fedcba9876543210").exists());
        assert!(profile.join("build/serde-fedcba9876543210").exists());
        // Prefix-sharing dependencies retained (no partial match).
        assert!(
            profile
                .join("deps/libscx_ktstr_helper-aaaaaaaaaaaaaaaa.rlib")
                .exists(),
            "a member stem must not partial-match a longer dependency stem",
        );
        assert!(
            profile
                .join(".fingerprint/scx-ktstr-helper-aaaaaaaaaaaaaaaa")
                .exists(),
            "member `scx-ktstr` must not match dependency `scx-ktstr-helper`",
        );
    }

    #[test]
    fn gc_reclaims_idle_shared_build_buckets_but_never_leased_or_recent_ones() {
        let temp = tempfile::tempdir().expect("scratch parent");
        let parent = temp.path().join("shared-build-scratch-v1");
        let lock_root = temp.path().join("locks");
        std::fs::create_dir_all(&lock_root).unwrap();
        let make_bucket = |name: &str| {
            let bucket = parent.join(name);
            std::fs::create_dir_all(bucket.join("release/deps")).unwrap();
            bucket
        };
        let idle = make_bucket("aaaaaaaaaaaaaaaa");
        let recent = make_bucket("bbbbbbbbbbbbbbbb");
        let leased = make_bucket("cccccccccccccccc");

        // A recent build keeps every bucket: nothing is older than the idle
        // window relative to "now".
        gc_stale_shared_build_scratch_in(&parent, &lock_root, std::time::SystemTime::now(), None);
        assert!(idle.exists() && recent.exists() && leased.exists());

        // Fast-forward past the idle window. Hold the build-output lease on
        // `leased`; it must survive even though it is idle.
        let far_future = std::time::SystemTime::now()
            + SHARED_BUILD_SCRATCH_MAX_IDLE
            + std::time::Duration::from_secs(60);
        let canonical_leased = std::fs::canonicalize(&leased).unwrap();
        let lock_path = cargo_build_output_lock_path(&lock_root, &canonical_leased);
        let _held = ktstr::flock::try_flock(&lock_path, ktstr::flock::FlockMode::Exclusive)
            .unwrap()
            .expect("acquire bucket lease for the test");
        gc_stale_shared_build_scratch_in(&parent, &lock_root, far_future, None);
        assert!(!idle.exists(), "idle unleased bucket must be reclaimed");
        assert!(
            !recent.exists(),
            "the second idle unleased bucket must be reclaimed"
        );
        assert!(
            leased.exists(),
            "a bucket under an active build-output lease must never be deleted",
        );
    }

    /// A bucket in continuous use can still present ancient mtimes: Cargo moves
    /// only `<bucket>/<profile>/deps` on a rebuild and nothing at all on a no-op
    /// build, so the last-use stamp — not the build output — is what records
    /// that a run took the bucket.
    #[test]
    fn gc_keeps_a_stamped_bucket_whose_build_output_mtimes_are_ancient() {
        let temp = tempfile::tempdir().expect("scratch parent");
        let parent = temp.path().join("shared-build-scratch-v1");
        let lock_root = temp.path().join("locks");
        std::fs::create_dir_all(&lock_root).unwrap();
        let now = std::time::SystemTime::now();
        let ancient = now - SHARED_BUILD_SCRATCH_MAX_IDLE * 2;
        let make_ancient_bucket = |name: &str| -> std::path::PathBuf {
            let bucket = parent.join(name);
            let deps = bucket.join("release/deps");
            std::fs::create_dir_all(&deps).unwrap();
            let product = deps.join("libdep.rlib");
            std::fs::write(&product, b"dep").unwrap();
            std::fs::File::options()
                .write(true)
                .open(&product)
                .unwrap()
                .set_modified(ancient)
                .unwrap();
            // Deepest directory first: every write above bumps its parent.
            for directory in [&deps, &bucket.join("release"), &bucket] {
                std::fs::File::open(directory)
                    .unwrap()
                    .set_modified(ancient)
                    .unwrap();
            }
            bucket
        };
        let stamped = make_ancient_bucket("aaaaaaaaaaaaaaaa");
        let unstamped = make_ancient_bucket("bbbbbbbbbbbbbbbb");
        stamp_shared_build_scratch_use(&stamped);
        let stamp = parent
            .join(SHARED_BUILD_SCRATCH_STAMP_DIR)
            .join("aaaaaaaaaaaaaaaa");
        assert!(
            stamp.is_file() && !stamped.join(SHARED_BUILD_SCRATCH_STAMP_DIR).exists(),
            "the stamp belongs beside the buckets, never inside the one Cargo builds into",
        );

        gc_stale_shared_build_scratch_in(&parent, &lock_root, now, None);
        assert!(
            stamped.exists(),
            "a bucket this run stamped must survive however old its build output is",
        );
        assert!(
            !unstamped.exists(),
            "a bucket with no stamp is still judged by the mtimes it does have",
        );

        // The stamp records use; it does not pin a bucket nobody comes back to.
        let far_future = now + SHARED_BUILD_SCRATCH_MAX_IDLE + std::time::Duration::from_secs(60);
        gc_stale_shared_build_scratch_in(&parent, &lock_root, far_future, None);
        assert!(!stamped.exists(), "a stamped bucket still ages out");
        assert!(
            !stamp.exists(),
            "reclaiming a bucket removes its last-use stamp",
        );
    }

    /// Why the post-lease re-check tests bucket contents rather than existence:
    /// taking a bucket's build-output lease creates the directory, so asking
    /// whether a reclaimed bucket exists answers itself.
    #[test]
    fn leasing_a_reclaimed_bucket_recreates_it_without_build_output() {
        let temp = tempfile::tempdir().expect("scratch parent");
        let parent = temp.path().join("shared-build-scratch-v1");
        let lock_root = temp.path().join("locks");
        std::fs::create_dir_all(&parent).unwrap();
        let bucket = parent.join("aaaaaaaaaaaaaaaa");
        assert!(!shared_build_scratch_has_build_output(&bucket));

        let lease = acquire_cargo_build_output_lease_at_root(
            &bucket,
            &lock_root,
            ktstr::flock::FlockMode::Shared,
            "cargo ktstr",
            &std::sync::atomic::AtomicBool::new(false),
        )
        .expect("lease a reclaimed bucket");
        assert!(
            bucket.is_dir() && !shared_build_scratch_has_build_output(&bucket),
            "the lease resurrects the directory but not the build output in it",
        );

        std::fs::write(bucket.join("obj.o"), b"obj").unwrap();
        assert!(
            shared_build_scratch_has_build_output(&bucket),
            "a bucket holding build output is usable",
        );
        drop(lease);
    }

    /// The sweep runs before the calling run takes its own bucket lease, so
    /// without the exemption a run's own GC pass deletes the bucket it is about
    /// to build in (or, on a cache hit, to read `OUT_DIR` artifacts from) —
    /// a same-process window no lease can cover.
    #[test]
    fn gc_spares_the_pending_bucket_of_the_run_that_swept() {
        let temp = tempfile::tempdir().expect("scratch parent");
        let parent = temp.path().join("shared-build-scratch-v1");
        let lock_root = temp.path().join("locks");
        std::fs::create_dir_all(&lock_root).unwrap();
        let make_bucket = |name: &str| -> std::path::PathBuf {
            let bucket = parent.join(name);
            std::fs::create_dir_all(bucket.join("release/deps")).unwrap();
            bucket
        };
        let pending = make_bucket("aaaaaaaaaaaaaaaa");
        let other = make_bucket("bbbbbbbbbbbbbbbb");

        let far_future = std::time::SystemTime::now()
            + SHARED_BUILD_SCRATCH_MAX_IDLE
            + std::time::Duration::from_secs(60);
        gc_stale_shared_build_scratch_in(
            &parent,
            &lock_root,
            far_future,
            Some(0xaaaa_aaaa_aaaa_aaaa),
        );
        assert!(
            pending.exists(),
            "the bucket this run is about to lease is never reclaimed by its own sweep",
        );
        assert!(
            !other.exists(),
            "every other idle unleased bucket is still reclaimed",
        );
    }

    #[test]
    fn reclaim_sweeps_oldest_unleased_scratch_until_shortfall_clears() {
        let temp = tempfile::tempdir().expect("scratch parent");
        let parent = temp.path().join("shared-build-scratch-v1");
        let lock_root = temp.path().join("locks");
        std::fs::create_dir_all(&lock_root).unwrap();
        let now = std::time::SystemTime::now();
        let make = |name: &str, bytes: usize, age_secs: u64| -> std::path::PathBuf {
            let bucket = parent.join(name);
            std::fs::create_dir_all(&bucket).unwrap();
            let marker = bucket.join("obj.o");
            std::fs::write(&marker, vec![0u8; bytes]).unwrap();
            let target = now - std::time::Duration::from_secs(age_secs);
            std::fs::File::options()
                .write(true)
                .open(&marker)
                .unwrap()
                .set_modified(target)
                .unwrap();
            // Set the bucket dir mtime LAST: adding the marker bumped it to ~now,
            // which would otherwise dominate `bucket_last_used`.
            std::fs::File::open(&bucket)
                .unwrap()
                .set_modified(target)
                .unwrap();
            bucket
        };
        let old = make("aaaaaaaaaaaaaaaa", 4096, 300);
        let mid = make("bbbbbbbbbbbbbbbb", 4096, 200);
        let new = make("cccccccccccccccc", 4096, 100);

        // A one-byte shortfall reclaims exactly the single oldest idle bucket and
        // then stops — proving both the oldest-first order and the bounded sweep.
        let reclaimed =
            reclaim_shared_build_scratch_under_pressure_in(&parent, &lock_root, 1, None);
        assert!(
            reclaimed >= 4096,
            "reclaimed at least the oldest bucket's bytes: {reclaimed}"
        );
        assert!(
            !old.exists(),
            "the oldest idle bucket is swept first under pressure"
        );
        assert!(
            mid.exists() && new.exists(),
            "the sweep stops once the shortfall is covered",
        );
    }

    #[test]
    fn reclaim_spares_leased_and_exempt_buckets_and_no_ops_without_shortfall() {
        let temp = tempfile::tempdir().expect("scratch parent");
        let parent = temp.path().join("shared-build-scratch-v1");
        let lock_root = temp.path().join("locks");
        std::fs::create_dir_all(&lock_root).unwrap();
        let make = |name: &str| -> std::path::PathBuf {
            let bucket = parent.join(name);
            std::fs::create_dir_all(&bucket).unwrap();
            std::fs::write(bucket.join("obj.o"), vec![0u8; 4096]).unwrap();
            bucket
        };
        let leased = make("aaaaaaaaaaaaaaaa");
        let exempt = make("bbbbbbbbbbbbbbbb");
        let free = make("cccccccccccccccc");

        // No shortfall: never sweep, even with reclaimable buckets present.
        assert_eq!(
            reclaim_shared_build_scratch_under_pressure_in(&parent, &lock_root, 0, None),
            0,
            "a zero shortfall must not sweep anything",
        );
        assert!(leased.exists() && exempt.exists() && free.exists());

        // Hold the build-output lease on `leased` and exempt `exempt` (the
        // pending build's own bucket). Even an unbounded shortfall may reclaim
        // only `free`.
        let canonical_leased = std::fs::canonicalize(&leased).unwrap();
        let held = ktstr::flock::try_flock(
            cargo_build_output_lock_path(&lock_root, &canonical_leased),
            ktstr::flock::FlockMode::Exclusive,
        )
        .unwrap()
        .expect("hold bucket lease for the test");
        let reclaimed = reclaim_shared_build_scratch_under_pressure_in(
            &parent,
            &lock_root,
            u64::MAX,
            Some(0xbbbb_bbbb_bbbb_bbbb),
        );
        drop(held);
        assert!(
            !free.exists(),
            "an unleased, non-exempt bucket is reclaimed under pressure",
        );
        assert!(reclaimed >= 4096);
        assert!(
            leased.exists(),
            "a bucket under an active build-output lease is never swept",
        );
        assert!(
            exempt.exists(),
            "the pending build's own bucket is exempt from the sweep",
        );
    }

    /// A runtime dependent (a live nextest test process reading `probe.o` from
    /// the bucket's baked `env!("OUT_DIR")`) holds the build-output lease SHARED
    /// for the whole run. Both reclaimers take a non-blocking EXCLUSIVE lease,
    /// which fails against that SHARED hold, so neither the pressure sweep nor
    /// the aged GC may reclaim the bucket until the run ends and the lease drops.
    #[test]
    fn shared_runtime_lease_shields_bucket_from_pressure_sweep_and_aged_gc() {
        let temp = tempfile::tempdir().expect("scratch parent");
        let parent = temp.path().join("shared-build-scratch-v1");
        let lock_root = temp.path().join("locks");
        std::fs::create_dir_all(&lock_root).unwrap();
        let bucket = parent.join("aaaaaaaaaaaaaaaa");
        std::fs::create_dir_all(&bucket).unwrap();
        std::fs::write(bucket.join("obj.o"), vec![0u8; 4096]).unwrap();

        // Hold the bucket's build-output lease SHARED, as the run does for the
        // lifetime of the test binaries built from it.
        let canonical = std::fs::canonicalize(&bucket).unwrap();
        let lock_path = cargo_build_output_lock_path(&lock_root, &canonical);
        let runtime = ktstr::flock::try_flock(&lock_path, ktstr::flock::FlockMode::Shared)
            .unwrap()
            .expect("hold the shared runtime lease");

        // Pressure sweep: an unbounded shortfall still cannot take the bucket.
        assert_eq!(
            reclaim_shared_build_scratch_under_pressure_in(&parent, &lock_root, u64::MAX, None),
            0,
            "a SHARED-leased bucket is never swept under disk pressure",
        );
        assert!(bucket.exists());

        // Aged GC: even far past the idle window the SHARED hold blocks it.
        let far_future = std::time::SystemTime::now()
            + SHARED_BUILD_SCRATCH_MAX_IDLE
            + std::time::Duration::from_secs(60);
        gc_stale_shared_build_scratch_in(&parent, &lock_root, far_future, None);
        assert!(
            bucket.exists(),
            "a SHARED-leased bucket is never reclaimed by the aged GC",
        );

        // Once the run ends and the lease drops, both reclaimers may take it.
        drop(runtime);
        assert!(
            reclaim_shared_build_scratch_under_pressure_in(&parent, &lock_root, u64::MAX, None)
                >= 4096,
            "an unleased bucket is reclaimed under pressure",
        );
        assert!(
            !bucket.exists(),
            "the bucket is reclaimable once no runtime lease remains",
        );
    }

    /// Metadata with one workspace member (`scx-ktstr`, `source == null`) and
    /// one registry dependency (`serde`, `source != null`), so a member purge
    /// has both something to remove and something it must keep.
    fn member_and_registry_metadata() -> cargo_metadata::Metadata {
        let package = |name: &str, id: &str, source: &str| -> String {
            format!(
                r#"{{
                    "name":"{name}",
                    "version":"1.0.0",
                    "id":"{id}",
                    "source":{source},
                    "description":null,
                    "dependencies":[],
                    "license":null,
                    "license_file":null,
                    "targets":[],
                    "features":{{}},
                    "manifest_path":"/w/{name}/Cargo.toml",
                    "readme":null,
                    "repository":null,
                    "homepage":null,
                    "documentation":null,
                    "links":null,
                    "publish":null,
                    "default_run":null
                }}"#
            )
        };
        let member = "scx-ktstr 1.0.0 (path+file:///w/scx-ktstr)";
        serde_json::from_str(&format!(
            r#"{{
                "packages":[{member_package},{dep_package}],
                "workspace_members":["{member}"],
                "workspace_default_members":["{member}"],
                "resolve":null,
                "workspace_root":"/w",
                "target_directory":"/w/target",
                "version":1
            }}"#,
            member_package = package("scx-ktstr", member, "null"),
            dep_package = package(
                "serde",
                "serde 1.0.0 (registry+https://github.com/rust-lang/crates.io-index)",
                r#""registry+https://github.com/rust-lang/crates.io-index""#,
            ),
        ))
        .expect("member/registry metadata fixture deserializes")
    }

    /// `load_or_build_nextest_artifacts` runs one acquire-lease -> purge-members
    /// sequence before Cargo for BOTH producer modes: that lease/purge sits
    /// outside the `if mode == CachedNextestMode::Coverage` branch, so a coverage
    /// producer purges stale workspace members from its own (mode-separated)
    /// shared bucket exactly as a plain producer does. Prove that composition on
    /// a coverage-style bucket: under the exclusive build-output lease, a
    /// workspace member is purged while a registry dependency survives.
    #[test]
    fn coverage_bucket_purges_members_under_the_build_output_lease() {
        // Anchors the mode label the coverage producer keys its bucket on; if it
        // ever drifts, the plain/coverage bucket separation silently collapses.
        assert_eq!(
            CachedNextestMode::Coverage.identity_label(),
            "llvm-cov-nextest",
        );

        let metadata = member_and_registry_metadata();
        let temp = tempfile::tempdir().expect("coverage bucket scratch");
        let bucket = temp
            .path()
            .join("shared-build-scratch-v1")
            .join("0f0f0f0f0f0f0f0f");
        let lock_root = temp.path().join("output-locks");
        let deps = bucket.join("ci/deps");
        let fingerprint = bucket.join("ci/.fingerprint");
        std::fs::create_dir_all(&deps).unwrap();
        std::fs::create_dir_all(&fingerprint).unwrap();
        // Member `scx-ktstr`: underscored in deps/, dashed in .fingerprint/.
        std::fs::write(deps.join("libscx_ktstr-0123456789abcdef.rlib"), b"member").unwrap();
        std::fs::create_dir_all(fingerprint.join("scx-ktstr-0123456789abcdef")).unwrap();
        // Registry dependency that must survive the purge.
        std::fs::write(deps.join("libserde-fedcba9876543210.rlib"), b"dep").unwrap();
        std::fs::create_dir_all(fingerprint.join("serde-fedcba9876543210")).unwrap();

        let interrupted = std::sync::atomic::AtomicBool::new(false);
        let lease = acquire_cargo_build_output_lease_at_root(
            &bucket,
            &lock_root,
            ktstr::flock::FlockMode::Exclusive,
            "coverage producer",
            &interrupted,
        )
        .expect("coverage producer owns its shared build bucket");

        purge_shared_build_dir_workspace_members(&bucket, &metadata)
            .expect("purge coverage bucket members under the build-output lease");

        assert!(
            !deps.join("libscx_ktstr-0123456789abcdef.rlib").exists(),
            "coverage bucket must purge the stale member product",
        );
        assert!(
            !fingerprint.join("scx-ktstr-0123456789abcdef").exists(),
            "coverage bucket must purge the stale member fingerprint",
        );
        assert!(
            deps.join("libserde-fedcba9876543210.rlib").exists(),
            "coverage bucket must retain the registry dependency it exists to reuse",
        );
        assert!(
            fingerprint.join("serde-fedcba9876543210").exists(),
            "coverage bucket must retain the registry dependency fingerprint",
        );

        drop(lease);
    }

    fn llvm_cov_feature_metadata() -> cargo_metadata::Metadata {
        let renamed = "renamed-scheduler 1.0.0 (path+file:///w/renamed-scheduler)";
        let ordinary = "ordinary-scheduler 1.0.0 (path+file:///w/ordinary-scheduler)";
        let package = |name: &str, id: &str, rename: &str, features: &str| -> String {
            format!(
                r#"{{
                    "name":"{name}",
                    "version":"1.0.0",
                    "id":"{id}",
                    "source":null,
                    "description":null,
                    "dependencies":[{{
                        "name":"ktstr",
                        "source":null,
                        "req":"=0.42.0",
                        "kind":null,
                        "rename":{rename},
                        "optional":true,
                        "uses_default_features":true,
                        "features":[],
                        "target":null,
                        "registry":null,
                        "path":null
                    }}],
                    "license":null,
                    "license_file":null,
                    "targets":[],
                    "features":{features},
                    "manifest_path":"/w/{name}/Cargo.toml",
                    "readme":null,
                    "repository":null,
                    "homepage":null,
                    "documentation":null,
                    "links":null,
                    "publish":null,
                    "default_run":null
                }}"#
            )
        };
        serde_json::from_str(&format!(
            r#"{{
                "packages":[{renamed_package},{ordinary_package}],
                "workspace_members":["{renamed}","{ordinary}"],
                "workspace_default_members":["{renamed}"],
                "resolve":null,
                "workspace_root":"/w",
                "target_directory":"/w/target",
                "version":1
            }}"#,
            renamed_package = package(
                "renamed-scheduler",
                renamed,
                r#""test-harness""#,
                r#"{"operator-choice":[],"verify-schedulers":["dep:test-harness"]}"#,
            ),
            ordinary_package = package(
                "ordinary-scheduler",
                ordinary,
                "null",
                r#"{"ktstr-tests":["dep:ktstr"]}"#,
            ),
        ))
        .expect("raw llvm-cov feature metadata fixture deserializes")
    }

    #[test]
    fn parse_nextest_version_extracts_the_semver_token() {
        assert_eq!(
            parse_nextest_version("cargo-nextest 0.9.98 (fc97e97bb 2026-06-21)\nrelease\n")
                .unwrap(),
            Version::new(0, 9, 98),
        );
    }

    #[test]
    fn parse_nextest_version_rejects_malformed_output() {
        for output in ["", "garbage", "cargo-nextest not-a-version"] {
            let error = parse_nextest_version(output).unwrap_err();
            assert!(error.contains("cargo ktstr:"), "{error}");
        }
    }

    #[test]
    fn nextest_version_floor_names_versions_and_remedy() {
        let error =
            nextest_version_floor_error(&Version::new(0, 9, 48)).expect("below floor must error");
        assert!(error.contains("0.9.48"), "{error}");
        assert!(error.contains(&MIN_NEXTEST_VERSION.to_string()), "{error}");
        assert!(
            error.contains("cargo install --locked cargo-nextest"),
            "{error}"
        );
        assert!(nextest_version_floor_error(&MIN_NEXTEST_VERSION).is_none());
        assert!(nextest_version_floor_error(&Version::new(0, 9, 98)).is_none());
    }

    #[test]
    fn version_guard_equal_is_ok() {
        assert_eq!(version_guard(&v("0.19.0"), &v("0.19.0")), VersionGuard::Ok);
    }

    #[test]
    fn cargo_output_lease_covers_exact_artifact_pin_boundary() {
        use std::os::unix::fs::PermissionsExt as _;
        use std::sync::atomic::AtomicBool;

        let directory = tempfile::tempdir().expect("temporary output-lock fixture");
        let target_dir = directory.path().join("target");
        let lock_root = directory.path().join("output-locks");
        std::fs::create_dir_all(&target_dir).expect("create fixture target dir");
        let artifact = target_dir.join("scheduler");
        std::fs::write(&artifact, b"first scheduler revision").expect("write first revision");
        std::fs::set_permissions(&artifact, std::fs::Permissions::from_mode(0o755))
            .expect("make first revision executable");

        let interrupted = Arc::new(AtomicBool::new(false));
        let first = acquire_cargo_build_output_lease_at_root(
            &target_dir,
            &lock_root,
            ktstr::flock::FlockMode::Exclusive,
            "first writer",
            &interrupted,
        )
        .expect("first writer owns target output");
        let pinned = ktstr::cache::pin_content_file(&artifact)
            .expect("pin Cargo artifact before ending output ownership");

        let canonical_target =
            std::fs::canonicalize(&target_dir).expect("canonical fixture target dir");
        let lock_path = cargo_build_output_lock_path(&lock_root, &canonical_target);
        let (contended_tx, contended_rx) = std::sync::mpsc::channel();
        let (replaced_tx, replaced_rx) = std::sync::mpsc::channel();
        let target_for_thread = target_dir.clone();
        let lock_root_for_thread = lock_root.clone();
        let artifact_for_thread = artifact.clone();
        let interrupted_for_thread = Arc::clone(&interrupted);
        let contender = std::thread::spawn(move || {
            let probe = ktstr::flock::try_flock(&lock_path, ktstr::flock::FlockMode::Exclusive)
                .expect("probe first writer's output lock");
            contended_tx
                .send(probe.is_none())
                .expect("report lock contention");
            drop(probe);

            let _second = acquire_cargo_build_output_lease_at_root(
                &target_for_thread,
                &lock_root_for_thread,
                ktstr::flock::FlockMode::Exclusive,
                "second writer",
                &interrupted_for_thread,
            )
            .expect("second writer acquires after first releases");
            let replacement = target_for_thread.join("scheduler.next");
            std::fs::write(&replacement, b"second scheduler revision")
                .expect("write replacement revision");
            std::fs::set_permissions(&replacement, std::fs::Permissions::from_mode(0o755))
                .expect("make replacement executable");
            std::fs::rename(replacement, artifact_for_thread)
                .expect("atomically replace Cargo output");
            replaced_tx.send(()).expect("report replacement");
        });

        assert!(
            contended_rx
                .recv_timeout(std::time::Duration::from_secs(5))
                .expect("receive contention observation before deadline"),
            "a second writer must not cross the artifact pin boundary",
        );
        assert!(
            replaced_rx.try_recv().is_err(),
            "the target pathname must remain stable while the first lease is held",
        );
        drop(first);
        replaced_rx
            .recv_timeout(std::time::Duration::from_secs(5))
            .expect("second writer replaces output before deadline");
        contender.join().expect("join output-lock contender");

        let snapshot = ktstr::scheduler_artifact::snapshot_pinned_scheduler_artifact(pinned)
            .expect("snapshot exact pinned revision after pathname replacement");
        assert_eq!(
            std::fs::read(snapshot.path()).expect("read exact scheduler snapshot"),
            b"first scheduler revision",
        );
        assert_eq!(
            std::fs::read(&artifact).expect("read current Cargo output"),
            b"second scheduler revision",
        );
    }

    #[test]
    fn cargo_output_lease_excludes_same_target_across_cache_roots() {
        const HOLDER_CASE: &str = "cargo-output-lock-cross-cache-holder";
        const CONTENDER_CASE: &str = "cargo-output-lock-cross-cache-contender";
        const TARGET_ENV: &str = "__KTSTR_TEST_CARGO_OUTPUT_TARGET";
        const CONTENDER_CACHE_ENV: &str = "__KTSTR_TEST_CARGO_OUTPUT_CONTENDER_CACHE";

        let target_dir = std::env::var_os(TARGET_ENV).map(PathBuf::from);
        if is_reexec_case(CONTENDER_CASE) {
            let target_dir = target_dir.expect("contender target directory");
            let root =
                ktstr::cache::cargo_build_output_lock_root().expect("contender output-lock root");
            let canonical_target =
                canonical_cargo_target_dir(&target_dir).expect("canonical contender target");
            let lock_path = cargo_build_output_lock_path(&root, &canonical_target);
            assert!(
                ktstr::flock::try_flock(&lock_path, ktstr::flock::FlockMode::Exclusive)
                    .expect("probe holder's output lock")
                    .is_none(),
                "a distinct KTSTR_CACHE_DIR must not create a distinct lock for the same target",
            );
            return;
        }
        if is_reexec_case(HOLDER_CASE) {
            let target_dir = target_dir.expect("holder target directory");
            let contender_cache =
                std::env::var_os(CONTENDER_CACHE_ENV).expect("contender cache directory");
            let _lease = acquire_cargo_build_output_lease(&target_dir, "holder")
                .expect("holder acquires output lease");
            reexec_current_test(
                CONTENDER_CASE,
                [ChildEnv::set(ktstr::KTSTR_CACHE_DIR_ENV, contender_cache)],
            );
            return;
        }

        let directory = tempfile::tempdir().expect("cross-cache output-lock fixture");
        let lock_dir = directory.path().join("locks");
        let first_cache = directory.path().join("cache-first");
        let second_cache = directory.path().join("cache-second");
        let target_dir = directory.path().join("target");
        reexec_current_test(
            HOLDER_CASE,
            [
                ChildEnv::set(ktstr::KTSTR_LOCK_DIR_ENV, &lock_dir),
                ChildEnv::set(ktstr::KTSTR_CACHE_DIR_ENV, &first_cache),
                ChildEnv::set(TARGET_ENV, &target_dir),
                ChildEnv::set(CONTENDER_CACHE_ENV, &second_cache),
            ],
        );
    }

    #[test]
    fn cargo_output_lease_precedes_build_reservation() {
        let events = std::cell::RefCell::new(Vec::new());
        let resources = acquire_cargo_build_resources(
            || {
                events.borrow_mut().push("target-output-lease");
                Ok::<_, String>("lease")
            },
            || {
                events.borrow_mut().push("build-reservation");
                Ok::<_, String>("reservation")
            },
        )
        .expect("acquire ordered build resources");

        assert_eq!(resources, ("lease", "reservation"));
        assert_eq!(
            events.into_inner(),
            ["target-output-lease", "build-reservation"],
            "same-target waiters must not claim build capacity before they own Cargo output",
        );
    }

    #[test]
    fn descriptor_backed_probe_executes_the_pinned_elf_revision() {
        let directory = tempfile::tempdir().expect("temporary pinned-exec fixture");
        let artifact = directory.path().join("test-executable");
        std::fs::copy("/bin/true", &artifact).expect("copy successful executable revision");
        let pinned = ktstr::cache::pin_content_file(&artifact).expect("pin executable revision");

        let replacement = directory.path().join("replacement");
        std::fs::copy("/bin/false", &replacement).expect("copy failing replacement revision");
        std::fs::rename(replacement, &artifact).expect("atomically replace executable pathname");

        let status = Command::new(pinned.proc_fd_path())
            .status()
            .expect("execute pinned ELF through /proc/self/fd");
        assert!(
            status.success(),
            "descriptor-backed probe must execute the pinned /bin/true inode",
        );
        assert!(
            !Command::new(&artifact)
                .status()
                .expect("execute replacement /bin/false")
                .success(),
            "the ordinary pathname must now select the replacement revision",
        );
    }

    #[test]
    fn version_guard_test_older_warns() {
        // test 0.18 < CLI 0.19 -> warn (skew; the newer CLI can usually
        // still drive an older test).
        assert!(matches!(
            version_guard(&v("0.19.0"), &v("0.18.0")),
            VersionGuard::Warn(_)
        ));
    }

    #[test]
    fn version_guard_test_newer_errors() {
        // test 0.20 > CLI 0.19 -> error (the CLI predates the test's ktstr
        // and cannot drive it).
        assert!(matches!(
            version_guard(&v("0.19.0"), &v("0.20.0")),
            VersionGuard::Error(_)
        ));
    }

    #[test]
    fn version_guard_patch_delta_full_version_compare() {
        // ktstr is pre-1.0, so the guard compares the full version — even
        // a patch delta is significant. test newer-by-patch -> error;
        // older-by-patch -> warn.
        assert!(matches!(
            version_guard(&v("0.19.0"), &v("0.19.1")),
            VersionGuard::Error(_)
        ));
        assert!(matches!(
            version_guard(&v("0.19.1"), &v("0.19.0")),
            VersionGuard::Warn(_)
        ));
    }

    #[test]
    fn version_guard_ignores_build_metadata() {
        // Semver precedence ignores build metadata, so a `+build`-tagged
        // path/git ktstr dep against a plain-version CLI is the SAME release
        // — Ok, not a spurious abort/warn. Regression guard for the
        // derived-`Ord` bug (it ordered `BuildMetadata::EMPTY < non-empty`,
        // making these compare unequal: `0.19.0+abc` > `0.19.0` -> Error).
        assert_eq!(
            version_guard(&v("0.19.0"), &v("0.19.0+abc")),
            VersionGuard::Ok,
        );
        assert_eq!(
            version_guard(&v("0.19.0+abc"), &v("0.19.0")),
            VersionGuard::Ok,
        );
    }

    #[test]
    fn llvm_cov_feature_inference_is_nextest_only() {
        for args in [
            strs(&["nextest", "--workspace"]),
            strs(&["--locked", "nextest"]),
            strs(&["--workspace", "nextest"]),
            strs(&["--manifest-path", "consumer/Cargo.toml", "nextest"]),
            strs(&["--features=integration", "nextest"]),
        ] {
            assert!(
                llvm_cov_uses_nextest(&args),
                "explicit nextest subcommand should receive inference: {args:?}",
            );
        }
        for args in [
            strs(&[]),
            strs(&["test"]),
            strs(&["report", "--lcov"]),
            strs(&["clean"]),
            strs(&["show-env"]),
            strs(&["run", "--bin", "scheduler"]),
            strs(&["--features", "nextest"]),
            strs(&["--manifest-path", "nextest"]),
            strs(&["--", "nextest"]),
        ] {
            assert!(
                !llvm_cov_uses_nextest(&args),
                "raw/report mode must preserve its exact feature selection: {args:?}",
            );
        }
    }

    #[test]
    fn tool_config_injection_reaches_every_nextest_backed_cargo_route_only() {
        let inject = |args| {
            crate::nextest_config::inject_with_path(
                args,
                std::path::Path::new("/tmp/ktstr-nextest.toml"),
            )
        };

        for (sub_argv, args) in [
            (TEST_SUB_ARGV, strs(&["-j", "77"])),
            (COVERAGE_SUB_ARGV, strs(&["--workspace"])),
            (
                LLVM_COV_SUB_ARGV,
                strs(&["--manifest-path", "Cargo.toml", "nextest"]),
            ),
        ] {
            let got = inject_nextest_tool_config_with(sub_argv, args, inject)
                .expect("nextest tool-config injection succeeds");
            assert_eq!(
                got.iter()
                    .filter(|argument| argument.starts_with("--tool-config-file=ktstr:"))
                    .count(),
                1,
                "nextest-backed route {sub_argv:?} must receive exactly one ktstr tool config",
            );
        }

        for raw_args in [
            strs(&[]),
            strs(&["test"]),
            strs(&["report", "--lcov"]),
            strs(&["clean"]),
            strs(&["show-env"]),
        ] {
            let original = raw_args.clone();
            let got = inject_nextest_tool_config_with(LLVM_COV_SUB_ARGV, raw_args, |_| {
                panic!("non-nextest llvm-cov mode must not call the injector")
            })
            .expect("raw llvm-cov passthrough succeeds");
            assert_eq!(got, original);
        }
    }

    #[test]
    fn llvm_cov_no_run_bypasses_metadata_tool_config_and_admission() {
        for (sub_argv, args) in [
            (
                COVERAGE_SUB_ARGV,
                strs(&["--no-run", "--test-threads", "7"]),
            ),
            (
                LLVM_COV_SUB_ARGV,
                strs(&["nextest", "--no-run", "--test-threads", "7"]),
            ),
        ] {
            assert!(!cargo_sub_uses_nextest(sub_argv, &args));
            let uninjected = inject_nextest_tool_config_with(sub_argv, args.clone(), |_| {
                panic!("report-only --no-run must not inject nextest config")
            })
            .unwrap();
            assert_eq!(uninjected, args);
        }
        let raw = strs(&["nextest", "--no-run", "--features", "explicit"]);
        let prepared = prepare_llvm_cov_args_with(raw.clone(), |_| {
            panic!("report-only --no-run must not query Cargo metadata")
        })
        .unwrap();
        assert_eq!(prepared, raw);
    }

    #[test]
    fn reserved_warmup_reuses_the_final_runs_injected_tool_config() {
        let run_args = normalize_nextest_admission(
            TEST_SUB_ARGV,
            inject_nextest_tool_config_with(TEST_SUB_ARGV, strs(&["-j", "77"]), |args| {
                crate::nextest_config::inject_with_path(
                    args,
                    std::path::Path::new("/tmp/ktstr-nextest.toml"),
                )
            })
            .expect("tool-config injection succeeds"),
        );
        let warm_args = prebuild_no_run_args(&run_args);

        for args in [&run_args, &warm_args] {
            assert_eq!(
                args.iter()
                    .filter(|argument| {
                        argument.as_str() == "--tool-config-file=ktstr:/tmp/ktstr-nextest.toml"
                    })
                    .count(),
                1,
            );
            assert_eq!(
                args.iter()
                    .filter(|argument| {
                        argument.as_str()
                            == format!("--test-threads={ORCHESTRATED_NEXTEST_TEST_THREADS}")
                    })
                    .count(),
                1,
            );
            assert!(!args.iter().any(|argument| argument == "-j"));
        }
        assert!(warm_args.iter().any(|argument| argument == "--no-run"));
    }

    #[test]
    fn cached_nextest_store_is_authoritatively_remapped_out_of_stable_source() {
        let fixture = tempfile::tempdir().expect("temporary nextest output-remap fixture");
        let stable_workspace = fixture.path().join("stable-source/source/primary");
        let invocation = stable_workspace.join("member");
        let config_dir = stable_workspace.join(".config");
        let output_target = fixture.path().join("stable-build/target");
        std::fs::create_dir_all(&invocation).expect("create stable invocation");
        std::fs::create_dir_all(&config_dir).expect("create stable nextest config directory");
        std::fs::write(
            config_dir.join("nextest.toml"),
            r#"
[store]
dir = "repo-relative-store"

[profile.ci]
test-threads = 77

[profile.ci.junit]
path = "junit.xml"
"#,
        )
        .expect("write stable nextest config");

        let args = strs(&["--profile", "ci", "--", "--nocapture"]);
        let remapped =
            remap_nextest_store_output(&args, &stable_workspace, &invocation, &output_target)
                .expect("remap nextest store");
        let separator = remapped
            .iter()
            .position(|argument| argument == "--")
            .expect("test-binary separator retained");
        let overlay = remapped[..separator]
            .iter()
            .find_map(|argument| argument.strip_prefix("--config-file="))
            .map(PathBuf::from)
            .expect("authoritative config-file injected before suffix");
        assert_eq!(&remapped[separator..], &["--", "--nocapture"]);
        assert!(overlay.starts_with(&output_target));

        let config = std::fs::read_to_string(&overlay)
            .expect("read generated nextest config")
            .parse::<toml::Table>()
            .expect("parse generated nextest config");
        assert_eq!(
            config["store"]["dir"].as_str(),
            output_target.join("nextest").to_str(),
            "repo-relative store must be overridden by the private writable target",
        );
        assert_eq!(
            config["profile"]["ci"]["test-threads"].as_integer(),
            Some(77),
            "all non-output repository config semantics must be preserved",
        );
        assert_eq!(
            config["profile"]["ci"]["junit"]["path"].as_str(),
            Some("junit.xml"),
        );
        assert!(
            std::fs::read_to_string(config_dir.join("nextest.toml"))
                .expect("original config remains readable")
                .contains("repo-relative-store"),
            "immutable source config must not be modified",
        );
    }

    #[test]
    fn explicit_nextest_config_is_rewritten_in_place_without_touching_suffix() {
        let fixture = tempfile::tempdir().expect("temporary explicit nextest config fixture");
        let workspace = fixture.path().join("workspace");
        let invocation = workspace.join("member");
        let output_target = fixture.path().join("output/target");
        std::fs::create_dir_all(&invocation).expect("create invocation directory");
        std::fs::write(
            invocation.join("custom-nextest.toml"),
            "[profile.default]\nretries = 9\n",
        )
        .expect("write explicit nextest config");
        let args = strs(&[
            "--config-file",
            "custom-nextest.toml",
            "--run-ignored",
            "all",
            "--",
            "--config-file",
            "opaque-to-nextest",
        ]);

        let remapped = remap_nextest_store_output(&args, &workspace, &invocation, &output_target)
            .expect("remap explicit nextest config");
        assert!(
            !remapped
                .iter()
                .any(|argument| argument == "custom-nextest.toml")
        );
        assert_eq!(
            &remapped[remapped
                .iter()
                .position(|argument| argument == "--")
                .expect("suffix separator retained")..],
            &["--", "--config-file", "opaque-to-nextest"],
        );
        let overlay = remapped
            .iter()
            .find_map(|argument| argument.strip_prefix("--config-file="))
            .expect("explicit config replaced by overlay");
        let config = std::fs::read_to_string(overlay)
            .expect("read explicit overlay")
            .parse::<toml::Table>()
            .expect("parse explicit overlay");
        assert_eq!(
            config["profile"]["default"]["retries"].as_integer(),
            Some(9),
        );
        assert_eq!(
            config["store"]["dir"].as_str(),
            output_target.join("nextest").to_str(),
        );
    }

    #[test]
    fn admission_normalization_replaces_every_nextest_slot_spelling() {
        let expected_threads = format!("--test-threads={ORCHESTRATED_NEXTEST_TEST_THREADS}");
        for sub_argv in [TEST_SUB_ARGV, COVERAGE_SUB_ARGV] {
            assert_eq!(
                normalize_nextest_admission(
                    sub_argv,
                    strs(&[
                        "-j",
                        "192",
                        "-j64",
                        "-j=32",
                        "--test-threads",
                        "16",
                        "--test-threads=8",
                        "--locked",
                        "--",
                        "-j",
                        "2",
                    ]),
                ),
                vec![
                    "--locked".to_string(),
                    expected_threads.clone(),
                    "--".to_string(),
                    "-j".to_string(),
                    "2".to_string(),
                ],
                "{sub_argv:?} must have one admission scheduler and preserve the opaque suffix",
            );
        }
    }

    #[test]
    fn admission_normalization_preserves_invalid_values_for_nextest_to_reject() {
        let expected_threads = format!("--test-threads={ORCHESTRATED_NEXTEST_TEST_THREADS}");
        let overflow = "999999999999999999999999999999999999999999";
        assert_eq!(
            normalize_nextest_admission(
                TEST_SUB_ARGV,
                vec![
                    "-j".into(),
                    "--locked".into(),
                    "--test-threads".into(),
                    "junk".into(),
                    "-j0".into(),
                    "-j=0".into(),
                    "--test-threads=0".into(),
                    format!("-j{overflow}"),
                    format!("--test-threads={overflow}"),
                ],
            ),
            vec![
                "-j".to_string(),
                "--locked".to_string(),
                "--test-threads".to_string(),
                "junk".to_string(),
                "-j0".to_string(),
                "-j=0".to_string(),
                "--test-threads=0".to_string(),
                format!("-j{overflow}"),
                format!("--test-threads={overflow}"),
                expected_threads,
            ],
            "the wrapper must not turn malformed user input into a successful invocation",
        );
        assert_eq!(
            normalize_nextest_admission(TEST_SUB_ARGV, strs(&["--test-threads"])),
            vec![
                "--test-threads".to_string(),
                format!("--test-threads={ORCHESTRATED_NEXTEST_TEST_THREADS}"),
            ],
            "a missing separate value remains visible to nextest",
        );
    }

    #[test]
    fn admission_normalization_accepts_nextest_negative_and_num_cpus_values() {
        assert_eq!(
            normalize_nextest_admission(
                TEST_SUB_ARGV,
                strs(&[
                    "-j",
                    "-2",
                    "-j-3",
                    "-j=-4",
                    "--test-threads",
                    "num-cpus",
                    "--test-threads=-5",
                ]),
            ),
            vec![format!(
                "--test-threads={ORCHESTRATED_NEXTEST_TEST_THREADS}"
            )],
        );
    }

    #[test]
    fn raw_llvm_cov_translates_global_build_jobs_and_normalizes_nextest_region() {
        let got = normalize_nextest_admission(
            LLVM_COV_SUB_ARGV,
            strs(&[
                "-j",
                "12",
                "--jobs=10",
                "--manifest-path",
                "Cargo.toml",
                "nextest",
                "-j192",
                "--test-threads",
                "7",
                "--features",
                "integration",
                "--",
                "-j",
                "3",
            ]),
        );
        assert_eq!(
            got,
            vec![
                "--manifest-path".to_string(),
                "Cargo.toml".to_string(),
                "nextest".to_string(),
                "--features".to_string(),
                "integration".to_string(),
                "--build-jobs".to_string(),
                "10".to_string(),
                format!("--test-threads={ORCHESTRATED_NEXTEST_TEST_THREADS}"),
                "--".to_string(),
                "-j".to_string(),
                "3".to_string(),
            ],
        );
        let report = strs(&["-j", "12", "report", "--test-threads=7"]);
        assert_eq!(
            normalize_nextest_admission(LLVM_COV_SUB_ARGV, report.clone()),
            report,
            "non-nextest raw llvm-cov must remain byte-for-byte passthrough"
        );
    }

    #[test]
    fn direct_cached_build_projects_only_cargo_build_surface() {
        let args = normalize_nextest_admission(
            LLVM_COV_SUB_ARGV,
            strs(&[
                "--branch",
                "--manifest-path",
                "Cargo.toml",
                "nextest",
                "--workspace",
                "--exclude-from-test",
                "slow-package",
                "--features",
                "integration,wprof",
                "--lcov",
                "--output-path",
                "coverage/lcov.info",
                "--retries",
                "2",
                "-E",
                "test(/scheduler/)",
                "name-filter",
            ]),
        );
        assert_eq!(
            direct_nextest_build_surface_args(LLVM_COV_SUB_ARGV, &args).unwrap(),
            strs(&[
                "--manifest-path",
                "Cargo.toml",
                "--workspace",
                "--exclude",
                "slow-package",
                "--features",
                "integration,wprof",
            ]),
            "the producer receives Cargo selectors, not coverage/report/run controls",
        );
    }

    #[test]
    fn cached_reuse_run_keeps_only_nextest_runtime_surface() {
        let args = strs(&[
            "--branch",
            "--manifest-path",
            "Cargo.toml",
            "nextest",
            "--workspace",
            "--features",
            "integration,wprof",
            "--timings=html",
            "--profile",
            "ci",
            "--tool-config-file=ktstr:/tmp/nextest.toml",
            "--test-threads=1000000",
            "--retries",
            "2",
            "--lcov",
            "--output-path",
            "lcov.info",
            "-E",
            "test(/scheduler/)",
            "name-filter",
            "--",
            "--nocapture",
        ]);
        assert_eq!(
            cached_nextest_run_args(LLVM_COV_SUB_ARGV, &args),
            strs(&[
                "--profile",
                "ci",
                "--tool-config-file=ktstr:/tmp/nextest.toml",
                "--test-threads=1000000",
                "--retries",
                "2",
                "-E",
                "test(/scheduler/)",
                "name-filter",
                "--",
                "--nocapture",
            ]),
            "reuse-build conflicts are removed while every run selector remains",
        );
    }

    #[test]
    fn ignore_run_fail_forces_complete_nextest_run_without_touching_test_args() {
        assert_eq!(
            force_nextest_no_fail_fast(strs(&[
                "--fail-fast",
                "--max-fail",
                "2",
                "-E",
                "all()",
                "--",
                "--fail-fast",
                "--max-fail=1",
            ])),
            strs(&[
                "-E",
                "all()",
                "--no-fail-fast",
                "--",
                "--fail-fast",
                "--max-fail=1",
            ]),
        );
    }

    #[test]
    fn cached_reuse_args_are_inserted_before_test_binary_suffix() {
        assert_eq!(
            inject_nextest_reuse_args(
                TEST_SUB_ARGV,
                strs(&["-E", "all()", "--", "--nocapture"]),
                &strs(&["--cargo-metadata", "/cache/cargo.json"]),
            ),
            strs(&[
                "-E",
                "all()",
                "--cargo-metadata",
                "/cache/cargo.json",
                "--",
                "--nocapture",
            ]),
        );
    }

    #[test]
    fn llvm_cov_show_env_projection_is_instrumentation_only() {
        assert_eq!(
            llvm_cov_show_env_args(
                LLVM_COV_SUB_ARGV,
                &strs(&[
                    "--branch",
                    "--dep-coverage",
                    "serde,anyhow",
                    "--target",
                    "x86_64-unknown-linux-gnu",
                    "nextest",
                    "--workspace",
                    "--features",
                    "integration",
                    "--profile",
                    "ci",
                    "--lcov",
                    "--include-build-script",
                    "--no-cfg-coverage-nightly",
                ]),
            ),
            strs(&[
                "--branch",
                "--dep-coverage",
                "serde,anyhow",
                "--target",
                "x86_64-unknown-linux-gnu",
                "--no-cfg-coverage-nightly",
            ]),
            "dependency coverage changes the rustc-wrapper allow-list, while include-build-script is report-only",
        );
    }

    #[test]
    fn llvm_cov_show_env_parser_decodes_shell_escape_without_exec() {
        let parsed = parse_llvm_cov_show_env(
            b"LLVM_PROFILE_FILE='/tmp/a b/'ktstr-%p.profraw\nRUSTFLAGS=-Cinstrument-coverage\n",
        )
        .unwrap();
        assert_eq!(
            parsed,
            vec![
                (
                    OsString::from("LLVM_PROFILE_FILE"),
                    OsString::from("/tmp/a b/ktstr-%p.profraw"),
                ),
                (
                    OsString::from("RUSTFLAGS"),
                    OsString::from("-Cinstrument-coverage"),
                ),
            ],
        );
    }

    #[test]
    fn cached_coverage_retargets_profile_writes_to_the_report_scan_root() {
        let target = Path::new("/cache/materialized/target");
        let build = Path::new("/cache/materialized/build");
        let mut environment = vec![
            (
                OsString::from("CARGO_LLVM_COV_TARGET_DIR"),
                OsString::from("/producer/target"),
            ),
            (
                OsString::from("CARGO_LLVM_COV_BUILD_DIR"),
                OsString::from("/producer/build"),
            ),
            (
                OsString::from("LLVM_PROFILE_FILE"),
                OsString::from("/producer/private/fixture-%p-%m.profraw"),
            ),
        ];

        retarget_llvm_cov_environment(&mut environment, target, build);

        let environment = environment.into_iter().collect::<BTreeMap<_, _>>();
        assert_eq!(
            environment
                .get(OsStr::new("CARGO_LLVM_COV_TARGET_DIR"))
                .map(OsString::as_os_str),
            Some(target.as_os_str()),
        );
        assert_eq!(
            environment
                .get(OsStr::new("CARGO_LLVM_COV_BUILD_DIR"))
                .map(OsString::as_os_str),
            Some(build.as_os_str()),
        );
        let profile = environment
            .get(OsStr::new("LLVM_PROFILE_FILE"))
            .expect("retargeted profile environment");
        assert_eq!(
            profile,
            &target.join("fixture-%p-%m.profraw").into_os_string(),
        );
        assert_eq!(
            Path::new(profile).parent(),
            Some(target),
            "cargo-llvm-cov report scans only the immediate target directory",
        );
    }

    #[test]
    fn cached_build_paths_are_rebased_into_the_stable_source() {
        assert_eq!(
            remap_cached_build_paths(
                &strs(&[
                    "--manifest-path=/work/member/Cargo.toml",
                    "--target",
                    "targets/custom.json",
                    "--features",
                    "integration",
                ]),
                Path::new("/work"),
                Path::new("/work/member"),
                Path::new("/cache/source"),
            ),
            strs(&[
                "--manifest-path=/cache/source/member/Cargo.toml",
                "--target",
                "/cache/source/member/targets/custom.json",
                "--features",
                "integration",
            ]),
        );
    }

    #[test]
    fn cached_coverage_report_is_report_only_and_writes_to_the_invocation_tree() {
        let metadata = llvm_cov_feature_metadata();
        let report = cached_llvm_cov_report_args(
            LLVM_COV_SUB_ARGV,
            &strs(&[
                "--cargo-profile",
                "release",
                "nextest",
                "--workspace",
                "--features",
                "integration,wprof",
                "--profile",
                "ci",
                "--test-threads=1000000",
                "--lcov",
                "--output-path",
                "coverage/lcov.info",
            ]),
            false,
            &metadata,
            Path::new("/work"),
        );
        assert_eq!(
            report,
            strs(&[
                "report",
                "--profile",
                "release",
                "--lcov",
                "--output-path",
                "/work/coverage/lcov.info",
                "--package",
                "ordinary-scheduler",
                "--package",
                "renamed-scheduler",
            ]),
            "nextest runtime controls stay out of the separate report command",
        );
    }

    #[test]
    fn cached_coverage_report_partitions_build_and_run_options_from_report_options() {
        let metadata = llvm_cov_feature_metadata();
        assert_eq!(
            cached_llvm_cov_report_args(
                LLVM_COV_SUB_ARGV,
                &strs(&[
                    "nextest",
                    "--workspace",
                    "--features",
                    "integration,wprof",
                    "-Fextra",
                    "--all-features",
                    "--no-default-features",
                    "--build-jobs",
                    "32",
                    "--profile",
                    "ci",
                    "--test-threads=1000000",
                    "--retries",
                    "2",
                    "--branch",
                    "--lcov",
                    "--fail-under-lines",
                    "80",
                    "--output-path=coverage/lcov.info",
                ]),
                false,
                &metadata,
                Path::new("/work"),
            ),
            strs(&[
                "report",
                "--branch",
                "--lcov",
                "--fail-under-lines",
                "80",
                "--output-path=/work/coverage/lcov.info",
                "--package",
                "ordinary-scheduler",
                "--package",
                "renamed-scheduler",
            ]),
            "the report receives only report/package-selection controls, never build features or nextest runtime controls",
        );
    }

    #[test]
    fn coverage_failure_footer_attributes_only_authoritative_nextest_failures() {
        assert!(should_render_nextest_failure_footer(false, false, false));
        assert!(
            !should_render_nextest_failure_footer(false, true, false),
            "a report-only failure must not be blamed on nextest",
        );
        assert!(
            !should_render_nextest_failure_footer(false, false, true),
            "an ignored test failure must not replace a report failure's attribution",
        );
        assert!(
            !should_render_nextest_failure_footer(true, false, false),
            "no failure footer is emitted when the invocation succeeds",
        );
    }

    #[test]
    fn cached_coverage_report_rebases_joined_output_and_manifest_paths() {
        let metadata = llvm_cov_feature_metadata();
        assert_eq!(
            cached_llvm_cov_report_args(
                LLVM_COV_SUB_ARGV,
                &strs(&[
                    "--manifest-path=/w/renamed-scheduler/Cargo.toml",
                    "nextest",
                    "--lcov",
                    "--output-path=coverage/lcov.info",
                    "--output-dir=/tmp/html",
                ]),
                false,
                &metadata,
                Path::new("/w"),
            ),
            strs(&[
                "report",
                "--manifest-path=/w/renamed-scheduler/Cargo.toml",
                "--lcov",
                "--output-path=/w/coverage/lcov.info",
                "--output-dir=/tmp/html",
                "--package",
                "renamed-scheduler",
            ]),
            "an explicit member manifest keeps coverage reporting on that member",
        );
    }

    #[test]
    fn cached_coverage_report_keeps_test_and_report_exclusions_independent() {
        let metadata = llvm_cov_feature_metadata();
        assert_eq!(
            cached_llvm_cov_report_args(
                LLVM_COV_SUB_ARGV,
                &strs(&[
                    "nextest",
                    "--workspace",
                    "--exclude-from-test=ordinary-scheduler",
                    "--exclude-from-report",
                    "renamed-scheduler",
                    "--lcov",
                ]),
                false,
                &metadata,
                Path::new("/w"),
            ),
            strs(&["report", "--lcov", "--package", "ordinary-scheduler",]),
            "report selection ignores test-only exclusions and applies report-only exclusions",
        );
    }

    #[test]
    fn cached_html_report_defaults_to_persistent_original_target_directory() {
        let metadata = llvm_cov_feature_metadata();
        let report = cached_llvm_cov_report_args(
            LLVM_COV_SUB_ARGV,
            &strs(&["nextest", "--html"]),
            false,
            &metadata,
            Path::new("/w/member"),
        );
        assert!(report.windows(2).any(|pair| {
            pair == ["--output-dir".to_string(), "/w/target/llvm-cov".to_string()]
        }));
    }

    #[test]
    fn cached_report_path_equivalence_preserves_existing_llvm_cov_flags() {
        assert_eq!(
            llvm_cov_flags_with_path_equivalence(
                Some(OsString::from("-show-branches=count")),
                Path::new("/cache/source"),
                Path::new("/work/source"),
            )
            .unwrap(),
            OsString::from("-show-branches=count --path-equivalence=/cache/source,/work/source"),
        );
    }

    #[test]
    fn cached_coverage_cleanup_removes_only_regular_profraw_shards() {
        use std::os::unix::fs::symlink;

        let root = tempfile::tempdir().expect("coverage cleanup fixture");
        let raw_a = root.path().join("ktstr-a.profraw");
        let raw_b = root.path().join("ktstr-b.profraw");
        let merged = root.path().join("ktstr.profdata");
        let report = root.path().join("lcov.info");
        let named_directory = root.path().join("keep.profraw");
        let named_symlink = root.path().join("keep-link.profraw");
        std::fs::write(&raw_a, b"raw-a").expect("write first raw shard");
        std::fs::write(&raw_b, b"raw-b").expect("write second raw shard");
        std::fs::write(&merged, b"merged").expect("write merged profile");
        std::fs::write(&report, b"report").expect("write report");
        std::fs::create_dir(&named_directory).expect("create profraw-named directory");
        std::fs::write(named_directory.join("nested.profraw"), b"nested")
            .expect("write nested raw fixture");
        symlink("ktstr-a.profraw", &named_symlink).expect("create profraw-named symlink");

        let cleanup = remove_cached_profraw_shards(root.path());

        assert_eq!(cleanup.removed, 2);
        assert_eq!(cleanup.failures, 0, "{cleanup:?}");
        assert!(!raw_a.exists());
        assert!(!raw_b.exists());
        assert_eq!(std::fs::read(merged).unwrap(), b"merged");
        assert_eq!(std::fs::read(report).unwrap(), b"report");
        assert!(named_directory.join("nested.profraw").exists());
        assert!(
            std::fs::symlink_metadata(named_symlink)
                .expect("profraw-named symlink remains")
                .file_type()
                .is_symlink()
        );
    }

    #[test]
    fn cached_coverage_cleanup_tolerates_an_already_deleted_directory() {
        let root = tempfile::tempdir().expect("coverage cleanup fixture");
        let deleted = root.path().join("already-deleted");
        assert_eq!(
            remove_cached_profraw_shards(&deleted),
            ProfrawCleanup::default(),
        );
    }

    #[test]
    fn cached_coverage_cleanup_diagnostics_are_count_and_length_bounded() {
        let mut cleanup = ProfrawCleanup::default();
        for index in 0..10 {
            cleanup.record_failure(format!("{index}:{}", "x".repeat(1000)));
        }
        assert_eq!(cleanup.failures, 10);
        assert_eq!(cleanup.diagnostics.len(), PROFRAW_CLEANUP_DIAGNOSTIC_LIMIT,);
        assert!(cleanup.diagnostics.iter().all(|diagnostic| {
            diagnostic.chars().count() <= PROFRAW_CLEANUP_DIAGNOSTIC_CHARS + 1
        }));
        let message = cleanup
            .failure_message(Path::new("/cache/profraw"))
            .expect("cleanup failures are rendered");
        assert!(message.contains("10 failure(s)"));
        assert!(message.contains("6 additional failure(s) omitted"));
    }

    #[test]
    fn cached_coverage_stages_raw_shards_until_success() {
        let root = tempfile::tempdir().expect("coverage staging fixture");
        let profiles = root.path().join("profiles");
        let recovery = root.path().join("recovery");
        std::fs::create_dir(&profiles).unwrap();
        let raw_a = profiles.join("a.profraw");
        let raw_b = profiles.join("b.profraw");
        let merged = profiles.join("workspace.profdata");
        std::fs::write(&raw_a, b"raw-a").unwrap();
        std::fs::write(&raw_b, b"raw-b").unwrap();
        std::fs::write(&merged, b"merged").unwrap();

        let staged = stage_profraw_for_report_in(&profiles, recovery).unwrap();
        assert_eq!(staged.count, 2);
        assert!(!raw_a.exists());
        assert!(!raw_b.exists());
        let staging_path = staged.directory.as_ref().unwrap().path().to_path_buf();
        assert_eq!(
            std::fs::read(staging_path.join("a.profraw")).unwrap(),
            b"raw-a"
        );
        drop(staged);

        assert!(
            !staging_path.exists(),
            "successful report drops staged raw data"
        );
        assert_eq!(std::fs::read(merged).unwrap(), b"merged");
    }

    #[test]
    fn cached_coverage_failure_persists_raw_and_merged_profiles() {
        use std::os::unix::fs::MetadataExt as _;

        let root = tempfile::tempdir().expect("coverage recovery fixture");
        let profiles = root.path().join("profiles");
        let recovery = root.path().join("recovery");
        std::fs::create_dir(&profiles).unwrap();
        let raw = profiles.join("runtime.profraw");
        let merged = profiles.join("workspace.profdata");
        std::fs::write(&raw, b"runtime-raw").unwrap();
        std::fs::write(&merged, b"merged-profile").unwrap();

        let staged = stage_profraw_for_report_in(&profiles, recovery).unwrap();
        let retained = staged
            .persist(&merged)
            .unwrap()
            .expect("a failed report retains its inputs");

        assert!(!raw.exists());
        assert_eq!(
            std::fs::read(retained.join("runtime.profraw")).unwrap(),
            b"runtime-raw",
        );
        assert_eq!(
            std::fs::read(retained.join("merged.profdata")).unwrap(),
            b"merged-profile",
        );
        let source_identity = std::fs::metadata(&merged).unwrap();
        let retained_identity = std::fs::metadata(retained.join("merged.profdata")).unwrap();
        assert_eq!(
            source_identity.dev(),
            retained_identity.dev(),
            "strict recovery reflinks stay on the cache filesystem",
        );
        assert_ne!(
            source_identity.ino(),
            retained_identity.ino(),
            "the retained profile must be an independent COW inode",
        );
        std::fs::write(retained.join("merged.profdata"), b"private-write").unwrap();
        assert_eq!(
            std::fs::read(&merged).unwrap(),
            b"merged-profile",
            "writes to the retained COW inode must not alter the live merged profile",
        );
    }

    #[test]
    fn cached_coverage_failure_survives_deleted_merged_profile() {
        let root = tempfile::tempdir().expect("coverage deletion recovery fixture");
        let profiles = root.path().join("profiles");
        let recovery = root.path().join("recovery");
        std::fs::create_dir(&profiles).unwrap();
        let raw = profiles.join("runtime.profraw");
        let merged = profiles.join("workspace.profdata");
        std::fs::write(&raw, b"runtime-raw").unwrap();
        std::fs::write(&merged, b"merged-profile").unwrap();

        let staged = stage_profraw_for_report_in(&profiles, recovery).unwrap();
        std::fs::remove_file(&merged).unwrap();
        let retained = staged
            .persist(&merged)
            .unwrap()
            .expect("raw recovery remains useful after merged-profile deletion");

        assert_eq!(
            std::fs::read(retained.join("runtime.profraw")).unwrap(),
            b"runtime-raw",
        );
        assert!(
            !retained.join("merged.profdata").exists(),
            "a deleted merged profile is not recreated from stale state",
        );
    }

    fn recovery_bundle_fixture(
        parent: &Path,
        name: &str,
        bytes: usize,
        modified: std::time::SystemTime,
    ) -> PathBuf {
        let bundle = parent.join(name);
        std::fs::create_dir_all(&bundle).unwrap();
        std::fs::write(bundle.join("payload.profraw"), vec![b'x'; bytes]).unwrap();
        std::fs::File::open(&bundle)
            .unwrap()
            .set_times(std::fs::FileTimes::new().set_modified(modified))
            .unwrap();
        bundle
    }

    #[test]
    fn coverage_recovery_gc_removes_expired_bundles_deterministically() {
        let root = tempfile::tempdir().unwrap();
        let now = std::time::SystemTime::UNIX_EPOCH + std::time::Duration::from_secs(1_000);
        let expired = recovery_bundle_fixture(
            root.path(),
            "report-expired",
            4,
            now - std::time::Duration::from_secs(500),
        );
        let recent = recovery_bundle_fixture(
            root.path(),
            "report-recent",
            4,
            now - std::time::Duration::from_secs(10),
        );

        gc_coverage_recovery_bundles_in(
            root.path(),
            now,
            CoverageRecoveryLimits {
                max_age: std::time::Duration::from_secs(100),
                max_bundles: 8,
                max_bytes: 1_024,
            },
            None,
        )
        .unwrap();

        assert!(!expired.exists());
        assert!(recent.exists());
    }

    #[test]
    fn coverage_recovery_gc_enforces_count_and_bytes_oldest_first() {
        let root = tempfile::tempdir().unwrap();
        let now = std::time::SystemTime::UNIX_EPOCH + std::time::Duration::from_secs(2_000);
        let oldest = recovery_bundle_fixture(
            root.path(),
            "report-a",
            4,
            now - std::time::Duration::from_secs(30),
        );
        let middle = recovery_bundle_fixture(
            root.path(),
            "report-b",
            5,
            now - std::time::Duration::from_secs(20),
        );
        let newest = recovery_bundle_fixture(
            root.path(),
            "report-c",
            6,
            now - std::time::Duration::from_secs(10),
        );

        gc_coverage_recovery_bundles_in(
            root.path(),
            now,
            CoverageRecoveryLimits {
                max_age: std::time::Duration::from_secs(1_000),
                max_bundles: 2,
                max_bytes: 10,
            },
            None,
        )
        .unwrap();

        assert!(!oldest.exists(), "count bound removes the oldest bundle");
        assert!(
            !middle.exists(),
            "byte bound continues oldest-first pruning"
        );
        assert!(newest.exists(), "the newest diagnostic bundle is retained");
    }

    #[test]
    fn coverage_recovery_gc_never_reclaims_a_live_bundle() {
        let root = tempfile::tempdir().unwrap();
        let live = create_coverage_recovery_dir(root.path()).unwrap();
        let live_path = live.path().to_path_buf();
        std::fs::write(live.path().join("active.profraw"), b"active").unwrap();
        let now = std::time::SystemTime::now() + std::time::Duration::from_secs(60);

        gc_coverage_recovery_bundles_in(
            root.path(),
            now,
            CoverageRecoveryLimits {
                max_age: std::time::Duration::ZERO,
                max_bundles: 1,
                max_bytes: 1,
            },
            None,
        )
        .unwrap();

        assert!(live_path.exists());
        drop(live);
        assert!(
            !live_path.exists(),
            "green-path TempDir cleanup remains intact"
        );
    }

    #[test]
    fn no_report_bundle_is_persistent_and_contains_replayable_report_script() {
        use std::os::unix::fs::PermissionsExt as _;

        let root = tempfile::tempdir().unwrap();
        let profiles = root.path().join("profiles");
        let recovery = root.path().join("recovery");
        std::fs::create_dir_all(&profiles).unwrap();
        std::fs::write(profiles.join("runtime.profraw"), b"raw").unwrap();
        let merged = profiles.join("workspace.profdata");
        std::fs::write(&merged, b"merged").unwrap();
        let staged = stage_profraw_for_report_in(&profiles, recovery).unwrap();

        let retained = staged
            .persist_with(&merged, true, |bundle| {
                let target = bundle.join("artifacts/target");
                let build = target.join("build");
                std::fs::create_dir_all(&build).unwrap();
                write_retained_coverage_report_script(
                    bundle,
                    Path::new("/work/it's-here"),
                    &[(
                        OsString::from("LLVM_PROFILE_FILE"),
                        OsString::from("/ephemeral/old-%p.profraw"),
                    )],
                    OsStr::new("--path-equivalence=/cache,/work"),
                    &strs(&["report", "--lcov", "--output-path=/work/lcov.info"]),
                    &target,
                    &build,
                )?;
                Ok(())
            })
            .unwrap()
            .unwrap();

        assert_eq!(
            std::fs::read(retained.join("runtime.profraw")).unwrap(),
            b"raw"
        );
        assert_eq!(
            std::fs::read(retained.join("merged.profdata")).unwrap(),
            b"merged"
        );
        let script_path = retained.join("report.sh");
        let script = std::fs::read(&script_path).unwrap();
        assert!(
            script
                .windows(b"artifacts/target".len())
                .any(|window| window == b"artifacts/target")
        );
        assert!(
            !script
                .windows(b"/ephemeral/".len())
                .any(|window| window == b"/ephemeral/")
        );
        assert!(
            script
                .windows(b"'\"'\"'".len())
                .any(|window| window == b"'\"'\"'")
        );
        assert!(
            script
                .windows(b"flock --exclusive 9".len())
                .any(|window| window == b"flock --exclusive 9"),
            "replay must hold the same liveness lock as recovery GC",
        );
        assert_eq!(
            std::fs::metadata(script_path).unwrap().permissions().mode() & 0o777,
            0o700,
        );
        assert!(
            retained.exists(),
            "TempDir ownership was detached on retention"
        );
    }

    #[test]
    fn cached_coverage_missing_profile_directory_is_an_empty_input_set() {
        let root = tempfile::tempdir().expect("coverage deletion fixture");
        let missing = root.path().join("deleted-materialization");
        assert!(profraw_files(&missing).unwrap().is_empty());
        let staged = stage_profraw_for_report_in(&missing, root.path().join("recovery")).unwrap();
        assert_eq!(staged.count, 0);
        assert!(staged.directory.is_none());
    }

    #[test]
    fn cached_coverage_rejects_non_regular_raw_profile_paths() {
        use std::os::unix::fs::symlink;

        let root = tempfile::tempdir().expect("coverage input fixture");
        let target = root.path().join("target");
        std::fs::create_dir(&target).unwrap();
        std::fs::write(target.join("actual"), b"raw").unwrap();
        symlink("actual", target.join("alias.profraw")).unwrap();
        let error = profraw_files(&target).unwrap_err();
        assert!(error.contains("not a regular file"), "{error}");
    }

    #[test]
    fn llvm_profdata_path_matches_rustup_llvm_tools_layout() {
        assert_eq!(
            llvm_profdata_from_target_libdir(Path::new(
                "/toolchain/lib/rustlib/x86_64-unknown-linux-gnu/lib",
            ))
            .unwrap(),
            PathBuf::from("/toolchain/lib/rustlib/x86_64-unknown-linux-gnu/bin/llvm-profdata",),
        );
    }

    #[test]
    fn cached_profdata_name_matches_non_ktstr_workspace_basename() {
        assert_eq!(
            cargo_llvm_cov_profdata_path(
                Path::new("/cache/private-target"),
                Path::new("/work/checkouts/acme-schedulers"),
            ),
            PathBuf::from("/cache/private-target/acme-schedulers.profdata"),
        );
    }

    #[test]
    fn llvm_profdata_merge_argv_and_explicit_flags_are_exact() {
        let environment = vec![(
            OsString::from("LLVM_PROFDATA_FLAGS"),
            OsString::from("-num-threads=4 -debug-info-correlate"),
        )];
        let flags = llvm_profdata_flags(&environment).unwrap();
        let mut command = Command::new("/tool/llvm-profdata");
        configure_llvm_profdata_merge(
            &mut command,
            Path::new("/profiles/inputs"),
            Path::new("/profiles/merged.profdata"),
            Some("any"),
            flags,
        );
        assert_eq!(command.get_program(), OsStr::new("/tool/llvm-profdata"));
        assert_eq!(
            command.get_args().collect::<Vec<_>>(),
            vec![
                OsStr::new("merge"),
                OsStr::new("-sparse"),
                OsStr::new("-f"),
                OsStr::new("/profiles/inputs"),
                OsStr::new("-o"),
                OsStr::new("/profiles/merged.profdata"),
                OsStr::new("-failure-mode=any"),
                OsStr::new("-num-threads=4"),
                OsStr::new("-debug-info-correlate"),
            ],
        );
    }

    #[test]
    fn dropped_shards_names_the_warned_inputs_only() {
        let inputs = vec![
            PathBuf::from("/p/valid.profdata"),
            PathBuf::from("/p/torn.profraw"),
            PathBuf::from("/p/ok.profraw"),
        ];
        // The reason text differs per corruption kind (header-corrupt here,
        // encoding-format elsewhere); matching keys on the input path.
        let stderr = "warning: /p/torn.profraw: invalid instrumentation \
             profile data (file header is corrupt)\n";
        assert_eq!(
            dropped_shards_from_stderr(stderr, &inputs),
            vec![Path::new("/p/torn.profraw")],
        );
    }

    #[test]
    fn dropped_shards_empty_without_warnings() {
        let inputs = vec![PathBuf::from("/p/a.profraw")];
        assert!(dropped_shards_from_stderr("", &inputs).is_empty());
    }

    #[test]
    fn dropped_shards_ignores_warnings_for_foreign_paths() {
        let inputs = vec![PathBuf::from("/p/a.profraw")];
        let stderr = "warning: /other/x.profraw: unrecognized instrumentation \
             profile encoding format\n";
        assert!(dropped_shards_from_stderr(stderr, &inputs).is_empty());
    }

    /// Build a valid indexed `.profdata` from a text profile using the pinned
    /// llvm-profdata (always present via the `llvm-tools-preview` component).
    fn valid_profdata_fixture(dir: &Path, name: &str) -> PathBuf {
        let tool = resolve_llvm_profdata(dir, &[]).expect("resolve llvm-profdata");
        let text = dir.join("fixture.proftext");
        std::fs::write(&text, "somefunc\n0x0\n1\n1\n\n").unwrap();
        let out = dir.join(name);
        let status = Command::new(&tool)
            .arg("merge")
            .arg("-o")
            .arg(&out)
            .arg(&text)
            .status()
            .expect("spawn llvm-profdata to build a valid fixture");
        assert!(status.success(), "build valid profdata fixture");
        out
    }

    /// A torn shard (watchdog SIGKILL mid-write) must cost only that
    /// process's coverage: the merge succeeds on the surviving valid input.
    #[test]
    fn merge_profdata_tolerates_a_corrupt_shard_beside_a_valid_one() {
        let dir = tempfile::tempdir().unwrap();
        let valid = valid_profdata_fixture(dir.path(), "valid.profdata");
        let corrupt = dir.path().join("torn.profraw");
        std::fs::write(&corrupt, b"not a profile").unwrap();
        let out = dir.path().join("merged.profdata");
        merge_profdata(dir.path(), &[], &[valid, corrupt], &out, None)
            .expect("a valid shard must survive a torn sibling");
        assert!(out.is_file(), "the merged profile must be published");
    }

    /// A genuinely all-corrupt set has no coverage to salvage and must still
    /// fail loudly rather than publish an empty profile.
    #[test]
    fn merge_profdata_fails_when_every_shard_is_corrupt() {
        let dir = tempfile::tempdir().unwrap();
        let a = dir.path().join("a.profraw");
        let b = dir.path().join("b.profraw");
        std::fs::write(&a, b"garbage-a").unwrap();
        std::fs::write(&b, b"garbage-b").unwrap();
        let out = dir.path().join("merged.profdata");
        let error = merge_profdata(dir.path(), &[], &[a, b], &out, None)
            .expect_err("an all-corrupt set must fail");
        assert!(
            error.contains("llvm-profdata merge exited"),
            "expected a merge-exit failure, got: {error}",
        );
        assert!(!out.exists(), "no partial profile on total failure");
    }

    /// An explicit caller failure mode (from the report's `--failure-mode`)
    /// overrides the tolerant default, so a strict `any` still rejects a torn
    /// shard beside a valid one.
    #[test]
    fn merge_profdata_honors_an_explicit_strict_failure_mode() {
        let dir = tempfile::tempdir().unwrap();
        let valid = valid_profdata_fixture(dir.path(), "valid.profdata");
        let corrupt = dir.path().join("torn.profraw");
        std::fs::write(&corrupt, b"not a profile").unwrap();
        let out = dir.path().join("merged.profdata");
        let error = merge_profdata(dir.path(), &[], &[valid, corrupt], &out, Some("any"))
            .expect_err("explicit -failure-mode=any must reject a torn shard");
        assert!(
            error.contains("llvm-profdata merge exited"),
            "expected a merge-exit failure, got: {error}",
        );
    }

    #[test]
    fn merge_profdata_rejects_an_empty_input_set() {
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("merged.profdata");
        let error = merge_profdata(dir.path(), &[], &[], &out, None)
            .expect_err("an empty input set must fail");
        assert!(error.contains("no coverage profile inputs"), "{error}");
    }

    #[test]
    fn llvm_cov_nextest_detector_skips_every_global_option_value() {
        for option in LLVM_COV_GLOBAL_VALUE_OPTIONS {
            let value_named_nextest = vec![(*option).to_string(), "nextest".to_string()];
            assert!(
                !llvm_cov_uses_nextest(&value_named_nextest),
                "{option} consumes the following `nextest` token as its value",
            );

            let followed_by_subcommand = vec![
                (*option).to_string(),
                "ordinary-value".to_string(),
                "nextest".to_string(),
            ];
            assert!(
                llvm_cov_uses_nextest(&followed_by_subcommand),
                "{option}'s value must be skipped before detecting the real subcommand",
            );
        }
        assert!(
            llvm_cov_uses_nextest(&strs(&[
                "--output-path=nextest",
                "--features=nextest",
                "nextest",
            ])),
            "equals spellings carry their value in-token and cannot hide the subcommand",
        );
    }

    #[test]
    fn llvm_cov_nextest_forwards_the_same_targeted_features_as_coverage() {
        let activations = [crate::feature_discovery::PackageFeatureActivation {
            package: "scx_lavd".to_string(),
            features: vec!["ktstr-tests".to_string()],
        }];
        let coverage_args = strs(&[
            "--workspace",
            "--target",
            "aarch64-unknown-linux-gnu",
            "--features",
            "integration",
        ]);
        let coverage_args =
            crate::feature_discovery::inject_feature_activations(coverage_args, &activations);
        let raw_args = prepare_llvm_cov_args_with(
            strs(&[
                "nextest",
                "--workspace",
                "--target",
                "aarch64-unknown-linux-gnu",
                "--features",
                "integration",
            ]),
            |args| {
                Ok(crate::feature_discovery::inject_feature_activations(
                    args,
                    &activations,
                ))
            },
        )
        .expect("nextest preparation succeeds");

        let coverage = build_cargo_command(
            COVERAGE_SUB_ARGV,
            false,
            None,
            None,
            false,
            false,
            &coverage_args,
        );
        let raw = build_cargo_command(
            LLVM_COV_SUB_ARGV,
            false,
            None,
            None,
            false,
            false,
            &raw_args,
        );
        let coverage_argv = coverage.get_args().collect::<Vec<_>>();
        let raw_argv = raw.get_args().collect::<Vec<_>>();
        assert_eq!(
            raw_argv, coverage_argv,
            "both frontends must hand cargo-llvm-cov the same nextest argv",
        );
        assert_eq!(
            raw_argv,
            [
                "llvm-cov",
                "--no-clean",
                "nextest",
                "--workspace",
                "--target",
                "aarch64-unknown-linux-gnu",
                "--features",
                "integration",
                "--features",
                "scx_lavd/ktstr-tests",
            ]
            .map(std::ffi::OsStr::new),
        );
        assert!(
            !raw_argv
                .iter()
                .any(|argument| *argument == std::ffi::OsStr::new("--all-features")),
            "targeted inference must never widen into --all-features",
        );

        let explicit_all = prepare_llvm_cov_args_with(
            strs(&["nextest", "--workspace", "--all-features"]),
            |args| {
                Ok(crate::feature_discovery::inject_feature_activations(
                    args,
                    &activations,
                ))
            },
        )
        .expect("explicit all-features remains valid");
        assert_eq!(
            explicit_all,
            strs(&["nextest", "--workspace", "--all-features"]),
            "an explicit user --all-features remains authoritative",
        );
    }

    #[test]
    fn llvm_cov_nextest_inference_honors_no_default_rename_package_and_exclude_scope() {
        let metadata = llvm_cov_feature_metadata();
        let selected_renamed = prepare_llvm_cov_args_with(
            strs(&[
                "nextest",
                "-p",
                "renamed-scheduler",
                "--no-default-features",
                "--features",
                "operator-choice",
            ]),
            |args| {
                Ok(crate::feature_discovery::augment_test_features_from_metadata(args, &metadata))
            },
        )
        .expect("renamed package preparation succeeds");
        assert_eq!(
            selected_renamed,
            [
                "nextest",
                "-p",
                "renamed-scheduler",
                "--no-default-features",
                "--features",
                "operator-choice",
                "--features",
                "renamed-scheduler/verify-schedulers",
            ]
            .map(ToString::to_string),
            "raw llvm-cov preserves explicit/no-default modes and infers through dep:<rename>",
        );

        let excluded_renamed = prepare_llvm_cov_args_with(
            strs(&["nextest", "--workspace", "--exclude", "renamed-scheduler"]),
            |args| {
                Ok(crate::feature_discovery::augment_test_features_from_metadata(args, &metadata))
            },
        )
        .expect("workspace exclusion preparation succeeds");
        assert_eq!(
            excluded_renamed,
            [
                "nextest",
                "--workspace",
                "--exclude",
                "renamed-scheduler",
                "--features",
                "ordinary-scheduler/ktstr-tests",
            ]
            .map(ToString::to_string),
            "excluded packages receive no inferred selector while selected workspace peers do",
        );
    }

    #[test]
    fn llvm_cov_nextest_honors_target_specific_test_only_dependency_and_attached_selector() {
        let mut metadata = llvm_cov_feature_metadata();
        let renamed = metadata
            .packages
            .iter_mut()
            .find(|package| package.name.as_str() == "renamed-scheduler")
            .expect("renamed fixture package");
        let dependency = renamed
            .dependencies
            .iter_mut()
            .find(|dependency| dependency.name == "ktstr")
            .expect("renamed fixture ktstr dependency");
        dependency.kind = cargo_metadata::DependencyKind::Development;
        dependency.target = Some(
            r#"cfg(target_os = "linux")"#
                .parse()
                .expect("valid Linux target predicate"),
        );

        let args = strs(&[
            "nextest",
            "--package=renamed-scheduler",
            "--no-default-features",
            "--target=x86_64-unknown-linux-gnu",
            "--test=scheduler-registry",
        ]);
        let linux = crate::feature_discovery::TargetContext::named(
            "x86_64-unknown-linux-gnu",
            vec![
                cargo_platform::Cfg::Name("unix".to_string()),
                cargo_platform::Cfg::KeyPair("target_os".to_string(), "linux".to_string()),
            ],
        );
        let selected = prepare_llvm_cov_args_with(args.clone(), |args| {
            Ok(
                crate::feature_discovery::augment_test_features_from_metadata_for_context(
                    args,
                    &metadata,
                    Some(&linux),
                ),
            )
        })
        .expect("Linux nextest preparation succeeds");
        assert_eq!(
            selected,
            [
                "nextest",
                "--package=renamed-scheduler",
                "--no-default-features",
                "--target=x86_64-unknown-linux-gnu",
                "--test=scheduler-registry",
                "--features",
                "renamed-scheduler/verify-schedulers",
            ]
            .map(ToString::to_string),
            "raw llvm-cov nextest must infer a renamed optional dev-dependency only on its \
             matching target while preserving attached target selection",
        );

        let windows = crate::feature_discovery::TargetContext::named(
            "x86_64-pc-windows-msvc",
            vec![
                cargo_platform::Cfg::Name("windows".to_string()),
                cargo_platform::Cfg::KeyPair("target_os".to_string(), "windows".to_string()),
            ],
        );
        let unselected = prepare_llvm_cov_args_with(args.clone(), |args| {
            Ok(
                crate::feature_discovery::augment_test_features_from_metadata_for_context(
                    args,
                    &metadata,
                    Some(&windows),
                ),
            )
        })
        .expect("Windows nextest preparation succeeds");
        assert_eq!(
            unselected, args,
            "the same test-only dependency must not activate for an opposite target cfg",
        );
    }

    #[test]
    fn llvm_cov_non_test_modes_never_run_feature_preparation() {
        for args in [
            strs(&[]),
            strs(&["test", "--features", "explicit"]),
            strs(&["report", "--features", "explicit"]),
            strs(&["clean", "--workspace"]),
            strs(&["show-env"]),
            strs(&["run", "--bin", "scheduler"]),
        ] {
            let prepared = prepare_llvm_cov_args_with(args.clone(), |_| {
                panic!("non-nextest llvm-cov mode must not run metadata preparation")
            })
            .expect("raw passthrough cannot fail");
            assert_eq!(prepared, args);
        }
    }

    fn strs(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| s.to_string()).collect()
    }

    /// `extract_nextest_filtersets` pulls every filterset spelling (`-E` /
    /// `--filterset` / legacy `--filter-expr`, space / `=` / glued-short) out of
    /// the argv while leaving positional filters and other flags in `rest`.
    #[test]
    fn extract_filtersets_all_spellings() {
        let (filters, rest) = extract_nextest_filtersets(strs(&[
            "-E",
            "test(a)",
            "--filterset",
            "test(b)",
            "--filter-expr=test(c)",
            "-E=test(d)",
            "-Etest(e)",
            "--filterset=test(f)",
            "positional_name",
            "--features",
            "integration",
        ]));
        assert_eq!(
            filters,
            strs(&[
                "test(a)", "test(b)", "test(c)", "test(d)", "test(e)", "test(f)",
            ]),
        );
        // Positional filter + unrelated flag survive untouched (nextest ANDs
        // positional filters with the filterset dimension, so they need no fold).
        assert_eq!(
            rest,
            strs(&["positional_name", "--features", "integration"])
        );
    }

    #[test]
    fn extract_filtersets_preserves_inner_separator_suffix() {
        let args = strs(&[
            "-E",
            "test(cargo_side)",
            "--",
            "-E",
            "test_binary_value",
            "--filterset=opaque",
        ]);
        let (filters, rest) = extract_nextest_filtersets(args);
        assert_eq!(filters, strs(&["test(cargo_side)"]));
        assert_eq!(
            rest,
            strs(&["--", "-E", "test_binary_value", "--filterset=opaque"]),
        );
    }

    /// With no user filterset, the relevant expression is injected verbatim as a
    /// single `-E`, and non-filter passthrough is preserved.
    #[test]
    fn compose_relevant_no_user_filterset() {
        let out = compose_relevant_filter(strs(&["--features", "x"]), "test(r1) | test(r2)");
        assert_eq!(out, strs(&["--features", "x", "-E", "test(r1) | test(r2)"]),);
    }

    /// With user filtersets present, the composition INTERSECTS
    /// (`(relevant) & (u1 | u2)`) so the net effect narrows, never widens
    /// (nextest UNIONs multiple `-E`). The user's `-E` tokens are removed.
    #[test]
    fn compose_relevant_intersects_user_filtersets() {
        let out = compose_relevant_filter(
            strs(&[
                "-E",
                "test(u1)",
                "--features",
                "x",
                "--filterset",
                "test(u2)",
            ]),
            "test(r)",
        );
        assert_eq!(
            out,
            strs(&[
                "--features",
                "x",
                "-E",
                "(test(r)) & ((test(u1)) | (test(u2)))",
            ]),
        );
    }

    #[test]
    fn compose_relevant_inserts_filter_before_inner_separator() {
        let out = compose_relevant_filter(
            strs(&["--features", "x", "--", "-E", "test_binary_value"]),
            "test(relevant)",
        );
        assert_eq!(
            out,
            strs(&[
                "--features",
                "x",
                "-E",
                "test(relevant)",
                "--",
                "-E",
                "test_binary_value",
            ]),
        );
    }

    /// A `none()` relevant expression (the docs-only Empty outcome) composes to
    /// a zero-match filter, so a `--relevant` run of a docs-only change runs no
    /// tests.
    #[test]
    fn compose_relevant_none_expression() {
        let out = compose_relevant_filter(Vec::new(), "none()");
        assert_eq!(out, strs(&["-E", "none()"]));
    }

    /// `apply_relevant_narrowing` is a no-op when `--relevant` is off — the argv
    /// (including any user `-E`) passes through byte-for-byte.
    #[test]
    fn apply_relevant_narrowing_off_is_noop() {
        let args = strs(&["-E", "test(x)", "--features", "y"]);
        let out =
            apply_relevant_narrowing(args.clone(), false, None, None, "main".to_string(), false)
                .expect("no-op path never errors");
        assert_eq!(out, args);
    }

    /// A `cargo metadata` Package JSON object with every required field
    /// (Options as `null`, collections empty); only name/version/id/source
    /// vary across the fixture's packages.
    fn pkg_json(name: &str, version: &str, id: &str, source: &str) -> String {
        format!(
            r#"{{"name":"{name}","version":"{version}","id":"{id}","source":{source},"description":null,"dependencies":[],"license":null,"license_file":null,"targets":[],"features":{{}},"manifest_path":"/w/{name}/Cargo.toml","readme":null,"repository":null,"homepage":null,"documentation":null,"links":null,"publish":null,"default_run":null}}"#
        )
    }

    /// Regression: in a dual-ktstr resolve graph,
    /// `resolved_ktstr_version` must return the ktstr the ROOT's targets
    /// LINK, not the `.max()` over the graph. The user project links ktstr
    /// 0.16 directly while an unrelated dep pulls ktstr 0.20 transitively
    /// (ktstr is pre-1.0, so the two semver-incompatible 0.x coexist).
    /// `.max()` would pick 0.20 and false-abort a run the CLI can drive;
    /// the walk must pick 0.16.
    #[test]
    fn resolved_ktstr_version_picks_linked_not_max() {
        let crates_io = r#""registry+https://github.com/rust-lang/crates.io-index""#;
        let root = "userproj 0.1.0 (path+file:///w/userproj)";
        let k016 = "ktstr 0.16.0 (registry+https://github.com/rust-lang/crates.io-index)";
        let k020 = "ktstr 0.20.0 (registry+https://github.com/rust-lang/crates.io-index)";
        let other = "otherdep 0.1.0 (registry+https://github.com/rust-lang/crates.io-index)";
        let json = format!(
            r#"{{
              "packages":[{up},{a},{b},{o}],
              "workspace_members":["{root}"],
              "resolve":{{
                "root":"{root}",
                "nodes":[
                  {{"id":"{root}","deps":[{{"name":"ktstr","pkg":"{k016}","dep_kinds":[{{"kind":null,"target":null}}]}},{{"name":"otherdep","pkg":"{other}","dep_kinds":[{{"kind":null,"target":null}}]}}],"dependencies":["{k016}","{other}"],"features":[]}},
                  {{"id":"{other}","deps":[{{"name":"ktstr","pkg":"{k020}","dep_kinds":[{{"kind":null,"target":null}}]}}],"dependencies":["{k020}"],"features":[]}},
                  {{"id":"{k016}","deps":[],"dependencies":[],"features":[]}},
                  {{"id":"{k020}","deps":[],"dependencies":[],"features":[]}}
                ]
              }},
              "workspace_root":"/w","target_directory":"/w/target","version":1
            }}"#,
            up = pkg_json("userproj", "0.1.0", root, "null"),
            a = pkg_json("ktstr", "0.16.0", k016, crates_io),
            b = pkg_json("ktstr", "0.20.0", k020, crates_io),
            o = pkg_json("otherdep", "0.1.0", other, crates_io),
        );
        let meta: cargo_metadata::Metadata =
            serde_json::from_str(&json).expect("fixture deserializes as cargo metadata");
        assert_eq!(
            resolved_ktstr_version(&meta),
            Some(&v("0.16.0")),
            "must pick the LINKED ktstr (root's direct dep), not the .max() (0.20.0)",
        );
    }

    /// `resolved_ktstr_version` when the ROOT package IS ktstr (the in-repo
    /// workspace, whose own bins/tests link itself) → its own version.
    #[test]
    fn resolved_ktstr_version_root_is_ktstr() {
        let root = "ktstr 0.19.0 (path+file:///w)";
        let json = format!(
            r#"{{
              "packages":[{k}],
              "workspace_members":["{root}"],
              "resolve":{{"root":"{root}","nodes":[{{"id":"{root}","deps":[],"dependencies":[],"features":[]}}]}},
              "workspace_root":"/w","target_directory":"/w/target","version":1
            }}"#,
            k = pkg_json("ktstr", "0.19.0", root, "null"),
        );
        let meta: cargo_metadata::Metadata =
            serde_json::from_str(&json).expect("fixture deserializes");
        assert_eq!(resolved_ktstr_version(&meta), Some(&v("0.19.0")));
    }

    /// Never-false-abort: a ktstr edge that is ONLY a build-dependency
    /// (the test/bin targets do not link it) must NOT be picked — yields
    /// `None` so the guard SKIPS rather than aborting on an unlinked
    /// version. This is the load-bearing safe-direction branch.
    #[test]
    fn resolved_ktstr_version_skips_build_only_edge() {
        let crates_io = r#""registry+https://github.com/rust-lang/crates.io-index""#;
        let root = "userproj 0.1.0 (path+file:///w/userproj)";
        let kb = "ktstr 0.20.0 (registry+https://github.com/rust-lang/crates.io-index)";
        let json = format!(
            r#"{{
              "packages":[{up},{k}],
              "workspace_members":["{root}"],
              "resolve":{{
                "root":"{root}",
                "nodes":[
                  {{"id":"{root}","deps":[{{"name":"ktstr","pkg":"{kb}","dep_kinds":[{{"kind":"build","target":null}}]}}],"dependencies":["{kb}"],"features":[]}},
                  {{"id":"{kb}","deps":[],"dependencies":[],"features":[]}}
                ]
              }},
              "workspace_root":"/w","target_directory":"/w/target","version":1
            }}"#,
            up = pkg_json("userproj", "0.1.0", root, "null"),
            k = pkg_json("ktstr", "0.20.0", kb, crates_io),
        );
        let meta: cargo_metadata::Metadata =
            serde_json::from_str(&json).expect("fixture deserializes");
        assert_eq!(
            resolved_ktstr_version(&meta),
            None,
            "a build-only ktstr edge is not linked by test/bin targets — skip, not abort",
        );
    }

    #[test]
    fn resolved_ktstr_versions_follow_effective_target_cfg() {
        let crates_io = r#""registry+https://github.com/rust-lang/crates.io-index""#;
        let root = "userproj 0.1.0 (path+file:///w/userproj)";
        let linux_ktstr = "ktstr 0.42.0 (registry+https://github.com/rust-lang/crates.io-index)";
        let windows_ktstr = "ktstr 0.43.0 (registry+https://github.com/rust-lang/crates.io-index)";
        let json = format!(
            r#"{{
              "packages":[{up},{linux},{windows}],
              "workspace_members":["{root}"],
              "resolve":{{
                "root":"{root}",
                "nodes":[
                  {{"id":"{root}","deps":[
                    {{"name":"linux_ktstr","pkg":"{linux_ktstr}","dep_kinds":[{{"kind":null,"target":"cfg(target_os = \"linux\")"}}]}},
                    {{"name":"windows_ktstr","pkg":"{windows_ktstr}","dep_kinds":[{{"kind":null,"target":"cfg(target_os = \"windows\")"}}]}}
                  ],"dependencies":["{linux_ktstr}","{windows_ktstr}"],"features":[]}},
                  {{"id":"{linux_ktstr}","deps":[],"dependencies":[],"features":[]}},
                  {{"id":"{windows_ktstr}","deps":[],"dependencies":[],"features":[]}}
                ]
              }},
              "workspace_root":"/w","target_directory":"/w/target","version":1
            }}"#,
            up = pkg_json("userproj", "0.1.0", root, "null"),
            linux = pkg_json("ktstr", "0.42.0", linux_ktstr, crates_io),
            windows = pkg_json("ktstr", "0.43.0", windows_ktstr, crates_io),
        );
        let meta: cargo_metadata::Metadata =
            serde_json::from_str(&json).expect("target fixture deserializes");
        let root_id = meta.workspace_members.first().expect("workspace root");
        let linux = crate::feature_discovery::TargetContext::named(
            "x86_64-unknown-linux-gnu",
            vec![cargo_platform::Cfg::KeyPair(
                "target_os".to_string(),
                "linux".to_string(),
            )],
        );
        let windows = crate::feature_discovery::TargetContext::named(
            "x86_64-pc-windows-msvc",
            vec![cargo_platform::Cfg::KeyPair(
                "target_os".to_string(),
                "windows".to_string(),
            )],
        );
        assert_eq!(
            linked_ktstr_versions_for_context(&meta, root_id, Some(&linux)),
            vec![&v("0.42.0")],
        );
        assert_eq!(
            linked_ktstr_versions_for_context(&meta, root_id, Some(&windows)),
            vec![&v("0.43.0")],
        );
    }

    /// Virtual workspace: no root package (`resolve.root` = null) — the
    /// walk falls back to every workspace member; a member linking ktstr
    /// resolves to that version.
    #[test]
    fn resolved_ktstr_version_virtual_workspace_walks_members() {
        let crates_io = r#""registry+https://github.com/rust-lang/crates.io-index""#;
        let member = "memberproj 0.1.0 (path+file:///w/memberproj)";
        let k = "ktstr 0.17.0 (registry+https://github.com/rust-lang/crates.io-index)";
        let json = format!(
            r#"{{
              "packages":[{m},{kp}],
              "workspace_members":["{member}"],
              "resolve":{{
                "root":null,
                "nodes":[
                  {{"id":"{member}","deps":[{{"name":"ktstr","pkg":"{k}","dep_kinds":[{{"kind":null,"target":null}}]}}],"dependencies":["{k}"],"features":[]}},
                  {{"id":"{k}","deps":[],"dependencies":[],"features":[]}}
                ]
              }},
              "workspace_root":"/w","target_directory":"/w/target","version":1
            }}"#,
            m = pkg_json("memberproj", "0.1.0", member, "null"),
            kp = pkg_json("ktstr", "0.17.0", k, crates_io),
        );
        let meta: cargo_metadata::Metadata =
            serde_json::from_str(&json).expect("fixture deserializes");
        assert_eq!(resolved_ktstr_version(&meta), Some(&v("0.17.0")));
    }

    #[test]
    fn selected_resolved_versions_follow_exact_glob_and_workspace_selection() {
        let crates_io = r#""registry+https://github.com/rust-lang/crates.io-index""#;
        let alpha = "alpha 0.1.0 (path+file:///w/alpha)";
        let beta = "beta 0.1.0 (path+file:///w/beta)";
        let old = "ktstr 0.18.0 (registry+https://github.com/rust-lang/crates.io-index)";
        let new = "ktstr 0.43.0 (registry+https://github.com/rust-lang/crates.io-index)";
        let json = format!(
            r#"{{
              "packages":[{alpha_pkg},{beta_pkg},{old_pkg},{new_pkg}],
              "workspace_members":["{alpha}","{beta}"],
              "workspace_default_members":["{alpha}"],
              "resolve":{{
                "root":null,
                "nodes":[
                  {{"id":"{alpha}","deps":[{{"name":"ktstr","pkg":"{old}","dep_kinds":[{{"kind":null,"target":null}}]}}],"dependencies":["{old}"],"features":[]}},
                  {{"id":"{beta}","deps":[{{"name":"ktstr","pkg":"{new}","dep_kinds":[{{"kind":null,"target":null}}]}}],"dependencies":["{new}"],"features":[]}},
                  {{"id":"{old}","deps":[],"dependencies":[],"features":[]}},
                  {{"id":"{new}","deps":[],"dependencies":[],"features":[]}}
                ]
              }},
              "workspace_root":"/w","target_directory":"/w/target","version":1
            }}"#,
            alpha_pkg = pkg_json("alpha", "0.1.0", alpha, "null"),
            beta_pkg = pkg_json("beta", "0.1.0", beta, "null"),
            old_pkg = pkg_json("ktstr", "0.18.0", old, crates_io),
            new_pkg = pkg_json("ktstr", "0.43.0", new, crates_io),
        );
        let meta: cargo_metadata::Metadata =
            serde_json::from_str(&json).expect("fixture deserializes");

        assert_eq!(
            selected_resolved_ktstr_versions(&meta, &[]),
            vec![&v("0.18.0")],
            "unscoped commands guard only Cargo's default members",
        );
        assert_eq!(
            selected_resolved_ktstr_versions(&meta, &strs(&["-p", "b*"])),
            vec![&v("0.43.0")],
            "a globbed package selection must not guard an unrelated member",
        );
        assert_eq!(
            selected_resolved_ktstr_versions(&meta, &strs(&["--workspace"])),
            vec![&v("0.18.0"), &v("0.43.0")],
            "workspace runs must guard every distinct linked ktstr version",
        );
    }

    /// The orchestrator stamps `KTSTR_RUNS_ROOT` = `{target}/ktstr`
    /// (absolute) so child test processes' sidecar writes AND the
    /// post-run footer reader resolve the SAME dir — the workspace
    /// empty-footer fix. With an absolute `CARGO_TARGET_DIR` and no
    /// pre-set override, `install_runs_root_env` must export that path.
    #[test]
    fn install_runs_root_env_stamps_absolute_target_subdir() {
        const CASE: &str = "install-runs-root-absolute";
        if !is_reexec_case(CASE) {
            let tmp = tempfile::TempDir::new().unwrap();
            reexec_current_test(
                CASE,
                [
                    ChildEnv::remove(ktstr::KTSTR_RUNS_ROOT_ENV),
                    ChildEnv::set("CARGO_TARGET_DIR", tmp.path()),
                ],
            );
            return;
        }

        let target_dir = std::path::PathBuf::from(std::env::var_os("CARGO_TARGET_DIR").unwrap());
        install_runs_root_env();
        assert_eq!(
            std::env::var_os(ktstr::KTSTR_RUNS_ROOT_ENV).map(std::path::PathBuf::from),
            Some(target_dir.join("ktstr")),
            "orchestrator must export the absolute {{target}}/ktstr so the \
             child writers and the footer reader agree on one dir",
        );
    }

    /// A pre-set `KTSTR_RUNS_ROOT` (operator override, or a test that
    /// pinned its own root) must win — `install_runs_root_env` no-ops
    /// rather than clobbering it with the cargo target dir.
    #[test]
    fn install_runs_root_env_is_idempotent_when_already_set() {
        const CASE: &str = "install-runs-root-idempotent";
        if !is_reexec_case(CASE) {
            let tmp = tempfile::TempDir::new().unwrap();
            let preset = tmp.path().join("operator-chosen-root");
            reexec_current_test(
                CASE,
                [
                    ChildEnv::set(ktstr::KTSTR_RUNS_ROOT_ENV, &preset),
                    ChildEnv::set("CARGO_TARGET_DIR", tmp.path()),
                ],
            );
            return;
        }

        let preset =
            std::path::PathBuf::from(std::env::var_os(ktstr::KTSTR_RUNS_ROOT_ENV).unwrap());
        install_runs_root_env();
        assert_eq!(
            std::env::var_os(ktstr::KTSTR_RUNS_ROOT_ENV).map(std::path::PathBuf::from),
            Some(preset),
            "a pre-set KTSTR_RUNS_ROOT must survive install (no clobber)",
        );
    }

    #[test]
    fn sidecar_run_coordinate_parses_only_the_ci_layout() {
        assert_eq!(
            parse_sidecar_run_coordinate("100-1-x64-base"),
            Some((100, 1))
        );
        assert_eq!(
            parse_sidecar_run_coordinate("999-2-coverage-arm64"),
            Some((999, 2)),
        );
        // A bare `<run_id>-<attempt>` with no lane suffix is not the CI shape.
        assert_eq!(parse_sidecar_run_coordinate("100-1"), None);
        // Operator-chosen / non-numeric leaves never parse into a run order.
        assert_eq!(parse_sidecar_run_coordinate("my-scratch-dir"), None);
        assert_eq!(parse_sidecar_run_coordinate("v6.12-abcdef0"), None);
    }

    /// A persistent runner's sidecar parent accumulates one directory per
    /// (run, attempt, lane). Startup pruning must reclaim only STRICTLY OLDER,
    /// idle runs while leaving this run's concurrent lanes, newer/overlapping
    /// runs, non-CI directories, and not-yet-idle prior runs untouched.
    #[test]
    fn prune_removes_only_strictly_older_idle_ci_sidecar_dirs() {
        let parent = tempfile::TempDir::new().unwrap();
        let make = |leaf: &str| {
            let dir = parent.path().join(leaf);
            std::fs::create_dir_all(&dir).unwrap();
            dir
        };
        let current = "100-1-x64-base";
        make(current);
        let older = make("50-1-x64-base"); // strictly older run
        let older_attempt = make("100-0-x64-base"); // older attempt of this run
        let sibling_lane = make("100-1-arm64-cov"); // same (run, attempt): concurrent lane
        let newer_attempt = make("100-2-x64-base"); // retry of this run (newer)
        let newer_run = make("200-1-x64-base"); // a later/overlapping run
        let foreign = make("checkout-scratch"); // non-CI directory

        // Not yet idle: every directory was just created, so nothing is
        // reclaimed even though older runs exist.
        prune_prior_ci_sidecar_dirs_in(parent.path(), current, std::time::SystemTime::now());
        for dir in [
            &older,
            &older_attempt,
            &sibling_lane,
            &newer_attempt,
            &newer_run,
            &foreign,
        ] {
            assert!(dir.is_dir(), "nothing is removed before the idle grace");
        }

        // Well past the idle grace: only the strictly-older, idle runs go.
        let future = std::time::SystemTime::now() + PRIOR_RUN_SIDECAR_GRACE * 2;
        prune_prior_ci_sidecar_dirs_in(parent.path(), current, future);
        assert!(!older.is_dir(), "a strictly older idle run is reclaimed");
        assert!(
            !older_attempt.is_dir(),
            "an older attempt of this run is reclaimed",
        );
        assert!(
            sibling_lane.is_dir(),
            "a concurrent same-run lane must never be removed",
        );
        assert!(
            newer_attempt.is_dir(),
            "a newer attempt of this run must never be removed",
        );
        assert!(
            newer_run.is_dir(),
            "a newer / overlapping run must never be removed",
        );
        assert!(
            foreign.is_dir(),
            "a non-CI-layout directory must never be removed",
        );
        assert!(
            parent.path().join(current).is_dir(),
            "this run's own directory must survive",
        );
    }

    /// A RELATIVE `CARGO_TARGET_DIR` is anchored to the orchestrator's
    /// cwd and exported ABSOLUTE — so child test processes (which run
    /// with a different cwd under nextest in a workspace) resolve the
    /// SAME runs root. A relative export would reintroduce the
    /// empty-footer split.
    #[test]
    fn install_runs_root_env_absolutizes_relative_target() {
        const CASE: &str = "install-runs-root-relative";
        if !is_reexec_case(CASE) {
            reexec_current_test(
                CASE,
                [
                    ChildEnv::remove(ktstr::KTSTR_RUNS_ROOT_ENV),
                    ChildEnv::set("CARGO_TARGET_DIR", "relative/target"),
                ],
            );
            return;
        }

        install_runs_root_env();
        let got = std::env::var_os(ktstr::KTSTR_RUNS_ROOT_ENV).map(std::path::PathBuf::from);
        let expected = std::env::current_dir()
            .unwrap()
            .join("relative/target/ktstr");
        assert_eq!(
            got,
            Some(expected),
            "a relative CARGO_TARGET_DIR must be anchored to the orchestrator cwd \
             and exported absolute, not skipped",
        );
    }

    #[test]
    fn item_progress_accounting_and_rendering_are_deterministic() {
        let state = ItemProgressState::new();
        assert!(
            !state.wait_until_finished(std::time::Instant::now()),
            "an unfinished reporter times out instead of inventing completion",
        );
        state.mark_finished();
        assert!(
            state.wait_until_finished(
                std::time::Instant::now() + std::time::Duration::from_secs(60),
            ),
            "completion published before a wait remains observable without sleeping",
        );
        assert_eq!(
            state.snapshot(),
            ItemProgressSnapshot {
                completed: 0,
                failed: 0,
            },
        );
        assert_eq!(
            state.record(true),
            ItemProgressSnapshot {
                completed: 1,
                failed: 0,
            },
        );
        let snapshot = state.record(false);
        assert_eq!(
            snapshot,
            ItemProgressSnapshot {
                completed: 2,
                failed: 1,
            },
        );
        assert_eq!(
            item_progress_line(
                "cargo ktstr probe",
                3,
                snapshot,
                std::time::Duration::from_secs(70),
                "failed",
                Some("bad\nchild"),
            ),
            "cargo ktstr probe: failed; completed=2/3; failed=1; elapsed=1m 10s; \
             error=bad child",
            "the renderer uses injected elapsed time and escapes terminal detail",
        );
    }

    #[test]
    fn cast_precompute_zero_and_one_stay_on_caller_without_pool() {
        let empty: Vec<std::path::PathBuf> = Vec::new();
        precompute_cast_paths_with_pool_builder(
            &empty,
            192,
            |_| panic!("empty scheduler set must not build a pool"),
            |_| panic!("empty scheduler set must not invoke precompute"),
        )
        .unwrap();

        let caller = std::thread::current().id();
        let one = vec![std::path::PathBuf::from("/fake/scx_one")];
        let visited = std::sync::Mutex::new(Vec::new());
        precompute_cast_paths_with_pool_builder(
            &one,
            192,
            |_| panic!("single scheduler must not build a pool"),
            |path| {
                assert_eq!(
                    std::thread::current().id(),
                    caller,
                    "single cast precompute escaped onto a worker"
                );
                visited.lock().unwrap().push(path.to_path_buf());
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(*visited.lock().unwrap(), one);
    }

    #[test]
    fn cast_precompute_pool_is_work_sized_and_joined_before_return() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let binaries: Vec<_> = (0..3)
            .map(|index| std::path::PathBuf::from(format!("/fake/scx_{index}")))
            .collect();
        let requested_threads = std::cell::Cell::new(0);
        let completed = AtomicUsize::new(0);
        precompute_cast_paths_with_pool_builder(
            &binaries,
            192,
            |threads| {
                requested_threads.set(threads);
                rayon::ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .build()
                    .map_err(|error| error.to_string())
            },
            |_| {
                std::thread::sleep(std::time::Duration::from_millis(10));
                completed.fetch_add(1, Ordering::Relaxed);
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(requested_threads.get(), binaries.len());
        assert_eq!(
            completed.load(Ordering::Relaxed),
            binaries.len(),
            "all private-pool work must be joined before precompute returns"
        );
    }

    #[test]
    fn cast_precompute_pool_failure_is_caller_sequential() {
        let binaries: Vec<_> = (0..3)
            .map(|index| std::path::PathBuf::from(format!("/fake/scx_{index}")))
            .collect();
        let caller = std::thread::current().id();
        let requested_threads = std::cell::Cell::new(0);
        let visited = std::sync::Mutex::new(Vec::new());
        precompute_cast_paths_with_pool_builder(
            &binaries,
            2,
            |threads| {
                requested_threads.set(threads);
                Err("forced private-pool failure".to_string())
            },
            |path| {
                assert_eq!(
                    std::thread::current().id(),
                    caller,
                    "private-pool failure initialized parallel fallback work"
                );
                visited.lock().unwrap().push(path.to_path_buf());
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(requested_threads.get(), 2);
        assert_eq!(*visited.lock().unwrap(), binaries);
    }

    #[test]
    fn cast_precompute_attempts_all_work_and_reports_first_input_error() {
        let binaries: Vec<_> = (0..4)
            .map(|index| std::path::PathBuf::from(format!("/fake/scx_{index}")))
            .collect();
        let visited = std::sync::Mutex::new(Vec::new());
        let error = precompute_cast_paths_with_pool_builder(
            &binaries,
            4,
            |threads| {
                rayon::ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .build()
                    .map_err(|error| error.to_string())
            },
            |path| {
                visited.lock().unwrap().push(path.to_path_buf());
                match path.file_name().and_then(std::ffi::OsStr::to_str) {
                    Some("scx_1") => Err("first input error".to_string()),
                    Some("scx_3") => Err("later input error".to_string()),
                    _ => Ok(()),
                }
            },
        )
        .unwrap_err();
        assert_eq!(error, "first input error");
        let mut visited = visited.into_inner().unwrap();
        visited.sort();
        assert_eq!(visited, binaries, "every cast precompute must be attempted");
    }

    /// Byte-exact pin on the three `*_SUB_ARGV` constants that drive
    /// `run_test`, `run_coverage`, and `run_llvm_cov` into
    /// `run_cargo_sub`. A regression that re-ordered the Coverage
    /// tokens (e.g. swapped `["llvm-cov","nextest"]` → `["nextest",
    /// "llvm-cov"]`) would exec `cargo nextest llvm-cov` which is
    /// not a valid cargo subcommand, silently failing coverage
    /// runs. A regression that added a second token to
    /// `LLVM_COV_SUB_ARGV` (e.g. `["llvm-cov","test"]`) would
    /// prepend an implicit subcommand and override the user's
    /// trailing args. Both are caught here.
    #[test]
    fn cargo_sub_argv_constants_are_pinned() {
        assert_eq!(TEST_SUB_ARGV, &["nextest", "run"]);
        assert_eq!(COVERAGE_SUB_ARGV, &["llvm-cov", "nextest"]);
        assert_eq!(LLVM_COV_SUB_ARGV, &["llvm-cov"]);
    }

    // -- profraw_inject_for --
    //
    // The injection must fire for `test` (so an instrumented test
    // binary cannot drop `default.profraw` in cwd), and must NOT
    // fire for `coverage` (cargo-llvm-cov manages
    // `LLVM_PROFILE_FILE` itself) or `llvm-cov` (raw passthrough,
    // user-controlled). An operator-supplied `LLVM_PROFILE_FILE`
    // must always win.

    /// `test` path with no operator override: returns a workspace-
    /// relative pattern ending in the `default-%p-%m.profraw`
    /// expansion tokens.
    #[test]
    fn profraw_inject_for_test_path_returns_pattern() {
        let pat = profraw_inject_for(TEST_SUB_ARGV, None)
            .expect("test path without LLVM_PROFILE_FILE must inject");
        assert!(
            pat.ends_with("default-%p-%m.profraw"),
            "injected pattern must end with default-%%p-%%m.profraw, got {}",
            pat.display(),
        );
        assert_ne!(
            pat.as_os_str(),
            "default-%p-%m.profraw",
            "pattern must be absolute (carry a target dir prefix), \
             not bare so the LLVM runtime never falls back to cwd",
        );
    }

    /// `coverage` path: cargo-llvm-cov manages the env itself.
    #[test]
    fn profraw_inject_for_coverage_path_skips() {
        assert!(
            profraw_inject_for(COVERAGE_SUB_ARGV, None).is_none(),
            "coverage path must not inject — cargo-llvm-cov owns LLVM_PROFILE_FILE",
        );
    }

    /// `llvm-cov` raw passthrough: user-controlled by contract.
    #[test]
    fn profraw_inject_for_llvm_cov_path_skips() {
        assert!(
            profraw_inject_for(LLVM_COV_SUB_ARGV, None).is_none(),
            "llvm-cov passthrough path must not inject — user owns env decisions",
        );
    }

    /// Operator already exported `LLVM_PROFILE_FILE` — explicit
    /// override stays authoritative even on the `test` path.
    #[test]
    fn profraw_inject_for_respects_operator_override() {
        let existing = std::ffi::OsString::from("/tmp/operator-pinned-%p.profraw");
        assert!(
            profraw_inject_for(TEST_SUB_ARGV, Some(existing)).is_none(),
            "an operator-set LLVM_PROFILE_FILE must not be overridden",
        );
    }

    #[test]
    fn ordinary_profraw_environment_participates_in_producer_identity() {
        let identity = producer_environment_identity(&[(
            OsString::from("LLVM_PROFILE_FILE"),
            OsString::from("/tmp/operator-%p.profraw"),
        )])
        .unwrap();
        assert_eq!(
            identity,
            vec!["producer-env:LLVM_PROFILE_FILE=/tmp/operator-%p.profraw".to_string()],
        );
    }

    #[test]
    fn stable_nextest_producer_disables_incremental_before_command_and_identity() {
        let environment = stable_cargo_producer_environment(&[
            (OsString::from("CARGO_INCREMENTAL"), OsString::from("1")),
            (
                OsString::from(ktstr::KTSTR_KERNEL_ENV),
                OsString::from("/runtime/guest-kernel"),
            ),
            (
                OsString::from(ktstr::KTSTR_BUSYBOX_PATH_ENV),
                OsString::from("/runtime/busybox"),
            ),
            (
                OsString::from("SCHEDULER_FIXTURE_MODE"),
                OsString::from("semantic"),
            ),
        ]);
        let command = nextest_binary_list_command(
            Path::new("/stable/workspace"),
            Path::new("/stable/output"),
            &[],
            false,
            &environment,
        );
        let command_environment = cmd_env_map(&command);

        assert_eq!(
            command_environment.get(OsStr::new("CARGO_INCREMENTAL")),
            Some(&Some(OsString::from("0"))),
            "the stable nextest producer must override an inherited incremental setting",
        );
        let identity = producer_environment_identity(&environment).unwrap();
        assert!(
            identity
                .iter()
                .any(|entry| entry == "producer-env:CARGO_INCREMENTAL=0"),
            "producer identity must describe the same non-incremental environment it executes",
        );
        assert!(
            !identity
                .iter()
                .any(|entry| entry == "producer-env:CARGO_INCREMENTAL=1"),
            "the inherited incremental setting must not survive normalization",
        );
        for runtime in [ktstr::KTSTR_KERNEL_ENV, ktstr::KTSTR_BUSYBOX_PATH_ENV] {
            assert!(
                environment
                    .iter()
                    .all(|(name, _)| name != OsStr::new(runtime)),
                "{runtime} must remain runtime-only for the cached producer",
            );
            assert!(
                !matches!(command_environment.get(OsStr::new(runtime)), Some(Some(_))),
                "{runtime} must be absent from the Cargo child",
            );
        }
    }

    fn producer_pair_output(
        primary_raw_status: libc::c_int,
        secondary_raw_status: libc::c_int,
        failed_first: Option<crate::interrupt::CommandPairSide>,
    ) -> crate::interrupt::CommandOutputPair {
        use std::os::unix::process::ExitStatusExt as _;

        crate::interrupt::CommandOutputPair {
            primary: std::process::Output {
                status: std::process::ExitStatus::from_raw(primary_raw_status),
                stdout: b"nextest-json".to_vec(),
                stderr: b"build failed".to_vec(),
            },
            secondary: std::process::Output {
                status: std::process::ExitStatus::from_raw(secondary_raw_status),
                stdout: b"cargo-metadata-json".to_vec(),
                stderr: b"metadata failed".to_vec(),
            },
            failed_first,
        }
    }

    #[test]
    fn paired_nextest_producer_returns_successful_metadata_bytes() {
        let output = producer_pair_output(0, 0, None);
        assert_eq!(
            validate_nextest_producer_pair(&output, "cargo ktstr test").unwrap(),
            b"cargo-metadata-json",
        );
    }

    #[test]
    fn paired_nextest_producer_reports_the_failure_that_cancelled_its_sibling() {
        let metadata_first = producer_pair_output(
            libc::SIGTERM,
            17 << 8,
            Some(crate::interrupt::CommandPairSide::Secondary),
        );
        let metadata_error =
            validate_nextest_producer_pair(&metadata_first, "cargo ktstr test").unwrap_err();
        assert!(metadata_error.contains("Cargo metadata for nextest reuse failed"));
        assert!(metadata_error.contains("metadata failed"));
        assert!(!metadata_error.contains("binary-only build failed"));

        let build_first = producer_pair_output(
            19 << 8,
            libc::SIGTERM,
            Some(crate::interrupt::CommandPairSide::Primary),
        );
        let build_error =
            validate_nextest_producer_pair(&build_first, "cargo ktstr test").unwrap_err();
        assert!(build_error.contains("nextest binary-only build failed (19)"));
        assert!(!build_error.contains("metadata failed"));
    }

    #[test]
    fn harness_producer_ignores_and_removes_nextest_retry_coordinates() {
        let environment = |attempt: &str, slot: &str, test_name: &str, fixture: &str| {
            vec![
                (OsString::from("NEXTEST"), OsString::from("1")),
                (OsString::from("NEXTEST_ATTEMPT"), OsString::from(attempt)),
                (
                    OsString::from("NEXTEST_TEST_GLOBAL_SLOT"),
                    OsString::from(slot),
                ),
                (
                    OsString::from("NEXTEST_TEST_NAME"),
                    OsString::from(test_name),
                ),
                (
                    OsString::from("SCHEDULER_FIXTURE_MODE"),
                    OsString::from(fixture),
                ),
                (
                    OsString::from(ktstr::KTSTR_KERNEL_ENV),
                    OsString::from(format!("/runtime/kernel-{attempt}")),
                ),
                (
                    OsString::from(ktstr::KTSTR_RUN_EPOCH_ENV),
                    OsString::from(format!("epoch-{slot}")),
                ),
            ]
        };
        let first = environment("1", "3", "ktstr::nested_retry", "semantic-a");
        let retry = environment("7", "91", "ktstr::nested_retry/retry", "semantic-a");
        assert_eq!(
            producer_environment_identity(&retry).unwrap(),
            producer_environment_identity(&first).unwrap(),
            "nextest retry coordinates must not split the harness Cargo cache",
        );
        assert_ne!(
            producer_environment_identity(&environment(
                "7",
                "91",
                "ktstr::nested_retry/retry",
                "semantic-b",
            ))
            .unwrap(),
            producer_environment_identity(&first).unwrap(),
            "an arbitrary inherited build-script input must still split the harness cache",
        );

        let command = nextest_binary_list_command(
            Path::new("/stable/workspace"),
            Path::new("/stable/output"),
            &[],
            false,
            &retry,
        );
        let command_environment = cmd_env_map(&command);
        for removed in [
            "NEXTEST",
            "NEXTEST_ATTEMPT",
            "NEXTEST_TEST_GLOBAL_SLOT",
            "NEXTEST_TEST_NAME",
            ktstr::KTSTR_KERNEL_ENV,
            ktstr::KTSTR_RUN_EPOCH_ENV,
        ] {
            assert_eq!(
                command_environment.get(OsStr::new(removed)),
                Some(&None),
                "{removed} must be explicitly absent from the harness Cargo producer",
            );
        }
        assert_eq!(
            command_environment.get(OsStr::new("SCHEDULER_FIXTURE_MODE")),
            Some(&Some(OsString::from("semantic-a"))),
            "arbitrary build-script inputs must remain visible to the harness producer",
        );

        let stable = stable_cargo_producer_environment(&retry);
        assert!(
            stable.iter().all(|(name, _)| {
                !crate::verifier::cached_cargo_build_environment_is_runtime(name)
            }),
            "normalized producer overlays must not retain nextest or ktstr runtime coordinates",
        );
    }

    #[test]
    fn harness_producer_ignores_and_removes_service_and_ci_coordinates() {
        let environment = |runner: &str| {
            let mut environment = vec![(
                OsString::from("SCHEDULER_FIXTURE_MODE"),
                OsString::from("semantic"),
            )];
            environment.extend(
                crate::verifier::CACHED_CARGO_BUILD_SYSTEMD_RUNTIME_ENVIRONMENT
                    .iter()
                    .map(|name| {
                        (
                            OsString::from(*name),
                            OsString::from(format!("/run/{runner}/{name}")),
                        )
                    }),
            );
            environment.extend(
                crate::verifier::CACHED_CARGO_BUILD_CI_RUNTIME_ENVIRONMENT
                    .iter()
                    .map(|name| {
                        (
                            OsString::from(*name),
                            OsString::from(format!("{runner}-{name}")),
                        )
                    }),
            );
            environment
        };
        let first = environment("runner-1");
        let second = environment("runner-9");

        assert_eq!(
            producer_environment_identity(&first).unwrap(),
            producer_environment_identity(&second).unwrap(),
            "service and CI control-plane coordinates must not split the harness cache",
        );

        let stable = stable_cargo_producer_environment(&second);
        assert_eq!(
            stable,
            vec![
                (
                    OsString::from("SCHEDULER_FIXTURE_MODE"),
                    OsString::from("semantic"),
                ),
                (OsString::from("CARGO_INCREMENTAL"), OsString::from("0")),
            ],
            "only semantic inputs and the forced incremental setting reach the normalized producer",
        );

        let command = nextest_binary_list_command(
            Path::new("/stable/workspace"),
            Path::new("/stable/output"),
            &[],
            false,
            &second,
        );
        let command_environment = cmd_env_map(&command);
        for &removed in crate::verifier::CACHED_CARGO_BUILD_SYSTEMD_RUNTIME_ENVIRONMENT
            .iter()
            .chain(crate::verifier::CACHED_CARGO_BUILD_CI_RUNTIME_ENVIRONMENT)
        {
            assert_eq!(
                command_environment.get(OsStr::new(removed)),
                Some(&None),
                "{removed} must be explicitly absent from the harness Cargo producer",
            );
        }
        assert_eq!(
            command_environment.get(OsStr::new("SCHEDULER_FIXTURE_MODE")),
            Some(&Some(OsString::from("semantic"))),
            "arbitrary build-script inputs must remain visible to the harness producer",
        );
    }

    // -- prebuilt_blob_bin_envs --
    //
    // cargo-ktstr re-exports its extracted busybox / wprof paths to the
    // child build as KTSTR_*_BIN so build.rs copies them instead of
    // re-fetching. Busybox is emitted whenever its source path is present;
    // wprof additionally requires the exact resolved ktstr feature.

    fn blob_feature_metadata(active_ktstr_features: &[&str]) -> cargo_metadata::Metadata {
        let ktstr = "ktstr 0.42.0 (path+file:///w/ktstr)";
        let features = serde_json::to_string(active_ktstr_features).expect("serialize features");
        let json = format!(
            r#"{{
              "packages":[{package}],
              "workspace_members":["{ktstr}"],
              "resolve":{{
                "root":"{ktstr}",
                "nodes":[{{"id":"{ktstr}","deps":[],"dependencies":[],"features":{features}}}]
              }},
              "workspace_root":"/w/ktstr",
              "target_directory":"/w/ktstr/target",
              "version":1
            }}"#,
            package = pkg_json("ktstr", "0.42.0", ktstr, "null"),
        );
        serde_json::from_str(&json).expect("blob feature metadata deserializes")
    }

    /// Both paths present → both `KTSTR_*_BIN` pairs, busybox first,
    /// carrying the exact path values.
    #[test]
    fn prebuilt_blob_bin_envs_sets_present_paths() {
        let metadata = blob_feature_metadata(&["default", "wprof"]);
        let pairs = prebuilt_blob_bin_envs(
            Some(std::ffi::OsString::from("/run/bb")),
            Some(std::ffi::OsString::from("/run/wp")),
            Some(&metadata),
        );
        assert_eq!(
            pairs,
            vec![
                ("KTSTR_BUSYBOX_BIN", std::ffi::OsString::from("/run/bb")),
                ("KTSTR_WPROF_BIN", std::ffi::OsString::from("/run/wp")),
            ],
        );
    }

    /// An absent path yields no pair for that blob — so a cargo-ktstr
    /// built without a blob never tells the child build to copy a
    /// nonexistent binary; it falls back to its own fetch path.
    #[test]
    fn prebuilt_blob_bin_envs_omits_absent_paths() {
        assert!(
            prebuilt_blob_bin_envs(None, None, None).is_empty(),
            "no source paths → no env pairs",
        );
        assert_eq!(
            prebuilt_blob_bin_envs(Some(std::ffi::OsString::from("/run/bb")), None, None,),
            vec![("KTSTR_BUSYBOX_BIN", std::ffi::OsString::from("/run/bb"))],
            "busybox present, wprof absent → only the busybox pair",
        );
    }

    /// Available resolved metadata is authoritative: a ktstr node without the
    /// `wprof` feature omits only that optional handoff while busybox remains
    /// byte-for-byte unchanged.
    #[test]
    fn prebuilt_blob_bin_envs_omits_wprof_when_resolved_feature_is_disabled() {
        let metadata = blob_feature_metadata(&["default"]);
        assert_eq!(
            prebuilt_blob_bin_envs(
                Some(std::ffi::OsString::from("/run/bb")),
                Some(std::ffi::OsString::from("/run/wp")),
                Some(&metadata),
            ),
            vec![("KTSTR_BUSYBOX_BIN", std::ffi::OsString::from("/run/bb"))],
        );
    }

    /// Metadata failure must never make a wprof-enabled child rebuild fetch
    /// the tool: retaining the historical handoff is the conservative path.
    #[test]
    fn prebuilt_blob_bin_envs_retains_wprof_when_metadata_is_unavailable() {
        assert_eq!(
            prebuilt_blob_bin_envs(
                Some(std::ffi::OsString::from("/run/bb")),
                Some(std::ffi::OsString::from("/run/wp")),
                None,
            ),
            vec![
                ("KTSTR_BUSYBOX_BIN", std::ffi::OsString::from("/run/bb")),
                ("KTSTR_WPROF_BIN", std::ffi::OsString::from("/run/wp")),
            ],
        );
    }

    /// A metadata value without Cargo's resolve graph is just as
    /// inconclusive as a failed query and therefore keeps wprof.
    #[test]
    fn prebuilt_blob_bin_envs_retains_wprof_when_resolve_graph_is_unavailable() {
        let mut metadata = blob_feature_metadata(&[]);
        metadata.resolve = None;
        assert_eq!(
            prebuilt_blob_bin_envs(
                None,
                Some(std::ffi::OsString::from("/run/wp")),
                Some(&metadata),
            ),
            vec![("KTSTR_WPROF_BIN", std::ffi::OsString::from("/run/wp"))],
        );
    }

    // -- build_cargo_command --
    //
    // The pure `Command` factory split out of `run_cargo_sub` so the
    // argv ordering and flag->env wiring are inspectable via
    // `Command::get_args` / `Command::get_envs` without execing cargo or
    // mutating process env. `get_envs` reflects only the explicit
    // `.env()` mutations on the Command (not the inherited process env),
    // so these assertions are deterministic under parallel nextest.

    /// Collect a `Command`'s explicitly-set env mutations into a map for
    /// exact presence/value/absence assertions.
    fn cmd_env_map(
        cmd: &Command,
    ) -> std::collections::BTreeMap<std::ffi::OsString, Option<std::ffi::OsString>> {
        cmd.get_envs()
            .map(|(k, v)| (k.to_os_string(), v.map(|v| v.to_os_string())))
            .collect()
    }

    /// `release=true` prepends `--cargo-profile release` BEFORE the
    /// user's trailing args so the profile applies to the whole
    /// invocation. Byte-exact full argv (order included) — a regression
    /// that appended the profile after the user args, or dropped the
    /// prepend, flips this vector.
    #[test]
    fn build_cargo_command_release_prepends_profile_before_user_args() {
        let cmd = build_cargo_command(
            TEST_SUB_ARGV,
            true,
            None,
            None,
            false,
            false,
            &["-E".to_string(), "test(foo)".to_string()],
        );
        let argv: Vec<&std::ffi::OsStr> = cmd.get_args().collect();
        assert_eq!(
            argv,
            [
                "nextest",
                "run",
                "--cargo-profile",
                "release",
                "-E",
                "test(foo)"
            ]
            .map(std::ffi::OsStr::new),
        );
    }

    /// `release=false` injects no `--cargo-profile` token, so the user's
    /// args follow `sub_argv` directly. This is the `run_llvm_cov`
    /// raw-passthrough contract (`release` hardcoded false): a regression
    /// that always-prepended the profile would add two tokens and corrupt
    /// the passthrough argv the user fully controls.
    #[test]
    fn build_cargo_command_no_release_omits_profile_flag() {
        let cmd = build_cargo_command(
            LLVM_COV_SUB_ARGV,
            false,
            None,
            None,
            false,
            false,
            &["report".to_string()],
        );
        let argv: Vec<&std::ffi::OsStr> = cmd.get_args().collect();
        assert_eq!(argv, ["llvm-cov", "report"].map(std::ffi::OsStr::new));
    }

    #[test]
    fn llvm_cov_final_no_clean_is_owned_only_by_a_build_warm() {
        let default = build_cargo_command(
            COVERAGE_SUB_ARGV,
            false,
            None,
            None,
            false,
            false,
            &strs(&["--features", "integration"]),
        );
        assert_eq!(
            default
                .get_args()
                .map(|argument| argument.to_string_lossy().into_owned())
                .collect::<Vec<_>>(),
            strs(&[
                "llvm-cov",
                "--no-clean",
                "nextest",
                "--features",
                "integration",
            ]),
        );

        let archive_reuse = build_cargo_command(
            COVERAGE_SUB_ARGV,
            false,
            None,
            None,
            false,
            false,
            &strs(&["--archive-file", "reuse.tar.zst"]),
        );
        assert_eq!(
            archive_reuse
                .get_args()
                .map(|argument| argument.to_string_lossy().into_owned())
                .collect::<Vec<_>>(),
            strs(&["llvm-cov", "nextest", "--archive-file", "reuse.tar.zst",]),
            "archive reuse did not run ktstr's clean/warm lifecycle",
        );

        for retained in ["--no-clean", "--no-report", "--no-run"] {
            let command = build_cargo_command(
                COVERAGE_SUB_ARGV,
                false,
                None,
                None,
                false,
                false,
                &[retained.to_string()],
            );
            let argv = command
                .get_args()
                .map(|argument| argument.to_string_lossy().into_owned())
                .collect::<Vec<_>>();
            assert_eq!(
                argv.iter().filter(|argument| *argument == retained).count(),
                1,
                "explicit lifecycle flag must be preserved without duplication",
            );
            assert_eq!(argv, strs(&["llvm-cov", "nextest", retained]));
        }
    }

    #[test]
    fn cached_coverage_retains_lone_no_report_but_not_mutable_target_modes() {
        assert!(
            !llvm_cov_requires_ordinary_retained_target(&strs(&["--no-report"])),
            "a lone --no-report is retained as one bounded complete COW bundle",
        );
        assert!(llvm_cov_requires_ordinary_retained_target(&strs(&[
            "--no-clean"
        ])));
        assert!(llvm_cov_requires_ordinary_retained_target(&strs(&[
            "--no-run"
        ])));
        assert!(llvm_cov_requires_ordinary_retained_target(&strs(&[
            "--no-report",
            "--no-clean",
        ])));
    }

    #[test]
    fn raw_llvm_cov_final_injects_no_clean_before_nextest_only() {
        let command = build_cargo_command(
            LLVM_COV_SUB_ARGV,
            false,
            None,
            None,
            false,
            false,
            &strs(&[
                "--manifest-path",
                "Cargo.toml",
                "nextest",
                "--features",
                "integration",
            ]),
        );
        assert_eq!(
            command
                .get_args()
                .map(|argument| argument.to_string_lossy().into_owned())
                .collect::<Vec<_>>(),
            strs(&[
                "llvm-cov",
                "--manifest-path",
                "Cargo.toml",
                "--no-clean",
                "nextest",
                "--features",
                "integration",
            ]),
        );

        let report = build_cargo_command(
            LLVM_COV_SUB_ARGV,
            false,
            None,
            None,
            false,
            false,
            &strs(&["report", "--lcov"]),
        );
        assert_eq!(
            report
                .get_args()
                .map(|argument| argument.to_string_lossy().into_owned())
                .collect::<Vec<_>>(),
            strs(&["llvm-cov", "report", "--lcov"]),
            "non-nextest llvm-cov remains an exact passthrough",
        );
    }

    /// Each of `--no-perf-mode` / `--no-skip-mode` independently gates
    /// one env var to the literal "1", absent when its flag is false.
    /// Two invocations — (perf=true,skip=false) and (perf=false,skip=true)
    /// — pin all four (var, gate-outcome) states: each asserts both the
    /// presence+value of its own env var AND the absence of the other, so
    /// a regression swapping the two env names, or dropping a gate, is
    /// caught.
    #[test]
    fn build_cargo_command_perf_and_skip_env_gates() {
        // perf=true, skip=false → only KTSTR_NO_PERF_MODE="1".
        let cmd = build_cargo_command(TEST_SUB_ARGV, false, None, None, true, false, &[]);
        let map = cmd_env_map(&cmd);
        assert_eq!(
            map.get(std::ffi::OsStr::new(ktstr::KTSTR_NO_PERF_MODE_ENV)),
            Some(&Some(std::ffi::OsString::from("1"))),
        );
        assert!(!map.contains_key(std::ffi::OsStr::new(ktstr::KTSTR_NO_SKIP_MODE_ENV)));

        // perf=false, skip=true → only KTSTR_NO_SKIP_MODE="1".
        let cmd = build_cargo_command(TEST_SUB_ARGV, false, None, None, false, true, &[]);
        let map = cmd_env_map(&cmd);
        assert_eq!(
            map.get(std::ffi::OsStr::new(ktstr::KTSTR_NO_SKIP_MODE_ENV)),
            Some(&Some(std::ffi::OsString::from("1"))),
        );
        assert!(!map.contains_key(std::ffi::OsStr::new(ktstr::KTSTR_NO_PERF_MODE_ENV)));
    }

    /// The `profile` Option (the `--profile <NAME>` scheduler BUILD
    /// profile) is wired into the `KTSTR_SCHEDULER_PROFILE` env:
    /// `Some("dev")` sets the var to "dev"; `None` leaves it absent (the
    /// scheduler build then defaults to release inside
    /// `build_and_find_binary`). Pins the WIRING (correct var name +
    /// value, and the absent corner) that a dropped `.env()` call would
    /// otherwise slip through.
    #[test]
    fn build_cargo_command_scheduler_profile_wired_from_flag() {
        // Some("dev") → var set to "dev".
        let cmd = build_cargo_command(TEST_SUB_ARGV, false, Some("dev"), None, false, false, &[]);
        let map = cmd_env_map(&cmd);
        assert_eq!(
            map.get(std::ffi::OsStr::new(ktstr::KTSTR_SCHEDULER_PROFILE_ENV)),
            Some(&Some(std::ffi::OsString::from("dev"))),
        );

        // None → var absent (scheduler build defaults to release).
        let cmd = build_cargo_command(TEST_SUB_ARGV, false, None, None, false, false, &[]);
        let map = cmd_env_map(&cmd);
        assert!(!map.contains_key(std::ffi::OsStr::new(ktstr::KTSTR_SCHEDULER_PROFILE_ENV)));
    }

    /// The `nextest_profile` Option (the `--nextest-profile <NAME>`
    /// NEXTEST test profile) is emitted as a `--profile <NAME>` argv token
    /// AFTER `sub_argv` and BEFORE the user's trailing args — nextest and
    /// `cargo llvm-cov nextest` read `--profile` to select the test
    /// profile. `None` injects nothing. Byte-exact argv pins the token
    /// name + placement; a regression emitting the wrong flag, or
    /// appending after the user args, flips the vector.
    #[test]
    fn build_cargo_command_nextest_profile_injects_profile_flag() {
        // Some("ci") → `--profile ci` after sub_argv, before user args.
        let cmd = build_cargo_command(
            TEST_SUB_ARGV,
            false,
            None,
            Some("ci"),
            false,
            false,
            &["-E".to_string(), "test(foo)".to_string()],
        );
        let argv: Vec<&std::ffi::OsStr> = cmd.get_args().collect();
        assert_eq!(
            argv,
            ["nextest", "run", "--profile", "ci", "-E", "test(foo)"].map(std::ffi::OsStr::new),
        );

        // None → no `--profile` token.
        let cmd = build_cargo_command(TEST_SUB_ARGV, false, None, None, false, false, &[]);
        let argv: Vec<&std::ffi::OsStr> = cmd.get_args().collect();
        assert_eq!(argv, ["nextest", "run"].map(std::ffi::OsStr::new));
    }

    // -- kernel_set_or_bail --
    //
    // resolve_kernel_set drops whitespace-only specs before any
    // KernelId::parse, so all-whitespace `--kernel` input does no
    // network/build I/O and the bail is host-isolable.

    /// A non-empty `--kernel` whose every value trims to empty resolves
    /// to nothing → actionable bail instead of silently falling through
    /// to auto-discovery (which would mask the operator's intent).
    #[test]
    fn kernel_set_or_bail_all_whitespace_bails() {
        let err = kernel_set_or_bail(&["".to_string(), "  \t ".to_string()], false)
            .expect_err("all-whitespace --kernel must bail, not auto-discover");
        assert!(
            err.starts_with("--kernel: every supplied value parsed to empty"),
            "unexpected bail message: {err}",
        );
    }

    /// An omitted `--kernel` flag (empty input vec) is the auto-discovery
    /// path, NOT an error: returns `Ok(empty)` so the caller falls
    /// through to the `find_kernel` chain without exporting KTSTR_KERNEL.
    #[test]
    fn kernel_set_or_bail_empty_input_is_ok_empty() {
        assert_eq!(kernel_set_or_bail(&[], false), Ok(Vec::new()));
    }

    // -- generate_btf_anchor (no-bpf early return) --
    //
    // Only the "no .bpf.o objects found -> None" path is host-isolable:
    // it returns at the `bpf_object_dirs.is_empty()` gate BEFORE any
    // env read (BPF_BASE_CFLAGS / BPF_CLANG / ...) and BEFORE the
    // delegation to `btf_catalog::generate_btf_anchor`, which execs
    // clang. Driving it past the gate would require a real `bpf.bpf.o`
    // and would spawn clang, so the populated path is not host-testable
    // here. These tests exercise the dir scan over a tempdir target so
    // they touch no env and no subprocess.

    /// A target dir with no `<profile>/build` directory at all: the
    /// `read_dir(&build_root)` errors, the `if let Ok` is skipped,
    /// `bpf_object_dirs` stays empty, and the fn returns `None` — for
    /// BOTH profiles. Pins the `release -> "release"` / `!release ->
    /// "debug"` selector reaching a missing build root in each case.
    #[test]
    fn generate_btf_anchor_missing_build_root_is_none() {
        let dir = tempfile::tempdir().expect("tempdir");
        assert_eq!(generate_btf_anchor(dir.path(), false), None);
        assert_eq!(generate_btf_anchor(dir.path(), true), None);
    }

    /// An existing but empty `debug/build` directory: `read_dir` now
    /// succeeds (Ok branch taken), the entry loop runs zero iterations,
    /// `bpf_object_dirs` is still empty, so the fn returns `None`.
    /// Distinct from the missing-root case — exercises the loop body's
    /// guard rather than the `read_dir` Err short-circuit.
    #[test]
    fn generate_btf_anchor_empty_build_root_is_none() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::create_dir_all(dir.path().join("debug").join("build"))
            .expect("create debug/build");
        assert_eq!(generate_btf_anchor(dir.path(), false), None);
    }

    /// A build-output entry whose `out/` directory exists but lacks the
    /// `bpf.bpf.o` gate file is NOT collected: `out.join("bpf.bpf.o")
    /// .is_file()` is false, the entry is skipped, `bpf_object_dirs`
    /// ends empty, and the fn returns `None`. Pins the gate-file name
    /// (`bpf.bpf.o`) — a sibling `.o` or a directory by that name must
    /// not satisfy the `is_file()` check.
    #[test]
    fn generate_btf_anchor_build_entry_without_bpf_object_is_none() {
        let dir = tempfile::tempdir().expect("tempdir");
        let out = dir
            .path()
            .join("debug")
            .join("build")
            .join("scx_utils-abc123")
            .join("out");
        std::fs::create_dir_all(&out).expect("create out");
        // A non-gate object and a non-empty file with a near-miss name —
        // neither is `bpf.bpf.o`, so the entry must be skipped.
        std::fs::write(out.join("other.bpf.o"), b"x").expect("write other.bpf.o");
        std::fs::write(out.join("bpf.o"), b"x").expect("write bpf.o");
        assert_eq!(generate_btf_anchor(dir.path(), false), None);
    }

    // ---------------------------------------------------------------
    // Reserved harness-compile warm-up wiring.
    // ---------------------------------------------------------------

    /// Every nextest-backed route warms its own exact command; non-test
    /// llvm-cov passthroughs remain untouched.
    #[test]
    fn reserved_prebuild_gates_every_nextest_route() {
        assert!(
            cargo_sub_uses_nextest(TEST_SUB_ARGV, &[]),
            "the `test` path must warm up under the reservation",
        );
        assert!(
            cargo_sub_uses_nextest(COVERAGE_SUB_ARGV, &[]),
            "coverage must warm its llvm-cov-instrumented command",
        );
        assert!(
            cargo_sub_uses_nextest(LLVM_COV_SUB_ARGV, &strs(&["nextest"])),
            "raw llvm-cov nextest must use the same warm-up path",
        );
        assert!(
            !cargo_sub_uses_nextest(LLVM_COV_SUB_ARGV, &strs(&["report"])),
            "raw llvm-cov report must remain a pure passthrough",
        );
        assert!(!cargo_sub_needs_reserved_prebuild(
            COVERAGE_SUB_ARGV,
            &strs(&["--no-run"]),
        ));
        assert!(!cargo_sub_needs_reserved_prebuild(
            COVERAGE_SUB_ARGV,
            &strs(&["--archive-file", "reuse.tar.zst"]),
        ));
        assert!(cargo_sub_needs_reserved_prebuild(
            COVERAGE_SUB_ARGV,
            &strs(&["--no-report"]),
        ));
    }

    #[test]
    fn archive_reuse_detection_is_nextest_region_and_value_aware() {
        assert!(llvm_cov_reuses_archive(
            LLVM_COV_SUB_ARGV,
            &strs(&[
                "--output-path",
                "report.info",
                "nextest",
                "--archive-file",
                "reuse.tar.zst",
            ]),
        ));
        assert!(!llvm_cov_reuses_archive(
            LLVM_COV_SUB_ARGV,
            &strs(&[
                "--output-path",
                "--archive-file",
                "nextest",
                "--features",
                "integration",
            ]),
        ));
        assert!(!llvm_cov_reuses_archive(
            COVERAGE_SUB_ARGV,
            &strs(&["--tool-config-file", "--archive-file"]),
        ));
        assert!(!llvm_cov_reuses_archive(
            COVERAGE_SUB_ARGV,
            &strs(&["--nextest-archive-file", "report.tar.zst"]),
        ));
        assert!(!llvm_cov_reuses_archive(
            COVERAGE_SUB_ARGV,
            &strs(&["--", "--archive-file", "opaque"]),
        ));
        for args in [
            strs(&["--archive-file", "reuse.tar.zst", "nextest"]),
            strs(&["--archive-file=reuse.tar.zst", "nextest"]),
        ] {
            assert!(
                llvm_cov_reuses_archive(LLVM_COV_SUB_ARGV, &args),
                "raw llvm-cov accepts archive reuse before its nextest subcommand",
            );
        }
        for args in [
            strs(&["--archive-file", "-", "nextest", "--archive-format=tar-zst"]),
            strs(&["--archive-file=-", "nextest", "--archive-format=tar-zst"]),
        ] {
            assert_eq!(
                nextest_archive_reuse(LLVM_COV_SUB_ARGV, &args)
                    .unwrap()
                    .unwrap()
                    .file
                    .value,
                "-",
                "a dash is nextest's literal filename in either spelling",
            );
        }
        assert!(
            nextest_archive_reuse(LLVM_COV_SUB_ARGV, &strs(&["--archive-file=-", "nextest"]),)
                .unwrap_err()
                .contains("format auto"),
            "the literal dash has no auto-detectable archive extension",
        );
        assert!(
            nextest_archive_reuse(
                LLVM_COV_SUB_ARGV,
                &strs(&[
                    "--archive-file",
                    "first.tar.zst",
                    "nextest",
                    "--archive-file=second.tar.zst",
                ]),
            )
            .unwrap_err()
            .contains("duplicate --archive-file"),
        );
        assert!(
            nextest_archive_reuse(
                COVERAGE_SUB_ARGV,
                &strs(&["--archive-file", "reuse.tar.zst", "--archive-format", "zip",]),
            )
            .unwrap_err()
            .contains("does not resolve to tar-zst"),
        );
    }

    #[test]
    fn archive_reuse_rewrite_preserves_occurrence_and_forces_tar_zst() {
        let rewritten = rewrite_nextest_archive_reuse(
            LLVM_COV_SUB_ARGV,
            strs(&["--archive-file", "reuse.tar.zst", "nextest", "--locked"]),
            std::path::Path::new("/cache/deadbeef.object"),
        )
        .unwrap();
        assert_eq!(
            rewritten,
            strs(&[
                "--archive-file",
                "/cache/deadbeef.object",
                "nextest",
                "--locked",
                "--archive-format=tar-zst",
            ]),
        );
    }

    #[test]
    fn archive_warm_projection_rejects_unknown_post_nextest_options() {
        let error = llvm_cov_nextest_archive_args(
            LLVM_COV_SUB_ARGV,
            &strs(&["nextest", "--future-option"]),
            std::path::Path::new("/tmp/warm.tar.zst"),
        )
        .unwrap_err();
        assert!(error.contains("unknown post-nextest option"));
    }

    #[test]
    fn llvm_cov_archive_warm_projection_is_build_exact_and_run_clean() {
        let command = build_llvm_cov_archive_warm_command(
            LLVM_COV_SUB_ARGV,
            true,
            Some("scheduler-dev"),
            Some("ci"),
            true,
            true,
            &strs(&[
                "--manifest-path",
                "Cargo.toml",
                "-j",
                "12",
                "--lcov",
                "--output-path",
                "report.info",
                "nextest",
                "--features",
                "integration",
                "-j",
                "7",
                "--show-progress",
                "bar",
                "--flaky-result",
                "fail",
                "--no-fail-fast",
                "-R",
                "previous",
                "-E",
                "test(x)",
                "positional",
                "--no-report",
                "--exclude-from-report",
                "vendor",
                "--branch",
                "--",
                "--exact",
                "opaque",
            ]),
            std::path::Path::new("/tmp/warm-build.tar.zst"),
        )
        .expect("known llvm-cov/nextest options project onto archive warm-up");
        assert_eq!(
            command
                .get_args()
                .map(|argument| argument.to_string_lossy().into_owned())
                .collect::<Vec<_>>(),
            strs(&[
                "llvm-cov",
                "--manifest-path",
                "Cargo.toml",
                "--no-clean",
                "nextest-archive",
                "--cargo-profile",
                "release",
                "--profile",
                "ci",
                "--features",
                "integration",
                "--exclude-from-report",
                "vendor",
                "--branch",
                "--build-jobs",
                "12",
                "--archive-file",
                "/tmp/warm-build.tar.zst",
                "--zstd-level=-7",
                "-E",
                "none()",
                "--cargo-message-format=json-render-diagnostics",
            ]),
        );
        let env = cmd_env_map(&command);
        assert_eq!(
            env.get(std::ffi::OsStr::new(ktstr::KTSTR_NO_PERF_MODE_ENV)),
            Some(&Some(std::ffi::OsString::from("1"))),
        );
        assert_eq!(
            env.get(std::ffi::OsStr::new(ktstr::KTSTR_NO_SKIP_MODE_ENV)),
            Some(&Some(std::ffi::OsString::from("1"))),
        );
        assert_eq!(
            env.get(std::ffi::OsStr::new(ktstr::KTSTR_SCHEDULER_PROFILE_ENV)),
            Some(&Some(std::ffi::OsString::from("scheduler-dev"))),
        );
    }

    /// `prebuild_no_run_args` appends exactly `--no-run` and preserves
    /// every user token (filtersets, features, positional filters) in
    /// order — the warm-up builds the whole test set so the filtered
    /// combined run finds all artifacts cached.
    #[test]
    fn prebuild_no_run_args_appends_no_run_preserving_rest() {
        let args = strs(&["-E", "test(eevdf)", "--features", "x", "positional"]);
        let out = prebuild_no_run_args(&args);
        assert_eq!(
            out,
            strs(&[
                "-E",
                "test(eevdf)",
                "--features",
                "x",
                "positional",
                "--no-run"
            ]),
        );
    }

    /// The run-phase flags nextest hard-rejects beside `--no-run` are
    /// stripped from the warm-up argv — `just test` passes
    /// `--no-fail-fast`, which killed the reserved pre-build with
    /// "the argument '--no-fail-fast' cannot be used with '--no-run'".
    /// Both `--max-fail` forms drop their value; build-affecting args
    /// around them survive in order.
    #[test]
    fn prebuild_no_run_args_strips_no_run_incompatible_flags() {
        let args = strs(&[
            "--features",
            "integration",
            "--no-fail-fast",
            "-j",
            "8",
            "--fail-fast",
            "--max-fail",
            "3",
            "--max-fail=2",
            "-E",
            "test(x)",
        ]);
        let out = prebuild_no_run_args(&args);
        assert_eq!(
            out,
            strs(&[
                "--features",
                "integration",
                "-j",
                "8",
                "-E",
                "test(x)",
                "--no-run"
            ]),
            "fail-fast family stripped (values included); -j kept (warns only)",
        );
    }

    #[test]
    fn prebuild_no_run_args_preserves_inner_separator_suffix() {
        let args = strs(&[
            "--no-fail-fast",
            "--features",
            "integration",
            "--",
            "--fail-fast",
            "--max-fail",
            "3",
        ]);
        assert_eq!(
            prebuild_no_run_args(&args),
            strs(&[
                "--features",
                "integration",
                "--no-run",
                "--",
                "--fail-fast",
                "--max-fail",
                "3",
            ]),
        );
    }

    #[test]
    fn prebuild_json_args_force_format_before_inner_separator() {
        let args = strs(&[
            "--cargo-message-format",
            "human",
            "--features",
            "integration",
            "--",
            "--cargo-message-format=json",
        ]);
        assert_eq!(
            prebuild_no_run_json_args(&args),
            strs(&[
                "--features",
                "integration",
                "--no-run",
                "--cargo-message-format=json-render-diagnostics",
                "--",
                "--cargo-message-format=json",
            ]),
        );
    }

    #[test]
    fn cargo_json_test_executable_filter_sorts_and_deduplicates() {
        let stream = br#"
{"reason":"compiler-artifact","executable":"/tmp/z-unit","target":{"kind":["lib"]},"profile":{"test":true}}
{"reason":"compiler-artifact","executable":"/tmp/a-integration","target":{"kind":["test"]},"profile":{"test":false}}
{"reason":"compiler-artifact","executable":"/tmp/z-unit","target":{"kind":["bin"]},"profile":{"test":true}}
{"reason":"compiler-artifact","executable":"/tmp/not-a-test","target":{"kind":["bin"]},"profile":{"test":false}}
{"reason":"build-finished","success":true}
nextest non-json output is ignored
"#;
        assert_eq!(
            test_executables_from_cargo_json(stream),
            vec![
                PathBuf::from("/tmp/a-integration"),
                PathBuf::from("/tmp/z-unit"),
            ],
        );
    }

    #[test]
    fn cargo_json_discovery_accepts_stderr_redirect_and_requires_build_finished() {
        let stderr = br#"
cargo-llvm-cov diagnostic
{"reason":"compiler-artifact","executable":"/tmp/from-stderr","target":{"kind":["test"]},"profile":{"test":false}}
{"reason":"build-finished","success":true}
"#;
        assert_eq!(
            validated_test_executables_from_cargo_output(b"", stderr).unwrap(),
            vec![PathBuf::from("/tmp/from-stderr")],
        );
        assert!(
            validated_test_executables_from_cargo_output(
                br#"{"reason":"compiler-artifact","executable":"/tmp/untrusted","target":{"kind":["test"]},"profile":{"test":false}}"#,
                b"plain diagnostics only",
            )
            .is_err(),
            "successful discovery without Cargo's build-finished record is untrusted",
        );
        assert_eq!(
            validated_test_executables_from_cargo_output(
                br#"{"reason":"build-finished","success":true}"#,
                b"",
            )
            .unwrap(),
            Vec::<PathBuf>::new(),
            "a confirmed build with zero test executables is valid",
        );
    }

    #[test]
    fn reserved_build_diagnostics_preserve_both_captured_streams() {
        use std::os::unix::process::ExitStatusExt as _;

        let directory = tempfile::tempdir().expect("diagnostics tempdir");
        let output = std::process::Output {
            status: std::process::ExitStatus::from_raw(0),
            stdout: b"{\"reason\":\"compiler-artifact\"}\n".to_vec(),
            stderr: b"warning: human diagnostic\n".to_vec(),
        };
        let root = persist_reserved_build_diagnostics_to(
            directory.path(),
            &output,
            "cargo ktstr: verifier",
            "scheduler workspace pre-build",
        )
        .expect("persist diagnostics");
        assert_eq!(root, directory.path());

        let mut entries = std::fs::read_dir(directory.path())
            .expect("read diagnostics")
            .map(|entry| entry.expect("diagnostic entry").path())
            .collect::<Vec<_>>();
        entries.sort();
        assert_eq!(entries.len(), 2);
        let stdout = entries
            .iter()
            .find(|path| path.to_string_lossy().ends_with(".stdout.log"))
            .expect("stdout artifact");
        let stderr = entries
            .iter()
            .find(|path| path.to_string_lossy().ends_with(".stderr.log"))
            .expect("stderr artifact");
        assert_eq!(
            std::fs::read(stdout).expect("read stdout artifact"),
            output.stdout,
        );
        assert_eq!(
            std::fs::read(stderr).expect("read stderr artifact"),
            output.stderr,
        );
        assert!(
            entries.iter().all(|path| {
                path.file_name()
                    .expect("artifact filename")
                    .to_string_lossy()
                    .starts_with("cargo-ktstr-verifier-scheduler-workspace-pre-build-")
            }),
            "artifact names must remain path-safe and identify their phase",
        );
    }

    #[test]
    fn reserved_build_diagnostic_bound_retains_stream_head_and_tail() {
        let directory = tempfile::tempdir().expect("diagnostics tempdir");
        let path = directory.path().join("bounded.stderr.log");
        let bytes = b"0123456789abcdefghijklmnop";
        write_bounded_diagnostic_stream_with_limit(&path, bytes, 12)
            .expect("write bounded diagnostics");
        let written = std::fs::read_to_string(path).expect("read bounded diagnostics");
        assert!(written.starts_with("012345"));
        assert!(written.contains("ktstr omitted 14 diagnostic bytes"));
        assert!(written.ends_with("klmnop"));
    }

    #[test]
    fn diagnostic_filename_component_is_safe_bounded_and_never_empty() {
        assert_eq!(
            diagnostic_filename_component(" Cargo ktstr:\nVerifier / Build! "),
            "cargo-ktstr-verifier-build",
        );
        assert_eq!(diagnostic_filename_component("\n/"), "build");
        assert!(
            diagnostic_filename_component(&"x".repeat(100)).len() <= 64,
            "diagnostic filenames must stay bounded",
        );
    }

    fn append_nextest_archive_file<W: std::io::Write>(
        archive: &mut tar::Builder<W>,
        path: &str,
        bytes: &[u8],
        mode: u32,
    ) {
        let mut header = tar::Header::new_gnu();
        header.set_size(bytes.len() as u64);
        header.set_mode(mode);
        header.set_mtime(0);
        header.set_cksum();
        archive
            .append_data(&mut header, path, bytes)
            .expect("append nextest archive fixture");
    }

    fn write_nextest_archive_fixture(
        path: &std::path::Path,
        metadata: &[u8],
        binaries: &[(&str, &[u8])],
        links: &[(&str, &str, bool)],
    ) {
        let file = std::fs::File::create(path).expect("create nextest archive fixture");
        let encoder = zstd::Encoder::new(file, 0).expect("create zstd encoder");
        let mut archive = tar::Builder::new(encoder);
        append_nextest_archive_file(
            &mut archive,
            NEXTEST_ARCHIVE_BINARIES_METADATA,
            metadata,
            0o644,
        );
        for (binary_path, bytes) in binaries {
            append_nextest_archive_file(&mut archive, binary_path, bytes, 0o755);
        }
        for (path, target, hard_link) in links {
            let mut header = tar::Header::new_gnu();
            header.set_size(0);
            header.set_mode(0o755);
            header.set_mtime(0);
            header.set_entry_type(if *hard_link {
                tar::EntryType::Link
            } else {
                tar::EntryType::Symlink
            });
            header
                .set_link_name(target)
                .expect("set archive fixture link target");
            header.set_cksum();
            archive
                .append_data(&mut header, path, std::io::empty())
                .expect("append nextest archive link fixture");
        }
        let encoder = archive.into_inner().expect("finish tar archive");
        encoder.finish().expect("finish zstd archive");
    }

    #[test]
    fn nextest_archive_probe_selects_exact_test_binaries_from_full_target_closure() {
        let temp = tempfile::tempdir().expect("archive fixture tempdir");
        let archive_path = temp.path().join("reuse.tar.zst");
        let metadata = br#"{
            "rust-build-meta": {"target-directory": "/original/target"},
            "rust-binaries": {
                "pkg::one": {"binary-path": "/original/target/debug/deps/one"},
                "pkg::two": {"binary-path": "/original/target/debug/deps/two"}
            }
        }"#;
        write_nextest_archive_fixture(
            &archive_path,
            metadata,
            &[
                ("target/debug/deps/one", b"first"),
                ("target/debug/deps/two", b"second"),
                ("target/debug/deps/unlisted", b"must not extract"),
            ],
            &[],
        );

        let extracted =
            extract_nextest_archive_test_binaries(&archive_path).expect("extract test binaries");
        assert_eq!(extracted.test_binaries.len(), 2);
        assert_eq!(
            extracted
                .test_binaries
                .iter()
                .map(std::fs::read)
                .collect::<Result<Vec<_>, _>>()
                .unwrap(),
            vec![b"first".to_vec(), b"second".to_vec()],
        );
        assert!(
            extracted
                ._owner
                .path()
                .join("target/debug/deps/unlisted")
                .exists(),
            "the archived target closure is retained so loader links cannot dangle",
        );
    }

    #[test]
    fn nextest_archive_probe_extracts_the_whole_loader_closure() {
        let temp = tempfile::tempdir().expect("archive fixture tempdir");
        let archive_path = temp.path().join("reuse.tar.zst");
        let metadata = br#"{
            "rust-build-meta": {
                "target-directory": "/original/target",
                "base-output-directories": ["debug"],
                "linked-paths": ["vendor/lib"]
            },
            "rust-binaries": {
                "pkg::one": {"binary-path": "/original/target/debug/deps/one"}
            }
        }"#;
        write_nextest_archive_fixture(
            &archive_path,
            metadata,
            &[
                ("target/debug/deps/one", b"test"),
                ("target/vendor/lib/liblinked.so", b"linked"),
                ("target/debug/deps/libdep.so", b"dep"),
                ("target/debug/libbase.so", b"base"),
                ("target/nextest/libdirs/host/libstd.so", b"std"),
                ("target/unrelated/libignored.so", b"ignored"),
            ],
            &[
                (
                    "target/debug/deps/libhard.so",
                    "target/debug/deps/libdep.so",
                    true,
                ),
                ("target/debug/deps/libdep-link.so", "libdep.so", false),
            ],
        );

        let extracted =
            extract_nextest_archive_test_binaries(&archive_path).expect("extract probe closure");
        for retained in [
            "target/vendor/lib/liblinked.so",
            "target/debug/libbase.so",
            "target/nextest/libdirs/host/libstd.so",
        ] {
            assert!(
                extracted._owner.path().join(retained).exists(),
                "{retained} belongs to the archived loader closure",
            );
        }
        assert!(
            extracted
                ._owner
                .path()
                .join("target/unrelated/libignored.so")
                .exists()
        );
        let deps = extracted._owner.path().join("target/debug/deps");
        assert_eq!(std::fs::read(deps.join("libdep-link.so")).unwrap(), b"dep",);
        use std::os::unix::fs::MetadataExt as _;
        assert_eq!(
            std::fs::metadata(deps.join("libdep.so")).unwrap().ino(),
            std::fs::metadata(deps.join("libhard.so")).unwrap().ino(),
            "archive-root hard links must resolve inside the extraction owner",
        );
    }

    #[test]
    fn nextest_archive_loader_links_must_remain_under_target_tree() {
        let entry = std::path::Path::new("target/debug/deps/libfoo.so");
        validate_archive_loader_link(entry, std::path::Path::new("libfoo.so.1"), false).unwrap();
        validate_archive_loader_link(
            entry,
            std::path::Path::new("target/debug/deps/libfoo.so.1"),
            true,
        )
        .unwrap();
        assert!(
            validate_archive_loader_link(
                entry,
                std::path::Path::new("../../../../etc/passwd"),
                false,
            )
            .unwrap_err()
            .contains("escapes target tree"),
        );
        assert!(
            validate_archive_loader_link(entry, std::path::Path::new("/etc/passwd"), false,)
                .is_err(),
        );
    }

    #[test]
    fn nextest_archive_probe_rejects_binary_outside_target_directory() {
        let metadata: NextestArchiveBinaryMetadata = serde_json::from_slice(
            br#"{
            "rust-build-meta": {"target-directory": "/original/target"},
            "rust-binaries": {
                "pkg::escape": {"binary-path": "/original/other/escape"}
            }
        }"#,
        )
        .expect("parse archive metadata fixture");
        let error = nextest_archive_binary_entry_paths(&metadata)
            .expect_err("archive binary outside target directory must fail");
        assert!(error.contains("outside build-directory"));
    }

    /// The warm-up command built for the `test` path carries the SAME
    /// build-affecting argv as the combined run (profile injection,
    /// nextest profile) PLUS a trailing `--no-run`, so the combined run
    /// finds the identical build cached. Inspects the pure
    /// `build_cargo_command` factory the warm-up shares with the run.
    #[test]
    fn warmup_command_mirrors_run_argv_plus_no_run() {
        let user = strs(&["--features", "consumer/ktstr-tests"]);
        let warm = build_cargo_command(
            TEST_SUB_ARGV,
            true, // release → `--cargo-profile release`
            None,
            Some("ci"),
            false,
            false,
            &prebuild_no_run_args(&user),
        );
        let argv: Vec<String> = warm
            .get_args()
            .map(|a| a.to_string_lossy().into_owned())
            .collect();
        assert_eq!(
            argv,
            strs(&[
                "nextest",
                "run",
                "--cargo-profile",
                "release",
                "--profile",
                "ci",
                "--features",
                "consumer/ktstr-tests",
                "--no-run",
            ]),
            "warm-up must equal the run argv (profiles/features) + trailing --no-run",
        );
    }

    /// Integration: the harness-compile reservation the warm-up takes
    /// grabs a machine-global LLC `LOCK_SH` under `KTSTR_LOCK_DIR`
    /// isolation. Mirrors the host_topology planning tests' flock-presence
    /// idiom, but drives the exact
    /// `ktstr::cli::acquire_build_reservation_waiting`
    /// call `run_reserved_prebuild` makes — proving the cross-binary
    /// reservation wiring is live, not just the argv shape.
    ///
    /// Gated on a plan actually being acquired (sysfs-readable host with
    /// a free LLC): a sysfs-less or fully-contended host returns no plan
    /// / an Unavailable error, which the test tolerates (skips) rather
    /// than fails — same defensive shape as the lib-side
    /// `acquire_build_reservation_plan_and_make_jobs_consistent`.
    #[test]
    fn reserved_prebuild_takes_llc_lock_sh_under_lock_dir_isolation() {
        const CASE: &str = "reserved-prebuild-lock-isolation";
        if !is_reexec_case(CASE) {
            let lock_dir = tempfile::TempDir::new().expect("tempdir");
            // Isolate the flock namespace into the tempdir and clear every
            // short-circuit so the reservation actually attempts the flock.
            reexec_current_test(
                CASE,
                [
                    ChildEnv::set(ktstr::KTSTR_LOCK_DIR_ENV, lock_dir.path()),
                    ChildEnv::remove(ktstr::KTSTR_BYPASS_LLC_LOCKS_ENV),
                    ChildEnv::remove(ktstr::KTSTR_CARGO_TEST_MODE_ENV),
                    ChildEnv::remove(ktstr::KTSTR_CPU_CAP_ENV),
                ],
            );
            return;
        }
        let lock_dir =
            std::path::PathBuf::from(std::env::var_os(ktstr::KTSTR_LOCK_DIR_ENV).unwrap());

        match ktstr::cli::acquire_build_reservation_waiting("test", None) {
            Ok(reservation) => {
                if reservation.make_jobs().is_none() {
                    eprintln!(
                        "no LLC plan on this host (sysfs unreadable / bypass); \
                         reservation wiring reached but flock skipped — tolerated"
                    );
                    return;
                }
                // A plan was acquired: its LOCK_SH fds pin one lockfile per
                // reserved LLC into the isolated dir. Assert at least one
                // `ktstr-llc-*.lock` landed there while the reservation is
                // still held.
                let has_llc_lock = std::fs::read_dir(&lock_dir)
                    .expect("read isolated lock dir")
                    .flatten()
                    .any(|e| {
                        e.file_name()
                            .to_str()
                            .is_some_and(|n| n.starts_with("ktstr-llc-") && n.ends_with(".lock"))
                    });
                assert!(
                    has_llc_lock,
                    "an acquired harness-build reservation must leave a \
                     ktstr-llc-*.lock in KTSTR_LOCK_DIR while held",
                );
                // reservation drops here → flocks release, cgroup rmdir.
            }
            Err(e) => {
                // Contended LLCs / sysfs error: the wiring was reached; the
                // flock outcome is host-dependent, so tolerate.
                eprintln!("acquire_build_reservation unavailable on this host: {e:#}");
            }
        }
    }
}
