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
use std::path::PathBuf;
use std::process::Command;

use cargo_metadata::semver::Version;

use crate::feature_discovery::{
    MetadataMode, augment_test_features_from_metadata_for_context, effective_target_context,
    query_metadata_for_target, query_resolved_metadata, selected_workspace_packages,
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
fn cargo_sub_uses_nextest(sub_argv: &[&str], args: &[String]) -> bool {
    let selects_nextest = sub_argv == TEST_SUB_ARGV
        || sub_argv == COVERAGE_SUB_ARGV
        || (sub_argv == LLVM_COV_SUB_ARGV && llvm_cov_uses_nextest(args));
    selects_nextest
        && !(sub_argv != TEST_SUB_ARGV
            && llvm_cov_has_lifecycle_flag(args, "--no-run"))
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
                return Err(
                    "cargo ktstr: split --archive-format is not valid before \
                     cargo-llvm-cov's nextest subcommand; use \
                     --archive-format=tar-zst or place it after nextest"
                        .to_string(),
                );
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
                value.as_str() != "--"
                    && (archive_file || !value.starts_with('-'))
            });
            let Some(value) = value else {
                return Err(format!(
                    "cargo ktstr: {argument} requires a nonempty value"
                ));
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
                return Err(
                    "cargo ktstr: --archive-file requires a nonempty value".to_string()
                );
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
                return Err(
                    "cargo ktstr: --archive-format requires a nonempty value".to_string()
                );
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
                return Err(format!(
                    "cargo ktstr: duplicate {option} is ambiguous"
                ));
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
            return Err(
                "cargo ktstr: --archive-format requires --archive-file".to_string()
            );
        }
        return Ok(None);
    };
    let supported = format.as_ref().map_or_else(
        || file.value.ends_with(".tar.zst"),
        |format| {
            matches!(format.value.as_str(), "tar-zst" | "tar-zstd")
                || (format.value == "auto"
                    && file.value.ends_with(".tar.zst"))
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
        "cargo ktstr: cannot rewrite a nextest invocation without --archive-file"
            .to_string()
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
        let region_start = nextest_region_start(sub_argv, &args)
            .expect("validated archive reuse selects nextest");
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
    !llvm_cov_has_lifecycle_flag(args, "--no-run")
        && !llvm_cov_reuses_archive(sub_argv, args)
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
        if matches!(argument.as_str(), "-j" | "--test-threads") {
            if args
                .get(index + 1)
                .filter(|_| index + 1 < separator)
                .is_some_and(|value| valid_nextest_test_threads(value))
            {
                index += 2;
                continue;
            }
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
        if let Some(value) = argument
            .strip_prefix("--jobs=")
            .or_else(|| {
                argument
                    .strip_prefix("-j")
                    .filter(|value| !value.is_empty())
                    .map(|value| value.strip_prefix('=').unwrap_or(value))
            })
        {
            build_jobs = Some(value.to_string());
            index += 1;
            continue;
        }
        out.push(argument.clone());
        index += 1;
        if LLVM_COV_GLOBAL_VALUE_OPTIONS.contains(&argument.as_str())
            && index < nextest_index
        {
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
    if args.get(0).map(String::as_str) == Some("nextest")
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

/// Build-time env vars handing `cargo-ktstr`'s already-extracted
/// busybox / wprof binaries to the child build, so the downstream
/// `ktstr` `build.rs` copies them into `$OUT_DIR` instead of
/// re-fetching + recompiling (see `install_prebuilt_blob` in
/// `build_helpers.rs`). `cargo-ktstr` exported `KTSTR_BUSYBOX_PATH` /
/// `KTSTR_WPROF_PATH` at startup (`bin/cargo_ktstr/blobs.rs`
/// `install_env`) pointing at the extracted blobs; this re-exports each
/// present path under the build-time `KTSTR_*_BIN` name `build.rs`
/// reads. A path var is present only when the embedded blob was
/// non-empty, so an absent var (cargo-ktstr built without that blob)
/// yields no pair and the child build falls back to its fetch path.
/// Pure with respect to its args so a unit test can drive every
/// present/absent combination. `pub(crate)` because the verifier
/// dispatcher's reserved warm-up (`verifier.rs`) applies the same
/// pairs so its pre-build and combined run share one build
/// fingerprint (`build.rs` watches `KTSTR_BUSYBOX_BIN` /
/// `KTSTR_WPROF_BIN` via `rerun-if-env-changed`).
pub(crate) fn prebuilt_blob_bin_envs(
    busybox_path: Option<std::ffi::OsString>,
    wprof_path: Option<std::ffi::OsString>,
) -> Vec<(&'static str, std::ffi::OsString)> {
    let mut pairs = Vec::new();
    if let Some(p) = busybox_path {
        pairs.push(("KTSTR_BUSYBOX_BIN", p));
    }
    if let Some(p) = wprof_path {
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
        argument.strip_prefix(option).is_some_and(|suffix| {
            suffix.starts_with('=') || (short && !suffix.is_empty())
        })
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
            argument.strip_prefix(option).is_some_and(|suffix| {
                suffix.starts_with('=') || (short && !suffix.is_empty())
            })
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
fn llvm_cov_nextest_archive_args(
    sub_argv: &[&str],
    args: &[String],
    archive_path: &std::path::Path,
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
            if let Some(value) = argument
                .strip_prefix("--jobs=")
                .or_else(|| {
                    argument
                        .strip_prefix("-j")
                        .filter(|value| !value.is_empty())
                        .map(|value| value.strip_prefix('=').unwrap_or(value))
                })
            {
                raw_build_jobs = Some(value.to_string());
                index += 1;
                continue;
            }
            globals.push(argument.clone());
            index += 1;
            if LLVM_COV_GLOBAL_VALUE_OPTIONS.contains(&argument.as_str())
                && index < nextest_index
            {
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
    let (globals, archive_args) =
        llvm_cov_nextest_archive_args(sub_argv, args, archive_path)?;
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

const NEXTEST_ARCHIVE_BINARIES_METADATA: &str =
    "target/nextest/binaries-metadata.json";
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
    #[serde(default)]
    base_output_directories: std::collections::BTreeSet<PathBuf>,
    #[serde(default)]
    linked_paths: std::collections::BTreeSet<PathBuf>,
}

#[derive(serde::Deserialize)]
#[serde(rename_all = "kebab-case")]
struct NextestArchiveBinary {
    binary_path: PathBuf,
}

struct ExtractedNextestArchive {
    _owner: tempfile::TempDir,
    test_binaries: Vec<PathBuf>,
    loader_paths: Vec<PathBuf>,
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

fn nextest_archive_loader_entry_dirs(
    metadata: &NextestArchiveBinaryMetadata,
) -> Result<Vec<PathBuf>, String> {
    let mut paths = Vec::new();
    let mut push_unique = |path| {
        if !paths.contains(&path) {
            paths.push(path);
        }
    };
    for relative in &metadata.rust_build_meta.linked_paths {
        push_unique(validated_archive_target_path(relative)?);
    }
    for relative in &metadata.rust_build_meta.base_output_directories {
        let base = validated_archive_target_path(relative)?;
        push_unique(base.join("deps"));
        push_unique(base);
    }
    push_unique(PathBuf::from("target/nextest/libdirs/host"));
    push_unique(PathBuf::from("target/nextest/libdirs/target/0"));
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
        let mut entry = entry.map_err(|error| {
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
    let loader_entry_dirs = nextest_archive_loader_entry_dirs(&metadata)?;
    let owner = tempfile::Builder::new()
        .prefix("ktstr-nextest-archive-probe-")
        .tempdir()
        .map_err(|error| {
            format!(
                "cargo ktstr: create nextest archive probe directory: {error}"
            )
        })?;
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
        let under_target = path
            .components()
            .next()
            .is_some_and(|component| {
                component == std::path::Component::Normal(std::ffi::OsStr::new("target"))
            })
            && path.components().all(|component| {
                matches!(component, std::path::Component::Normal(_))
            });
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
    let loader_paths = loader_entry_dirs
        .into_iter()
        .map(|path| owner.path().join(path))
        .filter(|path| path.is_dir())
        .collect();
    Ok(ExtractedNextestArchive {
        _owner: owner,
        test_binaries,
        loader_paths,
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
    args: Vec<String>,
) -> Result<(), String> {
    let invocation_dir = std::env::current_dir()
        .map_err(|error| format!("cargo ktstr: read invocation directory: {error}"))?;
    // Archive reuse must be one immutable revision across metadata parsing,
    // scheduler-declaration probing, and nextest's eventual extraction. Pin
    // and publish the validated source before constructing either command,
    // rewrite the user's exact archive option to the leased CAS pathname, and
    // retain the owner through the final child.
    let report_only_llvm_cov = sub_argv != TEST_SUB_ARGV
        && llvm_cov_has_lifecycle_flag(&args, "--no-run");
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
    );
    for (var, val) in &blob_envs {
        cmd.env(var, val);
    }

    if let Some(pat) = profraw_inject_for(sub_argv, std::env::var_os("LLVM_PROFILE_FILE")) {
        cmd.env("LLVM_PROFILE_FILE", pat);
    }

    // Empty `kernel` (flag omitted) -> empty set, auto-discovery path.
    // Non-empty but all-whitespace -> actionable bail (see
    // `kernel_set_or_bail`). Otherwise the resolved (label, dir) set.
    let resolved = kernel_set_or_bail(&kernel, include_eol)?;
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

        // Probe each resolved kernel's commit ONCE here, in the
        // orchestrator, and pass a `dir=commit;...` map down via
        // KTSTR_KERNEL_COMMIT so the sidecar writer skips a redundant gix
        // HEAD + dirty-walk in every per-test nextest process (that walk
        // is memoized per process but not across processes — N tests
        // would re-pay it). Keyed by the same dir string exported as
        // KTSTR_KERNEL / KTSTR_KERNEL_LIST so each sidecar can look
        // itself up. `;` joins entries, `=` splits dir from commit;
        // neither appears in a short hash (hex + optional `-dirty`), and
        // a kernel path containing either would already have broken
        // KTSTR_KERNEL_LIST's own encoding. The commit is resolved via
        // source_dir_for (the same resolution the sidecar uses) then
        // detect_kernel_commit, so the value matches the sidecar's
        // fallback exactly — including clean Path kernels whose resolved
        // dir is a cache entry, not a git tree. Kernels with no
        // recoverable source (transient Range/Git, or a Version/CacheKey
        // cache miss) probe to None and are omitted; their sidecar falls
        // back to the same (correct) None.
        let commit_map = resolved
            .iter()
            .filter_map(|(_, dir)| {
                let raw = dir.display().to_string();
                let commit = ktstr::test_support::source_dir_for(&raw)
                    .and_then(|src| ktstr::test_support::detect_kernel_commit(&src))?;
                Some(format!("{raw}={commit}"))
            })
            .collect::<Vec<_>>()
            .join(";");
        if !commit_map.is_empty() {
            cmd.env(ktstr::KTSTR_KERNEL_COMMIT_ENV, commit_map);
        }

        if resolved.len() > 1 {
            let encoded = encode_kernel_list(&resolved)?;
            eprintln!(
                "cargo ktstr: fanning gauntlet across {n} kernels",
                n = resolved.len(),
            );
            cmd.env(ktstr::KTSTR_KERNEL_LIST_ENV, encoded);
        }
    }

    let target_dir_path =
        resolve_cargo_target_dir_for_args(&args, &invocation_dir)?;

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
            eprintln!("cargo ktstr: BTF type anchor at {}", anchor_path.display());
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
    // acquire_build_reservation uses Consolidate placement (packs onto
    // already-held LLCs, leaving whole LLCs free for exclusive perf-mode
    // reservations — right for a throughput-elastic compile).
    let needs_prebuild = cargo_sub_needs_reserved_prebuild(sub_argv, &args);
    let archive_reuse_path = if cargo_sub_uses_nextest(sub_argv, &args)
        && !llvm_cov_has_lifecycle_flag(&args, "--no-run")
    {
        llvm_cov_archive_path(sub_argv, &args, &invocation_dir)?
    } else {
        None
    };
    let archive_dir = if needs_prebuild && sub_argv != TEST_SUB_ARGV {
        Some(
            tempfile::tempdir()
                .map_err(|error| format!("cargo ktstr: create coverage warm archive dir: {error}"))?,
        )
    } else {
        None
    };
    let archive_probe = archive_reuse_path
        .as_deref()
        .map(|archive_path| {
            eprintln!(
                "cargo ktstr: extracting test binaries for scheduler discovery from {}",
                archive_path.display()
            );
            extract_nextest_archive_test_binaries(archive_path)
        })
        .transpose()?;
    let (test_bins, pinned_test_bins) = if needs_prebuild {
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
    let prepared_scheduler_artifacts = if needs_prebuild || archive_probe.is_some() {
        Some(crate::verifier::prepare_scheduler_artifacts(
            &test_bins,
            archive_probe
                .as_ref()
                .map_or(&[][..], |archive| archive.loader_paths.as_slice()),
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

    // Analyze only requirement-derived, parent-snapshotted scheduler
    // artifacts. Export the same strict manifest to ordinary tests, coverage,
    // and raw llvm-cov nextest; the owner remains alive through run_status.
    if let Some(prepared) = &prepared_scheduler_artifacts {
        cmd.env(
            ktstr::KTSTR_SCHEDULER_MANIFEST_ENV,
            &prepared.manifest_path,
        );
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
    let status = crate::interrupt::run_status(cmd)
        .map_err(|e| format!("spawn cargo {}: {e}", sub_argv.join(" ")))?;
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
    if !status.success() {
        // nextest is the authoritative pass/fail signal. The footer
        // above lists per-test artifacts for failures that produced
        // them; a failure that left NO artifact — a build / vm.run
        // error, a pre-build host error (kvm probe, kernel/scheduler
        // resolve, validation), a host panic, or an unparseable guest
        // result — never reaches the dump / sidecar write sites, so it
        // has no entry above. Defer to the nextest summary for the
        // authoritative failed-test set rather than implying the
        // artifact list is exhaustive.
        eprintln!(
            "\ncargo ktstr: nextest reported failures (see its summary above); \
             per-test artifacts for failures that produced them are listed above. \
             Artifacts under {}.",
            runs_root.display(),
        );
    }
    if status.success() {
        Ok(())
    } else {
        Err(format!(
            "cargo {} exited with {}",
            sub_argv.join(" "),
            status
                .code()
                .map_or("signal".to_string(), |c| c.to_string()),
        ))
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
            && message.get("success").is_some_and(serde_json::Value::is_boolean)
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
            discovery
                .test_executables
                .push(PathBuf::from(executable));
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
    let reservation =
        ktstr::cli::acquire_build_reservation_waiting_interruptible_with_progress(
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

pub(crate) fn acquire_cargo_build_output_lease(
    target_dir: &std::path::Path,
    cli_label: &str,
) -> Result<CargoBuildOutputLease, String> {
    let root = ktstr::cache::cargo_build_output_lock_root()
        .map_err(|error| format!("{cli_label}: resolve Cargo output lock root: {error:#}"))?;
    acquire_cargo_build_output_lease_at_root(
        target_dir,
        &root,
        cli_label,
        &crate::interrupt::INTERRUPTED,
    )
}

fn acquire_cargo_build_output_lease_at_root(
    target_dir: &std::path::Path,
    root: &std::path::Path,
    cli_label: &str,
    interrupted: &std::sync::atomic::AtomicBool,
) -> Result<CargoBuildOutputLease, String> {
    let canonical_target_dir = canonical_cargo_target_dir(target_dir)?;
    std::fs::create_dir_all(&root).map_err(|error| {
        format!(
            "{cli_label}: create Cargo output lock root {}: {error}",
            root.display()
        )
    })?;
    let lock_path = cargo_build_output_lock_path(&root, &canonical_target_dir);
    let started = std::time::Instant::now();
    let mut next_heartbeat = started + std::time::Duration::from_secs(10);
    loop {
        if interrupted.load(std::sync::atomic::Ordering::Acquire) {
            return Err(format!(
                "{cli_label}: interrupted while waiting to own Cargo output directory {}",
                canonical_target_dir.display()
            ));
        }
        match ktstr::flock::try_flock(&lock_path, ktstr::flock::FlockMode::Exclusive)
            .map_err(|error| {
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
                    eprintln!(
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
) -> Result<std::process::Output, String> {
    tracing::debug!("{cli_label}: reserved {description}");
    let progress =
        crate::reserved_build_progress::ReservedBuildProgress::start(cli_label, description);
    crate::interrupt::run_output_observed(command, progress)
        .map_err(|error| format!("{cli_label}: spawn {description}: {error}"))
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
    target_dir: &std::path::Path,
    postprocess: impl FnOnce(&std::process::Output) -> Result<T, String>,
) -> Result<T, String> {
    // Reservation comes first: a queued compile must not own and idle-block
    // its target directory before Cargo is ready to become that directory's
    // writer. Every leased build follows this one ordering.
    let reservation = prepare_reserved_prebuild(&mut command, cli_label)?;
    let lease = acquire_cargo_build_output_lease(target_dir, cli_label)?;
    tracing::debug!(
        target_dir = %lease.target_dir().display(),
        "{cli_label}: acquired Cargo build-output ownership",
    );
    let output = run_prepared_reserved_build_output(command, cli_label, description)?;
    let processed = postprocess(&output);
    drop(lease);
    drop(reservation);
    processed
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
        "harness pre-build with Cargo artifact discovery",
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

/// Precompute cast analysis for exact declaration-derived scheduler
/// artifacts so the first test needing one does not pay the analysis cost.
pub(crate) fn precompute_cast_cache(binaries: &[std::path::PathBuf]) -> Result<(), String> {
    if binaries.is_empty() {
        return Ok(());
    }
    eprintln!(
        "cargo ktstr: precomputing cast analysis for {} scheduler binaries",
        binaries.len()
    );
    let configured_limit = std::thread::available_parallelism()
        .map(|count| count.get())
        .unwrap_or(1);
    precompute_cast_paths_with_pool_builder(
        binaries,
        configured_limit,
        |threads| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .map_err(|error| error.to_string())
        },
        |path| {
            ktstr::precompute_cast_analysis(path).map_err(|error| {
                format!("precompute cast analysis for {}: {error:#}", path.display())
            })
        },
    )
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

fn explicit_cargo_target_dir(
    args: &[String],
    invocation_dir: &std::path::Path,
) -> Option<PathBuf> {
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
        .other_options(crate::feature_discovery::metadata_passthrough_options(
            args,
        ))
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

fn selected_resolved_ktstr_versions_for_context<'metadata>(
    meta: &'metadata cargo_metadata::Metadata,
    args: &[String],
    target: Option<&crate::feature_discovery::TargetContext>,
) -> Vec<&'metadata Version> {
    let Some(packages) = selected_workspace_packages(meta, args) else {
        // Malformed/unsupported selection is left for Cargo to diagnose.
        return Vec::new();
    };
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
) -> Result<(), String> {
    let cli = Version::parse(env!("CARGO_PKG_VERSION"))
        .expect("cargo-ktstr's own CARGO_PKG_VERSION is valid semver");
    let tests = selected_resolved_ktstr_versions_for_context(meta, args, Some(target));
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
            eprintln!("cargo ktstr: warning: {message}");
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
pub(crate) fn prepare_nextest_args(args: Vec<String>) -> Result<Vec<String>, String> {
    let target = match effective_target_context(&args) {
        Ok(target) => target,
        Err(error) => {
            tracing::warn!(
                error,
                "ktstr target discovery/version guard failed; forwarding original Cargo args"
            );
            return Ok(args);
        }
    };
    let manifests = match query_metadata_for_target(&args, MetadataMode::NoDeps, &target) {
        Ok(metadata) => metadata,
        Err(error) => {
            tracing::warn!(
                error,
                "ktstr feature discovery/version guard failed; forwarding original Cargo args"
            );
            return Ok(args);
        }
    };
    let augmented =
        augment_test_features_from_metadata_for_context(args.clone(), &manifests, Some(&target));
    let resolved = match query_resolved_metadata(&augmented, &augmented, &manifests, &target) {
        Ok(metadata) => metadata,
        Err(error) => {
            tracing::warn!(
                error,
                "ktstr version guard could not resolve inferred features; forwarding Cargo args"
            );
            return Ok(augmented);
        }
    };
    check_ktstr_version_compat(&resolved, &augmented, &target)?;
    Ok(augmented)
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

#[allow(clippy::too_many_arguments)]
pub(crate) fn run_test(
    kernel: Vec<String>,
    no_perf_mode: bool,
    no_skip_mode: bool,
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
    nextest_archive_reuse(TEST_SUB_ARGV, &args)?;
    ktstr::cli::check_kvm().map_err(|e| format!("{e:#}"))?;
    ktstr::cli::check_tools(&["cargo-nextest"]).map_err(|e| format!("{e:#}"))?;
    // Smart feature inference must precede registry discovery: `--relevant`
    // probes the exact feature/package/target/profile selection the eventual
    // nextest run will build, rather than a second workspace-wide default.
    let args = if llvm_cov_reuses_archive(TEST_SUB_ARGV, &args) {
        args
    } else {
        prepare_nextest_args(args)?
    };
    let args = apply_relevant_narrowing(args, relevant, base, base_ref, default_branch, release)?;
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
        args,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn run_coverage(
    kernel: Vec<String>,
    no_perf_mode: bool,
    no_skip_mode: bool,
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
    // `coverage` runs the same suite through `cargo llvm-cov nextest`, so use
    // the same version guard and targeted feature inference as `test`.
    let args = if llvm_cov_reuses_archive(COVERAGE_SUB_ARGV, &args) {
        args
    } else {
        prepare_nextest_args(args)?
    };
    let args =
        apply_relevant_narrowing(args, relevant, base, base_ref, default_branch, release)?;
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
        args,
    )
}

pub(crate) fn run_llvm_cov(
    kernel: Vec<String>,
    no_perf_mode: bool,
    no_skip_mode: bool,
    include_eol: bool,
    args: Vec<String>,
) -> Result<(), String> {
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
    let args = prepare_llvm_cov_args_with(args, prepare_nextest_args)
        .map_err(|error| format!("cargo ktstr llvm-cov nextest: {error}"))?;
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
        args,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::OsString;
    use std::sync::{Arc, Mutex, MutexGuard, OnceLock};

    fn v(s: &str) -> Version {
        Version::parse(s).expect("test version literal is valid semver")
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
            let probe =
                ktstr::flock::try_flock(&lock_path, ktstr::flock::FlockMode::Exclusive)
                    .expect("probe first writer's output lock");
            contended_tx
                .send(probe.is_none())
                .expect("report lock contention");
            drop(probe);

            let _second = acquire_cargo_build_output_lease_at_root(
                &target_for_thread,
                &lock_root_for_thread,
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

        let snapshot =
            ktstr::scheduler_artifact::snapshot_pinned_scheduler_artifact(pinned)
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
            let uninjected = inject_nextest_tool_config_with(
                sub_argv,
                args.clone(),
                |_| panic!("report-only --no-run must not inject nextest config"),
            )
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

    /// Serialize env mutation across this binary's tests. nextest runs
    /// tests as parallel threads in ONE process, and `set_var` is
    /// process-global; the lib's `lock_env`/`EnvVarGuard` are
    /// `pub(crate)` to the `ktstr` crate and unreachable from this
    /// binary crate, so the orchestrator-env tests carry their own
    /// lock + save/restore guard.
    fn env_lock() -> MutexGuard<'static, ()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
            .lock()
            .unwrap_or_else(|e| e.into_inner())
    }

    /// Save-and-restore guard for a single env var; restores the prior
    /// value (or absence) on drop. Hold the [`env_lock`] for its life.
    struct EnvVar {
        key: &'static str,
        prev: Option<OsString>,
    }

    impl EnvVar {
        fn set(key: &'static str, val: impl AsRef<std::ffi::OsStr>) -> Self {
            let prev = std::env::var_os(key);
            // SAFETY: `env_lock` serializes env-mutating tests; no other
            // test reads CARGO_TARGET_DIR / KTSTR_RUNS_ROOT concurrently.
            unsafe { std::env::set_var(key, val) };
            Self { key, prev }
        }
        fn remove(key: &'static str) -> Self {
            let prev = std::env::var_os(key);
            // SAFETY: see `set`.
            unsafe { std::env::remove_var(key) };
            Self { key, prev }
        }
    }

    impl Drop for EnvVar {
        fn drop(&mut self) {
            // SAFETY: see `EnvVar::set`.
            unsafe {
                match &self.prev {
                    Some(v) => std::env::set_var(self.key, v),
                    None => std::env::remove_var(self.key),
                }
            }
        }
    }

    /// The orchestrator stamps `KTSTR_RUNS_ROOT` = `{target}/ktstr`
    /// (absolute) so child test processes' sidecar writes AND the
    /// post-run footer reader resolve the SAME dir — the workspace
    /// empty-footer fix. With an absolute `CARGO_TARGET_DIR` and no
    /// pre-set override, `install_runs_root_env` must export that path.
    #[test]
    fn install_runs_root_env_stamps_absolute_target_subdir() {
        let _lock = env_lock();
        let tmp = tempfile::TempDir::new().unwrap();
        let _g0 = EnvVar::remove(ktstr::KTSTR_RUNS_ROOT_ENV);
        let _g1 = EnvVar::set("CARGO_TARGET_DIR", tmp.path());
        install_runs_root_env();
        assert_eq!(
            std::env::var_os(ktstr::KTSTR_RUNS_ROOT_ENV).map(std::path::PathBuf::from),
            Some(tmp.path().join("ktstr")),
            "orchestrator must export the absolute {{target}}/ktstr so the \
             child writers and the footer reader agree on one dir",
        );
    }

    /// A pre-set `KTSTR_RUNS_ROOT` (operator override, or a test that
    /// pinned its own root) must win — `install_runs_root_env` no-ops
    /// rather than clobbering it with the cargo target dir.
    #[test]
    fn install_runs_root_env_is_idempotent_when_already_set() {
        let _lock = env_lock();
        let tmp = tempfile::TempDir::new().unwrap();
        let preset = tmp.path().join("operator-chosen-root");
        let _g0 = EnvVar::set(ktstr::KTSTR_RUNS_ROOT_ENV, &preset);
        let _g1 = EnvVar::set("CARGO_TARGET_DIR", tmp.path());
        install_runs_root_env();
        assert_eq!(
            std::env::var_os(ktstr::KTSTR_RUNS_ROOT_ENV).map(std::path::PathBuf::from),
            Some(preset),
            "a pre-set KTSTR_RUNS_ROOT must survive install (no clobber)",
        );
    }

    /// A RELATIVE `CARGO_TARGET_DIR` is anchored to the orchestrator's
    /// cwd and exported ABSOLUTE — so child test processes (which run
    /// with a different cwd under nextest in a workspace) resolve the
    /// SAME runs root. A relative export would reintroduce the
    /// empty-footer split.
    #[test]
    fn install_runs_root_env_absolutizes_relative_target() {
        let _lock = env_lock();
        let _g0 = EnvVar::remove(ktstr::KTSTR_RUNS_ROOT_ENV);
        let _g1 = EnvVar::set("CARGO_TARGET_DIR", "relative/target");
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

    // -- prebuilt_blob_bin_envs --
    //
    // cargo-ktstr re-exports its extracted busybox / wprof paths to the
    // child build as KTSTR_*_BIN so build.rs copies them instead of
    // re-fetching. A pair is emitted only for a present source path.

    /// Both paths present → both `KTSTR_*_BIN` pairs, busybox first,
    /// carrying the exact path values.
    #[test]
    fn prebuilt_blob_bin_envs_sets_present_paths() {
        let pairs = prebuilt_blob_bin_envs(
            Some(std::ffi::OsString::from("/run/bb")),
            Some(std::ffi::OsString::from("/run/wp")),
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
            prebuilt_blob_bin_envs(None, None).is_empty(),
            "no source paths → no env pairs",
        );
        assert_eq!(
            prebuilt_blob_bin_envs(Some(std::ffi::OsString::from("/run/bb")), None),
            vec![("KTSTR_BUSYBOX_BIN", std::ffi::OsString::from("/run/bb"))],
            "busybox present, wprof absent → only the busybox pair",
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
            strs(&[
                "llvm-cov",
                "nextest",
                "--archive-file",
                "reuse.tar.zst",
            ]),
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
            strs(&[
                "--archive-file",
                "-",
                "nextest",
                "--archive-format=tar-zst",
            ]),
            strs(&[
                "--archive-file=-",
                "nextest",
                "--archive-format=tar-zst",
            ]),
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
            nextest_archive_reuse(
                LLVM_COV_SUB_ARGV,
                &strs(&["--archive-file=-", "nextest"]),
            )
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
                &strs(&[
                    "--archive-file",
                    "reuse.tar.zst",
                    "--archive-format",
                    "zip",
                ]),
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
    fn nextest_archive_probe_extracts_and_orders_loader_directories() {
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
                (
                    "target/debug/deps/libdep-link.so",
                    "libdep.so",
                    false,
                ),
            ],
        );

        let extracted =
            extract_nextest_archive_test_binaries(&archive_path).expect("extract probe closure");
        let relative = extracted
            .loader_paths
            .iter()
            .map(|path| {
                path.strip_prefix(extracted._owner.path())
                    .unwrap()
                    .to_path_buf()
            })
            .collect::<Vec<_>>();
        assert_eq!(
            relative,
            [
                "target/vendor/lib",
                "target/debug/deps",
                "target/debug",
                "target/nextest/libdirs/host",
            ]
            .into_iter()
            .map(PathBuf::from)
            .collect::<Vec<_>>(),
            "probe loader order must match nextest: linked, deps/base, Rust libdirs",
        );
        assert!(extracted
            ._owner
            .path()
            .join("target/unrelated/libignored.so")
            .exists());
        let deps = extracted._owner.path().join("target/debug/deps");
        assert_eq!(
            std::fs::read(deps.join("libdep-link.so")).unwrap(),
            b"dep",
        );
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
        validate_archive_loader_link(
            entry,
            std::path::Path::new("libfoo.so.1"),
            false,
        )
        .unwrap();
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
            validate_archive_loader_link(
                entry,
                std::path::Path::new("/etc/passwd"),
                false,
            )
            .is_err(),
        );
    }

    #[test]
    fn nextest_archive_probe_rejects_binary_outside_target_directory() {
        let metadata: NextestArchiveBinaryMetadata = serde_json::from_slice(br#"{
            "rust-build-meta": {"target-directory": "/original/target"},
            "rust-binaries": {
                "pkg::escape": {"binary-path": "/original/other/escape"}
            }
        }"#)
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
        let _lock = env_lock();
        let lock_dir = tempfile::TempDir::new().expect("tempdir");
        // Isolate the flock namespace into the tempdir and clear every
        // short-circuit so the reservation actually attempts the flock.
        let _g_lockdir = EnvVar::set(ktstr::KTSTR_LOCK_DIR_ENV, lock_dir.path());
        let _g_bypass = EnvVar::remove(ktstr::KTSTR_BYPASS_LLC_LOCKS_ENV);
        let _g_testmode = EnvVar::remove(ktstr::KTSTR_CARGO_TEST_MODE_ENV);
        let _g_cap = EnvVar::remove(ktstr::KTSTR_CPU_CAP_ENV);

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
                let has_llc_lock = std::fs::read_dir(lock_dir.path())
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
