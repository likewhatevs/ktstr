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

use std::path::PathBuf;
use std::process::Command;

use cargo_metadata::semver::Version;

use crate::feature_discovery::{
    MetadataMode, augment_test_features, augment_test_features_from_metadata, query_metadata,
    selected_workspace_packages,
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

/// Whether the raw `cargo ktstr llvm-cov` passthrough explicitly selects
/// cargo-llvm-cov's nextest subcommand.
///
/// Bare `llvm-cov` and its `test` subcommand use Cargo's standard test
/// harness, not ktstr's nextest listing/dispatch protocol. Report, clean,
/// show-env, and run do not enumerate ktstr test binaries either. Keep this
/// deliberately narrow: the documented test-producing form is
/// `cargo ktstr llvm-cov nextest ...`.
fn llvm_cov_uses_nextest(args: &[String]) -> bool {
    let mut index = 0;
    while index < args.len() {
        let argument = &args[index];
        if argument == "--" {
            return false;
        }
        if !argument.starts_with('-') {
            return argument == "nextest";
        }

        // cargo-llvm-cov accepts its global options before the subcommand.
        // Skip the value of every current value-taking global option so a
        // value literally named `nextest` is not mistaken for the subcommand.
        let takes_separate_value = matches!(
            argument.as_str(),
            "--output-path"
                | "--output-dir"
                | "--failure-mode"
                | "--ignore-filename-regex"
                | "--fail-under-functions"
                | "--fail-under-lines"
                | "--fail-under-file-lines"
                | "--fail-under-regions"
                | "--fail-uncovered-lines"
                | "--fail-uncovered-regions"
                | "--fail-uncovered-functions"
                | "--dep-coverage"
                | "--bin"
                | "--example"
                | "--test"
                | "--bench"
                | "-p"
                | "--package"
                | "--exclude"
                | "--exclude-from-test"
                | "--exclude-from-report"
                | "-j"
                | "--jobs"
                | "--profile"
                | "-F"
                | "--features"
                | "--target"
                | "--color"
                | "--manifest-path"
                | "-Z"
        );
        index += if takes_separate_value { 2 } else { 1 };
    }
    false
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
/// re-fetching + recompiling (see `copy_prebuilt_blob` in
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
/// `args` so the profile selection applies to the whole invocation.
/// nextest reads `--cargo-profile` directly; `cargo llvm-cov nextest`
/// forwards it to its inner nextest invocation. For `cargo llvm-cov
/// <sub>` (the raw-passthrough binding) the caller passes `release ==
/// false` and `profile` / `nextest_profile` `None`, so nothing is
/// injected — the raw path relies on user-supplied flags.
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
    cmd.args(sub_argv);
    if release {
        cmd.args(["--cargo-profile", "release"]);
    }
    // `--nextest-profile <NAME>` selects the NEXTEST test profile
    // (`.config/nextest.toml`); nextest's own flag for it is `--profile`,
    // and `cargo llvm-cov nextest` forwards it to its inner nextest.
    if let Some(np) = nextest_profile {
        cmd.args(["--profile", np]);
    }
    cmd.args(args);
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

    let target_dir_path = resolve_target_dir();

    precompute_cast_cache(&target_dir_path);

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

    // Reserve + cgroup-confine the harness COMPILE phase only — NOT the
    // test-running phase (each VM cell takes its own LLC reservation; a
    // reservation still held here would deadlock/starve them). `cargo
    // nextest run` builds then runs in ONE process and the build is not
    // separable, so we run an explicit `cargo nextest run --no-run` warm-up
    // UNDER the reservation: it compiles every test binary while holding
    // machine-global LLC LOCK_SH + a cpuset cgroup, releases both, and the
    // combined run below then finds the build cached — never compiling
    // UNRESERVED where a colocated runner's harness build could invade a
    // perf-mode reservation. acquire_build_reservation uses Consolidate
    // placement (packs onto already-held LLCs, leaving whole LLCs free for
    // exclusive perf-mode reservations — right for a throughput-elastic
    // compile). Only the bare-`nextest` `test` path warms up; see
    // `wants_reserved_prebuild`.
    if wants_reserved_prebuild(sub_argv) {
        let mut warm = build_cargo_command(
            sub_argv,
            release,
            profile.as_deref(),
            nextest_profile.as_deref(),
            no_perf_mode,
            no_skip_mode,
            &prebuild_no_run_args(&args),
        );
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
        run_reserved_prebuild(warm, "cargo ktstr")?;
    }

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
    // Survive Ctrl-C / SIGTERM for the duration of the child run so the
    // parent reaches its cleanup below instead of dying at the default
    // disposition. nextest, in the same foreground process group,
    // receives its own SIGINT and tears down every per-test child
    // independently; this guard only stops the PARENT from dying so the
    // shm sweep + artifact footer still run. See `crate::interrupt`.
    let interrupt_guard = crate::interrupt::InterruptGuard::install();
    let status = cmd
        .status()
        .map_err(|e| format!("spawn cargo {}: {e}", sub_argv.join(" ")))?;
    cleanup_shm();
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
    // Cleanup + footer are done. If a Ctrl-C / SIGTERM arrived during the
    // run, propagate it as the conventional 128+signal exit (130 / 143)
    // now that teardown has completed. Drop the guard first so the signal
    // is back at its prior disposition before we re-raise.
    let caught = interrupt_guard.interrupted();
    drop(interrupt_guard);
    if let Some(sig) = caught {
        crate::interrupt::reraise(sig);
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

/// Does `sub_argv` select a build-then-run-in-one-process path whose
/// COMPILE phase we warm up under an LLC reservation + cgroup sandbox?
///
/// Only the bare `nextest run` (`test`) path. The `coverage`
/// (`llvm-cov nextest`) and raw `llvm-cov` paths run an
/// llvm-cov-INSTRUMENTED build that a plain `cargo nextest run --no-run`
/// warm-up would not match (different RUSTFLAGS) — warming them up would
/// cache-miss and double-build, so they are intentionally left
/// unreserved rather than risk that regression.
fn wants_reserved_prebuild(sub_argv: &[&str]) -> bool {
    sub_argv == TEST_SUB_ARGV
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

/// Run `warm_cmd` (a `cargo … --no-run` compile-only invocation) under a
/// machine-global LLC flock reservation + cgroup-v2 cpuset sandbox, then
/// release BOTH before returning so the caller's test-running phase
/// starts unreserved.
///
/// Mirrors the kernel build's reservation via the shared
/// [`ktstr::cli::acquire_build_reservation_waiting`]: same Consolidate
/// placement, `KTSTR_BYPASS_LLC_LOCKS` / `KTSTR_CARGO_TEST_MODE` /
/// degraded-sysfs short-circuits, and cgroup-degrade gating as kernel builds,
/// plus progress-aware waiting when perf-mode tests temporarily own the
/// required capacity
/// (hard error iff an explicit cap was set — here only via
/// `KTSTR_CPU_CAP`, since `cargo ktstr test` exposes no `--cpu-cap` flag;
/// warn-and-proceed otherwise). Errors abort the run BEFORE any test
/// starts, matching the kernel build's fail-hard-on-cap semantics.
/// `pub(crate)`: shared with the verifier dispatcher's warm-up
/// (`verifier.rs`), whose sweep is the primary colocated-CI workload.
pub(crate) fn run_reserved_prebuild(mut warm_cmd: Command, cli_label: &str) -> Result<(), String> {
    // `cargo ktstr test` has no `--cpu-cap` flag; resolve from
    // `KTSTR_CPU_CAP` (env), else `None` → the same 30%-of-allowed default
    // a cap-less kernel build uses. acquire_build_reservation_waiting enforces
    // the `KTSTR_BYPASS_LLC_LOCKS` + cap conflict identically.
    let cpu_cap = ktstr::cli::CpuCap::resolve(None)
        .map_err(|e| format!("{cli_label}: resolve harness-build CPU cap: {e:#}"))?;
    // RAII: `_reservation`'s cgroup sandbox drops before its LLC flocks per
    // `BuildReservation` field order; both are released when this fn
    // returns, ahead of the combined build+run the caller then spawns.
    let reservation = ktstr::cli::acquire_build_reservation_waiting(cli_label, cpu_cap)
        .map_err(|e| format!("{cli_label}: acquire harness-build reservation: {e:#}"))?;
    // `CARGO_BUILD_JOBS` is the harness-compile analog of the kernel
    // build's `make -jN`: cap parallel rustc to the reserved CPU count so
    // `nproc`-many rustc don't oversubscribe the confined cpuset. `None`
    // (bypass / degraded sysfs → no plan) leaves cargo on its own default.
    if let Some(jobs) = reservation.make_jobs() {
        warm_cmd.env("CARGO_BUILD_JOBS", jobs.to_string());
    }
    tracing::debug!("{cli_label}: reserved harness pre-build (cargo … --no-run)");
    let status = warm_cmd
        .status()
        .map_err(|e| format!("{cli_label}: spawn reserved pre-build: {e}"))?;
    if !status.success() {
        return Err(format!(
            "{cli_label}: reserved pre-build failed ({}) — see cargo output above",
            status
                .code()
                .map_or("signal".to_string(), |c| c.to_string()),
        ));
    }
    Ok(())
    // `reservation` drops here → cgroup rmdir, then LLC flocks release.
}

/// Precompute cast analysis for the built scheduler binaries so the
/// first test needing it doesn't pay the analysis cost inline.
///
/// `target_dir` is the resolved (absolute) cargo target dir, passed in
/// rather than re-read from `CARGO_TARGET_DIR`/"target": in a Cargo
/// workspace invoked from a member-dir CWD a relative "target" points
/// at a package dir that holds no built binaries (they live under the
/// shared workspace target), so the scan would silently find nothing.
/// The caller resolves it once via [`resolve_target_dir`] and shares it
/// with [`generate_btf_anchor`] — no extra `cargo metadata` spawn here.
fn precompute_cast_cache(target_dir: &std::path::Path) {
    let binaries = collect_scheduler_binaries(target_dir);
    if binaries.is_empty() {
        return;
    }
    eprintln!(
        "cargo ktstr: precomputing cast analysis for {} scheduler binaries",
        binaries.len()
    );
    for binary in binaries {
        let path = binary.clone();
        std::thread::spawn(move || {
            ktstr::precompute_cast_analysis(&path);
        });
    }
}

/// Collect built scheduler binaries (`scx_*`, no extension) under
/// `{target_dir}/{debug,release}`. The no-dot filter rejects build
/// byproducts that share the `scx_` prefix (e.g. `scx_foo.d` depfiles,
/// `scx_foo.rlib`); only the bare executable matches, and the
/// `is_file` gate excludes same-named directories. Split out from
/// [`precompute_cast_cache`] so the scan/filter is unit-testable
/// without spawning the per-binary analysis threads.
fn collect_scheduler_binaries(target_dir: &std::path::Path) -> Vec<std::path::PathBuf> {
    let mut binaries = Vec::new();
    for profile in ["debug", "release"] {
        let dir = target_dir.join(profile);
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let name = entry.file_name();
            let Some(name_str) = name.to_str() else {
                continue;
            };
            if name_str.starts_with("scx_") && !name_str.contains('.') {
                let path = entry.path();
                if path.is_file() {
                    binaries.push(path);
                }
            }
        }
    }
    binaries
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

fn resolve_target_dir() -> std::path::PathBuf {
    if let Ok(d) = std::env::var("CARGO_TARGET_DIR") {
        return std::path::PathBuf::from(d);
    }
    if let Ok(output) = Command::new("cargo")
        .args(["metadata", "--format-version=1", "--no-deps"])
        .output()
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
/// LIMITATION: a platform-gated `ktstr` (a `[target.'cfg(...)'.dependencies]`
/// edge) is matched by name regardless of `DepKindInfo.target`; host-triple
/// `Platform` matching is intentionally omitted (it needs the full host cfg
/// set). A cfg-gated ktstr test-driver dependency the host does not link is
/// exotic; if hit, the failure direction is a false-abort — accepted over
/// the matching complexity for that case.
fn linked_ktstr_versions<'metadata>(
    meta: &'metadata cargo_metadata::Metadata,
    root_id: &cargo_metadata::PackageId,
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
                )
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
fn selected_resolved_ktstr_versions<'metadata>(
    meta: &'metadata cargo_metadata::Metadata,
    args: &[String],
) -> Vec<&'metadata Version> {
    let Some(packages) = selected_workspace_packages(meta, args) else {
        // Malformed/unsupported selection is left for Cargo to diagnose.
        return Vec::new();
    };
    let mut versions = packages
        .into_iter()
        .flat_map(|package| linked_ktstr_versions(meta, &package.id))
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
) -> Result<(), String> {
    let cli = Version::parse(env!("CARGO_PKG_VERSION"))
        .expect("cargo-ktstr's own CARGO_PKG_VERSION is valid semver");
    let tests = selected_resolved_ktstr_versions(meta, args);
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

/// Reuse the initial requested-feature metadata result for targeted optional
/// ktstr feature inference and, when nothing is added, the version guard.
///
/// Metadata inspection has historically been best-effort on these general test
/// paths. Preserve that behavior: if it fails, warn and let the underlying
/// Cargo command diagnose (or successfully handle) its own arguments.
fn prepare_test_args(args: Vec<String>) -> Result<Vec<String>, String> {
    let metadata = match query_metadata(&args, MetadataMode::Default) {
        Ok(metadata) => metadata,
        Err(error) => {
            tracing::warn!(
                error,
                "ktstr feature discovery/version guard failed; forwarding original Cargo args"
            );
            return Ok(args);
        }
    };
    let augmented = augment_test_features_from_metadata(args.clone(), &metadata);
    if augmented == args {
        check_ktstr_version_compat(&metadata, &args)?;
        return Ok(args);
    }

    // The first resolve cannot contain an optional ktstr dependency that
    // inference has only just activated. Resolve once more with the targeted
    // package-qualified features so the newer-than-CLI guard inspects the
    // graph Cargo is actually about to build.
    let resolved = match query_metadata(&augmented, MetadataMode::Default) {
        Ok(metadata) => metadata,
        Err(error) => {
            tracing::warn!(
                error,
                "ktstr version guard could not resolve inferred features; forwarding Cargo args"
            );
            return Ok(augmented);
        }
    };
    check_ktstr_version_compat(&resolved, &augmented)?;
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
) -> Result<Vec<String>, String> {
    if !relevant {
        return Ok(args);
    }
    match crate::affected::relevant_test_filter(
        base.as_deref(),
        base_ref.as_deref(),
        &default_branch,
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
    ktstr::cli::check_kvm().map_err(|e| format!("{e:#}"))?;
    ktstr::cli::check_tools(&["cargo-nextest"]).map_err(|e| format!("{e:#}"))?;
    let args = apply_relevant_narrowing(args, relevant, base, base_ref, default_branch)?;
    let args = prepare_test_args(args)?;
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
    ktstr::cli::check_kvm().map_err(|e| format!("{e:#}"))?;
    ktstr::cli::check_tools(&["cargo-nextest", "cargo-llvm-cov"]).map_err(|e| format!("{e:#}"))?;
    let args = apply_relevant_narrowing(args, relevant, base, base_ref, default_branch)?;
    // `coverage` runs the same suite through `cargo llvm-cov nextest`, so use
    // the same version guard and targeted feature inference as `test`.
    let args = prepare_test_args(args)?;
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
    // `llvm-cov` is raw passthrough — the user supplies every
    // argument after the subcommand name, including any profile
    // selection. `release: false` / `profile: None` / `nextest_profile:
    // None` here mean "don't inject any profile ourselves"; the user
    // decides via the raw args.
    //
    // No ktstr version guard here (unlike `test` / `coverage`): a
    // passthrough subcommand like `llvm-cov report` does NOT rebuild or
    // run the tests, so guarding would wrongly block a pure-report
    // invocation under a version-skewed project. A `cargo ktstr llvm-cov
    // nextest` that DOES run tests is the user's explicit raw-passthrough
    // choice; they own the version in that case.
    let args = if llvm_cov_uses_nextest(&args) {
        augment_test_features(args)
            .map_err(|error| format!("cargo ktstr llvm-cov nextest: {error}"))?
    } else {
        args
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
        args,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::OsString;
    use std::sync::{Mutex, MutexGuard, OnceLock};

    fn v(s: &str) -> Version {
        Version::parse(s).expect("test version literal is valid semver")
    }

    #[test]
    fn version_guard_equal_is_ok() {
        assert_eq!(version_guard(&v("0.19.0"), &v("0.19.0")), VersionGuard::Ok);
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
        let out = apply_relevant_narrowing(args.clone(), false, None, None, "main".to_string())
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

    /// `collect_scheduler_binaries` returns only bare `scx_*`
    /// executables under `{target}/{debug,release}`: build byproducts
    /// sharing the prefix (`scx_foo.d`), non-`scx_` files, and
    /// same-named directories are excluded; a missing profile dir is a
    /// no-op (not an error).
    #[test]
    fn collect_scheduler_binaries_filters_to_bare_scx_executables() {
        let tmp = tempfile::TempDir::new().unwrap();
        // No debug/release dirs yet -> nothing collected.
        assert!(collect_scheduler_binaries(tmp.path()).is_empty());

        let debug = tmp.path().join("debug");
        std::fs::create_dir(&debug).unwrap();
        std::fs::write(debug.join("scx_ktstr"), b"x").unwrap(); // collected
        std::fs::write(debug.join("scx_ktstr.d"), b"x").unwrap(); // .d byproduct -> excluded
        std::fs::write(debug.join("ktstr"), b"x").unwrap(); // no scx_ prefix -> excluded
        std::fs::create_dir(debug.join("scx_subdir")).unwrap(); // dir -> excluded by is_file

        assert_eq!(
            collect_scheduler_binaries(tmp.path()),
            vec![debug.join("scx_ktstr")],
            "only the bare scx_* executable is collected",
        );
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

    /// Only the bare `nextest run` (`test`) path warms up under a
    /// reservation. The `coverage` / raw `llvm-cov` paths run an
    /// instrumented build a plain `--no-run` would not match, so they
    /// must NOT warm up (a mismatch would cache-miss and double-build).
    #[test]
    fn wants_reserved_prebuild_gates_test_path_only() {
        assert!(
            wants_reserved_prebuild(TEST_SUB_ARGV),
            "the `test` path must warm up under the reservation",
        );
        assert!(
            !wants_reserved_prebuild(COVERAGE_SUB_ARGV),
            "coverage's llvm-cov-instrumented build must NOT plain-warm-up",
        );
        assert!(
            !wants_reserved_prebuild(LLVM_COV_SUB_ARGV),
            "raw llvm-cov passthrough must NOT warm up",
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
