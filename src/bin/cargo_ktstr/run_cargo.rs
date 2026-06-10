//! Dispatch helpers for the `test`, `coverage`, and `llvm-cov`
//! subcommands.
//!
//! All three subcommands share the `cargo nextest`/`cargo
//! llvm-cov` execve plumbing, the `--no-perf-mode` env-var pass-
//! through, and the multi-kernel
//! [`ktstr::KTSTR_KERNEL_LIST_ENV`] export. The differences live
//! in the leading `cargo` subcommand argv (`{nextest run}` vs
//! `{llvm-cov nextest}` vs `{llvm-cov}`) and the optional
//! `--cargo-profile release` injection on the test/coverage
//! paths. [`run_cargo_sub`] folds the shared shape; thin
//! per-subcommand wrappers fix the argv constants.

use std::path::PathBuf;
use std::process::Command;

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
/// present/absent combination.
fn prebuilt_blob_bin_envs(
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
/// append the user's trailing args, and `cmd.exec()` once. The
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
/// `release` is always `false` for the raw `llvm-cov` passthrough —
/// that subcommand hands every argument to the user, so the profile
/// is set via the user's trailing args (or not at all). `test` and
/// `coverage` wire their `--release` flag through to this argument.
fn run_cargo_sub(
    sub_argv: &[&str],
    label: &str,
    kernel: Vec<String>,
    no_perf_mode: bool,
    no_skip_mode: bool,
    release: bool,
    args: Vec<String>,
) -> Result<(), String> {
    let mut cmd = Command::new("cargo");
    cmd.args(sub_argv);
    if release {
        // Prepend `--cargo-profile release` BEFORE the user's
        // trailing args so the profile selection applies to the
        // whole invocation. nextest reads `--cargo-profile` directly;
        // `cargo llvm-cov nextest` forwards it to its inner nextest
        // invocation. For `cargo llvm-cov <sub>` (the raw-passthrough
        // binding), the release arg is never passed here — the raw
        // path relies on user-supplied `--release` / `--profile`.
        cmd.args(["--cargo-profile", "release"]);
    }
    cmd.args(&args);
    if no_perf_mode {
        cmd.env(ktstr::KTSTR_NO_PERF_MODE_ENV, "1");
    }
    if no_skip_mode {
        cmd.env(ktstr::KTSTR_NO_SKIP_MODE_ENV, "1");
    }

    // Hand the child build cargo-ktstr's embedded busybox / wprof so its
    // build.rs copies them instead of re-downloading (see
    // prebuilt_blob_bin_envs). KTSTR_WPROF_PATH uses the literal name —
    // ktstr::KTSTR_WPROF_PATH_ENV is `#[cfg(feature = "wprof")]`, and
    // this propagation is a harmless no-op when wprof was not embedded.
    for (var, val) in prebuilt_blob_bin_envs(
        std::env::var_os(ktstr::KTSTR_BUSYBOX_PATH_ENV),
        std::env::var_os("KTSTR_WPROF_PATH"),
    ) {
        cmd.env(var, val);
    }

    if let Some(pat) = profraw_inject_for(sub_argv, std::env::var_os("LLVM_PROFILE_FILE")) {
        cmd.env("LLVM_PROFILE_FILE", pat);
    }

    if !kernel.is_empty() {
        let resolved = resolve_kernel_set(&kernel)?;
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

        if resolved.len() > 1 {
            let encoded = encode_kernel_list(&resolved)?;
            eprintln!(
                "cargo ktstr: fanning gauntlet across {n} kernels",
                n = resolved.len(),
            );
            cmd.env(ktstr::KTSTR_KERNEL_LIST_ENV, encoded);
        }
    }

    precompute_cast_cache();

    let target_dir_path = resolve_target_dir();

    // BTF type anchor: if a prior build left .bpf.o files, extract
    // struct definitions from the BPF source tree and generate a
    // -include header with weak global anchors that force clang to
    // retain struct types that inlining + DCE would eliminate. The
    // anchor is cached in target/ktstr_btf_anchor.h. First build
    // has no anchor (no prior .bpf.o files); second build onward
    // always uses it. Delete the header to regenerate.
    if let Some(anchor_path) = generate_btf_anchor(&target_dir_path, release) {
        let existing = std::env::var("BPF_EXTRA_CFLAGS_PRE_INCL").unwrap_or_default();
        let inject = format!("-include {} {existing}", anchor_path.display());
        cmd.env("BPF_EXTRA_CFLAGS_PRE_INCL", inject.trim());
        eprintln!("cargo ktstr: BTF type anchor at {}", anchor_path.display());
    }

    tracing::debug!("cargo ktstr: running {label}");
    let status = cmd
        .status()
        .map_err(|e| format!("spawn cargo {}: {e}", sub_argv.join(" ")))?;
    cleanup_shm();
    // Surface where per-test debugging artefacts live so an operator
    // investigating a failure does not have to grep through `target/`
    // to find them. Every test that boots a VM may write any of the
    // artefacts listed in the eprintln block below. The dir is per
    // (kernel, project commit) so failures from different runs of
    // the same test don't stomp each other. Printed on BOTH success
    // and failure: even passing runs leave per-test stats sidecars
    // — and wprof-tagged passing runs leave a Perfetto `.pb` trace
    // — worth inspecting.
    let sidecar_dir = ktstr::test_support::sidecar_dir();
    eprintln!();
    eprintln!("cargo ktstr: test outputs in:");
    eprintln!("  {}", sidecar_dir.display());
    eprintln!("    *.failure-dump.json       — VM-state JSON when a test crashed or asserted");
    eprintln!("    *.repro.failure-dump.json — VM-state JSON from the auto-repro retry");
    eprintln!("    *.sidecar.json            — per-scenario stats + scheduler metadata");
    eprintln!("    *.wprof.pb                — Perfetto trace from #[ktstr_test(wprof)] tests");
    eprintln!("    *.repro.wprof.pb          — Perfetto trace from the auto-repro retry");
    eprintln!(
        "  (run `cargo ktstr replay --dir {}` to step through a captured failure)",
        sidecar_dir.display()
    );
    eprintln!();
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

fn precompute_cast_cache() {
    let target_dir = std::env::var("CARGO_TARGET_DIR").unwrap_or_else(|_| "target".to_string());
    let mut binaries = Vec::new();
    for profile in ["debug", "release"] {
        let dir = std::path::Path::new(&target_dir).join(profile);
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

pub(crate) fn run_test(
    kernel: Vec<String>,
    no_perf_mode: bool,
    no_skip_mode: bool,
    release: bool,
    args: Vec<String>,
) -> Result<(), String> {
    ktstr::cli::check_kvm().map_err(|e| format!("{e:#}"))?;
    ktstr::cli::check_tools(&["cargo-nextest"]).map_err(|e| format!("{e:#}"))?;
    run_cargo_sub(
        TEST_SUB_ARGV,
        "tests",
        kernel,
        no_perf_mode,
        no_skip_mode,
        release,
        args,
    )
}

pub(crate) fn run_coverage(
    kernel: Vec<String>,
    no_perf_mode: bool,
    no_skip_mode: bool,
    release: bool,
    args: Vec<String>,
) -> Result<(), String> {
    ktstr::cli::check_kvm().map_err(|e| format!("{e:#}"))?;
    ktstr::cli::check_tools(&["cargo-nextest", "cargo-llvm-cov"]).map_err(|e| format!("{e:#}"))?;
    run_cargo_sub(
        COVERAGE_SUB_ARGV,
        "coverage",
        kernel,
        no_perf_mode,
        no_skip_mode,
        release,
        args,
    )
}

pub(crate) fn run_llvm_cov(
    kernel: Vec<String>,
    no_perf_mode: bool,
    no_skip_mode: bool,
    args: Vec<String>,
) -> Result<(), String> {
    // `llvm-cov` is raw passthrough — the user supplies every
    // argument after the subcommand name, including any profile
    // selection. `release: false` here means "don't inject a profile
    // ourselves"; the user decides.
    run_cargo_sub(
        LLVM_COV_SUB_ARGV,
        "llvm-cov",
        kernel,
        no_perf_mode,
        no_skip_mode,
        false,
        args,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
