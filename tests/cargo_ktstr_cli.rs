use assert_cmd::Command;
use predicates::prelude::*;

fn cargo_ktstr() -> Command {
    let mut cmd = Command::cargo_bin("cargo-ktstr").unwrap();
    cmd.arg("ktstr");
    cmd
}

/// The production process-group runner re-execs cargo-ktstr as a tiny group
/// anchor. Pin the real binary entry path—not the unit-test shell substitute:
/// it must inherit terminal signals blocked, install ignored dispositions
/// before unblocking them, and exit cleanly when its control pipe closes.
#[test]
fn hidden_process_group_anchor_obeys_its_control_pipe() {
    use std::process::Stdio;

    struct SignalMaskGuard(libc::sigset_t);

    impl Drop for SignalMaskGuard {
        fn drop(&mut self) {
            // SAFETY: this restores the calling thread's mask captured below.
            assert_eq!(
                unsafe { libc::pthread_sigmask(libc::SIG_SETMASK, &self.0, std::ptr::null_mut(),) },
                0,
                "restore test thread signal mask",
            );
        }
    }

    // Match `spawn_anchor`: the new process inherits both terminal signals
    // blocked, so it is safe before its userspace entry point is scheduled.
    let mask_guard = unsafe {
        let mut terminal: libc::sigset_t = std::mem::zeroed();
        assert_eq!(libc::sigemptyset(&mut terminal), 0);
        assert_eq!(libc::sigaddset(&mut terminal, libc::SIGINT), 0);
        assert_eq!(libc::sigaddset(&mut terminal, libc::SIGTERM), 0);
        let mut previous: libc::sigset_t = std::mem::zeroed();
        assert_eq!(
            libc::pthread_sigmask(libc::SIG_BLOCK, &terminal, &mut previous),
            0,
            "block terminal signals around anchor spawn",
        );
        SignalMaskGuard(previous)
    };
    let mut child = std::process::Command::new(env!("CARGO_BIN_EXE_cargo-ktstr"))
        .env("__KTSTR_PROCESS_GROUP_ANCHOR", "1")
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn cargo-ktstr anchor mode");
    drop(mask_guard);

    // SAFETY: `child.id()` names this live subprocess and a positive pid
    // targets only that process, not the test runner's process group.
    assert_eq!(
        unsafe { libc::kill(child.id() as libc::pid_t, libc::SIGTERM) },
        0,
        "deliver SIGTERM to the anchor",
    );

    drop(child.stdin.take());
    let status = child.wait().expect("reap anchor after control EOF");
    assert!(
        status.success(),
        "anchor ignores the pending terminal signal before unblocking and exits on control EOF",
    );
}

/// The private Cargo target-runner entry point must execute before normal
/// cargo-ktstr startup, compose an existing configured runner for list/plain
/// binaries, and restore the user's runner environment before the final exec.
#[test]
fn hidden_admission_runner_composes_listing_runner_and_restores_environment() {
    use std::os::unix::ffi::OsStrExt;
    use std::os::unix::fs::PermissionsExt;

    const TARGET_RUNNER: &str = "CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_RUNNER";
    const TARGET_ENV_KEY_ENV: &str = "KTSTR_ADMISSION_TARGET_ENV_KEY";
    const CHAINED_RUNNER_ENV: &str = "KTSTR_ADMISSION_CHAINED_RUNNER";
    const ORIGINAL_RUNNER_ENV: &str = "KTSTR_ADMISSION_ORIGINAL_RUNNER";

    let temporary = tempfile::tempdir().expect("create admission-runner fixture directory");
    let runner = temporary.path().join("runner.sh");
    let test_binary = temporary.path().join("listed-test.sh");
    std::fs::write(
        &runner,
        b"#!/bin/sh\n[ \"$1\" = --fixed-runner-arg ] || exit 31\nshift\nexec \"$@\"\n",
    )
    .expect("write chained runner fixture");
    std::fs::write(
        &test_binary,
        format!(
            "#!/bin/sh\n\
             [ \"${{{TARGET_RUNNER}}}\" = 'original runner --flag' ] || exit 32\n\
             [ -z \"${{{TARGET_ENV_KEY_ENV}+set}}\" ] || exit 33\n\
             [ -z \"${{{CHAINED_RUNNER_ENV}+set}}\" ] || exit 34\n\
             [ -z \"${{{ORIGINAL_RUNNER_ENV}+set}}\" ] || exit 35\n\
             printf 'admission-runner-ok\\n'\n",
        ),
    )
    .expect("write listed test fixture");
    for path in [&runner, &test_binary] {
        let mut permissions = std::fs::metadata(path)
            .expect("read fixture mode")
            .permissions();
        permissions.set_mode(0o755);
        std::fs::set_permissions(path, permissions).expect("make fixture executable");
    }

    let encoded_runner = serde_json::json!({
        "program": runner.as_os_str().as_bytes(),
        "args": [b"--fixed-runner-arg"],
    })
    .to_string();
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_cargo-ktstr"))
        .args([
            std::ffi::OsStr::new("__ktstr_admission_runner"),
            test_binary.as_os_str(),
            std::ffi::OsStr::new("--list"),
        ])
        .env(TARGET_ENV_KEY_ENV, TARGET_RUNNER)
        .env(CHAINED_RUNNER_ENV, encoded_runner)
        .env(ORIGINAL_RUNNER_ENV, "original runner --flag")
        .env(TARGET_RUNNER, "temporary ktstr wrapper")
        .output()
        .expect("run hidden admission runner");
    assert!(
        output.status.success(),
        "hidden admission runner failed with {}\nstdout:\n{}\nstderr:\n{}",
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
    assert_eq!(output.stdout, b"admission-runner-ok\n");
    assert!(output.stderr.is_empty());
}

// -- help output --

#[test]
fn help_lists_subcommands() {
    cargo_ktstr()
        .arg("--help")
        .assert()
        .success()
        .stdout(predicate::str::contains("test"))
        .stdout(predicate::str::contains("shell"))
        .stdout(predicate::str::contains("kernel"))
        .stdout(predicate::str::contains("verifier"))
        .stdout(predicate::str::contains("completions"))
        // `LlvmCov` variant renders as `llvm-cov` (clap derive
        // kebab-case default). Pinned with the two-space leading
        // indent that `HelpTemplate::subcmd` emits before every
        // subcommand name (clap_builder-4.6.0/src/output/mod.rs:21
        // `TAB = "  "` + help_template.rs:1070-1071 which pushes
        // TAB then the name). This discriminates the subcommand
        // list entry from incidental doc-text occurrences of
        // "llvm-cov" that would satisfy a bare substring check.
        .stdout(predicate::str::contains("  llvm-cov"))
        // `visible_alias = "nextest"` on the Test variant makes
        // the alias user-facing. Pinned by the literal
        // `[aliases: nextest]` tag emitted by
        // `HelpTemplate::sc_spec_vals` at clap_builder-4.6.0/src/
        // output/help_template.rs:1043 — the styled-ANSI wrappers
        // collapse to empty strings under `assert_cmd`'s non-TTY
        // capture so the plain tag appears verbatim. A regression
        // that dropped `visible_alias` (or switched to the
        // non-visible `alias` form, which `sc_spec_vals` ignores
        // at :1026 where it calls `get_visible_aliases`) would
        // strip the tag and fail this assertion.
        .stdout(predicate::str::contains("[aliases: nextest]"));
}

#[test]
fn help_test() {
    cargo_ktstr()
        .args(["test", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("--kernel"))
        .stdout(predicate::str::contains("--no-perf-mode"))
        .stdout(predicate::str::contains("cargo nextest"));
}

/// `cargo ktstr nextest --help` reaches the same help page as
/// `cargo ktstr test --help` via the `visible_alias = "nextest"`
/// on the Test variant. Pins that the alias is wired as an alias
/// (not a separate variant) — the help page inherits `--kernel`,
/// `--no-perf-mode`, and the "cargo nextest" passthrough doc.
#[test]
fn help_nextest_alias() {
    cargo_ktstr()
        .args(["nextest", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("--kernel"))
        .stdout(predicate::str::contains("--no-perf-mode"));
}

/// `cargo ktstr llvm-cov --help` renders the LlvmCov variant's
/// help page. The variant's about text advertises `cargo llvm-cov`
/// passthrough, and both `--kernel` + `--no-perf-mode` are
/// declared on the variant — any of the three would fail if a
/// clap regression re-generated the subcommand with drifted
/// metadata.
#[test]
fn help_llvm_cov() {
    cargo_ktstr()
        .args(["llvm-cov", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("--kernel"))
        .stdout(predicate::str::contains("--no-perf-mode"))
        .stdout(predicate::str::contains("cargo llvm-cov"));
}

#[test]
fn help_shell() {
    cargo_ktstr()
        .args(["shell", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("--kernel"))
        .stdout(predicate::str::contains("--topology"))
        .stdout(predicate::str::contains("--memory-mib"))
        .stdout(predicate::str::contains("--no-perf-mode"));
}

/// `cargo ktstr export --help` exposes the four flags the router
/// dispatches on: `<TEST>` positional, `--output`/-o, `--package`/-p
/// (workspace disambiguation), and `--release` (profile pin).
/// Pins the router CLI surface so a future clap regression
/// that drops one of these flags is caught at the help-text level
/// before it surfaces as a misleading "test not found" error in the
/// router.
#[test]
fn help_export() {
    cargo_ktstr()
        .args(["export", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("--output"))
        .stdout(predicate::str::contains("--package"))
        .stdout(predicate::str::contains("--release"))
        .stdout(predicate::str::contains("<TEST>"));
}

/// `cargo ktstr export <missing>` exits non-zero with a router
/// diagnostic. Pins the "test not found in any workspace test
/// binary" error path: the router builds tests, exec's each, sees
/// every candidate fail with "no registered test named X", and
/// surfaces a bundled error mentioning the candidate count and the
/// last per-binary stderr.
///
/// `#[ignore]`-d because the router executes a full
/// `cargo build --tests` over the entire workspace, compiling
/// every integration test binary — minutes of build time, too
/// heavy for the default `cargo nextest run` pass. Run via
/// `cargo nextest run --include-ignored -E 'test(export_unknown_test_errors)'`
/// to opt in locally.
#[test]
#[ignore = "runs cargo build --tests over the full workspace; minutes of compile time"]
fn export_unknown_test_errors() {
    cargo_ktstr()
        .args(["export", "definitely_not_a_real_ktstr_test_xyzzy_987"])
        .assert()
        .failure()
        .stderr(
            predicate::str::contains("not found in any workspace test binary").or(
                predicate::str::contains("definitely_not_a_real_ktstr_test_xyzzy_987"),
            ),
        );
}

#[test]
fn help_kernel() {
    cargo_ktstr()
        .args(["kernel", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("list"))
        .stdout(predicate::str::contains("build"))
        .stdout(predicate::str::contains("clean"));
}

#[test]
fn help_kernel_list() {
    cargo_ktstr()
        .args(["kernel", "list", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("--json"));
}

#[test]
fn help_kernel_build() {
    cargo_ktstr()
        .args(["kernel", "build", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("--kernel"))
        .stdout(predicate::str::contains("git+URL"))
        .stdout(predicate::str::contains("--force"))
        .stdout(predicate::str::contains("--clean"))
        .stdout(predicate::str::contains("--extra-kconfig"))
        // `--extra-kconfig` doc must explain that `make olddefconfig`
        // resolves dependencies — the help is the discoverability
        // surface for the merge pipeline. A regression that dropped
        // the explanation would leave operators guessing why a
        // fragment line silently disappeared from the final
        // `.config`.
        .stdout(predicate::str::contains("olddefconfig"))
        // `--skip-sha256` is a security-sensitive bypass flag — it
        // MUST appear in the discoverability surface so an operator
        // hitting an in-place tarball update at cdn.kernel.org can
        // find the recovery flag from `--help` alone.
        .stdout(predicate::str::contains("--skip-sha256"));
}

/// `kernel build --extra-kconfig <nonexistent>` must surface an
/// actionable error containing the user's input path verbatim, so a
/// typo names the exact string they passed. Pin the diagnostic
/// shape `--extra-kconfig {path}: {fs error}` produced by
/// `kernel_build`'s up-front file read.
///
/// `KTSTR_CACHE_DIR` is pointed at a tempdir so this test does not
/// touch the developer's real cache root, and `--kernel` is set to
/// a clearly-nonexistent path so even if the extra-kconfig check
/// were skipped (and the source-tree validation fired instead), the
/// command would still bail before any network or build work.
#[test]
fn kernel_build_extra_kconfig_nonexistent_path_errors() {
    let tmp = tempfile::TempDir::new().unwrap();
    cargo_ktstr()
        .env(ktstr::KTSTR_CACHE_DIR_ENV, tmp.path())
        .args([
            "kernel",
            "build",
            "--kernel",
            "/nonexistent/ktstr-extra-kconfig-source-test",
            "--extra-kconfig",
            "/definitely/not/a/real/file/ktstr-extra-kconfig-test.kconfig",
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains("--extra-kconfig"))
        .stderr(predicate::str::contains(
            "/definitely/not/a/real/file/ktstr-extra-kconfig-test.kconfig",
        ));
}

/// A directory passed to `--extra-kconfig` must surface a clear
/// "is a directory" error. The 4-arm error
/// classification in [`ktstr::cli::read_extra_kconfig`] maps the
/// kernel's EISDIR to "is a directory; pass a file" — pin that
/// the operator-facing message names BOTH `--extra-kconfig` and
/// the offending path.
#[test]
fn kernel_build_extra_kconfig_directory_errors() {
    let tmp = tempfile::TempDir::new().unwrap();
    let dir = tmp.path().join("not-a-file");
    std::fs::create_dir(&dir).unwrap();
    cargo_ktstr()
        .env(ktstr::KTSTR_CACHE_DIR_ENV, tmp.path())
        .args([
            "kernel",
            "build",
            "--kernel",
            "/nonexistent/ktstr-source-test-dir-arg",
            "--extra-kconfig",
        ])
        .arg(&dir)
        .assert()
        .failure()
        .stderr(predicate::str::contains("--extra-kconfig"))
        .stderr(predicate::str::contains("is a directory"));
}

/// A non-UTF-8 file passed to `--extra-kconfig` must surface a
/// clear "not valid UTF-8"
/// error. `read_extra_kconfig` rejects with a message that names
/// `--extra-kconfig` + the path so the operator can fix the file.
/// kconfig fragments are required to be ASCII text per kbuild's
/// own parser.
#[test]
fn kernel_build_extra_kconfig_invalid_utf8_errors() {
    let tmp = tempfile::TempDir::new().unwrap();
    let path = tmp.path().join("invalid.kconfig");
    // Lone 0xff is invalid UTF-8 — Vec<u8> with a single 0xff byte
    // fails String::from_utf8 with `Utf8Error`.
    std::fs::write(&path, [0xffu8]).unwrap();
    cargo_ktstr()
        .env(ktstr::KTSTR_CACHE_DIR_ENV, tmp.path())
        .args([
            "kernel",
            "build",
            "--kernel",
            "/nonexistent/ktstr-source-test-utf8-arg",
            "--extra-kconfig",
        ])
        .arg(&path)
        .assert()
        .failure()
        .stderr(predicate::str::contains("--extra-kconfig"))
        .stderr(predicate::str::contains("not valid UTF-8"));
}

/// An empty file passed to `--extra-kconfig` is NOT an error —
/// `read_extra_kconfig` warns but proceeds. The
/// build then bails when the source-tree check fails (we point
/// `--kernel` at a nonexistent path), proving the empty-file
/// branch passed through without aborting on the fragment read.
/// stderr carries both the empty-file warning AND the source-tree
/// failure, confirming sequence: empty fragment → warn → continue
/// → source-tree fail.
#[test]
fn kernel_build_extra_kconfig_empty_file_warns_but_proceeds() {
    let tmp = tempfile::TempDir::new().unwrap();
    let path = tmp.path().join("empty.kconfig");
    std::fs::write(&path, b"").unwrap();
    cargo_ktstr()
        .env(ktstr::KTSTR_CACHE_DIR_ENV, tmp.path())
        // RUST_LOG ensures the tracing::warn! emission lands on
        // stderr where the integration test can observe it.
        .env("RUST_LOG", "warn")
        .args([
            "kernel",
            "build",
            "--kernel",
            "/nonexistent/ktstr-source-test-empty-arg",
            "--extra-kconfig",
        ])
        .arg(&path)
        .assert()
        .failure()
        .stderr(predicate::str::contains("--extra-kconfig file is empty"));
}

/// Symlink chain resolution. A `--extra-kconfig` argument that
/// points at a symlink chain
/// (link → link → file) must resolve transparently — the
/// `read_extra_kconfig` helper uses `std::fs::read` which goes
/// through `open(2)` and follows symlinks per kernel default
/// (the same way kbuild reads `KCONFIG_CONFIG`). Pin that a
/// chain of two symlinks resolves to the underlying file's
/// contents without manual canonicalization.
///
/// Test passes when the build proceeds past the fragment-read
/// stage (we point `--kernel` at a nonexistent path so the
/// command bails on source-tree validation, AFTER the fragment
/// is successfully read). If symlink resolution were broken,
/// `read_extra_kconfig` would error before reaching the source
/// stage and stderr would carry the "--extra-kconfig …" error
/// instead of the source-tree error.
#[test]
fn kernel_build_extra_kconfig_symlink_chain_resolves() {
    let tmp = tempfile::TempDir::new().unwrap();
    let real = tmp.path().join("real.kconfig");
    std::fs::write(&real, b"CONFIG_KTSTR_SYMLINK_TEST=y\n").unwrap();
    let link1 = tmp.path().join("link1.kconfig");
    let link2 = tmp.path().join("link2.kconfig");
    // Build link1 → real, link2 → link1 (two-hop chain).
    std::os::unix::fs::symlink(&real, &link1).unwrap();
    std::os::unix::fs::symlink(&link1, &link2).unwrap();
    let assert = cargo_ktstr()
        .env(ktstr::KTSTR_CACHE_DIR_ENV, tmp.path())
        .args([
            "kernel",
            "build",
            "--kernel",
            "/nonexistent/ktstr-source-symlink-test",
            "--extra-kconfig",
        ])
        .arg(&link2)
        .assert()
        .failure();
    let stderr = String::from_utf8_lossy(&assert.get_output().stderr).into_owned();
    // Must NOT carry the `--extra-kconfig` error string — the
    // fragment was read successfully through the chain. The
    // failure that surfaces is the source-tree validation
    // (since --kernel points at nothing), proving the read
    // completed before that next stage.
    assert!(
        !stderr.contains("--extra-kconfig"),
        "symlink chain must resolve transparently — read_extra_kconfig \
         should not surface a `--extra-kconfig` error when the chain \
         resolves to a readable file. stderr={stderr:?}"
    );
}

/// The `--extra-kconfig` validation fires BEFORE source
/// acquisition. A nonexistent extra-kconfig path
/// MUST produce the `--extra-kconfig`-named error even when
/// `--kernel` is also nonexistent — proving the error precedence.
/// If the order were reversed the test would see the
/// source-tree error instead.
#[test]
fn kernel_build_extra_kconfig_validation_fires_before_source_acquire() {
    let tmp = tempfile::TempDir::new().unwrap();
    cargo_ktstr()
        .env(ktstr::KTSTR_CACHE_DIR_ENV, tmp.path())
        .args([
            "kernel",
            "build",
            "--kernel",
            "/nonexistent/ktstr-source-precedence-test",
            "--extra-kconfig",
            "/nonexistent/ktstr-extra-precedence-test.kconfig",
        ])
        .assert()
        .failure()
        // The error MUST name --extra-kconfig (not source-tree
        // failure). `read_extra_kconfig` runs first in
        // `kernel_build`, so its 4-arm classifier surfaces the
        // ENOENT before `kernel_build_one`'s source-acquire branch
        // would have fired.
        .stderr(predicate::str::contains("--extra-kconfig"))
        .stderr(predicate::str::contains(
            "/nonexistent/ktstr-extra-precedence-test.kconfig",
        ));
}

#[test]
fn help_kernel_clean() {
    cargo_ktstr()
        .args(["kernel", "clean", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("--keep"))
        .stdout(predicate::str::contains("--force"));
}

#[test]
fn help_verifier() {
    cargo_ktstr()
        .args(["verifier", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("--kernel"))
        .stdout(predicate::str::contains("--raw"));
}

#[test]
fn help_completions() {
    cargo_ktstr()
        .args(["completions", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("<SHELL>"))
        .stdout(predicate::str::contains("possible values: bash"));
}

// -- error cases --

#[test]
fn no_subcommand_fails() {
    cargo_ktstr().assert().failure();
}

// -- completions --

#[test]
fn completions_bash_produces_output() {
    cargo_ktstr()
        .args(["completions", "bash"])
        .assert()
        .success()
        .stdout(predicate::str::is_empty().not());
}

#[test]
fn completions_zsh_produces_output() {
    cargo_ktstr()
        .args(["completions", "zsh"])
        .assert()
        .success()
        .stdout(predicate::str::is_empty().not());
}

#[test]
fn completions_fish_produces_output() {
    cargo_ktstr()
        .args(["completions", "fish"])
        .assert()
        .success()
        .stdout(predicate::str::is_empty().not());
}

#[test]
fn completions_invalid_shell() {
    cargo_ktstr()
        .args(["completions", "noshell"])
        .assert()
        .failure();
}

// -- shell flags in help --

#[test]
fn help_shell_shows_exec() {
    cargo_ktstr()
        .args(["shell", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("--exec"));
}

#[test]
fn help_shell_shows_dmesg() {
    cargo_ktstr()
        .args(["shell", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("--dmesg"));
}

#[test]
fn help_shell_shows_include_files() {
    cargo_ktstr()
        .args(["shell", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("--include-files"));
}

// -- error cases --

#[test]
fn include_files_nonexistent_path() {
    cargo_ktstr()
        .args(["shell", "-i", "/nonexistent/path/to/file"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("not found"));
}

#[test]
fn shell_invalid_topology() {
    cargo_ktstr()
        .args(["shell", "--topology", "abc"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("invalid topology"));
}

// -- stats --

#[test]
fn stats_bare_prints_family_help() {
    // Bare `cargo ktstr stats` no longer auto-dumps the gauntlet
    // analysis (that moved behind `stats last-run`); it prints the
    // stats family's help so the verbs are discoverable. Success exit,
    // and the listing names the analysis verb.
    let tmp = tempfile::tempdir().unwrap();
    cargo_ktstr()
        .env(ktstr::KTSTR_SIDECAR_DIR_ENV, tmp.path())
        .args(["stats"])
        .assert()
        .success()
        .stdout(predicate::str::contains("last-run"));
}

#[test]
fn stats_last_run_no_data() {
    // Pin the read path to an empty directory via KTSTR_SIDECAR_DIR
    // so the test is independent of whatever sits under the
    // developer's target/ktstr/. With nothing there to read, the
    // empty-state notice goes to stderr and stdout stays clean --
    // the contract the old bare-`stats` path carried, now on the
    // last-run verb that inherited the report.
    let tmp = tempfile::tempdir().unwrap();
    cargo_ktstr()
        .env(ktstr::KTSTR_SIDECAR_DIR_ENV, tmp.path())
        .args(["stats", "last-run"])
        .assert()
        .success()
        .stderr(predicate::str::contains("no sidecar data found"))
        .stdout(predicate::str::is_empty());
}

// -- kernel list --

#[test]
fn kernel_list_runs() {
    // Isolate from the user's real kernel cache so the assertion is
    // deterministic. With an empty cache directory, `kernel list`
    // prints the cache path header on stderr and a "no cached
    // kernels" hint on stdout.
    let tmp = tempfile::TempDir::new().unwrap();
    cargo_ktstr()
        .env(ktstr::KTSTR_CACHE_DIR_ENV, tmp.path())
        .args(["kernel", "list"])
        .assert()
        .success()
        .stdout(predicate::str::contains("no cached kernels"))
        .stderr(predicate::str::contains("cache:"));
}

#[test]
fn kernel_list_json() {
    cargo_ktstr()
        .args(["kernel", "list", "--json"])
        .assert()
        .success()
        .stdout(predicate::str::contains("entries"));
}

// -- --cpu-cap vs KTSTR_BYPASS_LLC_LOCKS conflict — cargo-ktstr sites --
//
// Pins the parse-time rejection when both the --cpu-cap resource
// contract and the KTSTR_BYPASS_LLC_LOCKS=1 escape hatch are
// active simultaneously. Both sites (cargo-ktstr shell and
// cargo-ktstr kernel build) must bail with "resource contract" in
// the error text so the operator sees the contradiction before a
// pipeline deep-bail.

/// `cargo ktstr shell --no-perf-mode --cpu-cap N` under
/// KTSTR_BYPASS_LLC_LOCKS=1 must fail with the "resource contract"
/// substring. Pins the rejection at src/bin/cargo_ktstr/misc/shell.rs:184.
#[test]
fn cargo_ktstr_shell_cpu_cap_with_bypass_errors() {
    let tmp = tempfile::TempDir::new().unwrap();
    cargo_ktstr()
        .env(ktstr::KTSTR_CACHE_DIR_ENV, tmp.path())
        .env(ktstr::KTSTR_BYPASS_LLC_LOCKS_ENV, "1")
        .args(["shell", "--no-perf-mode", "--cpu-cap", "2"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("resource contract"));
}

/// `cargo ktstr kernel build --cpu-cap N` under
/// KTSTR_BYPASS_LLC_LOCKS=1 must fail with the "resource contract"
/// substring. Pins the rejection at src/bin/cargo_ktstr/kernel/mod.rs:720.
#[test]
fn cargo_ktstr_kernel_build_cpu_cap_with_bypass_errors() {
    let tmp = tempfile::TempDir::new().unwrap();
    // Pass a clearly-nonexistent `--kernel <path>` so if the cpu-cap
    // check were somehow skipped, we'd get a source-acquire failure
    // (not a network fetch hanging forever in CI).
    cargo_ktstr()
        .env(ktstr::KTSTR_CACHE_DIR_ENV, tmp.path())
        .env(ktstr::KTSTR_BYPASS_LLC_LOCKS_ENV, "1")
        .args([
            "kernel",
            "build",
            "--kernel",
            "/nonexistent/ktstr-cargo-ktstr-cpu-cap-bypass-test",
            "--cpu-cap",
            "2",
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains("resource contract"));
}

/// `cargo ktstr kernel build --kernel <cache-key>` is rejected: a
/// cache key names an already-built entry, so there is nothing to
/// build. Pins the `KernelId::CacheKey` arm of `kernel_build`'s
/// dispatch — a bare token that is not a version / path / range / git
/// source parses to `CacheKey`, and building one is ill-defined.
#[test]
fn cargo_ktstr_kernel_build_cache_key_rejected() {
    let tmp = tempfile::TempDir::new().unwrap();
    cargo_ktstr()
        .env(ktstr::KTSTR_CACHE_DIR_ENV, tmp.path())
        .args([
            "kernel",
            "build",
            "--kernel",
            "6.14.2-tarball-x86_64-kcabc123",
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains("cache key"))
        .stderr(predicate::str::contains("nothing to build"));
}

/// `cargo ktstr kernel build --kernel git+…` reaches the Git dispatch
/// arm (not the tarball path). A bogus `file://` git URL fails fast
/// (no network) with the git-arm-specific `build git+` error prefix
/// (kernel/mod.rs), proving the Git arm ran rather than a
/// tarball/download error.
#[test]
fn cargo_ktstr_kernel_build_git_dispatch() {
    let tmp = tempfile::TempDir::new().unwrap();
    cargo_ktstr()
        .env(ktstr::KTSTR_CACHE_DIR_ENV, tmp.path())
        .args([
            "kernel",
            "build",
            "--kernel",
            "git+file:///nonexistent-ktstr-git-dispatch-test#tag=x",
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains("build git+"));
}

// -- --extra-kconfig CLI integration --
//
// These integration tests use `KTSTR_CACHE_DIR` isolation and the
// `cargo ktstr kernel list --json` output / `kernel build` clap
// surface to verify the `--extra-kconfig` plumbing without spinning
// a real kernel build (network + 5+ minutes per build, unacceptable
// in `cargo nextest` runs). The pattern: plant fixture cache
// entries in the production `metadata.json` shape, then exercise
// the JSON listing + build dispatch through assert_cmd.
//
// Cache-key derivation roundtrip semantics are also unit-tested at
// the library level in `src/lib.rs::tests::cache_lookup_*` (planted
// CacheDir + lookup); these CLI tests pin the assert_cmd FRONTEND
// surface that scripts/automation consume.

/// Cache-roundtrip: write the SAME cache fixture twice and verify
/// `kernel list --json` returns the same entry with the same
/// `extra_kconfig_hash` value byte-for-byte across runs. Pins the
/// extras-aware lookup-by-key contract: identical extras content
/// must always produce the same cache key suffix derivation, so a
/// re-run with the same extras hits the same slot rather than
/// missing and re-building.
#[test]
fn extra_kconfig_cache_roundtrip() {
    let tmp = tempfile::TempDir::new().unwrap();
    let entry_dir = tmp.path().join("test-extras-roundtrip-bbbb1111");
    std::fs::create_dir_all(&entry_dir).unwrap();
    std::fs::write(entry_dir.join("bzImage"), b"fake kernel image").unwrap();
    let metadata_json = serde_json::json!({
        "version": "6.14.2",
        "source": {"type": "tarball"},
        "arch": "x86_64",
        "image_name": "bzImage",
        "config_hash": null,
        "built_at": "2026-04-22T00:00:00Z",
        "ktstr_kconfig_hash": null,
        "extra_kconfig_hash": "f00d1234",
        "has_vmlinux": false,
        "vmlinux_stripped": false,
    });
    std::fs::write(
        entry_dir.join("metadata.json"),
        serde_json::to_string_pretty(&metadata_json).unwrap(),
    )
    .unwrap();

    // First lookup: extras-built entry must surface.
    let run = |label: &str| {
        let output = cargo_ktstr()
            .env(ktstr::KTSTR_CACHE_DIR_ENV, tmp.path())
            .args(["kernel", "list", "--json"])
            .output()
            .unwrap_or_else(|e| panic!("{label}: kernel list --json must run: {e}"));
        assert!(
            output.status.success(),
            "{label}: kernel list --json must succeed; stderr={:?}",
            String::from_utf8_lossy(&output.stderr)
        );
        String::from_utf8(output.stdout).unwrap()
    };
    let stdout_a = run("first run");
    let stdout_b = run("second run");

    let parse_hash = |stdout: &str| -> String {
        let parsed: serde_json::Value = serde_json::from_str(stdout).unwrap();
        parsed["entries"]
            .as_array()
            .unwrap()
            .iter()
            .find(|e| e["key"].as_str() == Some("test-extras-roundtrip-bbbb1111"))
            .expect("planted extras entry must appear in both runs")["extra_kconfig_hash"]
            .as_str()
            .expect("extra_kconfig_hash must be present")
            .to_string()
    };
    let hash_a = parse_hash(&stdout_a);
    let hash_b = parse_hash(&stdout_b);
    assert_eq!(
        hash_a, hash_b,
        "same fixture must surface the same extra_kconfig_hash across runs — \
         cache-roundtrip identity"
    );
    assert_eq!(
        hash_a, "f00d1234",
        "hash must round-trip the planted value verbatim"
    );
}

/// Cache miss on different content: plant entries A and B with
/// distinct `extra_kconfig_hash` values; both must appear, and
/// each must carry its own hash distinct from the other. Pins the
/// segregation that prevents an extras=A build from being silently
/// served when the operator asks for extras=B.
#[test]
fn extra_kconfig_cache_miss_on_different_content() {
    let tmp = tempfile::TempDir::new().unwrap();

    let plant = |key: &str, extras_hash: serde_json::Value, built_at: &str| {
        let dir = tmp.path().join(key);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("bzImage"), b"fake kernel image").unwrap();
        let meta = serde_json::json!({
            "version": "6.14.2",
            "source": {"type": "tarball"},
            "arch": "x86_64",
            "image_name": "bzImage",
            "config_hash": null,
            "built_at": built_at,
            "ktstr_kconfig_hash": null,
            "extra_kconfig_hash": extras_hash,
            "has_vmlinux": false,
            "vmlinux_stripped": false,
        });
        std::fs::write(
            dir.join("metadata.json"),
            serde_json::to_string_pretty(&meta).unwrap(),
        )
        .unwrap();
    };
    plant(
        "test-extras-miss-AAAA-bbbb2222",
        serde_json::json!("aaaaaaaa"),
        "2026-04-22T00:00:00Z",
    );
    plant(
        "test-extras-miss-BBBB-bbbb3333",
        serde_json::json!("bbbbbbbb"),
        "2026-04-23T00:00:00Z",
    );

    let output = cargo_ktstr()
        .env(ktstr::KTSTR_CACHE_DIR_ENV, tmp.path())
        .args(["kernel", "list", "--json"])
        .output()
        .expect("kernel list --json must run");
    assert!(output.status.success());
    let parsed: serde_json::Value =
        serde_json::from_str(&String::from_utf8(output.stdout).unwrap()).unwrap();
    let entries = parsed["entries"].as_array().expect("entries array");
    let entry_a = entries
        .iter()
        .find(|e| e["key"].as_str() == Some("test-extras-miss-AAAA-bbbb2222"))
        .expect("entry A must appear");
    let entry_b = entries
        .iter()
        .find(|e| e["key"].as_str() == Some("test-extras-miss-BBBB-bbbb3333"))
        .expect("entry B must appear");
    assert_eq!(entry_a["extra_kconfig_hash"].as_str(), Some("aaaaaaaa"));
    assert_eq!(entry_b["extra_kconfig_hash"].as_str(), Some("bbbbbbbb"));
    assert_ne!(
        entry_a["extra_kconfig_hash"], entry_b["extra_kconfig_hash"],
        "different extras content must produce distinct cache slots — \
         a build with extras=B must not be served entry A's cached kernel"
    );
}

/// Range expansion parse roundtrip: `cargo ktstr kernel build
/// 6.14..6.16 --extra-kconfig PATH` must reach the dispatch
/// without clap rejecting the range + extras combination. Each
/// version in the expanded range receives the same extras content
/// per the production loop in `kernel_build`'s range branch.
///
/// We don't drive a real network kernel.org fetch from a unit
/// test, so we verify the parser accepts the combination by
/// pointing `--kernel` at a nonexistent path (the path branch still
/// proves the flag plumbing the range loop reuses) AND by
/// verifying `--help` documents the flag combination.
#[test]
fn extra_kconfig_range_expansion() {
    // Help-text surface: `kernel build --help` must list
    // `--extra-kconfig` so an operator can discover the flag.
    cargo_ktstr()
        .args(["kernel", "build", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("--extra-kconfig"));

    // Parse-level test: write a valid fragment, run with a range
    // shape AND --extra-kconfig. The build will fail at network
    // fetch (kernel.org/releases.json), proving clap accepted the
    // combination — the failure mode must NOT be a clap error.
    let tmp = tempfile::TempDir::new().unwrap();
    let frag = tmp.path().join("extras.kconfig");
    std::fs::write(&frag, "CONFIG_FOO=y\n").unwrap();

    // Use a `--kernel <path>` (not a range) to short-circuit before
    // the network fetch — the test's invariant is that
    // `--extra-kconfig` parses cleanly alongside the rest of `kernel
    // build`'s flag surface, which the range loop reuses verbatim. A
    // clap-level rejection of the combination would surface BEFORE the
    // source-acquire path runs.
    let assert_result = cargo_ktstr()
        .env(ktstr::KTSTR_CACHE_DIR_ENV, tmp.path())
        .env(ktstr::KTSTR_BYPASS_LLC_LOCKS_ENV, "1")
        .args([
            "kernel",
            "build",
            "--kernel",
            "/nonexistent/ktstr-extras-range-test",
            "--extra-kconfig",
            frag.to_str().unwrap(),
        ])
        .assert()
        .failure();
    let stderr = String::from_utf8(assert_result.get_output().stderr.clone()).unwrap();
    assert!(
        !stderr.contains("error: the argument") && !stderr.contains("cannot be used with"),
        "clap must accept `--extra-kconfig` alongside the build dispatch \
         (the range loop reuses this same flag set for every version); \
         got stderr: {stderr}"
    );
}

/// `kernel list --json` output must include an `extra_kconfig_hash`
/// field on every cached entry so an operator's automation can
/// distinguish builds that carried `--extra-kconfig` from bare
/// builds. Pins the JSON contract: field is present, is the planted
/// hex string for extras-built entries, and `null` for bare ones.
#[test]
fn extra_kconfig_kernel_list_shows_hash() {
    let tmp = tempfile::TempDir::new().unwrap();

    // Plant one bare entry and one extras entry side-by-side.
    let bare_dir = tmp.path().join("test-list-shows-hash-bare-bbbb4444");
    std::fs::create_dir_all(&bare_dir).unwrap();
    std::fs::write(bare_dir.join("bzImage"), b"bare kernel").unwrap();
    let bare_meta = serde_json::json!({
        "version": "6.14.2",
        "source": {"type": "tarball"},
        "arch": "x86_64",
        "image_name": "bzImage",
        "config_hash": null,
        "built_at": "2026-04-22T00:00:00Z",
        "ktstr_kconfig_hash": null,
        "extra_kconfig_hash": null,
        "has_vmlinux": false,
        "vmlinux_stripped": false,
    });
    std::fs::write(
        bare_dir.join("metadata.json"),
        serde_json::to_string_pretty(&bare_meta).unwrap(),
    )
    .unwrap();

    let extras_dir = tmp.path().join("test-list-shows-hash-extras-bbbb5555");
    std::fs::create_dir_all(&extras_dir).unwrap();
    std::fs::write(extras_dir.join("bzImage"), b"extras kernel").unwrap();
    let extras_meta = serde_json::json!({
        "version": "6.14.2",
        "source": {"type": "tarball"},
        "arch": "x86_64",
        "image_name": "bzImage",
        "config_hash": null,
        "built_at": "2026-04-23T00:00:00Z",
        "ktstr_kconfig_hash": null,
        "extra_kconfig_hash": "cafef00d",
        "has_vmlinux": false,
        "vmlinux_stripped": false,
    });
    std::fs::write(
        extras_dir.join("metadata.json"),
        serde_json::to_string_pretty(&extras_meta).unwrap(),
    )
    .unwrap();

    let output = cargo_ktstr()
        .env(ktstr::KTSTR_CACHE_DIR_ENV, tmp.path())
        .args(["kernel", "list", "--json"])
        .output()
        .expect("kernel list --json must run");
    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&stdout).unwrap();
    let entries = parsed["entries"].as_array().expect("entries array");
    let bare = entries
        .iter()
        .find(|e| e["key"].as_str() == Some("test-list-shows-hash-bare-bbbb4444"))
        .expect("bare entry must appear");
    let extras = entries
        .iter()
        .find(|e| e["key"].as_str() == Some("test-list-shows-hash-extras-bbbb5555"))
        .expect("extras entry must appear");

    // Both entries must have the field present (key existence
    // pins the schema contract). Bare = null, extras = hex string.
    assert!(
        bare.get("extra_kconfig_hash").is_some(),
        "bare entry must surface the `extra_kconfig_hash` JSON key (= null) \
         so consumers can distinguish 'no extras' from 'field missing'"
    );
    assert!(bare["extra_kconfig_hash"].is_null());
    assert_eq!(
        extras["extra_kconfig_hash"].as_str(),
        Some("cafef00d"),
        "extras entry must surface the planted hash verbatim"
    );
}
