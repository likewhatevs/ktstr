//! End-to-end: `cargo ktstr shell --test <name> --exec "..."`
//! boots a VM with the named test's topology + memory, executes
//! the command, exits cleanly.
//!
//! Locks in the `--test` arg's full chain:
//! 1. CLI parses `--test <name>` (`Shell.test: Option<String>`).
//! 2. cargo-ktstr probes every test binary under
//!    `cargo build --tests` for a `KtstrTestEntry::name == <NAME>`.
//! 3. The discovered entry's topology axes (numa_nodes, llcs,
//!    cores, threads), memory_mib (with wprof floor if
//!    applicable), and `extra_include_files` flow into the shell
//!    VM builder.
//! 4. The VM boots with that topology, runs `--exec`, exits.
//!
//! Without the wiring, the test fails one of three ways:
//! - probe fails to discover the named test (exit non-zero with
//!   diagnostic naming the test name)
//! - VM boots with default topology instead of the test's
//!   (verifiable by counting CPUs via `nproc` inside the guest)
//! - VM never reaches the `--exec` command
//!
//! Runs on the self-hosted CI runners (`[ktstr-x64]` /
//! `[ktstr-arm64]`). ktstr supplies the guest kernel itself via
//! its kernel-build cache; cargo-ktstr is built during the CI
//! test step.

#![cfg(unix)]

mod common;

use common::cargo_ktstr_subprocess::{CARGO_KTSTR_BINARY, run_cargo_ktstr_shell};

/// `cargo ktstr shell --test cgroup_ops_compose_in_real_vm
/// --exec "nproc"` boots a VM matching the test's topology
/// (`llcs=1, cores=2, threads=1` → 2 vCPUs), runs `nproc`,
/// stdout shows `2`, exit zero.
#[test]
fn shell_test_resolves_topology_from_named_test() {
    let output = run_cargo_ktstr_shell("cgroup_ops_compose_in_real_vm", "nproc");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "cargo ktstr shell --test exited non-zero (exit={:?})\n\
         STDOUT:\n{stdout}\n\nSTDERR:\n{stderr}",
        output.status.code(),
    );

    // Banner assertion: the test-resolution path prints a one-line
    // header to stderr BEFORE VM boot naming the test, scheduler,
    // memory, topology, and include-source counts. Pins the
    // operator-facing format so a regression that drops the banner
    // (or its `includes=test:M+cli:N` suffix) surfaces here. The
    // two `--exec` args produce stdout (nproc result);
    // banner stays on stderr.
    assert!(
        stderr.contains("ktstr shell: test=cgroup_ops_compose_in_real_vm"),
        "banner must name the resolved test on stderr; got:\n{stderr}",
    );
    assert!(
        stderr.contains("includes=test:") && stderr.contains("+cli:"),
        "banner must surface the include-source counts \
         (test:M+cli:N); got:\n{stderr}",
    );

    // Topology assertion: cgroup_ops_compose_in_real_vm declares
    // llcs=1, cores=2, threads=1 → 2 vCPUs. `nproc` in the guest
    // must report exactly 2. A default-topology boot would report
    // 1 (the cli default "1,1,1,1").
    let nproc_line = stdout
        .lines()
        .find(|l| !l.trim().is_empty())
        .expect("guest nproc produced empty stdout");
    let cpu_count: u32 = nproc_line
        .trim()
        .parse()
        .unwrap_or_else(|_| panic!("guest nproc output {nproc_line:?} not a number"));
    assert_eq!(
        cpu_count, 2,
        "shell --test did NOT inherit cgroup_ops_compose_in_real_vm's \
         topology (llcs=1 × cores=2 × threads=1 = 2 vCPUs); guest \
         nproc reported {cpu_count}. The probe → topology-resolve → \
         builder chain is broken — `nproc=1` means a default \
         topology took effect."
    );
}

/// `cargo ktstr shell --test <unknown>` must bail with a
/// diagnostic naming the unknown test name. Pins the operator's
/// error path — a silent fallback to default topology would hide
/// a typo in the test name.
#[test]
fn shell_test_unknown_test_bails_with_actionable_diagnostic() {
    // Uses raw `Command::new` rather than `run_cargo_ktstr_shell`
    // because the unknown-test bail fires before any kernel-resolve
    // path — no `--kernel` flag needed and no VM boot to gate on.
    let output = std::process::Command::new(CARGO_KTSTR_BINARY)
        .arg("ktstr")
        .arg("shell")
        .arg("--no-perf-mode")
        .arg("--test")
        .arg("nonexistent_test_name_xyzzy_42")
        .arg("--exec")
        .arg("true")
        .output()
        .expect("spawn cargo-ktstr shell --test");

    assert!(
        !output.status.success(),
        "unknown --test must fail; got exit {:?}",
        output.status.code(),
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("nonexistent_test_name_xyzzy_42"),
        "diagnostic must cite the unknown test name so the operator \
         sees their typo: stderr=\n{stderr}",
    );
}
