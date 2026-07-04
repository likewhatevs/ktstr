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

use anyhow::Result;
use common::cargo_ktstr_subprocess::{CARGO_KTSTR_BINARY, run_cargo_ktstr_shell};
use ktstr::assert::AssertResult;
use ktstr::ktstr_test;
use ktstr::prelude::{Scheduler, SchedulerSpec};
use ktstr::scenario::Ctx;

// Local fixture for `shell_test_resolves_topology_from_named_test`. The
// e2e harness sets `KTSTR_TEST_BINARY=current_exe` (the capture speedup
// that avoids a cold `cargo build --tests` nextest SIGTERMs mid-compile),
// so `--test` resolution searches ONLY this binary — the resolved fixture
// must live HERE, not in a sibling test binary. `llcs=1 × cores=2 ×
// threads=1 = 2 vCPUs` so the guest `nproc` is a non-default 2. `ignore`
// keeps it out of regular nextest runs while staying discoverable by
// `cargo ktstr shell --test` (the shell-mode dispatcher filters
// `host_only`, never `is_ignored`). The cross-binary probe path (probing
// sibling test binaries) is covered by probe.rs unit tests
// (`probe_collect_with_bins_*`) + the unknown-bail test below.
const SHELL_TEST_TOPO_FIXTURE: Scheduler =
    Scheduler::named("shell_test_topo_fixture").binary(SchedulerSpec::KernelBuiltin {
        enable: &[],
        disable: &[],
    });

#[ktstr_test(
    scheduler = SHELL_TEST_TOPO_FIXTURE,
    llcs = 1,
    cores = 2,
    threads = 1,
    memory_mib = 128,
    duration_s = 1,
    watchdog_timeout_s = 10,
    auto_repro = false,
    ignore,
)]
fn shell_test_topo_fixture(_ctx: &Ctx) -> Result<AssertResult> {
    Ok(AssertResult::pass())
}

/// `cargo ktstr shell --test shell_test_topo_fixture
/// --exec "nproc"` boots a VM matching the test's topology
/// (`llcs=1, cores=2, threads=1` → 2 vCPUs), runs `nproc`,
/// stdout shows `2`, exit zero.
#[test]
fn shell_test_resolves_topology_from_named_test() {
    let output = run_cargo_ktstr_shell("shell_test_topo_fixture", "nproc");

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
        stderr.contains("ktstr shell: test=shell_test_topo_fixture"),
        "banner must name the resolved test on stderr; got:\n{stderr}",
    );
    assert!(
        stderr.contains("includes=test:") && stderr.contains("+cli:"),
        "banner must surface the include-source counts \
         (test:M+cli:N); got:\n{stderr}",
    );

    // Topology assertion: shell_test_topo_fixture declares
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
        "shell --test did NOT inherit shell_test_topo_fixture's \
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
    // so the `--exec` payload is `true` (a no-op) and the `--test`
    // name is a fixed nonexistent one. Kernel resolution
    // (run_shell → resolve_kernel_image, shell.rs:201) runs BEFORE
    // the unknown-test probe (resolve_shell_from_test_entry,
    // shell.rs:233); with no `--kernel` passed it resolves via the
    // runner's kernel cache (find_kernel, resolve.rs:435), so the
    // run reaches the probe and the unknown name bails — which is
    // what this test pins. No VM boots because the bail precedes it.
    let output = std::process::Command::new(CARGO_KTSTR_BINARY)
        .arg("ktstr")
        .arg("shell")
        .arg("--no-perf-mode")
        .arg("--test")
        .arg("nonexistent_test_name_xyzzy_42")
        .arg("--exec")
        .arg("true")
        // Reuse this test binary for resolution (same speedup as
        // run_cargo_ktstr_shell): without it the unknown-test probe
        // triggers a cold `cargo build --tests` that nextest SIGTERMs
        // mid-compile. The unknown name is absent from this binary too,
        // so the bail still fires — exactly what this test pins.
        .env(
            "KTSTR_TEST_BINARY",
            std::env::current_exe().expect("current_exe for KTSTR_TEST_BINARY"),
        )
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
