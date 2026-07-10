use anyhow::Result;
use ktstr::assert::AssertResult;
use ktstr::prelude::VmResult;
use ktstr::scenario::Ctx;
use ktstr::scenario::ops::{CgroupDef, HoldSpec, Step, execute_steps_with};
use ktstr::test_support::{KtstrTestEntry, Scheduler, SchedulerSpec};

/// Build a scheduler package and resolve paths for verifier tests.
/// Returns `Ok(None)` when no kernel is available (CI without a custom
/// kernel) — callers should skip the test, not fail.
fn resolve_verifier_paths(
    package: &str,
) -> Result<Option<(std::path::PathBuf, std::path::PathBuf, std::path::PathBuf)>> {
    let Some(kernel) = ktstr::find_kernel()? else {
        return Ok(None);
    };
    let sched_bin = ktstr::build_and_find_binary(package, env!("CARGO_MANIFEST_DIR"))?;
    let ktstr_bin = std::env::current_exe()?;
    Ok(Some((sched_bin, ktstr_bin, kernel)))
}

fn __ktstr_inner_demo_verifier_brief(_ctx: &Ctx) -> Result<AssertResult> {
    let Some((sched_bin, ktstr_bin, kernel)) = resolve_verifier_paths("scx-ktstr")? else {
        return Ok(AssertResult::pass());
    };
    let result = ktstr::verifier::collect_verifier_output(
        &sched_bin,
        &ktstr_bin,
        &kernel,
        &[],
        ktstr::test_support::TopologyJson::SINGLE_CPU,
    )?;
    let output = ktstr::verifier::format_verifier_output("scx-ktstr", &result, false);
    anyhow::ensure!(
        output.contains("ktstr_enqueue"),
        "output should list ktstr_enqueue"
    );
    anyhow::ensure!(
        output.contains("ktstr_dispatch"),
        "output should list ktstr_dispatch"
    );
    anyhow::ensure!(
        output.contains("verified_insns="),
        "output should contain verified_insns="
    );
    Ok(AssertResult::pass())
}

#[ktstr::distributed_slice(ktstr::test_support::KTSTR_TESTS)]
#[linkme(crate = ktstr::linkme)]
static __KTSTR_ENTRY_VERIFIER_BRIEF: KtstrTestEntry = KtstrTestEntry {
    name: "demo_verifier_brief",
    func: __ktstr_inner_demo_verifier_brief,
    auto_repro: false,
    host_only: true,
    ..KtstrTestEntry::DEFAULT
};

fn __ktstr_inner_demo_verifier_diff(_ctx: &Ctx) -> Result<AssertResult> {
    let Some((sched_bin, ktstr_bin, kernel)) = resolve_verifier_paths("scx-ktstr")? else {
        return Ok(AssertResult::pass());
    };
    let result_a = ktstr::verifier::collect_verifier_output(
        &sched_bin,
        &ktstr_bin,
        &kernel,
        &[],
        ktstr::test_support::TopologyJson::SINGLE_CPU,
    )?;
    let result_b = ktstr::verifier::collect_verifier_output(
        &sched_bin,
        &ktstr_bin,
        &kernel,
        &[],
        ktstr::test_support::TopologyJson::SINGLE_CPU,
    )?;
    let output = ktstr::verifier::format_verifier_diff(
        "scx-ktstr",
        &result_a.stats,
        "scx-ktstr",
        &result_b.stats,
    );
    anyhow::ensure!(
        output.contains("delta"),
        "diff output should contain 'delta' header"
    );
    anyhow::ensure!(
        output.contains("program"),
        "diff output should contain 'program' column"
    );
    anyhow::ensure!(output.contains("+0"), "self-comparison deltas should be 0");
    Ok(AssertResult::pass())
}

#[ktstr::distributed_slice(ktstr::test_support::KTSTR_TESTS)]
#[linkme(crate = ktstr::linkme)]
static __KTSTR_ENTRY_VERIFIER_DIFF: KtstrTestEntry = KtstrTestEntry {
    name: "demo_verifier_diff",
    func: __ktstr_inner_demo_verifier_diff,
    auto_repro: false,
    host_only: true,
    ..KtstrTestEntry::DEFAULT
};

fn __ktstr_inner_verifier_cycle_collapse(_ctx: &Ctx) -> Result<AssertResult> {
    let Some((sched_bin, ktstr_bin, kernel)) = resolve_verifier_paths("scx-ktstr")? else {
        return Ok(AssertResult::pass());
    };
    let sched_args = vec!["--verify-loop".to_string()];
    let result = ktstr::verifier::collect_verifier_output(
        &sched_bin,
        &ktstr_bin,
        &kernel,
        &sched_args,
        ktstr::test_support::TopologyJson::SINGLE_CPU,
    )?;
    // The test scheduler (like most scx schedulers) logs through the
    // log crate, so the libbpf verifier trace lands on the live STDERR
    // stream: the cycle collapse must appear in the stderr rendering,
    // and the stdout rendering must NOT re-print the merged teardown
    // dump (that would duplicate the stderr section).
    let output = ktstr::verifier::format_verifier_output("scx-ktstr", &result, false);
    let stderr_out = ktstr::verifier::format_verifier_stderr("scx-ktstr", &result, false);
    anyhow::ensure!(
        stderr_out.contains("scheduler stderr"),
        "stderr rendering should contain the scheduler stderr section"
    );
    anyhow::ensure!(
        stderr_out.contains("identical iterations omitted"),
        "cycle collapse should compress verifier loop traces"
    );
    anyhow::ensure!(
        !output.contains("scheduler log"),
        "stdout rendering must not duplicate the merged dump when live \
         stderr arrived; got:\n{output}"
    );
    Ok(AssertResult::pass())
}

#[ktstr::distributed_slice(ktstr::test_support::KTSTR_TESTS)]
#[linkme(crate = ktstr::linkme)]
static __KTSTR_ENTRY_CYCLE_COLLAPSE: KtstrTestEntry = KtstrTestEntry {
    name: "verifier_cycle_collapse",
    func: __ktstr_inner_verifier_cycle_collapse,
    auto_repro: false,
    host_only: true,
    ..KtstrTestEntry::DEFAULT
};

// -- verifier BPF-load-rejection scenarios: --fail-verify and --verify-loop --

const FAIL_SCHED: Scheduler =
    Scheduler::named("ktstr_sched").binary(SchedulerSpec::Discover("scx-ktstr"));

fn scenario_fail_verify(ctx: &Ctx) -> Result<AssertResult> {
    let steps = vec![Step {
        setup: vec![CgroupDef::named("cg_0").workers(2)].into(),
        ops: vec![],
        hold: HoldSpec::FULL,
    }];
    execute_steps_with(ctx, steps, None)
}

/// Pin the SPECIFIC verifier rejection these cells demonstrate. Both
/// --fail-verify and --verify-loop make the BPF verifier reject
/// ktstr_dispatch, so libbpf wraps the kernel verifier trace in
/// `-- BEGIN PROG LOAD LOG --` in the scheduler's captured log. Asserting
/// that marker proves the reject actually happened (not merely "some
/// failure"), catching a changed failure mode. Wired as
/// `post_vm_unconditional` so it runs even though these cells `expect_err`
/// -- that hook bypasses the guest-fail suppression, and an Err here is a
/// hard failure `expect_err` does not invert.
fn assert_verifier_rejected(result: &VmResult) -> Result<()> {
    let log = result.scheduler_log();
    anyhow::ensure!(
        log.contains("-- BEGIN PROG LOAD LOG --"),
        "scheduler log missing the libbpf verifier-reject marker \
         `-- BEGIN PROG LOAD LOG --`; got:\n{log}"
    );
    Ok(())
}

#[ktstr::distributed_slice(ktstr::test_support::KTSTR_TESTS)]
#[linkme(crate = ktstr::linkme)]
static __KTSTR_ENTRY_FAIL_VERIFY: KtstrTestEntry = KtstrTestEntry {
    name: "demo_verifier_fail_verify",
    func: scenario_fail_verify,
    scheduler: &FAIL_SCHED,
    extra_sched_args: &["--fail-verify"],
    post_vm_unconditional: Some(assert_verifier_rejected),
    duration: std::time::Duration::from_secs(5),
    // The scheduler deliberately fails to load its BPF (--fail-verify
    // injects a verifier-rejected null store), so the guest diagnoses a
    // BPF-load rejection and never dispatches the workload -- that load
    // failure is the EXPECTED outcome of this demonstration cell. Invert
    // the guest-side failure verdict to PASS; otherwise the cell
    // hard-fails whenever it actually runs (it previously only "passed"
    // by skipping under LLC-lock contention, masking the hard failure).
    expect_err: true,
    // No auto-repro: the BPF-load rejection is the EXPECTED outcome, so
    // reproducing it in a second probe VM is wasted work -- and on cold
    // topos that extra boot compounded the timing pressure that first
    // surfaced this cell's flake.
    auto_repro: false,
    ..KtstrTestEntry::DEFAULT
};

#[ktstr::distributed_slice(ktstr::test_support::KTSTR_TESTS)]
#[linkme(crate = ktstr::linkme)]
static __KTSTR_ENTRY_VERIFY_REJECT: KtstrTestEntry = KtstrTestEntry {
    name: "demo_verifier_cycle_collapse",
    func: scenario_fail_verify,
    scheduler: &FAIL_SCHED,
    extra_sched_args: &["--verify-loop"],
    post_vm_unconditional: Some(assert_verifier_rejected),
    duration: std::time::Duration::from_secs(5),
    // Same expected-outcome as the --fail-verify sibling: --verify-loop
    // makes the BPF verifier reject ktstr_dispatch (an unrolled loop
    // then a verifier-rejected null store), so the scheduler never binds
    // and the guest diagnoses a BPF-load rejection. That load failure is
    // the EXPECTED outcome, so invert the guest-side failure verdict to
    // PASS. The slow reject also crosses the 1s liveness gate on cold
    // topos (StartupDied->NotAttached), but both frames now invert
    // identically, so the timing no longer flips the verdict.
    expect_err: true,
    // No auto-repro: the BPF-load rejection is the EXPECTED outcome, so
    // reproducing it in a second probe VM is wasted work -- and on cold
    // topos that extra boot compounded the timing pressure that first
    // surfaced this cell's flake.
    auto_repro: false,
    ..KtstrTestEntry::DEFAULT
};
