//! VM-backed end-to-end test for the placement-before-exec
//! invariant on the `Op::RunPayload` cgroup-placement path.
//!
//! ## Invariant
//!
//! [`CgroupOps::place_task_during_handshake`](ktstr::cgroup::CgroupOps::place_task_during_handshake)
//! guarantees that when a payload spawn carries an explicit cgroup
//! target, the framework writes the child's pid into the destination
//! cgroup's `cgroup.procs` BEFORE releasing the child from its
//! pre-exec pause, so the child's `execve(2)` lands inside the
//! destination cgroup. The first scheduled userspace instruction
//! of the spawned binary is therefore observably IN the
//! destination cgroup — never in the runner's own cgroup.
//!
//! Without the placement-during-handshake step, the child
//! would `execve` while still in its parent's cgroup; a
//! post-spawn `cgroup.procs` write would only migrate the task
//! AFTER it had executed an arbitrary amount of code. Tests
//! that assume "the entire payload ran in cg_dst" would silently
//! observe partial coverage — schedstat counters,
//! BPF-trace points, and other scheduler observables for the
//! pre-migration window would show the runner's cgroup, not the
//! destination cgroup.
//!
//! ## Strategy
//!
//! Spawn a `/bin/sh -c` payload into `cg_dst` whose first
//! action is to read `/proc/self/cgroup` and persist it to a
//! tmpfs path the test body can read after the scenario
//! drains. Because the framework guarantees the child's
//! `execve` lands in `cg_dst`, the very first userspace read
//! of `/proc/self/cgroup` observes the destination cgroup.
//!
//! The payload also persists its own pid so a regression that
//! degraded the handshake to "spawn-then-move" would produce
//! a `/proc/self/cgroup` line referencing the runner's cgroup
//! at the moment of the read while the post-spawn move would
//! later place the task in `cg_dst`. The recorded snapshot
//! captures the pre-move state — exactly the kernel state the
//! placement-before-exec invariant guarantees.
//!
//! ## Shell payload provisioning
//!
//! `#[ktstr_test]`'s initramfs ships only the test binary as
//! `/init`; there is no default `/bin/sh`. The test packs the
//! host's `/bin/sh` AND `/bin/cat` via `extra_include_files`,
//! landing at `/include-files/sh` + `/include-files/cat` inside
//! the guest. `/include-files` is prepended to the guest's
//! `PATH` by [`build_include_path`](crate::vmm::rust_init), so
//! [`Payload::binary`](ktstr::Payload::binary) with `binary =
//! "sh"` resolves cleanly via `Command::new("sh")`'s PATH
//! lookup, and the inline `cat /proc/self/cgroup` invocation
//! inside the probe script resolves via sh's own PATH lookup.
//! The host-side `/bin/sh` is present on every FHS-compliant
//! Linux distro (typically a symlink to `dash`, `bash`, or
//! `busybox`). Without `cat` packed, the `>` redirect would
//! still create an empty output file but `cat` would fail with
//! exit 127, surfacing the regression as a harder-to-diagnose
//! empty-file read in the test body rather than a clean
//! missing-binary diagnostic.

use anyhow::Result;
use ktstr::assert::{AssertDetail, AssertResult, DetailKind};
use ktstr::ktstr_test;
use ktstr::prelude::{Backdrop, CgroupDef, Payload, Scheduler, SchedulerSpec};
use ktstr::scenario::Ctx;
use ktstr::scenario::ops::{HoldSpec, Op, Step, execute_scenario};
use std::time::Duration;

const KTSTR_SCHED: Scheduler =
    Scheduler::named("ktstr_sched").binary(SchedulerSpec::Discover("scx-ktstr"));

/// Inline shell payload. `binary = "sh"` is resolved via the
/// guest's `PATH` against `/include-files/sh` (provisioned by
/// the `extra_include_files` attribute below). The framework's
/// `Op::RunPayload` dispatcher routes the spawn through
/// `CgroupOps::place_task_during_handshake` when an explicit
/// cgroup placement is supplied, so the resulting `execve` lands
/// inside the destination cgroup before any userspace
/// instruction runs.
const SHELL_PROBE: Payload = Payload::binary("cgroup_placement_probe", "sh");

/// Tmpfs path the payload writes its own `/proc/self/cgroup`
/// contents to. The test body reads this file after the
/// scenario drains, comparing the recorded cgroup line against
/// the expected `cg_dst` placement.
const CGROUP_SNAPSHOT_PATH: &str = "/tmp/ktstr-placement-check-cgroup";

/// Sibling tmpfs path that the payload also writes its own pid
/// to. Not required for the placement assertion itself but
/// included so a debugging operator can correlate a failing
/// `/proc/self/cgroup` snapshot back to a specific guest-side
/// pid in the same VM run.
const PID_SNAPSHOT_PATH: &str = "/tmp/ktstr-placement-check-pid";

#[ktstr_test(
    scheduler = KTSTR_SCHED,
    workload_root_cgroup = "/ktstr-placement-pre-exec-e2e",
    extra_include_files = ["/bin/sh", "/bin/cat"],
    llcs = 1,
    cores = 2,
    threads = 1,
    memory_mib = 256,
    duration_s = 3,
    watchdog_timeout_s = 30,
    auto_repro = false,
)]
fn op_runpayload_places_child_in_cgroup_before_exec(ctx: &Ctx) -> Result<AssertResult> {
    // Empty cg_dst lives on the Backdrop so it survives the
    // step boundary. `Op::add_cgroup` (rather than a CgroupDef)
    // because the test never spawns workers into the cgroup —
    // the only occupant is the shell-probe payload, whose
    // placement is the focus of the assertion.
    let backdrop = Backdrop::new().push_op(Op::add_cgroup("cg_dst"));
    // The shell snippet runs as the payload's `execve`d body.
    // First action: `cat /proc/self/cgroup > /tmp/...` —
    // captures the cgroup membership of the just-`execve`d
    // process. If the framework's handshake correctly placed
    // the child in `cg_dst` BEFORE releasing it to `execve`,
    // the recorded line names cg_dst's full path. A regression
    // that placed the child AFTER `execve` (or never) would
    // record the runner's cgroup at the moment of the read.
    //
    // `echo $$` writes the payload's own pid before the cgroup
    // read so a failed snapshot can be cross-referenced to a
    // concrete guest-side pid for debugging — orthogonal to the
    // assertion below.
    let probe_script = format!(
        "echo $$ > {pid_path} && cat /proc/self/cgroup > {cgroup_path}",
        pid_path = PID_SNAPSHOT_PATH,
        cgroup_path = CGROUP_SNAPSHOT_PATH,
    );
    let steps = vec![Step {
        setup: Vec::<CgroupDef>::new().into(),
        ops: vec![
            Op::run_payload_in_cgroup(&SHELL_PROBE, vec!["-c".to_string(), probe_script], "cg_dst"),
            // Block until the probe exits so the snapshot file
            // is fully written by the time the test body reads
            // it. `Op::wait_payload` is event-driven (waits on
            // the payload's pid); no sleep involved.
            Op::wait_payload(SHELL_PROBE.name),
        ],
        // `Op::wait_payload` already provides synchronous
        // gating on the probe; the step-level hold has no
        // additional work to do. ZERO valid per `HoldSpec::validate`.
        hold: HoldSpec::fixed(Duration::ZERO),
    }];
    let _ = execute_scenario(ctx, backdrop, steps)?;

    // Read the cgroup snapshot the payload recorded. The file
    // must exist (payload spawned + waited successfully) and
    // its sole line must equal `0::<expected_suffix>` exactly —
    // proving the payload's first userspace read of
    // `/proc/self/cgroup` observed it inside cg_dst, not in
    // some other (possibly nested) cgroup.
    let snapshot = match std::fs::read_to_string(CGROUP_SNAPSHOT_PATH) {
        Ok(s) => s,
        Err(e) => {
            return Ok(AssertResult::fail(AssertDetail::new(
                DetailKind::Other,
                format!(
                    "read {CGROUP_SNAPSHOT_PATH}: {e}. The shell probe \
                     payload was expected to write its `/proc/self/cgroup` \
                     contents to this path before exiting; a missing file \
                     means either the payload never ran (spawn failure) or \
                     the `Op::wait_payload` returned before the shell's \
                     redirect completed."
                ),
            )));
        }
    };
    // cgroup v2 renders /proc/<pid>/cgroup as a single line
    // `0::/path/to/cgroup` (the `0` hierarchy id and the
    // empty controller list are constants for the unified
    // hierarchy). Trim trailing newline before testing.
    let line = snapshot.trim();
    // Build the expected line: literal `0::` prefix (the
    // cgroup-v2 hierarchy id + empty-controller marker that
    // `cgroup_show_path` in kernel/cgroup/cgroup.c always
    // emits) followed by the test's declared
    // `workload_root_cgroup` + `/cg_dst`. `parent_path()` is
    // rooted under `/sys/fs/cgroup` for the guest-VM context
    // this test runs in (Mode A — no cgroup-v2 delegation
    // walk_root override), so stripping the mount prefix
    // yields the kernel's relative-path rendering as it
    // appears in `/proc/self/cgroup` (`0::/path`). A regression
    // that placed the task in a different cgroup (e.g., a
    // nested `cg_dst/inner/cg_dst`) would fail the equality
    // check; a regression that placed the task in the runner's
    // cgroup would yield a different prefix entirely.
    let relative = ctx
        .cgroups
        .parent_path()
        .strip_prefix("/sys/fs/cgroup")
        .expect(
            "workload_root_cgroup parent_path lives under /sys/fs/cgroup in \
             the guest-VM context this test runs in (Mode A — no cgroup-v2 \
             delegation walk_root override)",
        );
    let expected_line = format!("0::/{}/cg_dst", relative.display());
    if line != expected_line {
        return Ok(AssertResult::fail(AssertDetail::new(
            DetailKind::Other,
            format!(
                "{CGROUP_SNAPSHOT_PATH} contents = {line:?}; expected \
                 exactly {expected_line:?} (proving the payload's first \
                 userspace read of `/proc/self/cgroup` observed it inside \
                 cg_dst). A different line means \
                 `CgroupOps::place_task_during_handshake` either did not \
                 fire or fired after the child's `execve` — the \
                 placement-before-exec invariant has regressed and the \
                 spawn is no longer placement-before-exec."
            ),
        )));
    }
    Ok(AssertResult::pass())
}
