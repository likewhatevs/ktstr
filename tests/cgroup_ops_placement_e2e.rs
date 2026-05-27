//! VM-backed end-to-end test for the apply_ops cgroup-placement
//! dispatch path: `Op::Spawn` (with both `SpawnPlacement::Cgroup`
//! and `SpawnPlacement::RunnerCgroup`) and `Op::MoveAllTasks`.
//!
//! The MockCgroupOps unit tests in `src/scenario/ops/mod.rs`'s
//! `mod tests` (`op_spawn_cgroup_moves_tasks_into_named_cgroup`,
//! `op_spawn_runner_cgroup_emits_zero_cgroup_calls`,
//! `op_move_all_tasks_clears_subtree_control_then_moves_to_dst`,
//! `op_spawn_cgroup_after_addcgroupdef_sets_cpuset_before_move_tasks`)
//! pin the host-side dispatch call streams. This test pins the
//! end-to-end composition against a real cgroup hierarchy and a
//! real sched_ext scheduler — proving the three ops compose
//! successfully when the underlying cgroupfs ops actually fire
//! against `/sys/fs/cgroup/...` (not the mock).
//!
//! Composition: a setup pass declares `cg_src` (with 2 workers
//! from `CgroupDef.workers(2)`) and an empty destination
//! `cg_dst` on the [`Backdrop`]. The op sequence then:
//! 1. `Op::spawn_workers("cg_dst", ...)` — adds 1 worker into
//!    the named workload cgroup via the apply_ops path
//!    (separate code site from the apply_setup-driven worker in
//!    `cg_src`). The handle is step-local at spawn time but
//!    transfers into the Backdrop slot when the later
//!    `MoveAllTasks` re-keys it onto a Backdrop-owned cgroup
//!    (see `rename_handles` at `src/scenario/ops/mod.rs`).
//! 2. `Op::spawn_host(...)` — spawns 1 worker outside any
//!    managed workload cgroup (in the runner's own cgroup).
//! 3. `Op::move_all_tasks("cg_src", "cg_dst")` — migrates the
//!    2 workers from `cg_src` into `cg_dst`. After this op,
//!    `cg_dst` holds the 1 from step 1 plus the 2 from
//!    `cg_src`, and `cg_src` is empty.
//! 4. `Op::run_payload(SHELL_PROBE, ...)` — a shell probe that
//!    reads `cg_dst/cgroup.procs` AND each member's
//!    `/proc/<pid>/comm`, persisting both to tmpfs paths.
//!    Runs in the runner's own cgroup (the default for
//!    `Op::run_payload` with `cgroup = None`), so the probe's
//!    own pid does NOT join `cg_dst` and the expected pid count
//!    stays at 3. The probe must run INSIDE the scenario,
//!    because the per-scenario `BackdropState` (which owns
//!    `cg_dst` for the lifetime of the scenario) is dropped
//!    when `run_scenario` returns; that drop fires
//!    `CgroupGroup::drop`'s `remove_cgroup` walk, which
//!    rmdir's `cg_dst` before any post-`execute_scenario` read
//!    from the test body could observe its contents.
//! 5. `Op::wait_payload(SHELL_PROBE.name)` — blocks until the
//!    shell exits, so the tmpfs files are fully written before
//!    the scenario advances past this step.
//!
//! Bail in any of the three placement handlers (e.g. cgroup.procs
//! ENOENT from a stale cgroup name, EBUSY from a forgotten
//! clear_subtree_control, or migration policy failure) surfaces
//! as `execute_scenario` returning `Err`. The MockCgroupOps tests
//! cannot catch those — only a real VM with a real scheduler can.
//!
//! ## Identity assertion (catches false-positive count match)
//!
//! Each spawn carries a distinct `WorkSpec::comm`:
//!   * cg_src workers → `WORKER_COMM_SRC`
//!   * Op::Spawn(Cgroup) cg_dst worker → `WORKER_COMM_DST`
//!   * Op::Spawn(RunnerCgroup) worker → `WORKER_COMM_HOST`
//!
//! The probe captures `cg_dst/cgroup.procs` AND each member's
//! `/proc/<pid>/comm`. The test asserts:
//!
//!   * Exactly 3 pids in `cg_dst` (1 from Op::Spawn(Cgroup) into
//!     cg_dst + 2 from `MoveAllTasks cg_src→cg_dst`).
//!   * Comm MULTISET equals `{SRC × 2, DST × 1}` — proves the
//!     pids are the expected workers, NOT an arbitrary count
//!     match (e.g. Op::Spawn(RunnerCgroup) workers leaking into
//!     cg_dst would surface `HOST` in the multiset; a partial
//!     `MoveAllTasks` would yield fewer `SRC`).
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
//! lookup, and the inline `cat` invocations inside the probe
//! script resolve via sh's own PATH lookup. The probe script
//! invokes `cat` four times (lexemes): once to snapshot
//! `cg_dst/cgroup.procs` to tmpfs, once for `cg_src/cgroup.procs`
//! (the diagnostic supplementation that surfaces whether
//! `Op::move_all_tasks` migrated the cg_src workers), once
//! more to re-read `cg_dst/cgroup.procs` inside the for-loop
//! `$(...)` subshell to iterate the captured pids, and once
//! per pid to read `/proc/<pid>/comm` for the comm-multiset
//! identity assertion. `echo` and the `for ... in $(...)`
//! loop construct are shell builtins (no external binary
//! needed). Without `cat` packed, every invocation would fail
//! with exit 127 ("command not found"); the `>` redirect
//! still creates an empty output file (POSIX shell opens the
//! redirection target BEFORE executing the command), so the
//! test would surface as the harder-to-diagnose "0 pids in
//! cgroup.procs" assertion instead of a missing-binary
//! diagnostic.

use anyhow::Result;
use ktstr::assert::{AssertDetail, AssertResult, DetailKind};
use ktstr::ktstr_test;
use ktstr::prelude::{Backdrop, Payload, Scheduler, SchedulerSpec, WorkSpec, WorkType};
use ktstr::scenario::Ctx;
use ktstr::scenario::ops::{HoldSpec, Op, Step, execute_scenario};
use std::time::Duration;

const KTSTR_SCHED: Scheduler =
    Scheduler::named("ktstr_sched").binary(SchedulerSpec::Discover("scx-ktstr"));

/// Shell probe used to snapshot `cg_dst/cgroup.procs` + comms
/// inside the scenario. Resolved via the guest PATH against
/// `/include-files/sh` (provisioned by `extra_include_files`).
const SHELL_PROBE: Payload = Payload::binary("cgroup_ops_probe", "sh");

/// Tmpfs path the shell probe writes `cg_dst/cgroup.procs` to.
/// Persists past `BackdropState`'s end-of-scenario rmdir, so the
/// test body can read it after `execute_scenario` returns.
const PROCS_SNAPSHOT_PATH: &str = "/tmp/ktstr-cgroup-ops-procs";

/// Tmpfs path the probe writes each cg_dst member's
/// `/proc/<pid>/comm` to (one comm per line, in `cgroup.procs`
/// order). Used for the identity assertion below.
const COMMS_SNAPSHOT_PATH: &str = "/tmp/ktstr-cgroup-ops-comms";

/// Tmpfs path the probe writes `cg_src/cgroup.procs` to.
/// Diagnostic-only — surfaced in the failure message when the
/// `cg_dst` pid count is wrong, so the operator can distinguish
/// "Op::move_all_tasks did not migrate" (cg_src non-empty) from
/// "Op::spawn_workers / move_all_tasks both ran but cg_dst was
/// rmdir'd mid-read" (cg_src empty too) without re-running.
const SRC_PROCS_SNAPSHOT_PATH: &str = "/tmp/ktstr-cgroup-ops-src-procs";

/// Distinct `WorkSpec::comm` per spawn site so the identity
/// assertion can distinguish migrated cg_src workers from the
/// spawn-time cg_dst worker AND catch a RunnerCgroup-placement-leak
/// regression. All three are ≤ 15 bytes (TASK_COMM_LEN - 1)
/// per `validate_task_comm_string`.
const WORKER_COMM_SRC: &str = "ktstr_cgsrc";
const WORKER_COMM_DST: &str = "ktstr_cgdst";
const WORKER_COMM_HOST: &str = "ktstr_host";

#[ktstr_test(
    scheduler = KTSTR_SCHED,
    workload_root_cgroup = "/ktstr-cgroup-ops-e2e",
    extra_include_files = ["/bin/sh", "/bin/cat"],
    llcs = 1,
    cores = 2,
    threads = 1,
    memory_mib = 256,
    duration_s = 3,
    watchdog_timeout_s = 30,
    auto_repro = false,
)]
fn cgroup_ops_compose_in_real_vm(ctx: &Ctx) -> Result<AssertResult> {
    // Backdrop-declare cg_dst AND cg_src so both exist before any
    // Step::ops run. cg_dst is registered via `push_op(Op::add_cgroup)`
    // because it must start EMPTY — a CgroupDef would force a worker
    // spawn and corrupt the final pid count (the test exercises
    // spawn_workers + move_all_tasks placement, not a pre-populated
    // destination). cg_src is registered via `push_cgroup` so its 2
    // SpinWait workers are placed at backdrop setup time, before
    // any Step::ops fire. `run_step` (src/scenario/ops/mod.rs:1292)
    // executes ops BEFORE Step::setup for non-Loop hold modes; a
    // CgroupDef in Step::setup would mean MoveAllTasks fires with
    // cg_src unmigrated and missing on the cgroupfs, defeating the
    // test. Backdrop guarantees the setup-before-ops ordering this
    // test requires.
    let backdrop = Backdrop::new()
        .push_op(Op::add_cgroup("cg_dst"))
        .push_cgroup(
            ctx.cgroup_def("cg_src")
                .workers(2)
                .comm(WORKER_COMM_SRC)
                .work_type(WorkType::SpinWait),
        );
    // Path the shell probe reads. Derived from the test's
    // `workload_root_cgroup` so the assertion tracks whatever the
    // attribute declares; hard-coding `/sys/fs/cgroup/ktstr-cgroup-ops-e2e`
    // would drift silently if the attribute changed.
    let cg_dst_procs = ctx
        .cgroups
        .parent_path()
        .join("cg_dst")
        .join("cgroup.procs");
    let cg_src_procs = ctx
        .cgroups
        .parent_path()
        .join("cg_src")
        .join("cgroup.procs");
    // Shell snippet: snapshot cg_dst.procs + cg_src.procs (the
    // latter is diagnostic-only — surfaced in the failure
    // message when the cg_dst count is wrong), then iterate
    // cg_dst's pids and capture each one's /proc/<pid>/comm.
    // All redirects complete inside `sh -c` before exit; no
    // sleep, no background. `Op::wait_payload` below blocks
    // on the exit so the tmpfs files are fully written before
    // the test body reads them. The `||:` (no-op fallback)
    // on the cg_src cat keeps the script's exit code tied to
    // the cg_dst probe — cg_src may have been rmdir'd by the
    // time the script runs if all workers migrated out
    // successfully (cg_src is step-local via CgroupDef and
    // gets torn down at step boundary, but the probe runs
    // BEFORE step boundary so the dir should still exist; the
    // fallback is defense-in-depth so a transient ENOENT on
    // cg_src doesn't mask the primary cg_dst diagnostic).
    let probe_script = format!(
        "cat {procs} > {procs_out} && \
         (cat {src_procs} > {src_procs_out} 2>/dev/null ||:) && \
         for p in $(cat {procs}); do cat /proc/$p/comm; done > {comms_out}",
        procs = cg_dst_procs.display(),
        procs_out = PROCS_SNAPSHOT_PATH,
        src_procs = cg_src_procs.display(),
        src_procs_out = SRC_PROCS_SNAPSHOT_PATH,
        comms_out = COMMS_SNAPSHOT_PATH,
    );
    let steps = vec![Step {
        // cg_src now lives in the Backdrop above (see the rationale
        // there); Step::setup stays empty so apply_setup is a no-op
        // and the step only runs the ops below.
        setup: Default::default(),
        ops: vec![
            // No `Op::AddCgroup { name: "cg_dst" }` — the
            // Backdrop above already creates cg_dst, and a
            // step-local AddCgroup against a name already
            // tracked by the Backdrop bails with the collision
            // diagnostic in `Op::AddCgroup`'s apply_ops arm.
            Op::spawn_workers(
                "cg_dst",
                WorkSpec::default()
                    .workers(1)
                    .comm(WORKER_COMM_DST)
                    .work_type(WorkType::SpinWait),
            ),
            Op::spawn_host(
                WorkSpec::default()
                    .workers(1)
                    .comm(WORKER_COMM_HOST)
                    .work_type(WorkType::SpinWait),
            ),
            Op::move_all_tasks("cg_src", "cg_dst"),
            // Shell probe runs in the runner's cgroup (the
            // default for `Op::run_payload` with `cgroup =
            // None`), so its pid does NOT join cg_dst and the
            // expected pid count stays at 3. Critically, the
            // read happens INSIDE the scenario — the
            // post-scenario `BackdropState::drop` will rmdir
            // cg_dst before the test body resumes.
            Op::run_payload(&SHELL_PROBE, vec!["-c".to_string(), probe_script]),
            // Block until the probe exits so the tmpfs files
            // are fully written by the time the test body
            // reads them. `Op::wait_payload` is event-driven
            // (waits on the payload's pid); no sleep involved.
            Op::wait_payload(SHELL_PROBE.name),
        ],
        // `Op::wait_payload` already provides synchronous
        // gating on the probe; the step-level hold has no
        // additional work to do. ZERO valid per `HoldSpec::validate`.
        hold: HoldSpec::fixed(Duration::ZERO),
    }];
    let _ = execute_scenario(ctx, backdrop, steps)?;

    // Read the procs snapshot the probe wrote inside the scenario.
    // Unlike a direct read of cg_dst/cgroup.procs (which would
    // race the BackdropState's end-of-scenario rmdir), the
    // tmpfs file persists past teardown.
    let procs_contents = match std::fs::read_to_string(PROCS_SNAPSHOT_PATH) {
        Ok(s) => s,
        Err(e) => {
            return Ok(AssertResult::fail(AssertDetail::new(
                DetailKind::Other,
                format!(
                    "read {PROCS_SNAPSHOT_PATH}: {e}. The shell probe \
                     was expected to write cg_dst/cgroup.procs to this \
                     path before `Op::wait_payload` returned; a missing \
                     file means the probe never ran (spawn failure), \
                     `wait_payload` returned before the redirect \
                     completed, or the inner `cat` failed (cg_dst \
                     gone, permission denied, ...)."
                ),
            )));
        }
    };
    // `cgroup.procs` renders one pid per line (kernel:
    // `cgroup_procs_show` in kernel/cgroup/cgroup.c iterates the
    // css_set's tasks and writes one decimal pid + '\n' per
    // entry). `.lines()` skips a trailing empty line cleanly;
    // the extra `is_empty` filter guards against blank middle
    // lines that the kernel never emits but cost nothing to
    // tolerate.
    let pids: Vec<&str> = procs_contents.lines().filter(|s| !s.is_empty()).collect();
    const EXPECTED_PIDS: usize = 3;
    if pids.len() != EXPECTED_PIDS {
        // Pull supplementary state into the diagnostic so an
        // operator can split spawn-failed vs migrate-failed
        // without re-running:
        //   * cg_src pid list — empty means MoveAllTasks
        //     migrated whatever it observed (so a deficit in
        //     cg_dst points at spawn_workers); non-empty means
        //     MoveAllTasks left workers behind.
        //   * cg_dst comm multiset — distinguishes
        //     spawn_workers's worker (DST comm) from
        //     migrated cg_src workers (SRC comm).
        let src_pids = std::fs::read_to_string(SRC_PROCS_SNAPSHOT_PATH)
            .map(|s| {
                s.lines()
                    .filter(|l| !l.is_empty())
                    .map(str::to_string)
                    .collect::<Vec<_>>()
            })
            .unwrap_or_else(|_| Vec::new());
        let observed_comms = std::fs::read_to_string(COMMS_SNAPSHOT_PATH)
            .map(|s| {
                s.lines()
                    .filter(|l| !l.is_empty())
                    .map(str::to_string)
                    .collect::<Vec<_>>()
            })
            .unwrap_or_else(|_| Vec::new());
        // Surface the apply_setup placement log written by the
        // ktstr scenario engine. The file records one line per
        // cgroup-def spawn batch with the placed PIDs; an
        // "apply_setup: spawned non-pcomm workers cgroup=cg_src
        // count=2 pids=[...]" line confirms apply_setup placed the
        // 2 cg_src workers as expected — a missing or zero-count
        // line for cg_src would prove the placement itself failed
        // and shift the investigation from "did move_all_tasks
        // migrate" to "did apply_setup spawn." A read failure is
        // recorded inline rather than masked because its absence
        // is itself diagnostic.
        let apply_setup_log = std::fs::read_to_string(ktstr::scenario::ops::PLACEMENT_LOG_PATH)
            .unwrap_or_else(|e| {
                format!(
                    "<read {} failed: {e}>",
                    ktstr::scenario::ops::PLACEMENT_LOG_PATH
                )
            });
        return Ok(AssertResult::fail(AssertDetail::new(
            DetailKind::Other,
            format!(
                "{PROCS_SNAPSHOT_PATH} contains {} pids ({:?}); expected \
                 exactly {EXPECTED_PIDS} (1 from Op::spawn_workers(\"cg_dst\", \
                 ...) + 2 migrated from cg_src by Op::move_all_tasks). \
                 cg_src.procs snapshot at the same instant: {} pids \
                 ({:?}) — NB: \"0 pids\" here is ambiguous between \
                 (a) Op::move_all_tasks correctly migrated everything, \
                 and (b) cg_src directory was never created OR was \
                 rmdir'd before the snapshot (the shell probe's \
                 `cat ... 2>/dev/null ||:` suppresses cg_src ENOENT \
                 to keep the diagnostic-supplement non-fatal — see \
                 file-header script-doc). Observed cg_dst comms: {:?}. \
                 A deficit with cg_src non-empty means \
                 Op::move_all_tasks did not migrate the cg_src \
                 workers; a deficit with cg_src empty AND DST present \
                 in cg_dst comms means the cg_src workers vanished \
                 (apply_setup never placed them, OR they exited \
                 before move_all_tasks ran); a deficit with cg_src \
                 empty AND DST absent from cg_dst comms means \
                 Op::spawn_workers(\"cg_dst\", ...) did not place \
                 either; a surplus means Op::spawn_host's \
                 runner-cgroup worker leaked into cg_dst, or the \
                 shell probe was spawned in cg_dst instead of the \
                 runner's cgroup.\n\
                 --- apply_setup placement log ---\n{}",
                pids.len(),
                pids,
                src_pids.len(),
                src_pids,
                observed_comms,
                apply_setup_log,
            ),
        )));
    }
    // Identity assertion: each pid in cg_dst should have one of
    // the expected worker comms, and the multiset should match
    // {SRC × 2, DST × 1}. An Op::Spawn(RunnerCgroup) leak surfaces
    // as `HOST` in the comm set; a partial MoveAllTasks yields
    // fewer `SRC`.
    let comms_contents = match std::fs::read_to_string(COMMS_SNAPSHOT_PATH) {
        Ok(s) => s,
        Err(e) => {
            return Ok(AssertResult::fail(AssertDetail::new(
                DetailKind::Other,
                format!(
                    "read {COMMS_SNAPSHOT_PATH}: {e}. The probe was expected \
                     to write each cg_dst pid's /proc/<pid>/comm to this \
                     path. A missing file means the inner `for p in ...; \
                     do cat /proc/$p/comm; done` loop failed before \
                     completion."
                ),
            )));
        }
    };
    let comms: Vec<&str> = comms_contents.lines().filter(|s| !s.is_empty()).collect();
    if comms.len() != EXPECTED_PIDS {
        return Ok(AssertResult::fail(AssertDetail::new(
            DetailKind::Other,
            format!(
                "{COMMS_SNAPSHOT_PATH} contains {} comms ({:?}); expected \
                 exactly {EXPECTED_PIDS} (one per cg_dst pid). A mismatch \
                 means one of the pids vanished between the cgroup.procs \
                 read and the per-pid /proc/<pid>/comm cat (process exit \
                 between the two reads — unexpected for SpinWait workers \
                 within a 0-hold step).",
                comms.len(),
                comms,
            ),
        )));
    }
    let count_src = comms.iter().filter(|c| **c == WORKER_COMM_SRC).count();
    let count_dst = comms.iter().filter(|c| **c == WORKER_COMM_DST).count();
    let count_host = comms.iter().filter(|c| **c == WORKER_COMM_HOST).count();
    let count_other = comms.len() - count_src - count_dst - count_host;
    if count_src != 2 || count_dst != 1 || count_host != 0 || count_other != 0 {
        return Ok(AssertResult::fail(AssertDetail::new(
            DetailKind::Other,
            format!(
                "cg_dst comm multiset = {{{WORKER_COMM_SRC}: {count_src}, \
                 {WORKER_COMM_DST}: {count_dst}, {WORKER_COMM_HOST}: \
                 {count_host}, other: {count_other}}}; expected \
                 {{SRC: 2, DST: 1, HOST: 0, other: 0}}. SRC≠2 means \
                 Op::move_all_tasks did not migrate both cg_src workers \
                 (partial migration). DST≠1 means Op::spawn_workers \
                 did not place its worker in cg_dst. HOST≠0 means \
                 Op::spawn_host leaked its worker into cg_dst instead \
                 of the runner cgroup. other≠0 means an unexpected \
                 task landed in cg_dst (kthread, scheduler helper, \
                 etc.). Raw comms observed: {comms:?}",
            ),
        )));
    }
    Ok(AssertResult::pass())
}
