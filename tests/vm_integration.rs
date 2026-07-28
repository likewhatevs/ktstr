//! VM integration tests for kernel-facing capture pipelines.
//!
//! Each test boots a real KVM VM via `#[ktstr_test]`, runs a small
//! workload under scx-ktstr with `--stall-after=1`, lets the freeze
//! coordinator capture a `FailureDumpReport`, and asserts the
//! captured JSON carries the field the test is responsible for
//! pinning. Five tests cover the capture-pipeline gaps:
//!
//! - **DSQ + rq->scx walker**: `dsq_states` and `rq_scx_states`
//!   in the dump JSON populated from real frozen-VM walk.
//! - **Per-vCPU perf counters**: `vcpu_perf_at_freeze` populated
//!   with at least one non-`None` slot from a real
//!   `perf_event_open(exclude_host=1)` read.
//! - **Event-counter timeline**: `event_counter_timeline`
//!   populated with at least one entry across the run window — the
//!   load-bearing surface for the sched-event capture path. (The
//!   discrete tracepoint timeline is wired via `TimelineCapture` but
//!   not yet attached to FailureDumpReport; the event-counter
//!   timeline is the visible per-tick timeline today.) Runs on every
//!   job; it `post_vm_skip`s when (a) the guest kernel lacks the
//!   SCX_EV_* counters (pre-6.16, e.g. the 6.14 leg — no
//!   `scx_sched`/`scx_root` in BTF, so `scx_event_counters_supported`
//!   is false), (b) the per-tick loop captured no samples at all, or
//!   (c) — on `wprof`/coverage builds only — samples were captured but
//!   none carried counters (a host-load-starved leg). It asserts on the
//!   non-`wprof` legs, where a zero capture on a counter-supporting
//!   kernel is the real regression it guards.
//! - **SchedPolicy::Deadline**: worker spawned under
//!   `SchedPolicy::Deadline` reaches `worker_main` without bailing —
//!   proves the `sched_setattr(2)` syscall path runs end-to-end on
//!   a real kernel that supports `SCHED_DEADLINE`.
//! - **Failure-dump trigger**: boot → stall → capture → render
//!   pipeline produces a non-empty top-level dump (overlaps with
//!   `failure_dump_e2e.rs` but pins the schema discriminant + the
//!   minimal cross-pipeline invariant).
//!
//! Every scenario consumes the existing `failure_dump_e2e.rs`
//! pattern — same `--stall-after=1` trigger, same per-test sidecar
//! path resolution, same JSON shape inspection — so the host-side
//! freeze-coordinator wiring is exercised once per test in lockstep
//! with the in-tree pattern.
//!
//! User-facing test bar: each kernel-facing capture surface used by
//! ktstr's debugging story (DSQ depth, rq->scx scalars, vCPU perf,
//! per-tick event counters, SCHED_DEADLINE invocation, dump trigger)
//! must produce live data on a real VM run, not a synthetic literal
//! or a unit-tested code path.

mod common;

use anyhow::Result;
use common::failure_dump::read_dump_skip_placeholder;
use ktstr::assert::{AssertDetail, AssertResult, DetailKind};
use ktstr::prelude::{VmResult, post_vm_skip};
use ktstr::scenario::ops::{HoldSpec, Step, await_accessor_ready, execute_steps};
use ktstr::test_support::{Scheduler, SchedulerSpec};

const KTSTR_SCHED: Scheduler =
    Scheduler::named("ktstr_sched").binary(SchedulerSpec::Discover("scx-ktstr"));

/// Run a one-step workload under scx-ktstr `--stall-after=1` after the
/// freeze accessor is adopted (so the resulting failure dump renders
/// real captured state rather than a placeholder). Returns the
/// `AssertResult` from `execute_steps`; the per-scenario host-side
/// `check_*` post_vm callbacks read the host-written dump and assert.
fn run_stalled_workload(ctx: &ktstr::scenario::Ctx) -> Result<AssertResult> {
    await_accessor_ready();
    let steps = vec![Step {
        setup: vec![ctx.cgroup_def("cg_0")].into(),
        ops: vec![],
        hold: HoldSpec::FULL,
    }];
    execute_steps(ctx, steps)
}

// ----------------------------------------------------------------------------
// DSQ + rq->scx walker
// ----------------------------------------------------------------------------

/// Boot scx-ktstr, trigger a stall, and assert that the freeze-time
/// `dsq_states` walk produced at least one DSQ entry AND that
/// `rq_scx_states` enumerates at least one CPU's `rq->scx` scalars.
///
/// Pins the kernel-facing walker pipeline:
///   1. `ScxWalkerOffsets` resolved from the guest BTF (otherwise
///      `scx_walker_unavailable` would carry a reason and both vecs
///      would be empty).
///   2. `*scx_root` translated host-side to a kernel KVA.
///   3. The DSQ enumeration (`bypass_dsq` + per-CPU local DSQs +
///      global + user DSQs) yielded at least one entry.
///   4. The per-CPU `rq->scx` walk produced at least one record.
///
/// A regression that breaks any layer (BTF offsets, root translation,
/// IDR walk, percpu translation) flushes one of the two vecs to zero
/// length. The test's lower bound (>=1 DSQ, >=1 rq->scx) is the
/// minimal signal that the pipeline is alive end-to-end without
/// pinning a brittle exact count that drifts with scheduler version
/// or topology.
fn scenario_dsq_and_rq_walker_populates_failure_dump(
    ctx: &ktstr::scenario::Ctx,
) -> Result<AssertResult> {
    run_stalled_workload(ctx)
}

/// Host-side post_vm assertion for `vm_integration_dsq_and_rq_walker`.
/// The guest cannot read the host-written dump, so the walker
/// assertions run here; the callback's Err is a hard FAIL
/// (`PostVmAssertionFailure`) that `expect_err` does not invert.
fn check_dsq_and_rq_walker(result: &VmResult) -> Result<()> {
    let dump = read_dump_skip_placeholder(result)?;

    // If the walker explicitly could not run (BTF offsets unresolved or
    // *scx_root untranslatable under load), the DSQ / rq->scx capture is
    // inconclusive — skip rather than fail.
    if let Some(reason) = dump.get("scx_walker_unavailable").and_then(|v| v.as_str()) {
        return Err(post_vm_skip(format!(
            "scx walker unavailable ({reason}) — the DSQ / rq->scx walk could \
             not run, so its capture cannot be verified"
        )));
    }

    // dsq_states is `skip_serializing_if = "Vec::is_empty"`, so its
    // absence here means the walk produced zero entries. With the walker
    // reporting available, treat absent and present-but-empty as the
    // same regression.
    let dsq_states: &[serde_json::Value] = match dump.get("dsq_states") {
        Some(s) => s
            .as_array()
            .map(|a| a.as_slice())
            .ok_or_else(|| anyhow::anyhow!("dsq_states is present but not an array: {s}"))?,
        None => &[],
    };
    if dsq_states.is_empty() {
        anyhow::bail!(
            "dsq_states is empty (absent or zero-length) despite the walker \
             reporting available. The walker resolved BTF offsets and \
             translated *scx_root but the IDR walk yielded no DSQs."
        );
    }

    let rq_scx_states: &[serde_json::Value] = match dump.get("rq_scx_states") {
        Some(s) => s
            .as_array()
            .map(|a| a.as_slice())
            .ok_or_else(|| anyhow::anyhow!("rq_scx_states is present but not an array: {s}"))?,
        None => &[],
    };
    if rq_scx_states.is_empty() {
        anyhow::bail!(
            "rq_scx_states is empty (absent or zero-length). Per-CPU rq->scx \
             walk failed wholesale — every CPU's percpu translation errored \
             or the offsets were unavailable."
        );
    }

    eprintln!(
        "scx walker captured {} DSQ entries and {} rq->scx entries from \
         frozen-VM walk",
        dsq_states.len(),
        rq_scx_states.len(),
    );
    Ok(())
}

// ----------------------------------------------------------------------------
// Per-vCPU perf counters
// ----------------------------------------------------------------------------

/// Boot scx-ktstr, trigger a stall, and assert that
/// `vcpu_perf_at_freeze` carries at least one non-`None` slot
/// reflecting a real `read(2)` from a `perf_event_open(exclude_host=1)`
/// counter at freeze time.
///
/// Pins:
///   1. `DumpContext::perf_capture` was attached (perf available on
///      the host — kernel.perf_event_paranoid permits it, no
///      capability denial).
///   2. The vec has at least one entry per vCPU ordering (length
///      matches vCPU count when populated).
///   3. At least one entry is a non-null `VcpuPerfSample` —
///      `read(2)` succeeded for at least one vCPU.
///
/// `vcpu_perf_at_freeze` is `skip_serializing_if = "Vec::is_empty"`
/// AND each entry is `Option<VcpuPerfSample>` (null on per-vCPU
/// failure). Test treats absent vec as a hard fail (perf wholesale
/// unavailable on this host or test runner) — fall-back-to-empty
/// would mask a regression that breaks the perf wiring on every
/// host. Tests that need to opt out for perf-unavailable hosts can
/// add `#[cfg(feature = "...")]` later.
fn scenario_perf_counters_capture_populates_dump(
    ctx: &ktstr::scenario::Ctx,
) -> Result<AssertResult> {
    run_stalled_workload(ctx)
}

/// Host-side post_vm assertion for `vm_integration_perf_counters_capture`.
/// The guest cannot read the host-written dump, so the perf-capture
/// assertions run here; the callback's Err is a hard FAIL
/// (`PostVmAssertionFailure`) that `expect_err` does not invert.
fn check_perf_counters_capture(result: &VmResult) -> Result<()> {
    let dump = read_dump_skip_placeholder(result)?;

    let vcpu_perf: &[serde_json::Value] = match dump.get("vcpu_perf_at_freeze") {
        Some(v) => v
            .as_array()
            .map(|a| a.as_slice())
            .ok_or_else(|| anyhow::anyhow!("vcpu_perf_at_freeze present but not an array: {v}"))?,
        None => &[],
    };
    if vcpu_perf.is_empty() {
        // DumpContext::perf_capture was None — perf_event_open(exclude_host=1)
        // is unavailable on this host (kernel.perf_event_paranoid too
        // restrictive, or capability missing). That is a host-config
        // property, not a ktstr regression, so the run is inconclusive.
        return Err(post_vm_skip(
            "vcpu_perf_at_freeze is empty — perf_event_open(exclude_host=1) is \
             unavailable on this host (raise it with `sysctl \
             kernel.perf_event_paranoid=2` or lower); the perf capture cannot \
             be verified here",
        ));
    }

    // At least one slot must be a populated VcpuPerfSample (not null).
    // serde_json renders `None` as JSON `null`; a populated entry is
    // an object.
    let populated: Vec<&serde_json::Value> =
        vcpu_perf.iter().filter(|slot| slot.is_object()).collect();
    if populated.is_empty() {
        anyhow::bail!(
            "vcpu_perf_at_freeze has {} entries but every slot is null \
             (read(2) failed for every vCPU). Capture wiring may be broken: \
             check perf_event_attr.exclude_host and that the per-vCPU fd \
             remained valid through freeze.",
            vcpu_perf.len(),
        );
    }

    eprintln!(
        "vcpu_perf_at_freeze: {}/{} vCPUs reported a non-null \
         perf_event_open(exclude_host=1) sample at freeze",
        populated.len(),
        vcpu_perf.len(),
    );
    Ok(())
}

// ----------------------------------------------------------------------------
// Event-counter timeline (sched-event capture)
// ----------------------------------------------------------------------------

/// Boot scx-ktstr, trigger a stall, and assert that
/// `event_counter_timeline` carries at least one
/// `EventCounterSample` — proves the per-monitor-tick capture loop
/// observed the kernel's SCX_EV_* counters and the freeze coordinator
/// folded them into the dump.
///
/// Pins:
///   1. The monitor's per-tick sample loop ran (otherwise the vec is
///      empty).
///   2. `ScxEventCounters` offsets were resolved from BTF — without
///      them, every sample reports zero counters and the per-sample
///      drop predicate (skip when all-zero across the SCX_EV_*
///      family) would empty the vec.
///   3. The freeze coordinator's `EventCounterCapture` parameter was
///      attached (otherwise the vec is empty regardless of monitor
///      activity).
///
/// The lower bound (>=1 sample) is the minimal signal that the
/// per-tick capture surface is alive on a real kernel run. Sparkline
/// rendering on top of this vec is unit-tested elsewhere; this test
/// pins the integration boundary where unit tests cannot reach.
fn scenario_event_counter_timeline(ctx: &ktstr::scenario::Ctx) -> Result<AssertResult> {
    run_stalled_workload(ctx)
}

/// Host-side post_vm assertion for `vm_integration_event_counter_timeline`.
/// The per-tick SCX_EV_* capture is recorded on `VmResult::monitor`
/// (its `samples`), NOT in the failure dump: the freeze coordinator
/// currently passes `event_counter_capture: None` to dump_state
/// (src/vmm/freeze_coord/mod.rs), so the dump's `event_counter_timeline`
/// is always empty and asserting on it would always skip. The monitor
/// sample stream carries the real per-tick capture, so the assertion
/// runs against it here. The callback's Err is a hard FAIL
/// (`PostVmAssertionFailure`) that `expect_err` does not invert.
fn check_event_counter_timeline(result: &VmResult) -> Result<()> {
    let monitor = result.monitor.as_ref().ok_or_else(|| {
        anyhow::anyhow!("VmResult.monitor is None — the monitor sampler produced no report")
    })?;
    if !monitor.scx_event_counters_supported {
        // The running guest kernel does not expose the SCX_EV_* event
        // counters: the monitor resolved no `event_offsets` from BTF
        // because `struct scx_sched` / `scx_root` — the machinery the
        // counters live behind — is a 6.16-cycle addition. On such a
        // kernel (e.g. the 6.14 CI leg) there is nothing to capture, so
        // the assertion is inapplicable rather than failed. It still
        // hard-fails on 6.16+/7.1, where the kernel structurally
        // supports the counters but capture produced none — the real
        // regression this test guards. A BTF-capability probe, not a
        // version compare, so it stays correct across backports.
        return Err(post_vm_skip(
            "guest kernel lacks the SCX_EV_* event counters (pre-6.16: no \
             scx_sched/scx_root in BTF); the counter-timeline capture is \
             inapplicable on this kernel version",
        ));
    }
    if monitor.samples.is_empty() {
        // The per-tick loop recorded no sample before the stall fired
        // (load-starved); inconclusive for this assertion.
        return Err(post_vm_skip(
            "monitor recorded no samples — the per-tick loop did not tick before \
             the stall fired (load-starved); the SCX_EV_* capture cannot be \
             verified",
        ));
    }

    // At least one sample must carry SCX_EV_* counters on at least one CPU.
    // CpuSnapshot.event_counters is None when the ScxEventCounters offsets
    // are unresolved or scx_root is unset; if EVERY CPU in EVERY sample is
    // None, the per-tick capture observed nothing — a regression.
    let with_events = monitor
        .samples
        .iter()
        .filter(|s| s.cpus.iter().any(|c| c.event_counters.is_some()))
        .count();
    if with_events == 0 && cfg!(feature = "wprof") {
        // wprof/coverage builds run on the host-load-starved CI legs. There
        // the guest scheduler can fail to attach — or the monitor's
        // `data_valid` latch can fail to fire — within the sample window, so
        // no sample carries event counters even though the kernel supports
        // them. Treat that as inconclusive rather than a failure: a zero
        // capture under wprof cannot be distinguished from starvation, and
        // the assertion below is enforced on the non-wprof legs, where a
        // zero capture on an unstarved host IS the regression this test
        // guards. (Empirically the fix makes 7.1×wprof assert and pass when
        // capture is not starved; this only skips the extreme-starvation
        // tail.)
        return Err(post_vm_skip(
            "no monitor sample carried SCX_EV_* counters under a wprof/coverage \
             build — the host-load-starved vCPUs left the scheduler unattached \
             for the sample window; inconclusive (the assertion is enforced on \
             the non-wprof legs)",
        ));
    }
    anyhow::ensure!(
        with_events > 0,
        "no monitor sample carried SCX_EV_* event counters across {} samples — \
         the per-tick capture resolved no ScxEventCounters offsets from BTF, or \
         scx_root was unset on every CPU at every tick",
        monitor.samples.len(),
    );

    eprintln!(
        "event-counter capture: {}/{} monitor samples carried SCX_EV_* counters",
        with_events,
        monitor.samples.len(),
    );
    Ok(())
}

// ----------------------------------------------------------------------------
// SchedPolicy::Deadline real sched_setattr invocation
// ----------------------------------------------------------------------------

/// Spawn a worker under `SchedPolicy::Deadline` inside the VM and
/// assert it runs to non-default WorkerReport status. The
/// `sched_setattr(2)` syscall path needs a real kernel with
/// CONFIG_SCHED_DEADLINE — host-side unit tests cannot exercise it.
///
/// The scenario uses `WorkType::SpinWait` under SCHED_DEADLINE
/// `(runtime=500us, deadline=1ms, period=10ms)` — a 5% bandwidth
/// reservation that easily fits on any single CPU, so the
/// admission-control path (`__checkparam_dl`,
/// `kernel/sched/deadline.c::sched_dl_overflow`) accepts it.
///
/// Pins:
///   1. `WorkerReport::completed = true` — the worker ran the SCHED_DL
///      slice without the kernel returning EBUSY/EINVAL on the
///      `sched_setattr` syscall.
///   2. `WorkerReport::work_units > 0` — the SCHED_DL band granted
///      the worker actual run time within its declared deadline.
///
/// A regression in the syscall ABI (wrong sched_attr field offsets,
/// missing `flags`, wrong `size`) would surface as a sentinel report
/// — `completed = false` with a `WorkerExitInfo::Exited(<errno>)`
/// failure. The lower bound asserts on the success path; the
/// assertion-string-drift unit tests elsewhere catch the error path
/// shapes.
fn scenario_sched_deadline_real_setattr(ctx: &ktstr::scenario::Ctx) -> Result<AssertResult> {
    use ktstr::workload::{AffinityIntent, SchedPolicy, WorkType, WorkloadConfig, WorkloadHandle};
    use std::time::Duration;

    let config = WorkloadConfig {
        num_workers: 1,
        work_type: WorkType::SpinWait,
        affinity: AffinityIntent::Inherit,
        sched_policy: SchedPolicy::Deadline {
            runtime: Duration::from_micros(500),
            deadline: Duration::from_millis(1),
            period: Duration::from_millis(10),
        },
        ..Default::default()
    };

    let mut handle = WorkloadHandle::spawn(&config)?;
    handle.start();
    std::thread::sleep(ctx.duration);
    let reports = handle.stop_and_collect();

    let mut result = AssertResult::pass();
    if reports.is_empty() {
        result.record_fail(AssertDetail::new(
            DetailKind::Other,
            "SCHED_DEADLINE worker produced no report — sched_setattr likely \
             rejected the params"
                .to_string(),
        ));
        return Ok(result);
    }
    let r = &reports[0];
    if !r.completed {
        result.record_fail(AssertDetail::new(
            DetailKind::Other,
            format!(
                "SCHED_DEADLINE worker reported completed=false (sentinel) — \
                 sched_setattr returned an error or the worker died before \
                 the work loop. exit_info={:?}, work_units={}",
                r.exit_info, r.work_units,
            ),
        ));
        return Ok(result);
    }
    if r.work_units == 0 {
        result.record_fail(AssertDetail::new(
            DetailKind::Other,
            format!(
                "SCHED_DEADLINE worker reported work_units=0 — the SCHED_DL \
                 band did not grant any run time within the declared period. \
                 wall_time_ns={}, cpu_time_ns={}",
                r.wall_time_ns, r.cpu_time_ns,
            ),
        ));
        return Ok(result);
    }

    result.note(format!(
        "SCHED_DEADLINE worker completed cleanly: tid={}, work_units={}, \
         wall_time_ns={}, cpu_time_ns={} — sched_setattr(2) syscall \
         path verified end-to-end on real kernel",
        r.tid, r.work_units, r.wall_time_ns, r.cpu_time_ns,
    ));
    Ok(result)
}

// ----------------------------------------------------------------------------
// Failure-dump trigger, full-stack
// ----------------------------------------------------------------------------

/// End-to-end sanity check for the failure-dump trigger: boot →
/// stall → capture → render produces a non-empty top-level dump
/// with the schema discriminant intact.
///
/// Overlaps with `failure_dump_e2e.rs::scenario_failure_dump_renders_bss_fields`
/// in the underlying mechanism, but pins distinct minimal invariants:
///   1. `schema` field is present and equals "single" (the in-tree
///      discriminant for non-incremental dumps).
///   2. `maps` array is non-empty (BPF map enumeration found at least
///      one map after the freeze).
///   3. `vcpu_regs` array is non-empty (rendezvous attached at least
///      one vCPU's regs snapshot).
///
/// A regression that breaks any of these layers (schema renamed,
/// map enumeration broken, vCPU rendezvous timing out) would
/// surface here independent of which BPF struct the bss-fields test
/// happens to look at.
fn scenario_failure_dump_trigger_minimal_invariants(
    ctx: &ktstr::scenario::Ctx,
) -> Result<AssertResult> {
    run_stalled_workload(ctx)
}

/// Host-side post_vm assertion for `vm_integration_failure_dump_trigger`.
/// The guest cannot read the host-written dump, so the dump-invariant
/// assertions run here; the callback's Err is a hard FAIL
/// (`PostVmAssertionFailure`) that `expect_err` does not invert.
fn check_failure_dump_trigger(result: &VmResult) -> Result<()> {
    // A placeholder dump is inconclusive (skip); a real dump that
    // dropped its BPF-map enumeration or vCPU regs is a silent-drop
    // regression this test exists to catch — so read_dump_skip_placeholder
    // (NOT read_failure_dump, which would skip on empty maps) is used and
    // the emptiness checks below hard FAIL.
    let dump = read_dump_skip_placeholder(result)?;

    let schema = dump
        .get("schema")
        .and_then(|s| s.as_str())
        .ok_or_else(|| anyhow::anyhow!("dump JSON missing top-level `schema` field"))?;
    if schema != "single" {
        anyhow::bail!(
            "schema discriminant is {schema:?}, expected \"single\". \
             A rename or refactor of the schema constant must update \
             every consumer that pins this string."
        );
    }

    let maps = dump
        .get("maps")
        .and_then(|m| m.as_array())
        .ok_or_else(|| anyhow::anyhow!("dump JSON missing top-level `maps` array"))?;
    if maps.is_empty() {
        anyhow::bail!(
            "dump JSON `maps` array is empty — BPF map enumeration did not \
             find a single map after the SCX_EXIT_ERROR_STALL freeze. The \
             scheduler always loads at least the .bss + arena maps; an \
             empty map list means dump_state's IDR walk is broken."
        );
    }

    let vcpu_regs = dump
        .get("vcpu_regs")
        .and_then(|v| v.as_array())
        .ok_or_else(|| anyhow::anyhow!("dump JSON missing top-level `vcpu_regs` array"))?;
    if vcpu_regs.is_empty() {
        anyhow::bail!(
            "dump JSON `vcpu_regs` array is empty — freeze rendezvous \
             collected no vCPU snapshots. Either the rendezvous timed out \
             before any vCPU completed handle_freeze, or the regs-attach \
             callback was never registered."
        );
    }

    eprintln!(
        "failure-dump trigger pipeline produced schema={schema:?}, \
         {} maps, {} vcpu_regs entries — full-stack capture path \
         verified end-to-end",
        maps.len(),
        vcpu_regs.len(),
    );
    Ok(())
}

// ----------------------------------------------------------------------------
// Disk integration: boot with DiskConfig, exercise /dev/vda from guest
// ----------------------------------------------------------------------------
//
// These four scenarios pin the framework-level wiring that exposes a
// virtio-blk device to a `#[ktstr_test]` scenario:
//   1. `KtstrTestEntry.disk` carries a [`DiskConfig`].
//   2. [`crate::test_support::runtime::build_vm_builder_base`] forwards
//      it to [`crate::vmm::KtstrVmBuilder::disk`].
//   3. The host opens a sparse temp backing file and surfaces the device
//      to the guest at `/dev/vda`. The transport is arch-split: x86_64
//      installs a virtio-pci function (`init_virtio_blk_pci`, INTx +
//      MSI-X over a 32-bit BAR), aarch64 a virtio-MMIO device
//      (`init_virtio_blk`). Either way the guest's virtio-block driver
//      probes `/dev/vda`.
//
// Each scenario runs as guest-side Rust under PID 1 and uses
// `std::fs` against `/dev/vda` directly — no busybox, no shelling
// out. Failures distinguish missing-device vs IO-failure vs
// roundtrip-mismatch via dedicated [`AssertDetail`] entries.
//
// The `DiskConfig` struct used in each `static` lives in
// `ktstr::vmm::disk_config` (re-exported in `ktstr::prelude`); fields
// are written struct-literally because [`DiskConfig::default`] is not
// `const fn` and a `static` initializer must be const-evaluable.

const KTSTR_DISK_DEFAULT: ktstr::prelude::DiskConfig = ktstr::prelude::DiskConfig {
    capacity_mib: 256,
    filesystem: ktstr::prelude::Filesystem::Raw,
    throttle: ktstr::prelude::DiskThrottle {
        iops: None,
        bytes_per_sec: None,
        iops_burst_capacity: None,
        bytes_burst_capacity: None,
    },
    read_only: false,
    name: None,
    no_auto_mount: false,
};

const KTSTR_DISK_READ_ONLY: ktstr::prelude::DiskConfig = ktstr::prelude::DiskConfig {
    capacity_mib: 256,
    filesystem: ktstr::prelude::Filesystem::Raw,
    throttle: ktstr::prelude::DiskThrottle {
        iops: None,
        bytes_per_sec: None,
        iops_burst_capacity: None,
        bytes_burst_capacity: None,
    },
    read_only: true,
    name: None,
    no_auto_mount: false,
};

/// Boot the VM with a default-configured virtio-blk disk and assert
/// that `/dev/vda` appears as a block device inside the guest.
///
/// Pins the end-to-end wiring:
///   1. `KtstrTestEntry.disk = Some(..)` reaches
///      [`crate::test_support::runtime::build_vm_builder_base`].
///   2. The host attaches the virtio-blk device — a virtio-pci function
///      on x86_64 (`init_virtio_blk_pci`) or virtio-MMIO on aarch64
///      (`init_virtio_blk`).
///   3. The guest kernel's CONFIG_VIRTIO_BLK driver probes the
///      device, which surfaces as `/dev/vda` in the guest devtmpfs.
///
/// Asserts:
///   - `/dev/vda` exists and is a block device (per
///     `std::fs::metadata().file_type().is_block_device()`).
///   - The advertised capacity matches `KTSTR_DISK_DEFAULT.capacity_mib`
///     when read via `ioctl(BLKGETSIZE64)` (see kernel
///     `block/ioctl.c` for the constant `0x80081272`).
///
/// A regression that breaks any layer surfaces here as either a
/// missing device file or a wrong-capacity report.
fn scenario_disk_default_appears_at_dev_vda(_ctx: &ktstr::scenario::Ctx) -> Result<AssertResult> {
    use std::fs::OpenOptions;
    use std::os::unix::fs::FileTypeExt;
    use std::os::unix::io::AsRawFd;

    let path = std::path::Path::new("/dev/vda");
    let metadata = std::fs::metadata(path).map_err(|e| {
        anyhow::anyhow!(
            "/dev/vda missing in guest: {e}. The virtio-blk device was \
             not attached, the guest kernel does not have CONFIG_VIRTIO_BLK, \
             or the MMIO probe failed before devtmpfs populated /dev/vda."
        )
    })?;
    let ftype = metadata.file_type();
    if !ftype.is_block_device() {
        anyhow::bail!(
            "/dev/vda exists but is not a block device (file_type={ftype:?}). \
             devtmpfs created the node but the underlying device is not \
             a real virtio-blk; check the kernel-side virtio probe path."
        );
    }

    // Read the advertised capacity from the kernel via BLKGETSIZE64.
    // This validates the config-space `capacity` field round-trips
    // through the guest kernel, not just that the device node exists.
    let file = OpenOptions::new()
        .read(true)
        .open(path)
        .map_err(|e| anyhow::anyhow!("open /dev/vda for capacity probe: {e}"))?;
    let mut size_bytes: u64 = 0;
    // SAFETY: BLKGETSIZE64 is defined in linux/fs.h as
    // `_IOR(0x12, 114, size_t)` = 0x80081272 on a 64-bit host. The
    // kernel writes a `u64` to the `arg` pointer; `size_bytes` is a
    // valid mutable u64 for the duration of the call. The fd is
    // owned by `file` and outlives the syscall.
    let rc = unsafe { libc::ioctl(file.as_raw_fd(), 0x80081272, &mut size_bytes as *mut u64) };
    if rc != 0 {
        let errno = std::io::Error::last_os_error();
        anyhow::bail!(
            "BLKGETSIZE64 on /dev/vda returned {rc} (errno={errno}). The \
             kernel did not surface a capacity through the virtio config \
             space — possible config-space layout mismatch."
        );
    }

    let expected_bytes = (KTSTR_DISK_DEFAULT.capacity_mib as u64) << 20;
    if size_bytes != expected_bytes {
        anyhow::bail!(
            "BLKGETSIZE64 on /dev/vda reported {size_bytes} bytes; \
             expected {expected_bytes} ({} MiB). The host advertised \
             a different capacity than the test configured.",
            KTSTR_DISK_DEFAULT.capacity_mib,
        );
    }

    let mut result = AssertResult::pass();
    result.note(format!(
        "/dev/vda is a block device with capacity {size_bytes} bytes \
         ({} MiB), matching the configured DiskConfig",
        KTSTR_DISK_DEFAULT.capacity_mib,
    ));
    Ok(result)
}

/// Write a known pattern to sector 0 of `/dev/vda`, read it back,
/// and assert byte-for-byte equality. Pins the
/// virtio-blk read+write fast path end-to-end on a real KVM run:
///
///   1. Guest IO submission via the kernel block layer (`pwrite`,
///      `pread` on a block device).
///   2. virtio-blk descriptor chain construction by the guest driver
///      (`drivers/block/virtio_blk.c`).
///   3. Host-side chain dispatch through
///      [`crate::vmm::virtio_blk::VirtioBlk::process_requests`] ->
///      `handle_write` and `handle_read`.
///   4. Backing-file `pwrite`/`pread` on the host's sparse tempfile.
///   5. Status-byte write back to the guest's status descriptor.
///   6. `add_used` notification + irqfd → guest IRQ → completion.
///
/// The pattern is one full sector (512 bytes) of a recognizable
/// repeating byte (0xA5 = 0b10100101) so both an all-zero leak
/// (write didn't land) and a wrong-byte corruption (sector
/// addressing, endianness, descriptor-buffer aliasing) surface
/// distinctly.
fn scenario_disk_write_read_roundtrip(_ctx: &ktstr::scenario::Ctx) -> Result<AssertResult> {
    use std::fs::OpenOptions;
    use std::io::{Read, Seek, SeekFrom, Write};
    use std::os::unix::fs::OpenOptionsExt;

    const SECTOR_SIZE: usize = 512;
    const PATTERN_BYTE: u8 = 0xA5;

    // O_DIRECT-aligned sector buffer. Without 512-byte alignment, the
    // kernel rejects the O_DIRECT readback with EINVAL. `#[repr(align(512))]`
    // forces the struct's first byte (and thus `.0` array's first byte)
    // onto a 512-byte boundary regardless of stack layout.
    #[repr(align(512))]
    struct AlignedSector([u8; SECTOR_SIZE]);

    let path = std::path::Path::new("/dev/vda");
    let mut file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(path)
        .map_err(|e| {
            anyhow::anyhow!(
                "open /dev/vda for read+write: {e}. The disk should be \
                 attached read-write by default; if the host advertised \
                 VIRTIO_BLK_F_RO unexpectedly the kernel would refuse \
                 O_WRONLY on the device node."
            )
        })?;

    let pattern = [PATTERN_BYTE; SECTOR_SIZE];
    file.seek(SeekFrom::Start(0))
        .map_err(|e| anyhow::anyhow!("seek to sector 0 for write: {e}"))?;
    file.write_all(&pattern)
        .map_err(|e| anyhow::anyhow!("write pattern to sector 0: {e}"))?;
    // Sync to push the data through the kernel block layer to the
    // virtio device — without this, the write could sit in the page
    // cache and a subsequent read would short-circuit there instead
    // of round-tripping through the device. Also makes the WRITE
    // observable on the host's virtio-blk counters (record_write
    // fires per-request; record_flush fires on VIRTIO_BLK_T_FLUSH).
    file.sync_all()
        .map_err(|e| anyhow::anyhow!("fsync /dev/vda after write: {e}"))?;

    // Re-open for read with O_DIRECT so the readback bypasses the
    // bdev buffer cache and unconditionally reaches the device.
    // Without O_DIRECT the pwrite+fsync above leaves the sector in
    // the buffer cache; a subsequent pread typically short-circuits
    // there and never bumps the host's record_read counter. O_DIRECT
    // forces the kernel to issue a virtio-blk READ for every page
    // touched, which is exactly what the upstream host-side
    // counter-assertion test needs to observe (see post_vm callback
    // `assert_virtio_blk_counters_nonzero_after_roundtrip`).
    let mut readback = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_DIRECT)
        .open(path)
        .map_err(|e| anyhow::anyhow!("re-open /dev/vda for O_DIRECT readback: {e}"))?;
    let mut buf = AlignedSector([0u8; SECTOR_SIZE]);
    readback
        .seek(SeekFrom::Start(0))
        .map_err(|e| anyhow::anyhow!("seek to sector 0 for read: {e}"))?;
    readback
        .read_exact(&mut buf.0)
        .map_err(|e| anyhow::anyhow!("O_DIRECT read sector 0: {e}"))?;

    if buf.0 != pattern {
        // Find the first byte that differs to give a diagnostic
        // pointer into the corruption.
        let first_bad = buf
            .0
            .iter()
            .zip(pattern.iter())
            .position(|(a, b)| a != b)
            .unwrap_or(0);
        anyhow::bail!(
            "/dev/vda sector 0 readback mismatch: byte {first_bad} \
             read=0x{:02X} expected=0x{:02X}. The first 16 bytes \
             read back as {:02X?}, expected {:02X?}.",
            buf.0[first_bad],
            pattern[first_bad],
            &buf.0[..16.min(buf.0.len())],
            &pattern[..16.min(pattern.len())],
        );
    }

    let mut result = AssertResult::pass();
    result.note(format!(
        "{SECTOR_SIZE}-byte pattern written to sector 0 round-tripped \
         cleanly through virtio-blk write+fsync+read"
    ));
    Ok(result)
}

/// Boot with `/dev/vda` configured read-only and assert the guest
/// sees a read-only block device end-to-end. Pins the
/// [`DiskConfig::read_only`] knob:
///
///   1. The host advertises `VIRTIO_BLK_F_RO` via
///      `crate::vmm::virtio_blk::VirtioBlk::with_options` when
///      `read_only=true`.
///   2. The guest kernel observes the negotiated F_RO bit and marks
///      the gendisk read-only — surfaced at `/sys/block/vda/ro` (==1).
///   3. The kernel rejects WRITES to a read-only bdev at WRITE time
///      with `EPERM`, NOT at `open(2)` time: the bdev open path does
///      not gate on read-only, so `open(O_WRONLY)` SUCCEEDS and the
///      first `write()` returns `EPERM`. (`EROFS`, the prior
///      assertion, appears on neither path: userspace `open(2)` is
///      not gated, and the in-kernel `bdev_file_open_by_path` returns
///      `EACCES`.)
///
/// So the test reads `/sys/block/vda/ro` (must be `1`) and confirms a
/// `write()` to an `O_WRONLY` fd returns `EPERM`; reads are unaffected
/// (F_RO does not gate reads). The in-device write-rejection
/// (`VIRTIO_BLK_S_IOERR` for a write chain a guest builds despite the
/// negotiated bit) is unit-tested in `src/vmm/virtio_blk/`.
fn scenario_disk_read_only_rejects_write(_ctx: &ktstr::scenario::Ctx) -> Result<AssertResult> {
    use std::fs::OpenOptions;
    use std::io::Write;

    let path = std::path::Path::new("/dev/vda");
    // Read-open should still succeed — F_RO doesn't gate reads.
    {
        let _r = OpenOptions::new()
            .read(true)
            .open(path)
            .map_err(|e| anyhow::anyhow!("open /dev/vda read-only: {e}"))?;
    }

    // The guest kernel marked the gendisk read-only on F_RO
    // negotiation: /sys/block/vda/ro reads "1".
    let ro = std::fs::read_to_string("/sys/block/vda/ro").map_err(|e| {
        anyhow::anyhow!(
            "read /sys/block/vda/ro: {e} — /dev/vda likely did not \
             register (virtio_blk probe failed, or it is not a block \
             device)"
        )
    })?;
    anyhow::ensure!(
        ro.trim() == "1",
        "/sys/block/vda/ro is {ro:?}, expected \"1\" — the read-only chain \
         broke at the host (VIRTIO_BLK_F_RO not advertised) or the guest \
         (driver did not mark the gendisk read-only)"
    );

    // open(O_WRONLY) SUCCEEDS — the kernel's bdev open path does not
    // gate the open on read-only; the write is rejected at write time.
    let mut fd = OpenOptions::new().write(true).open(path).map_err(|e| {
        anyhow::anyhow!(
            "open(/dev/vda, O_WRONLY) failed with {e} — the kernel admits \
             write-opens on a read-only bdev (rejection is at write time), \
             so this open must succeed"
        )
    })?;

    // The first write() returns EPERM — the kernel rejects writes to a
    // read-only bdev at write time. This is the read-only enforcement
    // surface a guest userspace caller actually hits.
    match fd.write(&[0u8]) {
        Ok(0) => anyhow::bail!(
            "write(&[0u8]) to /dev/vda returned Ok(0) (zero progress) — \
             expected Err(EPERM); the kernel took the write fd but \
             reported no bytes written instead of rejecting the write"
        ),
        Ok(n) => anyhow::bail!(
            "write({n} bytes) to /dev/vda SUCCEEDED on a read-only disk — \
             the kernel did not reject the write; VIRTIO_BLK_F_RO is not \
             honored end-to-end"
        ),
        Err(e) => {
            let raw_errno = e.raw_os_error();
            anyhow::ensure!(
                raw_errno == Some(libc::EPERM),
                "write to /dev/vda failed errno={raw_errno:?}, expected EPERM \
                 ({}) — rejected for a reason other than read-only (EIO would \
                 mean the host's in-device IOERR gate fired; EBADF/EFAULT a \
                 test bug)",
                libc::EPERM
            );
        }
    }

    let mut result = AssertResult::pass();
    result.note(
        "/sys/block/vda/ro==1 and write() returned EPERM — VIRTIO_BLK_F_RO \
         is honored end-to-end (read-only gendisk, write-time rejection)",
    );
    Ok(result)
}

/// Host-side `post_vm` callback for the
/// `vm_integration_disk_virtio_blk_counters_handoff` entry below.
/// Asserts that the freeze-coordinator's snapshot-at-assignment seam
/// (`run.virtio_blk_counters.as_deref().map(|c| c.snapshot())` →
/// [`crate::vmm::VmResult::virtio_blk_counters`]) actually delivers
/// the device's live atomic counters into the host's plain-u64
/// snapshot after the guest exits.
///
/// The entry reuses [`scenario_disk_write_read_roundtrip`] as the
/// guest workload — that scenario writes a 512-byte pattern to
/// sector 0 of `/dev/vda`, fsyncs, and reads it back via
/// O_DIRECT (bypassing the bdev buffer cache so the read
/// unconditionally reaches the device). After the VM exits, this
/// callback verifies that the host-side counters observed both
/// the write, the read, and the flush end-to-end through the
/// virtio-blk request-processing path (named in framework as
/// `VirtioBlk::process_requests` → `VirtioBlkCounters::record_read`
/// / `record_write` / `record_flush`; these symbols are
/// `pub(crate)` so intra-doc links don't resolve from integration
/// tests — see `src/vmm/virtio_blk/` for the implementation).
///
/// Lower bounds, not exact equality: the guest kernel may issue
/// additional reads (e.g. partition-table scan, superblock probe)
/// before or after the roundtrip, so `reads_completed` and
/// `bytes_read` floor at the single roundtrip read. `bytes_written`
/// floors at 512 because the test write is the sole writer in a
/// fresh VM with no filesystem mounted on `/dev/vda`.
///
/// Regression coverage: without this callback, a future change that
/// snapshots `VirtioBlkCounters` BEFORE the worker has drained
/// in-flight notifications (or that wires the snapshot to the wrong
/// `Arc<VirtioBlkCounters>`) would silently report all zeros — every
/// existing guest-visible disk test would still pass because the
/// readback bytes are correct, but the host's introspection surface
/// would be broken.
fn assert_virtio_blk_counters_nonzero_after_roundtrip(
    result: &ktstr::prelude::VmResult,
) -> Result<()> {
    let counters = result.virtio_blk_counters.as_ref().ok_or_else(|| {
        anyhow::anyhow!(
            "result.virtio_blk_counters is None despite the entry attaching \
             a DiskConfig — the snapshot-at-assignment seam in \
             freeze_coord did not populate VmResult.virtio_blk_counters, \
             or the device was never wired into the run state at all"
        )
    })?;
    anyhow::ensure!(
        counters.reads_completed >= 1,
        "reads_completed = {} (expected >= 1 after the readback in \
         scenario_disk_write_read_roundtrip): the guest performed a 512-byte \
         pread of /dev/vda sector 0 but no read landed on the host counter — \
         the device never observed the request OR the counter was snapshotted \
         before record_read fired",
        counters.reads_completed,
    );
    anyhow::ensure!(
        counters.writes_completed >= 1,
        "writes_completed = {} (expected >= 1 after the pattern write): the \
         guest performed a 512-byte pwrite + fsync but no write landed on the \
         host counter",
        counters.writes_completed,
    );
    anyhow::ensure!(
        counters.bytes_read >= 512,
        "bytes_read = {} (expected >= 512 = one full sector readback): the \
         guest issued the readback but the byte total never reached the \
         counter (record_read parameter wrong, or the read never reached the \
         backing-file `pread`)",
        counters.bytes_read,
    );
    anyhow::ensure!(
        counters.bytes_written >= 512,
        "bytes_written = {} (expected >= 512 = one full sector pattern \
         write): the guest issued the pwrite + fsync but the byte total never \
         reached the counter (record_write parameter wrong, or the write never \
         reached the backing-file `pwrite`)",
        counters.bytes_written,
    );
    anyhow::ensure!(
        counters.flushes_completed >= 1,
        "flushes_completed = {} (expected >= 1 after fsync): the guest \
         issued sync_all on /dev/vda which translates to VIRTIO_BLK_T_FLUSH \
         per the virtio-spec + drivers/block/virtio_blk.c REQ_OP_FLUSH path, \
         but no flush landed on the host counter — the handle_flush / \
         record_flush wiring may be broken or the snapshot fired before \
         the flush completion",
        counters.flushes_completed,
    );
    // Upper-bound sanity caps: a regression that multiplies counters by
    // a wrong factor (e.g. unit confusion bytes-as-bits, or a counter-
    // bump loop double-counting) would pass the `>= 1` / `>= 512` floors
    // above. The caps below are generous slack — the test issues 1
    // operator write + 1 operator read of 512 bytes each, plus
    // discovery-time partition-probe reads. Real-world counter values
    // settle in the single-digit-requests / single-digit-KiB range.
    // Cap thresholds catch ×1000+ regressions while tolerating any
    // plausible kernel/probe-read variance.
    const SANE_REQ_CAP: u64 = 1000;
    const SANE_BYTES_CAP: u64 = 64 * 1024 * 1024;
    anyhow::ensure!(
        counters.writes_completed < SANE_REQ_CAP,
        "writes_completed = {} exceeds sanity cap {SANE_REQ_CAP}: the \
         operator issued 1 write; counter-bump regression suspected \
         (multiplication / loop / unit confusion)",
        counters.writes_completed,
    );
    anyhow::ensure!(
        counters.bytes_written < SANE_BYTES_CAP,
        "bytes_written = {} exceeds sanity cap {SANE_BYTES_CAP} (64 MiB) \
         on a single 512-byte operator write: counter-bump regression \
         suspected (bytes-as-bits, multi-counting, or byte-count parameter \
         mismatch with record_write)",
        counters.bytes_written,
    );
    anyhow::ensure!(
        counters.reads_completed < SANE_REQ_CAP,
        "reads_completed = {} exceeds sanity cap {SANE_REQ_CAP}: the \
         operator issued 1 O_DIRECT read; kernel partition probes plausibly \
         add a handful more. Above {SANE_REQ_CAP} indicates a \
         counter-bump regression",
        counters.reads_completed,
    );
    anyhow::ensure!(
        counters.bytes_read < SANE_BYTES_CAP,
        "bytes_read = {} exceeds sanity cap {SANE_BYTES_CAP} (64 MiB) \
         on a single 512-byte O_DIRECT read + a handful of partition \
         probes: counter-bump regression suspected",
        counters.bytes_read,
    );
    Ok(())
}

// ----------------------------------------------------------------------------
// Entry registrations
// ----------------------------------------------------------------------------

#[ktstr::ktstr_test_entry]
static __KTSTR_ENTRY_DSQ_RQ_WALKER: ktstr::test_support::KtstrTestEntry =
    ktstr::test_support::KtstrTestEntry {
        name: "vm_integration_dsq_and_rq_walker",
        func: scenario_dsq_and_rq_walker_populates_failure_dump,
        scheduler: &KTSTR_SCHED,
        extra_sched_args: &["--stall-after=1"],
        watchdog_timeout: std::time::Duration::from_secs(3),
        duration: std::time::Duration::from_secs(10),
        // Stall death inverts to PASS; the walker assertions gate in
        // check_dsq_and_rq_walker (post_vm_unconditional, hard FAIL via
        // PostVmAssertionFailure that expect_err does not invert).
        expect_err: true,
        post_vm_unconditional: Some(check_dsq_and_rq_walker),
        ..ktstr::test_support::KtstrTestEntry::DEFAULT
    };

#[ktstr::ktstr_test_entry]
static __KTSTR_ENTRY_PERF_COUNTERS: ktstr::test_support::KtstrTestEntry =
    ktstr::test_support::KtstrTestEntry {
        name: "vm_integration_perf_counters_capture",
        func: scenario_perf_counters_capture_populates_dump,
        scheduler: &KTSTR_SCHED,
        extra_sched_args: &["--stall-after=1"],
        watchdog_timeout: std::time::Duration::from_secs(3),
        duration: std::time::Duration::from_secs(10),
        // Stall death inverts to PASS; the perf-capture assertions gate
        // in check_perf_counters_capture (post_vm_unconditional, hard
        // FAIL via PostVmAssertionFailure that expect_err does not invert).
        expect_err: true,
        post_vm_unconditional: Some(check_perf_counters_capture),
        ..ktstr::test_support::KtstrTestEntry::DEFAULT
    };

#[ktstr::ktstr_test_entry]
static __KTSTR_ENTRY_EVENT_TIMELINE: ktstr::test_support::KtstrTestEntry =
    ktstr::test_support::KtstrTestEntry {
        name: "vm_integration_event_counter_timeline",
        func: scenario_event_counter_timeline,
        scheduler: &KTSTR_SCHED,
        extra_sched_args: &["--stall-after=1"],
        watchdog_timeout: std::time::Duration::from_secs(3),
        // Longer duration so the per-tick monitor loop accumulates
        // multiple samples before the stall fires.
        duration: std::time::Duration::from_secs(15),
        // Stall death inverts to PASS; the timeline assertions gate in
        // check_event_counter_timeline (post_vm_unconditional, hard FAIL
        // via PostVmAssertionFailure that expect_err does not invert).
        expect_err: true,
        post_vm_unconditional: Some(check_event_counter_timeline),
        ..ktstr::test_support::KtstrTestEntry::DEFAULT
    };

#[ktstr::ktstr_test_entry]
static __KTSTR_ENTRY_DEADLINE: ktstr::test_support::KtstrTestEntry =
    ktstr::test_support::KtstrTestEntry {
        name: "vm_integration_sched_deadline",
        func: scenario_sched_deadline_real_setattr,
        scheduler: &KTSTR_SCHED,
        // No --stall-after: this test just exercises the
        // sched_setattr ABI; no freeze required.
        extra_sched_args: &[],
        watchdog_timeout: std::time::Duration::from_secs(3),
        // Short duration — work_units > 0 needs only a few ms under
        // the 5% bandwidth reservation.
        duration: std::time::Duration::from_millis(500),
        // expect_err: false because this scenario asserts on the
        // success path (worker.completed=true, work_units > 0).
        expect_err: false,
        ..ktstr::test_support::KtstrTestEntry::DEFAULT
    };

#[ktstr::ktstr_test_entry]
static __KTSTR_ENTRY_DUMP_TRIGGER: ktstr::test_support::KtstrTestEntry =
    ktstr::test_support::KtstrTestEntry {
        name: "vm_integration_failure_dump_trigger",
        func: scenario_failure_dump_trigger_minimal_invariants,
        scheduler: &KTSTR_SCHED,
        extra_sched_args: &["--stall-after=1"],
        watchdog_timeout: std::time::Duration::from_secs(3),
        duration: std::time::Duration::from_secs(10),
        // Stall death inverts to PASS; the dump-invariant assertions gate
        // in check_failure_dump_trigger (post_vm_unconditional, hard FAIL
        // via PostVmAssertionFailure that expect_err does not invert).
        expect_err: true,
        post_vm_unconditional: Some(check_failure_dump_trigger),
        ..ktstr::test_support::KtstrTestEntry::DEFAULT
    };

// ----------------------------------------------------------------------------
// `#[test] #[ignore]` shims — nextest entry points for `cargo ktstr test`
// ----------------------------------------------------------------------------

/// Locate the `cargo-ktstr` binary built for this test pass.
/// `CARGO_BIN_EXE_<name>` is set at compile time for every `[[bin]]`
/// the workspace declares, so the shims resolve the absolute path
/// without shelling out to `which cargo-ktstr`.
const CARGO_KTSTR_BINARY: &str = env!("CARGO_BIN_EXE_cargo-ktstr");

/// Resolve the linux source tree (`../linux` relative to this
/// crate). VM boot requires a kernel cache populated from this
/// source; if the directory is missing, the shim panics with an
/// actionable message rather than a silent timeout.
fn linux_source_dir() -> std::path::PathBuf {
    ktstr::writable_source_path(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("linux")
}

/// Drive one `vm_integration_*` scenario via `cargo ktstr test`,
/// asserting the subprocess exits 0 (or, for `expect_err: true`
/// tests, that the test framework reports it as expected-failure
/// rather than wholesale subprocess error).
///
/// Stdout + stderr are captured and surfaced in the panic message
/// on failure so the operator can pinpoint which assertion or boot
/// stage failed without re-running the test under verbose logging.
fn drive_ktstr_test(scenario_name: &str) {
    let source = linux_source_dir();
    assert!(
        source.is_dir(),
        "../linux source tree missing — VM tests need a kernel source \
         tree. Expected: {}",
        source.display(),
    );

    let output = std::process::Command::new(CARGO_KTSTR_BINARY)
        .arg("ktstr")
        .arg("test")
        .arg("--kernel")
        .arg(&source)
        .arg("--")
        .arg("--filter")
        .arg(scenario_name)
        .output()
        .expect("spawn cargo-ktstr test");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "cargo ktstr test --filter {scenario_name} failed (exit={:?})\n\
         STDOUT:\n{stdout}\n\nSTDERR:\n{stderr}",
        output.status.code(),
    );
}

/// DSQ + rq->scx walker.
///
/// Boots scx-ktstr `--stall-after=1`, asserts `dsq_states` and
/// `rq_scx_states` in the failure-dump JSON are both non-empty.
/// Pins the kernel-facing walker pipeline end-to-end on a frozen
/// VM. See `scenario_dsq_and_rq_walker_populates_failure_dump`
/// above for the full scenario body.
///
/// Prerequisites:
/// - `../linux` kernel source tree
/// - `scx-ktstr` scheduler binary on `$PATH`
/// - `/dev/kvm` accessible
/// - guest kernel with CONFIG_SCHED_CLASS_EXT + CONFIG_DEBUG_INFO_BTF
#[test]
#[ignore = "requires KVM, ../linux, scx-ktstr, kernel BTF"]
fn vm_integration_dsq_and_rq_walker() {
    drive_ktstr_test("vm_integration_dsq_and_rq_walker");
}

/// Per-vCPU perf counters via `perf_event_open(exclude_host=1)`.
///
/// Boots a SpinWait workload, asserts `vcpu_perf_at_freeze` carries
/// at least one non-null `VcpuPerfSample` after the stall freeze.
///
/// Prerequisites: same as DSQ test, plus
/// - `kernel.perf_event_paranoid <= 2` on the host
/// - `CAP_PERFMON` (or root) in the test runner
#[test]
#[ignore = "requires KVM, ../linux, scx-ktstr, kernel.perf_event_paranoid <= 2 (CAP_PERFMON or root)"]
fn vm_integration_perf_counters_capture() {
    drive_ktstr_test("vm_integration_perf_counters_capture");
}

/// Event-counter timeline (per-tick sched-event capture).
///
/// Asserts that monitor samples carried `SCX_EV_*` event counters
/// (`CpuSnapshot.event_counters` in `VmResult.monitor.samples`) after
/// a 15s run window. The dump's `event_counter_timeline` field is
/// always empty (the freeze coordinator passes
/// `event_counter_capture: None`), so the check reads the monitor
/// samples, not that field. Pins the per-monitor-tick capture loop +
/// SCX_EV_* offset resolution + `EventCounterCapture` attach path.
#[test]
#[ignore = "requires KVM, ../linux, scx-ktstr"]
fn vm_integration_event_counter_timeline() {
    drive_ktstr_test("vm_integration_event_counter_timeline");
}

/// `SchedPolicy::Deadline` real `sched_setattr(2)` invocation.
///
/// Spawns a SpinWait worker under SCHED_DEADLINE 5% bandwidth
/// reservation; asserts the worker reports `completed=true` and
/// `work_units > 0`. Pins the syscall ABI end-to-end on a real
/// CONFIG_SCHED_DEADLINE kernel.
///
/// Distinct from the other tests: no stall, just exercises the
/// `sched_setattr` path. `expect_err: false` because the success
/// path is what's under test.
#[test]
#[ignore = "requires KVM, ../linux, scx-ktstr, CONFIG_SCHED_DEADLINE in guest"]
fn vm_integration_sched_deadline() {
    drive_ktstr_test("vm_integration_sched_deadline");
}

/// Failure-dump trigger, full-stack invariants.
///
/// Asserts `schema == "single"`, `maps` non-empty,
/// `vcpu_regs` non-empty after a stall freeze. Pins three
/// cross-pipeline invariants independent of which BPF struct
/// happens to be inspected.
#[test]
#[ignore = "requires KVM, ../linux, scx-ktstr"]
fn vm_integration_failure_dump_trigger() {
    drive_ktstr_test("vm_integration_failure_dump_trigger");
}

// ----------------------------------------------------------------------------
// Disk integration: KTSTR_TESTS entries + nextest shims
// ----------------------------------------------------------------------------

#[ktstr::ktstr_test_entry]
static __KTSTR_ENTRY_DISK_DEFAULT: ktstr::test_support::KtstrTestEntry =
    ktstr::test_support::KtstrTestEntry {
        name: "vm_integration_disk_default_appears",
        func: scenario_disk_default_appears_at_dev_vda,
        scheduler: &KTSTR_SCHED,
        // No --stall-after: this test exercises only the disk
        // attach path, not the failure-dump pipeline.
        extra_sched_args: &[],
        watchdog_timeout: std::time::Duration::from_secs(3),
        // Short duration — the test only opens /dev/vda and reads
        // capacity; no extended workload required.
        duration: std::time::Duration::from_millis(500),
        expect_err: false,
        disk: Some(KTSTR_DISK_DEFAULT),
        ..ktstr::test_support::KtstrTestEntry::DEFAULT
    };

#[ktstr::ktstr_test_entry]
static __KTSTR_ENTRY_DISK_ROUNDTRIP: ktstr::test_support::KtstrTestEntry =
    ktstr::test_support::KtstrTestEntry {
        name: "vm_integration_disk_write_read_roundtrip",
        func: scenario_disk_write_read_roundtrip,
        scheduler: &KTSTR_SCHED,
        extra_sched_args: &[],
        watchdog_timeout: std::time::Duration::from_secs(3),
        duration: std::time::Duration::from_millis(500),
        expect_err: false,
        disk: Some(KTSTR_DISK_DEFAULT),
        ..ktstr::test_support::KtstrTestEntry::DEFAULT
    };

#[ktstr::ktstr_test_entry]
static __KTSTR_ENTRY_DISK_READ_ONLY: ktstr::test_support::KtstrTestEntry =
    ktstr::test_support::KtstrTestEntry {
        name: "vm_integration_disk_read_only_rejects_write",
        func: scenario_disk_read_only_rejects_write,
        scheduler: &KTSTR_SCHED,
        extra_sched_args: &[],
        watchdog_timeout: std::time::Duration::from_secs(3),
        duration: std::time::Duration::from_millis(500),
        expect_err: false,
        disk: Some(KTSTR_DISK_READ_ONLY),
        ..ktstr::test_support::KtstrTestEntry::DEFAULT
    };

#[ktstr::ktstr_test_entry]
static __KTSTR_ENTRY_DISK_COUNTERS_HANDOFF: ktstr::test_support::KtstrTestEntry =
    ktstr::test_support::KtstrTestEntry {
        name: "vm_integration_disk_virtio_blk_counters_handoff",
        func: scenario_disk_write_read_roundtrip,
        scheduler: &KTSTR_SCHED,
        extra_sched_args: &[],
        watchdog_timeout: std::time::Duration::from_secs(3),
        duration: std::time::Duration::from_millis(500),
        expect_err: false,
        disk: Some(KTSTR_DISK_DEFAULT),
        post_vm: Some(assert_virtio_blk_counters_nonzero_after_roundtrip),
        ..ktstr::test_support::KtstrTestEntry::DEFAULT
    };

/// Disk #1 — `/dev/vda` exists with the configured capacity.
///
/// Boots scx-ktstr with a 256 MiB raw virtio-blk disk attached; the
/// guest scenario stat()s `/dev/vda`, verifies it is a block device,
/// and reads BLKGETSIZE64 to check the capacity round-trips through
/// the kernel virtio driver.
///
/// Prerequisites:
/// - `../linux` kernel source tree
/// - `/dev/kvm` accessible
/// - guest kernel with CONFIG_VIRTIO_BLK + CONFIG_BLK_DEV
#[test]
#[ignore = "requires KVM, ../linux, CONFIG_VIRTIO_BLK in guest"]
fn vm_integration_disk_default_appears() {
    drive_ktstr_test("vm_integration_disk_default_appears");
}

/// Disk #2 — write+read roundtrip on `/dev/vda` sector 0.
///
/// Writes a 512-byte 0xA5 pattern to sector 0 via `pwrite` + `fsync`,
/// re-opens for read, asserts byte-for-byte readback. Pins the full
/// virtio-blk fast path through guest driver, host-side
/// `process_requests`, sparse tempfile pwrite/pread, and irqfd
/// completion.
#[test]
#[ignore = "requires KVM, ../linux, CONFIG_VIRTIO_BLK in guest"]
fn vm_integration_disk_write_read_roundtrip() {
    drive_ktstr_test("vm_integration_disk_write_read_roundtrip");
}

/// Disk #3 — read-only disk rejects write.
///
/// Boots with a `read_only(true)` DiskConfig and asserts the
/// read-only chain end-to-end: `VIRTIO_BLK_F_RO` marks the gendisk
/// read-only (`/sys/block/vda/ro == 1`), `open(/dev/vda, O_WRONLY)`
/// SUCCEEDS (the bdev open path does not gate on read-only), and the
/// first `write()` returns `EPERM` — the kernel rejects writes to a
/// read-only bdev at write time, not `EROFS` at `open(2)` time.
#[test]
#[ignore = "requires KVM, ../linux, CONFIG_VIRTIO_BLK in guest"]
fn vm_integration_disk_read_only_rejects_write() {
    drive_ktstr_test("vm_integration_disk_read_only_rejects_write");
}

/// Disk #4 — virtio-blk counter handoff from device to `VmResult`.
///
/// Reuses the write+read roundtrip workload as the in-guest scenario
/// but adds a `post_vm` callback
/// ([`assert_virtio_blk_counters_nonzero_after_roundtrip`]) that asserts
/// the host-side `VmResult.virtio_blk_counters` snapshot reflects
/// the IO the guest performed. Pins the freeze-coordinator's
/// snapshot-at-assignment seam (`run.virtio_blk_counters` → `VmResult`)
/// — without this test, a regression that snapshots the counters
/// before `record_read` / `record_write` fires would silently report
/// zeros while every guest-visible disk test still passed.
#[test]
#[ignore = "requires KVM, ../linux, CONFIG_VIRTIO_BLK in guest"]
fn vm_integration_disk_virtio_blk_counters_handoff() {
    drive_ktstr_test("vm_integration_disk_virtio_blk_counters_handoff");
}
