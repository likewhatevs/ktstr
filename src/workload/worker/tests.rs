//! Worker-side tests: worker_main helpers (clock, schedstat, IO
//! backings, set_sched_policy, reservoir_push, matrix_multiply,
//! resolve_alu_width). Co-located with their production code in
//! `worker/mod.rs`.

#![cfg(test)]
#![allow(unused_imports)]

use super::super::affinity::*;
use super::super::config::*;
use super::super::spawn::*;
use super::super::types::*;
use super::*;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

/// [`MAX_WAKE_SAMPLES`] MUST be 100_000 — the value
/// `doc/guide/src/architecture/workers.md` cites verbatim ("clamped
/// to at most 100_000 entries (`MAX_WAKE_SAMPLES`)"). The doc cite
/// is operator-load-bearing (sets per-worker memory expectations for
/// long-running scenarios); pinning the constant at the source
/// catches a silent rot where the const changes but the doc doesn't.
///
/// A regression that bumped the cap to 1_000_000 — perhaps to chase
/// a percentile-precision request — would 10x per-worker memory
/// silently while leaving the doc claim stale. The pin trips with a
/// clear assertion message so the developer who changed the cap
/// updates the doc at the same time.
#[test]
fn max_wake_samples_pins_doc_value() {
    assert_eq!(
        MAX_WAKE_SAMPLES, 100_000,
        "MAX_WAKE_SAMPLES MUST stay 100_000 to match the doc cite in \
         doc/guide/src/architecture/workers.md (\"clamped to at most \
         100_000 entries (`MAX_WAKE_SAMPLES`)\"). If you intentionally \
         changed the cap, update both this assertion AND the doc \
         cite — they are pinned together to prevent silent drift.",
    );
}

/// [`defaults::PAGE_FAULT_CHURN_REGION_KIB`] /
/// [`defaults::PAGE_FAULT_CHURN_TOUCHES_PER_CYCLE`] /
/// [`defaults::PAGE_FAULT_CHURN_SPIN_ITERS`] MUST stay at the values
/// `doc/guide/src/architecture/workers.md` cites verbatim
/// ("region_kib=4096, touches_per_cycle=256, spin_iters=64"). The doc
/// cite gates operator expectations of per-iteration workload cost;
/// a silent change to any of the three would leave the doc stale and
/// mislead readers tuning page-fault-driven test scenarios.
#[test]
fn page_fault_churn_defaults_pin_doc_values() {
    assert_eq!(
        defaults::PAGE_FAULT_CHURN_REGION_KIB,
        4096,
        "PAGE_FAULT_CHURN_REGION_KIB MUST stay 4096 to match the doc \
         cite in doc/guide/src/architecture/workers.md \
         (\"region_kib=4096\"). Update both this assertion AND the doc \
         cite if you intentionally changed the default.",
    );
    assert_eq!(
        defaults::PAGE_FAULT_CHURN_TOUCHES_PER_CYCLE,
        256,
        "PAGE_FAULT_CHURN_TOUCHES_PER_CYCLE MUST stay 256 to match \
         the doc cite in doc/guide/src/architecture/workers.md \
         (\"touches_per_cycle=256\"). Update both this assertion AND \
         the doc cite if you intentionally changed the default.",
    );
    assert_eq!(
        defaults::PAGE_FAULT_CHURN_SPIN_ITERS,
        64,
        "PAGE_FAULT_CHURN_SPIN_ITERS MUST stay 64 to match the doc \
         cite in doc/guide/src/architecture/workers.md \
         (\"spin_iters=64\"). Update both this assertion AND the doc \
         cite if you intentionally changed the default.",
    );
}

/// `clock_gettime_ns(CLOCK_MONOTONIC)` must never observe time
/// moving backwards between two sequential calls on the same
/// thread. Pins the non-decreasing contract the wake-latency
/// reservoirs depend on: the messenger stamps `wake_ns` into
/// shared memory and the worker subtracts to compute
/// `now_ns - wake_ns`; a backward step would saturate to zero
/// in the subtractor and silently discard a valid sample, or
/// (without the saturator) wrap to `u64::MAX`.
///
/// A 2-sample test would miss a backward step that only
/// appears under load; the 1000-sample tight loop here burns
/// a few microseconds of CPU and catches any regression that
/// makes the clock non-monotonic under reasonable contention
/// (timer drift on a virtualised guest, or a helper swap
/// from `CLOCK_MONOTONIC` to `CLOCK_REALTIME` which is NOT
/// monotonic). Every adjacent pair in the 999-element diff
/// list is checked for non-decreasing order so a mid-run
/// regression is localised to the offending index, not just
/// "some pair somewhere".
#[test]
fn clock_gettime_ns_monotonic_non_decreasing() {
    const N: usize = 1000;
    let samples: Vec<u64> = (0..N)
        .map(|i| {
            clock_gettime_ns(libc::CLOCK_MONOTONIC).unwrap_or_else(|| {
                panic!(
                    "CLOCK_MONOTONIC must be readable on any Linux host; \
                     sample {i}/{N} returned None"
                )
            })
        })
        .collect();
    for i in 1..N {
        assert!(
            samples[i] >= samples[i - 1],
            "CLOCK_MONOTONIC went backwards at sample {i}: \
             prev={prev} curr={curr} (delta={delta})",
            prev = samples[i - 1],
            curr = samples[i],
            delta = samples[i - 1] - samples[i],
        );
    }
}
// -- matrix_multiply --

#[test]
fn matrix_multiply_1x1_produces_product() {
    // Size=1: A=[a], B=[b], expected C=[a*b]. The `black_box` calls
    // prevent constant folding, so the test directly exercises the
    // wrapping_mul path without any compiler optimization eating
    // the multiplication.
    let mut data = vec![0u64; 3];
    data[0] = 3; // A
    data[1] = 5; // B
    let mut work_units = 0u64;
    matrix_multiply(&mut data, 1, &mut work_units);
    assert_eq!(data[2], 15, "C = A * B for 1x1 matrix");
    // Read-back sink consumed C[0] (= 15) into work_units.
    assert_eq!(work_units, 15, "post-loop sink folds C[0] into work_units");
}
#[test]
fn matrix_multiply_2x2_against_reference() {
    // A = [[1, 2], [3, 4]], B = [[5, 6], [7, 8]]
    // C = A * B = [[19, 22], [43, 50]]
    let size = 2;
    let stride = size * size;
    let mut data = vec![0u64; 3 * stride];
    data[0] = 1;
    data[1] = 2;
    data[2] = 3;
    data[3] = 4;
    data[stride] = 5;
    data[stride + 1] = 6;
    data[stride + 2] = 7;
    data[stride + 3] = 8;
    let mut work_units = 0u64;
    matrix_multiply(&mut data, size, &mut work_units);
    assert_eq!(data[2 * stride], 19);
    assert_eq!(data[2 * stride + 1], 22);
    assert_eq!(data[2 * stride + 2], 43);
    assert_eq!(data[2 * stride + 3], 50);
}
#[test]
fn matrix_multiply_3x3_diagonal() {
    // Identity-like: A = diag(2, 3, 5), B = diag(1, 1, 1) = I.
    // Expected C = A = diag(2, 3, 5).
    let size = 3;
    let stride = size * size;
    let mut data = vec![0u64; 3 * stride];
    data[0] = 2;
    data[4] = 3;
    data[8] = 5;
    data[stride] = 1;
    data[stride + 4] = 1;
    data[stride + 8] = 1;
    let mut work_units = 0u64;
    matrix_multiply(&mut data, size, &mut work_units);
    let c = &data[2 * stride..3 * stride];
    // Diagonal entries carry A's diagonal because B = I.
    assert_eq!(c[0], 2);
    assert_eq!(c[4], 3);
    assert_eq!(c[8], 5);
    // All 6 off-diagonal entries must be 0 for A*I. Sparse
    // coverage (just c[1], c[3]) left 4 positions unverified,
    // which would mask a transposition bug that mis-writes
    // rows/columns of an identity product — this assertion
    // fingerprints the full matrix identity.
    assert_eq!(c[1], 0);
    assert_eq!(c[2], 0);
    assert_eq!(c[3], 0);
    assert_eq!(c[5], 0);
    assert_eq!(c[6], 0);
    assert_eq!(c[7], 0);
}
#[test]
#[cfg(debug_assertions)]
#[should_panic(expected = "assertion")]
fn matrix_multiply_mismatched_len_panics_in_debug() {
    // debug_assert_eq!(data.len(), 3 * size * size) guards the
    // bounds contract. Under cfg(debug_assertions) this panics.
    // Release builds skip the assert (no panic), so the test
    // itself is gated on `cfg(debug_assertions)` — otherwise
    // `cargo nextest run --release` would run the test expecting
    // a panic the release binary can't raise.
    let mut data = vec![0u64; 5]; // 3 * 2 * 2 = 12, so 5 is wrong.
    let mut work_units = 0u64;
    matrix_multiply(&mut data, 2, &mut work_units);
}
// -- RAII unit tests for the IO scratch / backing wrappers --
//
// Pin the contracts each Drop documents: DirectIoBuf returns a
// logical-block-aligned heap buffer, IoBacking unlinks its
// tempfile path on Drop (only when one was set), and
// PhaseIoTempfile unlinks unconditionally. The `ensure_*`
// helpers must be lazy-init — second call returns the same fd /
// pointer rather than re-opening / re-allocating.

/// `DirectIoBuf::alloc` returns a 4 KiB buffer aligned to the
/// logical-block boundary `O_DIRECT` requires. Writing the full
/// region and reading it back proves the allocation is mapped
/// for both reads and writes; the 0xAA pattern is arbitrary,
/// any non-zero bit pattern that survives the round trip is
/// sufficient.
#[test]
fn direct_io_buf_alloc_aligned() {
    let buf = DirectIoBuf::alloc()
        .expect("DirectIoBuf::alloc must succeed under normal allocator pressure");
    let addr = buf.as_ptr() as usize;
    assert_eq!(
        addr % IO_BLOCK_SIZE,
        0,
        "DirectIoBuf must be IO_BLOCK_SIZE-aligned (got addr={addr:#x})"
    );
    // SAFETY: alloc returned a non-null pointer to IO_BLOCK_SIZE
    // bytes of writable heap; the slice is fully covered by the
    // allocation, no aliasing live, and the pointer is unique
    // to this test scope.
    let slice = unsafe { std::slice::from_raw_parts_mut(buf.as_ptr(), IO_BLOCK_SIZE) };
    slice.fill(0xAA);
    assert!(
        slice.iter().all(|&b| b == 0xAA),
        "round-trip pattern must persist across the buffer",
    );
    // Drop runs at end of scope and dealloc's the layout. The
    // test itself can't observe dealloc; it only proves no
    // panic / no UB on the freed pointer.
}
/// `IoBacking` with `tempfile_path: Some(_)` unlinks the path on
/// Drop. Constructs a real on-disk file with a unique name,
/// drops the wrapper inside a scope, then asserts the path no
/// longer exists.
#[test]
fn io_backing_tempfile_unlinked_on_drop() {
    let path = std::env::temp_dir()
        .join(format!(
            "ktstr_iobacking_unlink_{}_{}",
            std::process::id(),
            unsafe { libc::syscall(libc::SYS_gettid) },
        ))
        .to_string_lossy()
        .to_string();
    let file = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(true)
        .open(&path)
        .expect("create real tempfile for IoBacking test");
    assert!(
        std::path::Path::new(&path).exists(),
        "precondition: file exists"
    );
    {
        let _backing = IoBacking {
            file,
            capacity_bytes: 0,
            tempfile_path: Some(path.clone()),
        };
        // Path still exists inside scope.
        assert!(std::path::Path::new(&path).exists());
    }
    assert!(
        !std::path::Path::new(&path).exists(),
        "IoBacking::Drop must unlink {path}",
    );
}
/// `IoBacking` with `tempfile_path: None` (the `/dev/vda` case)
/// must NOT call `remove_file` — block devices are never
/// deleted. Drop in this shape only closes the file fd. Use a
/// host tempfile as the backing fd so the test is self-contained
/// (running outside a VM where /dev/vda exists), but pass
/// `tempfile_path: None` to exercise the "block device" arm.
#[test]
fn io_backing_none_path_no_unlink() {
    let path = std::env::temp_dir()
        .join(format!(
            "ktstr_iobacking_nounlink_{}_{}",
            std::process::id(),
            unsafe { libc::syscall(libc::SYS_gettid) },
        ))
        .to_string_lossy()
        .to_string();
    let file = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(true)
        .open(&path)
        .expect("create stand-in for /dev/vda");
    {
        let _backing = IoBacking {
            file,
            capacity_bytes: 0,
            tempfile_path: None,
        };
        // Drop fires here.
    }
    // File still exists because tempfile_path was None.
    assert!(
        std::path::Path::new(&path).exists(),
        "IoBacking::Drop must NOT unlink when tempfile_path is None",
    );
    // Cleanup the stand-in we created for the test.
    let _ = std::fs::remove_file(&path);
}
/// `PhaseIoTempfile::Drop` unconditionally unlinks `path`. Same
/// shape as the IoBacking test, simpler invariants (no
/// optional path).
#[test]
fn phase_io_tempfile_unlinked_on_drop() {
    let path = std::env::temp_dir()
        .join(format!(
            "ktstr_phaseio_unlink_{}_{}",
            std::process::id(),
            unsafe { libc::syscall(libc::SYS_gettid) },
        ))
        .to_string_lossy()
        .to_string();
    let file = std::fs::OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(&path)
        .expect("create real tempfile for PhaseIoTempfile test");
    assert!(std::path::Path::new(&path).exists(), "precondition");
    {
        let _tf = PhaseIoTempfile {
            file,
            path: path.clone(),
        };
    }
    assert!(
        !std::path::Path::new(&path).exists(),
        "PhaseIoTempfile::Drop must unlink {path}",
    );
}
/// `ensure_io_disk` is lazy-init — calling it twice on the same
/// `Option<IoBacking>` slot opens the backing once and returns
/// the same fd on the second call. Compares `as_raw_fd()` across
/// both calls.
#[test]
fn ensure_io_disk_lazy_init() {
    use std::os::unix::io::AsRawFd;
    let tid: libc::pid_t = unsafe { libc::syscall(libc::SYS_gettid) as libc::pid_t };
    let mut io_disk: Option<IoBacking> = None;
    // First call: opens the backing.
    assert!(
        ensure_io_disk(&mut io_disk, 0, tid),
        "first ensure_io_disk must succeed (host can open tempfile fallback)",
    );
    let fd1 = io_disk
        .as_ref()
        .expect("io_disk Some after first call")
        .file
        .as_raw_fd();
    // Second call: must be a no-op, return the same fd.
    assert!(ensure_io_disk(&mut io_disk, 0, tid));
    let fd2 = io_disk.as_ref().unwrap().file.as_raw_fd();
    assert_eq!(
        fd1, fd2,
        "ensure_io_disk must be lazy-init — second call must not re-open",
    );
    // Drop unlinks the tempfile (if fallback path was used).
}
/// `ensure_io_buf` is lazy-init — calling it twice on the same
/// `Option<DirectIoBuf>` allocates once and returns the same
/// pointer on the second call.
#[test]
fn ensure_io_buf_lazy_init() {
    let mut io_buf: Option<DirectIoBuf> = None;
    assert!(
        ensure_io_buf(&mut io_buf),
        "first ensure_io_buf must succeed under normal allocator pressure",
    );
    let ptr1 = io_buf
        .as_ref()
        .expect("io_buf Some after first call")
        .as_ptr();
    assert!(ensure_io_buf(&mut io_buf));
    let ptr2 = io_buf.as_ref().unwrap().as_ptr();
    assert_eq!(
        ptr1, ptr2,
        "ensure_io_buf must be lazy-init — second call must not re-allocate",
    );
}
#[test]
fn thread_cpu_time_positive() {
    // Do some work so CPU time is non-zero
    let mut x = 0u64;
    for i in 0..100_000 {
        x = x.wrapping_add(i);
    }
    std::hint::black_box(x);
    let t = super::thread_cpu_time_ns();
    assert!(t > 0);
}
#[test]
fn set_sched_policy_normal_succeeds() {
    let pid: libc::pid_t = unsafe { libc::getpid() };
    let result = set_sched_policy(pid, SchedPolicy::Normal);
    assert!(result.is_ok());
}
#[test]
#[ignore]
fn set_sched_policy_fifo_returns_result() {
    let pid: libc::pid_t = unsafe { libc::getpid() };
    let result = set_sched_policy(pid, SchedPolicy::Fifo(1));
    // SCHED_FIFO requires CAP_SYS_NICE; succeeds when the runner holds it.
    assert!(
        result.is_ok(),
        "SCHED_FIFO should succeed with CAP_SYS_NICE"
    );
    restore_normal(pid);
}
#[test]
#[ignore]
fn set_sched_policy_rr_returns_result() {
    let pid: libc::pid_t = unsafe { libc::getpid() };
    let result = set_sched_policy(pid, SchedPolicy::RoundRobin(1));
    // SCHED_RR requires CAP_SYS_NICE; succeeds when the runner holds it.
    assert!(result.is_ok(), "SCHED_RR should succeed with CAP_SYS_NICE");
    restore_normal(pid);
}
// -- SchedPolicy variants --

/// Restore SCHED_NORMAL via the raw syscall. `set_sched_policy(Normal)`
/// is a no-op, so tests that change policy must use this to restore.
fn restore_normal(pid: libc::pid_t) {
    let param = libc::sched_param { sched_priority: 0 };
    unsafe { libc::sched_setscheduler(pid, libc::SCHED_OTHER, &param) };
}
#[test]
fn set_sched_policy_batch_returns_valid_result() {
    let pid: libc::pid_t = unsafe { libc::getpid() };
    let result = set_sched_policy(pid, SchedPolicy::Batch);
    // SCHED_BATCH does NOT require CAP_SYS_NICE:
    // `user_check_sched_setscheduler` routes only rt_policy /
    // dl_policy / negative-nice / leaving-IDLE through
    // req_priv; a fair-policy → fair-policy transition that
    // does not reduce nice never reaches the capable() check.
    // `scx_check_setscheduler` (kernel/sched/ext.c) does not
    // reject BATCH either — it only rejects transitions INTO
    // SCHED_EXT when `p->scx.disallow` is set, which BATCH is
    // not. Failure is therefore expected only on environments
    // that introduce extra LSM / security-module gates; the
    // test tolerates both outcomes.
    match result {
        Ok(()) => {
            let pol = unsafe { libc::sched_getscheduler(pid) };
            // Under sched_ext switch-all (`task_should_scx`
            // returns true for any policy when
            // `scx_switching_all` is set), `__setscheduler_class`
            // routes BATCH to `ext_sched_class`. Reading back
            // via `sched_getscheduler` returns the requested
            // policy regardless — this just sanity-checks the
            // syscall returned a non-negative policy id.
            assert!(
                pol >= 0,
                "sched_getscheduler must return a valid policy, got {pol}",
            );
            restore_normal(pid);
        }
        Err(ref e) => {
            let msg = format!("{e:#}");
            assert!(
                msg.contains("sched_setscheduler"),
                "error must name the syscall: {msg}"
            );
        }
    }
}
#[test]
fn set_sched_policy_idle_returns_valid_result() {
    let pid: libc::pid_t = unsafe { libc::getpid() };
    let result = set_sched_policy(pid, SchedPolicy::Idle);
    // SCHED_IDLE does NOT require CAP_SYS_NICE for *entering*
    // IDLE: `user_check_sched_setscheduler` gates the
    // IDLE-related capability check on `task_has_idle_policy(p)
    // && !idle_policy(policy)` — i.e. CAP_SYS_NICE is required
    // only when *leaving* SCHED_IDLE for a non-idle class
    // without RLIMIT_NICE permission, not when entering it.
    // `scx_check_setscheduler` (kernel/sched/ext.c) does not
    // reject IDLE either — same reasoning as the BATCH test
    // above. Failure is expected only on environments with
    // extra LSM / security-module gates.
    match result {
        Ok(()) => {
            let pol = unsafe { libc::sched_getscheduler(pid) };
            // Same switch-all reasoning as the BATCH test —
            // IDLE routes to `ext_sched_class` under switch-all
            // but the syscall return is the requested policy id.
            assert!(
                pol >= 0,
                "sched_getscheduler must return a valid policy, got {pol}",
            );
            restore_normal(pid);
        }
        Err(ref e) => {
            let msg = format!("{e:#}");
            assert!(
                msg.contains("sched_setscheduler"),
                "error must name the syscall: {msg}"
            );
        }
    }
}
// -- SCHED_DEADLINE validation tests --
//
// The five rejection tests below exercise the structural
// pre-validation that `set_sched_policy` performs before
// issuing the `sched_setattr` syscall. Each invariant mirrors
// a `__checkparam_dl` clause (`kernel/sched/deadline.c`); the
// tests pin user-space rejection so a malformed `Deadline`
// surfaces a named field rather than a generic kernel
// `EINVAL`. None of these tests require `CAP_SYS_NICE`
// because the bail!s fire before the syscall.

/// `deadline == Duration::ZERO` must be rejected:
/// `__checkparam_dl` returns false on `attr->sched_deadline ==
/// 0`. The runtime floor is satisfied here so the failure
/// pins the zero-deadline check, not the DL_SCALE check.
#[test]
fn set_sched_policy_deadline_zero_deadline_rejected() {
    let pid: libc::pid_t = unsafe { libc::getpid() };
    let result = set_sched_policy(
        pid,
        SchedPolicy::Deadline {
            runtime: Duration::from_nanos(1024),
            deadline: Duration::ZERO,
            period: Duration::from_nanos(1_000_000),
        },
    );
    let err = result.expect_err("zero deadline must be rejected");
    let msg = format!("{err:#}");
    assert!(
        msg.contains("deadline"),
        "error must name deadline field: {msg}"
    );
    assert!(
        msg.contains("must be > 0") || msg.contains("zero"),
        "error must explain zero rejection: {msg}"
    );
}
/// `runtime` shorter than 1024 ns must be rejected per the
/// `DL_SCALE` floor in `__checkparam_dl`.
#[test]
fn set_sched_policy_deadline_runtime_below_dl_scale_rejected() {
    let pid: libc::pid_t = unsafe { libc::getpid() };
    let result = set_sched_policy(
        pid,
        SchedPolicy::Deadline {
            runtime: Duration::from_nanos(1023),
            deadline: Duration::from_nanos(100_000),
            period: Duration::from_nanos(1_000_000),
        },
    );
    let err = result.expect_err("runtime below DL_SCALE must be rejected");
    let msg = format!("{err:#}");
    assert!(
        msg.contains("runtime"),
        "error must name runtime field: {msg}"
    );
    assert!(
        msg.contains("DL_SCALE") || msg.contains("1024"),
        "error must reference the floor: {msg}"
    );
}
/// `runtime > deadline` must be rejected per the
/// `runtime <= deadline` clause of `__checkparam_dl`.
#[test]
fn set_sched_policy_deadline_runtime_exceeds_deadline_rejected() {
    let pid: libc::pid_t = unsafe { libc::getpid() };
    let result = set_sched_policy(
        pid,
        SchedPolicy::Deadline {
            runtime: Duration::from_nanos(200_000),
            deadline: Duration::from_nanos(100_000),
            period: Duration::from_nanos(1_000_000),
        },
    );
    let err = result.expect_err("runtime > deadline must be rejected");
    let msg = format!("{err:#}");
    assert!(
        msg.contains("runtime") && msg.contains("deadline"),
        "error must name both fields: {msg}"
    );
}
/// `deadline > period` must be rejected when `period` is
/// non-zero. Pairs with
/// `set_sched_policy_deadline_period_zero_passes_validation`
/// which proves the gate is conditional on a non-zero period.
#[test]
fn set_sched_policy_deadline_deadline_exceeds_period_rejected() {
    let pid: libc::pid_t = unsafe { libc::getpid() };
    let result = set_sched_policy(
        pid,
        SchedPolicy::Deadline {
            runtime: Duration::from_nanos(1024),
            deadline: Duration::from_nanos(2_000_000),
            period: Duration::from_nanos(1_000_000),
        },
    );
    let err = result.expect_err("deadline > period must be rejected");
    let msg = format!("{err:#}");
    assert!(
        msg.contains("deadline") && msg.contains("period"),
        "error must name both fields: {msg}"
    );
}
/// A `deadline` whose nanosecond count exceeds `i64::MAX` must
/// be rejected. The kernel's `__checkparam_dl` clause `if
/// (attr->sched_deadline & (1ULL << 63)) return false;`
/// requires bit 63 to be clear; `duration_to_kernel_ns`
/// enforces this as a single i64::MAX overflow check on
/// `Duration::as_nanos()` (u128). The error message names the
/// offending field via the `field` argument so the diagnostic
/// points at `deadline` and not `runtime`/`period`.
#[test]
fn set_sched_policy_deadline_top_bit_set_rejected() {
    let pid: libc::pid_t = unsafe { libc::getpid() };
    let result = set_sched_policy(
        pid,
        SchedPolicy::Deadline {
            runtime: Duration::from_nanos(1024),
            // 1e12 seconds = 1e21 ns >> i64::MAX (≈ 9.2e18 ns)
            // — guaranteed to trip the overflow guard. Picked
            // far above the threshold so any future tweak to
            // the constraint still fires this test.
            deadline: Duration::from_secs(1_000_000_000_000),
            period: Duration::from_nanos(1_000_000),
        },
    );
    let err = result.expect_err("deadline exceeding i64::MAX must be rejected");
    let msg = format!("{err:#}");
    assert!(
        msg.contains("deadline") && (msg.contains("i64::MAX") || msg.contains("63 bits")),
        "error must name deadline field and the bit-63 / i64::MAX bound: {msg}"
    );
    // Per-field message: must NOT name `period` since only
    // `deadline` overflowed. `period` ordering matters —
    // `duration_to_kernel_ns` is called runtime → deadline →
    // period, so a deadline overflow short-circuits before
    // period is touched.
    assert!(
        !msg.contains("period"),
        "deadline-only overflow error must not mention period: {msg}"
    );
}
/// Happy-path: a structurally valid `Deadline` with
/// `period == Duration::ZERO` reaches the `sched_setattr`
/// syscall. The kernel substitutes `deadline` for the period
/// in this case (see `if (!period) period = attr->sched_deadline;`
/// in `__checkparam_dl`). Without `CAP_SYS_NICE` the syscall
/// fails with EPERM at the kernel-side capability check;
/// either Ok(()) or an Err whose message names
/// `sched_setattr` confirms we cleared the user-space
/// pre-validation. Marked `#[ignore]` so unprivileged CI
/// doesn't see EPERM as a hard failure — runners with
/// CAP_SYS_NICE can opt in.
#[test]
#[ignore]
fn set_sched_policy_deadline_period_zero_passes_validation() {
    let pid: libc::pid_t = unsafe { libc::getpid() };
    let result = set_sched_policy(
        pid,
        SchedPolicy::Deadline {
            runtime: Duration::from_nanos(1024),
            deadline: Duration::from_nanos(200_000),
            period: Duration::ZERO,
        },
    );
    match result {
        Ok(()) => {
            // Restore SCHED_NORMAL so the test process leaves
            // its run with default policy.
            restore_normal(pid);
        }
        Err(e) => {
            let msg = format!("{e:#}");
            assert!(
                msg.contains("sched_setattr"),
                "validation must have passed (error from kernel must name sched_setattr): {msg}"
            );
        }
    }
}
// -- reservoir_push tests --

#[test]
fn reservoir_push_empty_buf() {
    let mut buf = Vec::new();
    let mut count = 0u64;
    reservoir_push(&mut buf, &mut count, 42, 10);
    assert_eq!(buf, vec![42]);
    assert_eq!(count, 1);
}
#[test]
fn reservoir_push_under_cap() {
    let mut buf = Vec::new();
    let mut count = 0u64;
    for i in 0..5 {
        reservoir_push(&mut buf, &mut count, i * 100, 10);
    }
    assert_eq!(buf.len(), 5);
    assert_eq!(count, 5);
    assert_eq!(buf, vec![0, 100, 200, 300, 400]);
}
#[test]
fn reservoir_push_at_cap() {
    let mut buf = Vec::new();
    let mut count = 0u64;
    for i in 0..10 {
        reservoir_push(&mut buf, &mut count, i, 10);
    }
    assert_eq!(buf.len(), 10);
    assert_eq!(count, 10);
    // All values should be present since we're exactly at cap.
    for i in 0..10 {
        assert!(buf.contains(&i), "missing {i}");
    }
}
#[test]
fn reservoir_push_over_cap_maintains_size() {
    let mut buf = Vec::new();
    let mut count = 0u64;
    let cap = 5;
    for i in 0..1000 {
        reservoir_push(&mut buf, &mut count, i, cap);
    }
    assert_eq!(buf.len(), cap);
    assert_eq!(count, 1000);
}
#[test]
fn reservoir_push_uniform_sampling() {
    // Statistical test: push 10000 values into cap=100 reservoir.
    // Each value should have roughly equal probability of being present.
    // We test that the reservoir contains values from the full range.
    let mut buf = Vec::new();
    let mut count = 0u64;
    let cap = 100;
    let total = 10_000u64;
    for i in 0..total {
        reservoir_push(&mut buf, &mut count, i, cap);
    }
    assert_eq!(buf.len(), cap);
    assert_eq!(count, total);
    // The reservoir should contain values from different parts of the range.
    let has_early = buf.iter().any(|&v| v < total / 4);
    let has_late = buf.iter().any(|&v| v > total * 3 / 4);
    assert!(has_early, "reservoir should contain early values");
    assert!(has_late, "reservoir should contain late values");
}
#[test]
fn reservoir_push_cap_zero() {
    // Zero-capacity reservoir: buf.len() < 0 is never true (usize),
    // falls through to else branch where random_range(0..1) returns 0,
    // and 0 < 0 is false — sample is discarded.
    let mut buf = Vec::new();
    let mut count = 0u64;
    for i in 0..10 {
        reservoir_push(&mut buf, &mut count, i, 0);
    }
    assert!(buf.is_empty(), "cap=0 should never store samples");
    assert_eq!(count, 10, "count incremented regardless");
}
#[test]
fn reservoir_push_cap_one() {
    // Single-element reservoir. First sample always stored.
    // Subsequent samples replace with probability 1/count.
    let mut buf = Vec::new();
    let mut count = 0u64;
    reservoir_push(&mut buf, &mut count, 42, 1);
    assert_eq!(buf, vec![42]);
    assert_eq!(count, 1);
    // Push more — buf stays length 1.
    for i in 1..100 {
        reservoir_push(&mut buf, &mut count, i * 100, 1);
    }
    assert_eq!(buf.len(), 1);
    assert_eq!(count, 100);
}
// -- read_schedstat tests --

#[test]
fn read_schedstat_returns_finite_triple() {
    // The calling thread has been scheduled at least once by the
    // time this test runs (it's executing right now), so cpu_time
    // and timeslices must be strictly positive. run_delay can
    // legitimately be zero on an idle host where the test thread
    // never waited for a runqueue slot, so it is left unchecked.
    //
    // `None` is a legitimate outcome when the host kernel is
    // built without `CONFIG_SCHEDSTATS` — treat that as a skip
    // rather than a test failure.
    let Some((cpu_time, _run_delay, timeslices)) = read_schedstat(None) else {
        eprintln!("skipping: /proc/self/schedstat not available (CONFIG_SCHEDSTATS off)");
        return;
    };
    assert!(cpu_time > 0);
    assert!(timeslices > 0);
}
#[test]
fn parse_schedstat_line_happy_path() {
    // A well-formed line has at least three whitespace-separated
    // u64 fields; extra trailing fields are ignored.
    let (cpu_time, run_delay, timeslices) = parse_schedstat_line("100 200 300 999 extra").unwrap();
    assert_eq!(cpu_time, 100);
    assert_eq!(run_delay, 200);
    assert_eq!(timeslices, 300);
}
#[test]
fn parse_schedstat_line_tab_and_newline_separators() {
    // `split_whitespace` treats any run of whitespace as one
    // separator, so tabs and trailing newlines must parse.
    let parsed = parse_schedstat_line("1\t2\t3\n").unwrap();
    assert_eq!(parsed, (1, 2, 3));
}
#[test]
fn parse_schedstat_line_missing_field_returns_none() {
    // Two fields is one short — the third `?` bails.
    assert!(parse_schedstat_line("100 200").is_none());
    // One field short of two.
    assert!(parse_schedstat_line("100").is_none());
    // Empty input — zero fields.
    assert!(parse_schedstat_line("").is_none());
    // Whitespace-only input — zero tokens after split.
    assert!(parse_schedstat_line("   \t\n  ").is_none());
}
#[test]
fn parse_schedstat_line_non_u64_token_returns_none() {
    // Any non-u64 token fails the `.parse::<u64>().ok()?` chain.
    assert!(parse_schedstat_line("not-a-number 200 300").is_none());
    assert!(parse_schedstat_line("100 abc 300").is_none());
    assert!(parse_schedstat_line("100 200 nan").is_none());
    // Negative numbers parse to u64 as an error.
    assert!(parse_schedstat_line("-1 200 300").is_none());
    // Overflow beyond u64::MAX.
    assert!(parse_schedstat_line("99999999999999999999 2 3").is_none());
}
#[test]
fn claim_warn_slot_grants_first_call_only() {
    // The once-only warning contract lives in `claim_warn_slot`:
    // the warning helper emits its eprintln only when this returns
    // `true`, and it must return `true` exactly once per `Once`.
    // Drive it against a fresh `Once` so the assertion is
    // independent of any process-global state that peer tests may
    // have already tripped — direct stderr-emission capture is not
    // possible because `#[test]` threads share fd 2. A regression
    // that dropped the gate (emitting on every call) would make
    // this return `true` repeatedly and fail the false-on-repeat
    // assertions below.
    let once = std::sync::Once::new();
    assert!(
        claim_warn_slot(&once),
        "first call on a fresh Once must claim the slot (return true)",
    );
    for i in 0..10 {
        assert!(
            !claim_warn_slot(&once),
            "call {i} after the first must NOT re-claim the slot \
                 (the at-most-once emission contract)",
        );
    }
}

#[test]
fn warn_schedstat_unavailable_once_does_not_panic_on_repeat() {
    // The process-wide warning helper must tolerate the 2N calls a
    // full worker fan-out makes (each worker calls read_schedstat
    // twice). The once-only emission itself is pinned by
    // `claim_warn_slot_grants_first_call_only`; here we only
    // confirm the public wrapper is panic-free across repeats
    // (stderr-capture is unavailable since `#[test]` threads share
    // fd 2).
    for _ in 0..10 {
        warn_schedstat_unavailable_once();
    }
}
/// [`resolve_alu_width`] never returns `Widest`. Every
/// concrete variant resolves to itself (when the host
/// supports it) or to a downgrade — the sentinel must
/// disappear before reaching [`alu_hot_chain`].
#[test]
fn alu_width_resolve_never_returns_widest() {
    for &w in &[
        AluWidth::Scalar,
        AluWidth::Vec128,
        AluWidth::Vec256,
        AluWidth::Vec512,
        AluWidth::Amx,
        AluWidth::Widest,
    ] {
        let r = resolve_alu_width(w);
        assert!(
            !matches!(r, AluWidth::Widest),
            "resolve_alu_width({w:?}) returned Widest; \
             caller invariant violated",
        );
    }
}

/// [`alu_hot_chain`] MUST bump `work_units` by exactly `steps` per
/// invocation. The per-iteration body (its `for _ in 0..steps` loop) reads
/// `*work_units = std::hint::black_box(work_units.wrapping_add(1))`
/// — the bump is wrapped in `black_box` so the optimizer can't
/// elide it as a dead-store, but a refactor that removed the
/// `*work_units = ...` write entirely, moved it outside the loop,
/// or changed the increment constant would silently break the
/// work-units accounting that downstream metrics (worker progress
/// reporting, schedstat correlation) depend on. Test counts
/// work_units before and after a known-N call to catch:
/// - bump removed (post == 0)
/// - bump moved outside loop (post == 1)
/// - wrong increment constant (post != N)
/// - off-by-one in loop bounds (post == N-1 or N+1)
///
/// `AluWidth::Scalar` is the path the production code resolves to
/// on hosts without SIMD; using it sidesteps the
/// `resolve_alu_width(Widest)` call chain and tests the chain body
/// directly. The 1000-step count is large enough that an off-by-one
/// surfaces unambiguously without making the test slow (the chain
/// is pure ALU + black_box, ~microseconds at this size).
#[test]
fn alu_hot_chain_bumps_work_units_by_exact_step_count() {
    let mut work_units: u64 = 0;
    let steps: u64 = 1000;
    alu_hot_chain(AluWidth::Scalar, steps, &mut work_units);
    assert_eq!(
        work_units, steps,
        "alu_hot_chain(Scalar, {steps}, &mut 0) MUST leave work_units \
         == {steps}; got {work_units}. A regression that removed the \
         per-step bump (the loop body's `*work_units = ...wrapping_add(1)`) would surface here as 0; one \
         that moved the bump outside the loop would surface as 1; \
         one that changed the increment constant would surface as a \
         multiple of {steps}; one that introduced an off-by-one in \
         the loop bounds (e.g. `1..=steps` instead of `0..steps`) \
         would surface as {steps}-1 or {steps}+1.",
    );
}

/// [`alu_hot_chain`] accumulates into an externally-provided
/// `work_units` counter, so successive calls compound the bumps.
/// A regression that reset the counter inside the function (e.g.
/// `*work_units = N` instead of `*work_units = work_units.wrapping_add(1)`)
/// would surface as `work_units == steps` on the second call —
/// matching the single-call expectation but breaking the
/// compound-accumulation contract. Two back-to-back 500-step calls
/// must leave `work_units == 1000`.
#[test]
fn alu_hot_chain_compound_calls_accumulate_work_units() {
    let mut work_units: u64 = 0;
    alu_hot_chain(AluWidth::Scalar, 500, &mut work_units);
    alu_hot_chain(AluWidth::Scalar, 500, &mut work_units);
    assert_eq!(
        work_units, 1000,
        "two back-to-back alu_hot_chain(Scalar, 500, ...) calls MUST \
         leave work_units == 1000; got {work_units}. A regression that \
         reset the counter inside the function instead of bumping would \
         surface as 500 (the second call overwrites the first call's \
         total).",
    );
}

/// [`alu_hot_chain`] MUST bump `work_units` uniformly across every
/// concrete [`AluWidth`] variant. Today every arm runs the same
/// scalar multiply path per `alu_hot_chain`'s width-retention comment — `width`
/// is retained on the signature so a future SIMD-intrinsic drop can
/// pivot per-arm without changing the call shape. This test is
/// mechanically redundant today (all arms execute the same code) but
/// becomes load-bearing the moment per-arm SIMD dispatch lands: a
/// regression that breaks the `work_units` bump in ONE arm
/// (e.g. forgets the `*work_units = ...` write in the Vec256 branch
/// of a future per-arm match) would surface here as a single-variant
/// failure. `AluWidth::Widest` is excluded because `alu_hot_chain`
/// `debug_assert!`s against it (its `debug_assert!(!matches!(width, AluWidth::Widest), ...)`,
/// resolved away by
/// `resolve_alu_width` before reaching the chain).
#[test]
fn alu_hot_chain_bumps_work_units_uniformly_across_widths() {
    for &width in &[
        AluWidth::Scalar,
        AluWidth::Vec128,
        AluWidth::Vec256,
        AluWidth::Vec512,
        AluWidth::Amx,
    ] {
        let mut work_units: u64 = 0;
        let steps: u64 = 100;
        alu_hot_chain(width, steps, &mut work_units);
        assert_eq!(
            work_units, steps,
            "alu_hot_chain({width:?}, {steps}, &mut 0) MUST leave \
             work_units == {steps}; got {work_units}. A regression in \
             the per-arm dispatch (once SIMD intrinsics land) surfaces \
             here as a single-variant divergence — the assertion \
             message names the failing variant via `{{width:?}}`, \
             distinguishing it from sibling-arm regressions.",
        );
    }
}

/// [`alu_hot_chain`] uses `wrapping_add(1)` in its `for _ in 0..steps`
/// loop body — the
/// per-step bump must wrap silently on `u64::MAX` overflow rather
/// than panic or saturate. A regression that switched to
/// `checked_add(1).unwrap()` would panic; one that switched to
/// plain `+= 1` would panic in debug mode (integer overflow trap)
/// and silently wrap in release. The assert catches both by
/// starting from `u64::MAX` and verifying the wrap to 0.
/// Long-running workloads (multi-hour scheduler tests) WILL approach
/// `u64::MAX` given enough iterations; the wrapping contract is
/// load-bearing for those scenarios.
#[test]
fn alu_hot_chain_wraps_work_units_at_u64_max() {
    let mut work_units: u64 = u64::MAX;
    alu_hot_chain(AluWidth::Scalar, 1, &mut work_units);
    assert_eq!(
        work_units, 0,
        "alu_hot_chain bumping work_units past u64::MAX MUST wrap to 0; \
         got {work_units}. A regression that switched `wrapping_add(1)` \
         to `checked_add(1).unwrap()` would panic at this assertion; one \
         that switched to plain `+= 1` would panic in debug mode \
         (integer overflow trap) and produce a different value in release.",
    );
}

/// [`alu_hot_chain`] with `steps=0` must be a no-op: the for-loop
/// body (its `for _ in 0..steps` loop) executes 0 times, work_units stays
/// unchanged. A regression that off-by-one'd the loop bound
/// (e.g. `for _ in 0..=steps` instead of `for _ in 0..steps`)
/// would surface here as work_units == 43 (one bump from the
/// initial 42).
#[test]
fn alu_hot_chain_zero_steps_is_noop() {
    let mut work_units: u64 = 42;
    alu_hot_chain(AluWidth::Scalar, 0, &mut work_units);
    assert_eq!(
        work_units, 42,
        "alu_hot_chain(Scalar, 0, &mut 42) MUST leave work_units == 42; \
         got {work_units}. A regression that off-by-one'd the loop bound \
         (`0..=steps` instead of `0..steps`) would surface as 43.",
    );
}

/// [`alu_hot_chain`] with `steps=1` must bump exactly once. A
/// regression that started the loop at `for _ in 1..steps` (off-by-one
/// in lower bound) would surface here as work_units == 0 (zero
/// iterations from the empty `1..1` range).
#[test]
fn alu_hot_chain_single_step_bumps_by_one() {
    let mut work_units: u64 = 0;
    alu_hot_chain(AluWidth::Scalar, 1, &mut work_units);
    assert_eq!(
        work_units, 1,
        "alu_hot_chain(Scalar, 1, &mut 0) MUST leave work_units == 1; \
         got {work_units}. A regression that started the loop at \
         `1..steps` instead of `0..steps` would surface as 0.",
    );
}

/// [`alu_hot_chain`] has a `debug_assert!` (its
/// `debug_assert!(!matches!(width, AluWidth::Widest), ...)`) that
/// rejects `AluWidth::Widest` — callers MUST resolve `Widest` via
/// `resolve_alu_width` before reaching here. Test pins the
/// debug-build panic contract: a regression that removed the
/// debug_assert would surface as this test passing (no panic)
/// instead of expected_panic. Release builds elide debug_assert so
/// the test is gated on `cfg(debug_assertions)` to only run in
/// debug-mode test builds.
#[test]
#[cfg(debug_assertions)]
#[should_panic(expected = "alu_hot_chain reached with AluWidth::Widest")]
fn alu_hot_chain_panics_on_widest_in_debug_build() {
    let mut work_units: u64 = 0;
    alu_hot_chain(AluWidth::Widest, 1, &mut work_units);
}

/// `WorkPhase::AluHot` must drive observable retired-instructions counter
/// progress via the host PMU. A regression that turned the loop into a
/// no-op (e.g. compiler optimization stripping the work, or a refactor
/// that defaulted to Duration::ZERO) would silently pass functional
/// tests but produce zero ALU pressure — useless for noise-reduction
/// or fairness benchmarking. This test opens a `perf_event_open` fd
/// against `PERF_COUNT_HW_INSTRUCTIONS`, runs `alu_hot_chain` for a
/// fixed iteration count, and asserts the counter delta exceeds a
/// conservative floor.
///
/// **Panic-on-PMU-less host** per the user directive: if
/// perf_event_open returns -1 (no PMU, `CONFIG_PERF_EVENTS=n`, kernel
/// rejection, paranoid sysctl), the test FAILS LOUD instead of
/// skipping. ktstr is a benchmarking framework; silently degrading on
/// hosts without PMU access would undermine the entire
/// perf-measurement subsystem.
///
/// **CI host expectations.** This test requires unprivileged
/// userspace `perf_event_open` access. On hosts where
/// `kernel.perf_event_paranoid >= 2` (default on stock Debian/Ubuntu
/// since 2019) OR the process lacks `CAP_PERFMON`, the syscall
/// returns `EACCES` and the test panics. Restricted-container CI
/// (Docker without `--privileged`, GKE without
/// `securityContext.privileged: true`) cannot run this test — the
/// harness MUST filter it out via nextest (`--skip
/// alu_hot_chain_drives_instructions_retired`) or by setting
/// `kernel.perf_event_paranoid` to a value `<= 1` (kernel events) or
/// `0` (CPU-wide) or `-1` (unrestricted) in the runner setup. The
/// directive against silent skipping is honored: the test panics on
/// PMU-less, and the operator deals with the filter at the harness
/// layer.
///
/// Test guarantees retired-instructions semantics on both x86_64
/// (CPUID 0x0A PMU v2 synthesized in `x86_64::topology::generate_cpuid`'s
/// Leaf 0xA arm)
/// and aarch64 (KVM_ARM_VCPU_PMU_V3 wired by
/// `aarch64::kvm`'s `init_pmuv3` plus the
/// `kvi.features[0] |= 1 << KVM_ARM_VCPU_PMU_V3` feature set). The test runs on the HOST
/// (this is a #[test] in a host-test module, not a guest workload),
/// so the relevant PMU is the host's — Linux is required.
///
/// Uses `perf_event_open_sys` (Cargo.toml direct dep) rather than
/// raw libc because libc deliberately omits `perf_event_open` /
/// `perf_event_attr` bindings. `exclude_kernel = 1` so only userspace
/// ALU work counts (filters out interrupt-handler noise). `disabled
/// = 1` + `IOC_ENABLE` is the standard pattern to gate counter
/// activation until the workload starts.
#[test]
fn alu_hot_chain_drives_instructions_retired() {
    use perf_event_open_sys as pe;
    use std::io;
    use std::mem;
    use std::os::unix::io::RawFd;

    let mut attr = pe::bindings::perf_event_attr {
        size: mem::size_of::<pe::bindings::perf_event_attr>() as u32,
        type_: pe::bindings::PERF_TYPE_HARDWARE,
        config: pe::bindings::PERF_COUNT_HW_INSTRUCTIONS as u64,
        ..Default::default()
    };
    attr.set_disabled(1);
    attr.set_exclude_kernel(1);
    attr.set_exclude_hv(1);

    // perf_event_open: pid=0 (self process), cpu=-1 (any), group_fd=-1.
    // SAFETY: attrs points to a properly initialised perf_event_attr;
    // sentinel args are the standard form for "self process, any CPU,
    // no group". fd is checked for -1 before use.
    let fd = unsafe { pe::perf_event_open(&mut attr, 0, -1, -1, 0) };
    if fd < 0 {
        panic!(
            "perf_event_open(PERF_COUNT_HW_INSTRUCTIONS) failed: {}. \
             ktstr requires a PMU-capable host or guest. Check: \
             (1) CPUID leaf 0x0A (x86_64) / KVM_CAP_ARM_PMU_V3 \
             (aarch64) — host PMU presence; \
             (2) CONFIG_PERF_EVENTS=y — kernel build config; \
             (3) `cat /proc/sys/kernel/perf_event_paranoid` — must be \
             <= 2 for unprivileged hardware-counter access; defaults to \
             2 or 3 on stock Debian/Ubuntu/Fedora and BLOCKS this test. \
             If you cannot lower the sysctl (e.g. unprivileged \
             container), filter this test out via nextest \
             `--skip alu_hot_chain_drives_instructions_retired`.",
            io::Error::last_os_error(),
        );
    }
    let fd: RawFd = fd;

    // Reset + enable the counter. SAFETY: fd verified valid above;
    // ioctls are standard perf-event control ops.
    unsafe {
        pe::ioctls::RESET(fd, 0);
        pe::ioctls::ENABLE(fd, 0);
    }

    // Drive the AluHot work directly on the calling thread (the
    // perf fd was opened with pid=0, so only this thread is
    // measured). Use Scalar width to keep the test deterministic —
    // wider variants depend on host SIMD capability.
    let mut work_units: u64 = 0;
    // 10_000 outer chain entry/exit cycles × 1_000 inner steps per
    // call. Per-call retired-instruction count in debug (opt-level=0,
    // the default for `cargo ktstr test` / `cargo nextest run`) is
    // ~17 inst per inner step × 1_000 = ~17_000; × 10_000 outer
    // iterations = ~170M instructions. The 10M threshold sits ~17×
    // below this for comfortable debug-mode margin. Pre-fix tests
    // used `steps=1` and underdelivered the threshold by ~30×
    // (~500K actual vs 10M asserted) — a calibration defect.
    for _ in 0..10_000 {
        alu_hot_chain(AluWidth::Scalar, 1_000, &mut work_units);
    }

    // SAFETY: fd verified valid above; DISABLE ioctl is the
    // counter-pair to the ENABLE above (kernel/events/core.c
    // perf_ioctl maps both to the atomic counter-state transitions).
    unsafe {
        pe::ioctls::DISABLE(fd, 0);
    }
    let mut count: u64 = 0;
    // SAFETY: fd is valid; reading 8 bytes from a perf_event_open
    // fd yields the u64 counter value (default read format).
    let n = unsafe {
        libc::read(
            fd,
            &mut count as *mut u64 as *mut libc::c_void,
            mem::size_of::<u64>(),
        )
    };
    // SAFETY: fd is owned by this fn; close before any assertion
    // panic to avoid leaks across test runs.
    unsafe {
        libc::close(fd);
    }
    assert_eq!(
        n as usize,
        mem::size_of::<u64>(),
        "perf read must yield full u64 count; got {n} bytes",
    );

    // 10M instructions floor — conservative. 10_000 outer iterations
    // × `alu_hot_chain(Scalar, 1_000, ...)` emits ~170M debug-mode
    // instructions; threshold sits 17× below for margin. A counter at
    // zero or near-zero indicates the loop body was optimized away or
    // the PMU isn't counting. The diagnostic distinguishes the three
    // failure axes via work_units cross-check.
    assert!(
        count > 10_000_000,
        "alu_hot_chain must retire > 10M instructions across \
         10_000 iterations × 1_000 inner steps; got {count}. \
         Diagnostic axes: \
         (a) work_units = 0 → loop never ran (build/threading bug). \
         (b) work_units > 0 but count == 0 → PMU not counting \
         (paranoid sysctl, container restrictions, paravirt). \
         (c) work_units > 0 and 0 < count < 10M → threshold \
         mis-tuned OR loop got partially optimized (e.g. compiler \
         vectorized the scalar path). \
         Observed work_units = {work_units}.",
    );
}

/// `build_cross_affinity_masks` must yield two masks differing by
/// EXACTLY one CPU — the invariant that makes every
/// `CrossAffinityChurn` `sched_setaffinity` a genuine mask change, so
/// the kernel never `cpumask_equal`-early-returns and the
/// set_cpus_allowed path runs each iteration. Pins the "genuine mask
/// change" contract the WorkType's whole effect depends on.
#[test]
fn build_cross_affinity_masks_differ_by_exactly_one_cpu() {
    let cpus = [0usize, 1, 2, 3];
    let (a, b) = build_cross_affinity_masks(&cpus).expect(">=2 cpus yields a mask pair");
    let mut a_count = 0usize;
    let mut b_count = 0usize;
    let mut differ = 0usize;
    for &c in &cpus {
        let in_a = unsafe { libc::CPU_ISSET(c, &a) };
        let in_b = unsafe { libc::CPU_ISSET(c, &b) };
        a_count += in_a as usize;
        b_count += in_b as usize;
        differ += (in_a != in_b) as usize;
    }
    assert_eq!(a_count, cpus.len(), "mask_a must cover every cpu");
    assert_eq!(b_count, cpus.len() - 1, "mask_b must drop exactly one cpu");
    assert_eq!(differ, 1, "the two masks must differ by exactly one cpu");
}

/// A cpuset too small for two distinct masks yields `None` —
/// `CrossAffinityChurn` is a no-op for such a worker.
#[test]
fn build_cross_affinity_masks_none_below_two_cpus() {
    assert!(build_cross_affinity_masks(&[]).is_none());
    assert!(build_cross_affinity_masks(&[0]).is_none());
}

/// `read_sibling_pids` must never include the caller's own pid: the
/// `CrossAffinityChurn` arm and `WorkerCtx::sibling_pids` both rely on
/// self being filtered (a worker must not flip its own affinity via
/// the sibling path; a Custom worker expects peers only). Runs in the
/// test process's cgroup — the result may be empty where
/// `cgroup.procs` is unreadable, but must never contain `getpid()`.
#[test]
fn read_sibling_pids_excludes_self() {
    let me = unsafe { libc::getpid() };
    let sibs = read_sibling_pids();
    assert!(
        !sibs.contains(&me),
        "read_sibling_pids must exclude the caller's own pid {me}; got {sibs:?}"
    );
}

/// CgroupChurn rotation paths are built under the resolved workload
/// root (not the former hardcoded top-level path), as single-component leaf
/// cgroups. Production (`cgroup_churn_paths` setup) and this test share
/// `churn_cgroup_procs_paths`, so the layout is pinned at one site.
#[test]
fn churn_cgroup_procs_paths_are_under_the_root() {
    assert_eq!(
        churn_cgroup_procs_paths("/sys/fs/cgroup/ktstr", 2),
        vec![
            "/sys/fs/cgroup/ktstr/wt-cgroup-churn-0/cgroup.procs".to_string(),
            "/sys/fs/cgroup/ktstr/wt-cgroup-churn-1/cgroup.procs".to_string(),
        ]
    );
    // groups == 0 still yields one target (the .max(1) floor) so the
    // dispatch arm's index-by-target is always in-bounds.
    assert_eq!(churn_cgroup_procs_paths("/r", 0).len(), 1);
    assert_eq!(churn_cgroup_name(3), "wt-cgroup-churn-3");
}

/// `parse_cgroup_rel` extracts the cgroup-v2 `0::/<rel>` path, and
/// `read_self_cgroup_dir` maps it to `/sys/fs/cgroup<rel>`, returning
/// `None` for the root cgroup so a worker never mistakes the root for a
/// dedicated cgroup. Pins the parse + rel->dir mapping that
/// `WorkerCtx::cgroup_dir` is built on (its only consumer).
#[test]
fn parse_cgroup_rel_and_self_dir_discipline() {
    // A v2 single `0::` line yields the trimmed `<rel>`.
    assert_eq!(
        parse_cgroup_rel("0::/ktstr/cgA\n").as_deref(),
        Some("/ktstr/cgA")
    );
    // The root cgroup line parses to "/".
    assert_eq!(parse_cgroup_rel("0::/\n").as_deref(), Some("/"));
    // A cgroup-v1-only file (no `0::` line) yields None.
    assert_eq!(
        parse_cgroup_rel("3:cpu,cpuacct:/foo\n1:name=systemd:/bar\n"),
        None
    );
    // Hybrid host: v1 controller lines interleaved with the unified
    // `0::` line — the `0::` line is found regardless of its position.
    assert_eq!(
        parse_cgroup_rel("3:cpu,cpuacct:/foo\n0::/ktstr/cgA\n1:name=systemd:/bar\n")
            .as_deref(),
        Some("/ktstr/cgA")
    );
    // The rel->dir mapping pinned deterministically on literals (the
    // live check below only exercises whichever branch its own cgroup
    // hits): the root maps to None — never a bogus "/sys/fs/cgroup" —
    // and a dedicated rel maps under it.
    assert_eq!(cgroup_dir_from_rel("/"), None);
    // An empty rel (reachable from a `0::\n` literal via parse_cgroup_rel)
    // also maps to None — never the bare /sys/fs/cgroup sentinel.
    assert_eq!(cgroup_dir_from_rel(""), None);
    assert_eq!(
        cgroup_dir_from_rel("/ktstr/cgA"),
        Some(std::path::PathBuf::from("/sys/fs/cgroup/ktstr/cgA"))
    );
    // The live process: read_self_cgroup_dir is either None (root /
    // non-v2) or an absolute path strictly under /sys/fs/cgroup — never
    // the bare "/sys/fs/cgroup" root sentinel.
    if let Some(dir) = read_self_cgroup_dir() {
        assert!(
            dir.starts_with("/sys/fs/cgroup")
                && dir != std::path::Path::new("/sys/fs/cgroup"),
            "read_self_cgroup_dir must be a dedicated cgroup dir, got {dir:?}"
        );
    }
}

/// WorkerCtx::open_sibling_cgroup_procs: resolves
/// `cgroup_dir().parent()/<name>/cgroup.procs`, opens it write-mode (the
/// writability precheck), rejects escaping names before any fs access, and
/// surfaces NotFound (missing sibling / no cgroup_dir) loudly. Uses a
/// synthetic cgroup tree under a tempdir (no real cgroupfs).
#[test]
fn open_sibling_cgroup_procs_resolves_opens_and_validates() {
    use std::io::{ErrorKind, Write};
    let tmp = tempfile::TempDir::new().expect("tempdir");
    let root = tmp.path().join("root");
    let worker = root.join("worker");
    std::fs::create_dir_all(&worker).unwrap();
    std::fs::create_dir_all(root.join("dest")).unwrap();
    std::fs::write(root.join("dest").join("cgroup.procs"), b"").unwrap();
    let stop = AtomicBool::new(false);
    let ctx = WorkerCtx::new(&stop, &[], &[], Some(&worker), CustomCfg::default());

    // Resolves + opens the sibling's cgroup.procs write-mode.
    let mut f = ctx
        .open_sibling_cgroup_procs("dest")
        .expect("dest sibling cgroup.procs must open write-mode");
    assert!(f.write_all(b"").is_ok(), "the returned handle is writable");

    // Missing sibling -> NotFound, with the resolved path enriched into the
    // message (the .map_err layer the doc promises).
    let miss = ctx.open_sibling_cgroup_procs("no_such").unwrap_err();
    assert_eq!(miss.kind(), ErrorKind::NotFound);
    assert!(
        miss.to_string().contains("no_such/cgroup.procs"),
        "the open error must name the resolved sibling path; got: {miss}"
    );

    // Escaping / malformed names rejected as InvalidInput (the
    // validation-before-fs class — a dropped validation would instead surface
    // these as NotFound, since parent.join(name) would resolve).
    for bad in ["", "..", "a/../b", "/abs", ".hidden"] {
        assert_eq!(
            ctx.open_sibling_cgroup_procs(bad).unwrap_err().kind(),
            ErrorKind::InvalidInput,
            "name {bad:?} must be rejected as InvalidInput"
        );
    }

    // No cgroup_dir (root / non-v2 worker) -> NotFound, never a fabricated path.
    let ctx_root = WorkerCtx::new(&stop, &[], &[], None, CustomCfg::default());
    assert_eq!(
        ctx_root.open_sibling_cgroup_procs("dest").unwrap_err().kind(),
        ErrorKind::NotFound
    );

    // cgroup_dir == "/" (parent() is None) -> NotFound (the documented
    // no-parent branch), not a malformed join.
    let ctx_slash = WorkerCtx::new(
        &stop,
        &[],
        &[],
        Some(std::path::Path::new("/")),
        CustomCfg::default(),
    );
    assert_eq!(
        ctx_slash.open_sibling_cgroup_procs("dest").unwrap_err().kind(),
        ErrorKind::NotFound
    );
}

/// An unwritable sibling `cgroup.procs` surfaces as PermissionDenied
/// (errno-kind preserved). Gated on non-root — root bypasses DAC mode bits
/// on a regular file, so the EACCES path is only observable unprivileged
/// (ktstr's VM runs as root, where this skips).
#[test]
fn open_sibling_cgroup_procs_unwritable_is_permission_denied() {
    if unsafe { libc::geteuid() } == 0 {
        return; // root bypasses the 0o400 mode bit; EACCES is unobservable.
    }
    use std::io::ErrorKind;
    use std::os::unix::fs::PermissionsExt;
    let tmp = tempfile::TempDir::new().expect("tempdir");
    let worker = tmp.path().join("root").join("worker");
    let dest = tmp.path().join("root").join("dest");
    std::fs::create_dir_all(&worker).unwrap();
    std::fs::create_dir_all(&dest).unwrap();
    let procs = dest.join("cgroup.procs");
    std::fs::write(&procs, b"").unwrap();
    std::fs::set_permissions(&procs, std::fs::Permissions::from_mode(0o400)).unwrap();
    let stop = AtomicBool::new(false);
    let ctx = WorkerCtx::new(&stop, &[], &[], Some(&worker), CustomCfg::default());
    assert_eq!(
        ctx.open_sibling_cgroup_procs("dest").unwrap_err().kind(),
        ErrorKind::PermissionDenied
    );
}

/// `custom_iterations_telltale` flags ONLY the `work_units > 0` +
/// `iterations == 0` footgun shape (the naive-Custom-author trap the dispatch
/// warn-once targets), and nothing else — so the advisory never fires on a
/// balanced, throughput-only, or empty report.
#[test]
fn custom_iterations_telltale_flags_only_work_units_without_iterations() {
    let mk = |work_units, iterations| WorkerReport {
        work_units,
        iterations,
        ..Default::default()
    };
    assert!(custom_iterations_telltale(&mk(1000, 0)));
    assert!(!custom_iterations_telltale(&mk(0, 1000)));
    assert!(!custom_iterations_telltale(&mk(1000, 1000)));
    assert!(!custom_iterations_telltale(&mk(0, 0)));
}

/// `WorkerCtx` accessors return the borrowed worker state verbatim,
/// and `stop()` exposes the live flag — a flip is observable through
/// the ctx. Pins the read-only contract a `Custom` worker relies on.
#[test]
fn worker_ctx_accessors_return_borrowed_state() {
    let stop = AtomicBool::new(false);
    let cpus = [3usize, 7, 11];
    let siblings: [libc::pid_t; 2] = [100, 200];
    let cg = std::path::Path::new("/sys/fs/cgroup/ktstr/cgA");
    let ctx = WorkerCtx::new(&stop, &cpus, &siblings, Some(cg), CustomCfg::default());
    assert_eq!(
        ctx.cpus(),
        cpus.as_slice(),
        "cpus() returns the borrowed cpuset"
    );
    assert_eq!(
        ctx.sibling_pids(),
        siblings.as_slice(),
        "sibling_pids() returns the borrowed peer set"
    );
    assert_eq!(
        ctx.cgroup_dir(),
        Some(cg),
        "cgroup_dir() returns the borrowed cgroup path"
    );
    assert_eq!(
        WorkerCtx::new(&stop, &cpus, &siblings, None, CustomCfg::default()).cgroup_dir(),
        None,
        "cgroup_dir() is None for the root / non-v2 case"
    );
    assert!(!ctx.stop().load(Ordering::Relaxed));
    stop.store(true, Ordering::Relaxed);
    assert!(
        ctx.stop().load(Ordering::Relaxed),
        "stop() exposes the live flag — a flip is visible through the ctx"
    );
}

/// `read_effective_cpus` reports the caller's effective cpuset.
/// `sched_getaffinity(0)` always yields a non-empty set for a running
/// task, so the helper the `Custom` ctx and the affinity-churn arms
/// share must report at least one CPU.
#[test]
fn read_effective_cpus_reports_runner_cpuset() {
    assert!(
        !read_effective_cpus().is_empty(),
        "read_effective_cpus must report the runner's non-empty cpuset"
    );
}

// -- build_phase_slice: the per-phase finalize delta math.
// Both the per-transition drain and the final end-of-run drain call this,
// so these pin the math the two paths share. The production drains read the
// clock / `/proc` to produce the snapshots; this helper is pure, so the
// delta math is testable in isolation without booting a VM. --

/// Full delta math, every field non-default. The key invariant: `off_cpu_ns`
/// uses the thread-CPU delta (args 3->4) while `schedstat_cpu_time_ns` and
/// `run_delay_ns` use the schedstat `sum_exec_runtime` / `run_delay` deltas —
/// the two CPU measures are independent and must not be conflated.
#[test]
fn build_phase_slice_full_delta_math() {
    let cpus: BTreeSet<usize> = [0, 2].into_iter().collect();
    let numa: BTreeMap<usize, u64> = [(0, 100), (1, 50)].into_iter().collect();
    let s = build_phase_slice(
        7,                        // phase_epoch
        1_000_000,                // wall_ns
        500_000,                  // cpu_start_ns (thread cpu)
        900_000,                  // cpu_end_ns -> thread-cpu delta 400_000
        Some((10_000, 2_000, 1)), // schedstat_start (sum_exec, run_delay, ts)
        Some((13_000, 5_000, 9)), // schedstat_end -> ss_cpu 3_000, run_delay 3_000
        40,                       // vmstat_migrated_start
        70,                       // vmstat_migrated_end -> 30
        100,                      // iterations_delta (already a delta)
        2,                        // migration_count
        45_000_000,               // max_gap_ns -> 45 ms
        2,                        // max_gap_cpu
        cpus.clone(),
        numa.clone(),
        (vec![11, 22, 33], 5), // wake (reservoir, total)
    );
    assert_eq!(s.phase_epoch, 7);
    assert_eq!(s.wall_ns, 1_000_000);
    // off_cpu = wall - thread-cpu delta = 1_000_000 - 400_000.
    assert_eq!(s.off_cpu_ns, 600_000);
    // schedstat_cpu_time is the schedstat sum_exec_runtime delta (3_000),
    // NOT the thread-cpu delta (400_000) — the two are independent measures.
    assert_eq!(s.schedstat_cpu_time_ns, 3_000);
    assert_eq!(s.run_delay_ns, 3_000);
    assert_eq!(s.vmstat_numa_pages_migrated, 30);
    assert_eq!(s.iterations, 100);
    assert_eq!(s.migration_count, 2);
    assert_eq!(s.max_gap_ms, 45); // 45_000_000 ns floored to ms
    assert_eq!(s.max_gap_cpu, 2);
    assert_eq!(s.cpus_used, cpus);
    assert_eq!(s.numa_pages, numa);
    assert_eq!(s.wake_latencies_ns, vec![11, 22, 33]);
    assert_eq!(s.wake_sample_total, 5);
}

/// `off_cpu_ns` saturates at 0 when the thread-CPU delta exceeds wall — a
/// coarse wall clock against a finer thread-CPU clock must never underflow
/// into a near-`u64::MAX` value that would poison the host off-CPU% fold.
#[test]
fn build_phase_slice_off_cpu_saturates_at_zero() {
    let s = build_phase_slice(
        1,
        100, // wall_ns
        0,
        500, // cpu_delta 500 > wall 100
        None,
        None,
        0,
        0,
        0,
        0,
        0,
        0,
        BTreeSet::new(),
        BTreeMap::new(),
        (vec![], 0),
    );
    assert_eq!(s.off_cpu_ns, 0);
}

/// A schedstat snapshot missing at either end yields zero run_delay and zero
/// schedstat cpu time (the `_ => (0, 0)` arm) — never a partial read off one
/// endpoint.
#[test]
fn build_phase_slice_missing_schedstat_is_zero() {
    let mk = |start, end| {
        build_phase_slice(
            1,
            1000,
            0,
            100,
            start,
            end,
            0,
            0,
            0,
            0,
            0,
            0,
            BTreeSet::new(),
            BTreeMap::new(),
            (vec![], 0),
        )
    };
    let only_start = mk(Some((10, 20, 30)), None);
    assert_eq!(
        (only_start.run_delay_ns, only_start.schedstat_cpu_time_ns),
        (0, 0)
    );
    let only_end = mk(None, Some((10, 20, 30)));
    assert_eq!(
        (only_end.run_delay_ns, only_end.schedstat_cpu_time_ns),
        (0, 0)
    );
    let neither = mk(None, None);
    assert_eq!(
        (neither.run_delay_ns, neither.schedstat_cpu_time_ns),
        (0, 0)
    );
}

/// Every counter delta saturates when end < start (a counter read across a
/// reset, or a non-monotone `/proc` snapshot) — yields 0, never a wrapped
/// near-`u64::MAX` value that would poison the host counter fold.
#[test]
fn build_phase_slice_counter_deltas_saturate() {
    let s = build_phase_slice(
        1,
        1000,
        0,
        100,
        Some((100, 100, 1)), // schedstat_start sum_exec/run_delay 100
        Some((10, 10, 0)),   // schedstat_end 10 (< start)
        100,                 // vmstat_migrated_start 100
        10,                  // vmstat_migrated_end 10 (< start)
        0,
        0,
        0,
        0,
        BTreeSet::new(),
        BTreeMap::new(),
        (vec![], 0),
    );
    assert_eq!(s.run_delay_ns, 0);
    assert_eq!(s.schedstat_cpu_time_ns, 0);
    assert_eq!(s.vmstat_numa_pages_migrated, 0);
}

// -- WakeRec: the twin wake-latency reservoir. push()
// records every sample into BOTH the whole-run and per-phase reservoirs;
// drain_phase() takes the per-phase samples+total and resets ONLY the phase
// side, leaving the whole-run side intact. A swapped field or a missed reset
// would silently corrupt both the per-phase and whole-run wake distributions
// (wake_sample_total is the population weight in the cross-phase de-skew). --

/// push records into both reservoirs; under both caps every sample is kept.
#[test]
fn wakerec_push_records_both_reservoirs() {
    let mut w = WakeRec::new();
    for s in [10u64, 20, 30] {
        w.push(s);
    }
    let mut run = w.run.clone();
    run.sort_unstable();
    assert_eq!(run, vec![10, 20, 30]);
    assert_eq!(w.run_total, 3);
    let mut phase = w.phase.clone();
    phase.sort_unstable();
    assert_eq!(phase, vec![10, 20, 30]);
    assert_eq!(w.phase_total, 3);
}

/// drain_phase takes the per-phase samples+total and resets ONLY the phase
/// side; the whole-run reservoir is untouched, and a fresh phase accumulates
/// independently after the reset.
#[test]
fn wakerec_drain_phase_takes_phase_leaves_run() {
    let mut w = WakeRec::new();
    for s in [10u64, 20, 30] {
        w.push(s);
    }
    let (mut samples, total) = w.drain_phase();
    samples.sort_unstable();
    assert_eq!(samples, vec![10, 20, 30]);
    assert_eq!(total, 3);
    // Whole-run reservoir untouched by the phase drain.
    let mut run = w.run.clone();
    run.sort_unstable();
    assert_eq!(run, vec![10, 20, 30]);
    assert_eq!(w.run_total, 3);
    // Phase side reset: a second drain with no intervening push is empty.
    let (samples2, total2) = w.drain_phase();
    assert!(samples2.is_empty());
    assert_eq!(total2, 0);
    // A fresh phase accumulates from zero after the reset...
    w.push(99);
    let (samples3, total3) = w.drain_phase();
    assert_eq!(samples3, vec![99]);
    assert_eq!(total3, 1);
    // ...while the whole-run reservoir kept every sample across both phases.
    assert_eq!(w.run.len(), 4);
    assert_eq!(w.run_total, 4);
}

/// The phase and run caps are independent: pushing past MAX_PHASE_WAKE_SAMPLES
/// (but below MAX_WAKE_SAMPLES) caps the drained phase vec at the phase cap
/// while the whole-run reservoir keeps every sample. Both totals count every
/// push regardless of the sampled vec length.
#[test]
fn wakerec_phase_and_run_caps_are_independent() {
    let mut w = WakeRec::new();
    let n = MAX_PHASE_WAKE_SAMPLES + 808; // over the phase cap, under the run cap
    assert!(n < MAX_WAKE_SAMPLES);
    for i in 0..n as u64 {
        w.push(i);
    }
    let (samples, total) = w.drain_phase();
    // Phase reservoir samples down to its cap but counts every push.
    assert_eq!(samples.len(), MAX_PHASE_WAKE_SAMPLES);
    assert_eq!(total, n as u64);
    // Run reservoir is capped independently (and n is below its cap), so it
    // kept all n while the phase reservoir sampled down.
    assert_eq!(w.run.len(), n);
    assert_eq!(w.run_total, n as u64);
}
