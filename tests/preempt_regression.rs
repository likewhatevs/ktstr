use anyhow::Result;
use ktstr::assert::AssertResult;
use ktstr::ktstr_test;
use ktstr::scenario::Ctx;
use ktstr::scenario::ops::{CgroupDef, Step, execute_steps};
use ktstr::workload::{CustomCfg, WorkType, WorkerCtx, WorkerReport};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::Instant;

/// Allocate the shared futex word in the test process before workers
/// fork; returns its address. The `MAP_SHARED` page is inherited by every
/// forked worker at the same virtual address, so passing the address
/// through a [`CustomCfg`] `u64` slot (Copy POD) reaches each worker
/// post-fork — no `static`/global needed.
fn init_shared_futex() -> u64 {
    let ptr = unsafe {
        libc::mmap(
            std::ptr::null_mut(),
            std::mem::size_of::<u32>(),
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_SHARED | libc::MAP_ANONYMOUS,
            -1,
            0,
        )
    };
    assert_ne!(ptr, libc::MAP_FAILED, "mmap failed for shared futex");
    unsafe {
        std::ptr::write_bytes(ptr as *mut u8, 0, std::mem::size_of::<u32>());
    }
    ptr as u64
}

/// Combined lock-holding + page-faulting workload.
///
/// Each iteration: spin work, then CAS acquire a shared futex mutex
/// (FUTEX_WAIT on contention), touch random cold pages while holding
/// the lock, then atomic Release store + FUTEX_WAKE(1). When the lock
/// holder is preempted during page faults, all contenders stall — the
/// cascade that caused PostgreSQL regressions under preemption-heavy
/// schedulers. Neither MutexContention (no memory pressure under lock)
/// nor PageFaultChurn (no contention) alone reproduces this interaction.
fn fault_under_lock(ctx: &WorkerCtx) -> WorkerReport {
    let stop = ctx.stop();
    let tid: libc::pid_t = unsafe { libc::getpid() };
    let start = Instant::now();
    let mut work_units = 0u64;
    let mut iterations = 0u64;

    // Config arrives via CustomCfg (Copy POD inherited across fork): the
    // shared MAP_SHARED futex address in u64s[0], the fault region size in
    // usizes[0], touches-per-hold in usizes[1]. Replaces the former
    // `static mut FUTEX_PTR` + hardcoded consts.
    let cfg = ctx.cfg();
    let futex_addr = cfg.u64s[0];
    let region_size: usize = cfg.usizes[0];
    let touches_per_hold: usize = cfg.usizes[1];
    if futex_addr == 0 || region_size == 0 {
        return zeroed_report(tid, start);
    }
    let futex_ptr = futex_addr as *mut u32;
    let atom = unsafe { &*(futex_ptr as *const AtomicU32) };
    let page_count = (region_size / 4096).max(1);

    let ptr = unsafe {
        libc::mmap(
            std::ptr::null_mut(),
            region_size,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
            -1,
            0,
        )
    };
    if ptr == libc::MAP_FAILED {
        return zeroed_report(tid, start);
    }
    unsafe {
        libc::madvise(ptr, region_size, libc::MADV_NOHUGEPAGE);
    }

    let mut rng_state = (tid as u64) | 1;
    let xorshift64 = |state: &mut u64| -> u64 {
        let mut x = *state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        *state = x;
        x
    };

    while !stop.load(Ordering::Relaxed) {
        // Spin work between lock acquisitions.
        for _ in 0..256u64 {
            work_units = std::hint::black_box(work_units.wrapping_add(1));
            std::hint::spin_loop();
        }

        // Acquire: CAS 0 -> 1, FUTEX_WAIT on contention.
        loop {
            if stop.load(Ordering::Relaxed) {
                break;
            }
            if atom
                .compare_exchange_weak(0, 1, Ordering::Acquire, Ordering::Relaxed)
                .is_ok()
            {
                break;
            }
            let ts = libc::timespec {
                tv_sec: 0,
                tv_nsec: 100_000_000, // 100ms
            };
            unsafe {
                libc::syscall(
                    libc::SYS_futex,
                    futex_ptr,
                    libc::FUTEX_WAIT,
                    1u32,
                    &ts as *const libc::timespec,
                    std::ptr::null::<u32>(),
                    0u32,
                );
            }
        }

        // Critical section: fault cold pages while holding the lock.
        for _ in 0..touches_per_hold {
            let page_idx = (xorshift64(&mut rng_state) as usize) % page_count;
            let page_ptr = unsafe { (ptr as *mut u8).add(page_idx * 4096) };
            unsafe { std::ptr::write_volatile(page_ptr, 1u8) };
            work_units = work_units.wrapping_add(1);
        }

        // Release: atomic store + FUTEX_WAKE(1).
        atom.store(0, Ordering::Release);
        unsafe {
            libc::syscall(
                libc::SYS_futex,
                futex_ptr,
                libc::FUTEX_WAKE,
                1,
                std::ptr::null::<libc::timespec>(),
                std::ptr::null::<u32>(),
                0u32,
            );
        }

        // Zap PTEs so next iteration faults again.
        unsafe {
            libc::madvise(ptr, region_size, libc::MADV_DONTNEED);
        }

        iterations += 1;
    }

    unsafe {
        libc::munmap(ptr, region_size);
    }

    let wall_time_ns = start.elapsed().as_nanos() as u64;
    WorkerReport {
        tid,
        work_units,
        cpu_time_ns: 0,
        wall_time_ns,
        off_cpu_ns: 0,
        migration_count: 0,
        cpus_used: BTreeSet::new(),
        migrations: vec![],
        max_gap_ms: 0,
        max_gap_cpu: 0,
        max_gap_at_ms: 0,
        wake_latencies_ns: vec![],
        wake_sample_total: 0,
        iteration_costs_ns: vec![],
        iteration_cost_sample_total: 0,
        timer_latencies_ns: vec![],
        timer_sample_total: 0,
        iterations,
        schedstat_run_delay_ns: 0,
        schedstat_run_count: 0,
        schedstat_cpu_time_ns: 0,
        completed: true,
        numa_pages: BTreeMap::new(),
        vmstat_numa_pages_migrated: 0,
        exit_info: None,
        is_messenger: false,
        group_idx: 0,
        affinity_error: None,
        phase_slices: vec![],
    }
}

fn zeroed_report(tid: libc::pid_t, start: Instant) -> WorkerReport {
    WorkerReport {
        tid,
        work_units: 0,
        cpu_time_ns: 0,
        wall_time_ns: start.elapsed().as_nanos() as u64,
        off_cpu_ns: 0,
        migration_count: 0,
        cpus_used: BTreeSet::new(),
        migrations: vec![],
        max_gap_ms: 0,
        max_gap_cpu: 0,
        max_gap_at_ms: 0,
        wake_latencies_ns: vec![],
        wake_sample_total: 0,
        iteration_costs_ns: vec![],
        iteration_cost_sample_total: 0,
        timer_latencies_ns: vec![],
        timer_sample_total: 0,
        iterations: 0,
        schedstat_run_delay_ns: 0,
        schedstat_run_count: 0,
        schedstat_cpu_time_ns: 0,
        completed: true,
        numa_pages: BTreeMap::new(),
        vmstat_numa_pages_migrated: 0,
        exit_info: None,
        is_messenger: false,
        group_idx: 0,
        affinity_error: None,
        phase_slices: vec![],
    }
}

/// Reproduces the preemption-under-lock regression pattern observed in
/// PostgreSQL workloads. Multiple cgroups run the combined fault+lock
/// workload alongside pure CPU workers competing for the same CPUs.
///
/// To compare across kernel versions: run this test on both kernels,
/// compare `total_iterations` and `schedstat_run_delay_ns` from the
/// worker reports. A regression shows as lower throughput and higher
/// run delay on the affected kernel.
#[ktstr_test(llcs = 1, cores = 4, threads = 1, memory_mib = 2048)]
fn preempt_regression_fault_under_load(ctx: &Ctx) -> Result<AssertResult> {
    let futex_addr = init_shared_futex();
    // Pass the shared-futex address + fault-region geometry through the
    // fork-safe CustomCfg payload (replaces the former `static mut`).
    let cfg = CustomCfg::default()
        .u64_slot(0, futex_addr)
        .usize_slot(0, 256 * 1024) // region_size: 256 KB
        .usize_slot(1, 32); // touches_per_hold
    let fault_lock_wt = WorkType::custom_with("fault_under_lock", fault_under_lock, cfg);

    let steps = vec![Step::with_defs(
        vec![
            CgroupDef::named("fault_lock_workers")
                .workers(4)
                .work_type(fault_lock_wt),
            CgroupDef::named("cpu_contenders")
                .workers(4)
                .work_type(WorkType::SpinWait),
        ],
        ctx.settled_hold(1.0),
    )];

    execute_steps(ctx, steps)
}
