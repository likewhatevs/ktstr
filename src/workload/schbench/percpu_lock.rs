//! schbench's per-CPU lock stressor, ported faithfully from `schbench.c`
//! (`struct per_cpu_lock` :93, `lock_this_cpu` :1340, the init loop :1882, the
//! `pthread_mutex_unlock` in `do_work` :1411). Each CPU gets a
//! `pthread_mutex_t`; a worker locks the mutex for the CPU it is currently
//! running on, holds it across its matrix work, then unlocks. Under
//! lock-holder preemption, other workers that land on that CPU spin on its
//! mutex -- the scheduler pathology schbench induces. schbench's `-L`
//! (skip_locking) bypasses the lock entirely; that decision lives with the
//! caller (the `Schbench` workload), not here.
//!
//! # Fidelity
//!
//! - The lock is a real [`libc::pthread_mutex_t`] (a default NORMAL mutex) in
//!   a 16-byte-aligned wrapper, matching schbench's
//!   `struct per_cpu_lock { pthread_mutex_t lock; } __attribute__((aligned))`
//!   (`schbench.c:93-95`). The no-argument `aligned` resolves to
//!   `__BIGGEST_ALIGNMENT__` (16 on both supported arches, x86_64 and
//!   aarch64), so each per-lock slot has a 48-byte stride; `repr(C, align(16))`
//!   reproduces it (the `pthread_mutex_t` is 40 bytes + 8 tail padding on
//!   x86_64, 48 bytes with no padding on aarch64). The 48-byte stride is the
//!   load-bearing fact: it reproduces schbench's lock-word cache-line layout
//!   -- including the false sharing between adjacent CPUs' lock words -- so the
//!   workload's cache traffic matches. This is faithfulness to schbench's
//!   *workload behavior* (which the perf-identity comparison checks), distinct
//!   from fixing schbench's output-corrupting bugs.
//!
//! - [`spin_loop`] is the busy-wait hint, identical to schbench's `nop` macro
//!   (`schbench.c:1150-1156`: x86 `rep; nop` = PAUSE, aarch64 `yield`).
//!
//! - The per-CPU association holds only *at lock-acquire time*, not for the
//!   whole critical section. [`libc::sched_getcpu`] reads the current CPU (via
//!   the vDSO; never cached thread state), and the thread can migrate
//!   mid-section -- no syscall prevents that without affinity-pinning or rseq,
//!   which schbench deliberately does not use. The acquire-time recheck only
//!   ensures the lock taken is the lock for the CPU the thread was on when it
//!   acquired; it is self-contention shaping, not a hard per-CPU-residency
//!   guarantee. This matches schbench's intent (kernel-source-verified against
//!   `sched_getcpu`/`getcpu(2)`, which need no special CONFIG).

use core::hint::spin_loop;

/// One per-CPU lock. `repr(C, align(16))` matches schbench's
/// `struct per_cpu_lock { pthread_mutex_t lock; } __attribute__((aligned))`
/// (`schbench.c:93-95`): a 48-byte, 16-aligned slot (the `pthread_mutex_t` is
/// 40 bytes + 8 padding on x86_64, 48 bytes on aarch64), so an array of these
/// reproduces schbench's lock-word false-sharing layout on both arches.
#[repr(C, align(16))]
struct PerCpuLock {
    lock: libc::pthread_mutex_t,
}

/// schbench's per-CPU lock array (`per_cpu_locks`, `schbench.c:97`): one
/// `pthread_mutex_t` per CPU, indexed by [`libc::sched_getcpu`], shared across
/// every schbench worker thread.
pub(crate) struct PerCpuLocks {
    locks: Box<[PerCpuLock]>,
}

// SAFETY: `pthread_mutex_t` exists precisely to be shared and operated on
// concurrently from multiple threads. The mutexes live at stable heap
// addresses (`Box<[_]>`, never moved after `pthread_mutex_init`), and every
// access goes through the glibc/kernel mutex primitives, which are
// thread-safe. So `&PerCpuLocks` is safe to share across threads.
unsafe impl Sync for PerCpuLocks {}
// SAFETY: the array of pthread mutexes carries no thread-affine state, so
// ownership of a `PerCpuLocks` can move between threads.
unsafe impl Send for PerCpuLocks {}

impl PerCpuLocks {
    /// Allocate and initialize `num_cpus` per-CPU mutexes. Faithful to
    /// schbench's `calloc(num_cpu_locks, ...)` + per-lock `pthread_mutex_init`
    /// with a NULL attr (the default NORMAL mutex) at `schbench.c:1882-1896`.
    /// `num_cpus` is the number of CPUs the workload's threads can run on (the
    /// guest cpuset); [`lock_this_cpu`](Self::lock_this_cpu) indexes it by
    /// `sched_getcpu`.
    pub(crate) fn new(num_cpus: usize) -> Self {
        assert!(num_cpus > 0, "PerCpuLocks needs at least one CPU");
        // Zero each slot (schbench `calloc`s the array) then run
        // pthread_mutex_init at the final heap address. A zeroed
        // pthread_mutex_t is already glibc's valid unlocked NORMAL state, but
        // call pthread_mutex_init to match schbench exactly and stay correct
        // on any libc.
        let mut locks: Box<[PerCpuLock]> = (0..num_cpus)
            .map(|_| PerCpuLock {
                // SAFETY: an all-zero `pthread_mutex_t` is the documented
                // zero-state; the `pthread_mutex_init` below installs the real
                // initial state before any lock operation can run.
                lock: unsafe { core::mem::zeroed() },
            })
            .collect();
        for slot in locks.iter_mut() {
            // SAFETY: `&mut slot.lock` is a live, owned, stably-located (heap)
            // `pthread_mutex_t`; a NULL attr selects the default NORMAL mutex.
            // init of a fresh zeroed mutex with default attrs returns 0 on
            // Linux.
            let rc = unsafe { libc::pthread_mutex_init(&mut slot.lock, core::ptr::null()) };
            assert_eq!(rc, 0, "pthread_mutex_init failed: {rc}");
        }
        Self { locks }
    }

    /// Number of per-CPU locks. Production sizes the array via the
    /// `lock_array_size` passed to [`new`](Self::new) and indexes with
    /// `self.locks.len()` internally; this accessor gives the lock-count test
    /// clean read access without reaching into the private field.
    #[allow(dead_code)]
    pub(crate) fn len(&self) -> usize {
        self.locks.len()
    }

    /// Lock the mutex for the CPU this thread is currently on, returning a
    /// guard that unlocks on drop. Faithful port of `schbench.c`
    /// `lock_this_cpu` (:1340): read the current CPU, spin on its mutex via
    /// `trylock`, then re-read the CPU and -- if the thread migrated while
    /// spinning -- release and retry on the new CPU's mutex.
    ///
    /// The returned [`PerCpuLockGuard`]'s scope is the critical section,
    /// matching schbench's `lock = lock_this_cpu(); ...;
    /// pthread_mutex_unlock(lock)` bracket in `do_work` (`schbench.c:1387`,
    /// `:1411`). Per the module-level fidelity note, the per-CPU association
    /// holds only at the moment of acquire.
    pub(crate) fn lock_this_cpu(&self) -> PerCpuLockGuard<'_> {
        loop {
            let cpu = current_cpu(self.locks.len());
            let lock = core::ptr::addr_of!(self.locks[cpu].lock) as *mut libc::pthread_mutex_t;
            // Busy `trylock` spin, identical to schbench's
            // `while (pthread_mutex_trylock(lock) != 0) nop;` (:1353-1354).
            // SAFETY: `lock` points to an initialized, stably-located mutex
            // owned by `self`.
            while unsafe { libc::pthread_mutex_trylock(lock) } != 0 {
                spin_loop();
            }
            // Re-read the CPU: if we migrated while spinning we hold the wrong
            // CPU's lock, so release and retry (`schbench.c:1356-1366`).
            if current_cpu(self.locks.len()) == cpu {
                return PerCpuLockGuard {
                    lock,
                    _marker: core::marker::PhantomData,
                };
            }
            // SAFETY: we hold `lock` -- `trylock` returned 0 just above.
            unsafe { libc::pthread_mutex_unlock(lock) };
        }
    }
}

impl Drop for PerCpuLocks {
    fn drop(&mut self) {
        for slot in self.locks.iter_mut() {
            // SAFETY: a `PerCpuLockGuard` borrows `&self`, so none can outlive
            // this `PerCpuLocks`; no mutex is held here. Each mutex was
            // initialized in `new`. `pthread_mutex_destroy` is the teardown
            // counterpart to `pthread_mutex_init`.
            unsafe { libc::pthread_mutex_destroy(&mut slot.lock) };
        }
    }
}

/// The CPU this thread is currently on, via [`libc::sched_getcpu`]
/// (`schbench.c:1347`). Clamped to `len` so a result outside the allocated
/// range (a sparse cpuset or CPU-hotplug edge) indexes a valid lock instead of
/// reading out of bounds; in the common dense-cpuset case the clamp never
/// fires (the guest sees CPUs `0..len`).
fn current_cpu(len: usize) -> usize {
    // SAFETY: `sched_getcpu` takes no arguments and only reads the current CPU
    // id; it does not fail on Linux (no special CONFIG required).
    let cpu = unsafe { libc::sched_getcpu() };
    if cpu < 0 {
        // Not expected on Linux; CPU 0 always exists, so fall back to it
        // rather than panicking on the workload hot path.
        0
    } else {
        (cpu as usize).min(len - 1)
    }
}

/// RAII guard holding a per-CPU mutex; unlocks on drop. Returned by
/// [`PerCpuLocks::lock_this_cpu`].
pub(crate) struct PerCpuLockGuard<'a> {
    lock: *mut libc::pthread_mutex_t,
    _marker: core::marker::PhantomData<&'a PerCpuLocks>,
}

impl Drop for PerCpuLockGuard<'_> {
    fn drop(&mut self) {
        // SAFETY: the guard is only constructed after a successful `trylock` on
        // `self.lock`, so this thread holds it; the mutex is stably located in
        // the parent `PerCpuLocks`, which outlives the guard ('a). Faithful to
        // schbench's `pthread_mutex_unlock(lock)` at the end of `do_work`
        // (`schbench.c:1411`).
        unsafe { libc::pthread_mutex_unlock(self.lock) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, Ordering};

    /// Non-atomic counter guarded only by the per-CPU lock; correctness of the
    /// final value depends entirely on the lock providing mutual exclusion.
    struct RawCounter(core::cell::UnsafeCell<u64>);
    // SAFETY: the inner u64 is only ever touched while a PerCpuLockGuard for
    // the relevant CPU lock is held; the test pins all threads to one CPU so
    // they contend on the same lock. Test-only.
    unsafe impl Sync for RawCounter {}

    impl RawCounter {
        /// SAFETY: the caller must hold a [`PerCpuLockGuard`] for the lock all
        /// threads share, so this is the only thread touching the cell. Taking
        /// `&self` makes the closure capture the whole (`Sync`) `RawCounter`
        /// rather than the inner non-`Sync` `UnsafeCell`.
        unsafe fn incr(&self) {
            let p = self.0.get();
            unsafe { *p += 1 };
        }
        /// Read the final value; call only after all writers have joined.
        fn get(&self) -> u64 {
            // SAFETY: no concurrent writers remain at the read site.
            unsafe { *self.0.get() }
        }
    }

    /// Pin the calling thread to `cpu` so its `sched_getcpu` is deterministic.
    fn pin_to_cpu(cpu: usize) -> bool {
        // SAFETY: a zeroed cpu_set_t with one CPU set, applied to the calling
        // thread (pid 0). Returns whether the syscall succeeded.
        unsafe {
            let mut set: libc::cpu_set_t = core::mem::zeroed();
            libc::CPU_ZERO(&mut set);
            libc::CPU_SET(cpu, &mut set);
            libc::sched_setaffinity(0, core::mem::size_of::<libc::cpu_set_t>(), &set) == 0
        }
    }

    #[test]
    fn new_initializes_requested_lock_count() {
        let locks = PerCpuLocks::new(4);
        assert_eq!(locks.len(), 4);
    }

    #[test]
    fn lock_then_drop_allows_reacquire_same_thread() {
        // A NORMAL mutex would deadlock if re-locked while held; this proves
        // the guard's Drop actually unlocks. No hang => unlock works.
        let locks = PerCpuLocks::new(2);
        {
            let _g = locks.lock_this_cpu();
        } // guard drops, unlocks
        let _g2 = locks.lock_this_cpu();
        // Reaching here without blocking is the assertion.
    }

    #[test]
    fn same_cpu_threads_get_mutual_exclusion() {
        // Pin every thread to CPU 0 so they all dispatch to lock[0] and the
        // per-CPU lock degenerates to one shared mutex. A non-atomic counter
        // incremented under the guard must reach exactly N*M with no lost
        // updates -- which holds only if the lock serializes the threads.
        const THREADS: usize = 4;
        const ITERS: u64 = 20_000;

        let locks = PerCpuLocks::new(8);
        let counter = RawCounter(core::cell::UnsafeCell::new(0));
        // If pinning is unavailable, threads may spread across CPUs and take
        // different locks (no MX) -- skip rather than report a false failure.
        let pinned_ok = AtomicBool::new(true);

        std::thread::scope(|s| {
            for _ in 0..THREADS {
                s.spawn(|| {
                    if !pin_to_cpu(0) {
                        pinned_ok.store(false, Ordering::Relaxed);
                        return;
                    }
                    for _ in 0..ITERS {
                        let _g = locks.lock_this_cpu();
                        // SAFETY: all threads are pinned to CPU 0, so every
                        // lock_this_cpu returns lock[0]; the increment runs
                        // under that lock => exclusive access.
                        unsafe { counter.incr() };
                    }
                });
            }
        });

        if pinned_ok.load(Ordering::Relaxed) {
            let total = counter.get();
            assert_eq!(
                total,
                THREADS as u64 * ITERS,
                "lost updates => the per-CPU lock did not provide mutual exclusion"
            );
        }
    }
}
