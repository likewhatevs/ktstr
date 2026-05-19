//! On-demand freeze rendezvous gate.
//!
//! A single coordinator-thread `AtomicBool` that serializes the
//! on-demand capture paths (TLV CAPTURE, periodic snapshot,
//! user-watchpoint late-trigger, and cold-path kernel-memory op
//! dispatch) so concurrent freeze rendezvous cannot interleave on
//! the same VM. Each path acquires the gate before invoking the
//! `freeze_and_dispatch` closure and releases on scope exit via
//! RAII `Drop` — including panic unwind.
//!
//! # Panic safety
//!
//! The `Drop` body is a single [`AtomicBool::store`] which is
//! non-panicking. The gate clears on every scope exit: normal
//! return, early return, AND panic unwind. This closes the
//! permanent-deadlock failure mode that the prior unconditional
//! `store(false)` trail-release pattern exposed — a panic in
//! per-slot cleanup (e.g. `lock_unpoisoned`, `tag_guard.clear()`)
//! would skip the gate release and permanently block the rest of
//! the run because every future CAPTURE / periodic / watchpoint /
//! ColdOp dispatch would reject on the gate-held check.
//!
//! Contrast with `cloud-hypervisor`'s
//! `vmm/src/device_manager.rs::DeviceManager::drop` which CAN
//! panic mid-`Drop` via `.lock().unwrap()`. CH accepts that risk
//! because `Drop` runs only at terminal VMM teardown; ktstr's
//! guard runs every freeze rendezvous and cannot tolerate the
//! same risk.
//!
//! # Exclusivity invariant
//!
//! The guard's lifetime IS the exclusivity period. While a
//! [`OnDemandGateGuard`] is alive, no other thread can acquire
//! the gate (their [`OnDemandGateGuard::try_acquire`] returns
//! `None`). The `Drop`'s `store` is the precise moment ownership
//! transfers. Two-phase release (drop guard, then immediately
//! re-acquire) succeeds because the second acquire reads the
//! post-`Drop`-`store` state.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

/// RAII guard for the `freeze_coord_on_demand_in_flight` gate.
///
/// Holding this guard means "this thread owns the freeze
/// rendezvous gate" — no other CAPTURE / periodic / watchpoint /
/// ColdOp dispatch path can run a rendezvous until this guard is
/// dropped. See module-level doc for panic-safety + exclusivity
/// invariants.
pub(super) struct OnDemandGateGuard<'a> {
    gate: &'a Arc<AtomicBool>,
}

impl<'a> OnDemandGateGuard<'a> {
    /// Try to acquire the gate. Returns `Some(guard)` if the gate
    /// was free (swap-from-`false` to `true` succeeded); returns
    /// `None` if the gate was already held.
    ///
    /// The returned guard releases the gate on `Drop`.
    pub(super) fn try_acquire(gate: &'a Arc<AtomicBool>) -> Option<Self> {
        if gate.swap(true, Ordering::AcqRel) {
            None
        } else {
            Some(Self { gate })
        }
    }
}

impl Drop for OnDemandGateGuard<'_> {
    fn drop(&mut self) {
        // Release pairs with the next `try_acquire`'s AcqRel
        // swap (Acquire half), establishing the inter-guard
        // happens-before edge. `AtomicBool::store` is infallible;
        // this body cannot panic even mid-unwind. See module doc
        // for the panic-safety + CH contrast.
        self.gate.store(false, Ordering::Release);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `try_acquire` on a free gate returns Some and flips the
    /// atomic to true.
    #[test]
    fn try_acquire_on_free_gate_succeeds_and_holds() {
        let gate = Arc::new(AtomicBool::new(false));
        let guard = OnDemandGateGuard::try_acquire(&gate).expect("free gate");
        assert!(gate.load(Ordering::Acquire), "gate held after acquire");
        drop(guard);
    }

    /// `try_acquire` on a held gate returns None without flipping.
    #[test]
    fn try_acquire_on_held_gate_returns_none() {
        let gate = Arc::new(AtomicBool::new(true));
        let opt = OnDemandGateGuard::try_acquire(&gate);
        assert!(opt.is_none(), "held gate must reject");
        assert!(gate.load(Ordering::Acquire), "gate still held");
    }

    /// `Drop` releases the gate so the next `try_acquire` succeeds.
    #[test]
    fn drop_releases_gate_for_next_acquire() {
        let gate = Arc::new(AtomicBool::new(false));
        let guard = OnDemandGateGuard::try_acquire(&gate).expect("acquire 1");
        drop(guard);
        assert!(!gate.load(Ordering::Acquire), "gate released by Drop");
        let _guard2 = OnDemandGateGuard::try_acquire(&gate).expect("acquire 2 after release");
        assert!(gate.load(Ordering::Acquire), "gate re-held by acquire 2");
    }

    /// `Drop` releases the gate during panic unwind. This is the
    /// load-bearing test for the deadlock-on-panic class: without
    /// it, a panic after a successful atomic acquire would
    /// permanently block subsequent rendezvous.
    #[test]
    fn drop_on_panic_releases_gate() {
        let gate = Arc::new(AtomicBool::new(false));
        let gate_for_catch = gate.clone();
        let result = std::panic::catch_unwind(move || {
            let _guard =
                OnDemandGateGuard::try_acquire(&gate_for_catch).expect("acquire before panic");
            panic!("simulated panic mid-critical-section");
        });
        assert!(result.is_err(), "catch_unwind must see the panic");
        assert!(
            !gate.load(Ordering::Acquire),
            "gate released even under panic unwind"
        );
    }

    /// Sequential acquire/drop cycles correctly mutate the gate
    /// across iterations — pins that the guard's lifetime IS the
    /// exclusivity period.
    #[test]
    fn sequential_acquire_drop_cycles() {
        let gate = Arc::new(AtomicBool::new(false));
        for _ in 0..10 {
            let guard = OnDemandGateGuard::try_acquire(&gate).expect("acquire in loop");
            assert!(gate.load(Ordering::Acquire));
            drop(guard);
            assert!(!gate.load(Ordering::Acquire));
        }
    }

    /// Source-grep regression test: no bare-atomic `.swap()` or
    /// `.store()` calls on `freeze_coord_on_demand_in_flight`
    /// exist anywhere in `mod.rs`. A future contributor who added
    /// a new gate-mutation site bypassing [`OnDemandGateGuard`]
    /// would silently re-open the deadlock-on-panic class because
    /// `Drop` wouldn't fire on the bypassed path. Mirrors the
    /// `dispatch_arms_call_validate_target_helpers` source-grep
    /// pattern in `kernel_op_dispatch.rs`.
    #[test]
    fn no_bare_gate_atomic_mutations_in_mod_rs() {
        let src = include_str!("mod.rs");
        let needle = "freeze_coord_on_demand_in_flight";
        let mut cursor = 0;
        let mut bare_swap = 0_usize;
        let mut bare_store = 0_usize;
        while let Some(pos) = src[cursor..].find(needle) {
            cursor += pos + needle.len();
            let tail = src[cursor..].trim_start();
            if tail.starts_with(".swap(") {
                bare_swap += 1;
            } else if tail.starts_with(".store(") {
                bare_store += 1;
            }
        }
        assert_eq!(
            bare_swap, 0,
            "mod.rs must not call freeze_coord_on_demand_in_flight.swap() — \
             use OnDemandGateGuard::try_acquire() instead"
        );
        assert_eq!(
            bare_store, 0,
            "mod.rs must not call freeze_coord_on_demand_in_flight.store() — \
             the guard's Drop is the only release path"
        );
    }

    /// Concurrent `try_acquire` from two threads: only one wins
    /// each time. Probabilistic stress for ~1000 iterations.
    #[test]
    fn concurrent_try_acquire_is_mutually_exclusive() {
        use std::sync::atomic::AtomicUsize;
        use std::thread;

        let gate = Arc::new(AtomicBool::new(false));
        let in_critical = Arc::new(AtomicUsize::new(0));
        let max_concurrent = Arc::new(AtomicUsize::new(0));

        let mut handles = vec![];
        for _ in 0..2 {
            let gate = gate.clone();
            let in_critical = in_critical.clone();
            let max_concurrent = max_concurrent.clone();
            handles.push(thread::spawn(move || {
                for _ in 0..1000 {
                    if let Some(_guard) = OnDemandGateGuard::try_acquire(&gate) {
                        let cur = in_critical.fetch_add(1, Ordering::AcqRel) + 1;
                        let mut prev_max = max_concurrent.load(Ordering::Acquire);
                        while cur > prev_max {
                            match max_concurrent.compare_exchange(
                                prev_max,
                                cur,
                                Ordering::AcqRel,
                                Ordering::Acquire,
                            ) {
                                Ok(_) => break,
                                Err(actual) => prev_max = actual,
                            }
                        }
                        in_critical.fetch_sub(1, Ordering::AcqRel);
                    }
                }
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
        assert_eq!(
            max_concurrent.load(Ordering::Acquire),
            1,
            "at most one guard alive at any time"
        );
    }
}
