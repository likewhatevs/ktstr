//! Watchpoint publish / disarm / rebind helpers.
//!
//! The freeze coordinator's scan-tick logic publishes the resolved
//! `*scx_root + exit_kind_offset` KVA into [`super::super::vcpu::WatchpointArm`]
//! so every vCPU thread arms DR0 (x86_64) / DBGWVR0 (aarch64) on the
//! next `self_arm_watchpoint` call. This module owns the state-
//! transition logic that compares `*scx_root` against the cached
//! `last_sched_kva` and either disarms (slab freed), holds (no
//! change), or republishes (fresh attach).
//!
//! The A → B rebind case is a known race where DR0 still holds A's
//! KVA between the atomic publish and each vCPU's next
//! `KVM_SET_GUEST_DEBUG` reissue. To avoid firing
//! DR0 on A's address while reading B's `kind_host_ptr`, the rebind
//! is split into disarm-this-tick + republish-next-tick: clearing
//! `request_kva` first forces every vCPU's `self_arm_watchpoint` to
//! see `requests[0] = 0` and reissue without slot 0's enable bits,
//! which clears DR0 / DR7's L0/G0. The next scan tick observes
//! `last_sched_kva == 0 && sched_kva == B (non-zero)` and falls
//! into the fresh-attach arm.
//!
//! Splitting the logic into `pub(super) fn` lets unit tests drive
//! the state machine over arbitrary `(last_sched_kva, sched_kva)`
//! transitions and assert the resulting WatchpointArm slot 0 state
//! without booting a VM. The closure-local form was untestable.

use std::sync::Arc;
use std::sync::atomic::Ordering;

use super::super::vcpu::WatchpointArm;
use crate::monitor;

/// Outcome of a single scan-tick `*scx_root` poll. Returned to the
/// caller so the run-loop can update its `last_sched_kva` cache and
/// log appropriately. The variants mirror the three transition
/// shapes the rebind block handles.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum WatchpointPublishResult {
    /// `sched_kva == last_sched_kva` — no transition, no work done.
    /// Caller leaves its cache unchanged.
    Unchanged,
    /// `sched_kva == 0 && last_sched_kva != 0` — scheduler detached;
    /// `request_kva` cleared and `kind_host_ptr` nulled.
    /// Caller stores `last_sched_kva = 0`.
    Detached,
    /// `last_sched_kva != 0 && sched_kva != 0 && sched_kva !=
    /// last_sched_kva` — A → B rebind. This tick disarms only;
    /// the next tick observes `last_sched_kva == 0` and republishes.
    /// Caller stores `last_sched_kva = 0`.
    RebindDisarmed { previous: u64, next: u64 },
    /// `last_sched_kva == 0 && sched_kva != 0` — fresh attach (or
    /// post-rebind republish). `kind_host_ptr` and `request_kva`
    /// stored; arm gate flipped. Caller stores
    /// `last_sched_kva = sched_kva`.
    Published { exit_kind_kva: u64, kind_pa: u64 },
    /// Same as [`Self::Published`] in shape but the exit_kind KVA →
    /// PA → host pointer chain failed at the page-table walk or the
    /// `host_ptr_for_pa` lookup. Nothing was published; caller leaves
    /// `last_sched_kva` unchanged so the next scan tick retries
    /// against the same `*scx_root` value.
    PublishDeferred { exit_kind_kva: u64 },
}

/// Drive the watchpoint publish state machine over the resolved
/// `sched_kva` (= `*scx_root`, read by the caller). Pure function
/// over `last_sched_kva` and the resolved `sched_kva` — the caller
/// owns the cache and the read.
///
/// # Store ordering
///
/// All ordered writes follow the contract documented on
/// [`super::super::vcpu::WatchpointArm`]:
///
/// * Detach / RebindDisarmed: `request_kva` Release THEN
///   `kind_host_ptr` Release. Clearing `request_kva` first so a
///   racing vCPU's Acquire load returns 0 and `self_arm_watchpoint`
///   reissues `KVM_SET_GUEST_DEBUG` without slot 0's enable bits.
/// * Published: `kind_host_ptr` Release THEN `request_kva` Release.
///   Publishing the host pointer first so the vCPU thread's Acquire
///   load on `request_kva` synchronizes-with both stores — without
///   this ordering a vCPU could observe non-zero `request_kva` and
///   read a stale (or null) `kind_host_ptr` from the prior epoch.
///
/// # Rebind safety
///
/// The A → B rebind splits into two scan ticks. Tick N observes
/// `sched_kva = B (non-zero)` with `last_sched_kva = A (non-zero,
/// !=B)` and returns [`WatchpointPublishResult::RebindDisarmed`] —
/// caller resets cache to 0 but DOES NOT publish B. Tick N+1
/// observes `sched_kva = B` with `last_sched_kva = 0` and falls
/// into the fresh-attach `Published` arm. The 250 ms scan interval
/// is much larger than worst-case KVM_RUN slice, so every vCPU has
/// re-armed (or at least exited and re-entered KVM_RUN) by the
/// next tick — DR0 is cleared before B is published.
///
/// During the disarm-then-arm window, fires on A's address read
/// `kind_host_ptr = null` and bail in `latch_slot0_with_gate`
/// without dereferencing. The BPF .bss late-trigger fallback
/// remains active throughout.
pub(super) fn republish_watchpoint_on_rebind(
    sched_kva: u64,
    last_sched_kva: u64,
    exit_kind_offset: u32,
    watchpoint: &Arc<WatchpointArm>,
    kernel: &monitor::guest::GuestKernel,
    mem: &monitor::reader::GuestMem,
) -> WatchpointPublishResult {
    if sched_kva == last_sched_kva {
        return WatchpointPublishResult::Unchanged;
    }
    if sched_kva == 0 {
        // Scheduler detached (`*scx_root == 0`): the `scx_sched` the
        // slot watched is RCU-freed (scx_root_disable's
        // `RCU_INIT_POINTER(scx_root, NULL)`), so DR0 must not keep
        // firing into the now-freed (and possibly slab-recycled) page.
        // Disarm slot 0; the caller stores `last_sched_kva = 0` so the
        // next scan tick republishes on the next attach. See
        // [`WatchpointArm::disarm`] for the `request_kva`-first store
        // ordering and the hit-clear rationale (a stale fire from this
        // scheduler's teardown must not poison the next scheduler's
        // late-trigger arm; real errors still latch via the BPF `.bss`
        // `err_triggered` path, independent of `hit`).
        watchpoint.disarm();
        return WatchpointPublishResult::Detached;
    }
    if last_sched_kva != 0 {
        // A → B rebind: disarm-only this tick — DR0 may still hold A's
        // KVA between the publish and each vCPU's next
        // `KVM_SET_GUEST_DEBUG` reissue. The caller resets
        // `last_sched_kva = 0` so the next scan tick's `sched_kva !=
        // last_sched_kva` check sees `0 → B (non-zero)` and falls
        // through to the fresh-attach publish path below. Same disarm
        // (and same hit-clear rationale — the pre-rebind fire was on
        // the OLD scheduler's exit_kind write) as the Detached arm; see
        // [`WatchpointArm::disarm`].
        watchpoint.disarm();
        return WatchpointPublishResult::RebindDisarmed {
            previous: last_sched_kva,
            next: sched_kva,
        };
    }
    // 0 → non-zero: fresh attach (or post-rebind republish).
    // exit_kind field KVA = base of scx_sched (vmalloc/slab) +
    // BTF-resolved field offset. The kernel writes a 4-byte
    // atomic_t at this address via `atomic_try_cmpxchg` in
    // scx_claim_exit (error classes) / scx_disable_workfn
    // (SCX_EXIT_DONE); the hardware watchpoint catches every such
    // write regardless of the SCX_EXIT_* class.
    let exit_kind_kva = sched_kva.wrapping_add(exit_kind_offset as u64);
    // Translate the field's KVA to a host pointer so the vCPU
    // thread can `read_volatile` the post-store value at fire time
    // and gate `watchpoint.hit` on the error-class threshold
    // (1024). Without this, the watchpoint fires on every
    // exit_kind transition — including the clean
    // `KIND -> SCX_EXIT_DONE` write that `scx_unregister` issues
    // at end of every test — and produces a bogus failure dump on
    // every clean shutdown.
    //
    // The kva lives in scx_sched's slab/vmalloc page; translate
    // via the same direct-mapping-or-page-walk path the BPF .bss
    // poll uses, then look up the host pointer via
    // `host_ptr_for_pa`. `field_size` is 4 (the atomic_t holding
    // exit_kind is a u32). On any resolve failure we skip
    // publication this iteration so the caller leaves
    // `last_sched_kva` unchanged and the next scan retries against
    // the same `*scx_root` value.
    let walk = kernel.walk_context();
    let resolve = (|| {
        let kind_pa = crate::monitor::idr::translate_any_kva(
            mem,
            walk.cr3_pa,
            walk.page_offset,
            exit_kind_kva,
            walk.l5,
            walk.tcr_el1,
        )?;
        let host_ptr = mem.host_ptr_for_pa(kind_pa, 4)? as *mut u32;
        Some((kind_pa, host_ptr))
    })();
    match resolve {
        Some((kind_pa, kind_host_ptr)) => {
            // Publication ordering: store `kind_host_ptr` BEFORE
            // `request_kva`. The vCPU thread loads `request_kva`
            // with Acquire and only reads `kind_host_ptr` after
            // — the Release ordering on `request_kva` makes the
            // earlier `kind_host_ptr` store visible. Without this
            // ordering a vCPU could observe a non-zero
            // `request_kva`, arm the watchpoint, fire on the very
            // next instruction, and read a stale (or null)
            // `kind_host_ptr` from the prior epoch.
            watchpoint
                .kind_host_ptr
                .store(kind_host_ptr, Ordering::Release);
            watchpoint
                .request_kva
                .store(exit_kind_kva, Ordering::Release);
            // Flip the fast-path gate so every vCPU's
            // `self_arm_watchpoint` stops short-circuiting on
            // `any_armed == 0` and falls through to the per-slot
            // Acquire load on `request_kva`. Idempotent —
            // repeated calls keep the gate at 1, including across
            // re-bind cycles. Must follow the Release on
            // `request_kva`: `mark_armed` is `Relaxed`, so the
            // synchronizes-with edge that publishes the new KVA
            // value comes from the slot's Release/Acquire pair,
            // not the gate.
            watchpoint.mark_armed();
            WatchpointPublishResult::Published {
                exit_kind_kva,
                kind_pa,
            }
        }
        None => WatchpointPublishResult::PublishDeferred { exit_kind_kva },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// [`WatchpointArm::disarm`] clears slot 0: `request_kva` → 0,
    /// `kind_host_ptr` → null, `hit` → false. This is the teardown the
    /// scan-tick Detached / RebindDisarmed arms AND the synchronous
    /// sched-swap-notify path in `freeze_coord::run_vm` both rely on so
    /// a stale DR0 cannot keep firing into a now-RCU-freed (and
    /// possibly slab-recycled) `scx_sched` page and latch a phantom
    /// failure dump. Regression-pins the swap-notify teardown's
    /// watchpoint half (the rest — `cached_exit_kind_pa` /
    /// `last_sched_kva` / the owned accessor — are run-loop locals
    /// cleared inline alongside this call).
    #[test]
    fn disarm_clears_slot0_state() {
        let wp = WatchpointArm::new().expect("WatchpointArm::new");
        // Arm slot 0 with a plausible exit_kind KVA, a non-null host
        // pointer, and a latched hit — the state a Published transition
        // leaves behind. (disarm only stores/clears the pointer value;
        // it never dereferences it, so a stack address is fine here.)
        let mut dummy: u32 = 0;
        wp.kind_host_ptr
            .store(&mut dummy as *mut u32, Ordering::Release);
        wp.request_kva
            .store(0xffff_8000_dead_beef, Ordering::Release);
        wp.hit.store(true, Ordering::Release);
        wp.mark_armed();

        wp.disarm();

        assert_eq!(
            wp.request_kva.load(Ordering::Acquire),
            0,
            "request_kva must clear so self_arm_watchpoint reissues without slot 0"
        );
        assert!(
            wp.kind_host_ptr.load(Ordering::Acquire).is_null(),
            "kind_host_ptr must null so latch_slot0_with_gate bails on a stale fire"
        );
        assert!(
            !wp.hit.load(Ordering::Acquire),
            "sticky hit must clear so the late-trigger arm doesn't re-fire on the \
             closed-out scheduler's teardown write"
        );
    }
}
