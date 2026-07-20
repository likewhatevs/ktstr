//! Host-side bounded retry of a whole guest cold boot on an
//! AP-bring-up infra fault, detected from either side of the boot.
//!
//! On oversubscribed CI hosts an application processor can miss its
//! INIT-SIPI alive-window (the kernel's `cpuhp_wait_for_sync_state`
//! gives it 10 s of guest WALL-CLOCK, which host stalls burn while the
//! AP thread gets no cycles) and land present-but-offline. Two detectors
//! feed this retry:
//!
//!   * Guest side — the guest PID-1 checks every possible CPU is online
//!     BEFORE any test or scheduler runs
//!     (`crate::vmm::rust_init::topology::all_possible_cpus_online`) and
//!     PANICs on a gap; the panic reaches the host as
//!     [`crate::vmm::VmResult::crash_message`] carrying
//!     [`AP_BRINGUP_GAP_MARKER`].
//!   * Host side — the progressive AP-ready gate in
//!     `KtstrVm::spawn_ap_threads` bails with
//!     [`crate::vmm::freeze_coord::ApGateTimeout`] when an AP host thread
//!     receives excessive setup/blocked-observer service without reaching
//!     `KVM_RUN`, so the guest is never even released to boot.
//!
//! Recovery is a fresh cold boot — exactly what bare metal does — so the
//! scheduler only ever sees a topology assembled by a clean boot, never
//! a hotplug-healed one. Both faults are pre-test, so retrying is safe
//! and idempotent: no test or scheduler side effects exist yet.

/// Identifying substring of the guest PID-1 AP-bring-up-gap panic
/// message. The guest formats the full message in
/// `crate::vmm::rust_init::topology` around this exact substring, and
/// the host keys its boot retry on `crash_message.contains(MARKER)`.
/// Kept crate-internal and referenced by both sides so a reword on one
/// end cannot silently desync from the other — a sync test in
/// `rust_init/tests.rs` pins the guest format against it.
pub(crate) const AP_BRINGUP_GAP_MARKER: &str = "failed to come online (AP bring-up failed";

/// Kernel log evidence that a guest rejected ktstr's x2APIC setup because
/// it has neither interrupt remapping nor the KVM guest callback which
/// advertises extended MSI destination IDs.
const X2APIC_REJECTED_MARKER: &str = "IRQ remapping doesn't support X2APIC mode";

/// Kernel log evidence that the AP gap is the deterministic 8-bit xAPIC
/// ceiling, not an AP which transiently missed its INIT-SIPI window.
const XAPIC_ID_LIMIT_MARKER: &str = "has invalid APIC ID";

/// Whether a guest-side AP gap is a deterministic guest-kernel capability
/// mismatch rather than a transient AP scheduling miss.
///
/// ktstr enables x2APIC for a topology whose APIC IDs exceed 254. Some
/// distro kernels subsequently disable it when built without the KVM guest
/// hooks and without an interrupt-remapping IOMMU. Linux then rejects the
/// first AP at the 8-bit xAPIC ceiling. A cold retry cannot change either
/// kernel capability; requiring the AP-gap panic plus both explicit kernel
/// messages keeps this classification narrower than a generic AP failure.
pub(crate) fn guest_kernel_rejected_wide_apic(result: &crate::vmm::VmResult) -> bool {
    result
        .crash_message
        .as_deref()
        .is_some_and(|m| m.contains(AP_BRINGUP_GAP_MARKER))
        && result.output.contains(X2APIC_REJECTED_MARKER)
        && result.output.contains(XAPIC_ID_LIMIT_MARKER)
}

/// Total boot attempts (including the first) before giving up on the
/// AP-bring-up-gap fault. On the final attempt the result is returned
/// as-is — every downstream path behaves exactly as it would without
/// the retry.
pub(crate) const AP_GAP_BOOT_ATTEMPTS: u32 = 3;

/// Run a cell VM with bounded retry on the AP-bring-up-gap infra fault.
///
/// `build_and_run` builds a FRESH VM and runs it, returning the run's
/// [`crate::vmm::VmResult`]. Each call must produce a new VM: `build`
/// consumes the builder and `run` consumes the boot, so a retry cannot
/// reuse the prior VM.
///
/// Two distinct AP-bring-up failures are retried, both drawing on the
/// same [`AP_GAP_BOOT_ATTEMPTS`] budget (a boot that alternates between
/// the two still caps at that many total attempts):
///
///   1. The guest side: the returned result carries a `crash_message`
///      containing [`AP_BRINGUP_GAP_MARKER`] — the guest PID-1 panicked
///      pre-test because an AP came up present-but-offline.
///   2. The host side: the build/run `Err` chain downcasts to
///      [`crate::vmm::freeze_coord::ApGateTimeout`] — the host-side
///      AP-ready boot gate tripped because an AP thread never reached
///      `KVM_RUN`. (`downcast_ref` traverses the anyhow context chain,
///      so the production call site's `.context(...)` layers do not hide it.)
///
/// Both are pre-test infra faults with no test/scheduler side effects yet,
/// so a fresh cold boot is a safe, idempotent recovery. The one
/// deterministic AP-gap class — an explicit x2APIC rejection followed by
/// an invalid 8-bit APIC ID — returns immediately because another cold boot
/// cannot change the guest kernel's capabilities. EVERY other outcome
/// returns immediately: a clean run, any other `crash_message` (timeout,
/// scheduler crash), and any other `Err` all propagate as-is. The common
/// all-online path pays nothing beyond the single `crash_message`
/// inspection.
pub(crate) fn run_vm_with_ap_gap_retry<F>(
    mut build_and_run: F,
) -> anyhow::Result<crate::vmm::VmResult>
where
    F: FnMut() -> anyhow::Result<crate::vmm::VmResult>,
{
    let mut attempt = 1u32;
    loop {
        let result = match build_and_run() {
            Ok(result) => result,
            Err(err) => {
                // Host-side AP-ready gate trip: retry the cold boot, same
                // as the guest-side marker below. Any other Err — and the
                // gate trip once the attempt budget is spent — propagates.
                if let Some(gate) = err.downcast_ref::<crate::vmm::freeze_coord::ApGateTimeout>()
                    && attempt < AP_GAP_BOOT_ATTEMPTS
                {
                    eprintln!(
                        "ktstr: vCPU bring-up gate tripped ({gate}); retrying boot (attempt {}/{})",
                        attempt + 1,
                        AP_GAP_BOOT_ATTEMPTS,
                    );
                    attempt += 1;
                    continue;
                }
                return Err(err);
            }
        };
        let ap_gap = result
            .crash_message
            .as_deref()
            .is_some_and(|m| m.contains(AP_BRINGUP_GAP_MARKER));
        if !ap_gap || attempt >= AP_GAP_BOOT_ATTEMPTS {
            return Ok(result);
        }
        if guest_kernel_rejected_wide_apic(&result) {
            eprintln!(
                "ktstr: guest kernel rejected x2APIC and hit the 8-bit APIC-ID \
                 ceiling; another cold boot cannot change this kernel capability"
            );
            return Ok(result);
        }
        // Failure-story diagnostic (ungated), mirroring the watchdog /
        // send_sys_rdy timeout WARNs. Only fires on the rare AP-gap
        // path — the all-online case never reaches here.
        eprintln!(
            "ktstr: guest booted with offline CPUs ({}); retrying boot (attempt {}/{})",
            result.crash_message.as_deref().unwrap_or_default().trim(),
            attempt + 1,
            AP_GAP_BOOT_ATTEMPTS,
        );
        attempt += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    /// A `VmResult` whose `crash_message` is `msg` (everything else is
    /// the neutral test fixture). Drives the retry-decision paths
    /// without booting a VM.
    fn result_with_crash(msg: Option<&str>) -> crate::vmm::VmResult {
        let mut r = crate::test_support::test_helpers::make_vm_result("", "", 1, false);
        r.crash_message = msg.map(str::to_string);
        r
    }

    /// A deterministic x2APIC capability rejection is returned after the
    /// first boot. Retrying cannot change the guest kernel config.
    #[test]
    fn no_retry_on_explicit_xapic_ceiling() {
        let calls = Cell::new(0u32);
        let msg = format!("CPUs [144] {AP_BRINGUP_GAP_MARKER}; 144/192 online)");
        let out = run_vm_with_ap_gap_retry(|| {
            calls.set(calls.get() + 1);
            let mut result = result_with_crash(Some(&msg));
            result.output = format!(
                "x2apic: {X2APIC_REJECTED_MARKER}\n\
                 smpboot: CPU 144 {XAPIC_ID_LIMIT_MARKER} 100. Aborting bringup"
            );
            Ok(result)
        })
        .unwrap();
        assert_eq!(
            calls.get(),
            1,
            "a guest kernel capability mismatch must not consume cold retries"
        );
        assert!(guest_kernel_rejected_wide_apic(&out));
    }

    /// One suggestive kernel line is not sufficient to suppress the normal
    /// AP-gap retry; the narrow classifier requires the complete story.
    #[test]
    fn partial_xapic_evidence_still_retries() {
        let calls = Cell::new(0u32);
        let msg = format!("CPUs [144] {AP_BRINGUP_GAP_MARKER}; 144/192 online)");
        let out = run_vm_with_ap_gap_retry(|| {
            calls.set(calls.get() + 1);
            let mut result = result_with_crash(Some(&msg));
            result.output = X2APIC_REJECTED_MARKER.to_string();
            Ok(result)
        })
        .unwrap();
        assert_eq!(calls.get(), AP_GAP_BOOT_ATTEMPTS);
        assert!(!guest_kernel_rejected_wide_apic(&out));
    }

    /// A clean run (no crash_message) returns on the first attempt — the
    /// closure is invoked exactly once, no retry.
    #[test]
    fn no_retry_on_clean_run() {
        let calls = Cell::new(0u32);
        let out = run_vm_with_ap_gap_retry(|| {
            calls.set(calls.get() + 1);
            Ok(result_with_crash(None))
        })
        .unwrap();
        assert_eq!(calls.get(), 1, "clean run must not retry");
        assert!(out.crash_message.is_none());
    }

    /// A non-AP-gap crash (a real scheduler failure) returns on the
    /// first attempt — retry keys STRICTLY on the marker.
    #[test]
    fn no_retry_on_other_failure() {
        let calls = Cell::new(0u32);
        let out = run_vm_with_ap_gap_retry(|| {
            calls.set(calls.get() + 1);
            Ok(result_with_crash(Some(
                "panicked at src/foo.rs:1: unrelated scheduler crash",
            )))
        })
        .unwrap();
        assert_eq!(calls.get(), 1, "a non-AP-gap crash must not retry");
        assert!(out.crash_message.as_deref().unwrap().contains("unrelated"));
    }

    /// A persistent AP-gap gives up after exactly [`AP_GAP_BOOT_ATTEMPTS`]
    /// attempts and returns the last (still-failing) result.
    #[test]
    fn gives_up_after_max_attempts() {
        let calls = Cell::new(0u32);
        let msg = format!("CPUs [4] {AP_BRINGUP_GAP_MARKER}; 127/128 online)");
        let out = run_vm_with_ap_gap_retry(|| {
            calls.set(calls.get() + 1);
            Ok(result_with_crash(Some(&msg)))
        })
        .unwrap();
        assert_eq!(
            calls.get(),
            AP_GAP_BOOT_ATTEMPTS,
            "a persistent AP gap must be retried up to the attempt cap",
        );
        assert!(
            out.crash_message
                .as_deref()
                .unwrap()
                .contains(AP_BRINGUP_GAP_MARKER),
            "the final result is returned as-is on give-up",
        );
    }

    /// An AP gap that clears on a later attempt stops retrying the moment
    /// a clean result arrives and returns it.
    #[test]
    fn stops_retrying_once_gap_clears() {
        let calls = Cell::new(0u32);
        let msg = format!("CPUs [4] {AP_BRINGUP_GAP_MARKER}; 127/128 online)");
        let out = run_vm_with_ap_gap_retry(|| {
            let n = calls.get() + 1;
            calls.set(n);
            // Fail with the AP gap on attempt 1, succeed on attempt 2.
            if n == 1 {
                Ok(result_with_crash(Some(&msg)))
            } else {
                Ok(result_with_crash(None))
            }
        })
        .unwrap();
        assert_eq!(
            calls.get(),
            2,
            "second boot cleared the gap; no third attempt"
        );
        assert!(
            out.crash_message.is_none(),
            "the clean retry result is returned"
        );
    }

    /// A build/run `Err` that is NOT a gate trip propagates immediately —
    /// it is neither a marker-bearing result nor an [`ApGateTimeout`], so
    /// it is never retried.
    #[test]
    fn propagates_err_without_retry() {
        let calls = Cell::new(0u32);
        let out = run_vm_with_ap_gap_retry(|| {
            calls.set(calls.get() + 1);
            Err(anyhow::anyhow!("build failed"))
        });
        assert_eq!(calls.get(), 1, "an unrelated Err must not retry");
        assert!(out.is_err());
    }

    /// An `anyhow::Error` wrapping an [`ApGateTimeout`] — the host-side
    /// AP-ready gate trip the retry now recognizes. `ctx`, when set, adds
    /// a `.context(...)` layer to prove the downcast still traverses the
    /// chain (the production path wraps the error with context).
    fn gate_timeout_err(ctx: Option<&str>) -> anyhow::Error {
        let e = anyhow::Error::new(crate::vmm::freeze_coord::ApGateTimeout {
            not_ready: vec![3],
            elapsed: std::time::Duration::from_secs(30),
            delivered_service: std::time::Duration::from_secs(2),
            blocked_observer_service: std::time::Duration::ZERO,
            killed: false,
            evidence: "  vCPU 3: never scheduled (no TID stamped)\n".to_string(),
        });
        match ctx {
            Some(c) => e.context(c.to_string()),
            None => e,
        }
    }

    /// A persistent host-side gate trip is retried up to the shared
    /// attempt cap, then the final `Err` propagates as-is.
    #[test]
    fn retries_gate_timeout_up_to_cap() {
        let calls = Cell::new(0u32);
        let out = run_vm_with_ap_gap_retry(|| {
            calls.set(calls.get() + 1);
            Err(gate_timeout_err(None))
        });
        assert_eq!(
            calls.get(),
            AP_GAP_BOOT_ATTEMPTS,
            "a persistent gate trip must retry up to the attempt cap",
        );
        assert!(
            out.unwrap_err()
                .downcast_ref::<crate::vmm::freeze_coord::ApGateTimeout>()
                .is_some(),
            "the final gate-trip Err is returned as-is on give-up",
        );
    }

    /// A gate trip that clears on a later attempt stops retrying and
    /// returns the clean result.
    #[test]
    fn stops_retrying_once_gate_clears() {
        let calls = Cell::new(0u32);
        let out = run_vm_with_ap_gap_retry(|| {
            let n = calls.get() + 1;
            calls.set(n);
            if n == 1 {
                Err(gate_timeout_err(None))
            } else {
                Ok(result_with_crash(None))
            }
        })
        .unwrap();
        assert_eq!(
            calls.get(),
            2,
            "second boot cleared the gate; no third attempt"
        );
        assert!(
            out.crash_message.is_none(),
            "the clean retry result is returned"
        );
    }

    /// A context-wrapped gate trip still downcasts through the anyhow
    /// chain, so the production path's `.context(...)` layers do not defeat
    /// the retry.
    #[test]
    fn retries_context_wrapped_gate_timeout() {
        let calls = Cell::new(0u32);
        let out = run_vm_with_ap_gap_retry(|| {
            let n = calls.get() + 1;
            calls.set(n);
            if n == 1 {
                Err(gate_timeout_err(Some("build and run cell VM")))
            } else {
                Ok(result_with_crash(None))
            }
        })
        .unwrap();
        assert_eq!(
            calls.get(),
            2,
            "a context-wrapped gate trip is still recognized and retried"
        );
        assert!(out.crash_message.is_none());
    }
}
