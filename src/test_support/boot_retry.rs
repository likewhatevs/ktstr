//! Host-side bounded retry of a whole guest cold boot when the guest
//! PID-1 reports the AP-bring-up-gap infra fault.
//!
//! On oversubscribed CI hosts an application processor can miss its
//! INIT-SIPI alive-window (the kernel's `cpuhp_wait_for_sync_state`
//! gives it 10 s of guest WALL-CLOCK, which host stalls burn while the
//! AP thread gets no cycles) and land present-but-offline. The guest
//! PID-1 detects that gap BEFORE any test or scheduler runs
//! (`crate::vmm::rust_init::topology::all_possible_cpus_online`) and
//! PANICs — the panic reaches the host as
//! [`crate::vmm::VmResult::crash_message`] carrying
//! [`AP_BRINGUP_GAP_MARKER`].
//!
//! Recovery is a fresh cold boot — exactly what bare metal does — so the
//! scheduler only ever sees a topology assembled by a clean boot, never
//! a hotplug-healed one. The fault is pre-test, so retrying is safe and
//! idempotent: no test or scheduler side effects exist yet.

/// Identifying substring of the guest PID-1 AP-bring-up-gap panic
/// message. The guest formats the full message in
/// `crate::vmm::rust_init::topology` around this exact substring, and
/// the host keys its boot retry on `crash_message.contains(MARKER)`.
/// Kept crate-internal and referenced by both sides so a reword on one
/// end cannot silently desync from the other — a sync test in
/// `rust_init/tests.rs` pins the guest format against it.
pub(crate) const AP_BRINGUP_GAP_MARKER: &str = "failed to come online (AP bring-up failed";

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
/// When the returned result carries a `crash_message` containing
/// [`AP_BRINGUP_GAP_MARKER`] — the guest PID-1 panicked pre-test because
/// an AP missed its INIT-SIPI window — the whole boot is re-run, up to
/// [`AP_GAP_BOOT_ATTEMPTS`] total. EVERY other outcome (a clean run, or
/// ANY other failure — timeout, scheduler crash, boot failure) returns
/// immediately: the retry keys STRICTLY on the marker. A build/run
/// `Err` also propagates immediately (it is not a marker-bearing
/// result). The common all-online path pays nothing beyond the single
/// `crash_message` inspection.
pub(crate) fn run_vm_with_ap_gap_retry<F>(
    mut build_and_run: F,
) -> anyhow::Result<crate::vmm::VmResult>
where
    F: FnMut() -> anyhow::Result<crate::vmm::VmResult>,
{
    let mut attempt = 1u32;
    loop {
        let result = build_and_run()?;
        let ap_gap = result
            .crash_message
            .as_deref()
            .is_some_and(|m| m.contains(AP_BRINGUP_GAP_MARKER));
        if !ap_gap || attempt >= AP_GAP_BOOT_ATTEMPTS {
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

    /// A build/run `Err` propagates immediately — it is not a
    /// marker-bearing result, so it is never retried.
    #[test]
    fn propagates_err_without_retry() {
        let calls = Cell::new(0u32);
        let out = run_vm_with_ap_gap_retry(|| {
            calls.set(calls.get() + 1);
            Err(anyhow::anyhow!("build failed"))
        });
        assert_eq!(calls.get(), 1, "an Err must not retry");
        assert!(out.is_err());
    }
}
