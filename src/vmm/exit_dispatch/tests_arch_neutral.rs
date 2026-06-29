use super::*;

#[test]
fn classify_exit_hlt_returns_none() {
    // VcpuExit::Hlt is the AP-thread idle marker — classify_exit
    // returns None to signal "caller handles the kill-flag check
    // and continues". Pinning here so a future change that
    // accidentally maps Hlt to ExitAction::Continue (which would
    // skip the per-iteration kill recheck) is caught.
    let com1 = PiMutex::new(console::Serial::new(console::COM1_BASE));
    let com2 = PiMutex::new(console::Serial::new(console::COM2_BASE));
    let mut exit = VcpuExit::Hlt;
    let action = classify_exit(&com1, &com2, None, None, None, None, None, &mut exit);
    assert!(action.is_none(), "Hlt must classify as None");
}

#[test]
fn classify_exit_shutdown_variant_is_shutdown() {
    let com1 = PiMutex::new(console::Serial::new(console::COM1_BASE));
    let com2 = PiMutex::new(console::Serial::new(console::COM2_BASE));
    let mut exit = VcpuExit::Shutdown;
    let action = classify_exit(&com1, &com2, None, None, None, None, None, &mut exit);
    assert!(
        matches!(action, Some(ExitAction::Shutdown)),
        "Shutdown variant must classify as ExitAction::Shutdown"
    );
}

#[test]
fn classify_exit_system_event_shutdown_is_shutdown() {
    // KVM_SYSTEM_EVENT_SHUTDOWN (1) is the PSCI-style clean shutdown
    // signal both arches surface for guest-initiated reboot. Must
    // classify as Shutdown so the run loop stops the BSP and APs.
    let com1 = PiMutex::new(console::Serial::new(console::COM1_BASE));
    let com2 = PiMutex::new(console::Serial::new(console::COM2_BASE));
    let data: [u64; 0] = [];
    let mut exit = VcpuExit::SystemEvent(KVM_SYSTEM_EVENT_SHUTDOWN, &data);
    let action = classify_exit(&com1, &com2, None, None, None, None, None, &mut exit);
    assert!(
        matches!(action, Some(ExitAction::Shutdown)),
        "SystemEvent(SHUTDOWN=1) must classify as Shutdown"
    );
}

#[test]
fn classify_exit_system_event_reset_is_shutdown() {
    // KVM_SYSTEM_EVENT_RESET (2) is the guest reboot signal. ktstr
    // treats reboot-as-shutdown — there is no warm-reboot path —
    // so reset and shutdown collapse to the same ExitAction.
    let com1 = PiMutex::new(console::Serial::new(console::COM1_BASE));
    let com2 = PiMutex::new(console::Serial::new(console::COM2_BASE));
    let data: [u64; 0] = [];
    let mut exit = VcpuExit::SystemEvent(KVM_SYSTEM_EVENT_RESET, &data);
    let action = classify_exit(&com1, &com2, None, None, None, None, None, &mut exit);
    assert!(
        matches!(action, Some(ExitAction::Shutdown)),
        "SystemEvent(RESET=2) must classify as Shutdown"
    );
}

#[test]
fn classify_exit_system_event_unknown_type_is_continue() {
    // Unknown SystemEvent types (CRASH=3, WAKEUP=4, future codes)
    // must NOT terminate the run loop — KVM may evolve the event
    // namespace, and ktstr must not falsely shut down on a value
    // it does not recognize. Pinning a non-{1,2} type returns
    // Continue so the BSP polls forward.
    let com1 = PiMutex::new(console::Serial::new(console::COM1_BASE));
    let com2 = PiMutex::new(console::Serial::new(console::COM2_BASE));
    let data: [u64; 0] = [];
    // 99 is well outside KVM_SYSTEM_EVENT_{SHUTDOWN, RESET, CRASH,
    // WAKEUP, S2IDLE, SUSPEND} — picked to defend against future
    // expansion of the legitimate set without flipping this test.
    let mut exit = VcpuExit::SystemEvent(99, &data);
    let action = classify_exit(&com1, &com2, None, None, None, None, None, &mut exit);
    assert!(
        matches!(action, Some(ExitAction::Continue)),
        "SystemEvent with unknown type must classify as Continue, \
         not Shutdown — the run loop must not terminate on \
         unknown KVM event codes"
    );
}

#[test]
fn classify_exit_fail_entry_is_fatal_with_reason() {
    // KVM_EXIT_FAIL_ENTRY surfaces a hardware-side entry failure
    // (typically VMX/SVM consistency check failed). The reason
    // must propagate through to ExitAction::Fatal so the failure
    // dump records the architectural code an operator can decode.
    let com1 = PiMutex::new(console::Serial::new(console::COM1_BASE));
    let com2 = PiMutex::new(console::Serial::new(console::COM2_BASE));
    let mut exit = VcpuExit::FailEntry(0xdead_beef, 7);
    let action = classify_exit(&com1, &com2, None, None, None, None, None, &mut exit);
    match action {
        Some(ExitAction::Fatal(Some(reason))) => assert_eq!(
            reason, 0xdead_beef,
            "FailEntry reason must round-trip into Fatal(Some(_))"
        ),
        other => panic!(
            "FailEntry must classify as Fatal(Some(reason)); got tag {}",
            action_tag(&other)
        ),
    }
}

#[test]
fn classify_exit_internal_error_is_fatal_none() {
    // KVM_EXIT_INTERNAL_ERROR has no architectural reason code —
    // classify as Fatal(None) so the failure-dump emitter can
    // distinguish it from FailEntry and emit the correct prose.
    let com1 = PiMutex::new(console::Serial::new(console::COM1_BASE));
    let com2 = PiMutex::new(console::Serial::new(console::COM2_BASE));
    let mut exit = VcpuExit::InternalError;
    let action = classify_exit(&com1, &com2, None, None, None, None, None, &mut exit);
    assert!(
        matches!(action, Some(ExitAction::Fatal(None))),
        "InternalError must classify as Fatal(None)"
    );
}

/// Helper: erase the inner data of an ExitAction so panic messages
/// don't try to format the reason payload (Fatal carries Option<u64>).
fn action_tag(a: &Option<ExitAction>) -> u8 {
    match a {
        None => 0,
        Some(ExitAction::Continue) => 1,
        Some(ExitAction::Shutdown) => 2,
        Some(ExitAction::Fatal(_)) => 3,
    }
}
