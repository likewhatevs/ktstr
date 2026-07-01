use super::super::super::*;
use super::*;

// ----------------------------------------------------------------
// Hostile-guest control message defenses (handle_control_event).
//
// The kernel's virtio-console driver
// (drivers/char/virtio_console.c `virtcons_probe`) sends
// DEVICE_READY exactly once and PORT_READY exactly once per port.
// A hostile or buggy guest that re-sends either message would
// re-enqueue PORT_ADD / CONSOLE_PORT / PORT_OPEN / PORT_NAME and
// grow `control_out` without bound, exhausting host memory. The
// device gates each repeat behind `device_ready` / `ports[id].readied`
// flags; these tests pin the gates against regressions.
// ----------------------------------------------------------------

/// DEVICE_READY repeats must be ignored — the second message
/// must NOT enqueue a second batch of PORT_ADD frames. Pins the
/// `if self.device_ready` early-return at handle_control_event
/// (the DEVICE_READY arm). Without the gate, a guest spamming
/// DEVICE_READY would grow `control_out` by NUM_PORTS entries
/// per message until the host OOMs.
#[test]
fn handle_device_ready_repeat_ignored() {
    let mut dev = VirtioConsole::new();
    // First DEVICE_READY: enqueues NUM_PORTS PORT_ADD frames.
    dev.handle_control_event(VirtioConsoleControl {
        id: 0,
        event: VIRTIO_CONSOLE_DEVICE_READY,
        value: 1,
    });
    assert!(
        dev.device_ready,
        "device_ready must be set after first message"
    );
    let after_first = dev.control_out.len();
    assert_eq!(
        after_first, NUM_PORTS as usize,
        "first DEVICE_READY must enqueue exactly NUM_PORTS PORT_ADD frames",
    );

    // Second DEVICE_READY: the gate must reject; control_out
    // must NOT grow.
    dev.handle_control_event(VirtioConsoleControl {
        id: 0,
        event: VIRTIO_CONSOLE_DEVICE_READY,
        value: 1,
    });
    assert_eq!(
        dev.control_out.len(),
        after_first,
        "DEVICE_READY repeat must be a no-op — control_out length must \
         remain at NUM_PORTS, otherwise a hostile guest can grow it \
         unboundedly",
    );
    // device_ready stays true — the repeat does not flip it back.
    assert!(
        dev.device_ready,
        "device_ready must remain set after repeat"
    );
}

/// PORT_READY repeat for the same port must be ignored — the
/// second message must NOT re-enqueue
/// CONSOLE_PORT/PORT_NAME/PORT_OPEN. Pins the
/// `if self.ports[id as usize].readied` early-return at
/// handle_control_event (the PORT_READY arm). Mirrors the
/// DEVICE_READY gate but per-port — readying port 0 twice would
/// otherwise enqueue 6 frames (3 per call) instead of 3.
#[test]
fn handle_port_ready_repeat_ignored_port0() {
    let mut dev = VirtioConsole::new();
    // First PORT_READY for port 0: enqueues 3 frames
    // (CONSOLE_PORT, PORT_NAME, PORT_OPEN).
    dev.handle_control_event(VirtioConsoleControl {
        id: 0,
        event: VIRTIO_CONSOLE_PORT_READY,
        value: 1,
    });
    assert!(
        dev.ports[0].readied,
        "ports[0].readied must be set after first PORT_READY"
    );
    let after_first = dev.control_out.len();
    assert_eq!(
        after_first, 3,
        "first PORT_READY for port 0 must enqueue 3 frames \
         (CONSOLE_PORT, PORT_NAME, PORT_OPEN)",
    );

    // Second PORT_READY for port 0: the gate must reject;
    // control_out must NOT grow.
    dev.handle_control_event(VirtioConsoleControl {
        id: 0,
        event: VIRTIO_CONSOLE_PORT_READY,
        value: 1,
    });
    assert_eq!(
        dev.control_out.len(),
        after_first,
        "PORT_READY repeat for the same port must be a no-op — \
         control_out length must remain at 3, otherwise a hostile \
         guest can re-enqueue announce frames unboundedly",
    );
}

/// PORT_READY repeat for port 1 must be ignored — symmetric to
/// the port-0 case but exercises the port-1 branch (PORT_NAME
/// then PORT_OPEN, 2 frames per legitimate message). A regression
/// that scoped the gate per port-0 only would surface here.
#[test]
fn handle_port_ready_repeat_ignored_port1() {
    let mut dev = VirtioConsole::new();
    dev.handle_control_event(VirtioConsoleControl {
        id: 1,
        event: VIRTIO_CONSOLE_PORT_READY,
        value: 1,
    });
    assert!(dev.ports[1].readied);
    let after_first = dev.control_out.len();
    assert_eq!(
        after_first, 2,
        "first PORT_READY for port 1 must enqueue 2 frames (PORT_NAME, PORT_OPEN)",
    );

    dev.handle_control_event(VirtioConsoleControl {
        id: 1,
        event: VIRTIO_CONSOLE_PORT_READY,
        value: 1,
    });
    assert_eq!(
        dev.control_out.len(),
        after_first,
        "PORT_READY repeat for port 1 must be a no-op",
    );
}

/// PORT_READY for port 0 must NOT inhibit a subsequent PORT_READY
/// for port 1 — the gate is per-port, not global. Pins the array
/// indexing in `ports[id as usize].readied`. A regression that
/// used a single global flag would let only one port's announce
/// frames go through.
#[test]
fn handle_port_ready_per_port_not_global() {
    let mut dev = VirtioConsole::new();
    dev.handle_control_event(VirtioConsoleControl {
        id: 0,
        event: VIRTIO_CONSOLE_PORT_READY,
        value: 1,
    });
    let after_port0 = dev.control_out.len();
    assert_eq!(after_port0, 3, "port 0 announce: 3 frames");

    dev.handle_control_event(VirtioConsoleControl {
        id: 1,
        event: VIRTIO_CONSOLE_PORT_READY,
        value: 1,
    });
    // Now both ports have enqueued: 3 (port 0) + 2 (port 1) = 5.
    assert_eq!(
        dev.control_out.len(),
        5,
        "PORT_READY for port 1 after PORT_READY for port 0 must \
         enqueue port 1's announce frames — the gate is per-port",
    );
    assert!(dev.ports[0].readied);
    assert!(dev.ports[1].readied);
}

/// PORT_READY with value=0 must log an error and skip the
/// announce-frame enqueue. value=0 is the kernel's
/// `add_port` failed signal (drivers/char/virtio_console.c
/// `add_port` error path). Pins the early-return without
/// setting `ports[id].readied` so a future legitimate PORT_READY
/// (after recovery) is not blocked by the gate.
#[test]
fn handle_port_ready_value_zero_skipped() {
    let mut dev = VirtioConsole::new();
    dev.handle_control_event(VirtioConsoleControl {
        id: 0,
        event: VIRTIO_CONSOLE_PORT_READY,
        value: 0,
    });
    assert!(
        dev.control_out.is_empty(),
        "PORT_READY value=0 must NOT enqueue announce frames",
    );
    // ports[0].readied must remain false — the early-return for
    // value=0 happens BEFORE the gate flag is set, so a future
    // PORT_READY value=1 can still complete.
    assert!(
        !dev.ports[0].readied,
        "PORT_READY value=0 must NOT set ports[id].readied — the kernel \
         may legitimately retry with value=1 after the host fixes \
         the underlying issue",
    );
}

/// PORT_READY value=0 must NOT block a subsequent PORT_READY
/// value=1 from completing the handshake. Pins the rule that
/// the value=0 early-return precedes the `port_readied` flag
/// flip.
#[test]
fn handle_port_ready_value_zero_then_one_completes() {
    let mut dev = VirtioConsole::new();
    // value=0 first: skipped, no announce frames.
    dev.handle_control_event(VirtioConsoleControl {
        id: 0,
        event: VIRTIO_CONSOLE_PORT_READY,
        value: 0,
    });
    assert!(dev.control_out.is_empty());
    assert!(!dev.ports[0].readied);
    // value=1 next: must fire the announce as normal.
    dev.handle_control_event(VirtioConsoleControl {
        id: 0,
        event: VIRTIO_CONSOLE_PORT_READY,
        value: 1,
    });
    assert!(dev.ports[0].readied);
    assert_eq!(
        dev.control_out.len(),
        3,
        "PORT_READY value=1 after value=0 must enqueue the announce \
         — the value=0 path must not poison the per-port gate",
    );
}

/// PORT_READY for an unknown port id (>= NUM_PORTS) must be
/// ignored. The existing `handle_port_ready_unknown_port_ignored`
/// covers value=1; this pins that the port-id bounds check gates
/// before the repeat-check (the value-check runs first).
/// Verifies ports[id].readied stays all-false and control_out is empty.
#[test]
fn handle_port_ready_unknown_port_state_unchanged() {
    let mut dev = VirtioConsole::new();
    dev.handle_control_event(VirtioConsoleControl {
        id: NUM_PORTS, // first invalid id (NUM_PORTS == 3)
        event: VIRTIO_CONSOLE_PORT_READY,
        value: 1,
    });
    assert!(dev.control_out.is_empty());
    for p in &dev.ports {
        assert!(
            !p.readied,
            "unknown-port PORT_READY must not flip any port readied flag",
        );
    }
}

/// PORT_OPEN for an unknown port id (>= NUM_PORTS) must be
/// ignored. Pins the `if id >= NUM_PORTS` gate at the head of
/// the PORT_OPEN arm. Without the gate, an out-of-bounds
/// `self.ports[id as usize].opened` index would panic with
/// "index out of bounds" — far worse than a tracing warning.
#[test]
fn handle_port_open_unknown_port_ignored() {
    let mut dev = VirtioConsole::new();
    // Try multiple invalid ids — anything >= NUM_PORTS.
    dev.handle_control_event(VirtioConsoleControl {
        id: NUM_PORTS,
        event: VIRTIO_CONSOLE_PORT_OPEN,
        value: 1,
    });
    dev.handle_control_event(VirtioConsoleControl {
        id: 0xFFFF_FFFF,
        event: VIRTIO_CONSOLE_PORT_OPEN,
        value: 1,
    });
    // No panic. Every port's `opened` stays at the initial false.
    for p in &dev.ports {
        assert!(
            !p.opened,
            "unknown-port PORT_OPEN must not flip any port opened \
             flag — the gate prevents out-of-bounds array access on \
             a hostile id",
        );
    }
}

/// Unhandled c_ovq event values must be absorbed silently — the
/// `other` arm in handle_control_event logs a debug line and
/// returns. The kernel today only sends DEVICE_READY, PORT_READY,
/// and PORT_OPEN on c_ovq (drivers/char/virtio_console.c
/// `send_control_msg` callers); a future kernel may add new
/// event types. Pins that an unrecognised event does not panic,
/// does not enqueue control_out, and does not flip any FSM flag.
#[test]
fn handle_unhandled_event_absorbed() {
    let mut dev = VirtioConsole::new();
    // Pick events that are valid wire values but the device
    // does not act on. PORT_ADD is the kernel's PORT_ADD —
    // the host emits it (PORT_ADD on c_ivq), the guest does not
    // send it back on c_ovq, so seeing it here is "the guest
    // sent something we don't handle." PORT_REMOVE / RESIZE
    // / PORT_NAME / CONSOLE_PORT round out the wire-defined
    // events that hit the `other` arm.
    let unhandled = [
        VIRTIO_CONSOLE_PORT_ADD,
        VIRTIO_CONSOLE_PORT_REMOVE,
        VIRTIO_CONSOLE_CONSOLE_PORT,
        VIRTIO_CONSOLE_RESIZE,
        VIRTIO_CONSOLE_PORT_NAME,
        // Completely synthetic event id — pins that even values
        // beyond the wire-defined set fall through cleanly.
        0xBEEF,
    ];
    for ev in unhandled {
        dev.handle_control_event(VirtioConsoleControl {
            id: 0,
            event: ev,
            value: 1,
        });
    }
    // Nothing must have been enqueued, no FSM flag must have
    // been flipped.
    assert!(
        dev.control_out.is_empty(),
        "unhandled events must NOT enqueue control_out",
    );
    assert!(
        !dev.device_ready,
        "unhandled events must NOT flip device_ready",
    );
    for p in &dev.ports {
        assert!(
            !p.opened,
            "unhandled events must NOT flip any port opened flag"
        );
        assert!(
            !p.readied,
            "unhandled events must NOT flip any port readied flag"
        );
    }
}
