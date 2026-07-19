//! Host-authoritative scheduler attach-attempt watchdog.
//!
//! The guest reports generation-tagged begin/end boundaries, but does not
//! decide the service budget. The host reuses the monitor's existing
//! max-single-vCPU CPU currency, which is measured outside the VM and is
//! therefore invariant to KVM steal/descheduling on both x86_64 and aarch64.
//! This overlay is independent of the forward-only coarse lifecycle stage so
//! Attach/Restart/Replace Ops can be charged while that stage remains `Body`.

use std::sync::Arc;

use crate::monitor::{CPU_CURRENCY_NONE, LedgerSnapshot, ProgressLedger};
use crate::vmm::pi_mutex::PiMutex;
use crate::vmm::wire::{
    AttachAttemptEvent, AttachAttemptKind, AttachAttemptTransition, AttachCancelCause,
};

use super::watchdog_step;

type AttachControlConsole =
    Arc<crate::vmm::pi_mutex::PiMutex<crate::vmm::virtio_console::VirtioConsole>>;

/// Queue one boundary ACK while the tracker mutex is still held.
///
/// Every host path follows the fixed tracker→virtio-console lock order. This
/// makes state promotion and packet order atomic relative to watchdog
/// cancellation.
///
/// This nested lock cannot inherit guest backpressure: `queue_input` appends
/// the fixed 17-byte control packet to a host-owned deque and makes at most
/// one bounded `drain_pending_rx(0)` pass. That pass has an explicit
/// `RX_CHAINS_PER_CALL_MAX` cap (including zero-progress descriptor chains),
/// and every other `VirtioConsole` critical section is likewise a bounded
/// device operation: no condition wait, blocking read/write, thread join, or
/// guest-response wait is performed while its `PiMutex` is held. The mutex
/// uses priority inheritance, so retaining tracker→console ordering gives
/// deterministic packet order without letting a blocked guest reader pin the
/// watchdog behind guest I/O.
fn acknowledge_boundary_locked(console: &AttachControlConsole, event: AttachAttemptEvent) {
    match event.transition {
        AttachAttemptTransition::Started => {
            crate::vmm::host_comms::acknowledge_attach_started(console, event.generation);
        }
        AttachAttemptTransition::Finished => {
            crate::vmm::host_comms::acknowledge_attach_finished(console, event.generation);
        }
        AttachAttemptTransition::Settled => {}
    }
}

/// Raw PMU-currency allowance. Pthread currency is widened by the same 3/2
/// factor as ordinary phase Tier-1 accounting.
const ATTACH_SERVICE_BUDGET_NS: u64 = 35_000_000_000;

/// Additional delivered max-vCPU service allowed after cancellation before
/// the host fails closed. This is service, not wall time: a massively
/// oversubscribed cell gets the same opportunity to consume the cancellation
/// as an otherwise idle cell.
const ATTACH_CANCEL_SERVICE_GRACE_NS: u64 = 5_000_000_000;

/// Delivered max-vCPU service allowed for the FinishedAck/Settled
/// rendezvous. `Finished` starts a fresh accounting epoch, so wall time while
/// the VM is host-starved is free, but a guest which continues to receive
/// service without consuming the ACK and publishing `Settled` cannot suppress
/// the ordinary lifecycle watchdog forever.
const ATTACH_FINISH_SERVICE_GRACE_NS: u64 = 5_000_000_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum AttachWatchdogDecision {
    None,
    RequestCancel {
        generation: u64,
        kind: AttachAttemptKind,
        cause: AttachCancelCause,
        service_ns: u64,
        trigger_elapsed_ns: u64,
        trigger_budget_ns: u64,
    },
    FailClosed {
        generation: u64,
        kind: AttachAttemptKind,
        cause: AttachCancelCause,
        service_after_cancel_ns: u64,
        grace_budget_ns: u64,
    },
    /// `Finished` was accepted and acknowledged, but the guest consumed more
    /// than the bounded delivered-service grace without publishing the FIFO
    /// `Settled` boundary which proves it consumed that ACK.
    FinishUnsettled {
        generation: u64,
        kind: AttachAttemptKind,
        service_after_finished_ns: u64,
        grace_budget_ns: u64,
    },
    /// The host monitor reached a known terminal state while the attach
    /// overlay still owned the lifecycle watchdog. Unlike a stale heartbeat,
    /// this is an explicit terminal edge, so waiting for service accounting
    /// or a cancellation grace cannot recover the attempt.
    SensorTerminal {
        generation: u64,
        kind: AttachAttemptKind,
        finishing: bool,
    },
}

/// One watchdog-tick result. `active` lets the caller suppress every ordinary
/// lifecycle watchdog path while this more precise overlay owns the attach
/// attempt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct AttachWatchdogStep {
    pub active: bool,
    pub decision: AttachWatchdogDecision,
}

/// Coherent host watchdog view for one tick.
///
/// The attempt state and ledger snapshot are read under the tracker mutex, so
/// a Started/Finished boundary cannot be paired with the CPU epoch from the
/// opposite side of that boundary.
#[derive(Debug, Clone, Copy)]
pub(super) struct AttachWatchdogTick {
    pub snapshot: LedgerSnapshot,
    pub monitor_live: bool,
    pub attach: AttachWatchdogStep,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum AttachEventDisposition {
    Started,
    Finished,
    Settled,
    Duplicate,
    Stale,
    Conflict,
    UnexpectedFinish,
    UnexpectedSettle,
}

#[derive(Debug, Clone, Copy)]
struct CancelIssued {
    cause: AttachCancelCause,
}

#[derive(Debug, Clone, Copy)]
struct ActiveAttempt {
    generation: u64,
    kind: AttachAttemptKind,
    accounting_epoch: u32,
    cancel: Option<CancelIssued>,
}

#[derive(Debug, Clone, Copy)]
struct FinishingAttempt {
    generation: u64,
    kind: AttachAttemptKind,
    accounting_epoch: u32,
}

#[derive(Debug, Default)]
struct AttachAttemptState {
    last_generation: u64,
    last_settled: Option<(u64, AttachAttemptKind)>,
    active: Option<ActiveAttempt>,
    finishing: Option<FinishingAttempt>,
}

/// Single-active-attempt state shared by bulk dispatch and the watchdog.
pub(super) struct AttachAttemptTracker {
    state: PiMutex<AttachAttemptState>,
}

impl Default for AttachAttemptTracker {
    fn default() -> Self {
        Self {
            state: PiMutex::new(AttachAttemptState::default()),
        }
    }
}

impl std::fmt::Debug for AttachAttemptTracker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AttachAttemptTracker")
            .finish_non_exhaustive()
    }
}

impl AttachAttemptTracker {
    fn observe_event_with_ack(
        &self,
        event: AttachAttemptEvent,
        ledger: &ProgressLedger,
        wall_ns: u64,
        mut acknowledge: impl FnMut(AttachAttemptEvent),
    ) -> AttachEventDisposition {
        let mut state = self.state.lock();
        match event.transition {
            AttachAttemptTransition::Started => {
                if let Some(active) = state.active {
                    if active.generation == event.generation && active.kind == event.kind {
                        acknowledge(event);
                        return AttachEventDisposition::Duplicate;
                    }
                    return AttachEventDisposition::Conflict;
                }
                if state.finishing.is_some() {
                    return AttachEventDisposition::Conflict;
                }
                if event.generation <= state.last_generation {
                    return AttachEventDisposition::Stale;
                }

                // Re-anchor before publishing `active` while holding the mutex.
                // The watchdog cannot observe the new attempt paired with the
                // preceding lifecycle epoch.
                let accounting_epoch = ledger.reanchor_phase_cpu(wall_ns);
                state.last_generation = event.generation;
                state.active = Some(ActiveAttempt {
                    generation: event.generation,
                    kind: event.kind,
                    accounting_epoch,
                    cancel: None,
                });
                acknowledge(event);
                AttachEventDisposition::Started
            }
            AttachAttemptTransition::Finished => {
                if let Some(finishing) = state.finishing {
                    if finishing.generation == event.generation && finishing.kind == event.kind {
                        acknowledge(event);
                        return AttachEventDisposition::Duplicate;
                    }
                    return if event.generation < finishing.generation {
                        AttachEventDisposition::Stale
                    } else {
                        AttachEventDisposition::Conflict
                    };
                }
                let Some(active) = state.active else {
                    if state.last_settled == Some((event.generation, event.kind)) {
                        acknowledge(event);
                        return AttachEventDisposition::Duplicate;
                    }
                    return if event.generation <= state.last_generation {
                        AttachEventDisposition::Stale
                    } else {
                        AttachEventDisposition::UnexpectedFinish
                    };
                };
                if active.generation != event.generation || active.kind != event.kind {
                    return if event.generation < active.generation {
                        AttachEventDisposition::Stale
                    } else {
                        AttachEventDisposition::Conflict
                    };
                }

                // Queue FinishedAck first while the tracker still serializes
                // every boundary and watchdog cancellation. Only after that
                // enqueue completes do we establish the finishing epoch:
                // console-mutex contention belongs to the active attempt, not
                // to the guest's delivered-service allowance for consuming an
                // ACK which did not yet exist.
                acknowledge(event);
                let accounting_epoch = ledger.reanchor_phase_cpu(wall_ns);
                state.active = None;
                state.finishing = Some(FinishingAttempt {
                    generation: event.generation,
                    kind: event.kind,
                    accounting_epoch,
                });
                AttachEventDisposition::Finished
            }
            AttachAttemptTransition::Settled => {
                if let Some(finishing) = state.finishing {
                    if finishing.generation != event.generation || finishing.kind != event.kind {
                        return if event.generation < finishing.generation {
                            AttachEventDisposition::Stale
                        } else {
                            AttachEventDisposition::Conflict
                        };
                    }

                    // The guest consumed FinishedAck before publishing this
                    // FIFO boundary. Only now can coarse lifecycle accounting
                    // resume without racing a starved control reader.
                    ledger.reanchor_phase_cpu(wall_ns);
                    state.finishing = None;
                    state.last_settled = Some((event.generation, event.kind));
                    AttachEventDisposition::Settled
                } else if state.last_settled == Some((event.generation, event.kind)) {
                    AttachEventDisposition::Duplicate
                } else if state.active.is_some() {
                    AttachEventDisposition::UnexpectedSettle
                } else if event.generation <= state.last_generation {
                    AttachEventDisposition::Stale
                } else {
                    AttachEventDisposition::UnexpectedSettle
                }
            }
        }
    }

    /// Apply one CRC- and shape-validated guest boundary.
    ///
    /// Accepted starts and matching settles each re-anchor the monitor's
    /// max-vCPU tracker without changing the coarse lifecycle stage. Finished
    /// moves the overlay into a fresh, service-accounted rendezvous until the
    /// matching Settled proves the guest consumed its ACK. Duplicate, stale,
    /// conflicting, and unmatched events never move the anchor, so a buggy
    /// guest cannot evade the service budget by replaying `Started`.
    pub(super) fn observe_event(
        &self,
        event: AttachAttemptEvent,
        ledger: &ProgressLedger,
        wall_ns: u64,
        control_console: &AttachControlConsole,
    ) -> AttachEventDisposition {
        self.observe_event_with_ack(event, ledger, wall_ns, |event| {
            acknowledge_boundary_locked(control_console, event);
        })
    }

    /// Take one state/ledger-coherent host watchdog tick and atomically latch
    /// the first cancellation request.
    pub(super) fn watchdog_tick(
        &self,
        ledger: &ProgressLedger,
        now_wall_ns: u64,
        monitor_liveness: &mut watchdog_step::MonitorLiveness,
        control_console: &AttachControlConsole,
    ) -> AttachWatchdogTick {
        let mut state = self.state.lock();
        let snapshot = ledger.snapshot();
        let monitor_live = monitor_liveness.observe(snapshot.monitor_heartbeat);
        if snapshot.monitor_terminal {
            let attempt = state
                .active
                .map(|active| {
                    (
                        active.generation,
                        active.kind,
                        false, // still charging attach service
                    )
                })
                .or_else(|| {
                    state.finishing.map(|finishing| {
                        (
                            finishing.generation,
                            finishing.kind,
                            true, // waiting for FinishedAck/Settled
                        )
                    })
                });
            if let Some((generation, kind, finishing)) = attempt {
                return AttachWatchdogTick {
                    snapshot,
                    monitor_live,
                    attach: AttachWatchdogStep {
                        active: true,
                        decision: AttachWatchdogDecision::SensorTerminal {
                            generation,
                            kind,
                            finishing,
                        },
                    },
                };
            }
        }
        if let Some(mut finishing) = state.finishing {
            // A lifecycle transition racing Finished can publish a different
            // monitor epoch. As for an active attach/cancellation grace, do
            // not charge a sample from the wrong anchor and do not return
            // None forever: rebase once to the observed generation, then
            // charge subsequent delivered service directly.
            if snapshot.phase_epoch != finishing.accounting_epoch {
                finishing.accounting_epoch = ledger.reanchor_phase_cpu(now_wall_ns);
                state.finishing = Some(finishing);
                return AttachWatchdogTick {
                    snapshot: ledger.snapshot(),
                    monitor_live,
                    attach: AttachWatchdogStep {
                        active: true,
                        decision: AttachWatchdogDecision::None,
                    },
                };
            }

            let service_grace = watchdog_step::widen_budget_for_currency(
                ATTACH_FINISH_SERVICE_GRACE_NS,
                snapshot.cpu_currency,
            );
            let service_grace_exhausted = snapshot.cpu_currency != CPU_CURRENCY_NONE
                && snapshot.max_vcpu_cpu_in_phase_ns > service_grace;
            return AttachWatchdogTick {
                snapshot,
                monitor_live,
                attach: AttachWatchdogStep {
                    active: true,
                    decision: if service_grace_exhausted {
                        AttachWatchdogDecision::FinishUnsettled {
                            generation: finishing.generation,
                            kind: finishing.kind,
                            service_after_finished_ns: snapshot.max_vcpu_cpu_in_phase_ns,
                            grace_budget_ns: service_grace,
                        }
                    } else {
                        AttachWatchdogDecision::None
                    },
                },
            };
        }

        let Some(mut active) = state.active else {
            return AttachWatchdogTick {
                snapshot,
                monitor_live,
                attach: AttachWatchdogStep {
                    active: false,
                    decision: AttachWatchdogDecision::None,
                },
            };
        };

        // A monitor publication for any other epoch cannot be charged to this
        // attempt. Rebase once under the tracker lock instead of returning
        // None forever: subsequent service is measured directly against the
        // conservative fresh anchor and can still cancel/fail closed.
        if snapshot.phase_epoch != active.accounting_epoch {
            active.accounting_epoch = ledger.reanchor_phase_cpu(now_wall_ns);
            state.active = Some(active);
            return AttachWatchdogTick {
                snapshot: ledger.snapshot(),
                monitor_live,
                attach: AttachWatchdogStep {
                    active: true,
                    decision: AttachWatchdogDecision::None,
                },
            };
        }

        let service_ns = snapshot.max_vcpu_cpu_in_phase_ns;
        if let Some(cancel) = active.cancel {
            let service_grace = watchdog_step::widen_budget_for_currency(
                ATTACH_CANCEL_SERVICE_GRACE_NS,
                snapshot.cpu_currency,
            );
            let service_grace_exhausted =
                snapshot.cpu_currency != CPU_CURRENCY_NONE && service_ns > service_grace;
            let decision = if service_grace_exhausted {
                AttachWatchdogDecision::FailClosed {
                    generation: active.generation,
                    kind: active.kind,
                    cause: cancel.cause,
                    service_after_cancel_ns: service_ns,
                    grace_budget_ns: service_grace,
                }
            } else {
                AttachWatchdogDecision::None
            };
            return AttachWatchdogTick {
                snapshot,
                monitor_live,
                attach: AttachWatchdogStep {
                    active: true,
                    decision,
                },
            };
        }

        let service_budget = watchdog_step::widen_budget_for_currency(
            ATTACH_SERVICE_BUDGET_NS,
            snapshot.cpu_currency,
        );
        if snapshot.cpu_currency == CPU_CURRENCY_NONE || service_ns <= service_budget {
            return AttachWatchdogTick {
                snapshot,
                monitor_live,
                attach: AttachWatchdogStep {
                    active: true,
                    decision: AttachWatchdogDecision::None,
                },
            };
        }
        let cause = AttachCancelCause::ServiceBudget;
        let trigger_elapsed_ns = service_ns;
        let trigger_budget_ns = service_budget;

        // Cancellation receives a fresh max-per-vCPU epoch. Directly charging
        // that epoch is essential: `max_i(C_i(now)-A_i) -
        // max_i(C_i(cancel)-A_i)` is not `max_i(C_i(now)-C_i(cancel))` when
        // the busiest vCPU changes. Re-anchoring also tags the grace so a
        // pre-cancel watchdog snapshot can never exhaust it.
        crate::vmm::host_comms::request_attach_cancel(control_console, active.generation, cause);
        active.accounting_epoch = ledger.reanchor_phase_cpu(now_wall_ns);
        active.cancel = Some(CancelIssued { cause });
        state.active = Some(active);
        // The boundary above changed both phase_epoch and the milestone wall
        // anchor. Return that post-cancel generation to the caller so Tier-3
        // cannot consume the pre-cancel max/wall values in this same tick and
        // bypass the promised cancellation grace.
        let post_cancel_snapshot = ledger.snapshot();
        AttachWatchdogTick {
            snapshot: post_cancel_snapshot,
            monitor_live,
            attach: AttachWatchdogStep {
                active: true,
                decision: AttachWatchdogDecision::RequestCancel {
                    generation: active.generation,
                    kind: active.kind,
                    cause,
                    service_ns,
                    trigger_elapsed_ns,
                    trigger_budget_ns,
                },
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::monitor::{
        CPU_CURRENCY_NONE, CPU_CURRENCY_PMU, CPU_CURRENCY_PTHREAD, LifecycleStage,
    };
    use std::sync::atomic::Ordering;

    fn event(
        transition: AttachAttemptTransition,
        kind: AttachAttemptKind,
        generation: u64,
    ) -> AttachAttemptEvent {
        AttachAttemptEvent {
            transition,
            kind,
            generation,
        }
    }

    fn publish(
        ledger: &ProgressLedger,
        service_ns: u64,
        currency: u8,
        demand: bool,
        channels: bool,
    ) {
        let epoch = ledger.phase_epoch.load(Ordering::Acquire);
        ledger.record_liveness(
            service_ns, service_ns, epoch, false, 0, currency, demand, channels,
        );
    }

    fn tick(
        tracker: &AttachAttemptTracker,
        ledger: &ProgressLedger,
        now_wall_ns: u64,
        liveness: &mut watchdog_step::MonitorLiveness,
    ) -> AttachWatchdogTick {
        tracker.watchdog_tick(ledger, now_wall_ns, liveness, &test_control_console())
    }

    fn test_control_console() -> AttachControlConsole {
        Arc::new(crate::vmm::pi_mutex::PiMutex::new(
            crate::vmm::virtio_console::VirtioConsole::new(),
        ))
    }

    fn observe(
        tracker: &AttachAttemptTracker,
        event: AttachAttemptEvent,
        ledger: &ProgressLedger,
        wall_ns: u64,
    ) -> AttachEventDisposition {
        tracker.observe_event(event, ledger, wall_ns, &test_control_console())
    }

    #[test]
    fn boundaries_reanchor_cpu_without_regressing_coarse_phase() {
        let tracker = AttachAttemptTracker::default();
        let ledger = ProgressLedger::default();
        ledger
            .phase
            .store(LifecycleStage::Body as u8, Ordering::Relaxed);

        assert_eq!(
            observe(
                &tracker,
                event(
                    AttachAttemptTransition::Started,
                    AttachAttemptKind::Replace,
                    7,
                ),
                &ledger,
                100,
            ),
            AttachEventDisposition::Started
        );
        assert_eq!(
            ledger.phase.load(Ordering::Relaxed),
            LifecycleStage::Body as u8,
        );
        assert_eq!(ledger.phase_epoch.load(Ordering::Acquire), 1);

        // A duplicate cannot reset the service anchor.
        assert_eq!(
            observe(
                &tracker,
                event(
                    AttachAttemptTransition::Started,
                    AttachAttemptKind::Replace,
                    7,
                ),
                &ledger,
                200,
            ),
            AttachEventDisposition::Duplicate
        );
        assert_eq!(ledger.phase_epoch.load(Ordering::Acquire), 1);

        assert_eq!(
            observe(
                &tracker,
                event(
                    AttachAttemptTransition::Finished,
                    AttachAttemptKind::Replace,
                    7,
                ),
                &ledger,
                300,
            ),
            AttachEventDisposition::Finished
        );
        assert_eq!(
            ledger.phase.load(Ordering::Relaxed),
            LifecycleStage::Body as u8,
        );
        assert_eq!(
            ledger.phase_epoch.load(Ordering::Acquire),
            2,
            "Finished must reanchor the service-accounted ACK rendezvous",
        );

        // A retried terminal boundary is accepted idempotently so the host
        // can re-ACK it, but it must not move either accounting anchor.
        assert_eq!(
            observe(
                &tracker,
                event(
                    AttachAttemptTransition::Finished,
                    AttachAttemptKind::Replace,
                    7,
                ),
                &ledger,
                400,
            ),
            AttachEventDisposition::Duplicate
        );
        assert_eq!(ledger.phase_epoch.load(Ordering::Acquire), 2);

        assert_eq!(
            observe(
                &tracker,
                event(
                    AttachAttemptTransition::Settled,
                    AttachAttemptKind::Replace,
                    7,
                ),
                &ledger,
                500,
            ),
            AttachEventDisposition::Settled
        );
        assert_eq!(ledger.phase_epoch.load(Ordering::Acquire), 3);

        // Settled is FIFO terminal state, so retrying it is a no-op.
        assert_eq!(
            observe(
                &tracker,
                event(
                    AttachAttemptTransition::Settled,
                    AttachAttemptKind::Replace,
                    7,
                ),
                &ledger,
                600,
            ),
            AttachEventDisposition::Duplicate
        );
        assert_eq!(ledger.phase_epoch.load(Ordering::Acquire), 3);
    }

    #[test]
    fn started_and_finished_same_drain_leave_fresh_overlay_until_settled() {
        let tracker = AttachAttemptTracker::default();
        let ledger = ProgressLedger::default();
        let console = test_control_console();
        let mut liveness = watchdog_step::MonitorLiveness::new();
        let started = event(AttachAttemptTransition::Started, AttachAttemptKind::Boot, 9);
        let finished = event(
            AttachAttemptTransition::Finished,
            AttachAttemptKind::Boot,
            9,
        );

        // The console deliberately has no guest memory, ready queue, or
        // reader. Accepted boundaries must still return after appending to
        // the host-owned pending deque; queue_input must never wait for guest
        // RX capacity while tracker→console serialization is held.
        assert_eq!(
            tracker.observe_event(started, &ledger, 100, &console),
            AttachEventDisposition::Started
        );
        assert_eq!(
            tracker.observe_event(finished, &ledger, 100, &console),
            AttachEventDisposition::Finished
        );
        let finishing = tracker.watchdog_tick(&ledger, u64::MAX, &mut liveness, &console);
        assert!(finishing.attach.active);
        assert_eq!(finishing.attach.decision, AttachWatchdogDecision::None);

        let mut expected = crate::vmm::wire::encode_attach_started_ack(9).to_vec();
        expected.extend_from_slice(&crate::vmm::wire::encode_attach_finished_ack(9));
        assert_eq!(console.lock().pending_rx_bytes_for_test(), expected);
    }

    #[test]
    fn finished_grace_starts_only_after_delayed_console_enqueue() {
        let tracker = Arc::new(AttachAttemptTracker::default());
        let ledger = Arc::new(ProgressLedger::default());
        let console = test_control_console();
        let mut liveness = watchdog_step::MonitorLiveness::new();
        assert_eq!(
            tracker.observe_event(
                event(
                    AttachAttemptTransition::Started,
                    AttachAttemptKind::Replace,
                    12,
                ),
                &ledger,
                0,
                &console,
            ),
            AttachEventDisposition::Started,
        );
        assert_eq!(ledger.phase_epoch.load(Ordering::Acquire), 1);

        // Hold the actual console mutex, then use the testable serialized
        // enqueue seam to signal precisely after the finisher acquired the
        // tracker and immediately before it tries the production ACK path.
        // This makes the console delay deterministic without sleeps.
        let console_guard = console.lock();
        let (enqueue_entered_tx, enqueue_entered_rx) = std::sync::mpsc::channel();
        let tracker_for_finish = Arc::clone(&tracker);
        let ledger_for_finish = Arc::clone(&ledger);
        let console_for_finish = Arc::clone(&console);
        let finisher = std::thread::spawn(move || {
            tracker_for_finish.observe_event_with_ack(
                event(
                    AttachAttemptTransition::Finished,
                    AttachAttemptKind::Replace,
                    12,
                ),
                &ledger_for_finish,
                1,
                |event| {
                    enqueue_entered_tx.send(()).expect("signal enqueue entry");
                    acknowledge_boundary_locked(&console_for_finish, event);
                },
            )
        });
        enqueue_entered_rx
            .recv_timeout(std::time::Duration::from_secs(2))
            .expect("finisher reached delayed console enqueue");

        // Service published while FinishedAck is still blocked belongs to the
        // active attempt. The finishing epoch must not exist yet.
        publish(
            &ledger,
            ATTACH_FINISH_SERVICE_GRACE_NS + 1,
            CPU_CURRENCY_PMU,
            true,
            true,
        );
        assert_eq!(
            ledger.phase_epoch.load(Ordering::Acquire),
            1,
            "console delay must precede the finishing reanchor",
        );

        drop(console_guard);
        assert_eq!(
            finisher.join().expect("finisher thread"),
            AttachEventDisposition::Finished,
        );
        assert_eq!(ledger.phase_epoch.load(Ordering::Acquire), 2);
        let just_acked = tick(&tracker, &ledger, 2, &mut liveness);
        assert_eq!(
            just_acked.snapshot.max_vcpu_cpu_in_phase_ns, 0,
            "pre-ACK service must be discarded by the post-enqueue reanchor",
        );
        assert_eq!(just_acked.attach.decision, AttachWatchdogDecision::None,);

        let mut expected = crate::vmm::wire::encode_attach_started_ack(12).to_vec();
        expected.extend_from_slice(&crate::vmm::wire::encode_attach_finished_ack(12));
        assert_eq!(console.lock().pending_rx_bytes_for_test(), expected);
    }

    #[test]
    fn cancel_and_boundary_acks_follow_tracker_serialization_order() {
        let tracker = AttachAttemptTracker::default();
        let ledger = ProgressLedger::default();
        let console = test_control_console();
        let mut liveness = watchdog_step::MonitorLiveness::new();
        tracker.observe_event(
            event(
                AttachAttemptTransition::Started,
                AttachAttemptKind::Restart,
                10,
            ),
            &ledger,
            0,
            &console,
        );
        publish(
            &ledger,
            ATTACH_SERVICE_BUDGET_NS + 1,
            CPU_CURRENCY_PMU,
            true,
            true,
        );
        assert!(matches!(
            tracker
                .watchdog_tick(&ledger, 1, &mut liveness, &console)
                .attach
                .decision,
            AttachWatchdogDecision::RequestCancel { generation: 10, .. }
        ));
        assert_eq!(
            tracker.observe_event(
                event(
                    AttachAttemptTransition::Finished,
                    AttachAttemptKind::Restart,
                    10,
                ),
                &ledger,
                2,
                &console,
            ),
            AttachEventDisposition::Finished
        );

        let mut expected = crate::vmm::wire::encode_attach_started_ack(10).to_vec();
        expected.extend_from_slice(&crate::vmm::wire::encode_attach_cancel(
            10,
            AttachCancelCause::ServiceBudget,
        ));
        expected.extend_from_slice(&crate::vmm::wire::encode_attach_finished_ack(10));
        assert_eq!(console.lock().pending_rx_bytes_for_test(), expected);
    }

    #[test]
    fn service_budget_requests_one_generation_tagged_cancel() {
        let tracker = AttachAttemptTracker::default();
        let ledger = ProgressLedger::default();
        let mut liveness = watchdog_step::MonitorLiveness::new();
        observe(
            &tracker,
            event(AttachAttemptTransition::Started, AttachAttemptKind::Boot, 1),
            &ledger,
            0,
        );

        publish(
            &ledger,
            ATTACH_SERVICE_BUDGET_NS,
            CPU_CURRENCY_PMU,
            true,
            true,
        );
        let at_budget = tick(&tracker, &ledger, 1, &mut liveness);
        assert_eq!(at_budget.attach.decision, AttachWatchdogDecision::None);

        publish(
            &ledger,
            ATTACH_SERVICE_BUDGET_NS + 1,
            CPU_CURRENCY_PMU,
            true,
            true,
        );
        let over = tick(&tracker, &ledger, 2, &mut liveness);
        assert!(matches!(
            over.attach.decision,
            AttachWatchdogDecision::RequestCancel {
                generation: 1,
                kind: AttachAttemptKind::Boot,
                cause: AttachCancelCause::ServiceBudget,
                ..
            }
        ));
        assert_eq!(over.snapshot.max_vcpu_cpu_in_phase_ns, 0);
        assert_eq!(over.snapshot.wall_ns_at_progress, 2);
        // Cancellation starts a distinct max-vCPU generation. A later sample
        // is direct service since cancellation, not subtraction of two max
        // scalars whose busiest vCPU may differ.
        publish(&ledger, 1, CPU_CURRENCY_PMU, true, true);
        assert_eq!(
            tick(&tracker, &ledger, 3, &mut liveness).attach.decision,
            AttachWatchdogDecision::None,
            "cancellation must be edge-triggered once per generation",
        );
    }

    #[test]
    fn finishing_suppresses_coarse_watchdog_but_charges_its_fresh_service_epoch() {
        let tracker = AttachAttemptTracker::default();
        let ledger = ProgressLedger::default();
        let mut liveness = watchdog_step::MonitorLiveness::new();
        observe(
            &tracker,
            event(
                AttachAttemptTransition::Started,
                AttachAttemptKind::Attach,
                2,
            ),
            &ledger,
            0,
        );
        publish(
            &ledger,
            ATTACH_SERVICE_BUDGET_NS + 1,
            CPU_CURRENCY_PMU,
            false,
            true,
        );
        assert_eq!(
            observe(
                &tracker,
                event(
                    AttachAttemptTransition::Finished,
                    AttachAttemptKind::Attach,
                    2,
                ),
                &ledger,
                1,
            ),
            AttachEventDisposition::Finished,
        );
        let after = tick(&tracker, &ledger, 2, &mut liveness);
        assert!(
            after.attach.active,
            "Finishing must retain the overlay until FinishedAck is consumed"
        );
        assert_eq!(after.attach.decision, AttachWatchdogDecision::None);

        // Arbitrary host wall time, no runnable demand, dead channels, and no
        // delivered service cannot exhaust an ACK rendezvous.
        publish(&ledger, 0, CPU_CURRENCY_NONE, false, false);
        let starved = tick(&tracker, &ledger, u64::MAX, &mut liveness);
        assert!(starved.attach.active);
        assert_eq!(starved.attach.decision, AttachWatchdogDecision::None);

        publish(
            &ledger,
            ATTACH_FINISH_SERVICE_GRACE_NS,
            CPU_CURRENCY_PMU,
            true,
            true,
        );
        assert_eq!(
            tick(&tracker, &ledger, u64::MAX, &mut liveness)
                .attach
                .decision,
            AttachWatchdogDecision::None,
            "the exact delivered-service budget remains permitted",
        );
        publish(
            &ledger,
            ATTACH_FINISH_SERVICE_GRACE_NS + 1,
            CPU_CURRENCY_PMU,
            true,
            true,
        );
        assert_eq!(
            tick(&tracker, &ledger, u64::MAX, &mut liveness)
                .attach
                .decision,
            AttachWatchdogDecision::FinishUnsettled {
                generation: 2,
                kind: AttachAttemptKind::Attach,
                service_after_finished_ns: ATTACH_FINISH_SERVICE_GRACE_NS + 1,
                grace_budget_ns: ATTACH_FINISH_SERVICE_GRACE_NS,
            },
        );

        assert_eq!(
            observe(
                &tracker,
                event(
                    AttachAttemptTransition::Settled,
                    AttachAttemptKind::Attach,
                    2,
                ),
                &ledger,
                u64::MAX,
            ),
            AttachEventDisposition::Settled,
        );
        let settled = tick(&tracker, &ledger, u64::MAX, &mut liveness);
        assert!(!settled.attach.active);
        assert_eq!(settled.snapshot.max_vcpu_cpu_in_phase_ns, 0);
    }

    #[test]
    fn finishing_epoch_mismatch_rebases_once_then_fails_on_fresh_service() {
        let tracker = AttachAttemptTracker::default();
        let ledger = ProgressLedger::default();
        let mut liveness = watchdog_step::MonitorLiveness::new();
        for transition in [
            AttachAttemptTransition::Started,
            AttachAttemptTransition::Finished,
        ] {
            assert!(matches!(
                observe(
                    &tracker,
                    event(transition, AttachAttemptKind::Replace, 22),
                    &ledger,
                    0,
                ),
                AttachEventDisposition::Started | AttachEventDisposition::Finished,
            ));
        }

        // Model an unrelated lifecycle epoch advance which raced the
        // rendezvous, followed by a large publication for that foreign
        // anchor. The first tick must discard it and establish a fresh epoch.
        ledger.reanchor_phase_cpu(1);
        publish(
            &ledger,
            ATTACH_FINISH_SERVICE_GRACE_NS + 1,
            CPU_CURRENCY_PMU,
            true,
            true,
        );
        let rebased = tick(&tracker, &ledger, 2, &mut liveness);
        assert_eq!(rebased.attach.decision, AttachWatchdogDecision::None);
        assert_eq!(
            rebased.snapshot.max_vcpu_cpu_in_phase_ns, 0,
            "a foreign epoch's service must not be charged to finishing",
        );

        publish(
            &ledger,
            ATTACH_FINISH_SERVICE_GRACE_NS + 1,
            CPU_CURRENCY_PMU,
            true,
            true,
        );
        assert!(matches!(
            tick(&tracker, &ledger, 3, &mut liveness).attach.decision,
            AttachWatchdogDecision::FinishUnsettled {
                generation: 22,
                kind: AttachAttemptKind::Replace,
                ..
            },
        ));
    }

    #[test]
    fn stale_monitor_heartbeat_defers_but_explicit_terminal_fails_active_attempt() {
        let tracker = AttachAttemptTracker::default();
        let ledger = ProgressLedger::default();
        let mut liveness = watchdog_step::MonitorLiveness::new();
        observe(
            &tracker,
            event(
                AttachAttemptTransition::Started,
                AttachAttemptKind::Restart,
                30,
            ),
            &ledger,
            0,
        );

        let mut stale = tick(&tracker, &ledger, 1, &mut liveness);
        for now in 2..=u64::from(watchdog_step::WATCHDOG_MONITOR_LIVENESS_MISS_TICKS) {
            stale = tick(&tracker, &ledger, now, &mut liveness);
        }
        assert!(!stale.monitor_live);
        assert!(
            stale.attach.active,
            "a stale heartbeat must leave the service-accounted overlay installed",
        );
        assert_eq!(stale.attach.decision, AttachWatchdogDecision::None);

        ledger.publish_monitor_terminal();
        let terminal = tick(&tracker, &ledger, u64::MAX, &mut liveness);
        assert_eq!(
            terminal.attach.decision,
            AttachWatchdogDecision::SensorTerminal {
                generation: 30,
                kind: AttachAttemptKind::Restart,
                finishing: false,
            },
        );
    }

    #[test]
    fn explicit_monitor_terminal_fails_finishing_rendezvous_immediately() {
        let tracker = AttachAttemptTracker::default();
        let ledger = ProgressLedger::default();
        let mut liveness = watchdog_step::MonitorLiveness::new();
        for transition in [
            AttachAttemptTransition::Started,
            AttachAttemptTransition::Finished,
        ] {
            observe(
                &tracker,
                event(transition, AttachAttemptKind::Replace, 31),
                &ledger,
                0,
            );
        }

        ledger.publish_monitor_terminal();
        let terminal = tick(&tracker, &ledger, 1, &mut liveness);
        assert!(terminal.attach.active);
        assert_eq!(
            terminal.attach.decision,
            AttachWatchdogDecision::SensorTerminal {
                generation: 31,
                kind: AttachAttemptKind::Replace,
                finishing: true,
            },
        );
    }

    #[test]
    fn unexpected_monitor_epoch_rebases_once_then_reaches_cancel_and_fail_closed() {
        let tracker = AttachAttemptTracker::default();
        let ledger = ProgressLedger::default();
        let console = test_control_console();
        let mut liveness = watchdog_step::MonitorLiveness::new();
        tracker.observe_event(
            event(
                AttachAttemptTransition::Started,
                AttachAttemptKind::Attach,
                12,
            ),
            &ledger,
            0,
            &console,
        );

        ledger.reanchor_phase_cpu(1);
        publish(
            &ledger,
            ATTACH_SERVICE_BUDGET_NS + 1,
            CPU_CURRENCY_PMU,
            true,
            true,
        );
        let rebased = tracker.watchdog_tick(&ledger, 2, &mut liveness, &console);
        assert_eq!(rebased.attach.decision, AttachWatchdogDecision::None);
        assert_eq!(rebased.snapshot.max_vcpu_cpu_in_phase_ns, 0);

        publish(
            &ledger,
            ATTACH_SERVICE_BUDGET_NS + 1,
            CPU_CURRENCY_PMU,
            true,
            true,
        );
        assert!(matches!(
            tracker
                .watchdog_tick(&ledger, 3, &mut liveness, &console)
                .attach
                .decision,
            AttachWatchdogDecision::RequestCancel { generation: 12, .. }
        ));

        ledger.reanchor_phase_cpu(4);
        publish(
            &ledger,
            ATTACH_CANCEL_SERVICE_GRACE_NS + 1,
            CPU_CURRENCY_PMU,
            true,
            true,
        );
        assert_eq!(
            tracker
                .watchdog_tick(&ledger, 5, &mut liveness, &console)
                .attach
                .decision,
            AttachWatchdogDecision::None,
        );
        publish(
            &ledger,
            ATTACH_CANCEL_SERVICE_GRACE_NS + 1,
            CPU_CURRENCY_PMU,
            true,
            true,
        );
        assert!(matches!(
            tracker
                .watchdog_tick(&ledger, 6, &mut liveness, &console)
                .attach
                .decision,
            AttachWatchdogDecision::FailClosed { generation: 12, .. }
        ));
    }

    #[test]
    fn pthread_currency_uses_same_three_halves_widening_as_tier1() {
        let tracker = AttachAttemptTracker::default();
        let ledger = ProgressLedger::default();
        let mut liveness = watchdog_step::MonitorLiveness::new();
        observe(
            &tracker,
            event(
                AttachAttemptTransition::Started,
                AttachAttemptKind::Attach,
                2,
            ),
            &ledger,
            0,
        );
        let widened = ATTACH_SERVICE_BUDGET_NS + ATTACH_SERVICE_BUDGET_NS / 2;
        publish(&ledger, widened, CPU_CURRENCY_PTHREAD, true, true);
        assert_eq!(
            tick(&tracker, &ledger, 1, &mut liveness).attach.decision,
            AttachWatchdogDecision::None,
        );
        publish(&ledger, widened + 1, CPU_CURRENCY_PTHREAD, true, true);
        assert!(matches!(
            tick(&tracker, &ledger, 2, &mut liveness).attach.decision,
            AttachWatchdogDecision::RequestCancel { .. }
        ));

        let tracker = AttachAttemptTracker::default();
        let ledger = ProgressLedger::default();
        let mut liveness = watchdog_step::MonitorLiveness::new();
        for transition in [
            AttachAttemptTransition::Started,
            AttachAttemptTransition::Finished,
        ] {
            observe(
                &tracker,
                event(transition, AttachAttemptKind::Attach, 3),
                &ledger,
                0,
            );
        }
        let widened_finish = ATTACH_FINISH_SERVICE_GRACE_NS + ATTACH_FINISH_SERVICE_GRACE_NS / 2;
        publish(&ledger, widened_finish, CPU_CURRENCY_PTHREAD, true, true);
        assert_eq!(
            tick(&tracker, &ledger, 1, &mut liveness).attach.decision,
            AttachWatchdogDecision::None,
        );
        publish(
            &ledger,
            widened_finish + 1,
            CPU_CURRENCY_PTHREAD,
            true,
            true,
        );
        assert!(matches!(
            tick(&tracker, &ledger, 2, &mut liveness)
                .attach
                .decision,
            AttachWatchdogDecision::FinishUnsettled {
                grace_budget_ns,
                ..
            } if grace_budget_ns == widened_finish
        ));
    }

    #[test]
    fn ack_wait_with_no_delivered_service_never_consumes_attach_budget() {
        let tracker = AttachAttemptTracker::default();
        let ledger = ProgressLedger::default();
        let mut liveness = watchdog_step::MonitorLiveness::new();
        observe(
            &tracker,
            event(
                AttachAttemptTransition::Started,
                AttachAttemptKind::Restart,
                4,
            ),
            &ledger,
            0,
        );
        publish(&ledger, 0, CPU_CURRENCY_NONE, false, true);
        assert_eq!(
            tick(&tracker, &ledger, u64::MAX, &mut liveness,)
                .attach
                .decision,
            AttachWatchdogDecision::None,
        );
    }

    #[test]
    fn unacked_cancel_fails_closed_only_after_its_own_service_grace() {
        let start_cancelled = |generation| {
            let tracker = AttachAttemptTracker::default();
            let ledger = ProgressLedger::default();
            let mut liveness = watchdog_step::MonitorLiveness::new();
            observe(
                &tracker,
                event(
                    AttachAttemptTransition::Started,
                    AttachAttemptKind::Replace,
                    generation,
                ),
                &ledger,
                0,
            );
            publish(
                &ledger,
                ATTACH_SERVICE_BUDGET_NS + 1,
                CPU_CURRENCY_PMU,
                true,
                true,
            );
            assert!(matches!(
                tick(&tracker, &ledger, 100, &mut liveness).attach.decision,
                AttachWatchdogDecision::RequestCancel { .. }
            ));
            (tracker, ledger, liveness)
        };

        let (tracker, ledger, mut liveness) = start_cancelled(10);
        publish(
            &ledger,
            ATTACH_CANCEL_SERVICE_GRACE_NS + 1,
            CPU_CURRENCY_PMU,
            true,
            true,
        );
        assert!(matches!(
            tick(&tracker, &ledger, 101, &mut liveness).attach.decision,
            AttachWatchdogDecision::FailClosed { generation: 10, .. }
        ));

        let (tracker, ledger, mut liveness) = start_cancelled(11);
        publish(&ledger, 0, CPU_CURRENCY_PMU, false, true);
        assert_eq!(
            tick(&tracker, &ledger, u64::MAX, &mut liveness)
                .attach
                .decision,
            AttachWatchdogDecision::None,
            "wall time without delivered vCPU service cannot consume cancellation grace",
        );
    }

    #[test]
    fn stale_finish_cannot_close_new_generation() {
        let tracker = AttachAttemptTracker::default();
        let ledger = ProgressLedger::default();
        let mut liveness = watchdog_step::MonitorLiveness::new();
        for generation in [20, 21] {
            assert_eq!(
                observe(
                    &tracker,
                    event(
                        AttachAttemptTransition::Started,
                        AttachAttemptKind::Attach,
                        generation,
                    ),
                    &ledger,
                    generation,
                ),
                AttachEventDisposition::Started,
            );
            if generation == 21 {
                break;
            }
            assert_eq!(
                observe(
                    &tracker,
                    event(
                        AttachAttemptTransition::Finished,
                        AttachAttemptKind::Attach,
                        generation,
                    ),
                    &ledger,
                    generation,
                ),
                AttachEventDisposition::Finished,
            );
            assert_eq!(
                observe(
                    &tracker,
                    event(
                        AttachAttemptTransition::Settled,
                        AttachAttemptKind::Attach,
                        generation,
                    ),
                    &ledger,
                    generation,
                ),
                AttachEventDisposition::Settled,
            );
        }

        assert_eq!(
            observe(
                &tracker,
                event(
                    AttachAttemptTransition::Finished,
                    AttachAttemptKind::Attach,
                    20,
                ),
                &ledger,
                99,
            ),
            AttachEventDisposition::Stale,
        );
        publish(
            &ledger,
            ATTACH_SERVICE_BUDGET_NS + 1,
            CPU_CURRENCY_PMU,
            true,
            true,
        );
        assert!(matches!(
            tick(&tracker, &ledger, 100, &mut liveness).attach.decision,
            AttachWatchdogDecision::RequestCancel { generation: 21, .. }
        ));
    }
}
