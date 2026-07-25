//! Generic finite guest prerequisite/readiness-wait ownership.
//!
//! A guest publishes a generation-tagged `Started` boundary immediately
//! before it blocks on a finite prerequisite. While the matching generation is
//! active, that wait—not the coarse lifecycle phase—owns progress. The host
//! charges only delivered max-vCPU service, or its own observer CPU while the
//! responsible vCPU task set remains stably blocked. Runnable-but-starved
//! tasks spend nothing.

use std::sync::Mutex;

use crate::monitor::{CPU_CURRENCY_NONE, LedgerSnapshot, ProgressLedger};
use crate::vmm::wire::{ReadinessWaitEvent, ReadinessWaitKind, ReadinessWaitTransition};

use super::watchdog_step;

/// Raw PMU-currency and blocked-observer allowance for one readiness
/// generation. Pthread vCPU currency is widened by the shared 3/2 policy;
/// observer thread CPU is already exact and uses this value directly.
///
/// The two arms are NOT equally reachable, and only the delivered-service one
/// bounds a wait on its own. Delivered vCPU service accrues at up to wall
/// rate, so a still-serviced wait fails closed on a wall scale near the
/// budget. Observer CPU accrues only from the watchdog's own per-tick work: its
/// width-scaling term is the O(vCPU) two-file `/proc` walk, ~4.5 µs per vCPU
/// per 100 ms tick (measured, x86-64). Charging that walk as the whole duty —
/// an upper bound on wall-to-fire — 75 observer-CPU seconds need ~8e3 s of
/// wall even at 200 vCPUs, and would only come inside
/// [`watchdog_step::WALL_NET_ABSOLUTE_CEILING_NS`] past ~1600 vCPUs. So a
/// readiness wait whose vCPU tasks stay stably blocked is terminated by the
/// unconditional wall net ([`watchdog_step::wall_net_tripped`]), not by this
/// budget, at every width these tests run.
pub(super) const READINESS_WAIT_SERVICE_BUDGET_NS: u64 = 75_000_000_000;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct ActiveReadinessWait {
    pub generation: u64,
    pub kind: ReadinessWaitKind,
    accounting_epoch: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum ReadinessWaitUpdate {
    Started(ActiveReadinessWait),
    Replaced {
        prior: ActiveReadinessWait,
        current: ActiveReadinessWait,
    },
    Finished(ActiveReadinessWait),
    Duplicate,
    Stale,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum ReadinessWaitFailureCause {
    DeliveredVcpuService,
    BlockedObserverService,
    MonitorTerminal,
    ObserverClockUnavailable,
    ObserverClockRegressed,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum ReadinessWaitDecision {
    None,
    Reanchored {
        generation: u64,
        kind: ReadinessWaitKind,
    },
    FailClosed {
        generation: u64,
        kind: ReadinessWaitKind,
        cause: ReadinessWaitFailureCause,
        service_ns: u64,
        budget_ns: u64,
    },
}

#[derive(Debug)]
struct ReadinessWaitState {
    last_generation: u64,
    active: Option<ActiveReadinessWait>,
    blocked_service: watchdog_step::DeadmanHostService,
}

impl Default for ReadinessWaitState {
    fn default() -> Self {
        Self {
            last_generation: 0,
            active: None,
            blocked_service: watchdog_step::DeadmanHostService::new(),
        }
    }
}

/// One run's active finite prerequisite.
///
/// The same mutex serializes dispatch transitions and watchdog accounting, so
/// an accepted boundary cannot be paired with the opposite side's CPU epoch.
#[derive(Default, Debug)]
pub(super) struct ReadinessWaitOverlay {
    state: Mutex<ReadinessWaitState>,
}

impl ReadinessWaitOverlay {
    /// Apply one CRC- and shape-validated boundary.
    ///
    /// `reanchor` is invoked only for an accepted state transition. Duplicate
    /// and stale frames therefore cannot reset either service budget.
    pub(super) fn apply(
        &self,
        event: ReadinessWaitEvent,
        mut reanchor: impl FnMut() -> u32,
    ) -> ReadinessWaitUpdate {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        match event.transition {
            ReadinessWaitTransition::Started => {
                if let Some(current) = state.active {
                    if current.generation == event.generation && current.kind == event.kind {
                        return ReadinessWaitUpdate::Duplicate;
                    }
                    if event.generation <= current.generation {
                        return ReadinessWaitUpdate::Stale;
                    }
                    let incoming = ActiveReadinessWait {
                        generation: event.generation,
                        kind: event.kind,
                        accounting_epoch: reanchor(),
                    };
                    state.last_generation = event.generation;
                    state.active = Some(incoming);
                    state.blocked_service.reset();
                    return ReadinessWaitUpdate::Replaced {
                        prior: current,
                        current: incoming,
                    };
                }
                if event.generation <= state.last_generation {
                    return ReadinessWaitUpdate::Stale;
                }
                let incoming = ActiveReadinessWait {
                    generation: event.generation,
                    kind: event.kind,
                    accounting_epoch: reanchor(),
                };
                state.last_generation = event.generation;
                state.active = Some(incoming);
                state.blocked_service.reset();
                ReadinessWaitUpdate::Started(incoming)
            }
            ReadinessWaitTransition::Finished => {
                let Some(current) = state.active else {
                    return ReadinessWaitUpdate::Stale;
                };
                if current.generation != event.generation || current.kind != event.kind {
                    return ReadinessWaitUpdate::Stale;
                }
                // Re-anchor before clearing while still holding the state
                // mutex. The watchdog can observe either active+old epoch or
                // inactive+new epoch, never inactive+pre-wait service.
                reanchor();
                state.active = None;
                state.blocked_service.reset();
                ReadinessWaitUpdate::Finished(current)
            }
        }
    }

    /// Close the active wait of `kind` at the host readiness publication
    /// point. Re-anchors before clearing for the same atomicity contract as a
    /// guest Finished boundary.
    pub(super) fn finish_kind(
        &self,
        kind: ReadinessWaitKind,
        mut reanchor: impl FnMut() -> u32,
    ) -> Option<u64> {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        let current = state.active?;
        if current.kind != kind {
            return None;
        }
        reanchor();
        state.active = None;
        state.blocked_service.reset();
        Some(current.generation)
    }

    pub(super) fn active(&self) -> Option<ActiveReadinessWait> {
        self.state.lock().unwrap_or_else(|e| e.into_inner()).active
    }

    /// Read overlay ownership and its paired ledger generation atomically
    /// with respect to Started/Finished transitions.
    ///
    /// The watchdog's attach state machine takes its own earlier ledger
    /// snapshot. A host readiness publisher can close this overlay between
    /// that read and the ordinary-tier decision; reusing the earlier snapshot
    /// would pair inactive ownership with pre-wait CPU. Transition writers
    /// take this mutex before re-anchoring, so this method returns either the
    /// complete before-state or complete after-state.
    pub(super) fn coherent_snapshot(
        &self,
        ledger: &ProgressLedger,
    ) -> (Option<ActiveReadinessWait>, LedgerSnapshot) {
        let state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        (state.active, ledger.snapshot())
    }

    /// Charge one active generation using host-delivered service only.
    ///
    /// On exhaustion the generation is re-anchored and cleared under the same
    /// mutex before the fail-closed decision is returned. Ordinary watchdog
    /// policy may therefore resume on the next tick without inheriting the
    /// readiness interval.
    pub(super) fn watchdog_tick(
        &self,
        ledger: &ProgressLedger,
        now_wall_ns: u64,
        observer_cpu: watchdog_step::DeadmanObserverClock,
        vcpu_tasks: &[watchdog_step::HostVcpuTaskSample],
    ) -> ReadinessWaitDecision {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        let Some(mut active) = state.active else {
            state.blocked_service.reset();
            return ReadinessWaitDecision::None;
        };
        let snapshot = ledger.snapshot();

        if snapshot.monitor_terminal {
            return Self::fail_locked(
                &mut state,
                ledger,
                now_wall_ns,
                active,
                ReadinessWaitFailureCause::MonitorTerminal,
                0,
                READINESS_WAIT_SERVICE_BUDGET_NS,
            );
        }

        if snapshot.phase_epoch != active.accounting_epoch {
            active.accounting_epoch = ledger.reanchor_phase_cpu(now_wall_ns);
            state.active = Some(active);
            state.blocked_service.reset();
            return ReadinessWaitDecision::Reanchored {
                generation: active.generation,
                kind: active.kind,
            };
        }

        let delivered_budget = watchdog_step::widen_budget_for_currency(
            READINESS_WAIT_SERVICE_BUDGET_NS,
            snapshot.cpu_currency,
        );
        if snapshot.cpu_currency != CPU_CURRENCY_NONE
            && snapshot.max_vcpu_cpu_in_phase_ns >= delivered_budget
        {
            return Self::fail_locked(
                &mut state,
                ledger,
                now_wall_ns,
                active,
                ReadinessWaitFailureCause::DeliveredVcpuService,
                snapshot.max_vcpu_cpu_in_phase_ns,
                delivered_budget,
            );
        }

        match state.blocked_service.observe_with_budget(
            watchdog_step::DeadmanHostServiceInput {
                monitor_terminal: false,
                vcpu_cpu_budget_exhausted: false,
                observer_cpu,
                vcpu_tasks,
            },
            READINESS_WAIT_SERVICE_BUDGET_NS,
        ) {
            watchdog_step::DeadmanHostDecision::Fire(
                watchdog_step::DeadmanHostFire::BlockedObserverService {
                    observer_service_ns,
                    budget_ns,
                },
            ) => Self::fail_locked(
                &mut state,
                ledger,
                now_wall_ns,
                active,
                ReadinessWaitFailureCause::BlockedObserverService,
                observer_service_ns,
                budget_ns,
            ),
            watchdog_step::DeadmanHostDecision::Fire(
                watchdog_step::DeadmanHostFire::SensorFailure(
                    watchdog_step::DeadmanHostSensorFailure::ObserverClockUnavailable,
                ),
            ) => Self::fail_locked(
                &mut state,
                ledger,
                now_wall_ns,
                active,
                ReadinessWaitFailureCause::ObserverClockUnavailable,
                0,
                READINESS_WAIT_SERVICE_BUDGET_NS,
            ),
            watchdog_step::DeadmanHostDecision::Fire(
                watchdog_step::DeadmanHostFire::SensorFailure(
                    watchdog_step::DeadmanHostSensorFailure::ObserverClockRegressed { .. },
                ),
            ) => Self::fail_locked(
                &mut state,
                ledger,
                now_wall_ns,
                active,
                ReadinessWaitFailureCause::ObserverClockRegressed,
                0,
                READINESS_WAIT_SERVICE_BUDGET_NS,
            ),
            watchdog_step::DeadmanHostDecision::Defer(_)
            | watchdog_step::DeadmanHostDecision::Fire(
                watchdog_step::DeadmanHostFire::MonitorTerminal
                | watchdog_step::DeadmanHostFire::VcpuCpuBudget,
            ) => ReadinessWaitDecision::None,
        }
    }

    fn fail_locked(
        state: &mut ReadinessWaitState,
        ledger: &ProgressLedger,
        now_wall_ns: u64,
        active: ActiveReadinessWait,
        cause: ReadinessWaitFailureCause,
        service_ns: u64,
        budget_ns: u64,
    ) -> ReadinessWaitDecision {
        ledger.reanchor_phase_cpu(now_wall_ns);
        state.active = None;
        state.blocked_service.reset();
        ReadinessWaitDecision::FailClosed {
            generation: active.generation,
            kind: active.kind,
            cause,
            service_ns,
            budget_ns,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::monitor::{CPU_CURRENCY_PMU, LifecycleStage};
    use std::cell::Cell;
    use std::sync::atomic::Ordering;

    fn event(transition: ReadinessWaitTransition, generation: u64) -> ReadinessWaitEvent {
        ReadinessWaitEvent {
            transition,
            kind: ReadinessWaitKind::ProbeDump,
            generation,
        }
    }

    fn publish(ledger: &ProgressLedger, service_ns: u64) {
        let epoch = ledger.phase_epoch.load(Ordering::Acquire);
        ledger.record_liveness(
            service_ns,
            service_ns,
            epoch,
            false,
            0,
            CPU_CURRENCY_PMU,
            false,
            true,
        );
    }

    fn task(
        cpu_ns: u64,
        run_state: watchdog_step::HostVcpuRunState,
    ) -> watchdog_step::HostVcpuTaskSample {
        watchdog_step::HostVcpuTaskSample {
            task_id: Some(7),
            cpu_ns: Some(cpu_ns),
            run_state,
        }
    }

    fn start(overlay: &ReadinessWaitOverlay, ledger: &ProgressLedger, generation: u64) {
        assert!(matches!(
            overlay.apply(event(ReadinessWaitTransition::Started, generation), || {
                ledger.reanchor_phase_cpu(1)
            }),
            ReadinessWaitUpdate::Started(_)
        ));
    }

    #[test]
    fn matching_finish_closes_wait_but_stale_finish_cannot() {
        let overlay = ReadinessWaitOverlay::default();
        let ledger = ProgressLedger::default();
        start(&overlay, &ledger, 7);
        assert_eq!(
            overlay.apply(event(ReadinessWaitTransition::Finished, 6), || {
                ledger.reanchor_phase_cpu(2)
            }),
            ReadinessWaitUpdate::Stale
        );
        assert_eq!(overlay.active().map(|wait| wait.generation), Some(7));
        assert!(matches!(
            overlay.apply(event(ReadinessWaitTransition::Finished, 7), || {
                ledger.reanchor_phase_cpu(3)
            }),
            ReadinessWaitUpdate::Finished(_)
        ));
        assert_eq!(overlay.active(), None);
    }

    #[test]
    fn duplicate_and_stale_frames_cannot_reanchor_or_revive_a_host_closed_wait() {
        let overlay = ReadinessWaitOverlay::default();
        let ledger = ProgressLedger::default();
        let reanchors = Cell::new(0u32);
        let mut reanchor = || {
            reanchors.set(reanchors.get() + 1);
            ledger.reanchor_phase_cpu(u64::from(reanchors.get()))
        };

        assert!(matches!(
            overlay.apply(event(ReadinessWaitTransition::Started, 11), &mut reanchor),
            ReadinessWaitUpdate::Started(_)
        ));
        assert_eq!(reanchors.get(), 1);
        assert_eq!(
            overlay.apply(event(ReadinessWaitTransition::Started, 11), &mut reanchor),
            ReadinessWaitUpdate::Duplicate
        );
        assert_eq!(
            overlay.apply(event(ReadinessWaitTransition::Finished, 10), &mut reanchor),
            ReadinessWaitUpdate::Stale
        );
        assert_eq!(reanchors.get(), 1);

        assert_eq!(
            overlay.finish_kind(ReadinessWaitKind::ProbeDump, &mut reanchor),
            Some(11)
        );
        assert_eq!(reanchors.get(), 2);
        assert_eq!(
            overlay.apply(event(ReadinessWaitTransition::Finished, 11), &mut reanchor),
            ReadinessWaitUpdate::Stale
        );
        assert_eq!(
            overlay.apply(event(ReadinessWaitTransition::Started, 11), &mut reanchor),
            ReadinessWaitUpdate::Stale
        );
        assert_eq!(reanchors.get(), 2);
        assert_eq!(overlay.active(), None);
    }

    #[test]
    fn coherent_close_snapshot_cannot_pair_inactive_wait_with_pre_wait_service() {
        let overlay = ReadinessWaitOverlay::default();
        let ledger = ProgressLedger::default();
        start(&overlay, &ledger, 12);
        publish(&ledger, READINESS_WAIT_SERVICE_BUDGET_NS);
        assert_eq!(
            overlay.finish_kind(ReadinessWaitKind::ProbeDump, || {
                ledger.reanchor_phase_cpu(2)
            }),
            Some(12)
        );

        let (active, snapshot) = overlay.coherent_snapshot(&ledger);
        assert_eq!(active, None);
        assert_eq!(
            snapshot.max_vcpu_cpu_in_phase_ns, 0,
            "the monitor has not published the post-close epoch yet, so the \
             pre-wait service must be rejected"
        );
    }

    #[test]
    fn runnable_starved_task_spends_no_blocked_observer_budget() {
        let overlay = ReadinessWaitOverlay::default();
        let ledger = ProgressLedger::default();
        start(&overlay, &ledger, 1);
        publish(&ledger, 0);
        for observer in [1, READINESS_WAIT_SERVICE_BUDGET_NS * 2] {
            assert_eq!(
                overlay.watchdog_tick(
                    &ledger,
                    observer,
                    watchdog_step::DeadmanObserverClock::Reading(observer),
                    &[task(0, watchdog_step::HostVcpuRunState::Runnable)],
                ),
                ReadinessWaitDecision::None
            );
        }
        assert!(overlay.active().is_some());
    }

    #[test]
    fn delivered_vcpu_service_exhaustion_fails_closed() {
        let overlay = ReadinessWaitOverlay::default();
        let ledger = ProgressLedger::default();
        start(&overlay, &ledger, 2);
        publish(&ledger, READINESS_WAIT_SERVICE_BUDGET_NS + 1);
        assert!(matches!(
            overlay.watchdog_tick(
                &ledger,
                2,
                watchdog_step::DeadmanObserverClock::Reading(1),
                &[task(
                    READINESS_WAIT_SERVICE_BUDGET_NS + 1,
                    watchdog_step::HostVcpuRunState::NonRunnable,
                )],
            ),
            ReadinessWaitDecision::FailClosed {
                cause: ReadinessWaitFailureCause::DeliveredVcpuService,
                ..
            }
        ));
        assert!(overlay.active().is_none());
    }

    #[test]
    fn blocked_observer_service_exhaustion_fails_closed() {
        let overlay = ReadinessWaitOverlay::default();
        let ledger = ProgressLedger::default();
        start(&overlay, &ledger, 3);
        publish(&ledger, 0);
        let blocked = [task(0, watchdog_step::HostVcpuRunState::NonRunnable)];
        assert_eq!(
            overlay.watchdog_tick(
                &ledger,
                1,
                watchdog_step::DeadmanObserverClock::Reading(10),
                &blocked,
            ),
            ReadinessWaitDecision::None
        );
        assert!(matches!(
            overlay.watchdog_tick(
                &ledger,
                2,
                watchdog_step::DeadmanObserverClock::Reading(
                    10 + READINESS_WAIT_SERVICE_BUDGET_NS,
                ),
                &blocked,
            ),
            ReadinessWaitDecision::FailClosed {
                cause: ReadinessWaitFailureCause::BlockedObserverService,
                ..
            }
        ));
    }

    #[test]
    fn foreign_phase_progress_reanchors_before_service_is_charged() {
        let overlay = ReadinessWaitOverlay::default();
        let ledger = ProgressLedger::default();
        start(&overlay, &ledger, 4);
        ledger.advance_phase(LifecycleStage::Attach as u8, 2);
        publish(&ledger, READINESS_WAIT_SERVICE_BUDGET_NS + 1);
        assert!(matches!(
            overlay.watchdog_tick(
                &ledger,
                3,
                watchdog_step::DeadmanObserverClock::Reading(1),
                &[task(1, watchdog_step::HostVcpuRunState::NonRunnable)],
            ),
            ReadinessWaitDecision::Reanchored { generation: 4, .. }
        ));
        assert!(overlay.active().is_some());
    }
}
