//! schbench_rs — a faithful native re-expression of schbench in ktstr.
//! No binary, no subprocess: the schbench algorithm is
//! re-expressed in ktstr's own workload / scenario / metric primitives so
//! its numbers flow natively through the metric API (phases, assertions,
//! perf-delta).
//!
//! Landed so far: [`plat`] (schbench's bit-exact fio log2 histogram +
//! percentiles), [`percpu_lock`] (the per-CPU mutex stressor), [`handshake`]
//! (the futex message<->worker handshake), and [`run`] (the default-mode run
//! engine: topology, lockless wait-list, and the wakeup + request latency
//! loop). Later phases (the `Schbench` `WorkType` that wires [`run`] into a
//! workload, the RPS injector + auto-rps, scheduler-delay (schedstat) capture,
//! and the phasic metric path) consume these items; until then they are
//! `dead_code` in a non-test build.
#![allow(dead_code)]

pub(crate) mod handshake;
pub(crate) mod percpu_lock;
pub(crate) mod plat;
pub(crate) mod run;
