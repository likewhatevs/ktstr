//! schbench_rs — a faithful native re-expression of schbench in ktstr.
//! No binary, no subprocess: the schbench algorithm is
//! re-expressed in ktstr's own workload / scenario / metric primitives so
//! its numbers flow natively through the metric API (phases, assertions,
//! perf-delta).
//!
//! Phase 1 lands the metric foundation: [`plat`], schbench's bit-exact fio
//! log2 histogram + percentiles. Later phases (the per-CPU-lock stressor,
//! the futex handshake, the `Schbench` `WorkType`, RPS / request-latency /
//! schedstat capture, and the phasic metric path) consume these items;
//! until then they are `dead_code` in a non-test build.
#![allow(dead_code)]

pub(crate) mod plat;
