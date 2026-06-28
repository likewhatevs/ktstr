//! taobench_rs — a native re-expression of the taobench key-value object-cache
//! workload in ktstr. No real cache server, no RPC framework, no subprocess, no sockets: the
//! taobench ACCESS PATTERN — a GET-dominated key-value object cache (a
//! steady-state hit ratio engineered by key-range-vs-cache-capacity, a fast
//! in-cache request tier plus a slow backing-store-miss tier, small-heavy
//! long-tail value sizes) — is re-expressed in ktstr's own workload / metric
//! primitives so its qps / hit-ratio flow natively through the metric API
//! (phases, assertions, perf-delta). Backs the
//! [`Taobench`](crate::workload::WorkType::Taobench) workload.
//!
//! This models the access PATTERN and the CPU / memory / scheduling
//! characteristic, NOT the wire protocol: it is in-process (no real sockets or
//! TLS), with a bounded evicting cache rather than a production LRU, and
//! the slow-tier backing-store fetch is a workload-defined sleep (the same
//! think-sleep model schbench uses). The per-aspect real-vs-port fidelity
//! comparison — what matches, what is approximated, what diverges by design — is
//! in `validation.md` alongside this module.
//!
//! Modules: [`run`] (the engine — a bounded sharded cache with eviction, client
//! load threads, fast-hit worker threads and slow-miss dispatcher threads, and
//! per-phase stats reduction).

pub(crate) mod run;

/// User-facing config for the [`Taobench`](crate::workload::WorkType::Taobench)
/// workload.
pub use run::TaobenchConfig;

/// Whole-run / per-phase taobench engine counters, the carried aggregate on
/// [`crate::workload::WorkerReport::taobench_whole`] and
/// [`crate::assert::CgroupStats::taobench_whole`]. The host derives the run-level
/// `taobench_*` Rate metrics from it; test authors normally assert those metrics
/// rather than reading this struct directly.
pub use run::TaobenchStats;

/// Host-side standalone runner + its report, backing the `ktstr-taobench-validate`
/// validation driver (the analog of schbench's `run_standalone`).
pub use run::{TaobenchStandaloneReport, run_standalone};
