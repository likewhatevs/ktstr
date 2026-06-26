//! schbench_rs — a faithful native re-expression of schbench in ktstr.
//! No binary, no subprocess: the schbench algorithm is
//! re-expressed in ktstr's own workload / scenario / metric primitives so
//! its numbers flow natively through the metric API (phases, assertions,
//! perf-delta).
//!
//! Modules: [`plat`] (schbench's bit-exact fio log2 histogram + percentiles),
//! [`percpu_lock`] (the per-CPU mutex stressor), [`handshake`] (the futex
//! message<->worker handshake), and [`run`] (the run engine: topology, lockless
//! wait-list, the wakeup + request latency loop, per-phase histogram snapshots,
//! and schedstat run-delay capture). [`run::run`] backs the
//! [`Schbench`](crate::workload::WorkType::Schbench) workload and the per-phase
//! metric path; [`run_standalone`] drives the same engine host-side, outside a
//! VM, for the side-by-side validation against the reference schbench. The
//! RPS-injector mode (`-R`) and its auto-RPS rate control (`-A`) are part of the
//! engine; the control thread also samples the per-second RPS distribution.

pub(crate) mod handshake;
pub(crate) mod percpu_lock;
pub(crate) mod plat;
pub(crate) mod run;

/// User-facing config for the [`Schbench`](crate::workload::WorkType::Schbench) workload.
pub use run::SchbenchConfig;

/// The five latency percentiles reported by [`StandaloneReport`] and the
/// per-phase metric path, in column order: 20.0, 50.0, 90.0, 99.0, 99.9. Matches
/// schbench's percentile rows (`schbench.c` `show_latencies`). Callers label the
/// [`StandaloneReport`] percentile arrays by zipping with this slice rather than
/// hard-coding an index-to-percentile mapping.
pub const SCHBENCH_PERCENTILES: [f64; 5] = plat::PLIST;

/// Whole-run result of a standalone (no-VM) schbench engine run, projected for
/// the side-by-side comparison against the reference schbench. The percentile
/// arrays index in [`SCHBENCH_PERCENTILES`] order (20.0, 50.0, 90.0, 99.0,
/// 99.9), in microseconds. The sample counts are carried so a zero-sample run is
/// visible rather than silently reported as an all-zero distribution.
#[derive(Debug, Clone, Copy)]
pub struct StandaloneReport {
    /// Wakeup-latency percentiles (µs), in [`SCHBENCH_PERCENTILES`] order.
    pub wakeup_pcts_us: [u32; 5],
    /// Differenced per-bucket sample count at each percentile (schbench's
    /// per-row `(N samples)`), in [`SCHBENCH_PERCENTILES`] order.
    pub wakeup_counts: [u64; 5],
    /// Minimum observed wakeup latency (µs).
    pub wakeup_min_us: u32,
    /// Maximum observed wakeup latency (µs).
    pub wakeup_max_us: u32,
    /// Number of wakeup-latency samples folded into the percentiles.
    pub nr_wakeup_samples: u64,
    /// Request-latency percentiles (µs), in [`SCHBENCH_PERCENTILES`] order.
    pub request_pcts_us: [u32; 5],
    /// Differenced per-bucket sample count at each percentile (schbench's
    /// per-row `(N samples)`), in [`SCHBENCH_PERCENTILES`] order.
    pub request_counts: [u64; 5],
    /// Minimum observed request latency (µs).
    pub request_min_us: u32,
    /// Maximum observed request latency (µs).
    pub request_max_us: u32,
    /// Number of request-latency samples folded into the percentiles.
    pub nr_request_samples: u64,
    /// Per-second achieved-RPS percentiles (requests/sec), in
    /// [`SCHBENCH_PERCENTILES`] order — schbench's `rps_stats` table sampled once
    /// per second by the control thread (`schbench.c:1777`). Unitless rate, not
    /// µs, so no `_us` suffix.
    pub rps_pcts: [u32; 5],
    /// Differenced per-bucket sample count at each RPS percentile, in
    /// [`SCHBENCH_PERCENTILES`] order.
    pub rps_counts: [u64; 5],
    /// Minimum observed per-second RPS sample.
    pub rps_min: u32,
    /// Maximum observed per-second RPS sample.
    pub rps_max: u32,
    /// Number of per-second RPS samples folded into the percentiles.
    pub nr_rps_samples: u64,
    /// Auto-RPS final TOTAL target rate at run exit (per-thread live rate *
    /// message_threads), schbench's `final rps goal` (`schbench.c:1995`). Equal to
    /// the seeded total for fixed `-R`/default mode; diverges only under auto-RPS.
    pub final_rps_goal: usize,
    /// Completed work cycles per second over the TRUE elapsed run window
    /// (`loop_count / elapsed`). NOT schbench's `average rps` summary line, which
    /// divides by the integer `-r` runtime — `schbench_validate` prints that
    /// (`loop_count / runtime_secs`) separately; this field is the measured
    /// elapsed-window rate.
    pub achieved_rps: f64,
    /// Mean message-thread run-queue wait (ns), schedstat mean-of-means.
    pub sched_delay_msg_ns: u64,
    /// Mean worker-thread run-queue wait (ns), schedstat mean-of-means.
    pub sched_delay_worker_ns: u64,
    /// Total work-loop iterations across all worker threads.
    pub loop_count: u64,
}

/// Run the schbench engine standalone — host-side, no VM, no phases — for
/// `run_secs` seconds and project the whole-run result into a
/// [`StandaloneReport`] for the side-by-side validation against the reference
/// schbench.
///
/// The `run_secs` window mirrors schbench's `-r <secs>`: it is the benchmark's
/// own defined runtime — the workload behavior, like the per-request think-sleep
/// in [`run`] — not a harness poll or synchronization wait. The engine itself
/// stays stop-gated and event-driven; this wrapper is the only place a
/// wall-clock timer drives it, bounding the benchmark window the way `-r` does
/// upstream.
///
/// Non-phasic: `phase_epoch` is `None`, so the engine produces a single
/// whole-run aggregate — the reference schbench has no phases, and the
/// comparison is whole-run to whole-run.
pub fn run_standalone(config: &SchbenchConfig, run_secs: u64) -> StandaloneReport {
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

    let stop = AtomicBool::new(false);
    let progress = AtomicU64::new(0);
    let outcome = std::thread::scope(|s| {
        let runner = s.spawn(|| run::run(config, &stop, &progress, None));
        // The `-r` benchmark window: the workload's defined runtime, not a
        // synchronization sleep. The engine runs until `stop` is set here.
        std::thread::sleep(std::time::Duration::from_secs(run_secs));
        stop.store(true, Ordering::Release);
        runner.join().expect("schbench standalone runner panicked")
    });

    let w = &outcome.whole_run;
    StandaloneReport {
        wakeup_pcts_us: w.wakeup.values,
        wakeup_counts: w.wakeup.counts,
        wakeup_min_us: w.wakeup.min,
        wakeup_max_us: w.wakeup.max,
        nr_wakeup_samples: w.wakeup.nr_samples,
        request_pcts_us: w.request.values,
        request_counts: w.request.counts,
        request_min_us: w.request.min,
        request_max_us: w.request.max,
        nr_request_samples: w.request.nr_samples,
        rps_pcts: w.rps.values,
        rps_counts: w.rps.counts,
        rps_min: w.rps.min,
        rps_max: w.rps.max,
        nr_rps_samples: w.rps.nr_samples,
        final_rps_goal: w.final_rps_goal,
        achieved_rps: w.achieved_rps,
        sched_delay_msg_ns: w.sched_delay_msg_ns,
        sched_delay_worker_ns: w.sched_delay_worker_ns,
        loop_count: w.loop_count,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `run_standalone` runs the host engine for the benchmark window and the
    /// projection fills every [`StandaloneReport`] field from the whole-run
    /// aggregate. Pins the pub entry the `ktstr-schbench-validate` bin depends
    /// on. Minimal topology (1 message thread, 1 worker, no think-sleep) over a
    /// 2-second window — the wakeup/request loop records latency samples, and the
    /// control thread fires at the 1-second tick so at least one per-second RPS
    /// sample lands before stop (a 1s window would race the single tick).
    #[test]
    fn run_standalone_fills_report_from_a_real_run() {
        let config = SchbenchConfig::default()
            .message_threads(1)
            .worker_threads(1)
            .sleep_usec(0);
        let report = run_standalone(&config, 2);

        // The engine did work and paced requests over the window.
        assert!(report.loop_count > 0, "loop_count: {}", report.loop_count);
        assert!(
            report.achieved_rps > 0.0,
            "achieved_rps: {}",
            report.achieved_rps
        );

        // Both latency distributions recorded samples.
        assert!(report.nr_wakeup_samples > 0, "wakeup samples");
        assert!(report.nr_request_samples > 0, "request samples");

        // The control thread sampled the per-second RPS distribution (default
        // mode samples every second; the 1s tick lands inside the 2s window).
        assert!(
            report.nr_rps_samples > 0,
            "rps samples: {}",
            report.nr_rps_samples
        );

        // Percentile values projected in order: a distribution is monotonic
        // non-decreasing across p20..p99.9 (a higher percentile sits at a
        // higher histogram bucket). Catches a transposed/garbled values array.
        for w in report.wakeup_pcts_us.windows(2) {
            assert!(
                w[0] <= w[1],
                "wakeup percentiles monotonic: {:?}",
                report.wakeup_pcts_us
            );
        }
        for w in report.request_pcts_us.windows(2) {
            assert!(
                w[0] <= w[1],
                "request percentiles monotonic: {:?}",
                report.request_pcts_us
            );
        }
        for w in report.rps_pcts.windows(2) {
            assert!(
                w[0] <= w[1],
                "rps percentiles monotonic: {:?}",
                report.rps_pcts
            );
        }

        // Per-row counts carried through (not zeroed by the projection). They are
        // schbench's DIFFERENCED per-band counts, so they sum to the cumulative
        // count at p99.9 -- positive and bounded above by the total, not equal to
        // it.
        let wc: u64 = report.wakeup_counts.iter().sum();
        assert!(
            wc > 0 && wc <= report.nr_wakeup_samples,
            "wakeup counts {wc} in (0, {}]",
            report.nr_wakeup_samples
        );
        let rc: u64 = report.request_counts.iter().sum();
        assert!(
            rc > 0 && rc <= report.nr_request_samples,
            "request counts {rc} in (0, {}]",
            report.nr_request_samples
        );
        let rpsc: u64 = report.rps_counts.iter().sum();
        assert!(
            rpsc > 0 && rpsc <= report.nr_rps_samples,
            "rps counts {rpsc} in (0, {}]",
            report.nr_rps_samples
        );

        // min/max projected (carried, not swapped); min <= max is the
        // invariant. NOT `min <= pcts[0]` / `pcts[4] <= max`: min/max are exact
        // sample values, but a percentile value is the log-bucket MIDPOINT for
        // tails (plat_idx_to_val for idx >= 512), so a >=512µs sample in the
        // p99.9 bucket can put pcts[4] above the exact max (and min above
        // pcts[0]) -- a real value, not a bug, so bracketing would flake.
        assert!(report.wakeup_min_us <= report.wakeup_max_us);
        assert!(report.request_min_us <= report.request_max_us);
        assert!(report.rps_min <= report.rps_max);
    }
}
