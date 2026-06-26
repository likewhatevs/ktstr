//! Standalone schbench_rs validation driver.
//!
//! Runs the native schbench engine ([`ktstr::workload::run_standalone`])
//! host-side, outside a VM, with schbench-compatible CLI flags, and prints a
//! latency report in the same shape as the reference schbench's `show_latencies`
//! (`schbench.c:552`). It is the artifact for the side-by-side validation:
//! invoke this and the reference `schbench` with identical flags and compare the
//! percentile tables (output identity), and compare `perf stat` / `strace -c` of
//! the two processes (effects identity). The output mirrors schbench's final
//! summary (`schbench.c:1985-2004`): the Wakeup, Request, and RPS percentile
//! tables, then either `average rps` (fixed/default mode) or `final rps goal was`
//! (auto-RPS), then the `sched delay` line.
//!
//! The trailing `sched delay` line diverges, and the engine's value is the
//! more accurate one: schbench's FINAL summary runs `collect_sched_delay` after
//! every thread has exited — each message thread joins its workers
//! (`schbench.c:1599-1602`) before main joins the message threads
//! (`schbench.c:1931-1934`) — so each thread's `/proc/<tid>/schedstat` is already
//! gone and the read fails to 0 (`schbench.c:1664-1670`, `:1128-1130`); schbench
//! prints `message 0 worker 0`. The engine reads each thread's schedstat from
//! inside the thread before it exits, so it reports the live per-thread run-delay.
//!
//! Gated behind the `integration` feature so it is not built by the default
//! `cargo install` — it is a validation tool, not a shipped CLI.

use clap::Parser;

use ktstr::workload::{run_standalone, SchbenchConfig, StandaloneReport, SCHBENCH_PERCENTILES};

/// schbench-compatible flags. The short forms match schbench (`-m`/`-t`/`-F`/
/// `-n`/`-s`/`-L`/`-r`/`-R`) so one invocation drives both programs identically.
#[derive(Parser)]
#[command(
    name = "ktstr-schbench-validate",
    about = "Run the native schbench_rs engine host-side and print a schbench-comparable latency report"
)]
struct Args {
    /// Message threads (schbench `-m`).
    #[arg(short = 'm', long, default_value_t = 1)]
    message_threads: usize,
    /// Worker threads per message thread (schbench `-t`); 0 = one per CPU.
    #[arg(short = 't', long, default_value_t = 0)]
    worker_threads: usize,
    /// Per-worker matrix cache footprint in KiB (schbench `-F`).
    #[arg(short = 'F', long, default_value_t = 256)]
    cache_footprint_kib: usize,
    /// Matrix multiplications per work cycle (schbench `-n`).
    #[arg(short = 'n', long, default_value_t = 5)]
    operations: usize,
    /// Think-time sleep before the matrix work, microseconds (schbench `-s`).
    #[arg(short = 's', long, default_value_t = 100)]
    sleep_usec: u64,
    /// Skip the per-CPU lock around the matrix work (schbench `-L`).
    #[arg(short = 'L', long, default_value_t = false)]
    skip_locking: bool,
    /// Benchmark runtime in seconds (schbench `-r`, default 30 — same as schbench).
    #[arg(short = 'r', long, default_value_t = 30)]
    runtime_secs: u64,
    /// Fixed request rate, requests/second (schbench `-R`); 0 = off (the default
    /// message-handshake mode), non-zero drives the RPS-injector mode.
    #[arg(short = 'R', long, default_value_t = 0)]
    rps: usize,
    /// Auto-RPS target CPU-busy percentage (schbench `-A`); 0 = off. Non-zero
    /// turns on closed-loop rate control (seeds the rate to 10 when `-R` is 0).
    #[arg(short = 'A', long, default_value_t = 0)]
    auto_rps: usize,
}

fn main() {
    let args = Args::parse();
    let config = SchbenchConfig::default()
        .message_threads(args.message_threads)
        .worker_threads(args.worker_threads)
        .cache_footprint_kib(args.cache_footprint_kib)
        .operations(args.operations)
        .sleep_usec(args.sleep_usec)
        .skip_locking(args.skip_locking)
        .requests_per_sec(args.rps)
        .auto_rps(args.auto_rps);

    let report = run_standalone(&config, args.runtime_secs);
    print_report(&report, args.runtime_secs, args.auto_rps);
}

/// schbench's `PLIST_FOR_LAT` masks the 20.0th percentile off latency tables
/// (`schbench.c:129`), showing 50/90/99/99.9 starred at p99 (`PLIST_99`, index 3
/// — `schbench.c:126,1801`).
const LAT_ROW_INDICES: [usize; 4] = [1, 2, 3, 4];
const LAT_STAR_INDEX: usize = 3;

/// schbench's `PLIST_FOR_RPS` shows 20/50/90 starred at p50 (`PLIST_50`, index 1
/// — `schbench.c:130,1808`).
const RPS_ROW_INDICES: [usize; 3] = [0, 1, 2];
const RPS_STAR_INDEX: usize = 1;

/// Print the report in schbench's final-summary shape (`schbench.c:1985-2004`):
/// the Wakeup, Request, and RPS percentile tables, then the RPS-goal/average
/// line, then sched delay. `auto_rps` selects the schbench branch for that line.
fn print_report(r: &StandaloneReport, runtime_secs: u64, auto_rps: usize) {
    print_distribution(
        "Wakeup Latencies",
        "usec",
        runtime_secs,
        r.nr_wakeup_samples,
        &r.wakeup_pcts_us,
        &r.wakeup_counts,
        r.wakeup_min_us,
        r.wakeup_max_us,
        &LAT_ROW_INDICES,
        LAT_STAR_INDEX,
    );
    print_distribution(
        "Request Latencies",
        "usec",
        runtime_secs,
        r.nr_request_samples,
        &r.request_pcts_us,
        &r.request_counts,
        r.request_min_us,
        r.request_max_us,
        &LAT_ROW_INDICES,
        LAT_STAR_INDEX,
    );
    print_distribution(
        "RPS",
        "requests",
        runtime_secs,
        r.nr_rps_samples,
        &r.rps_pcts,
        &r.rps_counts,
        r.rps_min,
        r.rps_max,
        &RPS_ROW_INDICES,
        RPS_STAR_INDEX,
    );
    // schbench prints `final rps goal was N` under auto-RPS, else `average rps`
    // as loop_count / runtime (`schbench.c:1991-1997`).
    if auto_rps != 0 {
        println!("final rps goal was {}", r.final_rps_goal);
    } else {
        println!("average rps: {:.2}", r.loop_count as f64 / runtime_secs as f64);
    }
    // schbench prints sched delay in usec on one line (`schbench.c:2001-2004`,
    // the run_delay ns / 1000); mirror that exactly.
    println!(
        "sched delay: message {} (usec) worker {} (usec)",
        r.sched_delay_msg_ns / 1000,
        r.sched_delay_worker_ns / 1000
    );
}

#[allow(clippy::too_many_arguments)]
fn print_distribution(
    label: &str,
    units: &str,
    runtime_secs: u64,
    nr_samples: u64,
    pcts: &[u32; 5],
    counts: &[u64; 5],
    min: u32,
    max: u32,
    rows: &[usize],
    star: usize,
) {
    println!(
        "{label} percentiles ({units}) runtime {runtime_secs} (s) ({nr_samples} total samples)"
    );
    // SCHBENCH_PERCENTILES labels each row; `rows` applies schbench's per-table
    // mask (PLIST_FOR_LAT / PLIST_FOR_RPS) — no hard-coded labels.
    for &i in rows {
        let marker = if i == star { "* " } else { "  " };
        println!(
            "\t{marker}{:.1}th: {:<10} ({} samples)",
            SCHBENCH_PERCENTILES[i], pcts[i], counts[i]
        );
    }
    println!("\t  min={min}, max={max}");
}
