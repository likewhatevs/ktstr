//! Standalone schbench_rs validation driver.
//!
//! Runs the native schbench engine ([`ktstr::workload::run_standalone`])
//! host-side, outside a VM, with schbench-compatible CLI flags, and prints a
//! latency report in the same shape as the reference schbench's `show_latencies`
//! (`schbench.c:552`). It is the artifact for the side-by-side validation:
//! invoke this and the reference `schbench` with identical flags and compare the
//! percentile tables (output identity), and compare `perf stat` / `strace -c` of
//! the two processes (effects identity).
//!
//! Divergence from the reference output: schbench samples a *per-second* RPS
//! distribution and prints its percentiles; the engine reports a single
//! whole-run mean RPS (`loop_count / runtime`), printed here as `average rps`.
//! The per-second RPS distribution is part of the RPS-injector follow-up.
//!
//! The trailing `sched delay` line also diverges, and the engine's value is the
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
/// `-n`/`-s`/`-L`/`-r`) so one invocation drives both programs identically.
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
}

fn main() {
    let args = Args::parse();
    let config = SchbenchConfig::default()
        .message_threads(args.message_threads)
        .worker_threads(args.worker_threads)
        .cache_footprint_kib(args.cache_footprint_kib)
        .operations(args.operations)
        .sleep_usec(args.sleep_usec)
        .skip_locking(args.skip_locking);

    let report = run_standalone(&config, args.runtime_secs);
    print_report(&report, args.runtime_secs);
}

/// schbench's `PLIST_FOR_LAT` masks the 20.0th percentile off latency tables
/// (`schbench.c:129`) — only the RPS table uses it (`PLIST_FOR_RPS`). The bin
/// renders no RPS table, so the 20.0th (index 0) is never printed; latency
/// tables show 50/90/99/99.9 starred at p99 (`PLIST_99`, index 3 —
/// `schbench.c:126,1801`).
const LAT_ROW_INDICES: [usize; 4] = [1, 2, 3, 4];
const LAT_STAR_INDEX: usize = 3;

/// Print the report in schbench's `show_latencies` shape (`schbench.c:552`).
fn print_report(r: &StandaloneReport, runtime_secs: u64) {
    print_distribution(
        "Wakeup Latencies",
        runtime_secs,
        r.nr_wakeup_samples,
        &r.wakeup_pcts_us,
        &r.wakeup_counts,
        r.wakeup_min_us,
        r.wakeup_max_us,
    );
    print_distribution(
        "Request Latencies",
        runtime_secs,
        r.nr_request_samples,
        &r.request_pcts_us,
        &r.request_counts,
        r.request_min_us,
        r.request_max_us,
    );
    // The engine reports a scalar mean RPS, not schbench's per-second RPS
    // distribution (see the module doc); print it as schbench's trailing
    // `average rps` line.
    println!("average rps: {:.2}", r.achieved_rps);
    // schbench prints sched delay in usec on one line (`schbench.c:1809-1812`,
    // the run_delay ns / 1000); mirror that exactly.
    println!(
        "sched delay: message {} (usec) worker {} (usec)",
        r.sched_delay_msg_ns / 1000,
        r.sched_delay_worker_ns / 1000
    );
}

fn print_distribution(
    label: &str,
    runtime_secs: u64,
    nr_samples: u64,
    pcts: &[u32; 5],
    counts: &[u64; 5],
    min: u32,
    max: u32,
) {
    println!(
        "{label} percentiles (usec) runtime {runtime_secs} (s) ({nr_samples} total samples)"
    );
    // SCHBENCH_PERCENTILES labels each row; LAT_ROW_INDICES applies schbench's
    // PLIST_FOR_LAT mask (no 20.0th for latency tables) — no hard-coded labels.
    for &i in &LAT_ROW_INDICES {
        let marker = if i == LAT_STAR_INDEX { "* " } else { "  " };
        println!(
            "\t{marker}{:.1}th: {:<10} ({} samples)",
            SCHBENCH_PERCENTILES[i], pcts[i], counts[i]
        );
    }
    println!("\t  min={min}, max={max}");
}
