//! Standalone taobench_rs validation driver.
//!
//! Runs the native taobench engine
//! ([`ktstr::workload::taobench_run_standalone`]) host-side, outside a VM, and
//! prints a summary in the shape the reference taobench server reports:
//! `fast_qps` / `hit_rate` / `slow_qps`, plus the derived `total_qps`
//! (= fast + slow) and `hit_ratio` (= fast / total). This is the artifact for the
//! side-by-side validation against the reference taobench: run this and
//! the reference taobench single-host, then compare.
//!
//! Magnitude is environment-bound — the reference's published numbers are a
//! many-core, many-instance figure, not head-to-head with a single host-side
//! engine run. The comparison is STRUCTURAL: the fast≫slow split, the
//! steady-state hit ratio, and that both compute the same metric formulas.
//!
//! Gated behind the `integration` feature so it is not built by the default
//! `cargo install` — it is a validation tool, not a shipped CLI.

#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

use clap::Parser;

use ktstr::workload::{TaobenchConfig, TaobenchStandaloneReport, taobench_run_standalone};

/// taobench-port validation flags — the `TaobenchConfig` knobs (the real taobench
/// is driven separately via its own runner; these configure a comparable port
/// run).
#[derive(Parser)]
#[command(
    name = "ktstr-taobench-validate",
    about = "Run the native taobench_rs engine host-side and print a taobench-comparable qps/hit-ratio report"
)]
struct Args {
    /// Client threads (0 = one per allocated CPU).
    #[arg(short = 'c', long, default_value_t = 0)]
    client_threads: usize,
    /// Slow dispatcher threads (0 = max(1, client_threads / 3)).
    #[arg(short = 's', long, default_value_t = 0)]
    slow_threads: usize,
    /// Resident cache budget in MiB.
    #[arg(short = 'm', long, default_value_t = 64)]
    cache_capacity_mib: usize,
    /// Target steady-state hit ratio, percent (1..=99).
    #[arg(short = 'H', long, default_value_t = 90)]
    target_hit_pct: usize,
    /// Simulated backing-store fetch latency on a miss, microseconds (the MEDIAN
    /// when --slow-path-p99-us enables the heavy tail).
    #[arg(short = 'u', long, default_value_t = 100)]
    slow_path_sleep_us: u64,
    /// Heavy-tailed slow-path service-time p99 in microseconds (0, <=
    /// slow_path_sleep_us, or slow_path_sleep_us == 0 = fixed latency; a zero
    /// median has no Pareto scale). When larger than a non-zero slow_path_sleep_us,
    /// each fetch is a Pareto draw with median slow_path_sleep_us and this p99 —
    /// under -R the serve-latency line below then exhibits the tail.
    #[arg(short = 'p', long, default_value_t = 0)]
    slow_path_p99_us: u64,
    /// Benchmark runtime in seconds.
    #[arg(short = 'r', long, default_value_t = 30)]
    runtime_secs: u64,
    /// Open-loop AGGREGATE arrival rate in ops/sec across all clients (0 = closed
    /// loop). Non-zero enables coordinated-omission serve-latency measurement.
    #[arg(short = 'R', long, default_value_t = 0)]
    arrival_rate: usize,
}

fn main() {
    ktstr::host_heap::mark_jemalloc_global_allocator();
    let args = Args::parse();
    let config = TaobenchConfig::default()
        .client_threads(args.client_threads)
        .slow_threads(args.slow_threads)
        .cache_capacity_mib(args.cache_capacity_mib)
        .target_hit_pct(args.target_hit_pct)
        .slow_path_sleep_us(args.slow_path_sleep_us)
        .slow_path_p99_us(args.slow_path_p99_us)
        .arrival_rate(args.arrival_rate);

    let report = taobench_run_standalone(&config, args.runtime_secs);
    print_report(&report, args.runtime_secs);
}

/// Print the report in the reference taobench's metric shape (the parser keys
/// `fast_qps` / `hit_rate` / `slow_qps`, plus the derived `total_qps` /
/// `hit_ratio`).
fn print_report(r: &TaobenchStandaloneReport, runtime_secs: u64) {
    println!(
        "taobench_rs standalone — runtime {runtime_secs}s, {} client threads, {} slow dispatchers, {:.2}s measured window",
        r.nr_client_threads, r.nr_slow_threads, r.elapsed_secs
    );
    println!("total_qps = {:.1}", r.total_qps);
    println!("fast_qps  = {:.1}", r.fast_qps);
    println!("slow_qps  = {:.1}", r.slow_qps);
    println!("hit_ratio = {:.4}  (fast / (fast + slow))", r.hit_ratio);
    println!(
        "hit_rate  = {:.4}  (1 - get_misses / get_cmds, command-time)",
        r.hit_rate
    );
    println!(
        "ops: total={} fast={} slow={}",
        r.total_ops, r.fast_ops, r.slow_ops
    );
    // Open-loop coordinated-omission serve latency (absent in closed loop).
    if let (Some(p50), Some(p99), Some(p999), Some(min), Some(max)) = (
        r.serve_p50_us,
        r.serve_p99_us,
        r.serve_p999_us,
        r.serve_min_us,
        r.serve_max_us,
    ) {
        println!(
            "serve_us (coordinated-omission, from intended arrival): min={min} p50={p50} p99={p99} p99.9={p999} max={max}"
        );
    }
}
