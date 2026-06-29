//! Failure-output formatting for eval: the `--- monitor ---` section,
//! boot-settle sample trimming, and the scheduler label. Split out of
//! eval/mod.rs to keep the module under the size ceiling.

use crate::test_support::SchedulerSpec;

/// Format the `--- monitor ---` section for failure output.
///
/// Shows peak values, averaged metrics, event counter rates, schedstat
/// rates, and the monitor verdict. All values are from the post-warmup
/// evaluation window (boot-settle samples trimmed).
pub(crate) fn format_monitor_section(
    monitor: &crate::monitor::MonitorReport,
    merged_assert: &crate::assert::Assert,
) -> String {
    let eval_report = trim_settle_samples(monitor);
    let s = &eval_report.summary;
    let thresholds = merged_assert.monitor_thresholds();
    let verdict = thresholds.evaluate(&eval_report);
    let verdict_line = if verdict.is_pass() || verdict.is_inconclusive() {
        // Inconclusive arms (no samples / uninitialized data)
        // carry their narrative in `summary`, not `details` —
        // appending `: details.join` would render
        // "no monitor samples: " with an empty trailer.
        verdict.summary.clone()
    } else {
        format!("{}: {}", verdict.summary, verdict.details.join("; "))
    };

    let mut lines = vec![
        format!(
            "samples={} max_imbalance={:.2} max_dsq_depth={} stuck={}",
            s.total_samples, s.max_imbalance_ratio, s.max_local_dsq_depth, s.stuck_count,
        ),
        format!(
            "avg: imbalance={:.2} nr_running/cpu={:.1} dsq/cpu={:.1}",
            s.avg_imbalance_ratio, s.avg_nr_running, s.avg_local_dsq_depth,
        ),
    ];

    if let Some(ref ev) = s.event_deltas {
        lines.push(format!(
            "events: fallback={} ({:.1}/s) keep_last={} ({:.1}/s) offline={}",
            ev.total_fallback,
            ev.fallback_rate,
            ev.total_dispatch_keep_last,
            ev.keep_last_rate,
            ev.total_dispatch_offline,
        ));
        let mut extra = Vec::new();
        if ev.total_reenq_immed != 0 {
            extra.push(format!("reenq_immed={}", ev.total_reenq_immed));
        }
        if ev.total_reenq_local_repeat != 0 {
            extra.push(format!(
                "reenq_local_repeat={}",
                ev.total_reenq_local_repeat
            ));
        }
        if ev.total_refill_slice_dfl != 0 {
            extra.push(format!("refill_slice_dfl={}", ev.total_refill_slice_dfl));
        }
        if ev.total_bypass_activate != 0 {
            extra.push(format!("bypass_activate={}", ev.total_bypass_activate));
        }
        if ev.total_bypass_dispatch != 0 {
            extra.push(format!("bypass_dispatch={}", ev.total_bypass_dispatch));
        }
        if ev.total_bypass_duration != 0 {
            extra.push(format!("bypass_duration={}ns", ev.total_bypass_duration));
        }
        if ev.total_insert_not_owned != 0 {
            extra.push(format!("insert_not_owned={}", ev.total_insert_not_owned));
        }
        if ev.total_sub_bypass_dispatch != 0 {
            extra.push(format!(
                "sub_bypass_dispatch={}",
                ev.total_sub_bypass_dispatch
            ));
        }
        if !extra.is_empty() {
            lines.push(format!("events+: {}", extra.join(" ")));
        }
    }

    if let Some(ref ss) = s.schedstat_deltas {
        // Per-second rates derived inline from the counters / the schedstat
        // window. The cross-run-foldable forms live in the metric registry as
        // *_per_sec Rates (Σnum/Σden); this single-run display computes the same
        // quotient. 0.0 on a degenerate (single-sample) window.
        let per_sec = |n: u64| {
            if ss.total_schedstat_wall_sec > 0.0 {
                n as f64 / ss.total_schedstat_wall_sec
            } else {
                0.0
            }
        };
        lines.push(format!(
            "schedstat: csw={} ({:.0}/s) run_delay={:.0}ns/s ttwu={} goidle={}",
            ss.total_sched_count,
            per_sec(ss.total_sched_count),
            per_sec(ss.total_run_delay),
            ss.total_ttwu_count,
            ss.total_sched_goidle,
        ));
    }

    if let Some(ref progs) = s.prog_stats_deltas {
        for p in progs {
            if p.cnt > 0 {
                lines.push(format!(
                    "bpf: {} cnt={} {:.0}ns/call",
                    p.name, p.cnt, p.nsecs_per_call,
                ));
            }
        }
    }

    lines.push(format!("verdict: {verdict_line}"));

    format!("\n\n--- monitor ---\n{}", lines.join("\n"))
}

/// Number of monitor samples to skip at the start of evaluation.
///
/// During VM boot the kernel performs BPF verification, initramfs
/// unpacking, and scheduler loading. These memory-intensive operations
/// cause the scheduler tick to stall for hundreds of milliseconds.
/// The stalls are real but transient — evaluating them produces false
/// positives, especially in low-memory VMs.
///
/// 20 samples at ~100ms interval = ~2 seconds of warmup. This covers
/// the boot settling period after the scheduler attaches.
const MONITOR_WARMUP_SAMPLES: usize = 20;

/// Skip boot-settle samples from a MonitorReport for threshold evaluation.
///
/// Returns a report with the first `MONITOR_WARMUP_SAMPLES` removed so
/// that transient boot-time stalls don't trigger sustained-window
/// violations.
pub(crate) fn trim_settle_samples(
    report: &crate::monitor::MonitorReport,
) -> crate::monitor::MonitorReport {
    if report.samples.len() <= MONITOR_WARMUP_SAMPLES {
        return report.clone();
    }

    let trimmed = report.samples[MONITOR_WARMUP_SAMPLES..].to_vec();
    let summary = crate::monitor::MonitorSummary::from_samples_with_threshold(
        &trimmed,
        report.preemption_threshold_ns,
    );
    crate::monitor::MonitorReport {
        samples: trimmed,
        summary,
        preemption_threshold_ns: report.preemption_threshold_ns,
        watchdog_observation: report.watchdog_observation,
        page_offset: report.page_offset,
        boot_wait_outcome: report.boot_wait_outcome,
    }
}

/// Format a label for the scheduler spec, for use in test output.
///
/// Returns an empty string for `SchedulerSpec::Eevdf` so the failure
/// header reads `ktstr_test 'name' [topo=...]` with no sched
/// bracket — every other variant renders `" [sched=X]"` where `X`
/// comes from [`SchedulerSpec::display_name`].
pub(crate) fn scheduler_label(spec: &SchedulerSpec) -> String {
    if matches!(spec, SchedulerSpec::Eevdf) {
        String::new()
    } else {
        format!(" [sched={}]", spec.display_name())
    }
}
