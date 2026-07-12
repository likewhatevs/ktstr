use super::*;

fn make_row(scenario: &str, topo: &str, passed: bool, spread: f64) -> GauntletRow {
    GauntletRow {
        scenario: scenario.into(),
        perf_delta_assertions: Vec::new(),
        topology: topo.into(),
        work_type: "SpinWait".into(),
        scheduler: String::new(),
        kernel_version: None,
        cpu_budget: None,
        vcpus: None,
        throughput_denomination: Default::default(),
        commit: None,
        kernel_commit: None,
        run_source: None,
        resolve_source: None,
        skipped: false,
        passed,
        inconclusive: false,
        expected_failure: false,
        run_sample_count: 0,
        spread,
        gap_ms: 50,
        migrations: 10,
        migration_ratio: 0.0,
        imbalance_ratio: 1.0,
        max_dsq_depth: 2,
        stuck_count: 0.0,
        fallback_count: 0,
        keep_last_count: 0,
        total_iterations: 0,
        ext_metrics: BTreeMap::new(),
        ext_counter_keys: BTreeSet::new(),
        phases: Vec::new(),
    }
}

/// Build a [`crate::assert::PhaseBucket`] for a step with the given per-metric
/// values. Used by the per-phase noise tests (compare_core).
fn make_phase_bucket(
    step_index: u16,
    label: &str,
    metrics: &[(&str, f64)],
) -> crate::assert::PhaseBucket {
    let metrics_map = metrics.iter().map(|(k, v)| (k.to_string(), *v)).collect();
    crate::assert::PhaseBucket {
        per_cgroup: Default::default(),
        step_index,
        label: label.to_string(),
        start_ms: 0,
        end_ms: 100,
        sample_count: 1,
        metrics: metrics_map,
    }
}

/// Helper that builds a `GauntletRow` with controllable
/// scheduler / topology / work_type / kernel_version for the
/// filter tests. The metric fields default to harmless
/// passing values; tests are interested in identity-field
/// matching, not metrics.
fn make_filter_row(
    scenario: &str,
    scheduler: &str,
    topology: &str,
    work_type: &str,
    kernel_version: Option<&str>,
) -> GauntletRow {
    GauntletRow {
        scenario: scenario.into(),
        perf_delta_assertions: Vec::new(),
        topology: topology.into(),
        work_type: work_type.into(),
        scheduler: scheduler.into(),
        kernel_version: kernel_version.map(str::to_owned),
        cpu_budget: None,
        vcpus: None,
        throughput_denomination: Default::default(),
        commit: None,
        kernel_commit: None,
        run_source: None,
        resolve_source: None,
        passed: true,
        skipped: false,
        inconclusive: false,
        expected_failure: false,
        run_sample_count: 0,
        spread: 0.0,
        gap_ms: 0,
        migrations: 0,
        migration_ratio: 0.0,
        imbalance_ratio: 0.0,
        max_dsq_depth: 0,
        stuck_count: 0.0,
        fallback_count: 0,
        keep_last_count: 0,
        total_iterations: 0,
        ext_metrics: BTreeMap::new(),
        ext_counter_keys: BTreeSet::new(),
        phases: Vec::new(),
    }
}

mod aggregation_and_analyze;
mod compare_core;
mod compare_phase;
mod dims_pairing_runs;
mod dynamic_counter_fold;
mod rowfilter_and_group_avg;
mod sidecar_and_metric_def;
