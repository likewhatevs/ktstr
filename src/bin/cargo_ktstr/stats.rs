//! `cargo ktstr stats` subcommand dispatch helpers.
//!
//! Houses the dispatcher [`run_stats`] for the read-only stats
//! inspection surface (list, list-metrics, list-values, show-host,
//! explain-sidecar) and the [`BuildCompareFilters`] sugar resolver
//! that folds shared `--X` and per-side `--{a,b}-X` version-axis
//! filter flags into the two `RowFilter` instances
//! `cli::compare_partitions` consumes for `perf-delta`.

use crate::cli::StatsCommand;
use ktstr::cli;

pub(crate) fn run_stats(command: &Option<StatsCommand>) -> Result<(), String> {
    match command {
        None => {
            if let Some(output) = cli::print_stats_report() {
                print!("{output}");
            }
            Ok(())
        }
        Some(StatsCommand::List) => cli::list_runs().map_err(|e| format!("{e:#}")),
        Some(StatsCommand::ListMetrics { json }) => match cli::list_metrics(*json) {
            Ok(s) => {
                print!("{s}");
                Ok(())
            }
            Err(e) => Err(format!("{e:#}")),
        },
        Some(StatsCommand::ListValues { json, dir }) => {
            match cli::list_values(*json, dir.as_deref()) {
                Ok(s) => {
                    print!("{s}");
                    Ok(())
                }
                Err(e) => Err(format!("{e:#}")),
            }
        }
        Some(StatsCommand::ShowHost { run, dir }) => {
            match cli::show_run_host(run, dir.as_deref()) {
                Ok(s) => {
                    print!("{s}");
                    Ok(())
                }
                Err(e) => Err(format!("{e:#}")),
            }
        }
        Some(StatsCommand::ExplainSidecar { run, dir, json }) => {
            match cli::explain_sidecar(run, dir.as_deref(), *json) {
                Ok(s) => {
                    print!("{s}");
                    Ok(())
                }
                Err(e) => Err(format!("{e:#}")),
            }
        }
    }
}

/// Resolve shared / per-side commit-axis filter values into the two
/// `RowFilter` instances `perf-delta` compares.
///
/// Only the SLICEABLE version axes (kernel / project-commit / kernel-commit)
/// carry a per-side `a_`/`b_` override that forms an A/B CONTRAST; every other
/// dimension is FILTER + PAIRING only (see `Dimension::SLICEABLE`) and pins
/// BOTH sides to the same shared value. "More-specific replaces": a per-side
/// Vec is applied verbatim when non-empty, otherwise the shared Vec is used.
///
/// Commit fields hold ALREADY-EXPANDED commit lists — any git revspecs
/// (HEAD~1, tags, branches, `A..B` ranges) are resolved to 7-char short hashes
/// BEFORE construction — so `build()` stays a pure data-shuffler with no repo
/// I/O, unit-testable in isolation.
#[derive(Debug, Clone, Default)]
pub(crate) struct BuildCompareFilters {
    pub(crate) shared_kernel: Vec<String>,
    pub(crate) shared_project_commit: Vec<String>,
    pub(crate) shared_kernel_commit: Vec<String>,
    pub(crate) shared_run_source: Vec<String>,
    pub(crate) shared_resolve_source: Vec<String>,
    pub(crate) shared_cpu_budget: Vec<String>,
    pub(crate) shared_scheduler: Vec<String>,
    pub(crate) shared_topology: Vec<String>,
    pub(crate) shared_work_type: Vec<String>,
    // Per-side A/B overrides exist ONLY for the sliceable version axes.
    pub(crate) a_kernel: Vec<String>,
    pub(crate) a_project_commit: Vec<String>,
    pub(crate) a_kernel_commit: Vec<String>,
    pub(crate) b_kernel: Vec<String>,
    pub(crate) b_project_commit: Vec<String>,
    pub(crate) b_kernel_commit: Vec<String>,
}

impl BuildCompareFilters {
    /// Resolve into per-side `RowFilter` instances. A sliceable axis uses
    /// "more-specific replaces" (`pick_vec`: per-side Vec when non-empty, else
    /// shared); a non-sliceable dimension pins BOTH sides to its shared Vec, so
    /// A and B never differ on it (it is filter + pairing only).
    pub(crate) fn build(&self) -> (ktstr::cli::RowFilter, ktstr::cli::RowFilter) {
        let pick_vec = |a: &[String], shared: &[String]| -> Vec<String> {
            if a.is_empty() {
                shared.to_vec()
            } else {
                a.to_vec()
            }
        };
        let filter_a = ktstr::cli::RowFilter {
            kernels: pick_vec(&self.a_kernel, &self.shared_kernel),
            project_commits: pick_vec(&self.a_project_commit, &self.shared_project_commit),
            kernel_commits: pick_vec(&self.a_kernel_commit, &self.shared_kernel_commit),
            run_sources: self.shared_run_source.clone(),
            resolve_sources: self.shared_resolve_source.clone(),
            cpu_budgets: self.shared_cpu_budget.clone(),
            schedulers: self.shared_scheduler.clone(),
            topologies: self.shared_topology.clone(),
            work_types: self.shared_work_type.clone(),
        };
        let filter_b = ktstr::cli::RowFilter {
            kernels: pick_vec(&self.b_kernel, &self.shared_kernel),
            project_commits: pick_vec(&self.b_project_commit, &self.shared_project_commit),
            kernel_commits: pick_vec(&self.b_kernel_commit, &self.shared_kernel_commit),
            run_sources: self.shared_run_source.clone(),
            resolve_sources: self.shared_resolve_source.clone(),
            cpu_budgets: self.shared_cpu_budget.clone(),
            schedulers: self.shared_scheduler.clone(),
            topologies: self.shared_topology.clone(),
            work_types: self.shared_work_type.clone(),
        };
        (filter_a, filter_b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -- BuildCompareFilters: symmetric sugar resolution --

    /// Empty input → both sides default. No filters populated
    /// anywhere; the dispatch site rejects this with the
    /// "specify at least one --a-X" error, but the builder
    /// itself just returns two empty filters.
    #[test]
    fn build_compare_filters_empty_yields_default_default() {
        let b = BuildCompareFilters::default();
        let (fa, fb) = b.build();
        assert!(fa.kernels.is_empty());
        assert!(fa.project_commits.is_empty());
        assert!(fa.kernel_commits.is_empty());
        assert!(fa.run_sources.is_empty());
        assert!(fa.schedulers.is_empty());
        assert!(fa.topologies.is_empty());
        assert!(fa.work_types.is_empty());
        assert_eq!(fa.kernels, fb.kernels);
        assert_eq!(fa.project_commits, fb.project_commits);
        assert_eq!(fa.kernel_commits, fb.kernel_commits);
        assert_eq!(fa.run_sources, fb.run_sources);
        assert_eq!(fa.schedulers, fb.schedulers);
        assert_eq!(fa.topologies, fb.topologies);
        assert_eq!(fa.work_types, fb.work_types);
    }

    /// Per-side `--a-kernel-commit` overrides shared
    /// `--kernel-commit` for A only; B retains the shared value.
    /// Same "more-specific replaces" semantics as `--a-kernel`.
    /// The per-side override path is what populates the slicing
    /// dim on `KernelCommit` — without it, two sides with
    /// different live kernel HEADs cannot be contrasted in one
    /// `compare` invocation.
    #[test]
    fn build_compare_filters_per_side_kernel_commit_overrides_shared() {
        let b = BuildCompareFilters {
            shared_kernel_commit: vec!["abcdef1".to_string(), "fedcba2".to_string()],
            a_kernel_commit: vec!["111aaaa".to_string()],
            ..BuildCompareFilters::default()
        };
        let (fa, fb) = b.build();
        assert_eq!(
            fa.kernel_commits,
            vec!["111aaaa"],
            "A overrides shared kernel-commit",
        );
        assert_eq!(
            fb.kernel_commits,
            vec!["abcdef1", "fedcba2"],
            "B retains shared kernel-commit default",
        );
    }

    /// `--a-kernel-commit X --b-kernel-commit Y` slices on the
    /// `KernelCommit` dimension. Pins the slicing-dim derivation
    /// for the kernel-commit axis so a regression that dropped
    /// the dim from `derive_slicing_dims` lands here.
    #[test]
    fn build_compare_filters_disjoint_per_side_kernel_commit_slices() {
        let b = BuildCompareFilters {
            a_kernel_commit: vec!["abcdef1".to_string()],
            b_kernel_commit: vec!["fedcba2".to_string()],
            ..BuildCompareFilters::default()
        };
        let (fa, fb) = b.build();
        assert_eq!(fa.kernel_commits, vec!["abcdef1"]);
        assert_eq!(fb.kernel_commits, vec!["fedcba2"]);
        let slicing = ktstr::cli::derive_slicing_dims(&fa, &fb);
        assert_eq!(
            slicing,
            vec![ktstr::cli::Dimension::KernelCommit],
            "differing per-side kernel-commit must derive as a single \
             KernelCommit slicing dim",
        );
    }


    /// Shared `--kernel V` pins BOTH sides to the same vec.
    /// Sugar for `--a-kernel V --b-kernel V`.
    #[test]
    fn build_compare_filters_shared_kernel_pins_both_sides() {
        let b = BuildCompareFilters {
            shared_kernel: vec!["6.14".to_string()],
            ..BuildCompareFilters::default()
        };
        let (fa, fb) = b.build();
        assert_eq!(fa.kernels, vec!["6.14"]);
        assert_eq!(fb.kernels, vec!["6.14"]);
    }

    /// Per-side `--a-kernel` overrides shared `--kernel` for A
    /// only; B retains the shared value. "More-specific
    /// replaces" semantics.
    #[test]
    fn build_compare_filters_per_side_overrides_shared_for_that_side_only() {
        let b = BuildCompareFilters {
            shared_kernel: vec!["6.14".to_string(), "6.15".to_string()],
            a_kernel: vec!["6.13".to_string()],
            ..BuildCompareFilters::default()
        };
        let (fa, fb) = b.build();
        assert_eq!(fa.kernels, vec!["6.13"], "A overrides shared");
        assert_eq!(fb.kernels, vec!["6.14", "6.15"], "B retains shared default",);
    }

    /// Per-side overrides on the SAME dimension on BOTH sides
    /// produce the disjoint per-side filters the dispatch
    /// expects. This is the typical "slice on kernel" call shape:
    /// `--a-kernel A --b-kernel B`.
    #[test]
    fn build_compare_filters_disjoint_per_side_kernel_yields_two_filters() {
        let b = BuildCompareFilters {
            a_kernel: vec!["6.14".to_string()],
            b_kernel: vec!["6.15".to_string()],
            ..BuildCompareFilters::default()
        };
        let (fa, fb) = b.build();
        assert_eq!(fa.kernels, vec!["6.14"]);
        assert_eq!(fb.kernels, vec!["6.15"]);
    }


    /// Sibling of `build_compare_filters_empty_yields_default_default`
    /// for the `run_sources` field. The existing empty-default test
    /// asserts on `run_sources` already — this companion adds the
    /// cross-side equality check that ensures
    /// `fa.run_sources == fb.run_sources` under the empty default,
    /// matching the same pattern other dimensions have. A regression
    /// that diverged the per-side `run_sources` defaults (e.g. by
    /// forgetting to thread `shared_run_source` into BOTH
    /// constructors in `BuildCompareFilters::build`) would surface
    /// here.
    #[test]
    fn build_compare_filters_empty_run_sources_field_equal_on_both_sides() {
        let b = BuildCompareFilters::default();
        let (fa, fb) = b.build();
        assert!(
            fa.run_sources.is_empty(),
            "empty BuildCompareFilters must produce A-side filter with empty run_sources",
        );
        assert!(
            fb.run_sources.is_empty(),
            "empty BuildCompareFilters must produce B-side filter with empty run_sources",
        );
        assert_eq!(
            fa.run_sources, fb.run_sources,
            "both sides must agree on empty run_sources",
        );
    }


    /// Shared `--run-source` pins BOTH sides to the same vec (run-source is
    /// filter + pairing only — no per-side override). Mirrors
    /// `build_compare_filters_shared_kernel_pins_both_sides` for
    /// the run-source dimension.
    #[test]
    fn build_compare_filters_shared_source_pins_both_sides() {
        let b = BuildCompareFilters {
            shared_run_source: vec!["ci".to_string()],
            ..BuildCompareFilters::default()
        };
        let (fa, fb) = b.build();
        assert_eq!(fa.run_sources, vec!["ci".to_string()]);
        assert_eq!(fb.run_sources, vec!["ci".to_string()]);
    }


}
