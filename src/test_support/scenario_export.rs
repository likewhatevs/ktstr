//! Export every scenario a test binary registers, from any test binary.
//!
//! # The gap this exists to close
//!
//! [`KTSTR_SCENARIOS`](super::KTSTR_SCENARIOS) is a linkme distributed slice,
//! and a distributed slice is **per link unit**. Each integration test file
//! under `tests/` is its own binary, so each gets its OWN slice containing only
//! the scenarios declared in that file.
//!
//! The exporter used to live inside `tests/ktstr_sched_tests.rs` and iterate
//! that binary's slice. A scenario declared anywhere else was therefore
//! invisible to it: the conversion compiled, the test passed, and no record was
//! ever written. **It refused silently** — no error, no warning, no absent-file
//! diagnostic, just a scenario that never reached the second backend. That cost
//! a full round of work, in which eight conversions were planned against
//! candidates every one of which lives in a different binary.
//!
//! # The fix, and why this one
//!
//! The export loop moved here so ANY binary can run it, and each binary that
//! declares scenarios calls it. Three alternatives were weighed:
//!
//! * *Move the exporter to a binary that sees everything* — there is no such
//!   binary. That is the whole problem.
//! * *Aggregate the slices across binaries* — linkme cannot; the slices are
//!   resolved at link time and the binaries never share an address space. A
//!   runner-level merge in `cargo-ktstr` could collect records afterwards, but
//!   it would not make a MISSING one visible, which is the actual defect.
//! * *Replace linkme with a runtime registry* — same per-binary limitation,
//!   plus it would lose the registration-at-definition-site property that keeps
//!   the enumeration exact.
//!
//! # Making the failure loud
//!
//! Moving the loop is only half of it. A helper nobody calls is exactly as
//! silent as the bug. So `every_scenario_binary_exports` in
//! `tests/ktstr_sched_tests.rs` reads the test sources and fails if any file
//! declares a `#[ktstr_scenario]` without also calling this function. That
//! check is what converts "your scenario quietly never exported" into a named,
//! actionable failure, and it keeps working for a binary that does not exist
//! yet.

use std::path::{Path, PathBuf};

/// The environment variable that opts a run into exporting.
pub const EXPORT_DIR_ENV: &str = "KTSTR_SCENARIO_EXPORT_DIR";

/// The `default_workers_per_cgroup` stamped into every exported record.
///
/// ONE constant, shared by every exporting binary, deliberately. Each binary
/// hardcoding its own would give the suite several independently-maintained
/// copies of a number the simulator treats as authoritative -- and there is
/// already one such coupling worth removing rather than multiplying.
///
/// THE COUPLING, stated so it is not rediscovered: once a scenario uses
/// `CgroupDef::named` rather than `ctx.cgroup_def`, the two backends resolve a
/// worker count from DIFFERENT places. The VM resolves an unset `num_workers`
/// through `resolve_num_workers(work, ctx.workers_per_cgroup, ..)`, following
/// `Ctx`. The record carries `workers: null`, and the simulator binds it to
/// THIS value. They agree at 1 and diverge silently at anything else, because
/// the cross-backend check compares CPU *shares* -- two runs at different
/// worker counts still agree. Keep this equal to `Ctx::builder`'s
/// `workers_per_cgroup` default.
pub const DEFAULT_WORKERS_PER_CGROUP: u32 = 1;

/// What one call to [`export_registered_scenarios`] did.
#[derive(Debug)]
pub struct ExportOutcome {
    /// Records written this call.
    pub written: Vec<String>,
    /// Where they went.
    pub dir: PathBuf,
    /// Total gaps reported across all records. Non-zero is not a failure — a
    /// gap is the exporter naming something it could not carry — but a caller
    /// that expected a faithful record should look.
    pub gaps: usize,
}

/// Export every scenario registered **in the calling binary**.
///
/// Returns `None` when [`EXPORT_DIR_ENV`] is unset, which is the normal case:
/// an ordinary test run must not write files. Returns `Some` with what was
/// written otherwise.
///
/// `workers_per_cgroup` is the value stamped into each record's
/// `default_workers_per_cgroup`. See the caller in `ktstr_sched_tests.rs` for
/// why that number is load-bearing and what it is coupled to.
///
/// # Panics
///
/// Panics if a registered scenario has no paired `KtstrTestEntry`, if the
/// directory cannot be created, or if a record cannot be serialised or
/// written. All four are bugs rather than conditions, and a silent skip is
/// what this module exists to eliminate.
#[must_use]
pub fn export_registered_scenarios(workers_per_cgroup: u32) -> Option<ExportOutcome> {
    let dir = PathBuf::from(std::env::var_os(EXPORT_DIR_ENV)?);
    std::fs::create_dir_all(&dir).expect("create export dir");

    let mut written = Vec::new();
    let mut gaps = 0usize;

    for scenario in super::KTSTR_SCENARIOS {
        let entry = super::find_test(scenario.name)
            .unwrap_or_else(|| panic!("{} has no KtstrTestEntry", scenario.name));
        let def = (scenario.build)();
        let out = crate::scenario::export::export_scenario(
            scenario.name,
            &def,
            &entry.topology,
            entry.duration,
            workers_per_cgroup,
        );

        // Gaps are printed, never silently dropped: a record the simulator
        // consumes must not omit part of the workload without saying so.
        for gap in &out.gaps {
            println!(
                "EXPORT GAP {}: {} at {} — {}",
                scenario.name, gap.construct, gap.where_, gap.reason,
            );
        }
        gaps += out.gaps.len();

        let path = dir.join(format!("{}.json", scenario.name));
        // A name collision across two test binaries would silently overwrite,
        // and the records are keyed only by scenario name. Refuse instead --
        // this is the same silent-overwrite shape the module exists to remove.
        assert!(
            !path.exists(),
            "{} already exists: two test binaries registered a scenario named \
             {:?}, and the record would be overwritten. Scenario names must be \
             unique across the whole test suite, not just within one binary.",
            path.display(),
            scenario.name,
        );
        std::fs::write(
            &path,
            serde_json::to_vec_pretty(&out.record).expect("serialize"),
        )
        .unwrap_or_else(|e| panic!("write {}: {e}", path.display()));

        println!(
            "exported {} -> {} ({} gap(s))",
            scenario.name,
            path.display(),
            out.gaps.len(),
        );
        written.push(scenario.name.to_string());
    }

    Some(ExportOutcome { written, dir, gaps })
}

/// Every `tests/*.rs` that declares a scenario, and whether it also exports.
///
/// Source-level on purpose. The property is "this binary calls the exporter",
/// and a binary that does not cannot report on itself — the absent call is
/// precisely what there is no code to run. Reading the sources is the only
/// vantage point from which a missing caller is visible at all.
#[must_use]
pub fn scenario_binaries_missing_export(tests_dir: &Path) -> Vec<String> {
    let mut missing = Vec::new();
    let Ok(entries) = std::fs::read_dir(tests_dir) else {
        return missing;
    };
    for e in entries.flatten() {
        let p = e.path();
        if p.extension().is_some_and(|x| x == "rs") {
            let Ok(src) = std::fs::read_to_string(&p) else {
                continue;
            };
            let declares = src.contains("#[ktstr_scenario");
            let exports = src.contains("export_registered_scenarios");
            if declares && !exports {
                missing.push(
                    p.file_name()
                        .map(|f| f.to_string_lossy().to_string())
                        .unwrap_or_default(),
                );
            }
        }
    }
    missing.sort();
    missing
}
