use super::*;

/// Detected outlier helper: extract one numeric metric off a
/// [`GauntletRow`]. The same accessor is used to compute the overall
/// (cohort-wide) mean/std and the per-scenario mean — keeping the
/// "what counts as `imbalance`?" decision in one place avoids
/// drift between the two passes that polars-side handled implicitly
/// via the `df!` column name.
pub(crate) type MetricAccessor = fn(&GauntletRow) -> f64;

/// Pinned list of `(display_name, accessor)` for every metric that
/// outlier detection considers. The display name appears in
/// [`Outlier`] output verbatim ("scenario: imbalance 4.5 ..."); the
/// accessor pulls the f64 value off a `GauntletRow`. Mirrors the
/// `metrics` slice the old polars code keyed off DataFrame column
/// names, so the outlier set surfaces the same metrics under the same
/// names.
pub(crate) const OUTLIER_METRICS: &[(&str, MetricAccessor)] = &[
    ("spread", |r| r.spread),
    ("gap_ms", |r| r.gap_ms as f64),
    ("migrations", |r| r.migrations as f64),
    ("migration_ratio", |r| r.migration_ratio),
    ("imbalance", |r| r.imbalance_ratio),
    ("dsq_depth", |r| r.max_dsq_depth as f64),
    ("stuck", |r| r.stuck_count),
    ("fallback", |r| r.fallback_count as f64),
    ("keep_last", |r| r.keep_last_count as f64),
    // Distribution-kind roll-ups are ext_metrics-sourced (no typed field):
    // read them through the ext map, 0.0 when absent (the prior typed-field
    // default), mirroring the deleted `worst_*` accessors. The 0.0-on-absent
    // here is INTENTIONALLY distinct from the cross-RUN `group_and_average_by`
    // fold (via `aggregate_finite`'s Distribution arm), which EXCLUDES an absent
    // key from the mean (no-false-zero). The key is present whenever any cgroup
    // or carrier contributed; the 0.0-on-absent path is still reachable for a
    // telemetry-free pass (no phases AND no cgroups, e.g. host_only). A 0.0
    // here not only escapes being flagged as its own scenario's outlier — it
    // also ENTERS the cross-scenario overall_mean/overall_std baseline
    // `find_outliers` builds over all is_pass() rows, so it can shift the
    // 2-sigma threshold for the whole cohort. Both effects are benign ONLY
    // because every OUTLIER_METRICS Distribution entry is LowerBetter (a 0.0
    // reads as best, never the high tail outlier detection targets, and a
    // telemetry-free row carries no measured signal to begin with) — a coupling
    // the Distribution=>LowerBetter registry gate enforces. A HigherBetter ext
    // metric added here would NOT be benign (a 0.0 would depress the baseline
    // AND could itself read as a low outlier). So the two consumers diverge by
    // design, not by accident.
    ("worst_p99_wake_latency_us", |r| {
        r.ext_metrics
            .get("worst_p99_wake_latency_us")
            .copied()
            .unwrap_or(0.0)
    }),
    ("worst_wake_latency_cv", |r| {
        r.ext_metrics
            .get("worst_wake_latency_cv")
            .copied()
            .unwrap_or(0.0)
    }),
    ("worst_mean_run_delay_us", |r| {
        r.ext_metrics
            .get("worst_mean_run_delay_us")
            .copied()
            .unwrap_or(0.0)
    }),
    ("worst_run_delay_us", |r| {
        r.ext_metrics
            .get("worst_run_delay_us")
            .copied()
            .unwrap_or(0.0)
    }),
];

/// Arithmetic mean over the finite values produced by `iter`.
/// Non-finite values (NaN, ±inf) are excluded so a single outlier
/// or sentinel can't poison the mean. Returns 0.0 on an empty
/// (post-filter) input — matches what polars's `.mean()` does on a
/// chunked array of length zero.
pub(crate) fn mean<I: Iterator<Item = f64>>(iter: I) -> f64 {
    let (sum, count) = iter
        .filter(|x| x.is_finite())
        .fold((0.0_f64, 0usize), |(s, c), x| (s + x, c + 1));
    if count == 0 { 0.0 } else { sum / count as f64 }
}

/// Sample standard deviation (Bessel-corrected, ddof = 1) over the
/// finite values produced by `iter`. Returns 0.0 when fewer than two
/// finite values remain — matches polars's `.std(1)` semantics on a
/// 0- or 1-element chunked array. Requires `Iterator + Clone` because
/// the computation needs two passes (mean, then squared deviations).
pub(crate) fn std_dev<I: Iterator<Item = f64> + Clone>(iter: I) -> f64 {
    let m = mean(iter.clone());
    let (sum_sq, count) = iter
        .filter(|x| x.is_finite())
        .fold((0.0_f64, 0usize), |(s, c), x| {
            let d = x - m;
            (s + d * d, c + 1)
        });
    if count < 2 {
        0.0
    } else {
        (sum_sq / (count - 1) as f64).sqrt()
    }
}

/// Extract a grouping dimension's `&str` field off a [`GauntletRow`].
/// Replaces the polars `col(group_col)` lookup with a fn-pointer
/// dispatch over the three accepted dimension names. Returns `None`
/// for any other column name — `analyze_rows` and
/// `format_dimension_summary` both restrict the dimension to one of
/// the three documented columns, so the `None` arm is unreachable in
/// production but kept as defense-in-depth against a stray call site.
pub(crate) fn group_field<'a>(row: &'a GauntletRow, col: &str) -> Option<&'a str> {
    match col {
        "scenario" => Some(row.scenario.as_str()),
        "topology" => Some(row.topology.as_str()),
        "work_type" => Some(row.work_type.as_str()),
        _ => None,
    }
}

/// Detected outlier: a scenario with an anomalous stat.
pub(crate) struct Outlier {
    pub(crate) scenario: String,
    pub(crate) metric: &'static str,
    pub(crate) value: f64,
    pub(crate) overall_mean: f64,
    pub(crate) sigma: f64,
    pub(crate) worst_topos: Vec<String>,
}

impl std::fmt::Display for Outlier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}: {} {:.1} (overall avg {:.1}, +{:.1}\u{03c3})",
            self.scenario, self.metric, self.value, self.overall_mean, self.sigma
        )?;
        if !self.worst_topos.is_empty() {
            write!(f, "\n    worst on: {}", self.worst_topos.join(", "))?;
        }
        Ok(())
    }
}

// `col_f64`, `col_u32`, `col_str`, `col_mean_std`: removed alongside
// the polars dep. The aggregation paths now read metrics directly off
// `&GauntletRow` via the [`MetricAccessor`] / [`group_field`] dispatch
// at the top of this section, and per-iterator mean / std go through
// the standalone [`mean`] / [`std_dev`] helpers.

/// Find outlier scenarios where a metric exceeds 2 sigma.
///
/// For every metric in [`OUTLIER_METRICS`]: compute the
/// cohort-wide (overall_mean, overall_std), set the threshold at
/// `overall_mean + 2 * overall_std`, then check every per-scenario
/// mean against that threshold. Scenarios whose mean exceeds the
/// threshold get an [`Outlier`] entry annotated with the topology
/// rows (via [`find_worst_topos`]) that drove the excursion.
///
/// 4-state lattice filtering: only real-pass rows
/// (`passed && !skipped && !inconclusive`, matching
/// `GauntletRow::is_pass`) contribute to the per-scenario mean
/// AND the overall mean/std baseline. Skipped / inconclusive /
/// failed rows carry default-zero metric values (for typed fields
/// `sidecar_to_row` substitutes zero for non-finite + missing
/// fields; for the ext-sourced Distribution entries the
/// `OUTLIER_METRICS` accessor's own `.unwrap_or(0.0)` supplies the
/// zero, since `sidecar_to_row` copies only present finite ext keys
/// and never zero-fills a missing one; see the block comment on
/// those entries), and
/// including them would silently depress every measured mean — a
/// scenario with 1 real-pass run (value=100) and 9 inconclusive
/// runs (value=0) would otherwise report a per-scenario mean of
/// 10.0 and the same pollution would deflate the cohort-wide
/// baseline used for the 2-sigma threshold. Filtering on
/// `is_pass` matches the same defense-in-depth as
/// `format_dimension_summary`'s pass_count + the
/// `compare_rows_by` regression-math gate.
pub(crate) fn find_outliers(rows: &[GauntletRow]) -> Vec<Outlier> {
    let pass_rows: Vec<&GauntletRow> = rows.iter().filter(|r| r.is_pass()).collect();
    if pass_rows.is_empty() {
        return Vec::new();
    }

    // Bucket pass rows by scenario name. BTreeMap iterates in sorted
    // order so the outlier vector before the final sigma-sort is
    // already alphabetic per (metric, scenario) — deterministic across
    // runs even when multiple scenarios tie on sigma. Borrowing `&str`
    // out of the row avoids cloning the scenario names just to bucket.
    let mut by_scenario: BTreeMap<&str, Vec<&GauntletRow>> = BTreeMap::new();
    for r in &pass_rows {
        by_scenario.entry(r.scenario.as_str()).or_default().push(r);
    }

    let mut outliers = Vec::new();
    for &(name, accessor) in OUTLIER_METRICS {
        let overall_mean = mean(pass_rows.iter().map(|r| accessor(r)));
        let overall_std = std_dev(pass_rows.iter().map(|r| accessor(r)));
        // Drop metrics with std below epsilon. The cohort produced no
        // measurable spread on this metric, so flagging "outliers"
        // against a near-zero baseline would surface noise. Mirrors the
        // pre-polars `active.filter_map` short-circuit.
        if overall_std < f64::EPSILON {
            continue;
        }
        let threshold = overall_mean + 2.0 * overall_std;

        for (&scenario, rows_in_scenario) in &by_scenario {
            let scenario_mean = mean(rows_in_scenario.iter().map(|r| accessor(r)));
            if scenario_mean <= threshold {
                continue;
            }
            let sigma = (scenario_mean - overall_mean) / overall_std;
            // Worst topologies are computed against the full row set
            // (not the pass-only subset) so a failure cluster on a
            // particular topology still surfaces even when its rows
            // failed the is_pass gate — the outlier line is the place
            // where the operator first sees that topology, and gating
            // it on is_pass would hide the worst offenders.
            let worst = find_worst_topos(rows, scenario, accessor, threshold);
            outliers.push(Outlier {
                scenario: scenario.to_string(),
                metric: name,
                value: scenario_mean,
                overall_mean,
                sigma,
                worst_topos: worst,
            });
        }
    }

    outliers.sort_by(|a, b| {
        b.sigma
            .partial_cmp(&a.sigma)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    outliers
}

/// Topology names of rows in `scenario` whose metric value exceeds
/// `threshold`. Used by [`find_outliers`] to attribute a per-scenario
/// outlier to the specific topologies driving the excursion.
pub(crate) fn find_worst_topos(
    rows: &[GauntletRow],
    scenario: &str,
    accessor: MetricAccessor,
    threshold: f64,
) -> Vec<String> {
    // `filter` passes `&Self::Item`, so inside the closure `r` is
    // `&&GauntletRow`. Destructure with `|&r|` so `r: &GauntletRow`
    // and `accessor(r)` matches its `fn(&GauntletRow) -> f64`
    // signature without an explicit deref.
    rows.iter()
        .filter(|&r| r.scenario == scenario && accessor(r) > threshold)
        .map(|r| r.topology.clone())
        .collect()
}

/// Format a group-by summary for one dimension (`scenario`,
/// `topology`, or `work_type`). For each value of the dimension,
/// renders one line carrying:
///
/// `{name:<25} {pass}/{total} passed ({skip} skipped, {inconc} inconclusive, {fail} failed)
///  avg_spread={spread:.1}%  avg_gap={gap:.0}ms[  imbal=…][  dsq=…][  stuck=…][  fallback=…]`
///
/// Sorted by `avg_spread` descending so the worst dimension values
/// land at the top. Pass / skip / inconclusive / fail follow the
/// 4-state mutex documented in detail on the prior polars-side
/// pass_count aggregation: `fail` is the residual after subtracting
/// pass / skip / inconc from total, NOT a separate count. Splitting
/// out `inconclusive` from `failed` is what keeps a zero-denominator
/// inconclusive run from silently rendering as "failed."
///
/// `imbal` / `dsq` / `stuck` / `fallback` tail tokens render only
/// when their value crosses a per-metric threshold (`> 1.0` for
/// imbalance, `> 0.0` for the others) so a healthy dimension's line
/// stays terse.
///
/// Returns an empty string when `group_col` is not one of the three
/// accepted dimension names — matches the prior behavior of bailing
/// without panicking on a stray column name.
pub(crate) fn format_dimension_summary(rows: &[GauntletRow], group_col: &str) -> String {
    // Reject unknown dimension names up-front via `group_field`'s
    // None arm. The match exhausts at the first row — every row
    // resolves identically given the same `group_col` — so the
    // probe is O(1).
    if rows.is_empty()
        || rows
            .first()
            .and_then(|r| group_field(r, group_col))
            .is_none()
    {
        return String::new();
    }

    let mut by_dim: BTreeMap<&str, Vec<&GauntletRow>> = BTreeMap::new();
    for r in rows {
        if let Some(key) = group_field(r, group_col) {
            by_dim.entry(key).or_default().push(r);
        }
    }

    struct GroupStats<'a> {
        name: &'a str,
        pass_count: usize,
        skip_count: usize,
        inconc_count: usize,
        total: usize,
        avg_spread: f64,
        avg_gap_ms: f64,
        avg_imbalance: f64,
        avg_dsq_depth: f64,
        /// Sum of per-run `stuck_count` across the group — total stall
        /// observations, NOT the number of runs that stalled. Now that
        /// `stuck_count` is a true per-run count, a single run can
        /// contribute more than 1.
        total_stuck: f64,
        avg_fallback: f64,
    }

    let mut groups: Vec<GroupStats> = by_dim
        .iter()
        .map(|(name, group_rows)| GroupStats {
            name,
            pass_count: group_rows.iter().filter(|r| r.is_pass()).count(),
            skip_count: group_rows.iter().filter(|r| r.is_skip()).count(),
            inconc_count: group_rows.iter().filter(|r| r.is_inconclusive()).count(),
            total: group_rows.len(),
            avg_spread: mean(group_rows.iter().map(|r| r.spread)),
            avg_gap_ms: mean(group_rows.iter().map(|r| r.gap_ms as f64)),
            avg_imbalance: mean(group_rows.iter().map(|r| r.imbalance_ratio)),
            avg_dsq_depth: mean(group_rows.iter().map(|r| r.max_dsq_depth as f64)),
            total_stuck: group_rows
                .iter()
                .map(|r| r.stuck_count)
                .filter(|x| x.is_finite())
                .sum(),
            avg_fallback: mean(group_rows.iter().map(|r| r.fallback_count as f64)),
        })
        .collect();

    // Descending sort by avg_spread so the dimension value with the
    // worst spread reads first. Tie-breaker is the input order from
    // BTreeMap (alphabetic) — same deterministic shape as the polars
    // `sort([avg_spread], descending=true)` form.
    groups.sort_by(|a, b| {
        b.avg_spread
            .partial_cmp(&a.avg_spread)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut out = String::new();
    for g in &groups {
        let fail = g
            .total
            .saturating_sub(g.pass_count)
            .saturating_sub(g.skip_count)
            .saturating_sub(g.inconc_count);
        let mut line = format!(
            "  {:<25} {}/{} passed ({} skipped, {} inconclusive, {} failed)  avg_spread={:.1}%  avg_gap={:.0}ms",
            g.name,
            g.pass_count,
            g.total,
            g.skip_count,
            g.inconc_count,
            fail,
            g.avg_spread,
            g.avg_gap_ms,
        );
        if g.avg_imbalance > 1.0 {
            line.push_str(&format!("  imbal={:.1}", g.avg_imbalance));
        }
        if g.avg_dsq_depth > 0.0 {
            line.push_str(&format!("  dsq={:.0}", g.avg_dsq_depth));
        }
        if g.total_stuck > 0.0 {
            line.push_str(&format!("  stuck={}", g.total_stuck as u64));
        }
        if g.avg_fallback > 0.0 {
            line.push_str(&format!("  fallback={:.0}", g.avg_fallback));
        }
        line.push('\n');
        out.push_str(&line);
    }
    out
}

/// Analyze pre-built gauntlet rows and return a formatted report.
pub fn analyze_rows(rows: &[GauntletRow]) -> String {
    if rows.is_empty() {
        return String::new();
    }

    let mut report = String::from("\n=== GAUNTLET ANALYSIS ===\n\n");

    let outliers = find_outliers(rows);
    if outliers.is_empty() {
        report.push_str("No outliers detected.\n");
    } else {
        report.push_str("Outliers detected:\n");
        for o in &outliers {
            report.push_str(&format!("  {o}\n"));
        }
    }

    report.push_str("\nBy scenario (worst first):\n");
    report.push_str(&format_dimension_summary(rows, "scenario"));

    report.push_str("\nBy topology:\n");
    report.push_str(&format_dimension_summary(rows, "topology"));

    // Surface a "By work_type" pane only when the input carries
    // more than one work_type value; a single-work_type cohort
    // would render an identical "all rows" pane to the scenario
    // pane and add visual noise.
    let work_types: std::collections::BTreeSet<&str> =
        rows.iter().map(|r| r.work_type.as_str()).collect();
    if work_types.len() > 1 {
        report.push_str("\nBy work_type:\n");
        report.push_str(&format_dimension_summary(rows, "work_type"));
    }

    report
}

// ---------------------------------------------------------------------------
// Test-run enumeration and A/B comparison
// ---------------------------------------------------------------------------

/// List the test-run directories under
/// `{CARGO_TARGET_DIR or "target"}/ktstr/`.
///
/// Each subdirectory is one run keyed `{kernel}-{project_commit}`
/// where `{project_commit}` is the project HEAD short hex with
/// `-dirty` suffix when the worktree differs. Two runs sharing the
/// same key reuse the same directory: the second run pre-clears
/// prior `*.ktstr.json` files at first write so the directory is a
/// last-writer-wins snapshot of (kernel, project commit) rather
/// than an append-only archive of every invocation.
///
/// The rendered table carries four columns:
/// - `RUN`: the run-directory leaf name
///   (`{kernel}-{project_commit}` per the keying above).
/// - `TESTS`: number of `*.ktstr.json` sidecars in the directory
///   (and one level of subdirectories for per-job gauntlet
///   layouts that `collect_sidecars` walks).
/// - `DATE`: the earliest sidecar timestamp in the directory.
///   Under last-writer-wins this equals the most recent run's
///   first sidecar timestamp because the prior run's sidecars
///   were pre-cleared at the new run's first write.
/// - `ARCH`: the `host.arch` value (`x86_64`, `aarch64`, …) from
///   the run's first sidecar that carries a populated host
///   field. Renders as `-` when no sidecar carries a host
///   (pre-host-context-landing archives, host-only test stubs
///   that never populate host) so the column reads consistently
///   with the `DATE` sentinel.
///
/// Rows are sorted by directory mtime, **most recent first**, so
/// the latest run lands at the top of the table — the operator's
/// usual interest. Sorting by `file_name()` would produce
/// alphabetical-by-hex output (the `{project_commit}` half is a
/// hex short-hash with no temporal ordering), which scatters
/// chronologically-adjacent runs across the listing. `file_name`
/// is the tiebreaker on the secondary axis: pairs with equal
/// mtimes (or both unreadable) collapse onto a deterministic
/// alphabetical order so the listing stays stable across
/// re-invocations. Entries whose mtime cannot be read at all
/// (filesystem error, permission issue) sort to the END of the
/// listing — `Reverse(None) > Reverse(Some(_))` lands them after
/// every dated entry, with the file_name tiebreaker keeping the
/// undated group itself stable.
pub fn list_runs() -> anyhow::Result<()> {
    let root = crate::test_support::runs_root();
    // Both the missing-root case (target/ktstr/ never created) and
    // the empty-rows case (directory exists but no run
    // subdirectories pass `is_run_directory`) reduce to the same
    // operator state: no test data is on disk yet. Surface the
    // same actionable hint in both branches so a fresh checkout
    // sees consistent guidance regardless of whether
    // `target/ktstr/` was ever created.
    let hint = "Run `cargo ktstr test` to generate sidecar data.";
    if !root.exists() {
        eprintln!("no runs found at {}. {hint}", root.display());
        return Ok(());
    }
    let rows = sorted_run_entries(&root)?;
    if rows.is_empty() {
        eprintln!("no runs found at {}. {hint}", root.display());
        return Ok(());
    }
    let mut table = crate::cli::new_table();
    table.set_header(vec!["RUN", "TESTS", "DATE", "ARCH"]);
    for (path, count, date, arch) in rows {
        let key = path
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();
        let date_cell = date.unwrap_or_else(|| "-".to_string());
        // ARCH is sourced from `host.arch` on the run's first
        // sidecar; renders as `-` when no sidecar carries a host
        // (pre-host-context-landing archives, host-only test stubs
        // that never populate host) so the column reads consistently
        // with the DATE sentinel.
        let arch_cell = arch.unwrap_or_else(|| "-".to_string());
        table.add_row(vec![key, count.to_string(), date_cell, arch_cell]);
    }
    println!("{table}");
    Ok(())
}

/// One row produced by [`sorted_run_entries`] — see its doc for
/// the field-position contract. Aliased to keep the production
/// caller's `for (path, count, date, arch) in rows` destructure
/// readable while satisfying clippy's `type_complexity` lint
/// (a bare `Vec<(.., .., .., ..)>` over 4 elements crosses the
/// default complexity threshold).
type RunEntryRow = (std::path::PathBuf, usize, Option<String>, Option<String>);

/// Pure-IO sort + collection step of [`list_runs`]. Reads `root`,
/// filters to subdirectories, computes each entry's mtime + sidecar
/// pool, and returns the rows sorted by mtime DESCENDING (most
/// recent first), with `file_name` as a tiebreaker for equal mtimes
/// or both-unreadable mtime pairs. Entries whose mtime cannot be
/// read sort to the END of the returned vec.
///
/// Each row is a [`RunEntryRow`] = `(PathBuf, usize, Option<String>, Option<String>)`:
/// - `PathBuf`: the run-directory path (caller derives `file_name`
///   for display).
/// - `usize`: number of sidecars under the run dir (one level deep
///   per [`crate::test_support::collect_sidecars`]).
/// - `Option<String>`: earliest non-empty sidecar timestamp present
///   in the directory, or `None` when no sidecar carries a
///   non-empty timestamp (caller substitutes a display sentinel
///   like `"-"`).
/// - `Option<String>`: arch (e.g. `"x86_64"`, `"aarch64"`) from the
///   first sidecar that carries `host.arch`; `None` when no
///   sidecar in the dir has a `host`-populated arch (pre-
///   host-context-landing archives, host-only-stub paths that
///   never populate host). Caller substitutes a display sentinel
///   for the missing case the same way it does for date.
///
/// Factored out of [`list_runs`] so unit tests can exercise the
/// sort + row-shape contract without tee'ing stdout. Called once
/// per `cargo ktstr stats list` invocation; takes `&Path` so the
/// test harness can drive it against a tempdir-backed fixture
/// without mutating env vars.
pub(crate) fn sorted_run_entries(root: &std::path::Path) -> std::io::Result<Vec<RunEntryRow>> {
    use std::fs;
    use std::time::SystemTime;
    // Collect (entry, mtime) pairs so the sort key is computed once
    // per entry rather than per pairwise comparison. Entries whose
    // metadata or mtime cannot be read fall through with `None` and
    // sort to the end of the listing — the secondary `file_name`
    // tiebreaker keeps that group stable.
    let mut entries: Vec<(fs::DirEntry, Option<SystemTime>)> = fs::read_dir(root)?
        .filter_map(|e| e.ok())
        .filter(crate::test_support::is_run_directory)
        .map(|e| {
            let mtime = e.metadata().ok().and_then(|m| m.modified().ok());
            (e, mtime)
        })
        .collect();
    // Sort by mtime DESCENDING (newest first), with file_name as a
    // deterministic tiebreaker. `None` mtimes (metadata read
    // failure) sort to the END so good rows lead the listing —
    // `Reverse(Some(t))` orders larger times first, and `None`
    // wraps as `Reverse(None)` which compares as greater than
    // any `Reverse(Some(_))`, landing it at the end.
    entries.sort_by(|(a, a_mtime), (b, b_mtime)| {
        use std::cmp::Reverse;
        Reverse(*a_mtime)
            .cmp(&Reverse(*b_mtime))
            .then_with(|| a.file_name().cmp(&b.file_name()))
    });

    let rows = entries
        .into_iter()
        .map(|(entry, _)| {
            let path = entry.path();
            let sidecars = crate::test_support::collect_sidecars(&path);
            let count = sidecars.len();
            let date = sidecars
                .iter()
                .map(|s| s.timestamp.as_str())
                .filter(|t| !t.is_empty())
                .min()
                .map(|s| s.to_string());
            // Arch from the first sidecar that carries
            // `host.arch`. A run is on one machine so every
            // sidecar in the dir agrees on arch; taking the first
            // non-None reading keeps the lookup O(1) under the
            // common case (host-populated sidecar at the head
            // of `collect_sidecars`'s walk).
            let arch = sidecars
                .iter()
                .find_map(|s| s.host.as_ref().and_then(|h| h.arch.clone()));
            (path, count, date, arch)
        })
        .collect();
    Ok(rows)
}

/// Pool every sidecar under the runs root (or `dir` when set) and
/// emit the distinct values present on each filterable dimension.
///
/// Nine dimensions are reported: `kernel` (from
/// `SidecarResult::kernel_version`), `scheduler`, `topology`,
/// `work_type`, `commit` (from `SidecarResult::project_commit`),
/// `kernel_commit` (from `SidecarResult::kernel_commit`), `source`
/// (from `SidecarResult::run_source`), `resolve_source` (from
/// `SidecarResult::resolve_source`), and `cpu_budget` (from
/// `SidecarResult::cpu_budget`). The dimension catalogue here matches
/// what `cargo ktstr stats compare` accepts as `--X` and `--a-X` /
/// `--b-X` filter flags — the command exists so an operator can answer
/// "what kernel versions are in the pool?" before crafting a compare
/// invocation. The JSON keys `commit` and `source` are the wire
/// contract; the corresponding per-side filter flags spell
/// `--project-commit` and `--run-source`.
///
/// `cpu_budget` is the sole NUMERIC dimension: its JSON value is an
/// array of integers (every other dimension is a string array), and
/// budget-0 skip rows (never-booted) are excluded — a non-empty pool
/// of only skips renders the `(all runs skipped — no budget recorded)`
/// sentinel rather than `null` / `unknown`.
///
/// `kernel_version`, `project_commit`, `kernel_commit`,
/// `run_source`, and `resolve_source` are `Option<String>` on the
/// source sidecar; absence is reported as a literal JSON `null` in
/// the JSON shape and the textual sentinel `unknown` in the table shape.
/// The set is sorted by the type's natural ordering (`BTreeSet`);
/// `None` collates before any populated value in `Option<String>`
/// ordering, so `null` / `unknown` always lands at the top of the
/// per-dimension listing.
///
/// `json=true` emits a JSON object keyed by dimension name with
/// arrays of values (with `null` interleaved for absent
/// `kernel`, `commit`, `kernel_commit`, `source`, or
/// `resolve_source` entries — the five optional dimensions);
/// `json=false` emits a
/// per-dimension human-readable block with the values one per
/// line.
///
/// `dir` mirrors `compare_partitions` / `show_run_host` semantics:
/// when `Some(d)`, `d` replaces `runs_root()` as the pool source;
/// when `None`, `runs_root()` is used.
pub fn list_values(json: bool, dir: Option<&std::path::Path>) -> anyhow::Result<String> {
    use std::collections::BTreeSet;

    let (root, override_archive) = match dir {
        Some(d) => (d.to_path_buf(), true),
        None => (crate::test_support::runs_root(), false),
    };
    let mut pool = crate::test_support::collect_pool(&root);
    if override_archive {
        // `--dir` points at a non-default pool root. Stats tooling
        // treats those sidecars as `"archive"` regardless of the
        // tag they were written with — see
        // `apply_archive_source_override` for the rewrite contract.
        crate::test_support::apply_archive_source_override(&mut pool);
    }

    let mut kernels: BTreeSet<Option<String>> = BTreeSet::new();
    let mut project_commits: BTreeSet<Option<String>> = BTreeSet::new();
    let mut kernel_commits: BTreeSet<Option<String>> = BTreeSet::new();
    let mut run_sources: BTreeSet<Option<String>> = BTreeSet::new();
    let mut resolve_sources: BTreeSet<Option<String>> = BTreeSet::new();
    let mut cpu_budgets: BTreeSet<u32> = BTreeSet::new();
    let mut schedulers: BTreeSet<String> = BTreeSet::new();
    let mut topologies: BTreeSet<String> = BTreeSet::new();
    let mut work_types: BTreeSet<String> = BTreeSet::new();

    for sc in &pool {
        kernels.insert(sc.kernel_version.clone());
        project_commits.insert(sc.project_commit.clone());
        kernel_commits.insert(sc.kernel_commit.clone());
        run_sources.insert(sc.run_source.clone());
        resolve_sources.insert(sc.resolve_source.clone());
        // 0 = skip rows (never booted); exclude — they carry no budget.
        if sc.cpu_budget != 0 {
            cpu_budgets.insert(sc.cpu_budget);
        }
        schedulers.insert(sc.scheduler.clone());
        topologies.insert(sc.topology.clone());
        work_types.insert(sc.work_type.clone());
    }

    if json {
        let kernels_json: Vec<serde_json::Value> = kernels
            .iter()
            .map(|opt| match opt {
                Some(s) => serde_json::Value::String(s.clone()),
                None => serde_json::Value::Null,
            })
            .collect();
        let project_commits_json: Vec<serde_json::Value> = project_commits
            .iter()
            .map(|opt| match opt {
                Some(s) => serde_json::Value::String(s.clone()),
                None => serde_json::Value::Null,
            })
            .collect();
        let kernel_commits_json: Vec<serde_json::Value> = kernel_commits
            .iter()
            .map(|opt| match opt {
                Some(s) => serde_json::Value::String(s.clone()),
                None => serde_json::Value::Null,
            })
            .collect();
        let run_sources_json: Vec<serde_json::Value> = run_sources
            .iter()
            .map(|opt| match opt {
                Some(s) => serde_json::Value::String(s.clone()),
                None => serde_json::Value::Null,
            })
            .collect();
        let resolve_sources_json: Vec<serde_json::Value> = resolve_sources
            .iter()
            .map(|opt| match opt {
                Some(s) => serde_json::Value::String(s.clone()),
                None => serde_json::Value::Null,
            })
            .collect();
        // JSON keys stay as `commit` / `source` — operator-visible
        // wire contract for `cargo ktstr stats list-values --json`
        // does not rename when the internal field/variable does.
        // Note: the per-side filter flags on `compare` spell as
        // `--project-commit` / `--run-source` (longer-form
        // disambiguating names), so the JSON keys here intentionally
        // diverge from the CLI flag names. The wire contract is the
        // shorter form because that's what every external consumer
        // (CI scripts, archive readers) has been parsing since the
        // sidecar format was first introduced.
        let payload = serde_json::json!({
            "kernel": kernels_json,
            "commit": project_commits_json,
            "kernel_commit": kernel_commits_json,
            "source": run_sources_json,
            "resolve_source": resolve_sources_json,
            "cpu_budget": cpu_budgets.iter().collect::<Vec<_>>(),
            "scheduler": schedulers.iter().collect::<Vec<_>>(),
            "topology": topologies.iter().collect::<Vec<_>>(),
            "work_type": work_types.iter().collect::<Vec<_>>(),
        });
        return serde_json::to_string_pretty(&payload)
            .map(|mut s| {
                s.push('\n');
                s
            })
            .map_err(|e| anyhow::anyhow!("serialize list-values JSON: {e}"));
    }

    let mut out = String::new();
    let render_opt_set = |out: &mut String, label: &str, set: &BTreeSet<Option<String>>| {
        out.push_str(label);
        out.push('\n');
        if set.is_empty() {
            out.push_str("  (no sidecars in pool)\n");
        } else {
            for opt in set {
                match opt {
                    Some(s) => {
                        out.push_str("  ");
                        out.push_str(s);
                        out.push('\n');
                    }
                    None => out.push_str("  unknown\n"),
                }
            }
        }
        out.push('\n');
    };
    let render_str_set = |out: &mut String, label: &str, set: &BTreeSet<String>| {
        out.push_str(label);
        out.push('\n');
        if set.is_empty() {
            out.push_str("  (no sidecars in pool)\n");
        } else {
            for s in set {
                out.push_str("  ");
                out.push_str(s);
                out.push('\n');
            }
        }
        out.push('\n');
    };
    render_opt_set(&mut out, "kernel:", &kernels);
    render_opt_set(&mut out, "commit:", &project_commits);
    render_opt_set(&mut out, "kernel_commit:", &kernel_commits);
    render_opt_set(&mut out, "source:", &run_sources);
    render_opt_set(&mut out, "resolve_source:", &resolve_sources);
    out.push_str("cpu_budget:\n");
    if cpu_budgets.is_empty() {
        // cpu_budgets excludes budget-0 skip rows, so an empty set on a
        // NON-empty pool means every sidecar was a skip — distinguish
        // that from a genuinely empty pool (the other dims always insert
        // a value per sidecar, so they never hit this).
        if pool.is_empty() {
            out.push_str("  (no sidecars in pool)\n");
        } else {
            out.push_str("  (all runs skipped — no budget recorded)\n");
        }
    } else {
        for b in &cpu_budgets {
            out.push_str(&format!("  {b}\n"));
        }
    }
    out.push('\n');
    render_str_set(&mut out, "scheduler:", &schedulers);
    render_str_set(&mut out, "topology:", &topologies);
    render_str_set(&mut out, "work_type:", &work_types);
    Ok(out)
}
