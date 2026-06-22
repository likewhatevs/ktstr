//! End-to-end comparison pipeline: groups, fudges, derives,
//! sorts, and lifts the [`crate::ctprof::CtprofSnapshot`] pair into a
//! materialized [`super::CtprofDiff`].
//!
//! The pipeline:
//!
//! 1. [`compare`] is the public orchestrator. It builds per-side
//!    thread groups via [`super::build_groups`] (with
//!    [`super::build_cgroup_key_map`] for cgroup-mode tightening),
//!    flattens cgroup-stats per [`flatten_cgroup_stats`] when
//!    grouping by cgroup, fudges baseline-only / candidate-only
//!    cgroups together via the Jaccard thread-population overlap
//!    rule, emits one [`super::DiffRow`] per `(matched group,
//!    metric)` pair, builds derived rows per
//!    [`super::CTPROF_DERIVED_METRICS`], and finally hands the
//!    populated [`super::CtprofDiff`] back. Group keys present
//!    only on one side surface in
//!    [`super::CtprofDiff::only_baseline`] /
//!    [`super::CtprofDiff::only_candidate`] AFTER fudging
//!    consumed any pairs that joined.
//!
//! 2. [`emit_fudged_rows`] handles the N:1 fudge merge: every
//!    candidate group matched to a single baseline group has its
//!    metrics merged via `super::merge_aggregated_into`; one
//!    row per metric is then emitted with the merged candidate
//!    side. Display key carries the `[fudged: <leaf>]` marker
//!    so the renderer can flag the merged row visually.
//!
//! 3. [`build_derived_row`] computes one [`super::DerivedRow`]
//!    per matched group per derivation. `None` propagates when
//!    inputs are missing (CONFIG gate not set, jemalloc not
//!    linked) or denominator is zero.
//!
//! 4. [`sort_diff_rows_by_keys`] applies the `--sort-by` multi-key
//!    sort, ranking groups lexicographically per the
//!    [`super::SortKey`] tuple before re-ordering the row vec.
//!    Within a group, registry order is preserved as a stable
//!    tiebreak.
//!
//! 5. [`flatten_cgroup_stats`] collapses cgroup paths via
//!    glob-pattern flatten + auto-normalize key map; per-
//!    controller merges live in [`super::cgroup_merge`].
//!
//! Pointer-hash identity on [`super::DiffRow`] in downstream
//! consumers means the entire pipeline passes
//! `&super::CtprofDiff` by reference — never by value — so row
//! addresses stay stable across renderer passes.

use std::collections::BTreeMap;

use crate::ctprof::{CgroupStats, CtprofSnapshot};

use super::{
    Aggregated, CTPROF_DERIVED_METRICS, CTPROF_METRICS, CompareOptions, CtprofDiff,
    DerivedMetricDef, DerivedRow, DiffRow, FudgedPair, GroupBy, SortKey, ThreadGroup,
    aggregate::merge_aggregated_into,
    cgroup_merge::{merge_cgroup_cpu, merge_cgroup_memory, merge_cgroup_pids, merge_psi},
    format_value_cell,
    groups::{
        build_cgroup_key_map, build_groups, build_row, collect_smaps_rollup,
        collect_smaps_rollup_hierarchical, compile_flatten_patterns, flatten_cgroup_path,
    },
    pattern::{pattern_counts_union, pattern_display_label, pattern_key},
};

/// Apply a multi-key sort to `rows` per `sort_keys`. Computes a
/// per-group sort tuple by looking up the requested metrics'
/// deltas from the existing rows, ranks groups lexicographically
/// (with per-key direction), then orders rows by
/// `(group_rank, metric_registry_idx)` so rows for a given
/// group cluster together in registry order. Mirrors the default
/// sort's stability guarantee (within a group, registry order is
/// preserved; across groups, deterministic by tuple).
///
/// Missing values (a group has no row for the named metric, or
/// the row's `delta` is `None` because the metric is categorical
/// — even though `parse_sort_by` now rejects categorical
/// metrics at the CLI boundary, a programmatic caller can still
/// construct a [`SortKey`] over a `Mode*` metric directly) are
/// treated as `f64::NEG_INFINITY` for descending sort and
/// `f64::INFINITY` for ascending sort — they sink to the bottom
/// either way.
///
/// Caller must supply at least one sort key — an empty slice is a
/// programming error (the empty-spec case is handled at the
/// caller via the `sort_by.is_empty()` branch in
/// [`compare`] / `write_show`).
pub(super) fn sort_diff_rows_by_keys(
    rows: &mut [DiffRow],
    derived_rows: &mut [DerivedRow],
    sort_keys: &[SortKey],
) {
    debug_assert!(
        !sort_keys.is_empty(),
        "sort_diff_rows_by_keys called with empty sort_keys; \
         caller must short-circuit before invoking the multi-key \
         sort path",
    );
    use std::collections::{BTreeMap, BTreeSet};
    // metric name → registry index (for stable within-group
    // ordering after sort). Both sides are `&'static str` so this
    // map is allocation-free at the key layer.
    let metric_idx: BTreeMap<&'static str, usize> = CTPROF_METRICS
        .iter()
        .enumerate()
        .map(|(i, m)| (m.name, i))
        .collect();
    let derived_idx: BTreeMap<&'static str, usize> = CTPROF_DERIVED_METRICS
        .iter()
        .enumerate()
        .map(|(i, m)| (m.name, i))
        .collect();
    // group_key → (metric_name → delta). The inner key is
    // `&'static str` borrowed from `row.metric_name` (itself a
    // `&'static str` pointing into `CTPROF_METRICS.name` or
    // `CTPROF_DERIVED_METRICS.name`), so no per-row
    // allocation is needed for the metric axis. Derived deltas
    // populate the same map; sort_by treats primary and derived
    // names uniformly for ranking.
    let mut group_metrics: BTreeMap<String, BTreeMap<&'static str, f64>> = BTreeMap::new();
    for row in rows.iter() {
        if let Some(d) = row.delta {
            group_metrics
                .entry(row.group_key.clone())
                .or_default()
                .insert(row.metric_name, d);
        }
    }
    for row in derived_rows.iter() {
        if let Some(d) = row.delta {
            group_metrics
                .entry(row.group_key.clone())
                .or_default()
                .insert(row.metric_name, d);
        }
    }
    // Unique group set: every key from group_metrics PLUS every
    // group_key from `rows` that had no numeric delta (every row
    // was Mode/etc). BTreeSet handles dedup-on-insert without a
    // separate sort+dedup pass.
    let mut unique_groups: BTreeSet<String> = group_metrics.keys().cloned().collect();
    for row in rows.iter() {
        unique_groups.insert(row.group_key.clone());
    }
    for row in derived_rows.iter() {
        unique_groups.insert(row.group_key.clone());
    }
    // Precompute (group_key, sort_tuple) pairs once. Avoids
    // recomputing the tuple inside the comparator on every
    // comparison; with N groups and a non-trivial tuple this
    // saves O(N log N) tuple builds.
    let mut groups_with_tuples: Vec<(String, Vec<f64>)> = unique_groups
        .into_iter()
        .map(|g| {
            let metrics = group_metrics.get(&g);
            let tuple: Vec<f64> = sort_keys
                .iter()
                .map(|k| {
                    metrics
                        .and_then(|m| m.get(k.metric).copied())
                        .unwrap_or(if k.descending {
                            f64::NEG_INFINITY
                        } else {
                            f64::INFINITY
                        })
                })
                .collect();
            (g, tuple)
        })
        .collect();
    // Sort with the precomputed tuples: comparator does only
    // O(sort_keys.len()) f64 comparisons per call, no map
    // lookups.
    groups_with_tuples.sort_by(|(ga, ta), (gb, tb)| {
        for (i, key) in sort_keys.iter().enumerate() {
            let (va, vb) = (ta[i], tb[i]);
            let ord = if key.descending {
                vb.partial_cmp(&va).unwrap_or(std::cmp::Ordering::Equal)
            } else {
                va.partial_cmp(&vb).unwrap_or(std::cmp::Ordering::Equal)
            };
            if ord != std::cmp::Ordering::Equal {
                return ord;
            }
        }
        // Final tie-break: ascending group_key for determinism.
        ga.cmp(gb)
    });
    let group_ranks: BTreeMap<String, usize> = groups_with_tuples
        .into_iter()
        .enumerate()
        .map(|(i, (g, _))| (g, i))
        .collect();
    rows.sort_by(|a, b| {
        let ra = group_ranks.get(&a.group_key).copied().unwrap_or(usize::MAX);
        let rb = group_ranks.get(&b.group_key).copied().unwrap_or(usize::MAX);
        ra.cmp(&rb).then_with(|| {
            let ia = metric_idx.get(a.metric_name).copied().unwrap_or(usize::MAX);
            let ib = metric_idx.get(b.metric_name).copied().unwrap_or(usize::MAX);
            ia.cmp(&ib)
        })
    });
    derived_rows.sort_by(|a, b| {
        let ra = group_ranks.get(&a.group_key).copied().unwrap_or(usize::MAX);
        let rb = group_ranks.get(&b.group_key).copied().unwrap_or(usize::MAX);
        ra.cmp(&rb).then_with(|| {
            let ia = derived_idx
                .get(a.metric_name)
                .copied()
                .unwrap_or(usize::MAX);
            let ib = derived_idx
                .get(b.metric_name)
                .copied()
                .unwrap_or(usize::MAX);
            ia.cmp(&ib)
        })
    });
}

/// Compute one [`DerivedRow`] for a matched group. Called per
/// derivation in [`compare`]; the produced row carries `None`
/// values when the formula's inputs are missing or the
/// denominator is zero on either side.
pub(super) fn build_derived_row(
    key: &str,
    display_key: &str,
    n_a: usize,
    n_b: usize,
    def: &DerivedMetricDef,
    metrics_a: &BTreeMap<String, Aggregated>,
    metrics_b: &BTreeMap<String, Aggregated>,
) -> DerivedRow {
    let baseline = (def.compute)(metrics_a);
    let candidate = (def.compute)(metrics_b);
    let (delta, delta_pct) = match (baseline, candidate) {
        (Some(a), Some(b)) => {
            let va = a.as_f64();
            let vb = b.as_f64();
            let d = vb - va;
            // Suppress delta_pct for ratio rows per the design
            // call: `+20%` on a `[0, 1]` ratio is misleading.
            let pct = if def.is_ratio {
                None
            } else if va.abs() > f64::EPSILON {
                Some(d / va)
            } else {
                None
            };
            (Some(d), pct)
        }
        _ => (None, None),
    };
    DerivedRow {
        group_key: key.to_string(),
        display_key: display_key.to_string(),
        thread_count_a: n_a,
        thread_count_b: n_b,
        metric_name: def.name,
        metric_ladder: def.ladder,
        is_ratio: def.is_ratio,
        baseline,
        candidate,
        delta,
        delta_pct,
        sort_by_cell: None,
        sort_by_delta: None,
    }
}

/// Emit one DiffRow per CTPROF_METRICS entry (and one DerivedRow
/// per CTPROF_DERIVED_METRICS entry) for each fudged baseline
/// key, with candidate values aggregated across the N matched
/// candidate groups (N:1 merge). The shared display key
/// `[fudged]` flags the row in the renderer.
///
/// `matches` keys are baseline group keys; the values are the
/// list of candidate group keys that matched the baseline. A
/// missing baseline group skips the entry; a missing candidate
/// group is skipped from the merge but does not abort. Values
/// from each ckey are merged via [`merge_aggregated_into`].
pub(super) fn emit_fudged_rows(
    diff: &mut CtprofDiff,
    matches: &BTreeMap<String, Vec<String>>,
    groups_a: &BTreeMap<String, ThreadGroup>,
    groups_b: &BTreeMap<String, ThreadGroup>,
) {
    for (bkey, ckeys) in matches {
        let Some(ga) = groups_a.get(bkey) else {
            continue;
        };
        let mut merged_metrics: BTreeMap<String, Aggregated> = BTreeMap::new();
        let mut merged_thread_count: usize = 0;
        for ckey in ckeys {
            let Some(gb) = groups_b.get(ckey) else {
                continue;
            };
            merged_thread_count += gb.thread_count;
            for (name, val) in &gb.metrics {
                let entry = merged_metrics.entry(name.clone());
                match entry {
                    std::collections::btree_map::Entry::Vacant(e) => {
                        e.insert(val.clone());
                    }
                    std::collections::btree_map::Entry::Occupied(mut e) => {
                        let existing = e.get_mut();
                        merge_aggregated_into(existing, val);
                    }
                }
            }
        }
        // Display key for the fudged row: include the
        // baseline cgroup leaf so multiple fudged pairs in the
        // same diff stay distinguishable. `[fudged]` alone
        // would render N rows that all look identical when
        // viewed by an operator scanning the table; appending
        // the leaf path component (the bcg's last `/`-segment)
        // disambiguates without bloating the column.
        let bcg = bkey.split_once('\x00').map_or(bkey.as_str(), |(cg, _)| cg);
        let leaf = bcg.rsplit_once('/').map_or(bcg, |(_, l)| l);
        let display_key = if leaf.is_empty() {
            "[fudged]".to_string()
        } else {
            format!("[fudged: {leaf}]")
        };
        for metric in CTPROF_METRICS {
            let Some(a) = ga.metrics.get(metric.name).cloned() else {
                continue;
            };
            let Some(b) = merged_metrics.get(metric.name).cloned() else {
                continue;
            };
            diff.rows.push(build_row(
                bkey,
                &display_key,
                ga.thread_count,
                merged_thread_count,
                metric,
                a,
                b,
                None,
            ));
        }
        for def in CTPROF_DERIVED_METRICS {
            diff.derived_rows.push(build_derived_row(
                bkey,
                &display_key,
                ga.thread_count,
                merged_thread_count,
                def,
                &ga.metrics,
                &merged_metrics,
            ));
        }
    }
}

/// Read-only context threaded through the phase helpers of
/// [`compare`]. Holds borrows of the snapshot pair, the resolved
/// options, and the per-side thread groups so each phase signature
/// stays short. Built once by [`compare`] after
/// [`build_compare_groups`] returns the owned groups (kept as
/// orchestrator locals so the borrows in this struct stay valid).
struct CompareCtx<'a> {
    baseline: &'a CtprofSnapshot,
    candidate: &'a CtprofSnapshot,
    opts: &'a CompareOptions,
    /// Resolved grouping axis (`opts.group_by.0`), cached so the
    /// phases don't re-unwrap the default wrapper.
    group_by: GroupBy,
    /// Compiled `cgroup_flatten` glob patterns.
    flatten: &'a [glob::Pattern],
    /// Auto-normalize cgroup-key map, present only under
    /// `GroupBy::Cgroup | GroupBy::All` with `no_cg_normalize`
    /// unset.
    cgroup_key_map: Option<&'a BTreeMap<String, String>>,
    groups_a: &'a BTreeMap<String, ThreadGroup>,
    groups_b: &'a BTreeMap<String, ThreadGroup>,
    /// Per-snapshot "now" for lifetime calc: newest candidate
    /// thread's `start_time_clock_ticks`.
    now_b: u64,
}

/// Build both sides' thread groups plus the auto-normalize cgroup
/// key map. Resolves the grouping axis, seeds the per-thread
/// frequency-count gate (only under `GroupBy::Comm | GroupBy::Pcomm`
/// with normalization on), and runs [`build_groups`] for each side.
/// Returns `(group_by, cgroup_key_map, groups_a, groups_b)` by
/// value so [`compare`] can keep them as locals and borrow them
/// into [`CompareCtx`]. `flatten` is computed by the caller and
/// passed in so the patterns outlive the returned groups.
fn build_compare_groups(
    baseline: &CtprofSnapshot,
    candidate: &CtprofSnapshot,
    opts: &CompareOptions,
    flatten: &[glob::Pattern],
) -> CompareGroups {
    let group_by = opts.group_by.0;
    // For `GroupBy::Comm` and `GroupBy::Pcomm`, the frequency gate
    // that promotes a pattern_key from per-thread literal to a
    // clustered bucket must be evaluated against the UNION of both
    // snapshots' threads — otherwise a pattern that has 1 thread
    // in baseline + 3 threads in candidate would join under a
    // `worker-{N}` key on the candidate side but a literal
    // `worker-7` key on the baseline side, and the row would
    // surface as only-in-candidate. Computing the count from
    // the union ensures the same key is used on both sides.
    //
    // The Pcomm path is structurally identical: process names that
    // share a normalized skeleton across snapshots (e.g. ephemeral
    // worker pools whose pcomm differs only by a digit suffix)
    // collapse into one bucket, keyed by the skeleton. The accessor
    // selects which `ThreadState` field feeds the count — `t.comm`
    // for Comm, `t.pcomm` for Pcomm — so one helper covers both
    // axes.
    //
    // Skipped when `no_thread_normalize` is set — under literal
    // grouping, the key IS the comm/pcomm and there is no
    // promotion gate to evaluate.
    // Pattern_counts is only consulted by [`build_groups`] under
    // GroupBy::Pcomm / GroupBy::Comm (as the singleton-revert
    // gate). GroupBy::All uses the compound `cg\x00pcomm\x00comm`
    // key and normalizes both pcomm and comm through `pattern_key`
    // unconditionally — there is no singleton revert under All
    // because every thread already disambiguates by cgroup +
    // process. Skipping the seeding under All saves the
    // baseline+candidate scan when neither side will read it.
    let pattern_counts: Option<BTreeMap<String, usize>> = match (group_by, opts.no_thread_normalize)
    {
        (GroupBy::Comm, false) => Some(pattern_counts_union(baseline, candidate, |t| {
            t.comm.as_str()
        })),
        (GroupBy::Pcomm, false) => Some(pattern_counts_union(baseline, candidate, |t| {
            t.pcomm.as_str()
        })),
        _ => None,
    };
    let cgroup_key_map: Option<BTreeMap<String, String>> =
        if matches!(group_by, GroupBy::Cgroup | GroupBy::All) && !opts.no_cg_normalize {
            Some(build_cgroup_key_map(baseline, candidate, flatten))
        } else {
            None
        };
    let groups_a = build_groups(
        baseline,
        group_by,
        flatten,
        pattern_counts.as_ref(),
        cgroup_key_map.as_ref(),
        opts.no_thread_normalize,
    );
    let groups_b = build_groups(
        candidate,
        group_by,
        flatten,
        pattern_counts.as_ref(),
        cgroup_key_map.as_ref(),
        opts.no_thread_normalize,
    );
    (group_by, cgroup_key_map, groups_a, groups_b)
}

/// Emit one [`DiffRow`] per `(matched group, metric)` plus one
/// [`DerivedRow`] per derivation, and populate
/// [`CtprofDiff::only_baseline`] / [`CtprofDiff::only_candidate`]
/// with the keys present on exactly one side. Display labels for
/// pattern-aware groupings union both sides' members and run grex;
/// every other grouping echoes the join key.
fn emit_matched_rows(diff: &mut CtprofDiff, ctx: &CompareCtx) {
    for (key, group_a) in ctx.groups_a {
        let Some(group_b) = ctx.groups_b.get(key) else {
            diff.only_baseline.push(key.clone());
            continue;
        };
        // Render label: pattern grouping (Comm or Pcomm under
        // auto-normalize) unions baseline+candidate members and
        // runs grex over the result; every other grouping just
        // echoes the join key. Computed once per matched group,
        // reused across every metric row built off it.
        let pattern_axis_active =
            matches!(ctx.group_by, GroupBy::Comm | GroupBy::Pcomm) && !ctx.opts.no_thread_normalize;
        let display_key = if pattern_axis_active {
            let mut union: Vec<String> = group_a.members.clone();
            union.extend(group_b.members.iter().cloned());
            union.sort();
            union.dedup();
            pattern_display_label(key, &union)
        } else {
            key.clone()
        };
        // uptime_pct filled in second pass after all groups processed

        for metric in CTPROF_METRICS {
            let Some(a) = group_a.metrics.get(metric.name).cloned() else {
                continue;
            };
            let Some(b) = group_b.metrics.get(metric.name).cloned() else {
                continue;
            };
            diff.rows.push(build_row(
                key,
                &display_key,
                group_a.thread_count,
                group_b.thread_count,
                metric,
                a,
                b,
                None, // uptime_pct filled in second pass
            ));
        }
        // Derived metrics: one row per derivation per matched
        // group. Each row carries `None`-valued sides when the
        // formula's inputs are missing or the denominator is
        // zero — operator sees `-` rather than a synthesized
        // zero or NaN.
        for def in CTPROF_DERIVED_METRICS {
            diff.derived_rows.push(build_derived_row(
                key,
                &display_key,
                group_a.thread_count,
                group_b.thread_count,
                def,
                &group_a.metrics,
                &group_b.metrics,
            ));
        }
    }
    for key in ctx.groups_b.keys() {
        if !ctx.groups_a.contains_key(key) {
            diff.only_candidate.push(key.clone());
        }
    }
}

/// Extract the cgroup prefix (the `cg` portion) from a compound
/// `cg\x00pcomm\x00comm` group key. Used throughout the fudge
/// phase to group one-sided keys by their owning cgroup.
fn cg_prefix(key: &str) -> &str {
    key.split_once('\x00').map_or(key, |(cg, _)| cg)
}

/// `(pcomm, comm)` thread-type set for one cgroup prefix. Fudge
/// matches cgroups by Jaccard similarity over these sets.
type TypeSet = std::collections::BTreeSet<(String, String)>;

/// Grouping outcome from [`build_compare_groups`]: the resolved
/// `GroupBy`, an optional cgroup-fudge map, and the per-side thread
/// groups (baseline, candidate).
type CompareGroups = (
    GroupBy,
    Option<BTreeMap<String, String>>,
    BTreeMap<String, ThreadGroup>,
    BTreeMap<String, ThreadGroup>,
);

/// Fudge-match outcome from [`fudge_match_prefixes`]: the matched
/// `(baseline_cg, candidate_cg)` prefix pairs plus the per-prefix
/// thread-type sets for each side (baseline, candidate).
type FudgeMatch = (
    Vec<(String, String)>,
    BTreeMap<String, TypeSet>,
    BTreeMap<String, TypeSet>,
);

/// Match baseline-only cgroup prefixes to candidate-only cgroup
/// prefixes by Jaccard similarity (≥ 0.90, overlap ≥ 10) over
/// their `(pcomm, comm)` thread-type sets, and return both the
/// matched `(baseline_cg, candidate_cg)` pairs and the per-prefix
/// type sets (reused downstream for the residual report). Excludes
/// any prefix that already has a key matched on both sides
/// (`matched_prefixes`).
fn fudge_match_prefixes(diff: &CtprofDiff, ctx: &CompareCtx) -> FudgeMatch {
    // Collect thread types per CGROUP PREFIX (not per compound key).
    let mut cg_types_a: BTreeMap<String, TypeSet> = BTreeMap::new();
    let mut cg_types_b: BTreeMap<String, TypeSet> = BTreeMap::new();

    // Collect cgroup prefixes that appear in BOTH groups (already matched).
    // These must be excluded from fudging.
    let matched_prefixes: std::collections::BTreeSet<String> = ctx
        .groups_a
        .keys()
        .filter(|k| ctx.groups_b.contains_key(*k))
        .map(|k| cg_prefix(k).to_string())
        .collect();

    // Collect unique cgroup prefixes from one-sided keys,
    // skipping any prefix that already has matched keys.
    let mut cg_prefixes_a: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    let mut cg_prefixes_b: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    for key in &diff.only_baseline {
        let pfx = cg_prefix(key).to_string();
        if !matched_prefixes.contains(&pfx) {
            cg_prefixes_a.insert(pfx);
        }
    }
    for key in &diff.only_candidate {
        let pfx = cg_prefix(key).to_string();
        if !matched_prefixes.contains(&pfx) {
            cg_prefixes_b.insert(pfx);
        }
    }

    // Populate thread types per cgroup prefix from snapshots.
    for t in &ctx.baseline.threads {
        let cg = flatten_cgroup_path(&t.cgroup, ctx.flatten);
        let cg_key = match ctx.cgroup_key_map.and_then(|m| m.get(&cg)) {
            Some(k) => k.clone(),
            None => cg,
        };
        if cg_prefixes_a.contains(&cg_key) {
            cg_types_a
                .entry(cg_key)
                .or_default()
                .insert((pattern_key(&t.pcomm), pattern_key(&t.comm)));
        }
    }
    for t in &ctx.candidate.threads {
        let cg = flatten_cgroup_path(&t.cgroup, ctx.flatten);
        let cg_key = match ctx.cgroup_key_map.and_then(|m| m.get(&cg)) {
            Some(k) => k.clone(),
            None => cg,
        };
        if cg_prefixes_b.contains(&cg_key) {
            cg_types_b
                .entry(cg_key)
                .or_default()
                .insert((pattern_key(&t.pcomm), pattern_key(&t.comm)));
        }
    }

    // Match cgroup prefixes by Jaccard similarity. Each
    // candidate finds its best baseline match independently —
    // multiple candidates can match the same baseline (N
    // vs 1 baseline).
    let mut fudged_cg: Vec<(String, String)> = Vec::new(); // (baseline_cg, candidate_cg)

    for ccg in &cg_prefixes_b {
        let Some(set_b) = cg_types_b.get(ccg) else {
            continue;
        };
        if set_b.len() < 10 {
            continue;
        }
        let mut best: Option<(&str, f64, usize)> = None;
        for bcg in &cg_prefixes_a {
            let Some(set_a) = cg_types_a.get(bcg) else {
                continue;
            };
            let intersection = set_a.intersection(set_b).count();
            if intersection < 10 {
                continue;
            }
            let union = set_a.union(set_b).count();
            let jaccard = intersection as f64 / union as f64;
            if jaccard >= 0.90 && best.is_none_or(|(_, bj, _)| jaccard > bj) {
                best = Some((bcg.as_str(), jaccard, intersection));
            }
        }
        if let Some((bcg, _jaccard, _overlap)) = best {
            fudged_cg.push((bcg.to_string(), ccg.clone()));
        }
    }

    (fudged_cg, cg_types_a, cg_types_b)
}

/// Tracks which one-sided keys have been consumed by fudging (so
/// the cascade pass and the post-fudge `retain` drop them) and
/// the per-baseline cascade root + child count for the report.
#[derive(Default)]
struct FudgeRemap {
    remove_baseline: std::collections::BTreeSet<String>,
    remove_candidate: std::collections::BTreeSet<String>,
    cascade_counts: BTreeMap<String, usize>,
    cascade_roots: BTreeMap<(String, String), (String, String)>,
}

/// Apply the direct (suffix-exact) remap and then the cascade
/// remap for every fudged cgroup pair, emitting aggregated rows
/// for both. Records consumed keys + cascade metadata in
/// [`FudgeRemap`] and appends every joined `(bkey, ckey)` to
/// `fudged_key_pairs`. Mirrors the two `emit_fudged_rows` calls of
/// the original block: first the direct suffix matches, then the
/// cascaded children.
fn fudge_cascade(
    diff: &mut CtprofDiff,
    ctx: &CompareCtx,
    fudged_cg: &[(String, String)],
    fudged_key_pairs: &mut Vec<(String, String)>,
) -> FudgeRemap {
    // For each fudged cgroup pair, remap ALL compound keys sharing
    // that prefix. Match baseline keys to candidate keys by their
    // pcomm\x00comm suffix.
    let mut remap = FudgeRemap::default();

    // Collect all (baseline_key, candidate_key) pairs across
    // all fudge pairs. Multiple candidate keys can map to the
    // same baseline key (N:1).
    let mut fudge_matches: BTreeMap<String, Vec<String>> = BTreeMap::new(); // bkey → [ckeys]
    for (bcg, ccg) in fudged_cg {
        let b_keys: Vec<&String> = diff
            .only_baseline
            .iter()
            .filter(|k| cg_prefix(k) == bcg.as_str())
            .collect();
        let c_keys: Vec<&String> = diff
            .only_candidate
            .iter()
            .filter(|k| cg_prefix(k) == ccg.as_str())
            .collect();
        let c_suffix_map: BTreeMap<&str, &String> = c_keys
            .iter()
            .map(|k| {
                let suffix = k.split_once('\x00').map_or("", |(_, s)| s);
                (suffix, *k)
            })
            .collect();
        for bkey in &b_keys {
            let b_suffix = bkey.split_once('\x00').map_or("", |(_, s)| s);
            if let Some(ckey) = c_suffix_map.get(b_suffix) {
                remap.remove_baseline.insert((*bkey).clone());
                remap.remove_candidate.insert((*ckey).clone());
                fudged_key_pairs.push(((*bkey).clone(), (*ckey).clone()));
                fudge_matches
                    .entry((*bkey).clone())
                    .or_default()
                    .push((*ckey).clone());
            }
        }
    }
    // Emit one row per baseline key with candidate values
    // aggregated across all N matched candidate groups.
    emit_fudged_rows(diff, &fudge_matches, ctx.groups_a, ctx.groups_b);

    // Cascade: for each fudged cgroup pair, compute cascade
    // roots by stripping the longest common suffix (by /
    // segments) from the pair. Use those shorter roots for
    // starts_with matching, not the full fudged paths.
    let mut cascade_matches: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for (bcg, ccg) in fudged_cg {
        let b_segs: Vec<&str> = bcg.split('/').collect();
        let c_segs: Vec<&str> = ccg.split('/').collect();
        let common_suffix_len = b_segs
            .iter()
            .rev()
            .zip(c_segs.iter().rev())
            .take_while(|(a, b)| a == b)
            .count();
        let b_root: String = b_segs[..b_segs.len().saturating_sub(common_suffix_len)].join("/");
        let c_root: String = c_segs[..c_segs.len().saturating_sub(common_suffix_len)].join("/");
        let b_root = if b_root.is_empty() {
            bcg.clone()
        } else {
            b_root
        };
        let c_root = if c_root.is_empty() {
            ccg.clone()
        } else {
            c_root
        };

        remap
            .cascade_roots
            .insert((bcg.clone(), ccg.clone()), (b_root.clone(), c_root.clone()));

        // Boundary-checked prefix match: accept either an
        // exact root match (tail.is_empty()) or a child path
        // (tail starts with '/'). Bare `starts_with` would
        // false-match siblings — `/svc-extra` would slip
        // through `b_root=/svc`. The non-matching keys would
        // ultimately fail downstream lookup (c_by_suffix
        // applies the same boundary filter), but skipping them
        // here keeps the cascade scan consistent with the
        // remap rule and avoids needless work.
        let is_root_or_child = |cg: &str, root: &str| {
            let Some(tail) = cg.strip_prefix(root) else {
                return false;
            };
            tail.is_empty() || tail.starts_with('/')
        };
        let remaining_b: Vec<String> = diff
            .only_baseline
            .iter()
            .filter(|k| {
                !remap.remove_baseline.contains(*k)
                    && is_root_or_child(cg_prefix(k), b_root.as_str())
            })
            .cloned()
            .collect();
        let remaining_c: Vec<String> = diff
            .only_candidate
            .iter()
            .filter(|k| {
                !remap.remove_candidate.contains(*k)
                    && is_root_or_child(cg_prefix(k), c_root.as_str())
            })
            .cloned()
            .collect();
        // Filter_map (not map) so boundary-rejected entries
        // drop instead of all colliding at the empty-string
        // key in the BTreeMap. Combined with the upstream
        // is_root_or_child filter on remaining_c, every entry
        // here should already qualify; this stays defensive
        // against future filter regressions.
        let c_by_suffix: BTreeMap<String, &String> = remaining_c
            .iter()
            .filter_map(|k| {
                let child_cg = cg_prefix(k);
                let tail = &child_cg[c_root.len()..];
                if !tail.is_empty() && !tail.starts_with('/') {
                    return None;
                }
                let rewritten = format!("{b_root}{tail}");
                let suffix = k.split_once('\x00').map_or("", |(_, s)| s);
                Some((format!("{rewritten}\x00{suffix}"), k))
            })
            .collect();
        for bkey in &remaining_b {
            if let Some(ckey) = c_by_suffix.get(bkey) {
                remap.remove_baseline.insert(bkey.clone());
                remap.remove_candidate.insert((*ckey).clone());
                fudged_key_pairs.push((bkey.clone(), (*ckey).clone()));
                *remap.cascade_counts.entry(bcg.clone()).or_insert(0) += 1;
                cascade_matches
                    .entry(bkey.clone())
                    .or_default()
                    .push((*ckey).clone());
            }
        }
    }

    // Emit aggregated rows for cascaded children (same N:1 merge).
    emit_fudged_rows(diff, &cascade_matches, ctx.groups_a, ctx.groups_b);

    remap
}

/// Build the [`FudgedPair`] report vector for every matched cgroup
/// pair: overlap / Jaccard over the two thread-type sets, residuals
/// computed against the per-bcg / per-ccg UNION of counterpart sets
/// (so a type missing from every counterpart surfaces once), and
/// the cascade child count + roots from [`FudgeRemap`].
fn build_fudged_pairs(
    fudged_cg: &[(String, String)],
    cg_types_a: &BTreeMap<String, TypeSet>,
    cg_types_b: &BTreeMap<String, TypeSet>,
    remap: &FudgeRemap,
) -> Vec<FudgedPair> {
    // Store fudge report per cgroup pair.
    //
    // Residual deduplication for N:1: when one bcg is matched
    // against multiple ccgs, the per-pair baseline_residual =
    // (set_a - set_b_for_this_pair) over-reports thread types
    // that are missing from EVERY ccg — they appear N times,
    // once per pair. Same for candidate_residual when one ccg
    // is matched against multiple bcgs (M:1 in the other
    // direction). Compute residuals against the UNION of all
    // counterpart sets so a missing-on-every-side type
    // surfaces exactly once across the bcg's pairs.
    let mut union_b_for_bcg: BTreeMap<String, TypeSet> = BTreeMap::new();
    let mut union_a_for_ccg: BTreeMap<String, TypeSet> = BTreeMap::new();
    for (bcg, ccg) in fudged_cg {
        if let Some(sb) = cg_types_b.get(ccg) {
            union_b_for_bcg
                .entry(bcg.clone())
                .or_default()
                .extend(sb.iter().cloned());
        }
        if let Some(sa) = cg_types_a.get(bcg) {
            union_a_for_ccg
                .entry(ccg.clone())
                .or_default()
                .extend(sa.iter().cloned());
        }
    }
    fudged_cg
        .iter()
        .map(|(bcg, ccg)| {
            let set_a = cg_types_a.get(bcg).cloned().unwrap_or_default();
            let set_b = cg_types_b.get(ccg).cloned().unwrap_or_default();
            // Residuals against the per-bcg / per-ccg unions
            // so a thread type missing from every counterpart
            // candidate appears exactly once.
            let union_b = union_b_for_bcg.get(bcg).cloned().unwrap_or_default();
            let union_a = union_a_for_ccg.get(ccg).cloned().unwrap_or_default();
            let residual_a: Vec<String> = set_a
                .difference(&union_b)
                .map(|(p, c)| format!("{p}:{c}"))
                .collect();
            let residual_b: Vec<String> = set_b
                .difference(&union_a)
                .map(|(p, c)| format!("{p}:{c}"))
                .collect();
            let intersection = set_a.intersection(&set_b).count();
            let union = set_a.union(&set_b).count();
            FudgedPair {
                baseline_cgroup: bcg.clone(),
                candidate_cgroup: ccg.clone(),
                overlap: intersection,
                jaccard: if union > 0 {
                    intersection as f64 / union as f64
                } else {
                    0.0
                },
                baseline_residual: residual_a,
                candidate_residual: residual_b,
                cascaded_children: remap.cascade_counts.get(bcg).copied().unwrap_or(0),
                baseline_root: remap
                    .cascade_roots
                    .get(&(bcg.clone(), ccg.clone()))
                    .map(|(b, _)| b.clone())
                    .unwrap_or_else(|| bcg.clone()),
                candidate_root: remap
                    .cascade_roots
                    .get(&(bcg.clone(), ccg.clone()))
                    .map(|(_, c)| c.clone())
                    .unwrap_or_else(|| ccg.clone()),
            }
        })
        .collect()
}

/// Content-based cgroup fudging: join one-sided groups whose
/// thread populations overlap (cgroup path changed but workload is
/// the same). Runs only under `GroupBy::All` with both unmatched
/// lists non-empty; otherwise returns an empty `fudged_key_pairs`
/// and leaves `diff` untouched. On the active path it emits the
/// merged rows (direct + cascade), prunes the joined keys from
/// `only_baseline` / `only_candidate`, populates
/// [`CtprofDiff::fudged_pairs`], and returns the `(bkey, ckey)`
/// pairs consumed downstream by [`fill_uptime_pct`] and
/// [`attach_enrichment`].
fn apply_cgroup_fudge(diff: &mut CtprofDiff, ctx: &CompareCtx) -> Vec<(String, String)> {
    let mut fudged_key_pairs: Vec<(String, String)> = Vec::new();
    if ctx.group_by != GroupBy::All
        || diff.only_baseline.is_empty()
        || diff.only_candidate.is_empty()
    {
        return fudged_key_pairs;
    }

    let (fudged_cg, cg_types_a, cg_types_b) = fudge_match_prefixes(diff, ctx);

    let remap = fudge_cascade(diff, ctx, &fudged_cg, &mut fudged_key_pairs);

    diff.only_baseline
        .retain(|k| !remap.remove_baseline.contains(k));
    diff.only_candidate
        .retain(|k| !remap.remove_candidate.contains(k));

    diff.fudged_pairs = build_fudged_pairs(&fudged_cg, &cg_types_a, &cg_types_b, &remap);

    fudged_key_pairs
}

/// Second pass: fill [`DiffRow::uptime_pct`]. Computes each
/// matched group's average candidate-side thread lifetime
/// (`now_b - avg_start_ticks`), folds the fudged pairs' candidate
/// lifetimes into their baseline key (averaged across the N
/// matched candidates), then expresses each as a percentage of the
/// longest-lived group.
fn fill_uptime_pct(diff: &mut CtprofDiff, ctx: &CompareCtx, fudged_key_pairs: &[(String, String)]) {
    let mut group_lifetime: BTreeMap<String, u64> = BTreeMap::new();
    for (key, group_b) in ctx.groups_b {
        if ctx.groups_a.contains_key(key) {
            group_lifetime.insert(
                key.clone(),
                ctx.now_b.saturating_sub(group_b.avg_start_ticks),
            );
        }
    }
    let mut fudge_lt_sum: BTreeMap<String, (u64, u64)> = BTreeMap::new();
    for (bkey, ckey) in fudged_key_pairs {
        if let Some(gb) = ctx.groups_b.get(ckey) {
            let lt = ctx.now_b.saturating_sub(gb.avg_start_ticks);
            let entry = fudge_lt_sum.entry(bkey.clone()).or_insert((0, 0));
            entry.0 += lt;
            entry.1 += 1;
        }
    }
    for (bkey, (sum, count)) in &fudge_lt_sum {
        if *count > 0 {
            group_lifetime.insert(bkey.clone(), sum / count);
        }
    }
    let max_lifetime = group_lifetime.values().copied().max().unwrap_or(1).max(1);
    for row in &mut diff.rows {
        if let Some(&lt) = group_lifetime.get(&row.group_key) {
            row.uptime_pct = Some(lt as f64 / max_lifetime as f64 * 100.0);
        }
    }
}

/// Order the diff + derived rows. With an empty `sort_by`, applies
/// the default `|delta_pct|`-descending sort (ties broken by
/// ascending group_key) to both row vecs. With a non-empty
/// `sort_by`, routes through [`sort_diff_rows_by_keys`] for the
/// multi-key rank, then fills [`DiffRow::sort_by_cell`] /
/// `sort_by_delta` (mirrored onto the derived rows) from the
/// primary sort metric's per-group cell.
fn sort_diff(diff: &mut CtprofDiff, sort_by: &[SortKey]) {
    if sort_by.is_empty() {
        // Default: stable-sort by descending |delta_pct|, ties
        // broken by ascending group_key + registry order of
        // metric. Apply the same shape to derived_rows so the
        // `## Derived metrics` section ranks by salient delta
        // rather than registry order — matches the operator's
        // expectation that the most-changed row sits at the
        // top of every section.
        diff.rows.sort_by(|a, b| {
            b.sort_key()
                .partial_cmp(&a.sort_key())
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.group_key.cmp(&b.group_key))
        });
        diff.derived_rows.sort_by(|a, b| {
            b.sort_key()
                .partial_cmp(&a.sort_key())
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.group_key.cmp(&b.group_key))
        });
    } else {
        // Multi-key sort: rank groups by tuple of named-metric
        // deltas, sort rows by (group_rank, metric_registry_idx).
        sort_diff_rows_by_keys(&mut diff.rows, &mut diff.derived_rows, sort_by);

        // Fill sort_by_cell: for each group, find the sort metric's
        // row and format its baseline→candidate (delta%).
        let sort_metric = sort_by.first().map(|sk| sk.metric);
        diff.sort_metric_name = sort_metric;
        if let Some(metric_name) = sort_metric {
            let mut group_cells: BTreeMap<String, (String, Option<f64>)> = BTreeMap::new();
            for row in &diff.rows {
                if row.metric_name == metric_name && !group_cells.contains_key(&row.group_key) {
                    let b = format_value_cell(&row.baseline, row.metric_ladder);
                    let c = format_value_cell(&row.candidate, row.metric_ladder);
                    let pct = match row.delta_pct {
                        Some(p) => format!(" ({:+.1}%)", p * 100.0),
                        None => String::new(),
                    };
                    group_cells.insert(
                        row.group_key.clone(),
                        (format!("{b}\u{2192}{c}{pct}"), row.delta),
                    );
                }
            }
            for row in &mut diff.rows {
                if let Some((cell, delta)) = group_cells.get(&row.group_key) {
                    row.sort_by_cell = Some(cell.clone());
                    row.sort_by_delta = *delta;
                }
            }
            // Mirror the SortBy fill onto derived_rows so the
            // derived section's SortBy column carries the sort
            // metric's per-group cell instead of "-". Same group
            // keying (group_cells is keyed by group_key, which
            // is the join key shared by primary and derived rows
            // for the same bucket).
            for row in &mut diff.derived_rows {
                if let Some((cell, delta)) = group_cells.get(&row.group_key) {
                    row.sort_by_cell = Some(cell.clone());
                    row.sort_by_delta = *delta;
                }
            }
        }
    }
}

/// Re-key candidate-side smaps data so fudged pairs join the
/// baseline side. Fudge pairs a baseline cgroup with a candidate
/// cgroup, but smaps keys are `cg\x00pcomm` — different cg means no
/// join. For each fudge pair (processed most-specific-root-first),
/// scan candidate smaps keys under the pair's `candidate_root`,
/// strip that root to a relative child path, then re-key under the
/// SAME pair's `baseline_root`. Restricts to candidate cgroups that
/// were actually fudge-matched (`fudged_key_pairs`) and sums values
/// when multiple candidates map to one baseline key.
fn remap_fudged_smaps(diff: &mut CtprofDiff, fudged_key_pairs: &[(String, String)]) {
    // Build a set of candidate cgroup paths that were
    // actually fudge-matched (extracted from the candidate
    // half of fudged_key_pairs). The smaps remap MUST
    // restrict to these paths — without the gate, any smaps
    // key that happens to share a prefix with a fudge pair's
    // candidate_root is remapped, even if its compound-key
    // counterpart was never fudge-matched. Sub-cgroups under
    // a cascade root that didn't actually match would have
    // their smaps data silently re-keyed under the baseline_root.
    let fudged_cg_set: std::collections::BTreeSet<&str> = fudged_key_pairs
        .iter()
        .map(|(_, ckey)| ckey.split_once('\x00').map_or(ckey.as_str(), |(cg, _)| cg))
        .collect();
    let mut sorted_pairs: Vec<&FudgedPair> = diff.fudged_pairs.iter().collect();
    sorted_pairs.sort_by(|a, b| b.candidate_root.len().cmp(&a.candidate_root.len()));
    for fp in sorted_pairs {
        let br = &fp.baseline_root;
        let cr = &fp.candidate_root;
        let cr_slash = format!("{cr}/");
        let cr_nul = format!("{cr}\x00");
        // (relative_child_path + \x00 + pcomm) → summed values,
        // SCOPED to this fudge pair so pairs with different
        // baseline_roots don't share a remap accumulator.
        let mut summed_by_rel: BTreeMap<String, BTreeMap<String, u64>> = BTreeMap::new();
        // Fudge runs only under GroupBy::All, which routes
        // through `collect_smaps_rollup_hierarchical` and
        // produces keys shaped `cgroup\x00pcomm` for every
        // entry (see [`collect_smaps_rollup_inner`] under
        // `compound_cgroup=true`). A bare-cgroup key
        // (no `\x00`) cannot appear here, so the in-root
        // filter only needs to consider the `/`-bounded
        // (cr_slash) and `\x00`-bounded (cr_nul) prefixes.
        let keys: Vec<String> = diff
            .smaps_rollup_b
            .keys()
            .filter(|k| {
                let in_root = k.starts_with(&cr_slash) || k.starts_with(&cr_nul);
                if !in_root {
                    return false;
                }
                // Gate on actual fudge match: the smaps key's
                // cg_path must appear in the fudged_cg_set.
                let cg_path = k.split_once('\x00').map_or(k.as_str(), |(cg, _)| cg);
                fudged_cg_set.contains(cg_path)
            })
            .cloned()
            .collect();
        for k in keys {
            if let Some(val) = diff.smaps_rollup_b.remove(&k) {
                // Split smaps key: cg_path \x00 pcomm. The
                // unwrap_or fallback would only fire on a
                // key with no `\x00`, which cannot reach
                // here (see filter above) — kept defensive.
                let (cg_path, pcomm) = k.split_once('\x00').unwrap_or((&k, ""));
                // Strip candidate root to get relative child path.
                // `cg_path == cr.as_str()` covers the exact-root
                // hit (e.g. `/svc-a\x00pcomm` against root
                // `/svc-a`); the strip_prefix branch covers
                // child paths.
                let child = if cg_path == cr.as_str() {
                    ""
                } else if let Some(rest) = cg_path.strip_prefix(&cr_slash) {
                    rest
                } else {
                    continue;
                };
                let rel_key = format!("{child}\x00{pcomm}");
                let entry = summed_by_rel.entry(rel_key).or_default();
                for (field, v) in &val {
                    let slot = entry.entry(field.clone()).or_insert(0);
                    *slot = slot.saturating_add(*v);
                }
            }
        }
        // Rebuild this pair's baseline-side keys and insert
        // summed data under THIS pair's baseline_root.
        for (rel_key, summed) in summed_by_rel {
            let (child, pcomm) = rel_key.split_once('\x00').unwrap_or((&rel_key, ""));
            let base_key = if child.is_empty() {
                format!("{br}\x00{pcomm}")
            } else {
                format!("{br}/{child}\x00{pcomm}")
            };
            diff.smaps_rollup_b.insert(base_key, summed);
        }
    }
}

/// Attach the non-row enrichment data to `diff`: per-cgroup stats
/// (only under `GroupBy::Cgroup`), host PSI snapshots,
/// per-process smaps rollups (hierarchical under `GroupBy::All`,
/// flat otherwise), the fudged-smaps re-key
/// ([`remap_fudged_smaps`]), and the global sched_ext sysfs
/// snapshot.
fn attach_enrichment(
    diff: &mut CtprofDiff,
    ctx: &CompareCtx,
    fudged_key_pairs: &[(String, String)],
) {
    if ctx.group_by == GroupBy::Cgroup {
        diff.cgroup_stats_a =
            flatten_cgroup_stats(&ctx.baseline.cgroup_stats, ctx.flatten, ctx.cgroup_key_map);
        diff.cgroup_stats_b =
            flatten_cgroup_stats(&ctx.candidate.cgroup_stats, ctx.flatten, ctx.cgroup_key_map);
    }

    diff.host_psi_a = ctx.baseline.psi;
    diff.host_psi_b = ctx.candidate.psi;

    if ctx.group_by == GroupBy::All {
        diff.smaps_rollup_a = collect_smaps_rollup_hierarchical(
            ctx.baseline,
            ctx.opts.no_thread_normalize,
            ctx.flatten,
            ctx.cgroup_key_map,
        );
        diff.smaps_rollup_b = collect_smaps_rollup_hierarchical(
            ctx.candidate,
            ctx.opts.no_thread_normalize,
            ctx.flatten,
            ctx.cgroup_key_map,
        );
    } else {
        diff.smaps_rollup_a = collect_smaps_rollup(ctx.baseline, ctx.opts.no_thread_normalize);
        diff.smaps_rollup_b = collect_smaps_rollup(ctx.candidate, ctx.opts.no_thread_normalize);
    }

    remap_fudged_smaps(diff, fudged_key_pairs);

    diff.sched_ext_a = ctx.baseline.sched_ext.clone();
    diff.sched_ext_b = ctx.candidate.sched_ext.clone();
}

/// Compare two snapshots and produce a [`CtprofDiff`]. Sequences
/// the comparison phases in data-flow order: build the per-side
/// thread groups, emit matched + one-sided rows, fudge one-sided
/// cgroups together (producing the `fudged_key_pairs` consumed by
/// the uptime and enrichment phases), sort the one-sided lists,
/// fill uptime%, order the rows, and attach enrichment. The
/// fudged-pair `(bkey, ckey)` threading is the only cross-phase
/// data dependency: built by `apply_cgroup_fudge`, read by
/// `fill_uptime_pct` and `attach_enrichment`.
pub fn compare(
    baseline: &CtprofSnapshot,
    candidate: &CtprofSnapshot,
    opts: &CompareOptions,
) -> CtprofDiff {
    let flatten = compile_flatten_patterns(&opts.cgroup_flatten);
    let (group_by, cgroup_key_map, groups_a, groups_b) =
        build_compare_groups(baseline, candidate, opts, &flatten);

    // Compute per-snapshot "now" for lifetime calculation:
    // newest thread's start_time approximates capture time in ticks.
    let now_b = candidate
        .threads
        .iter()
        .map(|t| t.start_time_clock_ticks)
        .max()
        .unwrap_or(0);

    let ctx = CompareCtx {
        baseline,
        candidate,
        opts,
        group_by,
        flatten: &flatten,
        cgroup_key_map: cgroup_key_map.as_ref(),
        groups_a: &groups_a,
        groups_b: &groups_b,
        now_b,
    };

    let mut diff = CtprofDiff::default();

    emit_matched_rows(&mut diff, &ctx);

    // Content-based cgroup fudging: match one-sided groups by
    // thread population overlap when cgroup paths differ but
    // the workload is the same (e.g. service re-deployed to a
    // new cgroup path between snapshots). The returned pairs are
    // read by the uptime and enrichment phases below.
    let fudged_key_pairs = apply_cgroup_fudge(&mut diff, &ctx);

    diff.only_baseline.sort();
    diff.only_candidate.sort();

    fill_uptime_pct(&mut diff, &ctx, &fudged_key_pairs);

    sort_diff(&mut diff, &opts.sort_by);

    attach_enrichment(&mut diff, &ctx, &fudged_key_pairs);

    diff
}

pub fn flatten_cgroup_stats(
    stats: &BTreeMap<String, CgroupStats>,
    patterns: &[glob::Pattern],
    cgroup_key_map: Option<&BTreeMap<String, String>>,
) -> BTreeMap<String, CgroupStats> {
    // When multiple input paths flatten to the same key, the
    // merge is per-controller and per-field-class:
    //
    // - **Counters** (`usage_usec`, `nr_throttled`,
    //   `throttled_usec`, `pids.current`, `memory.events` map
    //   values, AND counter-shaped `memory.stat` keys
    //   (workingset_*, pgfault, pgmajfault, pgsteal_*, etc.)):
    //   saturating_add. Cumulative across the merged bucket.
    // - **Instantaneous values / gauges** (`memory.current` AND
    //   gauge-shaped `memory.stat` keys per
    //   [`MEMORY_STAT_GAUGE_KEYS`]: anon, file, slab,
    //   active_anon, etc.): max. Summing point-in-time pool
    //   sizes overstates the merged-bucket gauge. Counter vs
    //   gauge dispatch lives in [`merge_memory_stat`].
    // - **Limits** (`memory.max`, `memory.high`, `pids.max`,
    //   `cpu.max` quota, `cpu.weight`, `cpu.weight.nice`):
    //   max-for-limits via [`merge_max_option`]. `None` ("no
    //   limit") propagates when EITHER side is unbounded — the
    //   merged bucket is unbounded if any contributor is, since
    //   no synthesized cap reflects the actual kernel-enforced
    //   reality.
    // - **Floors** (`memory.low`, `memory.min`): min-for-floors
    //   via [`merge_min_option`]. `None` ("no floor")
    //   propagates when EITHER side has no floor — the merged
    //   bucket is only as protected as its weakest contributor,
    //   for the same reason. The literal "max" token (full
    //   protection) parses to `Some(u64::MAX)` per
    //   [`parse_floor_value`] and merges via min-for-floors,
    //   correctly yielding the smaller concrete floor when one
    //   contributor has full protection and another has a
    //   numeric floor.
    // - **PSI**: avg fields max-across, total_usec
    //   saturating_add (per [`merge_psi`]).
    //
    // When `cgroup_key_map` is provided (auto-normalize is on),
    // each post-flatten path is further mapped to its final
    // tightened key — so the enrichment table renders against
    // the same labels as thread groups. When absent, the
    // post-flatten path itself is the key (matches the legacy
    // behavior with glob-only flatten).
    // First-iteration-replace semantics: the first contributor
    // for a key is inserted verbatim (clone). Subsequent
    // contributors are merged in via the per-domain merge fns.
    // Using `or_default()` + merge here would synthesize a
    // CgroupStats whose `Option<u64>` limits are all None and
    // None-poison every `merge_max_option`/`merge_min_option`
    // call against the first real contributor — yielding `None`
    // for limits/floors even when every contributor has a
    // concrete value. The replace-on-first / merge-on-rest split
    // ensures None-poisoning fires only when contributors
    // genuinely disagree (one None, one Some), never when
    // merging the synthetic seed.
    let mut out: BTreeMap<String, CgroupStats> = BTreeMap::new();
    for (path, cs) in stats {
        let post_flatten = flatten_cgroup_path(path, patterns);
        let key = match cgroup_key_map.and_then(|m| m.get(&post_flatten)) {
            Some(k) => k.clone(),
            None => post_flatten,
        };
        match out.get_mut(&key) {
            None => {
                out.insert(key, cs.clone());
            }
            Some(agg) => {
                merge_cgroup_cpu(&mut agg.cpu, &cs.cpu);
                merge_cgroup_memory(&mut agg.memory, &cs.memory);
                merge_cgroup_pids(&mut agg.pids, &cs.pids);
                agg.psi = merge_psi(agg.psi, cs.psi);
            }
        }
    }
    out
}
