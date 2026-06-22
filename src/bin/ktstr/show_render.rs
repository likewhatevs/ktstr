//! Per-section render helpers for the `ktstr ctprof show`
//! single-snapshot view. Extracted verbatim from the parent
//! `ktstr` bin to keep that file under the size guard; `write_show`
//! (the orchestrator) stays in the parent and calls these via
//! `show_render::NAME`. The PSI siblings (`psi_resources`,
//! `format_psi_avg`, `host_psi_has_data`, `psi_resource_has_data`)
//! and the `ctprof_compare` alias are reached through `use super::*`.

use super::*;

/// Emit the `## Primary metrics` table. One row per (group,
/// metric) pair whose `metric.section` is enabled. Two sections
/// share the table — [`ctprof_compare::Section::Primary`] (52
/// non-taskstats rows) and [`ctprof_compare::Section::TaskstatsDelay`]
/// (34 taskstats genetlink rows). The outer gate keeps the table
/// open while EITHER section is enabled. Mirrors the
/// `ctprof_compare::write_diff` outer-gate semantics so
/// `--sections taskstats-delay` works identically across compare
/// and show.
#[allow(clippy::too_many_arguments)]
pub(super) fn write_show_primary<W: std::fmt::Write>(
    w: &mut W,
    display_options: &ctprof_compare::DisplayOptions,
    groups: &std::collections::BTreeMap<String, ctprof_compare::ThreadGroup>,
    group_order: &[&String],
    resolved_columns: &[ctprof_compare::Column],
    group_header: &'static str,
    group_by: ctprof_compare::GroupBy,
    no_thread_normalize: bool,
) -> std::fmt::Result {
    if display_options.is_section_enabled(ctprof_compare::Section::Primary)
        || display_options.is_section_enabled(ctprof_compare::Section::TaskstatsDelay)
    {
        writeln!(w, "## Primary metrics")?;
        let mut table = display_options.new_table();
        let header_row: Vec<&str> = resolved_columns
            .iter()
            .map(|c| c.header(group_header))
            .collect();
        table.set_header(header_row);

        for key in group_order {
            let group = &groups[*key];
            // Display key: pattern grouping under Comm or Pcomm
            // uses grex to turn the join-key skeleton into a regex
            // label; every other grouping (CommExact, Cgroup, or
            // either pattern axis under `--no-thread-normalize`)
            // renders the join key directly.
            let display_key = if matches!(
                group_by,
                ctprof_compare::GroupBy::Comm | ctprof_compare::GroupBy::Pcomm
            ) && !no_thread_normalize
            {
                ctprof_compare::pattern_display_label(key, &group.members)
            } else {
                (*key).clone()
            };
            for metric in ctprof_compare::CTPROF_METRICS {
                // `--metrics` filter: skip metrics not on the
                // operator-supplied allowlist. Empty allowlist
                // = no filter (default) per
                // `is_metric_enabled`'s default-empty contract.
                if !display_options.is_metric_enabled(metric.name) {
                    continue;
                }
                // Per-row section gate: skip metrics whose
                // `section` is not enabled by `--sections`. The
                // outer gate above keeps the table open while
                // either section is enabled; this inner gate
                // restricts which rows appear inside the table.
                if !display_options.is_section_enabled(metric.section) {
                    continue;
                }
                let Some(agg) = group.metrics.get(metric.name) else {
                    continue;
                };
                let metric_name = ctprof_compare::metric_display_name(metric).to_string();
                let value_cell = ctprof_compare::format_value_cell(agg, metric.rule.ladder());
                let tags_cell = ctprof_compare::metric_tags(metric);
                let cells: Vec<String> = resolved_columns
                    .iter()
                    .map(|c| match c {
                        ctprof_compare::Column::Group => display_key.clone(),
                        ctprof_compare::Column::Threads => group.thread_count.to_string(),
                        ctprof_compare::Column::Metric => metric_name.clone(),
                        ctprof_compare::Column::Value => value_cell.clone(),
                        ctprof_compare::Column::Tags => tags_cell.clone(),
                        ctprof_compare::Column::Uptime => "-".to_string(),
                        _ => "-".to_string(),
                    })
                    .collect();
                table.add_row(cells);
            }
        }
        writeln!(w, "{table}")?;
    }
    Ok(())
}

/// Emit the `## Derived metrics` table. One row per (group,
/// derivation) pair. Mirrors the `## Derived metrics` section
/// emitted by `ctprof_compare::write_diff` but adapted for the
/// single-snapshot show layout (no baseline/candidate split, one
/// value cell per row). The outer gate mirrors write_diff: open
/// the table when EITHER [`ctprof_compare::Section::Derived`] OR
/// [`ctprof_compare::Section::TaskstatsDelay`] is enabled. Per-row
/// gating keeps `--sections taskstats-delay` from leaking
/// non-taskstats derivations.
#[allow(clippy::too_many_arguments)]
pub(super) fn write_show_derived<W: std::fmt::Write>(
    w: &mut W,
    display_options: &ctprof_compare::DisplayOptions,
    groups: &std::collections::BTreeMap<String, ctprof_compare::ThreadGroup>,
    group_order: &[&String],
    resolved_columns: &[ctprof_compare::Column],
    group_header: &'static str,
    group_by: ctprof_compare::GroupBy,
    no_thread_normalize: bool,
) -> std::fmt::Result {
    if (display_options.is_section_enabled(ctprof_compare::Section::Derived)
        || display_options.is_section_enabled(ctprof_compare::Section::TaskstatsDelay))
        && !groups.is_empty()
    {
        let mut dt = display_options.new_table();
        let header_row: Vec<&str> = resolved_columns
            .iter()
            .map(|c| c.header(group_header))
            .collect();
        dt.set_header(header_row);
        // Iterate groups in the same order as the main table —
        // group_order has been computed once and applies to
        // every section emitted afterwards.
        for key in group_order {
            let group = &groups[*key];
            let display_key = if matches!(
                group_by,
                ctprof_compare::GroupBy::Comm | ctprof_compare::GroupBy::Pcomm
            ) && !no_thread_normalize
            {
                ctprof_compare::pattern_display_label(key, &group.members)
            } else {
                (*key).clone()
            };
            for d in ctprof_compare::CTPROF_DERIVED_METRICS {
                if !display_options.is_metric_enabled(d.name) {
                    continue;
                }
                // Per-row section gate: same shape as the primary
                // table loop. Skip derivations whose section is
                // not enabled.
                if !display_options.is_section_enabled(d.section) {
                    continue;
                }
                let value_cell = match (d.compute)(&group.metrics) {
                    Some(v) => ctprof_compare::format_derived_value_cell(v, d.ladder, d.is_ratio),
                    None => "-".to_string(),
                };
                let cells: Vec<String> = resolved_columns
                    .iter()
                    .map(|c| match c {
                        ctprof_compare::Column::Group => display_key.clone(),
                        ctprof_compare::Column::Threads => group.thread_count.to_string(),
                        ctprof_compare::Column::Metric => d.name.to_string(),
                        ctprof_compare::Column::Value => value_cell.clone(),
                        ctprof_compare::Column::Tags => String::new(),
                        ctprof_compare::Column::Uptime => "-".to_string(),
                        _ => "-".to_string(),
                    })
                    .collect();
                dt.add_row(ctprof_compare::color_derived_cells(cells));
            }
        }
        writeln!(w)?;
        writeln!(w, "## Derived metrics")?;
        writeln!(w, "{dt}")?;
    }
    Ok(())
}

/// Parent dispatcher for the per-cgroup secondary tables. Cgroup
/// grouping carries cgroup_stats enrichment alongside the
/// per-thread aggregates; these tables mirror compare's two-table
/// layout for `--group-by cgroup`. Early-returns when grouping is
/// not [`ctprof_compare::GroupBy::Cgroup`], when the snapshot
/// carries no `cgroup_stats`, or when the flattened `stats` bucket
/// is empty. The `--sections` filter is re-checked per sub-table
/// so a user can request `--sections pressure` and get only the
/// PSI rollups even though the cgroup-stats prefix is present in
/// the snapshot.
pub(super) fn write_show_cgroup_sections<W: std::fmt::Write>(
    w: &mut W,
    display_options: &ctprof_compare::DisplayOptions,
    snap: &ktstr::ctprof::CtprofSnapshot,
    group_by: ctprof_compare::GroupBy,
    flatten: &[glob::Pattern],
    cgroup_key_map: Option<&std::collections::BTreeMap<String, String>>,
) -> std::fmt::Result {
    if group_by != ctprof_compare::GroupBy::Cgroup || snap.cgroup_stats.is_empty() {
        return Ok(());
    }
    let stats = ctprof_compare::flatten_cgroup_stats(&snap.cgroup_stats, flatten, cgroup_key_map);
    if stats.is_empty() {
        return Ok(());
    }
    write_show_cgroup_stats_table(w, display_options, &stats)?;
    write_show_cgroup_limits_table(w, display_options, &stats)?;
    write_show_memory_stat_table(w, display_options, &stats)?;
    write_show_memory_events_table(w, display_options, &stats)?;
    write_show_cgroup_pressure_tables(w, display_options, &stats)?;
    Ok(())
}

/// Emit the headline per-cgroup counters table (cpu_usage_usec,
/// nr_throttled, throttled_usec, memory_current).
pub(super) fn write_show_cgroup_stats_table<W: std::fmt::Write>(
    w: &mut W,
    display_options: &ctprof_compare::DisplayOptions,
    stats: &std::collections::BTreeMap<String, ktstr::ctprof::CgroupStats>,
) -> std::fmt::Result {
    if display_options.is_section_enabled(ctprof_compare::Section::CgroupStats) {
        writeln!(w)?;
        let mut ct = display_options.new_table();
        ct.set_header(vec![
            "cgroup",
            "cpu_usage_usec",
            "nr_throttled",
            "throttled_usec",
            "memory_current",
        ]);
        // Route every scalar through `format_scaled_u64` so
        // the same auto-scale ladder that compare's enrichment
        // table uses applies here too — `7.500GiB` instead of
        // `8053063680`, `1.235s` instead of `1234567` µs.
        // Compare's table renders a baseline→candidate→delta
        // triple via `cgroup_cell`; show has a single snapshot
        // so each cell stands alone — `format_scaled_u64`
        // gives just the scaled value with no `→` arrow and
        // no `(+0…)` zero-delta tail. Units mirror compare's
        // call sites:
        //   cpu_usage_usec, throttled_usec → "µs"
        //   memory_current                  → "B"
        //   nr_throttled                    → "" (unitless count)
        for (key, s) in stats {
            ct.add_row(vec![
                key.clone(),
                ctprof_compare::format_scaled_u64(
                    s.cpu.usage_usec,
                    ctprof_compare::ScaleLadder::Us,
                ),
                ctprof_compare::format_scaled_u64(
                    s.cpu.nr_throttled,
                    ctprof_compare::ScaleLadder::Unitless,
                ),
                ctprof_compare::format_scaled_u64(
                    s.cpu.throttled_usec,
                    ctprof_compare::ScaleLadder::Us,
                ),
                ctprof_compare::format_scaled_u64(
                    s.memory.current,
                    ctprof_compare::ScaleLadder::Bytes,
                ),
            ]);
        }
        writeln!(w, "{ct}")?;
    }
    Ok(())
}

/// Emit the per-cgroup limits / knobs sub-table — operator-set
/// configuration that's typically static across a run but matters
/// when comparing two snapshots that straddle a deployment.
/// `cpu.max`, `cpu.weight`, `memory.max`, `memory.high`,
/// `pids.current/max` per `CgroupCpuStats` / `CgroupMemoryStats` /
/// `CgroupPidsStats`. Suppressed entirely when no cgroup in the
/// bucket exposes any of these (root cgroup, or a host without
/// pids/memory controllers enabled).
pub(super) fn write_show_cgroup_limits_table<W: std::fmt::Write>(
    w: &mut W,
    display_options: &ctprof_compare::DisplayOptions,
    stats: &std::collections::BTreeMap<String, ktstr::ctprof::CgroupStats>,
) -> std::fmt::Result {
    if display_options.is_section_enabled(ctprof_compare::Section::Limits)
        && stats.values().any(|s| {
            s.cpu.max_quota_us.is_some()
                || s.cpu.weight.is_some()
                || s.memory.max.is_some()
                || s.memory.high.is_some()
                || s.pids.current.is_some()
                || s.pids.max.is_some()
        })
    {
        writeln!(w)?;
        writeln!(w, "## Cgroup limits / knobs")?;
        let mut lt = display_options.new_table();
        lt.set_header(vec![
            "cgroup",
            "cpu.max",
            "cpu.weight",
            "memory.max",
            "memory.high",
            "pids.current",
            "pids.max",
        ]);
        for (key, s) in stats {
            // Per-row gate: skip rows where every column
            // is unset (the cgroup has no caps, no
            // weight set, no pids accounting). Without
            // this, a system-wide table can render N
            // empty rows for every host-controller
            // cgroup that doesn't expose any of these.
            let row_has_data = s.cpu.max_quota_us.is_some()
                || s.cpu.weight.is_some()
                || s.memory.max.is_some()
                || s.memory.high.is_some()
                || s.pids.current.is_some()
                || s.pids.max.is_some();
            if !row_has_data {
                continue;
            }
            lt.add_row(vec![
                key.clone(),
                ctprof_compare::format_cpu_max(s.cpu.max_quota_us, s.cpu.max_period_us),
                s.cpu
                    .weight
                    .map(|v| {
                        ctprof_compare::format_scaled_u64(v, ctprof_compare::ScaleLadder::Unitless)
                    })
                    .unwrap_or_else(|| "-".to_string()),
                ctprof_compare::format_optional_limit(
                    s.memory.max,
                    ctprof_compare::ScaleLadder::Bytes,
                ),
                ctprof_compare::format_optional_limit(
                    s.memory.high,
                    ctprof_compare::ScaleLadder::Bytes,
                ),
                s.pids
                    .current
                    .map(|v| {
                        ctprof_compare::format_scaled_u64(v, ctprof_compare::ScaleLadder::Unitless)
                    })
                    .unwrap_or_else(|| "-".to_string()),
                ctprof_compare::format_optional_limit(
                    s.pids.max,
                    ctprof_compare::ScaleLadder::Unitless,
                ),
            ]);
        }
        writeln!(w, "{lt}")?;
    }
    Ok(())
}

/// Emit the per-cgroup memory.stat sub-table — kernel-emitted
/// memory counters per cgroup. Up to 71 keys on a recent kernel.
/// Renders as one row per (cgroup, key) pair to keep column width
/// bounded; sorted by key for stable output. Suppressed when every
/// bucketed cgroup has an empty `memory.stat` map. Show-side
/// zero-suppression: a typical workload touches only a handful of
/// memory.stat keys, so rendering all 71 rows × N cgroups creates a
/// massive table dominated by zeros. Skip rows where the value is
/// exactly 0; if every key in a cgroup is zero, that cgroup
/// contributes no rows. The section still renders if any cgroup has
/// any non-zero key. This trims output ~10x for typical runs.
pub(super) fn write_show_memory_stat_table<W: std::fmt::Write>(
    w: &mut W,
    display_options: &ctprof_compare::DisplayOptions,
    stats: &std::collections::BTreeMap<String, ktstr::ctprof::CgroupStats>,
) -> std::fmt::Result {
    if display_options.is_section_enabled(ctprof_compare::Section::MemoryStat)
        && stats
            .values()
            .any(|s| s.memory.stat.values().any(|v| *v != 0))
    {
        writeln!(w)?;
        writeln!(w, "## memory.stat")?;
        let mut mt = display_options.new_table();
        mt.set_header(vec!["cgroup", "key", "value"]);
        for (key, s) in stats {
            for (stat_key, stat_value) in &s.memory.stat {
                if *stat_value == 0 {
                    continue;
                }
                mt.add_row(vec![
                    key.clone(),
                    stat_key.clone(),
                    ctprof_compare::format_scaled_u64(
                        *stat_value,
                        ctprof_compare::ScaleLadder::Unitless,
                    ),
                ]);
            }
        }
        writeln!(w, "{mt}")?;
    }
    Ok(())
}

/// Emit the per-cgroup memory.events sub-table — pressure-event
/// counters (low / high / max / oom / oom_kill etc.). Same
/// long-table layout as memory.stat with the same zero-row
/// suppression.
pub(super) fn write_show_memory_events_table<W: std::fmt::Write>(
    w: &mut W,
    display_options: &ctprof_compare::DisplayOptions,
    stats: &std::collections::BTreeMap<String, ktstr::ctprof::CgroupStats>,
) -> std::fmt::Result {
    if display_options.is_section_enabled(ctprof_compare::Section::MemoryEvents)
        && stats
            .values()
            .any(|s| s.memory.events.values().any(|v| *v != 0))
    {
        writeln!(w)?;
        writeln!(w, "## memory.events")?;
        let mut et = display_options.new_table();
        et.set_header(vec!["cgroup", "event", "count"]);
        for (key, s) in stats {
            for (event_key, event_value) in &s.memory.events {
                if *event_value == 0 {
                    continue;
                }
                et.add_row(vec![
                    key.clone(),
                    event_key.clone(),
                    ctprof_compare::format_scaled_u64(
                        *event_value,
                        ctprof_compare::ScaleLadder::Unitless,
                    ),
                ]);
            }
        }
        writeln!(w, "{et}")?;
    }
    Ok(())
}

/// Emit the per-cgroup PSI sub-tables — one per resource. Q8
/// ruling: per-resource sub-tables, all fields rendered (no
/// --verbose gate). Each resource shows a `some` row + a `full`
/// row with `avg10/avg60/avg300/total` columns. avg fields are
/// stored as centi-percent (0..=10099, see `PsiHalf` doc for the
/// kernel's EWMA rounding ceiling); render as `N.NN%` for
/// human-friendliness. total is microseconds; the auto_scale "µs"
/// ladder applies via `format_scaled_u64`. Per-resource
/// zero-suppression mirrors the compare-side write_diff path: skip
/// a resource sub-table when no cgroup in the bucket has any
/// non-zero data for it.
pub(super) fn write_show_cgroup_pressure_tables<W: std::fmt::Write>(
    w: &mut W,
    display_options: &ctprof_compare::DisplayOptions,
    stats: &std::collections::BTreeMap<String, ktstr::ctprof::CgroupStats>,
) -> std::fmt::Result {
    if display_options.is_section_enabled(ctprof_compare::Section::Pressure) {
        for (resource_name, accessor) in psi_resources() {
            let any_data = stats.values().any(|s| {
                let r = accessor(&s.psi);
                psi_resource_has_data(&r)
            });
            if !any_data {
                continue;
            }
            writeln!(w)?;
            writeln!(w, "## Pressure / {resource_name}")?;
            let mut pt = display_options.new_table();
            pt.set_header(vec!["cgroup", "row", "avg10", "avg60", "avg300", "total"]);
            for (key, s) in stats {
                let r = accessor(&s.psi);
                pt.add_row(vec![
                    key.clone(),
                    "some".into(),
                    format_psi_avg(r.some.avg10),
                    format_psi_avg(r.some.avg60),
                    format_psi_avg(r.some.avg300),
                    ctprof_compare::format_scaled_u64(
                        r.some.total_usec,
                        ctprof_compare::ScaleLadder::Us,
                    ),
                ]);
                pt.add_row(vec![
                    key.clone(),
                    "full".into(),
                    format_psi_avg(r.full.avg10),
                    format_psi_avg(r.full.avg60),
                    format_psi_avg(r.full.avg300),
                    ctprof_compare::format_scaled_u64(
                        r.full.total_usec,
                        ctprof_compare::ScaleLadder::Us,
                    ),
                ]);
            }
            writeln!(w, "{pt}")?;
        }
    }
    Ok(())
}

/// Emit host-level PSI — surface above the per-thread table when
/// any resource has nonzero data. Renders as four per-resource
/// sub-tables (cpu / memory / io / irq) with a `some`+`full` row
/// each, matching the per-cgroup layout.
pub(super) fn write_show_host_pressure<W: std::fmt::Write>(
    w: &mut W,
    display_options: &ctprof_compare::DisplayOptions,
    snap: &ktstr::ctprof::CtprofSnapshot,
) -> std::fmt::Result {
    if display_options.is_section_enabled(ctprof_compare::Section::HostPressure)
        && host_psi_has_data(&snap.psi)
    {
        for (resource_name, accessor) in psi_resources() {
            let r = accessor(&snap.psi);
            if !psi_resource_has_data(&r) {
                continue;
            }
            writeln!(w)?;
            writeln!(w, "## Host pressure / {resource_name}")?;
            let mut pt = display_options.new_table();
            pt.set_header(vec!["row", "avg10", "avg60", "avg300", "total"]);
            pt.add_row(vec![
                "some".into(),
                format_psi_avg(r.some.avg10),
                format_psi_avg(r.some.avg60),
                format_psi_avg(r.some.avg300),
                ctprof_compare::format_scaled_u64(
                    r.some.total_usec,
                    ctprof_compare::ScaleLadder::Us,
                ),
            ]);
            pt.add_row(vec![
                "full".into(),
                format_psi_avg(r.full.avg10),
                format_psi_avg(r.full.avg60),
                format_psi_avg(r.full.avg300),
                ctprof_compare::format_scaled_u64(
                    r.full.total_usec,
                    ctprof_compare::ScaleLadder::Us,
                ),
            ]);
            writeln!(w, "{pt}")?;
        }
    }
    Ok(())
}

/// Emit the per-process smaps_rollup sub-table. Routes through the
/// shared [`ctprof_compare::collect_smaps_rollup`] so the show-side
/// keying and aggregation match the compare-side exactly: under
/// default normalization the key is `pattern_key(&t.pcomm)` (tgid
/// dropped) and per-PID rows sharing the same pcomm pattern have
/// their byte counts field-summed; under `--no-thread-normalize`
/// the literal `pcomm[tgid]` shape is preserved so each PID stays
/// attributable. Process iteration order: descending by Rss,
/// tiebreak descending Pss, final tiebreak alphabetical (mirrors
/// the compare-side sort). Skip zero-valued entries per-row to keep
/// output bounded — Pss for an unmapped process is meaningfully
/// zero, but ShmemPmdMapped=0 etc. are noise rows. Suppressed when
/// no captured thread has a populated map (older kernels, stripped
/// permissions, synthetic fixtures).
///
/// Smaps keys can differ from primary-table Pcomm group keys for
/// singleton digit pcomms — smaps always normalizes (`worker-{N}`
/// even when only one PID matches), while the primary table reverts
/// singletons to the literal pcomm (`worker-7`); see
/// [`ctprof_compare::collect_smaps_rollup`] for the asymmetry and
/// its rationale (cross-snapshot diff joining vs. intra-snapshot
/// fleet aggregation).
///
/// Smaps keying is independent of `--group-by`: the keys reflect
/// the per-process pcomm pattern regardless of whether the operator
/// selected `cgroup`, `comm`, `pcomm`, or `comm-exact` for the
/// primary table. The smaps section reads pcomm directly off each
/// leader thread, not the post-grouping bucket key.
pub(super) fn write_show_smaps<W: std::fmt::Write>(
    w: &mut W,
    display_options: &ctprof_compare::DisplayOptions,
    snap: &ktstr::ctprof::CtprofSnapshot,
    no_thread_normalize: bool,
) -> std::fmt::Result {
    if display_options.is_section_enabled(ctprof_compare::Section::Smaps) {
        let smaps = ctprof_compare::collect_smaps_rollup(snap, no_thread_normalize);
        if !smaps.is_empty() {
            let mut process_keys: Vec<&String> = smaps.keys().collect();
            process_keys.sort_by(|a, b| {
                let max_for = |pkey: &&String, field: &str| -> u64 {
                    smaps
                        .get(*pkey)
                        .and_then(|m| m.get(field).copied())
                        .unwrap_or(0)
                };
                max_for(b, "Rss")
                    .cmp(&max_for(a, "Rss"))
                    .then_with(|| max_for(b, "Pss").cmp(&max_for(a, "Pss")))
                    .then_with(|| a.cmp(b))
            });
            // Pre-pass: every (process, key) pair with non-zero
            // value emits a row. Suppress the section header when
            // no rows would render (e.g. every value is zero).
            let any_row = process_keys.iter().any(|pkey| {
                smaps
                    .get(*pkey)
                    .map(|m| m.values().any(|v| *v != 0))
                    .unwrap_or(false)
            });
            if any_row {
                writeln!(w)?;
                writeln!(w, "## smaps_rollup")?;
                let mut st = display_options.new_table();
                st.set_header(vec!["process", "key", "value"]);
                for pkey in &process_keys {
                    // `process_keys` is built from `smaps.keys()`,
                    // so every entry resolves — index directly to
                    // make the invariant explicit.
                    let m = &smaps[*pkey];
                    for (key, bytes) in m {
                        if *bytes == 0 {
                            continue;
                        }
                        st.add_row(vec![
                            (*pkey).clone(),
                            key.clone(),
                            ctprof_compare::format_scaled_u64(
                                *bytes,
                                ctprof_compare::ScaleLadder::Bytes,
                            ),
                        ]);
                    }
                }
                writeln!(w, "{st}")?;
            }
        }
    }
    Ok(())
}

/// Emit the global sched_ext sysfs section. Suppressed when the
/// snapshot's `sched_ext` field is None (CONFIG_SCHED_CLASS_EXT=n
/// build, or sysfs directory absent). Single 5-row table mirroring
/// the kernel's exposed scx_global_attrs[] surface.
pub(super) fn write_show_sched_ext<W: std::fmt::Write>(
    w: &mut W,
    display_options: &ctprof_compare::DisplayOptions,
    snap: &ktstr::ctprof::CtprofSnapshot,
) -> std::fmt::Result {
    if display_options.is_section_enabled(ctprof_compare::Section::SchedExt)
        && let Some(scx) = &snap.sched_ext
    {
        writeln!(w)?;
        writeln!(w, "## sched_ext")?;
        let mut at = display_options.new_table();
        at.set_header(vec!["attr", "value"]);
        // state cell: render "-" when the file was unreadable
        // (empty string) so "no observation" stays visually
        // distinct from an actual scx_enable_state_str[] value.
        // Mirrors the compare-side rendering.
        let state_cell = if scx.state.is_empty() {
            "-".to_string()
        } else {
            scx.state.clone()
        };
        at.add_row(vec!["state".into(), state_cell]);
        at.add_row(vec![
            "switch_all".into(),
            ctprof_compare::format_scaled_u64(
                scx.switch_all,
                ctprof_compare::ScaleLadder::Unitless,
            ),
        ]);
        at.add_row(vec![
            "nr_rejected".into(),
            ctprof_compare::format_scaled_u64(
                scx.nr_rejected,
                ctprof_compare::ScaleLadder::Unitless,
            ),
        ]);
        at.add_row(vec![
            "hotplug_seq".into(),
            ctprof_compare::format_scaled_u64(
                scx.hotplug_seq,
                ctprof_compare::ScaleLadder::Unitless,
            ),
        ]);
        at.add_row(vec![
            "enable_seq".into(),
            ctprof_compare::format_scaled_u64(
                scx.enable_seq,
                ctprof_compare::ScaleLadder::Unitless,
            ),
        ]);
        writeln!(w, "{at}")?;
    }

    Ok(())
}
