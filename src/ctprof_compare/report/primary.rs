//! Primary-metrics table emitter for [`super::write_diff`].
//!
//! Renders the per-thread primary table (52 non-taskstats rows
//! plus 34 taskstats genetlink rows). Two [`Section`] flags
//! share the table: [`Section::Primary`] gates the non-taskstats
//! rows; [`Section::TaskstatsDelay`] gates the genetlink rows.
//! Either enabled keeps the table open and the per-row gate
//! filters the matching subset.
//!
//! Three rendering branches — [`GroupBy::All`] (cgroup +
//! pcomm + comm hierarchy with ranked tree headings),
//! [`GroupBy::Cgroup`] (table-per-parent), and the
//! flat-table default — share the global column-width vector
//! computed by [`super::write_diff`] so column widths align
//! across primary and derived sub-tables.
//!
//! See [`super::write_diff`] for the orchestrator that calls
//! into [`write_primary_section`] before emitting derived,
//! smaps, cgroup, host-PSI, sched_ext, and the orphan / fudge
//! lists.
//!
//! Helper [`build_primary_hier`] is shared with
//! [`super::derived`] which performs the same cgroup +
//! pcomm grouping over `DerivedRow`s.

use std::collections::BTreeMap;
use std::fmt;

use super::super::CTPROF_METRICS;
use super::super::columns::{Column, Section};
use super::super::diff_types::{CtprofDiff, DiffRow};
use super::super::options::GroupBy;
use super::super::render::{
    cgroup_parent_leaf, color_diff_cell, colored_header_with_sort, render_diff_row_cells,
};
use super::super::runner::DisplayOptions;

/// Cgroup-tree heading color by depth — green at the root,
/// cyan at level 1, dark grey thereafter. Shared with
/// [`super::derived`] so the primary and derived hierarchies
/// match visually.
pub(super) fn depth_color(depth: usize) -> comfy_table::Color {
    match depth {
        0 => comfy_table::Color::Green,
        1 => comfy_table::Color::Cyan,
        _ => comfy_table::Color::DarkGrey,
    }
}

/// Render the `## Primary metrics` section (and its taskstats
/// subset when [`Section::TaskstatsDelay`] is enabled). The
/// outer gate keeps the table open while EITHER section is
/// enabled — `--sections taskstats-delay` alone still emits
/// the table containing only the 34 taskstats rows;
/// `--sections primary` alone emits the table containing only
/// the 52 non-taskstats rows; either combined or the empty
/// default ("all on") emits all rows.
pub(super) fn write_primary_section<W: fmt::Write>(
    w: &mut W,
    diff: &CtprofDiff,
    group_by: GroupBy,
    group_header: &'static str,
    columns: &[Column],
    display: &DisplayOptions,
    global_max_widths: &[u16],
) -> fmt::Result {
    if !display.is_section_enabled(Section::Primary)
        && !display.is_section_enabled(Section::TaskstatsDelay)
    {
        return Ok(());
    }

    // Filter rows first.
    let primary_rows: Vec<&DiffRow> = diff
        .rows
        .iter()
        .filter(|row| {
            if !display.is_metric_enabled(row.metric_name) {
                return false;
            }
            let metric = CTPROF_METRICS
                .iter()
                .find(|m| m.name == row.metric_name)
                .expect("metric_name comes from CTPROF_METRICS via build_row");
            display.is_section_enabled(metric.section)
        })
        .collect();

    if group_by == GroupBy::All {
        write_primary_all(
            w,
            &primary_rows,
            columns,
            display,
            global_max_widths,
            diff.sort_metric_name,
        )?;
    } else if group_by == GroupBy::Cgroup {
        write_primary_cgroup(w, &primary_rows, columns, display, diff.sort_metric_name)?;
    } else {
        write_primary_flat(
            w,
            &primary_rows,
            columns,
            display,
            group_header,
            diff.sort_metric_name,
        )?;
    }

    Ok(())
}

/// One row in the cgroup + pcomm + comm hierarchy used by the
/// `GroupBy::All` rendering of primary and (analogously) derived
/// rows. Stored references all point into the underlying
/// [`DiffRow::group_key`] string, which uses NUL bytes
/// (`\x00`) as cgroup / pcomm / comm field separators.
pub(super) struct HierRow<'a, R> {
    pub(super) cgroup: &'a str,
    pub(super) pcomm: &'a str,
    pub(super) comm: &'a str,
    pub(super) row: &'a R,
}

/// Build a sorted hierarchy from a slice of rows for
/// [`GroupBy::All`] rendering. Uses pre-sorted row index as
/// the rank within (cgroup, pcomm) and (cgroup) buckets so the
/// chosen `--sort-by` metric drives the tree heading order:
/// the cgroup containing the biggest mover floats to the top,
/// inside it the pcomm containing the biggest mover floats to
/// the top, and within that bucket rows render in sort order.
pub(super) fn build_primary_hier<'a, R>(
    rows: impl IntoIterator<Item = &'a R>,
    key_fn: impl Fn(&R) -> &str,
) -> Vec<HierRow<'a, R>>
where
    R: 'a,
{
    let mut hier: Vec<HierRow<'a, R>> = rows
        .into_iter()
        .map(|row| {
            let mut parts = key_fn(row).splitn(3, '\x00');
            let cgroup = parts.next().unwrap_or("");
            let pcomm = parts.next().unwrap_or("");
            let comm = parts.next().unwrap_or(pcomm);
            HierRow {
                cgroup,
                pcomm,
                comm,
                row,
            }
        })
        .collect();
    let row_rank: BTreeMap<*const R, usize> = hier
        .iter()
        .enumerate()
        .map(|(i, h)| (h.row as *const R, i))
        .collect();
    let mut leaf_rank: BTreeMap<(&str, &str), usize> = BTreeMap::new();
    let mut cg_rank: BTreeMap<&str, usize> = BTreeMap::new();
    for h in &hier {
        let rank = row_rank[&(h.row as *const R)];
        let le = leaf_rank.entry((h.cgroup, h.pcomm)).or_insert(usize::MAX);
        if rank < *le {
            *le = rank;
        }
        let ce = cg_rank.entry(h.cgroup).or_insert(usize::MAX);
        if rank < *ce {
            *ce = rank;
        }
    }
    hier.sort_by(|a, b| {
        let cga = cg_rank.get(a.cgroup).copied().unwrap_or(usize::MAX);
        let cgb = cg_rank.get(b.cgroup).copied().unwrap_or(usize::MAX);
        cga.cmp(&cgb)
            .then_with(|| {
                let sa = leaf_rank
                    .get(&(a.cgroup, a.pcomm))
                    .copied()
                    .unwrap_or(usize::MAX);
                let sb = leaf_rank
                    .get(&(b.cgroup, b.pcomm))
                    .copied()
                    .unwrap_or(usize::MAX);
                sa.cmp(&sb)
            })
            .then_with(|| {
                let ra = row_rank[&(a.row as *const R)];
                let rb = row_rank[&(b.row as *const R)];
                ra.cmp(&rb)
            })
    });
    hier
}

/// Emit a cgroup-segment heading row. The leaf segment of the
/// path renders at depth-colored bold; ancestor segments seen
/// in `last_segments` are skipped. Returns the new
/// `last_segments` vector for the caller to track.
pub(super) fn emit_cgroup_segments<'a>(
    table: &mut comfy_table::Table,
    cgroup: &'a str,
    last_segments: &[&'a str],
    columns: &[Column],
) -> Option<Vec<&'a str>> {
    let segments: Vec<&str> = cgroup.split('/').filter(|s| !s.is_empty()).collect();
    let common = segments
        .iter()
        .zip(last_segments.iter())
        .take_while(|(a, b)| a == b)
        .count();
    let cg_changed = common < last_segments.len() || segments.len() > last_segments.len();
    if cg_changed {
        for (depth, seg) in segments.iter().enumerate().skip(common) {
            let indent = "  ".repeat(depth);
            let label = format!("{indent}{seg}");
            let heading_cells: Vec<comfy_table::Cell> = columns
                .iter()
                .map(|c| {
                    if *c == Column::Group {
                        comfy_table::Cell::new(&label)
                            .fg(depth_color(depth))
                            .add_attribute(comfy_table::Attribute::Bold)
                    } else {
                        comfy_table::Cell::new("")
                    }
                })
                .collect();
            table.add_row(heading_cells);
        }
        Some(segments)
    } else {
        None
    }
}

/// Emit a pcomm heading row at the current cgroup-depth indent.
pub(super) fn emit_pcomm_heading(
    table: &mut comfy_table::Table,
    pcomm: &str,
    cg_depth: usize,
    columns: &[Column],
) {
    let indent = "  ".repeat(cg_depth);
    let label = format!("{indent}{pcomm}");
    let heading_cells: Vec<comfy_table::Cell> = columns
        .iter()
        .map(|c| {
            if *c == Column::Group {
                comfy_table::Cell::new(&label)
                    .fg(comfy_table::Color::White)
                    .add_attribute(comfy_table::Attribute::Bold)
            } else {
                comfy_table::Cell::new("")
            }
        })
        .collect();
    table.add_row(heading_cells);
}

fn write_primary_all<W: fmt::Write>(
    w: &mut W,
    primary_rows: &[&DiffRow],
    columns: &[Column],
    display: &DisplayOptions,
    global_max_widths: &[u16],
    sort_metric_name: Option<&'static str>,
) -> fmt::Result {
    // Sort + truncate BEFORE organizing into the cgroup
    // tree. primary_rows are already delta-sorted from
    // compare(). Apply the line limit here so the tree
    // only contains the top movers.
    //
    // Pin fudged rows: rows whose display_key starts
    // with `[fudged` represent N:1 merges that the
    // operator specifically asked the fudge stage to
    // produce. Truncating them silently buries the
    // most informative output. Partition first so
    // every fudged row survives the truncation, then
    // fill the remaining budget with the top non-fudged
    // movers in delta order.
    let (fudged_rows, normal_rows): (Vec<&DiffRow>, Vec<&DiffRow>) = primary_rows
        .iter()
        .copied()
        .partition(|r| r.display_key.starts_with("[fudged"));
    let limited_rows: Vec<&DiffRow> = if display.section_line_limit > 0 {
        let mut out: Vec<&DiffRow> = fudged_rows.clone();
        let budget = display.section_line_limit.saturating_sub(fudged_rows.len());
        out.extend(normal_rows.into_iter().take(budget));
        out
    } else {
        let mut out = fudged_rows;
        out.extend(normal_rows);
        out
    };

    let hier = build_primary_hier(limited_rows.iter().copied(), |row: &DiffRow| {
        row.group_key.as_str()
    });

    // Two-pass: first measure data-only widths, then build
    // the real table with heading rows constrained to those
    // widths so headings can't inflate columns.
    writeln!(w, "## Primary metrics")?;
    let mut last_segments: Vec<&str> = Vec::new();
    let mut last_pcomm = "";
    let mut table = display.new_constrained_table(global_max_widths);
    table.set_header(colored_header_with_sort(columns, "comm", sort_metric_name));

    for h in &hier {
        if let Some(new_segments) =
            emit_cgroup_segments(&mut table, h.cgroup, &last_segments, columns)
        {
            last_segments = new_segments;
            last_pcomm = "";
        }

        if h.pcomm != last_pcomm {
            emit_pcomm_heading(&mut table, h.pcomm, last_segments.len(), columns);
            last_pcomm = h.pcomm;
        }

        let mut string_cells = render_diff_row_cells(h.row, columns);
        if let Some(pos) = columns.iter().position(|c| *c == Column::Group) {
            let cg_depth = last_segments.len();
            // Preserve the [fudged] marker for rows that
            // came from the N:1 fudge merge — without
            // this preservation the comm-overwrite below
            // would silently strip the indicator and
            // fudged rows would render identically to
            // naturally-matched rows.
            let fudge_marker = if h.row.display_key.starts_with("[fudged") {
                h.row.display_key.as_str()
            } else {
                ""
            };
            // Insert a single space between marker and
            // comm; empty marker leaves the cell as
            // just `<indent>  <comm>` (matches
            // pre-fudge layout).
            let fudge_separator = if fudge_marker.is_empty() { "" } else { " " };
            string_cells[pos] = format!(
                "{}  {}{}{}",
                "  ".repeat(cg_depth + 1),
                fudge_marker,
                fudge_separator,
                h.comm,
            );
        }
        let cells: Vec<comfy_table::Cell> = string_cells
            .into_iter()
            .zip(columns.iter())
            .map(|(s, col)| {
                color_diff_cell(
                    s,
                    *col,
                    h.row.delta,
                    h.row.delta_pct,
                    h.row.uptime_pct,
                    h.row.sort_by_delta,
                )
            })
            .collect();
        table.add_row(cells);
    }
    writeln!(w, "{table}")?;
    Ok(())
}

fn write_primary_cgroup<W: fmt::Write>(
    w: &mut W,
    primary_rows: &[&DiffRow],
    columns: &[Column],
    display: &DisplayOptions,
    sort_metric_name: Option<&'static str>,
) -> fmt::Result {
    // Hierarchical cgroup rendering: group rows by parent
    // path, emit a sub-heading per parent, show only the
    // leaf segment in the group column.
    let mut by_parent: BTreeMap<&str, Vec<&DiffRow>> = BTreeMap::new();
    for row in primary_rows {
        let (parent, _) = cgroup_parent_leaf(&row.display_key);
        by_parent.entry(parent).or_default().push(row);
    }
    for (parent, rows) in &by_parent {
        writeln!(w)?;
        writeln!(w, "\x1b[1;32m## {parent}\x1b[0m")?;
        let mut table = display.new_table();
        table.set_header(colored_header_with_sort(
            columns,
            "cgroup",
            sort_metric_name,
        ));
        let cg_limit = if display.section_line_limit > 0 {
            &rows[..rows.len().min(display.section_line_limit)]
        } else {
            &rows[..]
        };
        for row in cg_limit {
            let (_, leaf) = cgroup_parent_leaf(&row.display_key);
            let mut string_cells = render_diff_row_cells(row, columns);
            // Replace group cell with leaf segment.
            if let Some(pos) = columns.iter().position(|c| *c == Column::Group) {
                string_cells[pos] = leaf.to_string();
            }
            let cells: Vec<comfy_table::Cell> = string_cells
                .into_iter()
                .zip(columns.iter())
                .map(|(s, col)| {
                    color_diff_cell(
                        s,
                        *col,
                        row.delta,
                        row.delta_pct,
                        row.uptime_pct,
                        row.sort_by_delta,
                    )
                })
                .collect();
            table.add_row(cells);
        }
        writeln!(w, "{table}")?;
    }
    Ok(())
}

fn write_primary_flat<W: fmt::Write>(
    w: &mut W,
    primary_rows: &[&DiffRow],
    columns: &[Column],
    display: &DisplayOptions,
    group_header: &'static str,
    sort_metric_name: Option<&'static str>,
) -> fmt::Result {
    writeln!(w, "## Primary metrics")?;
    let mut table = display.new_table();
    table.set_header(colored_header_with_sort(
        columns,
        group_header,
        sort_metric_name,
    ));
    let limit_iter = if display.section_line_limit > 0 {
        &primary_rows[..primary_rows.len().min(display.section_line_limit)]
    } else {
        primary_rows
    };
    for row in limit_iter {
        let string_cells = render_diff_row_cells(row, columns);
        let cells: Vec<comfy_table::Cell> = string_cells
            .into_iter()
            .zip(columns.iter())
            .map(|(s, col)| {
                color_diff_cell(
                    s,
                    *col,
                    row.delta,
                    row.delta_pct,
                    row.uptime_pct,
                    row.sort_by_delta,
                )
            })
            .collect();
        table.add_row(cells);
    }
    writeln!(w, "{table}")?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::super::aggregate::Aggregated;
    use super::super::super::scale::ScaleLadder;
    use super::*;

    /// Compare-side `Full`-format column vector
    /// (`[Group, Threads, Metric, Baseline, Candidate, Delta, Pct]`)
    /// — the resolved set the renderers iterate. `Column::Group`
    /// at index 0 is the heading slot every helper writes into.
    fn full_columns() -> Vec<Column> {
        super::super::super::columns::compare_columns_for(
            super::super::super::columns::DisplayFormat::Full,
        )
    }

    /// Build a `DiffRow` whose `group_key` carries the NUL-separated
    /// `cgroup\x00pcomm\x00comm` hierarchy that `build_primary_hier`
    /// splits on, with a real registry metric so the
    /// `CTPROF_METRICS` lookups in `render_diff_row_cells` /
    /// `write_primary_section` resolve. `display_key` is set
    /// separately so the cgroup / fudge-marker tests can drive the
    /// `[fudged …]`-prefix branch independently of `group_key`.
    fn mk_row(group_key: &str, display_key: &str, delta: f64) -> DiffRow {
        DiffRow {
            group_key: group_key.into(),
            thread_count_a: 1,
            thread_count_b: 1,
            metric_name: "run_time_ns",
            metric_ladder: ScaleLadder::Ns,
            baseline: Aggregated::Sum(100),
            candidate: Aggregated::Sum(200),
            delta: Some(delta),
            delta_pct: Some(delta / 100.0),
            display_key: display_key.into(),
            uptime_pct: None,
            sort_by_cell: None,
            sort_by_delta: None,
        }
    }

    /// `depth_color` is the shared cgroup-tree palette: green at
    /// the root, cyan at level 1, dark grey for every deeper level
    /// (and the saturating tail). Pin the exact `comfy_table::Color`
    /// at each boundary so a future palette edit can't silently
    /// recolor the primary / derived / smaps hierarchies that all
    /// route through this one function.
    #[test]
    fn depth_color_maps_each_depth_to_its_palette_entry() {
        use comfy_table::Color;
        let cases: &[(usize, Color)] = &[
            (0, Color::Green),
            (1, Color::Cyan),
            (2, Color::DarkGrey),
            (3, Color::DarkGrey),
            (7, Color::DarkGrey),
            (usize::MAX, Color::DarkGrey),
        ];
        for &(depth, expected) in cases {
            assert_eq!(
                depth_color(depth),
                expected,
                "depth {depth} must map to {expected:?}",
            );
        }
    }

    /// `build_primary_hier` splits each `group_key` on the NUL
    /// separator into exactly (cgroup, pcomm, comm). With all three
    /// fields present the comm field is the third token verbatim.
    #[test]
    fn build_primary_hier_splits_three_field_group_key() {
        let rows = [mk_row("/svc\x00alpha\x00alpha-w", "alpha-w", 10.0)];
        let hier = build_primary_hier(rows.iter(), |r: &DiffRow| r.group_key.as_str());
        assert_eq!(hier.len(), 1);
        assert_eq!(hier[0].cgroup, "/svc");
        assert_eq!(hier[0].pcomm, "alpha");
        assert_eq!(hier[0].comm, "alpha-w");
    }

    /// When the `group_key` carries only two NUL fields
    /// (`cgroup\x00pcomm`), the comm field falls back to the pcomm
    /// token (`parts.next().unwrap_or(pcomm)`). Pins the
    /// thread-leader case where comm == pcomm and the key is not
    /// triple-segmented.
    #[test]
    fn build_primary_hier_comm_defaults_to_pcomm_when_two_fields() {
        let rows = [mk_row("/svc\x00beta", "beta", 10.0)];
        let hier = build_primary_hier(rows.iter(), |r: &DiffRow| r.group_key.as_str());
        assert_eq!(hier[0].cgroup, "/svc");
        assert_eq!(hier[0].pcomm, "beta");
        assert_eq!(
            hier[0].comm, "beta",
            "comm must default to pcomm for a two-field key",
        );
    }

    /// An empty `group_key` yields empty cgroup / pcomm, and comm
    /// defaults to the (empty) pcomm. Boundary case for the
    /// `unwrap_or("")` / `unwrap_or(pcomm)` fallbacks.
    #[test]
    fn build_primary_hier_empty_key_yields_empty_fields() {
        let rows = [mk_row("", "", 10.0)];
        let hier = build_primary_hier(rows.iter(), |r: &DiffRow| r.group_key.as_str());
        assert_eq!(hier[0].cgroup, "");
        assert_eq!(hier[0].pcomm, "");
        assert_eq!(hier[0].comm, "");
    }

    /// `build_primary_hier` reorders rows so the cgroup containing
    /// the lowest-input-index (top mover) floats first, then within
    /// a cgroup the pcomm leaf containing the top mover, then rows
    /// in input order. Input is deliberately interleaved so a no-op
    /// pass-through would fail: the top mover (index 0) lives in
    /// `/cg-b`, so `/cg-b`'s whole block must precede `/cg-a`, and
    /// inside `/cg-b` the `p2` leaf (holding index 0) precedes `p1`.
    #[test]
    fn build_primary_hier_floats_top_mover_cgroup_and_leaf_first() {
        let rows = [
            // index 0: top mover — lives in /cg-b under pcomm p2.
            mk_row("/cg-b\x00p2\x00p2-w", "p2-w", 999.0),
            // index 1: /cg-a under p1.
            mk_row("/cg-a\x00p1\x00p1-w", "p1-w", 5.0),
            // index 2: /cg-b under p1 (same cgroup as index 0, but
            // a different, higher-ranked leaf).
            mk_row("/cg-b\x00p1\x00p1-w", "p1-w", 50.0),
        ];
        let hier = build_primary_hier(rows.iter(), |r: &DiffRow| r.group_key.as_str());
        let order: Vec<(&str, &str)> = hier.iter().map(|h| (h.cgroup, h.pcomm)).collect();
        assert_eq!(
            order,
            vec![("/cg-b", "p2"), ("/cg-b", "p1"), ("/cg-a", "p1")],
            "cgroup with the top mover (/cg-b) sorts first; within it \
             the leaf holding the top mover (p2) precedes p1; /cg-a sinks",
        );
    }

    /// Render a `comfy_table::Table` for assertion. The
    /// NOTHING-preset, force-no-tty table the helpers build emits
    /// whitespace-padded, ANSI-free rows under nextest's non-TTY
    /// stdout, so substring / indentation checks are exact.
    fn render(table: &comfy_table::Table) -> String {
        format!("{table}")
    }

    /// Count the leading spaces of the (single) heading-row line
    /// whose only content is `label`. A cgroup/pcomm heading row
    /// fills just the Group column with the (indented) label and
    /// leaves every other column blank, so the rendered line trims to
    /// exactly `label`. The NOTHING preset prefixes every cell with a
    /// fixed one-space pad, so a label at indent depth `d` (cell text
    /// `"  ".repeat(d) + label`) lands at `1 + 2 * d` leading spaces.
    /// Exact-`trim()` match (not `starts_with`) avoids colliding with
    /// the column-header line — e.g. label `c` is a prefix of the
    /// `comm` header. Panics if no such line exists, so a missing
    /// heading fails the test rather than silently reading as
    /// "indent 0".
    fn label_indent(out: &str, label: &str) -> usize {
        let line = out
            .lines()
            .find(|l| l.trim() == label)
            .unwrap_or_else(|| panic!("no heading row carries label `{label}`:\n{out}"));
        line.len() - line.trim_start().len()
    }

    /// `emit_cgroup_segments` from an empty `last_segments` emits one
    /// heading row per path segment, each indented by its depth
    /// (`"  ".repeat(depth)`), and returns the full segment vector
    /// for the caller to track. Two-segment path `/a/b` ⇒ `a` at
    /// depth 0 (1 leading pad space), `b` at depth 1 (1 pad + 2
    /// indent = 3 leading spaces).
    #[test]
    fn emit_cgroup_segments_emits_indented_headings_from_empty() {
        let cols = full_columns();
        let mut table = DisplayOptions::default().new_table();
        table.set_header(colored_header_with_sort(&cols, "comm", None));
        let returned = emit_cgroup_segments(&mut table, "/a/b", &[], &cols);
        assert_eq!(
            returned,
            Some(vec!["a", "b"]),
            "must return the full split segment vector",
        );
        let out = render(&table);
        assert_eq!(
            label_indent(&out, "a"),
            1,
            "depth-0 segment `a` must render flush-left (1 pad space):\n{out}",
        );
        assert_eq!(
            label_indent(&out, "b"),
            3,
            "depth-1 segment `b` must render at 1 pad + 2 indent spaces:\n{out}",
        );
    }

    /// When the new cgroup shares a common prefix with
    /// `last_segments`, only the diverging tail emits headings.
    /// `/a/b` → `/a/c`: the `a` heading is suppressed (common), only
    /// `c` (at depth 1) emits, and the returned vector is the full
    /// new path.
    #[test]
    fn emit_cgroup_segments_skips_common_prefix() {
        let cols = full_columns();
        let mut table = DisplayOptions::default().new_table();
        table.set_header(colored_header_with_sort(&cols, "comm", None));
        let returned = emit_cgroup_segments(&mut table, "/a/c", &["a", "b"], &cols);
        assert_eq!(returned, Some(vec!["a", "c"]));
        let out = render(&table);
        assert_eq!(
            label_indent(&out, "c"),
            3,
            "diverging tail `c` must emit at depth 1 (3 leading spaces):\n{out}",
        );
        // `a` was a common ancestor and must NOT re-emit as its own
        // heading row — no data line should carry it.
        assert!(
            !out.lines().any(|l| l.trim() == "a"),
            "common ancestor `a` must not re-emit a heading:\n{out}",
        );
    }

    /// Re-emitting the identical cgroup is a no-op: `common` equals
    /// `last_segments.len()` and the path is not deeper, so
    /// `cg_changed` is false. Returns `None` and adds zero rows
    /// (only the header line survives).
    #[test]
    fn emit_cgroup_segments_unchanged_returns_none_adds_no_rows() {
        let cols = full_columns();
        let mut table = DisplayOptions::default().new_table();
        table.set_header(colored_header_with_sort(&cols, "comm", None));
        let returned = emit_cgroup_segments(&mut table, "/a/b", &["a", "b"], &cols);
        assert_eq!(returned, None, "unchanged cgroup must return None");
        let out = render(&table);
        // The only non-empty line is the column header (`comm ...`);
        // no `a` / `b` heading rows were added.
        assert!(
            !out.lines().any(|l| l.trim() == "a" || l.trim() == "b"),
            "no heading rows may be added for an unchanged cgroup:\n{out}",
        );
    }

    /// Descending into a deeper path that extends the previous one
    /// emits the new deeper segments even though the shorter path is
    /// a full prefix: `/a` → `/a/b` has `common == last.len() == 1`
    /// but `segments.len() (2) > last.len() (1)`, so `cg_changed`
    /// fires and `b` emits at depth 1.
    #[test]
    fn emit_cgroup_segments_deepening_emits_new_tail() {
        let cols = full_columns();
        let mut table = DisplayOptions::default().new_table();
        table.set_header(colored_header_with_sort(&cols, "comm", None));
        let returned = emit_cgroup_segments(&mut table, "/a/b", &["a"], &cols);
        assert_eq!(returned, Some(vec!["a", "b"]));
        let out = render(&table);
        assert_eq!(
            label_indent(&out, "b"),
            3,
            "deepening must emit the new tail segment `b` at depth 1 \
             (3 leading spaces):\n{out}",
        );
        // The shared ancestor `a` was already emitted by a prior call
        // (common prefix) and must NOT re-emit here.
        assert!(
            !out.lines().any(|l| l.trim() == "a"),
            "common ancestor `a` must not re-emit on deepening:\n{out}",
        );
    }

    /// `emit_pcomm_heading` adds a single row carrying the pcomm
    /// label indented by the current cgroup depth
    /// (`"  ".repeat(cg_depth)`). At cg_depth 1 the pcomm `myproc`
    /// renders at 1 pad + 2 indent = 3 leading spaces.
    #[test]
    fn emit_pcomm_heading_indents_by_cgroup_depth() {
        let cols = full_columns();
        let mut table = DisplayOptions::default().new_table();
        table.set_header(colored_header_with_sort(&cols, "comm", None));
        emit_pcomm_heading(&mut table, "myproc", 1, &cols);
        let out = render(&table);
        assert_eq!(
            label_indent(&out, "myproc"),
            3,
            "pcomm `myproc` must render at cg_depth=1 (3 leading spaces):\n{out}",
        );
    }

    /// At cg_depth 0 the pcomm heading is flush-left (1 pad space).
    #[test]
    fn emit_pcomm_heading_flush_left_at_depth_zero() {
        let cols = full_columns();
        let mut table = DisplayOptions::default().new_table();
        table.set_header(colored_header_with_sort(&cols, "comm", None));
        emit_pcomm_heading(&mut table, "rootproc", 0, &cols);
        let out = render(&table);
        assert_eq!(
            label_indent(&out, "rootproc"),
            1,
            "pcomm `rootproc` must render flush-left at cg_depth=0 \
             (1 pad space):\n{out}",
        );
    }

    /// Build a minimal `CtprofDiff` carrying the given rows and a
    /// `run_time_ns` sort metric so the section renderers have a
    /// real registry-backed sort name to print in the header.
    fn diff_with(rows: Vec<DiffRow>) -> CtprofDiff {
        CtprofDiff {
            rows,
            ..Default::default()
        }
    }

    /// `write_primary_section` under `GroupBy::All` emits the
    /// `## Primary metrics` heading and renders the cgroup → pcomm →
    /// comm tree: the cgroup segment heading, the pcomm heading, and
    /// the deeply-indented comm leaf all surface for a single
    /// three-field row.
    #[test]
    fn write_primary_section_all_renders_full_hierarchy() {
        let diff = diff_with(vec![mk_row("/svc/api\x00gunicorn\x00gunicorn-w", "gunicorn-w", 10.0)]);
        let cols = full_columns();
        let mut out = String::new();
        write_primary_section(
            &mut out,
            &diff,
            GroupBy::All,
            "comm",
            &cols,
            &DisplayOptions::default(),
            &[],
        )
        .unwrap();
        assert!(
            out.starts_with("## Primary metrics"),
            "All-mode section must lead with the heading:\n{out}",
        );
        // Cgroup segments `svc` (depth 0) and `api` (depth 1).
        assert!(out.contains("svc"), "cgroup segment `svc` missing:\n{out}");
        assert!(out.contains("api"), "cgroup segment `api` missing:\n{out}");
        // pcomm heading and comm leaf both surface.
        assert!(
            out.contains("gunicorn"),
            "pcomm/comm `gunicorn` must surface in the tree:\n{out}",
        );
        // The metric row rendered (run_time_ns 100 -> 200, +10ns).
        assert!(
            out.contains("run_time_ns"),
            "metric row must render under the tree:\n{out}",
        );
    }

    /// `GroupBy::All` pins fudged rows outside the truncation budget:
    /// with `section_line_limit == 1` and one fudged + one normal
    /// row, the fudged row (whose `display_key` starts with
    /// `[fudged`) survives and renders its marker, while the normal
    /// row is dropped. Without the partition the top-N normal mover
    /// would consume the single-line budget and bury the fudged row.
    #[test]
    fn write_primary_section_all_pins_fudged_rows_past_limit() {
        // Normal row is the bigger mover (delta 999) so a naive
        // top-N truncation would keep IT and drop the fudged row.
        let normal = mk_row("/cg\x00normalproc\x00normalproc-w", "normalproc-w", 999.0);
        let fudged = mk_row(
            "/cg\x00fudgedproc\x00fudgedproc-w",
            "[fudged: leaf] fudgedproc-w",
            1.0,
        );
        let diff = diff_with(vec![fudged, normal]);
        let cols = full_columns();
        let display = DisplayOptions {
            section_line_limit: 1,
            ..Default::default()
        };
        let mut out = String::new();
        write_primary_section(&mut out, &diff, GroupBy::All, "comm", &cols, &display, &[]).unwrap();
        assert!(
            out.contains("[fudged: leaf]"),
            "fudged marker must survive truncation:\n{out}",
        );
        assert!(
            out.contains("fudgedproc"),
            "fudged comm must render alongside its marker:\n{out}",
        );
        assert!(
            !out.contains("normalproc"),
            "the normal row must be dropped once the budget is spent on \
             the pinned fudged row:\n{out}",
        );
    }

    /// `write_primary_section` short-circuits to `Ok(())` with no
    /// output when neither `Section::Primary` nor
    /// `Section::TaskstatsDelay` is enabled. A `--sections` filter
    /// naming only a non-primary section (here `derived`) leaves the
    /// outer gate closed.
    #[test]
    fn write_primary_section_suppressed_when_neither_section_enabled() {
        let diff = diff_with(vec![mk_row("/svc\x00p\x00p-w", "p-w", 10.0)]);
        let cols = full_columns();
        let display = DisplayOptions {
            sections: vec![super::super::super::columns::Section::Derived],
            ..Default::default()
        };
        let mut out = String::new();
        write_primary_section(&mut out, &diff, GroupBy::All, "comm", &cols, &display, &[]).unwrap();
        assert!(
            out.is_empty(),
            "primary section must emit nothing when its gate is closed:\n{out}",
        );
    }

    /// `--sections taskstats-delay` (Primary off, TaskstatsDelay on)
    /// keeps the outer gate OPEN but the per-row metric-section
    /// filter drops every `Section::Primary` row. So the table still
    /// renders with the heading, but a `run_time_ns`
    /// (`Section::Primary`) row is filtered out.
    #[test]
    fn write_primary_section_taskstats_only_keeps_table_drops_primary_rows() {
        let diff = diff_with(vec![mk_row("alpha\x00alpha\x00alpha-w", "alpha-w", 10.0)]);
        let cols = full_columns();
        let display = DisplayOptions {
            sections: vec![super::super::super::columns::Section::TaskstatsDelay],
            ..Default::default()
        };
        let mut out = String::new();
        // Flat-table branch (Pcomm) is simplest for the row-filter
        // assertion; the outer gate is shared with All/Cgroup.
        write_primary_section(
            &mut out,
            &diff,
            GroupBy::Pcomm,
            "pcomm",
            &cols,
            &display,
            &[],
        )
        .unwrap();
        assert!(
            out.contains("## Primary metrics"),
            "taskstats-delay alone must keep the table open:\n{out}",
        );
        assert!(
            !out.contains("run_time_ns"),
            "a Section::Primary metric row must be filtered out when only \
             taskstats-delay is enabled:\n{out}",
        );
    }

    /// `write_primary_cgroup` (the `GroupBy::Cgroup` path) groups
    /// rows by parent cgroup path, emits a bold `## <parent>`
    /// sub-heading per parent, and renders ONLY the leaf segment in
    /// the group column. Two rows under distinct parents produce two
    /// sub-headings; the leaf names appear, the full parent path does
    /// NOT bleed into the data cell.
    #[test]
    fn write_primary_section_cgroup_emits_parent_heading_and_leaf_cell() {
        // display_key drives both the parent grouping and the leaf
        // cell (cgroup_parent_leaf reads display_key).
        let r1 = mk_row(
            "/system.slice/foo.service",
            "/system.slice/foo.service",
            10.0,
        );
        let r2 = mk_row(
            "/system.slice/bar.service",
            "/system.slice/bar.service",
            20.0,
        );
        let diff = diff_with(vec![r1, r2]);
        let cols = full_columns();
        let mut out = String::new();
        write_primary_section(
            &mut out,
            &diff,
            GroupBy::Cgroup,
            "cgroup",
            &cols,
            &DisplayOptions::default(),
            &[],
        )
        .unwrap();
        // Single shared parent `/system.slice` → one sub-heading.
        assert!(
            out.contains("## /system.slice"),
            "parent sub-heading `## /system.slice` missing:\n{out}",
        );
        let heading_count = out.matches("## /system.slice").count();
        assert_eq!(
            heading_count, 1,
            "both rows share one parent → exactly one sub-heading, got {heading_count}:\n{out}",
        );
        // Leaf segments render in the group column.
        assert!(
            out.contains("foo.service") && out.contains("bar.service"),
            "leaf segments must render as the group cell:\n{out}",
        );
    }

    /// Two rows under DIFFERENT parents produce two distinct
    /// `## <parent>` sub-headings (BTreeMap-ordered: `/a` before
    /// `/b`). Pins the per-parent partitioning loop.
    #[test]
    fn write_primary_section_cgroup_one_heading_per_distinct_parent() {
        let r1 = mk_row("/a/x.service", "/a/x.service", 10.0);
        let r2 = mk_row("/b/y.service", "/b/y.service", 20.0);
        let diff = diff_with(vec![r1, r2]);
        let cols = full_columns();
        let mut out = String::new();
        write_primary_section(
            &mut out,
            &diff,
            GroupBy::Cgroup,
            "cgroup",
            &cols,
            &DisplayOptions::default(),
            &[],
        )
        .unwrap();
        let a_at = out.find("## /a").expect("`## /a` heading missing");
        let b_at = out.find("## /b").expect("`## /b` heading missing");
        assert!(
            a_at < b_at,
            "BTreeMap parent iteration must place `/a` before `/b`:\n{out}",
        );
    }
}