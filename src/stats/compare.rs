use super::*;

/// One significant per-metric finding produced by [`compare_rows_by`].
///
/// `pairing_key` carries the dynamic identity the row pair joined
/// on — `scenario` plus every NON-slicing dimension's value. The
/// table renderer in [`compare_partitions`] decodes the key against
/// the slicing-dim list to produce a label like
/// `scenario/topology/work_type` (when topology + work_type are
/// pairing dims) or just `scenario` (when every other dim slices).
///
/// The `scenario` / `topology` / `work_type` fields carry the
/// matched row's values verbatim for legacy-shape consumers and
/// test fixtures that pre-date the dimensional-slicing refactor.
/// New code should read [`Finding::pairing_key`] directly so the
/// slicing-dim variation stays visible.
///
/// `metric` is the registry entry the comparison ran against;
/// consumers read polarity, display unit, and name through it
/// directly without re-looking up [`metric_def`].
#[derive(Debug, Clone, serde::Serialize)]
pub(crate) struct Finding {
    pub pairing_key: PairingKey,
    pub scenario: String,
    pub topology: String,
    pub work_type: String,
    pub metric: &'static MetricDef,
    pub val_a: f64,
    pub val_b: f64,
    pub delta: f64,
    pub kind: FindingKind,
}

/// How a significant (past-dual-gate) delta is classified. A metric
/// becomes a [`Finding`] only after clearing the
/// dual gate; this says which kind. `Informational` is for a
/// [`Polarity::Informational`](crate::test_support::Polarity::Informational)
/// metric (`MetricDef::classify_direction` => `None`): the change is
/// shown but is NEVER a regression or improvement and never affects the
/// exit code.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub(crate) enum FindingKind {
    Regression,
    Improvement,
    Informational,
}

/// A metric present on exactly ONE side of a paired (scenario,
/// topology, work_type) row — a coverage difference, not a perf delta.
///
/// `MetricDef::read` returns `None` for a metric absent on a row and
/// `Some(v)` (including `Some(0.0)`) when present, so an absent metric
/// is distinguishable from a genuine zero. A metric present on one side
/// and absent on the other is NOT a regression/improvement: it never
/// had a comparable baseline. Recording it here (never gated, never
/// counted in `regressions`/`improvements`/`informational`) surfaces
/// the appear/disappear-between-runs case instead of either silently
/// dropping it or — as the pre-fix `read().unwrap_or(0.0)` did —
/// mis-flagging it as a directional verdict against a coerced-zero
/// side: an unbounded relative change when the absent side is the
/// baseline (rel-gate INFINITY), a bounded one otherwise (e.g. 5 -> 0
/// gives rel ~1.0) — either clears the gate and yields a phantom
/// regression or improvement (the direction follows the metric's
/// polarity, so it inverts between LowerBetter and HigherBetter).
#[derive(Debug, Clone, serde::Serialize)]
pub(crate) struct CoverageDiff {
    pub pairing_key: PairingKey,
    pub scenario: String,
    pub topology: String,
    pub work_type: String,
    pub metric: &'static MetricDef,
    /// The side that HAS the metric; the other side is absent.
    pub present_side: ComparePartition,
    /// The present side's value (the absent side has none).
    pub value: f64,
}

/// Aggregate result of comparing two row sets via [`compare_rows_by`].
///
/// `regressions` and `improvements` count significant entries in
/// `findings`; `unchanged` counts metrics that fell below the dual
/// gate; `excluded_pairs` counts paired (scenario, topology, work_type)
/// row pairs where either side is excluded from regression math —
/// `fail`, `inconclusive`, `skip`, or an inverted `expected_failure`
/// run (which passes but carries failure-mode-dominated telemetry) all
/// route here. The field name captures "excluded from regression math"
/// rather than encoding any of the four excluded states, because the
/// per-side disposition (which side, which state) is recoverable from
/// the individual `GauntletRow::is_*` / `expected_failure` accessors
/// when the operator drills in.
/// `new_in_b`
/// counts B-side rows whose key has no match on the A side; the
/// converse is `removed_from_a`. The filter (when set) applies to
/// every counter, so excluded rows do not contribute.
#[derive(Debug, Clone, Default, serde::Serialize)]
pub(crate) struct CompareReport {
    pub regressions: u32,
    pub improvements: u32,
    /// Significant changes in `Polarity::Informational` metrics — shown
    /// but never gated (excluded from `regressions`/`improvements` and
    /// the exit code).
    pub informational: u32,
    pub unchanged: u32,
    pub excluded_pairs: u32,
    pub new_in_b: u32,
    pub removed_from_a: u32,
    pub findings: Vec<Finding>,
    /// Metrics present on exactly one side of a paired row (a metric
    /// appeared or disappeared between runs A and B). Never gated — not
    /// counted in `regressions`/`improvements`/`informational` and no
    /// effect on the exit code; surfaced so a coverage change is
    /// visible rather than silently dropped or mis-flagged as a
    /// regression from a zero baseline. See [`CoverageDiff`].
    pub coverage_diffs: Vec<CoverageDiff>,
}

/// Which side of an A/B comparison a row belongs to. Typed surface
/// for the per-phase rows so new code does not propagate the
/// `"A"` / `"B"` string-literal pattern the scalar-finding path
/// uses (kept as-is at the existing `"A"` / `"B"` call sites in this
/// module — `render_side_label`, `zero_match_diagnostic`).
#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Serialize)]
pub(crate) enum ComparePartition {
    A,
    B,
}

impl ComparePartition {
    /// Render the side as the same one-letter label
    /// `render_side_label` produces for the scalar table headers,
    /// so the noise per-phase coverage rows and the scalar findings
    /// table share the same operator-facing side identifier.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::A => "A",
            Self::B => "B",
        }
    }
}

/// Per-metric threshold policy driving `compare_rows` /
/// `compare_partitions`.
///
/// Resolution priority for a given metric's relative significance
/// threshold, highest first:
///
/// 1. `per_metric_percent[metric_name]` — explicit override for
///    this metric.
/// 2. `default_percent` — uniform override across every metric
///    not listed in the map (equivalent to the old `--threshold N`
///    CLI flag).
/// 3. The metric's built-in `default_rel` from the `METRICS`
///    registry — the "no policy" fallback.
///
/// Values in the struct are stored as PERCENT (e.g. `10.0` meaning
/// 10%), NOT fractions. [`Self::rel_threshold`] does the `/100.0`
/// conversion so every caller inside `compare_rows` reads a
/// fraction without re-deriving the division.
///
/// Note on the registry-fallback branch: the `default_rel` field
/// on `MetricDef` is already a FRACTION (e.g. `0.25` for 25%),
/// not a percent. `rel_threshold` returns it verbatim — it
/// does NOT divide by 100. Only the override branches
/// (per-metric map, `default_percent`) do the percent-to-fraction
/// conversion because their inputs are percents. This asymmetry
/// is deliberate so callers supplying CLI/file-based overrides
/// work in human-intuitive percent units while the registry
/// defaults (which already ship in fraction form) pass through
/// unchanged.
///
/// The struct is `serde::Serialize` / `serde::Deserialize` so
/// `cargo ktstr perf-delta --policy <path>` can load a
/// JSON-persisted policy file. Default construction produces an
/// empty policy that uses every registry default; [`Self::uniform`]
/// reproduces the old `--threshold N` behaviour without any
/// per-metric override plumbing at the call site.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ComparisonPolicy {
    /// Uniform override: when `Some(p)`, every metric whose name is
    /// NOT in [`Self::per_metric_percent`] uses `p / 100.0` as its
    /// relative threshold. `None` falls through to the registry
    /// `default_rel`. Stored as percent (e.g. `10.0` for 10%).
    pub default_percent: Option<f64>,
    /// Per-metric overrides keyed by metric name. Each value is a
    /// percent (e.g. `15.0` → 15%). An entry here takes precedence
    /// over both [`Self::default_percent`] and the registry
    /// `default_rel`.
    pub per_metric_percent: BTreeMap<String, f64>,
}

/// CLI-controlled rendering of the per-phase spread block in
/// `cargo ktstr perf-delta --noise-adjust`. Bundled as a struct
/// so the 5-flag clap surface threads through
/// `compare_partitions_noise` as a single positional rather than
/// five. Default value renders every phase / every metric / every
/// paired row — equivalent to passing no phase flags. All 5 flags
/// require `--noise-adjust` (per-phase output exists only there).
///
/// The flags compose via AND on independent axes (block-level
/// suppression × phase-id × row-significance), with three
/// mutex constraints enforced at CLI parse time:
///
/// - `--no-phases` excludes every other phase flag (the whole
///   block is suppressed; refining what to render is a
///   contradiction).
/// - `--phases-only` excludes `--no-phases` (same reason).
/// - `--steps-only` excludes `--phase` (one of them collapses
///   to a single bucket; the other suppresses BASELINE — both
///   together are confused phrasing).
///
/// The 5 flags trigger renderer behaviour ONLY — the
/// `--noise-adjust` per-phase pass always computes the full set
/// of `NoisePhaseFinding`s and coverage entries so programmatic
/// consumers see the unfiltered surface. Filtering is render-time
/// projection.
#[derive(Debug, Default, Clone)]
pub struct PhaseDisplayOptions {
    /// `--no-phases`: suppress the `--noise-adjust` per-phase
    /// spread block entirely. The aggregate spread table and
    /// footer render unchanged; the only effect is hiding the
    /// per-phase block (and its footer hint). Mutually exclusive
    /// with every other phase flag at CLI parse time.
    pub no_phases: bool,
    /// `--phases-only`: suppress the aggregate spread table and
    /// the host-context delta; render ONLY the per-phase spread
    /// block. Useful for narrowing investigation to a phase
    /// regression when the aggregate rollup is noise. Composes
    /// with `--steps-only`, `--phase`, and `--phase-threshold`.
    pub phases_only: bool,
    /// `--steps-only`: within the per-phase block, suppress
    /// the BASELINE bucket (`step_index == 0`); render only
    /// scenario Step buckets. Useful when the BASELINE settle
    /// window is dominated by scheduler startup transients.
    /// Mutually exclusive with `--phase`.
    pub steps_only: bool,
    /// `--phase <N>`: within the per-phase block, render only
    /// rows whose `step_index == N`. `0` selects BASELINE;
    /// `1..=N` selects scenario Step ordinals (1 → Step\[0\],
    /// 2 → Step\[1\], ...). Integer chosen over label so a label
    /// rename (`"Step[0]"` → `"Step:0"`) doesn't break operator
    /// CI invocations. Mutually exclusive with `--steps-only`.
    pub phase: Option<u16>,
    /// `--phase-threshold <PCT>`: render-side relative-spread
    /// gate for the `--noise-adjust` per-phase pass. Suppresses
    /// paired rows where `|delta-mean| / |a.mean| < PCT / 100.0`;
    /// a value from a ~zero baseline (`|a.mean| < ZERO_MEAN_EPS`)
    /// is an unbounded relative change and clears any finite
    /// threshold. `0.0` shows every paired row; absence falls
    /// through to the registry's per-metric `default_rel`.
    /// Independent from `--threshold` — the aggregate and
    /// per-phase passes have separate filters so an operator can
    /// widen the per-phase view without widening the aggregate
    /// view.
    pub phase_threshold: Option<f64>,
}

impl PhaseDisplayOptions {
    /// Resolve the per-phase relative threshold for a given
    /// metric. Returns the override fraction when
    /// `phase_threshold` is set, else falls through to the
    /// `ComparisonPolicy` resolution the scalar pass uses. The
    /// `metric_name` + `default_rel` shape mirrors
    /// [`ComparisonPolicy::rel_threshold`] so the two surfaces
    /// stay symmetric.
    pub fn rel_threshold(
        &self,
        policy: &ComparisonPolicy,
        metric_name: &str,
        default_rel: f64,
    ) -> f64 {
        match self.phase_threshold {
            Some(pct) => pct / 100.0,
            None => policy.rel_threshold(metric_name, default_rel),
        }
    }

    /// True when a phase row at the given `step_index` should
    /// render under the current display flags. Combines the two
    /// step-axis predicates (`--phase <N>` filter and
    /// `--steps-only` BASELINE-suppressor) into a single
    /// row-level decision the renderer applies uniformly across
    /// the `--noise-adjust` per-phase findings and coverage rows.
    /// Returns `true` when no relevant flag is set (default
    /// path: every step renders).
    pub fn matches_phase(&self, step_index: u16) -> bool {
        if let Some(want) = self.phase
            && step_index != want
        {
            return false;
        }
        if self.steps_only && step_index == 0 {
            return false;
        }
        true
    }

    /// The `--phase-threshold` relative-spread gate for a
    /// [`NoisePhaseFinding`]'s verdict: `|b.mean - a.mean| / |a.mean| >=
    /// phase_threshold / 100`. A move from a ~zero baseline (`|a.mean| <
    /// ZERO_MEAN_EPS`) is unbounded → shown; both ~zero carries no signal →
    /// filtered by any positive threshold. Returns `true` when no flag is set.
    /// The noise row carries per-side means (a [`NoiseVerdict`]), so the gate
    /// works on the mean delta rather than a single-run row delta.
    pub(crate) fn passes_noise_spread_threshold(&self, verdict: &NoiseVerdict) -> bool {
        let Some(pct) = self.phase_threshold else {
            return true;
        };
        let a = verdict.a.mean.abs();
        let delta = (verdict.b.mean - verdict.a.mean).abs();
        let rel = if a > ZERO_MEAN_EPS {
            delta / a
        } else if delta > ZERO_MEAN_EPS {
            f64::INFINITY
        } else {
            0.0
        };
        rel >= pct / 100.0
    }
}

impl ComparisonPolicy {
    /// Empty policy — every metric uses its `METRICS` registry
    /// default. Equivalent to the old `--threshold None` CLI path.
    pub fn new() -> Self {
        Self::default()
    }

    /// Uniform override: every metric uses `percent / 100.0`.
    /// Mirrors the old `--threshold N` CLI behaviour; the CLI
    /// dispatch at `cargo ktstr perf-delta --threshold N`
    /// constructs a policy via this constructor.
    pub fn uniform(percent: f64) -> Self {
        Self {
            default_percent: Some(percent),
            per_metric_percent: BTreeMap::new(),
        }
    }

    /// Load a JSON-persisted policy from a file. Errors propagate
    /// the read / parse reason as an `anyhow::Error` with the file
    /// path in the context chain so a malformed `--policy path.json`
    /// surfaces an actionable message rather than a generic
    /// "invalid JSON."
    ///
    /// Validates after parsing via [`Self::validate`]: rejects
    /// negative thresholds (a misconfigured 10 vs -10 would
    /// invert the dual-gate logic at the `.abs() >= rel_thresh`
    /// check and silently classify every metric as significant)
    /// and rejects per-metric keys not registered in `METRICS`
    /// (a typo like `"wrost_spread"` would otherwise be silently
    /// ignored — the key simply never matches during resolution
    /// and the metric falls through to `default_percent`).
    pub fn load_json(path: &std::path::Path) -> anyhow::Result<Self> {
        use anyhow::Context;
        let data = std::fs::read_to_string(path)
            .with_context(|| format!("read comparison policy from {}", path.display()))?;
        let policy: ComparisonPolicy = serde_json::from_str(&data)
            .with_context(|| format!("parse comparison policy from {}", path.display()))?;
        policy
            .validate()
            .with_context(|| format!("validate comparison policy from {}", path.display()))?;
        Ok(policy)
    }

    /// Structural validation separate from parsing so both the
    /// `load_json` path and programmatic constructors (after
    /// [`Self::uniform`] with a user-supplied percent) can share
    /// one set of invariants without re-implementing checks at
    /// each call site. Called automatically by [`Self::load_json`];
    /// CLI dispatch should call it after constructing via
    /// [`Self::uniform`] to catch `--threshold -10` at the
    /// entry point rather than deep inside `compare_rows` where
    /// the dual-gate math silently misbehaves.
    ///
    /// Rejects:
    /// - Negative `default_percent` (nonsensical — thresholds are
    ///   absolute-value comparisons).
    /// - Negative entries in `per_metric_percent`.
    /// - Per-metric keys not in the `METRICS` registry (silent
    ///   typos would otherwise fall through to `default_percent`
    ///   unnoticed).
    pub fn validate(&self) -> anyhow::Result<()> {
        if let Some(p) = self.default_percent
            && p < 0.0
        {
            anyhow::bail!(
                "ComparisonPolicy: default_percent must be non-negative; got {p}. \
                 Thresholds are absolute-value comparisons — a negative value \
                 would invert the dual-gate logic and silently classify every \
                 delta as significant."
            );
        }
        for (name, p) in &self.per_metric_percent {
            if !METRICS.iter().any(|m| m.name == name) {
                let known: Vec<&str> = METRICS.iter().map(|m| m.name).collect();
                anyhow::bail!(
                    "ComparisonPolicy: per_metric_percent contains unknown \
                     metric `{name}`. A typo in the key would silently fall \
                     through to default_percent. Registered metrics: {}",
                    known.join(", "),
                );
            }
            if *p < 0.0 {
                anyhow::bail!(
                    "ComparisonPolicy: per_metric_percent[{name:?}] must be \
                     non-negative; got {p}",
                );
            }
        }
        Ok(())
    }

    /// Resolve the mutually-exclusive `--threshold` / `--policy` CLI
    /// pair into a policy: `--threshold N` is sugar for a uniform N%
    /// default (validated for sign); `--policy PATH` loads a
    /// per-metric JSON policy; neither falls through to the registry
    /// defaults. Shared by every subcommand that accepts the pair
    /// (`perf-delta`) so the resolution rules — and
    /// the "exactly one of the two" contract — live in one place.
    ///
    /// Both flags set is rejected with an error. At the CLI call
    /// sites clap `conflicts_with` makes that unreachable, but this is
    /// a library entry point and must not panic on its inputs; the
    /// error is the defence-in-depth backstop.
    pub fn from_cli_flags(
        threshold: Option<f64>,
        policy: Option<&std::path::Path>,
    ) -> anyhow::Result<Self> {
        match (threshold, policy) {
            (Some(t), None) => {
                let p = Self::uniform(t);
                p.validate()?;
                Ok(p)
            }
            (None, Some(path)) => Self::load_json(path),
            (None, None) => Ok(Self::default()),
            (Some(_), Some(_)) => anyhow::bail!(
                "--threshold and --policy are mutually exclusive; use --policy \
                 for per-metric overrides"
            ),
        }
    }

    /// Resolve the relative threshold (as a fraction, e.g. `0.10`
    /// for 10%) for `metric_name` with `default_rel` as the
    /// registry-level fallback. Handles the percent→fraction
    /// conversion so `compare_rows_by` does not need to re-derive
    /// `p / 100.0` at every call site.
    pub fn rel_threshold(&self, metric_name: &str, default_rel: f64) -> f64 {
        if let Some(p) = self.per_metric_percent.get(metric_name) {
            p / 100.0
        } else if let Some(p) = self.default_percent {
            p / 100.0
        } else {
            default_rel
        }
    }
}

/// Compare two row sets metric-by-metric, parametrised on
/// `pairing_dims`.
///
/// Pure function: no I/O, no globals. Two rows pair iff their
/// [`PairingKey`] (scenario + every value for each dimension in
/// `pairing_dims`) is equal — this is the dimensional-slicing
/// pipeline's join primitive, with slicing dims EXCLUDED from
/// `pairing_dims` so rows on the A/B sides that differ on those
/// dims still pair as long as they agree on every non-slicing
/// dim. When `filter` is `Some(s)`, a row is included only if
/// `s` appears as a substring of the joined `"scenario topology
/// scheduler work_type"` string. The scheduler is
/// searchable via the substring filter but is not part of the
/// pairing key by default (only when `Dimension::Scheduler` is
/// in `pairing_dims`), so the same scenario+topology+work_type
/// pair compares correctly across different scheduler binaries
/// when the filter does not constrain it.
///
/// Row-pair accounting:
/// - B-side rows with no A-side match are counted in `new_in_b`.
/// - A-side rows with no B-side match are counted in `removed_from_a`
///   (a separate pass over `rows_a`).
/// - Paired rows where either side has `passed=false` are dropped
///   from the regression math and counted in `excluded_pairs`: a
///   failed scenario's metrics reflect the failure mode (short run,
///   stalled workload, missing samples), not the scheduler's
///   behavior.
///
/// The filter (when set) applies to every counter -- excluded rows
/// never reach the matching, pass, or metric stages.
///
/// `policy` carries the comparison thresholds. See
/// [`ComparisonPolicy`] for the resolution rules — per-metric
/// override → `default_percent` → registry `default_rel`. The
/// absolute gate always uses the metric's `default_abs`. A delta
/// must clear both gates to count as significant.
pub(crate) fn compare_rows_by(
    rows_a: &[GauntletRow],
    rows_b: &[GauntletRow],
    pairing_dims: &[Dimension],
    filter: Option<&str>,
    policy: &ComparisonPolicy,
) -> CompareReport {
    let mut report = CompareReport::default();

    // Build a HashMap<PairingKey, &GauntletRow> from rows_a once so
    // each row_b lookup is O(1) instead of O(rows_a). `or_insert_with`
    // preserves first-match semantics from the prior `rows_a.iter().find()`
    // call: on the rare path where two A-side rows share a key (the
    // averaging path folds same-key rows into one mean, so a
    // shared key is not normally reachable), the
    // earlier-iterated row wins.
    let mut a_by_key: HashMap<PairingKey, &GauntletRow> = HashMap::with_capacity(rows_a.len());
    for row_a in rows_a {
        let key = PairingKey::from_row(row_a, pairing_dims);
        a_by_key.entry(key).or_insert(row_a);
    }

    // Hoist the per-metric relative threshold out of the row×metric
    // loop. `policy.rel_threshold(m.name, m.default_rel)` is a pure
    // function of the metric — recomputing it for every row pair was
    // O(rows_b × METRICS) BTreeMap probes for nothing.
    let rel_thresholds: Vec<f64> = METRICS
        .iter()
        .map(|m| policy.rel_threshold(m.name, m.default_rel))
        .collect();
    // Same hoist for the render-suppression predicate: it is a pure
    // function of the metric (a small fixed-slice membership scan), so
    // probing it per (row_b x metric) re-ran the scan for nothing.
    let suppressed: Vec<bool> = METRICS
        .iter()
        .map(|m| is_render_suppressed_component(m.name))
        .collect();

    for row_b in rows_b {
        // Dynamic pairing key: scenario + every NON-slicing
        // dimension's value. Two rows pair iff their dynamic keys
        // match.
        let key_b = PairingKey::from_row(row_b, pairing_dims);
        if let Some(f) = filter {
            // Substring filter joins all identity-bearing fields —
            // including the SLICING dim values — so an operator
            // can narrow by any visible field via `-E`.
            let joined = format!(
                "{} {} {} {}",
                row_b.scenario, row_b.topology, row_b.scheduler, row_b.work_type,
            );
            if !joined.contains(f) {
                continue;
            }
        }
        let Some(&row_a) = a_by_key.get(&key_b) else {
            report.new_in_b += 1;
            continue;
        };

        // Drop from regression math when either side is a skip,
        // inconclusive, failure, or an inverted expected_failure run.
        // Skips carry no executed metrics
        // (the run didn't happen); inconclusive runs ran but lacked
        // signal to evaluate (zero-denominator ratio gate); failures
        // carry telemetry dominated by the failure mode (short run,
        // stalled workload), not the scheduler's behavior. An
        // expected_failure run has `passed == true` but its telemetry
        // is likewise failure-mode-dominated (short / stalled run), so
        // it is excluded despite passing —
        // comparing any of these against a real run produces
        // meaningless deltas.
        if row_a.is_fail()
            || row_b.is_fail()
            || row_a.is_inconclusive()
            || row_b.is_inconclusive()
            || row_a.is_skip()
            || row_b.is_skip()
            || row_a.expected_failure
            || row_b.expected_failure
        {
            report.excluded_pairs += 1;
            continue;
        }

        push_scalar_findings(
            &mut report,
            row_a,
            row_b,
            &key_b,
            &rel_thresholds,
            &suppressed,
        );
    }

    // Second pass: A-side rows whose key has no match on the B side.
    // Filter applies here too, so rows excluded by the filter never
    // count as removed. Build a HashSet<PairingKey> from rows_b once
    // so the existence check is O(1) per row_a; rows_b are inserted
    // unfiltered to preserve prior behaviour where a row_b that fails
    // the substring filter still suppresses a same-key row_a's
    // removed_from_a increment (the substring filter compares against
    // identity-bearing fields including slicing dims, so two rows
    // sharing a pairing key can disagree on filter membership).
    let b_keys: HashSet<PairingKey> = rows_b
        .iter()
        .map(|r| PairingKey::from_row(r, pairing_dims))
        .collect();
    for row_a in rows_a {
        let key_a = PairingKey::from_row(row_a, pairing_dims);
        if let Some(f) = filter {
            let joined = format!(
                "{} {} {} {}",
                row_a.scenario, row_a.topology, row_a.scheduler, row_a.work_type,
            );
            if !joined.contains(f) {
                continue;
            }
        }
        if !b_keys.contains(&key_a) {
            report.removed_from_a += 1;
        }
    }

    report
}

/// Append the scalar per-metric findings for one matched `(row_a,
/// row_b)` pair to `report`. Indexed by the `METRICS` enumerate
/// position: `rel_thresholds[i]` is the hoisted relative threshold
/// and `suppressed[i]` the hoisted render-suppression flag for the
/// i-th metric (both built once by [`compare_rows_by`] over the same
/// `METRICS` order). Bumps `report.unchanged` for sub-dual-gate
/// deltas and `report.regressions` / `report.improvements` per
/// metric polarity for the rest, pushing a [`Finding`] for each
/// significant delta.
fn push_scalar_findings(
    report: &mut CompareReport,
    row_a: &GauntletRow,
    row_b: &GauntletRow,
    key_b: &PairingKey,
    rel_thresholds: &[f64],
    suppressed: &[bool],
) {
    for (i, m) in METRICS.iter().enumerate() {
        // Rate components are internal plumbing — suppressed from compare
        // output (they remain in storage for the cross-run re-pool).
        if suppressed[i] {
            continue;
        }
        // `read` returns `None` for a metric absent on a row and `Some(v)`
        // (including `Some(0.0)`) when present, so absent is distinguishable
        // from a genuine zero. A metric present on exactly one side is a
        // coverage difference, NOT a delta: record it (never gated) and skip
        // the directional verdict. The pre-fix `unwrap_or(0.0)` coerced an
        // absent side to 0.0, producing a phantom directional verdict: when
        // the absent side was the baseline (val_a==0) the rel-gate's INFINITY
        // branch fired (unbounded), and otherwise rel_delta was bounded (e.g.
        // 5 -> 0 gives |(-5)/5|=1.0); either cleared both gates and yielded a
        // phantom regression or improvement (direction per the metric's
        // polarity) for a metric simply not captured on one side.
        let (val_a, val_b) = match (m.read(row_a), m.read(row_b)) {
            (Some(a), Some(b)) => (a, b),
            (None, None) => continue,
            (Some(a), None) => {
                report.coverage_diffs.push(CoverageDiff {
                    pairing_key: key_b.clone(),
                    scenario: row_b.scenario.clone(),
                    topology: row_b.topology.clone(),
                    work_type: row_b.work_type.clone(),
                    metric: m,
                    present_side: ComparePartition::A,
                    value: a,
                });
                continue;
            }
            (None, Some(b)) => {
                report.coverage_diffs.push(CoverageDiff {
                    pairing_key: key_b.clone(),
                    scenario: row_b.scenario.clone(),
                    topology: row_b.topology.clone(),
                    work_type: row_b.work_type.clone(),
                    metric: m,
                    present_side: ComparePartition::B,
                    value: b,
                });
                continue;
            }
        };
        // Both sides negligible (under ZERO_MEAN_EPS, the domain zero epsilon —
        // not f64::EPSILON, the machine ulp near 1.0): no signal, skip without
        // counting.
        if val_a.abs() < ZERO_MEAN_EPS && val_b.abs() < ZERO_MEAN_EPS {
            continue;
        }

        let rel_thresh = rel_thresholds[i];

        let delta = val_b - val_a;
        let rel_delta = if val_a.abs() > ZERO_MEAN_EPS {
            (delta / val_a).abs()
        } else {
            // A non-negligible value (val_b — the both-zero case is skipped
            // above) appearing from a ~zero baseline is an unbounded relative
            // change, not "unchanged". INFINITY clears the rel gate so the
            // absolute gate alone decides whether this delta is significant.
            f64::INFINITY
        };

        if delta.abs() < m.default_abs || rel_delta < rel_thresh {
            report.unchanged += 1;
            continue;
        }

        // Verdict: dual-gate already passed above (significant). An
        // Informational metric (classify_direction => None) is recorded
        // and displayed but NEVER counted as regression/improvement and
        // NEVER affects the exit code; a directional metric splits on its
        // polarity.
        let kind = match m.classify_direction() {
            None => {
                report.informational += 1;
                FindingKind::Informational
            }
            Some(higher_is_worse) => {
                let is_regression = if higher_is_worse {
                    delta > 0.0
                } else {
                    delta < 0.0
                };
                if is_regression {
                    report.regressions += 1;
                    FindingKind::Regression
                } else {
                    report.improvements += 1;
                    FindingKind::Improvement
                }
            }
        };
        report.findings.push(Finding {
            pairing_key: key_b.clone(),
            scenario: row_b.scenario.clone(),
            topology: row_b.topology.clone(),
            work_type: row_b.work_type.clone(),
            metric: m,
            val_a,
            val_b,
            delta,
            kind,
        });
    }
}

/// Emit a stderr warning naming any `-dirty` commit values present
/// in the partitioned rows so the operator knows the comparison
/// includes builds whose source tree may not match the recorded
/// HEAD.
///
/// Scans `commit` (project HEAD) and `kernel_commit` (kernel source
/// tree HEAD) on both sides' rows, dedupes the surviving values,
/// and emits one warning block listing each distinct dirty value
/// per dimension. Emits at most one block — silent when no row
/// carries a `-dirty` suffix on either dimension.
///
/// Dirty runs reuse the same sidecar filename as their clean HEAD
/// (the variant hash excludes `commit` / `kernel_commit` per
/// `crate::test_support::sidecar`), so re-running the same test
/// from a dirty tree overwrites the previous record. The warning
/// surfaces this so an operator can decide whether to commit the
/// working tree before re-running for a reproducible comparison.
///
/// Splits collection from emission via [`render_dirty_warning`] so
/// unit tests can pin the rendered text without trapping `stderr`.
fn warn_on_dirty_builds(rows_a: &[GauntletRow], rows_b: &[GauntletRow]) {
    if let Some(text) = render_dirty_warning(rows_a, rows_b) {
        eprint!("{text}");
    }
}

/// Emit the CPU-budget hazard warning for a comparison, if any.
/// Pure-render half is [`render_overcommit_warning`]; this only
/// `eprint!`s it, mirroring [`warn_on_dirty_builds`].
fn warn_on_overcommit(rows_a: &[GauntletRow], rows_b: &[GauntletRow], pairing_dims: &[Dimension]) {
    if let Some(text) = render_overcommit_warning(rows_a, rows_b, pairing_dims) {
        eprint!("{text}");
    }
}

/// Build the CPU-budget hazard warning from the filtered compare
/// sides, or `None` when neither hazard is present.
///
/// Two independent hazards, both read from [`GauntletRow::cpu_budget`]
/// / [`GauntletRow::vcpus`] — the consumers that make those fields
/// load-bearing on the compare path:
///
/// - OVERCOMMIT (`cpu_budget < vcpus`): the host time-sliced that
///   run's vCPU threads, so its wake-latency / off-CPU / run-delay
///   timing metrics are host-contention artifacts, not scheduler
///   signal (see [`crate::vmm::host_topology::overcommit_warning`]).
///   Always flagged when present on either side: comparing raw timing
///   from an overcommitted run is the silent-wrong-answer the budget
///   stamp exists to surface.
/// - MIXED BUDGET: a single pairing group on a side holds more than
///   one distinct non-skip budget. [`group_and_average_by`] folds rows
///   that share a full [`PairingKey`], so this is exactly the set the
///   averaging fold combines across budgets. It only arises
///   when [`Dimension::CpuBudget`] is NOT a pairing dim (the operator
///   sliced on cpu-budget, dropping it from the key); when it IS a
///   pairing dim, each budget keys its own group and is never folded.
///   Detection is per pairing group, NOT side-wide: two rows of
///   different scenarios (or any differing pairing dim) carry different
///   keys and never average, so a side merely spanning budgets across
///   distinct groups is not flagged.
///
/// Skip rows (budget 0 -> `None` in [`sidecar_to_row`]) carry no
/// budget identity and are ignored by both checks. Split from
/// emission so a unit test pins the text and the `None`-when-clean
/// polarity without trapping stderr, mirroring [`render_dirty_warning`].
pub(crate) fn render_overcommit_warning(
    rows_a: &[GauntletRow],
    rows_b: &[GauntletRow],
    pairing_dims: &[Dimension],
) -> Option<String> {
    use std::collections::BTreeSet;
    use std::fmt::Write;

    // Side-wide: the distinct overcommitted (budget, vcpus) pairs.
    let overcommitted = |rows: &[GauntletRow]| -> BTreeSet<(u32, u32)> {
        let mut over = BTreeSet::new();
        for r in rows {
            if let (Some(b), Some(v)) = (r.cpu_budget, r.vcpus)
                && b < v
            {
                over.insert((b, v));
            }
        }
        over
    };

    // Per pairing group: the union of budgets across groups that hold
    // >1 distinct budget — exactly the budgets the averaging fold
    // combines into one mean. Empty when CpuBudget is a pairing dim (each budget keys
    // its own group, so no group ever holds two).
    let cpu_budget_is_pairing = pairing_dims.contains(&Dimension::CpuBudget);
    let mixed_folded = |rows: &[GauntletRow]| -> BTreeSet<u32> {
        let mut folded = BTreeSet::new();
        if cpu_budget_is_pairing {
            return folded;
        }
        let mut by_key: std::collections::HashMap<PairingKey, BTreeSet<u32>> =
            std::collections::HashMap::new();
        for r in rows {
            if let Some(b) = r.cpu_budget {
                by_key
                    .entry(PairingKey::from_row(r, pairing_dims))
                    .or_default()
                    .insert(b);
            }
        }
        for budgets in by_key.values() {
            if budgets.len() > 1 {
                folded.extend(budgets.iter().copied());
            }
        }
        folded
    };

    let over_a = overcommitted(rows_a);
    let over_b = overcommitted(rows_b);
    let mixed_a = mixed_folded(rows_a);
    let mixed_b = mixed_folded(rows_b);

    if over_a.is_empty() && over_b.is_empty() && mixed_a.is_empty() && mixed_b.is_empty() {
        return None;
    }

    let any_overcommit = !over_a.is_empty() || !over_b.is_empty();
    let mut out = String::new();
    if any_overcommit {
        // Host time-slicing actually occurred -> raw timing is confounded.
        let _ = writeln!(
            out,
            "ktstr: WARNING: CPU-budget hazard in this comparison — a run was \
             host-overcommitted, so its guest-scheduler timing metrics \
             (wake-latency / off-CPU / run-delay) are host-contention-confounded. \
             Compare the overcommit-invariant worst_iterations_per_cpu_sec metric \
             instead of raw \
             timing."
        );
    } else {
        // Mixed budgets with NO overcommit: no host contention, the hazard is
        // collapsing two different measurement conditions into one number.
        let _ = writeln!(
            out,
            "ktstr: WARNING: CPU-budget hazard in this comparison — runs of \
             different CPU budgets share a pairing group, mixing two measurement \
             conditions. Slice with --cpu-budget, or compare the budget-invariant \
             worst_iterations_per_cpu_sec metric."
        );
    }
    let mut emit_side = |label: &str, over: &BTreeSet<(u32, u32)>, mixed: &BTreeSet<u32>| {
        if !over.is_empty() {
            let list = over
                .iter()
                .map(|(b, v)| format!("{b}/{v}"))
                .collect::<Vec<_>>()
                .join(", ");
            let _ = writeln!(
                out,
                "  side {label}: host-overcommitted run(s) [budget/vcpus]: {list}"
            );
        }
        if !mixed.is_empty() {
            let list = mixed
                .iter()
                .map(|b| b.to_string())
                .collect::<Vec<_>>()
                .join(", ");
            let _ = writeln!(
                out,
                "  side {label}: CPU budgets [{list}] share a pairing group — \
                 the average fold collapses them into one mean; slice with --cpu-budget so cross-budget runs are \
                 not compared under one key"
            );
        }
    };
    emit_side("A", &over_a, &mixed_a);
    emit_side("B", &over_b, &mixed_b);
    Some(out)
}

/// Build the dirty-builds warning block from row data.
///
/// Returns `None` when no row on either side carries a `-dirty`
/// suffix on either `commit` or `kernel_commit`. Otherwise returns
/// the full multi-line warning text — the body emitted to stderr by
/// [`warn_on_dirty_builds`] — terminated with a trailing newline so
/// the caller can `eprint!` it without further formatting.
///
/// Dimensions render in fixed order ("kernel source" before
/// "project") so the same dirty hashes always produce byte-identical
/// output across runs; values within each dimension are
/// `BTreeSet`-deduped so multiple rows sharing one dirty hash list
/// it once, and multiple distinct dirty hashes on one dimension list
/// in lex order.
pub(crate) fn render_dirty_warning(
    rows_a: &[GauntletRow],
    rows_b: &[GauntletRow],
) -> Option<String> {
    use std::collections::BTreeSet;
    use std::fmt::Write;

    let mut dirty_kernel: BTreeSet<&str> = BTreeSet::new();
    let mut dirty_project: BTreeSet<&str> = BTreeSet::new();
    for row in rows_a.iter().chain(rows_b.iter()) {
        // `ends_with` matches the producer contract: `detect_kernel_commit`
        // and `detect_project_commit` (src/test_support/sidecar/mod.rs) append
        // `-dirty` as a SUFFIX to the 7-char hex via
        // `format!("{short_hash}-dirty")`, so the dirty marker is
        // always tail-positioned. `contains` would also match a
        // hex hash that legitimately contains the substring `-dirty`
        // somewhere in the middle (impossible for the current
        // 7-char hex prefix, but a future commit-ish format change
        // would let a non-dirty value flag itself dirty under
        // `contains`).
        if let Some(c) = row.kernel_commit.as_deref()
            && c.ends_with("-dirty")
        {
            dirty_kernel.insert(c);
        }
        if let Some(c) = row.commit.as_deref()
            && c.ends_with("-dirty")
        {
            dirty_project.insert(c);
        }
    }

    if dirty_kernel.is_empty() && dirty_project.is_empty() {
        return None;
    }

    let mut out = String::new();
    writeln!(out, "warning: comparison includes dirty builds:").unwrap();
    for v in &dirty_kernel {
        writeln!(
            out,
            "  - kernel source: {v} (working tree may have changed since this run)"
        )
        .unwrap();
    }
    for v in &dirty_project {
        writeln!(
            out,
            "  - project: {v} (working tree may have changed since this run)"
        )
        .unwrap();
    }
    writeln!(
        out,
        "  Dirty runs overwrite previous results with the same HEAD."
    )
    .unwrap();
    writeln!(out, "  Commit changes for reproducible-ish comparisons.").unwrap();
    Some(out)
}

/// Render the actionable bail message emitted when one side's filter
/// matches zero sidecars in the pool.
///
/// Beyond the generic "check filters / run `cargo ktstr stats list`"
/// redirect, this helper inspects WHY the filter matched nothing and
/// adds three operator-actionable hints when applicable:
///
/// 1. **Dirty-form hint**: when the user passed
///    `--project-commit X` (or per-side / kernel-commit equivalent)
///    and the pool contains a row whose `commit` (or `kernel_commit`)
///    is `X-dirty`, append "Did you mean `--project-commit X-dirty`?".
///    A clean-vs-dirty mismatch is the single most common cause of a
///    false-zero on the commit dims — `detect_project_commit` /
///    `detect_kernel_commit` append `-dirty` whenever HEAD-vs-index
///    or index-vs-worktree changes are observed, so an operator who
///    expected `abcdef1` but the recorded value is `abcdef1-dirty`
///    sees no rows match without realizing why.
///
/// 2. **Unknown run-source hint**: when the user passed
///    `--run-source X` (or per-side equivalent) and `X` is NOT
///    among the distinct `run_source` values present in the pool,
///    append a hint listing the actual values seen. The schema is
///    deliberately extensible (`"benchmark"` and other future tags
///    are valid), so this is a hint rather than a hard validator —
///    but a typo (`--run-source loca` for `local`, or `--run-source CI`
///    for `ci` since the values are case-sensitive) is the most
///    common cause of a false-zero on the source dim, and listing
///    the distinct values present is more actionable than asking
///    the operator to consult the schema doc.
///
/// 3. **list-values redirect for commit dims**: when the user
///    populated any commit dimension (`project_commits` /
///    `kernel_commits`), suggest `cargo ktstr stats list-values`
///    specifically — that command emits the exact distinct values
///    present per dimension, which is more actionable than the
///    generic `stats list` which only shows top-level run keys.
///
/// `side` is `"A"` or `"B"` for diagnostic context. `filter` is the
/// per-side `RowFilter`. `rows` is the sidecar-derived row vec
/// (post-`sidecar_to_row` mapping, pre-filtering). `pool_len` is
/// the raw pool count for the "(N pooled)" diagnostic context.
pub(crate) fn zero_match_diagnostic(
    side: &str,
    filter: &RowFilter,
    rows: &[GauntletRow],
    pool_len: usize,
) -> String {
    let mut msg = format!(
        "perf-delta: {side} side filter matched 0 sidecars in \
         pool ({pool_len} pooled). Check the per-side filters or \
         confirm the runs exist with `cargo ktstr stats list`."
    );

    // Dirty-form hint per commit dimension. Only fires when a
    // populated filter value's `-dirty` form is in the pool.
    let mut dirty_hints: Vec<String> = Vec::new();
    for want in &filter.project_commits {
        let dirty = format!("{want}-dirty");
        let found = rows
            .iter()
            .any(|r| r.commit.as_deref() == Some(dirty.as_str()));
        if found {
            dirty_hints.push(format!(
                "no rows match `--project-commit {want}` but `{dirty}` exists in the pool — \
                 did you mean `--project-commit {dirty}`?"
            ));
        }
    }
    for want in &filter.kernel_commits {
        let dirty = format!("{want}-dirty");
        let found = rows
            .iter()
            .any(|r| r.kernel_commit.as_deref() == Some(dirty.as_str()));
        if found {
            dirty_hints.push(format!(
                "no rows match `--kernel-commit {want}` but `{dirty}` exists in the pool — \
                 did you mean `--kernel-commit {dirty}`?"
            ));
        }
    }
    for hint in dirty_hints {
        msg.push_str("\nhint: ");
        msg.push_str(&hint);
    }

    // Unknown-run-source hint. Fires when a `--run-source X` value
    // is not present in the pool — typo / wrong casing is the most
    // common cause. Schema is intentionally extensible (operators
    // can write `"benchmark"` etc.), so this is a hint not a hard
    // validator: the bail still fires, the operator still sees the
    // distinct values present, and the producer side is free to
    // emit any tag.
    if !filter.run_sources.is_empty() {
        let pool_run_sources: std::collections::BTreeSet<&str> = rows
            .iter()
            .filter_map(|r| r.run_source.as_deref())
            .collect();
        let unknowns: Vec<&str> = filter
            .run_sources
            .iter()
            .map(String::as_str)
            .filter(|want| !pool_run_sources.contains(*want))
            .collect();
        if !unknowns.is_empty() {
            let mut present: Vec<&str> = pool_run_sources.iter().copied().collect();
            present.sort_unstable();
            let unknown_list = unknowns
                .iter()
                .map(|s| format!("`{s}`"))
                .collect::<Vec<_>>()
                .join(", ");
            let present_list = if present.is_empty() {
                "(none — every row has `run_source: null`)".to_string()
            } else {
                present
                    .iter()
                    .map(|s| format!("`{s}`"))
                    .collect::<Vec<_>>()
                    .join(", ")
            };
            msg.push_str(&format!(
                "\nhint: --run-source {unknown_list} not found in pool; \
                 distinct values present: {present_list}. Values are \
                 case-sensitive (`ci` ≠ `CI`)."
            ));
        }
    }

    // Unknown-resolve-source hint. Mirrors the run_sources hint for the
    // scheduler-resolution-path dimension: fires when a `--resolve-source`
    // value is not among the resolve_sources present in the pool.
    if !filter.resolve_sources.is_empty() {
        let pool_resolve_sources: std::collections::BTreeSet<&str> = rows
            .iter()
            .filter_map(|r| r.resolve_source.as_deref())
            .collect();
        let unknowns: Vec<&str> = filter
            .resolve_sources
            .iter()
            .map(String::as_str)
            .filter(|want| !pool_resolve_sources.contains(*want))
            .collect();
        if !unknowns.is_empty() {
            let mut present: Vec<&str> = pool_resolve_sources.iter().copied().collect();
            present.sort_unstable();
            let unknown_list = unknowns
                .iter()
                .map(|s| format!("`{s}`"))
                .collect::<Vec<_>>()
                .join(", ");
            let present_list = if present.is_empty() {
                "(none — every row has `resolve_source: null`)".to_string()
            } else {
                present
                    .iter()
                    .map(|s| format!("`{s}`"))
                    .collect::<Vec<_>>()
                    .join(", ")
            };
            msg.push_str(&format!(
                "\nhint: --resolve-source {unknown_list} not found in pool; \
                 distinct values present: {present_list}. Values are \
                 case-sensitive (`auto_built` \u{2260} `Auto_Built`)."
            ));
        }
    }

    // Unknown-cpu-budget hint. Mirrors the run_sources hint for the
    // numeric budget dimension: fires when a `--cpu-budget` value is
    // not among the budgets present in the pool (the budgets render
    // canonically as decimal via `cpu_budget.to_string()`, so a
    // non-canonical input like `032` lists as not-found against the
    // canonical present set). Skip rows (`cpu_budget == None`) carry no
    // budget and are excluded.
    if !filter.cpu_budgets.is_empty() {
        let pool_budgets: std::collections::BTreeSet<u32> =
            rows.iter().filter_map(|r| r.cpu_budget).collect();
        let present_strs: std::collections::BTreeSet<String> =
            pool_budgets.iter().map(|b| b.to_string()).collect();
        let unknowns: Vec<&str> = filter
            .cpu_budgets
            .iter()
            .map(String::as_str)
            .filter(|want| !present_strs.contains(*want))
            .collect();
        if !unknowns.is_empty() {
            let unknown_list = unknowns
                .iter()
                .map(|s| format!("`{s}`"))
                .collect::<Vec<_>>()
                .join(", ");
            let present_list = if pool_budgets.is_empty() {
                "(none — every row is a skip with no recorded budget)".to_string()
            } else {
                pool_budgets
                    .iter()
                    .map(|b| format!("`{b}`"))
                    .collect::<Vec<_>>()
                    .join(", ")
            };
            msg.push_str(&format!(
                "\nhint: --cpu-budget {unknown_list} not found in pool; \
                 distinct budgets present: {present_list}."
            ));
        }
    }

    // list-values redirect: only fires when the operator narrowed
    // on a commit dimension. Generic case (no commit filter) keeps
    // the existing `stats list` redirect at the top of the message
    // — `list-values` would emit a long per-dimension dump that
    // isn't more actionable than `stats list` for a kernel/scheduler
    // /topology miss.
    let touched_commit_dim =
        !filter.project_commits.is_empty() || !filter.kernel_commits.is_empty();
    if touched_commit_dim {
        msg.push_str(
            "\nhint: run `cargo ktstr stats list-values` to see every \
             distinct commit value present in the pool — the specific \
             value the filter expected may not have a sidecar yet, or \
             may differ from what was recorded by \
             `detect_project_commit` / `detect_kernel_commit`.",
        );
    }
    msg
}

/// Resolved inputs for the `perf-delta` render phase.
///
/// Produced by [`prepare_partitioned_comparison`] — the validation,
/// pooling, partitioning, and averaging steps of [`compare_partitions`]
/// extracted into an owned bundle so the render half reads from one
/// destructure rather than a long flat prelude. Every field carries
/// the exact value the prior in-function prelude bound; the render
/// half computes labels and headers from these, then runs the four
/// print helpers.
struct PartitionedComparison {
    /// Dimensions on which `filter_a` differs from `filter_b` — the
    /// A/B contrast axes. Guaranteed non-empty (the empty case bails).
    slicing_dims: Vec<Dimension>,
    /// Dimensions NOT in `slicing_dims`, in canonical
    /// [`Dimension::ALL`] order — the join axes for pairing.
    pairing_dims: Vec<Dimension>,
    /// Every sidecar under the runs root (or `--dir` override).
    /// Guaranteed non-empty (the empty pool bails).
    pool: Vec<crate::test_support::SidecarResult>,
    /// `pool` converted to rows, same length and iteration order.
    rows: Vec<GauntletRow>,
    /// A-side rows fed to [`compare_rows_by`]: averaged mean rows
    /// under [`RowPrep::Averaged`], the raw filtered rows under
    /// [`RowPrep::PerRunPooled`].
    rows_a_for_compare: Vec<GauntletRow>,
    /// B-side counterpart of `rows_a_for_compare`.
    rows_b_for_compare: Vec<GauntletRow>,
    /// A-side averaged groups under [`RowPrep::Averaged`]; `None` under
    /// [`RowPrep::PerRunPooled`]. Drives the per-group pass-count block.
    avg_a: Option<Vec<AveragedGroup>>,
    /// B-side counterpart of `avg_a`.
    avg_b: Option<Vec<AveragedGroup>>,
    /// Post-typed-filter A-side contributor row count (pre-aggregation)
    /// — the "averaged across N runs" header numerator.
    pre_agg_a: usize,
    /// B-side counterpart of `pre_agg_a`.
    pre_agg_b: usize,
}

/// How [`prepare_partitioned_comparison`] folds each side's rows before
/// the compare half consumes them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RowPrep {
    /// Fold same-pairing-key rows on each side into one arithmetic-mean
    /// row — the default (averaging) compare behavior.
    Averaged,
    /// Keep every row INCLUDING duplicate pairing keys. The
    /// `perf-delta --noise-adjust`
    /// path: [`noise_findings`] groups the N runs per key per side into
    /// a spread, so N-per-key is the intended input, not an error.
    PerRunPooled,
}

/// Validate, pool, partition, and average the inputs for
/// [`compare_partitions`]. Returns the owned [`PartitionedComparison`]
/// bundle the render half destructures, or bails with the same
/// diagnostics in the same order as the original in-function prelude:
/// identical-rows gate, empty-pool gate, then the two zero-match
/// gates. The multi-dim slicing warning and the dirty-build /
/// overcommit warnings are emitted here so they precede the render
/// half's header lines, preserving output order.
fn prepare_partitioned_comparison(
    filter_a: &RowFilter,
    filter_b: &RowFilter,
    dir: Option<&std::path::Path>,
    prep: RowPrep,
) -> anyhow::Result<PartitionedComparison> {
    // Validation gate 1: there must be at least one dimension
    // on which filter_a differs from filter_b — otherwise the
    // operator hasn't expressed a contrast and the function has
    // nothing to compare. Empty slicing dims OR identical filters
    // are both rejected here with actionable diagnostics so the
    // user knows which knob to turn.
    let slicing_dims = derive_slicing_dims(filter_a, filter_b);
    if slicing_dims.is_empty() {
        anyhow::bail!(
            "perf-delta: A and B select identical rows. \
             Specify at least one per-side filter (e.g. \
             --a-kernel 6.14 --b-kernel 6.15) to define what \
             dimension separates the two sides."
        );
    }

    // Validation gate 2: warn (not error) when slicing on
    // multiple dimensions. The result is still well-defined —
    // the comparison joins on remaining pairing dims and
    // collapses the slicing-dim cross-product into a single
    // A/B contrast — but the operator is asking for a multi-axis
    // delta which is harder to interpret. The warning surfaces
    // the dim list so they can confirm the cohort shape.
    if slicing_dims.len() > 1 {
        let dim_names: Vec<&str> = slicing_dims.iter().map(|d| d.name()).collect();
        eprintln!(
            "warning: perf-delta: slicing on {n} dimensions [{dims}]; \
             results compress multiple axes into a single A/B contrast.",
            n = slicing_dims.len(),
            dims = dim_names.join(", "),
        );
    }

    // Pairing dims = every dimension NOT in the slicing-dim set,
    // in canonical [`Dimension::ALL`] order. The dynamic key
    // shape `(scenario, *pairing_dims)` matches whatever
    // dimensions are currently NOT being contrasted across A
    // and B.
    let pairing_dims = Dimension::pairing_dims(&slicing_dims);

    // Pool every sidecar under the runs root (or the operator's
    // --dir override) and convert to rows. The full-scan cost
    // is acceptable for the single-comparison-per-session
    // workflow.
    //
    // `--dir`-loaded sidecars get their `source` field rewritten
    // to `"archive"` via `apply_archive_source_override` before
    // row conversion. The producer-side `"local"` / `"ci"`
    // distinction is meaningful on the host that wrote the
    // sidecars; once the files have been copied off, the only
    // useful classification is "this came from elsewhere", which
    // is what `--run-source archive` queries for. Operators who need
    // to retain the producer-side distinction read from the
    // default root (no `--dir`) so values pass through untouched.
    let (root, override_archive) = match dir {
        Some(d) => (d.to_path_buf(), true),
        None => (crate::test_support::runs_root(), false),
    };
    let mut pool = crate::test_support::collect_pool(&root);
    if override_archive {
        crate::test_support::apply_archive_source_override(&mut pool);
    }
    if pool.is_empty() {
        anyhow::bail!(
            "perf-delta: no sidecar data found under {}. \
             Run `cargo ktstr test` to generate runs, or pass \
             --dir to point at an archived sidecar tree.",
            root.display(),
        );
    }
    let rows: Vec<GauntletRow> = pool.iter().map(sidecar_to_row).collect();

    // Partition: apply each side's filter to the same pool. A
    // row may match both sides (e.g. when project_commit is the
    // slicing dim, a row whose `project_commit` is in
    // `filter_a.project_commits` matches A but NOT B unless
    // `filter_b.project_commits` also contains it).
    let rows_a = apply_row_filters(&rows, filter_a);
    let rows_b = apply_row_filters(&rows, filter_b);
    if rows_a.is_empty() {
        anyhow::bail!(
            "{}",
            zero_match_diagnostic("A", filter_a, &rows, pool.len()),
        );
    }
    if rows_b.is_empty() {
        anyhow::bail!(
            "{}",
            zero_match_diagnostic("B", filter_b, &rows, pool.len()),
        );
    }

    warn_on_dirty_builds(&rows_a, &rows_b);
    warn_on_overcommit(&rows_a, &rows_b, &pairing_dims);

    let pre_agg_a = rows_a.len();
    let pre_agg_b = rows_b.len();

    // Fold each side's rows per the caller's [`RowPrep`]. `Averaged`
    // collapses same-pairing-key rows into one mean row (the default);
    // PerRunPooled keeps every row distinct, including duplicate
    // pairing keys (N-per-key is the intended noise-adjust input).
    let (rows_a_for_compare, rows_b_for_compare, avg_a, avg_b) = match prep {
        RowPrep::Averaged => {
            let avg_a = group_and_average_by(&rows_a, &pairing_dims);
            let avg_b = group_and_average_by(&rows_b, &pairing_dims);
            let a_rows: Vec<GauntletRow> = avg_a.iter().map(|r| r.row.clone()).collect();
            let b_rows: Vec<GauntletRow> = avg_b.iter().map(|r| r.row.clone()).collect();
            (a_rows, b_rows, Some(avg_a), Some(avg_b))
        }
        RowPrep::PerRunPooled => {
            // The noise-spread compare groups the N runs per pairing key
            // per side (see `noise_findings`), so N-per-key is the
            // intended input. Keep every row including duplicate
            // pairing keys — N-per-key is expected here, not an error.
            (rows_a, rows_b, None, None)
        }
    };

    Ok(PartitionedComparison {
        slicing_dims,
        pairing_dims,
        pool,
        rows,
        rows_a_for_compare,
        rows_b_for_compare,
        avg_a,
        avg_b,
        pre_agg_a,
        pre_agg_b,
    })
}

/// Warning for the SCALAR compare path when compared tests declare
/// `perf_delta_assertions` gates it does not evaluate. Declared gates are a
/// CI-gating perf assertion; gating on single-run scalar data would flip CI on
/// noise, so they are honored ONLY under `perf-delta --noise-adjust` (multi-run,
/// Welch + separation). Returning the message instead of a bare no-op keeps the
/// gate from silently disappearing on the default `perf-delta` invocation.
/// `None` when no compared test declares a gate. Pure (no I/O) so the
/// count/message is unit-testable. Counts DISTINCT scenarios carrying a gate.
pub(crate) fn scalar_declared_gate_warning(rows_b: &[GauntletRow]) -> Option<String> {
    let n = rows_b
        .iter()
        .filter(|r| !r.perf_delta_assertions.is_empty())
        .map(|r| r.scenario.as_str())
        .collect::<std::collections::BTreeSet<_>>()
        .len();
    (n > 0).then(|| {
        format!(
            "NOTE — {n} compared test(s) declare perf_delta_assertions gate(s) that this \
             scalar comparison does NOT evaluate. Declared gates are enforced only under \
             `perf-delta --noise-adjust N` (single-run scalar gating would flip CI on \
             noise); re-run with --noise-adjust to gate on them."
        )
    })
}

/// Compare two filter-defined partitions of the sidecar pool and
/// report regressions across slicing dimensions.
///
/// `filter_a` and `filter_b` are the per-side row filters that
/// define the A/B contrast. The dimensions on which the two
/// filters DIFFER are the SLICING dimensions; the dimensions on
/// which they AGREE (or on which both are unconstrained) are the
/// PAIRING dimensions. Two rows pair across the A/B sides iff
/// their dynamic [`PairingKey`] (scenario plus every pairing-dim
/// value) is equal — so the comparison naturally ignores
/// differences on the slicing axes (those ARE the contrast) and
/// joins on everything else.
///
/// `dir` overrides the default `runs_root()` for pool collection.
/// Pass `Some(path)` to compare archived sidecar trees copied off
/// a CI host; pass `None` to walk `target/ktstr/` (or
/// `CARGO_TARGET_DIR/ktstr/`).
///
/// Validation:
/// - Empty slicing-dim set (every dimension is identical between
///   A and B): bail with "specify at least one --a-X / --b-X to
///   define what to compare". This includes the no-flags-at-all
///   case (both filters are the empty default).
/// - Identical effective filters with at least one slicing dim is
///   a contradiction caught by clap-level construction; the
///   downstream check is "every value in filter_a appears in
///   filter_b on the same dim and vice versa." We catch that as
///   "A and B select identical rows" — symmetric to the empty
///   case.
/// - More than one slicing dimension prints a warning to stderr
///   ("warning: slicing on N dimensions; results compress
///   multiple axes into a single A/B contrast") but does NOT
///   bail — multi-dim slicing is a deliberate feature for
///   comparing e.g. (kernel A + scheduler A) against (kernel B +
///   scheduler B).
///
/// Groups every matching sidecar within each side by pairing key
/// and averages the metrics across the group.
///
/// Returns 1 when the confident regressions fail the operator gate — their
/// count reaches `gate.fail_threshold` (default 5) or a `gate.must_fail`
/// metric regressed; 0 otherwise. See [`gate_fails`] / [`GateOptions`].
pub fn compare_partitions(
    filter_a: &RowFilter,
    filter_b: &RowFilter,
    filter: Option<&str>,
    policy: &ComparisonPolicy,
    dir: Option<&std::path::Path>,
    gate: &GateOptions,
) -> anyhow::Result<i32> {
    let prepared = prepare_partitioned_comparison(filter_a, filter_b, dir, RowPrep::Averaged)?;
    let PartitionedComparison {
        slicing_dims,
        pairing_dims,
        pool,
        rows,
        rows_a_for_compare,
        rows_b_for_compare,
        avg_a,
        avg_b,
        pre_agg_a,
        pre_agg_b,
    } = &prepared;

    let report = compare_rows_by(
        rows_a_for_compare,
        rows_b_for_compare,
        pairing_dims,
        filter,
        policy,
    );

    // Side labels derive from the slicing dims' filter values.
    // Single slicing dim: e.g. "6.14.2" / "6.15.0". Multi: e.g.
    // "6.14.2:scx_rusty" / "6.15.0:scx_alpha". >3 values per dim:
    // collapse to "A"/"B" to keep column headers readable.
    let label_a = render_side_label(filter_a, slicing_dims, "A");
    let label_b = render_side_label(filter_b, slicing_dims, "B");

    // Header lines: name the slicing and pairing axes so the
    // operator can confirm the comparison shape at a glance.
    let slice_names: Vec<&str> = slicing_dims.iter().map(|d| d.name()).collect();
    let pair_names: Vec<&str> = pairing_dims.iter().map(|d| d.name()).collect();
    println!("slicing dimensions: {}", slice_names.join(", "));
    println!(
        "pairing on: scenario{}{}",
        if pair_names.is_empty() { "" } else { ", " },
        pair_names.join(", "),
    );
    // Declared gates are not evaluated on the scalar path — warn rather than
    // silently ignore (they are honored only under `perf-delta --noise-adjust`).
    if let Some(warning) = scalar_declared_gate_warning(rows_b_for_compare) {
        println!("{warning}");
    }

    println!(
        "{}",
        format_average_header(*pre_agg_a, *pre_agg_b, &label_a, &label_b)
    );

    // Scalar findings table.
    print_scalar_findings_table(&report, &label_a, &label_b);

    // Scalar summary block — regressions / improvements /
    // unchanged + skipped-failed + per-group pass counts +
    // new_in_b / removed_from_a.
    print_summary_block(&report, avg_a, avg_b, &label_a, &label_b);

    // Host-context delta. Same first-Some(host) baseline
    // `compare_partitions` uses — picking representative hosts
    // off the partitioned sidecars rather than the full pool so
    // the delta reflects what actually fed the comparison.
    print_host_context_delta(pool, rows, filter_a, filter_b, &label_a, &label_b);

    // Operator gate: the significance policy above decided WHICH deltas are
    // confident regressions; the gate decides HOW MANY / WHICH-NAMED of them
    // fail the run. (--all-metrics is a no-op on this path — the scalar table
    // already lists every changed metric and the unchanged COUNT prints in
    // the summary; it reveals stable/noisy rows on the --noise-adjust table.)
    let regressing: Vec<&str> = report
        .findings
        .iter()
        .filter(|f| f.kind == FindingKind::Regression)
        .map(|f| f.metric.name)
        .collect();
    Ok(if gate_fails(&regressing, gate) { 1 } else { 0 })
}

/// Operator-level perf-delta failure gate + render options, layered on top
/// of the per-metric significance policy (which decides WHICH deltas are
/// confident regressions). These decide HOW MANY / WHICH-NAMED confident
/// regressions fail the run, and whether stable/noisy rows render.
#[derive(Debug, Clone, Default)]
pub struct GateOptions {
    /// Fail iff at least this many confident regressions occur. `None`
    /// means 5 — a handful of regressions is tolerated as run-to-run noise
    /// and the run fails only once several metrics regress; pass
    /// `--fail-threshold 1` for fail-on-any. `Some(0)` disables the count
    /// gate entirely — only [`Self::must_fail`] can then fail the run.
    pub fail_threshold: Option<usize>,
    /// Metric registry names that fail the run if ANY of them regresses,
    /// regardless of the count gate (ORed on top). Caller-validated against
    /// the metric registry.
    pub must_fail: Vec<String>,
    /// Render every compared metric row (stable + noisy included) on the
    /// `--noise-adjust` table instead of only the meaningful ones.
    /// Display-only — never affects the gate.
    pub show_all: bool,
}

/// Decide whether a perf-delta run FAILS, given the registry names of the
/// confident regressions it found. Fails iff the count meets
/// [`GateOptions::fail_threshold`] (default 5; `Some(0)` disables the count
/// gate), OR any regressing metric is in [`GateOptions::must_fail`] (ORed
/// on top). The caller passes the CLASSIFIED regressions, so display-hidden
/// (suppressed) rows still feed the gate.
pub(crate) fn gate_fails(regressing_metrics: &[&str], gate: &GateOptions) -> bool {
    let n = gate.fail_threshold.unwrap_or(5);
    let fail_on_count = n >= 1 && regressing_metrics.len() >= n;
    let fail_on_must = !gate.must_fail.is_empty()
        && regressing_metrics
            .iter()
            .any(|m| gate.must_fail.iter().any(|w| w.as_str() == *m));
    fail_on_count || fail_on_must
}

#[cfg(test)]
mod gate_option_tests {
    use super::*;

    #[test]
    fn default_gate_fails_only_at_five_regressions() {
        // Default (None) threshold is 5: a handful of regressions is
        // tolerated as run-to-run noise; the run fails once >= 5 regress.
        let g = GateOptions::default();
        assert!(!gate_fails(&[], &g), "0 regressions passes");
        assert!(
            !gate_fails(&["a", "b", "c", "d"], &g),
            "4 < 5 passes under the default gate"
        );
        assert!(
            gate_fails(&["a", "b", "c", "d", "e"], &g),
            "5 >= 5 fails under the default gate"
        );
    }

    #[test]
    fn count_threshold_requires_n() {
        let g = GateOptions {
            fail_threshold: Some(3),
            ..Default::default()
        };
        assert!(!gate_fails(&["a", "b"], &g), "2 < 3 passes");
        assert!(gate_fails(&["a", "b", "c"], &g), "3 >= 3 fails");
    }

    #[test]
    fn zero_threshold_disables_count_gate() {
        let g = GateOptions {
            fail_threshold: Some(0),
            ..Default::default()
        };
        assert!(
            !gate_fails(&["a", "b", "c"], &g),
            "N=0 never fails on the count"
        );
    }

    #[test]
    fn must_fail_fails_regardless_of_count() {
        let g = GateOptions {
            fail_threshold: Some(0),
            must_fail: vec!["worst_p99_wake_latency_us".to_string()],
            ..Default::default()
        };
        assert!(
            gate_fails(&["worst_p99_wake_latency_us"], &g),
            "a must-fail metric regressing fails even with the count gate off"
        );
        assert!(
            !gate_fails(&["some_other_metric"], &g),
            "a non-must-fail regression does not fail with the count gate off"
        );
    }

    #[test]
    fn must_fail_is_ored_above_the_count() {
        let g = GateOptions {
            fail_threshold: Some(10),
            must_fail: vec!["avg_dsq_depth".to_string()],
            ..Default::default()
        };
        assert!(
            gate_fails(&["avg_dsq_depth"], &g),
            "must-fail fires even below the count threshold"
        );
    }
}

/// How a metric's noise-adjusted verdict classifies for the gate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum NoiseKind {
    /// SEPARATED (Welch or disjoint bands) AND MATERIAL in the worsening
    /// direction (per polarity) — the only kind that fails the gate.
    Regression,
    /// Separated AND material in the improving direction.
    Improvement,
    /// A side realized fewer than 2 samples, so variance / Welch are undefined —
    /// verdict untrustworthy; flagged but does NOT fail the gate. A merely
    /// high-spread side no longer lands here: `high_spread` is ADVISORY and
    /// annotates a reported verdict (e.g. `REGRESSION (noisy spread)`) rather
    /// than suppressing it — the fix for the old spread-gate signal inversion.
    Noisy,
    /// Separated + material change in a directionless (`Polarity::Informational`)
    /// metric — shown but NEVER fails the gate (no good/bad direction to regress).
    Informational,
    /// Not separated, or separated but immaterial (below the registry dual-gate),
    /// with both sides having >= 2 samples. Shown in the full metrics table so the
    /// operator sees every metric's A/B stats, but never gates. Emitted only when
    /// the caller passes `include_stable = true` (the render path); the gate-only
    /// path omits it.
    Stable,
}

/// One metric's noise-adjusted finding for a paired scenario.
pub(crate) struct NoiseFinding {
    /// The pairing-key label ("scenario" plus the pairing dims like
    /// topology/work_type, joined with "/"), so groups that share a scenario
    /// name but differ in a pairing dim render distinctly.
    pub pairing_label: String,
    pub metric: &'static MetricDef,
    pub verdict: NoiseVerdict,
    pub kind: NoiseKind,
    /// A whole-run [`crate::test_support::PerfDeltaAssertionRecord`] is declared
    /// on this (test, metric) — its `min_abs` / `max_regression_pct` /
    /// `direction` override the registry defaults in [`classify_noise`] (a
    /// declared value that is out of range on a corrupt/stale sidecar is
    /// rejected there and falls back to the registry default, but the row still
    /// CARRIES the declared gate). Rendered as a `(declared gate)` verdict
    /// annotation so the operator sees the author declared a gate here. `false`
    /// = no declared assertion; the registry defaults classified this row.
    pub gated_by_assertion: bool,
}

/// One metric's noise-adjusted finding for a matched PHASE of a paired
/// scenario — the per-phase (`--noise-adjust` + per-phase) analog of
/// [`NoiseFinding`], carrying the `step_index`/`label` the render prints.
/// Emitted only for a `(step_index, metric)` present on BOTH sides across
/// the runs. Per-phase SPREAD findings are render-only EXCEPT one carrying
/// an author-declared phase-scoped gate (`gated_by_assertion`), which DOES
/// contribute to the exit via [`NoiseReport::declared_phase_regressions`].
pub(crate) struct NoisePhaseFinding {
    /// The pairing-key label (see [`NoiseFinding::pairing_label`]).
    pub pairing_label: String,
    /// `0` = BASELINE, `1..=N` = scenario Step ordinals (framework convention).
    pub step_index: u16,
    /// Mirrors [`crate::assert::PhaseBucket::label`] (`"BASELINE"` / `"Step[k]"`).
    pub label: String,
    pub metric: &'static MetricDef,
    pub verdict: NoiseVerdict,
    pub kind: NoiseKind,
    /// A phase-scoped declared assertion (`phase == Some(step_index)`) drove the
    /// gate for this per-phase row — see [`NoiseFinding::gated_by_assertion`].
    pub gated_by_assertion: bool,
}

/// A per-phase metric present on only ONE side of the noise comparison —
/// either a metric absent in the other side's matched-`step_index` buckets,
/// or a whole one-sided `step_index`. Both collapse into one row because
/// a one-sided metric has no band on the absent side either way. Never gated;
/// surfaced so a coverage asymmetry is not silently dropped.
pub(crate) struct NoisePhaseCoverage {
    /// The pairing-key label (see [`NoiseFinding::pairing_label`]).
    pub pairing_label: String,
    pub step_index: u16,
    pub label: String,
    /// The one-sided metric, or `None` for a whole one-sided phase that
    /// carried no readable (non-suppressed) metric — surfaced (rendered with
    /// `—`) so the "phase fired but produced no data on one side" shape is not
    /// silently dropped.
    pub metric: Option<&'static MetricDef>,
    /// The side that carries the metric/phase; the other has none.
    pub present_side: ComparePartition,
    /// The present side's per-side mean across its runs, or `None` for the
    /// metric-less empty-phase-shape row.
    pub value: Option<f64>,
}

/// A declared [`crate::test_support::PerfDeltaAssertionRecord`] that was NEVER
/// evaluated — its metric resolves in the registry (guaranteed by
/// `KtstrTestEntry::validate`) but produced no comparable value in THIS
/// comparison, so [`classify_noise`] never saw it. The runtime analog of the
/// validate-time typo check: the author declared a perf gate that silently did
/// not fire (metric absent from the captured data, a Rate with a zero pooled
/// denominator, or — for a phase-scoped assertion — a step that no matched run
/// carried). Surfaced (never gated) so a silently-inert declared gate is not
/// mistaken for a passing one.
pub(crate) struct NoiseAssertionCoverage {
    /// The pairing-key label the unmatched gate was declared under (see
    /// [`NoiseFinding::pairing_label`]).
    pub pairing_label: String,
    /// The declared gate that never evaluated — carries its metric, phase
    /// scope, and overridden thresholds for the warning message.
    pub assertion: crate::test_support::PerfDeltaAssertionRecord,
}

/// Result of [`noise_findings`]: the per-(scenario, metric) aggregate findings,
/// the per-(scenario, phase, metric) findings + one-sided coverage rows, the
/// declared gates that never evaluated, and the number of scenarios paired
/// across both sides (for the footer).
pub(crate) struct NoiseReport {
    pub findings: Vec<NoiseFinding>,
    pub phase_findings: Vec<NoisePhaseFinding>,
    pub phase_coverage: Vec<NoisePhaseCoverage>,
    pub assertion_coverage: Vec<NoiseAssertionCoverage>,
    pub paired_scenarios: usize,
}

impl NoiseReport {
    /// Confident AGGREGATE regressions — the gate's exit basis. Per-phase
    /// findings are render-only and deliberately excluded (see
    /// [`Self::phase_regressions`]).
    pub fn regressions(&self) -> usize {
        self.findings
            .iter()
            .filter(|f| f.kind == NoiseKind::Regression)
            .count()
    }
    /// Aggregate (whole-run) regressions on a metric the test author explicitly
    /// DECLARED a whole-run [`crate::test_support::PerfDeltaAssertion`] for
    /// (`phase: None`, `gated_by_assertion`). Like a declared PHASE gate
    /// ([`Self::declared_phase_regressions`]), a declared whole-run gate is an
    /// author opt-in that ALWAYS contributes to the exit, orthogonal to the
    /// operator's count / must-fail gate — an UNdeclared aggregate regression
    /// stays subject to that count gate (`gate_fails`).
    pub fn declared_regressions(&self) -> usize {
        self.findings
            .iter()
            .filter(|f| f.kind == NoiseKind::Regression && f.gated_by_assertion)
            .count()
    }
    /// Aggregate metrics flagged untrustworthy — a side had < 2 usable runs.
    pub fn noisy(&self) -> usize {
        self.findings
            .iter()
            .filter(|f| f.kind == NoiseKind::Noisy)
            .count()
    }
    /// Confident AGGREGATE improvements — a metric that moved MATERIALLY in the
    /// better direction and cleared the significance test. Render-only (an
    /// improvement never gates); cited in the summary footer alongside
    /// [`Self::regressions`] so the reader sees the composite verdict (what
    /// regressed AND what improved), with [`Self::stable`] as the residual.
    pub fn improvements(&self) -> usize {
        self.findings
            .iter()
            .filter(|f| f.kind == NoiseKind::Improvement)
            .count()
    }
    /// Aggregate metrics that did NOT move confidently+materially — the expected
    /// common outcome, since the noise gate (Welch / disjoint-bands + material
    /// dual-gate) is conservative. Cited as a residual COUNT in the footer (the
    /// low-value majority), never enumerated.
    pub fn stable(&self) -> usize {
        self.findings
            .iter()
            .filter(|f| f.kind == NoiseKind::Stable)
            .count()
    }
    /// Aggregate metrics that changed but carry no better/worse polarity
    /// (registry `Informational`) — cited in the footer only when present.
    pub fn informational(&self) -> usize {
        self.findings
            .iter()
            .filter(|f| f.kind == NoiseKind::Informational)
            .count()
    }
    /// Confident per-phase regressions — for the footer + tests ONLY, never
    /// the exit basis (per-phase SPREAD is render-only). A per-phase regression the author explicitly DECLARED via a
    /// phase-scoped [`crate::test_support::PerfDeltaAssertion`] DOES gate — see
    /// [`Self::declared_phase_regressions`].
    pub fn phase_regressions(&self) -> usize {
        self.phase_findings
            .iter()
            .filter(|f| f.kind == NoiseKind::Regression)
            .count()
    }
    /// Per-phase regressions on a metric the test author explicitly DECLARED a
    /// phase-scoped gate for. Unlike the render-only per-phase spread pass, a
    /// declared phase gate is an opt-in: the author accepted the narrower
    /// phase-window noise, so it contributes to the perf-delta EXIT alongside
    /// the aggregate [`Self::regressions`]. A per-phase regression WITHOUT a
    /// declared gate stays render-only (a narrow-window flake must not flip CI
    /// red on its own).
    pub fn declared_phase_regressions(&self) -> usize {
        self.phase_findings
            .iter()
            .filter(|f| f.kind == NoiseKind::Regression && f.gated_by_assertion)
            .count()
    }
    /// Per-phase metrics flagged untrustworthy — a side had < 2 usable runs.
    pub fn phase_noisy(&self) -> usize {
        self.phase_findings
            .iter()
            .filter(|f| f.kind == NoiseKind::Noisy)
            .count()
    }
}

/// Row-level noise-adjusted compare — the testable core of
/// [`compare_partitions_noise`] (which wraps it with sidecar pooling + render),
/// mirroring how [`compare_rows_by`] underlies [`compare_partitions`]. Groups
/// each side's per-run rows by pairing key — EXCLUDING non-pass runs
/// (fail / skip / inconclusive / expected_failure), whose failure-mode
/// telemetry would corrupt the spread, exactly as [`compare_rows_by`] excludes
/// them — then per metric summarizes the spread and classifies via
/// [`noise_verdict`]. Returns one [`NoiseFinding`] per
/// (scenario, metric) that is changed, noisy, or — when `include_stable` is
/// true — unchanged-and-clean ([`NoiseKind::Stable`], shown in the full metrics
/// table but never gating). With `include_stable = false` only changed/noisy
/// metrics are returned (the gate-only path). Metrics with no signal on either
/// side (both means under [`ZERO_MEAN_EPS`]) and render-suppressed rate
/// components are omitted regardless of `include_stable`.
pub(crate) fn noise_findings(
    rows_a: &[GauntletRow],
    rows_b: &[GauntletRow],
    pairing_dims: &[Dimension],
    spread_threshold_pct: f64,
    include_stable: bool,
) -> NoiseReport {
    use std::collections::{BTreeMap, BTreeSet};
    let group = |rows: &[GauntletRow]| {
        let mut by_key: BTreeMap<PairingKey, Vec<GauntletRow>> = BTreeMap::new();
        for r in rows {
            // Exclude non-pass runs from the spread pool, mirroring the
            // scalar `compare_rows_by` and the averaged `group_and_average_by`:
            // a failed / inconclusive / skipped / expected_failure run's
            // telemetry is failure-mode-dominated (zeroed or an outlier), so
            // pooling it would corrupt the per-side [min, max] band and mean
            // and could produce a confident FALSE gate — the exact
            // silent-wrong-verdict class the noise mode exists to prevent.
            // A side reduced below 2 real samples by this filter is then
            // caught by `noise_verdict`'s n<2 insufficient_samples guard.
            if r.is_fail() || r.is_inconclusive() || r.is_skip() || r.expected_failure {
                continue;
            }
            // Derive this run's Rate metrics (e.g. the schedstat rates whose
            // components only materialize in ext_metrics at sidecar_to_row
            // time) so the spread reads them — the Averaged path derives via
            // group_and_average_by -> derive_rate_metrics; the per-run paths
            // must derive per row or every schedstat Rate is silently absent
            // from the verdict. Per-run derivation (each run's own num/den) is
            // the correct band semantics for run-to-run spread.
            let mut row = r.clone();
            crate::stats::metric::derive_rate_metrics(&mut row.ext_metrics);
            by_key
                .entry(PairingKey::from_row(&row, pairing_dims))
                .or_default()
                .push(row);
        }
        by_key
    };
    let a_by_key = group(rows_a);
    let b_by_key = group(rows_b);

    let mut findings = Vec::new();
    let mut phase_findings = Vec::new();
    let mut phase_coverage = Vec::new();
    let mut assertion_coverage = Vec::new();
    let mut paired_scenarios = 0usize;
    for (key, a_rows) in &a_by_key {
        let Some(b_rows) = b_by_key.get(key) else {
            continue;
        };
        paired_scenarios += 1;
        // Metrics whose gate WAS evaluated this group, split by scope: whole-run
        // (aggregate pass) and phase-scoped `(step_index, metric)` (per-phase
        // pass). Diffed against the declared assertions after both passes to
        // surface any declared gate that never fired (metric absent from the
        // captured data).
        let mut consulted: BTreeSet<&'static str> = BTreeSet::new();
        let mut consulted_phase: BTreeSet<(u16, &'static str)> = BTreeSet::new();
        // Label by the FULL pairing key (scenario + pairing dims like
        // topology/work_type), joined via pairing_key.0.join. noise_findings groups by the full pairing key,
        // so a scenario run on multiple topologies/work_types forms distinct
        // groups; labeling by scenario alone would render them indistinguishably.
        let pairing_label = key.0.join("/");
        for m in METRICS {
            if is_render_suppressed_component(m.name) {
                continue;
            }
            let a_vals: Vec<f64> = a_rows.iter().filter_map(|r| m.read(r)).collect();
            let b_vals: Vec<f64> = b_rows.iter().filter_map(|r| m.read(r)).collect();
            if a_vals.is_empty() || b_vals.is_empty() {
                continue;
            }
            // Rate metrics: the per-run ratios (a_vals/b_vals) give the
            // run-to-run band, but the compared centroid is the pooled
            // Σnum/Σden (duration-weighted) — the cross-run Rate value the
            // registry documents (metric.rs), so --noise-adjust and the scalar
            // averaging compare agree on a Rate's central value while the band still measures
            // per-run variability. Non-Rate metrics summarize their samples.
            let verdict = match m.kind {
                MetricKind::Rate {
                    numerator,
                    denominator,
                } => {
                    let pooled = |rows: &[GauntletRow]| -> Option<f64> {
                        // Sum num/den only over runs that carry BOTH components,
                        // so a run missing one cannot skew the pooled ratio (the
                        // per-run rate is undefined for it anyway).
                        let (num, den) = rows.iter().fold((0.0, 0.0), |(sn, sd), r| {
                            match (r.ext_metrics.get(numerator), r.ext_metrics.get(denominator)) {
                                (Some(n), Some(d)) => (sn + n, sd + d),
                                _ => (sn, sd),
                            }
                        });
                        (den != 0.0).then(|| num / den)
                    };
                    // A zero pooled denominator means no rate to compare on that
                    // side — skip (matches the per-run derivation guard).
                    let (Some(a_pooled), Some(b_pooled)) = (pooled(a_rows), pooled(b_rows)) else {
                        continue;
                    };
                    noise_verdict_from(
                        SideSummary::of(&a_vals).with_pooled_mean(a_pooled),
                        SideSummary::of(&b_vals).with_pooled_mean(b_pooled),
                        spread_threshold_pct,
                    )
                }
                _ => noise_verdict(&a_vals, &b_vals, spread_threshold_pct),
            };
            // This metric produced a comparable verdict, so a declared gate on
            // it WAS evaluated — record it so the post-loop diff does not flag it
            // as never-evaluated. Recorded here (not at the empty-values / zero-
            // denominator `continue`s above) so those absent-data cases DO
            // surface as unmatched declared gates.
            consulted.insert(m.name);
            // Aggregate (whole-run) declared assertion for this test+metric
            // (phase: None). HEAD (B) side is authoritative on the declaration.
            let assertion = b_rows.first().and_then(|r| {
                r.perf_delta_assertions
                    .iter()
                    .find(|x| x.metric == m.name && x.phase.is_none())
            });
            let gated_by_assertion = assertion.is_some();
            let Some(kind) = classify_noise(&verdict, m, assertion, include_stable) else {
                continue;
            };
            findings.push(NoiseFinding {
                pairing_label: pairing_label.clone(),
                metric: m,
                verdict,
                kind,
                gated_by_assertion,
            });
        }
        // Per-phase sub-pass: mirror the aggregate spread for each matched
        // (step_index, metric), surfacing one-sided phases/metrics as coverage
        // rows. Render-only — never contributes to the gate exit.
        noise_phase_findings(
            a_rows,
            b_rows,
            &pairing_label,
            spread_threshold_pct,
            include_stable,
            &mut phase_findings,
            &mut phase_coverage,
            &mut consulted_phase,
        );
        // Diff the declared gates (HEAD/B authoritative) against what actually
        // evaluated this group. A whole-run gate (`phase: None`) matches when its
        // metric produced an aggregate verdict; a phase-scoped gate matches when
        // its `(phase, metric)` produced a per-phase verdict. Anything left is a
        // gate the author declared that silently never fired — surfaced, never
        // gated.
        if let Some(first) = b_rows.first() {
            for a in &first.perf_delta_assertions {
                let matched = match a.phase {
                    None => consulted.contains(a.metric.as_str()),
                    Some(step) => consulted_phase.contains(&(step, a.metric.as_str())),
                };
                if !matched {
                    assertion_coverage.push(NoiseAssertionCoverage {
                        pairing_label: pairing_label.clone(),
                        assertion: a.clone(),
                    });
                }
            }
        }
    }
    NoiseReport {
        findings,
        phase_findings,
        phase_coverage,
        assertion_coverage,
        paired_scenarios,
    }
}

/// Classify one metric's [`NoiseVerdict`] into a [`NoiseKind`], shared by the
/// aggregate and per-phase noise passes. Returns `None` to omit the row: both
/// sides ~zero (no signal), or unchanged/immaterial-and-clean when
/// `include_stable` is false (the gate-only path).
///
/// A `< 2`-sample side (`insufficient_samples`) is a HARD gate that precedes
/// significance (Noisy, never a confident regression); the ADVISORY
/// `high_spread` flag does NOT gate (that suppression was the signal-inverting
/// bug). A row is a confident regression only when SEPARATED (Welch or disjoint
/// bands) AND MATERIAL (the registry dual-gate) in the worsening polarity.
fn classify_noise(
    verdict: &NoiseVerdict,
    m: &MetricDef,
    assertion: Option<&crate::test_support::PerfDeltaAssertionRecord>,
    include_stable: bool,
) -> Option<NoiseKind> {
    // No signal on either side (both ~zero): skip. Same zero epsilon as the
    // spread ratio for one consistent "is this zero".
    if verdict.a.mean.abs() < ZERO_MEAN_EPS && verdict.b.mean.abs() < ZERO_MEAN_EPS {
        return None;
    }
    // HARD gate: a side realized < 2 samples, so variance / Welch are undefined.
    // Precedes significance — never a confident regression. (The ADVISORY
    // high_spread flag, by contrast, does NOT gate.)
    if verdict.insufficient_samples {
        return Some(NoiseKind::Noisy);
    }
    // MATERIALITY: mirror the scalar dual-gate (push_scalar_findings ~L899) so
    // noise and default modes agree on "is this delta large enough". A
    // statistically-separated but trivially-small move stays Stable. --noise-adjust
    // conflicts with --threshold/--policy (cli.rs), so the registry defaults ARE
    // the resolved thresholds (an empty ComparisonPolicy::rel_threshold returns
    // default_rel); reading them from `m` directly is equivalent, no policy needed.
    // For a Rate metric, `mean` is the pooled Σnum/Σden centroid (the
    // registry-authoritative cross-run value), while the Welch separation arm
    // reads `sample_mean` (mean-of-ratios, coherent with `var`). Materiality AND
    // the direction label below authoritatively follow `mean`; the two statistics
    // can order the sides oppositely only under orders-of-magnitude denominator
    // skew between runs of one config (physically implausible), where the pooled
    // direction is the correct label anyway.
    let a = verdict.a.mean;
    let b = verdict.b.mean;
    let delta = b - a;
    let rel_delta = if a.abs() > ZERO_MEAN_EPS {
        (delta / a).abs()
    } else if delta.abs() > ZERO_MEAN_EPS {
        // Non-negligible move from a ~zero baseline: unbounded relative change,
        // so the absolute gate alone decides (mirrors the scalar path).
        f64::INFINITY
    } else {
        0.0
    };
    // A declared PerfDeltaAssertion OVERRIDES the registry gate for THIS
    // (test, metric): an absolute floor (`min_abs`), a relative threshold
    // (`max_regression_pct`), and/or a pinned direction. Absent (None) =>
    // registry defaults, so the no-assertion path is byte-identical to the
    // default gate. A tighter declared threshold turns a default-Stable move
    // into a Regression; a pinned direction can assert a registry-Informational
    // metric.
    //
    // `min_abs` / `max_regression_pct` come from a pub serde
    // `PerfDeltaAssertionRecord`. `KtstrTestEntry::validate` rejects a negative
    // or NaN threshold on the entry-construction path, but a hand-edited or
    // stale sidecar could deserialize one — and `delta.abs()` / `rel_delta` are
    // non-negative, so a NEGATIVE gate makes `material` unconditionally true (a
    // phantom confident regression that flips the exit) while a NaN gate makes
    // every `>=` false (silently disabled). Reject out-of-range values here and
    // fall back to the registry default — symmetric with the `TargetValue`
    // direction guard below, defending the same untrusted deserialization path.
    let abs_gate = assertion
        .and_then(|x| x.min_abs)
        .filter(|v| v.is_finite() && *v >= 0.0)
        .unwrap_or(m.default_abs);
    let rel_gate = assertion
        .and_then(|x| x.max_regression_pct)
        .filter(|v| v.is_finite() && *v >= 0.0)
        .map(|pct| pct / 100.0)
        .unwrap_or(m.default_rel);
    let material = delta.abs() >= abs_gate && rel_delta >= rel_gate;
    // A declared direction override, else the registry polarity. `TargetValue`
    // is rejected at the `PerfDeltaAssertion::with_direction` builder, so a
    // validated entry never carries it — but `PerfDeltaAssertionRecord` is a
    // pub serde type, so a hand-edited or stale sidecar could deserialize a
    // `direction: TargetValue`. Symmetric target-distance gating is
    // unimplemented (the polarity path would misread it as increase-is-worse),
    // so ignore it here and inherit the registry polarity — matching the
    // entry-path guarantee on the sidecar-deserialization path.
    let classify = match assertion.and_then(|x| x.direction) {
        Some(crate::test_support::Polarity::TargetValue(_)) | None => m.classify_direction(),
        Some(p) => p.classify_direction(),
    };

    Some(if verdict.separated && material {
        match classify {
            // Directionless metric: a separated + material move with no good/bad
            // direction — shown, never fails the gate.
            None => NoiseKind::Informational,
            Some(higher_is_worse) => {
                // Polarity split by the SIGN of the mean delta (b vs a), NOT a
                // band position: separation can come from the Welch arm even when
                // b.mean sits inside a's [min, max] band.
                let worsened = if higher_is_worse { b > a } else { b < a };
                if worsened {
                    NoiseKind::Regression
                } else {
                    NoiseKind::Improvement
                }
            }
        }
    } else if include_stable {
        NoiseKind::Stable
    } else {
        return None; // unchanged/immaterial + clean: omit (gate-only path)
    })
}

/// Push one one-sided per-phase metric to `coverage` — a metric present in a
/// matched-`step_index` bucket on one side only. `value` is the present
/// side's per-side mean across its runs (the absent side has none).
fn push_noise_phase_coverage(
    coverage: &mut Vec<NoisePhaseCoverage>,
    pairing_label: &str,
    step_index: u16,
    label: &str,
    metric: &'static MetricDef,
    present_side: ComparePartition,
    vals: &[f64],
) {
    if vals.is_empty() {
        return;
    }
    let value = vals.iter().sum::<f64>() / vals.len() as f64;
    coverage.push(NoisePhaseCoverage {
        pairing_label: pairing_label.to_string(),
        step_index,
        label: label.to_string(),
        metric: Some(metric),
        present_side,
        value: Some(value),
    });
}

/// Surface every non-suppressed metric of a whole one-sided `step_index` (a
/// phase present on only one side's runs) as [`NoisePhaseCoverage`] rows, so a
/// scenario-shape asymmetry is not silently dropped.
fn push_noise_unpaired_step(
    coverage: &mut Vec<NoisePhaseCoverage>,
    pairing_label: &str,
    step_index: u16,
    side: ComparePartition,
    buckets: &[&crate::assert::PhaseBucket],
) {
    let label = buckets[0].label.clone();
    let names: std::collections::BTreeSet<&str> = buckets
        .iter()
        .flat_map(|p| p.metrics.keys())
        .map(String::as_str)
        .collect();
    let before = coverage.len();
    for name in names {
        if is_render_suppressed_component(name) {
            continue;
        }
        let Some(m) = metric_def(name) else {
            continue;
        };
        let vals: Vec<f64> = buckets
            .iter()
            .filter_map(|p| p.metrics.get(name).copied())
            .collect();
        push_noise_phase_coverage(coverage, pairing_label, step_index, &label, m, side, &vals);
    }
    if coverage.len() == before {
        // A one-sided phase with no readable (non-suppressed) metric — a
        // synthesized capture-free step can carry an empty metrics map. Surface
        // the empty shape (metric/value None -> rendered with `—`) so it is not
        // silently dropped.
        coverage.push(NoisePhaseCoverage {
            pairing_label: pairing_label.to_string(),
            step_index,
            label,
            metric: None,
            present_side: side,
            value: None,
        });
    }
}

/// Per-phase noise sub-pass for one matched `(a_rows, b_rows)` pair, over the
/// N per-run [`crate::assert::PhaseBucket`]s per side. For each matched
/// `step_index` it walks the union of metric names
/// and emits a [`NoisePhaseFinding`] per `(step, metric)` present on BOTH sides
/// (spread verdict via the same machinery as the aggregate pass, incl. the
/// pooled-`Σnum/Σden` Rate centroid — here summed WITHIN the phase from
/// `bucket.metrics`), or a [`NoisePhaseCoverage`] for a one-sided metric or a
/// whole one-sided `step_index`. Skips the pair unless BOTH sides have at least
/// one run carrying phases (single-phase scenarios). Render-only: nothing here
/// contributes to the gate exit.
#[allow(clippy::too_many_arguments)]
fn noise_phase_findings(
    a_rows: &[GauntletRow],
    b_rows: &[GauntletRow],
    pairing_label: &str,
    spread_threshold_pct: f64,
    include_stable: bool,
    findings: &mut Vec<NoisePhaseFinding>,
    coverage: &mut Vec<NoisePhaseCoverage>,
    consulted_phase: &mut std::collections::BTreeSet<(u16, &'static str)>,
) {
    use std::collections::BTreeSet;
    // Single-phase scenarios carry empty phases on every run; skip the per-phase
    // view if either side has no run with phases.
    let has_phases = |rows: &[GauntletRow]| rows.iter().any(|r| !r.phases.is_empty());
    if !has_phases(a_rows) || !has_phases(b_rows) {
        return;
    }
    // A nested fn (not a closure) so the elided output lifetime ties the
    // borrowed &PhaseBucket to `rows` — a closure can't express that linkage.
    fn by_step(
        rows: &[GauntletRow],
    ) -> std::collections::BTreeMap<u16, Vec<&crate::assert::PhaseBucket>> {
        let mut m: std::collections::BTreeMap<u16, Vec<&crate::assert::PhaseBucket>> =
            std::collections::BTreeMap::new();
        for r in rows {
            for p in &r.phases {
                m.entry(p.step_index).or_default().push(p);
            }
        }
        m
    }
    let a_by_step = by_step(a_rows);
    let b_by_step = by_step(b_rows);
    let steps: BTreeSet<u16> = a_by_step.keys().chain(b_by_step.keys()).copied().collect();
    for step_index in steps {
        match (a_by_step.get(&step_index), b_by_step.get(&step_index)) {
            (Some(a_buckets), Some(b_buckets)) => {
                let label = a_buckets[0].label.clone();
                let names: BTreeSet<&str> = a_buckets
                    .iter()
                    .chain(b_buckets.iter())
                    .flat_map(|p| p.metrics.keys())
                    .map(String::as_str)
                    .collect();
                for name in names {
                    if is_render_suppressed_component(name) {
                        continue;
                    }
                    let Some(m) = metric_def(name) else {
                        continue;
                    };
                    let a_vals: Vec<f64> = a_buckets
                        .iter()
                        .filter_map(|p| p.metrics.get(name).copied())
                        .collect();
                    let b_vals: Vec<f64> = b_buckets
                        .iter()
                        .filter_map(|p| p.metrics.get(name).copied())
                        .collect();
                    match (a_vals.is_empty(), b_vals.is_empty()) {
                        (true, true) => continue,
                        // Present on only one matched side: no band on the other
                        // — coverage, not a delta.
                        (false, true) => {
                            push_noise_phase_coverage(
                                coverage,
                                pairing_label,
                                step_index,
                                &label,
                                m,
                                ComparePartition::A,
                                &a_vals,
                            );
                            continue;
                        }
                        (true, false) => {
                            push_noise_phase_coverage(
                                coverage,
                                pairing_label,
                                step_index,
                                &label,
                                m,
                                ComparePartition::B,
                                &b_vals,
                            );
                            continue;
                        }
                        (false, false) => {}
                    }
                    let verdict = match m.kind {
                        MetricKind::Rate {
                            numerator,
                            denominator,
                        } => {
                            // Per-phase Rate: pooled Σnum/Σden WITHIN the phase
                            // from bucket.metrics (phase-derivable rates carry
                            // both components per phase), band from per-run ratios.
                            let pooled = |buckets: &[&crate::assert::PhaseBucket]| -> Option<f64> {
                                let (num, den) = buckets.iter().fold((0.0, 0.0), |(sn, sd), p| {
                                    match (p.metrics.get(numerator), p.metrics.get(denominator)) {
                                        (Some(n), Some(d)) => (sn + n, sd + d),
                                        _ => (sn, sd),
                                    }
                                });
                                (den != 0.0).then(|| num / den)
                            };
                            let (Some(a_pooled), Some(b_pooled)) =
                                (pooled(a_buckets), pooled(b_buckets))
                            else {
                                continue;
                            };
                            noise_verdict_from(
                                SideSummary::of(&a_vals).with_pooled_mean(a_pooled),
                                SideSummary::of(&b_vals).with_pooled_mean(b_pooled),
                                spread_threshold_pct,
                            )
                        }
                        _ => noise_verdict(&a_vals, &b_vals, spread_threshold_pct),
                    };
                    // This (step, metric) produced a comparable verdict, so a
                    // phase-scoped gate on it WAS evaluated (see the aggregate
                    // pass's `consulted`).
                    consulted_phase.insert((step_index, m.name));
                    // Phase-scoped declared assertion for this test+metric+step.
                    let assertion = b_rows.first().and_then(|r| {
                        r.perf_delta_assertions
                            .iter()
                            .find(|x| x.metric == m.name && x.phase == Some(step_index))
                    });
                    let gated_by_assertion = assertion.is_some();
                    let Some(kind) = classify_noise(&verdict, m, assertion, include_stable) else {
                        continue;
                    };
                    findings.push(NoisePhaseFinding {
                        pairing_label: pairing_label.to_string(),
                        step_index,
                        label: label.clone(),
                        metric: m,
                        verdict,
                        kind,
                        gated_by_assertion,
                    });
                }
            }
            (Some(a_buckets), None) => {
                push_noise_unpaired_step(
                    coverage,
                    pairing_label,
                    step_index,
                    ComparePartition::A,
                    a_buckets,
                );
            }
            (None, Some(b_buckets)) => {
                push_noise_unpaired_step(
                    coverage,
                    pairing_label,
                    step_index,
                    ComparePartition::B,
                    b_buckets,
                );
            }
            (None, None) => {}
        }
    }
}

/// Summarize a side's loaded runs by why each is or isn't comparable, using the
/// SAME exclusions [`noise_findings`] applies (skip / fail / inconclusive /
/// expected-failure). Returns `(comparable_count, human_summary)` so an empty
/// comparison can explain WHY — e.g. every run was skipped — instead of the bare
/// "nothing to compare". The `comparable_count` equals the number of rows
/// `noise_findings` would keep (a row is comparable iff none of the four
/// exclusions hold), so a zero here is exactly why the side produced no findings.
pub(crate) fn summarize_side_runs(rows: &[GauntletRow]) -> (usize, String) {
    let (mut skipped, mut failed, mut inconclusive, mut xfail, mut comparable) =
        (0usize, 0usize, 0usize, 0usize, 0usize);
    for r in rows {
        if r.is_skip() {
            skipped += 1;
        } else if r.is_fail() {
            failed += 1;
        } else if r.is_inconclusive() {
            inconclusive += 1;
        } else if r.expected_failure {
            xfail += 1;
        } else {
            comparable += 1;
        }
    }
    let mut excluded = Vec::new();
    if skipped > 0 {
        excluded.push(format!("{skipped} skipped"));
    }
    if failed > 0 {
        excluded.push(format!("{failed} failed"));
    }
    if inconclusive > 0 {
        excluded.push(format!("{inconclusive} inconclusive"));
    }
    if xfail > 0 {
        excluded.push(format!("{xfail} expected-failure"));
    }
    let breakdown = if excluded.is_empty() {
        "none excluded".to_string()
    } else {
        excluded.join(", ")
    };
    (
        comparable,
        format!(
            "{} run(s): {comparable} comparable ({breakdown})",
            rows.len()
        ),
    )
}

/// Noise-adjusted variant of [`compare_partitions`]: instead of averaging each
/// side's runs into one mean and gating on a fixed threshold, it keeps every run
/// ([`RowPrep::PerRunPooled`]), summarizes each side per metric, and decides
/// whether the two sides are distinguishable given their run-to-run variability
/// (see [`noise_findings`] for the row-level core + [`noise_verdict`] for the
/// per-metric decision). Used by `perf-delta --noise-adjust N`, which produces N
/// runs per side. A metric is a CONFIDENT REGRESSION (fed to the operator
/// failure gate below) when it is SEPARATED (a two-sided Welch t-test rejects equal means at
/// `NOISE_ALPHA`, OR the `[min, max]` bands are fully disjoint) AND MATERIAL (the
/// mean delta clears both the registry `default_abs` and `default_rel`, the same
/// dual-gate as the scalar path) in the worsening direction (per polarity). A
/// side that realized fewer than 2 runs is flagged `NOISY` and never gates; a
/// per-side relative spread over `spread_threshold_pct` is an ADVISORY
/// `(noisy spread)` annotation that NEVER suppresses a verdict. By default prints
/// only the MEANINGFUL rows (confident regression / improvement / informational),
/// each with its side's `mean [min-max] spread%` and verdict; stable (unchanged +
/// clean) and noisy (<2-run) rows are hidden unless `--all-metrics`, and a
/// one-line summary prints when every row is suppressed. The footer leads with an
/// overall verdict word ([`overall_verdict`]: `STABLE` unless the regressed or
/// improved count clears the significance cutoff — sub-cutoff moves are flagged
/// but likely noise) followed by the composite counts (regressed / improved /
/// stable / under-sampled, plus informational when present); and, when every
/// changed (non-stable) metric had a side with <2 usable runs, an explicit
/// inconclusive note.
///
/// When the scenarios carry phases, a per-phase spread block follows the
/// aggregate table (via [`format_noise_phase_findings_lines`]): the same
/// spread verdict per matched `(step_index, metric)`, plus a coverage table for
/// one-sided phases/metrics. Like the aggregate table, the per-phase block shows
/// only MEANINGFUL rows by default (regression / improvement / informational) and
/// hides stable / noisy rows unless `--all-metrics` (`gate.show_all`); whenever the
/// spread rows are all suppressed it surfaces a one-line hint naming
/// `--all-metrics` (even alongside a coverage table), so the suppressed rows are
/// never silently gone. The coverage (one-sided) table is itself never
/// suppressed. `phase_opts` controls the per-phase
/// render only (`--no-phases` / `--phases-only` / `--steps-only` / `--phase` /
/// `--phase-threshold`); under `--phases-only` the aggregate table is
/// suppressed. Per-phase SPREAD findings are RENDER-ONLY — classified and
/// colored but not gating — EXCEPT a
/// per-phase regression the author explicitly declared a phase-scoped gate for,
/// which DOES contribute to the exit alongside the operator gate on the aggregate
/// confident regressions ([`gate_fails`]: their count reaches `fail_threshold`
/// [default 5], or a `must_fail` metric regressed).
/// The footer appends per-phase counts only when per-phase data exists and no
/// phase filter is active.
///
/// When no scenario pairs, the aggregate note breaks down each side's loaded
/// runs via [`summarize_side_runs`] — naming skipped / failed / inconclusive
/// runs — so an all-skipped comparison (e.g. a non-`performance_mode` test under
/// perf-delta's `KTSTR_PERF_ONLY`) is explained, not silently reported as
/// "nothing to compare".
pub fn compare_partitions_noise(
    filter_a: &RowFilter,
    filter_b: &RowFilter,
    dir: Option<&std::path::Path>,
    spread_threshold_pct: f64,
    phase_opts: &PhaseDisplayOptions,
    gate: &GateOptions,
) -> anyhow::Result<i32> {
    // Keep every per-run row INCLUDING duplicate pairing keys so the
    // run-to-run spread is observable: noise_findings groups the N runs
    // per key per side. PerRunPooled keeps every per-run row including
    // duplicate pairing keys — the N-per-key spread is the intended input.
    let prepared = prepare_partitioned_comparison(filter_a, filter_b, dir, RowPrep::PerRunPooled)?;
    let label_a = render_side_label(filter_a, &prepared.slicing_dims, "A");
    let label_b = render_side_label(filter_b, &prepared.slicing_dims, "B");

    let report = noise_findings(
        &prepared.rows_a_for_compare,
        &prepared.rows_b_for_compare,
        &prepared.pairing_dims,
        spread_threshold_pct,
        // Include Stable (unchanged + clean) metrics so the rendered table
        // shows the full comparison, not just changed/noisy rows.
        true,
    );

    println!(
        "perf-delta --noise-adjust: {label_b} vs {label_a} (advisory noisy-spread threshold {spread_threshold_pct:.2}%)"
    );
    // Aggregate spread table — suppressed under --phases-only (renders ONLY
    // the per-phase block).
    if !phase_opts.phases_only {
        if report.findings.is_empty() {
            // Distinguish "nothing paired" from "paired but every metric omitted"
            // so the message never contradicts the paired-scenario footer below.
            // A metric is omitted when both sides' means are ~zero, it read on
            // only one side, or it is a render-suppressed rate component.
            if report.paired_scenarios == 0 {
                // Explain WHY nothing paired instead of a bare "nothing to
                // compare": prepare_partitioned_comparison already bailed if a
                // side loaded ZERO rows, so both sides have >=1 run here — the
                // pairing failure is either all-excluded runs (skip/fail/etc.)
                // or a genuine scenario mismatch. Break down each side so a
                // skipped-on-both-sides run (the common perf-delta case: a
                // non-performance_mode test under KTSTR_PERF_ONLY) is named.
                let (a_ok, a_desc) = summarize_side_runs(&prepared.rows_a_for_compare);
                let (b_ok, b_desc) = summarize_side_runs(&prepared.rows_b_for_compare);
                println!(
                    "perf-delta --noise-adjust: no comparable runs to pair across the two runs."
                );
                println!("  {label_a} — {a_desc}");
                println!("  {label_b} — {b_desc}");
                if a_ok == 0 || b_ok == 0 {
                    println!(
                        "  A skipped run carries no metrics: perf-delta runs with \
                         KTSTR_PERF_ONLY, which skips any test not marked \
                         #[ktstr_test(performance_mode = true)]; host-gated skips land here \
                         too. A failed / inconclusive run is excluded from the spread math."
                    );
                } else {
                    println!(
                        "  Both sides produced comparable runs but share no scenario / topology \
                         / work_type — the two selections have no common test to contrast."
                    );
                }
            } else {
                println!(
                    "perf-delta --noise-adjust: no metric to display — every compared metric \
                     was unchanged at zero, present on only one side, or render-suppressed"
                );
            }
        } else {
            print!(
                "{}",
                format_noise_findings_table(&report.findings, &label_a, &label_b, gate.show_all)
            );
        }
    }
    // Per-phase spread block — render-only (never gates), honoring --no-phases /
    // --phase / --steps-only / --phase-threshold. Under --phases-only it is the
    // ONLY table, so emit an explicit note rather than a silent blank.
    let phase_lines = format_noise_phase_findings_lines(
        &report.phase_findings,
        &report.phase_coverage,
        phase_opts,
        &label_a,
        &label_b,
        gate.show_all,
    );
    if phase_lines.is_empty() && phase_opts.phases_only {
        println!(
            "perf-delta --noise-adjust: no per-phase noise data to show (no matched \
             multi-phase scenario at the selected step, or every per-phase row was \
             filtered by --phase / --steps-only / --phase-threshold)"
        );
    }
    for line in phase_lines {
        println!("{line}");
    }
    // Declared gates that never evaluated (metric absent from the compared
    // data). Rendered regardless of --phases-only — a silently-inert declared
    // gate is important whether or not the aggregate table is shown. Never
    // gates the exit.
    for line in format_noise_assertion_coverage_lines(&report.assertion_coverage) {
        println!("{line}");
    }
    let regressions = report.regressions();
    // Per-phase regressions the author explicitly DECLARED a phase-scoped gate
    // for — these gate the exit alongside the aggregate regressions (spread-only
    // per-phase findings stay render-only). Computed unconditionally so the exit
    // is identical whether or not the phase footer is shown.
    let declared_phase_regressions = report.declared_phase_regressions();
    // The aggregate summary footer describes the (hidden) aggregate spread, so
    // suppress it under --phases-only — which renders ONLY the per-phase
    // block.
    // The exit still gates on gate_fails(aggregate regressions) +
    // `declared_phase_regressions` (computed above) when the footer is hidden.
    if !phase_opts.phases_only {
        let noisy = report.noisy();
        // Composite verdict counts: regressed / improved LEAD (the table shows
        // each with its magnitude), stable is the expected residual cited as a
        // count only, and informational (changed but no polarity) is cited only
        // when present. All render-only — the exit reads gate_fails + declared
        // gates, never these.
        let improvements = report.improvements();
        let stable = report.stable();
        let informational = report.informational();
        let info_clause = if informational > 0 {
            format!(", {informational} informational")
        } else {
            String::new()
        };
        // Overall verdict word: STABLE unless a direction clears the cutoff (see
        // overall_verdict). Sub-cutoff moves are flagged in the counts above but
        // do not shift the verdict off STABLE.
        let verdict = overall_verdict(&report, gate);
        let stable_note = if verdict == "STABLE" && regressions + improvements > 0 {
            " (moves are below the significance cutoff, more likely noise than signal)"
        } else {
            ""
        };
        // Per-phase footer counts are shown ONLY when there IS per-phase data
        // AND no phase filter is active (no --no-phases / --phase / --steps-only
        // / --phase-threshold) — so the counts always match the fully-rendered
        // per-phase table, and a single-phase run (no per-phase view) shows no
        // confusing 0/0 per-phase clause. Uses the same !any-flag-set
        // discipline as the aggregate paired-scenario hint.
        let has_phase_data = !report.phase_findings.is_empty() || !report.phase_coverage.is_empty();
        let show_phase_footer = has_phase_data
            && !phase_opts.no_phases
            && phase_opts.phase.is_none()
            && !phase_opts.steps_only
            && phase_opts.phase_threshold.is_none();
        // A REGRESSED verdict with ZERO aggregate regressions can only come from a
        // declared PER-PHASE gate (a declared whole-run regression and --must-fail
        // both require an aggregate Regression, so regressions >= 1 there). When
        // the phase footer below is hidden (a phase filter is active) name the
        // source so `overall REGRESSED: 0 regressed` is not self-contradictory;
        // when the phase footer shows, its "(declared-gated, exit-affecting)"
        // clause already explains it, so skip the redundant note.
        let verdict_source =
            if verdict.starts_with("REGRESSED") && regressions == 0 && !show_phase_footer {
                " (regression is a declared per-phase gate)"
            } else {
                ""
            };
        let phase_footer = if show_phase_footer {
            // Per-phase regressions are render-only EXCEPT the declared-gated
            // ones, which gate the exit — call that out so the count is not read
            // as fully render-only when a declared phase gate fired.
            let declared_note = if declared_phase_regressions > 0 {
                format!(" ({declared_phase_regressions} declared-gated, exit-affecting)")
            } else {
                " (render-only)".to_string()
            };
            format!(
                "; {} per-phase regression(s){declared_note}, {} per-phase under-sampled (<2 runs)",
                report.phase_regressions(),
                report.phase_noisy(),
            )
        } else {
            String::new()
        };
        // Declared gates that never evaluated — a single-line summary of the
        // warning block above (shown whenever any gate went unevaluated).
        let unevaluated_gates = report.assertion_coverage.len();
        let gate_footer = if unevaluated_gates > 0 {
            format!("; {unevaluated_gates} declared gate(s) not evaluated")
        } else {
            String::new()
        };
        println!(
            "perf-delta --noise-adjust: {} paired scenario(s); overall {verdict}: \
             {regressions} regressed, {improvements} improved, {stable} stable{info_clause}, \
             {noisy} under-sampled (<2 runs){stable_note}{verdict_source}{phase_footer}{gate_footer}",
            report.paired_scenarios,
        );
    }
    // Inconclusive: every CHANGED aggregate metric (excluding Stable rows, which
    // never gate) had a side with < 2 usable runs — no trustworthy signal either
    // way. Surfaced prominently for CI logs, but noise FLAGS, not FAILS, so the
    // exit stays 0 unless a confident AGGREGATE regression fired. Suppressed
    // under --phases-only (the aggregate table is hidden there).
    if !phase_opts.phases_only {
        let changed: Vec<&NoiseFinding> = report
            .findings
            .iter()
            .filter(|f| f.kind != NoiseKind::Stable)
            .collect();
        if !changed.is_empty() && changed.iter().all(|f| f.kind == NoiseKind::Noisy) {
            println!(
                "perf-delta --noise-adjust: NOTE -- every changed metric had a side with <2 usable \
                 runs; raise --noise-adjust N (or investigate why per-side runs failed) for a \
                 trustworthy verdict"
            );
        }
    }
    // Exit gates on AGGREGATE regressions plus DECLARED phase regressions.
    // Spread-only per-phase findings stay render-only (parity with the scalar
    // per-phase pass — a narrow-window phase flake must not flip CI red), but a
    // phase-scoped gate the author explicitly declared is an opt-in and DOES
    // gate (matches the `PerfDeltaAssertion::phase` doc).
    // Operator gate: the count / must-fail gate applies to the UNdeclared
    // aggregate confident regressions; an author-DECLARED gate — whole-run OR
    // phase-scoped — always fails (its own opt-in, orthogonal to the operator's
    // count gate).
    Ok(noise_exit_code(&report, gate))
}

/// Exit code for the noise-adjusted compare. Fails (`1`) when EITHER the
/// operator gate ([`gate_fails`]) trips on the aggregate confident regressions,
/// OR any author-DECLARED regression is present — whole-run
/// ([`NoiseReport::declared_regressions`]) or phase-scoped
/// ([`NoiseReport::declared_phase_regressions`]). A declared assertion is a
/// per-test opt-in that ALWAYS gates on its metric, independent of the
/// operator's count / must-fail gate, so a single declared regression fails
/// even below `--fail-threshold`. Extracted so the exit decision is
/// unit-testable.
pub(crate) fn noise_exit_code(report: &NoiseReport, gate: &GateOptions) -> i32 {
    let regressing: Vec<&str> = report
        .findings
        .iter()
        .filter(|f| f.kind == NoiseKind::Regression)
        .map(|f| f.metric.name)
        .collect();
    if gate_fails(&regressing, gate)
        || report.declared_regressions() > 0
        || report.declared_phase_regressions() > 0
    {
        1
    } else {
        0
    }
}

/// Overall run verdict word for the `--noise-adjust` summary, derived from the
/// confident move counts against the significance cutoff (`--fail-threshold`,
/// default 5). Sub-cutoff moves in EITHER direction are FLAGGED (shown in the
/// table, counted in the footer) but do NOT shift the verdict off `STABLE` —
/// below the cutoff a move is more likely noise than signal, which is the whole
/// point of the conservative noise gate. A direction only reads into the verdict
/// once it clears the cutoff, and both can hold at once (`REGRESSED + IMPROVED`).
///
/// The regressed side reuses the exit decision ([`noise_exit_code`]) so a
/// `--must-fail` or declared gate that fails the run also reads `REGRESSED` even
/// below the count cutoff; the improved side has no gate, so it uses the count
/// cutoff alone. Display-only — never the exit basis.
pub(crate) fn overall_verdict(report: &NoiseReport, gate: &GateOptions) -> &'static str {
    verdict_label(
        noise_exit_code(report, gate) == 1,
        report.improvements(),
        gate.fail_threshold,
    )
}

/// Pure verdict classifier: `regressed` is the run's regression-fail decision;
/// `improvements` is the confident-improvement count; `fail_threshold` is the
/// significance cutoff (`None` = 5, `Some(0)` disables count significance).
/// Split out from [`overall_verdict`] so the STABLE-below-cutoff policy is
/// unit-testable without building a full report.
pub(crate) fn verdict_label(
    regressed: bool,
    improvements: usize,
    fail_threshold: Option<usize>,
) -> &'static str {
    let n = fail_threshold.unwrap_or(5);
    let improved = n >= 1 && improvements >= n;
    match (regressed, improved) {
        (true, true) => "REGRESSED + IMPROVED",
        (true, false) => "REGRESSED",
        (false, true) => "IMPROVED",
        (false, false) => "STABLE",
    }
}

/// Compose a noise verdict cell's text: the base label plus any parenthesized,
/// comma-joined annotations. Shared by the aggregate and per-phase tables so
/// both annotate identically. `noisy spread` (advisory `high_spread`, suppressed
/// on a Noisy <2-runs row where it is redundant) and `declared gate` (the row
/// carries a declared [`crate::test_support::PerfDeltaAssertion`] — its
/// overrides drive the gate, or fall back to the registry defaults if rejected
/// as out-of-range on a corrupt sidecar) can co-occur → `REGRESSION (noisy
/// spread, declared gate)`.
fn compose_noise_verdict_text(
    base: &str,
    high_spread: bool,
    kind: NoiseKind,
    gated_by_assertion: bool,
) -> String {
    let mut annotations: Vec<&str> = Vec::new();
    if high_spread && kind != NoiseKind::Noisy {
        annotations.push("noisy spread");
    }
    if gated_by_assertion {
        annotations.push("declared gate");
    }
    if annotations.is_empty() {
        base.to_string()
    } else {
        format!("{base} ({})", annotations.join(", "))
    }
}

/// Render the per-metric noise-adjusted findings as a table for
/// `perf-delta --noise-adjust`: one row per (scenario, metric) with each
/// side's `mean [min-max] spread%` and the colored verdict. Includes
/// [`NoiseKind::Stable`] rows so the operator sees every compared metric,
/// not only the changed ones. Pure (returns the rendered string with a
/// trailing newline) so the row/verdict mapping is unit-testable without
/// capturing stdout.
pub(crate) fn format_noise_findings_table(
    findings: &[NoiseFinding],
    label_a: &str,
    label_b: &str,
    show_all: bool,
) -> String {
    use comfy_table::{Cell, Color};
    // Default view shows only MEANINGFUL rows (confident regression /
    // improvement / informational). Stable (unchanged / immaterial) and
    // Noisy (<2 usable runs) rows are hidden unless `show_all`; their COUNTS
    // still print in the caller's footer, and the gate reads the full
    // classified set, so this suppression is display-only.
    let visible: Vec<&NoiseFinding> = findings
        .iter()
        .filter(|f| show_all || !matches!(f.kind, NoiseKind::Stable | NoiseKind::Noisy))
        .collect();
    if visible.is_empty() {
        if findings.is_empty() {
            return String::new();
        }
        let noisy = findings
            .iter()
            .filter(|f| f.kind == NoiseKind::Noisy)
            .count();
        return format!(
            "perf-delta --noise-adjust: {} metric(s) compared, none meaningfully changed \
             ({noisy} under-sampled); re-run with --all-metrics to see them\n",
            findings.len(),
        );
    }
    let mut table = crate::cli::new_table();
    table.set_header(vec![
        "TEST / METRIC".to_string(),
        format!("{label_a} (A: mean [min-max] spread%)"),
        format!("{label_b} (B: mean [min-max] spread%)"),
        "VERDICT".to_string(),
    ]);
    for f in visible {
        let (base, color) = match f.kind {
            NoiseKind::Regression => ("REGRESSION", Color::Red),
            NoiseKind::Improvement => ("improvement", Color::Green),
            NoiseKind::Noisy => ("NOISY (<2 runs)", Color::Yellow),
            NoiseKind::Informational => ("informational", Color::Blue),
            NoiseKind::Stable => ("stable", Color::Grey),
        };
        // Verdict annotations (parenthesized, comma-joined). `noisy spread` is
        // ADVISORY (high_spread) — flags a noisy side without changing the
        // classification, redundant on a Noisy <2-runs row so omitted there.
        // `declared gate` marks a row that CARRIES a declared PerfDeltaAssertion
        // (its overrides drive the gate, or fall back to registry defaults if
        // rejected as out-of-range on a corrupt sidecar), so the operator can
        // tell an author-declared gate from a pure registry-default one.
        let verdict_text =
            compose_noise_verdict_text(base, f.verdict.high_spread, f.kind, f.gated_by_assertion);
        let v = &f.verdict;
        table.add_row(vec![
            Cell::new(format!("{} / {}", f.pairing_label, f.metric.name)),
            Cell::new(format!(
                "{:.1} [{:.1}-{:.1}] {:.2}%",
                v.a.mean, v.a.min, v.a.max, v.a.spread_pct
            )),
            Cell::new(format!(
                "{:.1} [{:.1}-{:.1}] {:.2}%",
                v.b.mean, v.b.min, v.b.max, v.b.spread_pct
            )),
            Cell::new(verdict_text).fg(color),
        ]);
    }
    format!("{table}\n")
}

/// One-line description of a declared gate's overrides for the
/// not-evaluated warning: the thresholds it WOULD have applied. All-`None`
/// (a bare `PerfDeltaAssertion::new(metric)`) renders `registry defaults` —
/// a presence-checked gate that inherits the registry `default_abs`/
/// `default_rel`/polarity.
fn describe_declared_gate(a: &crate::test_support::PerfDeltaAssertionRecord) -> String {
    let mut parts: Vec<String> = Vec::new();
    if let Some(pct) = a.max_regression_pct {
        parts.push(format!("max_regression_pct={pct}"));
    }
    if let Some(abs) = a.min_abs {
        parts.push(format!("min_abs={abs}"));
    }
    if let Some(dir) = a.direction {
        parts.push(format!("direction={dir:?}"));
    }
    if parts.is_empty() {
        "registry defaults".to_string()
    } else {
        parts.join(", ")
    }
}

/// Render the declared perf gates that never evaluated (a metric absent from
/// the compared data) as warning lines for `perf-delta --noise-adjust`: a
/// TEST | METRIC | PHASE | DECLARED GATE table, so an author whose declared
/// [`crate::test_support::PerfDeltaAssertion`] silently did not fire sees it
/// rather than mistaking a not-evaluated gate for a passing one. Never gates
/// the exit (a gate that could not evaluate is not a regression). Pure —
/// returns the lines (empty when there are none) so the mapping is
/// unit-testable without capturing stdout.
pub(crate) fn format_noise_assertion_coverage_lines(
    coverage: &[NoiseAssertionCoverage],
) -> Vec<String> {
    use comfy_table::{Cell, Color};
    let mut lines = Vec::new();
    if coverage.is_empty() {
        return lines;
    }
    let mut rows: Vec<&NoiseAssertionCoverage> = coverage.iter().collect();
    rows.sort_by(|a, b| {
        a.pairing_label
            .cmp(&b.pairing_label)
            .then_with(|| a.assertion.metric.cmp(&b.assertion.metric))
            .then_with(|| a.assertion.phase.cmp(&b.assertion.phase))
    });
    lines.push(String::new());
    lines.push(
        "declared perf gate(s) NOT evaluated — the metric was absent from the compared \
         data (workload no longer emits it, a one-sided/failed run, or a Rate with no \
         samples), so the declared gate silently did not fire:"
            .to_string(),
    );
    let mut table = crate::cli::new_table();
    table.set_header(vec![
        "TEST".to_string(),
        "METRIC".to_string(),
        "PHASE".to_string(),
        "DECLARED GATE".to_string(),
    ]);
    for c in rows {
        let phase = match c.assertion.phase {
            None => "aggregate".to_string(),
            Some(k) => k.to_string(),
        };
        table.add_row(vec![
            Cell::new(&c.pairing_label),
            Cell::new(&c.assertion.metric).fg(Color::Yellow),
            Cell::new(phase),
            Cell::new(describe_declared_gate(&c.assertion)),
        ]);
    }
    lines.push(format!("{table}"));
    lines
}

/// Render the per-phase noise-adjusted findings as lines for
/// `perf-delta --noise-adjust`: a per-phase spread table (PHASE | TEST /
/// METRIC | A mean[min-max] spread% | B ... | VERDICT) plus a one-sided
/// coverage table (SIDE | TEST | PHASE | METRIC | VALUE). The TEST column
/// carries the full pairing-key label (scenario plus every pairing dim),
/// matching the scalar compare path — a scenario shared across topologies
/// renders as distinct rows. Mirrors
/// [`format_noise_findings_table`], honoring
/// [`PhaseDisplayOptions`] (no_phases / phase / steps_only / phase_threshold).
/// Render-only: these rows never gate. Pure — returns the lines (empty when
/// suppressed / no per-phase data) so the row/verdict mapping is unit-testable
/// without capturing stdout.
pub(crate) fn format_noise_phase_findings_lines(
    phase_findings: &[NoisePhaseFinding],
    phase_coverage: &[NoisePhaseCoverage],
    phase_opts: &PhaseDisplayOptions,
    label_a: &str,
    label_b: &str,
    show_all: bool,
) -> Vec<String> {
    use comfy_table::{Cell, Color};
    let mut lines = Vec::new();
    if phase_opts.no_phases {
        return lines;
    }
    // Rows passing the phase-axis filters (`--phase` / `--steps-only`) and the
    // `--phase-threshold` spread gate, BEFORE the meaningful-only display filter.
    // Retained so the collapse summary below can report how many rows the default
    // view suppressed.
    let phase_filtered: Vec<&NoisePhaseFinding> = phase_findings
        .iter()
        .filter(|f| phase_opts.matches_phase(f.step_index))
        .filter(|f| phase_opts.passes_noise_spread_threshold(&f.verdict))
        .collect();
    // Default view shows only MEANINGFUL rows (regression / improvement /
    // informational); Stable + Noisy rows are hidden unless `show_all`
    // (`--all-metrics`), mirroring the aggregate table
    // ([`format_noise_findings_table`]). Display-only: the footer still reports
    // the per-phase regression / under-sampled COUNTS from the unfiltered report,
    // and the exit gate reads the unfiltered findings — so suppressing rows here
    // changes neither the counts nor the pass/fail.
    let mut findings: Vec<&NoisePhaseFinding> = phase_filtered
        .iter()
        .copied()
        .filter(|f| show_all || !matches!(f.kind, NoiseKind::Stable | NoiseKind::Noisy))
        .collect();
    let mut coverage: Vec<&NoisePhaseCoverage> = phase_coverage
        .iter()
        .filter(|c| phase_opts.matches_phase(c.step_index))
        .collect();
    let had_findings = !findings.is_empty();
    // Spread rows existed but were all suppressed as stable/noisy in the default
    // view: a one-line hint (naming `--all-metrics`, wording matched to the
    // aggregate collapse) keeps the suppression discoverable — surfaced whether
    // the block is otherwise empty OR a coverage table follows, so the suppressed
    // rows are never silently gone. Only in the default view: under `show_all`
    // nothing is suppressed, so an empty result there means there was genuinely no
    // per-phase spread data.
    let suppressed_hint: Option<String> =
        if findings.is_empty() && !phase_filtered.is_empty() && !show_all {
            let noisy = phase_filtered
                .iter()
                .filter(|f| f.kind == NoiseKind::Noisy)
                .count();
            Some(format!(
                "perf-delta --noise-adjust: {} per-phase metric(s) compared, none meaningfully \
                 changed ({noisy} under-sampled); re-run with --all-metrics to see them",
                phase_filtered.len(),
            ))
        } else {
            None
        };
    if findings.is_empty() && coverage.is_empty() {
        // Nothing else to render — the hint (if any) is the whole block.
        lines.extend(suppressed_hint);
        return lines;
    }
    lines.push(String::new());
    if had_findings {
        // "per-phase spread:" heads the findings table ONLY — a coverage-only
        // section (no spread rows) gets its own header below, so the label
        // never mislabels a table.
        lines.push("per-phase spread:".to_string());
        // step_index-first (BASELINE..Step[N] time order), then pairing label,
        // then metric — a stable, top-down-by-phase-boundary order.
        findings.sort_by(|a, b| {
            a.step_index
                .cmp(&b.step_index)
                .then_with(|| a.pairing_label.cmp(&b.pairing_label))
                .then_with(|| a.metric.name.cmp(b.metric.name))
        });
        let mut table = crate::cli::new_table();
        table.set_header(vec![
            "PHASE".to_string(),
            "TEST / METRIC".to_string(),
            format!("{label_a} (A: mean [min-max] spread%)"),
            format!("{label_b} (B: mean [min-max] spread%)"),
            "VERDICT".to_string(),
        ]);
        for f in findings {
            let (base, color) = match f.kind {
                NoiseKind::Regression => ("REGRESSION", Color::Red),
                NoiseKind::Improvement => ("improvement", Color::Green),
                NoiseKind::Noisy => ("NOISY (<2 runs)", Color::Yellow),
                NoiseKind::Informational => ("informational", Color::Blue),
                NoiseKind::Stable => ("stable", Color::Grey),
            };
            // Same annotation composition as the aggregate table (advisory
            // `noisy spread` + `declared gate`).
            let verdict_text = compose_noise_verdict_text(
                base,
                f.verdict.high_spread,
                f.kind,
                f.gated_by_assertion,
            );
            let v = &f.verdict;
            table.add_row(vec![
                Cell::new(format!("{}: {}", f.step_index, f.label)),
                Cell::new(format!("{} / {}", f.pairing_label, f.metric.name)),
                Cell::new(format!(
                    "{:.1} [{:.1}-{:.1}] {:.2}%",
                    v.a.mean, v.a.min, v.a.max, v.a.spread_pct
                )),
                Cell::new(format!(
                    "{:.1} [{:.1}-{:.1}] {:.2}%",
                    v.b.mean, v.b.min, v.b.max, v.b.spread_pct
                )),
                Cell::new(verdict_text).fg(color),
            ]);
        }
        lines.push(table.to_string());
    } else {
        // Spread rows were all suppressed but a coverage table follows: surface
        // the suppression hint before it so `--all-metrics` stays discoverable.
        lines.extend(suppressed_hint);
    }
    if !coverage.is_empty() {
        // Separate from the spread table above only when one was rendered; the
        // section-leading blank already precedes a coverage-only block.
        if had_findings {
            lines.push(String::new());
        }
        lines.push("per-phase coverage asymmetry (one-sided metrics):".to_string());
        coverage.sort_by(|a, b| {
            a.step_index
                .cmp(&b.step_index)
                .then_with(|| a.present_side.as_str().cmp(b.present_side.as_str()))
                .then_with(|| a.pairing_label.cmp(&b.pairing_label))
                .then_with(|| a.metric.map(|m| m.name).cmp(&b.metric.map(|m| m.name)))
        });
        let mut table = crate::cli::new_table();
        table.set_header(vec!["SIDE", "TEST", "PHASE", "METRIC", "VALUE"]);
        for c in coverage {
            // A whole one-sided phase with no readable metric renders `—` in the
            // METRIC + VALUE columns.
            let metric_cell = c.metric.map(|m| m.name).unwrap_or("—");
            // Bare {:.2} with NO display_unit — matching the noise aggregate
            // findings table,
            // so a unit-carrying metric renders consistently across all of them.
            let value_cell = match c.value {
                Some(v) => format!("{v:.2}"),
                None => "—".to_string(),
            };
            table.add_row(vec![
                Cell::new(c.present_side.as_str()),
                Cell::new(c.pairing_label.as_str()),
                Cell::new(format!("{}: {}", c.step_index, c.label)),
                Cell::new(metric_cell),
                Cell::new(value_cell),
            ]);
        }
        lines.push(table.to_string());
    }
    lines
}

/// Render the scalar findings table for `perf-delta`.
///
/// Extracted from [`compare_partitions`] verbatim; the
/// `--phases-only` gate stays at the call site so this prints
/// unconditionally when invoked.
fn print_scalar_findings_table(report: &CompareReport, label_a: &str, label_b: &str) {
    use comfy_table::{Cell, Color};
    let mut table = crate::cli::new_table();
    table.set_header(vec!["TEST", "METRIC", label_a, label_b, "DELTA", "VERDICT"]);
    for f in &report.findings {
        let (verdict_text, verdict_color) = match f.kind {
            FindingKind::Regression => ("REGRESSION", Color::Red),
            FindingKind::Improvement => ("improvement", Color::Green),
            // Directionless metric: shown, never gated. Neutral color.
            FindingKind::Informational => ("informational", Color::Blue),
        };
        // PairingKey's first slot is scenario; subsequent slots
        // are the pairing-dim values in canonical order. Joining
        // with `/` produces a label whose shape mirrors the
        // pairing-dim count — so a comparison that pairs on
        // (topology, work_type) renders a `scenario/topology/work_type`
        // label, while a comparison that slices on most dims
        // renders a shorter identifier. The operator can always
        // cross-reference the "pairing on:" header line above to
        // see what each segment means.
        let label = f.pairing_key.0.join("/");
        table.add_row(vec![
            Cell::new(label),
            Cell::new(f.metric.name),
            Cell::new(format!("{:.2}", f.val_a)),
            Cell::new(format!("{:.2}", f.val_b)),
            Cell::new(format!("{:+.2}{}", f.delta, f.metric.display_unit)),
            Cell::new(verdict_text).fg(verdict_color),
        ]);
    }
    println!("{table}");
}

/// Render the scalar summary block for `perf-delta` —
/// regressions / improvements / unchanged + skipped-failed +
/// per-group pass counts + new_in_b / removed_from_a. All lines
/// describe the scalar findings table; the `--phases-only` gate
/// stays at the call site so this prints unconditionally when
/// invoked.
fn print_summary_block(
    report: &CompareReport,
    avg_a: &Option<Vec<AveragedGroup>>,
    avg_b: &Option<Vec<AveragedGroup>>,
    label_a: &str,
    label_b: &str,
) {
    println!();
    println!(
        "summary: {} regressions, {} improvements, {} informational, {} unchanged",
        report.regressions, report.improvements, report.informational, report.unchanged,
    );
    if report.excluded_pairs > 0 {
        println!(
            "  {} pairing-key row pair(s) excluded from regression math because one \
             or both sides was excluded (failed, inconclusive, skipped, or an inverted expected-failure run)",
            report.excluded_pairs,
        );
    }
    if let (Some(avg_a), Some(avg_b)) = (avg_a, avg_b) {
        let block = format_per_group_pass_counts(avg_a, avg_b, label_a, label_b);
        if !block.is_empty() {
            print!("{block}");
        }
    }
    if report.new_in_b > 0 {
        println!(
            "  {} row(s) new in '{}' (no matching key in '{}')",
            report.new_in_b, label_b, label_a,
        );
    }
    if report.removed_from_a > 0 {
        println!(
            "  {} row(s) removed from '{}' (no matching key in '{}')",
            report.removed_from_a, label_a, label_b,
        );
    }
    for line in format_coverage_diff_lines(report, label_a, label_b) {
        println!("{line}");
    }
}

/// Render the coverage-diff lines (metrics present on exactly one side of a
/// paired row) for [`print_summary_block`]. Pure (returns the lines, empty
/// when there are no coverage diffs) so the present/absent label mapping by
/// [`ComparePartition`] is unit-testable without capturing stdout.
pub(crate) fn format_coverage_diff_lines(
    report: &CompareReport,
    label_a: &str,
    label_b: &str,
) -> Vec<String> {
    if report.coverage_diffs.is_empty() {
        return Vec::new();
    }
    let mut lines = vec![format!(
        "  {} metric(s) present on only one side (coverage difference, \
         not a regression):",
        report.coverage_diffs.len(),
    )];
    for cd in &report.coverage_diffs {
        // present_side names the side that HAS the metric; the other is absent.
        let (present, absent) = match cd.present_side {
            ComparePartition::A => (label_a, label_b),
            ComparePartition::B => (label_b, label_a),
        };
        lines.push(format!(
            "    {} / {} = {:.2} in '{}', absent in '{}'",
            cd.pairing_key.0.join("/"),
            cd.metric.name,
            cd.value,
            present,
            absent,
        ));
    }
    lines
}

/// Print the host-context delta for `perf-delta`. Same
/// first-Some(host) baseline `compare_partitions` uses — picking
/// representative hosts off the partitioned sidecars rather than
/// the full pool so the delta reflects what actually fed the
/// comparison.
fn print_host_context_delta(
    pool: &[crate::test_support::SidecarResult],
    rows: &[GauntletRow],
    filter_a: &RowFilter,
    filter_b: &RowFilter,
    label_a: &str,
    label_b: &str,
) {
    // Zip the pool with the pre-computed `rows` (built once above
    // via `pool.iter().map(sidecar_to_row).collect()`) so the
    // per-side filter reuses the existing row instead of calling
    // `sidecar_to_row` a second and third time. `pool` and `rows`
    // are the same length and same iteration order by construction.
    let sidecars_a: Vec<&crate::test_support::SidecarResult> = pool
        .iter()
        .zip(rows.iter())
        .filter(|(_, r)| filter_a.matches(r))
        .map(|(s, _)| s)
        .collect();
    let sidecars_b: Vec<&crate::test_support::SidecarResult> = pool
        .iter()
        .zip(rows.iter())
        .filter(|(_, r)| filter_b.matches(r))
        .map(|(s, _)| s)
        .collect();
    let host_a = sidecars_a.iter().find_map(|s| s.host.as_ref());
    let host_b = sidecars_b.iter().find_map(|s| s.host.as_ref());
    print!("{}", format_host_delta(host_a, host_b, label_a, label_b));
}

/// Render the host-context delta section of `perf-delta`
/// as a block of text ready to `print!`. Extracted as a pure
/// function of `(Option<&HostContext>, Option<&HostContext>, &str,
/// &str)` so the five match arms can be unit-tested without
/// fixturing a real run directory.
///
/// The returned string is either empty (when both sides have no
/// host data — nothing to print) or ends with a newline so callers
/// can chain further output. Single-side cases print a clear
/// "captured in X only, delta unavailable" message rather than
/// silently suppressing the section — a mixed-tooling-version run
/// comparison should surface the asymmetry.
/// Format the one-line averaging-mode header that prints above
/// the comparison table.
///
/// Pure function of (`pre_agg_a`, `pre_agg_b`, `a`, `b`) so the
/// exact-string contract — the operator-visible "averaged across
/// N runs (A) and M runs (B)" surface — can be unit-tested
/// without capturing stdout from `compare_partitions`.
///
/// `pre_agg_a` / `pre_agg_b` are the post-typed-filter contributor
/// row counts (i.e. the number of sidecar rows that fed
/// [`group_and_average_by`]), NOT the post-aggregation unique-key
/// counts. The two answer different operator questions; the
/// header surfaces the contributor count because that's the
/// "how many trials got folded?" intuition the averaging fold
/// is actually delivering.
pub(crate) fn format_average_header(
    pre_agg_a: usize,
    pre_agg_b: usize,
    a: &str,
    b: &str,
) -> String {
    format!("averaged across {pre_agg_a} runs ({a}) and {pre_agg_b} runs ({b})")
}

/// Format the per-group `passes_observed/total_observed` block
/// that prints below the summary line.
///
/// Pure function of (`avg_a`, `avg_b`, `a`, `b`) so the rendered
/// surface — one line per (scenario, topology, work_type) group
/// present on either side, with `N/M` per side and `-` for any
/// side that lacks the group — can be unit-tested without
/// capturing stdout. Returns the trailing-newline-terminated
/// block, or empty string when neither side has groups.
///
/// Line shape:
/// `  scenario/topology/work_type: {a}=N/M {b}=N/M`
///
/// The leading two-space indent matches the sibling
/// `summary:` block's continuation lines (e.g.
/// `"  N (scenario, topology, work_type) row pair(s) skipped..."`)
/// so the per-group block reads as a continuation of the same
/// summary section. A blank line separates this block from the
/// preceding `summary:` line for readability.
///
/// Groups present on only one side render `-` for the missing
/// side (also counted in `compare_rows`' `new_in_b` /
/// `removed_from_a` upstream — the per-group block surfaces the
/// asymmetry by name so the operator can see *which* groups went
/// missing without cross-referencing the summary counters).
pub(crate) fn format_per_group_pass_counts(
    avg_a: &[AveragedGroup],
    avg_b: &[AveragedGroup],
    a: &str,
    b: &str,
) -> String {
    type SummaryKey<'a> = (&'a str, &'a str, &'a str);
    type SummaryValue<'a> = (Option<&'a AveragedGroup>, Option<&'a AveragedGroup>);
    let mut keys: BTreeMap<SummaryKey<'_>, SummaryValue<'_>> = BTreeMap::new();
    for ar in avg_a {
        let k = (
            ar.row.scenario.as_str(),
            ar.row.topology.as_str(),
            ar.row.work_type.as_str(),
        );
        keys.entry(k).or_insert((None, None)).0 = Some(ar);
    }
    for br in avg_b {
        let k = (
            br.row.scenario.as_str(),
            br.row.topology.as_str(),
            br.row.work_type.as_str(),
        );
        keys.entry(k).or_insert((None, None)).1 = Some(br);
    }
    if keys.is_empty() {
        return String::new();
    }
    let mut out = String::new();
    out.push('\n');
    out.push_str(
        "per-group pass counts (passes/total + skip/inconc/fail breakdown when non-zero):\n",
    );
    for ((scn, topo, wt), (ka, kb)) in keys.into_iter() {
        let fmt_side = |r: Option<&AveragedGroup>| -> String {
            let Some(x) = r else {
                return "-".to_string();
            };
            // Mirror format_dimension_summary's 4-state breakdown —
            // operators reading per-group lines must be able to
            // distinguish skip / inconclusive / fail buckets, not
            // see them collapsed into the (total - pass) denominator
            // gap. Skip silently rendering buckets that are zero so
            // the common-case "all passed" line stays terse.
            let mut s = format!("{}/{}", x.passes_observed, x.total_observed);
            let mut extras: Vec<String> = Vec::with_capacity(3);
            if x.skips_observed > 0 {
                extras.push(format!("{} skip", x.skips_observed));
            }
            if x.inconclusives_observed > 0 {
                extras.push(format!("{} inc", x.inconclusives_observed));
            }
            if x.failures_observed > 0 {
                extras.push(format!("{} fail", x.failures_observed));
            }
            if !extras.is_empty() {
                s.push_str(&format!(" ({})", extras.join(", ")));
            }
            s
        };
        out.push_str(&format!(
            "  {scn}/{topo}/{wt}: {a}={pa} {b}={pb}\n",
            pa = fmt_side(ka),
            pb = fmt_side(kb),
        ));
    }
    out
}

pub(crate) fn format_host_delta(
    host_a: Option<&crate::host_context::HostContext>,
    host_b: Option<&crate::host_context::HostContext>,
    a: &str,
    b: &str,
) -> String {
    match (host_a, host_b) {
        (Some(ha), Some(hb)) => {
            let delta = ha.diff(hb);
            if delta.is_empty() {
                // Identical hosts: surface arch when both sides
                // carry it so the operator sees WHAT is identical
                // (the two runs share x86_64 vs both being aarch64
                // is the operator's question). When
                // either side leaves arch as `None` (pre-host-
                // context-landing archive, or arch probe failed
                // on at least one side), fall through to the
                // bare "identical" message — emitting a partial
                // hint would mislead the reader into thinking
                // the silent side disagreed.
                match (ha.arch.as_deref(), hb.arch.as_deref()) {
                    (Some(arch_a), Some(arch_b)) if arch_a == arch_b => {
                        format!("\nhost: identical between '{a}' and '{b}' (arch: {arch_a})\n",)
                    }
                    _ => format!("\nhost: identical between '{a}' and '{b}'\n"),
                }
            } else {
                format!("\nhost delta ('{a}' → '{b}'):\n{delta}")
            }
        }
        (Some(_), None) => {
            format!("\nhost: captured in '{a}' only, delta unavailable\n")
        }
        (None, Some(_)) => {
            format!("\nhost: captured in '{b}' only, delta unavailable\n")
        }
        (None, None) => String::new(),
    }
}
