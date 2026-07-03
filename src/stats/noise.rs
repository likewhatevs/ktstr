//! Noise-adjusted A/B comparison for perf-delta's `--noise-adjust N` mode.
//!
//! The default perf-delta compare runs each side ONCE and gates on a fixed
//! relative threshold, which conflates a real regression with run-to-run
//! measurement noise: a scheduling-dominated metric can swing several percent
//! between identical runs (e.g. schbench pipe throughput — see
//! `src/workload/schbench/validation.md`), tripping a fixed 1-5% threshold on noise
//! alone. The noise-adjusted mode runs each side N times and decides from the
//! OBSERVED spread instead:
//!
//! - each side is summarized as `mean` over its `[min, max]` band across N runs;
//! - a change is SIGNIFICANT only when B's mean falls OUTSIDE A's observed
//!   `[min, max]` band — a move larger than A's own run-to-run noise;
//! - a metric is TOO NOISY (verdict untrustworthy) when either side's relative
//!   spread `(max - min) / |mean|` exceeds a threshold (default 1%), OR either
//!   side realized fewer than 2 samples (a single-point band has no measurable
//!   spread — e.g. a per-side run failed). Range is the
//!   most outlier-sensitive dispersion measure: one anomalous run inflates it and
//!   trips the gate — deliberate for a conservative noise gate (a single wild run
//!   flags the metric), but it reacts more strongly than a stddev / CV would.
//!
//! This is a deliberately simple, non-parametric, range-based test (no
//! distributional assumption), matched to the small N (3-5) the mode runs. The
//! range UNDER-estimates the true spread at small N, so the significance test
//! gets MORE sensitive as N shrinks: under the null (B drawn from A's
//! distribution), a single new draw lands outside A's observed range with
//! probability ~`2/(N+1)` (~50% at N=3, ~33% at N=5). The test compares B's MEAN,
//! which concentrates (SD `σ/√N`) and so beats that single-point bound, but the
//! floor still rises at small N — and the `too_noisy` gate is NOT independent of
//! the significance test (both read the same spread), so a residual confident
//! false-positive tail (significant AND not-noisy on identical inputs) survives.
//! Recommendation: use `N >= 5` when the gate must be trusted; `too_noisy` bounds
//! generally-noisy metrics but is not a guarantee. A confidence-interval / t-test
//! variant is a possible follow-up.

/// Magnitude below which `|mean|` is treated as zero for the relative-spread
/// ratio. A metric whose mean is essentially zero but whose runs differ has
/// unbounded relative spread, reported as `f64::INFINITY` (which the gate always
/// flags); the epsilon avoids dividing by a true zero. Magnitude-bearing metrics
/// (rps, latencies, byte rates) sit far above this; count/ratio metrics that are
/// legitimately exactly 0 (migrations, stuck_count, off-cpu %, …) have
/// `max == min` and so report spread 0, never INFINITY — the INFINITY branch is
/// reachable only by a hypothetical signed metric straddling zero, where flagging
/// it noisy is the correct-conservative call.
pub(crate) const ZERO_MEAN_EPS: f64 = 1e-9;

/// Summary of one side's N runs of a single metric.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SideSummary {
    /// Number of runs summarized.
    pub n: usize,
    /// Arithmetic mean of the runs (`0.0` when `n == 0`).
    pub mean: f64,
    /// Smallest run value (`0.0` when `n == 0`).
    pub min: f64,
    /// Largest run value (`0.0` when `n == 0`).
    pub max: f64,
    /// Relative spread as a PERCENT: `(max - min) / |mean| * 100`. `0.0` when
    /// `n < 2` (no spread observable) or `max == min`. `f64::INFINITY` when the
    /// runs differ but `|mean|` is below `ZERO_MEAN_EPS` (a metric oscillating
    /// around zero is maximally noisy in relative terms).
    pub spread_pct: f64,
}

impl SideSummary {
    /// Summarize `samples` — one metric's value from each of N runs. Empty input
    /// yields an all-zero summary (`n == 0`); `--noise-adjust N` is expected to
    /// run with `N >= 2` for a meaningful spread (`N == 1` reports `spread 0` and
    /// a single-point band, so any B change reads as significant).
    pub fn of(samples: &[f64]) -> SideSummary {
        let n = samples.len();
        if n == 0 {
            return SideSummary {
                n: 0,
                mean: 0.0,
                min: 0.0,
                max: 0.0,
                spread_pct: 0.0,
            };
        }
        let mean = samples.iter().sum::<f64>() / n as f64;
        let min = samples.iter().copied().fold(f64::INFINITY, f64::min);
        let max = samples.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let spread_pct = if max == min {
            0.0
        } else if mean.abs() < ZERO_MEAN_EPS {
            // Nonzero range around a ~zero mean: relative spread is unbounded.
            f64::INFINITY
        } else {
            (max - min) / mean.abs() * 100.0
        };
        SideSummary {
            n,
            mean,
            min,
            max,
            spread_pct,
        }
    }

    /// Override the centroid with a pooled mean (`Σnumerator/Σdenominator`)
    /// while KEEPING the per-run `[min, max]` band. For Rate metrics under
    /// `--noise-adjust`: the band still measures run-to-run variability from
    /// the per-run ratios, but the compared centroid is the duration-weighted
    /// pooled rate the metric registry documents as the cross-run Rate value,
    /// so `--noise-adjust` and `--average` agree on a Rate's central value. The
    /// relative spread is recomputed against the pooled mean; `n` and the band
    /// are unchanged (so the `< 2` too-noisy guard still applies).
    pub fn with_pooled_mean(mut self, pooled_mean: f64) -> SideSummary {
        self.mean = pooled_mean;
        self.spread_pct = if self.max == self.min {
            0.0
        } else if pooled_mean.abs() < ZERO_MEAN_EPS {
            f64::INFINITY
        } else {
            (self.max - self.min) / pooled_mean.abs() * 100.0
        };
        self
    }
}

/// Where B's mean falls relative to A's observed `[min, max]` band.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    /// B's mean is above A's `max`.
    Higher,
    /// B's mean is below A's `min`.
    Lower,
    /// B's mean is within A's `[min, max]` — no change beyond A's noise.
    Within,
}

/// The noise-adjusted verdict for one metric across N runs per side.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NoiseVerdict {
    /// A side (baseline) summary.
    pub a: SideSummary,
    /// B side (candidate) summary.
    pub b: SideSummary,
    /// Where B's mean sits relative to A's `[min, max]` band.
    pub direction: Direction,
    /// B's mean is outside A's band (`direction != Within`) — a change larger
    /// than A's run-to-run noise. Direction-agnostic; the caller applies the
    /// metric's polarity to decide regression vs improvement.
    pub significant: bool,
    /// The verdict is not trustworthy: either side's `spread_pct` exceeds the
    /// threshold, OR either side realized fewer than 2 samples (a single-point
    /// band has no measurable spread). Such a metric is flagged, never gated.
    pub too_noisy: bool,
}

/// Decide the noise-adjusted verdict for one metric from A's and B's per-run
/// samples. `spread_threshold_pct` is the relative-spread limit in PERCENT
/// (e.g. `1.0` for 1%): a side whose spread STRICTLY exceeds it marks the
/// verdict `too_noisy`. A side with fewer than 2 samples is ALSO
/// `too_noisy` (a single-point band has no measurable spread; never gate
/// on it). Significance is range-based — B's mean outside A's
/// `[min, max]` — and direction-agnostic (either way is significant).
pub fn noise_verdict(
    a_samples: &[f64],
    b_samples: &[f64],
    spread_threshold_pct: f64,
) -> NoiseVerdict {
    noise_verdict_from(
        SideSummary::of(a_samples),
        SideSummary::of(b_samples),
        spread_threshold_pct,
    )
}

/// Decide the verdict from two already-summarized sides. Split from
/// [`noise_verdict`] so the Rate consumer in `noise_findings` can inject a
/// pooled `Σnum/Σden` centroid ([`SideSummary::with_pooled_mean`]) while the
/// `[min, max]` band stays per-run.
pub fn noise_verdict_from(
    a: SideSummary,
    b: SideSummary,
    spread_threshold_pct: f64,
) -> NoiseVerdict {
    let direction = if b.mean > a.max {
        Direction::Higher
    } else if b.mean < a.min {
        Direction::Lower
    } else {
        Direction::Within
    };
    let significant = direction != Direction::Within;
    // A side with fewer than 2 realized samples has NO measurable spread:
    // SideSummary::of([x]) reports spread_pct=0, a degenerate single-point
    // band that reads any B change as significant-and-clean — the exact
    // pure-noise confident verdict this mode exists to prevent. Runs are
    // REQUESTED >= 2 (the --noise-adjust clap gate), but a per-side run can
    // fail (noise_dual_run logs and continues), so a side can REALIZE < 2.
    // Treat that as too noisy to gate on rather than a trustworthy verdict.
    let too_noisy = a.spread_pct > spread_threshold_pct
        || b.spread_pct > spread_threshold_pct
        || a.n < 2
        || b.n < 2;
    NoiseVerdict {
        a,
        b,
        direction,
        significant,
        too_noisy,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(x: f64, y: f64) -> bool {
        (x - y).abs() < 1e-9
    }

    #[test]
    fn side_summary_empty_is_all_zero() {
        let s = SideSummary::of(&[]);
        assert_eq!(
            s,
            SideSummary {
                n: 0,
                mean: 0.0,
                min: 0.0,
                max: 0.0,
                spread_pct: 0.0
            }
        );
    }

    #[test]
    fn side_summary_single_has_zero_spread() {
        // One sample: min == max == mean, no spread observable.
        let s = SideSummary::of(&[42.0]);
        assert_eq!((s.n, s.mean, s.min, s.max), (1, 42.0, 42.0, 42.0));
        assert_eq!(s.spread_pct, 0.0);
    }

    #[test]
    fn noise_verdict_flags_side_with_fewer_than_two_samples_too_noisy() {
        // A per-side run can fail (noise_dual_run logs + continues), so a
        // side can realize <2 samples even though --noise-adjust requires
        // N>=2. A single-sample side has a degenerate zero-spread band that
        // would otherwise read any B change as a CONFIDENT verdict — the
        // exact pure-noise false gate the mode exists to prevent. It must be
        // too_noisy (flagged, never gated), even when B is "significant".
        // A realized 1 sample; B 3 clean samples with a real shift.
        let v = noise_verdict(&[100.0], &[130.0, 130.0, 130.0], 1.0);
        assert_eq!(v.a.n, 1);
        assert!(v.significant, "B mean is outside A's single-point band");
        assert!(
            v.too_noisy,
            "a side with <2 realized samples must be too_noisy, not confident",
        );
        // Symmetric: B realized 1 sample.
        let v = noise_verdict(&[100.0, 100.0, 100.0], &[130.0], 1.0);
        assert_eq!(v.b.n, 1);
        assert!(v.too_noisy, "a degenerate B side must also be too_noisy");
        // Control: both sides have >=2 clean samples (zero spread) with a
        // real shift — a trustworthy significant verdict, NOT over-flagged.
        let v = noise_verdict(&[100.0, 100.0], &[130.0, 130.0], 1.0);
        assert!(
            v.significant && !v.too_noisy,
            "n>=2 clean sides stay gateable; the <2 guard must not over-trigger",
        );
    }

    #[test]
    fn side_summary_mean_min_max_spread() {
        // mean (90+110+100)/3 = 100; range 20; spread 20/100 = 20%.
        let s = SideSummary::of(&[90.0, 110.0, 100.0]);
        assert!(approx(s.mean, 100.0));
        assert_eq!((s.min, s.max), (90.0, 110.0));
        assert!(approx(s.spread_pct, 20.0));
    }

    #[test]
    fn side_summary_identical_runs_have_zero_spread() {
        let s = SideSummary::of(&[7.0, 7.0, 7.0]);
        assert_eq!(
            s.spread_pct, 0.0,
            "identical runs => no spread, never noisy"
        );
    }

    #[test]
    fn side_summary_near_zero_mean_with_variance_is_infinite_spread() {
        // mean ~0 but runs differ => relative spread is unbounded -> INFINITY,
        // which any finite threshold flags as too_noisy.
        let s = SideSummary::of(&[-1.0, 1.0]);
        assert!(approx(s.mean, 0.0));
        assert_eq!(s.spread_pct, f64::INFINITY);
    }

    #[test]
    fn verdict_b_within_a_band_is_not_significant() {
        // A: [90,110] mean 100; B mean 100 within band -> Within, not significant.
        let v = noise_verdict(&[90.0, 110.0, 100.0], &[95.0, 105.0, 100.0], 100.0);
        assert_eq!(v.direction, Direction::Within);
        assert!(!v.significant);
    }

    #[test]
    fn verdict_b_above_a_max_is_higher_and_significant() {
        // A band [90,110]; B mean 150 > 110 -> Higher, significant.
        let v = noise_verdict(&[90.0, 110.0], &[150.0, 150.0], 100.0);
        assert_eq!(v.direction, Direction::Higher);
        assert!(v.significant);
    }

    #[test]
    fn verdict_b_below_a_min_is_lower_and_significant() {
        let v = noise_verdict(&[90.0, 110.0], &[50.0, 50.0], 100.0);
        assert_eq!(v.direction, Direction::Lower);
        assert!(v.significant);
    }

    #[test]
    fn verdict_b_exactly_on_boundary_is_within() {
        // B mean == A.max exactly: inclusive band, not significant.
        let v = noise_verdict(&[90.0, 110.0], &[110.0, 110.0], 100.0);
        assert_eq!(v.direction, Direction::Within);
        assert!(!v.significant);
    }

    #[test]
    fn verdict_too_noisy_when_a_spread_exceeds_threshold() {
        // A spread 20% > 1% threshold -> too_noisy even though B is within.
        let v = noise_verdict(&[90.0, 110.0, 100.0], &[100.0, 100.0, 100.0], 1.0);
        assert!(v.too_noisy);
        assert!(approx(v.a.spread_pct, 20.0));
        assert_eq!(v.b.spread_pct, 0.0);
    }

    #[test]
    fn verdict_too_noisy_when_b_spread_exceeds_threshold() {
        // A clean, B noisy -> still too_noisy (either side trips the gate).
        let v = noise_verdict(&[100.0, 100.0], &[80.0, 120.0], 1.0);
        assert!(v.too_noisy);
        assert!(approx(v.b.spread_pct, 40.0));
    }

    #[test]
    fn verdict_not_noisy_when_both_within_threshold() {
        // Both spreads 1% exactly; strict `>` => NOT noisy at a 1% threshold.
        let v = noise_verdict(&[99.5, 100.5], &[99.5, 100.5], 1.0);
        assert!(approx(v.a.spread_pct, 1.0));
        assert!(
            !v.too_noisy,
            "spread == threshold is not over it (strict >)"
        );
    }

    #[test]
    fn verdict_significant_and_clean_is_a_confident_change() {
        // The CONFIDENT-change case: B far outside A's band AND both sides clean.
        // This is what a real regression looks like (vs a noise-driven flag).
        let v = noise_verdict(&[100.0, 100.5, 99.5], &[150.0, 150.5, 149.5], 1.0);
        assert!(v.significant);
        assert!(
            !v.too_noisy,
            "a large change with low spread is a confident verdict"
        );
        assert_eq!(v.direction, Direction::Higher);
    }

    /// Demonstration on the real schbench engine (host-side `run_standalone`, no
    /// VM — each run is the short `-r` window, far under the 15s/run budget).
    /// Three windows per config feed the per-run request-p50 into `noise_verdict`,
    /// exercising both arms:
    ///  - A vs a much heavier B (10x matrix ops): a large real change the tool
    ///    must flag `significant` (the hard assertion + the true-positive arm).
    ///  - A vs A (identical config): the false-positive check — any `significant`
    ///    verdict must coincide with the `too_noisy` gate firing (the gate is the
    ///    backstop; a confident significant verdict on identical configs would be
    ///    a false positive).
    ///
    /// Prints the observed spreads + verdicts so the behavior is inspectable.
    #[test]
    fn schbench_noise_adjust_demonstration() {
        use crate::workload::{SchbenchConfig, run_standalone};

        const RUN_SECS: u64 = 2; // << 15s/run; host-side, no VM boot
        const N: usize = 3;

        // request-latency p50 (us): index 1 of [20,50,90,99,99.9]. sleep_usec(0)
        // makes it pure matrix work, so `operations` scales it ~linearly and the
        // heavy config separates cleanly from the light one.
        let p50 = |cfg: &SchbenchConfig| run_standalone(cfg, RUN_SECS).request_pcts_us[1] as f64;
        let samples = |cfg: &SchbenchConfig| (0..N).map(|_| p50(cfg)).collect::<Vec<f64>>();

        let light = SchbenchConfig::default()
            .worker_threads(2)
            .sleep_usec(0)
            .operations(5);
        let heavy = SchbenchConfig::default()
            .worker_threads(2)
            .sleep_usec(0)
            .operations(50);

        let a = samples(&light);
        let b_same = samples(&light); // identical config: the no-regression arm
        let b_heavy = samples(&heavy); // 10x ops: the real-regression arm

        let same = noise_verdict(&a, &b_same, 1.0);
        let diff = noise_verdict(&a, &b_heavy, 1.0);

        eprintln!(
            "\nnoise-adjust schbench demo (request p50 us, N={N}, {RUN_SECS}s/run):\n  \
             A       = {:.0} [{:.0}-{:.0}] spread {:.2}%\n  \
             A vs A  : B {:.0} [{:.0}-{:.0}] spread {:.2}% -> significant={} too_noisy={} {:?}\n  \
             A vs B' : B {:.0} [{:.0}-{:.0}] spread {:.2}% -> significant={} too_noisy={} {:?}",
            same.a.mean,
            same.a.min,
            same.a.max,
            same.a.spread_pct,
            same.b.mean,
            same.b.min,
            same.b.max,
            same.b.spread_pct,
            same.significant,
            same.too_noisy,
            same.direction,
            diff.b.mean,
            diff.b.min,
            diff.b.max,
            diff.b.spread_pct,
            diff.significant,
            diff.too_noisy,
            diff.direction,
        );

        // True positive: 10x matrix ops is a large real increase in request latency.
        assert!(
            diff.significant && diff.direction == Direction::Higher,
            "10x operations must read as a significant increase: A={:?} B={:?}",
            diff.a,
            diff.b,
        );
        // Real separation dwarfs the noise: the heavy mean clears A's whole
        // observed band (reliable at 10x, not flaky).
        assert!(
            diff.b.mean > a.iter().copied().fold(f64::NEG_INFINITY, f64::max),
            "heavy mean {:.0} should clear A's band max {:.0}",
            diff.b.mean,
            same.a.max,
        );
        // The A-vs-A arm is REPORTED above for real-data evaluation, not hard
        // asserted: at small N a clean-but-narrow A band can let B's mean drift
        // just outside it (the residual confident-FP tail the module doc notes),
        // so a hard `!(significant && !too_noisy)` assertion here would be flaky.
        // The no-false-positive logic is pinned deterministically by the unit
        // tests (verdict_b_within_a_band_is_not_significant et al.); this test only
        // hard-asserts the reliable true-positive and its separation from A's noise.
    }
}
