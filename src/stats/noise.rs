//! Noise-adjusted A/B comparison for perf-delta's `--noise-adjust N` mode.
//!
//! The default perf-delta compare runs each side ONCE and gates on a fixed
//! relative threshold, which conflates a real regression with run-to-run
//! measurement noise: a scheduling-dominated metric can swing several percent
//! between identical runs (e.g. schbench pipe throughput — see
//! `src/workload/schbench/validation.md`), tripping a fixed threshold on noise
//! alone. The noise-adjusted mode runs each side N times and decides whether the
//! two sides are distinguishable GIVEN their observed run-to-run variability.
//!
//! A metric is a confident regression when it is BOTH:
//!
//! - SEPARATED — the two sides' distributions do not overlap in a
//!   variability-aware sense. Two independent arms, OR'd: a two-sided Welch
//!   (unequal-variance) t-test rejecting equal means at [`NOISE_ALPHA`]
//!   (computed only when both sides have `n >= 2` and a finite, positive
//!   standard error), OR the observed `[min, max]` bands being fully disjoint.
//!   The Welch arm handles near-but-overlapping separations; the disjoint-band
//!   floor is a deterministic, distribution-free guarantee that a
//!   fully-separated move is NEVER hidden by a high-variance side.
//! - MATERIAL — the mean-to-mean delta clears BOTH the metric's `default_abs`
//!   AND its relative threshold (the same dual-gate the scalar
//!   [`crate::stats::compare`] pass applies), so a statistically-separated but
//!   trivially-small change stays Stable. Materiality is metric-specific, so the
//!   `classify_noise` caller applies it — this module reports only the
//!   metric-agnostic separation.
//!
//! Two trust flags are carried alongside, and — unlike the old range-spread gate
//! — NEITHER suppresses a separated + material regression:
//!
//! - `insufficient_samples` (a side realized `< 2` runs, so variance / Welch are
//!   undefined — a per-side run can fail even though `--noise-adjust` requests
//!   `N >= 2`) is a HARD gate: such a verdict is flagged Noisy and never gates.
//! - `high_spread` (a side's relative spread `(max - min) / |mean|` exceeds the
//!   threshold) is ADVISORY only: it annotates the row so the operator sees a
//!   noisy side, but a clearly-separated regression is still reported. The prior
//!   design let `high_spread` suppress the verdict, which INVERTED signal and
//!   noise — a real regression's degraded side is intrinsically higher-variance,
//!   so the biggest regressions were the ones dropped.
//!
//! The Welch t-test needs per-side variance, so [`SideSummary`] retains the
//! Bessel (N-1) sample variance; the `[min, max]` band still drives the
//! distribution-free floor and the advisory spread. Recommendation: use
//! `N >= 5` for a well-powered Welch arm. The disjoint-band floor keeps large
//! regressions detectable even at small N, but it cuts both ways: at `N < 5` two
//! same-distribution sides land on disjoint bands by chance ~33% of the time at
//! `N = 2` and ~10% at `N = 3` (well above the Welch arm's 0.05), so the floor
//! can also report a chance-separated move. The materiality dual-gate is the only
//! backstop against that, and it does NOT cover a chance-disjoint move whose mean
//! delta happens to clear `default_abs`/`default_rel` — another reason to prefer
//! `N >= 5`, where chance-disjoint bands fall below the Welch alpha. The floor is
//! outlier-fragile in the OTHER direction too: a single wild run bridging the gap
//! collapses the disjoint bands AND inflates that side's variance, weakening the
//! Welch arm at the same time, so a large real regression can read as
//! not-separated (a false negative) — the same variance-driven suppression this
//! change exists to defeat, arriving via one anomalous run instead of intrinsic
//! spread. `N >= 5` mitigates this too (a lone outlier shifts min/max and inflates
//! the variance less, relative to a real signal).

use statrs::distribution::{ContinuousCDF, StudentsT};
use statrs::statistics::Statistics;

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

/// Significance level for the Welch two-sample t-test arm of the separation
/// test: a metric whose two-sided Welch p-value is below this is SEPARATED on
/// the parametric arm (the disjoint-band floor is an independent OR that needs
/// no alpha). Fixed, not user-tunable — separation is a correctness signal, not
/// a knob; the operator tunes N (`--noise-adjust`) and the materiality
/// thresholds (`--noise-spread-threshold` is the advisory spread flag, not this).
pub(crate) const NOISE_ALPHA: f64 = 0.05;

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
    /// around zero is maximally noisy in relative terms). Drives the ADVISORY
    /// `high_spread` flag only — never gates on its own.
    pub spread_pct: f64,
    /// Bessel-corrected (N-1) sample variance of the runs — the dispersion the
    /// Welch t-test consumes (`var / n` is the per-side squared standard error).
    /// `f64::NAN` when `n < 2` (variance undefined for a single point), which the
    /// Welch arm guards against via `insufficient_samples`. Distinct from
    /// `spread_pct` (an outlier-sensitive RANGE ratio used only for the advisory
    /// flag); `var` is the full-sample dispersion the significance test needs.
    pub var: f64,
    /// Arithmetic mean of the RAW samples — equals [`Self::mean`] EXCEPT when a
    /// Rate centroid override ([`Self::with_pooled_mean`]) replaced `mean` with
    /// the pooled `Σnum/Σden`. The Welch arm uses THIS mean (not `mean`) so its
    /// numerator (the `sample_mean` difference) stays coherent with its
    /// denominator (`var`, the variance of the SAME raw samples): a hybrid
    /// pooled-mean-over-ratio-variance t-statistic is NOT a valid two-sample test.
    /// `mean` remains the reported / materiality centroid (the pooled value for a
    /// Rate); `sample_mean` and `var` describe the per-run distribution the
    /// parametric separation test is actually computed on.
    pub sample_mean: f64,
}

impl SideSummary {
    /// Summarize `samples` — one metric's value from each of N runs. Empty input
    /// yields an all-zero summary (`n == 0`); `--noise-adjust N` is expected to
    /// run with `N >= 2` (`N == 1` reports `spread 0`, undefined variance (NaN),
    /// and a single-point band, so `noise_verdict_from` flags it
    /// `insufficient_samples` and `classify_noise` returns Noisy — a single-run
    /// side never gates as a confident regression).
    pub fn of(samples: &[f64]) -> SideSummary {
        let n = samples.len();
        if n == 0 {
            return SideSummary {
                n: 0,
                mean: 0.0,
                min: 0.0,
                max: 0.0,
                spread_pct: 0.0,
                // No samples: variance is undefined, matching the `n < 2` NaN
                // the single-sample case produces below.
                var: f64::NAN,
                sample_mean: 0.0,
            };
        }
        let mean = samples.iter().sum::<f64>() / n as f64;
        let min = samples.iter().copied().fold(f64::INFINITY, f64::min);
        let max = samples.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        // Bessel-corrected (N-1) sample variance via statrs, for the Welch arm.
        // Returns `f64::NAN` for `n < 2` (variance undefined for a single
        // point); the Welch path guards that via `insufficient_samples`.
        let var = samples.variance();
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
            var,
            // Coherent with `var`/`min`/`max` (all from `samples`). Stays equal
            // to `mean` unless a later Rate centroid override shifts `mean` to the
            // pooled value; the Welch arm always reads `sample_mean`.
            sample_mean: mean,
        }
    }

    /// Override the centroid with a pooled mean (`Σnumerator/Σdenominator`)
    /// while KEEPING the per-run `[min, max]` band. For Rate metrics under
    /// `--noise-adjust`: the band still measures run-to-run variability from
    /// the per-run ratios, but the compared centroid is the duration-weighted
    /// pooled rate the metric registry documents as the cross-run Rate value,
    /// so `--noise-adjust` and the scalar averaging compare agree on a Rate's central value. The
    /// relative spread is recomputed against the pooled mean; `n`, the band, the
    /// variance, AND `sample_mean` are unchanged. The Welch arm reads
    /// `sample_mean` + `var` (both from the raw per-run ratios), so overriding
    /// only the reported `mean` keeps the parametric test COHERENT — its numerator
    /// (the `sample_mean` difference) and denominator (`var`) reference the same
    /// per-run distribution, never the pooled centroid (a pooled-mean-over-
    /// ratio-variance t-statistic is not a valid two-sample test). `n` unchanged
    /// keeps the `< 2` insufficient-samples guard applicable.
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

/// The noise-adjusted verdict for one metric across N runs per side.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NoiseVerdict {
    /// A side (baseline) summary.
    pub a: SideSummary,
    /// B side (candidate) summary.
    pub b: SideSummary,
    /// B's distribution is SEPARATED from A's: a two-sided Welch (unequal-
    /// variance) t-test rejects equal means at [`NOISE_ALPHA`] (computed only
    /// when both sides have `n >= 2` and a finite, positive standard error), OR
    /// the observed `[min, max]` bands are fully disjoint. Direction-agnostic;
    /// the caller applies the metric's polarity + `sign(b.mean - a.mean)` to
    /// split regression from improvement, and the materiality dual-gate to
    /// decide whether the separated move is large enough to report.
    pub separated: bool,
    /// HARD gate: a side realized fewer than 2 samples, so variance and the
    /// Welch statistic are undefined (a per-side run can fail even though
    /// `--noise-adjust` requests `N >= 2`). Such a verdict is never a confident
    /// regression — it renders Noisy and never fails the gate.
    pub insufficient_samples: bool,
    /// ADVISORY only: a side's `spread_pct` exceeds the spread threshold. Unlike
    /// the old `too_noisy`, this NEVER suppresses a separated + material
    /// regression (that suppression INVERTED signal and noise — a real
    /// regression's degraded side is intrinsically higher-variance); it only
    /// annotates the row so the operator sees a noisy side.
    pub high_spread: bool,
}

/// Decide the noise-adjusted verdict for one metric from A's and B's per-run
/// samples. `spread_threshold_pct` is the relative-spread limit in PERCENT
/// (e.g. `5.0` for 5%): a side whose spread STRICTLY exceeds it sets the
/// ADVISORY `high_spread` flag, which never suppresses a verdict. Separation is
/// the Welch t-test OR the disjoint-band test (see [`noise_verdict_from`]); the
/// caller (`classify_noise`) applies the metric's materiality dual-gate and
/// polarity on top.
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
///
/// `separated` is `welch_p < NOISE_ALPHA` OR fully-disjoint `[min, max]` bands.
/// The Welch arm is formed ONLY when both sides have `n >= 2` and a finite,
/// positive pooled squared standard error, AND the Welch-Satterthwaite dof is
/// finite and positive — so [`StudentsT::new`]'s freedom precondition holds at
/// every `unwrap`. Degenerate inputs (a side `< 2` samples, or zero within-side
/// variance on both sides → `se2 == 0`) make the Welch arm `false` WITHOUT
/// panicking, and the distribution-free disjoint-band floor decides.
pub fn noise_verdict_from(
    a: SideSummary,
    b: SideSummary,
    spread_threshold_pct: f64,
) -> NoiseVerdict {
    // HARD gate: a side with < 2 realized samples has undefined variance — the
    // Welch statistic cannot be formed, and a single-point band would read any
    // move as separated. Runs are REQUESTED >= 2 (the --noise-adjust clap gate),
    // but a per-side run can fail (noise_dual_run logs and continues), so a side
    // can REALIZE < 2. Flagged, never gated as a confident regression.
    let insufficient_samples = a.n < 2 || b.n < 2;
    // Distribution-free floor: fully disjoint observed bands are unambiguous
    // separation regardless of either side's within-side spread — the
    // deterministic fix for the signal-inverting bug (a large real move whose
    // degraded side is high-variance).
    let disjoint_bands = a.max < b.min || b.max < a.min;
    // Parametric arm: two-sided Welch (unequal-variance) t-test. Formed ONLY
    // when both sides have n >= 2 (variance defined) and the pooled squared
    // standard error `se2` is finite and strictly positive; then only when the
    // Welch-Satterthwaite dof is finite and positive. These guards make every
    // StudentsT::new freedom argument valid, so the unwrap cannot panic; on any
    // degenerate input the arm is `false` and the disjoint-band floor decides.
    let welch_separated = if insufficient_samples {
        false
    } else {
        let sa = a.var / a.n as f64;
        let sb = b.var / b.n as f64;
        let se2 = sa + sb;
        if se2.is_finite() && se2 > 0.0 {
            let dof = (se2 * se2) / (sa * sa / (a.n as f64 - 1.0) + sb * sb / (b.n as f64 - 1.0));
            if dof.is_finite() && dof > 0.0 {
                // Numerator uses sample_mean (coherent with var: both from the
                // raw samples). `mean` may be a Rate pooled centroid that does not
                // match the variance's samples, which would make t incoherent.
                let t = (a.sample_mean - b.sample_mean) / se2.sqrt();
                // freedom finite & > 0 => StudentsT::new(0, 1, dof) is Ok.
                let p = 2.0 * StudentsT::new(0.0, 1.0, dof).unwrap().sf(t.abs());
                p < NOISE_ALPHA
            } else {
                false
            }
        } else {
            false
        }
    };
    let separated = welch_separated || disjoint_bands;
    // ADVISORY only (see NoiseVerdict::high_spread): flags a noisy side, never
    // suppresses a separated verdict.
    let high_spread =
        a.spread_pct > spread_threshold_pct || b.spread_pct > spread_threshold_pct;
    NoiseVerdict {
        a,
        b,
        separated,
        insufficient_samples,
        high_spread,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(x: f64, y: f64) -> bool {
        (x - y).abs() < 1e-9
    }

    #[test]
    fn side_summary_empty_field_by_field() {
        // A NaN var precludes a whole-struct assert_eq (NaN != NaN); check
        // fields: an empty side is all-zero with an undefined (NaN) variance.
        let s = SideSummary::of(&[]);
        assert_eq!(
            (s.n, s.mean, s.min, s.max, s.spread_pct),
            (0, 0.0, 0.0, 0.0, 0.0)
        );
        assert!(s.var.is_nan(), "empty side has undefined variance");
    }

    #[test]
    fn side_summary_single_has_zero_spread_and_nan_var() {
        // One sample: min == max == mean, no spread observable, variance
        // undefined (Bessel N-1 needs >= 2 points).
        let s = SideSummary::of(&[42.0]);
        assert_eq!((s.n, s.mean, s.min, s.max), (1, 42.0, 42.0, 42.0));
        assert_eq!(s.spread_pct, 0.0);
        assert!(s.var.is_nan(), "single-sample side has undefined variance");
    }

    #[test]
    fn noise_verdict_flags_side_with_fewer_than_two_samples() {
        // A per-side run can fail (noise_dual_run logs + continues), so a side
        // can realize <2 samples even though --noise-adjust requires N>=2. Such
        // a side has undefined variance and a degenerate single-point band; it
        // is a HARD gate (insufficient_samples), never a confident regression.
        // The Welch arm must be SKIPPED (dof undefined) — this asserts no panic.
        let v = noise_verdict(&[100.0], &[130.0, 130.0, 130.0], 5.0);
        assert_eq!(v.a.n, 1);
        assert!(
            v.insufficient_samples,
            "a side with <2 realized samples must be flagged insufficient",
        );
        // Symmetric: B realized 1 sample.
        let v = noise_verdict(&[100.0, 100.0, 100.0], &[130.0], 5.0);
        assert_eq!(v.b.n, 1);
        assert!(
            v.insufficient_samples,
            "a degenerate B side is also insufficient",
        );
        // Control: both sides have >=2 samples with a real, fully-separated
        // shift — separated and NOT insufficient (gateable once material).
        let v = noise_verdict(&[100.0, 100.0], &[130.0, 130.0], 5.0);
        assert!(
            v.separated && !v.insufficient_samples,
            "n>=2 sides with disjoint bands stay separated + gateable",
        );
    }

    #[test]
    fn side_summary_mean_min_max_spread_var() {
        // mean (90+110+100)/3 = 100; range 20; spread 20/100 = 20%; Bessel
        // variance (100+100+0)/(3-1) = 100.
        let s = SideSummary::of(&[90.0, 110.0, 100.0]);
        assert!(approx(s.mean, 100.0));
        assert_eq!((s.min, s.max), (90.0, 110.0));
        assert!(approx(s.spread_pct, 20.0));
        assert!(approx(s.var, 100.0), "Bessel N-1 variance");
    }

    #[test]
    fn side_summary_identical_runs_have_zero_spread_and_var() {
        let s = SideSummary::of(&[7.0, 7.0, 7.0]);
        assert_eq!(s.spread_pct, 0.0, "identical runs => no spread");
        assert_eq!(s.var, 0.0, "identical runs => zero variance");
    }

    #[test]
    fn side_summary_near_zero_mean_with_variance_is_infinite_spread() {
        // mean ~0 but runs differ => relative spread is unbounded -> INFINITY
        // (the advisory high_spread flag). Variance stays finite (Bessel
        // (1+1)/1 = 2), so the Welch arm is unaffected by the near-zero mean.
        let s = SideSummary::of(&[-1.0, 1.0]);
        assert!(approx(s.mean, 0.0));
        assert_eq!(s.spread_pct, f64::INFINITY);
        assert!(approx(s.var, 2.0));
    }

    #[test]
    fn with_pooled_mean_keeps_sample_mean_and_var_for_a_coherent_welch_arm() {
        // Rate centroid override: `mean` becomes the pooled value (reported +
        // materiality), but `sample_mean`, `var`, and the band stay the raw
        // per-run values so the Welch arm's numerator (sample_mean diff) stays
        // coherent with its denominator (var). Guards the Rate Welch-coherence fix
        // — a pooled-mean-over-ratio-variance t-statistic is not a valid test.
        let s = SideSummary::of(&[10.0, 20.0, 30.0]); // raw mean 20, var 100
        assert!(approx(s.mean, 20.0) && approx(s.sample_mean, 20.0));
        let p = s.with_pooled_mean(18.18);
        assert!(approx(p.mean, 18.18), "reported mean is the pooled centroid");
        assert!(approx(p.sample_mean, 20.0), "Welch mean stays the raw sample mean");
        assert!(approx(p.var, 100.0), "variance stays the raw-sample variance");
        assert_eq!((p.min, p.max), (10.0, 30.0), "band stays raw");
    }

    #[test]
    fn verdict_overlapping_bands_equal_means_not_separated() {
        // A [90,110] mean 100; B [95,105] mean 100. Bands overlap and the means
        // are equal -> Welch t=0 (p=1) and no disjoint bands -> not separated.
        let v = noise_verdict(&[90.0, 110.0, 100.0], &[95.0, 105.0, 100.0], 100.0);
        assert!(!v.separated, "overlapping bands, equal means => not separated");
    }

    #[test]
    fn verdict_disjoint_bands_separated_higher() {
        // A band [90,110]; B [150,150] entirely above -> the disjoint-band floor
        // fires (separated) regardless of the Welch arm; b.mean > a.mean.
        let v = noise_verdict(&[90.0, 110.0], &[150.0, 150.0], 100.0);
        assert!(v.separated);
        assert!(v.b.mean > v.a.mean);
    }

    #[test]
    fn verdict_disjoint_bands_separated_lower() {
        let v = noise_verdict(&[90.0, 110.0], &[50.0, 50.0], 100.0);
        assert!(v.separated);
        assert!(v.b.mean < v.a.mean);
    }

    #[test]
    fn verdict_touching_bands_weak_welch_not_separated() {
        // B mean == A.max exactly: bands TOUCH (not strictly disjoint), and the
        // Welch arm at n=2 with this separation does not reach alpha -> not
        // separated. Guards the strict `<` in the disjoint-band test.
        let v = noise_verdict(&[90.0, 110.0], &[110.0, 110.0], 100.0);
        assert!(!v.separated, "touching (non-disjoint) bands + weak Welch");
    }

    #[test]
    fn verdict_welch_separates_overlapping_bands() {
        // The parametric arm in isolation: bands OVERLAP (A [96,104], B
        // [102,110]) so the disjoint floor does NOT fire, but with n=5 and tight
        // per-side variance the Welch t-test clears alpha (t=-3, dof=8,
        // two-sided p ~ 0.017) -> separated via Welch, not the floor.
        let a = [96.0, 98.0, 100.0, 102.0, 104.0];
        let b = [102.0, 104.0, 106.0, 108.0, 110.0];
        let v = noise_verdict(&a, &b, 100.0);
        assert!(
            v.a.max >= v.b.min,
            "bands must overlap so this exercises the Welch arm, not the floor",
        );
        assert!(v.separated, "Welch t-test must separate overlapping-band sides");
    }

    #[test]
    fn verdict_zero_variance_disjoint_no_panic() {
        // Both sides deterministic (zero within-side variance) with different
        // means: se2 == 0 so the Welch arm is SKIPPED (StudentsT::new would
        // otherwise get a NaN dof) — the disjoint-band floor separates. Pins the
        // no-panic guarantee on the common quiesced-counter case.
        let v = noise_verdict(&[100.0, 100.0, 100.0], &[130.0, 130.0, 130.0], 5.0);
        assert_eq!(v.a.var, 0.0);
        assert_eq!(v.b.var, 0.0);
        assert!(v.separated, "disjoint bands separate even when Welch is skipped");
        assert!(!v.insufficient_samples);
    }

    #[test]
    fn verdict_high_spread_is_advisory_not_suppressing() {
        // The signal-inversion regression test: a high-variance side must NOT
        // suppress a clearly-separated move. A's 20% spread (> 5% threshold) sets
        // the advisory high_spread flag, but the disjoint bands keep the verdict
        // separated — the old too_noisy gate would have dropped this.
        let v = noise_verdict(&[90.0, 110.0, 100.0], &[300.0, 300.0, 300.0], 5.0);
        assert!(v.high_spread, "A's 20% spread exceeds the 5% threshold");
        assert!(
            v.separated,
            "high_spread is advisory and must not suppress a separated verdict",
        );
    }

    #[test]
    fn verdict_spread_flag_is_strict_threshold() {
        // Both spreads exactly 1% at a 1% threshold: strict `>` => NOT flagged.
        let v = noise_verdict(&[99.5, 100.5], &[99.5, 100.5], 1.0);
        assert!(approx(v.a.spread_pct, 1.0));
        assert!(!v.high_spread, "spread == threshold is not over it (strict >)");
    }

    /// Demonstration on the real schbench engine (host-side `run_standalone`, no
    /// VM — each run is the short `-r` window, far under the 15s/run budget).
    /// Three windows per config feed the per-run request-p50 into `noise_verdict`,
    /// exercising both arms:
    ///  - A vs a much heavier B (10x matrix ops): a large real change the tool
    ///    must flag `separated` (the hard assertion + the true-positive arm).
    ///  - A vs A (identical config): the false-positive check — REPORTED for
    ///    real-data evaluation, not hard-asserted (the deterministic
    ///    no-false-positive logic is pinned by the unit tests above).
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

        let same = noise_verdict(&a, &b_same, 5.0);
        let diff = noise_verdict(&a, &b_heavy, 5.0);

        eprintln!(
            "\nnoise-adjust schbench demo (request p50 us, N={N}, {RUN_SECS}s/run):\n  \
             A       = {:.0} [{:.0}-{:.0}] spread {:.2}%\n  \
             A vs A  : B {:.0} [{:.0}-{:.0}] spread {:.2}% -> separated={} high_spread={}\n  \
             A vs B' : B {:.0} [{:.0}-{:.0}] spread {:.2}% -> separated={} high_spread={}",
            same.a.mean,
            same.a.min,
            same.a.max,
            same.a.spread_pct,
            same.b.mean,
            same.b.min,
            same.b.max,
            same.b.spread_pct,
            same.separated,
            same.high_spread,
            diff.b.mean,
            diff.b.min,
            diff.b.max,
            diff.b.spread_pct,
            diff.separated,
            diff.high_spread,
        );

        // True positive: 10x matrix ops is a large real increase in request
        // latency — separated (the disjoint-band floor and/or Welch) and clearing
        // A's whole observed band (reliable at 10x, not flaky).
        assert!(
            diff.separated
                && diff.b.mean > a.iter().copied().fold(f64::NEG_INFINITY, f64::max),
            "10x operations must read as a separated increase clearing A's band: A={:?} B={:?}",
            diff.a,
            diff.b,
        );
        // The A-vs-A arm is REPORTED above for real-data evaluation, not hard
        // asserted: at small N two identical configs can still drift enough for
        // the Welch arm to flag them, so a hard `!separated` assertion here would
        // be flaky. The deterministic no-false-positive logic is pinned by the
        // unit tests above (verdict_overlapping_bands_equal_means_not_separated
        // et al.); this test hard-asserts only the reliable true positive.
    }
}
