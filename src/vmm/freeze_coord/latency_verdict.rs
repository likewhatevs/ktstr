//! Pure verdict logic for the "weather witness" contention model.
//!
//! Guest-wall latency metrics (wakeup p99, request p99, ...) are measured
//! inside the guest and are inflated by whatever host-side scheduling delay
//! the vCPU threads ate DURING the measurement — there is no steal-adjusted
//! guest clock that could subtract it out. So a latency threshold cannot be
//! a plain `measured <= T` gate under host contention: a legitimate cell can
//! blow the threshold purely because a noisy host neighbour starved its
//! vCPUs. The approved design measures contention per MEASUREMENT PHASE (the
//! Body phase) and makes the latency verdict TRI-STATE:
//!
//!   - `measured <= T`                       → [`LatencyVerdict::Pass`]
//!     (always sound — host contention only ever INFLATES a guest-wall
//!     interval, so a value already under the threshold cannot be a
//!     contention artefact; nothing to explain).
//!   - `measured - T > W(measured)`          → [`LatencyVerdict::FailConfirmed`]
//!     (the excess over the threshold is larger than the WORST contamination
//!     any contiguous window of the measurement phase could contribute — no
//!     arrangement of the host contention we witnessed explains it, so it is
//!     a real scheduler-under-test failure).
//!   - otherwise                             →
//!     [`LatencyVerdict::ContentionIndeterminatePass`]
//!     (the excess is within the witnessed contamination bound — the failure
//!     MIGHT be host contention, so per the USER RULING indeterminate is
//!     non-blocking = a pass, but it carries the D / W evidence so the seam
//!     can annotate it).
//!
//! `W(L)` is the peak-window contamination bound: the maximum total host
//! run-delay accrued by the vCPU threads over any contiguous run of monitor
//! ticks spanning at least `L` ns ([`peak_window_delay_ns`]). The p99
//! exemplar is treated as a SINGLE interval of length `measured` — one
//! request cannot have waited longer than its own measured latency, so
//! bounding its contamination by the worst `measured`-long window is
//! conservative (a real p99 sample spans one request, never the whole
//! phase). Every rounding in this module is biased toward MORE indeterminate
//! / never a wrong `FailConfirmed`.
//!
//! CONSUMED BY THE EVAL SEAM: this module is the PURE verdict core; the seam
//! that calls it — `crate::test_support::eval::apply_contention_verdict` —
//! re-runs the tri-state over the guest's wall-latency gate failures with
//! the host-side Body-phase witness, demoting the indeterminate ones and
//! confirming the refutation-proof ones. `perf_isolation_violated` /
//! `PERF_ISOLATION_D_MAX` back the seam's perf-mode isolation-fault check.

/// Peak host-contention contamination (ns) over any contiguous run of
/// monitor ticks spanning at least `window_ns` — the `W(L)` bound.
///
/// `deltas[i]` is the per-tick TOTAL host run-delay delta (Σ over the vCPU
/// threads) the monitor accrued during tick `i` of the measurement phase;
/// `tick_ns` is the nominal monitor tick period (100 ms in production).
/// Returns the maximum sum of `deltas` over any sliding window of
/// `ceil(window_ns / tick_ns) + 1` consecutive ticks — the most host
/// run-delay that could have piled into any single interval of length
/// `window_ns`.
///
/// CONSERVATIVE PARTIAL-TICK ROUNDING: the window is rounded UP to whole
/// ticks (`div_ceil`) PLUS ONE for boundary straddle — an interval of
/// length `L` placed arbitrarily against the tick grid intersects up to
/// `ceil(L/tick) + 1` consecutive tick spans (e.g. a 150 ms interval
/// starting 99 ms into a tick touches three 100 ms ticks), and delay
/// attributed to a partially-overlapped boundary tick may all have landed
/// inside the interval. The tick-quantised window therefore always
/// OVER-covers the real `window_ns` — the returned bound can only be too
/// LARGE, never too small. A too-large `W` can only push a verdict toward
/// indeterminate (non-blocking), never toward a wrong `FailConfirmed`.
///
/// Degenerate inputs fold conservatively to "the whole phase's
/// contamination": an empty series is 0; a window that rounds to `>=` the
/// series length (window longer than the whole measured phase) returns the
/// SUM of every tick (any interval of that length contains at most the whole
/// phase, so the phase total is its own bound); a `tick_ns == 0` (which would
/// make `div_ceil` divide by zero) also folds to the phase total. Every add
/// saturates so a pathological all-`u64::MAX` series cannot wrap.
pub(crate) fn peak_window_delay_ns(deltas: &[u64], tick_ns: u64, window_ns: u64) -> u64 {
    if deltas.is_empty() {
        return 0;
    }
    // Whole phase total (saturating) — the fallback bound for every
    // degenerate case below and the base for the slide.
    let phase_total = || deltas.iter().fold(0u64, |a, &d| a.saturating_add(d));
    if tick_ns == 0 {
        return phase_total();
    }
    // Round the window UP to whole ticks, PLUS ONE boundary-straddle tick
    // (see the doc comment): an interval never aligns with the tick grid,
    // so it can intersect one more tick span than its rounded-up length.
    // Also guarantees a sub-tick window still spans >= 2 ticks, never zero.
    let window_ticks = (window_ns.div_ceil(tick_ns) as usize).saturating_add(1);
    if window_ticks >= deltas.len() {
        // The window is at least the whole phase — its worst contamination is
        // the entire phase's run-delay.
        return phase_total();
    }
    // Sliding window of exactly `window_ticks` consecutive ticks. Seed with
    // the first window, then slide by adding the incoming and subtracting the
    // outgoing tick. `saturating_sub` on the outgoing keeps the running sum
    // well-defined even if a saturating add earlier clamped it.
    let mut running = deltas[..window_ticks]
        .iter()
        .fold(0u64, |a, &d| a.saturating_add(d));
    let mut best = running;
    for i in window_ticks..deltas.len() {
        running = running.saturating_add(deltas[i]);
        running = running.saturating_sub(deltas[i - window_ticks]);
        best = best.max(running);
    }
    best
}

/// Tri-state outcome of a contention-aware latency threshold check.
///
/// Each variant carries the evidence the seam needs to render the verdict:
/// the measured value, its threshold, the excess over threshold, the `W`
/// bound the decision used, and (where relevant) the phase dilation `D` for
/// annotation. `Pass` carries no evidence — there is nothing to explain.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum LatencyVerdict {
    /// `measured <= threshold`. Always sound: contention only inflates a
    /// guest-wall interval, so a value already under threshold cannot be a
    /// contention artefact.
    Pass,
    /// `measured - threshold > W(measured)`: the excess exceeds the worst
    /// contamination any `measured`-long window of the measurement phase
    /// could contribute, so no witnessed host contention explains it — a
    /// real failure at any load.
    FailConfirmed {
        /// The measured guest-wall latency (ns).
        measured_ns: u64,
        /// The threshold that was exceeded (ns).
        threshold_ns: u64,
        /// `measured_ns - threshold_ns` (ns).
        excess_ns: u64,
        /// `W(measured_ns)` — the peak-window contamination bound the
        /// decision compared against (ns).
        window_bound_ns: u64,
        /// Body-phase host dilation `D` for annotation (`None` when no
        /// on-CPU time was sampled — see
        /// [`crate::vmm::result::HostVcpuSchedstat::dilation`]).
        phase_d: Option<f64>,
    },
    /// `threshold < measured` yet `measured - threshold <= W(measured)`: the
    /// excess is within the witnessed contamination bound, so the failure
    /// MIGHT be host contention. USER RULING: indeterminate == pass
    /// (non-blocking); the evidence rides along for the seam to annotate.
    ContentionIndeterminatePass {
        /// The measured guest-wall latency (ns).
        measured_ns: u64,
        /// The threshold that was exceeded (ns).
        threshold_ns: u64,
        /// `measured_ns - threshold_ns` (ns).
        excess_ns: u64,
        /// `W(measured_ns)` — the peak-window contamination bound that
        /// covers the excess (ns).
        window_bound_ns: u64,
        /// Body-phase host dilation `D` for annotation.
        phase_d: Option<f64>,
    },
}

impl LatencyVerdict {
    /// Whether this verdict blocks (fails) the run. Only
    /// [`LatencyVerdict::FailConfirmed`] blocks — `Pass` and
    /// `ContentionIndeterminatePass` are both non-blocking per the user
    /// ruling.
    pub(crate) fn is_blocking(self) -> bool {
        matches!(self, LatencyVerdict::FailConfirmed { .. })
    }
}

/// Contention-aware tri-state verdict for one latency threshold.
///
/// `measured_ns` is the guest-wall latency exemplar (e.g. the wakeup p99),
/// `threshold_ns` its gate, `phase_d` the Body-phase dilation for annotation
/// only (it does NOT enter the decision — the operative bound is `W`, which
/// is derived from the run-delay series directly; `D` is the summary the
/// seam renders alongside). `peak_window_fn` maps a window length `L` (ns)
/// to `W(L)` — bind it as `|l| peak_window_delay_ns(&deltas, tick_ns, l)`.
///
/// The exemplar is scored as ONE interval of length `measured_ns`: `W` is
/// evaluated at `L = measured_ns`, because a single request's contamination
/// is bounded by a window as long as the request's own measured latency (it
/// cannot have waited longer than it waited). This is the conservative
/// percentile treatment — a real p99 sample spans one request, never the
/// whole phase.
///
/// Boundary: `excess > W` confirms, `excess <= W` is indeterminate — so when
/// `W == 0` (a quiet host, `D ≈ 1`, empty/zero run-delay series) the
/// indeterminate band is empty and ANY excess confirms, exactly as a plain
/// threshold would behave with no contention to blame.
pub(crate) fn latency_verdict(
    measured_ns: u64,
    threshold_ns: u64,
    phase_d: Option<f64>,
    peak_window_fn: impl Fn(u64) -> u64,
) -> LatencyVerdict {
    if measured_ns <= threshold_ns {
        return LatencyVerdict::Pass;
    }
    let excess_ns = measured_ns - threshold_ns;
    let window_bound_ns = peak_window_fn(measured_ns);
    if excess_ns > window_bound_ns {
        LatencyVerdict::FailConfirmed {
            measured_ns,
            threshold_ns,
            excess_ns,
            window_bound_ns,
            phase_d,
        }
    } else {
        LatencyVerdict::ContentionIndeterminatePass {
            measured_ns,
            threshold_ns,
            excess_ns,
            window_bound_ns,
            phase_d,
        }
    }
}

/// Body-phase dilation `D` above which a PERFORMANCE-MODE cell is treated as
/// having lost host isolation.
///
/// DERIVATION (from `src/vmm/freeze_coord/dilation_validation.md`, the
/// captured perf-mode measured-quiet numbers). Perf mode 1:1-pins every vCPU
/// thread to its own host CPU, so a well-isolated perf cell on a quiet host
/// reads a near-1.0 `D`. The doc's perf-mode measurements on the quiet
/// 64-CPU host:
///   - `performance_mode_schbench_steady`: `D` avg 1.0771 (1.0764 / 1.0777 /
///     1.0773).
///   - `performance_mode_perphase_metrics_across_detach`: `D` avg 1.2041
///     (single reps up to 1.2064; re-runs 1.2031 / 1.2064 / 1.2056).
///
/// The worst measured-quiet perf-mode `D` is therefore ≈ 1.206 (the detach
/// demo, whose mid-run scheduler swap is the noisiest perf shape the suite
/// has). 1.5 sits ~24% above that worst measured-quiet value — clear of the
/// run-to-run spread — and `D > 1.5` on a 1:1-pinned perf cell means the
/// pinned vCPUs waited >50% on top of their run time, which for a dedicated
/// host CPU is unambiguous external contention (a noisy neighbour on the
/// pinned CPU), not measurement jitter.
///
/// Deliberately conservative (a HIGH bar): this predicate is a diagnostic
/// SIGNAL only. Per the task it is NOT wired into any pass/fail decision yet
/// — the seam integration is a later pass — so the const errs toward "only
/// flag a gross, unambiguous isolation break".
pub(crate) const PERF_ISOLATION_D_MAX: f64 = 1.5;

/// Whether a perf-mode cell's Body-phase dilation indicates lost host
/// isolation: `true` iff `phase_d > `[`PERF_ISOLATION_D_MAX`]. `None`
/// (no on-CPU time sampled — schedstats off, or no vCPU ran) is NOT a
/// violation: absence of evidence is not evidence of contention. Pure
/// predicate, no pass/fail wiring (a later seam pass consumes it).
pub(crate) fn perf_isolation_violated(phase_d: Option<f64>) -> bool {
    phase_d.is_some_and(|d| d > PERF_ISOLATION_D_MAX)
}

#[cfg(test)]
mod tests {
    use super::*;

    const MS: u64 = 1_000_000;
    const TICK: u64 = 100 * MS; // 100 ms production monitor tick.

    // ---- peak_window_delay_ns: window math, exhaustive edges ----

    #[test]
    fn peak_window_empty_series_is_zero() {
        assert_eq!(peak_window_delay_ns(&[], TICK, 5 * TICK), 0);
    }

    #[test]
    fn peak_window_single_tick() {
        // One tick, sub-tick window rounds up to 1 tick → that tick's delta.
        assert_eq!(peak_window_delay_ns(&[42], TICK, 1), 42);
        assert_eq!(peak_window_delay_ns(&[42], TICK, TICK), 42);
        // A multi-tick window over a one-tick phase → the whole phase.
        assert_eq!(peak_window_delay_ns(&[42], TICK, 10 * TICK), 42);
    }

    #[test]
    fn peak_window_longer_than_phase_returns_total() {
        // window (5 ticks) >= phase (3 ticks) → sum of all.
        let d = [10, 20, 30];
        assert_eq!(peak_window_delay_ns(&d, TICK, 5 * TICK), 60);
        // window exactly == phase length → still the full sum (one window).
        assert_eq!(peak_window_delay_ns(&d, TICK, 3 * TICK), 60);
    }

    #[test]
    fn peak_window_all_zero_is_zero() {
        assert_eq!(peak_window_delay_ns(&[0, 0, 0, 0], TICK, 2 * TICK), 0);
    }

    #[test]
    fn peak_window_burst_at_edges_and_middle() {
        // 2-tick nominal window → 3 ticks after the straddle tick. Bursts at
        // the leading, middle, and trailing edges must all be found by the
        // slide (zero padding keeps the expected sums burst-only here).
        let lead = [100, 100, 0, 0, 0];
        assert_eq!(peak_window_delay_ns(&lead, TICK, 2 * TICK), 200);
        let mid = [0, 0, 70, 80, 0];
        assert_eq!(peak_window_delay_ns(&mid, TICK, 2 * TICK), 150);
        let trail = [0, 0, 0, 90, 90];
        assert_eq!(peak_window_delay_ns(&trail, TICK, 2 * TICK), 180);
    }

    #[test]
    fn peak_window_rounds_up_conservatively() {
        // 250 ms window over 100 ms ticks → ceil(2.5) = 3 ticks + 1 straddle
        // = 4. The 4-tick sum (over-covering the 250 ms at any grid offset)
        // must be used, not the aligned 3-tick sum.
        let d = [10, 20, 30, 40, 50];
        // Best 4-tick window is [20,30,40,50] = 140; the (unsound) aligned
        // 3-tick best would be 120.
        assert_eq!(peak_window_delay_ns(&d, TICK, 250 * MS), 140);
    }

    #[test]
    fn peak_window_boundary_straddle_is_covered() {
        // THE straddle counterexample: a 200 ms interval placed at grid
        // offset 50 ms (spanning [50 ms, 250 ms)) touches ticks 0, 1 AND 2 —
        // and can absorb tick 0's delay (if it landed in [50,100)) plus
        // tick 2's (if it landed in [200,250)). An aligned ceil(L/tick)=2
        // window reads max(60,60)=60 and would under-bound the possible
        // 120 ms contamination, risking a wrong FailConfirmed; the +1
        // straddle tick covers it.
        let d = [60, 0, 60];
        assert_eq!(peak_window_delay_ns(&d, TICK, 2 * TICK), 120);
    }

    #[test]
    fn peak_window_tick_ns_zero_folds_to_total() {
        // Degenerate tick_ns must not divide-by-zero; folds to phase total.
        assert_eq!(peak_window_delay_ns(&[1, 2, 3], 0, 5 * MS), 6);
    }

    #[test]
    fn peak_window_saturates_not_wraps() {
        let d = [u64::MAX, u64::MAX];
        assert_eq!(peak_window_delay_ns(&d, TICK, 2 * TICK), u64::MAX);
    }

    // ---- latency_verdict: tri-state truth table ----

    #[test]
    fn verdict_under_threshold_is_pass() {
        // measured <= T → Pass regardless of any W (contention only inflates).
        let v = latency_verdict(50, 100, Some(3.0), |_| 1_000_000);
        assert_eq!(v, LatencyVerdict::Pass);
        assert!(!v.is_blocking());
        // Exactly at threshold is still a pass.
        assert_eq!(latency_verdict(100, 100, None, |_| 0), LatencyVerdict::Pass);
    }

    #[test]
    fn verdict_quiet_host_any_excess_confirms() {
        // D ≈ 1, empty run-delay series → W ≈ 0. The indeterminate band is
        // empty: any excess over threshold is FailConfirmed.
        let deltas: [u64; 0] = [];
        let v = latency_verdict(101, 100, Some(1.0), |l| {
            peak_window_delay_ns(&deltas, TICK, l)
        });
        match v {
            LatencyVerdict::FailConfirmed {
                excess_ns,
                window_bound_ns,
                ..
            } => {
                assert_eq!(excess_ns, 1);
                assert_eq!(window_bound_ns, 0);
            }
            other => panic!("expected FailConfirmed, got {other:?}"),
        }
        assert!(v.is_blocking());
    }

    #[test]
    fn verdict_heavy_burst_excess_under_w_is_indeterminate() {
        // A contended Body phase: one tick contributed 5 ms of host run-delay.
        // measured 8 ms, threshold 4 ms → excess 4 ms. W(8ms) covers the 5 ms
        // burst (window rounds up to >= 1 tick, catching the burst tick) →
        // excess (4 ms) <= W (5 ms) → indeterminate-pass, non-blocking.
        let deltas = [0, 5 * MS, 0, 0];
        let v = latency_verdict(8 * MS, 4 * MS, Some(1.9), |l| {
            peak_window_delay_ns(&deltas, TICK, l)
        });
        match v {
            LatencyVerdict::ContentionIndeterminatePass {
                excess_ns,
                window_bound_ns,
                phase_d,
                ..
            } => {
                assert_eq!(excess_ns, 4 * MS);
                assert_eq!(window_bound_ns, 5 * MS);
                assert_eq!(phase_d, Some(1.9));
            }
            other => panic!("expected indeterminate, got {other:?}"),
        }
        assert!(!v.is_blocking());
    }

    #[test]
    fn verdict_gross_violation_over_w_confirms_at_any_load() {
        // Even a heavily contended phase (5 ms burst) cannot explain a 50 ms
        // excess over a 4 ms threshold → FailConfirmed regardless of load.
        let deltas = [0, 5 * MS, 0, 0];
        let v = latency_verdict(54 * MS, 4 * MS, Some(1.9), |l| {
            peak_window_delay_ns(&deltas, TICK, l)
        });
        assert!(matches!(v, LatencyVerdict::FailConfirmed { .. }));
        assert!(v.is_blocking());
    }

    #[test]
    fn verdict_boundary_excess_equals_w_is_indeterminate() {
        // excess == W is INSIDE the band (`>` confirms, so `==` is
        // indeterminate) — the conservative side.
        let deltas = [3 * MS];
        let v = latency_verdict(7 * MS, 4 * MS, None, |l| {
            peak_window_delay_ns(&deltas, TICK, l)
        });
        assert!(matches!(
            v,
            LatencyVerdict::ContentionIndeterminatePass { .. }
        ));
    }

    // ---- perf_isolation_violated ----

    #[test]
    fn perf_isolation_threshold_and_none() {
        // None (no on-CPU sampled) is never a violation.
        assert!(!perf_isolation_violated(None));
        // Measured-quiet perf-mode D values (from dilation_validation.md) are
        // NOT violations — well under the bar.
        assert!(!perf_isolation_violated(Some(1.0771))); // steady avg
        assert!(!perf_isolation_violated(Some(1.2064))); // worst detach rep
        // At/just under the bar is not a violation; strictly above is.
        assert!(!perf_isolation_violated(Some(PERF_ISOLATION_D_MAX)));
        assert!(perf_isolation_violated(Some(1.6)));
        assert!(perf_isolation_violated(Some(2.0)));
    }
}
