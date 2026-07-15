//! Seam-level truth table for [`apply_contention_verdict`] — the host-side
//! contention re-evaluation of the guest's wall-latency gate failures.
//!
//! Child of `eval`: reaches the production core via `super::*`. Drives
//! synthetic [`VmResult`](crate::vmm::VmResult) fixtures (via
//! [`crate::vmm::VmResult::test_fixture`]) plus hand-built
//! [`AssertResult`](crate::assert::AssertResult) values so every arm —
//! confirm / demote / saturated / witness-absent / perf-isolation — is
//! exercised without booting a VM.
use super::*;
use crate::assert::{AssertDetail, AssertResult, DetailKind};
use crate::vmm::result::BODY_STAGE_INDEX;
use crate::vmm::{
    BodyContentionWindow, ContentionWitness, HostVcpuSchedstat, PerPhaseSchedstat, VmResult,
};

const MS: u64 = 1_000_000;
const TICK: u64 = 100 * MS;

/// A Body-phase witness with the given on-CPU / run-delay sums (which set
/// `body_dilation()`) and legacy fixed-width contention series (which sets
/// `W`). The Body
/// wall AND covered span are both set to the series' nominal span
/// (`len * TICK`), i.e. FULLY spanning — so the coverage soundness gate is
/// satisfied and these fixtures exercise the confirm/demote/saturated arms as
/// before. Under-coverage is exercised by [`witness_cover`].
fn witness(
    on_cpu_ns: u64,
    run_delay_ns: u64,
    tick_deltas: Vec<u64>,
    saturated: bool,
) -> ContentionWitness {
    let wall = tick_deltas.len() as u64 * TICK;
    witness_cover(on_cpu_ns, run_delay_ns, tick_deltas, saturated, wall, wall)
}

/// Like [`witness`] but with an EXPLICIT Body wall span AND covered span, so a
/// test can build an under-covering series (ticks spanning too little of a
/// long Body wall) that the coverage gate must refuse to FailConfirm on.
fn witness_cover(
    on_cpu_ns: u64,
    run_delay_ns: u64,
    tick_deltas: Vec<u64>,
    saturated: bool,
    body_wall_ns: u64,
    body_covered_ns: u64,
) -> ContentionWitness {
    let mut per_phase = PerPhaseSchedstat::default();
    per_phase.phases[BODY_STAGE_INDEX] = HostVcpuSchedstat {
        total_on_cpu_ns: on_cpu_ns,
        total_run_delay_ns: run_delay_ns,
        sampled_vcpus: 1,
    };
    ContentionWitness {
        per_phase,
        body_window: BodyContentionWindow {
            tick_deltas,
            tick_widths_ns: Vec::new(),
            tick_ns: TICK,
            saturated,
            complete: false,
            schedstat_cap_ns: 0,
            schedstat_cap_complete: false,
            body_wall_ns,
            body_covered_ns,
        },
    }
}

/// A VmResult carrying `w` as its contention witness (all else fixture
/// boilerplate). `None` → no witness.
fn vm_with_witness(w: Option<ContentionWitness>) -> VmResult {
    let mut r = VmResult::test_fixture();
    r.contention_witness = w;
    r
}

/// A one-failure AssertResult whose sole failing detail is a p99 latency
/// gate carrying `(measured, threshold)` evidence.
fn latency_gate_fail(measured_ns: u64, threshold_ns: u64) -> AssertResult {
    let mut r = AssertResult::pass();
    r.record_fail(
        AssertDetail::new(
            DetailKind::Benchmark,
            format!("p99 wake latency {measured_ns}ns exceeds limit {threshold_ns}ns"),
        )
        .with_latency_gate(measured_ns, threshold_ns),
    );
    r
}

// ---- witness present: confirm ----

#[test]
fn quiet_witness_confirms_and_annotates() {
    // Empty contention series → W ≈ 0. Any excess is refutation-proof:
    // FailConfirmed. The failure STAYS and its message gains the
    // contention-checked note.
    let mut cr = latency_gate_fail(101 * MS, 100 * MS);
    let vr = vm_with_witness(Some(witness(100 * MS, 0, vec![0, 0, 0], false)));
    apply_contention_verdict(&mut cr, &vr, false);
    assert!(
        cr.is_fail(),
        "quiet host: excess over W must stay a failure"
    );
    let msg = &cr.failure_details().next().unwrap().message;
    assert!(
        msg.contains("contention-checked"),
        "confirmed failure must carry the contention-checked note: {msg}"
    );
    assert!(cr.info_notes.is_empty(), "confirm must not demote");
}

#[test]
fn complete_schedstat_cap_removes_unrelated_psi_from_w() {
    let mut w = witness(1_000 * MS, 20 * MS, vec![350 * MS, 350 * MS], false);
    w.body_window.tick_widths_ns = vec![500 * MS, 500 * MS];
    w.body_window.complete = true;
    w.body_window.body_wall_ns = 1_000 * MS;
    w.body_window.body_covered_ns = 1_000 * MS;
    w.body_window.schedstat_cap_ns = 20 * MS;
    w.body_window.schedstat_cap_complete = true;

    let mut cr = latency_gate_fail(500 * MS, 1);
    let vr = vm_with_witness(Some(w));
    apply_contention_verdict(&mut cr, &vr, false);

    assert!(
        cr.is_fail(),
        "700ms of runner-cgroup PSI cannot demote a VM whose complete task-specific cap is 20ms"
    );
    let msg = &cr.failure_details().next().unwrap().message;
    assert!(
        msg.contains("W 20.0ms"),
        "the rendered bound must use the cap: {msg}"
    );
}

#[test]
fn gross_excess_over_w_confirms_under_heavy_contention() {
    // A heavily contended Body phase (5 ms burst) cannot explain a 50 ms
    // excess → FailConfirmed regardless of load.
    let mut cr = latency_gate_fail(54 * MS, 4 * MS);
    let vr = vm_with_witness(Some(witness(
        100 * MS,
        100 * MS,
        vec![0, 5 * MS, 0, 0],
        false,
    )));
    apply_contention_verdict(&mut cr, &vr, false);
    assert!(cr.is_fail());
    assert!(
        cr.failure_details()
            .next()
            .unwrap()
            .message
            .contains("contention-checked")
    );
}

// ---- witness present: demote ----

#[test]
fn burst_covers_excess_demotes_to_pass_with_annotation() {
    // measured 8 ms, threshold 4 ms → excess 4 ms; a 5 ms burst tick makes
    // W(8ms) >= 5 ms >= excess → indeterminate == pass. The sole failure is
    // dropped (whole result flips to passing) and a loud annotation rides
    // info_notes.
    let mut cr = latency_gate_fail(8 * MS, 4 * MS);
    let vr = vm_with_witness(Some(witness(
        100 * MS,
        100 * MS,
        vec![0, 5 * MS, 0, 0],
        false,
    )));
    apply_contention_verdict(&mut cr, &vr, false);
    assert!(
        cr.is_pass(),
        "indeterminate == pass: the failure must demote"
    );
    assert_eq!(cr.info_notes.len(), 1);
    let note = &cr.info_notes[0].message;
    assert!(note.starts_with(CONTENTION_INDETERMINATE_PREFIX), "{note}");
    assert!(note.contains("Body dilation D=2.00x"), "{note}");
    assert!(
        !note.contains("saturated"),
        "unsaturated must not note saturation"
    );
}

// ---- witness present: saturated never confirms ----

#[test]
fn saturated_series_never_confirms_even_on_gross_excess() {
    // A 50 ms excess over a tiny W would normally FailConfirm, but a
    // saturated Body series makes W a prefix-only lower bound → treat as
    // indeterminate (demote) and note the saturation.
    let mut cr = latency_gate_fail(54 * MS, 4 * MS);
    let vr = vm_with_witness(Some(witness(100 * MS, 100 * MS, vec![0, 0, 0], true)));
    apply_contention_verdict(&mut cr, &vr, false);
    assert!(cr.is_pass(), "saturated series must never FailConfirm");
    let note = &cr.info_notes[0].message;
    assert!(
        note.contains("saturated"),
        "must note the saturation: {note}"
    );
}

// ---- witness present: under-coverage never confirms (BUG 1 regression) ----

#[test]
fn empty_body_series_never_confirms_regression() {
    // THE arm64 CI regression (verifier-hang-fixes,
    // contention_indeterminate_pass_under_overcommit): 2 vCPUs, no_perf,
    // cpu_budget=1, sibling burner. The SCHED_OTHER monitor never ticked
    // INSIDE Body, so the Body series was EMPTY → peak_window(&[], ..) == 0 ==
    // W. Pre-fix the seam read excess (229.06ms) > W (0) and wrongly CONFIRMED
    // the failure — the one verdict the design forbids — even though the
    // whole-run host dilation was 11.70x. The coverage rule now DEMOTES: an
    // empty series (0 ticks, no Body wall observed) never covered Body, so W
    // is not a trustworthy refutation bound.
    let measured = 229_058_932; // ns — the real arm64 p99 wake latency
    let threshold = 5 * MS; // the 5 ms gate
    let mut cr = latency_gate_fail(measured, threshold);
    // Empty series ⇒ per-phase Body schedstat is default ⇒ body_dilation None
    // (the whole-run D=11.70x rides the render-time annotation, not the seam).
    // No Body tick ⇒ zero wall and zero covered span.
    let vr = vm_with_witness(Some(witness_cover(0, 0, vec![], false, 0, 0)));
    apply_contention_verdict(&mut cr, &vr, false);
    assert!(
        cr.is_pass(),
        "empty Body series (W==0) must demote, never FailConfirm"
    );
    let note = &cr.info_notes[0].message;
    assert!(note.starts_with(CONTENTION_INDETERMINATE_PREFIX), "{note}");
    assert!(
        note.contains("under-covered the Body phase"),
        "must carry the under-coverage note: {note}"
    );
    assert!(note.contains("0 ticks"), "empty series is 0 ticks: {note}");
    // Pin the arm64 excess (229.06 - 5.00 = 224.06 ms) in the annotation.
    assert!(note.contains("excess 224.1ms"), "{note}");
}

#[test]
fn sparse_body_series_below_tick_floor_demotes() {
    // A starved monitor that DID sample Body, but only once, over a ~100 s
    // Body wall (Body dilation D = 1 + 1070/100 = 11.70x). One tick is below
    // the 2-tick floor (zero span), so W built from it cannot refute the
    // 224 ms excess — demote, never confirm.
    let measured = 229_058_932;
    let threshold = 5 * MS;
    let mut cr = latency_gate_fail(measured, threshold);
    let wall = 1_000 * TICK; // ~100 s Body wall
    // Single tick ⇒ covered span 0.
    let vr = vm_with_witness(Some(witness_cover(
        100 * MS,
        1_070 * MS,
        vec![9 * MS],
        false,
        wall,
        0,
    )));
    apply_contention_verdict(&mut cr, &vr, false);
    assert!(cr.is_pass(), "single-tick series must demote");
    let note = &cr.info_notes[0].message;
    assert!(note.contains("under-covered the Body phase"), "{note}");
    assert!(note.contains("1 ticks"), "{note}");
    assert!(
        note.contains("D=11.70x"),
        "carries the Body dilation: {note}"
    );
}

#[test]
fn clustered_ticks_under_coverage_span_demotes() {
    // Several ticks, clustered in a 0.2 s slice of a 10 s Body wall (the
    // dense-then-silent shape): 2% span coverage, below the 50% floor. A
    // 50 ms excess over a zero-W cluster would normally FailConfirm; the span
    // gate demotes it because the ticks never reached across the phase.
    let mut cr = latency_gate_fail(54 * MS, 4 * MS);
    let wall = 100 * TICK; // 10 s Body wall
    let covered = 2 * TICK; // 0.2 s spanned
    let vr = vm_with_witness(Some(witness_cover(
        100 * MS,
        100 * MS,
        vec![0, 0, 0],
        false,
        wall,
        covered,
    )));
    apply_contention_verdict(&mut cr, &vr, false);
    assert!(cr.is_pass(), "under-span coverage must demote");
    let note = &cr.info_notes[0].message;
    assert!(note.contains("under-covered the Body phase"), "{note}");
    assert!(note.contains("3 ticks"), "{note}");
}

#[test]
fn sparse_but_spanning_series_still_confirms() {
    // The fix must NOT over-demote a COARSE-but-spanning series: only 2 ticks,
    // but they reach from Body start to Body end (9 s span over a 10 s wall).
    // The whole phase was watched, so a gross excess still confirms — density
    // is not coverage.
    let mut cr = latency_gate_fail(54 * MS, 4 * MS);
    let wall = 100 * TICK; // 10 s
    let covered = 90 * TICK; // 9 s spanned — 90% coverage
    let vr = vm_with_witness(Some(witness_cover(
        100 * MS,
        100 * MS,
        vec![0, 0],
        false,
        wall,
        covered,
    )));
    apply_contention_verdict(&mut cr, &vr, false);
    assert!(
        cr.is_fail(),
        "coarse-but-spanning gross excess must still confirm"
    );
    assert!(
        cr.failure_details()
            .next()
            .unwrap()
            .message
            .contains("contention-checked")
    );
    assert!(cr.info_notes.is_empty(), "confirm must not demote");
}

// ---- witness absent: unchanged ----

#[test]
fn no_witness_leaves_failure_untouched() {
    let mut cr = latency_gate_fail(54 * MS, 4 * MS);
    let vr = vm_with_witness(None);
    apply_contention_verdict(&mut cr, &vr, false);
    assert!(cr.is_fail(), "no witness → failure stands");
    let msg = &cr.failure_details().next().unwrap().message;
    assert!(
        !msg.contains("contention-checked"),
        "no witness → no re-evaluation note"
    );
    assert!(cr.info_notes.is_empty());
}

#[test]
fn non_latency_failure_passes_through_even_with_witness() {
    // A failure with no latency_gate evidence is never touched.
    let mut cr = AssertResult::pass();
    cr.record_fail(AssertDetail::new(
        DetailKind::NoProgress,
        "tid 7 made no progress",
    ));
    let vr = vm_with_witness(Some(witness(100 * MS, 100 * MS, vec![0, 9 * MS, 0], false)));
    apply_contention_verdict(&mut cr, &vr, false);
    assert!(cr.is_fail());
    assert!(cr.info_notes.is_empty());
}

// ---- perf-mode isolation fault ----

#[test]
fn perf_isolation_fires_only_in_perf_mode_over_threshold() {
    // D = 2.0 (> PERF_ISOLATION_D_MAX 1.5). A clean (passing) result plus a
    // violating witness: perf mode → loud PerfIsolation failure; default
    // mode → nothing.
    let w = || witness(100 * MS, 100 * MS, vec![0, 0], false);

    let mut perf = AssertResult::pass();
    apply_contention_verdict(&mut perf, &vm_with_witness(Some(w())), true);
    assert!(perf.is_fail(), "perf mode + D>ceiling must fail");
    let d = perf
        .failure_details()
        .find(|d| d.kind == DetailKind::PerfIsolation)
        .expect("PerfIsolation detail");
    assert!(
        d.message.starts_with(PERF_ISOLATION_VIOLATED_PREFIX),
        "{}",
        d.message
    );

    let mut deflt = AssertResult::pass();
    apply_contention_verdict(&mut deflt, &vm_with_witness(Some(w())), false);
    assert!(deflt.is_pass(), "default mode: no perf-isolation check");
}

#[test]
fn perf_isolation_silent_under_threshold() {
    // D = 1.2 (< 1.5) → no violation even in perf mode.
    let mut cr = AssertResult::pass();
    let vr = vm_with_witness(Some(witness(100 * MS, 20 * MS, vec![0, 0], false)));
    apply_contention_verdict(&mut cr, &vr, true);
    assert!(
        cr.is_pass(),
        "D under the ceiling is not an isolation fault"
    );
    assert!(
        cr.failure_details()
            .all(|d| d.kind != DetailKind::PerfIsolation)
    );
}

#[test]
fn perf_violation_fires_even_when_latency_gate_demotes() {
    // Perf mode, violating witness, AND a latency gate that the same
    // contention demotes: the demotion drops the gate failure, but the
    // isolation fault is recorded regardless of gate outcome, so the run
    // still fails loudly.
    let mut cr = latency_gate_fail(8 * MS, 4 * MS);
    let vr = vm_with_witness(Some(witness(
        100 * MS,
        100 * MS,
        vec![0, 5 * MS, 0, 0],
        true,
    )));
    apply_contention_verdict(&mut cr, &vr, true);
    assert!(
        cr.is_fail(),
        "perf isolation fault dominates a demoted gate"
    );
    assert!(
        cr.failure_details()
            .any(|d| d.kind == DetailKind::PerfIsolation)
    );
    // The demotion annotation is still present.
    assert!(
        cr.info_notes
            .iter()
            .any(|n| n.message.starts_with(CONTENTION_INDETERMINATE_PREFIX))
    );
}
