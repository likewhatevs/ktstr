//! Host-side tests for the scenario export record.
//!
//! These pin the record's *shape*. Whether that shape is the one
//! `scxsim-workload-ir` accepts is not knowable from inside ktstr and is not
//! asserted here — the consumer deserialises with the real `SourceScenario`
//! type, so a mismatch fails loudly there rather than passing quietly here.

use std::time::Duration;

use super::export_scenario;
use crate::scenario::ScenarioDef;
use crate::scenario::ops::{CgroupDef, CpusetSpec, HoldSpec, Setup, Step};
use crate::test_support::Topology;
use crate::workload::{WorkSpec, WorkType};

fn topo() -> Topology {
    Topology {
        llcs: 1,
        cores_per_llc: 2,
        threads_per_core: 1,
        numa_nodes: 1,
        nodes: None,
        distances: None,
        llc_cores: None,
    }
}

/// The `sched_basic_proportional` shape, end to end. This is the record the
/// simulator side actually consumes, so its exact content is the contract.
#[test]
fn exports_the_basic_proportional_shape() {
    let def = ScenarioDef::with_defs(vec![CgroupDef::named("cg_0"), CgroupDef::named("cg_1")]);
    let out = export_scenario(
        "sched_basic_proportional",
        &def,
        &topo(),
        Duration::from_secs(12),
        2,
    );
    assert!(out.is_complete(), "unexpected gaps: {:#?}", out.gaps);

    let r = &out.record;
    assert_eq!(r["name"], "sched_basic_proportional");
    assert_eq!(r["topology"]["llcs"], 1);
    assert_eq!(r["topology"]["cores"], 2);
    assert_eq!(r["topology"]["threads"], 1);
    assert_eq!(r["duration"], 12_000_000_000u64);
    assert_eq!(r["default_workers_per_cgroup"], 2);

    let steps = r["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["hold"]["frac"], 1.0);

    let setup = steps[0]["setup"].as_array().expect("setup array");
    assert_eq!(setup.len(), 2);
    assert_eq!(setup[0]["name"], "cg_0");
    assert_eq!(setup[1]["name"], "cg_1");

    // An empty `works` means one default WorkSpec, and the record says so
    // explicitly rather than leaving the consumer to reinterpret an empty list.
    let works = setup[0]["works"].as_array().expect("works array");
    assert_eq!(works.len(), 1);
    assert_eq!(works[0]["work_type"], "spin_wait");
    assert!(
        works[0]["workers"].is_null(),
        "num_workers stays symbolic so each backend binds its own default; \
         got {}",
        works[0]["workers"],
    );
}

/// Disjoint cpusets survive as the symbolic spec, not as a resolved CPU list.
/// Resolving here would bake this host's topology into a record whose whole
/// purpose is to be portable between backends.
#[test]
fn cpuset_stays_symbolic() {
    let def = ScenarioDef::with_defs(vec![
        CgroupDef::named("cg_0").cpuset(CpusetSpec::Disjoint { index: 0, of: 2 }),
    ]);
    let out = export_scenario("t", &def, &topo(), Duration::from_secs(1), 2);
    assert!(out.is_complete(), "gaps: {:#?}", out.gaps);
    let cs = &out.record["steps"][0]["setup"][0]["cpuset"];
    assert_eq!(cs["disjoint"]["index"], 0);
    assert_eq!(cs["disjoint"]["of"], 2);
}

/// Per-step holds are preserved individually — the `sched_dynamic_add` shape,
/// where collapsing two half-duration steps into one would change what the
/// scheduler sees.
#[test]
fn per_step_holds_are_preserved() {
    let def = ScenarioDef::new(vec![
        Step::with_defs(vec![CgroupDef::named("cg_0")], HoldSpec::frac(0.5)),
        Step::with_defs(vec![CgroupDef::named("cg_1")], HoldSpec::frac(0.5)),
    ]);
    let out = export_scenario("t", &def, &topo(), Duration::from_secs(10), 2);
    let steps = out.record["steps"].as_array().unwrap();
    assert_eq!(steps.len(), 2);
    assert_eq!(steps[0]["hold"]["frac"], 0.5);
    assert_eq!(steps[1]["hold"]["frac"], 0.5);
}

/// A fixed hold exports as nanoseconds.
#[test]
fn fixed_hold_exports_nanoseconds() {
    let def = ScenarioDef::new(vec![Step::with_defs(
        vec![CgroupDef::named("cg_0")],
        HoldSpec::fixed(Duration::from_millis(250)),
    )]);
    let out = export_scenario("t", &def, &topo(), Duration::from_secs(1), 1);
    assert_eq!(out.record["steps"][0]["hold"]["fixed"], 250_000_000u64);
}

/// An unmapped work type is a RECORDED GAP, not a substitution. This is the
/// test that would fail if someone made the exporter "more useful" by falling
/// back to spin_wait — which would hand the simulator a different workload
/// than the test declares while looking like a faithful reading of it.
#[test]
fn unmapped_work_type_is_a_gap_not_a_substitution() {
    let def = ScenarioDef::with_defs(vec![CgroupDef::named("cg_0").work(
        WorkSpec::default().work_type(WorkType::FutexPingPong {
            spin_iters: 20_000_000,
        }),
    )]);
    let out = export_scenario("t", &def, &topo(), Duration::from_secs(1), 2);
    assert!(!out.is_complete(), "an unmapped work type must be reported");
    assert!(
        out.gaps
            .iter()
            .any(|g| g.construct.contains("FutexPingPong")),
        "the gap must name the construct: {:#?}",
        out.gaps,
    );
    let works = out.record["steps"][0]["setup"][0]["works"]
        .as_array()
        .unwrap();
    assert!(
        works.is_empty(),
        "the unmapped work must be ABSENT, not replaced with a plausible \
         stand-in; got {works:?}",
    );
}

/// A `Setup::Factory` step cannot be read without a guest, so it is a gap and
/// its cgroups are absent rather than guessed at.
#[test]
fn factory_setup_is_a_gap() {
    fn per_ctx(_ctx: &crate::scenario::Ctx) -> Vec<CgroupDef> {
        vec![CgroupDef::named("cg_0")]
    }
    let mut s = Step::hold(HoldSpec::FULL);
    s.setup = Setup::with_factory(per_ctx);
    let out = export_scenario(
        "t",
        &ScenarioDef::new(vec![s]),
        &topo(),
        Duration::from_secs(1),
        2,
    );
    assert!(!out.is_complete());
    assert!(out.gaps.iter().any(|g| g.construct == "Setup::Factory"));
    assert!(
        out.record["steps"][0]["setup"]
            .as_array()
            .unwrap()
            .is_empty()
    );
}

/// The oracle is not smuggled through the workload record. A scenario with an
/// `Assert` override exports its workload and reports the override as a gap,
/// because a shared oracle has to be stated against both backends' outputs.
#[test]
fn assert_override_is_reported_not_encoded() {
    let checks = crate::assert::Assert::default_checks().min_iteration_rate(5000.0);
    let def = ScenarioDef::with_defs(vec![CgroupDef::named("cg_0")]).set_checks(checks);
    let out = export_scenario("t", &def, &topo(), Duration::from_secs(1), 2);
    assert!(out.gaps.iter().any(|g| g.construct == "Assert override"));
}

/// `nice` reaches the record.
///
/// It did not, for as long as this exporter has existed: the field was
/// hardcoded null while `WorkSpec` carried a value and `SourceWorkSpec` had a
/// field waiting for it. Nothing caught it because every scenario ported so far
/// leaves nice at the default, so the null was always the right answer by
/// accident.
#[test]
fn nice_is_carried_rather_than_nulled() {
    let def = ScenarioDef::with_defs(vec![
        CgroupDef::named("cg_0").work(WorkSpec::default().work_type(WorkType::SpinWait).nice(-5))
    ]);
    let out = export_scenario("nice", &def, &topo(), Duration::from_secs(1), 1);
    let works = &out.record["steps"][0]["setup"][0]["works"][0];
    assert_eq!(
        works["nice"], -5,
        "nice must reach the record; got {:#?}",
        out.record["steps"][0]["setup"][0]
    );
    assert!(out.is_complete(), "unexpected gaps: {:#?}", out.gaps);
}

/// A non-default scheduling policy is REPORTED as a gap, not dropped in
/// silence.
///
/// Unlike `nice`, this one cannot be fixed by carrying it: `SourceWorkSpec` has
/// no policy field, so the record genuinely cannot distinguish SCHED_NORMAL
/// from BATCH/IDLE/FIFO. The requirement is therefore that the loss is
/// VISIBLE. Without this, `custom_sched_mixed` — whose entire point is a
/// Normal/Batch/Idle/FIFO mix — would export as uniformly SCHED_NORMAL and
/// look complete.
#[test]
fn a_non_default_sched_policy_is_a_recorded_gap() {
    use crate::workload::SchedPolicy;
    let def = ScenarioDef::with_defs(vec![
        CgroupDef::named("cg_0")
            .work(WorkSpec::default().work_type(WorkType::SpinWait).sched_policy(SchedPolicy::Batch))
    ]);
    let out = export_scenario("policy", &def, &topo(), Duration::from_secs(1), 1);
    assert!(
        !out.is_complete(),
        "a Batch policy must surface as a gap, not vanish"
    );
    assert!(
        out.gaps.iter().any(|g| g.construct.contains("sched_policy")),
        "the gap must name sched_policy; got {:#?}",
        out.gaps
    );
}

/// The fieldless work types transfer verbatim.
///
/// `IoSyncWrite` is the one that matters today — it is what
/// `custom_cgroup_io_compute_imbalance` needs, and it was previously refused by
/// this exporter even though the IR has lowered it all along.
#[test]
fn fieldless_work_types_are_mapped() {
    for (wt, expected) in [
        (WorkType::IoSyncWrite, "io_sync_write"),
        (WorkType::IoRandRead, "io_rand_read"),
        (WorkType::IoConvoy, "io_convoy"),
        (WorkType::ForkExit, "fork_exit"),
        (WorkType::NiceSweep, "nice_sweep"),
        (WorkType::SmtSiblingSpin, "smt_sibling_spin"),
    ] {
        let def =
            ScenarioDef::with_defs(vec![CgroupDef::named("cg_0").work(WorkSpec::default().work_type(wt.clone()))]);
        let out = export_scenario("wt", &def, &topo(), Duration::from_secs(1), 1);
        assert_eq!(
            out.record["steps"][0]["setup"][0]["works"][0]["work_type"], expected,
            "WorkType::{wt:?} must export as {expected:?}"
        );
        assert!(out.is_complete(), "unexpected gaps for {wt:?}: {:#?}", out.gaps);
    }
}
