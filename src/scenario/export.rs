//! Export a [`ScenarioDef`] as a backend-neutral workload record.
//!
//! [`ScenarioDef`] made a test's workload a value; this makes it a value that
//! can leave the process. The record is the input to `scxsim-workload-ir`'s
//! `SourceScenario`, which lowers to a restricted IR and is ingested by the
//! simulator — so the same test definition can drive the VM backend and the
//! simulator without being written twice.
//!
//! # Why JSON and not a Cargo dependency
//!
//! ktstr and scx-sim live in different repositories, and the dependency runs
//! ktstr -> scx-sim, never the reverse. A path dependency between two checkouts
//! would bake a host-specific path into a committed manifest. JSON is the seam
//! two repos can share without either owning the other's build.
//!
//! The schema is not maintained by agreement. It is `SourceScenario`'s serde
//! representation, and the consumer deserialises with that exact type — so a
//! drift here surfaces as a hard deserialisation error naming the field, not as
//! a workload that silently means something else. That is the point: this file
//! is allowed to be wrong, but it is not allowed to be wrong quietly.
//!
//! # What is deliberately not exported
//!
//! Anything the record cannot represent faithfully is omitted with a recorded
//! reason rather than approximated here. Approximation is the *lowering's* job,
//! where it is classified and reported ([`FidelityReport`] on the IR side); an
//! approximation invented at export time would be invisible to that machinery
//! and would reach the simulator disguised as an exact reading of the test.

use serde::Serialize;

use crate::scenario::ScenarioDef;
use crate::scenario::ops::{CgroupDef, CpusetSpec, HoldSpec, Setup, Step};
use crate::test_support::Topology;
use crate::workload::{WorkSpec, WorkType};

/// A reason some part of a scenario could not be exported faithfully.
///
/// Carried out of [`export_scenario`] rather than logged, for the same reason
/// the IR carries its fidelity report: a caller deciding whether a simulator
/// run answers its question needs to see what the record does not say.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ExportGap {
    /// Where in the scenario the gap is, e.g. `step[1].setup[0] "cg_0"`.
    pub where_: String,
    /// The construct that could not be exported.
    pub construct: String,
    /// Why, in terms a reader can act on.
    pub reason: String,
}

/// The exported record plus everything it could not carry.
#[derive(Debug, Clone)]
pub struct Export {
    /// `SourceScenario`-shaped JSON.
    pub record: serde_json::Value,
    /// Constructs omitted from `record`, each with a reason.
    pub gaps: Vec<ExportGap>,
}

impl Export {
    /// Whether the record carries the scenario in full.
    #[must_use]
    pub fn is_complete(&self) -> bool {
        self.gaps.is_empty()
    }
}

/// Nanoseconds, matching `scxsim_workload_ir::units::DurationNs`'s
/// transparent-`u64` serde form.
fn ns(d: std::time::Duration) -> serde_json::Value {
    serde_json::json!(u64::try_from(d.as_nanos()).unwrap_or(u64::MAX))
}

/// ktstr's `CpusetSpec` -> `SourceCpuset`'s snake_case externally-tagged form.
///
/// Returns `None` for variants the record has no counterpart for; the caller
/// records a gap. Resolving one here (say, by flattening a topology-relative
/// cpuset into an explicit CPU list) would bind the scenario to this host's
/// topology, which is exactly the symbolic-ness the DSL exists to keep.
fn cpuset(spec: &CpusetSpec) -> Option<serde_json::Value> {
    use serde_json::json;
    Some(match spec {
        CpusetSpec::Llc(i) => json!({ "llc": i }),
        CpusetSpec::Numa(i) => json!({ "numa": i }),
        CpusetSpec::Disjoint { index, of } => json!({ "disjoint": { "index": index, "of": of } }),
        CpusetSpec::Range {
            start_frac,
            end_frac,
        } => {
            json!({ "range": { "start_frac": start_frac, "end_frac": end_frac } })
        }
        CpusetSpec::Overlap { index, of, frac } => {
            json!({ "overlap": { "index": index, "of": of, "frac": frac } })
        }
        _ => return None,
    })
}

/// ktstr's `WorkType` -> `SourceWorkType`.
///
/// Only the variants whose scheduler-visible meaning transfers verbatim are
/// mapped. Everything else is a gap: the lowering is the layer that decides how
/// a work type collapses onto run/sleep/yield and records the discarded
/// dimension, and it cannot do that for a construct this function has already
/// silently rewritten.
///
/// # Why this covers the FIELDLESS variants and no others
///
/// `SourceWorkType` mirrors this enum — same 45 names — so it is tempting to
/// map all 45 mechanically. Do not. The mirror is not exact: several variants
/// carry ktstr fields the IR has no home for, among them
/// `PriorityInversion::pi_mode`, `ProducerConsumerImbalance::queue_depth_target`
/// and `Custom::{run, cfg}`. Mapping those by name would drop a tuning knob
/// silently, which is the one thing this file exists not to do.
///
/// The nine fieldless variants have nothing to drop, so they transfer
/// verbatim and are safe. Two field-carrying variants are also mapped --
/// `FutexPingPong` and `CrossAffinityChurn` -- each checked field-by-field
/// against the IR first; both carry only `spin_iters: u64` on both sides. The
/// remaining 34 want a per-variant mapping that translates the fields it can
/// and records an [`ExportGap`] for the fields it cannot — worth doing, but it
/// is a per-variant judgement each time, not a loop.
fn work_type(wt: &WorkType) -> Option<serde_json::Value> {
    use serde_json::json;
    Some(match wt {
        WorkType::SpinWait => json!("spin_wait"),
        WorkType::YieldHeavy => json!("yield_heavy"),
        WorkType::Mixed => json!("mixed"),
        // FIELDLESS variants, verbatim and only verbatim: each names the same
        // fieldless `SourceWorkType`, so nothing is rewritten here. What the
        // simulator cannot model about them -- for `IoSyncWrite`, the block
        // device, queue depth and byte counts -- is discarded one layer down by
        // the lowering, which records the cause when it does.
        WorkType::IoSyncWrite => json!("io_sync_write"),
        WorkType::IoRandRead => json!("io_rand_read"),
        WorkType::IoConvoy => json!("io_convoy"),
        WorkType::ForkExit => json!("fork_exit"),
        WorkType::NiceSweep => json!("nice_sweep"),
        WorkType::SmtSiblingSpin => json!("smt_sibling_spin"),

        // FIELD-CARRYING variants, mapped one at a time with their fields
        // checked against the IR rather than by name. Both of these carry a
        // single `spin_iters: u64` on BOTH sides, so nothing is dropped and no
        // gap is recorded -- which is the bar a field-carrying mapping has to
        // meet before it belongs here. A variant whose ktstr fields have no IR
        // home must record an ExportGap instead; see the note above about
        // PriorityInversion::pi_mode and friends.
        WorkType::FutexPingPong { spin_iters } => {
            json!({ "futex_ping_pong": { "spin_iters": spin_iters } })
        }
        WorkType::CrossAffinityChurn { spin_iters } => {
            json!({ "cross_affinity_churn": { "spin_iters": spin_iters } })
        }

        _ => return None,
    })
}

fn work_spec(w: &WorkSpec, at: &str, gaps: &mut Vec<ExportGap>) -> Option<serde_json::Value> {
    let Some(wt) = work_type(&w.work_type) else {
        gaps.push(ExportGap {
            where_: at.to_string(),
            construct: format!("WorkType::{:?}", w.work_type),
            reason: "no SourceWorkType counterpart is mapped yet; the lowering, \
                     not this exporter, is where a work type is approximated and \
                     the discarded dimension recorded"
                .to_string(),
        });
        return None;
    };
    // `nice` used to be hardcoded null here while `WorkSpec` carried one and
    // `SourceWorkSpec` had a field waiting for it — a silent drop, in the file
    // whose whole premise is that it is allowed to be wrong but not quietly
    // wrong. It went unnoticed because every scenario ported so far leaves nice
    // at the default.
    //
    // The widths differ (ktstr i32, IR i8). Linux nice is -20..=19 so every
    // legal value fits, and an out-of-range one is recorded as a gap rather
    // than truncated into a different, plausible priority.
    let nice = match w.nice {
        None => serde_json::Value::Null,
        Some(n) => match i8::try_from(n) {
            Ok(v) => serde_json::json!(v),
            Err(_) => {
                gaps.push(ExportGap {
                    where_: at.to_string(),
                    construct: format!("WorkSpec::nice = {n}"),
                    reason: "outside i8, so outside the -20..=19 nice range the \
                             record can express; truncating would silently \
                             substitute a different priority"
                        .to_string(),
                });
                serde_json::Value::Null
            }
        },
    };

    // `sched_policy` has no counterpart in `SourceWorkSpec` at all, so this is
    // a gap in the SCHEMA rather than in this function — but it was previously
    // not even mentioned, which made a policy-mixing scenario export as though
    // every worker were SCHED_NORMAL. `custom_sched_mixed` is exactly that
    // scenario: its point is a Normal/Batch/Idle/FIFO mix.
    if w.sched_policy != Default::default() {
        gaps.push(ExportGap {
            where_: at.to_string(),
            construct: format!("WorkSpec::sched_policy = {:?}", w.sched_policy),
            reason: "SourceWorkSpec has no scheduling-policy field, so the \
                     record cannot distinguish SCHED_NORMAL from BATCH, IDLE, \
                     FIFO or DEADLINE; a policy-mixing scenario would otherwise \
                     export as uniformly SCHED_NORMAL"
                .to_string(),
        });
    }

    Some(serde_json::json!({
        "workers": w.num_workers.map(|n| u32::try_from(n).unwrap_or(u32::MAX)),
        "work_type": wt,
        "nice": nice,
    }))
}

fn cgroup_def(def: &CgroupDef, at: &str, gaps: &mut Vec<ExportGap>) -> serde_json::Value {
    let cs = match &def.cpuset {
        None => serde_json::Value::Null,
        Some(spec) => match cpuset(spec) {
            Some(v) => v,
            None => {
                gaps.push(ExportGap {
                    where_: at.to_string(),
                    construct: format!("CpusetSpec::{spec:?}"),
                    reason: "no SourceCpuset counterpart; resolving it here would \
                             bind the scenario to this host's topology"
                        .to_string(),
                });
                serde_json::Value::Null
            }
        },
    };

    // `works` empty means "one default WorkSpec", which is what the step runner
    // resolves it to (`merged_works`). Make that explicit in the record rather
    // than exporting an empty list the consumer would have to know to reinterpret.
    let works: Vec<serde_json::Value> = if def.works.is_empty() {
        work_spec(&WorkSpec::default(), at, gaps)
            .into_iter()
            .collect()
    } else {
        def.works
            .iter()
            .filter_map(|w| work_spec(w, at, gaps))
            .collect()
    };

    if def.payload.is_some() {
        gaps.push(ExportGap {
            where_: at.to_string(),
            construct: "CgroupDef::payload".to_string(),
            reason: "a payload runs an external binary; its behaviour is not \
                     derivable from the declaration"
                .to_string(),
        });
    }

    serde_json::json!({
        "name": def.name.as_ref(),
        "cpuset": cs,
        "works": works,
        "cpu_quota": serde_json::Value::Null,
        "cpu_weight": serde_json::Value::Null,
    })
}

fn hold(h: &HoldSpec) -> serde_json::Value {
    use serde_json::json;
    match h {
        HoldSpec::Frac(f) => json!({ "frac": f }),
        HoldSpec::Fixed(d) => json!({ "fixed": ns(*d) }),
        HoldSpec::Loop { interval } => json!({ "loop": { "interval": ns(*interval) } }),
    }
}

fn step(s: &Step, idx: usize, gaps: &mut Vec<ExportGap>) -> serde_json::Value {
    let setup: Vec<serde_json::Value> = match &s.setup {
        Setup::Defs(defs) => defs
            .iter()
            .enumerate()
            .map(|(i, d)| {
                let at = format!("step[{idx}].setup[{i}] {:?}", d.name.as_ref());
                cgroup_def(d, &at, gaps)
            })
            .collect(),
        Setup::Factory(_) => {
            gaps.push(ExportGap {
                where_: format!("step[{idx}].setup"),
                construct: "Setup::Factory".to_string(),
                reason: "the cgroup list is produced by an fn(&Ctx) and cannot be \
                         evaluated without a running guest"
                    .to_string(),
            });
            Vec::new()
        }
    };

    if !s.ops.is_empty() {
        gaps.push(ExportGap {
            where_: format!("step[{idx}].ops"),
            construct: format!("{} Op(s)", s.ops.len()),
            reason: "ops are not exported yet; the record would describe a \
                     scenario missing its mutations"
                .to_string(),
        });
    }

    serde_json::json!({
        "setup": setup,
        "ops": [],
        "hold": hold(&s.hold),
    })
}

/// Export `def` as a `SourceScenario`-shaped record.
///
/// `topology` and `duration` come from the test's `#[ktstr_scenario]`
/// attributes (via its `KtstrTestEntry`) rather than from the `ScenarioDef`,
/// because that is where they live — the scenario value carries the workload,
/// the attributes carry the machine it runs on.
///
/// `default_workers_per_cgroup` is recorded explicitly. A `WorkSpec` with
/// `num_workers: None` means "inherit", and the two backends would otherwise
/// inherit *different* defaults — which is precisely the kind of divergence
/// that makes a cross-backend comparison meaningless.
pub fn export_scenario(
    name: &str,
    def: &ScenarioDef,
    topology: &Topology,
    duration: std::time::Duration,
    default_workers_per_cgroup: u32,
) -> Export {
    let mut gaps = Vec::new();
    let steps: Vec<serde_json::Value> = def
        .steps()
        .iter()
        .enumerate()
        .map(|(i, s)| step(s, i, &mut gaps))
        .collect();

    if def.checks().is_some() {
        gaps.push(ExportGap {
            where_: "scenario".to_string(),
            construct: "Assert override".to_string(),
            reason: "the record carries the workload, not the oracle; a shared \
                     oracle has to be expressed against both backends' outputs, \
                     not smuggled through the workload record"
                .to_string(),
        });
    }

    let record = serde_json::json!({
        "name": name,
        "topology": {
            "numa_nodes": topology.numa_nodes,
            "llcs": topology.llcs,
            "cores": topology.cores_per_llc,
            "threads": topology.threads_per_core,
        },
        "duration": ns(duration),
        "steps": steps,
        "default_workers_per_cgroup": default_workers_per_cgroup,
    });

    Export { record, gaps }
}

#[cfg(test)]
mod tests;
