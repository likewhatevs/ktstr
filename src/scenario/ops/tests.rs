use std::borrow::Cow;
use std::ops::RangeInclusive;

use super::*;
use crate::workload::{AffinityIntent, WorkSpec, WorkType};
use strum::IntoEnumIterator;

/// Pin the `Op::ReplaceScheduler` worker-not-trying gate
/// deadline. The 5 s value is load-bearing per
/// `REPLACE_NOT_TRYING_DEADLINE_S`'s doc: bumping risks pushing
/// total dispatch latency past test `duration_s` budgets;
/// lowering risks spurious "stayed in TRYING past deadline"
/// bails on cold-cache CI. A regression that silently changes
/// it would surface as either a flake or a slow test, neither
/// of which gives an actionable signal — this pin is the canary.
#[test]
fn replace_not_trying_deadline_pinned_to_5s() {
    assert_eq!(
        REPLACE_NOT_TRYING_DEADLINE_S, 5,
        "REPLACE_NOT_TRYING_DEADLINE_S changed away from 5 s; \
             read its doc and update the dispatch-latency analysis \
             before relaxing this assertion."
    );
}

/// Exhaustiveness guard for [`OpKind::bit_index`]. A new [`Op`]
/// variant auto-generates a matching [`OpKind`] variant (via
/// `#[derive(strum::EnumDiscriminants)]`), which the match arms
/// in `bit_index` must cover — adding an `Op` variant without
/// extending `bit_index` fails compilation. But the arms could
/// still drift in a way the compiler cannot see: two variants
/// accidentally mapped to the same index, or the contiguous-
/// from-zero invariant broken by a typo.
///
/// This test iterates every `OpKind` via `EnumIter` and pins
/// both invariants:
/// - Every variant produces a distinct bit index.
/// - Indices are contiguous `0..N` where N = variant count.
///
/// A regression — duplicate index, gap, or an off-by-one —
/// surfaces here before it silently corrupts the `op_kinds`
/// bitmask semantics elsewhere in the crate.
#[test]
fn op_kind_bit_indices_are_unique_and_contiguous() {
    let kinds: Vec<OpKind> = OpKind::iter().collect();
    let indices: Vec<u32> = kinds.iter().copied().map(OpKind::bit_index).collect();

    // Unique: every kind has a distinct index.
    let unique: std::collections::BTreeSet<u32> = indices.iter().copied().collect();
    assert_eq!(
        unique.len(),
        indices.len(),
        "OpKind::bit_index produced duplicates. \
             Pairs (OpKind, bit_index): {:?}. Fix the match in \
             OpKind::bit_index so every variant maps to a distinct \
             bit.",
        kinds.iter().zip(&indices).collect::<Vec<_>>(),
    );

    // Contiguous: indices form `0..N`.
    let expected: Vec<u32> = (0..kinds.len() as u32).collect();
    let mut sorted = indices.clone();
    sorted.sort_unstable();
    assert_eq!(
        sorted,
        expected,
        "OpKind::bit_index indices must be contiguous from 0 \
             (no gaps, no duplicates). Got sorted indices {sorted:?} \
             for {} OpKind variants; expected {expected:?}.",
        kinds.len(),
    );
}

/// `OpKind::iter()` order matches `bit_index` ascending order.
/// strum's `EnumIter` derive follows declaration order by default
/// — this test pins that contract so a future strum upgrade or
/// an enum reorder that decouples the two orderings surfaces
/// here instead of silently reshuffling bitmask traversal.
///
/// Complements `op_kind_bit_indices_are_unique_and_contiguous`
/// (which proves bit_index forms 0..N but not that iter() yields
/// them ascending) and the discriminant tests (which don't
/// exercise iter order at all).
#[test]
fn op_kind_iter_order_matches_bit_index_ascending() {
    let kinds: Vec<OpKind> = OpKind::iter().collect();
    let pairs: Vec<(usize, u32)> = kinds
        .iter()
        .enumerate()
        .map(|(i, k)| (i, k.bit_index()))
        .collect();
    for (i, bit) in &pairs {
        assert_eq!(
            *bit as usize, *i,
            "OpKind::iter()[{i}] (variant {:?}) has bit_index {bit}; \
                 expected iter-index to match bit_index. Pairs: {pairs:?}",
            kinds[*i],
        );
    }
}

// -- Traverse combinator (test-only) --

/// Layout strategy for Traverse phases.
#[derive(Debug)]
enum Layout {
    Disjoint,
    /// Overlapping cpusets. (min_frac, max_frac) — PRNG picks a value in range.
    Overlap(f64, f64),
}

/// Generates a random walk of cgroup topology changes across phases.
///
/// Each phase picks a random (cgroup_count, layout) pair, generates SetCpuset
/// ops, spawns workers in new cgroups, and holds for phase_duration.
///
/// `persistent_cgroups` cgroups are created in phase 0 and never removed.
/// Only cgroups at index >= `persistent_cgroups` are added/removed by the
/// random walk. The `cgroup_count` range applies to the total cgroup count
/// (persistent + ephemeral).
///
/// `cgroup_workloads` controls the workload for each cgroup index. If the
/// vec has fewer entries than the cgroup index, the last entry repeats.
#[derive(Debug)]
struct Traverse {
    seed: Option<u64>,
    cgroup_count: RangeInclusive<usize>,
    layouts: Vec<Layout>,
    phases: usize,
    phase_duration: Duration,
    settle: Duration,
    /// Cgroups [0..persistent_cgroups) are created once and never removed.
    persistent_cgroups: usize,
    /// WorkSpec definition per cgroup index. Last entry repeats for higher indices.
    cgroup_workloads: Vec<WorkSpec>,
}

impl Traverse {
    /// Generate a `Vec<Step>` from the Traverse configuration.
    fn generate(&self, ctx: &Ctx) -> Vec<Step> {
        use rand::RngExt;

        let seed = self.seed.unwrap_or_else(|| std::process::id() as u64);
        let mut rng = seeded_rng(seed);

        let usable_len = ctx.topo.usable_cpus().len();
        let max_cgroups = (*self.cgroup_count.end()).min(usable_len / 2).max(1);
        let min_cgroups = (*self.cgroup_count.start()).max(1).min(max_cgroups);

        let mut steps = Vec::with_capacity(self.phases + 1);
        let mut live_cgroups: Vec<Cow<'static, str>> = Vec::new();

        let names: Vec<Cow<'static, str>> = (0..max_cgroups)
            .map(|i| Cow::Owned(format!("cg_{i}")))
            .collect();

        for phase in 0..self.phases {
            let range = max_cgroups - min_cgroups + 1;
            let target_count = min_cgroups + rng.random_range(0..range);
            let layout_idx = rng.random_range(0..self.layouts.len());
            let layout = &self.layouts[layout_idx];

            let mut ops = Vec::new();

            // Add cgroups if needed.
            while live_cgroups.len() < target_count {
                let idx = live_cgroups.len();
                let name = names[idx].clone();
                let w = self
                    .cgroup_workloads
                    .get(idx)
                    .or(self.cgroup_workloads.last())
                    .cloned()
                    .unwrap_or_default();
                ops.push(Op::AddCgroup { name: name.clone() });
                ops.push(Op::Spawn {
                    placement: SpawnPlacement::Cgroup(name.clone()),
                    work: w,
                });
                live_cgroups.push(name);
            }

            // Remove cgroups if needed (never remove persistent cgroups).
            while live_cgroups.len() > target_count && live_cgroups.len() > self.persistent_cgroups
            {
                if let Some(name) = live_cgroups.pop() {
                    ops.push(Op::StopCgroup {
                        cgroup: name.clone(),
                    });
                    ops.push(Op::RemoveCgroup { cgroup: name });
                }
            }

            // Apply cpuset layout.
            for (i, name) in live_cgroups.iter().enumerate() {
                let spec = match layout {
                    Layout::Disjoint => CpusetSpec::Disjoint {
                        index: i,
                        of: live_cgroups.len(),
                    },
                    Layout::Overlap(min_frac, max_frac) => {
                        let frac = min_frac
                            + rng.random_range(0..100) as f64 / 100.0 * (max_frac - min_frac);
                        CpusetSpec::Overlap {
                            index: i,
                            of: live_cgroups.len(),
                            frac,
                        }
                    }
                };
                ops.push(Op::SetCpuset {
                    cgroup: name.clone(),
                    cpus: spec,
                });
            }

            let hold = if phase == 0 {
                // First phase includes settle time.
                HoldSpec::fixed(self.settle + self.phase_duration)
            } else {
                HoldSpec::fixed(self.phase_duration)
            };

            steps.push(Step {
                setup: vec![].into(),
                ops,
                hold,
            });
        }

        steps
    }
}

/// Seeded PRNG for deterministic topology generation.
fn seeded_rng(seed: u64) -> rand::rngs::StdRng {
    use rand::SeedableRng;
    rand::rngs::StdRng::seed_from_u64(seed)
}

// -- validate_known_flags tests --

/// Declared allowlist, every `--flag` in args is on the
/// allowlist → `Ok(())`. Covers both `--foo` and
/// `--foo=value` shapes to pin the flag-body split.
#[test]
fn validate_known_flags_accepts_listed_long_flags() {
    use crate::test_support::{OutputFormat, Payload, PayloadKind};
    static WITH_ALLOWLIST: Payload = Payload {
        name: "with_allowlist",
        kind: PayloadKind::Binary("/bin/true"),
        output: OutputFormat::ExitCode,
        default_args: &[],
        default_checks: &[],
        metrics: &[],
        include_files: &[],
        uses_parent_pgrp: false,
        known_flags: Some(&["runtime", "threads", "verbose"]),
        metric_bounds: None,
    };
    let args: Vec<String> = vec![
        "--runtime=30".into(),
        "--threads".into(),
        "4".into(),
        "--verbose".into(),
        "positional_arg".into(),
        "-s".into(), // short flags aren't inspected
        // Degenerate forms: the bare `--` (end-of-flags
        // marker used by many CLIs) and `--=value` (empty
        // name before `=`) both skip the allowlist check
        // because the extracted flag name is empty. Pin the
        // empty-name skip path so a future refactor can't
        // accidentally treat them as unknown long flags.
        "--".into(),
        "--=value".into(),
    ];
    validate_known_flags(&WITH_ALLOWLIST, &args).expect("all long flags in allowlist must pass");
}

/// Fail-fast ordering: when args contain a known flag, a
/// typo, then another known flag, the error must name ONLY
/// the typo — the validator bails on the first unknown flag
/// without continuing to inspect later args.
#[test]
fn validate_known_flags_fails_fast_on_first_unknown() {
    use crate::test_support::{OutputFormat, Payload, PayloadKind};
    static WITH_ALLOWLIST: Payload = Payload {
        name: "with_allowlist",
        kind: PayloadKind::Binary("/bin/true"),
        output: OutputFormat::ExitCode,
        default_args: &[],
        default_checks: &[],
        metrics: &[],
        include_files: &[],
        uses_parent_pgrp: false,
        known_flags: Some(&["runtime", "threads", "verbose"]),
        metric_bounds: None,
    };
    let args = vec!["--runtime=30".into(), "--threds".into(), "--verbose".into()];
    let err = validate_known_flags(&WITH_ALLOWLIST, &args)
        .expect_err("typo between two known flags must be rejected");
    let msg = format!("{err:#}");
    assert!(msg.contains("--threds"), "error must name the typo: {msg}");
    assert!(
        !msg.contains("--verbose"),
        "error must not mention the later known flag '--verbose' \
             — fail-fast broke: {msg}",
    );
}

/// A `--flag` whose bare name is not on the allowlist bails
/// with a message naming both the offending flag and the
/// allowlist — the loud-typo-detection contract.
#[test]
fn validate_known_flags_rejects_unknown_long_flag() {
    use crate::test_support::{OutputFormat, Payload, PayloadKind};
    static WITH_ALLOWLIST: Payload = Payload {
        name: "with_allowlist",
        kind: PayloadKind::Binary("/bin/true"),
        output: OutputFormat::ExitCode,
        default_args: &[],
        default_checks: &[],
        metrics: &[],
        include_files: &[],
        uses_parent_pgrp: false,
        known_flags: Some(&["runtime", "threads"]),
        metric_bounds: None,
    };
    // "threds" is a typo for "threads" — the exact failure
    // the allowlist exists to catch.
    let args = vec!["--threds".to_string(), "4".to_string()];
    let err = validate_known_flags(&WITH_ALLOWLIST, &args).expect_err("typo must be rejected");
    let msg = format!("{err:#}");
    assert!(
        msg.contains("--threds"),
        "error must name the offending flag: {msg}",
    );
    assert!(
        msg.contains("known_flags allowlist"),
        "error must mention the allowlist surface: {msg}",
    );
}

/// `known_flags: None` (the default on every Payload that
/// doesn't opt in) lets every `--flag` through without
/// inspection. Required for payloads that wrap binaries with
/// open-ended flag surfaces (stress-ng, fio, schbench).
#[test]
fn validate_known_flags_none_is_permissive() {
    use crate::test_support::{OutputFormat, Payload, PayloadKind};
    static NO_ALLOWLIST: Payload = Payload {
        name: "no_allowlist",
        kind: PayloadKind::Binary("/bin/true"),
        output: OutputFormat::ExitCode,
        default_args: &[],
        default_checks: &[],
        metrics: &[],
        include_files: &[],
        uses_parent_pgrp: false,
        known_flags: None,
        metric_bounds: None,
    };
    let args: Vec<String> = vec![
        "--anything".into(),
        "--whatever=x".into(),
        "--threds".into(),
    ];
    validate_known_flags(&NO_ALLOWLIST, &args).expect("None allowlist must pass any flag");
}

// -- Op discriminant tests --

#[test]
fn op_discriminant_unique() {
    use crate::test_support::{OutputFormat, Payload, PayloadKind};
    static TRUE_BIN: Payload = Payload {
        name: "true_bin",
        kind: PayloadKind::Binary("/bin/true"),
        output: OutputFormat::ExitCode,
        default_args: &[],
        default_checks: &[],
        metrics: &[],
        include_files: &[],
        uses_parent_pgrp: false,
        known_flags: None,
        metric_bounds: None,
    };
    let ops: Vec<Op> = vec![
        Op::AddCgroup { name: "a".into() },
        Op::AddCgroupDef {
            def: CgroupDef::named("a"),
        },
        Op::RemoveCgroup { cgroup: "a".into() },
        Op::SetCpuset {
            cgroup: "a".into(),
            cpus: CpusetSpec::exact([]),
        },
        Op::ClearCpuset { cgroup: "a".into() },
        Op::SwapCpusets {
            a: "a".into(),
            b: "b".into(),
        },
        Op::Spawn {
            placement: SpawnPlacement::Cgroup("a".into()),
            work: Default::default(),
        },
        Op::StopCgroup { cgroup: "a".into() },
        Op::SetAffinity {
            cgroup: "a".into(),
            affinity: Default::default(),
        },
        Op::MoveAllTasks {
            from: "a".into(),
            to: "b".into(),
        },
        Op::RunPayload {
            payload: &TRUE_BIN,
            args: vec![],
            cgroup: None,
        },
        Op::WaitPayload {
            name: "p".into(),
            cgroup: None,
        },
        Op::KillPayload {
            name: "p".into(),
            cgroup: None,
        },
        Op::FreezeCgroup { cgroup: "a".into() },
        Op::UnfreezeCgroup { cgroup: "a".into() },
        Op::CaptureSnapshot {
            name: "snap".into(),
        },
        Op::WatchSnapshot {
            symbol: "kernel.x".into(),
        },
        Op::WriteKernelHot {
            writes: vec![(KernelTarget::symbol("x"), KernelValue::u64(0))],
        },
        Op::WriteKernelCold {
            writes: vec![(KernelTarget::symbol("x"), KernelValue::u64(0))],
        },
        Op::ReadKernelHot {
            tag: "t".into(),
            target: KernelTarget::symbol("x"),
            width: KernelValueWidth::u64(),
        },
        Op::ReadKernelCold {
            tag: "t".into(),
            target: KernelTarget::symbol("x"),
            width: KernelValueWidth::u64(),
        },
        Op::CaptureCgroupProcs {
            tag: "snap".into(),
            cgroup: "a".into(),
        },
    ];
    let mut seen = std::collections::BTreeSet::new();
    for op in &ops {
        assert!(seen.insert(op.discriminant()), "duplicate discriminant");
    }
}

/// Pin one Op variant's discriminant to `want`. `#[track_caller]`
/// reports a mismatch at the call site (the specific variant), and
/// `name` labels the variant in the panic message — so a
/// multi-variant failure is operator-readable from the cargo-test
/// output without a source cross-reference.
#[track_caller]
fn assert_discriminant(op: Op, want: u32, name: &str) {
    assert_eq!(op.discriminant(), want, "{name}");
}

/// Pins every Op variant's exact discriminant value against the
/// canonical `OpKind::bit_index` match in types.rs. A renumbering
/// or reordering surfaces here naming the specific variant that
/// moved (via [`assert_discriminant`]'s `name` label) —
/// complementing `op_kind_bit_indices_are_unique_and_contiguous`
/// (whose contiguity arm surfaces gaps as sorted indices only,
/// not the offending variant; its uniqueness arm DOES name
/// variants via the `{:?}` of `(OpKind, bit_index)` pairs) and
/// `op_discriminant_unique` (which proves no collisions via the
/// `BTreeSet::insert` "duplicate discriminant" panic).
///
/// The pin is split into theme-grouped sub-tests
/// (`op_discriminant_cgroup_ops`, `op_discriminant_workload_ops`,
/// `op_discriminant_payload_ops`,
/// `op_discriminant_freeze_snapshot_kernel_ops`,
/// `op_discriminant_scheduler_ops`); their union covers every Op
/// variant (discriminants 0..=26) exactly once, in source order.
#[test]
fn op_discriminant_cgroup_ops() {
    assert_discriminant(Op::AddCgroup { name: "a".into() }, 0, "AddCgroup");
    assert_discriminant(Op::AddCgroupDef { def: CgroupDef::named("a") }, 1, "AddCgroupDef");
    assert_discriminant(Op::RemoveCgroup { cgroup: "a".into() }, 2, "RemoveCgroup");
    assert_discriminant(
        Op::SetCpuset {
            cgroup: "a".into(),
            cpus: CpusetSpec::Llc(0),
        },
        3,
        "SetCpuset",
    );
    assert_discriminant(Op::ClearCpuset { cgroup: "a".into() }, 4, "ClearCpuset");
    assert_discriminant(
        Op::SwapCpusets {
            a: "a".into(),
            b: "b".into(),
        },
        5,
        "SwapCpusets",
    );
}

/// Discriminant pin for the workload/placement Op variants
/// (discriminants 6..=9). See [`op_discriminant_cgroup_ops`] for the
/// full pin rationale.
#[test]
fn op_discriminant_workload_ops() {
    assert_discriminant(Op::spawn(SpawnPlacement::cgroup("a"), WorkSpec::default()), 6, "Spawn");
    assert_discriminant(Op::StopCgroup { cgroup: "a".into() }, 7, "StopCgroup");
    assert_discriminant(
        Op::SetAffinity {
            cgroup: "a".into(),
            affinity: AffinityIntent::Inherit,
        },
        8,
        "SetAffinity",
    );
    assert_discriminant(
        Op::MoveAllTasks {
            from: "a".into(),
            to: "b".into(),
        },
        9,
        "MoveAllTasks",
    );
}

/// Discriminant pin for the payload Op variants (discriminants
/// 10..=12). See [`op_discriminant_cgroup_ops`] for the full pin
/// rationale.
#[test]
fn op_discriminant_payload_ops() {
    use crate::test_support::{OutputFormat, Payload, PayloadKind};
    static TRUE_BIN: Payload = Payload {
        name: "true_bin",
        kind: PayloadKind::Binary("/bin/true"),
        output: OutputFormat::ExitCode,
        default_args: &[],
        default_checks: &[],
        metrics: &[],
        include_files: &[],
        uses_parent_pgrp: false,
        known_flags: None,
        metric_bounds: None,
    };
    assert_discriminant(
        Op::RunPayload {
            payload: &TRUE_BIN,
            args: vec![],
            cgroup: None,
        },
        10,
        "RunPayload",
    );
    assert_discriminant(
        Op::WaitPayload {
            name: "p".into(),
            cgroup: None,
        },
        11,
        "WaitPayload",
    );
    assert_discriminant(
        Op::KillPayload {
            name: "p".into(),
            cgroup: None,
        },
        12,
        "KillPayload",
    );
}

/// Discriminant pin for the freeze, snapshot, and kernel-I/O Op
/// variants (discriminants 13..=20). See [`op_discriminant_cgroup_ops`]
/// for the full pin rationale.
#[test]
fn op_discriminant_freeze_snapshot_kernel_ops() {
    assert_discriminant(Op::FreezeCgroup { cgroup: "a".into() }, 13, "FreezeCgroup");
    assert_discriminant(Op::UnfreezeCgroup { cgroup: "a".into() }, 14, "UnfreezeCgroup");
    assert_discriminant(Op::CaptureSnapshot { name: "snap".into() }, 15, "Snapshot");
    assert_discriminant(Op::WatchSnapshot { symbol: "kernel.x".into() }, 16, "WatchSnapshot");
    assert_discriminant(
        Op::WriteKernelHot {
            writes: vec![(KernelTarget::symbol("x"), KernelValue::u64(0))],
        },
        17,
        "WriteKernelHot",
    );
    assert_discriminant(
        Op::WriteKernelCold {
            writes: vec![(KernelTarget::symbol("x"), KernelValue::u64(0))],
        },
        18,
        "WriteKernelCold",
    );
    assert_discriminant(
        Op::ReadKernelHot {
            tag: "t".into(),
            target: KernelTarget::symbol("x"),
            width: KernelValueWidth::u64(),
        },
        19,
        "ReadKernelHot",
    );
    assert_discriminant(
        Op::ReadKernelCold {
            tag: "t".into(),
            target: KernelTarget::symbol("x"),
            width: KernelValueWidth::u64(),
        },
        20,
        "ReadKernelCold",
    );
}

/// Discriminant pin for the scheduler-control Op variants plus
/// `PinBpfMap`/`CaptureCgroupProcs` (discriminants 21..=26). See
/// [`op_discriminant_cgroup_ops`] for the full pin rationale.
#[test]
fn op_discriminant_scheduler_ops() {
    static SCHED_FIXTURE: crate::test_support::Scheduler = crate::test_support::Scheduler::EEVDF;
    assert_discriminant(
        Op::AttachScheduler {
            scheduler: &SCHED_FIXTURE,
        },
        21,
        "AttachScheduler",
    );
    assert_discriminant(Op::DetachScheduler, 22, "DetachScheduler");
    assert_discriminant(Op::RestartScheduler, 23, "RestartScheduler");
    assert_discriminant(
        Op::ReplaceScheduler {
            scheduler: &SCHED_FIXTURE,
        },
        24,
        "ReplaceScheduler",
    );
    assert_discriminant(Op::PinBpfMap { name: "scx_test.bss".into() }, 25, "PinBpfMap");
    assert_discriminant(
        Op::CaptureCgroupProcs {
            tag: "snap".into(),
            cgroup: "a".into(),
        },
        26,
        "CaptureCgroupProcs",
    );
}

// -- seeded_rng tests --

#[test]
fn seeded_rng_deterministic() {
    use rand::RngExt;
    let mut rng1 = seeded_rng(42);
    let mut rng2 = seeded_rng(42);
    for _ in 0..100 {
        assert_eq!(rng1.random::<u64>(), rng2.random::<u64>());
    }
}

#[test]
fn seeded_rng_different_seeds_differ() {
    use rand::RngExt;
    let mut rng1 = seeded_rng(1);
    let mut rng2 = seeded_rng(2);
    let same = (0..10).all(|_| rng1.random::<u64>() == rng2.random::<u64>());
    assert!(!same);
}

// -- HoldSpec validate --

#[test]
fn holdspec_validate_accepts_valid() {
    HoldSpec::Frac(0.5).validate().unwrap();
    HoldSpec::Frac(1.0).validate().unwrap();
    HoldSpec::Fixed(Duration::from_millis(1))
        .validate()
        .unwrap();
    HoldSpec::Loop {
        interval: Duration::from_millis(100),
    }
    .validate()
    .unwrap();
}

#[test]
fn holdspec_validate_accepts_fixed_zero() {
    HoldSpec::Fixed(Duration::ZERO)
        .validate()
        .expect("Duration::ZERO is valid for settle/op-only steps");
}

#[test]
fn holdspec_validate_rejects_frac_zero() {
    let err = HoldSpec::Frac(0.0).validate().unwrap_err();
    assert!(err.contains("Frac") && err.contains("> 0"), "got: {err}");
}

#[test]
fn holdspec_validate_rejects_frac_negative() {
    let err = HoldSpec::Frac(-0.5).validate().unwrap_err();
    assert!(err.contains("Frac") && err.contains("> 0"), "got: {err}");
}

#[test]
fn holdspec_validate_rejects_frac_nan() {
    let err = HoldSpec::Frac(f64::NAN).validate().unwrap_err();
    assert!(
        err.contains("not finite") || err.contains("NaN"),
        "got: {err}"
    );
}

#[test]
fn holdspec_validate_rejects_frac_inf() {
    let err = HoldSpec::Frac(f64::INFINITY).validate().unwrap_err();
    assert!(
        err.contains("not finite") || err.contains("Inf"),
        "got: {err}"
    );
}

#[test]
fn holdspec_validate_rejects_loop_zero_interval() {
    let err = HoldSpec::Loop {
        interval: Duration::ZERO,
    }
    .validate()
    .unwrap_err();
    assert!(err.contains("Loop") && err.contains("busy"), "got: {err}");
}

// -- HoldSpec variants (exercise constructors + Step storage + PartialEq) --

#[test]
fn holdspec_frac() {
    let step = Step::new(vec![], HoldSpec::frac(0.5));
    assert_eq!(step.hold, HoldSpec::Frac(0.5));
}

#[test]
fn holdspec_fixed() {
    let step = Step::new(vec![], HoldSpec::fixed(Duration::from_secs(3)));
    assert_eq!(step.hold, HoldSpec::Fixed(Duration::from_secs(3)));
}

#[test]
fn holdspec_loop() {
    let step = Step::new(vec![], HoldSpec::loop_at(Duration::from_millis(100)));
    assert_eq!(
        step.hold,
        HoldSpec::Loop {
            interval: Duration::from_millis(100)
        }
    );
}

/// Drive `HoldSpec::Loop` end-to-end via `execute_steps` against
/// the mock CgroupOps. The `HoldSpec::Loop { interval }` arm of `run_step`
/// fires `apply_ops` repeatedly at `interval` until `ctx.duration`
/// elapses; each iteration's SetCpuset op records a
/// `CgroupCall::SetCpuset` in the mock. After the scenario
/// completes, the mock's SetCpuset count proves the loop actually
/// repeated — distinguishing the Loop path from the Fixed/Frac
/// single-apply path. `sched_pid = None` (inherited from `mock_ctx`)
/// makes `hold_or_sched_died` a plain sleep with no liveness probe
/// (verified by `hold_or_sched_died`'s `let Some(pid) = sched_pid else { ... }`
/// no-pid arm, which polls only the crash latch with no pidfd liveness
/// probe), so the loop exits cleanly on the
/// duration deadline rather than on a spurious dead-scheduler signal.
/// `duration` is overridden to 150ms (vs `mock_ctx`'s 1-second
/// default) to keep the unit-test runtime short. Lower bound is
/// loose (>= 2) to absorb CI timing variance — the contract being
/// pinned is "repeats at least once", not "fires exactly N times".
#[test]
fn holdspec_loop_apply_path_repeats_ops_until_duration_elapses() {
    let mock = MockCgroupOps::new();
    let topo = mock_topo();
    let mut ctx = mock_ctx(&mock, &topo);
    ctx.duration = Duration::from_millis(150);
    let steps = vec![Step::new(
        vec![Op::set_cpuset("loop_test", CpusetSpec::Llc(0))],
        HoldSpec::loop_at(Duration::from_millis(30)),
    )];
    let result = execute_steps(&ctx, steps)
        .expect("HoldSpec::Loop apply path must succeed against mock cgroups");
    assert!(
        result.is_pass(),
        "scenario must pass with no failing assertions; got: {:?}",
        result.outcomes,
    );
    let set_cpuset_calls = mock
        .calls()
        .iter()
        .filter(|c| matches!(c, CgroupCall::SetCpuset(name, _) if name == "loop_test"))
        .count();
    assert!(
        set_cpuset_calls >= 2,
        "HoldSpec::Loop with interval=30ms over duration=150ms must fire \
             SetCpuset at least twice; got {set_cpuset_calls} calls. The \
             `HoldSpec::Loop` arm of run_step must invoke apply_ops repeatedly \
             until the deadline; a regression that single-shotted the ops \
             would surface here as exactly 1 call.",
    );
}

/// The Loop arm's setup pass (its `if !step.setup.is_empty()` block,
/// placed before the `while` loop) runs `apply_setup`
/// ONCE before entering the while loop, NOT per-iteration. A
/// regression that moved the `if !step.setup.is_empty()` block
/// inside the loop would attempt to re-create the same cgroup
/// every iteration and bail on the second iteration's collision
/// check (apply_setup's `cgroup_name_is_tracked` guard).
/// Test pins this by counting `CreateCgroup` calls — must be
/// exactly 1 even though the loop body iterates multiple times
/// (verified separately via the SetCpuset count).
#[test]
fn holdspec_loop_apply_path_setup_runs_once_not_per_iteration() {
    let mock = MockCgroupOps::new();
    let topo = mock_topo();
    let mut ctx = mock_ctx(&mock, &topo);
    ctx.duration = Duration::from_millis(150);
    let steps = vec![
        Step::with_defs(
            vec![CgroupDef::named("setup_cg")],
            HoldSpec::loop_at(Duration::from_millis(30)),
        )
        .set_ops(vec![Op::set_cpuset("setup_cg", CpusetSpec::Llc(0))]),
    ];
    let result = execute_steps(&ctx, steps)
        .expect("HoldSpec::Loop with setup must succeed against mock cgroups");
    assert!(
        result.is_pass(),
        "scenario must pass with no failing assertions; got: {:?}",
        result.outcomes,
    );
    let create_calls = mock
        .calls()
        .iter()
        .filter(|c| matches!(c, CgroupCall::CreateCgroup(name) if name == "setup_cg"))
        .count();
    assert_eq!(
        create_calls, 1,
        "Loop arm's setup pass must run exactly ONCE before the loop body; \
             got {create_calls} CreateCgroup calls. A regression that moved \
             the `if !step.setup.is_empty()` block inside the while loop \
             (the Loop arm's pre-loop setup pass) would surface here as N > 1 calls (the second \
             iteration's apply_setup would also fail the collision check, \
             but counting reveals the bug source).",
    );
    let set_cpuset_calls = mock
        .calls()
        .iter()
        .filter(|c| matches!(c, CgroupCall::SetCpuset(name, _) if name == "setup_cg"))
        .count();
    assert!(
        set_cpuset_calls >= 2,
        "Loop body must repeat SetCpuset >= 2 times despite setup running \
             once; got {set_cpuset_calls}. Pairs with the create-once check \
             above to pin the full setup-once + ops-many contract.",
    );
}

/// `interval > duration` is a degenerate-but-valid Loop config:
/// the while loop body runs exactly ONCE (deadline reached after
/// the first apply_ops + sleep). Pins the exact-iteration
/// contract via `assert_eq!(..., 1)` — catches BOTH a regression
/// that skipped the first apply_ops (0 calls) AND a regression
/// in the deadline-min logic (the Loop arm's
/// `hold_or_sched_died(remaining.min(interval), ...)` call) that let the second
/// iteration's sleep underflow (2+ calls). That boundary behavior
/// (`hold_or_sched_died(remaining.min(interval), ...)`)
/// ensures sleep is capped at the remaining time so the loop
/// exits promptly on the next deadline check.
#[test]
fn holdspec_loop_apply_path_fires_exactly_once_when_interval_exceeds_duration() {
    let mock = MockCgroupOps::new();
    let topo = mock_topo();
    let mut ctx = mock_ctx(&mock, &topo);
    ctx.duration = Duration::from_millis(30);
    let steps = vec![Step::new(
        vec![Op::set_cpuset("brief_loop", CpusetSpec::Llc(0))],
        HoldSpec::loop_at(Duration::from_millis(100)),
    )];
    let result = execute_steps(&ctx, steps)
        .expect("HoldSpec::Loop with interval > duration must succeed against mock");
    assert!(
        result.is_pass(),
        "scenario must pass with no failing assertions; got: {:?}",
        result.outcomes,
    );
    let set_cpuset_calls = mock
        .calls()
        .iter()
        .filter(|c| matches!(c, CgroupCall::SetCpuset(name, _) if name == "brief_loop"))
        .count();
    assert_eq!(
        set_cpuset_calls, 1,
        "interval (100ms) > duration (30ms) must fire SetCpuset exactly \
             once; got {set_cpuset_calls}. The loop body should run a single \
             iteration: enter loop (now < deadline) → apply_ops → sleep \
             min(remaining, interval) = ~30ms → next deadline check fails. \
             0 calls = a regression that skipped the first apply_ops; 2+ \
             calls = a regression in the Loop arm's deadline-min logic \
             that let the second iteration's sleep underflow.",
    );
}

/// `Op::CaptureSnapshot` reached during the execution of a
/// `HoldSpec::Loop` step's ops vec must produce a hard error that
/// names the observer-effect rationale. A capture
/// forces a freeze rendezvous; inside a Loop generating a
/// high-rate pattern, even one capture per iteration destroys the
/// workload via N freezes/sec. Boundary captures emitted in a
/// non-Loop Step before/after the Loop step are the correct
/// pattern. The error message must explain the WHY so the test
/// author understands the redirect (not just "rejected").
#[test]
fn holdspec_loop_rejects_capture_snapshot_inside_ops_vec() {
    let mock = MockCgroupOps::new();
    let topo = mock_topo();
    let mut ctx = mock_ctx(&mock, &topo);
    ctx.duration = Duration::from_millis(60);
    let steps = vec![Step::new(
        vec![Op::capture_snapshot("inside_loop_capture")],
        HoldSpec::loop_at(Duration::from_millis(30)),
    )];
    // execute_steps catches per-step bail!() and surfaces them as
    // a Fail outcome on the returned AssertResult so the per-step
    // teardown (drain_on_err!) still runs. The bail message is
    // wrapped as `"step N failed: <bail message>"`. The test must
    // inspect the Fail outcome's message, not unwrap_err.
    let result = execute_steps(&ctx, steps)
        .expect("execute_steps returns Ok(AssertResult) even when a step's apply_ops bails");
    assert!(
        !result.is_pass(),
        "scenario with Op::CaptureSnapshot inside HoldSpec::Loop must NOT pass; got: {:?}",
        result.outcomes,
    );
    let fail_msg = result
        .outcomes
        .iter()
        .find_map(|o| match o {
            crate::assert::Outcome::Fail(detail) => Some(detail.message.clone()),
            _ => None,
        })
        .expect("at least one Fail outcome carrying the Loop+CaptureSnapshot reject message");
    assert!(
        fail_msg.contains("Op::CaptureSnapshot")
            && fail_msg.contains("HoldSpec::Loop")
            && fail_msg.contains("freezing every vCPU"),
        "Fail outcome must name the rejected op + the enclosing hold + the \
             concrete mechanism (every vCPU frozen per iteration) so the operator \
             understands the redirect from the message alone, not just see \
             'rejected' with no observer-effect rationale. got: {fail_msg}",
    );
    assert!(
        fail_msg.contains("non-Loop Step"),
        "Fail outcome must point to the correct fix (move capture into a non-Loop \
             Step before/after the Loop step); got: {fail_msg}",
    );
}

/// The Loop arm's sched-died-early-exit path (its
/// `if hold_or_sched_died(...) { *sched_died_during_hold = true; return Ok(()); }`)
/// fires when `hold_or_sched_died` observes the scheduler pid
/// has exited mid-loop. Setting `sched_died_during_hold = true`
/// and returning `Ok(())` is the contract — the outer caller
/// (`run_scenario`'s `if sched_died_during_hold` block) reads the flag
/// and stamps one of
/// `DetailKind::SchedulerCrashed` /
/// `DetailKind::SchedulerExitedCleanly` /
/// `DetailKind::SchedulerDiedUnknownReason` (chosen by
/// `sched_died_detail_kind` reading the probe BSS latch) with
/// `format_sched_died_during_workload`,
/// then marks the AssertResult `passed = false`.
///
/// Implementation: use `libc::pid_t::MAX` as the dead pid. The
/// kernel's PID_MAX_LIMIT (include/linux/threads.h) caps real
/// pids well below `i32::MAX`, so `pidfd_open` on `pid_t::MAX`
/// always returns ESRCH, which `hold_or_sched_died` maps to
/// "dead, return true." This pattern matches
/// [`crate::scenario::process_alive_nonexistent_pid`] (the same
/// trick is used to assert process-alive's no-such-pid path
/// without a fork+reap race window).
///
/// Pins: (1) `sched_pid` carrying a dead pid into the Loop arm
/// exits the while-loop after the first apply_ops iteration;
/// (2) the Loop arm's `*sched_died_during_hold = true` write
/// reaches the outer caller; (3) the outer caller pushes
/// one of the three sched-died `DetailKind` variants and marks
/// `passed = false`. A regression that DROPPED the early-exit
/// (loop runs all iterations after the death is observed) would
/// surface as multiple SetCpuset calls; a regression that
/// DROPPED the `sched_died_during_hold = true` write would
/// surface as passed=true with no sched-died detail. Note that
/// `return Ok(())` vs `break` produce identical observable
/// state here because the Loop arm is the last operation in
/// run_step's match block — both exit the while loop and fall
/// through to the same return — so the count assertion catches
/// loss of the early-exit BEHAVIOR, not the specific keyword
/// chosen to implement it.
#[test]
fn holdspec_loop_arm_exits_early_when_sched_dies_during_hold() {
    let mock = MockCgroupOps::new();
    let topo = mock_topo();
    let mut ctx = mock_ctx(&mock, &topo);
    // Short total duration keeps the test fast; the loop should
    // exit on iteration 1 long before this deadline anyway.
    ctx.duration = Duration::from_millis(150);
    // libc::pid_t::MAX is above kernel PID_MAX_LIMIT, so
    // pidfd_open inside hold_or_sched_died returns ESRCH
    // immediately. Same trick as scenario::process_alive's
    // no-such-pid test. Publishes via the SCHED_PID atomic
    // because the death-detection sites in apply_ops read
    // `crate::vmm::rust_init::sched_pid()` live (swap-aware
    // for Op::ReplaceScheduler) rather than the
    // `ctx.sched_pid` snapshot.
    ctx.sched_pid = Some(libc::pid_t::MAX);
    crate::vmm::rust_init::set_sched_pid(libc::pid_t::MAX);
    // SCHED_PID is a process-global atomic — restore to 0 on
    // exit so this test doesn't pollute the empty-pid contract
    // of neighbor tests (e.g. apply_ops_detach_scheduler_bails_when_no_scheduler_attached)
    // that read sched_pid() and expect None.
    struct ResetSchedPid;
    impl Drop for ResetSchedPid {
        fn drop(&mut self) {
            crate::vmm::rust_init::set_sched_pid(0);
        }
    }
    let _reset = ResetSchedPid;
    let steps = vec![Step::new(
        vec![Op::set_cpuset("died_test", CpusetSpec::Llc(0))],
        HoldSpec::loop_at(Duration::from_millis(30)),
    )];
    let result = execute_steps(&ctx, steps).expect(
            "Loop arm must return Ok even when sched dies — the death \
             is surfaced via sched_died_during_hold + one of the three sched-died DetailKind variants, \
             NOT as an Err out of run_step",
        );
    assert!(
        !result.is_pass(),
        "sched-died during the Loop hold must mark passed=false; \
             got passed=true with details: {:?}",
        result.outcomes,
    );
    let sched_died_details: Vec<_> = result
        .failure_details()
        .filter(|d| {
            matches!(
                d.kind,
                crate::assert::DetailKind::SchedulerCrashed
                    | crate::assert::DetailKind::SchedulerExitedCleanly
                    | crate::assert::DetailKind::SchedulerDiedUnknownReason
            )
        })
        .collect();
    assert_eq!(
        sched_died_details.len(),
        1,
        "must push exactly one sched-died DetailKind detail (from \
             `run_scenario`'s `if sched_died_during_hold` block); got {} sched-died failures out of {} total \
             failures: {:?}",
        sched_died_details.len(),
        result.failure_details().count(),
        result.outcomes,
    );
    let set_cpuset_calls = mock
        .calls()
        .iter()
        .filter(|c| matches!(c, CgroupCall::SetCpuset(name, _) if name == "died_test"))
        .count();
    // First iteration's `apply_ops` in the Loop arm fires BEFORE
    // that arm's `hold_or_sched_died` call, so a sched-died-from-
    // entry still records exactly one SetCpuset call. A
    // regression that DROPPED the early-exit (loop runs all
    // iterations after the death is observed) would surface as
    // multiple calls here; a regression that skipped the first
    // apply_ops would surface as zero.
    assert_eq!(
        set_cpuset_calls, 1,
        "sched-died-on-entry must apply ops once (iter 1) then exit; \
             got {set_cpuset_calls} SetCpuset calls. > 1 means the loop \
             continued past the sched-died signal (early-exit dropped); \
             0 means apply_ops was gated on liveness (would surface as \
             a missing-apply regression).",
    );
}

/// Custom worker that refuses the cooperative SIGUSR1 stop:
/// installs `SIG_IGN` for SIGUSR1 and clears any STOP the framework's
/// handler set before SIG_IGN took effect, then sleeps past
/// `stop_and_collect`'s 5s collection deadline. A small duplicate of
/// `spawn::testing::ignores_sigusr1_fn` (a `pub(super)` fixture in the
/// spawn test tree, not reachable from this module). The STOP clear
/// makes it race-immune: a SIGUSR1 that lands before SIG_IGN is
/// installed flips STOP via the default handler, but the clear undoes
/// it, so the worker keeps sleeping and only a SIGKILL terminates it.
fn ignores_sigusr1_spin(ctx: &crate::workload::WorkerCtx) -> crate::workload::WorkerReport {
    use std::sync::atomic::Ordering;
    let stop = ctx.stop();
    // SAFETY: runs in a freshly-forked, single-threaded worker child,
    // where `libc::signal` is async-signal-safe.
    unsafe {
        libc::signal(libc::SIGUSR1, libc::SIG_IGN);
    }
    stop.store(false, Ordering::Relaxed);
    let deadline = std::time::Instant::now() + Duration::from_secs(7);
    while !stop.load(Ordering::Relaxed) && std::time::Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(10));
    }
    crate::workload::WorkerReport::default()
}

/// Scheduler-death teardown is prompt: the during-hold crash path
/// SIGKILLs the step's workers BEFORE collecting them, so the
/// per-worker reap in `stop_and_collect` does not pay its 5s
/// cooperative-stop deadline.
///
/// Setup mirrors
/// `holdspec_loop_arm_exits_early_when_sched_dies_during_hold`: a dead
/// `sched_pid` (`pid_t::MAX`, above PID_MAX_LIMIT → `pidfd_open`
/// ESRCH) makes `hold_or_sched_died` report the scheduler dead on the
/// first hold, driving run_scenario's during-hold path. The workers
/// (`ignores_sigusr1_spin`) ignore SIGUSR1, so the cooperative stop
/// never lands; without the `sigkill_handles` pre-pass before
/// `collect_step`, the collect waits out the full 5s deadline (the
/// `sigkill_workers_makes_collect_prompt_for_sigusr1_ignoring_workers`
/// test pins that same shape at the WorkloadHandle layer). With the
/// pre-pass the workers are SIGKILLed up front and the reap returns
/// promptly.
///
/// The 2s bound is well under the 5s deadline and far above the
/// sub-second fast path (the workers sleep, so SIGKILL drops them
/// immediately). A regression that removes the during-hold
/// `sigkill_handles` call surfaces here as a ~5s teardown. This is the
/// path the scheduler-crash reproducers hit; the inter-step and final
/// paths apply the same `sigkill_handles` pre-pass to the backdrop
/// handles before `collect_backdrop`.
#[test]
fn during_hold_sched_death_sigkills_workers_before_collect() {
    let mock = MockCgroupOps::new();
    let topo = mock_topo();
    let mut ctx = mock_ctx(&mock, &topo);
    ctx.duration = Duration::from_secs(10);
    // Dead scheduler pid (above PID_MAX_LIMIT) → hold_or_sched_died
    // reports death on the first hold. SCHED_PID is the live read in
    // run_scenario; restore it on exit (panic-safe) so neighbor tests
    // keep their no-scheduler contract.
    ctx.sched_pid = Some(libc::pid_t::MAX);
    crate::vmm::rust_init::set_sched_pid(libc::pid_t::MAX);
    struct ResetSchedPid;
    impl Drop for ResetSchedPid {
        fn drop(&mut self) {
            crate::vmm::rust_init::set_sched_pid(0);
        }
    }
    let _reset = ResetSchedPid;
    // Op::spawn_host spawns real worker processes in the runner's own
    // cgroup (zero cgroup ops, so MockCgroupOps is untouched). The
    // workers ignore SIGUSR1, so only the sigkill pre-pass terminates
    // them promptly.
    let work = WorkSpec::default()
        .workers(6)
        .work_type(WorkType::custom("ignores_sigusr1", ignores_sigusr1_spin));
    let steps = vec![Step::new(
        vec![Op::spawn_host(work)],
        HoldSpec::Fixed(Duration::from_secs(1)),
    )];
    let start = std::time::Instant::now();
    let result =
        execute_steps(&ctx, steps).expect("execute_steps returns Ok even when the scheduler dies");
    let elapsed = start.elapsed();
    assert!(
        elapsed < Duration::from_secs(2),
        "during-hold scheduler-death teardown took {elapsed:?}; expected \
         < 2s. The SIG_IGN workers force stop_and_collect's ~5s \
         cooperative-stop deadline unless the during-hold path SIGKILLs \
         them before collecting (sigkill_handles pre-pass missing?).",
    );
    assert!(
        !result.is_pass(),
        "scheduler death during the hold must mark the scenario failed; \
         got passed=true: {:?}",
        result.outcomes,
    );
}

/// `hold_or_sched_died` aborts on the BPF err-exit latch (set at the
/// crash) WITHOUT waiting for the scheduler PROCESS to exit — the
/// fast-crash-detection path that keeps crash-repro tests from paying a
/// slow scheduler's process-exit latency (scx_lavd's process exit trails
/// the crash by ~1s: its userspace poll loop observes the disable, then
/// flushes its dump and exits). Forces the probe mirror to `Crashed` and
/// passes a LIVE pid (this test process, which
/// never exits during the test): a hold that only watched the pidfd
/// would block the full 30s, so a sub-2s return proves the latch poll
/// aborted it.
#[test]
fn hold_aborts_on_err_exit_latch_not_process_exit() {
    use crate::probe::process::{SchedExitKind, sched_exit_kind, set_probe_sched_exit_state};
    // The mirror is a process-global atomic; reset it (panic-safe) so a
    // forced `Crashed` does not leak into neighbor tests.
    struct ResetExitState;
    impl Drop for ResetExitState {
        fn drop(&mut self) {
            set_probe_sched_exit_state(SchedExitKind::Unknown);
        }
    }
    let _reset = ResetExitState;
    set_probe_sched_exit_state(SchedExitKind::Crashed);
    assert_eq!(
        sched_exit_kind(),
        SchedExitKind::Crashed,
        "setup: the probe mirror must read Crashed after the override",
    );
    // A live thread-group leader that does not exit during the test, so
    // any abort comes from the latch poll, not the pidfd backstop.
    let live_pid = unsafe { libc::getpid() };
    let start = std::time::Instant::now();
    let died = hold_or_sched_died(Duration::from_secs(30), Some(live_pid));
    let elapsed = start.elapsed();
    assert!(
        died,
        "hold_or_sched_died must report sched-died when the err-exit latch is Crashed",
    );
    assert!(
        elapsed < Duration::from_secs(2),
        "hold must abort on the latch within ~one poll interval, not wait for \
         the live process or the full 30s dur; took {elapsed:?}",
    );
}

/// The Loop arm's apply_ops error-propagation path: an
/// `apply_ops` Err on a Loop iteration (the arm's
/// `drain_on_err!(scenario, apply_ops(...))` call) exits the loop
/// via the `drain_on_err!` macro (defined at the top of `run_step`)
/// which
/// propagates the Err up through `run_step`. The outer caller
/// (`run_scenario`'s `if let Err(err) = step_res` block) converts the Err to
/// `Ok(AssertResult { passed: false, details: [...
/// DetailKind::Other ...] })` so a mid-scenario failure still
/// returns the merged prior-step results plus the error context
/// rather than an opaque Err.
///
/// Implementation: `MockCgroupOps::fail_call_at(2, "...")` fails
/// the third cgroup call. The cgroup call sequence is:
/// - Index 0: `Setup` (`run_scenario`'s `ctx.cgroups.setup(&required)`
///   call before any step runs)
/// - Index 1: Iteration 1's SetCpuset → Ok
/// - Index 2: Iteration 2's SetCpuset → Err (injected)
///
/// Expected post-state: exactly 2 SetCpuset calls (iter 1 ok +
/// iter 2 fail, no third iteration), result.passed=false,
/// DetailKind::Other detail containing the injected message.
/// A regression that allowed the loop to continue past the
/// failing iteration would surface as 3+ calls.
///
/// SCOPE NOTE: this test does NOT verify the
/// `scenario.drain_all_payloads()` side effect inside
/// `drain_on_err!` because the fixture has no live payloads.
/// The drain-on-err contract at the macro level is verified by
/// `apply_ops_error_does_not_lose_live_payload_handles`
/// (sibling test in this module — grep by name) which checks
/// `apply_ops` itself doesn't drain (so `execute_steps`'
/// `drain_on_err!` is responsible). A dedicated Loop-arm drain
/// test with a live payload fixture is a follow-up (see queue
/// task for "Loop-arm drain verification with live payload
/// fixture") because the test infrastructure requires a custom
/// PayloadHandle observer to distinguish drain-side `.kill()`
/// from `Drop`-side SIGKILL.
#[test]
fn holdspec_loop_arm_propagates_apply_ops_error() {
    let mock = MockCgroupOps::new();
    // Inject an error at the THIRD cgroup call (index 2). The
    // sequence is: Setup (index 0) + iter-1 SetCpuset (index 1,
    // Ok) + iter-2 SetCpuset (index 2, Err injected). See
    // `run_scenario`'s `ctx.cgroups.setup(&required)` call for the
    // Setup-first call.
    mock.fail_call_at(2, "injected SetCpuset error mid-iteration");
    let topo = mock_topo();
    let mut ctx = mock_ctx(&mock, &topo);
    // Long enough for >2 iterations at 30ms interval if the
    // loop incorrectly continued past the failing iteration.
    ctx.duration = Duration::from_millis(200);
    let steps = vec![Step::new(
        vec![Op::set_cpuset("err_drain_test", CpusetSpec::Llc(0))],
        HoldSpec::loop_at(Duration::from_millis(30)),
    )];
    let result = execute_steps(&ctx, steps).expect(
        "execute_steps converts step Err to Ok(passed=false) per \
             run_scenario's `if let Err(err) = step_res` block; the Err must NOT propagate to the caller",
    );
    assert!(
        !result.is_pass(),
        "injected apply_ops error must mark passed=false; got \
             passed=true with details: {:?}",
        result.outcomes,
    );
    let other_details: Vec<_> = result
        .failure_details()
        .filter(|d| {
            matches!(d.kind, crate::assert::DetailKind::Other)
                && d.message.contains("injected SetCpuset error mid-iteration")
        })
        .collect();
    assert_eq!(
        other_details.len(),
        1,
        "step Err must surface exactly once as DetailKind::Other \
             carrying the injected message; got {} matching details out \
             of {} total: {:?}",
        other_details.len(),
        result.outcomes.len(),
        result.outcomes,
    );
    let set_cpuset_calls = mock
        .calls()
        .iter()
        .filter(|c| matches!(c, CgroupCall::SetCpuset(name, _) if name == "err_drain_test"))
        .count();
    // 1 ok + 1 fail = 2. A third call would mean the loop body
    // ignored the Err and continued to the next interval —
    // a regression in either drain_on_err! propagation or the
    // run_step Loop arm's `?`-out behavior.
    assert_eq!(
        set_cpuset_calls, 2,
        "loop must stop at the failed iteration (1 ok + 1 fail = 2); \
             got {set_cpuset_calls} SetCpuset calls. Any value > 2 means \
             the apply_ops Err was swallowed and the Loop arm continued, \
             which would also bypass the drain_on_err! payload-kill path \
             (silent metric loss).",
    );
}

/// Loop-arm `drain_on_err!` invokes `.kill()` on every live
/// payload handle when `apply_ops` returns Err mid-iteration,
/// rather than letting `PayloadHandle::Drop` SIGKILL them via
/// the process-group fallback. The two paths differ in
/// observable behavior at `payload_run.rs::PayloadHandle::drop`:
/// `.kill()` calls `self.child.take()` before reaping, so by the
/// time Drop runs `self.child.is_none()` and the diagnostic
/// `eprintln!("ktstr: PayloadHandle for 'X' dropped without
/// wait/kill — process group SIGKILLed, metrics not recorded.")`
/// does NOT fire. If `drain_on_err!` regressed to a bare `?`-out
/// (or any path that doesn't drain), Drop would see
/// `self.child.is_some()`, fire the eprintln, and the captured
/// stderr would contain "dropped without wait/kill".
///
/// Pairs with `holdspec_loop_arm_propagates_apply_ops_error`:
/// that test verifies Err PROPAGATION via the macro (loop stops,
/// passed=false, exactly 2 SetCpuset calls). This test verifies
/// the SIDE EFFECT (kill-not-Drop) using a live `/bin/sleep`
/// fixture spawned by `Op::run_payload`. Both tests must pass —
/// drain_on_err! must both propagate Err AND drain payloads, and
/// covering only one half misses regressions in the other.
#[test]
fn holdspec_loop_arm_drain_on_err_kills_live_payload_via_kill_not_drop() {
    use crate::test_support::{OutputFormat, Payload, PayloadKind};
    static SLEEP: Payload = Payload {
        name: "drain_on_err_observer",
        kind: PayloadKind::Binary("/bin/sleep"),
        output: OutputFormat::ExitCode,
        default_args: &[],
        default_checks: &[],
        metrics: &[],
        include_files: &[],
        uses_parent_pgrp: false,
        known_flags: None,
        metric_bounds: None,
    };

    let mock = MockCgroupOps::new();
    // Index sequence (cgroup-op counts only):
    //   0 = run_scenario Setup (its `ctx.cgroups.setup(&required)` call)
    //   1 = iter-1 SetCpuset (Ok)
    //   2 = iter-2 SetCpuset (Err injected here)
    // Op::run_payload without an explicit cgroup arg does NOT go
    // through MockCgroupOps — the SLEEP child spawns directly,
    // so it's live in state.payload_handles when the iter-2
    // SetCpuset Err triggers drain_on_err!.
    mock.fail_call_at(2, "injected SetCpuset error to trigger drain_on_err");
    let topo = mock_topo();
    let mut ctx = mock_ctx(&mock, &topo);
    // Wide enough for iter-2 to reach the SetCpuset failure point
    // at the 30ms interval below; the Err short-circuits the
    // remaining iterations.
    ctx.duration = Duration::from_millis(200);

    let steps = vec![Step::new(
        vec![
            Op::run_payload(&SLEEP, ["3600"]),
            Op::set_cpuset("drain_observer_cg", CpusetSpec::Llc(0)),
        ],
        HoldSpec::loop_at(Duration::from_millis(30)),
    )];

    let (_, captured_stderr) = crate::test_support::test_helpers::capture_stderr(|| {
        let _ = execute_steps(&ctx, steps).expect(
            "execute_steps converts step Err to Ok(passed=false); the \
                     Err must NOT propagate to the caller",
        );
    });

    let stderr_text = String::from_utf8_lossy(&captured_stderr);
    assert!(
        !stderr_text.contains("dropped without wait/kill"),
        "drain_on_err! must invoke .kill() on every live payload \
             handle, not let them fall through to PayloadHandle::Drop's \
             process-group SIGKILL. Observed the Drop-path eprintln in \
             captured stderr — drain_on_err! regressed (Err propagated \
             but payloads were leaked to Drop). Full captured stderr: \
             {stderr_text:?}",
    );
}

// -- HoldSpec PartialEq (load-bearing semantic pins) --

/// Payload participates in equality across every variant. A
/// derived PartialEq guarantees this, but pinning it explicitly
/// catches a hypothetical hand-rolled `|_, _| true` regression
/// AND a partial-derive that ignores struct-variant field
/// contents.
#[test]
fn holdspec_partialeq_payload_participates_in_equality() {
    assert_ne!(HoldSpec::Frac(0.5), HoldSpec::Frac(0.75));
    assert_ne!(
        HoldSpec::Fixed(Duration::from_secs(1)),
        HoldSpec::Fixed(Duration::from_secs(2))
    );
    assert_ne!(
        HoldSpec::Loop {
            interval: Duration::from_millis(100)
        },
        HoldSpec::Loop {
            interval: Duration::from_millis(200)
        }
    );
}

/// IEEE 754: 0.1 + 0.2 != 0.3. PartialEq on Frac inherits strict
/// float equality so a Frac built from arithmetic does NOT
/// compare equal to a Frac with the rounded literal. Pins the
/// documented behavior so a future "fuzzy PartialEq" rewrite
/// doesn't silently change the contract.
#[test]
fn holdspec_partialeq_frac_float_strict_equality() {
    assert_ne!(HoldSpec::Frac(0.1 + 0.2), HoldSpec::Frac(0.3));
}

/// IEEE 754: NaN != NaN, even against itself. PartialEq on Frac
/// inherits the non-reflexive behavior. `HoldSpec::validate`
/// rejects Frac(NaN) at intake so production code paths don't
/// see this, but the type-level PartialEq contract must hold —
/// pinned against a future "treat NaN as reflexive" rewrite.
#[test]
fn holdspec_partialeq_frac_nan_self_unequal() {
    let nan = HoldSpec::Frac(f64::NAN);
    assert_ne!(nan, nan);
}

/// `FULL` is an alias for `Frac(1.0)`. The public-API const
/// shouldn't drift from the variant it expands to.
#[test]
fn holdspec_full_equals_frac_one() {
    assert_eq!(HoldSpec::FULL, HoldSpec::Frac(1.0));
}

// Compile-time proof the constructor signatures stay `const fn`.
// If a future refactor demotes any of these (e.g. by introducing
// a non-const call internally), the module fails to compile here
// — surfaces the regression at the layer where const usability
// matters. Discarded via `_` to avoid namespace pollution.
const _: HoldSpec = HoldSpec::fixed(Duration::from_secs(1));
const _: HoldSpec = HoldSpec::frac(0.5);
const _: HoldSpec = HoldSpec::loop_at(Duration::from_millis(50));

// -- CpusetSpec::Exact --

#[test]
fn cpusetspec_exact_is_passthrough() {
    let cpus: BTreeSet<usize> = [0, 2, 4].iter().copied().collect();
    let spec = CpusetSpec::Exact(cpus.clone());
    let topo = crate::topology::TestTopology::from_vm_topology(
        &crate::vmm::topology::Topology::new(1, 1, 4, 1),
    );
    let cgroups = crate::cgroup::CgroupManager::new("/nonexistent");
    let ctx = Ctx {
        cgroups: &cgroups,
        topo: &topo,
        duration: Duration::from_secs(10),
        workers_per_cgroup: 4,
        sched_pid: None,
        settle: Duration::from_millis(1000),
        work_type_override: None,
        assert: crate::assert::Assert::default_checks(),
        wait_for_map_write: false,
        current_step: std::sync::Arc::new(std::sync::atomic::AtomicU16::new(0)),
        entry_name: None,
    };
    let resolved = spec.resolve(&ctx);
    assert_eq!(resolved, cpus);
}

// -- Defense-in-depth: resolve must not panic on spec shapes that
// -- validate rejects. Each test exercises a concrete panic the
// -- resolver's hardening guards against.

#[test]
fn resolve_disjoint_of_zero_returns_empty_instead_of_panicking() {
    // `usable.len() / of` with of=0 would panic without hardening.
    // Current behavior: returns an empty BTreeSet with a
    // tracing::warn.
    let (cg, topo) = make_ctx(1, 4, 1);
    let ctx = ctx_from(&cg, &topo);
    let spec = CpusetSpec::Disjoint { index: 0, of: 0 };
    assert!(spec.resolve(&ctx).is_empty());
}

#[test]
fn resolve_overlap_of_zero_returns_empty_instead_of_panicking() {
    let (cg, topo) = make_ctx(1, 4, 1);
    let ctx = ctx_from(&cg, &topo);
    let spec = CpusetSpec::Overlap {
        index: 0,
        of: 0,
        frac: 0.5,
    };
    assert!(spec.resolve(&ctx).is_empty());
}

#[test]
fn resolve_range_inverted_fracs_returns_empty_instead_of_panicking() {
    // Without hardening, `usable[start.min(len)..end.min(len)]`
    // with start_frac > end_frac produced start > end after
    // clamping and panicked the slice operation. Current
    // behavior: the slice is clamped to length-zero instead.
    let (cg, topo) = make_ctx(1, 4, 1);
    let ctx = ctx_from(&cg, &topo);
    let spec = CpusetSpec::Range {
        start_frac: 0.8,
        end_frac: 0.2,
    };
    assert!(spec.resolve(&ctx).is_empty());
}

#[test]
fn resolve_range_nan_fracs_clamps_to_zero_instead_of_panicking() {
    // NaN as usize saturates to 0 on stable Rust, but inverted
    // start/end after both saturate is still fine post-fix.
    let (cg, topo) = make_ctx(1, 4, 1);
    let ctx = ctx_from(&cg, &topo);
    let spec = CpusetSpec::Range {
        start_frac: f64::NAN,
        end_frac: f64::NAN,
    };
    assert!(spec.resolve(&ctx).is_empty());
}

#[test]
fn resolve_overlap_nonfinite_frac_clamps_to_zero() {
    // NaN frac pre-fix flowed through `(chunk as f64 * frac) as
    // usize` and could produce an out-of-range overlap. Post-fix
    // clamps NaN to 0, yielding the same partition boundaries as
    // Disjoint.
    let (cg, topo) = make_ctx(1, 4, 1);
    let ctx = ctx_from(&cg, &topo);
    let spec = CpusetSpec::Overlap {
        index: 0,
        of: 2,
        frac: f64::NAN,
    };
    // No panic; result must be non-empty because index/of are valid.
    let result = spec.resolve(&ctx);
    assert!(!result.is_empty());
}

// -- CpusetSpec resolution helpers --

fn make_ctx(
    llcs: u32,
    cores: u32,
    threads: u32,
) -> (crate::cgroup::CgroupManager, crate::topology::TestTopology) {
    let cgroups = crate::cgroup::CgroupManager::new("/nonexistent");
    let topo = crate::topology::TestTopology::from_vm_topology(
        &crate::vmm::topology::Topology::new(1, llcs, cores, threads),
    );
    (cgroups, topo)
}

fn ctx_from<'a>(
    cgroups: &'a crate::cgroup::CgroupManager,
    topo: &'a crate::topology::TestTopology,
) -> Ctx<'a> {
    Ctx {
        cgroups,
        topo,
        duration: Duration::from_secs(10),
        workers_per_cgroup: 4,
        sched_pid: None,
        settle: Duration::ZERO,
        work_type_override: None,
        assert: crate::assert::Assert::default_checks(),
        wait_for_map_write: false,
        current_step: std::sync::Arc::new(std::sync::atomic::AtomicU16::new(0)),
        entry_name: None,
    }
}

// -- CpusetSpec::Disjoint --

#[test]
fn cpusetspec_disjoint_two_partitions() {
    let (cg, topo) = make_ctx(1, 8, 1);
    let ctx = ctx_from(&cg, &topo);
    let a = CpusetSpec::Disjoint { index: 0, of: 2 }.resolve(&ctx);
    let b = CpusetSpec::Disjoint { index: 1, of: 2 }.resolve(&ctx);
    // Partitions must be disjoint.
    assert!(a.is_disjoint(&b), "partitions overlap: {:?} vs {:?}", a, b);
    // Together they cover all usable CPUs.
    let usable = ctx.topo.usable_cpuset();
    let union: BTreeSet<usize> = a.union(&b).copied().collect();
    assert_eq!(union, usable);
}

#[test]
fn cpusetspec_disjoint_remainder_to_last() {
    // 8 CPUs, last reserved → 7 usable [0..6]. 7/3 = chunk 2, so
    // partition 0=[0,1], 1=[2,3], and the last partition (index 2)
    // gets `usable[start..usable.len()]` = [4,5,6] — chunk PLUS the
    // remainder CPU (6). Pin the exact set: a regression in the
    // last-partition branch that computed `end = (index+1)*chunk = 6`
    // instead of `usable.len() = 7` would drop CPU 6, yielding [4,5]
    // (len 2). An inequality `len >= chunk` would still accept that
    // bug; the exact-set assertion catches it.
    let (cg, topo) = make_ctx(1, 8, 1);
    let ctx = ctx_from(&cg, &topo);
    let usable = ctx.topo.usable_cpus();
    assert_eq!(
        usable,
        [0, 1, 2, 3, 4, 5, 6],
        "fixture assumption: 8 CPUs minus 1 reserved = 7 usable [0..6]"
    );
    let c = CpusetSpec::Disjoint { index: 2, of: 3 }.resolve(&ctx);
    let expected: BTreeSet<usize> = [4, 5, 6].into_iter().collect();
    assert_eq!(
        c, expected,
        "last partition must absorb the remainder CPU (the tail of \
         usable), got {c:?}"
    );
    // Explicitly pin that the tail usable CPU is in the last partition
    // — the property the remainder-to-last contract guarantees.
    assert!(
        c.contains(usable.last().unwrap()),
        "last partition must include the final usable CPU {}",
        usable.last().unwrap()
    );
}

#[test]
fn cpusetspec_disjoint_single_partition() {
    let (cg, topo) = make_ctx(1, 4, 1);
    let ctx = ctx_from(&cg, &topo);
    let all = CpusetSpec::Disjoint { index: 0, of: 1 }.resolve(&ctx);
    let usable = ctx.topo.usable_cpuset();
    assert_eq!(all, usable);
}

#[test]
fn cpusetspec_disjoint_index_beyond_of_returns_empty() {
    // Defense-in-depth: `validate` rejects index >= of with a clear
    // error, but callers that skip validation (e.g. programmatic
    // spec construction) must not hit the div-by-zero or panic in
    // `resolve`. With index = 5 and of = 3 on 3 usable CPUs
    // (4 total, 1 reserved by `usable_cpus`), chunk = 1 and
    // start = 5 clamps past `usable.len()` to yield an empty set
    // — a safe fallback, not a panic.
    let (cg, topo) = make_ctx(1, 4, 1);
    let ctx = ctx_from(&cg, &topo);
    let cpus = CpusetSpec::Disjoint { index: 5, of: 3 }.resolve(&ctx);
    assert!(
        cpus.is_empty(),
        "Disjoint with index beyond `of` must return an empty \
             cpuset rather than panicking, got: {cpus:?}",
    );
}

// -- CpusetSpec::Range --

#[test]
fn cpusetspec_range_first_half() {
    let (cg, topo) = make_ctx(1, 8, 1);
    let ctx = ctx_from(&cg, &topo);
    let cpus = CpusetSpec::Range {
        start_frac: 0.0,
        end_frac: 0.5,
    }
    .resolve(&ctx);
    let usable = ctx.topo.usable_cpus();
    let expected_len = usable.len() / 2;
    assert_eq!(cpus.len(), expected_len);
    // Should contain the first usable CPUs.
    for &cpu in &cpus {
        assert!(usable.contains(&cpu));
    }
}

#[test]
fn cpusetspec_range_second_half() {
    let (cg, topo) = make_ctx(1, 8, 1);
    let ctx = ctx_from(&cg, &topo);
    let a = CpusetSpec::Range {
        start_frac: 0.0,
        end_frac: 0.5,
    }
    .resolve(&ctx);
    let b = CpusetSpec::Range {
        start_frac: 0.5,
        end_frac: 1.0,
    }
    .resolve(&ctx);
    assert!(a.is_disjoint(&b));
}

#[test]
fn cpusetspec_range_full() {
    let (cg, topo) = make_ctx(1, 4, 1);
    let ctx = ctx_from(&cg, &topo);
    let cpus = CpusetSpec::Range {
        start_frac: 0.0,
        end_frac: 1.0,
    }
    .resolve(&ctx);
    let usable = ctx.topo.usable_cpuset();
    assert_eq!(cpus, usable);
}

#[test]
fn cpusetspec_range_empty() {
    let (cg, topo) = make_ctx(1, 4, 1);
    let ctx = ctx_from(&cg, &topo);
    let cpus = CpusetSpec::Range {
        start_frac: 0.5,
        end_frac: 0.5,
    }
    .resolve(&ctx);
    assert!(cpus.is_empty());
}

#[test]
fn cpusetspec_range_clamps_to_bounds() {
    let (cg, topo) = make_ctx(1, 4, 1);
    let ctx = ctx_from(&cg, &topo);
    // end_frac > 1.0 should be clamped to usable.len().
    let cpus = CpusetSpec::Range {
        start_frac: 0.0,
        end_frac: 2.0,
    }
    .resolve(&ctx);
    let usable = ctx.topo.usable_cpuset();
    assert_eq!(cpus, usable);
}

// -- CpusetSpec::Overlap --

#[test]
fn cpusetspec_overlap_neighbors_share_cpus() {
    let (cg, topo) = make_ctx(1, 8, 1);
    let ctx = ctx_from(&cg, &topo);
    let a = CpusetSpec::Overlap {
        index: 0,
        of: 2,
        frac: 0.5,
    }
    .resolve(&ctx);
    let b = CpusetSpec::Overlap {
        index: 1,
        of: 2,
        frac: 0.5,
    }
    .resolve(&ctx);
    let shared: BTreeSet<usize> = a.intersection(&b).copied().collect();
    assert!(!shared.is_empty(), "overlap=0.5 should produce shared CPUs");
}

#[test]
fn cpusetspec_overlap_zero_frac_is_disjoint() {
    let (cg, topo) = make_ctx(1, 8, 1);
    let ctx = ctx_from(&cg, &topo);
    let a = CpusetSpec::Overlap {
        index: 0,
        of: 2,
        frac: 0.0,
    }
    .resolve(&ctx);
    let b = CpusetSpec::Overlap {
        index: 1,
        of: 2,
        frac: 0.0,
    }
    .resolve(&ctx);
    assert!(a.is_disjoint(&b), "frac=0 should be disjoint");
}

#[test]
fn cpusetspec_overlap_last_partition_covers_tail() {
    let (cg, topo) = make_ctx(1, 8, 1);
    let ctx = ctx_from(&cg, &topo);
    let last = CpusetSpec::Overlap {
        index: 2,
        of: 3,
        frac: 0.5,
    }
    .resolve(&ctx);
    let usable = ctx.topo.usable_cpus();
    // Last partition should include the last usable CPU.
    assert!(last.contains(usable.last().unwrap()));
}

#[test]
fn cpusetspec_overlap_first_partition_starts_at_zero() {
    let (cg, topo) = make_ctx(1, 8, 1);
    let ctx = ctx_from(&cg, &topo);
    let first = CpusetSpec::Overlap {
        index: 0,
        of: 3,
        frac: 0.5,
    }
    .resolve(&ctx);
    let usable = ctx.topo.usable_cpus();
    assert!(first.contains(&usable[0]));
}

// -- CpusetSpec::Llc --

#[test]
fn cpusetspec_llc_index_zero() {
    let (cg, topo) = make_ctx(2, 4, 1);
    let ctx = ctx_from(&cg, &topo);
    let cpus = CpusetSpec::Llc(0).resolve(&ctx);
    assert!(!cpus.is_empty());
    // All CPUs in the set should belong to LLC 0.
    let llc0 = ctx.topo.llc_aligned_cpuset(0);
    assert_eq!(cpus, llc0);
}

#[test]
fn cpusetspec_llc_two_llcs_disjoint() {
    let (cg, topo) = make_ctx(2, 4, 1);
    let ctx = ctx_from(&cg, &topo);
    let llc0 = CpusetSpec::Llc(0).resolve(&ctx);
    let llc1 = CpusetSpec::Llc(1).resolve(&ctx);
    assert!(llc0.is_disjoint(&llc1), "LLCs should be disjoint");
}

// -- CpusetSpec::Numa --

fn make_numa_ctx(
    numa_nodes: u32,
    llcs: u32,
    cores: u32,
    threads: u32,
) -> (crate::cgroup::CgroupManager, crate::topology::TestTopology) {
    let cgroups = crate::cgroup::CgroupManager::new("/nonexistent");
    let topo = crate::topology::TestTopology::from_vm_topology(
        &crate::vmm::topology::Topology::new(numa_nodes, llcs, cores, threads),
    );
    (cgroups, topo)
}

#[test]
fn cpusetspec_numa_node_zero() {
    // 2 NUMA nodes, 4 LLCs (2 per NUMA), 4 cores, 1 thread
    // LLCs 0,1 -> NUMA 0 (CPUs 0-7), LLCs 2,3 -> NUMA 1 (CPUs 8-15)
    let (cg, topo) = make_numa_ctx(2, 4, 4, 1);
    let ctx = ctx_from(&cg, &topo);
    let cpus = CpusetSpec::Numa(0).resolve(&ctx);
    let expected: BTreeSet<usize> = (0..8).collect();
    assert_eq!(cpus, expected);
}

#[test]
fn cpusetspec_numa_node_one() {
    let (cg, topo) = make_numa_ctx(2, 4, 4, 1);
    let ctx = ctx_from(&cg, &topo);
    let cpus = CpusetSpec::Numa(1).resolve(&ctx);
    let expected: BTreeSet<usize> = (8..16).collect();
    assert_eq!(cpus, expected);
}

#[test]
fn cpusetspec_numa_disjoint() {
    let (cg, topo) = make_numa_ctx(2, 4, 4, 1);
    let ctx = ctx_from(&cg, &topo);
    let node0 = CpusetSpec::Numa(0).resolve(&ctx);
    let node1 = CpusetSpec::Numa(1).resolve(&ctx);
    assert!(
        node0.is_disjoint(&node1),
        "NUMA nodes should be disjoint: {:?} vs {:?}",
        node0,
        node1
    );
    let union: BTreeSet<usize> = node0.union(&node1).copied().collect();
    assert_eq!(union, ctx.topo.all_cpuset());
}

#[test]
fn cpusetspec_numa_single_node_returns_all() {
    let (cg, topo) = make_numa_ctx(1, 2, 4, 1);
    let ctx = ctx_from(&cg, &topo);
    let cpus = CpusetSpec::Numa(0).resolve(&ctx);
    assert_eq!(cpus, ctx.topo.all_cpuset());
}

#[test]
fn cpusetspec_numa_validate_out_of_range() {
    let (cg, topo) = make_numa_ctx(2, 4, 4, 1);
    let ctx = ctx_from(&cg, &topo);
    let spec = CpusetSpec::Numa(5);
    let err = spec.validate(&ctx).unwrap_err();
    assert!(err.contains("out of range"), "got: {err}");
}

#[test]
fn cpusetspec_numa_validate_valid() {
    let (cg, topo) = make_numa_ctx(2, 4, 4, 1);
    let ctx = ctx_from(&cg, &topo);
    assert!(CpusetSpec::Numa(0).validate(&ctx).is_ok());
    assert!(CpusetSpec::Numa(1).validate(&ctx).is_ok());
}

#[test]
fn cpusetspec_numa_convenience_constructor() {
    let spec = CpusetSpec::numa(0);
    assert!(matches!(spec, CpusetSpec::Numa(0)));
}

// -- Traverse::generate --

#[test]
fn traverse_generate_produces_steps() {
    let (cg, topo) = make_ctx(1, 8, 1);
    let ctx = ctx_from(&cg, &topo);
    let t = Traverse {
        seed: Some(42),
        cgroup_count: 2..=4,
        layouts: vec![Layout::Disjoint],
        phases: 3,
        phase_duration: Duration::from_millis(100),
        settle: Duration::from_millis(50),
        persistent_cgroups: 0,
        cgroup_workloads: vec![WorkSpec::default()],
    };
    let steps = t.generate(&ctx);
    assert_eq!(steps.len(), 3);
    for step in &steps {
        assert!(!step.ops.is_empty(), "each phase should have ops");
    }
}

#[test]
fn traverse_generate_deterministic() {
    let (cg, topo) = make_ctx(1, 8, 1);
    let ctx = ctx_from(&cg, &topo);
    let t = Traverse {
        seed: Some(99),
        cgroup_count: 2..=4,
        layouts: vec![Layout::Disjoint, Layout::Overlap(0.2, 0.5)],
        phases: 5,
        phase_duration: Duration::from_millis(100),
        settle: Duration::from_millis(50),
        persistent_cgroups: 1,
        cgroup_workloads: vec![WorkSpec::default()],
    };
    let steps1 = t.generate(&ctx);
    let steps2 = t.generate(&ctx);
    assert_eq!(steps1.len(), steps2.len());
    for (s1, s2) in steps1.iter().zip(steps2.iter()) {
        assert_eq!(
            s1.ops.len(),
            s2.ops.len(),
            "deterministic seed should produce same ops"
        );
    }
}

#[test]
fn traverse_generate_persistent_cgroups_preserved() {
    let (cg, topo) = make_ctx(1, 8, 1);
    let ctx = ctx_from(&cg, &topo);
    let t = Traverse {
        seed: Some(42),
        cgroup_count: 1..=4,
        layouts: vec![Layout::Disjoint],
        phases: 5,
        phase_duration: Duration::from_millis(100),
        settle: Duration::from_millis(50),
        persistent_cgroups: 2,
        cgroup_workloads: vec![WorkSpec::default()],
    };
    let steps = t.generate(&ctx);
    // Every phase should have at least persistent_cgroups worth of SetCpuset ops
    // (cg_0, cg_1 are never removed).
    for step in &steps {
        let remove_ops: Vec<&Op> = step.ops.iter()
                .filter(|op| matches!(op, Op::RemoveCgroup { cgroup } if cgroup == "cg_0" || cgroup == "cg_1"))
                .collect();
        assert!(
            remove_ops.is_empty(),
            "persistent cgroups should never be removed"
        );
    }
}

// -- CgroupDef builder --

#[test]
fn cgroup_def_builder_chain() {
    let d = CgroupDef::named("test")
        .cpuset(CpusetSpec::llc(0))
        .workers(8)
        .work_type(WorkType::bursty(
            Duration::from_millis(50),
            Duration::from_millis(100),
        ))
        .sched_policy(crate::workload::SchedPolicy::Batch)
        .swappable(true);
    assert_eq!(d.name, "test");
    assert!(d.cpuset.is_some());
    assert_eq!(d.works.len(), 1);
    assert_eq!(d.works[0].num_workers, Some(8));
    assert!(d.swappable);
}

#[test]
fn cgroup_def_multi_work() {
    let d = CgroupDef::named("multi")
        .work(WorkSpec::default().workers(4).work_type(WorkType::SpinWait))
        .work(
            WorkSpec::default()
                .workers(2)
                .work_type(WorkType::YieldHeavy),
        );
    assert_eq!(d.works.len(), 2);
    assert_eq!(d.works[0].num_workers, Some(4));
    assert_eq!(d.works[1].num_workers, Some(2));
}

#[test]
fn cgroup_def_old_api_then_work() {
    let d = CgroupDef::named("mixed")
        .workers(4)
        .work(WorkSpec::default().workers(2));
    assert_eq!(d.works.len(), 2);
    assert_eq!(d.works[0].num_workers, Some(4));
    assert_eq!(d.works[1].num_workers, Some(2));
}

#[test]
fn cgroup_def_work_only_no_phantom() {
    let d = CgroupDef::named("explicit").work(WorkSpec::default().workers(3));
    assert_eq!(d.works.len(), 1);
    assert_eq!(d.works[0].num_workers, Some(3));
}

// -- Setup --

#[test]
fn setup_defs_resolves() {
    let defs = vec![CgroupDef::named("a"), CgroupDef::named("b")];
    let setup = Setup::Defs(defs);
    let (cg, topo) = make_ctx(1, 4, 1);
    let ctx = ctx_from(&cg, &topo);
    let resolved = setup.resolve(&ctx);
    assert_eq!(resolved.len(), 2);
    assert!(!setup.is_empty());
}

#[test]
fn setup_defs_empty() {
    let setup = Setup::Defs(vec![]);
    assert!(setup.is_empty());
}

#[test]
fn setup_factory_not_empty() {
    let setup = Setup::Factory(|_| vec![CgroupDef::named("generated")]);
    assert!(!setup.is_empty());
}

// -- Step::with_defs / with_ops --

#[test]
fn step_with_defs_empty() {
    let step = Step::with_defs(vec![], HoldSpec::frac(0.5));
    assert!(step.setup.is_empty());
    assert!(step.ops.is_empty());
}

#[test]
fn step_with_defs_populated() {
    let step = Step::with_defs(
        vec![CgroupDef::named("cg_0"), CgroupDef::named("cg_1")],
        HoldSpec::fixed(Duration::from_secs(5)),
    );
    assert!(!step.setup.is_empty());
    assert!(step.ops.is_empty());
}

#[test]
fn step_with_defs_then_ops() {
    let step = Step::with_defs(vec![CgroupDef::named("cg_0")], HoldSpec::FULL).set_ops(vec![
        Op::AddCgroup {
            name: "cg_1".into(),
        },
    ]);
    assert!(!step.setup.is_empty());
    assert_eq!(step.ops.len(), 1);
}

#[test]
fn step_set_ops_replaces() {
    let step = Step::new(
        vec![Op::AddCgroup { name: "a".into() }],
        HoldSpec::frac(0.5),
    )
    .set_ops(vec![
        Op::AddCgroup { name: "b".into() },
        Op::RemoveCgroup { cgroup: "c".into() },
    ]);
    assert_eq!(step.ops.len(), 2);
}

// -- CpusetSpec::validate --

#[test]
fn cpusetspec_validate_disjoint_of_zero() {
    let (cg, topo) = make_ctx(1, 4, 1);
    let ctx = ctx_from(&cg, &topo);
    let spec = CpusetSpec::Disjoint { index: 0, of: 0 };
    let err = spec.validate(&ctx).unwrap_err();
    assert!(err.contains("must be > 0"), "got: {err}");
}

#[test]
fn cpusetspec_validate_disjoint_index_ge_of() {
    let (cg, topo) = make_ctx(1, 4, 1);
    let ctx = ctx_from(&cg, &topo);
    let spec = CpusetSpec::Disjoint { index: 3, of: 3 };
    let err = spec.validate(&ctx).unwrap_err();
    assert!(err.contains("index 3 >= partition count 3"), "got: {err}");
}

#[test]
fn cpusetspec_validate_overlap_of_zero() {
    let (cg, topo) = make_ctx(1, 4, 1);
    let ctx = ctx_from(&cg, &topo);
    let spec = CpusetSpec::Overlap {
        index: 0,
        of: 0,
        frac: 0.5,
    };
    let err = spec.validate(&ctx).unwrap_err();
    assert!(err.contains("must be > 0"), "got: {err}");
}

#[test]
fn cpusetspec_validate_overlap_index_ge_of() {
    let (cg, topo) = make_ctx(1, 4, 1);
    let ctx = ctx_from(&cg, &topo);
    let spec = CpusetSpec::Overlap {
        index: 2,
        of: 2,
        frac: 0.5,
    };
    let err = spec.validate(&ctx).unwrap_err();
    assert!(err.contains("index 2 >= partition count 2"), "got: {err}");
}

#[test]
fn cpusetspec_validate_range_start_ge_end() {
    let (cg, topo) = make_ctx(1, 4, 1);
    let ctx = ctx_from(&cg, &topo);
    let spec = CpusetSpec::Range {
        start_frac: 0.8,
        end_frac: 0.2,
    };
    let err = spec.validate(&ctx).unwrap_err();
    assert!(err.contains("start_frac"), "got: {err}");
}

#[test]
fn cpusetspec_validate_range_rejects_nan() {
    // Regression: IEEE 754 comparisons with NaN always return false, so
    // `start_frac >= end_frac` failed to reject it. validate() now
    // rejects non-finite fracs explicitly.
    let (cg, topo) = make_ctx(1, 4, 1);
    let ctx = ctx_from(&cg, &topo);
    let spec = CpusetSpec::Range {
        start_frac: 0.8,
        end_frac: f64::NAN,
    };
    let err = spec.validate(&ctx).unwrap_err();
    assert!(err.contains("not finite"), "got: {err}");
}

#[test]
fn cpusetspec_validate_range_rejects_infinity() {
    let (cg, topo) = make_ctx(1, 4, 1);
    let ctx = ctx_from(&cg, &topo);
    let spec = CpusetSpec::Range {
        start_frac: 0.0,
        end_frac: f64::INFINITY,
    };
    let err = spec.validate(&ctx).unwrap_err();
    assert!(err.contains("not finite"), "got: {err}");
}

#[test]
fn cpusetspec_validate_range_rejects_negative() {
    let (cg, topo) = make_ctx(1, 4, 1);
    let ctx = ctx_from(&cg, &topo);
    let spec = CpusetSpec::Range {
        start_frac: -0.5,
        end_frac: 0.5,
    };
    let err = spec.validate(&ctx).unwrap_err();
    assert!(err.contains("[0.0, 1.0]"), "got: {err}");
}

#[test]
fn cpusetspec_validate_range_rejects_above_one() {
    let (cg, topo) = make_ctx(1, 4, 1);
    let ctx = ctx_from(&cg, &topo);
    let spec = CpusetSpec::Range {
        start_frac: 0.5,
        end_frac: 1.5,
    };
    let err = spec.validate(&ctx).unwrap_err();
    assert!(err.contains("[0.0, 1.0]"), "got: {err}");
}

#[test]
fn cpusetspec_validate_overlap_rejects_nan_frac() {
    let (cg, topo) = make_ctx(1, 4, 1);
    let ctx = ctx_from(&cg, &topo);
    let spec = CpusetSpec::Overlap {
        index: 0,
        of: 2,
        frac: f64::NAN,
    };
    let err = spec.validate(&ctx).unwrap_err();
    assert!(err.contains("not finite"), "got: {err}");
}

#[test]
fn cpusetspec_validate_overlap_rejects_infinity_frac() {
    let (cg, topo) = make_ctx(1, 4, 1);
    let ctx = ctx_from(&cg, &topo);
    let spec = CpusetSpec::Overlap {
        index: 0,
        of: 2,
        frac: f64::INFINITY,
    };
    let err = spec.validate(&ctx).unwrap_err();
    assert!(err.contains("not finite"), "got: {err}");
}

#[test]
fn cpusetspec_validate_overlap_rejects_out_of_range_frac() {
    let (cg, topo) = make_ctx(1, 4, 1);
    let ctx = ctx_from(&cg, &topo);
    let spec = CpusetSpec::Overlap {
        index: 0,
        of: 2,
        frac: 1.5,
    };
    let err = spec.validate(&ctx).unwrap_err();
    assert!(err.contains("[0.0, 1.0]"), "got: {err}");
}

#[test]
fn cpusetspec_validate_too_few_cpus_for_partitions() {
    // 1 LLC, 2 cores, 1 thread => 2 total cpus, 2 usable
    let (cg, topo) = make_ctx(1, 2, 1);
    let ctx = ctx_from(&cg, &topo);
    let spec = CpusetSpec::Disjoint { index: 0, of: 5 };
    let err = spec.validate(&ctx).unwrap_err();
    assert!(err.contains("not enough usable CPUs"), "got: {err}");
}

#[test]
fn cpusetspec_validate_exact_in_range_ok() {
    // 1 LLC * 4 cores * 1 thread = CPUs 0..=3 physically present.
    let (cg, topo) = make_ctx(1, 4, 1);
    let ctx = ctx_from(&cg, &topo);
    let spec = CpusetSpec::exact([0, 2]);
    assert!(spec.validate(&ctx).is_ok());
}

#[test]
fn cpusetspec_validate_exact_empty_rejected() {
    let (cg, topo) = make_ctx(1, 4, 1);
    let ctx = ctx_from(&cg, &topo);
    let spec = CpusetSpec::Exact(BTreeSet::new());
    let err = spec.validate(&ctx).unwrap_err();
    assert!(err.contains("Exact") && err.contains("empty"), "got: {err}");
}

#[test]
fn cpusetspec_validate_exact_out_of_range_rejected() {
    // Topology has CPUs 0..=3; 99 is not physically present.
    let (cg, topo) = make_ctx(1, 4, 1);
    let ctx = ctx_from(&cg, &topo);
    let spec = CpusetSpec::exact([99]);
    let err = spec.validate(&ctx).unwrap_err();
    assert!(
        err.contains("99") && err.contains("physical CPU set"),
        "error must name the offending CPU and call it physical: {err}"
    );
}

/// Regression: the reserved last CPU (when `total_cpus > 2`,
/// `usable_cpus` drops the last one to leave the root cgroup a
/// home) is still PHYSICALLY present. A scheduler author pinning
/// a cgroup to that CPU for testing is legitimate — validate
/// must NOT reject on `usable_cpuset` membership. Accepting it
/// here is the contract that lets isolated-CPU tests compile.
#[test]
fn cpusetspec_validate_exact_accepts_reserved_last_cpu() {
    let (cg, topo) = make_ctx(1, 4, 1);
    let ctx = ctx_from(&cg, &topo);
    let total = ctx.topo.all_cpus().len();
    assert!(total > 2, "test requires a topology that reserves a CPU");
    let reserved_cpu = total - 1;
    assert!(
        !ctx.topo.usable_cpuset().contains(&reserved_cpu),
        "precondition: reserved CPU {reserved_cpu} must sit outside usable_cpuset",
    );
    assert!(
        ctx.topo.all_cpuset().contains(&reserved_cpu),
        "precondition: reserved CPU {reserved_cpu} must be physically present",
    );
    let spec = CpusetSpec::exact([reserved_cpu]);
    assert!(
        spec.validate(&ctx).is_ok(),
        "validate must accept the reserved CPU — physical presence, not \
             usable-set membership, is the bar",
    );
}

/// Regression guard for the HoldSpec pre-loop validation:
/// execute_steps_with must bail on a vacuous hold BEFORE running
/// any op. Failure mode without the pre-loop check: ops mutate
/// cgroup state, then `Duration::from_secs_f64` / `thread::sleep`
/// hit the downstream panic, leaving orphan cgroups on disk.
#[test]
fn execute_steps_with_bails_on_invalid_hold_before_ops() {
    let parent = std::env::temp_dir().join(format!("ktstr-hold-validate-{}", std::process::id()));
    // Pre-clean in case a prior failing test left a directory.
    let _ = std::fs::remove_dir_all(&parent);
    std::fs::create_dir_all(&parent).unwrap();
    let cgroups = crate::cgroup::CgroupManager::new(parent.to_str().unwrap());
    let topo = crate::topology::TestTopology::from_vm_topology(
        &crate::vmm::topology::Topology::new(1, 1, 4, 1),
    );
    let ctx = ctx_from(&cgroups, &topo);
    let cg_name = "should_never_exist";
    let step = Step::new(vec![Op::add_cgroup(cg_name)], HoldSpec::Frac(0.0));
    let err = execute_steps_with(&ctx, vec![step], None).unwrap_err();
    let msg = format!("{err:#}");
    assert!(
        msg.contains("hold validation") && msg.contains("Frac"),
        "error must cite hold validation + variant: {msg}"
    );
    assert!(
        !parent.join(cg_name).exists(),
        "AddCgroup op ran before hold validation — cgroup dir '{}' exists",
        parent.join(cg_name).display()
    );
    let _ = std::fs::remove_dir_all(&parent);
}

/// The SetAffinity dispatcher resolves its intent via the real
/// `crate::scenario::resolve_affinity_for_cgroup` (see
/// `Op::SetAffinity` in dispatch.rs) and the resulting
/// `ResolvedAffinity::Random` is consumed by the spawn-pipeline
/// guard `crate::workload::resolve_affinity`. Both production
/// functions reject the no-op conditions the test name names —
/// empty pool and `count == 0` — because an empty
/// `sched_setaffinity` mask is rejected by the kernel with EINVAL
/// (no-silent-drops invariant). This drives BOTH real guards
/// rather than a re-declared copy of the `!from.is_empty() &&
/// count > 0` classification, so flipping `||` to `&&`, dropping a
/// side of the condition, or removing the bail in either
/// production function fails this test.
#[test]
fn set_affinity_random_no_op_conditions() {
    use crate::workload::ResolvedAffinity;

    let (cg, topo) = make_ctx(1, 8, 1);
    let ctx = ctx_from(&cg, &topo);
    let pool: BTreeSet<usize> = [0, 1, 2].into_iter().collect();

    // -- Upstream resolver (the function dispatch.rs calls at
    // Op::SetAffinity) --

    // Valid pool + count>0 resolves to Random with the pool intact.
    let resolved = crate::scenario::resolve_affinity_for_cgroup(
        &AffinityIntent::random_subset(pool.iter().copied(), 2),
        None,
        ctx.topo,
    )
    .expect("non-empty pool + count>0 must resolve");
    match &resolved {
        ResolvedAffinity::Random { from, count } => {
            assert_eq!(*from, pool, "resolved pool must equal the intent pool");
            assert_eq!(*count, 2, "resolved count must equal the intent count");
        }
        other => panic!("expected ResolvedAffinity::Random, got {other:?}"),
    }

    // count == 0 bails before any allocation.
    let err = crate::scenario::resolve_affinity_for_cgroup(
        &AffinityIntent::random_subset(pool.iter().copied(), 0),
        None,
        ctx.topo,
    )
    .expect_err("count=0 must bail (no-op condition)");
    assert!(
        format!("{err:#}").contains("count=0"),
        "count=0 diagnostic expected, got: {err:#}"
    );

    // Empty pool with count>0 bails (no CPU to sample).
    let err = crate::scenario::resolve_affinity_for_cgroup(
        &AffinityIntent::random_subset(std::iter::empty::<usize>(), 1),
        None,
        ctx.topo,
    )
    .expect_err("empty pool must bail (no-op condition)");
    assert!(
        format!("{err:#}").contains("empty `from` pool"),
        "empty-pool diagnostic expected, got: {err:#}"
    );

    // -- Downstream consumer guard (the spawn pipeline's
    // ResolvedAffinity::Random consumer) — drives the SAME no-op
    // classification that dispatch.rs's per-worker Random arm
    // unreachable!()s on if it is ever reached. --

    // Valid Random samples exactly `count` CPUs, all from the pool.
    let sampled = crate::workload::resolve_affinity(&ResolvedAffinity::random([0, 1, 2], 2))
        .expect("valid Random must resolve")
        .expect("Random yields a concrete CPU set");
    assert_eq!(sampled.len(), 2, "must sample exactly count CPUs");
    assert!(
        sampled.is_subset(&pool),
        "sampled CPUs {sampled:?} must come from the pool {pool:?}"
    );

    // count == 0 bails.
    assert!(
        crate::workload::resolve_affinity(&ResolvedAffinity::random([0, 1, 2], 0)).is_err(),
        "count=0 Random must bail in the consumer guard"
    );
    // Empty pool with count>0 bails.
    assert!(
        crate::workload::resolve_affinity(&ResolvedAffinity::random(
            std::iter::empty::<usize>(),
            1
        ))
        .is_err(),
        "empty-pool Random must bail in the consumer guard"
    );
}

#[test]
fn cpusetspec_validate_llc_out_of_range() {
    let (cg, topo) = make_ctx(1, 4, 1);
    let ctx = ctx_from(&cg, &topo);
    let spec = CpusetSpec::Llc(5);
    let err = spec.validate(&ctx).unwrap_err();
    assert!(err.contains("out of range"), "got: {err}");
}

#[test]
fn cpusetspec_validate_valid_disjoint_ok() {
    let (cg, topo) = make_ctx(1, 8, 1);
    let ctx = ctx_from(&cg, &topo);
    let spec = CpusetSpec::Disjoint { index: 1, of: 2 };
    assert!(spec.validate(&ctx).is_ok());
}

// -- MemPolicy + cpuset validation tests --

#[test]
fn validate_mempolicy_default_always_ok() {
    // 2 NUMA nodes, 2 LLCs (1 per node), 4 cores, 1 thread = 8 CPUs
    let (cg, topo) = make_numa_ctx(2, 2, 4, 1);
    let ctx = ctx_from(&cg, &topo);
    let cpuset: BTreeSet<usize> = (0..4).collect();
    assert!(
        validate_mempolicy_cpuset(
            &MemPolicy::Default,
            crate::workload::MpolFlags::NONE,
            &cpuset,
            &ctx,
            "cg_0",
        )
        .is_ok()
    );
}

#[test]
fn validate_mempolicy_local_always_ok() {
    let (cg, topo) = make_numa_ctx(2, 2, 4, 1);
    let ctx = ctx_from(&cg, &topo);
    let cpuset: BTreeSet<usize> = (0..4).collect();
    assert!(
        validate_mempolicy_cpuset(
            &MemPolicy::Local,
            crate::workload::MpolFlags::NONE,
            &cpuset,
            &ctx,
            "cg_0",
        )
        .is_ok()
    );
}

#[test]
fn validate_mempolicy_bind_covered() {
    // 2 NUMA nodes, 2 LLCs, 4 cores each = 8 CPUs total
    // LLC 0 (CPUs 0-3) = NUMA 0, LLC 1 (CPUs 4-7) = NUMA 1
    let (cg, topo) = make_numa_ctx(2, 2, 4, 1);
    let ctx = ctx_from(&cg, &topo);
    let cpuset: BTreeSet<usize> = (0..8).collect(); // covers both nodes
    let policy = MemPolicy::Bind([0, 1].into_iter().collect());
    assert!(
        validate_mempolicy_cpuset(
            &policy,
            crate::workload::MpolFlags::NONE,
            &cpuset,
            &ctx,
            "cg_0",
        )
        .is_ok()
    );
}

#[test]
fn validate_mempolicy_bind_uncovered() {
    let (cg, topo) = make_numa_ctx(2, 2, 4, 1);
    let ctx = ctx_from(&cg, &topo);
    let cpuset: BTreeSet<usize> = (0..4).collect(); // NUMA node 0 only
    let policy = MemPolicy::Bind([1].into_iter().collect()); // node 1 not in cpuset
    let err = validate_mempolicy_cpuset(
        &policy,
        crate::workload::MpolFlags::NONE,
        &cpuset,
        &ctx,
        "cg_bind_test",
    )
    .unwrap_err()
    .to_string();
    // Cgroup name must appear so multi-cgroup scenarios can
    // triage which entry triggered the bail.
    assert!(err.contains("cg_bind_test"), "bail must name cgroup: {err}");
    // Uncovered node (1) and the covering cpuset node (0) must
    // both appear so the reader sees the exact disjoint pair.
    assert!(
        err.contains("[1]"),
        "bail must name uncovered node 1: {err}"
    );
    assert!(err.contains("{0}"), "bail must name cpuset node 0: {err}");
    // Both escape hatches must surface — pin the enumerated
    // `(a)` / `(b)` markers so a regression that collapses them
    // into one option trips this test before a user sees a
    // vague diagnostic.
    assert!(
        err.contains("(a) add .mpol_flags(MpolFlags::STATIC_NODES)"),
        "bail must call out hatch (a) STATIC_NODES opt-in by name: {err}",
    );
    assert!(
        err.contains("(b) widen the cpuset"),
        "bail must call out hatch (b) cpuset widening: {err}",
    );
    assert!(
        err.contains("CpusetSpec::Numa(N)"),
        "bail must name CpusetSpec::Numa(N) as a widening example: {err}",
    );
    assert!(
        err.contains("CpusetSpec::Exact"),
        "bail must name the CpusetSpec::Exact cpuset-widening escape hatch: {err}",
    );
    // The mismatch framing ("cross-socket allocation traffic
    // that is almost certainly unintended") must survive doc
    // edits — it's what makes the bail actionable for an
    // author who wrote the policy assuming the kernel would
    // silently intersect.
    assert!(
        err.contains("cross-socket allocation traffic"),
        "bail must name the cross-socket framing: {err}",
    );
    assert!(
        err.contains("almost certainly unintended"),
        "bail must frame the mismatch as unintended: {err}",
    );
}

#[test]
fn validate_mempolicy_preferred_covered() {
    let (cg, topo) = make_numa_ctx(2, 2, 4, 1);
    let ctx = ctx_from(&cg, &topo);
    let cpuset: BTreeSet<usize> = (4..8).collect(); // NUMA node 1
    let policy = MemPolicy::Preferred(1);
    assert!(
        validate_mempolicy_cpuset(
            &policy,
            crate::workload::MpolFlags::NONE,
            &cpuset,
            &ctx,
            "cg_0",
        )
        .is_ok()
    );
}

#[test]
fn validate_mempolicy_preferred_uncovered() {
    let (cg, topo) = make_numa_ctx(2, 2, 4, 1);
    let ctx = ctx_from(&cg, &topo);
    let cpuset: BTreeSet<usize> = (0..4).collect(); // NUMA node 0 only
    let policy = MemPolicy::Preferred(1);
    let err = validate_mempolicy_cpuset(
        &policy,
        crate::workload::MpolFlags::NONE,
        &cpuset,
        &ctx,
        "cg_preferred_test",
    )
    .unwrap_err()
    .to_string();
    assert!(
        err.contains("cg_preferred_test"),
        "bail must name cgroup: {err}"
    );
    assert!(
        err.contains("[1]"),
        "bail must name uncovered node 1: {err}"
    );
    assert!(err.contains("{0}"), "bail must name cpuset node 0: {err}");
    assert!(
        err.contains("(a) add .mpol_flags(MpolFlags::STATIC_NODES)"),
        "bail must enumerate hatch (a): {err}",
    );
    assert!(
        err.contains("(b) widen the cpuset"),
        "bail must enumerate hatch (b): {err}",
    );
    assert!(
        err.contains("CpusetSpec::Numa(N)"),
        "bail must cite CpusetSpec::Numa(N) example: {err}",
    );
    assert!(
        err.contains("CpusetSpec::Exact"),
        "bail must name CpusetSpec::Exact widening: {err}",
    );
    assert!(
        err.contains("almost certainly unintended"),
        "bail must frame mismatch as unintended: {err}",
    );
}

#[test]
fn validate_mempolicy_interleave_partial_uncovered() {
    let (cg, topo) = make_numa_ctx(2, 2, 4, 1);
    let ctx = ctx_from(&cg, &topo);
    let cpuset: BTreeSet<usize> = (0..4).collect(); // NUMA node 0 only
    let policy = MemPolicy::Interleave([0, 1].into_iter().collect());
    let err = validate_mempolicy_cpuset(
        &policy,
        crate::workload::MpolFlags::NONE,
        &cpuset,
        &ctx,
        "cg_interleave_test",
    )
    .unwrap_err()
    .to_string();
    assert!(
        err.contains("cg_interleave_test"),
        "bail must name cgroup: {err}"
    );
    // Only node 1 is uncovered (node 0 is covered by cpuset); the
    // bail should not list node 0 in the uncovered set.
    assert!(
        err.contains("[1]"),
        "bail must name uncovered node 1: {err}"
    );
    assert!(err.contains("{0}"), "bail must name cpuset node 0: {err}");
    assert!(
        err.contains("(a) add .mpol_flags(MpolFlags::STATIC_NODES)"),
        "bail must enumerate hatch (a): {err}",
    );
    assert!(
        err.contains("(b) widen the cpuset"),
        "bail must enumerate hatch (b): {err}",
    );
    assert!(
        err.contains("CpusetSpec::Numa(N)"),
        "bail must cite CpusetSpec::Numa(N) example: {err}",
    );
    assert!(
        err.contains("CpusetSpec::Exact"),
        "bail must name CpusetSpec::Exact widening: {err}",
    );
}

/// `MPOL_F_STATIC_NODES` is the kernel's explicit opt-in for
/// keeping a mempolicy nodemask absolute across cpuset changes,
/// so the validator must NOT reject a policy referencing nodes
/// outside the cpuset when that flag is set — the caller has
/// signaled intentional cross-node placement.
#[test]
fn validate_mempolicy_static_nodes_bypasses_cpuset_check() {
    let (cg, topo) = make_numa_ctx(2, 2, 4, 1);
    let ctx = ctx_from(&cg, &topo);
    let cpuset: BTreeSet<usize> = (0..4).collect(); // NUMA node 0 only
    let policy = MemPolicy::Interleave([0, 1].into_iter().collect());
    assert!(
        validate_mempolicy_cpuset(
            &policy,
            crate::workload::MpolFlags::STATIC_NODES,
            &cpuset,
            &ctx,
            "cg_0",
        )
        .is_ok()
    );
}

/// `STATIC_NODES | RELATIVE_NODES` is a kernel-rejected
/// combination — `sanitize_mpol_flags` in `mm/mempolicy.c`
/// returns `EINVAL` if both bits are set. The validator must
/// surface this with a named diagnostic before the syscall,
/// not let it collapse into a generic EINVAL at runtime.
#[test]
fn validate_mempolicy_rejects_static_and_relative_conflict() {
    let (cg, topo) = make_numa_ctx(2, 2, 4, 1);
    let ctx = ctx_from(&cg, &topo);
    let cpuset: BTreeSet<usize> = (0..4).collect();
    let policy = MemPolicy::Bind([0].into_iter().collect());
    let flags =
        crate::workload::MpolFlags::STATIC_NODES | crate::workload::MpolFlags::RELATIVE_NODES;
    let err = validate_mempolicy_cpuset(&policy, flags, &cpuset, &ctx, "cg_0")
        .expect_err("STATIC_NODES | RELATIVE_NODES must be rejected");
    let rendered = format!("{err:#}");
    assert!(
        rendered.contains("mutually exclusive"),
        "error must name the mutual-exclusion contract; got: {rendered}"
    );
}

/// The unknown-bit guard must reject any `MpolFlags` bit that
/// isn't one of the three documented constants. Without this
/// test, a regression that accidentally widened `known_bits` or
/// skipped the guard would land silently — the kernel would
/// either EINVAL or (worse) interpret the bit as a flag the
/// validator doesn't model. Uses the `#[cfg(test)]`
/// `from_bits_for_test` constructor to synthesize a bit pattern
/// (1 << 10) that no production `MpolFlags` call path can
/// produce via the named constants.
#[test]
fn validate_mempolicy_rejects_unknown_bits() {
    let (cg, topo) = make_numa_ctx(2, 2, 4, 1);
    let ctx = ctx_from(&cg, &topo);
    let cpuset: BTreeSet<usize> = (0..8).collect();
    let unknown = crate::workload::MpolFlags::from_bits_for_test(1 << 10);
    let err = validate_mempolicy_cpuset(
        &MemPolicy::Default,
        unknown,
        &cpuset,
        &ctx,
        "cg_unknown_bit",
    )
    .expect_err("unknown bit must bail");
    let s = err.to_string();
    assert!(s.contains("cg_unknown_bit"), "bail must name cgroup: {s}");
    assert!(
        s.contains("unknown bit"),
        "bail must name the unknown-bit contract: {s}"
    );
    assert!(
        s.contains("STATIC_NODES"),
        "bail must enumerate the known bits so the user sees what IS supported: {s}",
    );
}

/// `RELATIVE_NODES` treats the policy nodemask as an ordinal
/// into the cpuset's allowed-nodes set — the kernel performs
/// the relative→absolute remap internally, so cpuset coverage
/// in absolute-id terms does not apply. The validator must
/// bypass the uncovered-node bail path that the default
/// (no-flag) case enforces; otherwise every RELATIVE_NODES
/// policy referencing an ordinal beyond the cpuset's first
/// node would false-positive.
#[test]
fn validate_mempolicy_relative_nodes_bypasses_cpuset_check() {
    let (cg, topo) = make_numa_ctx(2, 2, 4, 1);
    let ctx = ctx_from(&cg, &topo);
    // cpuset covers NUMA node 0 only; policy references
    // "node 1" which would fail the absolute-id coverage
    // check in the default path. RELATIVE_NODES must bypass.
    let cpuset: BTreeSet<usize> = (0..4).collect();
    let policy = MemPolicy::Bind([1].into_iter().collect());
    assert!(
        validate_mempolicy_cpuset(
            &policy,
            crate::workload::MpolFlags::RELATIVE_NODES,
            &cpuset,
            &ctx,
            "cg_0",
        )
        .is_ok(),
        "RELATIVE_NODES must bypass the absolute-id cpuset coverage check"
    );
}

/// Under `STATIC_NODES` the nodemask is absolute, so the
/// validator must verify every referenced node actually exists
/// on the host topology. A policy pinning node 7 on a 2-node
/// host would fail at syscall time; surfacing it here names
/// the offender before the failure.
#[test]
fn validate_mempolicy_static_nodes_rejects_missing_host_node() {
    let (cg, topo) = make_numa_ctx(2, 2, 4, 1); // host has nodes {0, 1}
    let ctx = ctx_from(&cg, &topo);
    let cpuset: BTreeSet<usize> = (0..8).collect();
    // Reference a node that does not exist on this synthetic host.
    let policy = MemPolicy::Bind([7].into_iter().collect());
    let err = validate_mempolicy_cpuset(
        &policy,
        crate::workload::MpolFlags::STATIC_NODES,
        &cpuset,
        &ctx,
        "cg_0",
    )
    .expect_err("STATIC_NODES policy referencing missing host node must be rejected");
    let rendered = format!("{err:#}");
    assert!(
        rendered.contains("do not exist on this host"),
        "error must name the missing-host-node condition; got: {rendered}"
    );
}

#[test]
fn cgroupdef_mem_policy_builder() {
    let def = CgroupDef::named("test").mem_policy(MemPolicy::Bind([0].into_iter().collect()));
    assert!(matches!(def.works[0].mem_policy, MemPolicy::Bind(_)));
}

// ---------------------------------------------------------------
// apply_setup tests via MockCgroupOps
// ---------------------------------------------------------------
//
// MockCgroupOps is a recording implementor of crate::cgroup::CgroupOps
// that stores every call it receives in an internal Vec and can be
// primed to return an error from the next call. This lets
// apply_setup tests assert on the sequence of cgroup operations
// without touching /sys/fs/cgroup, so they run as regular userspace
// unit tests.
//
// apply_setup still calls WorkloadHandle::spawn, which forks real
// worker processes. That's intentional: fork does not require root,
// and the cgroup.procs write (which would require root in the real
// kernel) is abstracted behind the mock. The test subject is the
// orchestration logic — "for each def, call create_cgroup, then
// set_cpuset if spec.is_some(), then move_tasks after spawn".
//
// Parallel-nextest behavior: verified non-flaky over repeated
// `cargo nextest run --lib -E 'test(apply_setup)' --test-threads 8`
// invocations and back-to-back full-suite runs. Each `MockCgroupOps`
// owns its own `Mutex<Vec<CgroupCall>>`, so cross-test recording
// cannot contend. `apply_setup` does call `WorkloadHandle::start`
// (see top of this file) — workers wake, run briefly, and are then
// SIGKILL'd when the owning `WorkloadHandle` drops via
// `cleanup_state(&mut state)` / `state.handles.clear()` at the tail
// of each test. No test assertion depends on worker output, only
// on mock-recorded cgroup calls, so worker timing is not
// observable. Fd footprint is 4 pipes × `workers()` per test — 8
// fds for the 2-worker tests, well inside any RLIMIT_NOFILE the
// harness sets.

use crate::cgroup::CgroupOps;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Mutex;

/// A call captured by MockCgroupOps during apply_setup execution.
/// Equality-comparable so tests can assert on the exact sequence.
/// `MoveTasks` stores the pid count rather than the full `pids` Vec
/// because PIDs are unpredictable between runs.
#[derive(Debug, Clone, PartialEq, Eq)]
enum CgroupCall {
    Setup(BTreeSet<crate::cgroup::Controller>),
    CreateCgroup(String),
    RemoveCgroup(String),
    SetCpuset(String, BTreeSet<usize>),
    ClearCpuset(String),
    SetCpusetMems(String, BTreeSet<usize>),
    #[allow(dead_code)] // Emitted by CgroupOps::clear_cpuset_mems; no test asserts on it yet.
    ClearCpusetMems(String),
    // (name, quota_us, period_us); quota=None means "max".
    SetCpuMax(String, Option<u64>, u64),
    SetCpuWeight(String, u32),
    SetMemoryMax(String, Option<u64>),
    SetMemoryHigh(String, Option<u64>),
    SetMemoryLow(String, Option<u64>),
    SetMemorySwapMax(String, Option<u64>),
    SetIoWeight(String, u16),
    SetFreeze(String, bool),
    SetPidsMax(String, Option<u64>),
    MoveTask(String, libc::pid_t),
    MoveTasks(String, usize), // (cgroup name, number of pids)
    // (cgroup name, child pid). Records the placement-before-exec
    // write that the payload-spawn cgroup-sync handshake routes
    // through `CgroupOps::place_task_during_handshake`. Distinct
    // from `MoveTask` so tests can assert that
    // `Op::RunPayload { cgroup: Some(_), .. }` goes through the
    // handshake path (not the post-spawn `move_task` path used
    // for synthetic workers spawned via `WorkloadHandle::spawn`).
    PlaceTaskDuringHandshake(String, libc::pid_t),
    ClearSubtreeControl(String),
    DrainTasks(String),
    ReadProcs(String),
    CleanupAll,
}

/// Predicate over a CgroupCall, used by `fail_nth_call_matching`
/// to schedule a semantic-index failure. Boxed so the mock can
/// hold heterogeneous predicates without parameterizing the
/// struct on a generic type.
type CallPredicate = Box<dyn Fn(&CgroupCall) -> bool + Send + 'static>;

struct MockCgroupOps {
    parent: std::path::PathBuf,
    calls: Mutex<Vec<CgroupCall>>,
    // When Some, the Nth call (indexed from 0 at insertion time)
    // returns an error and decrements; otherwise all calls return Ok.
    fail_at: Mutex<Option<(usize, String)>>,
    // When Some, the Nth call (0-indexed) where the predicate
    // returns true returns the error. `matches_so_far` tracks
    // how many predicate-matches have been seen since the
    // scheduler was installed.
    fail_match: Mutex<Option<(usize, usize, String, CallPredicate)>>,
    // Pre-loaded responses for read_procs. Maps cgroup name to
    // the pid list the mock returns. Cgroups absent from the
    // map return Ok(vec![]) (the same shape the production impl
    // emits for an empty cgroup.procs file).
    procs_returns: Mutex<HashMap<String, Vec<libc::pid_t>>>,
    // When Some, the next read_procs call against the matching
    // cgroup name returns this error. Lets tests pin
    // error-propagation through the apply_ops dispatch arm.
    procs_err: Mutex<Option<(String, String)>>,
}

impl MockCgroupOps {
    fn new() -> Self {
        Self {
            parent: std::path::PathBuf::from("/mock/cgroup"),
            calls: Mutex::new(Vec::new()),
            fail_at: Mutex::new(None),
            fail_match: Mutex::new(None),
            procs_returns: Mutex::new(HashMap::new()),
            procs_err: Mutex::new(None),
        }
    }

    /// Pre-load the response `read_procs(cgroup)` returns. Subsequent
    /// `read_procs` invocations against `cgroup` clone the stored
    /// `pids` Vec. Cgroups not pre-loaded return `Ok(vec![])`.
    #[allow(dead_code)] // call-site is the Op::CaptureCgroupProcs dispatch tests
    fn set_procs(&self, cgroup: &str, pids: Vec<libc::pid_t>) {
        self.procs_returns
            .lock()
            .unwrap()
            .insert(cgroup.to_string(), pids);
    }

    /// Schedule `read_procs(cgroup)` to fail with `message`. One-shot:
    /// the error fires on the first matching call and clears.
    #[allow(dead_code)] // call-site is the Op::CaptureCgroupProcs dispatch tests
    fn fail_read_procs(&self, cgroup: &str, message: &str) {
        *self.procs_err.lock().unwrap() = Some((cgroup.to_string(), message.to_string()));
    }

    /// Return an error from the Nth call (0-indexed from now) with
    /// the given message. Used by tests that check error
    /// propagation through apply_setup.
    ///
    /// Brittle: absolute-index counting drifts when a handler
    /// refactor adds an unrelated kernel-op between setup and
    /// the targeted call. For new tests prefer
    /// [`Self::fail_nth_call_matching`] which counts SEMANTIC
    /// matches (e.g. "the 1st MoveTasks call after now")
    /// instead of absolute call indices.
    #[allow(dead_code)] // older tests still call this; the semantic-index helper is preferred for new sites
    fn fail_call_at(&self, index: usize, message: &str) {
        *self.fail_at.lock().unwrap() = Some((index, message.to_string()));
    }

    /// Return an error from the Nth call (0-indexed from now)
    /// where `predicate(call)` returns true. Semantic index —
    /// adversary-resistant to handler refactors that add or
    /// reorder unrelated kernel-ops, because the count only
    /// advances on predicate matches.
    ///
    /// Example: fail the 1st MoveTasks call after now (regardless
    /// of what other kernel ops the handler emits first):
    /// ```ignore
    /// mock.fail_nth_call_matching(
    ///     0,
    ///     |c| matches!(c, CgroupCall::MoveTasks(_, _)),
    ///     "injected ENOSPC",
    /// );
    /// ```
    ///
    /// # Predicate must not re-enter the mock
    ///
    /// The predicate runs inside `record`'s `fail_match` mutex
    /// guard. Calling any `&self` method on the mock from
    /// inside the predicate (`mock.calls()`, another
    /// `fail_nth_call_matching`, even an indirect `CgroupOps`
    /// trait method) attempts to re-acquire one of the mock's
    /// mutexes and **deadlocks**. In practice every realistic
    /// predicate is a `matches!` against the variant tag —
    /// no re-entry concern — but the signature `Fn(&CgroupCall)
    /// -> bool` permits any closure body, so the contract is
    /// pinned here explicitly.
    ///
    /// # Interaction with [`Self::fail_call_at`]
    ///
    /// Both schedulers can be installed simultaneously without
    /// panic. `record` checks `fail_match` first then
    /// `fail_at` — semantic-index has priority on the call
    /// where its predicate first matches; absolute-index fires
    /// independently on the call whose index it targets, so a
    /// test arming both will see TWO separate failures fire
    /// (one for each). Tests that genuinely want layered
    /// scheduling can rely on this deterministic interaction;
    /// tests that want a single scheduler should pick one and
    /// not install the other.
    #[allow(dead_code)] // called by MoveAllTasks failure-injection tests; future tests will adopt
    fn fail_nth_call_matching<F>(&self, n: usize, predicate: F, message: &str)
    where
        F: Fn(&CgroupCall) -> bool + Send + 'static,
    {
        *self.fail_match.lock().unwrap() = Some((n, 0, message.to_string(), Box::new(predicate)));
    }

    fn calls(&self) -> Vec<CgroupCall> {
        self.calls.lock().unwrap().clone()
    }

    /// Record a call and decide whether to return Ok or inject an
    /// error. Centralizes the fail-injection logic so every
    /// trait method gets it for free. Checks `fail_match` first
    /// (semantic-index) then `fail_at` (absolute-index); either
    /// scheduler clears itself after firing.
    fn record(&self, call: CgroupCall) -> Result<()> {
        let mut calls = self.calls.lock().unwrap();
        let current_index = calls.len();
        calls.push(call);
        // Clone the just-recorded call so the calls lock can
        // drop before predicate evaluation — the predicate
        // would otherwise need the calls lock to inspect call
        // history. The fail_match lock IS still held during
        // predicate evaluation below, so a predicate that
        // re-enters the mock via any &self method that touches
        // fail_match deadlocks; see fail_nth_call_matching's
        // "# Predicate must not re-enter the mock" doc.
        let recorded_idx = calls.len() - 1;
        let just_recorded = calls[recorded_idx].clone();
        drop(calls);
        let mut fail_match = self.fail_match.lock().unwrap();
        if let Some((target_n, ref mut matches_so_far, ref message, ref predicate)) = *fail_match
            && predicate(&just_recorded)
        {
            if *matches_so_far == target_n {
                let err_msg = message.clone();
                *fail_match = None;
                return Err(anyhow::anyhow!(err_msg));
            }
            *matches_so_far += 1;
        }
        drop(fail_match);
        let mut fail = self.fail_at.lock().unwrap();
        if let Some((index, ref message)) = *fail
            && current_index == index
        {
            let err_msg = message.clone();
            *fail = None;
            return Err(anyhow::anyhow!(err_msg));
        }
        Ok(())
    }
}

impl CgroupOps for MockCgroupOps {
    fn parent_path(&self) -> &Path {
        &self.parent
    }
    fn setup(&self, controllers: &BTreeSet<crate::cgroup::Controller>) -> Result<()> {
        self.record(CgroupCall::Setup(controllers.clone()))
    }
    fn create_cgroup(&self, name: &str) -> Result<()> {
        self.record(CgroupCall::CreateCgroup(name.to_string()))
    }
    fn remove_cgroup(&self, name: &str) -> Result<()> {
        self.record(CgroupCall::RemoveCgroup(name.to_string()))
    }
    fn set_cpuset(&self, name: &str, cpus: &BTreeSet<usize>) -> Result<()> {
        self.record(CgroupCall::SetCpuset(name.to_string(), cpus.clone()))
    }
    fn clear_cpuset(&self, name: &str) -> Result<()> {
        self.record(CgroupCall::ClearCpuset(name.to_string()))
    }
    fn set_cpuset_mems(&self, name: &str, nodes: &BTreeSet<usize>) -> Result<()> {
        self.record(CgroupCall::SetCpusetMems(name.to_string(), nodes.clone()))
    }
    fn clear_cpuset_mems(&self, name: &str) -> Result<()> {
        self.record(CgroupCall::ClearCpusetMems(name.to_string()))
    }
    fn set_cpu_max(&self, name: &str, quota_us: Option<u64>, period_us: u64) -> Result<()> {
        self.record(CgroupCall::SetCpuMax(name.to_string(), quota_us, period_us))
    }
    fn set_cpu_weight(&self, name: &str, weight: u32) -> Result<()> {
        self.record(CgroupCall::SetCpuWeight(name.to_string(), weight))
    }
    fn set_memory_max(&self, name: &str, bytes: Option<u64>) -> Result<()> {
        self.record(CgroupCall::SetMemoryMax(name.to_string(), bytes))
    }
    fn set_memory_high(&self, name: &str, bytes: Option<u64>) -> Result<()> {
        self.record(CgroupCall::SetMemoryHigh(name.to_string(), bytes))
    }
    fn set_memory_low(&self, name: &str, bytes: Option<u64>) -> Result<()> {
        self.record(CgroupCall::SetMemoryLow(name.to_string(), bytes))
    }
    fn set_io_weight(&self, name: &str, weight: u16) -> Result<()> {
        self.record(CgroupCall::SetIoWeight(name.to_string(), weight))
    }
    fn set_freeze(&self, name: &str, frozen: bool) -> Result<()> {
        self.record(CgroupCall::SetFreeze(name.to_string(), frozen))
    }
    fn set_pids_max(&self, name: &str, max: Option<u64>) -> Result<()> {
        self.record(CgroupCall::SetPidsMax(name.to_string(), max))
    }
    fn set_memory_swap_max(&self, name: &str, bytes: Option<u64>) -> Result<()> {
        self.record(CgroupCall::SetMemorySwapMax(name.to_string(), bytes))
    }
    fn move_task(&self, name: &str, pid: libc::pid_t) -> Result<()> {
        self.record(CgroupCall::MoveTask(name.to_string(), pid))
    }
    fn move_tasks(&self, name: &str, pids: &[libc::pid_t]) -> Result<()> {
        self.record(CgroupCall::MoveTasks(name.to_string(), pids.len()))
    }
    fn place_task_during_handshake(&self, cgroup_name: &str, child_pid: libc::pid_t) -> Result<()> {
        self.record(CgroupCall::PlaceTaskDuringHandshake(
            cgroup_name.to_string(),
            child_pid,
        ))
    }
    fn clear_subtree_control(&self, name: &str) -> Result<()> {
        self.record(CgroupCall::ClearSubtreeControl(name.to_string()))
    }
    fn drain_tasks(&self, name: &str) -> Result<()> {
        self.record(CgroupCall::DrainTasks(name.to_string()))
    }
    fn read_procs(&self, name: &str) -> Result<Vec<libc::pid_t>> {
        // Record the call first so failure-injection still leaves
        // the audit trail tests assert against.
        self.record(CgroupCall::ReadProcs(name.to_string()))?;
        // Fail-injection wins over pre-loaded returns: tests that
        // arm both pick the error path.
        let err = {
            let mut guard = self.procs_err.lock().unwrap();
            match guard.as_ref() {
                Some((target, _)) if target == name => guard.take(),
                _ => None,
            }
        };
        if let Some((_, msg)) = err {
            anyhow::bail!("{msg}");
        }
        let pids = self
            .procs_returns
            .lock()
            .unwrap()
            .get(name)
            .cloned()
            .unwrap_or_default();
        Ok(pids)
    }
    fn cleanup_all(&self) -> Result<()> {
        self.record(CgroupCall::CleanupAll)
    }
}

/// Build a Ctx backed by MockCgroupOps so apply_setup can be driven
/// without cgroup filesystem access. Topology fixed at 1 NUMA /
/// 1 LLC / 4 cores / 1 thread = 4 CPUs — enough range to cover
/// per-cpu cpuset assertions without making the mock brittle.
fn mock_ctx<'a>(mock: &'a MockCgroupOps, topo: &'a crate::topology::TestTopology) -> Ctx<'a> {
    Ctx {
        cgroups: mock,
        topo,
        duration: Duration::from_secs(1),
        workers_per_cgroup: 1,
        sched_pid: None,
        settle: Duration::ZERO,
        work_type_override: None,
        assert: crate::assert::Assert::default_checks(),
        wait_for_map_write: false,
        current_step: std::sync::Arc::new(std::sync::atomic::AtomicU16::new(0)),
        entry_name: None,
    }
}

fn mock_topo() -> crate::topology::TestTopology {
    crate::topology::TestTopology::from_vm_topology(&crate::vmm::topology::Topology::new(
        1, 1, 4, 1,
    ))
}

/// Drop workload + payload handles inside state so apply_setup
/// tests don't leak worker or payload processes. Synthetic
/// `WorkloadHandle`s SIGKILL their workers on Drop, so a
/// `handles.clear()` is enough; `PayloadHandle` likewise
/// SIGKILLs its child on Drop (with an eprintln warning about
/// metrics not being recorded — acceptable in the test path
/// where metrics aren't what's under test). Calling
/// `drain_all_payload_handles` routes through `.kill()` so the
/// metric-emission branch runs and the test doesn't trigger
/// the Drop-warning banner on stderr.
fn cleanup_state(state: &mut StepState<'_>) {
    state.handles.clear();
    drain_all_payload_handles(&mut state.payload_handles);
}

/// Test helper: call `apply_setup` against a step-local-only
/// [`ScenarioState`]. Constructs a throwaway backdrop state
/// pointing at the same mock-cgroups handle `state` uses so
/// tests that only exercise step-local semantics stay terse.
fn apply_setup_test<'a>(
    ctx: &'a Ctx<'a>,
    state: &mut StepState<'a>,
    defs: &[CgroupDef],
) -> Result<()> {
    let mut backdrop = BackdropState::empty(ctx);
    let mut scenario = ScenarioState::new(state, &mut backdrop);
    apply_setup(ctx, &mut scenario, defs)
}

/// Test helper: call `apply_ops` against a step-local-only
/// [`ScenarioState`]. Mirrors [`apply_setup_test`] for ops.
fn apply_ops_test<'a>(ctx: &'a Ctx<'a>, state: &mut StepState<'a>, ops: &[Op]) -> Result<()> {
    let mut backdrop = BackdropState::empty(ctx);
    let mut scenario = ScenarioState::new(state, &mut backdrop);
    apply_ops(ctx, &mut scenario, ops, false)
}

/// Common test preamble for MockCgroupOps tests in this module.
/// Builds the [`MockCgroupOps`], default [`TestTopology`] (1 NUMA /
/// 1 LLC / 4 cores / 1 thread), and [`Ctx`] triple — replacing the
/// 3-line `let mock = MockCgroupOps::new(); let topo = mock_topo();
/// let ctx = mock_ctx(&mock, &topo);` boilerplate that previously
/// opened every test in the apply_setup / apply_ops / execute_steps
/// suites. The local bindings are inlined (not returned from a
/// helper fn) because [`Ctx`] borrows `mock` + `topo` and
/// [`StepState`] borrows [`Ctx`] — a tuple-returning helper would
/// require self-referential lifetimes.
///
/// # Skip cases — sites that intentionally KEEP the manual preamble
///
/// - **Tests that need `mut ctx`** (override [`Ctx::duration`] /
///   [`Ctx::sched_pid`] post-construction) keep the explicit 3-line
///   form — the macro emits `let ctx` (immutable) only. ~7 such
///   sites in the holdspec / sched-dies suites.
/// - **Bare [`MockCgroupOps`] contract tests** (no topology / no
///   Ctx needed — e.g. the [`MockCgroupOps::fail_nth_call_matching`]
///   contract suite) construct only the mock; the macro would
///   force an unused topo+ctx pair.
/// - **Tests with non-default topology** (e.g. 2-NUMA configurations
///   built via [`TestTopology::from_vm_topology`]) — the macro
///   hard-codes [`mock_topo`] and can't be parameterized without
///   exploding the variant count.
/// - **Multi-case tests that reuse an outer `topo`** but reset
///   `mock` per sub-case — the macro re-binds `topo`, which would
///   shadow the outer binding.
macro_rules! mock_setup {
    ($mock:ident, $topo:ident, $ctx:ident) => {
        let $mock = MockCgroupOps::new();
        let $topo = mock_topo();
        let $ctx = mock_ctx(&$mock, &$topo);
    };
}

/// Variant of [`mock_setup`] that also constructs a [`StepState`].
/// Equivalent to the 4-line preamble (mock + topo + ctx + state)
/// that opens most apply_setup / apply_ops tests.
macro_rules! mock_setup_state {
    ($mock:ident, $topo:ident, $ctx:ident, $state:ident) => {
        mock_setup!($mock, $topo, $ctx);
        let mut $state = StepState::empty(&$ctx);
    };
}

/// Variant of [`mock_setup`] that also constructs the
/// `(StepState, BackdropState)` pair that the run-step /
/// scenario-state tests open with — distinct from
/// [`mock_setup_state`] because tests that build the pair
/// typically also wire a [`ScenarioState::new`] from them
/// (instead of going through `apply_setup_test` /
/// `apply_ops_test` which carry their own throwaway backdrop).
macro_rules! mock_setup_backdrop {
    ($mock:ident, $topo:ident, $ctx:ident, $step:ident, $backdrop:ident) => {
        mock_setup!($mock, $topo, $ctx);
        let mut $step = StepState::empty(&$ctx);
        let mut $backdrop = BackdropState::empty(&$ctx);
    };
}

// --- MockCgroupOps::fail_nth_call_matching contract tests ---

/// Non-matching calls MUST NOT advance the predicate-match
/// counter. A regression that incremented on every call
/// (rather than only on predicate matches) would fire the
/// failure on an unrelated kernel op — defeats the semantic-
/// index resilience that's the helper's reason to exist.
#[test]
fn fail_nth_call_matching_predicate_skipping_does_not_advance_counter() {
    let mock = MockCgroupOps::new();
    mock.fail_nth_call_matching(
        0,
        |c| matches!(c, CgroupCall::CreateCgroup(_)),
        "injected on first CreateCgroup",
    );
    // Setup + remove_cgroup don't match the predicate, so
    // they MUST succeed (counter stays at 0).
    mock.setup(&BTreeSet::new())
        .expect("non-matching setup() must succeed");
    mock.remove_cgroup("x")
        .expect("non-matching remove_cgroup() must succeed");
    // First CreateCgroup matches at counter=0 → fires.
    let err = mock
        .create_cgroup("first")
        .expect_err("first CreateCgroup must hit the injected failure");
    assert!(
        format!("{err:#}").contains("injected on first CreateCgroup"),
        "error must surface the injected message verbatim: {err:#}"
    );
}

/// The `n` parameter MUST count only predicate-matching calls.
/// Tests that non-matching calls interleaved between matches
/// don't shift the targeted match position — i.e. `n=1` means
/// "the 2nd MATCH" not "the 2nd CALL after the first match."
#[test]
fn fail_nth_call_matching_n_index_counts_only_matches() {
    let mock = MockCgroupOps::new();
    mock.fail_nth_call_matching(
        1,
        |c| matches!(c, CgroupCall::MoveTasks(_, _)),
        "injected on 2nd MoveTasks",
    );
    // Counter stays 0 throughout these non-matching calls.
    mock.create_cgroup("a")
        .expect("non-match create_cgroup must succeed");
    // First MoveTasks: n=0, no fire (target is n=1).
    mock.move_tasks("a", &[1])
        .expect("1st MoveTasks must succeed (counter advances to 1)");
    // Another non-matching call between matches — must not shift n.
    mock.create_cgroup("b")
        .expect("non-match between matches must succeed without advancing n");
    // Second MoveTasks: n=1, fires.
    let err = mock
        .move_tasks("b", &[2])
        .expect_err("2nd MoveTasks must hit the injected failure");
    assert!(
        format!("{err:#}").contains("injected on 2nd MoveTasks"),
        "error must surface the injected message: {err:#}"
    );
}

/// If the predicate never matches, the scheduler MUST NOT
/// fire — pins that the helper can't accidentally trigger on
/// the "Nth call regardless of predicate" interpretation that
/// would defeat the whole point of semantic indexing.
#[test]
fn fail_nth_call_matching_no_match_means_no_fire() {
    let mock = MockCgroupOps::new();
    mock.fail_nth_call_matching(
        0,
        |c| matches!(c, CgroupCall::MoveTasks(_, _)),
        "should never fire — no MoveTasks calls happen",
    );
    // Several non-MoveTasks calls — all must succeed.
    mock.create_cgroup("x")
        .expect("non-match create_cgroup must succeed");
    mock.set_cpuset("x", &BTreeSet::new())
        .expect("non-match set_cpuset must succeed");
    mock.remove_cgroup("x")
        .expect("non-match remove_cgroup must succeed");
    assert_eq!(
        mock.calls().len(),
        3,
        "all 3 non-matching calls must have been recorded without firing: {:?}",
        mock.calls()
    );
}

#[test]
fn apply_setup_empty_defs_is_noop() {
    mock_setup_state!(mock, topo, ctx, state);
    apply_setup_test(&ctx, &mut state, &[]).unwrap();
    assert!(
        mock.calls().is_empty(),
        "apply_setup on zero defs must not call any cgroup op, got: {:?}",
        mock.calls()
    );
    cleanup_state(&mut state);
}

#[test]
fn apply_setup_creates_cgroup_per_def() {
    mock_setup_state!(mock, topo, ctx, state);
    let defs = vec![
        CgroupDef::named("cg_a").workers(1),
        CgroupDef::named("cg_b").workers(1),
    ];
    apply_setup_test(&ctx, &mut state, &defs).unwrap();
    let calls = mock.calls();
    let creates: Vec<&CgroupCall> = calls
        .iter()
        .filter(|c| matches!(c, CgroupCall::CreateCgroup(_)))
        .collect();
    assert_eq!(
        creates,
        vec![
            &CgroupCall::CreateCgroup("cg_a".to_string()),
            &CgroupCall::CreateCgroup("cg_b".to_string()),
        ],
        "one create_cgroup call per def, in order"
    );
    cleanup_state(&mut state);
}

#[test]
fn apply_setup_sets_cpuset_when_spec_present() {
    mock_setup_state!(mock, topo, ctx, state);
    let cpus: BTreeSet<usize> = [0, 1].into_iter().collect();
    let defs = vec![
        CgroupDef::named("cg_0")
            .cpuset(CpusetSpec::Exact(cpus.clone()))
            .workers(1),
    ];
    apply_setup_test(&ctx, &mut state, &defs).unwrap();
    let calls = mock.calls();
    assert!(
        calls.contains(&CgroupCall::SetCpuset("cg_0".to_string(), cpus.clone())),
        "set_cpuset must be called with exactly the resolved cpu set, got: {calls:?}"
    );
    // state.cpusets should mirror the set so later SetAffinity /
    // MemPolicy checks see the resolved cpuset.
    assert_eq!(
        state.cpusets.get("cg_0"),
        Some(&cpus),
        "state.cpusets must record the resolved set"
    );
    cleanup_state(&mut state);
}

#[test]
fn apply_setup_skips_cpuset_when_none() {
    mock_setup_state!(mock, topo, ctx, state);
    // cpuset: None → inherit parent's set, apply_setup must not
    // emit a set_cpuset call.
    let defs = vec![CgroupDef::named("cg_inherit").workers(1)];
    apply_setup_test(&ctx, &mut state, &defs).unwrap();
    let calls = mock.calls();
    let has_set_cpuset = calls
        .iter()
        .any(|c| matches!(c, CgroupCall::SetCpuset(_, _)));
    assert!(
        !has_set_cpuset,
        "no set_cpuset should be emitted when CgroupDef.cpuset is None, got: {calls:?}"
    );
    assert!(
        state.cpusets.is_empty(),
        "state.cpusets should stay empty when no CpusetSpec was resolved"
    );
    cleanup_state(&mut state);
}

#[test]
fn apply_setup_moves_spawned_tasks_into_cgroup() {
    mock_setup_state!(mock, topo, ctx, state);
    // workers(2): after spawn, apply_setup must call move_tasks
    // with 2 pids.
    let defs = vec![CgroupDef::named("cg_move").workers(2)];
    apply_setup_test(&ctx, &mut state, &defs).unwrap();
    let calls = mock.calls();
    assert!(
        calls.contains(&CgroupCall::MoveTasks("cg_move".to_string(), 2)),
        "move_tasks must be called with the 2 spawned worker pids, got: {calls:?}"
    );
    // Ordering invariant: move_tasks follows create_cgroup, and
    // set_cpuset (when present) follows create_cgroup but precedes
    // move_tasks. Here with no cpuset, just assert create precedes
    // move.
    let create_idx = calls
        .iter()
        .position(|c| matches!(c, CgroupCall::CreateCgroup(n) if n == "cg_move"))
        .expect("create_cgroup for cg_move");
    let move_idx = calls
        .iter()
        .position(|c| matches!(c, CgroupCall::MoveTasks(n, _) if n == "cg_move"))
        .expect("move_tasks for cg_move");
    assert!(
        create_idx < move_idx,
        "create_cgroup must precede move_tasks for the same cgroup: {calls:?}"
    );
    cleanup_state(&mut state);
}

#[test]
fn apply_setup_sets_cpuset_before_move_tasks() {
    // Ordering invariant: for a cgroup with both a cpuset spec and
    // workers, `set_cpuset` MUST precede `move_tasks` so the
    // kernel enforces the cpu mask on the first scheduling
    // decision after the task enters the cgroup. Moving first
    // would let tasks briefly run on cpus outside the intended
    // set.
    mock_setup_state!(mock, topo, ctx, state);
    let cpus: BTreeSet<usize> = [0, 1].into_iter().collect();
    let defs = vec![
        CgroupDef::named("cg_ordered")
            .cpuset(CpusetSpec::Exact(cpus.clone()))
            .workers(2),
    ];
    apply_setup_test(&ctx, &mut state, &defs).unwrap();
    let calls = mock.calls();
    let set_idx = calls
        .iter()
        .position(|c| matches!(c, CgroupCall::SetCpuset(n, _) if n == "cg_ordered"))
        .expect("set_cpuset for cg_ordered");
    let move_idx = calls
        .iter()
        .position(|c| matches!(c, CgroupCall::MoveTasks(n, _) if n == "cg_ordered"))
        .expect("move_tasks for cg_ordered");
    assert!(
        set_idx < move_idx,
        "set_cpuset must precede move_tasks for the same cgroup: {calls:?}"
    );
    cleanup_state(&mut state);
}

#[test]
fn apply_setup_bails_on_invalid_cpuset_spec() {
    mock_setup_state!(mock, topo, ctx, state);
    // Llc(99) on a 1-LLC topology is out of range; CpusetSpec::validate
    // bails after create_cgroup runs but before set_cpuset / move_tasks
    // fire.
    let defs = vec![CgroupDef::named("cg_bad").cpuset(CpusetSpec::Llc(99))];
    let err = apply_setup_test(&ctx, &mut state, &defs).unwrap_err();
    let msg = format!("{err:#}");
    assert!(
        msg.contains("CpusetSpec validation failed"),
        "expected validation error, got: {msg}"
    );
    // create_cgroup runs before cpuset validation — record that
    // here so future refactors notice if the order flips.
    let calls = mock.calls();
    assert_eq!(
        calls,
        vec![CgroupCall::CreateCgroup("cg_bad".to_string())],
        "current ordering: create_cgroup first, then cpuset validation"
    );
    cleanup_state(&mut state);
}

#[test]
fn apply_setup_propagates_set_cpuset_error() {
    mock_setup_state!(mock, topo, ctx, state);
    // Inject failure at call index 1. Index 0 is the create_cgroup
    // emitted before the cpuset write; index 1 is the set_cpuset
    // itself.
    // Reordered after macro: fail_call_at is &self + only mutates
    // the fail-injection lock (not the call-index counter), so the
    // reorder is observationally identical.
    mock.fail_call_at(1, "set_cpuset kernel EBUSY");
    let cpus: BTreeSet<usize> = [0, 1].into_iter().collect();
    let defs = vec![
        CgroupDef::named("cg_setfail")
            .cpuset(CpusetSpec::Exact(cpus))
            .workers(1),
    ];
    let err = apply_setup_test(&ctx, &mut state, &defs).unwrap_err();
    let msg = format!("{err:#}");
    assert!(
        msg.contains("set_cpuset kernel EBUSY"),
        "set_cpuset error must propagate, got: {msg}"
    );
    // Check the failure halted apply_setup before reaching spawn:
    // no MoveTasks call should have been recorded.
    let calls = mock.calls();
    let has_move = calls
        .iter()
        .any(|c| matches!(c, CgroupCall::MoveTasks(_, _)));
    assert!(
        !has_move,
        "no move_tasks call should follow a failed set_cpuset, got: {calls:?}"
    );
    cleanup_state(&mut state);
}

#[test]
fn apply_setup_validates_mempolicy_against_cpuset() {
    let mock = MockCgroupOps::new();
    // 2 NUMA / 2 LLCs (1 per node) / 4 cores / 1 thread = 8 CPUs
    let topo = crate::topology::TestTopology::from_vm_topology(
        &crate::vmm::topology::Topology::new(2, 2, 4, 1),
    );
    let ctx = mock_ctx(&mock, &topo);
    let mut state = StepState::empty(&ctx);
    // cpuset = NUMA node 0 only (CPUs 0-3); mem_policy binds to
    // node 1 — must bail, no downstream spawn.
    let cpus: BTreeSet<usize> = (0..4).collect();
    let bind: BTreeSet<usize> = [1].into_iter().collect();
    let defs = vec![
        CgroupDef::named("cg_memfail")
            .cpuset(CpusetSpec::Exact(cpus))
            .mem_policy(MemPolicy::Bind(bind))
            .workers(1),
    ];
    let err = apply_setup_test(&ctx, &mut state, &defs).unwrap_err();
    let msg = format!("{err:#}");
    assert!(
        msg.contains("cg_memfail"),
        "error must name the bad cgroup, got: {msg}"
    );
    // set_cpuset was called before the mempolicy check (order
    // documented by apply_setup). Assert move_tasks did not run —
    // that would mean the pre-validation guard failed.
    let calls = mock.calls();
    let has_move = calls
        .iter()
        .any(|c| matches!(c, CgroupCall::MoveTasks(_, _)));
    assert!(
        !has_move,
        "mempolicy validation must bail before spawn, got: {calls:?}"
    );
    cleanup_state(&mut state);
}

// -- CgroupDef::workload --

/// Default CgroupDef has no payload attached — every test that
/// doesn't opt in stays Payload-free so the synthetic-workload
/// path is unaffected.
#[test]
fn cgroup_def_default_payload_is_none() {
    let def = CgroupDef::named("cg_0");
    assert!(def.payload.is_none());
}

/// The `.workload(&FIO)` builder stores the reference on the
/// CgroupDef so apply_setup can spawn it. Because `Payload` is
/// `Copy`, the builder preserves identity through pointer
/// equality after conversion to `&'static` refs.
#[test]
fn cgroup_def_workload_stores_payload() {
    use crate::test_support::{OutputFormat, Payload, PayloadKind};
    static FIO: Payload = Payload {
        name: "fio",
        kind: PayloadKind::Binary("fio"),
        output: OutputFormat::Json,
        default_args: &[],
        default_checks: &[],
        metrics: &[],
        include_files: &[],
        uses_parent_pgrp: false,
        known_flags: None,
        metric_bounds: None,
    };
    let def = CgroupDef::named("cg_0").workload(&FIO);
    let p = def.payload.expect("workload was attached");
    assert_eq!(p.name, "fio");
    assert!(!p.is_scheduler());
}

/// Scheduler-kind payloads are rejected at builder time — the
/// `workload` slot is exclusively for userspace binaries that
/// run *under* a scheduler, not for scheduler placement itself.
#[test]
#[should_panic(expected = "CgroupDef::workload called with a scheduler-kind Payload")]
fn cgroup_def_workload_rejects_scheduler_kind_payload() {
    use crate::test_support::Payload;
    let _ = CgroupDef::named("cg_0").workload(&Payload::KERNEL_DEFAULT);
}

/// The drain helper kills + removes entries whose cgroup name
/// matches the target. Non-matching entries stay in the vector
/// so subsequent step teardown (via `collect_step`) or scenario
/// end (via `collect_backdrop`) kills them in turn.
#[test]
fn drain_payload_handles_for_cgroup_removes_matching_only() {
    use crate::cgroup::CgroupManager;
    use crate::scenario::payload_run::PayloadRun;
    use crate::test_support::{OutputFormat, Payload, PayloadKind};
    use crate::topology::TestTopology;

    static TRUE_BIN: Payload = Payload {
        name: "true_bin",
        kind: PayloadKind::Binary("/bin/true"),
        output: OutputFormat::ExitCode,
        default_args: &[],
        default_checks: &[],
        metrics: &[],
        include_files: &[],
        uses_parent_pgrp: false,
        known_flags: None,
        metric_bounds: None,
    };

    let cgroups = CgroupManager::new("/nonexistent");
    let topo = TestTopology::synthetic(4, 1);
    let ctx = crate::scenario::Ctx::builder(&cgroups, &topo).build();

    let h_a = PayloadRun::new(&ctx, &TRUE_BIN)
        .spawn()
        .expect("spawn /bin/true for cg_a");
    let h_b = PayloadRun::new(&ctx, &TRUE_BIN)
        .spawn()
        .expect("spawn /bin/true for cg_b");

    let mut handles = vec![
        PayloadEntry {
            cgroup: "cg_a".to_string(),
            source: PayloadSource::CgroupDefWorkload,
            handle: h_a,
        },
        PayloadEntry {
            cgroup: "cg_b".to_string(),
            source: PayloadSource::CgroupDefWorkload,
            handle: h_b,
        },
    ];
    drain_payload_handles_for_cgroup(&mut handles, "cg_a");

    assert_eq!(handles.len(), 1);
    assert_eq!(handles[0].cgroup, "cg_b");

    drain_all_payload_handles(&mut handles);
    assert!(handles.is_empty());
}

// -- Step::with_payload + Op::RunPayload/WaitPayload/KillPayload --

/// Step::with_payload emits a step whose ops consist of a single
/// Op::RunPayload carrying the supplied payload. Hold passes
/// through unchanged.
#[test]
fn step_with_payload_emits_runpayload_op() {
    use crate::test_support::{OutputFormat, Payload, PayloadKind};
    static FIO: Payload = Payload {
        name: "fio",
        kind: PayloadKind::Binary("fio"),
        output: OutputFormat::Json,
        default_args: &[],
        default_checks: &[],
        metrics: &[],
        include_files: &[],
        uses_parent_pgrp: false,
        known_flags: None,
        metric_bounds: None,
    };
    let step = Step::with_payload(&FIO, HoldSpec::fixed(Duration::from_millis(50)));
    assert_eq!(step.ops.len(), 1);
    match &step.ops[0] {
        Op::RunPayload {
            payload,
            args,
            cgroup,
        } => {
            assert_eq!(payload.name, "fio");
            assert!(args.is_empty());
            assert!(cgroup.is_none());
        }
        other => panic!("expected RunPayload, got {other:?}"),
    }
    assert!(matches!(step.hold, HoldSpec::Fixed(d) if d == Duration::from_millis(50)));
    assert!(matches!(&step.setup, Setup::Defs(d) if d.is_empty()));
}

/// Op convenience constructors — `run_payload`, `wait_payload`,
/// `kill_payload`, `run_payload_in_cgroup` — build the expected
/// enum shapes with the right field contents.
#[test]
fn op_payload_constructors_produce_expected_variants() {
    use crate::test_support::{OutputFormat, Payload, PayloadKind};
    static FIO: Payload = Payload {
        name: "fio",
        kind: PayloadKind::Binary("fio"),
        output: OutputFormat::Json,
        default_args: &[],
        default_checks: &[],
        metrics: &[],
        include_files: &[],
        uses_parent_pgrp: false,
        known_flags: None,
        metric_bounds: None,
    };

    let op = Op::run_payload(&FIO, ["--warmup"]);
    match op {
        Op::RunPayload {
            payload,
            args,
            cgroup,
        } => {
            assert_eq!(payload.name, "fio");
            assert_eq!(args, vec!["--warmup".to_string()]);
            assert!(cgroup.is_none());
        }
        other => panic!("expected RunPayload, got {other:?}"),
    }

    let op = Op::run_payload_in_cgroup(&FIO, [] as [&str; 0], "cg_0");
    match op {
        Op::RunPayload {
            payload,
            args,
            cgroup,
        } => {
            assert_eq!(payload.name, "fio");
            assert!(args.is_empty());
            assert_eq!(cgroup.as_deref(), Some("cg_0"));
        }
        other => panic!("expected RunPayload, got {other:?}"),
    }

    let op = Op::wait_payload("fio");
    assert!(matches!(
        op,
        Op::WaitPayload { ref name, ref cgroup } if name.as_ref() == "fio" && cgroup.is_none(),
    ));

    let op = Op::kill_payload("fio");
    assert!(matches!(
        op,
        Op::KillPayload { ref name, ref cgroup } if name.as_ref() == "fio" && cgroup.is_none(),
    ));

    let op = Op::wait_payload_in_cgroup("fio", "cg_0");
    assert!(matches!(
        op,
        Op::WaitPayload { ref name, cgroup: Some(ref c) } if name.as_ref() == "fio" && c.as_ref() == "cg_0",
    ));

    let op = Op::kill_payload_in_cgroup("fio", "cg_0");
    assert!(matches!(
        op,
        Op::KillPayload { ref name, cgroup: Some(ref c) } if name.as_ref() == "fio" && c.as_ref() == "cg_0",
    ));
}

/// Op::RunPayload rejects scheduler-kind payloads at apply time
/// with an actionable error message. The existing CgroupDef
/// path panics at builder time; the Op path runs at scenario
/// time and must bail instead of panicking so one bad step in
/// a sequence doesn't crash the harness.
#[test]
fn apply_ops_runpayload_rejects_scheduler_kind() {
    use crate::test_support::Payload;
    mock_setup_state!(mock, topo, ctx, state);
    let ops = vec![Op::RunPayload {
        payload: &Payload::KERNEL_DEFAULT,
        args: vec![],
        cgroup: None,
    }];
    let err = apply_ops_test(&ctx, &mut state, &ops).unwrap_err();
    let msg = format!("{err:#}");
    assert!(
        msg.contains("scheduler-kind Payload") && msg.contains("kernel_default"),
        "error must name the scheduler-kind reason AND the payload name, got: {msg}"
    );
    assert!(
        state.payload_handles.is_empty(),
        "no handle should be stored when RunPayload rejects the kind"
    );
}

// -- CgroupOps::place_task_during_handshake coverage tests --
//
// Pins that `Op::RunPayload { cgroup: Some(_), .. }` and
// `CgroupDef::workload(&payload)` route the
// placement-before-exec write through
// `CgroupOps::place_task_during_handshake`. A regression that
// bypasses the trait via a bare `std::fs::OpenOptions` write at
// `spawn_with_cgroup_sync` would defeat both the mock recording
// AND any future fault-injection or audit layered behind the
// trait.

/// Op::RunPayload with `cgroup: Some(name)` must produce
/// exactly ONE `PlaceTaskDuringHandshake(name, pid)` call on
/// the cgroups mock, where `pid > 0` (the child's real pid
/// the handshake notify pipe carried up to the parent).
#[test]
fn op_runpayload_writes_pid_to_named_cgroup_via_placement_trait() {
    use crate::test_support::{OutputFormat, Payload, PayloadKind};
    static TRUE_BIN: Payload = Payload {
        name: "true_bin_t_f_named",
        kind: PayloadKind::Binary("/bin/true"),
        output: OutputFormat::ExitCode,
        default_args: &[],
        default_checks: &[],
        metrics: &[],
        include_files: &[],
        uses_parent_pgrp: false,
        known_flags: None,
        metric_bounds: None,
    };
    mock_setup_state!(mock, topo, ctx, state);
    let ops = vec![Op::RunPayload {
        payload: &TRUE_BIN,
        args: vec![],
        cgroup: Some("cg_a".into()),
    }];
    apply_ops_test(&ctx, &mut state, &ops)
        .expect("apply_ops Op::RunPayload { cgroup: Some(_) } must succeed under MockCgroupOps");
    let calls = mock.calls();
    let placements: Vec<&CgroupCall> = calls
        .iter()
        .filter(|c| matches!(c, CgroupCall::PlaceTaskDuringHandshake(_, _)))
        .collect();
    assert_eq!(
        placements.len(),
        1,
        "Op::RunPayload {{ cgroup: Some(_), .. }} must record exactly one \
             PlaceTaskDuringHandshake call, got: {calls:?}"
    );
    match placements[0] {
        CgroupCall::PlaceTaskDuringHandshake(name, pid) => {
            assert_eq!(
                name, "cg_a",
                "PlaceTaskDuringHandshake must carry the user-facing cgroup name"
            );
            assert!(
                *pid > 0,
                "PlaceTaskDuringHandshake must carry the real child pid (>0), got {pid}"
            );
        }
        other => panic!("expected PlaceTaskDuringHandshake, got {other:?}"),
    }
    cleanup_state(&mut state);
}

/// Op::RunPayload with `cgroup: None` MUST NOT invoke the
/// placement trait at all — the child inherits the parent's
/// cgroup directly via `Command::spawn` without any handshake.
/// Recording any `PlaceTaskDuringHandshake` entry in this case
/// would mean the no-cgroup path accidentally walked through
/// the handshake pipes, breaking the documented short-circuit
/// on `build_command` when no cgroup is requested.
#[test]
fn op_runpayload_without_cgroup_does_not_place() {
    use crate::test_support::{OutputFormat, Payload, PayloadKind};
    static TRUE_BIN: Payload = Payload {
        name: "true_bin_t_f_none",
        kind: PayloadKind::Binary("/bin/true"),
        output: OutputFormat::ExitCode,
        default_args: &[],
        default_checks: &[],
        metrics: &[],
        include_files: &[],
        uses_parent_pgrp: false,
        known_flags: None,
        metric_bounds: None,
    };
    mock_setup_state!(mock, topo, ctx, state);
    let ops = vec![Op::RunPayload {
        payload: &TRUE_BIN,
        args: vec![],
        cgroup: None,
    }];
    apply_ops_test(&ctx, &mut state, &ops)
        .expect("apply_ops Op::RunPayload { cgroup: None } must succeed");
    let calls = mock.calls();
    let placements: Vec<&CgroupCall> = calls
        .iter()
        .filter(|c| matches!(c, CgroupCall::PlaceTaskDuringHandshake(_, _)))
        .collect();
    assert!(
        placements.is_empty(),
        "Op::RunPayload {{ cgroup: None, .. }} MUST NOT record any \
             PlaceTaskDuringHandshake calls; got: {calls:?}"
    );
    cleanup_state(&mut state);
}

/// `CgroupDef::named("cg_def").workload(&payload)` routes the
/// apply_setup-time payload spawn through the same trait. The
/// `def.name` (NOT a derived path) must be the string passed
/// to `place_task_during_handshake`.
#[test]
fn cgroupdef_workload_with_payload_places_in_def_name() {
    use crate::test_support::{OutputFormat, Payload, PayloadKind};
    static TRUE_BIN: Payload = Payload {
        name: "true_bin_t_f_defworkload",
        kind: PayloadKind::Binary("/bin/true"),
        output: OutputFormat::ExitCode,
        default_args: &[],
        default_checks: &[],
        metrics: &[],
        include_files: &[],
        uses_parent_pgrp: false,
        known_flags: None,
        metric_bounds: None,
    };
    mock_setup_state!(mock, topo, ctx, state);
    let defs = vec![CgroupDef::named("cg_def").workload(&TRUE_BIN)];
    apply_setup_test(&ctx, &mut state, &defs)
        .expect("apply_setup with CgroupDef::workload must spawn the payload");
    let calls = mock.calls();
    let placements: Vec<&CgroupCall> = calls
        .iter()
        .filter(|c| matches!(c, CgroupCall::PlaceTaskDuringHandshake(_, _)))
        .collect();
    assert_eq!(
        placements.len(),
        1,
        "CgroupDef::workload(&payload) must record exactly one \
             PlaceTaskDuringHandshake call, got: {calls:?}"
    );
    match placements[0] {
        CgroupCall::PlaceTaskDuringHandshake(name, pid) => {
            assert_eq!(
                name, "cg_def",
                "PlaceTaskDuringHandshake must carry CgroupDef::name verbatim"
            );
            assert!(
                *pid > 0,
                "PlaceTaskDuringHandshake must carry the real child pid (>0), got {pid}"
            );
        }
        other => panic!("expected PlaceTaskDuringHandshake, got {other:?}"),
    }
    cleanup_state(&mut state);
}

/// NEGATIVE test: the pcomm-coalesced spawn path in
/// `apply_setup` (around `WorkloadHandle::spawn_pcomm_cgroup`)
/// migrates workers via `move_tasks`, NOT through the
/// handshake placement trait. Pcomm spawns happen for
/// synthetic workers whose pids are placed post-spawn — no
/// pre_exec handshake runs, so `place_task_during_handshake`
/// MUST NOT fire for that path. A regression that routed
/// pcomm-coalesced spawns through the handshake would
/// confuse the freeze coordinator (the synthetic workers
/// don't need pre-exec placement) and break the existing
/// pcomm-grouping semantics.
#[test]
fn pcomm_coalesced_spawn_uses_move_tasks() {
    mock_setup_state!(mock, topo, ctx, state);
    // `CgroupDef::pcomm` writes the same pcomm string into
    // every WorkSpec in the cgroup, which routes the spawn
    // through `WorkloadHandle::spawn_pcomm_cgroup` (the
    // pcomm-coalesce branch at the top of `apply_setup`).
    // 2 workers under the same pcomm coalesce into one tgid
    // leader; the kernel placement still happens via
    // post-spawn `move_tasks`, NOT the handshake path.
    let defs = vec![CgroupDef::named("cg_pcomm").workers(2).pcomm("shared")];
    apply_setup_test(&ctx, &mut state, &defs)
        .expect("apply_setup with pcomm-coalesced workers must succeed");
    let calls = mock.calls();
    let placements: Vec<&CgroupCall> = calls
        .iter()
        .filter(|c| matches!(c, CgroupCall::PlaceTaskDuringHandshake(_, _)))
        .collect();
    assert!(
        placements.is_empty(),
        "pcomm-coalesced WorkSpec spawn MUST NOT route through \
             place_task_during_handshake (post-spawn move_tasks owns \
             worker placement); got: {calls:?}"
    );
    let move_tasks: Vec<&CgroupCall> = calls
        .iter()
        .filter(|c| matches!(c, CgroupCall::MoveTasks(name, _) if name == "cg_pcomm"))
        .collect();
    assert!(
        !move_tasks.is_empty(),
        "pcomm-coalesced WorkSpec spawn MUST record at least one \
             MoveTasks call for 'cg_pcomm' (post-spawn placement); \
             got: {calls:?}"
    );
    cleanup_state(&mut state);
}

/// `PlaceTaskDuringHandshake` MUST receive the user-facing
/// `cgroup` NAME, not a derived absolute path or a substring
/// of one. A regression that passed the resolved
/// `<parent>/<name>/cgroup.procs` substring would: (a) defeat
/// the `CgroupManager::place_task_during_handshake` path
/// derivation (it would double-join the path), (b) break the
/// `validate_cgroup_name` gate inside the trait
/// implementation (absolute-leading-`/` rejection), and (c)
/// make every recorded call name-specific to the host's
/// `/sys/fs/cgroup/ktstr` layout, useless for assertions on
/// the user-facing API.
#[test]
fn runpayload_placement_uses_def_name_not_resolved_path() {
    use crate::test_support::{OutputFormat, Payload, PayloadKind};
    static TRUE_BIN: Payload = Payload {
        name: "true_bin_t_f_namecheck",
        kind: PayloadKind::Binary("/bin/true"),
        output: OutputFormat::ExitCode,
        default_args: &[],
        default_checks: &[],
        metrics: &[],
        include_files: &[],
        uses_parent_pgrp: false,
        known_flags: None,
        metric_bounds: None,
    };
    mock_setup_state!(mock, topo, ctx, state);
    // MockCgroupOps's parent_path() is "/mock/cgroup" (see
    // MockCgroupOps::new). If the implementation passed the
    // resolved path (or any substring of it) the assertion
    // below would fail with a name containing '/' or 'mock'.
    let ops = vec![Op::RunPayload {
        payload: &TRUE_BIN,
        args: vec![],
        cgroup: Some("namecheck_cg".into()),
    }];
    apply_ops_test(&ctx, &mut state, &ops)
        .expect("apply_ops Op::RunPayload { cgroup: Some(_) } must succeed");
    let calls = mock.calls();
    let placement_name = calls
        .iter()
        .find_map(|c| match c {
            CgroupCall::PlaceTaskDuringHandshake(name, _) => Some(name.clone()),
            _ => None,
        })
        .expect("Op::RunPayload { cgroup: Some(_) } must record a PlaceTaskDuringHandshake");
    assert_eq!(
        placement_name, "namecheck_cg",
        "trait method must receive the raw user-facing cgroup name verbatim; \
             received {placement_name:?}"
    );
    assert!(
        !placement_name.contains('/'),
        "trait method must receive a bare name, not a path with '/': \
             received {placement_name:?}"
    );
    assert!(
        !placement_name.contains("mock"),
        "trait method must NOT receive any substring of the mock's parent path \
             (/mock/cgroup); received {placement_name:?}"
    );
    cleanup_state(&mut state);
}

/// Op::RunPayload with `cgroup: Some(name)` where `name` is NOT
/// registered in the test setup pins the CURRENT behavior:
/// resolve_cgroup_path validates the name's syntax but does not
/// verify the cgroup exists in the framework's tracked set, and
/// the trait method silently routes the placement against the
/// unknown name. A future tightening (warn on unknown / bail on
/// unknown — parity with Op::MoveAllTasks's diagnostic) MUST
/// re-pin this test to the new shape, surfacing the behavior
/// change in the diff rather than silently shipping it.
#[test]
fn op_runpayload_unknown_cgroup_currently_silently_places_via_trait() {
    use crate::test_support::{OutputFormat, Payload, PayloadKind};
    static TRUE_BIN: Payload = Payload {
        name: "true_bin_unknown_cg",
        kind: PayloadKind::Binary("/bin/true"),
        output: OutputFormat::ExitCode,
        default_args: &[],
        default_checks: &[],
        metrics: &[],
        include_files: &[],
        uses_parent_pgrp: false,
        known_flags: None,
        metric_bounds: None,
    };
    mock_setup_state!(mock, topo, ctx, state);
    let ops = vec![Op::RunPayload {
        payload: &TRUE_BIN,
        args: vec![],
        cgroup: Some("cg_never_added".into()),
    }];
    apply_ops_test(&ctx, &mut state, &ops)
        .expect("apply_ops must currently succeed even for an unregistered cgroup name");
    let calls = mock.calls();
    let placements: Vec<&CgroupCall> = calls
        .iter()
        .filter(|c| matches!(c, CgroupCall::PlaceTaskDuringHandshake(_, _)))
        .collect();
    assert_eq!(
        placements.len(),
        1,
        "current behavior: trait method fires against the unknown name \
             with no upfront framework-level check; got: {calls:?}"
    );
    match placements[0] {
        CgroupCall::PlaceTaskDuringHandshake(name, _) => {
            assert_eq!(
                name, "cg_never_added",
                "name must be the unknown name verbatim — no silent renaming or path-stripping"
            );
        }
        other => panic!("expected PlaceTaskDuringHandshake, got {other:?}"),
    }
    cleanup_state(&mut state);
}

/// Op::WaitPayload with no matching handle surfaces a descriptive
/// error rather than silently no-op'ing. Ditto KillPayload. A
/// silent no-op would let test authors wait for ghosts and pass
/// scenarios that never ran what they claim.
#[test]
fn apply_ops_wait_unknown_payload_bails() {
    mock_setup_state!(mock, topo, ctx, state);
    let err = apply_ops_test(
        &ctx,
        &mut state,
        &[Op::WaitPayload {
            name: "ghost".into(),
            cgroup: None,
        }],
    )
    .unwrap_err();
    let msg = format!("{err:#}");
    assert!(
        msg.contains("no running payload named 'ghost'"),
        "error must name the missing payload, got: {msg}"
    );
}

#[test]
fn apply_ops_kill_unknown_payload_bails() {
    mock_setup_state!(mock, topo, ctx, state);
    let err = apply_ops_test(
        &ctx,
        &mut state,
        &[Op::KillPayload {
            name: "ghost".into(),
            cgroup: None,
        }],
    )
    .unwrap_err();
    let msg = format!("{err:#}");
    assert!(
        msg.contains("no running payload named 'ghost'"),
        "error must name the missing payload, got: {msg}"
    );
}

// -- Scheduler-lifecycle Op dispatch error-path tests --
//
// The 4 scheduler-lifecycle Op variants (AttachScheduler /
// DetachScheduler / RestartScheduler / ReplaceScheduler) dispatch
// through `dispatch_attach_scheduler` / `dispatch_detach_scheduler`
// / `dispatch_restart_scheduler` / `dispatch_replace_scheduler`
// helpers in this module. Unit tests cover the error paths that
// don't require a real running scheduler: AttachScheduler with a
// missing staged binary, and Detach / Restart / Replace with no
// scheduler attached (SCHED_PID == 0). The success paths require
// a real spawn + libbpf attach, which the e2e VM integration
// suite exercises with scx-ktstr as the staged target.
//
// Every error path emits an actionable message naming the op
// AND the specific failure mode — copy-paste regression across
// the 4 arms surfaces as a distinct substring per arm.

/// Op::AttachScheduler against a Scheduler whose `name` doesn't
/// resolve to a staged binary file must bail with an actionable
/// error naming the missing path + suggesting the staging
/// pipeline check. EEVDF.name = "eevdf" — its `/staging/schedulers/
/// eevdf/scheduler` path does NOT exist in the test environment
/// (test harness has no initramfs mounted), so the inline
/// existence probe in `spawn_scheduler_from_paths` returns
/// `(None, None)` and the dispatch arm bails.
#[test]
fn apply_ops_attach_scheduler_bails_when_staged_binary_missing() {
    static SCHED: crate::test_support::Scheduler = crate::test_support::Scheduler::EEVDF;
    mock_setup_state!(mock, topo, ctx, state);
    // Install a no-op SnapshotBridge so
    // wait_for_worker_state_not_trying_or_bail's None-arm doesn't
    // fire ahead of the staging-pipeline error this test is
    // verifying. Mirrors the production path where the VM-
    // orchestrated bridge is installed before dispatch reaches
    // AttachScheduler. The no-op callback returns None (no
    // snapshot data); the wait helper's bridge presence check
    // sees Some(...) and the underlying wait returns Ok(0)
    // immediately because no worker is in TRYING state.
    let cb: crate::scenario::snapshot::CaptureCallback = std::sync::Arc::new(|_| None);
    let _bridge_guard = crate::scenario::snapshot::SnapshotBridge::new(cb).set_thread_local();
    let err = apply_ops_test(&ctx, &mut state, &[Op::attach_scheduler(&SCHED)]).unwrap_err();
    let msg = format!("{err:#}");
    assert!(
        msg.contains("Op::AttachScheduler"),
        "error must name the op (catches copy-paste regression across the 4 arms): {msg}"
    );
    assert!(
        msg.contains("staging") || msg.contains("staged"),
        "error must point at the staging pipeline so the operator knows where to look: {msg}"
    );
    assert!(
        msg.contains("eevdf"),
        "error must include the scheduler name so the operator can identify which entry: {msg}"
    );
}

/// Op::DetachScheduler with no scheduler currently attached
/// (SCHED_PID == 0 sentinel) bails with an actionable error.
/// Distinct from AttachScheduler's pin because each arm's error
/// message must be per-variant.
#[test]
fn apply_ops_detach_scheduler_bails_when_no_scheduler_attached() {
    mock_setup_state!(mock, topo, ctx, state);
    // Test environment has no scheduler spawned — SCHED_PID is 0.
    let err = apply_ops_test(&ctx, &mut state, &[Op::detach_scheduler()]).unwrap_err();
    let msg = format!("{err:#}");
    assert!(
        msg.contains("Op::DetachScheduler"),
        "error must name the op: {msg}"
    );
    assert!(
        msg.contains("no scheduler attached") || msg.contains("SCHED_PID"),
        "error must name the no-scheduler failure mode: {msg}"
    );
}

/// Op::RestartScheduler with no scheduler attached bails with
/// an actionable error.
#[test]
fn apply_ops_restart_scheduler_bails_when_no_scheduler_attached() {
    mock_setup_state!(mock, topo, ctx, state);
    let err = apply_ops_test(&ctx, &mut state, &[Op::restart_scheduler()]).unwrap_err();
    let msg = format!("{err:#}");
    assert!(
        msg.contains("Op::RestartScheduler"),
        "error must name the op: {msg}"
    );
    assert!(
        msg.contains("no scheduler attached") || msg.contains("SCHED_PID"),
        "error must name the no-scheduler failure mode: {msg}"
    );
}

/// Op::ReplaceScheduler with no scheduler attached bails BEFORE
/// attempting to spawn the replacement — the detach phase fails
/// fast on the SCHED_PID == 0 check so the operator sees the
/// "no scheduler to replace" error rather than a confusing
/// post-spawn diagnostic.
#[test]
fn apply_ops_replace_scheduler_bails_when_no_scheduler_attached() {
    static SCHED: crate::test_support::Scheduler = crate::test_support::Scheduler::EEVDF;
    mock_setup_state!(mock, topo, ctx, state);
    let err = apply_ops_test(&ctx, &mut state, &[Op::replace_scheduler(&SCHED)]).unwrap_err();
    let msg = format!("{err:#}");
    assert!(
        msg.contains("Op::ReplaceScheduler"),
        "error must name the op: {msg}"
    );
    assert!(
        msg.contains("no scheduler attached") || msg.contains("SCHED_PID"),
        "error must name the no-scheduler failure mode (detach phase fails fast): {msg}"
    );
}

/// `staged_scheduler_log_path` produces collision-free per-name
/// plus per-seq paths so successive Op::AttachScheduler or
/// Op::ReplaceScheduler dispatches with the SAME staged name
/// don't overwrite each other's logs. Pins the
/// `/tmp/sched_<name>_<seq>.log` scheme against a regression
/// that drops either the per-name keying or the per-seq seq
/// suffix.
#[test]
fn staged_scheduler_log_path_is_per_name_and_seq_keyed() {
    let a1 = staged_scheduler_log_path("scx_mitosis_a");
    let a2 = staged_scheduler_log_path("scx_mitosis_a");
    let b1 = staged_scheduler_log_path("scx_mitosis_b");
    // Same name on consecutive calls must produce distinct
    // paths via the seq suffix — protects against repeated
    // Op::ReplaceScheduler with the same staged name losing
    // the first spawn's failure-dump payload.
    assert_ne!(a1, a2, "same-name consecutive calls must differ via seq");
    // Different names must also differ — name keying defends
    // against parallel dispatch with distinct staged entries.
    assert_ne!(a1, b1, "different names must produce distinct paths");
    // Path shape: prefix + name + underscore + numeric seq +
    // .log. Asserts the seq suffix is purely numeric.
    assert!(
        a1.starts_with("/tmp/sched_scx_mitosis_a_"),
        "missing name + underscore prefix: {a1}"
    );
    assert!(a1.ends_with(".log"), "missing .log extension: {a1}");
    let seq_part = a1
        .strip_prefix("/tmp/sched_scx_mitosis_a_")
        .unwrap()
        .strip_suffix(".log")
        .unwrap();
    assert!(
        seq_part.chars().all(|c| c.is_ascii_digit()),
        "seq suffix must be all digits: {seq_part:?}"
    );
}

/// End-to-end on a real payload binary: Op::RunPayload spawns
/// a long-running `/bin/sleep`, Op::KillPayload matches by
/// payload.name and consumes the handle. The handle should
/// disappear from state.payload_handles so later teardown
/// drains don't double-consume.
#[test]
fn apply_ops_run_then_kill_consumes_handle() {
    use crate::test_support::{OutputFormat, Payload, PayloadKind};
    static SLEEP: Payload = Payload {
        // Name distinct from binary so the payload_name lookup
        // path is exercised against a non-basename key.
        name: "sleeper",
        kind: PayloadKind::Binary("/bin/sleep"),
        output: OutputFormat::ExitCode,
        default_args: &[],
        default_checks: &[],
        metrics: &[],
        include_files: &[],
        uses_parent_pgrp: false,
        known_flags: None,
        metric_bounds: None,
    };

    mock_setup_state!(mock, topo, ctx, state);
    apply_ops_test(&ctx, &mut state, &[Op::run_payload(&SLEEP, ["3600"])])
        .expect("spawn /bin/sleep");
    assert_eq!(state.payload_handles.len(), 1, "one payload is live");
    assert_eq!(state.payload_handles[0].handle.payload_name(), "sleeper");

    apply_ops_test(&ctx, &mut state, &[Op::kill_payload("sleeper")])
        .expect("kill the live payload");
    assert!(
        state.payload_handles.is_empty(),
        "handle must be consumed by KillPayload"
    );
}

/// Spawning a second payload with the same name while the first
/// is still live is a caller bug — the `WaitPayload`/
/// `KillPayload` lookup would hit the first match and leave the
/// second leaked. Reject at RunPayload time.
#[test]
fn apply_ops_run_duplicate_payload_name_bails() {
    use crate::test_support::{OutputFormat, Payload, PayloadKind};
    static SLEEP: Payload = Payload {
        name: "sleeper",
        kind: PayloadKind::Binary("/bin/sleep"),
        output: OutputFormat::ExitCode,
        default_args: &[],
        default_checks: &[],
        metrics: &[],
        include_files: &[],
        uses_parent_pgrp: false,
        known_flags: None,
        metric_bounds: None,
    };

    mock_setup_state!(mock, topo, ctx, state);
    apply_ops_test(&ctx, &mut state, &[Op::run_payload(&SLEEP, ["3600"])]).expect("first spawn");

    let err = apply_ops_test(&ctx, &mut state, &[Op::run_payload(&SLEEP, ["3600"])]).unwrap_err();
    let msg = format!("{err:#}");
    assert!(
        msg.contains("payload 'sleeper' already running"),
        "error must flag the duplicate, got: {msg}"
    );
    // The dup error must identify the surface that spawned the
    // live handle so the user knows where to go to fix it. The
    // first spawn was via Op::RunPayload, not CgroupDef::workload.
    assert!(
        msg.contains("Op::RunPayload"),
        "dup error must name the originating surface, got: {msg}"
    );
    // The Op::RunPayload in this test ran without a
    // `cgroup = Some(..)`, so the rendered cgroup key must be
    // `(no cgroup)`, not an empty-quoted `''`.
    assert!(
        msg.contains("(no cgroup)"),
        "empty-cgroup key must render as '(no cgroup)', got: {msg}"
    );
    assert!(
        !msg.contains("cgroup ''"),
        "empty-cgroup key must not render as quoted empty, got: {msg}"
    );
    assert_eq!(
        state.payload_handles.len(),
        1,
        "second spawn must not add a handle on failure"
    );

    // Clean up the live handle so the test process doesn't leak
    // a /bin/sleep.
    apply_ops_test(&ctx, &mut state, &[Op::kill_payload("sleeper")]).expect("teardown kill");
}

/// When the first spawn came from `CgroupDef::workload` in
/// `cg_def` and a subsequent `Op::run_payload_in_cgroup` targets
/// the same `cg_def` with the same payload name, the composite-
/// key dup check fires and names `CgroupDef::workload` as the
/// originating surface. A cross-cgroup duplicate (same name,
/// different cgroup) is legitimate and tested separately.
#[test]
fn apply_ops_run_rejects_payload_already_owned_by_cgroup_def() {
    use crate::test_support::{OutputFormat, Payload, PayloadKind};
    static SLEEP: Payload = Payload {
        name: "sleeper",
        kind: PayloadKind::Binary("/bin/sleep"),
        output: OutputFormat::ExitCode,
        default_args: &[],
        default_checks: &[],
        metrics: &[],
        include_files: &[],
        uses_parent_pgrp: false,
        known_flags: None,
        metric_bounds: None,
    };

    mock_setup_state!(mock, topo, ctx, state);
    // Simulate the def-owned handle directly — apply_setup pushes
    // entries with PayloadSource::CgroupDefWorkload, so construct
    // the equivalent here without invoking the real spawn path
    // (apply_setup needs workers(N) and cgroupfs ops which MockCgroupOps
    // does not implement for this test shape).
    let h = crate::scenario::payload_run::PayloadRun::new(&ctx, &SLEEP)
        .args(["3600".to_string()])
        .spawn()
        .expect("manual def-source spawn");
    state.payload_handles.push(PayloadEntry {
        cgroup: "def_cg".to_string(),
        source: PayloadSource::CgroupDefWorkload,
        handle: h,
    });

    // Targeting the SAME cgroup as the pre-existing entry: dup.
    let err = apply_ops_test(
        &ctx,
        &mut state,
        &[Op::run_payload_in_cgroup(&SLEEP, ["1"], "def_cg")],
    )
    .unwrap_err();
    let msg = format!("{err:#}");
    assert!(
        msg.contains("CgroupDef::workload"),
        "dup error must name the def-source surface, got: {msg}"
    );
    assert!(
        msg.contains("'def_cg'"),
        "dup error must name the cgroup the live handle is in, got: {msg}"
    );
    // Only the original handle remains — op branch bailed pre-spawn.
    assert_eq!(state.payload_handles.len(), 1);

    apply_ops_test(
        &ctx,
        &mut state,
        &[Op::kill_payload_in_cgroup("sleeper", "def_cg")],
    )
    .expect("teardown kill");
}

/// [`render_cgroup_key`] renders an empty string as
/// `(no cgroup)` and a populated name as single-quoted prose.
/// Pins the formatting so every error path that echoes the
/// cgroup key through this helper stays consistent.
#[test]
fn render_cgroup_key_handles_empty_and_populated() {
    assert_eq!(render_cgroup_key(""), "(no cgroup)");
    assert_eq!(render_cgroup_key("cg_a"), "'cg_a'");
}

// -- payload_handles drain on error paths in execute_steps_with --

/// An Err return from `execute_steps_with` (here: a vacuous
/// `HoldSpec::Frac(0.0)` caught by up-front validation — `Frac`
/// rejects `f <= 0.0` per `HoldSpec::validate`'s
/// `HoldSpec::Frac(f) if *f <= 0.0` arm, while `Fixed(ZERO)` is
/// deliberately valid for op-only settle steps per the same fn's
/// `HoldSpec::Fixed(_) => Ok(())` arm)
/// leaves no live payload_handles because no setup/ops ran.
/// Pins the invariant that the pre-ops validation path does
/// not spawn anything that could then leak.
#[test]
fn execute_steps_with_early_validation_err_has_nothing_to_drain() {
    use crate::cgroup::CgroupManager;
    let cgroups = CgroupManager::new("/nonexistent");
    let topo = mock_topo();
    let ctx = crate::scenario::Ctx::builder(&cgroups, &topo).build();
    let step = Step::new(vec![], HoldSpec::Frac(0.0));
    let err = execute_steps_with(&ctx, vec![step], None).unwrap_err();
    let msg = format!("{err:#}");
    assert!(
        msg.contains("hold validation") && msg.contains("Frac"),
        "expected pre-ops validation err, got: {msg}"
    );
}

/// When a live payload has been spawned and a later op returns
/// Err, the drain-on-err path consumes the payload handles via
/// `.kill()` (which emits metrics) rather than leaking them to
/// `PayloadHandle::drop` (which SIGKILLs without recording).
///
/// This test exercises the drain path directly by spawning a
/// /bin/sleep, then calling `apply_ops` with an op that forces
/// an error (unknown-name `WaitPayload`). After the Err, the
/// state's payload_handles must still be consulted by the
/// drain — verified by checking the live count before +
/// explicit teardown after.
#[test]
fn apply_ops_error_does_not_lose_live_payload_handles() {
    use crate::test_support::{OutputFormat, Payload, PayloadKind};
    static SLEEP: Payload = Payload {
        name: "sleeper_drain",
        kind: PayloadKind::Binary("/bin/sleep"),
        output: OutputFormat::ExitCode,
        default_args: &[],
        default_checks: &[],
        metrics: &[],
        include_files: &[],
        uses_parent_pgrp: false,
        known_flags: None,
        metric_bounds: None,
    };
    mock_setup_state!(mock, topo, ctx, state);
    apply_ops_test(&ctx, &mut state, &[Op::run_payload(&SLEEP, ["3600"])]).expect("spawn");
    assert_eq!(state.payload_handles.len(), 1);
    // Trigger an Err via WaitPayload on an unknown name. Before
    // the fix, execute_steps_with would propagate the Err via
    // `?` and leave the SLEEP handle to be SIGKILLed by Drop
    // (losing the metric emission).
    let err = apply_ops_test(&ctx, &mut state, &[Op::wait_payload("never_spawned")]).unwrap_err();
    assert!(
        format!("{err:#}").contains("no running payload named 'never_spawned'"),
        "expected wait-unknown-name err",
    );
    // The live handle is still in state — apply_ops itself does
    // not drain on Err (that's execute_steps_with's
    // responsibility). Manually drain via the helper to
    // terminate the child cleanly.
    drain_all_payload_handles(&mut state.payload_handles);
    assert!(state.payload_handles.is_empty());
}

// ---------------------------------------------------------------
// Step/Backdrop ruling invariants
// ---------------------------------------------------------------

/// `Op::RemoveCgroup` and `Op::StopCgroup` reach the cgroup ops
/// for Backdrop-owned targets from both step-local apply and
/// Backdrop's own setup pass. RemoveCgroup also drops the
/// Backdrop tracking entry so a later AddCgroup with the same
/// name does not collide against a stale slot.
///
/// Regression class: a future re-introduction of the Backdrop-
/// target rejection (e.g. as a "safety" re-add by a contributor
/// who didn't see the rationale) would surface here as the
/// `apply_ops` call returning Err. The framework intentionally
/// trades the early-bail for permissive removal — tests that
/// mistype a cgroup name will silently succeed at the
/// RemoveCgroup site and surface the typo later as a kernel-
/// layer `cgroup missing` error on the next op that references
/// the name.
#[test]
fn remove_and_stop_cgroup_permit_backdrop_target_from_step() {
    mock_setup_backdrop!(mock, topo, ctx, step_state, backdrop_state);
    backdrop_state
        .cgroups
        .add_cgroup_no_cpuset("bd_cg")
        .expect("add backdrop cgroup");

    {
        let mut scenario = ScenarioState::new(&mut step_state, &mut backdrop_state);
        apply_ops(&ctx, &mut scenario, &[Op::remove_cgroup("bd_cg")], false)
            .expect("step-local RemoveCgroup permitted against Backdrop target");
    }
    let calls = mock.calls();
    assert!(
        calls
            .iter()
            .any(|c| matches!(c, CgroupCall::RemoveCgroup(n) if n == "bd_cg")),
        "step-local remove must reach the cgroup ops, got: {calls:?}"
    );
    assert!(
        !backdrop_state.cgroups.names().iter().any(|n| n == "bd_cg"),
        "post-RemoveCgroup must drop backdrop tracking entry, got: {:?}",
        backdrop_state.cgroups.names()
    );

    // Slot is free — re-adding the same name must succeed.
    {
        let mut scenario = ScenarioState::new(&mut step_state, &mut backdrop_state);
        apply_ops(&ctx, &mut scenario, &[Op::add_cgroup("bd_cg")], false)
            .expect("AddCgroup with previously-removed name must succeed");
    }

    backdrop_state
        .cgroups
        .add_cgroup_no_cpuset("bd_cg_2")
        .expect("add second backdrop cgroup");
    {
        let mut scenario = ScenarioState::new(&mut step_state, &mut backdrop_state);
        apply_ops(&ctx, &mut scenario, &[Op::stop_cgroup("bd_cg_2")], false)
            .expect("step-local StopCgroup permitted against Backdrop target");
    }

    backdrop_state
        .cgroups
        .add_cgroup_no_cpuset("bd_cg_3")
        .expect("add third backdrop cgroup");
    {
        let mut scenario = ScenarioState::new(&mut step_state, &mut backdrop_state);
        scenario
            .with_target_backdrop(|s| apply_ops(&ctx, s, &[Op::remove_cgroup("bd_cg_3")], false))
            .expect("backdrop-pass RemoveCgroup permitted against Backdrop target");
    }

    cleanup_state(&mut step_state);
}

/// `Op::MoveAllTasks` from a step-local cgroup to a Backdrop
/// cgroup must transfer the handle from step-local slot to
/// backdrop slot so the worker survives the step boundary. A
/// step-to-step move keeps ownership step-local. A backdrop-to-
/// step move keeps the handle in the backdrop slot (persistent
/// does not degrade).
#[test]
fn move_all_tasks_transfers_handle_ownership_step_to_backdrop() {
    use crate::workload::{WorkSpec, WorkloadConfig, WorkloadHandle};

    mock_setup_backdrop!(mock, topo, ctx, step_state, backdrop_state);
    // Backdrop owns "bd_cg"; the step owns "step_cg" and a
    // handle keyed under it.
    backdrop_state
        .cgroups
        .add_cgroup_no_cpuset("bd_cg")
        .unwrap();
    step_state.cgroups.add_cgroup_no_cpuset("step_cg").unwrap();
    // Mirror production: WorkloadConfig::for_scenario_engine pins
    // Fork in the constructor — same code path apply_setup uses.
    let w = WorkSpec::default();
    let wl = WorkloadConfig::for_scenario_engine(
        &w,
        1,
        crate::workload::AffinityIntent::Inherit,
        w.work_type.clone(),
    )
    .expect(
        "test fixture: pcomm must stay None for scenario-engine dispatch — \
             if a future fixture variant sets pcomm, route via spawn_pcomm_cgroup \
             instead of for_scenario_engine",
    );
    let h = WorkloadHandle::spawn(&wl).expect("spawn worker");
    step_state.handles.push(("step_cg".to_string(), h));
    assert_eq!(step_state.handles.len(), 1);
    assert_eq!(backdrop_state.handles.len(), 0);

    // Move tasks from step_cg to bd_cg: ownership transfers.
    {
        let mut scenario = ScenarioState::new(&mut step_state, &mut backdrop_state);
        apply_ops(
            &ctx,
            &mut scenario,
            &[Op::move_all_tasks("step_cg", "bd_cg")],
            false,
        )
        .expect("move into backdrop");
    }
    assert_eq!(
        step_state.handles.len(),
        0,
        "step-local handle must leave the step slot after transfer",
    );
    assert_eq!(
        backdrop_state.handles.len(),
        1,
        "backdrop slot must receive the transferred handle",
    );
    assert_eq!(
        backdrop_state.handles[0].0, "bd_cg",
        "transferred handle must be re-keyed to `to`",
    );

    // Clear the handles before the test drops (handles SIGKILL on
    // drop — avoid leaking the worker process).
    backdrop_state.handles.clear();
    step_state.handles.clear();
}

/// Step→step move does NOT cross state slots (companion to the
/// step→backdrop transfer test above).
#[test]
fn move_all_tasks_step_to_step_keeps_step_ownership() {
    use crate::workload::{WorkSpec, WorkloadConfig, WorkloadHandle};

    mock_setup_backdrop!(mock, topo, ctx, step_state, backdrop_state);
    step_state.cgroups.add_cgroup_no_cpuset("src").unwrap();
    step_state.cgroups.add_cgroup_no_cpuset("dst").unwrap();
    // Same Fork invariant as the step→backdrop transfer test
    // above; production scenario-engine synthesis routes through
    // for_scenario_engine which pins Fork.
    let w = WorkSpec::default();
    let wl = WorkloadConfig::for_scenario_engine(
        &w,
        1,
        crate::workload::AffinityIntent::Inherit,
        w.work_type.clone(),
    )
    .expect(
        "test fixture: pcomm must stay None for scenario-engine dispatch — \
             if a future fixture variant sets pcomm, route via spawn_pcomm_cgroup \
             instead of for_scenario_engine",
    );
    let h = WorkloadHandle::spawn(&wl).expect("spawn");
    step_state.handles.push(("src".to_string(), h));
    {
        let mut scenario = ScenarioState::new(&mut step_state, &mut backdrop_state);
        apply_ops(
            &ctx,
            &mut scenario,
            &[Op::move_all_tasks("src", "dst")],
            false,
        )
        .expect("step-to-step move");
    }
    assert_eq!(step_state.handles.len(), 1);
    assert_eq!(step_state.handles[0].0, "dst");
    assert_eq!(backdrop_state.handles.len(), 0);
    step_state.handles.clear();
}

/// A step-local `Op::MoveAllTasks` that
/// pulls from a Backdrop-owned cgroup into a step-local cgroup
/// must bail before touching cgroupfs. The persistent worker
/// would otherwise be stranded in a cgroup that gets rmdir'd at
/// the step boundary. Backdrop-setup ops (`target_backdrop`)
/// stay exempt.
#[test]
fn move_all_tasks_backdrop_to_step_rejected() {
    mock_setup_backdrop!(mock, topo, ctx, step_state, backdrop_state);
    backdrop_state.cgroups.add_cgroup_no_cpuset("bd").unwrap();
    step_state.cgroups.add_cgroup_no_cpuset("step").unwrap();

    let mut scenario = ScenarioState::new(&mut step_state, &mut backdrop_state);
    let err = apply_ops(
        &ctx,
        &mut scenario,
        &[Op::move_all_tasks("bd", "step")],
        false,
    )
    .unwrap_err();
    let msg = format!("{err:#}");
    assert!(
        msg.contains("Backdrop-owned 'bd'") && msg.contains("step-local 'step'"),
        "error must name both cgroups and the direction, got: {msg}"
    );
    // The mock must not have seen a cgroup.procs write — the
    // guard bails before any kernel-side work.
    let calls = mock.calls();
    assert!(
        !calls
            .iter()
            .any(|c| matches!(c, CgroupCall::MoveTasks(_, _))),
        "pre-bail path must not invoke move_tasks, got: {calls:?}"
    );
}

/// `Op::MoveAllTasks` with `from == to` is a silent kernel
/// no-op (cgroup.procs is idempotent on same-cgroup writes).
/// The handler bails so the test author either deletes the
/// stale op or fixes the typo that collapsed both sides
/// onto the same name; the diagnostic includes both names
/// so the operator can choose which to change. Pin the bail
/// and the no-kernel-side-effect contract so a future refactor
/// that drops the self-move check surfaces here.
#[test]
fn move_all_tasks_self_move_bails() {
    mock_setup_backdrop!(mock, topo, ctx, step_state, backdrop_state);
    step_state.cgroups.add_cgroup_no_cpuset("cg").unwrap();
    let mut scenario = ScenarioState::new(&mut step_state, &mut backdrop_state);
    let err = apply_ops(
        &ctx,
        &mut scenario,
        &[Op::move_all_tasks("cg", "cg")],
        false,
    )
    .unwrap_err();
    let msg = format!("{err:#}");
    assert!(
        msg.contains("Op::MoveAllTasks") && msg.contains("self-move"),
        "error must name the op + the self-move shape, got: {msg}"
    );
    assert!(
        msg.contains("'cg'"),
        "error must quote the colliding name so the operator sees which side to change, got: {msg}"
    );
    // No kernel-side writes — the bail fires before any
    // clear_subtree_control / move_tasks call.
    let calls = mock.calls();
    assert!(
        !calls
            .iter()
            .any(|c| matches!(c, CgroupCall::MoveTasks(_, _))),
        "self-move bail must not invoke move_tasks, got: {calls:?}"
    );
}

/// Symmetric self-move check on the empty-string
/// RunnerCgroup-placement key: `Op::move_all_tasks("", "")` is
/// also a no-op (would "move" RunnerCgroup-placement workers to
/// themselves) and bails on the same `from == to` path. Pin the
/// empty-string case so a future refactor that special-cases
/// empty doesn't regress.
#[test]
fn move_all_tasks_self_move_bails_on_empty_string_pair() {
    mock_setup_backdrop!(mock, topo, ctx, step_state, backdrop_state);
    let mut scenario = ScenarioState::new(&mut step_state, &mut backdrop_state);
    let err = apply_ops(&ctx, &mut scenario, &[Op::move_all_tasks("", "")], false).unwrap_err();
    let msg = format!("{err:#}");
    assert!(
        msg.contains("self-move"),
        "empty-string pair must hit the same self-move bail path, got: {msg}"
    );
}

/// `Op::MoveAllTasks` to a destination cgroup not in either
/// tracking set emits a typo-late-surfacing warn that dumps
/// both backdrop and step lists so the operator can compare
/// against the test source. Mirrors `Op::RemoveCgroup`'s
/// branch-2 warn. The move STILL ATTEMPTS the kernel write
/// (operator may be targeting an externally-managed cgroup);
/// the warn is informational. Pin both the warn fields and
/// the post-warn attempt so a future refactor that converts
/// the warn to a bail surfaces here.
#[test]
fn move_all_tasks_unknown_dst_warn_dumps_tracked_cgroups() {
    let events = capture_tracing_events(|| {
        mock_setup_backdrop!(mock, topo, ctx, step_state, backdrop_state);
        backdrop_state
            .cgroups
            .add_cgroup_no_cpuset("bd_real_name")
            .expect("add backdrop cgroup");
        step_state
            .cgroups
            .add_cgroup_no_cpuset("step_local_real")
            .expect("add step-local cgroup");
        // No handle keyed under "src" — the handler will
        // emit the warn for the unknown dst, then collect
        // zero matching handles, then the empty pid_batches
        // loop exits without any move_tasks calls. The
        // mock's clear_subtree_control records the "dts"
        // target so we can confirm the move was attempted.
        let mut scenario = ScenarioState::new(&mut step_state, &mut backdrop_state);
        apply_ops(
            &ctx,
            &mut scenario,
            &[Op::move_all_tasks("src", "dts")],
            false,
        )
        .expect("typo'd-dst MoveAllTasks must succeed (warn-then-attempt)");
    });
    let warns: Vec<&(tracing::Level, String)> = events
        .iter()
        .filter(|(lvl, _)| *lvl == tracing::Level::WARN)
        .collect();
    assert_eq!(
        warns.len(),
        1,
        "exactly one typo-warn expected on unknown dst; got: {warns:?}",
    );
    let body = &warns[0].1;
    assert!(
        body.contains("matches no step-local"),
        "warn must use the typo-late-surfacing phrasing, got: {body:?}",
    );
    assert!(
        body.contains("Op::MoveAllTasks"),
        "warn must name the op so the operator can grep, got: {body:?}",
    );
    assert!(
        body.contains("dts"),
        "warn must include the typo'd destination name, got: {body:?}",
    );
    assert!(
        body.contains("bd_real_name"),
        "warn must dump backdrop_cgroups so the operator sees real names, got: {body:?}",
    );
    assert!(
        body.contains("step_local_real"),
        "warn must dump step_cgroups so the operator sees real names, got: {body:?}",
    );
}

/// The typo-warn does NOT fire when `to` matches a tracked
/// step-local cgroup. Pin the suppression so a future refactor
/// that inverts the predicate (warning on the happy path)
/// surfaces here.
#[test]
fn move_all_tasks_emits_no_typo_warn_when_dst_tracked() {
    let events = capture_tracing_events(|| {
        mock_setup_backdrop!(mock, topo, ctx, step_state, backdrop_state);
        step_state.cgroups.add_cgroup_no_cpuset("src").unwrap();
        step_state.cgroups.add_cgroup_no_cpuset("dst").unwrap();
        let mut scenario = ScenarioState::new(&mut step_state, &mut backdrop_state);
        apply_ops(
            &ctx,
            &mut scenario,
            &[Op::move_all_tasks("src", "dst")],
            false,
        )
        .expect("tracked-dst MoveAllTasks must succeed without warn");
    });
    // Filter on diagnostic-class tokens that the typo-late-
    // surfacing contract owns ("Op::MoveAllTasks" + "matches
    // no") rather than the specific "destination" word: a
    // future warn-text refactor that keeps the diagnostic class
    // intact but rewords the surrounding prose still trips this
    // filter, so the suppression contract isn't bound to one
    // brittle string.
    let move_warns: Vec<&(tracing::Level, String)> = events
        .iter()
        .filter(|(lvl, body)| {
            *lvl == tracing::Level::WARN
                && body.contains("Op::MoveAllTasks")
                && body.contains("matches no")
        })
        .collect();
    assert!(
        move_warns.is_empty(),
        "happy-path move must emit zero typo-warns, got: {move_warns:?}",
    );
}

/// Companion to `move_all_tasks_self_move_bails`: pins the
/// bail-before-handle-walk ordering. The bare self-move test
/// exercises only the no-handle
/// path; a regression that reordered the self-move check
/// after `pid_batches` collection would let `move_tasks`
/// fire on the same src/dst pair (still kernel-side a no-op
/// per the kernel-source grounding, but the burn-cycle waste
/// plus the freeze-window blast radius the bail exists to
/// prevent would silently regress). Push a real handle keyed
/// under "cg" first so the regression would observably call
/// the mock; pin that the bail still fires with zero mock
/// MoveTasks calls.
#[test]
fn move_all_tasks_self_move_bails_with_handles_present_no_kernel_call() {
    use crate::workload::{WorkSpec, WorkloadConfig, WorkloadHandle};
    mock_setup_backdrop!(mock, topo, ctx, step_state, backdrop_state);
    step_state.cgroups.add_cgroup_no_cpuset("cg").unwrap();
    let w = WorkSpec::default();
    let wl = WorkloadConfig::for_scenario_engine(
        &w,
        1,
        crate::workload::AffinityIntent::Inherit,
        w.work_type.clone(),
    )
    .expect(
        "test fixture: pcomm must stay None for scenario-engine dispatch — \
             if a future fixture variant sets pcomm, route via spawn_pcomm_cgroup \
             instead of for_scenario_engine",
    );
    let h = WorkloadHandle::spawn(&wl).expect("spawn");
    step_state.handles.push(("cg".to_string(), h));
    let mut scenario = ScenarioState::new(&mut step_state, &mut backdrop_state);
    let err = apply_ops(
        &ctx,
        &mut scenario,
        &[Op::move_all_tasks("cg", "cg")],
        false,
    )
    .unwrap_err();
    let msg = format!("{err:#}");
    assert!(
        msg.contains("self-move"),
        "self-move bail must fire ahead of the handle walk, got: {msg}",
    );
    let calls = mock.calls();
    assert!(
        !calls
            .iter()
            .any(|c| matches!(c, CgroupCall::MoveTasks(_, _))),
        "bail-before-handle-walk ordering must skip every move_tasks call \
             even with a matching handle present, got: {calls:?}"
    );
    step_state.handles.clear();
}

/// Companion to `move_all_tasks_unknown_dst_warn_dumps_tracked_cgroups`:
/// pins the warn-is-informational contract — the post-warn
/// `move_tasks(typo_dst, pids)` actually fires against the
/// kernel side. The bare warn test has no handles keyed
/// under "src", so `pid_batches` is empty and `move_tasks`
/// is never invoked; that path can't observe the post-warn
/// attempt the handler doc comment claims. Push a real
/// handle keyed under "src" so the warn-then-attempt
/// contract is observable: the warn fires, then move_tasks
/// fires against the typo'd dst (the mock records the call;
/// in production the kernel would return ENOENT, but the
/// pin here is the *attempt*, not the kernel outcome).
#[test]
fn move_all_tasks_unknown_dst_with_handles_does_attempt_kernel_call() {
    use crate::workload::{WorkSpec, WorkloadConfig, WorkloadHandle};
    mock_setup_backdrop!(mock, topo, ctx, step_state, backdrop_state);
    backdrop_state
        .cgroups
        .add_cgroup_no_cpuset("bd_real_name")
        .expect("add backdrop cgroup");
    step_state.cgroups.add_cgroup_no_cpuset("src").unwrap();
    let w = WorkSpec::default();
    let wl = WorkloadConfig::for_scenario_engine(
        &w,
        1,
        crate::workload::AffinityIntent::Inherit,
        w.work_type.clone(),
    )
    .expect(
        "test fixture: pcomm must stay None for scenario-engine dispatch — \
             if a future fixture variant sets pcomm, route via spawn_pcomm_cgroup \
             instead of for_scenario_engine",
    );
    let h = WorkloadHandle::spawn(&wl).expect("spawn");
    step_state.handles.push(("src".to_string(), h));
    let mut scenario = ScenarioState::new(&mut step_state, &mut backdrop_state);
    // The mock's move_tasks succeeds unconditionally (no
    // kernel ENOENT — that's a real-kernel concern). The
    // pin is that the call HAPPENS despite the warn.
    apply_ops(
        &ctx,
        &mut scenario,
        &[Op::move_all_tasks("src", "dts")],
        false,
    )
    .expect("warn-then-attempt path must succeed against the mock");
    let calls = mock.calls();
    let move_calls: Vec<&CgroupCall> = calls
        .iter()
        .filter(|c| matches!(c, CgroupCall::MoveTasks(_, _)))
        .collect();
    assert_eq!(
        move_calls.len(),
        1,
        "warn-is-informational contract: exactly one move_tasks call \
             against the typo'd dst, got: {calls:?}",
    );
    match &move_calls[0] {
        CgroupCall::MoveTasks(dst, _) => assert_eq!(
            dst, "dts",
            "the warn doesn't intercept; move_tasks fires against the typo'd dst verbatim",
        ),
        _ => unreachable!(),
    }
    step_state.handles.clear();
}

/// `run_scenario` rejects a scheduler-kind payload in
/// `Backdrop::payloads` before running any setup.
#[test]
fn run_scenario_rejects_scheduler_kind_backdrop_payload() {
    use crate::cgroup::CgroupManager;
    use crate::test_support::Payload;
    let cgroups = CgroupManager::new("/nonexistent");
    let topo = mock_topo();
    let ctx = crate::scenario::Ctx::builder(&cgroups, &topo).build();
    let backdrop =
        crate::scenario::backdrop::Backdrop::new().push_payload(&Payload::KERNEL_DEFAULT);
    let err = execute_scenario_with(
        &ctx,
        backdrop,
        vec![Step::new(vec![], HoldSpec::fixed(Duration::from_millis(1)))],
        None,
    )
    .unwrap_err();
    let msg = format!("{err:#}");
    assert!(
        msg.contains("scheduler-kind") && msg.contains("Backdrop"),
        "error must name the kind mismatch and the Backdrop surface, got: {msg}"
    );
}

/// `apply_setup` rejects a step-local CgroupDef whose name
/// collides with a Backdrop-tracked cgroup.
#[test]
fn apply_setup_rejects_name_collision_with_backdrop() {
    mock_setup_backdrop!(mock, topo, ctx, step_state, backdrop_state);
    backdrop_state
        .cgroups
        .add_cgroup_no_cpuset("shared")
        .unwrap();
    let defs = vec![CgroupDef::named("shared").workers(1)];
    let mut scenario = ScenarioState::new(&mut step_state, &mut backdrop_state);
    let err = apply_setup(&ctx, &mut scenario, &defs).unwrap_err();
    let msg = format!("{err:#}");
    assert!(
        msg.contains("already tracked") && msg.contains("shared"),
        "error must cite the collision and the offending name, got: {msg}"
    );
    cleanup_state(&mut step_state);
}

// ---------------------------------------------------------------
// composite-key (name, cgroup) dedup for Op::RunPayload
// ---------------------------------------------------------------

/// Push a synthetic live PayloadEntry into `state`'s step slot
/// so tests can exercise dedup / lookup paths without paying
/// the cost of a real cgroupfs-backed spawn (which fails inside
/// the MockCgroupOps test harness because `/mock/cgroup/...`
/// doesn't exist on disk).
fn push_fake_payload_entry<'a>(
    ctx: &'a Ctx<'a>,
    state: &mut StepState<'a>,
    payload: &'static crate::test_support::Payload,
    cgroup: &str,
    source: PayloadSource,
) {
    let h = crate::scenario::payload_run::PayloadRun::new(ctx, payload)
        .args(["3600".to_string()])
        .spawn()
        .expect("manual spawn (no cgroup placement)");
    state.payload_handles.push(PayloadEntry {
        cgroup: cgroup.to_string(),
        source,
        handle: h,
    });
}

/// Same payload live in `cg_a` AND `cg_b`; a third
/// `Op::RunPayload` targeting a brand-new `cg_c` must NOT trip
/// the composite-key dedup because the (name, cgroup) pair is
/// fresh. Simulated via direct state injection so the test
/// doesn't depend on cgroupfs.
#[test]
fn apply_ops_run_duplicate_name_different_cgroups_allowed() {
    use crate::test_support::{OutputFormat, Payload, PayloadKind};
    static SLEEP: Payload = Payload {
        name: "sleeper",
        kind: PayloadKind::Binary("/bin/sleep"),
        output: OutputFormat::ExitCode,
        default_args: &[],
        default_checks: &[],
        metrics: &[],
        include_files: &[],
        uses_parent_pgrp: false,
        known_flags: None,
        metric_bounds: None,
    };
    mock_setup_state!(mock, topo, ctx, state);
    push_fake_payload_entry(
        &ctx,
        &mut state,
        &SLEEP,
        "cg_a",
        PayloadSource::OpRunPayload,
    );
    push_fake_payload_entry(
        &ctx,
        &mut state,
        &SLEEP,
        "cg_b",
        PayloadSource::OpRunPayload,
    );

    let mut backdrop = BackdropState::empty(&ctx);
    let scenario = ScenarioState::new(&mut state, &mut backdrop);
    // The `find_live_payload_with_cgroup` lookup for ("sleeper", "cg_c")
    // returns None because no live entry matches that pair — so
    // the dup check passes and run_scenario would let the spawn
    // proceed. We check the lookup directly (spawning against
    // MockCgroupOps would fail on the pre_exec cgroup write).
    assert!(
        scenario
            .find_live_payload_with_cgroup("sleeper", "cg_c")
            .is_none(),
        "fresh (name, cgroup) pair must not collide with live entries in other cgroups",
    );
    // And the existing same-cgroup entry still collides.
    assert!(
        scenario
            .find_live_payload_with_cgroup("sleeper", "cg_a")
            .is_some(),
        "same (name, cgroup) still matches — only the pair matters",
    );

    cleanup_state(&mut state);
}

/// `take_payload_by_name` in composite mode matches only the
/// exact `(name, cgroup)` pair and leaves sibling copies alone.
#[test]
fn take_payload_by_composite_key_matches_exact_cgroup() {
    use crate::test_support::{OutputFormat, Payload, PayloadKind};
    static SLEEP: Payload = Payload {
        name: "sleeper",
        kind: PayloadKind::Binary("/bin/sleep"),
        output: OutputFormat::ExitCode,
        default_args: &[],
        default_checks: &[],
        metrics: &[],
        include_files: &[],
        uses_parent_pgrp: false,
        known_flags: None,
        metric_bounds: None,
    };
    mock_setup_state!(mock, topo, ctx, state);
    push_fake_payload_entry(
        &ctx,
        &mut state,
        &SLEEP,
        "cg_a",
        PayloadSource::OpRunPayload,
    );
    push_fake_payload_entry(
        &ctx,
        &mut state,
        &SLEEP,
        "cg_b",
        PayloadSource::OpRunPayload,
    );

    let mut backdrop = BackdropState::empty(&ctx);
    let mut scenario = ScenarioState::new(&mut state, &mut backdrop);
    let taken = scenario
        .take_payload_by_name("sleeper", Some("cg_a"))
        .expect("composite lookup does not bail on ambiguity")
        .expect("one entry matches");
    assert_eq!(taken.cgroup, "cg_a");
    // The cg_b entry survives.
    assert_eq!(state.payload_handles.len(), 1);
    assert_eq!(state.payload_handles[0].cgroup, "cg_b");
    // Drain to avoid leaking the live child.
    drain_all_payload_handles(&mut state.payload_handles);
    let _ = taken.handle.kill();
}

/// Bare `take_payload_by_name(name, None)` returns
/// `Err(ambiguous_cgroups)` when two or more copies are live,
/// surfacing both cgroup keys so the caller can disambiguate.
#[test]
fn take_payload_by_bare_name_reports_ambiguous_cgroups() {
    use crate::test_support::{OutputFormat, Payload, PayloadKind};
    static SLEEP: Payload = Payload {
        name: "sleeper",
        kind: PayloadKind::Binary("/bin/sleep"),
        output: OutputFormat::ExitCode,
        default_args: &[],
        default_checks: &[],
        metrics: &[],
        include_files: &[],
        uses_parent_pgrp: false,
        known_flags: None,
        metric_bounds: None,
    };
    mock_setup_state!(mock, topo, ctx, state);
    push_fake_payload_entry(
        &ctx,
        &mut state,
        &SLEEP,
        "cg_a",
        PayloadSource::OpRunPayload,
    );
    push_fake_payload_entry(
        &ctx,
        &mut state,
        &SLEEP,
        "cg_b",
        PayloadSource::OpRunPayload,
    );

    let mut backdrop = BackdropState::empty(&ctx);
    let mut scenario = ScenarioState::new(&mut state, &mut backdrop);
    let err = match scenario.take_payload_by_name("sleeper", None) {
        Err(cgroups) => cgroups,
        Ok(_) => panic!("bare lookup over multi-copy must surface ambiguity"),
    };
    assert_eq!(err.len(), 2);
    assert!(err.contains(&"cg_a".to_string()) && err.contains(&"cg_b".to_string()));
    // No handle consumed — both still live.
    assert_eq!(state.payload_handles.len(), 2);
    drain_all_payload_handles(&mut state.payload_handles);
}

/// Bare `take_payload_by_name(name, None)` succeeds when
/// exactly one copy is live, so `Op::wait_payload(name)` and
/// `Op::kill_payload(name)` don't need to carry a cgroup
/// argument in the single-copy case.
#[test]
fn take_payload_by_bare_name_succeeds_on_single_copy() {
    use crate::test_support::{OutputFormat, Payload, PayloadKind};
    static SLEEP: Payload = Payload {
        name: "sleeper",
        kind: PayloadKind::Binary("/bin/sleep"),
        output: OutputFormat::ExitCode,
        default_args: &[],
        default_checks: &[],
        metrics: &[],
        include_files: &[],
        uses_parent_pgrp: false,
        known_flags: None,
        metric_bounds: None,
    };
    mock_setup_state!(mock, topo, ctx, state);
    push_fake_payload_entry(
        &ctx,
        &mut state,
        &SLEEP,
        "cg_a",
        PayloadSource::OpRunPayload,
    );

    let mut backdrop = BackdropState::empty(&ctx);
    let mut scenario = ScenarioState::new(&mut state, &mut backdrop);
    let taken = scenario
        .take_payload_by_name("sleeper", None)
        .expect("single-copy bare lookup returns Ok")
        .expect("one entry matches");
    assert_eq!(taken.cgroup, "cg_a");
    assert!(state.payload_handles.is_empty());
    let _ = taken.handle.kill();
}

/// The apply_ops ambiguity hint must spell the full snake_case
/// constructor path so a user copying the hint into source
/// writes something that actually compiles. Covers both
/// `Op::wait_payload` and `Op::kill_payload` entry points
/// because they route through the same helper.
#[test]
fn apply_ops_bare_wait_and_kill_ambiguity_hint_names_full_constructor() {
    use crate::test_support::{OutputFormat, Payload, PayloadKind};
    static SLEEP: Payload = Payload {
        name: "sleeper",
        kind: PayloadKind::Binary("/bin/sleep"),
        output: OutputFormat::ExitCode,
        default_args: &[],
        default_checks: &[],
        metrics: &[],
        include_files: &[],
        uses_parent_pgrp: false,
        known_flags: None,
        metric_bounds: None,
    };
    mock_setup!(mock, topo, ctx);

    // WaitPayload path.
    let mut state = StepState::empty(&ctx);
    push_fake_payload_entry(
        &ctx,
        &mut state,
        &SLEEP,
        "cg_a",
        PayloadSource::OpRunPayload,
    );
    push_fake_payload_entry(
        &ctx,
        &mut state,
        &SLEEP,
        "cg_b",
        PayloadSource::OpRunPayload,
    );
    let err = apply_ops_test(&ctx, &mut state, &[Op::wait_payload("sleeper")]).unwrap_err();
    let msg = format!("{err:#}");
    assert!(
        msg.contains("ambiguous"),
        "wait ambiguity message must flag ambiguity, got: {msg}"
    );
    assert!(
        msg.contains("Op::wait_payload_in_cgroup(name, cgroup)"),
        "wait ambiguity hint must name the full snake_case constructor \
             so a copy-paste into source compiles, got: {msg}"
    );
    drain_all_payload_handles(&mut state.payload_handles);

    // KillPayload path.
    let mut state = StepState::empty(&ctx);
    push_fake_payload_entry(
        &ctx,
        &mut state,
        &SLEEP,
        "cg_a",
        PayloadSource::OpRunPayload,
    );
    push_fake_payload_entry(
        &ctx,
        &mut state,
        &SLEEP,
        "cg_b",
        PayloadSource::OpRunPayload,
    );
    let err = apply_ops_test(&ctx, &mut state, &[Op::kill_payload("sleeper")]).unwrap_err();
    let msg = format!("{err:#}");
    assert!(
        msg.contains("Op::kill_payload_in_cgroup(name, cgroup)"),
        "kill ambiguity hint must name the full snake_case constructor, got: {msg}"
    );
    drain_all_payload_handles(&mut state.payload_handles);
}

/// The not-found arm uses `-ing` verb form ("before waiting" /
/// "before killing"), not the collapsed single-word lowercase
/// a previous implementation emitted.
#[test]
fn apply_ops_not_found_message_uses_gerund_verb() {
    mock_setup_state!(mock, topo, ctx, state);
    let err = apply_ops_test(&ctx, &mut state, &[Op::wait_payload("ghost")]).unwrap_err();
    let msg = format!("{err:#}");
    assert!(
        msg.contains("before waiting"),
        "wait not-found message must say 'before waiting', got: {msg}"
    );
    assert!(
        !msg.contains("before waitpayload"),
        "must not collapse 'wait payload' into 'waitpayload', got: {msg}"
    );

    let err = apply_ops_test(&ctx, &mut state, &[Op::kill_payload("ghost")]).unwrap_err();
    let msg = format!("{err:#}");
    assert!(
        msg.contains("before killing"),
        "kill not-found message must say 'before killing', got: {msg}"
    );
}

// ---------------------------------------------------------------
// Step-local vs Backdrop state invariants
// ---------------------------------------------------------------

/// Op::RemoveCgroup prunes the name from CgroupGroup's tracked
/// `names` vec via the `forget` helper BEFORE dispatching
/// `ctx.cgroups.remove_cgroup`. Without the prune, the stale
/// tracking entry would re-trigger the AddCgroup collision check
/// for a same-name re-create, and CgroupGroup's Drop would invoke
/// a redundant rmdir against an already-removed dir. Pin both:
/// names() reflects only the surviving cgroup, and the mock
/// observes exactly one RemoveCgroup call for the dropped name
/// (from the Op — Drop does not see it).
#[test]
fn remove_cgroup_forgets_name_so_drop_does_not_double_rmdir() {
    mock_setup_state!(mock, topo, ctx, state);
    apply_ops_test(
        &ctx,
        &mut state,
        &[Op::add_cgroup("cg_keep"), Op::add_cgroup("cg_drop")],
    )
    .unwrap();
    // Op::RemoveCgroup records on the mock AND prunes `cg_drop`
    // from the tracked names — only `cg_keep` survives.
    apply_ops_test(&ctx, &mut state, &[Op::remove_cgroup("cg_drop")]).unwrap();
    assert_eq!(
        state.cgroups.names(),
        &["cg_keep".to_string()],
        "Op::RemoveCgroup must prune the removed name from \
             CgroupGroup::names so a later AddCgroup with the same \
             name can re-create the cgroup without colliding against \
             a stale tracking entry",
    );
    // After Drop, the mock observed exactly one RemoveCgroup
    // call for cg_drop (from the Op itself). Drop iterates only
    // surviving names so it does not re-issue rmdir against
    // cg_drop, and it issues exactly one rmdir for the
    // surviving cg_keep.
    drop(state);
    let calls = mock.calls();
    let cg_drop_removes: Vec<&CgroupCall> = calls
        .iter()
        .filter(|c| matches!(c, CgroupCall::RemoveCgroup(n) if n == "cg_drop"))
        .collect();
    assert_eq!(
        cg_drop_removes.len(),
        1,
        "Op::RemoveCgroup must be the sole rmdir dispatcher for \
             cg_drop; Drop must not re-issue rmdir against a forgotten \
             name: {calls:?}",
    );
    let cg_keep_removes: Vec<&CgroupCall> = calls
        .iter()
        .filter(|c| matches!(c, CgroupCall::RemoveCgroup(n) if n == "cg_keep"))
        .collect();
    assert_eq!(
        cg_keep_removes.len(),
        1,
        "Drop must rmdir the surviving cg_keep exactly once: {calls:?}",
    );
}

/// Install a minimal tracing subscriber that records every
/// event's (Level, concatenated-field-values) pair, run `f`,
/// and return the captured events. The MessageVisitor handles
/// both record_debug (Debug-formatted `?` fields wrapped via
/// `DebugValue`, Display-formatted `%` fields wrapped via
/// `DisplayValue`, and the format-string message itself via
/// `fmt::Arguments` — all three route through record_debug) and
/// record_str (raw string field values), so the returned message
/// string contains the warn body concatenated with every
/// structured-field value.
fn capture_tracing_events<F: FnOnce()>(f: F) -> Vec<(tracing::Level, String)> {
    use std::sync::{Arc, Mutex};
    use tracing::field::{Field, Visit};
    use tracing::span::{Attributes, Id, Record};
    use tracing::{Event, Level, Metadata, Subscriber};

    #[derive(Default)]
    struct CaptureSubscriber {
        events: Arc<Mutex<Vec<(Level, String)>>>,
    }
    struct MessageVisitor<'a>(&'a mut String);
    impl<'a> Visit for MessageVisitor<'a> {
        fn record_debug(&mut self, _field: &Field, value: &dyn std::fmt::Debug) {
            use std::fmt::Write;
            let _ = write!(self.0, "{value:?} ");
        }
        fn record_str(&mut self, _field: &Field, value: &str) {
            use std::fmt::Write;
            let _ = write!(self.0, "{value} ");
        }
    }
    impl Subscriber for CaptureSubscriber {
        fn enabled(&self, _: &Metadata<'_>) -> bool {
            true
        }
        fn new_span(&self, _: &Attributes<'_>) -> Id {
            Id::from_u64(1)
        }
        fn record(&self, _: &Id, _: &Record<'_>) {}
        fn record_follows_from(&self, _: &Id, _: &Id) {}
        fn event(&self, event: &Event<'_>) {
            let mut msg = String::new();
            event.record(&mut MessageVisitor(&mut msg));
            self.events
                .lock()
                .unwrap()
                .push((*event.metadata().level(), msg));
        }
        fn enter(&self, _: &Id) {}
        fn exit(&self, _: &Id) {}
    }
    let events: Arc<Mutex<Vec<(Level, String)>>> = Arc::new(Mutex::new(Vec::new()));
    let sub = CaptureSubscriber {
        events: events.clone(),
    };
    tracing::subscriber::with_default(sub, f);
    events.lock().unwrap().clone()
}

/// Branch 1 of Op::RemoveCgroup's typo-late-surfacing warn
/// fires when the removed name was tracked in Backdrop. The
/// warn correlates a later kernel-level "cgroup missing" with
/// the intentional removal source. Pin the predicate so a
/// future refactor that inverts the membership check, swallows
/// the warn, or routes the Backdrop path to branch 2 surfaces
/// here.
#[test]
fn remove_cgroup_warn_branch_1_fires_on_backdrop_tracked_name() {
    let events = capture_tracing_events(|| {
        mock_setup_backdrop!(mock, topo, ctx, step_state, backdrop_state);
        backdrop_state
            .cgroups
            .add_cgroup_no_cpuset("bd_cg")
            .expect("add backdrop cgroup");
        let mut scenario = ScenarioState::new(&mut step_state, &mut backdrop_state);
        apply_ops(&ctx, &mut scenario, &[Op::remove_cgroup("bd_cg")], false)
            .expect("remove_cgroup of backdrop target must succeed");
    });
    let warns: Vec<&(tracing::Level, String)> = events
        .iter()
        .filter(|(lvl, _)| *lvl == tracing::Level::WARN)
        .collect();
    assert_eq!(
        warns.len(),
        1,
        "exactly one warn expected from branch 1; got: {warns:?}",
    );
    assert!(
        warns[0]
            .1
            .contains("removed a Backdrop-owned cgroup mid-scenario"),
        "warn must include branch-1 text identifying Backdrop-owned removal; got: {:?}",
        warns[0].1,
    );
    assert!(
        warns[0].1.contains("bd_cg"),
        "warn must include the cgroup name; got: {:?}",
        warns[0].1,
    );
}

/// Branch 2 of Op::RemoveCgroup's typo-late-surfacing warn
/// fires when the removed name matches NEITHER step-local NOR
/// Backdrop tracking. The warn dumps both lists so the
/// operator can compare against the test source and find a
/// typo. Pin the dump-on-typo behavior so a future refactor
/// that drops a list or fires on the wrong predicate surfaces
/// here.
#[test]
fn remove_cgroup_warn_branch_2_fires_on_unknown_typo_name() {
    let events = capture_tracing_events(|| {
        mock_setup_backdrop!(mock, topo, ctx, step_state, backdrop_state);
        backdrop_state
            .cgroups
            .add_cgroup_no_cpuset("bd_real_name")
            .expect("add backdrop cgroup");
        let mut scenario = ScenarioState::new(&mut step_state, &mut backdrop_state);
        // Add a step-local cgroup so the step_cgroups field
        // in the warn has a non-empty value to substring-match
        // against — guards against a future refactor dropping
        // the step_cgroups field from the warn.
        apply_ops(
            &ctx,
            &mut scenario,
            &[Op::add_cgroup("step_local_real")],
            false,
        )
        .expect("add step-local cgroup");
        apply_ops(
            &ctx,
            &mut scenario,
            &[Op::remove_cgroup("bd_typoed_name")],
            false,
        )
        .expect("remove_cgroup of unknown name must succeed (permissive)");
    });
    let warns: Vec<&(tracing::Level, String)> = events
        .iter()
        .filter(|(lvl, _)| *lvl == tracing::Level::WARN)
        .collect();
    assert_eq!(
        warns.len(),
        1,
        "exactly one warn expected from branch 2; got: {warns:?}",
    );
    assert!(
        warns[0].1.contains("matches no step-local"),
        "warn must include branch-2 text identifying unknown-name typo; got: {:?}",
        warns[0].1,
    );
    assert!(
        warns[0].1.contains("bd_real_name"),
        "warn must dump backdrop_cgroups list including the real name; got: {:?}",
        warns[0].1,
    );
    assert!(
        warns[0].1.contains("step_local_real"),
        "warn must dump step_cgroups list including the step-local name; got: {:?}",
        warns[0].1,
    );
    assert!(
        warns[0].1.contains("bd_typoed_name"),
        "warn must include the typo'd cgroup target name; got: {:?}",
        warns[0].1,
    );
}

/// Branch 2 also fires for a legitimate second-remove of a
/// name already pruned by a prior Op::RemoveCgroup. The
/// wording must acknowledge this as one of two possible
/// causes (typo or double-remove) so a test author seeing the
/// warn doesn't immediately assume bug. Pin both the warn
/// emission and the wording.
#[test]
fn remove_cgroup_warn_branch_2_fires_on_double_remove_already_forgotten() {
    let events = capture_tracing_events(|| {
        mock_setup_state!(mock, topo, ctx, state);
        apply_ops_test(&ctx, &mut state, &[Op::add_cgroup("cg_once")]).unwrap();
        // First remove: in_step is true → branch 2 gated off.
        apply_ops_test(&ctx, &mut state, &[Op::remove_cgroup("cg_once")]).unwrap();
        // Second remove: name already pruned by the prior
        // remove's `forget` → matches neither tracking set →
        // branch 2 fires.
        apply_ops_test(&ctx, &mut state, &[Op::remove_cgroup("cg_once")]).unwrap();
    });
    let warns: Vec<&(tracing::Level, String)> = events
        .iter()
        .filter(|(lvl, _)| *lvl == tracing::Level::WARN)
        .collect();
    assert_eq!(
        warns.len(),
        1,
        "exactly one warn expected — first remove gated by in_step, second remove fires branch 2 once; got: {warns:?}",
    );
    assert!(
        warns[0]
            .1
            .contains("second-remove of an already-forgotten name"),
        "branch-2 wording must acknowledge double-remove as legitimate cause alongside typo; got: {:?}",
        warns[0].1,
    );
}

/// Neither warn branch fires on the happy step-local
/// add-then-remove path. Pin the suppression so a future
/// refactor that flips the membership predicate (so step-local
/// removals would log "unknown name") surfaces here.
#[test]
fn remove_cgroup_emits_no_warn_on_happy_step_local_path() {
    let events = capture_tracing_events(|| {
        mock_setup_state!(mock, topo, ctx, state);
        apply_ops_test(&ctx, &mut state, &[Op::add_cgroup("cg_local")]).unwrap();
        apply_ops_test(&ctx, &mut state, &[Op::remove_cgroup("cg_local")]).unwrap();
    });
    let warns: Vec<&(tracing::Level, String)> = events
        .iter()
        .filter(|(lvl, _)| *lvl == tracing::Level::WARN)
        .collect();
    assert!(
        warns.is_empty(),
        "happy step-local add-then-remove path must emit zero warns; got: {warns:?}",
    );
}

/// Step-local `Op::AddCgroup` with a name that already lives
/// in the Backdrop must route through the same
/// `cgroup_name_is_tracked` collision guard as `apply_setup`
/// — otherwise the CgroupGroup would push a shadow step-local
/// entry that later steps could address, silently racing the
/// Backdrop's own writes to cpuset / subtree_control on the
/// same cgroupfs path.
#[test]
fn op_add_cgroup_step_local_rejects_collision_with_backdrop() {
    mock_setup_backdrop!(mock, topo, ctx, step_state, backdrop_state);
    backdrop_state
        .cgroups
        .add_cgroup_no_cpuset("shared")
        .expect("add backdrop cgroup");
    let mut scenario = ScenarioState::new(&mut step_state, &mut backdrop_state);
    let err = apply_ops(&ctx, &mut scenario, &[Op::add_cgroup("shared")], false).expect_err(
        "apply_ops must reject a step-local AddCgroup whose \
                         name already lives in the Backdrop",
    );
    let msg = format!("{err:?}");
    assert!(
        msg.contains("'shared'") && msg.contains("collides"),
        "error must name the colliding cgroup and explain the collision; got: {msg}",
    );
    // Step-local names must NOT gain a shadow entry after the
    // guard fires.
    assert!(
        step_state.cgroups.names().iter().all(|n| n != "shared"),
        "step-local names must not contain the rejected name; got: {:?}",
        step_state.cgroups.names(),
    );
    // Backdrop copy is untouched.
    assert!(
        backdrop_state.cgroups.names().iter().any(|n| n == "shared"),
        "backdrop copy must survive the rejected op",
    );
}

/// `Op::AddCgroup` applied twice in one step with the same name
/// is rejected by the `cgroup_name_is_tracked` collision guard.
/// The first op adds the name to step-local tracking; the second
/// sees it already tracked and bails, so the CgroupGroup's name
/// vec gains exactly one entry and Drop's remove_cgroup runs
/// once per unique name.
#[test]
fn op_add_cgroup_duplicate_in_same_step_is_rejected() {
    mock_setup_state!(mock, topo, ctx, state);
    let err = apply_ops_test(
        &ctx,
        &mut state,
        &[Op::add_cgroup("cg_dup"), Op::add_cgroup("cg_dup")],
    )
    .expect_err("second AddCgroup must fail against the same step-local name");
    let msg = format!("{err:?}");
    assert!(
        msg.contains("'cg_dup'") && msg.contains("collides"),
        "error must name the colliding cgroup and explain the collision; got: {msg}",
    );
    let names = state.cgroups.names();
    assert_eq!(
        names.iter().filter(|n| n.as_str() == "cg_dup").count(),
        1,
        "the first op must register the name exactly once; the second op \
             must not push a shadow entry; got: {names:?}",
    );
}

/// `Op::add_cgroup_def(def)` constructor wraps the [`CgroupDef`]
/// in the `AddCgroupDef` variant without mutation. Pins the
/// constructor contract so a future refactor that, e.g., merges
/// AddCgroup and AddCgroupDef into one variant or splits the def
/// into separate fields surfaces here.
#[test]
fn op_add_cgroup_def_constructor_wraps_def_unmutated() {
    let def = CgroupDef::named("midstep").workers(3);
    let op = Op::add_cgroup_def(def.clone());
    match op {
        Op::AddCgroupDef { def: out } => {
            assert_eq!(out.name, def.name);
            assert_eq!(out.merged_works().len(), def.merged_works().len());
            assert_eq!(out.merged_works()[0].num_workers, Some(3));
        }
        other => panic!("expected AddCgroupDef, got {other:?}"),
    }
}

/// `Op::AddCgroupDef` dispatches through `apply_setup` so the
/// cgroup is created in cgroupfs and the def's name is tracked
/// in step-local state — same observable result as declaring the
/// def in `Step::with_defs`, just at apply-ops time instead of
/// the step's setup pass.
#[test]
fn op_add_cgroup_def_creates_cgroup_through_apply_setup() {
    mock_setup_state!(mock, topo, ctx, state);
    apply_ops_test(
        &ctx,
        &mut state,
        &[Op::add_cgroup_def(CgroupDef::named("cg_midstep"))],
    )
    .expect("AddCgroupDef must succeed for a fresh name");
    assert!(
        state.cgroups.names().iter().any(|n| n == "cg_midstep"),
        "step-local tracking must record the AddCgroupDef name; got: {:?}",
        state.cgroups.names(),
    );
}

/// `Op::AddCgroupDef` reuses `apply_setup`'s dedup check, so a
/// name that already lives on the Backdrop is rejected with the
/// same collision diagnostic operators see from a step-local
/// `Step::with_defs` collision.
#[test]
fn op_add_cgroup_def_rejects_collision_with_backdrop() {
    mock_setup_backdrop!(mock, topo, ctx, step_state, backdrop_state);
    backdrop_state
        .cgroups
        .add_cgroup_no_cpuset("persistent")
        .expect("add backdrop cgroup");
    let mut scenario = ScenarioState::new(&mut step_state, &mut backdrop_state);
    let err = apply_ops(
        &ctx,
        &mut scenario,
        &[Op::add_cgroup_def(CgroupDef::named("persistent"))],
        false,
    )
    .expect_err("AddCgroupDef must reject a name already tracked by the Backdrop");
    let msg = format!("{err:?}");
    assert!(
        msg.contains("'persistent'") && msg.contains("collides"),
        "error must name the colliding cgroup and explain the collision; got: {msg}",
    );
}

/// `required_controllers` picks up the controllers needed by a
/// [`CgroupDef`] embedded in `Op::AddCgroupDef`, so the parent's
/// `subtree_control` enables Cpuset before the op runs and the
/// cpuset write at apply-ops time doesn't fail with ENOENT on
/// the controller file. Without this absorb pass, a scenario
/// whose only cpuset user is an `Op::AddCgroupDef` would skip
/// Cpuset controller enablement entirely.
#[test]
fn required_controllers_absorbs_add_cgroup_def_cpuset() {
    use crate::cgroup::Controller;
    mock_setup!(mock, topo, ctx);
    let steps = vec![Step::new(
        vec![Op::add_cgroup_def(
            CgroupDef::named("cg_pinned").cpuset(CpusetSpec::disjoint(0, 2)),
        )],
        HoldSpec::fixed(Duration::from_millis(1)),
    )];
    let needed = required_controllers(&ctx, &backdrop::Backdrop::new(), &steps);
    assert!(
        needed.contains(&Controller::Cpuset),
        "AddCgroupDef carrying a cpuset must require Cpuset controller; got: {needed:?}",
    );
}

/// `Op::AddCgroupDef` carrying `workers(N)` spawns N workers
/// and emits the resulting `MoveTasks(_, N)` call against the
/// embedded def's cgroup — proves the delegation to
/// `apply_setup` invokes the same worker-spawn + move-into-cgroup
/// path that step-local CgroupDefs use. Mirrors
/// `apply_setup_moves_spawned_tasks_into_cgroup` (which exercises
/// the setup-time entry) but enters via the apply-ops entry.
#[test]
fn op_add_cgroup_def_spawns_workers_and_moves_into_cgroup() {
    mock_setup_state!(mock, topo, ctx, state);
    apply_ops_test(
        &ctx,
        &mut state,
        &[Op::add_cgroup_def(
            CgroupDef::named("cg_workers").workers(2),
        )],
    )
    .expect("AddCgroupDef with workers must succeed against mock");
    let calls = mock.calls();
    assert!(
        calls.contains(&CgroupCall::MoveTasks("cg_workers".to_string(), 2)),
        "AddCgroupDef must move 2 spawned worker pids into 'cg_workers'; got: {calls:?}",
    );
}

/// `Op::AddCgroupDef` carrying a cpuset spec emits a SetCpuset
/// mock call against the embedded def's resolved CPU set —
/// proves the delegation to `apply_setup` writes the cpuset
/// through `CgroupOps::set_cpuset`, not just stages controller
/// state. Regression class: a future refactor that bypasses
/// apply_setup's cpuset-write loop for the AddCgroupDef path
/// would slip past `required_controllers_absorbs_add_cgroup_def_cpuset`
/// (which only verifies the controller-bitmask side) — this
/// test pins the actual write.
#[test]
fn op_add_cgroup_def_writes_embedded_cpuset_to_mock() {
    mock_setup_state!(mock, topo, ctx, state);
    apply_ops_test(
        &ctx,
        &mut state,
        &[Op::add_cgroup_def(
            CgroupDef::named("cg_pinned").cpuset(CpusetSpec::disjoint(0, 2)),
        )],
    )
    .expect("AddCgroupDef with cpuset must succeed against mock");
    let calls = mock.calls();
    let has_set_cpuset = calls
        .iter()
        .any(|c| matches!(c, CgroupCall::SetCpuset(name, _) if name == "cg_pinned"));
    assert!(
        has_set_cpuset,
        "AddCgroupDef must emit SetCpuset for 'cg_pinned' via apply_setup; got: {calls:?}",
    );
}

/// `Op::AddCgroupDef` whose embedded def configures
/// `workers_pct` against a cpuset that resolves to zero CPUs
/// surfaces the same diagnostic apply_setup produces in the
/// step-setup path — the per-pct error message naming the
/// cgroup, the pct value, and the empty-cpuset condition.
/// Regression class: a refactor that short-circuits the
/// workers_pct empty-cpuset check for the AddCgroupDef path
/// would let a misconfigured def silently spawn 0 workers.
#[test]
fn op_add_cgroup_def_workers_pct_empty_cpuset_bails() {
    mock_setup_state!(mock, topo, ctx, state);
    // CpusetSpec::Exact with an empty set resolves to 0 CPUs;
    // workers_pct on top of that hits the dedicated diagnostic.
    let err = apply_ops_test(
        &ctx,
        &mut state,
        &[Op::add_cgroup_def(
            CgroupDef::named("cg_pct")
                .cpuset(CpusetSpec::exact(std::iter::empty::<usize>()))
                .workers_pct(0.5),
        )],
    )
    .expect_err("workers_pct + empty cpuset must bail through AddCgroupDef");
    let msg = format!("{err:?}");
    assert!(
        msg.contains("cg_pct") && msg.contains("workers_pct"),
        "diagnostic must name the cgroup and the workers_pct condition; got: {msg}",
    );
}

/// `Op::AddCgroupDef` after a prior `Op::AddCgroup` with the
/// same name in one step is rejected via apply_setup's
/// collision check (delegation transmits the dedup contract).
#[test]
fn op_add_cgroup_def_collides_with_prior_add_cgroup_in_same_step() {
    mock_setup_state!(mock, topo, ctx, state);
    let err = apply_ops_test(
        &ctx,
        &mut state,
        &[
            Op::add_cgroup("shared"),
            Op::add_cgroup_def(CgroupDef::named("shared")),
        ],
    )
    .expect_err("AddCgroupDef must reject a name already tracked by a prior AddCgroup");
    let msg = format!("{err:?}");
    assert!(
        msg.contains("'shared'") && msg.contains("collides"),
        "error must name the colliding cgroup and explain the collision; got: {msg}",
    );
}

/// Two `Op::AddCgroupDef` ops with the same name in one step
/// are rejected — second op hits apply_setup's collision check
/// against the first op's step-local tracking entry.
#[test]
fn op_add_cgroup_def_collides_with_prior_add_cgroup_def_in_same_step() {
    mock_setup_state!(mock, topo, ctx, state);
    let err = apply_ops_test(
        &ctx,
        &mut state,
        &[
            Op::add_cgroup_def(CgroupDef::named("dup")),
            Op::add_cgroup_def(CgroupDef::named("dup")),
        ],
    )
    .expect_err("second AddCgroupDef must reject the duplicated name");
    let msg = format!("{err:?}");
    assert!(
        msg.contains("'dup'") && msg.contains("collides"),
        "error must name the duplicated cgroup and explain the collision; got: {msg}",
    );
}

/// `Op::AddCgroup` after a prior `Op::AddCgroupDef` with the
/// same name in one step is rejected via the AddCgroup arm's
/// `cgroup_name_is_tracked` check — symmetric of the
/// AddCgroup-then-AddCgroupDef ordering. Without symmetric
/// coverage, a refactor that scoped tracking differently
/// per-arm could let a name escape the dedup in one ordering
/// only.
#[test]
fn op_add_cgroup_collides_with_prior_add_cgroup_def_in_same_step() {
    mock_setup_state!(mock, topo, ctx, state);
    let err = apply_ops_test(
        &ctx,
        &mut state,
        &[
            Op::add_cgroup_def(CgroupDef::named("shared")),
            Op::add_cgroup("shared"),
        ],
    )
    .expect_err("AddCgroup must reject a name already tracked by a prior AddCgroupDef");
    let msg = format!("{err:?}");
    assert!(
        msg.contains("'shared'") && msg.contains("collides"),
        "error must name the colliding cgroup and explain the collision; got: {msg}",
    );
}

/// `MoveAllTasks` must re-key EVERY workload handle whose
/// current name matches `from`, not just the first. Multiple
/// handles on the same cgroup arise when a scenario issues two
/// `Op::Spawn(SpawnPlacement::Cgroup(_))` ops on the same cgroup
/// name.
#[test]
fn move_all_tasks_renames_every_handle_keyed_under_from() {
    use crate::workload::{AffinityIntent, WorkType, WorkloadConfig, WorkloadHandle};

    mock_setup_backdrop!(mock, topo, ctx, step_state, backdrop_state);
    step_state.cgroups.add_cgroup_no_cpuset("src").unwrap();
    step_state.cgroups.add_cgroup_no_cpuset("dst").unwrap();

    // Push THREE handles all keyed under "src" — simulates two
    // Op::Spawn(SpawnPlacement::Cgroup) ops in the same cgroup +
    // one from CgroupDef.
    for _ in 0..3 {
        let wl = WorkloadConfig {
            num_workers: 1,
            affinity: AffinityIntent::Inherit,
            work_type: WorkType::SpinWait,
            ..Default::default()
        };
        let h = WorkloadHandle::spawn(&wl).expect("spawn worker");
        step_state.handles.push(("src".to_string(), h));
    }
    assert_eq!(step_state.handles.len(), 3);

    {
        let mut scenario = ScenarioState::new(&mut step_state, &mut backdrop_state);
        apply_ops(
            &ctx,
            &mut scenario,
            &[Op::move_all_tasks("src", "dst")],
            false,
        )
        .expect("move");
    }

    assert_eq!(step_state.handles.len(), 3, "no handles lost");
    assert!(
        step_state.handles.iter().all(|(name, _)| name == "dst"),
        "every handle must be re-keyed to 'dst': {:?}",
        step_state
            .handles
            .iter()
            .map(|(n, _)| n.as_str())
            .collect::<Vec<_>>(),
    );
    // Per-handle move_tasks pin: the handler at MoveAllTasks
    // iterates pid_batches and emits one move_tasks call per
    // matching handle. With 3 handles keyed under "src", the
    // mock must record 3 move_tasks("dst", 1) calls. A
    // regression that collapsed the loop into one bulk write
    // would still pass the re-key assertion above but fail
    // here.
    let calls = mock.calls();
    let dst_moves = calls
        .iter()
        .filter(|c| matches!(c, CgroupCall::MoveTasks(n, _) if n == "dst"))
        .count();
    assert_eq!(
        dst_moves, 3,
        "expected 3 move_tasks(\"dst\", _) calls (one per handle), \
             got {dst_moves} in: {calls:?}"
    );
    // SIGKILL before drop so the synthetic workers don't leak.
    step_state.handles.clear();
}

/// Per-step teardown is observable via the mock's call log.
/// `execute_scenario` runs Step::cgroups Drop at step boundary;
/// with MockCgroupOps we can pin that the rmdir calls happen
/// (a) only on step-local cgroups, (b) in REVERSE order of
/// addition — nested-cgroup-safe teardown.
#[test]
fn per_step_teardown_removes_step_local_cgroups_in_reverse_order() {
    mock_setup_state!(mock, topo, ctx, state);
    apply_ops_test(
        &ctx,
        &mut state,
        &[
            Op::add_cgroup("cg_a"),
            Op::add_cgroup("cg_a/sub"),
            Op::add_cgroup("cg_b"),
        ],
    )
    .unwrap();
    // Simulate step boundary: drop the state to run CgroupGroup::Drop.
    drop(state);
    let calls = mock.calls();
    let removes: Vec<&str> = calls
        .iter()
        .filter_map(|c| match c {
            CgroupCall::RemoveCgroup(n) => Some(n.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(
        removes,
        vec!["cg_b", "cg_a/sub", "cg_a"],
        "per-step teardown must rmdir in reverse addition order so a \
             child cgroup's directory is gone before its parent's rmdir \
             runs",
    );
}

/// `build_stimulus` encodes the 1-indexed phase number
/// (`step_idx + 1`) into the wire `step_index` slot, saturating
/// to `u16::MAX` (with a `tracing::warn!`) when the +1 would
/// overflow u16. The 0 slot is reserved for the BASELINE
/// pre-first-Step window the framework never emits a stimulus
/// for, so the lowest wire value `build_stimulus` ever produces
/// is 1. Exercise the three interesting values:
///
/// - `step_idx == 0` -> wire `step_index == 1` (first Step,
///   not BASELINE).
/// - `step_idx == u16::MAX as usize - 1` -> wire
///   `step_index == u16::MAX` (highest 1-indexed value that
///   fits without saturation).
/// - `step_idx == u16::MAX as usize` -> wire
///   `step_index == u16::MAX` (the +1 overflows; must saturate
///   instead of wrapping to 0).
#[test]
fn build_stimulus_saturates_step_idx_at_u16_max() {
    mock_setup_backdrop!(mock, topo, ctx, step_state, backdrop_state);
    let scenario = ScenarioState::new(&mut step_state, &mut backdrop_state);
    let start = std::time::Instant::now();

    let zero = build_stimulus(&start, 0, &[], &scenario);
    assert_eq!(
        zero.step_index, 1,
        "scenario step_idx=0 publishes wire step_index=1 \
             per the 1-indexed phase encoding (BASELINE owns 0)",
    );

    let last_unsaturated = build_stimulus(&start, u16::MAX as usize - 1, &[], &scenario);
    assert_eq!(
        last_unsaturated.step_index,
        u16::MAX,
        "scenario step_idx=u16::MAX - 1 publishes wire step_index=u16::MAX \
             without saturation (highest 1-indexed value that fits)",
    );

    let overflow = build_stimulus(&start, u16::MAX as usize, &[], &scenario);
    assert_eq!(
        overflow.step_index,
        u16::MAX,
        "scenario step_idx=u16::MAX would publish wire step_index=u16::MAX+1 \
             after the 1-indexed +1, so the encoder must saturate to u16::MAX \
             rather than wrap to 0",
    );

    // Far-overflow smoke check: the helper handles values
    // orders of magnitude past u16::MAX without panicking or
    // returning nonsense. The saturated value is u16::MAX
    // regardless of how far past the boundary `step_idx`
    // landed.
    let far = build_stimulus(&start, u32::MAX as usize, &[], &scenario);
    assert_eq!(
        far.step_index,
        u16::MAX,
        "far-overflow step_idx must saturate to u16::MAX",
    );
}

/// Saturation without a warn would silently clip the wire field;
/// the `tracing::warn!` inside `to_u16` is the only observable
/// signal an operator gets when a scenario blew past `u16::MAX`.
/// Install a minimal capturing subscriber, run a saturation-
/// triggering call, and assert the warn event fired.
#[test]
fn build_stimulus_warns_on_step_idx_saturation() {
    use std::sync::{Arc, Mutex};
    use tracing::field::{Field, Visit};
    use tracing::span::{Attributes, Id, Record};
    use tracing::{Event, Subscriber};
    use tracing::{Level, Metadata};

    // Capturing subscriber that records `(level, message)` pairs
    // for every event. Span-related methods are implemented as
    // no-ops; the test only cares about event emission.
    #[derive(Default)]
    struct CaptureSubscriber {
        events: Arc<Mutex<Vec<(Level, String)>>>,
    }
    struct MessageVisitor<'a>(&'a mut String);
    impl<'a> Visit for MessageVisitor<'a> {
        fn record_debug(&mut self, _field: &Field, value: &dyn std::fmt::Debug) {
            use std::fmt::Write;
            let _ = write!(self.0, "{value:?} ");
        }
        fn record_str(&mut self, _field: &Field, value: &str) {
            use std::fmt::Write;
            let _ = write!(self.0, "{value} ");
        }
    }
    impl Subscriber for CaptureSubscriber {
        fn enabled(&self, _: &Metadata<'_>) -> bool {
            true
        }
        fn new_span(&self, _: &Attributes<'_>) -> Id {
            Id::from_u64(1)
        }
        fn record(&self, _: &Id, _: &Record<'_>) {}
        fn record_follows_from(&self, _: &Id, _: &Id) {}
        fn event(&self, event: &Event<'_>) {
            let mut msg = String::new();
            event.record(&mut MessageVisitor(&mut msg));
            self.events
                .lock()
                .unwrap()
                .push((*event.metadata().level(), msg));
        }
        fn enter(&self, _: &Id) {}
        fn exit(&self, _: &Id) {}
    }

    let events: Arc<Mutex<Vec<(Level, String)>>> = Arc::new(Mutex::new(Vec::new()));
    let sub = CaptureSubscriber {
        events: events.clone(),
    };

    tracing::subscriber::with_default(sub, || {
        mock_setup_backdrop!(mock, topo, ctx, step_state, backdrop_state);
        let scenario = ScenarioState::new(&mut step_state, &mut backdrop_state);
        let start = std::time::Instant::now();

        // In-range call: no saturation, no warn expected.
        let _ = build_stimulus(&start, 0, &[], &scenario);
        // Saturating call: must emit a warn naming the
        // overflowing field and the offending value. The
        // 1-indexed encoding (`step_idx + 1`) saturates when
        // the +1 would exceed u16::MAX, which kicks in at
        // `step_idx == u16::MAX as usize`.
        let _ = build_stimulus(&start, u16::MAX as usize, &[], &scenario);
    });

    let captured = events.lock().unwrap();
    let warn_hits: Vec<&String> = captured
        .iter()
        .filter(|(lvl, _)| *lvl == Level::WARN)
        .map(|(_, msg)| msg)
        .collect();
    assert!(
        warn_hits
            .iter()
            .any(|m| m.contains("step_index")
                && m.contains("StimulusPayload step_index overflowed u16")),
        "saturation must emit a tracing::warn naming step_index; got warns: {warn_hits:?}",
    );
    // Sanity: no warn should fire for the in-range 0 call.
    // Since we can't easily partition the two calls, we assert
    // the total count is exactly one: saturating call fires
    // once, in-range call fires zero.
    assert_eq!(
        warn_hits.len(),
        1,
        "exactly one saturation warn expected; got: {warn_hits:?}",
    );
}

// -- Op variant constructor coverage --
//
// `Op` is `#[non_exhaustive]` — its doc directs downstream
// authors to use the per-op constructors (`Op::add_cgroup`,
// `Op::run_payload`, …) rather than naming variants directly so
// new variants can land without breaking matchers. This test is
// the enforcement seam: it exercises every documented constructor
// once AND pattern-matches the produced value against every Op
// variant without a wildcard arm. Either half failing catches a
// different regression:
//
// - A new variant added without a constructor fails the match
//   compilation (non-exhaustive pattern).
// - A new variant with a constructor but no test coverage
//   survives compilation but the constructor block below won't
//   cover it — a reviewer adding a variant + constructor must
//   also add a call here.
//
// The guard is build-time rather than runtime: removing the
// wildcard `_ =>` arm makes the rustc exhaustiveness checker
// own the constructor-per-variant contract.

/// Static binary-kind Payload used only to address the
/// `RunPayload` / `WaitPayload` / `KillPayload` constructors.
/// The test never spawns or runs this payload — only the
/// `&'static Payload` reference is consumed.
static CONSTRUCTOR_TEST_PAYLOAD: crate::test_support::Payload =
    crate::test_support::Payload::binary("constructor-test", "/bin/true");

/// Static Scheduler used only to address the AttachScheduler /
/// ReplaceScheduler constructors. The test never spawns or
/// attaches this scheduler — only the `&'static Scheduler`
/// reference is consumed. `EEVDF` is the zero-binary baseline
/// so the fixture has no init-time cost.
static CONSTRUCTOR_TEST_SCHEDULER: crate::test_support::Scheduler =
    crate::test_support::Scheduler::EEVDF;

#[test]
fn op_constructor_coverage_is_exhaustive() {
    let w = WorkSpec::default();
    let constructed: Vec<Op> = vec![
        Op::add_cgroup("a"),
        Op::add_cgroup_def(CgroupDef::named("midstep")),
        Op::remove_cgroup("a"),
        Op::set_cpuset("a", CpusetSpec::Llc(0)),
        Op::clear_cpuset("a"),
        Op::swap_cpusets("a", "b"),
        Op::stop_cgroup("a"),
        Op::set_affinity("a", AffinityIntent::Inherit),
        Op::move_all_tasks("a", "b"),
        Op::run_payload(&CONSTRUCTOR_TEST_PAYLOAD, [] as [&str; 0]),
        Op::run_payload_in_cgroup(&CONSTRUCTOR_TEST_PAYLOAD, [] as [&str; 0], "a"),
        Op::wait_payload("constructor-test"),
        Op::wait_payload_in_cgroup("constructor-test", "a"),
        Op::kill_payload("constructor-test"),
        Op::kill_payload_in_cgroup("constructor-test", "a"),
        Op::freeze_cgroup("a"),
        Op::unfreeze_cgroup("a"),
        Op::capture_snapshot("constructor-test"),
        Op::watch_snapshot("kernel.constructor_test"),
        Op::write_kernel_hot(
            KernelTarget::symbol("constructor_test_symbol"),
            KernelValue::u64(0),
        ),
        Op::write_kernel_cold(
            KernelTarget::symbol("constructor_test_symbol"),
            KernelValue::u64(0),
        ),
        Op::read_kernel_hot(
            "constructor-test-hot",
            KernelTarget::symbol("constructor_test_symbol"),
            KernelValueWidth::u64(),
        ),
        Op::read_kernel_cold(
            "constructor-test-cold",
            KernelTarget::symbol("constructor_test_symbol"),
            KernelValueWidth::u32(),
        ),
        Op::attach_scheduler(&CONSTRUCTOR_TEST_SCHEDULER),
        Op::detach_scheduler(),
        Op::restart_scheduler(),
        Op::replace_scheduler(&CONSTRUCTOR_TEST_SCHEDULER),
        Op::pin_bpf_map("constructor_test.bss"),
        Op::spawn(SpawnPlacement::cgroup("a"), w.clone()),
        Op::capture_cgroup_procs("constructor-test", "a"),
    ];

    // Track which variants we observed. Adding a variant to `Op`
    // without a constructor call above leaves one slot `false`,
    // and adding a variant without a match arm below fails to
    // compile (no `_ =>` on purpose). **Bump the array size when
    // the bit_index high-water-mark in `OpKind::bit_index`
    // changes** — the runtime index check at `seen[idx] = true`
    // will panic if the new variant's index >= the array length.
    let mut seen = [false; 27];
    for op in &constructed {
        let idx = match op {
            Op::AddCgroup { .. } => 0,
            Op::AddCgroupDef { .. } => 1,
            Op::RemoveCgroup { .. } => 2,
            Op::SetCpuset { .. } => 3,
            Op::ClearCpuset { .. } => 4,
            Op::SwapCpusets { .. } => 5,
            Op::Spawn { .. } => 6,
            Op::StopCgroup { .. } => 7,
            Op::SetAffinity { .. } => 8,
            Op::MoveAllTasks { .. } => 9,
            Op::RunPayload { .. } => 10,
            Op::WaitPayload { .. } => 11,
            Op::KillPayload { .. } => 12,
            Op::FreezeCgroup { .. } => 13,
            Op::UnfreezeCgroup { .. } => 14,
            Op::CaptureSnapshot { .. } => 15,
            Op::WatchSnapshot { .. } => 16,
            Op::WriteKernelHot { .. } => 17,
            Op::WriteKernelCold { .. } => 18,
            Op::ReadKernelHot { .. } => 19,
            Op::ReadKernelCold { .. } => 20,
            Op::AttachScheduler { .. } => 21,
            Op::DetachScheduler => 22,
            Op::RestartScheduler => 23,
            Op::ReplaceScheduler { .. } => 24,
            Op::PinBpfMap { .. } => 25,
            Op::CaptureCgroupProcs { .. } => 26,
        };
        seen[idx] = true;
    }

    let missing: Vec<usize> = seen
        .iter()
        .enumerate()
        .filter(|(_, hit)| !**hit)
        .map(|(i, _)| i)
        .collect();
    assert!(
        missing.is_empty(),
        "Op variant discriminants with no constructor coverage: {missing:?}. \
             Every Op variant must have a public constructor under impl Op per the \
             non_exhaustive convention documented on the enum.",
    );
}

#[test]
fn cpuset_spec_constructor_coverage_is_exhaustive() {
    let constructed = [
        CpusetSpec::llc(0),
        CpusetSpec::numa(0),
        CpusetSpec::range(0.0, 1.0),
        CpusetSpec::disjoint(0, 2),
        CpusetSpec::overlap(0, 2, 0.25),
        CpusetSpec::exact([0usize]),
    ];
    let mut seen = [false; 6];
    for spec in &constructed {
        let idx = match spec {
            CpusetSpec::Llc(_) => 0,
            CpusetSpec::Numa(_) => 1,
            CpusetSpec::Range { .. } => 2,
            CpusetSpec::Disjoint { .. } => 3,
            CpusetSpec::Overlap { .. } => 4,
            CpusetSpec::Exact(_) => 5,
        };
        seen[idx] = true;
    }
    assert!(
        seen.iter().all(|s| *s),
        "every CpusetSpec variant must have a matching constructor, seen={seen:?}",
    );
}

// -- CgroupDef cgroup-v2 resource builders -----------------------

/// `.cpu_quota_pct(50)` populates `cpu.max_quota_us = 50_000`
/// with the default 100 ms period. Pins the percentage-to-µs
/// conversion factor so a future refactor that shifts to
/// nanoseconds trips this test.
#[test]
fn cgroup_def_cpu_quota_pct_uses_100ms_period_and_correct_quota() {
    let def = CgroupDef::named("cg_a").cpu_quota_pct(50);
    let cpu = def.cpu.expect("cpu_quota_pct must populate `cpu`");
    assert_eq!(cpu.max_quota_us, Some(50_000));
    assert_eq!(cpu.max_period_us, 100_000);
    assert!(cpu.weight.is_none(), "weight must remain unset");
}

/// `.cpu_quota(quota, period)` accepts arbitrary Durations and
/// converts to microseconds.
#[test]
fn cgroup_def_cpu_quota_accepts_explicit_durations() {
    let def =
        CgroupDef::named("cg_a").cpu_quota(Duration::from_micros(7_500), Duration::from_millis(10));
    let cpu = def.cpu.unwrap();
    assert_eq!(cpu.max_quota_us, Some(7_500));
    assert_eq!(cpu.max_period_us, 10_000);
}

/// `.cpu_unlimited()` clears the quota but preserves `weight`.
/// Pins the "weight survives clear" guarantee documented on
/// the builder.
#[test]
fn cgroup_def_cpu_unlimited_clears_quota_keeps_weight() {
    let def = CgroupDef::named("cg_a")
        .cpu_quota_pct(80)
        .cpu_weight(200)
        .cpu_unlimited();
    let cpu = def.cpu.unwrap();
    assert!(cpu.max_quota_us.is_none());
    assert_eq!(cpu.max_period_us, 100_000);
    assert_eq!(cpu.weight, Some(200));
}

/// All three memory builders compose into a single MemoryLimits.
#[test]
fn cgroup_def_memory_builders_compose() {
    let def = CgroupDef::named("cg_a")
        .memory_max(1_000_000)
        .memory_high(800_000)
        .memory_low(400_000);
    let m = def.memory.unwrap();
    assert_eq!(m.max, Some(1_000_000));
    assert_eq!(m.high, Some(800_000));
    assert_eq!(m.low, Some(400_000));
}

/// `.memory_unlimited()` resets every memory knob to None,
/// undoing prior `.memory_max/high/low` calls.
#[test]
fn cgroup_def_memory_unlimited_clears_all_three() {
    let def = CgroupDef::named("cg_a")
        .memory_max(1_000_000)
        .memory_high(800_000)
        .memory_low(400_000)
        .memory_unlimited();
    let m = def.memory.unwrap();
    assert!(m.max.is_none());
    assert!(m.high.is_none());
    assert!(m.low.is_none());
}

/// `.io_weight(N)` populates the IoLimits.
#[test]
fn cgroup_def_io_weight_populates() {
    let def = CgroupDef::named("cg_a").io_weight(750);
    assert_eq!(def.io.unwrap().weight, Some(750));
}

/// `.cpuset_mems(nodes)` populates the new field without
/// disturbing the cpuset.cpus side.
#[test]
fn cgroup_def_cpuset_mems_populates_independent_field() {
    let nodes: BTreeSet<usize> = [0usize, 1].into_iter().collect();
    let def = CgroupDef::named("cg_a").cpuset_mems(nodes.clone());
    assert_eq!(def.cpuset_mems, Some(nodes));
    assert!(def.cpuset.is_none());
}

// -- apply_setup wires builder values to CgroupOps calls ----------
//
// These tests drive `apply_setup` against a `MockCgroupOps` that
// records every call into the existing `CgroupCall` enum, then
// assert on the recorded sequence. The apply_setup site emits
// the new resource-control writes between cpuset assignment and
// worker spawn, so the tests pin both presence (the calls fire)
// and ordering (cpu/memory/io land BEFORE move_tasks so the
// limits are in effect when workers join).

/// A bare CgroupDef with `.cpu_quota_pct(50)` records exactly
/// one SetCpuMax call with the converted u64 quota and the
/// default 100 ms period.
#[test]
fn apply_setup_records_set_cpu_max_for_cpu_quota_pct_builder() {
    mock_setup_state!(mock, topo, ctx, state);
    let defs = vec![CgroupDef::named("cg_cap").cpu_quota_pct(75)];
    apply_setup_test(&ctx, &mut state, &defs).expect("apply_setup must succeed");
    let calls = mock.calls();
    assert!(
        calls.contains(&CgroupCall::SetCpuMax(
            "cg_cap".to_string(),
            Some(75_000),
            100_000,
        )),
        "expected SetCpuMax(cg_cap, Some(75000), 100000); got {calls:?}",
    );
    cleanup_state(&mut state);
}

/// `.memory_max(N)` records SetMemoryMax(Some(N)) AND clears
/// the unset high/low to None — the apply_setup loop emits all
/// three writes whenever the `memory` field is `Some` so a
/// prior cgroup's residue can't bleed through.
#[test]
fn apply_setup_records_three_memory_writes_when_memory_field_set() {
    mock_setup_state!(mock, topo, ctx, state);
    let defs = vec![CgroupDef::named("cg_mem").memory_max(1_000_000)];
    apply_setup_test(&ctx, &mut state, &defs).expect("apply_setup must succeed");
    let calls = mock.calls();
    let max_idx = calls
        .iter()
        .position(|c| matches!(c, CgroupCall::SetMemoryMax(n, _) if n == "cg_mem"))
        .expect("SetMemoryMax must fire");
    let high_idx = calls
        .iter()
        .position(|c| matches!(c, CgroupCall::SetMemoryHigh(n, _) if n == "cg_mem"))
        .expect("SetMemoryHigh must fire");
    let low_idx = calls
        .iter()
        .position(|c| matches!(c, CgroupCall::SetMemoryLow(n, _) if n == "cg_mem"))
        .expect("SetMemoryLow must fire");
    assert!(
        max_idx < high_idx && high_idx < low_idx,
        "memory writes must land in (max, high, low) order; got max={max_idx} high={high_idx} low={low_idx}",
    );
    // Specific values: max=Some, high=None (writes "max"),
    // low=None (writes "0") — pin both the SET and the
    // implicit-clear.
    assert!(calls.contains(&CgroupCall::SetMemoryMax(
        "cg_mem".to_string(),
        Some(1_000_000)
    )),);
    assert!(calls.contains(&CgroupCall::SetMemoryHigh("cg_mem".to_string(), None)));
    assert!(calls.contains(&CgroupCall::SetMemoryLow("cg_mem".to_string(), None)));
    cleanup_state(&mut state);
}

/// Ordering pin: every resource-control write MUST land before
/// the first MoveTasks for the same cgroup so workers join an
/// already-configured environment. Reverse ordering is a kernel
/// race per Documentation/admin-guide/cgroup-v2.rst — tasks
/// admitted before cpuset.mems is set may fail allocation per
/// `cpuset_update_task_spread`.
#[test]
fn apply_setup_resource_writes_land_before_move_tasks() {
    mock_setup_state!(mock, topo, ctx, state);
    let mems: BTreeSet<usize> = [0usize].into_iter().collect();
    let defs = vec![
        CgroupDef::named("cg_full")
            .cpuset_mems(mems)
            .cpu_quota_pct(40)
            .cpu_weight(200)
            .memory_max(2_000_000)
            .io_weight(150),
    ];
    apply_setup_test(&ctx, &mut state, &defs).expect("apply_setup must succeed");
    let calls = mock.calls();
    let move_idx = calls
        .iter()
        .position(|c| matches!(c, CgroupCall::MoveTasks(n, _) if n == "cg_full"));
    // No workers here means no MoveTasks — but every resource
    // write must still appear, in the documented order. Pin
    // each kind's presence and then assert the inter-kind
    // ordering relative to the (possibly absent) MoveTasks.
    let kinds: Vec<usize> = calls
        .iter()
        .enumerate()
        .filter_map(|(i, c)| match c {
            CgroupCall::SetCpusetMems(n, _) if n == "cg_full" => Some(i),
            CgroupCall::SetCpuMax(n, _, _) if n == "cg_full" => Some(i),
            CgroupCall::SetCpuWeight(n, _) if n == "cg_full" => Some(i),
            CgroupCall::SetMemoryMax(n, _) if n == "cg_full" => Some(i),
            CgroupCall::SetMemoryHigh(n, _) if n == "cg_full" => Some(i),
            CgroupCall::SetMemoryLow(n, _) if n == "cg_full" => Some(i),
            CgroupCall::SetIoWeight(n, _) if n == "cg_full" => Some(i),
            _ => None,
        })
        .collect();
    assert!(
        kinds.len() >= 7,
        "expected at least 7 resource writes (mems + cpu.max + cpu.weight + 3 memory + io.weight); got {} ({calls:?})",
        kinds.len(),
    );
    if let Some(mi) = move_idx {
        assert!(
            kinds.iter().all(|k| *k < mi),
            "every resource write must precede MoveTasks; kinds={kinds:?} move_idx={mi}",
        );
    }
    cleanup_state(&mut state);
}

/// `cpu.weight = 0` (out of kernel range 1..=10000) MUST be
/// rejected at apply_setup with a clear error message naming
/// the cgroup and the offending value.
#[test]
fn apply_setup_rejects_cpu_weight_out_of_range() {
    mock_setup_state!(mock, topo, ctx, state);
    let defs = vec![CgroupDef::named("cg_bad").cpu_weight(0)];
    let err =
        apply_setup_test(&ctx, &mut state, &defs).expect_err("apply_setup must reject weight=0");
    let msg = format!("{err:#}");
    assert!(
        msg.contains("cg_bad") && msg.contains("cpu.weight"),
        "error must name cgroup and field; got: {msg}",
    );
    cleanup_state(&mut state);
}

/// `cpu.max` with `period_us = 0` MUST be rejected — the
/// kernel writes `quota period` and divide-by-zero in the CFS
/// scheduler is a guaranteed bug.
#[test]
fn apply_setup_rejects_cpu_max_period_zero() {
    mock_setup_state!(mock, topo, ctx, state);
    let defs =
        vec![CgroupDef::named("cg_bad").cpu_quota(Duration::from_millis(50), Duration::ZERO)];
    let err =
        apply_setup_test(&ctx, &mut state, &defs).expect_err("apply_setup must reject period=0");
    let msg = format!("{err:#}");
    assert!(
        msg.contains("cg_bad") && msg.contains("period"),
        "error must name cgroup and period; got: {msg}",
    );
    cleanup_state(&mut state);
}

// -- pids.max + memory.swap.max + cgroup.freeze ------------------

/// `.memory_swap_max(bytes)` populates the swap_max field on the
/// MemoryLimits inner; `.memory_swap_unlimited()` clears it back
/// to None. Mirrors the cpu_quota_pct / cpu_unlimited convention.
#[test]
fn cgroup_def_memory_swap_max_builder_round_trip() {
    let d = CgroupDef::named("cg_a").memory_swap_max(2 * 1024 * 1024);
    assert_eq!(d.memory.as_ref().unwrap().swap_max, Some(2 * 1024 * 1024));

    let d = d.memory_swap_unlimited();
    assert_eq!(d.memory.as_ref().unwrap().swap_max, None);
}

/// `memory_swap_unlimited()` on a fresh CgroupDef (no prior
/// `memory_*` calls) MUST NOT inflate `self.memory` from `None`
/// to `Some(MemoryLimits::default())` — that would trigger 3
/// unwanted apply_setup writes (`memory.max`, `memory.high`,
/// `memory.low`) for a user who only asked to clear the swap
/// cap. Pin the no-op short-circuit so a regression that drops
/// the `if let Some` guard surfaces here.
#[test]
fn cgroup_def_memory_swap_unlimited_on_fresh_def_is_noop() {
    let d = CgroupDef::named("cg_a").memory_swap_unlimited();
    assert!(
        d.memory.is_none(),
        "memory_swap_unlimited() on a fresh CgroupDef must leave \
             self.memory == None; got: {:?}",
        d.memory,
    );
}

/// `memory_unlimited()` then `memory_swap_unlimited()` — the
/// chain cln-preread flagged. memory_unlimited sets
/// `self.memory = Some(MemoryLimits::default())` (already has
/// `swap_max = None`); the subsequent memory_swap_unlimited
/// must not redundantly recreate the MemoryLimits. After both
/// calls, the inner is `Some(default)` with all four knobs
/// `None`, mirroring memory_unlimited's intent. Pin both ends
/// of the chain.
#[test]
fn cgroup_def_memory_unlimited_then_swap_unlimited_is_idempotent() {
    let d = CgroupDef::named("cg_a")
        .memory_unlimited()
        .memory_swap_unlimited();
    let m = d.memory.expect("memory_unlimited installs Some(default)");
    assert!(m.max.is_none());
    assert!(m.high.is_none());
    assert!(m.low.is_none());
    assert!(m.swap_max.is_none());
}

/// `apply_setup` against a CgroupDef with `memory_swap_unlimited()`
/// alone (no other memory builders) must NOT emit any memory
/// writes — the no-op short-circuit keeps `self.memory == None`,
/// so the apply_setup `if let Some(ref mem)` block is skipped.
/// Without the fix, a fresh `MemoryLimits::default()` would land
/// in `self.memory` and fire `set_memory_max(None)` +
/// `set_memory_high(None)` + `set_memory_low(None)` — a silent
/// regression for tests that just want to clear a swap cap
/// inherited from a base CgroupDef factory.
#[test]
fn apply_setup_memory_swap_unlimited_on_fresh_def_emits_no_memory_writes() {
    mock_setup_state!(mock, topo, ctx, state);
    let defs = vec![CgroupDef::named("cg_swap_clear").memory_swap_unlimited()];
    apply_setup_test(&ctx, &mut state, &defs).expect("apply_setup must succeed");
    let calls = mock.calls();
    assert!(
        !calls.iter().any(|c| matches!(
            c,
            CgroupCall::SetMemoryMax(_, _)
                | CgroupCall::SetMemoryHigh(_, _)
                | CgroupCall::SetMemoryLow(_, _)
                | CgroupCall::SetMemorySwapMax(_, _)
        )),
        "memory_swap_unlimited() on a fresh CgroupDef must emit zero memory writes; got: {calls:?}"
    );
    cleanup_state(&mut state);
}

/// `.pids_max(n)` populates the pids field; `.pids_unlimited()`
/// clears it. The pids field is independent of memory/cpu/io.
#[test]
fn cgroup_def_pids_max_builder_round_trip() {
    let d = CgroupDef::named("cg_a").pids_max(1024);
    assert_eq!(d.pids.as_ref().unwrap().max, Some(1024));

    let d = d.pids_unlimited();
    assert_eq!(d.pids.as_ref().unwrap().max, None);
}

/// apply_setup with `.memory_swap_max(N)` records exactly one
/// SetMemorySwapMax call. swap_max defaults to None on a
/// MemoryLimits constructed by `memory_max` alone — pin both
/// shapes so a regression that always emits swap_max writes
/// (or never emits them) surfaces here.
#[test]
fn apply_setup_records_set_memory_swap_max() {
    mock_setup_state!(mock, topo, ctx, state);
    let defs = vec![CgroupDef::named("cg_swap").memory_swap_max(4 * 1024 * 1024)];
    apply_setup_test(&ctx, &mut state, &defs).expect("apply_setup must succeed");
    let calls = mock.calls();
    assert!(
        calls.contains(&CgroupCall::SetMemorySwapMax(
            "cg_swap".to_string(),
            Some(4 * 1024 * 1024),
        )),
        "swap_max with bytes must record SetMemorySwapMax(Some(N)), got: {calls:?}",
    );
    cleanup_state(&mut state);

    // memory_max alone: swap_max stays None — apply_setup must
    // SKIP the SetMemorySwapMax write entirely. memory.swap.max
    // only exists on CONFIG_SWAP kernels; the per-knob
    // explicit-set semantics (write only when the user opted in)
    // keeps swap-disabled kernels viable for tests that just set
    // memory_max. This mirrors the pids block's "only write when
    // pids.max.is_some()" gate.
    let mock = MockCgroupOps::new();
    let ctx = mock_ctx(&mock, &topo);
    let mut state = StepState::empty(&ctx);
    let defs = vec![CgroupDef::named("cg_nosw").memory_max(1_000_000)];
    apply_setup_test(&ctx, &mut state, &defs).expect("apply_setup must succeed");
    let calls = mock.calls();
    assert!(
        !calls.iter().any(|c| matches!(
            c,
            CgroupCall::SetMemorySwapMax(n, _) if n == "cg_nosw",
        )),
        "memory_max-only must NOT record SetMemorySwapMax (would \
             ENOENT on CONFIG_SWAP=n kernels); got: {calls:?}",
    );
    // Memory-write order pin: max → high → low. The ordering
    // matters because max must precede high so a high-above-max
    // user error surfaces with a clearer kernel error. swap_max
    // is excluded from the order check here because it only
    // emits when explicitly opted in (see test
    // `apply_setup_orders_memory_swap_max_after_low` for the
    // 4-write order pin).
    let max_idx = calls
        .iter()
        .position(|c| matches!(c, CgroupCall::SetMemoryMax(n, _) if n == "cg_nosw"))
        .expect("SetMemoryMax must fire");
    let high_idx = calls
        .iter()
        .position(|c| matches!(c, CgroupCall::SetMemoryHigh(n, _) if n == "cg_nosw"))
        .expect("SetMemoryHigh must fire");
    let low_idx = calls
        .iter()
        .position(|c| matches!(c, CgroupCall::SetMemoryLow(n, _) if n == "cg_nosw"))
        .expect("SetMemoryLow must fire");
    assert!(
        max_idx < high_idx && high_idx < low_idx,
        "memory writes must land in (max, high, low) order; \
             got max={max_idx} high={high_idx} low={low_idx}",
    );
    cleanup_state(&mut state);
}

/// When the user opts in via `.memory_swap_max(N)`, apply_setup
/// emits SetMemorySwapMax AFTER the max/high/low triple. Pins the
/// 4-write order across the full memory block so a regression
/// that re-orders swap_max relative to the other knobs surfaces
/// here. Distinct from `apply_setup_records_set_memory_swap_max`
/// which pins presence/absence under the swap-disabled-kernel
/// gate; this test pins ordering under the swap-enabled path.
#[test]
fn apply_setup_orders_memory_swap_max_after_low() {
    mock_setup_state!(mock, topo, ctx, state);
    let defs = vec![
        CgroupDef::named("cg_full_mem")
            .memory_max(2_000_000)
            .memory_high(1_500_000)
            .memory_low(500_000)
            .memory_swap_max(8 * 1024 * 1024),
    ];
    apply_setup_test(&ctx, &mut state, &defs).expect("apply_setup must succeed");
    let calls = mock.calls();
    let max_idx = calls
        .iter()
        .position(|c| matches!(c, CgroupCall::SetMemoryMax(n, _) if n == "cg_full_mem"))
        .expect("SetMemoryMax must fire");
    let high_idx = calls
        .iter()
        .position(|c| matches!(c, CgroupCall::SetMemoryHigh(n, _) if n == "cg_full_mem"))
        .expect("SetMemoryHigh must fire");
    let low_idx = calls
        .iter()
        .position(|c| matches!(c, CgroupCall::SetMemoryLow(n, _) if n == "cg_full_mem"))
        .expect("SetMemoryLow must fire");
    let swap_idx = calls
        .iter()
        .position(|c| matches!(c, CgroupCall::SetMemorySwapMax(n, _) if n == "cg_full_mem"))
        .expect("SetMemorySwapMax must fire when swap_max is opted in");
    assert!(
        max_idx < high_idx && high_idx < low_idx && low_idx < swap_idx,
        "memory writes must land in (max, high, low, swap_max) order; \
             got max={max_idx} high={high_idx} low={low_idx} swap={swap_idx}",
    );
    cleanup_state(&mut state);
}

/// apply_setup with `.pids_max(N)` records SetPidsMax(Some(N)).
/// Without `pids` set, no SetPidsMax call is emitted.
#[test]
fn apply_setup_records_set_pids_max_only_when_set() {
    mock_setup_state!(mock, topo, ctx, state);
    let defs = vec![CgroupDef::named("cg_pids").pids_max(512)];
    apply_setup_test(&ctx, &mut state, &defs).expect("apply_setup must succeed");
    let calls = mock.calls();
    assert!(
        calls.contains(&CgroupCall::SetPidsMax("cg_pids".to_string(), Some(512))),
        "pids_max(N) must record SetPidsMax(Some(N)), got: {calls:?}",
    );
    cleanup_state(&mut state);

    // No pids — no SetPidsMax call.
    let mock = MockCgroupOps::new();
    let ctx = mock_ctx(&mock, &topo);
    let mut state = StepState::empty(&ctx);
    let defs = vec![CgroupDef::named("cg_nopids").memory_max(1_000_000)];
    apply_setup_test(&ctx, &mut state, &defs).expect("apply_setup must succeed");
    let calls = mock.calls();
    assert!(
        !calls
            .iter()
            .any(|c| matches!(c, CgroupCall::SetPidsMax(_, _))),
        "no SetPidsMax expected when pids field is None, got: {calls:?}",
    );
    cleanup_state(&mut state);
}

/// `pids_max(0)` must be rejected at apply_setup with a clear
/// error naming the cgroup and the offending value. A 0-limit
/// cgroup silently halts every fork inside, including the
/// futex-helper threads spawned by some WorkType variants —
/// kernel accepts it but the workload would silently halt.
#[test]
fn apply_setup_rejects_pids_max_zero() {
    mock_setup_state!(mock, topo, ctx, state);
    let defs = vec![CgroupDef::named("cg_zero").pids_max(0)];
    let err =
        apply_setup_test(&ctx, &mut state, &defs).expect_err("apply_setup must reject pids_max(0)");
    let msg = format!("{err:#}");
    assert!(
        msg.contains("cg_zero") && msg.contains("pids.max"),
        "error must name cgroup and pids.max; got: {msg}",
    );
    // Pin the full diagnostic wording: the actionable hint
    // ("must be > 0" + "pids_unlimited") is what tells a user
    // to switch builders rather than rewrite their config.
    // Drift in either substring makes the diagnostic less
    // actionable; surface it here at test time, not at
    // user-debugging time.
    assert!(
        msg.contains("must be > 0"),
        "error must spell out the constraint; got: {msg}",
    );
    assert!(
        msg.contains("pids_unlimited"),
        "error must name the escape hatch (pids_unlimited()); got: {msg}",
    );
    cleanup_state(&mut state);
}

/// `Op::FreezeCgroup` against a cgroup the framework has never
/// created routes through `ctx.cgroups.set_freeze` and
/// surfaces the underlying kernel error as a step-level
/// failure. The MockCgroupOps double records the call but
/// returns Ok by default; pin the call sequence so a future
/// regression that swallows the FreezeCgroup op (or routes it
/// through a different code path that masks the error from a
/// real cgroupfs ENOENT) trips here. The "real" fail-on-ENOENT
/// path is exercised at the [`crate::cgroup`] layer's
/// `set_freeze_returns_err_with_enoent_when_freeze_file_missing`
/// test; this test pins the apply_ops dispatch shape.
#[test]
fn apply_ops_freeze_undefined_cgroup_dispatches_set_freeze() {
    mock_setup_state!(mock, topo, ctx, state);
    // The cgroup name "ghost_cg" is never declared via
    // CgroupDef or Op::AddCgroup. apply_ops still dispatches —
    // the framework does not gate FreezeCgroup on prior
    // creation; the kernel is the final authority on whether
    // the cgroup directory exists.
    apply_ops_test(&ctx, &mut state, &[Op::freeze_cgroup("ghost_cg")])
        .expect("apply_ops must dispatch FreezeCgroup even for an undeclared name");
    let calls = mock.calls();
    assert!(
        calls.contains(&CgroupCall::SetFreeze("ghost_cg".to_string(), true)),
        "FreezeCgroup must reach set_freeze regardless of declaration state, got: {calls:?}"
    );
}

/// `Op::FreezeCgroup` propagates the underlying ops error with
/// an `Op::FreezeCgroup: cgroup '<name>'` context prefix so a
/// failure dump names both the op and the offender. Inject an
/// error from the mock and verify the context chain.
#[test]
fn apply_ops_freeze_propagates_set_freeze_error_with_context() {
    mock_setup_state!(mock, topo, ctx, state);
    // Index 0 is the SetFreeze call from the FreezeCgroup op.
    // Reordered after macro: fail_call_at is &self + only mutates
    // the fail-injection lock (not the call-index counter), so the
    // reorder is observationally identical.
    mock.fail_call_at(0, "kernel ENOENT — cgroup directory does not exist");
    let err = apply_ops_test(&ctx, &mut state, &[Op::freeze_cgroup("ghost_cg")])
        .expect_err("set_freeze failure must surface as Err");
    let msg = format!("{err:#}");
    assert!(
        msg.contains("Op::FreezeCgroup") && msg.contains("ghost_cg"),
        "error must name the op and the cgroup, got: {msg}"
    );
    assert!(
        msg.contains("ENOENT"),
        "error must propagate the underlying cause, got: {msg}"
    );
}

/// `Op::FreezeCgroup` dispatches to set_freeze(true);
/// `Op::UnfreezeCgroup` to set_freeze(false). The mock records
/// both shapes verbatim so a regression that swaps the bool
/// surfaces here. Direct apply_ops dispatch — no workers needed.
#[test]
fn apply_ops_freeze_and_unfreeze_record_set_freeze() {
    mock_setup_state!(mock, topo, ctx, state);
    apply_ops_test(
        &ctx,
        &mut state,
        &[Op::freeze_cgroup("cg_x"), Op::unfreeze_cgroup("cg_x")],
    )
    .expect("freeze/unfreeze ops must succeed");
    let calls = mock.calls();
    assert!(
        calls.contains(&CgroupCall::SetFreeze("cg_x".to_string(), true)),
        "FreezeCgroup must dispatch SetFreeze(true), got: {calls:?}",
    );
    assert!(
        calls.contains(&CgroupCall::SetFreeze("cg_x".to_string(), false)),
        "UnfreezeCgroup must dispatch SetFreeze(false), got: {calls:?}",
    );
    // Sanity: the order must be (true, false) — the ops were
    // applied in that order.
    let true_idx = calls
        .iter()
        .position(|c| matches!(c, CgroupCall::SetFreeze(_, true)))
        .expect("found freeze");
    let false_idx = calls
        .iter()
        .position(|c| matches!(c, CgroupCall::SetFreeze(_, false)))
        .expect("found unfreeze");
    assert!(
        true_idx < false_idx,
        "freeze (true) must come before unfreeze (false): {calls:?}",
    );
    cleanup_state(&mut state);
}

/// `apply_setup` rejects `io.weight` outside the kernel's
/// `1..=10000` range BEFORE issuing the syscall. The kernel's
/// `cgrp_dfl_io_weight_write` parses via `kstrtouint` and
/// returns -ERANGE for values outside the documented bound; the
/// framework intercepts at apply-setup time so the operator
/// gets a structured error naming the offending cgroup and
/// value, rather than a raw ERANGE on cgroupfs.
///
/// Pin both ends (0 and 10001) so a refactor that loosens the
/// check in either direction surfaces here.
#[test]
fn apply_setup_rejects_io_weight_out_of_range() {
    for (weight, label) in [(0u16, "zero"), (10_001u16, "above-max")] {
        mock_setup_state!(mock, topo, ctx, state);
        let defs = vec![CgroupDef::named("cg_io").io_weight(weight)];
        let err = apply_setup_test(&ctx, &mut state, &defs)
            .expect_err(&format!("io.weight={weight} ({label}) must reject"));
        let msg = format!("{err:#}");
        assert!(
            msg.contains("io.weight") && msg.contains("out of range"),
            "error must name the offending knob and constraint; got: {msg}",
        );
        assert!(
            msg.contains("cg_io"),
            "error must name the offending cgroup; got: {msg}",
        );
        // The reject must fire BEFORE the kernel write — no
        // SetIoWeight call should have been recorded.
        let calls = mock.calls();
        assert!(
            !calls
                .iter()
                .any(|c| matches!(c, CgroupCall::SetIoWeight(n, _) if n == "cg_io")),
            "rejected weight must not reach the cgroupfs write: {calls:?}",
        );
        cleanup_state(&mut state);
    }
}

/// Range boundary acceptance: `io.weight=1` and `io.weight=10000`
/// (the kernel's documented endpoints) MUST be accepted by the
/// framework's range gate. Pinned alongside the rejection test so
/// a future refactor that flips a `<` to `<=` (or vice versa)
/// breaks one of the two tests instead of silently widening or
/// narrowing the accepted set.
#[test]
fn apply_setup_accepts_io_weight_range_endpoints() {
    for weight in [1u16, 10_000u16] {
        mock_setup_state!(mock, topo, ctx, state);
        let defs = vec![CgroupDef::named("cg_io").io_weight(weight)];
        apply_setup_test(&ctx, &mut state, &defs)
            .unwrap_or_else(|e| panic!("io.weight={weight} (boundary) must be accepted: {e:#}"));
        let calls = mock.calls();
        assert!(
            calls.contains(&CgroupCall::SetIoWeight("cg_io".to_string(), weight)),
            "boundary weight must reach the cgroupfs write; got: {calls:?}",
        );
        cleanup_state(&mut state);
    }
}

/// Empty `works` substitution: a `CgroupDef` declared without a
/// `.work(...)` or `.workload(...)` call falls back to a single
/// default [`WorkSpec`](crate::workload::WorkSpec) (SpinWait, Normal, ctx.workers_per_cgroup
/// workers) at apply-setup time. Pin the substitution by
/// asserting that workers were spawned and migrated into the
/// cgroup — without the fallback, no MoveTasks call would fire
/// and the cgroup would sit empty.
///
/// The tests above (e.g. `apply_setup_creates_cgroup_per_def`)
/// drive `CgroupDef::named` directly without a workload; this
/// test pins the fallback explicitly with a comment naming the
/// invariant so a future refactor that drops the default-work
/// substitution surfaces here with a clear failure message
/// rather than a generic "no MoveTasks" symptom.
#[test]
fn apply_setup_substitutes_default_workspec_when_works_empty() {
    mock_setup_state!(mock, topo, ctx, state);
    // No .work(...) and no .workload(...) — empty `works` vec.
    let def = CgroupDef::named("cg_default_work");
    assert!(
        def.works.is_empty(),
        "test premise: CgroupDef without .work() must start with empty works",
    );
    apply_setup_test(&ctx, &mut state, &[def])
        .expect("apply_setup with default-work substitution must succeed");
    let calls = mock.calls();
    // The substitution surfaces as a real worker spawn → at
    // least one MoveTasks call into the cgroup with a non-zero
    // pid count. MoveTasks records the count (usize) rather
    // than the Vec; matching `count > 0` pins both the call
    // presence and the fact that the spawned workload had
    // workers to migrate.
    assert!(
        calls.iter().any(|c| matches!(
            c,
            CgroupCall::MoveTasks(name, count) if name == "cg_default_work" && *count > 0
        )),
        "default-WorkSpec substitution must spawn workers and migrate them into the \
             cgroup; without it the empty `works` would leave the cgroup taskless. \
             Got: {calls:?}",
    );
    cleanup_state(&mut state);
}

// -- pcomm coalescing tests -----------------------------------------
//
// [`CgroupDef::pcomm`] propagates `pcomm` to every WorkSpec in the
// group AND records it on the CgroupDef itself. At apply_setup time,
// pcomm-bearing WorkSpecs trigger the fork-then-thread spawn path
// in [`WorkloadHandle::spawn`]: ONE container child is forked, its
// comm is set to `pcomm`, then N thread workers are spawned inside.
//
// Verification at this layer:
// - `move_tasks` receives a single PID per pcomm group (the
//   container), not one PID per thread.
// - `/proc/<container>/comm` carries `pcomm` byte-for-byte
//   (the builder rejects > 15 bytes — TASK_COMM_LEN-1 from
//   include/linux/sched.h — so the framework never feeds the
//   kernel a name that `__set_task_comm` would truncate).
// - Mixed pcomm/non-pcomm CgroupDefs in the same setup keep their
//   move_tasks shapes distinct: pcomm group → 1 PID, non-pcomm
//   group → N PIDs (one per worker fork).
// - pcomm + num_workers=0 is rejected by `resolve_num_workers`
//   like any other cgroup: a 0-worker cgroup emits no
//   `WorkerReport`s, so every downstream assertion would
//   vacuously pass. The pcomm path receives no exception —
//   the rejection happens before pcomm dispatch runs.

/// Read `/proc/<pid>/comm`. The kernel emits the comm bytes
/// followed by a single newline (see `comm_show` in
/// `fs/proc/base.c:1750`). The trailing newline is stripped.
fn read_proc_comm(pid: libc::pid_t) -> String {
    let raw = std::fs::read_to_string(format!("/proc/{pid}/comm"))
        .expect("/proc/<pid>/comm must be readable for live task");
    raw.trim_end_matches('\n').to_string()
}

/// `CgroupDef::named(...).pcomm("X").workers(2)` propagates
/// pcomm into the group's single (default) WorkSpec, and the
/// resulting spawn forks ONE container process — observable
/// here as a single PID delivered to `move_tasks`. Without
/// fork-then-thread coalescing, `move_tasks` would receive
/// 2 distinct fork-mode worker PIDs.
#[test]
fn apply_setup_pcomm_via_cgroup_def_forks_one_container() {
    mock_setup_state!(mock, topo, ctx, state);
    let defs = vec![CgroupDef::named("cg_pcomm").pcomm("leader").workers(2)];
    apply_setup_test(&ctx, &mut state, &defs).expect("pcomm apply_setup must succeed");
    let calls = mock.calls();
    // pcomm coalescing: exactly ONE PID is moved into the
    // cgroup (the container), not 2 (one per worker fork).
    assert!(
        calls.iter().any(|c| matches!(
            c,
            CgroupCall::MoveTasks(name, 1) if name == "cg_pcomm"
        )),
        "pcomm group must move exactly 1 PID (the container) into the cgroup; \
             got: {calls:?}",
    );
    cleanup_state(&mut state);
}

/// pcomm + per-thread comm coexist. The container holds `pcomm`
/// as its comm; each worker thread sets its own comm via the
/// post-spawn `prctl(PR_SET_NAME)`. Observable through
/// `/proc/<leader>/comm == pcomm` while each per-thread file
/// at `/proc/<leader>/task/<tid>/comm` carries the per-thread
/// `comm` (except the leader-thread's own task entry, whose
/// comm tracks `pcomm` since the leader called the
/// container-wide prctl).
///
/// `worker_pids()` for a pcomm group returns ONLY the leader
/// pid (the parent has no per-thread tids exported across the
/// process boundary). To verify per-thread comm we enumerate
/// `/proc/<leader>/task/` directly: every directory entry is
/// a kernel TID inside the container's tgid, and its `comm`
/// file is the kernel-side authoritative per-thread comm.
#[test]
fn apply_setup_pcomm_with_per_thread_comm() {
    mock_setup_state!(mock, topo, ctx, state);
    let defs = vec![
        CgroupDef::named("cg_named")
            .pcomm("leader")
            .comm("worker")
            .workers(2),
    ];
    apply_setup_test(&ctx, &mut state, &defs).expect("pcomm + comm apply_setup must succeed");
    // Wait for the container's post-fork init (prctl for leader
    // comm) before reading /proc. Same race as truncation test.
    std::thread::sleep(Duration::from_millis(200));
    // Take handles so the workers are observable before drop.
    let mut handles = std::mem::take(&mut state.handles);
    assert_eq!(handles.len(), 1, "one CgroupDef → one handle");
    let (_name, handle) = handles
        .pop()
        .expect("apply_setup must have pushed a handle for cg_named");
    let pids = handle.worker_pids();
    assert_eq!(
        pids.len(),
        1,
        "pcomm handle must report exactly 1 pid (the leader); got {}",
        pids.len(),
    );
    // For pcomm groups, worker_pids()[0] IS the leader pid
    // directly (the parent never observes the per-thread tids).
    let leader_pid = pids[0];
    // Container's leader-thread comm is the pcomm value.
    assert_eq!(
        read_proc_comm(leader_pid),
        "leader",
        "/proc/<leader>/comm must equal pcomm",
    );
    // Wait briefly for thread workers to install their
    // per-thread comm via prctl in worker_main. 100 ms is
    // generous against scheduler jitter on contended hosts;
    // sleep is intentional: waits for prctl to propagate.
    std::thread::sleep(Duration::from_millis(100));
    // Enumerate /proc/<leader>/task/ — every directory entry
    // is a TID inside the container's tgid. Read each TID's
    // comm. The leader-thread's own task entry tracks the
    // container-wide prctl (== "leader"); every other TID is
    // a worker thread that ran worker_main's prctl == "worker".
    let task_dir = format!("/proc/{leader_pid}/task");
    let entries: Vec<libc::pid_t> = std::fs::read_dir(&task_dir)
        .expect("/proc/<leader>/task must be readable for live container")
        .flatten()
        .filter_map(|e| e.file_name().to_str().and_then(|n| n.parse().ok()))
        .collect();
    assert!(
        entries.len() >= 3,
        "leader pid {leader_pid} must have leader + 2 worker threads in /proc/<leader>/task; \
             observed {} entries: {entries:?}",
        entries.len(),
    );
    let mut leader_seen = false;
    let mut worker_seen = 0usize;
    for tid in entries {
        let tcomm = read_proc_comm(tid);
        if tid == leader_pid {
            assert_eq!(
                tcomm, "leader",
                "/proc/<leader>/task/<leader>/comm must equal pcomm; got {tcomm:?}",
            );
            leader_seen = true;
        } else {
            assert_eq!(
                tcomm, "worker",
                "/proc/<leader>/task/{tid}/comm must equal per-thread comm 'worker'; \
                     got {tcomm:?}",
            );
            worker_seen += 1;
        }
    }
    assert!(
        leader_seen,
        "leader's own task entry must appear in /proc/<leader>/task",
    );
    assert_eq!(
        worker_seen, 2,
        "must observe exactly 2 worker threads with per-thread comm 'worker'; \
             saw {worker_seen}",
    );
    // Drop the handle (reaps container + threads) and clean up
    // the rest of state.
    drop(handle);
    cleanup_state(&mut state);
}

/// Mixed cgroup behavior: one CgroupDef has `pcomm`, another
/// does not. The pcomm group spawns via fork-then-thread
/// (1 PID into its cgroup), the non-pcomm group spawns via
/// normal fork mode (N PIDs into its cgroup). Pin both shapes
/// so the implementer cannot regress either path.
#[test]
fn apply_setup_mixed_pcomm_and_non_pcomm_groups() {
    mock_setup_state!(mock, topo, ctx, state);
    let defs = vec![
        // Group 1: pcomm — fork-then-thread, one container PID.
        CgroupDef::named("cg_pcomm").pcomm("threaded").workers(2),
        // Group 2: no pcomm — normal fork mode, two PIDs.
        CgroupDef::named("cg_fork").workers(2),
    ];
    apply_setup_test(&ctx, &mut state, &defs).expect("mixed apply_setup must succeed");
    let calls = mock.calls();
    // pcomm group: 1 PID move.
    assert!(
        calls.iter().any(|c| matches!(
            c,
            CgroupCall::MoveTasks(name, 1) if name == "cg_pcomm"
        )),
        "cg_pcomm must move 1 PID (container only) into its cgroup; \
             got: {calls:?}",
    );
    // Non-pcomm group: 2 PID move.
    assert!(
        calls.iter().any(|c| matches!(
            c,
            CgroupCall::MoveTasks(name, 2) if name == "cg_fork"
        )),
        "cg_fork must move 2 PIDs (one per fork worker) into its cgroup; \
             got: {calls:?}",
    );
    cleanup_state(&mut state);
}

/// `CgroupDef::pcomm("x").workers(0)` is rejected at
/// `apply_setup` like any other 0-worker cgroup. The pcomm
/// path receives no exception: `resolve_num_workers` runs
/// before pcomm dispatch and rejects `num_workers=0`
/// because a worker-less cgroup emits no [`WorkerReport`](crate::workload::WorkerReport)s,
/// vacuously passing every downstream assertion. The
/// rejection error names the cgroup and the offending
/// field so a typo'd worker count surfaces at setup
/// rather than as a silent green test.
///
/// Pin the rejection here so a regression that silently
/// no-ops the call (or forks an empty container) surfaces
/// as a passing `apply_setup_test` instead of the expected
/// `Err`.
#[test]
fn apply_setup_pcomm_with_zero_workers_is_rejected() {
    mock_setup_state!(mock, topo, ctx, state);
    let defs = vec![CgroupDef::named("cg_zero").pcomm("empty").workers(0)];
    let err = apply_setup_test(&ctx, &mut state, &defs)
        .expect_err("pcomm + 0 workers apply_setup must be rejected");
    let msg = format!("{err:#}");
    assert!(
        msg.contains("cg_zero"),
        "rejection error must name the cgroup: {msg}",
    );
    assert!(
        msg.contains("num_workers=0"),
        "rejection error must name the offending field: {msg}",
    );
    // No spawn ever happened: no PIDs were moved into the
    // cgroup. The cgroup itself may have been created
    // (`add_cgroup_no_cpuset` runs before the WorkSpec
    // resolution loop) — only `MoveTasks` is forbidden.
    let calls = mock.calls();
    let any_move = calls.iter().any(|c| {
        matches!(
            c,
            CgroupCall::MoveTasks(name, _) if name == "cg_zero"
        )
    });
    assert!(
        !any_move,
        "rejection must short-circuit before any move_tasks call \
             into cg_zero; got: {calls:?}",
    );
    cleanup_state(&mut state);
}

/// `CgroupDef::workers_pct(0.5)` on a cgroup with no explicit
/// cpuset resolves against the topology-usable cpuset and
/// produces `ceil(usable_cpus * 0.5)` workers. The mock_topo's
/// 4-CPU topology reserves the last CPU so usable=3 → ceil(3*0.5)=2.
/// Pins the no-cpuset path through the apply_setup pre-resolution.
#[test]
fn workers_pct_no_cpuset_resolves_against_usable_topology() {
    mock_setup_state!(mock, topo, ctx, state);
    let def = CgroupDef::named("cg_p").workers_pct(0.5);
    apply_setup_test(&ctx, &mut state, std::slice::from_ref(&def)).unwrap();
    let handle = &state
        .handles
        .iter()
        .find(|(n, _)| n == "cg_p")
        .expect("workload spawned")
        .1;
    assert_eq!(
        handle.worker_pids().len(),
        2,
        "workers_pct(0.5) on usable=3 CPUs must resolve to ceil(3*0.5)=2 \
             workers; got {} workers",
        handle.worker_pids().len(),
    );
    cleanup_state(&mut state);
}

/// `CgroupDef::workers_pct(0.34)` on an LLC-restricted cpuset
/// (size 4 here because the mock topology is 1 LLC × 4 cores)
/// resolves to ceil(4 * 0.34) = 2 workers. Pins the with-cpuset
/// path: workers_pct denominator is the resolved cpuset size,
/// not the full topology.
#[test]
fn workers_pct_with_cpuset_resolves_against_cpuset_size() {
    mock_setup_state!(mock, topo, ctx, state);
    let def = CgroupDef::named("cg_p")
        .cpuset(CpusetSpec::Llc(0))
        .workers_pct(0.34);
    apply_setup_test(&ctx, &mut state, std::slice::from_ref(&def)).unwrap();
    let handle = &state
        .handles
        .iter()
        .find(|(n, _)| n == "cg_p")
        .expect("workload spawned")
        .1;
    assert_eq!(
        handle.worker_pids().len(),
        2,
        "workers_pct(0.34) on Llc(0)=4 CPUs must resolve to ceil(4*0.34)=2 \
             workers; got {} workers",
        handle.worker_pids().len(),
    );
    cleanup_state(&mut state);
}

/// `workers_pct(2.0)` accepts oversubscription. 4-CPU LLC * 2.0 = 8.
/// Pins that >1.0 fractions are NOT rejected at apply time.
#[test]
fn workers_pct_above_one_accepts_oversubscription() {
    mock_setup_state!(mock, topo, ctx, state);
    let def = CgroupDef::named("cg_p")
        .cpuset(CpusetSpec::Llc(0))
        .workers_pct(2.0);
    apply_setup_test(&ctx, &mut state, std::slice::from_ref(&def)).unwrap();
    let handle = &state
        .handles
        .iter()
        .find(|(n, _)| n == "cg_p")
        .expect("workload spawned")
        .1;
    assert_eq!(
        handle.worker_pids().len(),
        8,
        "workers_pct(2.0) on Llc(0)=4 CPUs must resolve to ceil(4*2.0)=8 \
             workers (oversubscription); got {}",
        handle.worker_pids().len(),
    );
    cleanup_state(&mut state);
}

/// Setting both `workers(N)` and `workers_pct(p)` is rejected at
/// apply-setup time regardless of the builder-call order. Pins
/// BOTH orderings against the mutex-asymmetry concern.
#[test]
fn workers_pct_then_workers_rejected_at_apply() {
    mock_setup_state!(mock, topo, ctx, state);
    let def = CgroupDef::named("cg_p")
        .cpuset(CpusetSpec::Llc(0))
        .workers_pct(0.5)
        .workers(2);
    let err = apply_setup_test(&ctx, &mut state, std::slice::from_ref(&def))
        .expect_err("workers_pct + workers must be rejected");
    let msg = format!("{err}");
    assert!(
        msg.contains("workers_pct") && msg.contains("workers(2)"),
        "error must name both workers and workers_pct: {msg}",
    );
    cleanup_state(&mut state);
}

#[test]
fn workers_then_workers_pct_rejected_at_apply() {
    mock_setup_state!(mock, topo, ctx, state);
    let def = CgroupDef::named("cg_p")
        .cpuset(CpusetSpec::Llc(0))
        .workers(2)
        .workers_pct(0.5);
    let err = apply_setup_test(&ctx, &mut state, std::slice::from_ref(&def))
        .expect_err("workers + workers_pct must be rejected");
    let msg = format!("{err}");
    assert!(
        msg.contains("workers_pct") && msg.contains("workers(2)"),
        "error must name both workers and workers_pct: {msg}",
    );
    cleanup_state(&mut state);
}

/// `CgroupDef::workers_pct(p)` stores the fraction on
/// `works[0].workers_pct` without pre-resolving, and leaves
/// `works[0].num_workers` unset. Pins the construction-time
/// invariant: resolution is deferred to apply-setup (which has
/// access to the cpuset size). A future regression that
/// eagerly resolved at construction would silently break the
/// apply-time-resolution contract; this test catches it.
#[test]
fn workers_pct_construction_stores_pct_without_resolving() {
    let def = CgroupDef::named("cg_p").workers_pct(0.5);
    let work = &def.works[0];
    assert_eq!(
        work.workers_pct,
        Some(0.5),
        "workers_pct must be stored verbatim at construction; got {:?}",
        work.workers_pct,
    );
    assert_eq!(
        work.num_workers, None,
        "num_workers must be left unset at construction (apply-setup resolves); got {:?}",
        work.num_workers,
    );
}

/// `workers_pct` uses ceil() for the cpuset→worker count
/// resolution. Pin the rounding across four cases covering the
/// integer / fractional / just-above / just-below boundaries:
/// an exact integer product stays at that integer; any non-zero
/// remainder rounds UP regardless of which side of the half it
/// falls on. Catches a future regression to round() or floor()
/// rounding modes that would produce off-by-one worker counts at
/// boundary fractions (`round` and `floor` differ from `ceil`
/// in opposite directions, so these four cases pin ceil
/// uniquely).
#[test]
fn workers_pct_rounding_is_ceil_not_round_or_floor() {
    mock_setup!(mock, topo, ctx);

    // Exact integer product: 4 * 0.5 = 2.0 (exact in IEEE 754
    // because 0.5 = 2^-1 is exactly representable) → ceil(2.0) = 2.
    // round and floor also give 2 here; this case doesn't
    // distinguish ceil, it's a baseline.
    let mut state = StepState::empty(&ctx);
    let def_exact = CgroupDef::named("cg_exact")
        .cpuset(CpusetSpec::Llc(0))
        .workers_pct(0.5);
    apply_setup_test(&ctx, &mut state, std::slice::from_ref(&def_exact)).unwrap();
    let handle = &state
        .handles
        .iter()
        .find(|(n, _)| n == "cg_exact")
        .expect("workload spawned")
        .1;
    assert_eq!(
        handle.worker_pids().len(),
        2,
        "workers_pct(0.5) on 4 CPUs (exact 2.0) must produce 2 workers",
    );
    cleanup_state(&mut state);

    // Mid-fractional product: 4 * 0.6 ≈ 2.3999... → ceil = 3.
    // round (nearest) gives 2; floor gives 2. ceil gives 3.
    // Distinguishes ceil from round.
    let mut state = StepState::empty(&ctx);
    let def_mid = CgroupDef::named("cg_mid")
        .cpuset(CpusetSpec::Llc(0))
        .workers_pct(0.6);
    apply_setup_test(&ctx, &mut state, std::slice::from_ref(&def_mid)).unwrap();
    let handle = &state
        .handles
        .iter()
        .find(|(n, _)| n == "cg_mid")
        .expect("workload spawned")
        .1;
    assert_eq!(
        handle.worker_pids().len(),
        3,
        "workers_pct(0.6) on 4 CPUs (≈2.4) must ceil to 3 workers; round (2) or floor (2) would be wrong",
    );
    cleanup_state(&mut state);

    // Just above an integer: 4 * 0.51 ≈ 2.04 → ceil = 3. Pins
    // that ANY non-zero remainder rounds up, not just near-half.
    let mut state = StepState::empty(&ctx);
    let def_just_over = CgroupDef::named("cg_over")
        .cpuset(CpusetSpec::Llc(0))
        .workers_pct(0.51);
    apply_setup_test(&ctx, &mut state, std::slice::from_ref(&def_just_over)).unwrap();
    let handle = &state
        .handles
        .iter()
        .find(|(n, _)| n == "cg_over")
        .expect("workload spawned")
        .1;
    assert_eq!(
        handle.worker_pids().len(),
        3,
        "workers_pct(0.51) on 4 CPUs (≈2.04) must ceil to 3 workers; round (2) or floor (2) would be wrong",
    );
    cleanup_state(&mut state);

    // Just below an integer: 4 * 0.49 ≈ 1.96 → ceil = 2.
    // floor gives 1; round gives 2. Distinguishes ceil/round
    // from floor — completes the rounding-mode coverage.
    let mut state = StepState::empty(&ctx);
    let def_just_under = CgroupDef::named("cg_under")
        .cpuset(CpusetSpec::Llc(0))
        .workers_pct(0.49);
    apply_setup_test(&ctx, &mut state, std::slice::from_ref(&def_just_under)).unwrap();
    let handle = &state
        .handles
        .iter()
        .find(|(n, _)| n == "cg_under")
        .expect("workload spawned")
        .1;
    assert_eq!(
        handle.worker_pids().len(),
        2,
        "workers_pct(0.49) on 4 CPUs (≈1.96) must ceil to 2 workers; floor (1) would be wrong",
    );
    cleanup_state(&mut state);
}

/// Setup-spawned workers (workers from apply_setup-time
/// `workers_pct` resolution) keep their pid set across
/// subsequent `Op::SetCpuset` cpuset changes. Pins that
/// `Op::SetCpuset`'s apply arm (`Op::SetCpuset { cgroup, cpus }`
/// in `dispatch.rs`'s `apply_op`) is NOT a
/// `resolve_workers_pct` call site — the arm validates +
/// resolves the CpusetSpec, calls `ctx.cgroups.set_cpuset`,
/// and records the new cpuset via `state.record_cpuset`, but
/// touches no WorkSpec / handle state.
///
/// `resolve_workers_pct` does have TWO call sites overall
/// (apply_setup and the Op::Spawn(SpawnPlacement::Cgroup) arm),
/// so a test author who issues an `Op::spawn(SpawnPlacement::cgroup(_), _)`
/// AFTER an `Op::SetCpuset` will get fresh resolution against
/// the then-current cpuset — that's the Op::Spawn integration
/// layer's responsibility and is verified by `op_spawn_*` tests,
/// not here. This test catches a future regression that adds
/// re-resolution INTO the Op::SetCpuset apply branch
/// (re-counting apply-setup workers when the cpuset narrows).
///
/// The test drives Op::SetCpuset through `apply_ops_test` (the
/// real Op-dispatch wrapper) instead of calling
/// `ctx.cgroups.set_cpuset` directly — that distinction matters
/// because a regression that added `resolve_workers_pct` inside
/// the Op match arm would NOT be caught by a direct set_cpuset
/// call that bypasses dispatch.
#[test]
fn workers_pct_setup_workers_survive_op_setcpuset_narrowing() {
    mock_setup_state!(mock, topo, ctx, state);
    let def = CgroupDef::named("cg_stable")
        .cpuset(CpusetSpec::Llc(0))
        .workers_pct(0.5);
    apply_setup_test(&ctx, &mut state, std::slice::from_ref(&def)).unwrap();
    let initial_count = state
        .handles
        .iter()
        .find(|(n, _)| n == "cg_stable")
        .expect("workload spawned")
        .1
        .worker_pids()
        .len();
    assert_eq!(
        initial_count, 2,
        "baseline: workers_pct(0.5) on Llc(0)=4 CPUs → ceil(4*0.5)=2 workers",
    );

    // Drive Op::SetCpuset through the real apply_ops dispatch
    // (NOT just ctx.cgroups.set_cpuset, which would bypass the
    // Op match arm where a regression might add re-resolution).
    let narrower: std::collections::BTreeSet<usize> = [0usize, 1].into_iter().collect();
    apply_ops_test(
        &ctx,
        &mut state,
        &[Op::SetCpuset {
            cgroup: "cg_stable".into(),
            cpus: CpusetSpec::Exact(narrower.clone()),
        }],
    )
    .expect("Op::SetCpuset applies");

    // Verify the narrowing actually took effect — without this
    // assertion, a silently-no-op set_cpuset would make the
    // worker-stability claim trivially true. StepState's
    // `cpusets` HashMap is the step-local cpuset bookkeeping
    // that Op::SetCpuset's `state.record_cpuset` call writes to.
    assert_eq!(
        state
            .cpusets
            .get("cg_stable")
            .expect("cg_stable has recorded cpuset"),
        &narrower,
        "Op::SetCpuset must persist the narrower set in state.cpusets via record_cpuset",
    );

    let after_count = state
        .handles
        .iter()
        .find(|(n, _)| n == "cg_stable")
        .expect("workload still present")
        .1
        .worker_pids()
        .len();
    assert_eq!(
        after_count, initial_count,
        "Op::SetCpuset apply arm must NOT re-resolve workers_pct; \
             setup-spawned worker count must remain {initial_count}; got {after_count}",
    );
    cleanup_state(&mut state);
}

/// Pathological `workers_pct` values rejected at construction:
/// NaN, INFINITY, negative values, and zero all panic via
/// `CgroupDef::workers_pct`'s `assert!` (its `pct must be finite and > 0.0` check).
/// Pin all four rejection paths so a future regression that
/// loosens the gate (e.g. accepts NaN as "use default") fails
/// here loudly.
#[test]
fn workers_pct_pathological_finite_rejected_at_construction() {
    // Non-finite NaN → CgroupDef::workers_pct panics; std::panic::catch_unwind verifies.
    let nan_panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = CgroupDef::named("cg_nan").workers_pct(f64::NAN);
    }));
    assert!(
        nan_panic.is_err(),
        "CgroupDef::workers_pct(NaN) must panic at construction",
    );

    let inf_panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = CgroupDef::named("cg_inf").workers_pct(f64::INFINITY);
    }));
    assert!(
        inf_panic.is_err(),
        "CgroupDef::workers_pct(INFINITY) must panic at construction",
    );

    let neg_panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = CgroupDef::named("cg_neg").workers_pct(-1.0);
    }));
    assert!(
        neg_panic.is_err(),
        "CgroupDef::workers_pct(-1.0) must panic at construction",
    );

    let zero_panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = CgroupDef::named("cg_zero").workers_pct(0.0);
    }));
    assert!(
        zero_panic.is_err(),
        "CgroupDef::workers_pct(0.0) must panic at construction",
    );
}

/// A very large finite `workers_pct` (e.g. `1e100`) passes the
/// finite + positive construction gate but produces `usize::MAX`
/// when `resolve_workers_pct` evaluates
/// `(cpuset_cpus as f64 * pct).ceil() as usize` — Rust's
/// saturating float-to-int cast (RFC 2484, stable since 1.45)
/// clamps any finite f64 exceeding the integer range to the
/// bound. The product `4.0 * 1e100 = 4e100` is finite (well
/// below `f64::MAX ≈ 1.798e308`) but far exceeds `usize::MAX`
/// (~`1.844e19` on 64-bit), so the cast saturates.
///
/// Calls `resolve_workers_pct` directly on a constructed
/// [`WorkSpec`](crate::workload::WorkSpec) — pins the FRAMEWORK contract (current behavior
/// returns `Ok(num_workers=Some(usize::MAX))`). A future
/// regression that added a saturation guard
/// (e.g. `if scaled == usize::MAX { bail!("too large") }`)
/// would flip this to `Err` and trip the test. The spawn path
/// is NOT exercised — spawning `usize::MAX` workers would hang
/// the host.
#[test]
fn workers_pct_pathological_finite_large_saturates_usize() {
    let work = crate::workload::WorkSpec::default().workers_pct(1e100);
    let resolved = work
        .resolve_workers_pct(4, "cg_saturate")
        .expect("current framework does not gate against usize::MAX saturation");
    assert_eq!(
        resolved.num_workers,
        Some(usize::MAX),
        "extreme pct saturates `num_workers` to `usize::MAX` per Rust's saturating \
             float-to-int `as` cast (RFC 2484); got {:?}",
        resolved.num_workers,
    );
}

/// Empty cpuset + MULTIPLE [`WorkSpec`](crate::workload::WorkSpec)s with distinct `workers_pct`
/// values: the diagnostic must enumerate ALL pct values, not just
/// the first. An earlier diagnostic used
/// `find_map(|w| w.workers_pct)` which dropped subsequent pcts and
/// hid that other WorkSpecs in the cgroup also had pct configured.
#[test]
fn workers_pct_empty_cpuset_multi_workspec_lists_all_pcts() {
    mock_setup_state!(mock, topo, ctx, state);
    let def = CgroupDef::named("cg_multi")
        .cpuset(CpusetSpec::Exact(std::collections::BTreeSet::new()))
        .workers_pct(0.3)
        .work(crate::workload::WorkSpec::default().workers_pct(0.7))
        .work(crate::workload::WorkSpec::default().workers_pct(0.5));
    let err = apply_setup_test(&ctx, &mut state, std::slice::from_ref(&def))
        .expect_err("multi-workspec workers_pct on empty cpuset must reject");
    let msg = format!("{err}");
    assert!(
        msg.contains("0.3") && msg.contains("0.7") && msg.contains("0.5"),
        "diagnostic must name ALL configured workers_pct values, not just the first: {msg}",
    );
    assert!(
        msg.contains("cpuset of 0"),
        "diagnostic must still name the empty cpuset size: {msg}",
    );
    cleanup_state(&mut state);
}

/// Empty cpuset + a single [`WorkSpec`](crate::workload::WorkSpec) that sets BOTH `workers(N)`
/// AND `workers_pct(p)`: the framework must emit a dual-set-specific
/// bail (the more fundamental misconfiguration) rather than letting
/// validate's empty-Exact mask preempt it OR the workers_pct-only
/// empty-cpuset diagnostic claim "would resolve to 0 workers" (which
/// is misleading when workers(N) explicitly sets the count). The
/// operator must pick one of `workers` or `workers_pct` before the
/// empty-cpuset question is meaningful. Case (1) of the
/// empty-cpuset handling in apply_setup surfaces this bail inline
/// with the "BOTH workers ... empty cpuset would otherwise mask"
/// wording.
#[test]
fn workers_pct_empty_cpuset_dual_set_bails_with_dedicated_error() {
    mock_setup_state!(mock, topo, ctx, state);
    let def = CgroupDef::named("cg_both")
        .cpuset(CpusetSpec::Exact(std::collections::BTreeSet::new()))
        .workers(2)
        .workers_pct(0.5);
    let err = apply_setup_test(&ctx, &mut state, std::slice::from_ref(&def))
        .expect_err("workers + workers_pct on empty cpuset must reject");
    let msg = format!("{err}");
    assert!(
        msg.contains("BOTH workers"),
        "dual-set error must fire first; got the empty-cpuset diagnostic instead: {msg}",
    );
    assert!(
        !msg.contains("cpuset of 0"),
        "workers_pct-only empty-cpuset diagnostic must NOT preempt the more fundamental dual-set error: {msg}",
    );
    assert!(
        msg.contains("empty cpuset would otherwise mask"),
        "dual-set bail must include the case-(1)-specific trailing context that \
             explains why this fired at apply_setup rather than at the deeper resolve \
             path: {msg}",
    );
    cleanup_state(&mut state);
}

/// A cgroup whose `cpuset_spec` resolves to an empty CPU set
/// AND that does NOT use `workers_pct` must still bail at
/// apply_setup — silently writing an empty mask would leave the
/// cgroup with no CPUs assigned, downstream worker spawns would
/// fail or produce vacuous assertions, and the operator would
/// have no signal that they misconfigured the spec. Uses
/// `Range { 0.0, 0.1 }` on a 4-CPU mock topology: validate
/// accepts because `0.0 < 0.1` and both fracs are in `[0, 1]`,
/// but resolve computes `start = 4 * 0.0 = 0` and `end =
/// (4 * 0.1) as usize = 0`, yielding an empty slice. This is
/// the canonical "passes validate but resolves to empty" case
/// — `Range { 0.0, 0.0 }` would be rejected by validate's
/// `start_frac >= end_frac` guard in `CpusetSpec::validate`, so the
/// fraction must be small but non-zero to thread the needle.
/// Distinct from the `workers_pct`-driven empty-cpuset bails:
/// no fraction is set, so the diagnostic should cite the
/// cpuset_spec itself rather than a fraction-on-zero-CPUs
/// framing.
#[test]
fn empty_resolved_cpuset_without_workers_pct_bails_in_apply_setup() {
    mock_setup_state!(mock, topo, ctx, state);
    let def = CgroupDef::named("cg_empty_range").cpuset(CpusetSpec::Range {
        start_frac: 0.0,
        end_frac: 0.1,
    });
    let err = apply_setup_test(&ctx, &mut state, std::slice::from_ref(&def))
        .expect_err("empty-resolved cpuset must reject even without workers_pct");
    let msg = format!("{err}");
    assert!(
        msg.contains("cg_empty_range"),
        "diagnostic must name the cgroup: {msg}",
    );
    assert!(
        msg.contains("resolved to 0 CPU(s)"),
        "diagnostic must name the zero-CPU resolution: {msg}",
    );
    assert!(
        !msg.contains("workers_pct"),
        "diagnostic must NOT cite workers_pct when none is set; \
             that would mis-direct the operator to a knob they didn't \
             configure: {msg}",
    );
    cleanup_state(&mut state);
}

/// `Op::SetCpuset` mid-scenario must also bail when the new spec
/// resolves to an empty CPU set, symmetric with apply_setup.
/// Silently re-masking a live cgroup to empty would leave its
/// running workers without CPUs and downstream assertions would
/// vacuously pass. The diagnostic must cite the target cgroup
/// and the spec that resolved to empty so the operator knows
/// which mid-scenario narrow produced the empty resolution.
#[test]
fn op_set_cpuset_narrow_to_empty_bails() {
    mock_setup_state!(mock, topo, ctx, state);
    // Establish a live cgroup with a valid cpuset first.
    apply_setup_test(
        &ctx,
        &mut state,
        std::slice::from_ref(&CgroupDef::named("cg_narrow").cpuset(CpusetSpec::Llc(0))),
    )
    .unwrap();
    // Now try to narrow it via Op::SetCpuset to an empty range.
    // `Range { 0.0, 0.1 }` passes validate (start < end, both in
    // [0, 1]) but resolves empty on a 4-CPU topology: end =
    // (4 * 0.1) as usize = 0, so the slice is [0..0] = empty.
    let err = apply_ops_test(
        &ctx,
        &mut state,
        &[Op::SetCpuset {
            cgroup: std::borrow::Cow::Borrowed("cg_narrow"),
            cpus: CpusetSpec::Range {
                start_frac: 0.0,
                end_frac: 0.1,
            },
        }],
    )
    .expect_err("Op::SetCpuset narrowing to empty must reject");
    let msg = format!("{err}");
    assert!(
        msg.contains("cg_narrow"),
        "diagnostic must name the target cgroup: {msg}",
    );
    assert!(
        msg.contains("resolved to 0 CPU(s)"),
        "diagnostic must name the zero-CPU resolution: {msg}",
    );
    assert!(
        msg.contains("Op::SetCpuset"),
        "diagnostic must identify the Op layer so the operator \
             knows this came from a mid-scenario narrow, not setup: \
             {msg}",
    );
    assert!(
        msg.contains("Op::ClearCpuset"),
        "diagnostic must point the operator at the right \
             primitive for the 'release cpuset restriction' intent \
             so a regression that drops the Op::ClearCpuset \
             direction (leading users to the workaround \
             `Range {{ 0.0, 1.0 }}` instead) is caught: {msg}",
    );
    cleanup_state(&mut state);
}

/// `workers_pct` against an empty cpuset (Exact({})) resolves to
/// 0 workers and bails with a diagnostic that names the cpuset
/// size and the requested fraction. Pin the
/// "loud reject with diagnostic" caveat — the message must carry
/// the diagnostic fields so a future refactor that drops them is
/// caught here, not by a confused user.
#[test]
fn workers_pct_empty_cpuset_rejects_with_diagnostic() {
    mock_setup_state!(mock, topo, ctx, state);
    let def = CgroupDef::named("cg_e")
        .cpuset(CpusetSpec::Exact(std::collections::BTreeSet::new()))
        .workers_pct(0.9);
    let err = apply_setup_test(&ctx, &mut state, std::slice::from_ref(&def))
        .expect_err("workers_pct on empty cpuset must reject");
    let msg = format!("{err}");
    assert!(
        msg.contains("workers_pct(0.9)") && msg.contains("cpuset of 0"),
        "diagnostic must name the requested fraction AND cpuset size: {msg}",
    );
    cleanup_state(&mut state);
}

/// `Op::Spawn` with `SpawnPlacement::Cgroup` and `WorkSpec::workers_pct`
/// resolves against the cgroup's currently-recorded cpuset,
/// mirroring the apply_setup path. Pin so a future regression that
/// drops the workers_pct pre-resolution from the Op::Spawn(Cgroup)
/// arm (silently falling back to `ctx.workers_per_cgroup` and
/// ignoring the user's fraction) is caught.
#[test]
fn op_spawn_cgroup_pct_resolves_against_cgroup_cpuset() {
    mock_setup_state!(mock, topo, ctx, state);
    // Set up an empty cgroup with an explicit cpuset first.
    apply_setup_test(
        &ctx,
        &mut state,
        std::slice::from_ref(&CgroupDef::named("cg_spawn").cpuset(CpusetSpec::Llc(0))),
    )
    .unwrap();
    // Drop the apply_setup default-spawned workload so the Spawn
    // we issue below is the only handle for cg_spawn.
    state.handles.clear();
    // Now Spawn a WorkSpec that uses workers_pct(0.5).
    // Llc(0) = 4 CPUs → ceil(4 * 0.5) = 2 workers.
    let work = crate::workload::WorkSpec::default().workers_pct(0.5);
    apply_ops_test(
        &ctx,
        &mut state,
        &[Op::spawn(SpawnPlacement::cgroup("cg_spawn"), work)],
    )
    .unwrap();
    let handle = &state
        .handles
        .iter()
        .find(|(n, _)| n == "cg_spawn")
        .expect("Op::Spawn(Cgroup) workload registered")
        .1;
    assert_eq!(
        handle.worker_pids().len(),
        2,
        "Op::Spawn(Cgroup) workers_pct(0.5) on Llc(0)=4 must resolve to 2 workers; \
             got {}",
        handle.worker_pids().len(),
    );
    cleanup_state(&mut state);
}

/// `Op::Spawn` with `SpawnPlacement::Cgroup` and BOTH workers and
/// workers_pct set is rejected the same way apply_setup rejects it
/// — the resolution helper is shared so the diagnostic is
/// identical.
#[test]
fn op_spawn_cgroup_pct_dual_set_rejected() {
    mock_setup_state!(mock, topo, ctx, state);
    apply_setup_test(
        &ctx,
        &mut state,
        std::slice::from_ref(&CgroupDef::named("cg_x").cpuset(CpusetSpec::Llc(0))),
    )
    .unwrap();
    state.handles.clear();
    let work = crate::workload::WorkSpec::default()
        .workers(2)
        .workers_pct(0.5);
    let err = apply_ops_test(
        &ctx,
        &mut state,
        &[Op::spawn(SpawnPlacement::cgroup("cg_x"), work)],
    )
    .expect_err("Op::Spawn(Cgroup) dual-set must reject");
    let msg = format!("{err}");
    assert!(
        msg.contains("workers_pct") && msg.contains("workers(2)"),
        "Op::Spawn(Cgroup) diagnostic must name both knobs: {msg}",
    );
    cleanup_state(&mut state);
}

/// `Ctx::cpuset_cpus(&spec)` returns the size of
/// `spec.resolve(ctx)` for every CpusetSpec variant. Pinned via a
/// single-pass equivalence check across all variants so a future
/// CpusetSpec variant added without updating cpuset_cpus stays
/// detectable.
#[test]
fn ctx_cpuset_cpus_matches_resolve_len() {
    mock_setup!(mock, topo, ctx);
    let specs = [
        CpusetSpec::Llc(0),
        CpusetSpec::Numa(0),
        CpusetSpec::Range {
            start_frac: 0.0,
            end_frac: 0.5,
        },
        CpusetSpec::Disjoint { index: 0, of: 2 },
        CpusetSpec::Overlap {
            index: 0,
            of: 2,
            frac: 0.5,
        },
        CpusetSpec::Exact([0usize, 1, 2].iter().copied().collect()),
    ];
    for spec in &specs {
        assert_eq!(
            ctx.cpuset_cpus(spec),
            spec.resolve(&ctx).len(),
            "ctx.cpuset_cpus drift on {spec:?}",
        );
    }
}

// -----------------------------------------------------------------
// Kernel-op integration: 4 Op::*Kernel* arms dispatch through
// apply_ops + a thread-local SnapshotBridge kernel-op callback.
// -----------------------------------------------------------------

/// `Op::WriteKernelHot` dispatched via `apply_ops` invokes the
/// installed bridge kernel-op callback with the correct mode +
/// direction + entries, and the bridge's drain log records the
/// reply. Pins the executor arm's mapping from variant fields
/// to wire payload — a regression that flipped Hot↔Cold or
/// Write↔Read or dropped a write entry surfaces here.
#[test]
fn apply_ops_write_kernel_hot_dispatches_via_bridge() {
    use std::sync::Arc;
    let captured = Arc::new(std::sync::Mutex::new(
        None::<crate::vmm::wire::KernelOpRequestPayload>,
    ));
    let captured_clone = captured.clone();
    let kernel_op_cb: crate::scenario::snapshot::KernelOpCallback = Arc::new(move |req| {
        *captured_clone.lock().unwrap() = Some(req.clone());
        crate::vmm::wire::KernelOpReplyPayload {
            request_id: req.request_id,
            success: true,
            reason: String::new(),
            read_values: vec![],
        }
    });
    let bridge = crate::scenario::snapshot::SnapshotBridge::new(Arc::new(|_| None))
        .with_kernel_op(kernel_op_cb);
    let bridge_clone = bridge.clone();
    let _bg = bridge.set_thread_local();
    mock_setup_state!(mock, topo, ctx, state);
    let ops = vec![Op::write_kernel_hot(
        KernelTarget::symbol("test_field"),
        KernelValue::u64(42),
    )];
    apply_ops_test(&ctx, &mut state, &ops).expect("WriteKernelHot must dispatch");
    let req = captured.lock().unwrap().take().expect("callback must fire");
    assert_eq!(req.mode, crate::vmm::wire::KernelOpMode::Hot);
    assert_eq!(req.direction, crate::vmm::wire::KernelOpDirection::Write);
    assert_eq!(req.entries.len(), 1);
    match &req.entries[0].target {
        crate::vmm::wire::KernelOpTarget::Symbol(s) => assert_eq!(s, "test_field"),
        other => panic!("unexpected target shape: {other:?}"),
    }
    match req.entries[0].value {
        crate::vmm::wire::KernelOpValue::U64(42) => {}
        ref other => panic!("unexpected value shape: {other:?}"),
    }
    assert_eq!(bridge_clone.drain_kernel_ops().len(), 1);
    cleanup_state(&mut state);
}

/// `Op::WriteKernelCold` dispatches with `KernelOpMode::Cold`
/// (vs Hot) — pins the per-arm mode mapping. A regression that
/// reused Hot's payload-build path for Cold would surface here.
#[test]
fn apply_ops_write_kernel_cold_dispatches_with_cold_mode() {
    use std::sync::Arc;
    let captured = Arc::new(std::sync::Mutex::new(
        None::<crate::vmm::wire::KernelOpRequestPayload>,
    ));
    let captured_clone = captured.clone();
    let kernel_op_cb: crate::scenario::snapshot::KernelOpCallback = Arc::new(move |req| {
        *captured_clone.lock().unwrap() = Some(req.clone());
        crate::vmm::wire::KernelOpReplyPayload {
            request_id: req.request_id,
            success: true,
            reason: String::new(),
            read_values: vec![],
        }
    });
    let bridge = crate::scenario::snapshot::SnapshotBridge::new(Arc::new(|_| None))
        .with_kernel_op(kernel_op_cb);
    let _bg = bridge.set_thread_local();
    mock_setup_state!(mock, topo, ctx, state);
    let ops = vec![Op::write_kernel_cold_batch(vec![
        (
            KernelTarget::per_cpu_field("runqueues", "clock", 0),
            KernelValue::u64(100),
        ),
        (
            KernelTarget::per_cpu_field("runqueues", "clock", 1),
            KernelValue::u64(200),
        ),
    ])];
    apply_ops_test(&ctx, &mut state, &ops).expect("WriteKernelCold must dispatch");
    let req = captured.lock().unwrap().take().expect("callback must fire");
    assert_eq!(req.mode, crate::vmm::wire::KernelOpMode::Cold);
    assert_eq!(req.direction, crate::vmm::wire::KernelOpDirection::Write);
    assert_eq!(req.entries.len(), 2, "batch must carry both entries");
    cleanup_state(&mut state);
}

/// `Op::ReadKernelHot` dispatches with the right tag + width
/// hint. The wire payload's value-slot mirrors the
/// `KernelValueWidth` chosen at the variant level: U32 picks
/// the u32 read family, U64 picks u64, Bytes(N) picks the
/// N-byte read.
#[test]
fn apply_ops_read_kernel_hot_dispatches_with_width_u32() {
    use std::sync::Arc;
    let captured = Arc::new(std::sync::Mutex::new(
        None::<crate::vmm::wire::KernelOpRequestPayload>,
    ));
    let captured_clone = captured.clone();
    let kernel_op_cb: crate::scenario::snapshot::KernelOpCallback = Arc::new(move |req| {
        *captured_clone.lock().unwrap() = Some(req.clone());
        crate::vmm::wire::KernelOpReplyPayload {
            request_id: req.request_id,
            success: true,
            reason: String::new(),
            read_values: vec![crate::vmm::wire::KernelOpValue::U32(7)],
        }
    });
    let bridge = crate::scenario::snapshot::SnapshotBridge::new(Arc::new(|_| None))
        .with_kernel_op(kernel_op_cb);
    let bridge_clone = bridge.clone();
    let _bg = bridge.set_thread_local();
    mock_setup_state!(mock, topo, ctx, state);
    let ops = vec![Op::read_kernel_hot(
        "scratch_u32",
        KernelTarget::symbol("some_u32"),
        KernelValueWidth::u32(),
    )];
    apply_ops_test(&ctx, &mut state, &ops).expect("ReadKernelHot must dispatch");
    let req = captured.lock().unwrap().take().expect("callback must fire");
    assert_eq!(req.mode, crate::vmm::wire::KernelOpMode::Hot);
    assert_eq!(req.direction, crate::vmm::wire::KernelOpDirection::Read);
    assert_eq!(req.tag, "scratch_u32");
    match req.entries[0].value {
        crate::vmm::wire::KernelOpValue::U32(_) => {}
        ref other => panic!("u32 width hint must emit U32 slot, got {other:?}"),
    }
    // Single-tag convenience accessor returns the U32 read-back.
    match bridge_clone.kernel_op_value("scratch_u32") {
        Some(crate::vmm::wire::KernelOpValue::U32(7)) => {}
        other => panic!("kernel_op_value lookup mismatch: {other:?}"),
    }
    cleanup_state(&mut state);
}

/// `Op::ReadKernelCold` mirrors `Op::ReadKernelHot` with cold
/// mode + Bytes width. Pins the Bytes width hint passing
/// through to the wire payload's value slot.
#[test]
fn apply_ops_read_kernel_cold_dispatches_with_width_bytes() {
    use std::sync::Arc;
    let captured = Arc::new(std::sync::Mutex::new(
        None::<crate::vmm::wire::KernelOpRequestPayload>,
    ));
    let captured_clone = captured.clone();
    let kernel_op_cb: crate::scenario::snapshot::KernelOpCallback = Arc::new(move |req| {
        *captured_clone.lock().unwrap() = Some(req.clone());
        crate::vmm::wire::KernelOpReplyPayload {
            request_id: req.request_id,
            success: true,
            reason: String::new(),
            read_values: vec![crate::vmm::wire::KernelOpValue::Bytes(vec![0xAA; 16])],
        }
    });
    let bridge = crate::scenario::snapshot::SnapshotBridge::new(Arc::new(|_| None))
        .with_kernel_op(kernel_op_cb);
    let _bg = bridge.set_thread_local();
    mock_setup_state!(mock, topo, ctx, state);
    let ops = vec![Op::read_kernel_cold(
        "scratch_bytes",
        KernelTarget::kva(0xffff_c900_0000_1000),
        KernelValueWidth::bytes(16),
    )];
    apply_ops_test(&ctx, &mut state, &ops).expect("ReadKernelCold must dispatch");
    let req = captured.lock().unwrap().take().expect("callback must fire");
    assert_eq!(req.mode, crate::vmm::wire::KernelOpMode::Cold);
    assert_eq!(req.direction, crate::vmm::wire::KernelOpDirection::Read);
    match &req.entries[0].value {
        crate::vmm::wire::KernelOpValue::Bytes(b) => {
            assert_eq!(b.len(), 16, "Bytes(16) width hint must emit a 16-byte slot");
        }
        other => panic!("Bytes width hint must emit Bytes slot, got {other:?}"),
    }
    cleanup_state(&mut state);
}

/// Three singleton `Op::WriteKernelCold` ops dispatched
/// through `apply_ops` produce ONE bridge callback with all 3
/// writes — confirms the executor's pre-pass folds adjacent
/// singletons into a single freeze rendezvous end-to-end,
/// not just at the helper level. Pins the freeze-rendezvous-
/// batching contract the [`Op::WriteKernelCold`] doc names
/// as a "hard correctness requirement" (no inter-CPU skew).
#[test]
fn apply_ops_merges_three_adjacent_cold_write_singletons_into_one_dispatch() {
    use std::sync::Arc;
    let captured = Arc::new(std::sync::Mutex::new(Vec::<
        crate::vmm::wire::KernelOpRequestPayload,
    >::new()));
    let captured_clone = captured.clone();
    let kernel_op_cb: crate::scenario::snapshot::KernelOpCallback = Arc::new(move |req| {
        captured_clone.lock().unwrap().push(req.clone());
        crate::vmm::wire::KernelOpReplyPayload {
            request_id: req.request_id,
            success: true,
            reason: String::new(),
            read_values: vec![],
        }
    });
    let bridge = crate::scenario::snapshot::SnapshotBridge::new(Arc::new(|_| None))
        .with_kernel_op(kernel_op_cb);
    let _bg = bridge.set_thread_local();
    mock_setup_state!(mock, topo, ctx, state);
    let ops = vec![
        Op::write_kernel_cold(
            KernelTarget::per_cpu_field("runqueues", "clock", 0),
            KernelValue::u64(100),
        ),
        Op::write_kernel_cold(
            KernelTarget::per_cpu_field("runqueues", "clock", 1),
            KernelValue::u64(200),
        ),
        Op::write_kernel_cold(
            KernelTarget::per_cpu_field("runqueues", "clock", 2),
            KernelValue::u64(300),
        ),
    ];
    apply_ops_test(&ctx, &mut state, &ops).expect("merged cold-write batch must dispatch");
    let payloads = captured.lock().unwrap();
    assert_eq!(
        payloads.len(),
        1,
        "3 adjacent singletons must collapse into ONE bridge dispatch, got {} dispatches",
        payloads.len()
    );
    assert_eq!(payloads[0].mode, crate::vmm::wire::KernelOpMode::Cold);
    assert_eq!(
        payloads[0].direction,
        crate::vmm::wire::KernelOpDirection::Write
    );
    assert_eq!(
        payloads[0].entries.len(),
        3,
        "merged batch must carry all 3 writes in input order"
    );
    cleanup_state(&mut state);
}

/// `Op::CaptureSnapshot` between two cold-write singletons
/// acts as a hard barrier — the snapshot must observe state
/// AFTER the first write but BEFORE the second. Pins the
/// "any non-cold-write op is a barrier" generalization in the
/// pre-pass: a regression that narrowed the predicate to
/// "only kernel ops barrier" would fold across CaptureSnapshot
/// and silently break the captured-state-between-writes
/// contract.
#[test]
fn merge_adjacent_cold_writes_capture_snapshot_is_barrier() {
    use super::merge_adjacent_cold_writes;
    let ops = vec![
        Op::write_kernel_cold(KernelTarget::symbol("a"), KernelValue::u64(1)),
        Op::CaptureSnapshot { name: "mid".into() },
        Op::write_kernel_cold(KernelTarget::symbol("b"), KernelValue::u64(2)),
    ];
    let merged = merge_adjacent_cold_writes(&ops);
    assert_eq!(merged.len(), 3, "CaptureSnapshot must split cold writes");
    assert!(matches!(merged[0], Op::WriteKernelCold { ref writes } if writes.len() == 1));
    assert!(matches!(merged[1], Op::CaptureSnapshot { .. }));
    assert!(matches!(merged[2], Op::WriteKernelCold { ref writes } if writes.len() == 1));
}

/// A generic non-kernel op (e.g. `Op::AddCgroup`) between two
/// cold-write singletons acts as a hard barrier. Pins the
/// "any non-cold-write op is a barrier" predicate so a
/// regression that narrowed to "only kernel ops barrier" or
/// "only Op::Write/Read variants barrier" silently breaks
/// sequencing with cgroup setup / payload spawn / etc.
#[test]
fn merge_adjacent_cold_writes_non_kernel_op_is_barrier() {
    use super::merge_adjacent_cold_writes;
    let ops = vec![
        Op::write_kernel_cold(KernelTarget::symbol("a"), KernelValue::u64(1)),
        Op::AddCgroup {
            name: "cg_mid".into(),
        },
        Op::write_kernel_cold(KernelTarget::symbol("b"), KernelValue::u64(2)),
    ];
    let merged = merge_adjacent_cold_writes(&ops);
    assert_eq!(
        merged.len(),
        3,
        "non-kernel cgroup op must split cold writes"
    );
    assert!(matches!(merged[0], Op::WriteKernelCold { ref writes } if writes.len() == 1));
    assert!(matches!(merged[1], Op::AddCgroup { .. }));
    assert!(matches!(merged[2], Op::WriteKernelCold { ref writes } if writes.len() == 1));
}

// ---- cgroup-placement invariants for the apply_ops dispatch path ----
//
// The apply_setup path (CgroupDef.workers(N)) is covered by
// apply_setup_* tests above. These pin the parallel invariants
// for the apply_ops path: Op::Spawn (both SpawnPlacement::Cgroup
// and SpawnPlacement::RunnerCgroup arms) and Op::MoveAllTasks.

/// `Op::Spawn { placement: SpawnPlacement::Cgroup(cgroup), .. }`
/// MUST call `move_tasks(cgroup, pids)` after the workers spawn.
/// A regression where the Cgroup arm stopped issuing move_tasks
/// would silently strand workers in the runner's own cgroup —
/// the same placement the RunnerCgroup arm intentionally uses —
/// making the regression invisible without this guard.
#[test]
fn op_spawn_cgroup_moves_tasks_into_named_cgroup() {
    mock_setup_state!(mock, topo, ctx, state);
    state.cgroups.add_cgroup_no_cpuset("cg_test").unwrap();
    let work = WorkSpec::default().workers(2).work_type(WorkType::SpinWait);
    apply_ops_test(&ctx, &mut state, &[Op::spawn_workers("cg_test", work)])
        .expect("Op::Spawn(Cgroup) should succeed");
    let calls = mock.calls();
    assert!(
        calls
            .iter()
            .any(|c| matches!(c, CgroupCall::MoveTasks(n, 2) if n == "cg_test")),
        "Op::Spawn(Cgroup) must call move_tasks(\"cg_test\", 2 pids), \
             got: {calls:?}"
    );
    cleanup_state(&mut state);
}

/// `Op::Spawn { placement: SpawnPlacement::RunnerCgroup, .. }`
/// MUST emit ZERO cgroup ops. The handler arm deliberately omits
/// any `ctx.cgroups.*` call so workers inherit the spawner's own
/// cgroup — see the `SpawnPlacement::RunnerCgroup` doc for the
/// "runner-cgroup vs workload-cgroup" rationale. The stronger
/// `calls.is_empty()` assertion (vs the narrower
/// `!any(MoveTasks)`) catches a future regression that adds
/// any unrelated cgroup op to the RunnerCgroup arm.
#[test]
fn op_spawn_runner_cgroup_emits_zero_cgroup_calls() {
    mock_setup_state!(mock, topo, ctx, state);
    let work = WorkSpec::default().workers(1).work_type(WorkType::SpinWait);
    apply_ops_test(&ctx, &mut state, &[Op::spawn_host(work)])
        .expect("Op::Spawn(RunnerCgroup) should succeed");
    let calls = mock.calls();
    assert!(
        calls.is_empty(),
        "Op::Spawn(RunnerCgroup) must NOT touch the cgroup ops surface — \
             workers stay in the spawner's own cgroup. Got: {calls:?}"
    );
    cleanup_state(&mut state);
}

/// `Op::Spawn { placement: SpawnPlacement::RunnerCgroup, work:
/// WorkSpec.workers(0) }` MUST bail with the `resolve_num_workers`
/// diagnostic ("num_workers=0 is not allowed"). The runner-path
/// label is `<runner>` (literal angle-brackets — pinned because
/// the operator sees this string in the error and grep-finds the
/// RunnerCgroup spawn call site from it). A regression that
/// bypassed `resolve_num_workers` on the runner path (e.g.
/// "runner-cgroup workers don't need report assertions, skip the
/// check") would silently let 0-worker spawns through, vacuously
/// passing downstream assertions.
#[test]
fn op_spawn_runner_cgroup_workers_zero_bails_with_actionable_diagnostic() {
    mock_setup_state!(mock, topo, ctx, state);
    let work = WorkSpec::default().workers(0).work_type(WorkType::SpinWait);
    let err = apply_ops_test(&ctx, &mut state, &[Op::spawn_host(work)])
        .expect_err("Op::Spawn(RunnerCgroup) workers(0) must bail");
    let msg = format!("{err:#}");
    assert!(
        msg.contains("num_workers=0 is not allowed"),
        "error must cite the resolve_num_workers diagnostic: {msg}"
    );
    assert!(
        msg.contains("<runner>"),
        "error must label the runner-cgroup path as `<runner>` \
             (literal angle-brackets) so the operator can grep the \
             RunnerCgroup spawn call site from the error msg: {msg}"
    );
    assert!(
        mock.calls().is_empty(),
        "bail path must not invoke cgroup ops: {:?}",
        mock.calls()
    );
    cleanup_state(&mut state);
}

/// `Op::Spawn` with `SpawnPlacement::RunnerCgroup` against a
/// `WorkSpec` with `workers_pct = Some(_)` MUST bail. workers_pct
/// semantics depend on a per-cgroup cpuset denominator
/// (`ceil(cpuset_cpus * pct)`) that RunnerCgroup placement
/// doesn't have — spawning happens into the spawner's own cgroup,
/// with no managed cpuset to scale against. The earlier handler
/// silently fell back to `ctx.workers_per_cgroup`, discarding the
/// operator's fraction intent. Bail loud with both recovery paths
/// (.workers(N) or SpawnPlacement::Cgroup(name)) so the test
/// author surfaces the right fix instead of debugging an
/// unexpected worker count.
#[test]
fn op_spawn_runner_cgroup_bails_when_workspec_workers_pct_set() {
    mock_setup_state!(mock, topo, ctx, state);
    let work = WorkSpec::default()
        .workers_pct(0.5)
        .work_type(WorkType::SpinWait);
    let err = apply_ops_test(&ctx, &mut state, &[Op::spawn_host(work)])
        .expect_err("Op::Spawn(RunnerCgroup) with WorkSpec::workers_pct must bail");
    let msg = format!("{err:#}");
    assert!(
        msg.contains("Op::Spawn") && msg.contains("workers_pct"),
        "error must name the op + the rejected field: {msg}"
    );
    // Pin the literal-as-emitted recovery hints. A regression
    // that drifted `.workers(N)` to `.workers(n)` or dropped
    // the leading dot would otherwise sneak through an OR-chain
    // brittle filter; assert the exact production string.
    assert!(
        msg.contains(".workers(N)"),
        "error must name the explicit-count recovery as `.workers(N)`: {msg}"
    );
    assert!(
        msg.contains("SpawnPlacement::Cgroup"),
        "error must name the named-cgroup recovery path: {msg}"
    );
    // Construction-time bail: no cgroup ops should run.
    assert!(
        mock.calls().is_empty(),
        "bail must skip every cgroup op: {:?}",
        mock.calls()
    );
    cleanup_state(&mut state);
}

/// Positive companion to
/// `op_spawn_runner_cgroup_bails_when_workspec_workers_pct_set`:
/// `Op::Spawn(SpawnPlacement::Cgroup)` DOES honor
/// `WorkSpec::workers_pct` against the cgroup's resolved
/// cpuset. The RunnerCgroup arm's bail rejects workers_pct
/// precisely because RunnerCgroup has no cpuset denominator;
/// the Cgroup arm's `resolve_workers_pct` path is the entire
/// reason that bail is acceptable. Pin the happy path so a
/// regression in the Cgroup arm's cpuset → ceil(cpuset * pct)
/// → num_workers routing surfaces here (and doesn't silently
/// regress to the "bail-only test suite" failure mode where
/// every test asserts a bail and no test asserts the working
/// path).
#[test]
fn op_spawn_cgroup_honors_workspec_workers_pct_against_cgroup_cpuset() {
    mock_setup_state!(mock, topo, ctx, state);
    // mock_topo gives 4 CPUs; record the same as the cgroup's
    // cpuset so workers_pct(0.5) resolves to ceil(4 * 0.5) = 2.
    let cpus: std::collections::BTreeSet<usize> = [0, 1, 2, 3].into_iter().collect();
    state.cgroups.add_cgroup_no_cpuset("cg").unwrap();
    // ScenarioState::record_cpuset is the production path, but
    // for a one-off StepState fixture we insert directly into
    // the step-local cpuset map (record_cpuset only routes; the
    // backing storage is the same HashMap).
    state.cpusets.insert("cg".to_string(), cpus);
    let work = WorkSpec::default()
        .workers_pct(0.5)
        .work_type(WorkType::SpinWait);
    apply_ops_test(&ctx, &mut state, &[Op::spawn_workers("cg", work)]).expect(
        "Op::Spawn(SpawnPlacement::Cgroup) + workers_pct(0.5) on 4-CPU cpuset must succeed",
    );
    // Resolve happened: exactly one move_tasks fired against
    // "cg" with 2 worker PIDs (ceil(4 * 0.5) = 2).
    let calls = mock.calls();
    let moves: Vec<&CgroupCall> = calls
        .iter()
        .filter(|c| matches!(c, CgroupCall::MoveTasks(n, _) if n == "cg"))
        .collect();
    assert_eq!(
        moves.len(),
        1,
        "exactly one move_tasks expected for the spawn; got: {moves:?}",
    );
    match &moves[0] {
        CgroupCall::MoveTasks(_, n_pids) => assert_eq!(
            *n_pids, 2,
            "workers_pct(0.5) × 4 CPUs = ceil(2.0) = 2 workers; \
                 a regression in resolve_workers_pct's ceil-then-write surfaces here",
        ),
        _ => unreachable!(),
    }
    cleanup_state(&mut state);
}

/// `Op::Spawn(SpawnPlacement::Cgroup)` against a `WorkSpec` with
/// `pcomm = Some(_)` MUST bail at construction time. The
/// scenario-engine spawn dispatch routes through
/// `WorkloadConfig::for_scenario_engine` which forks one process
/// per worker — `task->group_leader->comm` stays at the binary
/// name, so the operator's requested pcomm silently fails to
/// apply. Scheduler matchers filtering on the leader's comm see
/// zero matches; the workload silently fails to reproduce its
/// intended fixture. Bail loud at the construction boundary
/// instead. Companion to the existing composed[i].pcomm.is_some()
/// bail at `WorkloadHandle::spawn` — same contract, parallel
/// surface.
#[test]
fn op_spawn_cgroup_bails_when_workspec_pcomm_set() {
    mock_setup_state!(mock, topo, ctx, state);
    state.cgroups.add_cgroup_no_cpuset("cg").unwrap();
    let work = WorkSpec::default()
        .workers(1)
        .work_type(WorkType::SpinWait)
        .pcomm("chrome");
    let err = apply_ops_test(&ctx, &mut state, &[Op::spawn_workers("cg", work)])
        .expect_err("Op::Spawn(SpawnPlacement::Cgroup) with WorkSpec::pcomm must bail");
    let msg = format!("{err:#}");
    assert!(
        msg.contains("pcomm"),
        "error must cite the pcomm rejection: {msg}"
    );
    // Pin BOTH recovery hints separately so a regression that
    // drops either one surfaces here. The actionable-diagnostic
    // contract requires both paths be named so the operator can
    // choose between the cgroup-def + apply_setup route and the
    // direct spawn_pcomm_cgroup call.
    assert!(
        msg.contains("CgroupDef::pcomm"),
        "error must name the CgroupDef::pcomm recovery path: {msg}"
    );
    assert!(
        msg.contains("spawn_pcomm_cgroup"),
        "error must name the WorkloadHandle::spawn_pcomm_cgroup recovery path: {msg}"
    );
    // No mock kernel calls: the bail fires at the constructor
    // boundary before any cgroup op runs.
    let move_calls: Vec<_> = mock
        .calls()
        .iter()
        .filter(|c| matches!(c, CgroupCall::MoveTasks(_, _)))
        .cloned()
        .collect();
    assert!(
        move_calls.is_empty(),
        "construction-time bail must skip every move_tasks call: {move_calls:?}"
    );
    cleanup_state(&mut state);
}

/// `Op::Spawn(SpawnPlacement::RunnerCgroup)` against a `WorkSpec`
/// with `pcomm = Some(_)` MUST also bail. The RunnerCgroup arm
/// uses the same `for_scenario_engine` constructor as the Cgroup
/// arm (forks one process per worker in the spawner's own
/// cgroup), so the same pcomm-silent-drop hazard applies. Pinning
/// both placements separately surfaces a regression in either
/// arm individually.
#[test]
fn op_spawn_runner_cgroup_bails_when_workspec_pcomm_set() {
    mock_setup_state!(mock, topo, ctx, state);
    let work = WorkSpec::default()
        .workers(1)
        .work_type(WorkType::SpinWait)
        .pcomm("java");
    let err = apply_ops_test(&ctx, &mut state, &[Op::spawn_host(work)])
        .expect_err("Op::Spawn(SpawnPlacement::RunnerCgroup) with WorkSpec::pcomm must bail");
    let msg = format!("{err:#}");
    assert!(
        msg.contains("pcomm"),
        "error must cite the pcomm rejection: {msg}"
    );
    // Same actionable-diagnostic contract as the Cgroup arm: pin
    // both recovery hints individually rather than ORing them.
    assert!(
        msg.contains("CgroupDef::pcomm"),
        "error must name the CgroupDef::pcomm recovery path: {msg}"
    );
    assert!(
        msg.contains("spawn_pcomm_cgroup"),
        "error must name the WorkloadHandle::spawn_pcomm_cgroup recovery path: {msg}"
    );
    assert!(
        mock.calls().is_empty(),
        "construction-time bail must skip every cgroup op: {:?}",
        mock.calls()
    );
    cleanup_state(&mut state);
}

/// `Op::MoveAllTasks` rolls back its state mutation when the
/// kernel `move_tasks` call fails partway through. The handler
/// collects `pid_batches` first (read-only on state), then runs
/// `move_tasks(to, pids)?` per batch, and only calls
/// `rename_handles(from, to)` AFTER every kernel write
/// succeeded. A failure in `move_tasks` propagates the error
/// via `?` and leaves `state` un-mutated — handles remain
/// keyed under `from`, so subsequent ops looking up by `from`
/// find the same set they would have found before this op
/// ran. The kernel side may still be partially migrated
/// (writes before the failing pid succeeded), but the
/// in-process tracking does not also drift.
///
/// Pin the rollback contract: a regression that called
/// `rename_handles` before the move loop (or unconditionally
/// after it) would silently re-key the handles to `to` even
/// when no kernel migration occurred, so later ops would
/// fail to find them under `from` AND find a phantom entry
/// under `to` that doesn't reflect the kernel state.
#[test]
fn move_all_tasks_preserves_state_when_move_tasks_fails() {
    use crate::workload::{WorkSpec, WorkloadConfig, WorkloadHandle};

    mock_setup_backdrop!(mock, topo, ctx, step_state, backdrop_state);
    step_state.cgroups.add_cgroup_no_cpuset("src").unwrap();
    step_state.cgroups.add_cgroup_no_cpuset("dst").unwrap();
    let w = WorkSpec::default();
    let wl = WorkloadConfig::for_scenario_engine(
        &w,
        1,
        crate::workload::AffinityIntent::Inherit,
        w.work_type.clone(),
    )
    .expect(
        "test fixture: pcomm must stay None for scenario-engine dispatch — \
             if a future fixture variant sets pcomm, route via spawn_pcomm_cgroup \
             instead of for_scenario_engine",
    );
    let h = WorkloadHandle::spawn(&wl).expect("spawn worker");
    step_state.handles.push(("src".to_string(), h));

    // Schedule the next-encountered move_tasks call to fail.
    // Semantic-index scheduling (`fail_nth_call_matching`) is
    // adversary-resistant to handler refactors that add an
    // unrelated kernel-op between setup and the targeted
    // call — the count only advances on predicate matches, so
    // "the 0th MoveTasks after now" stays the right target
    // regardless of how many ClearSubtreeControl / SetCpuset
    // / etc. fire first. setup_call_count below feeds only
    // the post-loop call-sequence asserts that pin the
    // handler's emit order.
    let setup_call_count = mock.calls().len();
    mock.fail_nth_call_matching(
        0,
        |c| matches!(c, CgroupCall::MoveTasks(_, _)),
        "injected kernel ENOSPC mid-move",
    );

    {
        let mut scenario = ScenarioState::new(&mut step_state, &mut backdrop_state);
        let err = apply_ops(
            &ctx,
            &mut scenario,
            &[Op::move_all_tasks("src", "dst")],
            false,
        )
        .expect_err("move_tasks failure must propagate");
        let msg = format!("{err:#}");
        assert!(
            msg.contains("injected kernel ENOSPC mid-move"),
            "error must surface the injected failure verbatim: {msg}"
        );
    }

    // Handle MUST remain keyed under "src" (NOT re-keyed to
    // "dst"); rename_handles never ran. A regression that
    // reordered rename_handles before the move loop would
    // surface here as "dst" instead of "src".
    assert_eq!(
        step_state.handles.len(),
        1,
        "handle must survive the failed move; got: {n}",
        n = step_state.handles.len(),
    );
    assert_eq!(
        step_state.handles[0].0, "src",
        "partial-failure rollback contract: handle stays keyed under \
             `from` when move_tasks errored before rename_handles ran"
    );
    // Backdrop slot must NOT have picked up the handle either.
    assert_eq!(
        backdrop_state.handles.len(),
        0,
        "step-to-backdrop ownership transfer must not have run on \
             the failure path; got: {n}",
        n = backdrop_state.handles.len(),
    );
    // FE1 guard: assert the exact 2-call sequence the handler
    // emitted matches the `setup_call_count + 1` index we
    // targeted. A future handler refactor that adds a kernel
    // call between clear_subtree_control and move_tasks would
    // shift the move_tasks to setup_call_count + 2, and the
    // fail-injection would land on the wrong call. Surface
    // that drift here instead of letting it produce a
    // mysteriously-passing test that no longer pins the
    // rollback contract.
    let recorded = mock.calls();
    let post_setup_calls: Vec<&CgroupCall> = recorded.iter().skip(setup_call_count).collect();
    assert!(
        matches!(post_setup_calls.first(), Some(CgroupCall::ClearSubtreeControl(n)) if n == "dst"),
        "expected first post-setup call = clear_subtree_control(\"dst\"); \
             got: {post_setup_calls:?}"
    );
    assert!(
        matches!(post_setup_calls.get(1), Some(CgroupCall::MoveTasks(n, _)) if n == "dst"),
        "expected second post-setup call = move_tasks(\"dst\", _) (the one \
             we targeted via fail_nth_call_matching for the 0th MoveTasks); \
             got: {post_setup_calls:?}"
    );
    step_state.handles.clear();
}

/// Companion to `move_all_tasks_preserves_state_when_move_tasks_fails`:
/// the step→backdrop ownership-transfer site at the handler
/// L2727 rename_handles call must ALSO not run when move_tasks
/// fails mid-loop. Pins the cross-state-slot atomicity contract
/// — a regression that ran the transfer before the move loop
/// would push the handle into backdrop_state.handles even
/// though no kernel migration occurred, so a subsequent
/// scenario teardown would find a phantom backdrop handle.
#[test]
fn move_all_tasks_step_to_backdrop_failure_preserves_step_ownership() {
    use crate::workload::{WorkSpec, WorkloadConfig, WorkloadHandle};

    mock_setup_backdrop!(mock, topo, ctx, step_state, backdrop_state);
    backdrop_state
        .cgroups
        .add_cgroup_no_cpuset("bd_dst")
        .unwrap();
    step_state.cgroups.add_cgroup_no_cpuset("src").unwrap();
    let w = WorkSpec::default();
    let wl = WorkloadConfig::for_scenario_engine(
        &w,
        1,
        crate::workload::AffinityIntent::Inherit,
        w.work_type.clone(),
    )
    .expect(
        "test fixture: pcomm must stay None for scenario-engine dispatch — \
             if a future fixture variant sets pcomm, route via spawn_pcomm_cgroup \
             instead of for_scenario_engine",
    );
    let h = WorkloadHandle::spawn(&wl).expect("spawn worker");
    step_state.handles.push(("src".to_string(), h));

    mock.fail_nth_call_matching(
        0,
        |c| matches!(c, CgroupCall::MoveTasks(_, _)),
        "injected kernel ENOSPC mid-move (step→backdrop)",
    );

    {
        let mut scenario = ScenarioState::new(&mut step_state, &mut backdrop_state);
        let err = apply_ops(
            &ctx,
            &mut scenario,
            &[Op::move_all_tasks("src", "bd_dst")],
            false,
        )
        .expect_err("step→backdrop move_tasks failure must propagate");
        let msg = format!("{err:#}");
        assert!(
            msg.contains("ENOSPC mid-move (step→backdrop)"),
            "error must surface the injected failure verbatim: {msg}"
        );
    }

    assert_eq!(
        step_state.handles.len(),
        1,
        "handle must stay in step_state on failure; got: {n}",
        n = step_state.handles.len(),
    );
    assert_eq!(
        step_state.handles[0].0, "src",
        "handle must stay keyed under 'src' (no rename ran)"
    );
    assert!(
        backdrop_state.handles.is_empty(),
        "no ownership transfer on failure path; got: {:?}",
        backdrop_state
            .handles
            .iter()
            .map(|(n, _)| n.as_str())
            .collect::<Vec<_>>()
    );
    step_state.handles.clear();
}

/// Companion to `move_all_tasks_preserves_state_when_move_tasks_fails`:
/// when MULTIPLE handles are keyed under `from`, a mid-loop
/// move_tasks failure must leave ALL handles keyed under
/// `from` (per the handler comment at L2697-2705: "the kernel
/// side may still be partially migrated, but the in-process
/// tracking does not also drift"). The first N-1 batches
/// migrated kernel-side; the Nth failed; rename_handles never
/// ran for any of them. Subsequent ops looking up `from` find
/// the same set as pre-op. Pins the all-or-nothing in-process
/// re-key contract.
#[test]
fn move_all_tasks_multi_handle_partial_failure_keeps_all_under_src() {
    use crate::workload::{WorkSpec, WorkloadConfig, WorkloadHandle};

    mock_setup_backdrop!(mock, topo, ctx, step_state, backdrop_state);
    step_state.cgroups.add_cgroup_no_cpuset("src").unwrap();
    step_state.cgroups.add_cgroup_no_cpuset("dst").unwrap();
    for _ in 0..2 {
        let w = WorkSpec::default();
        let wl = WorkloadConfig::for_scenario_engine(
            &w,
            1,
            crate::workload::AffinityIntent::Inherit,
            w.work_type.clone(),
        )
        .expect(
            "test fixture: pcomm must stay None for scenario-engine dispatch — \
                 if a future fixture variant sets pcomm, route via spawn_pcomm_cgroup \
                 instead of for_scenario_engine",
        );
        let h = WorkloadHandle::spawn(&wl).expect("spawn worker");
        step_state.handles.push(("src".to_string(), h));
    }

    // Semantic-index scheduling: fail the 2nd (index 1)
    // MoveTasks call after now. Handler emits one move_tasks
    // per matching handle; with 2 handles under "src" the
    // 1st-indexed MoveTasks is the second handle's migration.
    // First handle's kernel migration succeeded, second
    // failed — in-process state must NOT drift for either.
    mock.fail_nth_call_matching(
        1,
        |c| matches!(c, CgroupCall::MoveTasks(_, _)),
        "injected kernel ENOSPC on second handle's move_tasks",
    );

    {
        let mut scenario = ScenarioState::new(&mut step_state, &mut backdrop_state);
        let err = apply_ops(
            &ctx,
            &mut scenario,
            &[Op::move_all_tasks("src", "dst")],
            false,
        )
        .expect_err("multi-handle partial-failure must propagate");
        let msg = format!("{err:#}");
        assert!(
            msg.contains("ENOSPC on second handle"),
            "error must surface the injected failure verbatim: {msg}"
        );
    }

    assert_eq!(
        step_state.handles.len(),
        2,
        "no handle dropped on partial failure; got: {n}",
        n = step_state.handles.len(),
    );
    assert!(
        step_state.handles.iter().all(|(n, _)| n == "src"),
        "all handles MUST stay keyed under 'src' (no partial re-key); \
             got: {:?}",
        step_state
            .handles
            .iter()
            .map(|(n, _)| n.as_str())
            .collect::<Vec<_>>(),
    );
    step_state.handles.clear();
}

/// `Op::MoveAllTasks { from, to }` MUST call
/// `clear_subtree_control(to)` BEFORE `move_tasks(to, ..)`.
/// The kernel's cgroup-v2 no-internal-process constraint
/// (`cgroup_migrate_vet_dst` in `kernel/cgroup/cgroup.c`)
/// returns EBUSY on `cgroup.procs` writes to a cgroup whose
/// `cgroup.subtree_control` is non-empty. The handler clears
/// subtree_control first to avoid that path.
#[test]
fn op_move_all_tasks_clears_subtree_control_then_moves_to_dst() {
    use crate::workload::{WorkloadConfig, WorkloadHandle};

    mock_setup_state!(mock, topo, ctx, state);
    state.cgroups.add_cgroup_no_cpuset("src").unwrap();
    state.cgroups.add_cgroup_no_cpuset("dst").unwrap();
    let wl = WorkloadConfig {
        num_workers: 1,
        affinity: AffinityIntent::Inherit,
        work_type: WorkType::SpinWait,
        ..Default::default()
    };
    let h = WorkloadHandle::spawn(&wl).expect("spawn worker");
    state.handles.push(("src".to_string(), h));
    apply_ops_test(&ctx, &mut state, &[Op::move_all_tasks("src", "dst")])
        .expect("MoveAllTasks should succeed");
    let calls = mock.calls();
    let clear_idx = calls
        .iter()
        .position(|c| matches!(c, CgroupCall::ClearSubtreeControl(n) if n == "dst"))
        .expect("MoveAllTasks must call clear_subtree_control(\"dst\")");
    let move_idx = calls
        .iter()
        .position(|c| matches!(c, CgroupCall::MoveTasks(n, _) if n == "dst"))
        .expect("MoveAllTasks must call move_tasks(\"dst\", _)");
    assert!(
        clear_idx < move_idx,
        "clear_subtree_control must precede move_tasks for the same \
             cgroup or the kernel rejects the cgroup.procs write with \
             EBUSY (no-internal-process invariant). Got: {calls:?}"
    );
    cleanup_state(&mut state);
}

/// Composing `Op::AddCgroupDef` (with a cpuset) and
/// `Op::Spawn(SpawnPlacement::Cgroup)` in the same apply_ops
/// batch MUST result in `set_cpuset` running before `move_tasks`
/// — same kernel-correctness invariant pinned for apply_setup at
/// `apply_setup_sets_cpuset_before_move_tasks` above. Moving
/// tasks before the cpuset would let them briefly run on
/// CPUs outside the intended set.
#[test]
fn op_spawn_cgroup_after_addcgroupdef_sets_cpuset_before_move_tasks() {
    mock_setup_state!(mock, topo, ctx, state);
    let cpus: BTreeSet<usize> = [0, 1].into_iter().collect();
    apply_ops_test(
        &ctx,
        &mut state,
        &[
            Op::AddCgroupDef {
                def: CgroupDef::named("cg_ordered").cpuset(CpusetSpec::Exact(cpus.clone())),
            },
            Op::spawn_workers(
                "cg_ordered",
                WorkSpec::default().workers(2).work_type(WorkType::SpinWait),
            ),
        ],
    )
    .expect("AddCgroupDef + Op::Spawn(SpawnPlacement::Cgroup) should succeed");
    let calls = mock.calls();
    let set_idx = calls
        .iter()
        .position(|c| matches!(c, CgroupCall::SetCpuset(n, _) if n == "cg_ordered"))
        .expect("set_cpuset for cg_ordered");
    let move_idx = calls
        .iter()
        .position(|c| matches!(c, CgroupCall::MoveTasks(n, _) if n == "cg_ordered"))
        .expect("move_tasks for cg_ordered");
    assert!(
        set_idx < move_idx,
        "set_cpuset must precede move_tasks for the same cgroup \
             across the AddCgroupDef → Op::Spawn(SpawnPlacement::Cgroup) boundary: {calls:?}"
    );
    cleanup_state(&mut state);
}

// -- Op::CaptureCgroupProcs dispatch tests --

/// Happy path: the dispatch arm calls `CgroupOps::read_procs(cgroup)`
/// with the supplied name and records the resulting pid list on
/// the active bridge under the supplied tag. Mock pre-loads 3 pids
/// for `"cg_x"`; the test asserts on both the trait-method
/// invocation (via mock's call log) and the bridge-side drain
/// (the snapshot's tag/cgroup/pids triple).
#[test]
fn op_capture_cgroup_procs_records_snapshot_on_active_bridge() {
    use std::sync::Arc;
    mock_setup_state!(mock, topo, ctx, state);
    mock.set_procs("cg_x", vec![100, 200, 300]);
    let bridge = crate::scenario::snapshot::SnapshotBridge::new(Arc::new(|_| None));
    let bridge_for_drain = bridge.clone();
    let _guard = bridge.set_thread_local();
    apply_ops_test(
        &ctx,
        &mut state,
        &[Op::capture_cgroup_procs("snap_tag", "cg_x")],
    )
    .expect("Op::CaptureCgroupProcs should succeed");
    // Mock call-log pins the trait invocation shape: exactly one
    // read_procs call against "cg_x".
    let procs_calls: Vec<String> = mock
        .calls()
        .iter()
        .filter_map(|c| match c {
            CgroupCall::ReadProcs(name) => Some(name.clone()),
            _ => None,
        })
        .collect();
    assert_eq!(
        procs_calls,
        vec!["cg_x".to_string()],
        "exactly one read_procs(\"cg_x\") expected; got: {procs_calls:?}",
    );
    // Bridge drain pins the delivery contract: the snapshot
    // carries the supplied tag, the supplied cgroup name, and
    // the pids the mock returned — in mock-provided order.
    let snaps = bridge_for_drain.drain_cgroup_procs();
    assert_eq!(snaps.len(), 1);
    assert_eq!(snaps[0].tag, "snap_tag");
    assert_eq!(snaps[0].cgroup, "cg_x");
    assert_eq!(snaps[0].pids, vec![100, 200, 300]);
    cleanup_state(&mut state);
}

/// `read_procs` failures must propagate as an `Err` from
/// apply_ops with the actionable context naming the op + tag +
/// cgroup. No snapshot is recorded on the bridge when the read
/// fails (the bridge `record_cgroup_procs` is gated behind the
/// `?` early-exit).
#[test]
fn op_capture_cgroup_procs_propagates_read_procs_error() {
    use std::sync::Arc;
    mock_setup_state!(mock, topo, ctx, state);
    mock.fail_read_procs("cg_x", "injected ENOENT from cgroup.procs");
    let bridge = crate::scenario::snapshot::SnapshotBridge::new(Arc::new(|_| None));
    let bridge_for_drain = bridge.clone();
    let _guard = bridge.set_thread_local();
    let err = apply_ops_test(
        &ctx,
        &mut state,
        &[Op::capture_cgroup_procs("snap_tag", "cg_x")],
    )
    .expect_err("read_procs Err must surface as apply_ops Err");
    let msg = format!("{err:#}");
    assert!(
        msg.contains("Op::CaptureCgroupProcs"),
        "diagnostic must name the op; got: {msg}",
    );
    assert!(
        msg.contains("snap_tag") && msg.contains("cg_x"),
        "diagnostic must echo tag + cgroup; got: {msg}",
    );
    assert!(
        msg.contains("injected ENOENT"),
        "diagnostic must chain the inner read_procs error; got: {msg}",
    );
    assert!(
        bridge_for_drain.drain_cgroup_procs().is_empty(),
        "no snapshot must be recorded when the read fails",
    );
    cleanup_state(&mut state);
}

/// Empty `tag` must bail at the dispatch arm BEFORE invoking
/// `read_procs`. Mirrors the `Op::Spawn(SpawnPlacement::Cgroup(""))`
/// empty-string bail pattern: the tag is the snapshot key
/// consumers use to find the capture, and an empty key would
/// silently alias multiple captures into the same drain entry
/// (per the bridge's insertion-order append semantic).
#[test]
fn op_capture_cgroup_procs_empty_tag_bails_before_read() {
    use std::sync::Arc;
    mock_setup_state!(mock, topo, ctx, state);
    let bridge = crate::scenario::snapshot::SnapshotBridge::new(Arc::new(|_| None));
    let _guard = bridge.set_thread_local();
    let err = apply_ops_test(&ctx, &mut state, &[Op::capture_cgroup_procs("", "cg_x")])
        .expect_err("empty tag must bail");
    let msg = format!("{err:#}");
    assert!(
        msg.contains("tag is empty"),
        "diagnostic must cite the empty-tag bail; got: {msg}",
    );
    // No mock invocation: the bail fires before read_procs is
    // called.
    let procs_calls: Vec<String> = mock
        .calls()
        .iter()
        .filter_map(|c| match c {
            CgroupCall::ReadProcs(name) => Some(name.clone()),
            _ => None,
        })
        .collect();
    assert!(
        procs_calls.is_empty(),
        "empty-tag bail must skip read_procs; got: {procs_calls:?}",
    );
    cleanup_state(&mut state);
}

/// Empty `cgroup` must bail at the dispatch arm BEFORE invoking
/// `read_procs`. Mirrors the `Op::Spawn(SpawnPlacement::Cgroup(""))`
/// pattern pinned in tests/op_spawn_cgroup_empty_string_bail_e2e.rs.
#[test]
fn op_capture_cgroup_procs_empty_cgroup_bails_before_read() {
    use std::sync::Arc;
    mock_setup_state!(mock, topo, ctx, state);
    let bridge = crate::scenario::snapshot::SnapshotBridge::new(Arc::new(|_| None));
    let _guard = bridge.set_thread_local();
    let err = apply_ops_test(
        &ctx,
        &mut state,
        &[Op::capture_cgroup_procs("snap_tag", "")],
    )
    .expect_err("empty cgroup name must bail");
    let msg = format!("{err:#}");
    assert!(
        msg.contains("cgroup name is empty"),
        "diagnostic must cite the empty-cgroup bail; got: {msg}",
    );
    let procs_calls: Vec<String> = mock
        .calls()
        .iter()
        .filter_map(|c| match c {
            CgroupCall::ReadProcs(name) => Some(name.clone()),
            _ => None,
        })
        .collect();
    assert!(
        procs_calls.is_empty(),
        "empty-cgroup bail must skip read_procs; got: {procs_calls:?}",
    );
    cleanup_state(&mut state);
}

/// Op fires without an installed bridge: the read succeeds but
/// there is no recipient for the pids. Per the no-silent-drops
/// policy, the dispatch arm bails loud with a diagnostic naming the
/// missing bridge — silently dropping the snapshot would make
/// a missing-bridge misconfiguration look identical to an empty
/// cgroup on drain.
#[test]
fn op_capture_cgroup_procs_bails_when_no_bridge_installed() {
    mock_setup_state!(mock, topo, ctx, state);
    mock.set_procs("cg_x", vec![42]);
    // NOTE: no bridge installed — ACTIVE_BRIDGE is None.
    let err = apply_ops_test(
        &ctx,
        &mut state,
        &[Op::capture_cgroup_procs("snap_tag", "cg_x")],
    )
    .expect_err("missing bridge must surface as Err");
    let msg = format!("{err:#}");
    assert!(
        msg.contains("no SnapshotBridge installed"),
        "diagnostic must cite the missing bridge; got: {msg}",
    );
    assert!(
        msg.contains("set_thread_local"),
        "diagnostic must point to the install API; got: {msg}",
    );
    // Pin the bail-before-read-procs invariant: the dispatch arm
    // MUST hoist the bridge-presence check above the read_procs
    // syscall so a missing-bridge misconfiguration surfaces
    // directly instead of being shadowed by any unrelated read
    // failure (matches Op::Spawn(SpawnPlacement::Cgroup("")) bail-
    // before-spawn pattern). If a future refactor moves the read
    // before the bridge check, this assertion fails loud.
    let read_calls: Vec<String> = mock
        .calls()
        .iter()
        .filter_map(|c| match c {
            CgroupCall::ReadProcs(name) => Some(name.clone()),
            _ => None,
        })
        .collect();
    assert!(
        read_calls.is_empty(),
        "no-bridge bail must hoist above read_procs; got read_procs calls: {read_calls:?}",
    );
    cleanup_state(&mut state);
}

/// Multiple captures of the SAME cgroup under DIFFERENT tags
/// must append separately to the drain log, preserving
/// insertion order. Lets a scenario capture pre/post snapshots
/// of the same cgroup and disambiguate them on drain.
#[test]
fn op_capture_cgroup_procs_multiple_tags_same_cgroup_preserve_order() {
    use std::sync::Arc;
    mock_setup_state!(mock, topo, ctx, state);
    mock.set_procs("cg_x", vec![100, 200]);
    let bridge = crate::scenario::snapshot::SnapshotBridge::new(Arc::new(|_| None));
    let bridge_for_drain = bridge.clone();
    let _guard = bridge.set_thread_local();
    apply_ops_test(
        &ctx,
        &mut state,
        &[
            Op::capture_cgroup_procs("before", "cg_x"),
            Op::capture_cgroup_procs("after", "cg_x"),
        ],
    )
    .expect("two captures should succeed");
    let snaps = bridge_for_drain.drain_cgroup_procs();
    assert_eq!(snaps.len(), 2);
    assert_eq!(snaps[0].tag, "before");
    assert_eq!(snaps[1].tag, "after");
    assert_eq!(snaps[0].cgroup, "cg_x");
    assert_eq!(snaps[1].cgroup, "cg_x");
    assert_eq!(snaps[0].pids, vec![100, 200]);
    assert_eq!(snaps[1].pids, vec![100, 200]);
    cleanup_state(&mut state);
}

/// Multiple captures with the SAME `(tag, cgroup)` pair MUST
/// append rather than overwrite — pins the
/// `record_cgroup_procs` `Vec.push` semantic against a future
/// refactor that switches to a HashMap-keyed-by-(tag, cgroup)
/// dedup store. Without this test, a refactor that silently
/// dropped the second capture would pass the existing
/// preserve_order test (which uses two different tags) without
/// surfacing the regression.
#[test]
fn op_capture_cgroup_procs_same_tag_same_cgroup_appends_not_overwrites() {
    use std::sync::Arc;
    mock_setup_state!(mock, topo, ctx, state);
    mock.set_procs("cg_x", vec![42]);
    let bridge = crate::scenario::snapshot::SnapshotBridge::new(Arc::new(|_| None));
    let bridge_for_drain = bridge.clone();
    let _guard = bridge.set_thread_local();
    apply_ops_test(
        &ctx,
        &mut state,
        &[
            Op::capture_cgroup_procs("snap", "cg_x"),
            Op::capture_cgroup_procs("snap", "cg_x"),
        ],
    )
    .expect("duplicate (tag, cgroup) captures should succeed");
    let snaps = bridge_for_drain.drain_cgroup_procs();
    assert_eq!(
        snaps.len(),
        2,
        "same (tag, cgroup) MUST append both captures; HashMap-style \
             overwrite would yield len=1 and silently drop a capture",
    );
    assert_eq!(snaps[0].tag, "snap");
    assert_eq!(snaps[1].tag, "snap");
    cleanup_state(&mut state);
}
