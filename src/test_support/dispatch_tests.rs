//! Unit tests for [`super`] (the `test_support::dispatch` module).
//! Co-located via the `tests` submodule pattern (sibling file).

#![cfg(test)]

use super::*;
use crate::sync::MutexExt;
use crate::test_support::Scheduler;

// ---------------------------------------------------------------
// is_test_sentinel — convention-based sentinel-name predicate
// ---------------------------------------------------------------

/// Accepted shapes: `__unit_test_*__` (the established
/// sentinel convention — double-underscore prefix with
/// `unit_test_` tag, arbitrary inner suffix, double-underscore
/// suffix).
#[test]
fn is_test_sentinel_accepts_convention_shaped_names() {
    assert!(is_test_sentinel("__unit_test_dummy__"));
    assert!(is_test_sentinel("__unit_test_panics__"));
    // Any inner body after the prefix is accepted, as long as
    // the `__` suffix is also present.
    assert!(is_test_sentinel("__unit_test_foo_bar_baz__"));
}

/// Rejected shapes: real user names, unrelated
/// double-underscore names, and partial matches.
#[test]
fn is_test_sentinel_rejects_non_convention_names() {
    // Real user-authored name.
    assert!(!is_test_sentinel("my_test"));
    // Double-underscore wrapping but not the `__unit_test_` tag.
    assert!(!is_test_sentinel("__foo__"));
    // Empty string.
    assert!(!is_test_sentinel(""));
    // Has the prefix but no `__` suffix (ends with just `_`).
    assert!(!is_test_sentinel("__unit_test_"));
    // Has the prefix, has `__` suffix, but the prefix itself
    // is truncated — missing the trailing `_` of `__unit_test_`.
    assert!(!is_test_sentinel("__unit__"));
}

// ---------------------------------------------------------------
// run_named_test / run_gauntlet_test — nextest dispatch routing
// ---------------------------------------------------------------
//
// These tests cover the `test_name → function` routing without
// booting a VM. The happy paths require KVM and a kernel image,
// so the assertions here target the failure branches that return
// exit code 1 before any VM spawn:
//   - `ktstr/` prefix with unknown bare name
//   - `gauntlet/` prefix with malformed parts / unknown preset
//   - bare names fall through to `ktstr/` lookup
//
// The routing invariant: `gauntlet/` always delegates to
// `run_gauntlet_test`, every other prefix (including none)
// delegates to the base-test path inside `run_named_test`.

#[test]
fn run_named_test_gauntlet_prefix_routes_to_run_gauntlet_test() {
    // Gauntlet names require two slash-separated parts after the
    // prefix (`{name}/{preset}`); a single-segment name is rejected
    // by `run_gauntlet_test` with "invalid gauntlet test name: ...".
    // The base-test path would instead print "unknown test: ..." —
    // both return exit 1, so capturing stderr is what proves the
    // `gauntlet/` prefix routed to `run_gauntlet_test` rather than
    // falling through to `find_test`.
    use crate::test_support::test_helpers::{EnvVarGuard, capture_stderr, lock_env};
    // run_named_test reads KTSTR_KERNEL_LIST lock-free; hold the env
    // lock and clear the list so a concurrent multi-kernel setter
    // can't make strip_kernel_suffix reject the name (exit 1 via the
    // wrong branch). Mirrors run_named_test_perf_mode_test_skips_*.
    let _env_lock = lock_env();
    let _kernel_list = EnvVarGuard::remove(crate::KTSTR_KERNEL_LIST_ENV);

    let (exit, captured) = capture_stderr(|| run_named_test("gauntlet/__unit_test_dummy__"));
    assert_eq!(exit, 1, "malformed gauntlet names must exit 1");
    let stderr = String::from_utf8(captured).expect("stderr is utf-8");
    assert!(
        stderr.contains("invalid gauntlet test name: gauntlet/__unit_test_dummy__"),
        "the gauntlet/ prefix must route to run_gauntlet_test (its \
         format error), not the base-path 'unknown test'; got: {stderr}",
    );
    assert!(
        !stderr.contains("unknown test:"),
        "must NOT fall through to the base-test 'unknown test' path; \
         got: {stderr}",
    );
}

#[test]
fn run_named_test_bare_unknown_exits_nonzero() {
    // `run_named_test` strips `ktstr/` when present; a bare unknown
    // name falls through to `find_test` which returns None,
    // producing exit code 1. Hold the env lock and clear
    // KTSTR_KERNEL_LIST so a concurrent multi-kernel setter can't
    // make strip_kernel_suffix reject the name first (also exit 1,
    // but via the wrong branch); capture stderr to prove the
    // find_test-None path is what produced the exit.
    use crate::test_support::test_helpers::{EnvVarGuard, capture_stderr, lock_env};
    let _env_lock = lock_env();
    let _kernel_list = EnvVarGuard::remove(crate::KTSTR_KERNEL_LIST_ENV);
    let (exit, captured) = capture_stderr(|| run_named_test("__definitely_not_a_real_test__"));
    assert_eq!(exit, 1);
    let stderr = String::from_utf8(captured).expect("stderr is utf-8");
    assert!(
        stderr.contains("unknown test:") && stderr.contains("__definitely_not_a_real_test__"),
        "exit 1 must come from the find_test None path; got: {stderr}",
    );
}

#[test]
fn run_named_test_ktstr_prefix_unknown_exits_nonzero() {
    // `ktstr/` prefix is stripped; the bare name (also unknown)
    // returns 1 via the find_test None path. Same env isolation and
    // stderr check as the bare-name case so the asserted branch is
    // provably the cause.
    use crate::test_support::test_helpers::{EnvVarGuard, capture_stderr, lock_env};
    let _env_lock = lock_env();
    let _kernel_list = EnvVarGuard::remove(crate::KTSTR_KERNEL_LIST_ENV);
    let (exit, captured) =
        capture_stderr(|| run_named_test("ktstr/__definitely_not_a_real_test__"));
    assert_eq!(exit, 1);
    let stderr = String::from_utf8(captured).expect("stderr is utf-8");
    assert!(
        stderr.contains("unknown test:") && stderr.contains("__definitely_not_a_real_test__"),
        "exit 1 must come from the find_test None path (after ktstr/ \
         strip); got: {stderr}",
    );
}

#[test]
fn run_gauntlet_test_rejects_name_with_fewer_than_two_parts() {
    // `rest` must split into exactly 2 parts (`{name}/{preset}`).
    // A single-segment name has no preset and is a format error.
    let exit = run_gauntlet_test("some_test_no_preset");
    assert_eq!(exit, 1);
}

#[test]
fn run_gauntlet_test_rejects_empty_rest() {
    // Empty rest splits into one empty string — also a format
    // error.
    let exit = run_gauntlet_test("");
    assert_eq!(exit, 1);
}

#[test]
fn run_gauntlet_test_rejects_unknown_test_name() {
    // Well-formed two-part name whose test is not registered
    // in KTSTR_TESTS. Returns 1 via the find_test None branch,
    // never reaching preset lookup or VM spawn.
    let exit = run_gauntlet_test("__not_a_test__/tiny-1llc");
    assert_eq!(exit, 1);
}

// ---------------------------------------------------------------
// is_topology_insufficient — typed-error classification
// ---------------------------------------------------------------

/// A [`TopologyInsufficient`] is recognised, including when wrapped
/// in `.context(...)` layers (the eval build/run wrappers) — the
/// predicate walks the full error chain.
#[test]
fn is_topology_insufficient_recognizes_typed_error_through_context() {
    use crate::vmm::host_topology::TopologyInsufficient;
    let direct: anyhow::Error = anyhow::Error::new(TopologyInsufficient {
        reason: "vCPU count 600 exceeds KVM_CAP_MAX_VCPUS 512; cannot boot a VM this wide".into(),
    });
    assert!(is_topology_insufficient(&direct), "direct must match");
    // Mirror the real production wrapping: a kvm-cap TopologyInsufficient
    // (the VM cannot boot) flows through the eval layer's
    // "build ktstr_test VM" / "run ktstr_test VM" .context wrappers. (A
    // perf-mode-too-small host is re-mapped to PerfModeUnavailable
    // before any wrapper, so it never reaches this predicate.)
    let wrapped = direct
        .context("build ktstr_test VM")
        .context("run ktstr_test VM");
    assert!(
        is_topology_insufficient(&wrapped),
        "context-wrapped TopologyInsufficient must still match through \
         the full production wrapper depth",
    );
}

/// Anti-fragility: an unrelated error whose message HAPPENS to
/// contain "need" + "CPU" must NOT be classified as a topology
/// skip. The replaced string-match (`"need"` + `"CPU"`/`"LLC"`)
/// would have wrongly skipped a real failure here. A
/// [`ResourceContention`] (a distinct typed class) is likewise not
/// topology-insufficient.
#[test]
fn is_topology_insufficient_rejects_unrelated_and_other_typed_errors() {
    let look_alike =
        anyhow::anyhow!("scheduler regression: workload did not get the CPU time it needs");
    assert!(
        !is_topology_insufficient(&look_alike),
        "a plain error mentioning need+CPU must NOT be a topology skip — \
         that is the string-match fragility this typed predicate fixes",
    );
    let contention: anyhow::Error =
        anyhow::Error::new(crate::vmm::host_topology::ResourceContention {
            reason: "no 4 consecutive CPUs available".into(),
        });
    assert!(
        !is_topology_insufficient(&contention),
        "ResourceContention is a distinct class, not topology-insufficient",
    );
}

/// `is_topology_unrepresentable` recognizes the typed hard-fault through
/// a `.context(...)` wrap (chain-aware, mirroring the eval VM-build
/// wrap) and rejects the sibling skip type — so `classify_host_error`
/// classifies it as `HostClass::Fail` for it and only it, and a
/// `TopologyInsufficient` skip is never promoted to the hard fault.
#[test]
fn is_topology_unrepresentable_is_chain_aware_and_distinct() {
    let direct: anyhow::Error =
        anyhow::Error::new(crate::vmm::host_topology::TopologyUnrepresentable {
            reason: "topology has 513 vCPUs, exceeding the maximum of 512".into(),
        });
    assert!(is_topology_unrepresentable(&direct), "direct must match");
    // `anyhow::Error::context` is the inherent method (no `Context`
    // trait import needed — that trait is for Result/Option).
    let wrapped = direct.context("build ktstr_test VM");
    assert!(
        is_topology_unrepresentable(&wrapped),
        "context-wrapped must still match (chain-aware)",
    );
    // The host-dependent skip counterpart must NOT match — the two
    // classify_host_error classifications are disjoint (one Fail, one Skip).
    let insufficient: anyhow::Error =
        anyhow::Error::new(crate::vmm::host_topology::TopologyInsufficient {
            reason: "host has too few CPUs".into(),
        });
    assert!(
        !is_topology_unrepresentable(&insufficient),
        "TopologyInsufficient (skip) is not TopologyUnrepresentable (fail)",
    );
}

#[test]
fn run_gauntlet_test_rejects_unknown_preset() {
    // `__unit_test_dummy__` is registered in test_support::tests;
    // combined with a preset name that is not in
    // `gauntlet_presets`, the function returns 1 at the preset-
    // lookup branch.
    let exit = run_gauntlet_test("__unit_test_dummy__/__no_such_preset__");
    assert_eq!(exit, 1);
}

// ---------------------------------------------------------------
// warn_duplicate_test_names_inner — pure duplicate-walker
// ---------------------------------------------------------------
//
// The OnceLock-gated `warn_duplicate_test_names_once` wrapper is
// process-wide and reads `KTSTR_TESTS` directly, so its emit-once
// semantics aren't observable from a unit test (the gate may
// already have fired from the production listing path during
// nextest discovery, or from a sibling test in this file). The
// pure inner walker is exposed exactly so its detection logic
// is testable without that global state — these tests pin the
// dedup invariants that actually matter:
//   1. No duplicates → no output (zero-emit on clean input).
//   2. Each duplicate name surfaces EXACTLY once even when the
//      same name appears 3+ times (the warned-set prevents
//      double-prints).
//   3. The emitted line embeds the offending name in
//      double-quoted form (the canonical `{name:?}` debug
//      format the production message uses) so tooling can
//      grep operator output for the collision.
//   4. Distinct duplicate names each produce one line —
//      independent collision groups must not blur into one.
//   5. An empty input is a no-op (defensive: exhaust early
//      before any HashSet alloc).

/// No duplicates → empty sink. Pins the zero-emit base case.
#[test]
fn warn_duplicate_test_names_inner_no_duplicates_writes_nothing() {
    let mut sink = Vec::<u8>::new();
    warn_duplicate_test_names_inner(["alpha", "beta", "gamma"], &mut sink);
    assert!(
        sink.is_empty(),
        "clean input must produce zero diagnostic bytes; got {:?}",
        String::from_utf8_lossy(&sink),
    );
}

/// Empty input → no walking, no output. Defensive against a
/// regression that would crash on a zero-element HashSet
/// allocation or emit a spurious line on the empty path.
#[test]
fn warn_duplicate_test_names_inner_empty_input_writes_nothing() {
    let mut sink = Vec::<u8>::new();
    warn_duplicate_test_names_inner(std::iter::empty::<&str>(), &mut sink);
    assert!(
        sink.is_empty(),
        "empty input must emit nothing; got {:?}",
        String::from_utf8_lossy(&sink),
    );
}

/// One duplicate name appearing twice → exactly one warning
/// line containing the duplicated name. Pins the basic
/// emit-on-collision contract.
#[test]
fn warn_duplicate_test_names_inner_emits_warning_for_duplicate() {
    let mut sink = Vec::<u8>::new();
    warn_duplicate_test_names_inner(["alpha", "beta", "alpha"], &mut sink);
    let out = String::from_utf8(sink).expect("sink is utf-8");
    let lines: Vec<&str> = out.lines().collect();
    assert_eq!(
        lines.len(),
        1,
        "single duplicate must emit exactly one line; got {lines:?}",
    );
    let line = lines[0];
    assert!(
        line.contains("warning: ktstr_test:"),
        "warning prefix must be present (operator-actionable signal); \
         got line: {line:?}",
    );
    // The duplicated name must appear in the canonical `{name:?}`
    // double-quoted form so grep tooling can pull it out.
    assert!(
        line.contains("\"alpha\""),
        "the duplicated name must appear in quoted form; got: {line:?}",
    );
    assert!(
        !line.contains("\"beta\""),
        "non-duplicate names must NOT appear in any warning; got: {line:?}",
    );
}

/// A triple-collision (same name appearing 3 times) emits the
/// warning EXACTLY ONCE — the inner `warned` HashSet
/// suppresses the second and third occurrences. Pins the
/// "one warning per duplicated name" contract documented on
/// the public wrapper.
#[test]
fn warn_duplicate_test_names_inner_triple_collision_emits_once() {
    let mut sink = Vec::<u8>::new();
    warn_duplicate_test_names_inner(["dup", "dup", "dup"], &mut sink);
    let out = String::from_utf8(sink).expect("sink is utf-8");
    let lines: Vec<&str> = out.lines().collect();
    assert_eq!(
        lines.len(),
        1,
        "triple-collision must emit exactly one warning, not one-per-extra; \
         got {lines:?} — a regression that drops the warned-set guard would \
         surface here as 2 lines (one for the second, one for the third).",
    );
    assert!(
        lines[0].contains("\"dup\""),
        "warning must name the duplicated entry; got: {:?}",
        lines[0],
    );
}

/// Two distinct duplicate names → two warning lines (one per
/// collision group). A regression that collapses every
/// duplicate into a single warning would surface here as one
/// line instead of two.
#[test]
fn warn_duplicate_test_names_inner_independent_duplicates_each_warn() {
    let mut sink = Vec::<u8>::new();
    warn_duplicate_test_names_inner(["alpha", "beta", "alpha", "gamma", "beta"], &mut sink);
    let out = String::from_utf8(sink).expect("sink is utf-8");
    let lines: Vec<&str> = out.lines().collect();
    assert_eq!(
        lines.len(),
        2,
        "two independent collision groups must produce two warnings; \
         got {lines:?}",
    );
    let body = lines.join("\n");
    assert!(
        body.contains("\"alpha\""),
        "first duplicate name must appear in output; got: {body:?}",
    );
    assert!(
        body.contains("\"beta\""),
        "second duplicate name must appear in output; got: {body:?}",
    );
    assert!(
        !body.contains("\"gamma\""),
        "non-duplicate `gamma` must NOT trigger a warning; got: {body:?}",
    );
}

// -- host_capacity --

#[test]
fn host_capacity_returns_plausible_triple() {
    // `host_capacity` reads `available_parallelism` and sysfs topology.
    // The exact values depend on the test host, but the invariants
    // hold on any sane Linux machine:
    //   - cpus >= 1
    //   - llcs >= 1 (at least one cache domain)
    //   - max_cpus_per_llc >= 1
    //   - max_cpus_per_llc <= cpus (no LLC wider than the whole host)
    let (cpus, llcs, max_cpus_per_llc) = super::super::host_capacity();
    assert!(cpus >= 1, "cpus >= 1, got {cpus}");
    assert!(llcs >= 1, "llcs >= 1, got {llcs}");
    assert!(
        max_cpus_per_llc >= 1,
        "max_cpus_per_llc >= 1, got {max_cpus_per_llc}"
    );
    assert!(
        max_cpus_per_llc <= cpus,
        "max_cpus_per_llc ({max_cpus_per_llc}) must not exceed cpus ({cpus})"
    );
}

// -- for_each_gauntlet_variant --

#[test]
fn for_each_gauntlet_variant_skips_presets_exceeding_host_capacity() {
    // Pass host_cpus=1/host_llcs=1 against the preset list: every
    // current preset has total_cpus >= 4 (see `gauntlet_presets()`
    // in src/gauntlet.rs), so every preset fails
    // `TopologyConstraints::accepts` and `visit` must never be
    // called. Any entry works since the constraint check runs
    // before the visit — use the test dummy.
    let presets = crate::gauntlet::gauntlet_presets();
    // Precondition for the assertion below: if a future preset
    // with total_cpus <= 1 is added, this test must be updated to
    // account for it instead of silently under-asserting.
    let every_preset_needs_more_than_one_cpu = presets
        .iter()
        .all(|p| p.topology.total_cpus() > 1 || p.topology.llcs > 1);
    assert!(
        presets.is_empty() || every_preset_needs_more_than_one_cpu,
        "test assumes every preset requires >1 CPU or >1 LLC; \
         found a single-CPU preset — update the assertion below"
    );

    let mut visited: Vec<String> = Vec::new();
    for_each_gauntlet_variant(
        find_test("__unit_test_dummy__").unwrap(),
        &presets,
        1,
        1,
        1,
        |preset| visited.push(preset.name.to_string()),
    );
    assert!(
        visited.is_empty(),
        "with host_cpus=1 host_llcs=1, no preset should be visited; \
         visited: {visited:?}"
    );
}

#[test]
fn for_each_gauntlet_variant_visit_count_equals_accepted_preset_count() {
    // With generous host capacity (u32::MAX cpus/llcs/per-LLC),
    // every preset that `entry.constraints.accepts(...)` admits
    // must yield exactly one visit — no profile multiplier, no
    // duplicate visits per preset. Computing the expected count
    // from the same `accepts` predicate the function calls means
    // this assertion catches both directions of regression:
    //
    //   - a regression that double-visits each accepted preset
    //     produces `count == 2 * expected` (the weaker `>= 1`
    //     assertion this test replaced would have silently
    //     passed),
    //   - a regression that skips accepted presets (e.g. an
    //     inverted condition) produces `count < expected`.
    let presets = crate::gauntlet::gauntlet_presets();
    let entry = find_test("__unit_test_dummy__").unwrap();
    let expected: usize = presets
        .iter()
        .filter(|p| {
            // Non-uniform (llc_cores) presets are verifier-only and skipped
            // by for_each_gauntlet_variant regardless of constraints.
            p.topology.llc_cores.is_none()
                && entry
                    .constraints
                    .accepts(&p.topology, u32::MAX, u32::MAX, u32::MAX)
        })
        .count();
    let mut count = 0;
    for_each_gauntlet_variant(entry, &presets, u32::MAX, u32::MAX, u32::MAX, |_| {
        count += 1
    });
    assert_eq!(
        count, expected,
        "post-flag-kill: visit count must equal the number of presets the \
         entry's constraints accept; one visit per preset, no profile multiplier",
    );
}

#[test]
fn for_each_gauntlet_variant_monotonic_in_host_capacity() {
    // Comparative-baseline: giving the function MORE host capacity
    // can only let MORE presets pass the cap-size filter, never
    // fewer. The upper-bound assertion in
    // `for_each_gauntlet_variant_skips_presets_exceeding_host_capacity`
    // and the lower-bound assertion in
    // `..._visits_every_fitting_preset` both check one extreme;
    // this test anchors the monotonic relationship between them.
    // A regression that inverted the host-cap comparison (e.g.
    // `host_cpus < preset_cpus` → accept) would pass both
    // endpoint tests but fail here.
    let presets = crate::gauntlet::gauntlet_presets();
    if presets.is_empty() {
        return;
    }
    let entry = find_test("__unit_test_dummy__").unwrap();
    let count_for = |cpus: u32, llcs: u32| {
        let mut n = 0;
        for_each_gauntlet_variant(entry, &presets, cpus, llcs, u32::MAX, |_| n += 1);
        n
    };
    let tight = count_for(1, 1);
    let loose = count_for(u32::MAX, u32::MAX);
    assert!(
        loose >= tight,
        "host-capacity monotonicity violated: tight=(1,1) yielded {tight} \
         visits, loose=(u32::MAX,u32::MAX) yielded {loose}; loose \
         must admit at least as many presets as tight",
    );
}

// ---------------------------------------------------------------
// KTSTR_KERNEL_LIST parsing + sanitization + suffix dispatch
// ---------------------------------------------------------------

#[test]
fn parse_kernel_list_empty_returns_empty() {
    assert!(parse_kernel_list("").is_empty());
    assert!(parse_kernel_list(";").is_empty());
    assert!(parse_kernel_list(";;;").is_empty());
    assert!(parse_kernel_list("   ").is_empty());
}

#[test]
fn parse_kernel_list_basic_pair() {
    // Producer emits semantic labels (the version string for
    // Version specs); the parser is shape-agnostic and just
    // splits on `;` and `=` then sanitizes. A version-only
    // label sanitizes to `kernel_6_14_2`.
    let entries = parse_kernel_list("6.14.2=/cache/foo");
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0].kernel_dir, PathBuf::from("/cache/foo"));
    assert_eq!(entries[0].sanitized, "kernel_6_14_2");
}

#[test]
fn parse_kernel_list_two_entries() {
    let entries = parse_kernel_list("6.14.2=/a;6.15.0=/b");
    assert_eq!(entries.len(), 2);
    assert_eq!(entries[0].kernel_dir, PathBuf::from("/a"));
    assert_eq!(entries[0].sanitized, "kernel_6_14_2");
    assert_eq!(entries[1].kernel_dir, PathBuf::from("/b"));
    assert_eq!(entries[1].sanitized, "kernel_6_15_0");
}

#[test]
fn parse_kernel_list_drops_malformed() {
    // Missing `=`, empty label, empty path — all silently
    // dropped. Producer is `cargo ktstr` which encodes the
    // format under our control; a malformed entry indicates a
    // regression in the producer rather than operator input
    // that deserves a clear error.
    let entries = parse_kernel_list("noeq;=onlypath;onlylabel=;valid=/foo");
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0].kernel_dir, PathBuf::from("/foo"));
}

#[test]
fn parse_kernel_list_trims_whitespace() {
    let entries = parse_kernel_list("  6.14.2=/a  ;  6.15.0=/b  ");
    assert_eq!(entries.len(), 2);
    assert_eq!(entries[0].sanitized, "kernel_6_14_2");
    assert_eq!(entries[1].sanitized, "kernel_6_15_0");
}

/// `KernelEntry.label` preserves the producer-side label
/// string verbatim. Pinned because the
/// `sched_kernel_filter_accepts` range-membership branch reads
/// the raw label to feed into `decompose_version_for_compare`
/// (the sanitized form has lost dot separators required for
/// version parsing).
#[test]
fn parse_kernel_list_preserves_label() {
    let entries = parse_kernel_list("6.14.2=/a;git_tj_sched_ext_branch_main=/b;6.15-rc3=/c");
    assert_eq!(entries.len(), 3);
    assert_eq!(entries[0].label, "6.14.2");
    assert_eq!(entries[1].label, "git_tj_sched_ext_branch_main");
    assert_eq!(entries[2].label, "6.15-rc3");
}

// ---------------------------------------------------------------
// sched_kernel_filter_accepts + entry_matches_spec
// (coverage for the per-scheduler kernel filter that gates
// verifier cell emission against KTSTR_KERNEL_LIST)
// ---------------------------------------------------------------

/// Build a `KernelEntry` for filter testing without round-
/// tripping through `parse_kernel_list`. Wraps the test-only
/// `SanitizedKernelLabel::from_pre_sanitized_for_test` so
/// fixtures can hand-write the exact label strings the
/// production parser would emit.
fn mk_entry(raw: &str, sanitized: &str, dir: &str) -> KernelEntry {
    KernelEntry {
        label: raw.to_string(),
        sanitized: SanitizedKernelLabel::from_pre_sanitized_for_test(sanitized),
        kernel_dir: PathBuf::from(dir),
    }
}

#[test]
fn filter_accepts_everything_when_declared_empty() {
    // Empty sched.kernels means "no per-scheduler filter" — every
    // KTSTR_KERNEL_LIST entry passes.
    let e = mk_entry("6.14.2", "kernel_6_14_2", "/a");
    assert!(sched_kernel_filter_accepts(&[], &e));
    let weird = mk_entry("anything", "kernel_anything", "/b");
    assert!(sched_kernel_filter_accepts(&[], &weird));
}

#[test]
fn filter_matches_version_by_label() {
    let e = mk_entry("6.14.2", "kernel_6_14_2", "/a");
    // Exact raw-label equality is the primary match path.
    assert!(entry_matches_spec(&e, "6.14.2"));
    // sched.kernels = ["6.14.2"] accepts this entry.
    assert!(sched_kernel_filter_accepts(&["6.14.2"], &e));
}

#[test]
fn filter_matches_version_by_sanitized_label() {
    // Different raw label but the spec sanitizes to the same
    // sanitized form — match via the sanitized-equality fallback.
    // Example: spec "6.14.2" sanitizes to "kernel_6_14_2" and
    // the entry's sanitized label is the same.
    let e = mk_entry("6.14.2-tarball-x86_64-kcabc", "kernel_6_14_2", "/a");
    // label != "6.14.2", but sanitized matches.
    assert!(entry_matches_spec(&e, "6.14.2"));
}

#[test]
fn filter_rejects_version_mismatch() {
    let e = mk_entry("6.15.0", "kernel_6_15_0", "/a");
    // Neither raw nor sanitized matches "6.14.2".
    assert!(!entry_matches_spec(&e, "6.14.2"));
    assert!(!sched_kernel_filter_accepts(&["6.14.2"], &e));
}

#[test]
fn filter_matches_range_membership_inclusive() {
    // Range "6.14..6.16" (both endpoints inclusive). Entries
    // inside the range match; outside reject.
    let inside_low = mk_entry("6.14", "kernel_6_14", "/a");
    let inside_mid = mk_entry("6.15.3", "kernel_6_15_3", "/b");
    let inside_high = mk_entry("6.16", "kernel_6_16", "/c");
    let below = mk_entry("6.13.7", "kernel_6_13_7", "/d");
    let above = mk_entry("6.17.0", "kernel_6_17_0", "/e");

    assert!(entry_matches_spec(&inside_low, "6.14..6.16"));
    assert!(entry_matches_spec(&inside_mid, "6.14..6.16"));
    assert!(entry_matches_spec(&inside_high, "6.14..6.16"));
    assert!(!entry_matches_spec(&below, "6.14..6.16"));
    assert!(!entry_matches_spec(&above, "6.14..6.16"));
}

#[test]
fn filter_matches_range_inclusive_form_too() {
    // `..=` spelling produces the same inclusive range as `..`
    // per KernelId::parse (both inclusive on both endpoints).
    let inside = mk_entry("6.15.0", "kernel_6_15_0", "/a");
    assert!(entry_matches_spec(&inside, "6.14..=6.16"));
    let above = mk_entry("6.17.0", "kernel_6_17_0", "/b");
    assert!(!entry_matches_spec(&above, "6.14..=6.16"));
}

#[test]
fn filter_handles_unparseable_entry_label_in_range() {
    // Entry whose label isn't version-shaped (e.g. a Git
    // label) can't be in a version range — reject.
    let git_entry = mk_entry(
        "git_tj_sched_ext_branch_main",
        "kernel_git_tj_sched_ext_branch_main",
        "/a",
    );
    assert!(!entry_matches_spec(&git_entry, "6.14..6.16"));
}

#[test]
fn filter_matches_path_spec_by_sanitized() {
    // Path specs match by sanitized-label equality.
    let e = mk_entry("path_linux_a3f2b1", "kernel_path_linux_a3f2b1", "/some/dir");
    // The matching spec for a Path uses the path-derived sanitized
    // form. A user-supplied "../linux" sanitizes differently from
    // the producer's path_kernel_label output, so a Path spec in
    // sched.kernels typically wouldn't be useful — but pin the
    // sanitized-equality path anyway.
    let same_path_spec = "/some/dir";
    // KernelId::parse("/some/dir") → Path. sanitize_kernel_label
    // turns it into "kernel_some_dir" — not equal to the entry's
    // "kernel_path_linux_a3f2b1". Reject.
    assert!(!entry_matches_spec(&e, same_path_spec));
}

#[test]
fn filter_matches_cache_key_spec_by_sanitized() {
    // CacheKey spec matches when sanitized labels align.
    let e = mk_entry(
        "6.14.2-tarball-x86_64-kcabc",
        "kernel_6_14_2_tarball_x86_64_kcabc",
        "/cache/foo",
    );
    // The spec parsed as CacheKey sanitizes to the same form.
    assert!(entry_matches_spec(&e, "6.14.2-tarball-x86_64-kcabc",));
}

#[test]
fn filter_accepts_when_any_declared_spec_matches() {
    // Multiple declared specs; entry matches one of them.
    let e = mk_entry("6.15.3", "kernel_6_15_3", "/a");
    assert!(sched_kernel_filter_accepts(
        &[
            "6.14.2",
            "6.14..6.16",
            "git+https://example.com/r#branch=main"
        ],
        &e,
    ));
}

#[test]
fn filter_rejects_when_no_declared_spec_matches() {
    let e = mk_entry("7.0.0", "kernel_7_0_0", "/a");
    assert!(!sched_kernel_filter_accepts(&["6.14.2", "6.14..6.16"], &e,));
}

// ---------------------------------------------------------------
// Pin the exact diagnostic strings emitted by run_verifier_cell
// when the kernel-list lookup fails. Tests exercise the formatter
// helpers directly — no need to spawn a separate test binary
// because the eprintln! call sites now route through these pure
// formatters.
// ---------------------------------------------------------------

#[test]
fn format_empty_kernel_list_error_names_cell_and_dispatcher() {
    let s = format_empty_kernel_list_error("verifier/sched_foo/kernel_6_14_2/tiny-1llc");
    // Cell name appears verbatim so the operator can grep their
    // own invocation for the failing cell.
    assert!(
        s.contains("verifier/sched_foo/kernel_6_14_2/tiny-1llc"),
        "missing cell name in: {s}",
    );
    // Root cause is named explicitly.
    assert!(
        s.contains("KTSTR_KERNEL_LIST is empty"),
        "missing cause: {s}"
    );
    // Actionable hint points back at the dispatcher subcommand
    // (the only supported entry point).
    assert!(
        s.contains("cargo ktstr verifier"),
        "missing actionable hint: {s}",
    );
}

#[test]
fn format_unknown_kernel_label_error_lists_present_labels_and_both_fix_paths() {
    let present = vec!["kernel_6_14_2", "kernel_6_15_0"];
    let s = format_unknown_kernel_label_error(
        "verifier/sched_foo/kernel_7_0_0/tiny-1llc",
        "kernel_7_0_0",
        "sched_foo",
        &present,
    );
    // Cell name + missing label appear so operators see exactly
    // which lookup failed.
    assert!(
        s.contains("verifier/sched_foo/kernel_7_0_0/tiny-1llc"),
        "missing cell name: {s}",
    );
    // Debug-formatted missing label (`{kernel_label:?}` produces
    // double-quoted output).
    assert!(s.contains("\"kernel_7_0_0\""), "missing debug label: {s}");
    // Present-labels enumeration: every entry must appear so the
    // operator can see what IS available.
    assert!(s.contains("kernel_6_14_2"), "missing present[0]: {s}");
    assert!(s.contains("kernel_6_15_0"), "missing present[1]: {s}");
    // Scheduler name surfaces in the declaration-side fix hint.
    assert!(s.contains("sched_foo"), "missing scheduler name: {s}");
    // Both fix paths are documented: add a kernel to the
    // dispatcher OR drop the matching entry from the declaration.
    assert!(
        s.contains("add --kernel"),
        "missing dispatcher-side fix: {s}"
    );
    assert!(
        s.contains("declare_scheduler!"),
        "missing declaration-side fix: {s}",
    );
}

#[test]
fn format_unknown_kernel_label_error_empty_present_renders_empty_brackets() {
    // Edge case: kernel_list has entries that fail the find()
    // (string equality drifted) but the present slice the caller
    // assembles is empty — still surfaces the bracket pair so the
    // diagnostic format is uniform with the non-empty case.
    let s = format_unknown_kernel_label_error("verifier/foo/kernel_x/tiny", "kernel_x", "foo", &[]);
    assert!(
        s.contains("Present labels: []"),
        "missing empty brackets: {s}"
    );
}

#[test]
fn format_unknown_kernel_label_error_joins_present_with_comma_space() {
    // Three-entry present slice must render comma-space separated
    // to match the `present.join(", ")` contract.
    let present = vec!["a", "b", "c"];
    let s = format_unknown_kernel_label_error(
        "verifier/foo/kernel_x/tiny",
        "kernel_x",
        "foo",
        &present,
    );
    assert!(
        s.contains("Present labels: [a, b, c]"),
        "wrong join delimiter: {s}",
    );
}

#[test]
fn sanitize_kernel_label_pure_version() {
    assert_eq!(sanitize_kernel_label("6.14.2"), "kernel_6_14_2");
}

#[test]
fn sanitize_kernel_label_rc_suffix() {
    assert_eq!(sanitize_kernel_label("6.15-rc3"), "kernel_6_15_rc3");
}

/// The sanitizer is shape-agnostic — it normalizes any input
/// that happens to flow in. The producer-side encoder now
/// emits semantic labels, but a future regression that
/// surfaced a raw cache-key basename would still produce a
/// valid (if uglier) nextest identifier rather than crashing.
/// Pinned via a synthetic full-cache-key input.
#[test]
fn sanitize_kernel_label_handles_full_cache_key_shape() {
    assert_eq!(
        sanitize_kernel_label("6.14.2-tarball-x86_64-kcabc1234"),
        "kernel_6_14_2_tarball_x86_64_kcabc1234",
    );
}

/// Git-source semantic label `git_tj_sched_ext_branch_for-next` from
/// the producer-side encoder maps to the dash-stripped form
/// the sanitizer produces.
#[test]
fn sanitize_kernel_label_git_semantic_label() {
    assert_eq!(
        sanitize_kernel_label("git_tj_sched_ext_branch_for-next"),
        "kernel_git_tj_sched_ext_branch_for_next",
    );
}

/// Path-source semantic label `path_linux_a3f2b1` is already
/// `[a-z0-9_]+` so the sanitizer only adds the `kernel_`
/// prefix.
#[test]
fn sanitize_kernel_label_path_semantic_label() {
    assert_eq!(
        sanitize_kernel_label("path_linux_a3f2b1"),
        "kernel_path_linux_a3f2b1",
    );
}

#[test]
fn sanitize_kernel_label_lowercases() {
    assert_eq!(sanitize_kernel_label("ABC-DEF"), "kernel_abc_def");
}

#[test]
fn sanitize_kernel_label_collapses_repeated_separators() {
    assert_eq!(sanitize_kernel_label("a..b...c"), "kernel_a_b_c");
}

#[test]
fn sanitize_kernel_label_strips_trailing_underscore() {
    assert_eq!(sanitize_kernel_label("for-next-"), "kernel_for_next");
}

#[test]
fn sanitize_kernel_label_empty_input() {
    assert_eq!(sanitize_kernel_label(""), "kernel_");
}

// ---------------------------------------------------------------
// SanitizedKernelLabel — newtype invariants
// ---------------------------------------------------------------
//
// `SanitizedKernelLabel::new(raw)` is the only production
// path that yields a value of the type, and it always routes
// through `sanitize_kernel_label`. The tests below pin each
// surface (constructor, accessors, `PartialEq` impls) directly
// so a regression that bypassed the sanitizer (e.g. a future
// `From<String>` impl that wraps verbatim) or dropped one of
// the comparison-ergonomics impls would surface as a unit-test
// failure rather than as a downstream filter mismatch.

/// `SanitizedKernelLabel::new(raw)` yields a value whose
/// `as_str()` equals `sanitize_kernel_label(raw)` byte-for-
/// byte. Pins the constructor's "always sanitize" contract:
/// the only path that builds a `SanitizedKernelLabel` MUST run
/// `sanitize_kernel_label`, otherwise a future caller could
/// stuff a raw label into the field via a regression that
/// forgot to invoke the sanitizer. Multiple inputs covered to
/// distinguish "happens to match" from "truly routes through
/// the sanitizer" — version, RC suffix, mixed case, embedded
/// dots-vs-dashes.
#[test]
fn sanitized_kernel_label_new_runs_sanitizer() {
    for raw in [
        "6.14.2",
        "6.15-rc3",
        "ABC-DEF",
        "git_tj_sched_ext_branch_for-next",
        "",
    ] {
        let label = SanitizedKernelLabel::new(raw);
        assert_eq!(
            label.as_str(),
            sanitize_kernel_label(raw),
            "SanitizedKernelLabel::new({raw:?}).as_str() must equal \
             sanitize_kernel_label({raw:?}); a regression that wrapped \
             raw input verbatim would surface here",
        );
    }
}

/// `as_str()` returns the sanitized inner string, NOT the raw
/// input. Round-trip through `as_str()` matches what the
/// sanitizer produced — distinct from
/// `sanitized_kernel_label_new_runs_sanitizer` which checks
/// the constructor wires through; this checks that read-side
/// access exposes the SAME bytes the constructor wrote.
#[test]
fn sanitized_kernel_label_as_str_returns_sanitized_form() {
    let label = SanitizedKernelLabel::new("6.14.2");
    assert_eq!(label.as_str(), "kernel_6_14_2");
    // A regression that returned the raw input from `as_str()`
    // would surface here because the raw input contains `.`
    // which is `_` after sanitization.
    assert_ne!(label.as_str(), "6.14.2");
}

/// `PartialEq<&str>` lets `assert_eq!(label, "kernel_6_14_2")`
/// stay readable at every consumer (and is what
/// `parse_kernel_list_*` tests already use). A regression that
/// dropped or narrowed the impl (e.g. switched to `PartialEq<
/// String>` only) would force every consumer to chain
/// `.as_str()` and break the existing test suite.
#[test]
fn sanitized_kernel_label_partial_eq_with_str_ref() {
    let label = SanitizedKernelLabel::new("6.14.2");
    let want: &str = "kernel_6_14_2";
    assert_eq!(label, want);
    // Symmetric inequality: distinct sanitization output must
    // NOT compare equal to a different `&str`.
    let other: &str = "kernel_6_15_0";
    assert_ne!(label, other);
}

/// `PartialEq<str>` covers the unsized-`str` comparison path
/// (e.g. dereferenced `String` slice) distinct from the
/// `&str` path above. Both impls are explicit because Rust's
/// auto-deref to `&str` does not bridge the `PartialEq<str>`
/// case for some assert macros / generic comparators. A
/// regression that dropped just the `PartialEq<str>` impl
/// would compile most consumer sites but break callers that
/// land on the unsized form.
#[test]
fn sanitized_kernel_label_partial_eq_with_str_unsized() {
    let label = SanitizedKernelLabel::new("6.14.2");
    let owned: String = "kernel_6_14_2".to_string();
    // `*owned` is `str` (unsized) — exercises `PartialEq<str>`
    // rather than `PartialEq<&str>`. Wrap with `&` to satisfy
    // `PartialEq::eq`'s `&Self`-vs-`&Other` shape; the impl
    // body still operates on the unsized `str`.
    assert!(
        label == *owned.as_str(),
        "PartialEq<str> impl missing — assert against unsized str failed",
    );
    // Symmetric inequality for the unsized path.
    let other: String = "kernel_6_15_0".to_string();
    assert!(label != *other.as_str());
}

/// `strip_kernel_suffix` is a no-op for single-kernel mode (0 or
/// 1 entries) — returns the input verbatim and signals "no
/// kernel override needed."
#[test]
fn strip_kernel_suffix_single_kernel_passthrough() {
    let kernel_list = vec![KernelEntry {
        label: "6.14.2".to_string(),
        sanitized: SanitizedKernelLabel::from_pre_sanitized_for_test("kernel_6_14_2"),
        kernel_dir: PathBuf::from("/a"),
    }];
    let (stripped, entry) = strip_kernel_suffix("gauntlet/eevdf/2llc", &kernel_list).unwrap();
    assert_eq!(stripped, "gauntlet/eevdf/2llc");
    assert!(entry.is_none());

    let (stripped, entry) = strip_kernel_suffix("ktstr/eevdf", &[]).unwrap();
    assert_eq!(stripped, "ktstr/eevdf");
    assert!(entry.is_none());
}

/// In multi-kernel mode (2+ entries), the suffix is required and
/// peeled off. The matching `KernelEntry` is returned.
#[test]
fn strip_kernel_suffix_multi_kernel_peels_suffix() {
    let kernel_list = vec![
        KernelEntry {
            label: "6.14.2".to_string(),
            sanitized: SanitizedKernelLabel::from_pre_sanitized_for_test("kernel_6_14_2"),
            kernel_dir: PathBuf::from("/a"),
        },
        KernelEntry {
            label: "6.15.0".to_string(),
            sanitized: SanitizedKernelLabel::from_pre_sanitized_for_test("kernel_6_15_0"),
            kernel_dir: PathBuf::from("/b"),
        },
    ];
    let (stripped, entry) =
        strip_kernel_suffix("gauntlet/eevdf/2llc/kernel_6_14_2", &kernel_list).unwrap();
    assert_eq!(stripped, "gauntlet/eevdf/2llc");
    assert_eq!(entry.unwrap().kernel_dir, PathBuf::from("/a"));

    let (stripped, entry) =
        strip_kernel_suffix("gauntlet/eevdf/2llc/kernel_6_15_0", &kernel_list).unwrap();
    assert_eq!(stripped, "gauntlet/eevdf/2llc");
    assert_eq!(entry.unwrap().kernel_dir, PathBuf::from("/b"));
}

/// In multi-kernel mode, a test name that lacks the kernel
/// suffix surfaces an actionable error rather than silently
/// using the first kernel — the suffix is part of every test
/// name `--list` emitted, so a missing suffix indicates
/// operator hand-construction or stale tooling.
#[test]
fn strip_kernel_suffix_multi_kernel_missing_suffix_errors() {
    let kernel_list = vec![
        KernelEntry {
            label: "6.14.2".to_string(),
            sanitized: SanitizedKernelLabel::from_pre_sanitized_for_test("kernel_6_14_2"),
            kernel_dir: PathBuf::from("/a"),
        },
        KernelEntry {
            label: "6.15.0".to_string(),
            sanitized: SanitizedKernelLabel::from_pre_sanitized_for_test("kernel_6_15_0"),
            kernel_dir: PathBuf::from("/b"),
        },
    ];
    let err = strip_kernel_suffix("gauntlet/eevdf/2llc", &kernel_list)
        .expect_err("missing suffix in multi-kernel mode must error");
    assert!(
        err.contains("no recognised kernel suffix"),
        "error must mention missing suffix, got: {err}",
    );
}

/// Suffix peeling is anchored at the end of the test name —
/// gauntlet variants whose body contains `/` (the test / preset
/// separator) are not accidentally peeled. A naive
/// `rsplit_once('/')` would peel the preset segment instead.
#[test]
fn strip_kernel_suffix_does_not_peel_preset_segment() {
    let kernel_list = vec![
        KernelEntry {
            label: "6.14.2".to_string(),
            sanitized: SanitizedKernelLabel::from_pre_sanitized_for_test("kernel_6_14_2"),
            kernel_dir: PathBuf::from("/a"),
        },
        KernelEntry {
            label: "6.15.0".to_string(),
            sanitized: SanitizedKernelLabel::from_pre_sanitized_for_test("kernel_6_15_0"),
            kernel_dir: PathBuf::from("/b"),
        },
    ];
    // The preset name is `2llc`, NOT `kernel_6_14_2` — the
    // peeler must require an EXACT match against a known
    // sanitized label, not just any `/<word>` ending.
    let (stripped, entry) =
        strip_kernel_suffix("gauntlet/eevdf/2llc/kernel_6_14_2", &kernel_list).unwrap();
    // Stripped name still contains both of the original path
    // segments (eevdf, 2llc).
    assert_eq!(stripped, "gauntlet/eevdf/2llc");
    assert!(entry.is_some());
}

// ---------------------------------------------------------------
// host_only kernel-suffix skip — multi-kernel listing
// ---------------------------------------------------------------
//
// `list_tests_all` and `list_tests_budget` short-circuit
// `host_only` entries: a host_only test never boots a VM, so the
// kernel never affects what runs. Both listers emit ONE entry per
// host_only test regardless of `KTSTR_KERNEL_LIST` cardinality —
// otherwise N identical copies of the same host-side function
// would land in nextest's plan.
//
// The tests below register a dedicated `host_only=true` entry in
// `KTSTR_TESTS` via `linkme::distributed_slice`, set
// `KTSTR_KERNEL_LIST` to a 2-entry payload, and capture stdout
// while invoking each lister. The capture asserts the host_only
// entry name appears EXACTLY once and never with a `/kernel_…`
// suffix.

/// Process-wide mutex serializing every stdout-capture call in
/// this module. `fd 1 → tempfile` redirection is a non-reentrant
/// process-global mutation — two concurrent callers would see
/// each other's output land in their sink. Mirrors the
/// `STDERR_CAPTURE_LOCK` pattern in `test_support::test_helpers`
/// (see `capture_stderr_serializes_concurrent_callers` for the
/// rationale and the failure mode without serialization).
static STDOUT_CAPTURE_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// RAII guard that restores the saved stdout fd on Drop, even if
/// the captured closure panics under the `panic = "unwind"` test
/// profile. Without this guard a panicking closure would leak the
/// fd-1 swap and every subsequent stdout write in the test
/// process would land in the orphaned tempfile. `saved` is
/// `Option` so Drop can `take()` and consume it without an `&mut`
/// borrow fight. Mirrors `StderrRestoreGuard` in
/// `test_support::test_helpers`.
struct StdoutRestoreGuard {
    saved: Option<std::os::fd::OwnedFd>,
}
impl Drop for StdoutRestoreGuard {
    fn drop(&mut self) {
        if let Some(saved) = self.saved.take() {
            let _ = nix::unistd::dup2_stdout(&saved);
        }
    }
}

/// Run `f` with stdout redirected to an in-memory tempfile;
/// return both `f`'s value and the captured bytes. Uses
/// [`STDOUT_CAPTURE_LOCK`] to serialize against every other
/// stdout-capture call in this module. The RAII
/// [`StdoutRestoreGuard`] restores fd 1 even if `f` panics
/// under `panic = "unwind"`. Mirrors `capture_stderr` in
/// `test_support::test_helpers`.
fn capture_stdout<R>(f: impl FnOnce() -> R) -> (R, Vec<u8>) {
    use std::io::{Read, Seek, SeekFrom, Write};
    let _lock = STDOUT_CAPTURE_LOCK.lock_unpoisoned();
    let mut sink = tempfile::tempfile().expect("create stdout-capture tempfile");
    // Flush before redirect: println! is line-buffered behind
    // the Stdout lock; pre-call bytes need to reach the
    // ORIGINAL fd 1 or they leak into the captured tempfile.
    std::io::stdout().flush().ok();
    let saved = nix::unistd::dup(std::io::stdout()).expect("dup(stdout)");
    nix::unistd::dup2_stdout(&sink).expect("dup2_stdout(sink)");
    let guard = StdoutRestoreGuard { saved: Some(saved) };
    let result = f();
    std::io::stdout().flush().ok();
    drop(guard);
    sink.seek(SeekFrom::Start(0)).expect("rewind sink");
    let mut bytes = Vec::new();
    sink.read_to_end(&mut bytes).expect("read sink");
    (result, bytes)
}

/// Stub func for the host_only listing-test entry. The listers
/// (`list_tests_all` / `list_tests_budget`) only iterate
/// `KTSTR_TESTS` to emit names and never call `func` — but every
/// `KTSTR_TESTS` entry is ALSO reachable by the run dispatcher: a
/// full `cargo ktstr test` run executes it, exactly like the sibling
/// `__unit_test_dummy__` sentinel. So the func must be a harmless
/// pass, not an error. `host_only = true` runs it on the host with
/// no VM and it asserts nothing. (An earlier Err-bailing stub
/// assumed the entry was never dispatched, which made every full
/// run fail when the dispatcher reached it.)
fn host_only_listing_stub(
    _ctx: &crate::scenario::Ctx,
) -> anyhow::Result<crate::assert::AssertResult> {
    Ok(crate::assert::AssertResult::pass())
}

/// Distinct sentinel name so the listing-output filters in the
/// tests below match this entry and not the `__unit_test_dummy__`
/// (also registered in `KTSTR_TESTS` from the
/// `test_support::tests` module) or any other entry that may be
/// added later. The `__unit_test_…__` shape collides with
/// `is_test_sentinel` (see the predicate at the top of this
/// module) so the `cargo test` harness still classifies it as a
/// sentinel and the early-dispatch warning logic does not
/// double-fire.
const HOST_ONLY_LISTING_NAME: &str = "__unit_test_host_only_listing__";

#[linkme::distributed_slice(KTSTR_TESTS)]
static __HOST_ONLY_LISTING_ENTRY: KtstrTestEntry = KtstrTestEntry {
    name: HOST_ONLY_LISTING_NAME,
    func: host_only_listing_stub,
    host_only: true,
    ..KtstrTestEntry::DEFAULT
};

/// Two-kernel KTSTR_KERNEL_LIST payload reused by the listing
/// tests below. Both labels sanitize to distinct nextest
/// suffixes (`kernel_6_14_2`, `kernel_6_15_0`), so a regression
/// that started emitting `/kernel_…` suffixes for the host_only
/// entry would surface as either `2` matches (one per kernel)
/// rather than the expected `1`, or as suffix substrings on the
/// emitted line.
const TWO_KERNEL_LIST: &str = "6.14.2=/cache/a;6.15.0=/cache/b";

/// Filter the captured listing output to only the lines that
/// reference `HOST_ONLY_LISTING_NAME`. Other lines from the
/// `__unit_test_dummy__` entry (and from any future entries
/// registered in this binary's `KTSTR_TESTS`) are intentionally
/// dropped so the assertions key on this fixture's behaviour
/// alone.
fn host_only_listing_lines(captured: &[u8]) -> Vec<String> {
    std::str::from_utf8(captured)
        .expect("capture must be UTF-8")
        .lines()
        .filter(|l| l.contains(HOST_ONLY_LISTING_NAME))
        .map(str::to_owned)
        .collect()
}

/// `list_tests_all` in multi-kernel mode emits exactly ONE line
/// for a `host_only` entry, with NO `/kernel_…` suffix. Pins the
/// `if entry.host_only { println!("ktstr/{}: test", entry.name); }`
/// branch at the top of `list_tests_all` against a regression
/// that fell through into the kernel-suffix loop. A regression
/// would yield 2 matches (one per kernel) and at least one line
/// would carry a `/kernel_6_14_2` or `/kernel_6_15_0` suffix.
///
/// Holds [`crate::test_support::test_helpers::lock_env`] for the
/// full save/mutate/restore window — `KTSTR_KERNEL_LIST` is
/// process-wide, and the budget-test sibling below also rewrites
/// env vars. Without the lock, a concurrent test mutating a
/// different env key could observe a transiently-corrupt
/// `KTSTR_KERNEL_LIST` value.
#[test]
fn list_tests_all_host_only_skips_kernel_suffix_under_multi_kernel() {
    use crate::test_support::test_helpers::{EnvVarGuard, lock_env};
    let _env_lock = lock_env();
    let _kernel_list = EnvVarGuard::set(crate::KTSTR_KERNEL_LIST_ENV, TWO_KERNEL_LIST);
    // Suppress the budget-mode branch — `KTSTR_BUDGET_SECS` would
    // route the dispatcher through `list_tests_budget` instead of
    // `list_tests_all`, but we are calling the lister directly so
    // the dispatcher path is irrelevant. Removing the env var here
    // is defensive against a parallel test that set it without
    // restoring (would not affect this call's output, but keeps
    // the test's runtime hypothesis explicit).
    let _budget_guard = EnvVarGuard::remove(crate::KTSTR_BUDGET_SECS_ENV);

    let (_, captured) = capture_stdout(|| list_tests_all(false));
    let lines = host_only_listing_lines(&captured);

    assert_eq!(
        lines.len(),
        1,
        "list_tests_all must emit exactly 1 line for a host_only entry \
         under multi-kernel mode (saw {n}): {lines:?}",
        n = lines.len(),
    );
    let line = &lines[0];
    // Expected exact form (mirrors the `println!("ktstr/{}: test", entry.name)`
    // in the host_only branch of `list_tests_all`).
    assert_eq!(
        line,
        &format!("ktstr/{HOST_ONLY_LISTING_NAME}: test"),
        "host_only line must be `ktstr/<name>: test` with no kernel suffix",
    );
    // Belt-and-suspenders: neither sanitized kernel label appears
    // anywhere on the line, even as a substring.
    assert!(
        !line.contains("kernel_6_14_2") && !line.contains("kernel_6_15_0"),
        "host_only line must carry NO sanitized kernel suffix — \
         a regression that emitted `/kernel_…` would surface here. line: {line:?}",
    );
}

/// `list_tests_budget` mirror: in multi-kernel mode, the
/// budget-selecting lister emits exactly ONE candidate for a
/// `host_only` entry without a kernel suffix. Pins the second
/// `if entry.host_only { … } else { … }` branch in
/// `list_tests_budget` against the same regression class as the
/// `list_tests_all` sibling.
///
/// Budget is set generously (10000 secs) so the greedy selector
/// in `crate::budget::select` picks every distinct-feature
/// candidate including this fixture (the `HOST_ONLY_SHIFT` bit
/// in `extract_features` makes the host_only entry's feature set
/// uniquely contributory — see `budget::extract_features`).
/// The selector prints to stdout AND `eprintln!`s a summary
/// line to stderr; only stdout is captured here, so the stderr
/// summary lands on the test runner's normal stderr.
#[test]
fn list_tests_budget_host_only_skips_kernel_suffix_under_multi_kernel() {
    use crate::test_support::test_helpers::{EnvVarGuard, lock_env};
    let _env_lock = lock_env();
    let _kernel_list = EnvVarGuard::set(crate::KTSTR_KERNEL_LIST_ENV, TWO_KERNEL_LIST);

    let (_, captured) = capture_stdout(|| list_tests_budget(false, 10_000.0));
    let lines = host_only_listing_lines(&captured);

    assert_eq!(
        lines.len(),
        1,
        "list_tests_budget must emit exactly 1 candidate line for a \
         host_only entry under multi-kernel mode (saw {n}): {lines:?}",
        n = lines.len(),
    );
    let line = &lines[0];
    assert_eq!(
        line,
        &format!("ktstr/{HOST_ONLY_LISTING_NAME}: test"),
        "host_only candidate name must be `ktstr/<name>: test` with no kernel suffix",
    );
    assert!(
        !line.contains("kernel_6_14_2") && !line.contains("kernel_6_15_0"),
        "host_only candidate must carry NO sanitized kernel suffix — \
         a regression that emitted `/kernel_…` would surface here. line: {line:?}",
    );
}

// ---------------------------------------------------------------
// KTSTR_CARGO_TEST_MODE listing behavior
// ---------------------------------------------------------------
//
// `list_tests_all` and `list_tests_budget` skip gauntlet
// emission when `KTSTR_CARGO_TEST_MODE` is active. Pins the
// dispatch contract: bare `cargo test` runs each test once
// with its declared topology, no per-preset fan-out.
// Multi-kernel suffix emission is also suppressed because the
// cargo-ktstr resolver that produces `KTSTR_KERNEL_LIST` is
// not on the cargo-test path.

/// Under `KTSTR_CARGO_TEST_MODE=1`, `list_tests_all` emits
/// exactly one `ktstr/{name}: test` line per registered entry
/// — no `gauntlet/...` lines. Pins the gauntlet-skip branch.
#[test]
fn list_tests_all_cargo_test_mode_skips_gauntlet() {
    use crate::test_support::test_helpers::{EnvVarGuard, lock_env};
    let _env_lock = lock_env();
    let _cargo = EnvVarGuard::set(crate::KTSTR_CARGO_TEST_MODE_ENV, "1");
    let _no_kernel_list = EnvVarGuard::remove(crate::KTSTR_KERNEL_LIST_ENV);
    let _budget_guard = EnvVarGuard::remove(crate::KTSTR_BUDGET_SECS_ENV);

    let (_, captured) = capture_stdout(|| list_tests_all(false));
    let stdout = std::str::from_utf8(&captured).expect("utf-8");
    let gauntlet_lines: Vec<&str> = stdout
        .lines()
        .filter(|l| l.starts_with("gauntlet/"))
        .collect();
    assert!(
        gauntlet_lines.is_empty(),
        "cargo-test-mode must suppress every `gauntlet/...` line; \
         got {} lines: {gauntlet_lines:?}",
        gauntlet_lines.len(),
    );
}

/// `result_to_exit_code` maps the 4-state verdict lattice
/// (`Fail > Inconclusive > Pass > Skip`) to 3 distinct
/// nextest-dispatch exit codes: Pass → 0, Inconclusive → 2,
/// Fail → 1. The distinct Inconclusive code lets CI tooling
/// triage zero-denominator runs separately from real
/// regressions. A regression that mapped Inconclusive into
/// the Pass branch (`Ok(_) => 0`) would silently let a test
/// whose ratio gate could not evaluate slip past the CI
/// summary as a green run.
#[test]
fn result_to_exit_code_inconclusive_maps_to_distinct_code() {
    use crate::assert::{AssertDetail, AssertResult, DetailKind};
    // Pass → 0 (no expect_err inversion, no allow_inconclusive).
    assert_eq!(
        result_to_exit_code(Ok(AssertResult::pass()), false, false),
        0
    );
    // Inconclusive → 2 — distinct from Pass and Fail when
    // allow_inconclusive is unset.
    let inc =
        AssertResult::inconclusive(AssertDetail::new(DetailKind::Benchmark, "zero-denominator"));
    assert_eq!(result_to_exit_code(Ok(inc), false, false), 2);
    // expect_err + Pass → 1 (the test was supposed to fail).
    assert_eq!(
        result_to_exit_code(Ok(AssertResult::pass()), true, false),
        1
    );
    // expect_err + Inconclusive → 1 (the test was supposed to
    // produce a real failure; an Inconclusive verdict does
    // not satisfy expect_err since the gate could not even
    // evaluate). Distinct surfaced message in stderr.
    let inc2 = AssertResult::inconclusive(AssertDetail::new(
        DetailKind::Benchmark,
        "zero-denominator under expect_err",
    ));
    assert_eq!(result_to_exit_code(Ok(inc2), true, false), 1);
}

/// Anti-drift truth table: `final_outcome(&r, e, a).to_exit_code()` MUST
/// equal `result_to_exit_code(r, e, a)` for every (result, expect_err,
/// allow_inconclusive). They are two encodings of one classification —
/// the sidecar finalize records [`final_outcome`], nextest reads
/// [`result_to_exit_code`] — so any divergence silently mis-records a
/// test's pass/fail across the footer, `stats`, and `replay`. Covers the
/// Ok arms (pass / skip / inconclusive) and the Err arms (plain + each
/// marker), exercising `final_outcome`'s replicated first-match
/// precedence and context-aware marker downcasts.
#[test]
fn final_outcome_projects_to_result_to_exit_code() {
    use crate::assert::{AssertDetail, AssertResult, DetailKind};
    type ResultBuilder = fn() -> Result<AssertResult>;
    let cases: [(ResultBuilder, &str); 9] = [
        (|| Ok(AssertResult::pass()), "ok_pass"),
        (|| Ok(AssertResult::skip("skip reason")), "ok_skip"),
        (
            || {
                Ok(AssertResult::inconclusive(AssertDetail::new(
                    DetailKind::Benchmark,
                    "zero-denominator",
                )))
            },
            "ok_inconclusive",
        ),
        (|| Err(anyhow::anyhow!("plain failure")), "err_plain"),
        (
            || Err(anyhow::anyhow!("f").context(crate::test_support::eval::PostVmAssertionFailure)),
            "err_post_vm_assertion",
        ),
        (
            || {
                Err(anyhow::anyhow!("f")
                    .context(crate::test_support::eval::ExpectAutoReproSatisfied))
            },
            "err_expect_auto_repro_satisfied",
        ),
        (
            || {
                Err(anyhow::anyhow!("f")
                    .context(crate::test_support::eval::ScxBpfErrorMatcherMismatch))
            },
            "err_scx_bpf_matcher_mismatch",
        ),
        (
            || Err(anyhow::anyhow!("f").context(crate::test_support::eval::SchedulerBuildRefused)),
            "err_scheduler_build_refused",
        ),
        (
            || Err(anyhow::anyhow!("f").context(crate::test_support::eval::SurvivesStormViolated)),
            "err_survives_storm_violated",
        ),
    ];
    for (build, label) in cases {
        for expect_err in [false, true] {
            for allow_inconclusive in [false, true] {
                let via_final =
                    final_outcome(&build(), expect_err, allow_inconclusive).to_exit_code();
                let via_dispatch = result_to_exit_code(build(), expect_err, allow_inconclusive);
                assert_eq!(
                    via_final, via_dispatch,
                    "verdict drift: {label} expect_err={expect_err} \
                     allow_inconclusive={allow_inconclusive}: final_outcome->{via_final} \
                     result_to_exit_code->{via_dispatch}",
                );
            }
        }
    }
}

/// `Verdict::sidecar_bits` mapping + the Pass-vs-Skip distinction in
/// `final_outcome`. The truth-table above asserts `to_exit_code()`
/// equivalence, but `to_exit_code` collapses BOTH Pass and Skip to
/// `EXIT_PASS` — so a Pass<->Skip drift in `final_outcome` is invisible
/// there yet would set the WRONG persisted sidecar bits (passed vs
/// skipped). This pins the four `sidecar_bits` tuples AND that
/// `final_outcome` keeps Pass and Skip distinct (incl. Skip dominating
/// `expect_err`).
#[test]
fn verdict_sidecar_bits_and_pass_vs_skip_distinction() {
    use crate::assert::AssertResult;
    assert_eq!(Verdict::Pass.sidecar_bits(), (true, false, false));
    assert_eq!(Verdict::Skip.sidecar_bits(), (false, true, false));
    assert_eq!(Verdict::Inconclusive.sidecar_bits(), (false, false, true));
    assert_eq!(Verdict::Fail.sidecar_bits(), (false, false, false));

    assert_eq!(
        final_outcome(&Ok(AssertResult::pass()), false, false),
        Verdict::Pass,
    );
    assert_eq!(
        final_outcome(&Ok(AssertResult::skip("s")), false, false),
        Verdict::Skip,
    );
    assert_eq!(
        final_outcome(&Ok(AssertResult::skip("s")), true, false),
        Verdict::Skip,
        "Skip dominates expect_err (the test never evaluated)",
    );
}

/// A Skip verdict maps to EXIT_PASS regardless of expect_err — the
/// test never evaluated, so there is no guest failure to "expect."
/// Regression guard: a `post_vm_skip` (e.g. a placeholder failure
/// dump under VM load) on an `expect_err` test previously fell into
/// the `Ok(r) if expect_err` arm and surfaced as "expected error but
/// test passed" (EXIT_FAIL), turning a load-starvation skip into a
/// flaky failure. The `is_skip()` arm must precede the expect_err
/// arms.
#[test]
fn result_to_exit_code_skip_maps_to_pass_even_under_expect_err() {
    use crate::assert::AssertResult;
    // Skip without expect_err → 0 (the `Ok(_)` arm already covers
    // this; pinned against a match-reorder regression).
    assert_eq!(
        result_to_exit_code(
            Ok(AssertResult::skip("inconclusive: placeholder dump")),
            false,
            false
        ),
        0
    );
    // Skip WITH expect_err → 0. The is_skip arm dominates: a skip is
    // never "the expected error failed to appear" because the test
    // never evaluated.
    assert_eq!(
        result_to_exit_code(
            Ok(AssertResult::skip("inconclusive: placeholder dump")),
            true,
            false
        ),
        0
    );
}

/// `PerfModeUnavailable` (direct and `.context`-wrapped) is a
/// host-insufficiency: it SKIPS (EXIT_PASS) when KTSTR_NO_SKIP_MODE is
/// unset — including under expect_err=true, because the skip arm precedes
/// the expect_err arm. Mirror of the ResourceContention / TopologyInsufficient
/// skip tests; pins the perf-mode skip routing through the shared
/// `classify_host_error` (`HostClass::Skip`), surfaced via
/// `result_to_exit_code` → `err_to_exit_code`. The no-skip-mode promotion to EXIT_FAIL is
/// pinned by `result_to_exit_code_skip_class_fails_under_no_skip_mode`. NOTE:
/// the 3rd `result_to_exit_code` arg is `allow_inconclusive`, NOT no_skip —
/// no_skip is read from KTSTR_NO_SKIP_MODE_ENV inside the fn.
#[test]
fn result_to_exit_code_perf_mode_unavailable_skips_when_not_no_skip() {
    use crate::test_support::test_helpers::{EnvVarGuard, lock_env};
    use anyhow::Context as _;
    let _env_lock = lock_env();
    let _no_skip = EnvVarGuard::remove(crate::KTSTR_NO_SKIP_MODE_ENV);

    let direct = || -> Result<crate::assert::AssertResult> {
        Err(anyhow::Error::new(
            crate::vmm::host_topology::PerfModeUnavailable {
                reason: "host too small for perf topology".to_string(),
            },
        ))
    };
    assert_eq!(result_to_exit_code(direct(), false, false), EXIT_PASS);
    // Skip arm precedes the expect_err arm: still EXIT_PASS.
    assert_eq!(result_to_exit_code(direct(), true, false), EXIT_PASS);

    // Context-wrapped (production shape) still recognised via the chain
    // walk → EXIT_PASS.
    let wrapped = || -> Result<crate::assert::AssertResult> {
        Err(anyhow::Error::new(
            crate::vmm::host_topology::PerfModeUnavailable {
                reason: "host too small for perf topology".to_string(),
            },
        ))
        .context("build ktstr_test VM")
    };
    assert_eq!(result_to_exit_code(wrapped(), false, false), EXIT_PASS);
}

/// `CpuBudgetUnsatisfiable` routes to EXIT_FAIL even under expect_err
/// (same arm-ordering invariant as the perf-mode case).
#[test]
fn result_to_exit_code_cpu_budget_unsatisfiable_fails_even_under_expect_err() {
    use anyhow::Context as _;
    let mk = || -> Result<crate::assert::AssertResult> {
        Err(anyhow::Error::new(
            crate::vmm::host_topology::CpuBudgetUnsatisfiable {
                reason: "--cpu-cap exceeds allowed CPUs".to_string(),
            },
        ))
        .context("build ktstr_test VM")
    };
    assert_eq!(result_to_exit_code(mk(), false, false), EXIT_FAIL);
    assert_eq!(result_to_exit_code(mk(), true, false), EXIT_FAIL);
}

/// `KernelUnavailable` on the dispatch exit-code path is a skip-class
/// host-insufficiency: `classify_host_error` maps it to `HostClass::Skip`
/// (default), which `err_to_exit_code` routes to EXIT_PASS — and because
/// the host-class match precedes the `expect_err` inversion, BOTH polarities
/// skip. Under nextest the plain `#[test]` wrapper is suppressed, so an
/// entry dispatches as `ktstr/{name}` via `run_named_test` ->
/// `err_to_exit_code`; this means a developer running `cargo nextest run`,
/// or `cargo ktstr test` without `--kernel`, on a kernel-less host gets a
/// clean skip rather than a hard fail on every entry. (A requested
/// `--kernel` that fails to build bails in cargo-ktstr before nextest
/// spawns, so this never masks a CI kernel-build failure.) The
/// `KTSTR_NO_SKIP_MODE` -> FAIL promotion is pinned by
/// `result_to_exit_code_skip_class_fails_under_no_skip_mode`.
#[test]
fn result_to_exit_code_kernel_unavailable_skips_on_dispatch_path() {
    use crate::test_support::test_helpers::{EnvVarGuard, lock_env};
    use anyhow::Context as _;
    let _env_lock = lock_env();
    let _no_skip = EnvVarGuard::remove(crate::KTSTR_NO_SKIP_MODE_ENV);
    let mk = || -> Result<crate::assert::AssertResult> {
        Err(anyhow::Error::new(crate::test_support::KernelUnavailable {
            diagnostic: "no kernel image resolved; run via `cargo ktstr test`".to_string(),
        }))
        .context("run ktstr_test VM")
    };
    // Skip-class: EXIT_PASS for both expect_err polarities — the host-class
    // skip arm precedes the expect_err inversion.
    assert_eq!(result_to_exit_code(mk(), false, false), EXIT_PASS);
    assert_eq!(result_to_exit_code(mk(), true, false), EXIT_PASS);
}

/// `TopologyUnrepresentable` routes to EXIT_FAIL via its DEDICATED
/// hard-fail arm (above the `expect_err` inversion) even under
/// expect_err — a fixed VMM-layout limit no host can satisfy is a test
/// misconfiguration, never "the expected error appeared". Without that
/// arm the generic `expect_err` arm would invert it to EXIT_PASS.
/// Crucially it is NOT a skip type: `is_topology_insufficient` must
/// reject it (chain-aware, even context-wrapped) so a too-wide aarch64
/// topology hard-fails rather than silently skipping. This is the
/// deliberate counterpart to the x86 over-`KVM_CAP_MAX_VCPUS` bail,
/// which IS a `TopologyInsufficient` skip (host-dependent cap).
#[test]
fn result_to_exit_code_topology_unrepresentable_fails_and_is_not_a_skip() {
    use anyhow::Context as _;
    let mk = || -> Result<crate::assert::AssertResult> {
        Err(anyhow::Error::new(
            crate::vmm::host_topology::TopologyUnrepresentable {
                reason: "topology has 513 vCPUs, exceeding the maximum of 512".to_string(),
            },
        ))
        .context("build ktstr_test VM")
    };
    assert_eq!(result_to_exit_code(mk(), false, false), EXIT_FAIL);
    assert_eq!(result_to_exit_code(mk(), true, false), EXIT_FAIL);
    // The dedicated is_topology_unrepresentable arm (above the skip
    // arms) is what routes it to FAIL; this assertion separately pins
    // type-distinctness — TopologyUnrepresentable is not a
    // TopologyInsufficient skip — so a future skip-matcher change can't
    // reclassify the type and silently turn the misconfiguration into a
    // pass.
    let wrapped: anyhow::Error =
        anyhow::Error::new(crate::vmm::host_topology::TopologyUnrepresentable {
            reason: "too wide".to_string(),
        })
        .context("build ktstr_test VM");
    assert!(
        !is_topology_insufficient(&wrapped),
        "TopologyUnrepresentable is a hard fault, not a topology skip",
    );
}

/// `allow_inconclusive = true` routes a terminal Inconclusive
/// verdict to EXIT_PASS at the dispatch layer, opting the test
/// out of the EXIT_INCONCLUSIVE projection. Used by tests that
/// have an explicit reason to accept "couldn't evaluate" as
/// not-a-failure (e.g. exploratory benchmarks under hosts
/// whose topology may legitimately starve the workload).
/// expect_err still dominates: an expect_err scenario whose
/// result is Inconclusive maps to EXIT_FAIL regardless of
/// allow_inconclusive — expect_err demands a real Fail.
#[test]
fn result_to_exit_code_allow_inconclusive_routes_to_pass() {
    use crate::assert::{AssertDetail, AssertResult, DetailKind};
    let inc = AssertResult::inconclusive(AssertDetail::new(
        DetailKind::Benchmark,
        "zero-denominator (allow_inconclusive=true)",
    ));
    // allow_inconclusive=true, expect_err=false: Inconclusive → 0.
    assert_eq!(result_to_exit_code(Ok(inc), false, true), 0);

    // Pass + allow_inconclusive=true: still Pass → 0 (no
    // behavior change on the Pass arm).
    assert_eq!(
        result_to_exit_code(Ok(AssertResult::pass()), false, true),
        0
    );

    // expect_err + Inconclusive + allow_inconclusive: still
    // → 1. The expect_err gate dominates; allow_inconclusive
    // only relaxes the EXIT_INCONCLUSIVE projection on the
    // no-expect_err path.
    let inc2 = AssertResult::inconclusive(AssertDetail::new(
        DetailKind::Benchmark,
        "zero-denominator under expect_err + allow_inconclusive",
    ));
    assert_eq!(result_to_exit_code(Ok(inc2), true, true), 1);
}

/// Multi-kernel suffix emission is suppressed in
/// cargo-test mode even when `KTSTR_KERNEL_LIST` is set —
/// the bare `cargo test` path doesn't drive the cargo-ktstr
/// resolver, so any `KTSTR_KERNEL_LIST` is a stale leftover
/// from a prior session and must not influence listing.
#[test]
fn list_tests_all_cargo_test_mode_ignores_kernel_list() {
    use crate::test_support::test_helpers::{EnvVarGuard, lock_env};
    let _env_lock = lock_env();
    let _cargo = EnvVarGuard::set(crate::KTSTR_CARGO_TEST_MODE_ENV, "1");
    let _kernel_list = EnvVarGuard::set(crate::KTSTR_KERNEL_LIST_ENV, TWO_KERNEL_LIST);
    let _budget_guard = EnvVarGuard::remove(crate::KTSTR_BUDGET_SECS_ENV);

    let (_, captured) = capture_stdout(|| list_tests_all(false));
    let stdout = std::str::from_utf8(&captured).expect("utf-8");
    assert!(
        !stdout.contains("kernel_6_14_2") && !stdout.contains("kernel_6_15_0"),
        "cargo-test-mode must suppress multi-kernel suffix emission \
         even when KTSTR_KERNEL_LIST is set; got stdout containing a \
         sanitized kernel label:\n{stdout}",
    );
}

// ---------------------------------------------------------------
// build_host_cgroup_manager — KTSTR_CGROUP_WALK_ROOT_ENV roundtrip
// ---------------------------------------------------------------

/// `build_host_cgroup_manager` threads a non-empty
/// `KTSTR_CGROUP_WALK_ROOT_ENV` value into
/// `CgroupManager::with_walk_root`. Pins the env-var → walk_root
/// wire-up against a refactor that drops the threading.
/// Requires root because the no-env-set fallback path bails for
/// non-root operators (delegated subtree without explicit walk_root
/// would EACCES downstream); skipped on non-root.
#[test]
fn build_host_cgroup_manager_threads_env_walk_root() {
    use crate::test_support::test_helpers::{EnvVarGuard, lock_env};
    let _env_lock = lock_env();
    let _g = EnvVarGuard::set(
        crate::KTSTR_CGROUP_WALK_ROOT_ENV,
        "/sys/fs/cgroup/delegated",
    );
    let cg = build_host_cgroup_manager("/sys/fs/cgroup/delegated/ktstr")
        .expect("env-set walk_root must thread into CgroupManager");
    assert_eq!(
        cg.walk_root(),
        std::path::Path::new("/sys/fs/cgroup/delegated"),
        "walk_root must match the env value verbatim",
    );
}

/// Empty `KTSTR_CGROUP_WALK_ROOT_ENV` is observationally
/// identical to unset (the empty-string-equals-unset contract).
/// Pins the
/// `Ok(walk_root) if !walk_root.is_empty()` gate at the
/// build_host_cgroup_manager match arm.
/// Root-gated: the unset arm bails for non-root operators.
#[test]
fn build_host_cgroup_manager_empty_env_treated_as_unset() {
    if unsafe { libc::geteuid() } != 0 {
        return;
    }
    use crate::test_support::test_helpers::{EnvVarGuard, lock_env};
    let _env_lock = lock_env();
    let _g = EnvVarGuard::set(crate::KTSTR_CGROUP_WALK_ROOT_ENV, "");
    let cg = build_host_cgroup_manager("/sys/fs/cgroup/ktstr")
        .expect("empty env must fall through to default (root-only)");
    assert_eq!(
        cg.walk_root(),
        std::path::Path::new("/sys/fs/cgroup"),
        "empty env must select the canonical-root default",
    );
}

/// Unset `KTSTR_CGROUP_WALK_ROOT_ENV` selects the canonical-root
/// default `/sys/fs/cgroup` (Mode A).
/// Root-gated: see sibling test for the rationale.
#[test]
fn build_host_cgroup_manager_unset_env_uses_default() {
    if unsafe { libc::geteuid() } != 0 {
        return;
    }
    use crate::test_support::test_helpers::{EnvVarGuard, lock_env};
    let _env_lock = lock_env();
    let _g = EnvVarGuard::remove(crate::KTSTR_CGROUP_WALK_ROOT_ENV);
    let cg = build_host_cgroup_manager("/sys/fs/cgroup/ktstr")
        .expect("unset env must fall through to default (root-only)");
    assert_eq!(
        cg.walk_root(),
        std::path::Path::new("/sys/fs/cgroup"),
        "unset env must select the canonical-root default",
    );
}

/// With `KTSTR_CGROUP_WALK_ROOT_ENV` unset, `build_host_cgroup_manager`
/// returns the manager unconditionally (NOT root-gated): the non-root
/// precondition moved to a lazy check at `CgroupManager::setup` (first
/// real cgroup use), so host_only tests that never create a cgroup run
/// without root. The lazy euid gate itself is locked in by
/// `CgroupManager::default_root_requires_root`'s truth table
/// (cgroup_tests.rs); this pins that CONSTRUCTION no longer bails for a
/// non-root caller (the prior eager bail at this layer was removed).
#[test]
fn build_host_cgroup_manager_unset_env_defers_non_root_check_to_setup() {
    use crate::test_support::test_helpers::{EnvVarGuard, lock_env};
    let _env_lock = lock_env();
    let _g = EnvVarGuard::remove(crate::KTSTR_CGROUP_WALK_ROOT_ENV);
    // No euid gate: construction must succeed regardless of euid (the
    // prior eager non-root bail at this layer was deleted).
    let cg = build_host_cgroup_manager("/sys/fs/cgroup/ktstr")
        .expect("build defers the non-root check to setup; construction must succeed");
    assert_eq!(
        cg.walk_root(),
        std::path::Path::new("/sys/fs/cgroup"),
        "unset env selects the canonical default walk root",
    );
}

/// `KTSTR_CGROUP_WALK_ROOT_ENV` values outside `/sys/fs/cgroup`
/// MUST bail upfront (defensive guard mirroring the
/// sibling `KTSTR_HOST_CGROUP_PARENT_ENV` check). Catches
/// operator typos like `/tmp/foo` at config-validation time
/// instead of as a downstream fs::write EACCES.
#[test]
fn build_host_cgroup_manager_env_outside_cgroupfs_bails() {
    use crate::test_support::test_helpers::{EnvVarGuard, lock_env};
    let _env_lock = lock_env();
    let _g = EnvVarGuard::set(crate::KTSTR_CGROUP_WALK_ROOT_ENV, "/tmp/foo");
    let err = build_host_cgroup_manager("/tmp/foo/ktstr")
        .expect_err("walk_root outside /sys/fs/cgroup MUST bail");
    let msg = format!("{err:#}");
    assert!(
        msg.contains("/sys/fs/cgroup") && msg.contains("KTSTR_CGROUP_WALK_ROOT"),
        "bail message must name both the required prefix and the \
         env var so the operator can fix the config; got: {msg}",
    );
}

/// `resolve_host_cgroup_parent` with `KTSTR_HOST_CGROUP_PARENT`
/// unset falls back to `DEFAULT_HOST_CGROUP_PARENT`. Pure env-read
/// and string logic (no fs, no syscall, no root), so unlike the
/// `build_host_cgroup_manager_*` tests above these need no geteuid
/// gate. Converted from the former host_mode_e2e.rs host_only test
/// (which ignored ctx and only exercised this pure cascade — a VM
/// boot for nothing).
#[test]
fn resolve_host_cgroup_parent_env_unset_returns_default() {
    use crate::test_support::test_helpers::{EnvVarGuard, lock_env};
    let _env_lock = lock_env();
    let _g = EnvVarGuard::remove(crate::KTSTR_HOST_CGROUP_PARENT_ENV);
    let resolved = resolve_host_cgroup_parent().expect("unset env resolves to default");
    assert_eq!(resolved, DEFAULT_HOST_CGROUP_PARENT);
}

/// Empty `KTSTR_HOST_CGROUP_PARENT` is treated as unset (the
/// `Ok(s) if !s.is_empty()` gate) and falls back to the default.
#[test]
fn resolve_host_cgroup_parent_env_empty_returns_default() {
    use crate::test_support::test_helpers::{EnvVarGuard, lock_env};
    let _env_lock = lock_env();
    let _g = EnvVarGuard::set(crate::KTSTR_HOST_CGROUP_PARENT_ENV, "");
    let resolved = resolve_host_cgroup_parent().expect("empty env resolves to default");
    assert_eq!(resolved, DEFAULT_HOST_CGROUP_PARENT);
}

/// A valid override rooted under `/sys/fs/cgroup` (and naming a
/// subdirectory) is returned verbatim.
#[test]
fn resolve_host_cgroup_parent_env_override_returns_value() {
    use crate::test_support::test_helpers::{EnvVarGuard, lock_env};
    let _env_lock = lock_env();
    let _g = EnvVarGuard::set(
        crate::KTSTR_HOST_CGROUP_PARENT_ENV,
        "/sys/fs/cgroup/ktstr-foo",
    );
    let resolved = resolve_host_cgroup_parent().expect("valid override resolves");
    assert_eq!(resolved, "/sys/fs/cgroup/ktstr-foo");
}

/// Both invalid-override branches bail: a path outside
/// `/sys/fs/cgroup`, and the bare mount root itself (which is not
/// a usable parent — it needs a subdirectory). The message names
/// the required prefix and the default fallback.
#[test]
fn resolve_host_cgroup_parent_env_invalid_bails() {
    use crate::test_support::test_helpers::{EnvVarGuard, lock_env};
    let _env_lock = lock_env();
    {
        let _g = EnvVarGuard::set(crate::KTSTR_HOST_CGROUP_PARENT_ENV, "/tmp/foo");
        let err =
            resolve_host_cgroup_parent().expect_err("override outside /sys/fs/cgroup must bail");
        let msg = format!("{err:#}");
        assert!(
            msg.contains("/sys/fs/cgroup") && msg.contains(DEFAULT_HOST_CGROUP_PARENT),
            "bail message must name the required prefix and the default \
             fallback so the operator can fix the config; got: {msg}",
        );
    }
    {
        let _g = EnvVarGuard::set(crate::KTSTR_HOST_CGROUP_PARENT_ENV, "/sys/fs/cgroup");
        resolve_host_cgroup_parent()
            .expect_err("the bare /sys/fs/cgroup mount root must bail (needs a subdir)");
    }
}

/// Value-drift canary: `DEFAULT_HOST_CGROUP_PARENT`'s literal must
/// stay `/sys/fs/cgroup/ktstr`. Catches an asymmetric change to
/// the const value (vs its name) that a rename-only refactor
/// wouldn't. Mirrors the LITERAL_DEFAULT canary the host_mode e2e
/// test carried before it was converted to these unit tests.
#[test]
fn default_host_cgroup_parent_literal_pin() {
    assert_eq!(DEFAULT_HOST_CGROUP_PARENT, "/sys/fs/cgroup/ktstr");
}

// -- result_to_exit_code: ExpectAutoReproSatisfied marker tests --
//
// Pin the dispatch-side verdict flip that consumes the marker
// wrapped onto the failure Err by run_ktstr_test_inner_impl
// when apply_expect_auto_repro_inversion signaled satisfaction.
// The eval-side helper's gates are exercised separately at the
// `crate::test_support::eval` tests module; these pin the dispatch-arm match-order
// contract.

/// Err with [`ExpectAutoReproSatisfied`] attached directly →
/// the dispatch arm downcasts and routes to [`EXIT_PASS`]. The
/// canonical happy-path: helper set the marker, no other arm
/// matches first, verdict flips.
#[test]
fn result_to_exit_code_expect_auto_repro_satisfied_routes_to_pass() {
    let err: Result<crate::assert::AssertResult> = Err(anyhow::anyhow!("primary VM failure")
        .context(crate::test_support::eval::ExpectAutoReproSatisfied));
    assert_eq!(
        result_to_exit_code(err, false, false),
        EXIT_PASS,
        "ExpectAutoReproSatisfied marker must route Err → EXIT_PASS"
    );
}

/// Err with the marker attached by `.context()` and then buried under
/// more `.context()` layers → anyhow's context-aware `downcast_ref`
/// still surfaces it. Pins the
/// `e.downcast_ref::<ExpectAutoReproSatisfied>().is_some()` contract:
/// `downcast_ref` walks the full context chain, so a nested marker is
/// found. A regression that swapped it for a `chain().any(|c| c.is::<_>())`
/// walk would MISS the marker — anyhow boxes a context-attached value as
/// `ContextError<C, E>`, which `is::<ExpectAutoReproSatisfied>()` never
/// matches (the anyhow-chain-walk footgun). The test passing proves
/// production takes the context-aware path. The eval layer's surrounding
/// context wrappers (`"build ktstr_test VM"`, `"run ktstr_test VM"`) make
/// nested-marker chains the production-typical shape.
#[test]
fn result_to_exit_code_expect_auto_repro_satisfied_works_through_nested_context() {
    let err: Result<crate::assert::AssertResult> = Err(anyhow::anyhow!("primary VM failure")
        .context(crate::test_support::eval::ExpectAutoReproSatisfied)
        .context("run ktstr_test VM")
        .context("dispatch wrapper"));
    assert_eq!(
        result_to_exit_code(err, false, false),
        EXIT_PASS,
        "nested ExpectAutoReproSatisfied must still be found by the context-aware downcast_ref"
    );
}

/// Baseline control: Err WITHOUT the marker → [`EXIT_FAIL`].
/// Guards against an arm regression that routes every Err to
/// EXIT_PASS — the marker's presence MUST be load-bearing for
/// the verdict flip. Pairs with the direct-marker test above
/// so a one-arm-too-broad regression fails this baseline
/// instead of silently green-lighting every failure.
#[test]
fn result_to_exit_code_plain_err_without_marker_routes_to_fail() {
    let err: Result<crate::assert::AssertResult> = Err(anyhow::anyhow!(
        "primary VM failure without inversion marker"
    ));
    assert_eq!(
        result_to_exit_code(err, false, false),
        EXIT_FAIL,
        "Err without ExpectAutoReproSatisfied must route to EXIT_FAIL"
    );
}

/// `expect_err = true` + both markers attached → the
/// ExpectAutoReproSatisfied arm wins because it is positioned
/// BEFORE the expect_err arm in the dispatch match. The two
/// inversion paths are mutually exclusive at macro-parse (the
/// cross-attr check in ktstr_test rejects the combination), so
/// this pin guards the order invariant rather than a
/// runtime-reachable combo: a programmatic-construction path
/// that bypassed the macro MUST still route consistently. The
/// scenario also attaches ScxBpfErrorMatcherMismatch so the
/// alternative arm has a distinguishing outcome
/// (EXIT_FAIL via the matcher-mismatch branch) — without the
/// second marker, both arms would converge on EXIT_PASS and the
/// precedence claim would not be falsifiable.
#[test]
fn result_to_exit_code_marker_arm_wins_over_expect_err_arm() {
    let err: Result<crate::assert::AssertResult> = Err(anyhow::anyhow!("primary VM failure")
        .context(crate::test_support::eval::ScxBpfErrorMatcherMismatch)
        .context(crate::test_support::eval::ExpectAutoReproSatisfied));
    assert_eq!(
        result_to_exit_code(err, true, false),
        EXIT_PASS,
        "marker arm must win over expect_err arm by match-order positioning \
         (expect_err arm with ScxBpfErrorMatcherMismatch would return EXIT_FAIL)"
    );
}

// -- result_to_exit_code: SurvivesStormViolated marker tests --
//
// Pin the dispatch-side EXIT_FAIL the survives_storm assertion produces.
// The marker is attached by render_failure_verdict_message ONLY when
// entry.survives_storm was set AND the failure carried a scheduler-death
// DetailKind, so its presence alone is load-bearing (no flag param,
// mirroring the other marker arms). The eval-side attach gate is exercised
// in the eval tests; these pin the dispatch-arm contract + match order.

/// Err with [`SurvivesStormViolated`] attached → the dispatch arm
/// downcasts and forces [`EXIT_FAIL`].
#[test]
fn result_to_exit_code_survives_storm_violated_routes_to_fail() {
    let err: Result<crate::assert::AssertResult> =
        Err(anyhow::anyhow!("scheduler died during hold")
            .context(crate::test_support::eval::SurvivesStormViolated));
    assert_eq!(
        result_to_exit_code(err, false, false),
        EXIT_FAIL,
        "SurvivesStormViolated marker must route Err → EXIT_FAIL"
    );
}

/// Marker buried under more `.context()` layers → the context-aware
/// `downcast_ref` still surfaces it (the anyhow-chain-walk footgun: a
/// `chain().any(|c| c.is::<_>())` walk would miss the ContextError-boxed
/// marker). The eval layer's surrounding context wrappers make
/// nested-marker chains the production-typical shape.
#[test]
fn result_to_exit_code_survives_storm_violated_through_nested_context() {
    let err: Result<crate::assert::AssertResult> =
        Err(anyhow::anyhow!("scheduler died during hold")
            .context(crate::test_support::eval::SurvivesStormViolated)
            .context("run ktstr_test VM")
            .context("dispatch wrapper"));
    assert_eq!(
        result_to_exit_code(err, false, false),
        EXIT_FAIL,
        "nested SurvivesStormViolated must still be found by the context-aware downcast_ref"
    );
}

/// `expect_err = true` + SurvivesStormViolated → the survives_storm arm
/// wins because it is positioned BEFORE the expect_err inversion arm in the
/// dispatch match. The two are mutually exclusive at validate
/// (survives_storm ⊕ expect_err), so this guards the order invariant for a
/// programmatic path that bypassed the macro: a survival violation must
/// never invert to PASS via expect_err. Falsifiable — without the
/// survives_storm arm, the expect_err arm would route this Err to
/// EXIT_PASS.
#[test]
fn result_to_exit_code_survives_storm_violated_wins_over_expect_err() {
    let err: Result<crate::assert::AssertResult> =
        Err(anyhow::anyhow!("scheduler died during hold")
            .context(crate::test_support::eval::SurvivesStormViolated));
    assert_eq!(
        result_to_exit_code(err, true, false),
        EXIT_FAIL,
        "SurvivesStormViolated must win over the expect_err inversion (positioned before it)"
    );
}

/// `SurvivesStormViolated` + `ExpectAutoReproSatisfied` both attached →
/// the survives_storm arm wins (positioned BEFORE the ExpectAutoReproSatisfied
/// arm), so the survival violation resolves to EXIT_FAIL and is NOT inverted
/// to PASS. The two are mutually exclusive at validate (survives_storm ⊕
/// expect_auto_repro), so this guards the order invariant for a programmatic
/// path that bypassed the macro. Falsifiable — without the survives_storm arm
/// (or if mispositioned after it), the ExpectAutoReproSatisfied arm would
/// route this Err to EXIT_PASS.
#[test]
fn result_to_exit_code_survives_storm_violated_wins_over_expect_auto_repro() {
    let err: Result<crate::assert::AssertResult> =
        Err(anyhow::anyhow!("scheduler died during hold")
            .context(crate::test_support::eval::ExpectAutoReproSatisfied)
            .context(crate::test_support::eval::SurvivesStormViolated));
    assert_eq!(
        result_to_exit_code(err, false, false),
        EXIT_FAIL,
        "SurvivesStormViolated must win over ExpectAutoReproSatisfied (positioned before it)"
    );
}

// -- result_to_exit_code: ScxBpfErrorMatcherMismatch (expect_err
// arm) tests --
//
// Pin the two-way verdict the expect_err arm produces depending
// on whether the scx_bpf_error matcher mismatch marker is
// attached. Until this batch, no test exercised the marker
// detection path; the chain walk shape was untested at the
// dispatch layer and only protected by the `crate::test_support::eval` structural
// source-pin.

/// `expect_err = true` + Err WITHOUT a matcher-mismatch marker
/// → the expect_err arm inverts to [`EXIT_PASS`]. The canonical
/// `expected error happened` path.
#[test]
fn result_to_exit_code_expect_err_without_matcher_marker_routes_to_pass() {
    let err: Result<crate::assert::AssertResult> =
        Err(anyhow::anyhow!("primary VM failure (no matcher mismatch)"));
    assert_eq!(
        result_to_exit_code(err, true, false),
        EXIT_PASS,
        "expect_err arm must invert plain Err to EXIT_PASS"
    );
}

/// `expect_err = true` + Err WITH the matcher-mismatch marker
/// → the expect_err arm REFUSES to invert and returns
/// [`EXIT_FAIL`]. The reproducer's matcher narrowed which bug
/// counts; a different bug firing must surface as a regression
/// even though `expect_err = true` would normally invert.
#[test]
fn result_to_exit_code_expect_err_with_matcher_mismatch_routes_to_fail() {
    let err: Result<crate::assert::AssertResult> =
        Err(anyhow::anyhow!("primary VM failure (matcher mismatch)")
            .context(crate::test_support::eval::ScxBpfErrorMatcherMismatch));
    assert_eq!(
        result_to_exit_code(err, true, false),
        EXIT_FAIL,
        "expect_err arm must refuse inversion when ScxBpfErrorMatcherMismatch is attached"
    );
}

/// `expect_err = true` + Err with the matcher-mismatch marker
/// nested INSIDE additional `.context()` wrapping → the
/// expect_err arm STILL refuses to invert because anyhow's
/// `downcast_ref` walks the context+source chain. Sibling pin
/// to `result_to_exit_code_expect_auto_repro_satisfied_works_through_nested_context`
/// — the eval-layer wrappers (`"build ktstr_test VM"`,
/// `"run ktstr_test VM"`) routinely wrap the inner Err with
/// additional context, so nested-marker chains are the
/// production-typical shape. A regression that swapped
/// `downcast_ref` for a top-level downcast would only match
/// the outermost context wrapper — passing the unnested
/// sibling test above but silently inverting nested-marker
/// regressions in production.
#[test]
fn result_to_exit_code_matcher_mismatch_through_nested_context_routes_to_fail() {
    let err: Result<crate::assert::AssertResult> = Err(anyhow::anyhow!("primary VM failure")
        .context(crate::test_support::eval::ScxBpfErrorMatcherMismatch)
        .context("run ktstr_test VM")
        .context("dispatch wrapper"));
    assert_eq!(
        result_to_exit_code(err, true, false),
        EXIT_FAIL,
        "nested ScxBpfErrorMatcherMismatch must still be downcast by anyhow's context-aware downcast_ref"
    );
}

// -- result_to_exit_code: SchedulerBuildRefused (expect_err arm) tests --
//
// A failed orchestrated scheduler build the resolver refused to serve stale
// carries the SchedulerBuildRefused marker; like the other host-side
// hard-fail markers it must force EXIT_FAIL even under expect_err — a broken
// build must never masquerade as the expect_err test's expected failure.

/// `expect_err = true` + Err WITH the SchedulerBuildRefused marker → the
/// dispatch guard forces [`EXIT_FAIL`], refusing the expect_err inversion.
/// Without the marker (the bug this fixes) a build refusal would be a plain
/// Err and the expect_err arm would invert it to a false PASS.
#[test]
fn result_to_exit_code_expect_err_with_scheduler_build_refused_routes_to_fail() {
    let err: Result<crate::assert::AssertResult> =
        Err(anyhow::anyhow!("cargo build -p scx_x failed")
            .context(crate::test_support::eval::SchedulerBuildRefused));
    assert_eq!(
        result_to_exit_code(err, true, false),
        EXIT_FAIL,
        "expect_err must NOT invert a SchedulerBuildRefused failure to PASS"
    );
}

/// Production-typical shape: the resolver attaches the marker INNER then wraps
/// the operator-facing refusal message OUTER
/// (`e.context(SchedulerBuildRefused).context("... refusing ...")`), so the
/// marker is nested. anyhow's context-aware `downcast_ref` still finds it →
/// [`EXIT_FAIL`] under expect_err.
#[test]
fn result_to_exit_code_scheduler_build_refused_through_nested_context_routes_to_fail() {
    let err: Result<crate::assert::AssertResult> =
        Err(anyhow::anyhow!("cargo build -p scx_x failed")
            .context(crate::test_support::eval::SchedulerBuildRefused)
            .context("ktstr_test: refusing to validate against a stale binary")
            .context("dispatch wrapper"));
    assert_eq!(
        result_to_exit_code(err, true, false),
        EXIT_FAIL,
        "nested SchedulerBuildRefused must still be downcast and force EXIT_FAIL"
    );
}

// -- result_to_exit_code: PostVmAssertionFailure marker tests --
//
// Pin the dispatch-side verdict the PostVmAssertionFailure arm
// produces: a host-side post_vm callback failure is honored even
// under expect_err (expect_err inverts a GUEST-side expected
// failure, not a HOST-side assertion), and wins over the
// ExpectAutoReproSatisfied arm by match-order positioning. The
// marker is wrapped onto the failure Err by run_ktstr_test_inner_impl
// when post_vm_err is Some.

/// `expect_err = true` + Err WITH the PostVmAssertionFailure marker
/// → the dispatch arm refuses to invert and returns [`EXIT_FAIL`].
/// The anti-hollow-test guarantee: a failure-dump render test
/// triggers an expected stall (which expect_err would invert to
/// PASS) and asserts the dump in post_vm — a wrong render MUST fail
/// the test, not silently invert.
#[test]
fn result_to_exit_code_post_vm_assertion_failure_under_expect_err_routes_to_fail() {
    let err: Result<crate::assert::AssertResult> = Err(anyhow::anyhow!(
        "post_vm callback returned Err: dump render wrong"
    )
    .context(crate::test_support::eval::PostVmAssertionFailure));
    assert_eq!(
        result_to_exit_code(err, true, false),
        EXIT_FAIL,
        "PostVmAssertionFailure marker must refuse expect_err inversion → EXIT_FAIL"
    );
}

/// Baseline control: `expect_err = true` + Err WITHOUT the marker
/// → the expect_err arm inverts to [`EXIT_PASS`]. Pairs with the
/// test above so the marker's presence is load-bearing (a
/// one-arm-too-broad regression that failed every expect_err Err
/// would fail this baseline).
#[test]
fn result_to_exit_code_expect_err_without_post_vm_marker_routes_to_pass() {
    let err: Result<crate::assert::AssertResult> = Err(anyhow::anyhow!(
        "expected guest-side stall, no host-side check"
    ));
    assert_eq!(
        result_to_exit_code(err, true, false),
        EXIT_PASS,
        "expect_err arm must invert a plain Err (no PostVmAssertionFailure) to EXIT_PASS"
    );
}

/// PostVmAssertionFailure nested INSIDE additional `.context()`
/// wrapping → the dispatch arm STILL refuses to invert because
/// anyhow's `downcast_ref` walks the context+source chain. The
/// eval-layer wrappers (`"build ktstr_test VM"`, `"run ktstr_test
/// VM"`) routinely wrap the inner Err, so nested-marker chains are
/// the production-typical shape.
#[test]
fn result_to_exit_code_post_vm_marker_through_nested_context_routes_to_fail() {
    let err: Result<crate::assert::AssertResult> = Err(anyhow::anyhow!(
        "post_vm callback returned Err: dump render wrong"
    )
    .context(crate::test_support::eval::PostVmAssertionFailure)
    .context("run ktstr_test VM")
    .context("dispatch wrapper"));
    assert_eq!(
        result_to_exit_code(err, true, false),
        EXIT_FAIL,
        "nested PostVmAssertionFailure must still be downcast by anyhow's context-aware downcast_ref"
    );
}

/// Both PostVmAssertionFailure AND ExpectAutoReproSatisfied
/// attached → the PostVmAssertionFailure arm wins (EXIT_FAIL)
/// because it is positioned BEFORE the ExpectAutoReproSatisfied arm
/// in the dispatch match. A real host-side regression must override
/// an auto-repro satisfaction inversion: the dump assertion failing
/// is a regression regardless of whether the repro artifact landed.
/// Without the PostVmAssertionFailure marker this same Err would
/// route to EXIT_PASS via the ExpectAutoReproSatisfied arm, so the
/// precedence claim is falsifiable.
#[test]
fn result_to_exit_code_post_vm_marker_wins_over_expect_auto_repro() {
    let err: Result<crate::assert::AssertResult> = Err(anyhow::anyhow!("primary VM failure")
        .context(crate::test_support::eval::ExpectAutoReproSatisfied)
        .context(crate::test_support::eval::PostVmAssertionFailure));
    assert_eq!(
        result_to_exit_code(err, false, false),
        EXIT_FAIL,
        "PostVmAssertionFailure arm must win over ExpectAutoReproSatisfied by match-order \
         positioning (ExpectAutoReproSatisfied alone would return EXIT_PASS)"
    );
}

// -- result_to_exit_code: ResourceContention + TopologyInsufficient
// arms --
//
// Both typed errors route through the shared `classify_host_error`
// (the `HostClass::Skip` / `Fail` mapping), surfaced via
// `result_to_exit_code` → `err_to_exit_code`.
// TopologyInsufficient: with KTSTR_NO_SKIP_MODE unset it maps to
// EXIT_PASS (the test never ran — `crate::report::test_skip` is a bare
// eprintln!, no VM needed); with KTSTR_NO_SKIP_MODE set it hard-FAILs
// (EXIT_FAIL). ResourceContention: ALWAYS EXIT_FAIL — the run path
// waits the holder out, so residual contention is a retryable failure
// nextest re-runs, never a skip (EXIT_PASS is un-retried and would
// silently drop coverage). The host-class arms sit ABOVE the
// `Err(e) if expect_err` arm, so even under expect_err=true these
// verdicts are not inverted. NOTE: the 3rd `result_to_exit_code`
// param is `allow_inconclusive`, NOT no_skip — no_skip is read from
// KTSTR_NO_SKIP_MODE_ENV inside the fn (its
// `let no_skip = std::env::var_os(crate::KTSTR_NO_SKIP_MODE_ENV).is_some();`
// binding).

/// `ResourceContention` (direct and `.context`-wrapped) routes to
/// EXIT_FAIL (retryable — nextest re-runs) even when
/// KTSTR_NO_SKIP_MODE is unset, and even under expect_err=true (the
/// host-class arm precedes the expect_err inversion). The wrapped
/// case exercises the reason extraction + the
/// `is_resource_contention` chain walk through the eval-layer
/// "build/run ktstr_test VM" wrappers. The banner must frame the
/// failure as transient contention, never host incapacity.
#[test]
fn result_to_exit_code_resource_contention_fails_retryably() {
    use crate::test_support::test_helpers::{EnvVarGuard, capture_stderr, lock_env};
    use anyhow::Context as _;
    let _env_lock = lock_env();
    let _no_skip = EnvVarGuard::remove(crate::KTSTR_NO_SKIP_MODE_ENV);

    let direct = || -> Result<crate::assert::AssertResult> {
        Err(anyhow::Error::new(
            crate::vmm::host_topology::ResourceContention {
                reason: "no 4 consecutive CPUs available".into(),
            },
        ))
    };
    let (code, captured) = capture_stderr(|| result_to_exit_code(direct(), false, false));
    assert_eq!(
        code, EXIT_FAIL,
        "contention must FAIL (retryable), not skip"
    );
    let stderr = String::from_utf8(captured).expect("stderr is utf-8");
    assert!(
        stderr.contains("resource contention")
            && stderr.contains("transient")
            && stderr.contains("no 4 consecutive CPUs available"),
        "banner must name transient contention + the typed reason; got: {stderr}",
    );
    assert!(
        !stderr.contains("cannot run"),
        "banner must never claim host incapacity for contention; got: {stderr}",
    );
    // Host-class arm precedes the expect_err arm: still EXIT_FAIL,
    // never inverted to a pass.
    assert_eq!(result_to_exit_code(direct(), true, false), EXIT_FAIL);

    // Context-wrapped (production shape) still recognised via the
    // chain walk → EXIT_FAIL.
    let wrapped = || -> Result<crate::assert::AssertResult> {
        Err(anyhow::Error::new(
            crate::vmm::host_topology::ResourceContention {
                reason: "no 4 consecutive CPUs available".into(),
            },
        ))
        .context("build ktstr_test VM")
        .context("run ktstr_test VM")
    };
    assert_eq!(result_to_exit_code(wrapped(), false, false), EXIT_FAIL);
}

/// `TopologyInsufficient` (direct and `.context`-wrapped) routes to
/// EXIT_PASS when KTSTR_NO_SKIP_MODE is unset — including under
/// expect_err=true. Mirror of the ResourceContention skip test;
/// pins the topology-insufficient skip via the shared `classify_host_error`
/// (`HostClass::Skip`), surfaced through `result_to_exit_code`.
#[test]
fn result_to_exit_code_topology_insufficient_skips_when_not_no_skip() {
    use crate::test_support::test_helpers::{EnvVarGuard, lock_env};
    use anyhow::Context as _;
    let _env_lock = lock_env();
    let _no_skip = EnvVarGuard::remove(crate::KTSTR_NO_SKIP_MODE_ENV);

    let direct = || -> Result<crate::assert::AssertResult> {
        Err(anyhow::Error::new(
            crate::vmm::host_topology::TopologyInsufficient {
                reason: "host has too few CPUs".into(),
            },
        ))
    };
    assert_eq!(result_to_exit_code(direct(), false, false), EXIT_PASS);
    assert_eq!(result_to_exit_code(direct(), true, false), EXIT_PASS);

    let wrapped = || -> Result<crate::assert::AssertResult> {
        Err(anyhow::Error::new(
            crate::vmm::host_topology::TopologyInsufficient {
                reason: "host has too few CPUs".into(),
            },
        ))
        .context("build ktstr_test VM")
        .context("run ktstr_test VM")
    };
    assert_eq!(result_to_exit_code(wrapped(), false, false), EXIT_PASS);
}

/// Under KTSTR_NO_SKIP_MODE, the skip-class errors hard-FAIL
/// (EXIT_FAIL) instead of skipping, and the stderr banner names
/// the cause/reason. Pins the no_skip branches (the `no_skip` →
/// `HostClass::Fail` arms in the shared `classify_host_error` for
/// kernel-unavailable, perf-mode, and topology-insufficient, surfaced
/// via `result_to_exit_code`) and the reason extraction.
/// ResourceContention also FAILs here, but not via a no_skip
/// promotion — it is unconditionally a retryable failure (see
/// `result_to_exit_code_resource_contention_fails_retryably`); this
/// test pins that no_skip does not change that verdict.
#[test]
fn result_to_exit_code_skip_class_fails_under_no_skip_mode() {
    use crate::test_support::test_helpers::{EnvVarGuard, capture_stderr, lock_env};
    let _env_lock = lock_env();
    let _no_skip = EnvVarGuard::set(crate::KTSTR_NO_SKIP_MODE_ENV, "1");

    let rc = || -> Result<crate::assert::AssertResult> {
        Err(anyhow::Error::new(
            crate::vmm::host_topology::ResourceContention {
                reason: "no 4 consecutive CPUs available".into(),
            },
        ))
    };
    let (code, captured) = capture_stderr(|| result_to_exit_code(rc(), false, false));
    assert_eq!(code, EXIT_FAIL);
    let stderr = String::from_utf8(captured).expect("stderr is utf-8");
    assert!(
        stderr.contains("resource contention")
            && stderr.contains("no 4 consecutive CPUs available"),
        "no-skip RC banner must name the cause + reason; got: {stderr}",
    );

    let ti = || -> Result<crate::assert::AssertResult> {
        Err(anyhow::Error::new(
            crate::vmm::host_topology::TopologyInsufficient {
                reason: "host has too few CPUs".into(),
            },
        ))
    };
    let (code, captured) = capture_stderr(|| result_to_exit_code(ti(), false, false));
    assert_eq!(code, EXIT_FAIL);
    let stderr = String::from_utf8(captured).expect("stderr is utf-8");
    assert!(
        stderr.contains("host topology insufficient under --no-skip-mode")
            && stderr.contains("host has too few CPUs"),
        "no-skip TI banner must name the cause + reason; got: {stderr}",
    );

    let perf = || -> Result<crate::assert::AssertResult> {
        Err(anyhow::Error::new(
            crate::vmm::host_topology::PerfModeUnavailable {
                reason: "host too small for perf topology".into(),
            },
        ))
    };
    let (code, captured) = capture_stderr(|| result_to_exit_code(perf(), false, false));
    assert_eq!(code, EXIT_FAIL);
    let stderr = String::from_utf8(captured).expect("stderr is utf-8");
    assert!(
        stderr.contains("performance mode unavailable under --no-skip-mode")
            && stderr.contains("host too small for perf topology"),
        "no-skip perf-mode banner must name the cause + reason; got: {stderr}",
    );

    let kernel = || -> Result<crate::assert::AssertResult> {
        Err(anyhow::Error::new(crate::test_support::KernelUnavailable {
            diagnostic: "no kernel image resolved".into(),
        }))
    };
    let (code, captured) = capture_stderr(|| result_to_exit_code(kernel(), false, false));
    assert_eq!(code, EXIT_FAIL);
    let stderr = String::from_utf8(captured).expect("stderr is utf-8");
    assert!(
        stderr.contains("harness not configured under --no-skip-mode")
            && stderr.contains("no kernel image resolved"),
        "no-skip kernel-unavailable banner must name the cause + reason; got: {stderr}",
    );
}

// ---------------------------------------------------------------
// analyze_sidecars — host-side sidecar collection + render
// ---------------------------------------------------------------

/// An empty / sidecar-free directory short-circuits to the empty
/// string. `collect_sidecars` on a directory with no
/// `*.ktstr.json` files yields an empty Vec (the read_dir is Ok
/// but the iterator yields nothing for an existing empty dir, so
/// `sidecars.is_empty()` is true), exercising
/// `analyze_sidecars`'s `if sidecars.is_empty() { return String::new(); }`
/// branch — pinned via exact-empty equality, not just
/// `.is_empty()`.
#[test]
fn analyze_sidecars_empty_dir_returns_empty_string() {
    let d = tempfile::tempdir().expect("create tempdir");
    assert_eq!(analyze_sidecars(Some(d.path())), "");
}

/// One synthetic `*.ktstr.json` sidecar (the canonical
/// `SidecarResult::test_fixture`, whose verifier_stats=[],
/// monitor=None, kvm_stats=None) renders EXACTLY
/// `analyze_rows(&[sidecar_to_row(loaded)])` — the verifier /
/// callback / kvm sub-sections all collapse to empty so the
/// concatenation reduces to the rows render alone. Pins the
/// collect → row → analyze_rows path plus the four-section
/// concatenation order (only the rows section emits here).
#[test]
fn analyze_sidecars_single_fixture_renders_rows_only() {
    use crate::test_support::SidecarResult;
    let d = tempfile::tempdir().expect("create tempdir");
    let fixture = SidecarResult::test_fixture();
    let json = serde_json::to_string(&fixture).expect("fixture serializes");
    // Filename must satisfy is_sidecar_filename: `*.ktstr.json`
    // with `.ktstr.` in the file-name component.
    std::fs::write(d.path().join("t-0001.ktstr.json"), json).expect("write sidecar");

    let out = analyze_sidecars(Some(d.path()));
    assert!(!out.is_empty(), "one valid sidecar must render non-empty");

    // Exact equality against the rows-only render computed from
    // the SAME deserialized value collect_sidecars produces. The
    // verifier/callback/kvm formatters return empty for the
    // fixture, so analyze_sidecars == analyze_rows(&[row]).
    let collected = crate::test_support::collect_sidecars(d.path());
    assert_eq!(collected.len(), 1, "exactly one sidecar must be collected");
    let row = crate::stats::sidecar_to_row(&collected[0]);
    assert_eq!(
        out,
        crate::stats::analyze_rows(std::slice::from_ref(&row)),
        "analyze_sidecars must equal analyze_rows of the single row \
         (verifier/callback/kvm sections empty for the fixture)",
    );

    // Pin a concrete fixture-derived substring (the scenario name
    // "t" rendered in the `By scenario` pane) and the section
    // headers, so a regression that dropped the rows render or
    // emitted a spurious verifier/callback/kvm section surfaces.
    assert!(
        out.contains("=== GAUNTLET ANALYSIS ==="),
        "missing rows-section header; got: {out}",
    );
    assert!(
        out.contains("By scenario (worst first):"),
        "missing scenario pane; got: {out}",
    );
    assert!(
        !out.contains("=== BPF VERIFIER STATS ===")
            && !out.contains("=== BPF CALLBACK PROFILE ===")
            && !out.contains("=== KVM STATS"),
        "fixture has no verifier/callback/kvm data — those sections must NOT \
         appear; got: {out}",
    );
}

// ---------------------------------------------------------------
// run_verifier_cell — pure name-parse error branches
// ---------------------------------------------------------------
//
// The two early guards (missing `verifier/` prefix, <3-part cell
// name) run BEFORE the cell banner, check_kvm(), and any scheduler
// / kernel resolution (`run_verifier_cell`'s
// `full_name.strip_prefix("verifier/")` None arm and its
// `if parts.len() != 3` arm), so they are
// host-reachable with no KVM, no scheduler binary, no kernel. A
// syntactically-valid 3-part name would fall through to
// check_kvm() and is intentionally NOT tested here.

/// A name lacking the `verifier/` prefix exits 1 with the
/// missing-prefix diagnostic. Pins the strip_prefix None arm.
#[test]
fn run_verifier_cell_missing_prefix_exits_one() {
    use crate::test_support::test_helpers::capture_stderr;
    let (code, captured) = capture_stderr(|| run_verifier_cell("no_verifier_prefix"));
    assert_eq!(code, 1);
    let stderr = String::from_utf8(captured).expect("stderr is utf-8");
    assert!(
        stderr.contains("missing 'verifier/' prefix"),
        "missing-prefix diagnostic must name the cause; got: {stderr}",
    );
}

/// A name with the prefix but fewer than 3 slash-separated segments
/// (splitn(3) on `only/two` yields 2 parts) exits 1 with the
/// malformed-cell diagnostic naming the expected shape. Pins the
/// parts.len() != 3 arm.
#[test]
fn run_verifier_cell_too_few_parts_exits_one() {
    use crate::test_support::test_helpers::capture_stderr;
    let (code, captured) = capture_stderr(|| run_verifier_cell("verifier/only/two"));
    assert_eq!(code, 1);
    let stderr = String::from_utf8(captured).expect("stderr is utf-8");
    assert!(
        stderr.contains("malformed cell name")
            && stderr.contains("expected verifier/<sched>/<kernel>/<preset>"),
        "malformed-cell diagnostic must name the expected shape; got: {stderr}",
    );
}

// ---------------------------------------------------------------
// AsRef<str> for SanitizedKernelLabel
// ---------------------------------------------------------------

/// The `AsRef<str>` impl exposes the SANITIZED inner string, not
/// the raw input. Mirror of `sanitized_kernel_label_as_str_returns_sanitized_form`
/// but routed through the trait so the `impl AsRef<str> for
/// SanitizedKernelLabel` is load-bearing rather than dead.
#[test]
fn sanitized_kernel_label_as_ref_returns_sanitized_form() {
    let label = SanitizedKernelLabel::new("6.14.2");
    // Disambiguate from the inherent `as_str()` by naming the trait.
    assert_eq!(
        <SanitizedKernelLabel as AsRef<str>>::as_ref(&label),
        "kernel_6_14_2",
    );
    // The raw input contained `.` which sanitizes to `_`, so the
    // AsRef view must NOT equal the raw input.
    let via_as_ref: &str = label.as_ref();
    assert_ne!(via_as_ref, "6.14.2");
}

// ---------------------------------------------------------------
// run_ktstr_test — entry.validate()? bail (pure host check)
// ---------------------------------------------------------------

/// `run_ktstr_test` calls `entry.validate()?` as its first
/// statement, so an entry that violates a
/// validate invariant returns Err BEFORE any VM / KVM work
/// (the `if entry.host_only` / `bpf_map_write` / `run_ktstr_test_inner`
/// body after that `entry.validate()?`). A non-empty name plus
/// `cpu_budget=Some(0)` reaches
/// the cpu_budget-zero bail: the empty-name check fires FIRST
/// (`KtstrTestEntry::validate`'s `self.name.is_empty()` check), so the
/// name MUST be set for this fixture to
/// reach the intended message. Pinning the exact bail string
/// proves the failure came from validate() and not a downstream
/// VM error.
#[test]
fn run_ktstr_test_validate_rejects_zero_cpu_budget() {
    let invalid = KtstrTestEntry {
        name: "__unit_test_run_ktstr_test_validate__",
        cpu_budget: Some(0),
        ..KtstrTestEntry::DEFAULT
    };
    let err = run_ktstr_test(&invalid).expect_err("validate must reject cpu_budget=Some(0)");
    let rendered = format!("{err:#}");
    assert!(
        rendered.contains("cpu_budget=Some(0)") && rendered.contains("zero host-CPU"),
        "bail must be the cpu_budget-zero validate message (proving the failure \
         came from entry.validate()? before any VM work); got: {rendered}",
    );
}

// ---------------------------------------------------------------
// run_named_test — performance_mode / KTSTR_PERF_ONLY skip gates
// ---------------------------------------------------------------
//
// These exercise the cleanly host-only skip branches in
// run_named_test (its `if entry.performance_mode &&
// no_perf_mode_active()` and `if perf_only_skips_entry(entry)` arms):
// a performance_mode test
// under --no-perf-mode, and a non-performance_mode test under
// KTSTR_PERF_ONLY. Both call `crate::report::test_skip` (a bare
// eprintln!) and `record_skip_sidecar`, then return 0 with no VM.
// The `__unit_test_*__` fixture names are excluded from full-run
// listing (is_test_sentinel), so registering them does not add VM
// boots to a real run; they are reachable here only via the direct
// run_named_test call. KTSTR_SIDECAR_DIR is redirected to a
// tempdir so record_skip_sidecar does not pollute the shared run
// dir.

/// Stub func for the perf-skip listing fixtures. Never invoked on
/// the skip path (the skip returns before reaching `func`), but
/// must be a harmless pass for the same dispatcher-reachability
/// reason as `host_only_listing_stub`.
fn perf_skip_listing_stub(
    _ctx: &crate::scenario::Ctx,
) -> anyhow::Result<crate::assert::AssertResult> {
    Ok(crate::assert::AssertResult::pass())
}

const PERF_MODE_SKIP_NAME: &str = "__unit_test_perf_mode_skip__";
const PERF_ONLY_SKIP_NAME: &str = "__unit_test_perf_only_skip__";

#[linkme::distributed_slice(KTSTR_TESTS)]
static __PERF_MODE_SKIP_ENTRY: KtstrTestEntry = KtstrTestEntry {
    name: PERF_MODE_SKIP_NAME,
    func: perf_skip_listing_stub,
    performance_mode: true,
    ..KtstrTestEntry::DEFAULT
};

#[linkme::distributed_slice(KTSTR_TESTS)]
static __PERF_ONLY_SKIP_ENTRY: KtstrTestEntry = KtstrTestEntry {
    name: PERF_ONLY_SKIP_NAME,
    func: perf_skip_listing_stub,
    ..KtstrTestEntry::DEFAULT
};

/// A `performance_mode` test under KTSTR_NO_PERF_MODE skips:
/// exit 0 + the canonical perf-mode skip banner. KTSTR_PERF_ONLY
/// is removed so it cannot pre-empt this branch, and
/// KTSTR_KERNEL_LIST is removed so strip_kernel_suffix is a
/// passthrough. Pins `run_named_test`'s `if entry.performance_mode &&
/// no_perf_mode_active()` skip arm.
#[test]
fn run_named_test_perf_mode_test_skips_under_no_perf_mode() {
    use crate::test_support::test_helpers::{EnvVarGuard, capture_stderr, lock_env};
    let _env_lock = lock_env();
    let _no_perf = EnvVarGuard::set(crate::KTSTR_NO_PERF_MODE_ENV, "1");
    let _perf_only = EnvVarGuard::remove(crate::KTSTR_PERF_ONLY_ENV);
    let _kernel_list = EnvVarGuard::remove(crate::KTSTR_KERNEL_LIST_ENV);
    let sidecar_dir = tempfile::tempdir().expect("create sidecar tempdir");
    let _sidecar = EnvVarGuard::set(crate::KTSTR_SIDECAR_DIR_ENV, sidecar_dir.path());

    let (code, captured) =
        capture_stderr(|| run_named_test(&format!("ktstr/{PERF_MODE_SKIP_NAME}")));
    assert_eq!(
        code, 0,
        "perf_mode test under --no-perf-mode must skip → exit 0"
    );
    let stderr = String::from_utf8(captured).expect("stderr is utf-8");
    assert!(
        stderr.contains("requires performance_mode but --no-perf-mode"),
        "perf-mode skip banner must explain the gate; got: {stderr}",
    );
}

/// A non-`performance_mode` test under KTSTR_PERF_ONLY skips:
/// exit 0 + the perf-only skip banner. KTSTR_NO_PERF_MODE is
/// removed so the perf_mode branch above does not interfere (it
/// is `false && ...` anyway for this fixture), and
/// KTSTR_KERNEL_LIST is removed for the passthrough. Pins
/// `run_named_test`'s `if perf_only_skips_entry(entry)` skip arm.
#[test]
fn run_named_test_non_perf_test_skips_under_perf_only() {
    use crate::test_support::test_helpers::{EnvVarGuard, capture_stderr, lock_env};
    let _env_lock = lock_env();
    let _perf_only = EnvVarGuard::set(crate::KTSTR_PERF_ONLY_ENV, "1");
    let _no_perf = EnvVarGuard::remove(crate::KTSTR_NO_PERF_MODE_ENV);
    let _kernel_list = EnvVarGuard::remove(crate::KTSTR_KERNEL_LIST_ENV);
    let sidecar_dir = tempfile::tempdir().expect("create sidecar tempdir");
    let _sidecar = EnvVarGuard::set(crate::KTSTR_SIDECAR_DIR_ENV, sidecar_dir.path());

    let (code, captured) =
        capture_stderr(|| run_named_test(&format!("ktstr/{PERF_ONLY_SKIP_NAME}")));
    assert_eq!(
        code, 0,
        "non-perf test under KTSTR_PERF_ONLY must skip → exit 0"
    );
    let stderr = String::from_utf8(captured).expect("stderr is utf-8");
    assert!(
        stderr.contains("KTSTR_PERF_ONLY is active"),
        "perf-only skip banner must explain the gate; got: {stderr}",
    );
}

// ---------------------------------------------------------------
// list_tests — KTSTR_BUDGET_SECS parse arms
// ---------------------------------------------------------------
//
// `list_tests` reads KTSTR_BUDGET_SECS (via its
// `s.parse::<f64>()` match populating `budget_secs`) and
// branches three ways:
//   - unset / unparseable / non-positive → fall back to
//     `list_tests_all` (after an `eprintln!` warning for the two
//     malformed cases),
//   - valid positive f64 → `list_tests_budget` (which emits a
//     `ktstr budget:` summary to stderr).
// Each test holds `lock_env()`, clears KTSTR_KERNEL_LIST so the
// single-kernel print path is taken, and captures stdout (the test
// names) plus stderr (the warning / budget summary).

/// A non-numeric KTSTR_BUDGET_SECS emits the parse-error warning
/// and falls through to `list_tests_all` — the base test names
/// still land on stdout. Pins the `Err(e) => { eprintln!(...) }`
/// arm of `list_tests`'s `s.parse::<f64>()` match.
#[test]
fn list_tests_budget_non_numeric_warns_and_lists_all() {
    use crate::test_support::test_helpers::{EnvVarGuard, capture_stderr, lock_env};
    let _env_lock = lock_env();
    let _budget = EnvVarGuard::set(crate::KTSTR_BUDGET_SECS_ENV, "not-a-number");
    let _kernel_list = EnvVarGuard::remove(crate::KTSTR_KERNEL_LIST_ENV);
    let _cargo = EnvVarGuard::remove(crate::KTSTR_CARGO_TEST_MODE_ENV);

    // Nest stdout-capture inside stderr-capture so both the warning
    // (stderr) and the fallthrough listing (stdout) are observed.
    let ((_, stdout), stderr) = capture_stderr(|| capture_stdout(|| list_tests(false)));
    let stderr = String::from_utf8(stderr).expect("stderr is utf-8");
    let stdout = String::from_utf8(stdout).expect("stdout is utf-8");
    assert!(
        stderr.contains("KTSTR_BUDGET_SECS=\"not-a-number\""),
        "non-numeric budget must warn with the raw value; got: {stderr}",
    );
    assert!(
        stderr.contains("ignoring"),
        "parse-error warning must say it is ignoring the value; got: {stderr}",
    );
    // No budget summary — the budget lister was never invoked.
    assert!(
        !stderr.contains("ktstr budget:"),
        "unparseable budget must NOT route through list_tests_budget; got: {stderr}",
    );
    // Fell through to list_tests_all: at least one base `ktstr/` line.
    assert!(
        stdout
            .lines()
            .any(|l| l.starts_with("ktstr/") && l.ends_with(": test")),
        "fallthrough must list base ktstr/ test names; got:\n{stdout}",
    );
}

/// A non-positive KTSTR_BUDGET_SECS (zero / negative) emits the
/// "must be positive" warning and falls through to
/// `list_tests_all`. Pins the `Ok(v) => { eprintln!("...must be positive...") }`
/// non-positive arm of `list_tests`'s `s.parse::<f64>()` match,
/// distinct from the parse-error arm above.
#[test]
fn list_tests_budget_non_positive_warns_and_lists_all() {
    use crate::test_support::test_helpers::{EnvVarGuard, capture_stderr, lock_env};
    let _env_lock = lock_env();
    let _budget = EnvVarGuard::set(crate::KTSTR_BUDGET_SECS_ENV, "0");
    let _kernel_list = EnvVarGuard::remove(crate::KTSTR_KERNEL_LIST_ENV);
    let _cargo = EnvVarGuard::remove(crate::KTSTR_CARGO_TEST_MODE_ENV);

    let ((_, stdout), stderr) = capture_stderr(|| capture_stdout(|| list_tests(false)));
    let stderr = String::from_utf8(stderr).expect("stderr is utf-8");
    let stdout = String::from_utf8(stdout).expect("stdout is utf-8");
    assert!(
        stderr.contains("KTSTR_BUDGET_SECS=0") && stderr.contains("must be positive"),
        "non-positive budget must warn it must be positive; got: {stderr}",
    );
    assert!(
        !stderr.contains("ktstr budget:"),
        "non-positive budget must NOT route through list_tests_budget; got: {stderr}",
    );
    assert!(
        stdout
            .lines()
            .any(|l| l.starts_with("ktstr/") && l.ends_with(": test")),
        "fallthrough must list base ktstr/ test names; got:\n{stdout}",
    );
}

/// A valid positive KTSTR_BUDGET_SECS routes through
/// `list_tests_budget`, which emits a `ktstr budget:` summary line
/// to stderr. Pins `list_tests`'s
/// `if let Some(budget) = budget_secs { list_tests_budget(...) }`
/// arm. A generous budget keeps the
/// selector inclusive so the path is exercised regardless of the
/// registered test set's estimated durations.
#[test]
fn list_tests_budget_valid_routes_to_budget_lister() {
    use crate::test_support::test_helpers::{EnvVarGuard, capture_stderr, lock_env};
    let _env_lock = lock_env();
    let _budget = EnvVarGuard::set(crate::KTSTR_BUDGET_SECS_ENV, "100000");
    let _kernel_list = EnvVarGuard::remove(crate::KTSTR_KERNEL_LIST_ENV);
    let _cargo = EnvVarGuard::remove(crate::KTSTR_CARGO_TEST_MODE_ENV);

    let ((_, _stdout), stderr) = capture_stderr(|| capture_stdout(|| list_tests(false)));
    let stderr = String::from_utf8(stderr).expect("stderr is utf-8");
    assert!(
        stderr.contains("ktstr budget:"),
        "a valid budget must route through list_tests_budget (its \
         `ktstr budget:` summary); got: {stderr}",
    );
}

// ---------------------------------------------------------------
// list_verifier_cells_all — empty / scheduler-free kernel list
// ---------------------------------------------------------------

#[test]
fn verifier_named_topology_exclusion_does_not_change_gauntlet_constraints() {
    let presets = crate::gauntlet::gauntlet_presets();
    let excluded = presets
        .iter()
        .find(|p| p.name == "medium-4llc-nosmt")
        .expect("named verifier exclusion fixture preset");
    let retained = presets
        .iter()
        .find(|p| p.name == "medium-8llc-nosmt")
        .expect("retained verifier fixture preset");
    let sched = Scheduler::named("verifier_exclusion_fixture")
        .verifier_exclude_topologies(&["medium-4llc-nosmt"]);

    assert!(
        sched.constraints.accepts_verifier(&excluded.topology),
        "the structural verifier constraints must still accept the preset",
    );
    assert!(
        !scheduler_accepts_verifier_preset(&sched, excluded),
        "the named verifier exception must suppress exactly that cell",
    );
    assert!(
        scheduler_accepts_verifier_preset(&sched, retained),
        "unlisted verifier topologies must remain eligible",
    );
    assert!(
        sched.constraints.accepts(
            &excluded.topology,
            excluded.topology.total_cpus(),
            excluded.topology.num_llcs(),
            excluded.topology.cores_per_llc * excluded.topology.threads_per_core,
        ),
        "the verifier-only exclusion must not alter ordinary gauntlet constraints",
    );
}

/// With KTSTR_KERNEL_LIST unset, `list_verifier_cells_all` returns
/// immediately and emits nothing. Pins the
/// `if kernel_list.is_empty() { return; }` early-return in
/// `list_verifier_cells_all` — the verifier matrix dimension is
/// KTSTR_KERNEL_LIST, so an absent list means zero cells.
#[test]
fn list_verifier_cells_all_empty_kernel_list_emits_nothing() {
    use crate::test_support::test_helpers::{EnvVarGuard, lock_env};
    let _env_lock = lock_env();
    let _kernel_list = EnvVarGuard::remove(crate::KTSTR_KERNEL_LIST_ENV);

    let (_, captured) = capture_stdout(list_verifier_cells_all);
    let stdout = std::str::from_utf8(&captured).expect("utf-8");
    assert!(
        stdout.is_empty(),
        "empty KTSTR_KERNEL_LIST must emit zero verifier cells; got:\n{stdout}",
    );
}

/// With a populated KTSTR_KERNEL_LIST but NO declared schedulers in
/// this binary's `KTSTR_SCHEDULERS` slice (the lib test binary
/// registers none), `list_verifier_cells_all` runs the post-early-
/// return setup (its `gauntlet_presets()` / `host_capacity()` /
/// `scheduler_filter` bindings) and then iterates zero schedulers — so
/// no `verifier/` line is ever printed. Pins that a kernel list alone
/// does not synthesize cells without a scheduler to pair them with.
#[test]
fn list_verifier_cells_all_no_schedulers_emits_no_cells() {
    use crate::test_support::test_helpers::{EnvVarGuard, lock_env};
    let _env_lock = lock_env();
    let _kernel_list = EnvVarGuard::set(crate::KTSTR_KERNEL_LIST_ENV, TWO_KERNEL_LIST);

    let (_, captured) = capture_stdout(list_verifier_cells_all);
    let stdout = std::str::from_utf8(&captured).expect("utf-8");
    assert!(
        !stdout.lines().any(|l| l.starts_with("verifier/")),
        "zero declared schedulers must yield zero `verifier/` cells even \
         with a populated kernel list; got:\n{stdout}",
    );
}

/// `workspace_member_packages` queried against ktstr's OWN workspace
/// (`CARGO_MANIFEST_DIR`) via `cargo metadata` includes the real
/// scheduler package (its cells emit) and EXCLUDES the macro-expansion
/// fixture scheduler names from tests/declare_scheduler.rs (their cells
/// are gated out of the sweep so `--run-ignored` doesn't fail on
/// `cargo build -p <fixture>`). Spawning `cargo metadata` here is
/// acceptable — sibling tests already spawn cargo builds.
#[test]
fn workspace_member_packages_includes_real_scheduler_excludes_fixtures() {
    let pkgs = workspace_member_packages(env!("CARGO_MANIFEST_DIR"));
    assert!(
        pkgs.contains("ktstr"),
        "the root ktstr crate must be a workspace member",
    );
    assert!(
        pkgs.contains("scx-ktstr"),
        "the real scheduler package must be a workspace member (emits verifier cells)",
    );
    assert!(
        !pkgs.contains("scx-full") && !pkgs.contains("scx-ee") && !pkgs.contains("scx-both"),
        "declare_scheduler.rs fixture schedulers must NOT be workspace members (cells gated out)",
    );
}

/// A nonexistent manifest dir yields the empty set: `cargo metadata`
/// fails (no `Cargo.toml` there) and the query degrades to "no members",
/// which gates out every `Discover` cell rooted in that dir rather than
/// emitting cells the runtime resolver could not build.
#[test]
fn workspace_member_packages_missing_manifest_dir_is_empty() {
    let pkgs = workspace_member_packages("/nonexistent/ktstr_workspace_xyz");
    assert!(
        pkgs.is_empty(),
        "a manifest dir with no Cargo.toml must yield an empty member set; got: {pkgs:?}",
    );
}

// ---------------------------------------------------------------
// run_verifier_cell — scheduler-not-found branch (post-check_kvm)
// ---------------------------------------------------------------

/// A syntactically-valid 3-part cell name whose scheduler is not in
/// `KTSTR_SCHEDULERS` (the lib test binary declares none) exits 1
/// via the "no declared scheduler" branch (`run_verifier_cell`'s
/// `let Some(sched) = KTSTR_SCHEDULERS.iter().find(|s| s.name == sched_name) else { ... }`).
/// This branch is gated behind `check_kvm()` (the preceding
/// `if let Err(e) = crate::cli::check_kvm()` preflight in the same fn), so
/// the test runs the branch only when KVM is actually available —
/// when `check_kvm()` errors (no /dev/kvm, or permission denied)
/// the run_verifier_cell call would exit 1 via the KVM-preflight
/// branch instead, so the test skips cleanly to keep the asserted
/// branch deterministic. The cell banner (printed before every
/// branch) lands on stdout; the not-found diagnostic on stderr.
#[test]
fn run_verifier_cell_unknown_scheduler_exits_one() {
    use crate::test_support::test_helpers::capture_stderr;
    // Gate on real KVM availability so the asserted branch is the
    // scheduler-not-found one, not the KVM-preflight one. Mirrors the
    // geteuid()-gate pattern the build_host_cgroup_manager tests use.
    if crate::cli::check_kvm().is_err() {
        return;
    }
    let (code, captured) = capture_stderr(|| {
        // Banner goes to stdout; swallow it so only the diagnostic
        // is examined. The 3 parts are sched/kernel/preset.
        let (code, _stdout) =
            capture_stdout(|| run_verifier_cell("verifier/__no_such_sched__/kernel_x/tiny-1llc"));
        code
    });
    assert_eq!(code, 1, "unknown scheduler cell must exit 1");
    let stderr = String::from_utf8(captured).expect("stderr is utf-8");
    assert!(
        stderr.contains("no declared scheduler") && stderr.contains("__no_such_sched__"),
        "exit 1 must come from the scheduler-not-found branch naming the \
         missing scheduler; got: {stderr}",
    );
}

/// Unconditional companion to the KVM-gated test above (which SKIPs on a
/// no-KVM runner and would otherwise read as PASS without exercising
/// anything). Pins the scheduler-lookup invariant the exit-1 branch
/// depends on: a name absent from `KTSTR_SCHEDULERS` does not resolve, so
/// the `find(|s| s.name == sched_name)` returns `None` and the cell exits
/// with code 1. The lib test binary declares zero schedulers, so any
/// name is absent; the gated test exercises the surrounding `run_verifier_cell`
/// exit path when KVM is present. This runs everywhere, so the
/// precondition is verified even where the gated test cannot.
#[test]
fn unknown_scheduler_is_absent_from_registry() {
    assert!(
        crate::test_support::KTSTR_SCHEDULERS
            .iter()
            .all(|s| s.name != "__no_such_sched__"),
        "an unknown scheduler name must not resolve in KTSTR_SCHEDULERS — \
         the precondition run_verifier_cell's scheduler-not-found exit-1 \
         branch relies on",
    );
}

// ---------------------------------------------------------------
// run_host_only_test — host-side dispatch (no VM)
// ---------------------------------------------------------------

/// `run_host_only_test` (the wrapper that calls
/// `run_host_only_test_inner` then `result_to_exit_code`) runs a host_only
/// entry on the host with no VM: it resolves real-host sysfs
/// topology, builds a cgroup manager (no setup), builds a Ctx, and
/// calls the entry's `func`, then projects the AssertResult through
/// `result_to_exit_code`. The registered `__HOST_ONLY_LISTING_ENTRY`
/// fixture's func returns `AssertResult::pass()` and touches no
/// cgroup, so the call needs neither root nor a VM and must exit 0.
/// Pins the host_only dispatch wrapper against a regression that
/// routed host_only tests through the VM path.
#[test]
fn run_host_only_test_passing_entry_exits_zero() {
    use crate::test_support::test_helpers::{EnvVarGuard, lock_env};
    // resolve_host_cgroup_parent reads KTSTR_HOST_CGROUP_PARENT;
    // KTSTR_CGROUP_WALK_ROOT feeds build_host_cgroup_manager. Hold the
    // env lock and clear both so the resolve cascade takes its default
    // path deterministically regardless of a concurrent setter.
    let _env_lock = lock_env();
    let _parent = EnvVarGuard::remove(crate::KTSTR_HOST_CGROUP_PARENT_ENV);
    let _walk = EnvVarGuard::remove(crate::KTSTR_CGROUP_WALK_ROOT_ENV);

    let entry = find_test(HOST_ONLY_LISTING_NAME)
        .expect("the host_only listing fixture must be registered in KTSTR_TESTS");
    assert!(entry.host_only, "fixture must be host_only");
    let code = run_host_only_test(entry);
    assert_eq!(
        code, 0,
        "a passing host_only entry must dispatch on the host and exit 0 \
         (no VM, no root)",
    );
}

// ---------------------------------------------------------------
// run_gauntlet_test — topo derivation reached before skip gate
// ---------------------------------------------------------------

/// `run_gauntlet_test` with a registered `performance_mode` entry
/// and a real preset name derives the per-preset TopoOverride
/// (its `derive_test_memory_mib(...)` + `TopoOverride { ... }`
/// bindings from the
/// preset topology) and THEN hits the perf-mode skip gate
/// (its `if entry.performance_mode && no_perf_mode_active()` arm)
/// under KTSTR_NO_PERF_MODE, returning 0
/// with no VM. Drives the topo-derivation lines that the existing
/// `run_gauntlet_test_*` error-branch tests never reach (they bail
/// at the name-parse / preset-lookup guards before topo derivation).
/// KTSTR_SIDECAR_DIR is redirected to a tempdir so the skip
/// sidecar `record_skip_sidecar` writes does not pollute the shared
/// run dir.
#[test]
fn run_gauntlet_test_perf_mode_entry_derives_topo_then_skips() {
    use crate::test_support::test_helpers::{EnvVarGuard, capture_stderr, lock_env};
    let _env_lock = lock_env();
    let _no_perf = EnvVarGuard::set(crate::KTSTR_NO_PERF_MODE_ENV, "1");
    let _perf_only = EnvVarGuard::remove(crate::KTSTR_PERF_ONLY_ENV);
    let sidecar_dir = tempfile::tempdir().expect("create sidecar tempdir");
    let _sidecar = EnvVarGuard::set(crate::KTSTR_SIDECAR_DIR_ENV, sidecar_dir.path());

    // `tiny-1llc` is the first gauntlet preset (the first tuple in
    // `gauntlet::gauntlet_presets()`) — a
    // real preset so the preset-lookup guard passes and topo
    // derivation runs.
    let (code, captured) =
        capture_stderr(|| run_gauntlet_test(&format!("{PERF_MODE_SKIP_NAME}/tiny-1llc")));
    assert_eq!(
        code, 0,
        "perf_mode gauntlet variant under --no-perf-mode must skip → exit 0 \
         after deriving the preset topology",
    );
    let stderr = String::from_utf8(captured).expect("stderr is utf-8");
    assert!(
        stderr.contains("requires performance_mode but --no-perf-mode"),
        "skip must come from the perf-mode gate (reached only AFTER topo \
         derivation); got: {stderr}",
    );
}

// ---------------------------------------------------------------
// ktstr_list_only / warn_duplicate_test_names_once
// ---------------------------------------------------------------

#[test]
fn is_ignored_honors_registered_flag_and_demo_convention() {
    let ordinary = KtstrTestEntry {
        name: "ordinary",
        ..KtstrTestEntry::DEFAULT
    };
    assert!(!is_ignored(&ordinary));

    let attributed = KtstrTestEntry {
        name: "compile_only_fixture",
        ignored: true,
        ..KtstrTestEntry::DEFAULT
    };
    assert!(is_ignored(&attributed));

    let demo = KtstrTestEntry {
        name: "demo_manual_benchmark",
        ..KtstrTestEntry::DEFAULT
    };
    assert!(is_ignored(&demo));
}

/// `ktstr_list_only` (whose body is
/// `args.iter().any(|a| a == "--ignored")` then `list_tests(ignored_only)`)
/// reads argv for
/// `--ignored`, then delegates to `list_tests`, which prints the
/// `ktstr/{name}: test` names to stdout and RETURNS (unlike
/// `ktstr_main`, which `process::exit`s). Pins the
/// return-don't-exit listing entry point against a regression that
/// routed it through the exiting handler. The bucket
/// (`--ignored`-only vs all) is read from the test process's own
/// argv exactly as `ktstr_list_only` reads it, so the assertion is
/// keyed on the SAME bucket the function selected: the host_only
/// fixture is non-`demo_` (not ignored), so it appears only in the
/// all-tests bucket — under an `--ignored` argv the bucket is empty
/// of this fixture and the assertion correctly relaxes to "function
/// returned without exiting".
#[test]
fn ktstr_list_only_prints_test_names_and_returns() {
    use crate::test_support::test_helpers::{EnvVarGuard, lock_env};
    let _env_lock = lock_env();
    // Single-kernel listing + no budget filtering so the plain
    // `ktstr/{name}: test` shape is emitted for the host_only fixture.
    let _kernel_list = EnvVarGuard::remove(crate::KTSTR_KERNEL_LIST_ENV);
    let _budget = EnvVarGuard::remove(crate::KTSTR_BUDGET_SECS_ENV);
    let _cargo = EnvVarGuard::remove(crate::KTSTR_CARGO_TEST_MODE_ENV);

    // Mirror the bucket selection inside ktstr_list_only so the
    // assertion is deterministic regardless of how the harness
    // invoked the binary (a `--ignored` argv selects the
    // ignored-only bucket, which excludes the non-`demo_` host_only
    // fixture).
    let ignored_only = std::env::args().any(|a| a == "--ignored");

    let (_, captured) = capture_stdout(ktstr_list_only);
    let stdout = std::str::from_utf8(&captured).expect("utf-8");
    let want = format!("ktstr/{HOST_ONLY_LISTING_NAME}: test");
    if ignored_only {
        // Ignored-only bucket: the non-ignored host_only fixture
        // must NOT appear; the function still returned (no exit).
        assert!(
            !stdout.contains(&want),
            "the non-ignored host_only fixture must be absent from the \
             --ignored bucket; got:\n{stdout}",
        );
    } else {
        // All-tests bucket: the registered host_only fixture must
        // appear with its bare `ktstr/<name>: test` form.
        assert!(
            stdout.contains(&want),
            "ktstr_list_only must print the registered host_only fixture's \
             base name in the all-tests bucket; got:\n{stdout}",
        );
    }
}

/// `warn_duplicate_test_names_once` (whose body is a
/// `CHECKED.get_or_init(...)` over a `static CHECKED: OnceLock<()>`) gates
/// its walk through a process-wide `OnceLock<()>`. The walk itself
/// (whose detection logic is pinned by the
/// `warn_duplicate_test_names_inner_*` tests above) is fired against
/// the real `KTSTR_TESTS` slice; this test only pins the wrapper's
/// idempotence contract: calling it (possibly again, after the
/// production listing path or a sibling test already fired the gate)
/// must complete without panicking — the `get_or_init` closure runs
/// at most once per process and is a no-op on every subsequent call.
#[test]
fn warn_duplicate_test_names_once_is_idempotent_and_panic_free() {
    use crate::test_support::test_helpers::capture_stderr;
    // Both calls are captured so the (at most one) emission does not
    // leak onto the test runner's stderr, and so a panic inside the
    // OnceLock closure would surface as a test failure rather than a
    // captured-output artifact.
    let (_, _first) = capture_stderr(warn_duplicate_test_names_once);
    // Second call must be a pure no-op: the OnceLock is already
    // initialized, so the closure does not run again.
    let (_, second) = capture_stderr(warn_duplicate_test_names_once);
    assert!(
        second.is_empty(),
        "the second warn_duplicate_test_names_once call must emit nothing \
         (OnceLock already fired); got: {:?}",
        String::from_utf8_lossy(&second),
    );
}
