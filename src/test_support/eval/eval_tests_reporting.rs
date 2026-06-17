//! Part of the eval module's unit-test suite, split across sibling
//! `eval_tests*.rs` files to keep each under the size ceiling. Child of
//! `eval`: reaches the production core via `super::` / `super::super::`.
//! The self-scan tests call `include_str!("mod.rs")`, which resolves to the sibling `eval/mod.rs` (the production core), so they scan the real code.
use super::super::output::STAGE_INIT_STARTED_NO_PAYLOAD;
use super::super::test_helpers::lifecycle_drain;
// EnvVarGuard / lock_env / sched_entry are only used by the
// `#[cfg(feature = "wprof")]` apply_expect_auto_repro tests below.
#[cfg(feature = "wprof")]
use super::super::test_helpers::{EnvVarGuard, lock_env, sched_entry};
use super::*;
use tempfile::TempDir;

// -- write_placeholder_failure_dump_if_missing --
//
// Pins the spec-promise that every failed test leaves a JSON
// failure-dump file on disk (real for scheduler-attached
// failures, placeholder for pre-attach failures). Tests
// exercise the helper directly without booting a VM, so the
// coverage matrix is deterministic and runs well under a
// second.

/// Failure + path missing → writes placeholder; file exists and
/// parses as a FailureDumpReport with is_placeholder=true.
#[test]
fn placeholder_dump_writes_when_path_missing() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test_writes.failure-dump.json");
    let result = vmm::VmResult {
        success: false,
        ..vmm::VmResult::test_fixture()
    };
    write_placeholder_failure_dump_if_missing(&path, &result);
    assert!(path.exists(), "placeholder must land at the canonical path");
    let body = std::fs::read_to_string(&path).expect("readable");
    let report: crate::monitor::dump::FailureDumpReport =
        serde_json::from_str(&body).expect("valid FailureDumpReport JSON");
    assert!(report.is_placeholder, "stub must carry is_placeholder=true",);
    let reason = report
        .sdt_alloc_unavailable
        .as_deref()
        .expect("placeholder sets sdt_alloc_unavailable");
    assert!(
        reason.contains("no BPF state captured"),
        "reason must explain why no real dump exists: {reason}",
    );
}

/// Failure + path already exists → don't overwrite the real dump.
#[test]
fn placeholder_dump_skipped_when_file_exists() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test_exists.failure-dump.json");
    let sentinel: &[u8] = br#"{"real":true,"is_placeholder":false}"#;
    std::fs::write(&path, sentinel).unwrap();
    let result = vmm::VmResult {
        success: false,
        ..vmm::VmResult::test_fixture()
    };
    write_placeholder_failure_dump_if_missing(&path, &result);
    let after = std::fs::read(&path).unwrap();
    assert_eq!(
        after, sentinel,
        "real dump must not be overwritten by placeholder",
    );
}

/// Atomic-publish via `.tmp` + `rename(2)` leaves no orphan.
/// Pin so a regression that drops the rename (and writes directly
/// to the canonical path) would let a half-written file leak;
/// the absence of any `.json.tmp` after a successful write
/// proves the rename completed.
#[test]
fn placeholder_dump_atomic_publish_no_tmp_orphan() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test_atomic.failure-dump.json");
    let result = vmm::VmResult {
        success: false,
        ..vmm::VmResult::test_fixture()
    };
    write_placeholder_failure_dump_if_missing(&path, &result);
    assert!(path.exists());
    let tmp = path.with_extension("json.tmp");
    assert!(
        !tmp.exists(),
        "atomic rename(2) must consume the .tmp file; orphan at: {}",
        tmp.display(),
    );
}

/// Reason embeds the lifecycle stage label from
/// `classify_init_stage`. Synthesize a drain with `InitStarted`
/// but no `PayloadStarting` → stage label is the
/// "init started but payload never ran" message, and that text
/// must appear in the placeholder reason. A regression that
/// drops the stage label propagation would surface here.
#[test]
fn placeholder_dump_reason_includes_lifecycle_stage_label() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test_stage.failure-dump.json");
    let drain = lifecycle_drain(&[crate::vmm::wire::LifecyclePhase::InitStarted]);
    let result = vmm::VmResult {
        success: false,
        guest_messages: Some(drain),
        ..vmm::VmResult::test_fixture()
    };
    write_placeholder_failure_dump_if_missing(&path, &result);
    let body = std::fs::read_to_string(&path).unwrap();
    let report: crate::monitor::dump::FailureDumpReport = serde_json::from_str(&body).unwrap();
    let reason = report.sdt_alloc_unavailable.as_deref().unwrap();
    assert!(
        reason.contains(STAGE_INIT_STARTED_NO_PAYLOAD),
        "reason must include the lifecycle stage label `{}`: {reason}",
        STAGE_INIT_STARTED_NO_PAYLOAD,
    );
}

/// Reason folds the `BUG SUMMARY` extraction (per the design
/// intent) so the on-disk artifact matches the
/// stderr summary instead of being less informative. Synthesize a
/// `result.output` carrying a `scx_bpf_error` line; the
/// extract_bug_summary fallback path picks it up and the
/// reason includes a `BUG SUMMARY: ...` clause.
#[test]
fn placeholder_dump_reason_includes_bug_summary_from_sched_log() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test_bug.failure-dump.json");
    let result = vmm::VmResult {
        success: false,
        output: "scx_bpf_error: apply_cell_config returned -EINVAL\n".to_string(),
        ..vmm::VmResult::test_fixture()
    };
    write_placeholder_failure_dump_if_missing(&path, &result);
    let body = std::fs::read_to_string(&path).unwrap();
    let report: crate::monitor::dump::FailureDumpReport = serde_json::from_str(&body).unwrap();
    let reason = report.sdt_alloc_unavailable.as_deref().unwrap();
    assert!(
        reason.contains("BUG SUMMARY:"),
        "reason must fold the BUG SUMMARY extraction: {reason}",
    );
    assert!(
        reason.contains("apply_cell_config returned -EINVAL"),
        "BUG SUMMARY text must surface the actionable scx_bpf_error: {reason}",
    );
}

/// Reason omits the `BUG SUMMARY:` clause when no actionable
/// text could be extracted. Pin so a regression that emits
/// `BUG SUMMARY: ` (empty) or a `BUG SUMMARY: None` literal
/// would surface here.
#[test]
fn placeholder_dump_reason_omits_bug_summary_when_none() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test_no_bug.failure-dump.json");
    let result = vmm::VmResult {
        success: false,
        ..vmm::VmResult::test_fixture()
    };
    write_placeholder_failure_dump_if_missing(&path, &result);
    let body = std::fs::read_to_string(&path).unwrap();
    let report: crate::monitor::dump::FailureDumpReport = serde_json::from_str(&body).unwrap();
    let reason = report.sdt_alloc_unavailable.as_deref().unwrap();
    assert!(
        !reason.contains("BUG SUMMARY"),
        "reason must not mention BUG SUMMARY when no actionable text was extracted: {reason}",
    );
}

/// Pin both production call sites of
/// `write_placeholder_failure_dump_if_missing` against regression
/// that removes the failure-gating. The helper has no internal
/// success check — callers are responsible for never invoking it
/// on a successful run. A regression that:
///   - removes the `if !result.success` gate at the post-`vm.run`
///     site → would write a stub on every test, including passing
///     ones (a stub never overwrites a real dump thanks to the
///     `_if_missing` early-return, but a stub on a passing test
///     means sidecar walkers see a placeholder `FailureDumpReport`
///     where none should exist), OR
///   - drops the call in the `post_vm` Err branch → spec-promise
///     parity breaks: host-side-overruled failures
///     (result.success=true but post_vm callback returned Err)
///     would land with no failure-dump artifact at all.
///
/// The 6 helper-only unit tests above (`placeholder_dump_*`) cover
/// what the helper does once invoked; this test covers when the
/// helper is invoked. Interim coverage until an
/// `evaluate_vm_result` mock harness lands. Once that harness
/// exists, replace this with an E2E test that asserts no stub
/// lands when `result.success = true`.
///
/// Fragile to source refactors: if production code wraps the call
/// sites in a helper function, or spells either gate keyword
/// differently (`if result.failed()` vs `if !result.success`,
/// `match post_vm(&result) { Err(...) => ... }` vs
/// `if let Err(e) = post_vm(&result)`), this test fails even
/// when the failure-gating semantics are preserved. Update the
/// patterns or migrate to the E2E mock-harness test in that
/// case. Site lookup is pattern-based (not order-based) so
/// swapping the source-line order of the two sites does not
/// false-positive — each gate is searched across all call sites.
#[test]
fn placeholder_dump_production_call_sites_are_failure_gated() {
    let src = include_str!("mod.rs");
    let lines: Vec<&str> = src.lines().collect();
    // Scan the whole production core: this test lives in
    // eval_tests.rs, so the scanned mod.rs (via
    // include_str!("mod.rs")) holds no test literals to exclude.
    // The `&primary_dump_path` filter below isolates the 2
    // production call sites — the helper definition uses `path`
    // and the sibling unit tests use `&path`.
    let scan_end = lines.len();
    let call_lines: Vec<usize> = lines
        .iter()
        .enumerate()
        .take(scan_end)
        .filter_map(|(i, l)| {
            l.contains("write_placeholder_failure_dump_if_missing(&primary_dump_path")
                .then_some(i)
        })
        .collect();
    let display_lines: Vec<usize> = call_lines.iter().map(|i| i + 1).collect();
    assert_eq!(
        call_lines.len(),
        2,
        "expected exactly 2 production call sites (matched on &primary_dump_path \
             to exclude helper definition and unit-test call sites which use &path); \
             found {} at lines {:?}",
        call_lines.len(),
        display_lines,
    );
    // Pattern-based site lookup: find the call line whose 10-line
    // lookback contains `if !result.success` (post-`vm.run` site)
    // and the call line whose 20-line lookback contains BOTH
    // halves of the post_vm Err branch gate. Robust to source
    // reordering — neither site is identified by its position
    // in the call_lines vec.
    let success_gated = call_lines.iter().copied().find(|&i| {
        lines[i.saturating_sub(10)..=i]
            .join("\n")
            .contains("if !result.success")
    });
    let post_vm_gated = call_lines.iter().copied().find(|&i| {
        let window = lines[i.saturating_sub(20)..=i].join("\n");
        // The gate's PRE-binding is the `run_post_vm_callbacks(entry,
        // &result, guest_already_failed)` call (which combines the
        // `post_vm` and `post_vm_unconditional` dispatch + panic-catch);
        // the gate itself is `post_vm_err.is_some()`. Match both.
        window.contains("run_post_vm_callbacks(entry, &result, guest_already_failed)")
            && window.contains("post_vm_err.is_some()")
    });
    let success_gated = success_gated.unwrap_or_else(|| {
        panic!(
            "no production call site is gated by `if !result.success` (post-`vm.run` \
                 placeholder emission); production sites at lines: {display_lines:?}",
        )
    });
    let post_vm_gated = post_vm_gated.unwrap_or_else(|| {
        panic!(
            "no production call site is gated by both \
                 `run_post_vm_callbacks(entry, &result, guest_already_failed)` \
                 (the post_vm_err binding source) AND `post_vm_err.is_some()` \
                 (the gate); production sites at lines: {display_lines:?}",
        )
    });
    // Each gate guards a distinct call site. If the same site
    // satisfies both patterns, one of the two semantic gates is
    // missing from the OTHER site even though both gate keywords
    // are present in the file.
    assert_ne!(
        success_gated,
        post_vm_gated,
        "the same call site (line {}) satisfies both gate patterns; \
             each gate should guard a different call site (sites: {display_lines:?})",
        success_gated + 1,
    );
}

/// Pin `bug_summary_line()` as positional arg 2 — immediately
/// after `fingerprint_line` — in every failure-message
/// `format!()` call. The 4 failure paths in `evaluate_vm_result`
/// (assert-fail, monitor-fail, timeout, no-result) each render
/// their stderr message via a `format!()` whose first two
/// positionals are `fingerprint_line` then `bug_summary_line()`,
/// so the operator scanning a CI log sees the BUG SUMMARY at
/// the top of the error block where the eye stops on the first
/// few lines. A regression that swaps the order, drops the
/// call, or moves it past `entry.name`/`topo` would push the
/// BUG SUMMARY below the test-name / topology line — exactly
/// the location the redesign moved it out of.
///
/// Fragile to source refactors: a refactor that wraps the
/// failure-message format!() blocks in a helper function would
/// drop the per-site positional check. In that case, the
/// helper itself takes both args by position; update this test
/// to walk the helper's `format!()` instead.
/// Strip a single leading `{name}` named-argument span from a
/// format-string literal so `assert!(starts_with("\"{}{}"))`
/// passes when the format string begins with `"{name}{}{}..."`.
/// Walks past the opening `"` and one balanced `{ident}` pair
/// if present, then re-prefixes the `"` so the caller's
/// starts_with check still sees the quote.
fn strip_named_arg_prefix(s: &str) -> String {
    let rest = match s.strip_prefix('"') {
        Some(r) => r,
        None => return s.to_string(),
    };
    let rest = match rest.strip_prefix('{') {
        Some(r) => r,
        None => return s.to_string(),
    };
    let end = match rest.find('}') {
        Some(e) => e,
        None => return s.to_string(),
    };
    let name = &rest[..end];
    let is_named_arg = !name.is_empty()
        && name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
        && name.chars().next().is_some_and(|c| !c.is_ascii_digit());
    if !is_named_arg {
        return s.to_string();
    }
    format!("\"{}", &rest[end + 1..])
}

#[test]
fn bug_summary_line_immediately_follows_fingerprint_line_in_all_failure_messages() {
    let src = include_str!("mod.rs");
    let lines: Vec<&str> = src.lines().collect();
    // Scan the whole production core: this test lives in
    // eval_tests.rs, so the scanned mod.rs holds no test literals
    // to double-count.
    let scan_end = lines.len();
    // Find every line that is exactly `fingerprint_line,`
    // (trimmed). The 4 failure-message format!() sites all
    // pass `fingerprint_line` then `bug_summary_line()` on the
    // next non-empty line. The single binding site
    // (`let fingerprint_line = ...`) does not trim to
    // `fingerprint_line,`, so it's excluded naturally.
    let fingerprint_arg_lines: Vec<usize> = lines
        .iter()
        .enumerate()
        .take(scan_end)
        .filter_map(|(i, l)| (l.trim() == "fingerprint_line,").then_some(i))
        .collect();
    let display_lines: Vec<usize> = fingerprint_arg_lines.iter().map(|i| i + 1).collect();
    assert_eq!(
        fingerprint_arg_lines.len(),
        4,
        "expected exactly 4 failure-message format!() sites passing \
             `fingerprint_line,` as a positional argument (assert-fail, \
             monitor-fail, timeout, no-result paths); found {} at lines {:?}. \
             If a 5th failure path was added, extend this test; if a path was \
             removed, update the expected count.",
        fingerprint_arg_lines.len(),
        display_lines,
    );
    for &i in &fingerprint_arg_lines {
        let next = lines
            .get(i + 1)
            .unwrap_or_else(|| panic!("no line after fingerprint_line, at {}", i + 1));
        assert_eq!(
            next.trim(),
            "bug_summary_line(),",
            "failure-message format!() at eval/mod.rs:{} passes `fingerprint_line,` \
                 but the next positional argument is `{}` (trimmed), not \
                 `bug_summary_line(),`. The BUG SUMMARY must render at the top \
                 of every failure message so it surfaces above the test-name / \
                 topology line in CI logs.",
            i + 1,
            next.trim(),
        );
        // Arg-list order alone is insufficient: a regression could
        // rearrange the format-string positional indices (e.g.
        // `"{4}{0}{1}..."`) and silently render topo at the top
        // while leaving `fingerprint_line,` + `bug_summary_line(),`
        // as args 0+1 in the call. Walk back from the
        // `fingerprint_line,` arg line to the format-string
        // literal (the previous non-empty source line) and assert
        // it starts with `"{}{}` (default positional indices for
        // args 0 and 1, in order). This pins rendered-output
        // order — the load-bearing invariant — not just call
        // order.
        let fmt_line_idx = (0..i)
            .rev()
            .find(|&k| !lines[k].trim().is_empty())
            .unwrap_or(0);
        let trimmed = lines[fmt_line_idx].trim();
        // Allow an optional `{name}` named-arg prefix before `{}{}`.
        // The named arg renders deterministic context (e.g.
        // `{post_vm_prefix}`) which the failure path may want to
        // surface BEFORE the BUG SUMMARY — what matters for the
        // invariant is that fingerprint_line and bug_summary_line
        // remain adjacent positional args 0+1, not that they are
        // literally the first characters of the rendered output.
        let stripped = strip_named_arg_prefix(trimmed);
        assert!(
            stripped.starts_with("\"{}{}"),
            "failure-message format!() at eval/mod.rs:{} passes `fingerprint_line,` \
                 then `bug_summary_line(),` as args 0+1, but the preceding format \
                 string literal at eval/mod.rs:{} (`{}`) does NOT start with `\"{{}}{{}}` \
                 (default positional indices for args 0+1 in order, optionally \
                 preceded by a single named-arg span like `{{post_vm_prefix}}`). \
                 A regression that reordered the format-string indices (e.g. \
                 `\"{{2}}{{0}}{{1}}...\"`) would render the BUG SUMMARY below \
                 the test-name / topology line even though the args list still \
                 threads `bug_summary_line()` as the second positional.",
            i + 1,
            fmt_line_idx + 1,
            trimmed,
        );
    }
}

/// Pin the literal `BUG SUMMARY:` prefix in BOTH arms of the
/// `bug_summary_line` closure (the `stderr_color()`-true ANSI-colored
/// arm and the plain arm). CI log greps and downstream
/// parsers key on the post-ANSI-strip byte sequence
/// `BUG SUMMARY:`; a regression that renamed the prefix
/// (e.g. `BPF ERROR:`, `BUG_SUMMARY:`, dropped the colon)
/// would silently break those consumers while the positional
/// pin above passes. Scoping the source-scan to the closure
/// body — from the `let bug_summary_line = || -> String {`
/// opener to its matching `};` — isolates both arms from
/// (a) the test's own docstring (which contains the literal
/// for explanatory purposes), (b) sibling tests that assert
/// `reason.contains("BUG SUMMARY:")` on synthesized output,
/// and (c) any future code that mentions the prefix without
/// rendering it.
#[test]
fn bug_summary_line_renders_bug_summary_prefix_in_both_arms() {
    let src = include_str!("mod.rs");
    let lines: Vec<&str> = src.lines().collect();
    let opener_idx = lines
        .iter()
        .position(|l| l.contains("let bug_summary_line = || -> String {"))
        .expect(
            "bug_summary_line closure opener `let bug_summary_line = || -> String {` \
                 must exist in eval/mod.rs",
        );
    let closer_idx = lines
        .iter()
        .enumerate()
        .skip(opener_idx + 1)
        .find(|(_, l)| l.trim() == "};")
        .map(|(i, _)| i)
        .expect("bug_summary_line closure must close with `};` on its own line");
    let body = &lines[opener_idx..=closer_idx];
    let prefix_lines: Vec<usize> = body
        .iter()
        .enumerate()
        .filter_map(|(off, l)| l.contains("BUG SUMMARY:").then_some(opener_idx + off + 1))
        .collect();
    assert_eq!(
        prefix_lines.len(),
        2,
        "expected the literal `BUG SUMMARY:` prefix to appear EXACTLY twice in \
             the `bug_summary_line` closure body (eval/mod.rs:{}-{}): once in the \
             ANSI-colored arm and once in the plain arm. Found {} occurrence(s) \
             at lines {:?}. A regression that renamed either arm's prefix (e.g. \
             `BPF ERROR:`, `BUG_SUMMARY:`, dropped the colon) would break the \
             post-ANSI-strip byte sequence that downstream CI log parsers grep for.",
        opener_idx + 1,
        closer_idx + 1,
        prefix_lines.len(),
        prefix_lines,
    );
}

// -- scx_bpf_error matcher dispatch wiring --

/// Pin the production wiring that surfaces
/// `expect_scx_bpf_error_contains` / `_matches` failures through
/// the test verdict. Three load-bearing source-level invariants:
///
/// 1. `merged_assert.evaluate_scx_bpf_error_match(&matcher_corpus,
///    entry.expect_err)` is dispatched from
///    [`evaluate_vm_result`] (the only production caller; the
///    matcher-fn itself lives on `Assert`).
/// 2. The dispatch is gated by `if matcher_configured` so the
///    common no-matcher path does not allocate the corpus
///    string for every test.
/// 3. When the matcher contributes a mismatch detail, the
///    returned `anyhow::Error` is wrapped with
///    [`ScxBpfErrorMatcherMismatch`] as `.context()`. The
///    dispatch-time `expect_err` inversion checks for this
///    marker via downcast and bypasses the failure-to-pass
///    inversion (see [`crate::test_support::dispatch`]) — a
///    reproducer with a matcher mismatch fails the test even
///    when `expect_err = true`.
///
/// A regression that:
///   - removes the `evaluate_scx_bpf_error_match` dispatch →
///     the matcher never runs; configured matchers produce zero
///     details; `matcher_mismatch` stays false; `expect_err`
///     inversion turns every Err into a pass, silently making
///     positive-matcher tests vacuous.
///   - hardcodes `matcher_configured = false` or removes the
///     `if matcher_configured` gate → matcher_details stays
///     `Vec::new()`; same vacuous outcome as above.
///   - removes `err.context(ScxBpfErrorMatcherMismatch)` or
///     removes the `matcher_mismatch` gate around it → the
///     marker is never attached; `expect_err = true` reproducers
///     would invert matcher-mismatch failures into passes.
///
/// Interim source-pattern coverage; the proper E2E coverage
/// belongs in an `evaluate_vm_result` mock harness.
/// Fragile to source refactors: if production code wraps the
/// dispatch in a helper function, or spells the matcher fn /
/// marker type differently, this test fails even when the
/// wiring's semantics are preserved. Update the patterns or
/// migrate to the E2E mock harness in that case.
#[test]
fn matcher_dispatch_and_mismatch_marker_wiring_pinned() {
    let src = include_str!("mod.rs");
    let lines: Vec<&str> = src.lines().collect();
    // Scan the whole production core: this test lives in
    // eval_tests.rs, so the scanned mod.rs holds no test literals.
    let scan_end = lines.len();
    // Collect production sites of a needle from mod.rs. `find_sites`
    // skips comment lines (`//`, `///`, `//!`) so production
    // docstrings citing identifiers verbatim don't false-positive.
    let find_sites = |needle: &str| -> Vec<usize> {
        lines
            .iter()
            .enumerate()
            .take(scan_end)
            .filter_map(|(i, l)| {
                // Skip all line comments (`//`, `///`, `//!`).
                // Production code is never a line-comment line;
                // excluding them avoids false-positive counts
                // from the test's docstring (citing production
                // identifiers verbatim) and from any future
                // `//` regular comment that mentions a needle.
                if l.trim_start().starts_with("//") {
                    return None;
                }
                l.contains(needle).then_some(i)
            })
            .collect()
    };
    // Assert a production site's 8-line lookback contains the
    // expected gate keyword. 8 lines is enough to capture
    // surrounding `if let`-style binding plus comments.
    let assert_gated = |site: usize, label: &str, gate: &str| {
        let window = lines[site.saturating_sub(8)..=site].join("\n");
        assert!(
            window.contains(gate),
            "{label} (line {}) must be gated by `{gate}`; \
                 8-line lookback window:\n{window}",
            site + 1,
        );
    };
    // Assert a needle has exactly one production occurrence and
    // return the line index. Used 4x — extract to DRY the
    // cardinality-check + iterator-to-1-based-line diagnostic.
    let assert_unique = |sites: &[usize], label: &str| -> usize {
        assert_eq!(
            sites.len(),
            1,
            "expected exactly 1 {label} site; found {} at lines {:?}",
            sites.len(),
            sites.iter().map(|i| i + 1).collect::<Vec<_>>(),
        );
        sites[0]
    };
    let dispatch_site = assert_unique(
        &find_sites("merged_assert.evaluate_scx_bpf_error_match("),
        "matcher dispatch",
    );
    assert_gated(dispatch_site, "matcher dispatch", "if matcher_configured");
    // Pin the inversion-arg pass-through so a regression that
    // hardcodes `true`/`false` for entry.expect_err (compile-OK,
    // runtime regression) is caught.
    assert!(
        lines[dispatch_site].contains("entry.expect_err"),
        "matcher dispatch (line {}) must forward `entry.expect_err` to drive \
             the inversion check; line: {}",
        dispatch_site + 1,
        lines[dispatch_site],
    );
    let marker_site = assert_unique(
        &find_sites("err.context(ScxBpfErrorMatcherMismatch)"),
        "marker attach",
    );
    assert_gated(marker_site, "marker attach", "if matcher_mismatch");
    // Catch the "hardcoded false" regression class for both
    // gate flags: the lookback assertion only checks that the
    // gate keyword appears in source. A regression that leaves
    // the gate intact but hardcodes the flag to `false` (e.g.
    // `let matcher_configured = false;`) would silently bypass
    // the matcher dispatch / marker attach. Pin the assignment
    // shape so the derivation from runtime state is preserved.
    let configured_site = assert_unique(
        &find_sites("let matcher_configured ="),
        "`let matcher_configured =` assignment",
    );
    // 3-line window covers production's multi-line `X.is_some()
    // || Y.is_some()` RHS at the `let matcher_configured =` assignment
    // in eval/mod.rs.
    let configured_window =
        lines[configured_site..=configured_site.saturating_add(2).min(scan_end - 1)].join("\n");
    assert!(
        configured_window.contains("expect_scx_bpf_error_contains.is_some()")
            && configured_window.contains("expect_scx_bpf_error_matches.is_some()"),
        "matcher_configured must derive from the matcher fields' \
             `.is_some()` checks, not a hardcoded literal; assignment window:\n\
             {configured_window}",
    );
    let mismatch_site = assert_unique(
        &find_sites("let matcher_mismatch ="),
        "`let matcher_mismatch =` assignment",
    );
    let mismatch_line = lines[mismatch_site];
    assert!(
        mismatch_line.contains("matcher_details.is_empty()"),
        "matcher_mismatch must derive from `matcher_details.is_empty()`, \
             not a hardcoded literal; assignment line: {mismatch_line}",
    );
}

/// `resolve_staged_schedulers_strict` MUST preserve
/// `entry.staged_schedulers` iteration order in its returned
/// Vec. The future initramfs packer iterates the result to
/// emit per-scheduler `/staging/schedulers/<name>/` archive
/// entries; a silent reorder (e.g. a refactor that uses
/// `.collect::<HashMap<_,_>>().into_iter()`) would silently
/// change initramfs staging layout.
///
/// Uses a synthetic resolver that returns the spec encoded as
/// a path so the order assertion can read back the original
/// order without touching the host filesystem.
#[test]
fn resolve_staged_schedulers_strict_preserves_entry_iteration_order() {
    use crate::test_support::Scheduler;
    static FIRST: Scheduler = Scheduler::named("scx_alpha").binary_discover("scx_alpha_bin");
    static SECOND: Scheduler = Scheduler::named("scx_beta").binary_discover("scx_beta_bin");
    static THIRD: Scheduler = Scheduler::named("scx_gamma").binary_discover("scx_gamma_bin");
    static SCHEDS: &[&Scheduler] = &[&FIRST, &SECOND, &THIRD];
    let entry = crate::test_support::entry::KtstrTestEntry {
        name: "order_pin",
        staged_schedulers: SCHEDS,
        ..crate::test_support::entry::KtstrTestEntry::DEFAULT
    };
    let resolved = resolve_staged_schedulers_strict(&entry, |spec| {
        // Encode the spec as a deterministic synthetic path so
        // the resolver is pure (no FS) and the test can pin
        // both the order AND that the resolver was called per
        // staged entry.
        let key = match spec {
            SchedulerSpec::Discover(s) => s.to_string(),
            _ => "unexpected_variant".to_string(),
        };
        Ok(Some(PathBuf::from(format!("/synthetic/{key}"))))
    })
    .expect("strict resolver succeeds on synthetic happy path");

    let names: Vec<&str> = resolved.iter().map(|(n, _, _)| n.as_str()).collect();
    assert_eq!(
        names,
        vec!["scx_alpha", "scx_beta", "scx_gamma"],
        "resolution MUST preserve entry.staged_schedulers declaration order; \
             a future refactor that collects via HashMap would silently scramble \
             initramfs staging layout"
    );
    let paths: Vec<String> = resolved
        .iter()
        .map(|(_, p, _)| p.display().to_string())
        .collect();
    assert_eq!(
        paths,
        vec![
            "/synthetic/scx_alpha_bin",
            "/synthetic/scx_beta_bin",
            "/synthetic/scx_gamma_bin",
        ],
        "synthetic resolver paths must align with iteration order — \
             confirms the per-entry resolver call happens in declaration order"
    );
}

/// `resolve_staged_schedulers_strict` drops entries whose
/// resolver returns `Ok(None)` — matches the
/// `KernelBuiltin` / `Eevdf` semantic (no binary to stage).
/// Pins the silent-drop behavior so a future refactor that
/// changes the dropped-entry handling (e.g. bails on None
/// instead of skipping) surfaces here.
#[test]
fn resolve_staged_schedulers_strict_skips_resolver_none() {
    use crate::test_support::Scheduler;
    static BINARY: Scheduler = Scheduler::named("scx_real").binary_discover("scx_real_bin");
    static BUILTIN: Scheduler = Scheduler::named("scx_builtin").binary_discover("scx_skip");
    static SCHEDS: &[&Scheduler] = &[&BINARY, &BUILTIN];
    let entry = crate::test_support::entry::KtstrTestEntry {
        name: "none_skip",
        staged_schedulers: SCHEDS,
        ..crate::test_support::entry::KtstrTestEntry::DEFAULT
    };
    let resolved = resolve_staged_schedulers_strict(&entry, |spec| match spec {
        SchedulerSpec::Discover("scx_skip") => Ok(None),
        SchedulerSpec::Discover(s) => Ok(Some(PathBuf::from(format!("/synthetic/{s}")))),
        _ => Ok(None),
    })
    .expect("strict resolver succeeds; None entries are dropped not errored");
    assert_eq!(resolved.len(), 1);
    assert_eq!(resolved[0].0, "scx_real");
}

/// `resolve_staged_schedulers_strict` propagates resolver
/// errors (vs the auto-repro path's log-and-skip). Pins the
/// strict semantic against a refactor that softens to
/// log-and-skip — primary-path staging failure MUST surface at
/// dispatch time, not silently degrade to "Op::AttachScheduler
/// will fail later inside the VM".
#[test]
fn resolve_staged_schedulers_strict_propagates_resolver_error() {
    use crate::test_support::Scheduler;
    static SCHED: Scheduler = Scheduler::named("scx_fail").binary_discover("scx_fail_bin");
    static SCHEDS: &[&Scheduler] = &[&SCHED];
    let entry = crate::test_support::entry::KtstrTestEntry {
        name: "err_propagate",
        staged_schedulers: SCHEDS,
        ..crate::test_support::entry::KtstrTestEntry::DEFAULT
    };
    let err = resolve_staged_schedulers_strict(&entry, |_spec| {
        Err::<Option<PathBuf>, _>(anyhow::anyhow!(
            "synthetic resolver error — staged binary not found on host"
        ))
    })
    .expect_err("strict resolver must propagate error, not swallow");
    assert!(
        err.to_string().contains("synthetic resolver error"),
        "error chain must preserve resolver's message, got: {err:#}"
    );
}

// -- apply_expect_auto_repro_inversion tests --
//
// Pin the gate matrix for the eval-layer helper that derives
// `result.expect_auto_repro_satisfied`. Each test exercises one
// bail arm or the satisfaction arm in isolation so a regression
// in a single condition surfaces by name. The dispatch-side
// verdict flip (Err → EXIT_PASS via the `ExpectAutoReproSatisfied`
// marker) is exercised separately at the dispatch.rs layer.

/// Build a `KtstrTestEntry` with `expect_auto_repro = true`
/// bound to the scx-style `SCHED_TEST` fixture. Mirrors
/// `sched_entry` but flips the field under test; tests that need
/// `expect_auto_repro = false` use `sched_entry(...)` directly.
#[cfg(feature = "wprof")]
fn expect_auto_repro_entry(name: &'static str) -> KtstrTestEntry {
    KtstrTestEntry {
        expect_auto_repro: true,
        ..sched_entry(name)
    }
}

/// Build a `VmResult` with `success = false` and the supplied
/// `entry_name`. Mirrors the prod call site in
/// `run_ktstr_test_inner_impl` where the helper sees a failed
/// VM with the macro-stamped entry name.
#[cfg(feature = "wprof")]
fn failing_vm_result_with_name(name: &'static str) -> crate::vmm::VmResult {
    crate::vmm::VmResult {
        success: false,
        entry_name: Some(name),
        ..crate::vmm::VmResult::test_fixture()
    }
}

#[cfg(feature = "wprof")]
fn write_valid_repro_artifact(sidecar_dir: &std::path::Path, name: &str) {
    use crate::test_support::wprof::{PERFETTO_TRACE_PACKETS_TAG, WPROF_PB_MIN_BYTES};
    let mut bytes = vec![PERFETTO_TRACE_PACKETS_TAG];
    bytes.resize(WPROF_PB_MIN_BYTES, 0);
    let path = sidecar_dir.join(format!("{name}.repro.wprof.pb"));
    std::fs::write(&path, &bytes).expect("write valid repro artifact");
}

#[cfg(feature = "wprof")]
#[test]
fn apply_expect_auto_repro_inversion_no_op_when_attr_unset() {
    let _lock = lock_env();
    let dir = TempDir::new().expect("tempdir");
    let _env = EnvVarGuard::set(
        crate::KTSTR_SIDECAR_DIR_ENV,
        dir.path().to_str().expect("utf8 tempdir"),
    );
    let entry = sched_entry("attr_unset");
    assert!(!entry.expect_auto_repro, "fixture must leave attr false");
    write_valid_repro_artifact(dir.path(), "attr_unset");
    let mut result = failing_vm_result_with_name("attr_unset");
    apply_expect_auto_repro_inversion(&entry, &mut result);
    assert!(
        !result.expect_auto_repro_satisfied,
        "attr-unset run must leave field false even when artifact is shape-valid"
    );
}

#[cfg(feature = "wprof")]
#[test]
fn apply_expect_auto_repro_inversion_no_op_when_success_true() {
    let _lock = lock_env();
    let dir = TempDir::new().expect("tempdir");
    let _env = EnvVarGuard::set(
        crate::KTSTR_SIDECAR_DIR_ENV,
        dir.path().to_str().expect("utf8 tempdir"),
    );
    let entry = expect_auto_repro_entry("success_true");
    write_valid_repro_artifact(dir.path(), "success_true");
    let mut result = crate::vmm::VmResult {
        success: true,
        entry_name: Some("success_true"),
        ..crate::vmm::VmResult::test_fixture()
    };
    apply_expect_auto_repro_inversion(&entry, &mut result);
    assert!(
        !result.expect_auto_repro_satisfied,
        "success=true run must leave field false even when artifact is shape-valid"
    );
}

#[cfg(feature = "wprof")]
#[test]
fn apply_expect_auto_repro_inversion_no_op_when_entry_name_none() {
    let _lock = lock_env();
    let dir = TempDir::new().expect("tempdir");
    let _env = EnvVarGuard::set(
        crate::KTSTR_SIDECAR_DIR_ENV,
        dir.path().to_str().expect("utf8 tempdir"),
    );
    let entry = expect_auto_repro_entry("entry_name_none");
    let mut result = crate::vmm::VmResult {
        success: false,
        entry_name: None,
        ..crate::vmm::VmResult::test_fixture()
    };
    apply_expect_auto_repro_inversion(&entry, &mut result);
    assert!(
        !result.expect_auto_repro_satisfied,
        "entry_name=None must trip the path-resolve bail and leave field false"
    );
}

#[cfg(feature = "wprof")]
#[test]
fn apply_expect_auto_repro_inversion_no_op_when_artifact_missing() {
    let _lock = lock_env();
    let dir = TempDir::new().expect("tempdir");
    let _env = EnvVarGuard::set(
        crate::KTSTR_SIDECAR_DIR_ENV,
        dir.path().to_str().expect("utf8 tempdir"),
    );
    let entry = expect_auto_repro_entry("artifact_missing");
    let mut result = failing_vm_result_with_name("artifact_missing");
    apply_expect_auto_repro_inversion(&entry, &mut result);
    assert!(
        !result.expect_auto_repro_satisfied,
        "missing artifact must trip the shape-check bail and leave field false"
    );
}

#[cfg(feature = "wprof")]
#[test]
fn apply_expect_auto_repro_inversion_no_op_when_artifact_truncated() {
    use crate::test_support::wprof::{PERFETTO_TRACE_PACKETS_TAG, WPROF_PB_MIN_BYTES};
    let _lock = lock_env();
    let dir = TempDir::new().expect("tempdir");
    let _env = EnvVarGuard::set(
        crate::KTSTR_SIDECAR_DIR_ENV,
        dir.path().to_str().expect("utf8 tempdir"),
    );
    let entry = expect_auto_repro_entry("artifact_truncated");
    let mut bytes = vec![PERFETTO_TRACE_PACKETS_TAG];
    bytes.resize(WPROF_PB_MIN_BYTES - 1, 0);
    std::fs::write(dir.path().join("artifact_truncated.repro.wprof.pb"), &bytes)
        .expect("write truncated artifact");
    let mut result = failing_vm_result_with_name("artifact_truncated");
    apply_expect_auto_repro_inversion(&entry, &mut result);
    assert!(
        !result.expect_auto_repro_satisfied,
        "truncated artifact must trip the size gate and leave field false"
    );
}

#[cfg(feature = "wprof")]
#[test]
fn apply_expect_auto_repro_inversion_no_op_when_artifact_wrong_tag() {
    use crate::test_support::wprof::WPROF_PB_MIN_BYTES;
    let _lock = lock_env();
    let dir = TempDir::new().expect("tempdir");
    let _env = EnvVarGuard::set(
        crate::KTSTR_SIDECAR_DIR_ENV,
        dir.path().to_str().expect("utf8 tempdir"),
    );
    let entry = expect_auto_repro_entry("artifact_wrong_tag");
    let mut bytes = vec![0xff]; // any byte != PERFETTO_TRACE_PACKETS_TAG
    bytes.resize(WPROF_PB_MIN_BYTES, 0);
    std::fs::write(dir.path().join("artifact_wrong_tag.repro.wprof.pb"), &bytes)
        .expect("write wrong-tag artifact");
    let mut result = failing_vm_result_with_name("artifact_wrong_tag");
    apply_expect_auto_repro_inversion(&entry, &mut result);
    assert!(
        !result.expect_auto_repro_satisfied,
        "wrong-tag artifact must trip the tag gate and leave field false"
    );
}

#[cfg(feature = "wprof")]
#[test]
fn apply_expect_auto_repro_inversion_sets_field_on_valid_artifact() {
    let _lock = lock_env();
    let dir = TempDir::new().expect("tempdir");
    let _env = EnvVarGuard::set(
        crate::KTSTR_SIDECAR_DIR_ENV,
        dir.path().to_str().expect("utf8 tempdir"),
    );
    let entry = expect_auto_repro_entry("valid_artifact");
    write_valid_repro_artifact(dir.path(), "valid_artifact");
    let mut result = failing_vm_result_with_name("valid_artifact");
    apply_expect_auto_repro_inversion(&entry, &mut result);
    assert!(
        result.expect_auto_repro_satisfied,
        "shape-valid artifact + every upstream gate satisfied must set field true"
    );
}
