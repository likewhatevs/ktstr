//! Unit tests for [`super`] (the `test_support::probe` module).
//! Co-located via the `tests` submodule pattern (sibling file).

#![cfg(test)]

use super::*;

/// Process-wide serialization lock for every test that touches the
/// global [`DEFERRED_PROBE_COLLECT`] static. Tests run in parallel
/// within one process, so a stash from one test interleaving with a
/// drain-and-assert in another would poison the assertion. Every
/// deferred-probe test acquires this single lock for the duration of
/// its stash/take exercises so the ordering is well defined.
static DEFERRED_PROBE_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// The auto-repro VM's host deadline must include PROBE_DRAIN_GRACE beyond the
/// base workload timeout, so the watchdog cannot fire during the post-trigger
/// probe-drain tail and truncate the captured-arg payload (the repro
/// arg-capture race). Pins
/// the budget = vm_timeout_from_entry(entry) + PROBE_DRAIN_GRACE; a regression
/// that drops the grace (repro VM reusing the bare base timeout) trips here.
#[test]
fn repro_vm_builder_adds_probe_drain_grace_to_deadline() {
    use std::path::Path;
    let entry = crate::test_support::test_helpers::eevdf_entry("repro-grace-test");
    let (builder, _dump_path) = build_repro_vm_builder(
        &entry,
        Path::new("/dummy/kernel"),
        None,
        Path::new("/dummy/ktstr"),
        None,
        &[],
    )
    .expect("repro builder builds for a no-wprof EEVDF entry");
    let base = crate::test_support::runtime::vm_timeout_from_entry(&entry, entry.topology.total_cpus());
    assert_eq!(
        builder.timeout,
        base + PROBE_DRAIN_GRACE,
        "repro VM deadline must be the base workload timeout plus the probe-drain grace",
    );
    assert!(
        builder.timeout > base,
        "the grace strictly extends the deadline beyond the workload budget",
    );
}

#[test]
fn extract_probe_output_valid_json() {
    use crate::probe::process::ProbeEvent;
    let payload = ProbeBytes {
        events: vec![ProbeEvent {
            func_idx: 0,
            task_ptr: 1,
            ts: 100,
            args: [0; 6],
            fields: vec![("p:task_struct.pid".to_string(), 42)],
            kstack: vec![],
            str_val: None,
            ..Default::default()
        }],
        func_names: vec![(0, "schedule".to_string())],
        bpf_source_locs: Default::default(),
        diagnostics: None,
        nr_cpus: None,
        param_names: Default::default(),
        render_hints: Default::default(),
    };
    let json = serde_json::to_string(&payload).unwrap();
    let output = format!("noise\n{PROBE_OUTPUT_START}\n{json}\n{PROBE_OUTPUT_END}\nmore");
    let parsed = extract_probe_output(&output, None, None);
    assert!(parsed.is_some());
    let formatted = parsed.unwrap();
    assert!(
        formatted.contains("schedule"),
        "should contain func name: {formatted}"
    );
    assert!(
        formatted.contains("pid"),
        "should contain field name: {formatted}"
    );
}

#[test]
fn extract_probe_output_missing() {
    assert!(extract_probe_output("no markers", None, None).is_none());
}

#[test]
fn extract_probe_output_empty() {
    let output = format!("{PROBE_OUTPUT_START}\n\n{PROBE_OUTPUT_END}");
    assert!(extract_probe_output(&output, None, None).is_none());
}

#[test]
fn extract_probe_output_invalid_json() {
    // Non-EOF malformed input: `recover_partial_events` finds no
    // `"events":` key so recovery yields nothing and the function
    // returns None. The raw bytes still get written to
    // `partial_dump_path` when one is supplied, which the
    // truncated-input tests below exercise.
    let output = format!("{PROBE_OUTPUT_START}\nnot valid json\n{PROBE_OUTPUT_END}");
    assert!(extract_probe_output(&output, None, None).is_none());
}

/// `exit_code_for_result` maps the 4-state verdict lattice
/// (`Fail > Inconclusive > Pass > Skip`) to 3 distinct probe-
/// dispatch exit codes: Pass → 0, Inconclusive → 2, every other
/// state (Fail, Skip, mixed Fail+Inconclusive) → 1. The third
/// code lets CI tooling triage zero-denominator probe runs as
/// "couldn't measure" rather than conflating them with real
/// probe failures. A regression that mapped Inconclusive into
/// the Pass branch would silently let a probe scenario whose
/// gate could not evaluate slip past the dispatch caller's
/// failure signal.
#[test]
fn exit_code_for_result_pass_inconc_fail_skip_lattice() {
    use crate::assert::{AssertDetail, AssertResult, DetailKind};
    // Pass → 0
    assert_eq!(exit_code_for_result(&AssertResult::pass()), 0);
    // Inconclusive → 2
    let inc =
        AssertResult::inconclusive(AssertDetail::new(DetailKind::Benchmark, "zero-denominator"));
    assert_eq!(exit_code_for_result(&inc), 2);
    // Fail → 1
    let fail = AssertResult::fail(AssertDetail::new(DetailKind::Other, "real failure"));
    assert_eq!(exit_code_for_result(&fail), 1);
    // Skip → 1 (the probe path treats a skip as "no signal —
    // surface the failure code for the dispatch caller")
    assert_eq!(exit_code_for_result(&AssertResult::skip("no signal")), 1);
    // Fail + Inconclusive → 1 (Fail dominates per the lattice)
    let mut fail_plus_inc =
        AssertResult::fail(AssertDetail::new(DetailKind::Other, "real failure first"));
    fail_plus_inc.record_inconclusive(AssertDetail::new(
        DetailKind::Benchmark,
        "and a zero-denom gate",
    ));
    assert_eq!(exit_code_for_result(&fail_plus_inc), 1);
    // Inconclusive + Pass → 2 (Inconclusive dominates Pass)
    let mut inc_plus_pass =
        AssertResult::inconclusive(AssertDetail::new(DetailKind::Benchmark, "zero-denom"));
    inc_plus_pass.record_pass();
    assert_eq!(exit_code_for_result(&inc_plus_pass), 2);
}

/// Process-wide serialization lock for tests that mutate the global
/// scheduler-liveness signals — `SCHED_PID`
/// ([`set_sched_pid`](crate::vmm::rust_init::set_sched_pid)) and the
/// `TEST_SCX_STATE` scx-state override. Mirrors [`DEFERRED_PROBE_TEST_LOCK`]:
/// tests run in parallel within one process, so a test that flips these
/// globals holds this lock and resets them on the way out so a peer test
/// never observes a poisoned signal.
static SCHED_LIVENESS_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// RAII reset for the scheduler-liveness globals: restores `SCHED_PID` to the
/// unset sentinel and clears the scx-state override on drop, even if the test
/// panics mid-way.
struct SchedLivenessGuard;
impl Drop for SchedLivenessGuard {
    fn drop(&mut self) {
        crate::vmm::rust_init::set_sched_pid(0);
        crate::scenario::ops::set_test_scx_state(None);
    }
}

/// `enforce_survives_storm_liveness` is the guest-side post-function probe that
/// closes the `survives_storm` vacuous-pass hole for scenarios which hand-roll
/// `Op` dispatch without an `execute_*` driver (so no in-hold liveness probe
/// runs). Host-runnable — it drives the `TEST_SCX_STATE` override plus a
/// synthetic `SCHED_PID`, so CI without `/dev/kvm` actively proves the fold
/// logic rather than relying on the VM e2e (which SKIPs there; see the
/// "skipping e2e masks bugs" lesson).
///
/// Pins both levers of the helper's `!process_alive(pid) || scx_down()` OR and
/// the pid-unset short-circuit. Cases 1-4 hold our own (live, signalable) pid
/// so `process_alive` reads true and the scx-state override is the sole lever;
/// case 5 clears the pid (the short-circuit); case 6 uses a nonexistent pid to
/// isolate the `!process_alive` lever while scx reads `enabled`:
/// 1. `survives_storm` off → never folds, even with the scheduler down.
/// 2. scheduler healthy (scx `enabled`, pid alive) → no fold; stays a pass.
/// 3. scheduler down (scx `disabling`, pid alive) → folds exactly one
///    `Scheduler*` fail, flipping `is_pass()` to false — the hole this closes.
/// 4. a result already carrying a `Scheduler*` detail (the `execute_*` in-hold
///    probe path) → the guard skips, so no double-record.
/// 5. no scheduler expected (`sched_pid` unset, e.g. a clean detach) → no fold
///    even when scx reads down.
/// 6. scheduler process gone (a nonexistent pid) while scx still reads
///    `enabled` → folds via the `!process_alive` lever alone.
#[test]
fn enforce_survives_storm_liveness_folds_only_on_unattributed_death() {
    use crate::assert::{AssertDetail, AssertResult, DetailKind};
    use crate::scenario::ops::{ScxState, set_test_scx_state};
    use crate::vmm::rust_init::set_sched_pid;

    let _lock = SCHED_LIVENESS_TEST_LOCK.lock().unwrap();
    let _reset = SchedLivenessGuard;

    let self_pid = std::process::id() as libc::pid_t;
    let scheduler_fail_count = |r: &AssertResult| {
        r.failure_details()
            .filter(|d| {
                matches!(
                    d.kind,
                    DetailKind::SchedulerCrashed
                        | DetailKind::SchedulerExitedCleanly
                        | DetailKind::SchedulerDiedUnknownReason
                )
            })
            .count()
    };

    // (1) survives_storm off → never folds, even with the scheduler down.
    set_sched_pid(self_pid);
    set_test_scx_state(Some(ScxState::Disabling));
    let mut r = AssertResult::pass();
    enforce_survives_storm_liveness(&mut r, false);
    assert!(r.is_pass(), "survives_storm off must not fold a fail");
    assert_eq!(scheduler_fail_count(&r), 0);

    // (2) scheduler healthy (enabled, pid alive) → no fold.
    set_test_scx_state(Some(ScxState::Enabled));
    let mut r = AssertResult::pass();
    enforce_survives_storm_liveness(&mut r, true);
    assert!(r.is_pass(), "a surviving scheduler must stay a pass");
    assert_eq!(scheduler_fail_count(&r), 0);

    // (3) scheduler down (disabling) with a live pid → folds exactly one
    //     Scheduler* fail onto an otherwise-passing result. The exact variant
    //     is left to sched_died_detail_kind (a separate global the BPF latch
    //     drives); this pins only that a death was recorded.
    set_test_scx_state(Some(ScxState::Disabling));
    let mut r = AssertResult::pass();
    enforce_survives_storm_liveness(&mut r, true);
    assert!(
        !r.is_pass(),
        "a downed scheduler must fail the survives_storm run"
    );
    assert_eq!(
        scheduler_fail_count(&r),
        1,
        "exactly one Scheduler* fail folded onto the passing result"
    );

    // (4) a result already carrying a Scheduler* detail (an execute_* in-hold
    //     probe recorded it) → guard skips, no double-record.
    set_test_scx_state(Some(ScxState::Disabled));
    let mut r = AssertResult::pass();
    r.record_fail(AssertDetail::new(
        DetailKind::SchedulerCrashed,
        "recorded by the execute_* in-hold probe",
    ));
    enforce_survives_storm_liveness(&mut r, true);
    assert_eq!(
        scheduler_fail_count(&r),
        1,
        "must not double-record when a Scheduler* detail is already present"
    );
    assert!(
        r.failure_details()
            .any(|d| d.kind == DetailKind::SchedulerCrashed),
        "the pre-existing crash detail is preserved"
    );

    // (5) no scheduler expected (sched_pid unset, e.g. a clean detach) → no
    //     fold even when scx reads down.
    set_sched_pid(0);
    set_test_scx_state(Some(ScxState::Disabled));
    let mut r = AssertResult::pass();
    enforce_survives_storm_liveness(&mut r, true);
    assert!(
        r.is_pass(),
        "an unset sched_pid (clean detach) must not trip the probe"
    );
    assert_eq!(scheduler_fail_count(&r), 0);

    // (6) scheduler process gone (pid set to a nonexistent pid) while scx still
    //     reads enabled → folds via the `!process_alive` lever ALONE, the OR's
    //     other branch (cases 1-5 only exercise scx_down). pid_t::MAX is
    //     reliably unused (mirrors scenario::tests::process_alive_nonexistent_pid).
    set_sched_pid(libc::pid_t::MAX);
    set_test_scx_state(Some(ScxState::Enabled));
    let mut r = AssertResult::pass();
    enforce_survives_storm_liveness(&mut r, true);
    assert!(
        !r.is_pass(),
        "a dead scheduler process must fail even when scx still reads enabled"
    );
    assert_eq!(
        scheduler_fail_count(&r),
        1,
        "the !process_alive lever alone must fold exactly one Scheduler* fail"
    );
}

#[test]
fn extract_probe_output_enriched_fields() {
    use crate::probe::process::ProbeEvent;
    let payload = ProbeBytes {
        events: vec![
            ProbeEvent {
                func_idx: 0,
                task_ptr: 1,
                ts: 100,
                args: [0xDEAD, 0, 0, 0, 0, 0],
                fields: vec![
                    ("prev:task_struct.pid".to_string(), 42),
                    ("prev:task_struct.scx_flags".to_string(), 0x1c),
                ],
                kstack: vec![],
                str_val: None,
                ..Default::default()
            },
            ProbeEvent {
                func_idx: 1,
                task_ptr: 1,
                ts: 200,
                args: [0; 6],
                fields: vec![("rq:rq.cpu".to_string(), 3)],
                kstack: vec![],
                str_val: None,
                ..Default::default()
            },
        ],
        func_names: vec![
            (0, "schedule".to_string()),
            (1, "pick_task_scx".to_string()),
        ],
        bpf_source_locs: Default::default(),
        diagnostics: None,
        nr_cpus: None,
        param_names: Default::default(),
        render_hints: Default::default(),
    };
    let json = serde_json::to_string(&payload).unwrap();
    let output = format!("{PROBE_OUTPUT_START}\n{json}\n{PROBE_OUTPUT_END}");
    let formatted = extract_probe_output(&output, None, None).unwrap();

    // Decoded fields present (not raw args).
    assert!(formatted.contains("pid"), "pid field: {formatted}");
    assert!(formatted.contains("42"), "pid value: {formatted}");
    assert!(
        formatted.contains("scx_flags"),
        "scx_flags field: {formatted}"
    );
    assert!(formatted.contains("cpu"), "cpu field: {formatted}");
    assert!(formatted.contains("3"), "cpu value: {formatted}");

    // Type header grouping for struct params.
    assert!(
        formatted.contains("task_struct *prev"),
        "type header for task_struct: {formatted}"
    );
    assert!(
        formatted.contains("rq *rq"),
        "type header for rq: {formatted}"
    );

    // Raw args suppressed when fields present.
    assert!(
        !formatted.contains("arg0"),
        "raw args should not appear when fields exist: {formatted}"
    );

    // Function names present.
    assert!(formatted.contains("schedule"), "func schedule: {formatted}");
    assert!(
        formatted.contains("pick_task_scx"),
        "func pick_task_scx: {formatted}"
    );
}

// -- truncated probe payload recovery --
//
// When the repro VM dies before the probe emitter finishes its
// `println!` of the JSON payload, COM2 captures a half-written
// object and `PROBE_OUTPUT_END` is missing entirely.
// `extract_section` returns the truncated JSON (stops at end-of-
// buffer when no end sentinel is present). These tests pin
// `parse_probe_payload` / `recover_partial_events` /
// `find_balanced_object_end` against that wire reality.

/// Build a `ProbeBytes` JSON serialization, then truncate it at
/// `cut_after` bytes. Used to manufacture the wire shape the bug
/// report describes ("EOF while parsing a string at line 1
/// column N").
fn truncate_probe_json(payload: &ProbeBytes, cut_after: usize) -> String {
    let json = serde_json::to_string(payload).expect("serialize ProbeBytes");
    json[..cut_after.min(json.len())].to_string()
}

#[test]
fn extract_probe_output_truncated_recovers_complete_events() {
    use crate::probe::process::ProbeEvent;
    // Three complete events, then we'll lop off the rest of the
    // payload mid-event so only the first two recover cleanly
    // (the third event gets cut while serializing its kstack
    // string).
    let payload = ProbeBytes {
        events: vec![
            ProbeEvent {
                func_idx: 0,
                task_ptr: 0xa,
                ts: 100,
                args: [0; 6],
                fields: vec![("p:task_struct.pid".to_string(), 11)],
                kstack: vec![],
                str_val: None,
                ..Default::default()
            },
            ProbeEvent {
                func_idx: 1,
                task_ptr: 0xb,
                ts: 200,
                args: [0; 6],
                fields: vec![("p:task_struct.pid".to_string(), 22)],
                kstack: vec![],
                str_val: None,
                ..Default::default()
            },
            ProbeEvent {
                func_idx: 2,
                task_ptr: 0xc,
                ts: 300,
                args: [0; 6],
                fields: vec![("p:task_struct.pid".to_string(), 33)],
                kstack: vec![],
                str_val: None,
                ..Default::default()
            },
        ],
        func_names: vec![
            (0, "first".to_string()),
            (1, "second".to_string()),
            (2, "third".to_string()),
        ],
        bpf_source_locs: Default::default(),
        diagnostics: None,
        nr_cpus: None,
        param_names: Default::default(),
        render_hints: Default::default(),
    };
    let full = serde_json::to_string(&payload).unwrap();
    // Find the start of the third event's `{` and truncate
    // halfway into it. We locate the third event by counting
    // top-level `{` occurrences inside `"events":[...]`.
    let events_start = full.find("\"events\":[").unwrap() + "\"events\":[".len();
    let mut depth: u32 = 0;
    let mut in_string = false;
    let mut escape = false;
    let mut event_starts: Vec<usize> = Vec::new();
    for (i, b) in full.bytes().enumerate().skip(events_start) {
        if in_string {
            if escape {
                escape = false;
            } else if b == b'\\' {
                escape = true;
            } else if b == b'"' {
                in_string = false;
            }
            continue;
        }
        match b {
            b'"' => in_string = true,
            b'{' => {
                if depth == 0 {
                    event_starts.push(i);
                }
                depth += 1;
            }
            b'}' => depth = depth.saturating_sub(1),
            b']' if depth == 0 => break,
            _ => {}
        }
    }
    assert_eq!(
        event_starts.len(),
        3,
        "test fixture should produce 3 events"
    );
    // Cut a few bytes into the third event's body. That kills
    // the third event AND everything after (including the
    // closing `]` and trailing fields), so recovery must yield
    // exactly the first two complete events.
    let cut = event_starts[2] + 5;
    let truncated = &full[..cut];
    let output = format!("{PROBE_OUTPUT_START}\n{truncated}\n");
    // No PROBE_OUTPUT_END — exactly what the buggy path sees.
    let dir = tempfile::tempdir().expect("tempdir");
    let partial = dir.path().join("payload.partial.json");
    let formatted = extract_probe_output(&output, None, Some(&partial))
        .expect("recovery must surface partial events");
    // First two events' func indices are 0 and 1; the recovery
    // path leaves func_names empty, so the formatter falls back
    // to the literal `unknown` for every event header.
    assert!(
        formatted.contains("unknown"),
        "recovered events should print under `unknown` func header (no func_names): {formatted}",
    );
    assert!(
        formatted.contains("11"),
        "first event's pid value should appear: {formatted}",
    );
    assert!(
        formatted.contains("22"),
        "second event's pid value should appear: {formatted}",
    );
    assert!(
        !formatted.contains("33"),
        "third (truncated) event's pid value must NOT appear: {formatted}",
    );
    // Partial dump file written.
    assert!(
        partial.exists(),
        "raw truncated payload must be written to partial dump path",
    );
    let dumped = std::fs::read_to_string(&partial).expect("read partial dump");
    // Dumped bytes match what extract_section returned — that's
    // the trimmed json between START and end-of-buffer.
    assert_eq!(
        dumped.trim(),
        truncated.trim(),
        "partial dump must contain the raw extracted JSON verbatim",
    );
}

#[test]
fn extract_probe_output_truncated_no_complete_events_returns_none() {
    // Truncate inside the FIRST event's body. No complete events
    // recoverable, so `parse_probe_payload` returns None.
    // `extract_probe_output` therefore returns None as well —
    // BUT the partial dump file still gets written, since the
    // bug report's "operator can inspect manually" requirement
    // applies even when no events survive.
    use crate::probe::process::ProbeEvent;
    let payload = ProbeBytes {
        events: vec![ProbeEvent {
            func_idx: 0,
            task_ptr: 0xa,
            ts: 100,
            args: [0; 6],
            fields: vec![("p:task_struct.pid".to_string(), 11)],
            kstack: vec![],
            str_val: None,
            ..Default::default()
        }],
        func_names: vec![(0, "first".to_string())],
        bpf_source_locs: Default::default(),
        diagnostics: None,
        nr_cpus: None,
        param_names: Default::default(),
        render_hints: Default::default(),
    };
    let full = serde_json::to_string(&payload).unwrap();
    // Truncate halfway through the events array (after `[` plus
    // a few bytes into the first event).
    let events_open = full.find("\"events\":[").unwrap() + "\"events\":[".len();
    let cut = events_open + 5;
    let truncated = &full[..cut];
    let output = format!("{PROBE_OUTPUT_START}\n{truncated}\n");
    let dir = tempfile::tempdir().expect("tempdir");
    let partial = dir.path().join("payload.partial.json");
    let result = extract_probe_output(&output, None, Some(&partial));
    assert!(
        result.is_none(),
        "no complete events recoverable should yield None: {result:?}",
    );
    assert!(
        partial.exists(),
        "partial dump must be written even when recovery yields zero events",
    );
    let dumped = std::fs::read_to_string(&partial).expect("read partial dump");
    assert_eq!(dumped.trim(), truncated.trim());
}

#[test]
fn extract_probe_output_truncated_string_value_recovers_prior() {
    // The bug report's exact failure: "EOF while parsing a string
    // at line 1 column 1078". Truncate inside a string value so
    // serde_json reports `EofWhileParsingString`. The events
    // before the truncated string still recover.
    use crate::probe::process::ProbeEvent;
    let payload = ProbeBytes {
        events: vec![
            ProbeEvent {
                func_idx: 0,
                task_ptr: 0xa,
                ts: 100,
                args: [0; 6],
                fields: vec![("p:task_struct.pid".to_string(), 11)],
                kstack: vec![],
                str_val: None,
                ..Default::default()
            },
            // Long string value: a partially-written `str_val`
            // is the realistic shape a mid-`println!` truncation
            // produces.
            ProbeEvent {
                func_idx: 1,
                task_ptr: 0xb,
                ts: 200,
                args: [0; 6],
                fields: vec![],
                kstack: vec![],
                str_val: Some("a".repeat(200)),
                ..Default::default()
            },
        ],
        func_names: vec![(0, "first".to_string()), (1, "second".to_string())],
        bpf_source_locs: Default::default(),
        diagnostics: None,
        nr_cpus: None,
        param_names: Default::default(),
        render_hints: Default::default(),
    };
    let full = serde_json::to_string(&payload).unwrap();
    // Truncate inside the long `aaa...` string of the second
    // event. We locate the run of `a` characters and chop near
    // its midpoint.
    let needle = "aaaaaaaaaa"; // first 10 chars of the str_val
    let idx = full
        .find(needle)
        .expect("fixture must contain the long string");
    let cut = idx + needle.len() + 50; // ~50 `a`s into the string
    let truncated = &full[..cut];
    let output = format!("{PROBE_OUTPUT_START}\n{truncated}\n");
    let formatted = extract_probe_output(&output, None, None)
        .expect("recovery must surface the first complete event");
    assert!(
        formatted.contains("11"),
        "first event's pid must survive truncation in the second event: {formatted}",
    );
}

#[test]
fn extract_probe_output_truncated_partial_dump_path_none_skips_write() {
    // When the caller passes None for partial_dump_path, no file
    // gets written — and no panic / error. The recovery path
    // still surfaces partial events.
    use crate::probe::process::ProbeEvent;
    let payload = ProbeBytes {
        events: vec![
            ProbeEvent {
                func_idx: 0,
                task_ptr: 0xa,
                ts: 100,
                args: [0; 6],
                fields: vec![("p:task_struct.pid".to_string(), 11)],
                kstack: vec![],
                str_val: None,
                ..Default::default()
            },
            ProbeEvent {
                func_idx: 1,
                task_ptr: 0xb,
                ts: 200,
                args: [0; 6],
                fields: vec![],
                kstack: vec![],
                str_val: None,
                ..Default::default()
            },
        ],
        func_names: vec![(0, "f0".to_string()), (1, "f1".to_string())],
        bpf_source_locs: Default::default(),
        diagnostics: None,
        nr_cpus: None,
        param_names: Default::default(),
        render_hints: Default::default(),
    };
    let full = serde_json::to_string(&payload).unwrap();
    // Truncate after the close-brace of the first event, mid-
    // serializing the second.
    let first_close = full.find("},{").expect("two events present");
    let cut = first_close + 3; // include the `,{` so recovery sees a starting `{`
    let truncated = &full[..cut];
    let output = format!("{PROBE_OUTPUT_START}\n{truncated}\n");
    let formatted = extract_probe_output(&output, None, None)
        .expect("recovery must yield the first event without a dump path");
    assert!(
        formatted.contains("11"),
        "first event recovered: {formatted}"
    );
}

#[test]
fn parse_probe_payload_strict_success() {
    // Direct unit test for the helper: a complete payload
    // round-trips without touching the recovery path.
    use crate::probe::process::ProbeEvent;
    let payload = ProbeBytes {
        events: vec![ProbeEvent {
            func_idx: 0,
            task_ptr: 0xa,
            ts: 100,
            args: [0; 6],
            fields: vec![],
            kstack: vec![],
            str_val: None,
            ..Default::default()
        }],
        func_names: vec![(0, "schedule".to_string())],
        bpf_source_locs: Default::default(),
        diagnostics: None,
        nr_cpus: None,
        param_names: Default::default(),
        render_hints: Default::default(),
    };
    let json = serde_json::to_string(&payload).unwrap();
    let parsed = parse_probe_payload(&json, None).expect("strict parse must succeed");
    assert_eq!(parsed.events.len(), 1);
    assert_eq!(parsed.func_names.len(), 1);
}

#[test]
fn parse_probe_payload_eof_with_dump_path_writes_file_returns_recovered() {
    use crate::probe::process::ProbeEvent;
    let payload = ProbeBytes {
        events: vec![ProbeEvent {
            func_idx: 0,
            task_ptr: 0xa,
            ts: 100,
            args: [0; 6],
            fields: vec![],
            kstack: vec![],
            str_val: None,
            ..Default::default()
        }],
        func_names: vec![],
        bpf_source_locs: Default::default(),
        diagnostics: None,
        nr_cpus: None,
        param_names: Default::default(),
        render_hints: Default::default(),
    };
    // Truncate mid-array, after the first event's closing `}`
    // but before the `]` and trailing fields.
    let full = serde_json::to_string(&payload).unwrap();
    let first_close = full.find("}]").expect("single-event array ends with `}]`");
    let truncated = &full[..first_close + 1]; // include the `}` only
    let dir = tempfile::tempdir().expect("tempdir");
    let partial = dir.path().join("dump.json");
    let parsed = parse_probe_payload(truncated, Some(&partial))
        .expect("EOF with one complete event must recover");
    assert_eq!(parsed.events.len(), 1);
    // Recovery path zeroes out non-events fields.
    assert!(parsed.func_names.is_empty());
    assert!(parsed.diagnostics.is_none());
    // Partial dump written verbatim.
    assert!(partial.exists());
    assert_eq!(std::fs::read_to_string(&partial).unwrap(), truncated);
}

#[test]
fn parse_probe_payload_non_eof_error_returns_none_and_dumps() {
    // Garbage JSON: serde_json reports a syntax error, NOT a
    // category Eof error. We still write the raw bytes to the
    // dump path, but recovery is not attempted.
    let dir = tempfile::tempdir().expect("tempdir");
    let partial = dir.path().join("dump.json");
    let result = parse_probe_payload("not json", Some(&partial));
    assert!(result.is_none());
    assert!(partial.exists());
    assert_eq!(std::fs::read_to_string(&partial).unwrap(), "not json");
}

#[test]
fn parse_probe_payload_truncated_no_dump_path_still_recovers() {
    // Recovery works without a dump-path sink — the partial-
    // payload-on-disk feature is independent of in-memory event
    // recovery.
    use crate::probe::process::ProbeEvent;
    let payload = ProbeBytes {
        events: vec![ProbeEvent {
            func_idx: 0,
            task_ptr: 1,
            ts: 100,
            args: [0; 6],
            fields: vec![],
            kstack: vec![],
            str_val: None,
            ..Default::default()
        }],
        func_names: vec![],
        bpf_source_locs: Default::default(),
        diagnostics: None,
        nr_cpus: None,
        param_names: Default::default(),
        render_hints: Default::default(),
    };
    let full = serde_json::to_string(&payload).unwrap();
    let first_close = full.find("}]").unwrap();
    let truncated = &full[..first_close + 1];
    let parsed = parse_probe_payload(truncated, None).expect("recovery without dump path");
    assert_eq!(parsed.events.len(), 1);
}

// -- recover_partial_events --

#[test]
fn recover_partial_events_no_events_key_returns_empty() {
    // A JSON payload that does not contain `"events":` cannot be
    // recovered. Helper short-circuits.
    assert!(recover_partial_events(r#"{"foo":1}"#).is_empty());
    assert!(recover_partial_events("").is_empty());
}

#[test]
fn recover_partial_events_empty_array() {
    // `"events":[]` produces zero events, even when truncated
    // immediately after.
    assert!(recover_partial_events(r#"{"events":[]"#).is_empty());
    assert!(recover_partial_events(r#"{"events":["#).is_empty());
}

#[test]
fn recover_partial_events_handles_braces_in_strings() {
    // Brace characters inside string fields must not unbalance
    // the depth counter. Build an event whose `str_val` carries
    // literal `{`/`}` and confirm the splitter still finds the
    // event boundary.
    use crate::probe::process::ProbeEvent;
    let event = ProbeEvent {
        func_idx: 0,
        task_ptr: 1,
        ts: 100,
        args: [0; 6],
        fields: vec![],
        kstack: vec![],
        str_val: Some("contains {nested} and \\\"quoted\\\"".to_string()),
        ..Default::default()
    };
    let event_json = serde_json::to_string(&event).unwrap();
    let payload = format!(r#"{{"events":[{event_json}]}}"#);
    // Truncate before the events array's closing `]` so the
    // splitter sees a complete event without the array terminator.
    // Use rfind to skip `]` characters inside the event itself
    // (e.g. from `args:[0,0,0,0,0,0]`).
    let cut = payload.rfind(']').unwrap();
    let truncated = &payload[..cut];
    let recovered = recover_partial_events(truncated);
    assert_eq!(
        recovered.len(),
        1,
        "one event should recover: {recovered:?}"
    );
    // Round-trip: the original `str_val` runtime value contained
    // a literal `\` followed by `"` (Rust source `\\\"` = `\` +
    // `"`). After serde serialize → splitter → serde deserialize
    // we expect the same runtime bytes back.
    assert_eq!(
        recovered[0].str_val.as_deref(),
        Some("contains {nested} and \\\"quoted\\\""),
    );
}

// -- find_balanced_object_end --

#[test]
fn find_balanced_object_end_simple_object() {
    assert_eq!(find_balanced_object_end("{}"), Some(2));
    assert_eq!(find_balanced_object_end("{}rest"), Some(2));
}

#[test]
fn find_balanced_object_end_nested_objects() {
    assert_eq!(find_balanced_object_end(r#"{"a":{"b":{}}}"#), Some(14));
}

#[test]
fn find_balanced_object_end_braces_in_strings_ignored() {
    // `{` and `}` inside a quoted string must not unbalance the
    // depth count.
    assert_eq!(find_balanced_object_end(r#"{"x":"{{}}"}"#), Some(12));
}

#[test]
fn find_balanced_object_end_escaped_quote_does_not_close_string() {
    // Escaped `\"` inside a string keeps the parser inside the
    // string until a real, unescaped `"` is seen.
    let s = r#"{"x":"\"}"}"#;
    assert_eq!(find_balanced_object_end(s), Some(s.len()));
}

#[test]
fn find_balanced_object_end_truncated_returns_none() {
    // Missing closing brace → None.
    assert_eq!(find_balanced_object_end(r#"{"a":1"#), None);
}

#[test]
fn find_balanced_object_end_truncated_in_string_returns_none() {
    // Truncated mid-string (no closing `"`) → in_string never
    // resets so the trailing `}` isn't seen → None.
    assert_eq!(find_balanced_object_end(r#"{"a":"hello"#), None);
}

#[test]
fn find_balanced_object_end_non_object_returns_none() {
    assert_eq!(find_balanced_object_end("[1,2]"), None);
    assert_eq!(find_balanced_object_end(""), None);
    assert_eq!(find_balanced_object_end("null"), None);
}

// truncate_probe_json is a fixture helper — exercise it once so
// its `min` clamp is pinned and a future caller passing an
// out-of-range cut doesn't silently read garbage.
#[test]
fn truncate_probe_json_clamps_to_full_length() {
    let payload = ProbeBytes {
        events: vec![],
        func_names: vec![],
        bpf_source_locs: Default::default(),
        diagnostics: None,
        nr_cpus: None,
        param_names: Default::default(),
        render_hints: Default::default(),
    };
    let full = serde_json::to_string(&payload).unwrap();
    // Cut past end → returns the full string unchanged.
    let s = truncate_probe_json(&payload, full.len() + 1024);
    assert_eq!(s, full);
    // Cut at zero → empty string.
    assert_eq!(truncate_probe_json(&payload, 0), "");
}

// -- format_tail --

#[test]
fn format_tail_empty_text_returns_none() {
    assert_eq!(format_tail("", 5, "scheduler"), None);
}

#[test]
fn format_tail_fewer_lines_than_n_returns_all() {
    let out = format_tail("one\ntwo\nthree", 10, "scheduler").unwrap();
    assert_eq!(out, "--- scheduler ---\none\ntwo\nthree");
}

#[test]
fn format_tail_trims_to_last_n_lines() {
    let out = format_tail("1\n2\n3\n4\n5", 3, "log").unwrap();
    assert_eq!(out, "--- log ---\n3\n4\n5");
}

#[test]
fn format_tail_zero_n_returns_empty_body_under_header() {
    // saturating_sub keeps `start == lines.len()`, so the joined
    // slice is empty — the header alone survives.
    let out = format_tail("a\nb", 0, "hdr").unwrap();
    assert_eq!(out, "--- hdr ---\n");
}

#[test]
fn format_tail_preserves_trailing_blank_lines() {
    // `str::lines` strips a single trailing newline but keeps
    // interior blanks. The tail should include the blank line.
    let out = format_tail("a\n\nb", 3, "hdr").unwrap();
    assert_eq!(out, "--- hdr ---\na\n\nb");
}

// -- parse_rust_env_from_cmdline --

#[test]
fn parse_rust_env_empty_cmdline_is_empty() {
    assert!(parse_rust_env_from_cmdline("").is_empty());
}

#[test]
fn parse_rust_env_no_matches() {
    assert!(parse_rust_env_from_cmdline("console=ttyS0 ro quiet").is_empty());
}

#[test]
fn parse_rust_env_backtrace_only() {
    let parsed = parse_rust_env_from_cmdline("console=ttyS0 RUST_BACKTRACE=1 ro");
    assert_eq!(parsed, vec![("RUST_BACKTRACE", "1")]);
}

#[test]
fn parse_rust_env_log_only() {
    let parsed = parse_rust_env_from_cmdline("RUST_LOG=debug other=x");
    assert_eq!(parsed, vec![("RUST_LOG", "debug")]);
}

#[test]
fn parse_rust_env_both() {
    let parsed = parse_rust_env_from_cmdline("RUST_BACKTRACE=full RUST_LOG=trace other=y");
    assert_eq!(
        parsed,
        vec![("RUST_BACKTRACE", "full"), ("RUST_LOG", "trace")]
    );
}

#[test]
fn parse_rust_env_preserves_token_order() {
    let parsed = parse_rust_env_from_cmdline("RUST_LOG=info RUST_BACKTRACE=1");
    assert_eq!(parsed, vec![("RUST_LOG", "info"), ("RUST_BACKTRACE", "1")]);
}

#[test]
fn parse_rust_env_empty_value() {
    // `RUST_LOG=` with no value yields an empty-string value,
    // matching the split semantics of `strip_prefix`.
    let parsed = parse_rust_env_from_cmdline("RUST_LOG=");
    assert_eq!(parsed, vec![("RUST_LOG", "")]);
}

#[test]
fn parse_rust_env_ignores_prefix_mismatch() {
    // Tokens that merely contain the key substring but do not
    // start with it are ignored (e.g. `xRUST_LOG=...`).
    assert!(parse_rust_env_from_cmdline("xRUST_LOG=x").is_empty());
}

#[test]
fn parse_rust_env_sidecar_dir() {
    // `KTSTR_SIDECAR_DIR` is propagated alongside RUST_BACKTRACE
    // / RUST_LOG so the guest's `sidecar_dir()` returns the
    // host's resolved override path.
    let parsed = parse_rust_env_from_cmdline(
        "console=ttyS0 KTSTR_SIDECAR_DIR=/host/target/ktstr/run-key ro",
    );
    assert_eq!(
        parsed,
        vec![(crate::KTSTR_SIDECAR_DIR_ENV, "/host/target/ktstr/run-key")]
    );
}

#[test]
fn parse_rust_env_all_three_keys() {
    // Order-preserving across all three keys — the guest applies
    // them in the order they appear, matching cmdline composition
    // semantics.
    let parsed =
        parse_rust_env_from_cmdline("RUST_LOG=info KTSTR_SIDECAR_DIR=/dir RUST_BACKTRACE=1");
    assert_eq!(
        parsed,
        vec![
            ("RUST_LOG", "info"),
            (crate::KTSTR_SIDECAR_DIR_ENV, "/dir"),
            ("RUST_BACKTRACE", "1")
        ]
    );
}

// -- extract_not_attached_reason --

fn lifecycle_drain(
    phase: crate::vmm::wire::LifecyclePhase,
    reason: &str,
) -> crate::vmm::host_comms::BulkDrainResult {
    let mut payload = vec![phase.wire_value()];
    payload.extend_from_slice(reason.as_bytes());
    crate::vmm::host_comms::BulkDrainResult {
        entries: vec![crate::vmm::wire::ShmEntry {
            msg_type: crate::vmm::wire::MSG_TYPE_LIFECYCLE,
            payload,
            crc_ok: true,
        }],
    }
}

#[test]
fn extract_not_attached_reason_timeout() {
    let drain = lifecycle_drain(
        crate::vmm::wire::LifecyclePhase::SchedulerNotAttached,
        "timeout",
    );
    assert_eq!(
        extract_not_attached_reason(Some(&drain)).as_deref(),
        Some("timeout"),
    );
}

#[test]
fn extract_not_attached_reason_sysfs_absent() {
    // Multi-word reason must survive through to the caller so the
    // user can distinguish "timeout" from "sched_ext sysfs absent".
    let drain = lifecycle_drain(
        crate::vmm::wire::LifecyclePhase::SchedulerNotAttached,
        "sched_ext sysfs absent",
    );
    assert_eq!(
        extract_not_attached_reason(Some(&drain)).as_deref(),
        Some("sched_ext sysfs absent"),
    );
}

#[test]
fn extract_not_attached_reason_trims_surrounding_whitespace() {
    // The lifecycle payload's reason is trimmed before
    // surfacing to the caller, so a stray space or newline
    // appended at the emit site does not leak into the
    // displayed reason string.
    let drain = lifecycle_drain(
        crate::vmm::wire::LifecyclePhase::SchedulerNotAttached,
        "  timeout  ",
    );
    assert_eq!(
        extract_not_attached_reason(Some(&drain)).as_deref(),
        Some("timeout"),
    );
}

#[test]
fn extract_not_attached_reason_absent_returns_none() {
    assert_eq!(extract_not_attached_reason(None), None);
    let died = lifecycle_drain(crate::vmm::wire::LifecyclePhase::SchedulerDied, "");
    assert_eq!(extract_not_attached_reason(Some(&died)), None);
}

#[test]
fn extract_not_attached_reason_empty_suffix_returns_none() {
    // A `SchedulerNotAttached` lifecycle with no reason bytes
    // carries no diagnostic value — `None` lets the caller
    // fall through to the generic abnormal-exit branch instead
    // of surfacing an empty reason.
    let drain = lifecycle_drain(crate::vmm::wire::LifecyclePhase::SchedulerNotAttached, "");
    assert_eq!(extract_not_attached_reason(Some(&drain)), None);
    let drain_ws = lifecycle_drain(
        crate::vmm::wire::LifecyclePhase::SchedulerNotAttached,
        "   ",
    );
    assert_eq!(extract_not_attached_reason(Some(&drain_ws)), None);
}

#[test]
fn extract_not_attached_reason_first_match_wins() {
    // Two `SchedulerNotAttached` frames should not be possible
    // in production (rust_init emits exactly one before
    // force_reboot), but if the bulk drain ever concatenates
    // multiple, pinning "first match" keeps the classification
    // stable.
    let drain = crate::vmm::host_comms::BulkDrainResult {
        entries: vec![
            lifecycle_drain(
                crate::vmm::wire::LifecyclePhase::SchedulerNotAttached,
                "timeout",
            )
            .entries
            .pop()
            .unwrap(),
            lifecycle_drain(
                crate::vmm::wire::LifecyclePhase::SchedulerNotAttached,
                "sched_ext sysfs absent",
            )
            .entries
            .pop()
            .unwrap(),
        ],
    };
    assert_eq!(
        extract_not_attached_reason(Some(&drain)).as_deref(),
        Some("timeout"),
    );
}

#[test]
fn extract_not_attached_reason_skips_crc_bad() {
    // CRC-bad lifecycle frames must be ignored — same rule as
    // every other host-side bulk-drain consumer.
    let mut bad = lifecycle_drain(
        crate::vmm::wire::LifecyclePhase::SchedulerNotAttached,
        "timeout",
    );
    bad.entries[0].crc_ok = false;
    assert_eq!(extract_not_attached_reason(Some(&bad)), None);
}

// -- classify_repro_vm_status --
//
// Lifecycle phases now travel as `MSG_TYPE_LIFECYCLE` TLV
// frames on the bulk data port, so each fixture builds a
// `BulkDrainResult` containing the relevant frame(s). The
// `lifecycle_drain` helper above produces a single-frame drain;
// multi-frame fixtures construct entries inline.

fn died_drain() -> crate::vmm::host_comms::BulkDrainResult {
    lifecycle_drain(crate::vmm::wire::LifecyclePhase::SchedulerDied, "")
}

fn not_attached_drain(reason: &str) -> crate::vmm::host_comms::BulkDrainResult {
    lifecycle_drain(
        crate::vmm::wire::LifecyclePhase::SchedulerNotAttached,
        reason,
    )
}

#[test]
fn classify_repro_vm_status_timeout_wins_over_other_signals() {
    // Even with a crash message present, the VM-level timeout is
    // the primary classification — a timed-out VM may have dumped
    // any signal on the way out.
    let drain = not_attached_drain("timeout");
    let status = classify_repro_vm_status(
        /*timed_out*/ true,
        /*has_crash_message*/ true,
        137,
        Some(&drain),
    );
    assert_eq!(status, "repro VM: timed out");
}

#[test]
fn classify_repro_vm_status_not_attached_with_reason() {
    let drain = not_attached_drain("sched_ext sysfs absent");
    let status = classify_repro_vm_status(false, false, 1, Some(&drain));
    assert_eq!(
        status,
        "repro VM: scheduler did not attach (sched_ext sysfs absent) (exit code 1)",
    );
}

#[test]
fn classify_repro_vm_status_not_attached_takes_precedence_over_crashed() {
    // The rust_init emission path writes SchedulerNotAttached and
    // then force-reboots before any SchedulerDied could happen.
    // If both ever appear in the same drain, NotAttached is the
    // more specific classification and wins.
    let mut drain = died_drain();
    drain
        .entries
        .push(not_attached_drain("timeout").entries.pop().unwrap());
    let status = classify_repro_vm_status(false, true, 1, Some(&drain));
    assert_eq!(
        status,
        "repro VM: scheduler did not attach (timeout) (exit code 1)",
    );
}

#[test]
fn classify_repro_vm_status_crashed_from_sentinel() {
    // Positive exit code on the crash-sentinel branch → qemu
    // propagated a non-zero exit alongside the guest crash
    // sentinel. Clause format: "exited with non-zero status (N)".
    let drain = died_drain();
    let status = classify_repro_vm_status(false, false, 139, Some(&drain));
    assert_eq!(
        status,
        "repro VM: scheduler crashed — exited with non-zero status (139)",
    );
}

#[test]
fn classify_repro_vm_status_crashed_from_crash_message() {
    // crash_message set without a SchedulerDied lifecycle frame
    // (e.g. a guest-side panic captured from COM2 by
    // `extract_panic_message`) still routes to the crashed
    // branch. Positive exit code → non-zero-status clause.
    let status = classify_repro_vm_status(false, true, 134, None);
    assert_eq!(
        status,
        "repro VM: scheduler crashed — exited with non-zero status (134)",
    );
}

/// exit_code == 0 with a SchedulerDied lifecycle frame is the
/// "guest panic handler + orderly reboot" case — qemu shut down
/// cleanly but the guest emitted SchedulerDied. The old format
/// conflated this with a true qemu-level crash; the branched
/// format makes it unambiguous.
#[test]
fn classify_repro_vm_status_crashed_from_sentinel_qemu_clean_exit() {
    let drain = died_drain();
    let status = classify_repro_vm_status(false, false, 0, Some(&drain));
    assert_eq!(status, "repro VM: scheduler crashed — exited cleanly");
}

/// Negative exit_code on the crash branch exercises the
/// `<0` arm. The sign convention is the VMM's, not
/// `std::process::ExitStatus`: `VmResult::exit_code`
/// (vmm::mod.rs) is seeded to `-1` in the BSP run loop and
/// left negative on watchdog-fire / non-normal exits, so
/// negatives that reach `classify_repro_vm_status` are VMM
/// sentinels rather than OS-reported signal codes
/// (`ExitStatus::code()` returns `None`, never a negative
/// i32, on signal-kill). Clause format: "killed by signal (N)".
#[test]
fn classify_repro_vm_status_crashed_from_sentinel_killed_by_signal() {
    let drain = died_drain();
    let status = classify_repro_vm_status(false, false, -9, Some(&drain));
    assert_eq!(
        status,
        "repro VM: scheduler crashed — killed by signal (-9)",
    );
}

/// `exit_code == -1` on the crash branch is the VMM sentinel —
/// `VmResult::exit_code` is seeded to `-1` at the top of the
/// boot-CPU run loop and left there when the scheduler did not
/// deliver its final exit message. Watchdog-fire is caught
/// earlier via the `timed_out` branch, so a `-1` here means the
/// boot-CPU ran a code-unsetting error path — not a signal-kill.
/// Distinct clause so users don't misread this as "signal 1"
/// (SIGHUP). The asserted string is phrased in end-user terms —
/// the internals are in the implementation comment so the
/// console output stays operator-readable without cross-
/// referencing VMM source. A regression that swapped the clause
/// back to "VMM exit-code sentinel" or any `BSP` /
/// `VmResult::exit_code` / `MSG_TYPE_EXIT` phrasing would fail
/// here.
#[test]
fn classify_repro_vm_status_crashed_from_sentinel_vmm_exit_code_unset() {
    let drain = died_drain();
    let status = classify_repro_vm_status(false, false, -1, Some(&drain));
    assert_eq!(
        status,
        "repro VM: scheduler crashed — VM host reported no final exit \
         status (the scheduler did not deliver an exit signal before \
         the VM ended)",
    );
    // Negative assertions: none of the internal vocabulary may
    // leak into the user-facing status — each term is a
    // usability bug and must stay out of the rendered string.
    assert!(
        !status.contains("BSP"),
        "user-facing status leaks BSP: {status}"
    );
    assert!(
        !status.contains("VmResult::exit_code"),
        "user-facing status leaks VmResult::exit_code: {status}",
    );
    assert!(
        !status.contains("MSG_TYPE_EXIT"),
        "user-facing status leaks MSG_TYPE_EXIT: {status}",
    );
}

#[test]
fn classify_repro_vm_status_abnormal_exit() {
    let status = classify_repro_vm_status(false, false, 2, None);
    assert_eq!(status, "repro VM: exited abnormally (exit code 2)");
}

#[test]
fn classify_repro_vm_status_clean_run() {
    let status = classify_repro_vm_status(false, false, 0, None);
    assert_eq!(
        status,
        "repro VM: scheduler ran normally (crash did not reproduce)",
    );
}

#[test]
fn classify_repro_vm_status_malformed_not_attached_falls_through() {
    // A SchedulerNotAttached lifecycle frame with no reason
    // bytes does not count as a classification signal. With no
    // crash signals and exit_code=1 the result should be the
    // abnormal-exit branch, not a NotAttached branch with an
    // empty reason.
    let drain = not_attached_drain("");
    let status = classify_repro_vm_status(false, false, 1, Some(&drain));
    assert_eq!(status, "repro VM: exited abnormally (exit code 1)");
}

// -- render_failure_dump_file -----------------------------------
//
// The auto-repro path reads its `{name}-{variant_hash}.repro.failure-dump.json`
// sidecar back, sniffs the `schema` discriminant to choose
// between [`FailureDumpReport`] and [`DualFailureDumpReport`],
// and emits the Display rendering as a tail block. These tests
// pin every branch of that helper: missing file, both schemas,
// absent schema (back-compat), unknown schema, malformed JSON.
// tempfile gives us scratch paths without polluting the working
// directory or relying on sidecar_dir() machinery.

#[test]
fn render_failure_dump_file_missing_returns_none() {
    // A path under temp_dir that we never create returns None
    // without panicking. Mirrors the auto-repro path when the
    // freeze coordinator never fired (no dump written).
    let nonexistent = std::env::temp_dir().join("ktstr-render-failure-dump-missing");
    // Best-effort: ensure the file does not exist if a prior
    // test left one behind.
    let _ = std::fs::remove_file(&nonexistent);
    assert!(render_failure_dump_file(&nonexistent).is_none());
}

#[test]
fn render_failure_dump_file_single_schema() {
    use crate::monitor::dump::{FailureDumpReport, SCHEMA_SINGLE};
    let report = FailureDumpReport {
        schema: SCHEMA_SINGLE.to_string(),
        ..Default::default()
    };
    let json = serde_json::to_string(&report).expect("serialize single");
    let tmp = tempfile::NamedTempFile::new().expect("tempfile");
    std::fs::write(tmp.path(), json).expect("write tempfile");

    let rendered = render_failure_dump_file(tmp.path()).expect("single-schema must render Some");
    assert!(
        rendered.starts_with("--- repro VM failure dump ---"),
        "header missing: {rendered}"
    );
    // FailureDumpReport's empty Display body is "(empty failure dump)";
    // pin the substring that the FailureDumpReport's own Display
    // emits for an empty-but-valid report so we know the body
    // came from FailureDumpReport::fmt and not a different path.
    assert!(
        rendered.contains("(empty failure dump)"),
        "single-schema body must come from FailureDumpReport Display: {rendered}"
    );
}

#[test]
fn render_failure_dump_file_dual_schema() {
    use crate::monitor::dump::{DualFailureDumpReport, FailureDumpReport, SCHEMA_DUAL};
    let dual = DualFailureDumpReport {
        schema: SCHEMA_DUAL.to_string(),
        early: None,
        late: FailureDumpReport::default(),
        early_max_age_jiffies: 0,
        early_threshold_jiffies: 0,
        early_skipped_reason: None,
    };
    let json = serde_json::to_string(&dual).expect("serialize dual");
    let tmp = tempfile::NamedTempFile::new().expect("tempfile");
    std::fs::write(tmp.path(), json).expect("write tempfile");

    let rendered = render_failure_dump_file(tmp.path()).expect("dual-schema must render Some");
    assert!(
        rendered.starts_with("--- repro VM failure dump ---"),
        "header missing: {rendered}"
    );
    assert!(
        rendered.contains("DualFailureDumpReport:"),
        "dual-schema body must come from DualFailureDumpReport Display: {rendered}"
    );
}

#[test]
fn render_failure_dump_file_absent_schema_returns_none() {
    // JSON without the `schema` field: the dispatcher at
    // `FailureDumpReportAny::from_json` requires an explicit
    // schema discriminant. Per the dispatcher doc, the previous
    // "absent ⇒ single" fallback was deliberately removed to
    // avoid silently mis-routing a richer wrapper as a lossy
    // single shape. So absent-schema JSON returns None — the
    // helper does not invent a schema choice.
    let json = r#"{"maps":[],"vcpu_regs":[],"sdt_allocations":[]}"#;
    let tmp = tempfile::NamedTempFile::new().expect("tempfile");
    std::fs::write(tmp.path(), json).expect("write tempfile");

    assert!(
        render_failure_dump_file(tmp.path()).is_none(),
        "absent-schema JSON must return None to avoid silent mis-routing"
    );
}

#[test]
fn render_failure_dump_file_unknown_schema_returns_none() {
    // A schema value the helper doesn't know about (e.g. a
    // future "triple" wrapper) returns None — the helper does
    // not silently fall back to single, since that could
    // mis-render a richer wrapper as the lossy single shape.
    let json = r#"{"schema":"triple","maps":[],"vcpu_regs":[],"sdt_allocations":[]}"#;
    let tmp = tempfile::NamedTempFile::new().expect("tempfile");
    std::fs::write(tmp.path(), json).expect("write tempfile");
    assert!(render_failure_dump_file(tmp.path()).is_none());
}

#[test]
fn render_failure_dump_file_invalid_json_returns_none() {
    // Garbage bytes on disk: the initial
    // `serde_json::from_str::<Value>` call returns Err, and the
    // helper short-circuits to None without panicking.
    let tmp = tempfile::NamedTempFile::new().expect("tempfile");
    std::fs::write(tmp.path(), "not json").expect("write tempfile");
    assert!(render_failure_dump_file(tmp.path()).is_none());
}

// -- stitch_drop_cause / format_probe_diagnostics events line --
//
// The user's symptom report from the lavd run was:
//   bpf_discover: 0 programs found
//   trigger:      not fired (tp_btf)
//   probe_data:   146 keys, 0 unmatched IPs
//   events:       146 captured, 0 after stitch
//   bpf_counts:   16567 kprobe fires, 0 trigger fires, 0 meta misses
//
// The renderer used to silently emit `events: 146 captured, 0
// after stitch` and stop. Operators had no way to disambiguate
// "trigger never fired" (lifecycle race) from "trigger fired
// with kind=STALL" (suppressed by the BPF handler) from "trigger
// fired with kind=BPF_ERROR but stitch found no match" (a real
// bug). These tests pin every branch of the new
// `stitch_drop_cause` ladder against synthetic
// `ProbeDiagnostics` fixtures so a future change to the BPF
// exit-kind values, the stitch logic, or the renderer surfaces
// here at compile time, not as a missing diagnostic in
// production.

fn diag_with_events(
    before: u32,
    after: u32,
    trigger_fires: u64,
    exit_kind: u32,
) -> crate::probe::process::ProbeDiagnostics {
    crate::probe::process::ProbeDiagnostics {
        events_before_stitch: before,
        events_after_stitch: after,
        bpf_trigger_fires: trigger_fires,
        bpf_exit_kind_snap: exit_kind,
        ..Default::default()
    }
}

#[test]
fn stitch_drop_cause_trigger_never_fired() {
    // bpf_trigger_fires == 0 → the tp_btf handler never executed.
    // Either the scheduler clean-exited (kind < SCX_EXIT_ERROR
    // hits the early-return at probe.bpf.c:565) or the scheduler
    // crashed before reaching the tracepoint at all. The cause
    // string MUST mention "trigger never fired" so an operator
    // grepping the section can land on the lifecycle bug rather
    // than chasing a stitch failure.
    let diag = diag_with_events(146, 0, 0, 0);
    let cause = stitch_drop_cause(&diag);
    assert!(
        cause.contains("trigger never fired"),
        "expected 'trigger never fired' branch, got: {cause}"
    );
}

#[test]
fn stitch_drop_cause_kind_stall() {
    // bpf_trigger_fires > 0 AND exit_kind_snap == ERROR_STALL →
    // probe.bpf.c:699 explicit early-return for STALL. Tracepoint
    // fired (counter incremented) but no ringbuf event submitted,
    // so target_tptr is None at the host. The cause string MUST
    // mention "kind=STALL" so the operator knows the watchdog
    // path was the cause rather than a stitch bug.
    use crate::probe::scx_defs::EXIT_ERROR_STALL;
    let diag = diag_with_events(146, 0, 1, EXIT_ERROR_STALL as u32);
    let cause = stitch_drop_cause(&diag);
    assert!(
        cause.contains("kind=STALL"),
        "expected 'kind=STALL' branch, got: {cause}"
    );
}

#[test]
fn stitch_drop_cause_kind_error_generic() {
    // bpf_trigger_fires > 0 AND exit_kind_snap == ERROR (1024) →
    // generic ERROR exit can fire from kworker context where
    // `current` is the worker thread, not the causal task. The
    // BPF handler sets args[0] = 0 (probe.bpf.c:731 ternary) so
    // target_tptr is None at the host. The cause string MUST
    // mention "kind=ERROR" without the BPF or STALL qualifier.
    use crate::probe::scx_defs::EXIT_ERROR;
    let diag = diag_with_events(146, 0, 1, EXIT_ERROR as u32);
    let cause = stitch_drop_cause(&diag);
    assert!(
        cause.contains("kind=ERROR"),
        "expected 'kind=ERROR' branch, got: {cause}"
    );
    // Negative assertion: must NOT match the STALL or BPF_ERROR
    // branches just because the substring "kind=" appears.
    assert!(
        !cause.contains("kind=STALL"),
        "kind=ERROR branch must not say STALL: {cause}"
    );
    assert!(
        !cause.contains("kind=BPF_ERROR"),
        "kind=ERROR branch must not say BPF_ERROR: {cause}"
    );
}

#[test]
fn stitch_drop_cause_kind_bpf_error() {
    // bpf_trigger_fires > 0 AND exit_kind_snap == ERROR_BPF →
    // BPF callback faulted in a real task's context, args[0]
    // resolved to bpf_get_current_task(). If we still have
    // events_after_stitch == 0, something is wrong with the
    // stitch logic itself (func_idx_offset bug, ID mismatch).
    // The cause string MUST flag this as a suspected bug
    // ("file a ticket") so the operator escalates rather than
    // chasing a misconfigured probe.
    use crate::probe::scx_defs::EXIT_ERROR_BPF;
    let diag = diag_with_events(146, 0, 1, EXIT_ERROR_BPF as u32);
    let cause = stitch_drop_cause(&diag);
    assert!(
        cause.contains("kind=BPF_ERROR"),
        "expected 'kind=BPF_ERROR' branch, got: {cause}"
    );
    assert!(
        cause.contains("file a ticket") || cause.contains("ID mismatch"),
        "kind=BPF_ERROR branch must signal a suspected bug: {cause}"
    );
}

#[test]
fn stitch_drop_cause_unrecognized_kind() {
    // bpf_trigger_fires > 0 AND exit_kind_snap == an unknown
    // value (e.g. a future kernel version, or a value not yet
    // wired into scx_defs.rs). The cause string falls through
    // to a generic "unrecognized" branch so future kernels
    // don't silently render with no diagnostic.
    let diag = diag_with_events(146, 0, 1, 9999);
    let cause = stitch_drop_cause(&diag);
    assert!(
        cause.contains("unrecognized") || cause.contains("no causal"),
        "unrecognized-kind branch must surface a diagnostic, got: {cause}"
    );
}

#[test]
fn format_probe_diagnostics_appends_cause_when_zero_after_stitch() {
    // End-to-end: the rendered probe pipeline summary must
    // include the cause explanation on the events line, not
    // just emit the bare counter pair. Reproduces the user's
    // lavd report shape: 146 captured, 0 after stitch, 0
    // trigger fires.
    let pipeline = PipelineDiagnostics::default();
    let skeleton = diag_with_events(146, 0, 0, 0);
    let rendered = format_probe_diagnostics(&pipeline, &skeleton);
    assert!(
        rendered.contains("146 captured, 0 after stitch"),
        "rendered output missing counter pair: {rendered}"
    );
    assert!(
        rendered.contains("trigger never fired"),
        "rendered output missing cause explanation: {rendered}"
    );
}

#[test]
fn format_probe_diagnostics_appends_kind_stall_diagnostic() {
    // Renderer must surface STALL as a distinct diagnostic so
    // the operator can immediately distinguish the watchdog
    // path from a generic timing race.
    use crate::probe::scx_defs::EXIT_ERROR_STALL;
    let pipeline = PipelineDiagnostics::default();
    let skeleton = diag_with_events(146, 0, 1, EXIT_ERROR_STALL as u32);
    let rendered = format_probe_diagnostics(&pipeline, &skeleton);
    assert!(
        rendered.contains("kind=STALL"),
        "rendered output missing kind=STALL diagnostic: {rendered}"
    );
}

#[test]
fn format_probe_diagnostics_no_cause_when_clean_run() {
    // Sanity check: when events_before_stitch == 0 (clean run,
    // no probe data captured), the cause-explanation logic
    // MUST NOT fire. The bare counter pair is the correct
    // output for a no-op run.
    let pipeline = PipelineDiagnostics::default();
    let skeleton = diag_with_events(0, 0, 0, 0);
    let rendered = format_probe_diagnostics(&pipeline, &skeleton);
    assert!(
        rendered.contains("0 captured, 0 after stitch"),
        "missing zero-zero counter line: {rendered}"
    );
    assert!(
        !rendered.contains("trigger never fired"),
        "must not append cause for clean run: {rendered}"
    );
    assert!(
        !rendered.contains("kind=STALL"),
        "must not append cause for clean run: {rendered}"
    );
}

#[test]
fn format_probe_diagnostics_no_cause_when_stitch_succeeded() {
    // Successful stitch: events_before_stitch > 0 and
    // events_after_stitch > 0 (146 -> 100). The cause line only fires
    // on events_after_stitch == 0, so a non-zero after-count
    // suppresses it — the events line is its own sufficient summary.
    let pipeline = PipelineDiagnostics::default();
    let skeleton = diag_with_events(146, 100, 1, 1025);
    let rendered = format_probe_diagnostics(&pipeline, &skeleton);
    assert!(
        rendered.contains("146 captured, 100 after stitch"),
        "missing counter line: {rendered}"
    );
    assert!(
        !rendered.contains("trigger never fired"),
        "must not append cause when stitch succeeded: {rendered}"
    );
    assert!(
        !rendered.contains("kind=STALL"),
        "must not append cause when stitch succeeded: {rendered}"
    );
}

#[test]
fn format_probe_diagnostics_appends_fallback_marker() {
    // When `stitch_fallback_used` is set and the counter is
    // non-zero, the renderer must mark the chain as best-effort
    // grouping so the operator does not mistake it for a verified
    // stitch. This protects the user-direction "no invalid data
    // made" — the candidate chain is real probe data, but the
    // labelling has to make clear it is grouped by frequency,
    // not anchored to a verified trigger task pointer.
    let pipeline = PipelineDiagnostics::default();
    let skeleton = crate::probe::process::ProbeDiagnostics {
        events_before_stitch: 146,
        events_after_stitch: 80,
        stitch_fallback_used: true,
        bpf_trigger_fires: 0,
        bpf_exit_kind_snap: 0,
        ..Default::default()
    };
    let rendered = format_probe_diagnostics(&pipeline, &skeleton);
    assert!(
        rendered.contains("trigger absent") && rendered.contains("frequency"),
        "fallback marker missing from rendered output: {rendered}"
    );
}

// -- classify_dmesg_corruption --
//
// The user's lavd report showed `dmesg: � (single 0xff byte)`.
// The renderer was emitting that opaque garbage as if it were a
// real kernel console excerpt. These tests pin the corruption
// classifier so empty / all-0xff / all-replacement-character
// inputs become operator-readable diagnostics, not garbage tail
// blocks.

#[test]
fn classify_dmesg_corruption_empty_text() {
    let diag = classify_dmesg_corruption("");
    assert!(diag.is_some());
    assert!(diag.unwrap().contains("empty"));
}

#[test]
fn classify_dmesg_corruption_only_whitespace() {
    // Whitespace-only stderr is the same operator outcome as
    // empty (kernel never wrote anything readable).
    let diag = classify_dmesg_corruption("   \n\n\t  \n");
    assert!(diag.is_some());
    assert!(diag.unwrap().contains("empty"));
}

#[test]
fn classify_dmesg_corruption_only_replacement_chars() {
    // A stream of U+FFFD characters is what the host's lossy
    // UTF-8 decoder emits for a UART buffer full of 0xff
    // bytes. The diagnostic must say "corrupt" not "empty"
    // so the operator knows the kernel TRIED to fill the
    // buffer but the bytes were uninitialised.
    let diag = classify_dmesg_corruption("\u{fffd}\u{fffd}\u{fffd}");
    assert!(diag.is_some());
    assert!(diag.unwrap().contains("corrupt"));
}

#[test]
fn classify_dmesg_corruption_only_control_chars() {
    // NUL bytes (0x00) and other C0 controls are valid UTF-8 and
    // arrive at the classifier as their Unicode codepoints (not
    // U+FFFD). An uninitialised UART buffer of zero bytes used to
    // slip past the classifier and render as silent garbage; the
    // control-char branch catches it.
    let diag = classify_dmesg_corruption("\0\0\0");
    assert!(diag.is_some());
    assert!(diag.unwrap().contains("corrupt"));

    // Mix of NUL + other C0 controls (0x01..0x08) — same outcome.
    let diag = classify_dmesg_corruption("\0\x01\x02\x07\x08");
    assert!(diag.is_some());
    assert!(diag.unwrap().contains("corrupt"));

    // DEL (U+007F) is also a non-whitespace control char per
    // [`char::is_control`].
    let diag = classify_dmesg_corruption("\x7f\x7f");
    assert!(diag.is_some());
    assert!(diag.unwrap().contains("corrupt"));
}

#[test]
fn classify_dmesg_corruption_only_replacement_and_control_mix() {
    // Mixed U+FFFD (lossy-decoded raw 0xFF from the UART) and
    // C0 control chars (uninitialised NUL bytes): still corrupt,
    // single diagnostic.
    let diag = classify_dmesg_corruption("\u{fffd}\0\u{fffd}");
    assert!(diag.is_some());
    assert!(diag.unwrap().contains("corrupt"));
}

#[test]
fn classify_dmesg_corruption_latin1_text_passes_through() {
    // Legitimate Latin-1 supplement characters (U+00A0..=U+00FF)
    // are valid kernel printk content — e.g. a hardware vendor
    // string that the firmware tagged with a Latin-1 letter.
    // MUST NOT be classified as corrupt. The historic check
    // for `c == '\u{ff}'` was a false-positive trap: raw 0xFF
    // bytes from an uninitialised UART arrive as U+FFFD after
    // `String::from_utf8_lossy`, never as U+00FF.
    for ch in ['\u{c0}', '\u{e9}', '\u{f1}', '\u{ff}'] {
        let s: String = std::iter::repeat_n(ch, 5).collect();
        let diag = classify_dmesg_corruption(&s);
        assert!(
            diag.is_none(),
            "Latin-1 char U+{:04X} must NOT be classified as corrupt: {diag:?}",
            ch as u32,
        );
    }
}

#[test]
fn classify_dmesg_corruption_real_kernel_text_passes_through() {
    // A real kernel printk excerpt MUST NOT be classified as
    // corrupt. The classifier returns None and the caller falls
    // through to format_tail.
    let diag = classify_dmesg_corruption("[    0.000000] Linux version 6.16.0\n");
    assert!(
        diag.is_none(),
        "real kernel text must not be classified as corrupt: {diag:?}"
    );
}

#[test]
fn classify_dmesg_corruption_one_control_amid_text_passes_through() {
    // A single control char buried in legitimate kernel printk
    // text does NOT trigger the classifier — there is still real
    // content for the operator to read. The classifier short-
    // circuits to None on the first non-corrupt, non-whitespace
    // char.
    let diag = classify_dmesg_corruption("[0.1] Linux\u{1}version");
    assert!(
        diag.is_none(),
        "one control char amid real text is not corruption: {diag:?}"
    );
}

#[test]
fn classify_dmesg_corruption_mixed_garbage_and_text_passes_through() {
    // Even one ordinary byte amidst 0xff noise means there's
    // SOMETHING for the operator to read. Don't suppress it.
    let diag = classify_dmesg_corruption("\u{fffd}A\u{fffd}");
    assert!(diag.is_none());
}

// -- render_dmesg_tail: filter-empty disambiguation --
//
// The wrapper around `classify_dmesg_corruption` must distinguish
// "VM produced no kernel printk" (genuinely-empty stderr — VM
// crashed before any boot line landed) from "VM ran cleanly and
// every stderr line is a sched_ext_dump record that's already
// rendered in its own tail section above" (the filter empties
// non-empty stderr). Without the disambiguation, the second case
// surfaces the misleading "scheduler crashed before kernel
// printk reached the UART buffer" diagnostic against a clean
// run — caught during code review.

#[test]
fn render_dmesg_tail_filter_empties_non_empty_stderr_emits_pointer_diag() {
    // Symptom: clean repro VM whose stderr contains only
    // sched_ext_dump records. The filter strips every line →
    // empty `filtered`, but pre_filter content is non-trivial.
    // Must emit the "no kernel printk other than sched_ext_dump"
    // pointer diagnostic, NOT the crash classifier's output.
    let stderr = "[  0.5] sched_ext_dump: header\n\
                  [  0.6] sched_ext_dump: body line A\n\
                  [  0.7] sched_ext_dump: body line B\n";
    let tail = render_dmesg_tail(stderr, 40);
    assert!(
        tail.contains("--- repro VM dmesg ---"),
        "tail must carry the section header: {tail}",
    );
    assert!(
        tail.contains("no kernel printk other than sched_ext_dump"),
        "tail must point operators at the sched_ext_dump section, \
         not falsely report a crash: {tail}",
    );
    assert!(
        !tail.contains("scheduler crashed"),
        "filter-emptied real output must NOT surface the crash \
         classifier's diagnostic: {tail}",
    );
}

#[test]
fn render_dmesg_tail_truly_empty_stderr_emits_crash_diagnostic() {
    // Pre-filter stderr really is empty — VM crashed before any
    // kernel printk fired. The crash classifier's "empty
    // (scheduler crashed...)" diagnostic must still surface.
    let tail = render_dmesg_tail("", 40);
    assert!(
        tail.contains("scheduler crashed before kernel printk"),
        "genuinely-empty stderr must surface the crash diagnostic: {tail}",
    );
}

#[test]
fn render_dmesg_tail_real_kernel_text_passes_through_to_format_tail() {
    // Mixed stderr with both sched_ext_dump and real kernel
    // printks. The filter strips dump lines but the remaining
    // text has content — falls through to format_tail.
    let stderr = "[  0.1] Linux version 6.16.0\n\
                  [  0.5] sched_ext_dump: header\n\
                  [  0.6] sched_ext_dump: body\n\
                  [  0.9] systemd: starting\n";
    let tail = render_dmesg_tail(stderr, 40);
    assert!(
        tail.contains("Linux version 6.16.0"),
        "real kernel text must survive the filter: {tail}",
    );
    assert!(
        tail.contains("systemd: starting"),
        "non-dump lines must survive the filter: {tail}",
    );
    assert!(
        !tail.contains("sched_ext_dump"),
        "dump lines must be stripped (rendered separately): {tail}",
    );
    assert!(
        !tail.contains("scheduler crashed"),
        "real kernel text must NOT surface the crash diagnostic: {tail}",
    );
}

#[test]
fn render_dmesg_tail_only_whitespace_emits_crash_diagnostic() {
    // Whitespace-only stderr — same operator outcome as empty.
    // The crash classifier's diagnostic must surface; the filter
    // dropped no lines so the corruption branch wins.
    let tail = render_dmesg_tail("   \n\n\t  \n", 40);
    assert!(
        tail.contains("scheduler crashed before kernel printk"),
        "whitespace-only stderr must surface the crash diagnostic: {tail}",
    );
}

#[test]
fn render_dmesg_tail_dump_plus_replacement_noise_emits_pointer_diag() {
    // stderr contains sched_ext_dump lines plus U+FFFD
    // corruption residue (lossy-decoded raw 0xFF bytes from the
    // UART). The filter strips dump lines; the residue triggers
    // `classify_dmesg_corruption`. The "scheduler crashed before
    // kernel printk" framing would be misleading because the
    // VM clearly DID produce real dump output. Point at the dump
    // section instead.
    let stderr = "[1] sched_ext_dump: header\n\u{fffd}\u{fffd}\u{fffd}\n";
    let tail = render_dmesg_tail(stderr, 40);
    assert!(
        tail.contains("no kernel printk other than sched_ext_dump"),
        "filter-dropped-real-lines + U+FFFD residue must point at \
         the dump section, not surface the crash diagnostic: {tail}",
    );
    assert!(
        !tail.contains("scheduler crashed"),
        "must NOT misclassify residue as a crash when real dump lines \
         were filtered: {tail}",
    );
}

#[test]
fn render_dmesg_tail_dump_plus_control_noise_emits_pointer_diag() {
    // Variant: residue is C0 control chars (uninitialised NUL
    // bytes — valid UTF-8 arriving as U+0000) instead of U+FFFD.
    // Same outcome: point at the dump section.
    let stderr = "[  0.5] sched_ext_dump: header\n\0\0\0\n";
    let tail = render_dmesg_tail(stderr, 40);
    assert!(
        tail.contains("no kernel printk other than sched_ext_dump"),
        "control-char residue with dump lines must point at the dump \
         section: {tail}",
    );
}

#[test]
fn render_dmesg_tail_dump_plus_whitespace_emits_pointer_diag() {
    // stderr like `"[1.0] sched_ext_dump: dump\n   \n\t\n"`
    // filters to whitespace-only residue, which the classifier
    // labels "empty (scheduler crashed...)". Surfacing that
    // diagnostic against real dump lines would be misleading.
    // Point at the dump section.
    let stderr = "[1.0] sched_ext_dump: dump\n   \n\t\n";
    let tail = render_dmesg_tail(stderr, 40);
    assert!(
        tail.contains("no kernel printk other than sched_ext_dump"),
        "whitespace residue with dump lines must point at the dump \
         section, not surface the crash diagnostic: {tail}",
    );
    assert!(
        !tail.contains("scheduler crashed"),
        "must NOT report a crash when real dump lines were filtered: {tail}",
    );
}

#[test]
fn render_dmesg_tail_pure_corruption_no_dump_emits_corrupt_diag() {
    // Regression guard: stderr is pure corruption with NO
    // sched_ext_dump lines — the filter drops nothing, the
    // corruption branch surfaces the crash diagnostic. The
    // pointer-diag tightening must NOT regress this.
    let tail = render_dmesg_tail("\u{fffd}\u{fffd}\u{fffd}", 40);
    assert!(
        tail.contains("scheduler crashed") || tail.contains("UART buffer"),
        "pure-corruption stderr must surface the corruption diagnostic: {tail}",
    );
}

#[test]
fn render_dmesg_tail_pure_whitespace_no_dump_emits_empty_diag() {
    // Regression guard for the whitespace path without dump line —
    // the empty/whitespace classifier branch must still surface.
    let tail = render_dmesg_tail("  \n\t\n", 40);
    assert!(
        tail.contains("scheduler crashed before kernel printk"),
        "whitespace-only stderr (no dump line) must surface the \
         crash diagnostic: {tail}",
    );
}

#[test]
fn render_dmesg_tail_latin1_residue_no_dump_passes_through() {
    // Latin-1 audit: legitimate Latin-1 bytes (U+00A0..=U+00FF
    // outside the C1 control range U+0080..=U+009F) are NOT
    // corruption. A kernel printk that mentions a Latin-1 letter
    // (e.g. an NLS-translated filename, a hardware vendor string,
    // a USB descriptor) must format-tail-render unchanged, not
    // surface a crash diagnostic.
    let stderr = "[0.1] hw vendor: foo\u{ff}bar\n";
    let tail = render_dmesg_tail(stderr, 40);
    assert!(
        tail.contains("foo\u{ff}bar"),
        "Latin-1 residue must format-tail-render unchanged: {tail}"
    );
    assert!(
        !tail.contains("scheduler crashed"),
        "Latin-1 residue must NOT trigger the corruption diag: {tail}"
    );
}

#[test]
fn render_dmesg_tail_uses_tightened_marker() {
    // A printk that contains the substring "sched_ext_dump" WITHOUT
    // the trailing colon must NOT be stripped by the filter. The
    // tightened marker is `sched_ext_dump:` (colon required) — see
    // [`SCHED_EXT_DUMP_MARKER`]. A regression that loosens this to
    // a substring match would pull kernel BUG / systemd unit
    // references into the dump section.
    let stderr = "[  0.1] BUG in sched_ext_dump_disable callback\n";
    let tail = render_dmesg_tail(stderr, 40);
    assert!(
        tail.contains("BUG in sched_ext_dump_disable callback"),
        "non-marker line (no colon after) must survive the filter: {tail}",
    );
}

#[test]
fn render_dmesg_tail_bug_line_and_real_dump_split_correctly() {
    // Combined stderr: a kernel BUG line that mentions
    // `sched_ext_dump_disable` (bare-word) AND a real
    // `sched_ext_dump:` tracepoint line. The tightened marker
    // splits them correctly:
    //   - the dump line is filtered (goes to the dump section
    //     rendered separately via extract_sched_ext_dump);
    //   - the BUG line survives the filter and lands in the
    //     dmesg tail.
    let stderr = "[  0.1] BUG in sched_ext_dump_disable callback\n\
                  ktstr-0 [001] 0.5: sched_ext_dump: scheduler state\n";
    let tail = render_dmesg_tail(stderr, 40);
    assert!(
        tail.contains("BUG in sched_ext_dump_disable callback"),
        "BUG line must survive the filter and land in dmesg: {tail}",
    );
    assert!(
        !tail.contains("scheduler state"),
        "real dump line must be stripped from dmesg (it goes to \
         the dump section rendered separately): {tail}",
    );
}

// -- end-to-end: extract_probe_output carries the new diagnostics --
//
// These tests are the host-only equivalent of a synthetic
// ProbeBytes E2E. They construct a full
// ProbeBytes payload that mirrors what a guest VM emits over
// COM2 between the PROBE_OUTPUT_START / _END sentinels, then
// assert the host-side extract_probe_output reproduces the
// diagnostic line through the normal parse → format pipeline
// (no shortcuts into format_probe_diagnostics directly). A
// regression that breaks the end-to-end wiring (e.g.
// ProbeBytesDiagnostics serde drift, format_probe_diagnostics
// getting bypassed in extract_probe_output) shows up here, not
// only in the unit-level helper tests.

#[test]
fn extract_probe_output_emits_kind_stall_diagnostic_end_to_end() {
    // Symptom that motivated the fix: `events: 146 captured, 0
    // after stitch` with `bpf_trigger_fires: 1` and
    // `bpf_exit_kind_snap: SCX_EXIT_ERROR_STALL`. Wire up the
    // payload as the guest would emit it; the host renderer
    // MUST surface "kind=STALL" in the diagnostics block.
    use crate::probe::process::ProbeDiagnostics;
    use crate::probe::scx_defs::EXIT_ERROR_STALL;
    let skeleton = ProbeDiagnostics {
        events_before_stitch: 146,
        events_after_stitch: 0,
        bpf_trigger_fires: 1,
        bpf_exit_kind_snap: EXIT_ERROR_STALL as u32,
        ..Default::default()
    };
    let payload = ProbeBytes {
        events: Vec::new(),
        func_names: Vec::new(),
        bpf_source_locs: Default::default(),
        diagnostics: Some(ProbeBytesDiagnostics {
            pipeline: PipelineDiagnostics::default(),
            skeleton,
        }),
        nr_cpus: None,
        param_names: Default::default(),
        render_hints: Default::default(),
    };
    let json = serde_json::to_string(&payload).unwrap();
    let output = format!("noise\n{PROBE_OUTPUT_START}\n{json}\n{PROBE_OUTPUT_END}\n");
    let formatted = extract_probe_output(&output, None, None)
        .expect("ProbeBytes with diagnostics must produce some output");
    assert!(
        formatted.contains("--- probe pipeline ---"),
        "missing pipeline header: {formatted}"
    );
    assert!(
        formatted.contains("146 captured, 0 after stitch"),
        "missing events counter line: {formatted}"
    );
    assert!(
        formatted.contains("kind=STALL"),
        "missing kind=STALL diagnostic in end-to-end output: {formatted}"
    );
}

#[test]
fn extract_probe_output_emits_trigger_never_fired_end_to_end() {
    // Lifecycle race symptom: `bpf_trigger_fires: 0`. The host
    // renderer MUST say "trigger never fired" in the diagnostics
    // block so the operator chases the lifecycle bug rather than
    // a stitch failure.
    use crate::probe::process::ProbeDiagnostics;
    let skeleton = ProbeDiagnostics {
        events_before_stitch: 146,
        events_after_stitch: 0,
        bpf_trigger_fires: 0,
        bpf_exit_kind_snap: 0,
        bpf_kprobe_fires: 16567,
        bpf_meta_misses: 0,
        ..Default::default()
    };
    let payload = ProbeBytes {
        events: Vec::new(),
        func_names: Vec::new(),
        bpf_source_locs: Default::default(),
        diagnostics: Some(ProbeBytesDiagnostics {
            pipeline: PipelineDiagnostics::default(),
            skeleton,
        }),
        nr_cpus: None,
        param_names: Default::default(),
        render_hints: Default::default(),
    };
    let json = serde_json::to_string(&payload).unwrap();
    let output = format!("{PROBE_OUTPUT_START}\n{json}\n{PROBE_OUTPUT_END}");
    let formatted = extract_probe_output(&output, None, None)
        .expect("ProbeBytes with diagnostics must produce some output");
    assert!(
        formatted.contains("trigger never fired"),
        "missing 'trigger never fired' diagnostic: {formatted}"
    );
    assert!(
        formatted.contains("16567 kprobe fires"),
        "missing bpf_counts line: {formatted}"
    );
}

#[test]
fn extract_probe_output_emits_fallback_marker_end_to_end() {
    // When the lifecycle race still loses (e.g. kernel doesn't
    // call scx_claim_exit at all in some edge case), the
    // best-effort fallback must produce events with the
    // grouped-by-frequency marker — NOT a silent empty section.
    use crate::probe::process::{ProbeDiagnostics, ProbeEvent};
    let skeleton = ProbeDiagnostics {
        events_before_stitch: 146,
        events_after_stitch: 80,
        stitch_fallback_used: true,
        bpf_trigger_fires: 0,
        ..Default::default()
    };
    // One synthetic event so the formatter actually emits an
    // events block — pin both the diagnostic AND the events
    // line being present together.
    let payload = ProbeBytes {
        events: vec![ProbeEvent {
            func_idx: 0,
            task_ptr: 0xa,
            ts: 100,
            args: [0; 6],
            fields: Vec::new(),
            kstack: Vec::new(),
            str_val: None,
            ..Default::default()
        }],
        func_names: vec![(0, "schedule".to_string())],
        bpf_source_locs: Default::default(),
        diagnostics: Some(ProbeBytesDiagnostics {
            pipeline: PipelineDiagnostics::default(),
            skeleton,
        }),
        nr_cpus: None,
        param_names: Default::default(),
        render_hints: Default::default(),
    };
    let json = serde_json::to_string(&payload).unwrap();
    let output = format!("{PROBE_OUTPUT_START}\n{json}\n{PROBE_OUTPUT_END}");
    let formatted = extract_probe_output(&output, None, None)
        .expect("ProbeBytes with events must produce output");
    assert!(
        formatted.contains("trigger absent") && formatted.contains("frequency"),
        "fallback marker missing from end-to-end output: {formatted}"
    );
}

// -- deferred probe stash / take --
//
// The lifecycle fix adds a process-wide stash for the probe
// stop+handle so the rust_init Phase 6 finaliser can drain it
// after `child.kill()`. Pin the stash/take API behaviour so a
// future refactor doesn't accidentally reintroduce the
// immediate-detach path that was the original bug.
//
// The mutex guarding the stash is process-wide and the tests
// run in parallel within one process, so all stash/take
// exercises live in a single test that holds an internal
// serialisation mutex for the duration of every assertion.
// This keeps the stash invariants pinned without creating
// cross-test interference.

#[test]
fn deferred_probe_stash_take_invariants() {
    // Process-wide guard: every assertion below executes under the
    // shared `DEFERRED_PROBE_TEST_LOCK` so concurrent unit tests
    // that also touch `DEFERRED_PROBE_COLLECT` (e.g.
    // `finalize_probe_after_unwind_noop_*`,
    // `publish_result_and_collect_stash_arm_*`) cannot interleave a
    // stash between this test's drain and its assertions.
    let _guard = DEFERRED_PROBE_TEST_LOCK.lock().unwrap();

    // Drain any prior state (paranoid; a prior test on the
    // same process should have cleared, but a re-entrant
    // future test could leave a stash behind).
    let _ = take_deferred_probe();

    // 1) Empty stash → take returns None.
    assert!(
        take_deferred_probe().is_none(),
        "empty stash must yield None"
    );

    // 2) Single stash → take round-trips and drains.
    let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    stash_deferred_probe(stop.clone(), None);
    let taken = take_deferred_probe().expect("stash must round-trip");
    assert!(
        !taken.stop.load(std::sync::atomic::Ordering::Relaxed),
        "stop flag must round-trip with original value"
    );
    assert!(taken.handle.is_none(), "handle round-trip preserved None");
    assert!(
        take_deferred_probe().is_none(),
        "drained: subsequent take must return None"
    );

    // 3) Re-entrant stash → second value wins (single-shot
    // semantics: a stale prior value would belong to a
    // phantom run).
    let stop_a = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let stop_b = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(true));
    stash_deferred_probe(stop_a, None);
    stash_deferred_probe(stop_b, None);
    let taken = take_deferred_probe().expect("stash present");
    assert!(
        taken.stop.load(std::sync::atomic::Ordering::Relaxed),
        "second stash must win — got stop_a value instead of stop_b"
    );

    // Final drain so a subsequent test (if any) starts clean.
    let _ = take_deferred_probe();
}

/// `label_repro_verdict_when_workload_not_reached` prepends the
/// cautionary label when the primary VM did NOT reach its
/// workload (no `PayloadStarting` frame). The repro verdict text
/// appears on the following line so operators see both.
#[test]
fn label_repro_verdict_wraps_when_primary_did_not_reach_workload() {
    use super::label_repro_verdict_when_workload_not_reached;
    let verdict = "repro VM: scheduler ran normally (crash did not reproduce)";
    let wrapped = label_repro_verdict_when_workload_not_reached(false, verdict);
    let lines: Vec<&str> = wrapped.lines().collect();
    assert!(
        lines.len() >= 2,
        "wrap must put the original verdict on a new line: {wrapped}",
    );
    assert!(
        lines[0].starts_with("PRIMARY DID NOT REACH WORKLOAD"),
        "first line must lead with the cautionary label: {wrapped}",
    );
    assert!(
        wrapped.contains("not load-bearing"),
        "wrap must call out that auto-repro is not load-bearing: {wrapped}",
    );
    assert_eq!(
        *lines.last().unwrap(),
        verdict,
        "last line must be the unmodified original verdict: {wrapped}",
    );
}

/// False-positive guard: when the primary DID reach its workload
/// (the `PayloadStarting` lifecycle frame fired), the repro
/// verdict passes through unchanged — a clean repro run is
/// load-bearing evidence about reproducibility in that case.
#[test]
fn label_repro_verdict_passthrough_when_primary_reached_workload() {
    use super::label_repro_verdict_when_workload_not_reached;
    let verdict = "repro VM: scheduler ran normally (crash did not reproduce)";
    let wrapped = label_repro_verdict_when_workload_not_reached(true, verdict);
    assert_eq!(
        wrapped, verdict,
        "primary reached workload; verdict must NOT be wrapped: got {wrapped}",
    );
}

/// `primary_reached_workload` returns true iff a
/// `LifecyclePhase::PayloadStarting` frame is present on the
/// primary's bulk-port drain. The check operates DIRECTLY on the
/// frame, NOT on `classify_init_stage`'s stage-string bucketing
/// — the bucketing lumps `SchedulerNotAttached` (pre-workload
/// failure) with `PayloadStarting`, so a stage-string-based gate
/// would mislabel "scheduler failed to attach" as "reached
/// workload."
#[test]
fn primary_reached_workload_distinguishes_payload_starting_from_scheduler_not_attached() {
    use crate::test_support::output::primary_reached_workload;
    use crate::vmm::host_comms::BulkDrainResult;
    use crate::vmm::wire::{LifecyclePhase, MSG_TYPE_LIFECYCLE, ShmEntry};

    let phase_entry = |phase: LifecyclePhase| ShmEntry {
        msg_type: MSG_TYPE_LIFECYCLE,
        crc_ok: true,
        payload: vec![phase.wire_value()],
    };

    // Drain with SchedulerNotAttached only — no PayloadStarting.
    // primary_reached_workload uses the PayloadStarting frame
    // DIRECTLY, not classify_init_stage's stage string. Decoupling
    // the auto-repro gate from the classifier ensures a future
    // regression to the classifier's bucketing can't silently
    // re-introduce the SchedulerNotAttached-misclassification bug
    // here.
    let drain_only_not_attached = BulkDrainResult {
        entries: vec![phase_entry(LifecyclePhase::SchedulerNotAttached)],
    };
    assert!(
        !primary_reached_workload(Some(&drain_only_not_attached)),
        "SchedulerNotAttached without PayloadStarting must NOT count as \
         reached workload — the auto-repro gate must check the frame \
         directly, not via any future stage-string bucketing",
    );

    // Drain with PayloadStarting present — DID reach workload.
    let drain_with_payload = BulkDrainResult {
        entries: vec![phase_entry(LifecyclePhase::PayloadStarting)],
    };
    assert!(
        primary_reached_workload(Some(&drain_with_payload)),
        "PayloadStarting frame must count as reached workload",
    );

    // No drain — counts as not reached.
    assert!(
        !primary_reached_workload(None),
        "missing drain must NOT count as reached workload",
    );

    // Drain with InitStarted only — not reached.
    let drain_only_init = BulkDrainResult {
        entries: vec![phase_entry(LifecyclePhase::InitStarted)],
    };
    assert!(
        !primary_reached_workload(Some(&drain_only_init)),
        "InitStarted without PayloadStarting must NOT count as reached workload",
    );

    // Empty drain (Some, no entries) — not reached. The
    // `.any()` predicate on an empty iter is false, so this
    // works by construction, but pin it so a future refactor
    // that adds a "fallback" branch for the empty case can't
    // silently flip the answer.
    let empty_drain = BulkDrainResult { entries: vec![] };
    assert!(
        !primary_reached_workload(Some(&empty_drain)),
        "empty drain (Some, but no entries) must NOT count as reached workload",
    );

    // CRC-bad PayloadStarting frame — not reached. The
    // predicate filters on `crc_ok` (`primary_reached_workload`'s
    // `e.crc_ok` filter) so a
    // corrupted PayloadStarting frame must not satisfy the
    // gate. Mirrors classify_init_stage_skips_crc_bad_lifecycle_frames
    // for primary_reached_workload's parallel filter.
    let mut crc_bad = BulkDrainResult {
        entries: vec![phase_entry(LifecyclePhase::PayloadStarting)],
    };
    crc_bad.entries[0].crc_ok = false;
    assert!(
        !primary_reached_workload(Some(&crc_bad)),
        "CRC-bad PayloadStarting frame must NOT count as reached workload",
    );
}

// -- write_auto_repro_sidecar_artifacts --

/// Build a minimal `VmResult` whose `guest_messages` is `Some(...)`
/// with the supplied entries. All other fields take fixture
/// defaults via [`crate::vmm::result::VmResult::test_fixture`].
///
/// Gated on `wprof` with the sidecar-writer tests it feeds:
/// [`write_auto_repro_sidecar_artifacts`] exists only under `wprof`, so
/// its callers here (and this shared fixture) are dead code otherwise.
#[cfg(feature = "wprof")]
fn vm_result_with_drain(entries: Vec<crate::vmm::wire::ShmEntry>) -> crate::vmm::result::VmResult {
    crate::vmm::result::VmResult {
        guest_messages: Some(crate::vmm::host_comms::BulkDrainResult { entries }),
        ..crate::vmm::result::VmResult::test_fixture()
    }
}

/// A `MsgType::WprofTrace` frame containing a 4-byte protobuf-shaped
/// payload (`0x0a 0x02 'h' 'i'` = field=1 wire-type=2 length=2
/// followed by "hi"). The bytes don't need to be a real Perfetto
/// proto for the helper's write-to-disk contract — the helper is
/// payload-opaque.
#[cfg(feature = "wprof")]
fn wprof_frame(payload: &[u8], crc_ok: bool) -> crate::vmm::wire::ShmEntry {
    crate::vmm::wire::ShmEntry {
        msg_type: crate::vmm::wire::MsgType::WprofTrace.wire_value(),
        payload: payload.to_vec(),
        crc_ok,
    }
}

/// CRC-OK WprofTrace frame writes
/// `${entry.name}-${variant_hash:016x}.repro.wprof.pb` to sidecar_dir
/// with the exact payload bytes. Pins the no-silent-drop contract on
/// the wprof bulk-drain dispatch arm.
#[cfg(feature = "wprof")]
#[test]
fn write_auto_repro_sidecar_artifacts_writes_wprof_pb() {
    let _env_lock = crate::test_support::test_helpers::lock_env();
    let tmp = tempfile::tempdir().expect("tempdir");
    let _sidecar = crate::test_support::test_helpers::EnvVarGuard::set(
        crate::KTSTR_SIDECAR_DIR_ENV,
        tmp.path(),
    );
    let entry = crate::test_support::test_helpers::eevdf_entry("write_auto_repro_wprof_fixture");
    let payload = b"\x0a\x02hi";
    let result = vm_result_with_drain(vec![wprof_frame(payload, true)]);
    write_auto_repro_sidecar_artifacts(&entry, &result);
    let pb = tmp
        .path()
        .join("write_auto_repro_wprof_fixture-0000000000000000.repro.wprof.pb");
    assert!(pb.exists(), "expected wprof .pb at {}", pb.display());
    assert_eq!(
        std::fs::read(&pb).expect("read wprof .pb"),
        payload,
        "payload must round-trip byte-for-byte",
    );
}

/// CRC-bad WprofTrace frames are skipped — a corrupted payload
/// would mask the corruption if written. Pins the CRC gate at
/// [`write_auto_repro_sidecar_artifacts`].
#[cfg(feature = "wprof")]
#[test]
fn write_auto_repro_sidecar_artifacts_skips_crc_bad_wprof() {
    let _env_lock = crate::test_support::test_helpers::lock_env();
    let tmp = tempfile::tempdir().expect("tempdir");
    let _sidecar = crate::test_support::test_helpers::EnvVarGuard::set(
        crate::KTSTR_SIDECAR_DIR_ENV,
        tmp.path(),
    );
    let entry = crate::test_support::test_helpers::eevdf_entry("write_auto_repro_crc_bad_fixture");
    let result = vm_result_with_drain(vec![wprof_frame(b"garbage", false)]);
    write_auto_repro_sidecar_artifacts(&entry, &result);
    let pb = tmp
        .path()
        .join("write_auto_repro_crc_bad_fixture-0000000000000000.repro.wprof.pb");
    assert!(
        !pb.exists(),
        "crc_ok=false WprofTrace must NOT produce a sidecar file at {}",
        pb.display(),
    );
}

/// Regression pin: the writer's on-disk repro wprof name must equal
/// `VmResult::repro_wprof_pb_path()` — the path the expect_auto_repro
/// inversion resolves. The non-zero `variant_hash` proves the hash
/// VALUE flows into the name, so a writer that drops or re-derives a
/// diverging name (the bug this fix closed) fails here.
#[cfg(feature = "wprof")]
#[test]
fn write_auto_repro_sidecar_artifacts_name_matches_resolver() {
    let _env_lock = crate::test_support::test_helpers::lock_env();
    let tmp = tempfile::tempdir().expect("tempdir");
    let _sidecar = crate::test_support::test_helpers::EnvVarGuard::set(
        crate::KTSTR_SIDECAR_DIR_ENV,
        tmp.path(),
    );
    let entry = crate::test_support::test_helpers::eevdf_entry("name_matches_resolver_fixture");
    let result = crate::vmm::result::VmResult {
        entry_name: Some("name_matches_resolver_fixture"),
        variant_hash: 0xab,
        ..vm_result_with_drain(vec![wprof_frame(b"\x0a\x02hi", true)])
    };
    write_auto_repro_sidecar_artifacts(&entry, &result);
    let resolver_path = result.repro_wprof_pb_path().expect("resolve repro path");
    assert!(
        resolver_path.exists(),
        "writer must write at the resolver path {} (writer==reader)",
        resolver_path.display(),
    );
}

// -- wait_for_sched_disabled --

/// `wait_for_sched_disabled_at` covers all three arms deterministically
/// through a controlled path. The live `/sys/kernel/sched_ext/state`
/// is unsuitable: a kernel built with `CONFIG_SCHED_CLASS_EXT` and no
/// attached scheduler reads `"disabled"`, so the file is neither
/// reliably absent nor reliably non-`"disabled"` across hosts.
///   - file unreadable → `false` immediately (the else arm), no spin
///   - contents trimming to `"disabled"` → `true` (the success arm)
///   - readable but non-`"disabled"` + tiny timeout → `false` (the
///     deadline arm), bounded by the deadline rather than spinning
#[test]
fn wait_for_sched_disabled_at_covers_all_arms() {
    let tiny = std::time::Duration::from_millis(1);

    // else arm: a path that does not exist → read_to_string Err →
    // false, returned without spinning.
    assert!(
        !wait_for_sched_disabled_at("/nonexistent/ktstr/sched_ext/state", tiny),
        "unreadable state path must yield a non-spinning false",
    );

    let tmp = tempfile::tempdir().expect("tempdir");

    // success arm: contents trim to "disabled" → true.
    let disabled = tmp.path().join("disabled_state");
    std::fs::write(&disabled, "disabled\n").expect("write disabled state");
    assert!(
        wait_for_sched_disabled_at(disabled.to_str().expect("utf8 path"), tiny),
        "state reading \"disabled\" must yield true",
    );

    // deadline arm: readable but not "disabled" → loop until the tiny
    // deadline elapses → false (never true, never an unbounded spin).
    let enabled = tmp.path().join("enabled_state");
    std::fs::write(&enabled, "enabled\n").expect("write enabled state");
    assert!(
        !wait_for_sched_disabled_at(enabled.to_str().expect("utf8 path"), tiny),
        "non-\"disabled\" state must time out to false within the deadline",
    );
}

// -- finalize_probe_after_unwind (no-op path) --

/// `finalize_probe_after_unwind` is a no-op when no `DeferredProbe`
/// was stashed: the `let Some(deferred) = take_deferred_probe()
/// else { return; }` early-return in `finalize_probe_after_unwind`
/// leaves the global `DEFERRED_PROBE_COLLECT` static untouched and
/// never reaches `wait_for_sched_disabled` /
/// `collect_and_print_probe_data`.
/// Drain to a known-empty precondition, call the finaliser, then
/// assert the static is still empty — the no-op path must NOT stash
/// anything and must NOT panic.
#[test]
fn finalize_probe_after_unwind_noop_when_nothing_stashed() {
    let _guard = DEFERRED_PROBE_TEST_LOCK.lock().unwrap();
    // Drain any residue so the empty precondition is known.
    let _ = take_deferred_probe();
    assert!(
        take_deferred_probe().is_none(),
        "precondition: stash must be empty before the no-op call",
    );
    finalize_probe_after_unwind();
    assert!(
        take_deferred_probe().is_none(),
        "no-op path must leave DEFERRED_PROBE_COLLECT empty \
         (the early-return in finalize_probe_after_unwind)",
    );
}

// -- collect_and_print_probe_data (handle:None path) --

/// `collect_and_print_probe_data` returns at the `let Some(ph) =
/// handle else { return; }` early-return when `handle` is `None`,
/// BEFORE the `stop.store(true, Release)`. Asserting the stop flag
/// stays `false` (not merely "no panic") distinguishes the
/// early-return from the join path, which would have set it to
/// `true`.
#[test]
fn collect_and_print_probe_data_none_handle_leaves_stop_false() {
    let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    collect_and_print_probe_data(stop.clone(), None);
    assert!(
        !stop.load(std::sync::atomic::Ordering::Acquire),
        "None-handle path must early-return before stop.store, \
         leaving stop == false",
    );
}

// -- publish_result_and_collect (host arm / stash arm) --

/// Host arm: with `is_guest()` forced `false`, `publish_result_and_collect`
/// takes the else branch
/// (`collect_and_print_probe_data(stop, None)`), which early-returns
/// on the `None` handle before `stop.store`. `try_flush_profraw`
/// (host no-op: the `cfg(coverage)` body early-returns when
/// `!is_guest()`, and is absent in non-coverage builds) and
/// `print_assert_result` -> `send_test_result` (host no-op via
/// `write_msg`'s `assert_guest_context` guard in `vmm::guest_comms`)
/// run without effect. Asserting `stop == false` proves the host
/// branch ran the collect-and-early-return path without a handle.
#[test]
fn publish_result_and_collect_host_arm_leaves_stop_false() {
    let _g = crate::vmm::guest_comms::IsGuestOverrideGuard::new(false);
    let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    publish_result_and_collect(&AssertResult::pass(), stop.clone(), None);
    assert!(
        !stop.load(std::sync::atomic::Ordering::Acquire),
        "host arm calls collect_and_print_probe_data(stop, None) which \
         early-returns before stop.store; stop must stay false",
    );
}

/// Stash arm: with `is_guest()` forced `true`, `publish_result_and_collect`
/// takes the `stash_deferred_probe(stop, handle)` branch — which
/// stores `Some(DeferredProbe)` into the global static even with
/// `handle = None` (`stash_deferred_probe` always wraps in `Some`).
/// After the call, `take_deferred_probe()` must return `Some`
/// (proves the stash branch executed); then drain it so a subsequent
/// deferred-probe test
/// starts clean. `IsGuestOverrideGuard` is thread-local
/// (guest_comms.rs), but the global static needs the shared
/// `DEFERRED_PROBE_TEST_LOCK`.
#[test]
fn publish_result_and_collect_stash_arm_stashes_deferred() {
    let _guard = DEFERRED_PROBE_TEST_LOCK.lock().unwrap();
    // Drain residue under the lock so the stash we observe is ours.
    let _ = take_deferred_probe();
    let _g = crate::vmm::guest_comms::IsGuestOverrideGuard::new(true);
    let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    publish_result_and_collect(&AssertResult::pass(), stop, None);
    let stashed = take_deferred_probe();
    assert!(
        stashed.is_some(),
        "guest arm must stash a DeferredProbe via stash_deferred_probe; \
         take_deferred_probe must return Some",
    );
    assert!(
        stashed.expect("stash present").handle.is_none(),
        "stashed handle must round-trip the None passed in",
    );
    // Final drain so a subsequent deferred-probe test starts clean.
    let _ = take_deferred_probe();
}

// -- emit_probe_payload (empty-events path) --

/// `emit_probe_payload` with empty `events` takes the
/// `let bpf_source_locs = if events.is_empty()` fast path — an empty
/// `bpf_source_locs` map, skipping the libbpf `discover_bpf_symbols`
/// / `resolve_bpf_source_locs` walk (that `if`'s `else` branch) that needs
/// a live BPF env. It then builds `ProbeBytes`, serializes, and
/// `println!`s the START/JSON/END markers (the tail of `emit_probe_payload`).
///
/// Two halves, landed together:
/// 1. DIRECT call for line coverage of `emit_probe_payload`'s
///    empty-events fast path + marker emission.
///    The fn writes to process stdout, which a std unit test cannot
///    cleanly capture, so the direct call alone can only assert
///    no-panic.
/// 2. ROUND-TRIP behavioral assertion: reconstruct the SAME
///    `ProbeBytes` the empty branch builds, wrap it in the
///    START/END markers, and run it through `extract_probe_output`.
///    With default diagnostics, `format_probe_diagnostics` always
///    pushes `"--- probe pipeline ---"` (its `out.push_str("--- probe pipeline ---\n")`)
///    and, since
///    `events_before_stitch == 0`, the bare
///    `"0 captured, 0 after stitch"` events line (its
///    `"  events:      {} captured, {} after stitch"` push, no
///    stitch-drop cause appended). `extract_probe_output` returns
///    `Some(out)` (diagnostics `Some` -> out non-empty, then events empty ->
///    `Some(out)` via its `if payload.events.is_empty()` arm).
///    Assert the exact substrings.
#[test]
fn emit_probe_payload_empty_events_round_trips_diagnostics_only() {
    // (1) Direct call: covers the empty-events fast path lines.
    // Output goes to process stdout (uncapturable in a std unit
    // test); this half pins no-panic / line coverage only.
    emit_probe_payload(
        &[],
        &[],
        &PipelineDiagnostics::default(),
        &crate::probe::process::ProbeDiagnostics::default(),
        &std::collections::HashMap::new(),
        &std::collections::HashMap::new(),
    );

    // (2) Round-trip: reconstruct the exact ProbeBytes the empty
    // branch builds and assert the host renderer's output shape.
    let payload = ProbeBytes {
        events: vec![],
        func_names: vec![],
        bpf_source_locs: std::collections::HashMap::new(),
        diagnostics: Some(ProbeBytesDiagnostics {
            pipeline: PipelineDiagnostics::default(),
            skeleton: crate::probe::process::ProbeDiagnostics::default(),
        }),
        nr_cpus: crate::probe::output::get_nr_cpus(),
        param_names: std::collections::HashMap::new(),
        render_hints: std::collections::HashMap::new(),
    };
    let json = serde_json::to_string(&payload).expect("serialize empty-events payload");
    let output = format!("{PROBE_OUTPUT_START}\n{json}\n{PROBE_OUTPUT_END}");
    // Default diagnostics make format_probe_diagnostics emit a
    // non-empty pipeline string, so extract_probe_output returns
    // Some(out) even though events is empty (NOT None).
    let formatted = extract_probe_output(&output, None, None)
        .expect("empty-events payload with default diagnostics must render the pipeline header");
    assert!(
        formatted.contains("--- probe pipeline ---"),
        "diagnostics header missing (format_probe_diagnostics's `--- probe pipeline ---` push): {formatted}",
    );
    assert!(
        formatted.contains("0 captured, 0 after stitch"),
        "empty-events stitch counters missing (format_probe_diagnostics's \
         `{{}} captured, {{}} after stitch` line, no cause \
         appended since events_before_stitch == 0): {formatted}",
    );
}
