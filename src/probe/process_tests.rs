//! Unit tests for [`super`] (the `probe::process` module).
//! Co-located via the `tests` submodule pattern (sibling file).

#![cfg(test)]

use super::*;

#[test]
fn bounded_libbpf_tail_keeps_specific_final_diagnostics_and_caps_output() {
    let log = format!(
        "{}\n\
         libbpf: prog 'ktstr_trigger_vexit_return': BPF program load failed\n\
         libbpf: failed to load raw scx_vexit return probe: -22\n",
        "unhelpful verifier prelude\n".repeat(300)
    );
    let tail = bounded_libbpf_tail(&log);
    assert!(
        tail.contains("prog 'ktstr_trigger_vexit_return'"),
        "program identity must survive: {tail}"
    );
    assert!(
        tail.contains("raw scx_vexit return probe"),
        "target failure must survive: {tail}"
    );
    assert!(
        tail.chars().count() <= 1_000,
        "early init error must stay bounded: {} chars",
        tail.chars().count()
    );
}

#[test]
fn scheduler_exit_trigger_selection_prefers_tracepoint_and_covers_older_shapes() {
    assert_eq!(
        select_scheduler_exit_trigger(true, true, true),
        Some(SchedulerExitTrigger::SchedExtExitTracepoint),
        "the exact post-claim tracepoint must win when all generations coexist"
    );
    assert_eq!(
        select_scheduler_exit_trigger(true, false, false),
        Some(SchedulerExitTrigger::SchedExtExitTracepoint)
    );
    assert_eq!(
        select_scheduler_exit_trigger(false, true, true),
        Some(SchedulerExitTrigger::ScxVexitKprobePair),
        "without the tracepoint, the raw post-claim entry/return pair wins"
    );
    assert_eq!(
        select_scheduler_exit_trigger(false, true, false),
        Some(SchedulerExitTrigger::ScxVexitKprobePair)
    );
    assert_eq!(
        select_scheduler_exit_trigger(false, false, true),
        Some(SchedulerExitTrigger::ScxDumpState),
        "6.14 must select its filtered global-era dump entry"
    );
    assert_eq!(
        select_scheduler_exit_trigger(false, false, false),
        None,
        "missing every compatible typed target must fail closed before object load"
    );
}

#[test]
fn scheduler_exit_autoload_selects_one_complete_generation() {
    assert_eq!(
        scheduler_exit_program_selection(SchedulerExitTrigger::SchedExtExitTracepoint),
        SchedulerExitProgramSelection {
            tracepoint: true,
            vexit_enter: false,
            vexit_return: false,
            dump_fentry: false,
        }
    );
    assert_eq!(
        scheduler_exit_program_selection(SchedulerExitTrigger::ScxVexitKprobePair),
        SchedulerExitProgramSelection {
            tracepoint: false,
            vexit_enter: true,
            vexit_return: true,
            dump_fentry: false,
        },
        "the raw scx_vexit path must load both correlation halves"
    );
    assert_eq!(
        scheduler_exit_program_selection(SchedulerExitTrigger::ScxDumpState),
        SchedulerExitProgramSelection {
            tracepoint: false,
            vexit_enter: false,
            vexit_return: false,
            dump_fentry: true,
        }
    );
}

/// Linux 7.3 introduces `sched_ext_exit` with
/// `TP_PROTO(struct scx_sched *sch, __u32 kind)`. `BPF_PROG` exposes those
/// explicit tracepoint arguments in order, so dropping the leading scheduler
/// pointer would reinterpret arg0 as `kind` and make the verifier reject the
/// resulting pointer-to-u32 operations. Keep the source-level ABI contract
/// pinned independently of the selector/autoload tests above.
#[test]
fn sched_ext_exit_source_matches_linux_7_3_tracepoint_abi() {
    let source = include_str!("../bpf/probe.bpf.c");
    let prototype = "int BPF_PROG(ktstr_trigger_tp, struct scx_sched *sch, unsigned int kind)";
    assert!(
        source.contains(prototype),
        "7.3 sched_ext_exit wrapper must bind `sch` to arg0 and `kind` to arg1"
    );
    assert!(
        !source.contains("int BPF_PROG(ktstr_trigger_tp, unsigned int kind)"),
        "one-argument wrapper reads the scheduler pointer as the exit kind"
    );

    let wrapper = source
        .split_once(prototype)
        .map(|(_, suffix)| suffix)
        .expect("sched_ext_exit wrapper prototype")
        .split_once("SEC(\"kprobe/scx_vexit\")")
        .map(|(body, _)| body)
        .expect("sched_ext_exit wrapper ends before the older-kernel fallback");
    for required in [
        "if (kind < SCX_EXIT_ERROR)",
        "ktstr_trigger_error(ctx, sch, kind,",
        "kind == SCX_EXIT_ERROR_BPF, true",
    ] {
        assert!(
            wrapper.contains(required),
            "7.3 sched_ext_exit wrapper lost `{required}`"
        );
    }
}

#[test]
fn raw_scx_vexit_source_pairs_entry_state_with_accepted_return() {
    let source = include_str!("../bpf/probe.bpf.c");
    for required in [
        "int ktstr_trigger_vexit_enter(struct pt_regs *ctx)",
        "bpf_map_update_elem(&ktstr_vexit_args, &pid_tgid, &args, BPF_ANY)",
        "int ktstr_trigger_vexit_return(struct pt_regs *ctx)",
        "bpf_map_lookup_elem(&ktstr_vexit_args, &pid_tgid)",
        "bpf_map_delete_elem(&ktstr_vexit_args, &pid_tgid)",
        "if (!PT_REGS_RC_CORE(ctx))",
        "args.kind == SCX_EXIT_ERROR_BPF",
        "args.kind == SCX_EXIT_ERROR_BPF, false",
    ] {
        assert!(
            source.contains(required),
            "raw scx_vexit entry/return contract lost `{required}`"
        );
    }

    let delete = source
        .find("bpf_map_delete_elem(&ktstr_vexit_args, &pid_tgid)")
        .expect("return probe deletes entry state");
    let accepted = source
        .find("if (!PT_REGS_RC_CORE(ctx))")
        .expect("return probe gates on successful scx_claim_exit");
    assert!(
        delete < accepted,
        "losing returns must still delete their entry-state slot"
    );
}

// -- parse_kallsyms --

#[test]
fn parse_kallsyms_happy_path() {
    // Canonical kallsyms layout: `HEX TYPE NAME`. The type column
    // is accepted but ignored by the parser.
    let raw = "ffffffff81000000 T _stext\n\
               ffffffff81000010 T schedule\n\
               ffffffff82000000 D init_mm\n";
    let map = parse_kallsyms(raw);
    assert_eq!(map.len(), 3);
    assert_eq!(map["_stext"], 0xffffffff81000000);
    assert_eq!(map["schedule"], 0xffffffff81000010);
    assert_eq!(map["init_mm"], 0xffffffff82000000);
}

#[test]
fn parse_kallsyms_skips_lines_missing_name() {
    // Lines with fewer than 3 tokens (addr, type, name — all
    // required) are dropped silently; the rest still parse.
    let raw = "\
        \n\
        ffffffff81000000\n\
        ffffffff81000010 T\n\
        ffffffff81000020 T real_sym\n";
    let map = parse_kallsyms(raw);
    assert_eq!(map.len(), 1);
    assert_eq!(map["real_sym"], 0xffffffff81000020);
}

#[test]
fn parse_kallsyms_skips_nonhex_addr() {
    // First token must parse as u64 hex; otherwise the line is
    // skipped and parsing continues on the next line.
    let raw = "garbage T should_skip\n\
               ffffffff81000000 T kept\n";
    let map = parse_kallsyms(raw);
    assert_eq!(map.len(), 1);
    assert_eq!(map["kept"], 0xffffffff81000000);
}

#[test]
fn parse_kallsyms_empty_input_yields_empty_map() {
    // Permanently-empty input is a valid parse, returning an empty
    // map rather than an error.
    let map = parse_kallsyms("");
    assert!(map.is_empty());
}

#[test]
fn parse_kallsyms_duplicate_name_keeps_last() {
    // HashMap::insert semantics: duplicate keys overwrite, so the
    // last occurrence wins. Callers that care about multiple
    // symbols with the same name would need a different parser.
    let raw = "ffffffff81000000 T dup\n\
               ffffffff82000000 T dup\n";
    let map = parse_kallsyms(raw);
    assert_eq!(map.len(), 1);
    assert_eq!(map["dup"], 0xffffffff82000000);
}

#[test]
fn parse_kallsyms_ignores_trailing_module_tag() {
    // Kernel-built-in symbols omit the trailing `[module]` tag;
    // module symbols include it. The parser only uses the first
    // 3 tokens, so trailing tokens (like `[mptcp]`) are dropped.
    let raw = "ffffffff81000000 T mod_sym\t[mptcp]\n";
    let map = parse_kallsyms(raw);
    assert_eq!(map.len(), 1);
    assert_eq!(map["mod_sym"], 0xffffffff81000000);
}

#[test]
fn build_field_keys_known_struct() {
    let func = super::BtfFunc {
        name: "test".into(),
        params: vec![super::super::btf::BtfParam {
            name: "p".into(),
            struct_name: Some("task_struct".into()),
            is_ptr: true,
            ..Default::default()
        }],
        ..Default::default()
    };
    let keys = build_field_keys(&func);
    assert!(
        keys.iter()
            .any(|(k, _)| k.contains("task_struct") && k.contains("pid"))
    );
    assert!(keys.iter().any(|(k, _)| k.contains("dsq_id")));
}

#[test]
fn build_field_keys_scalar_param() {
    let func = super::BtfFunc {
        name: "test".into(),
        params: vec![super::super::btf::BtfParam {
            name: "flags".into(),
            struct_name: None,
            is_ptr: false,
            ..Default::default()
        }],
        ..Default::default()
    };
    let keys = build_field_keys(&func);
    assert!(keys.iter().any(|(k, _)| k.contains("flags:val.flags")));
}

#[test]
fn build_field_keys_ptr_no_struct() {
    let func = super::BtfFunc {
        name: "test".into(),
        params: vec![super::super::btf::BtfParam {
            name: "ctx".into(),
            struct_name: None,
            is_ptr: true,
            ..Default::default()
        }],
        ..Default::default()
    };
    let keys = build_field_keys(&func);
    // Raw pointer with no struct info: no keys generated
    assert!(keys.is_empty());
}

#[test]
fn build_field_keys_empty_params() {
    let func = super::BtfFunc {
        name: "empty".into(),
        params: vec![],
        ..Default::default()
    };
    let keys = build_field_keys(&func);
    assert!(keys.is_empty());
}

#[test]
fn resolve_func_ip_nonexistent() {
    assert!(resolve_func_ip("__nonexistent_kernel_function_xyz__").is_none());
}

#[test]
fn build_field_keys_unknown_struct() {
    let func = super::BtfFunc {
        name: "test".into(),
        params: vec![super::super::btf::BtfParam {
            name: "p".into(),
            struct_name: Some("unknown_struct_xyz".into()),
            is_ptr: true,
            ..Default::default()
        }],
        ..Default::default()
    };
    let keys = build_field_keys(&func);
    assert!(keys.is_empty(), "unknown struct should produce no keys");
}

// -- detect_str_param --

#[test]
fn detect_str_param_btf_string_ptr() {
    // param[1] carries is_string_ptr (the BTF signal) but a name
    // ("p1") matching NO name heuristic; param[2] is a non-BTF ptr
    // whose name ("msg") DOES match the heuristic. The BTF loop runs
    // first and returns index 1; were the BTF is_string_ptr loop
    // deleted, the name fallback would instead return index 2. So
    // asserting == 1 pins BOTH the BTF branch and its precedence
    // over the name heuristic (a "fmt"-named string ptr could not —
    // "fmt" is itself a name-heuristic literal, masking the branch).
    let func = BtfFunc {
        name: "test".into(),
        params: vec![
            super::super::btf::BtfParam {
                name: "p".into(),
                struct_name: Some("task_struct".into()),
                is_ptr: true,
                ..Default::default()
            },
            super::super::btf::BtfParam {
                name: "p1".into(),
                struct_name: None,
                is_ptr: true,
                is_string_ptr: true,
                ..Default::default()
            },
            super::super::btf::BtfParam {
                name: "msg".into(),
                struct_name: None,
                is_ptr: true,
                ..Default::default()
            },
        ],
        ..Default::default()
    };
    assert_eq!(detect_str_param(&func), 1);
}

#[test]
fn detect_str_param_name_heuristic() {
    let func = BtfFunc {
        name: "test".into(),
        params: vec![
            super::super::btf::BtfParam {
                name: "flags".into(),
                struct_name: None,
                is_ptr: false,
                ..Default::default()
            },
            super::super::btf::BtfParam {
                name: "msg".into(),
                struct_name: None,
                is_ptr: true,
                ..Default::default()
            },
        ],
        ..Default::default()
    };
    assert_eq!(detect_str_param(&func), 1);
}

#[test]
fn detect_str_param_none() {
    let func = BtfFunc {
        name: "test".into(),
        params: vec![super::super::btf::BtfParam {
            name: "flags".into(),
            struct_name: None,
            is_ptr: false,
            ..Default::default()
        }],
        ..Default::default()
    };
    assert_eq!(detect_str_param(&func), 0xff);
}

#[test]
fn detect_str_param_struct_ptr_not_string() {
    let func = BtfFunc {
        name: "test".into(),
        params: vec![super::super::btf::BtfParam {
            name: "rq".into(),
            struct_name: Some("rq".into()),
            is_ptr: true,
            ..Default::default()
        }],
        ..Default::default()
    };
    assert_eq!(detect_str_param(&func), 0xff);
}

#[test]
fn detect_str_param_name_contains_str() {
    let func = BtfFunc {
        name: "test".into(),
        params: vec![super::super::btf::BtfParam {
            name: "my_str_ptr".into(),
            struct_name: None,
            is_ptr: true,
            ..Default::default()
        }],
        ..Default::default()
    };
    assert_eq!(detect_str_param(&func), 0);
}

// -- build_field_keys with auto_fields --

#[test]
fn build_field_keys_auto_fields() {
    let func = BtfFunc {
        name: "test".into(),
        params: vec![super::super::btf::BtfParam {
            name: "ctx".into(),
            struct_name: None,
            is_ptr: true,
            auto_fields: vec![
                ("field_a".into(), "->field_a".into(), RenderHint::Bool),
                ("field_b".into(), "->field_b".into(), RenderHint::Signed),
            ],
            type_name: Some("task_ctx".into()),
            ..Default::default()
        }],
        ..Default::default()
    };
    let keys = build_field_keys(&func);
    assert_eq!(keys.len(), 2);
    assert!(keys[0].0.contains("task_ctx"));
    assert!(keys[0].0.contains("field_a"));
    assert_eq!(keys[0].1, RenderHint::Bool);
    assert!(keys[1].0.contains("field_b"));
    assert_eq!(keys[1].1, RenderHint::Signed);
}

// -- build_field_keys with cpumask fields --

#[test]
fn build_field_keys_includes_cpumask_words() {
    let func = BtfFunc {
        name: "test".into(),
        params: vec![super::super::btf::BtfParam {
            name: "p".into(),
            struct_name: Some("task_struct".into()),
            is_ptr: true,
            ..Default::default()
        }],
        ..Default::default()
    };
    let keys = build_field_keys(&func);
    assert!(
        keys.iter().any(|(k, _)| k.contains("cpumask_0")),
        "should have cpumask_0: {keys:?}",
    );
    assert!(
        keys.iter().any(|(k, _)| k.contains("cpumask_3")),
        "should have cpumask_3: {keys:?}",
    );
}

#[test]
fn build_field_keys_max_six_params() {
    let params: Vec<_> = (0..8)
        .map(|i| super::super::btf::BtfParam {
            name: format!("p{i}"),
            struct_name: None,
            is_ptr: false,
            ..Default::default()
        })
        .collect();
    let func = super::BtfFunc {
        name: "many".into(),
        params,
        ..Default::default()
    };
    let keys = build_field_keys(&func);
    // Only first 6 params processed
    assert!(keys.len() <= 6);
    assert!(keys.iter().any(|(k, _)| k.contains("p5")));
    assert!(!keys.iter().any(|(k, _)| k.contains("p6")));
}

// ---- args[0] kind-conditional filter ----------------------------
//
// The args[0] conditional in the common BPF scheduler-exit
// trigger handler sets
// `event->args[0] = (kind == SCX_EXIT_ERROR_BPF)
// ? bpf_get_current_task() : 0;` — current-task is emitted
// ONLY for SCX_EXIT_ERROR_BPF (1025). For SCX_EXIT_ERROR
// (1024) the field is 0 because the exit can fire from
// kworker / sysrq context where `current` is unrelated.
//
// The `causal_tptr` filter in run_probe_skeleton returns
// None for events whose `task_ptr` (sourced from args[0]) is
// 0, dropping the verified-stitch path. run_probe_skeleton
// then falls back to frequency-grouped candidate task_ptrs
// and suppresses probe output entirely only when no candidate
// task_ptrs remain. These tests pin the host-side filter
// against both sides of that contract.
//
// SCX_EXIT_ERROR enum values (mirrored from kernel
// ext_internal.h, also defined in src/bpf/intf.h):
const SCX_EXIT_ERROR: u64 = 1024;
const SCX_EXIT_ERROR_BPF: u64 = 1025;

/// Build a synthetic `ProbeEvent` mirroring what the
/// ringbuf callback inside `run_probe_skeleton` constructs
/// from a trigger event. `args[0]` is the causal task
/// pointer the BPF side emitted (per the args[0] conditional
/// in the common trigger handler); `args[1]` is the exit kind.
/// `task_ptr` is set from `args[0]` in the trigger event
/// constructor in run_probe_skeleton.
fn make_trigger_event(args0: u64, kind: u64) -> ProbeEvent {
    let mut args = [0u64; 6];
    args[0] = args0;
    args[1] = kind;
    ProbeEvent {
        func_idx: 0,
        task_ptr: args0,
        ts: 0,
        args,
        fields: Vec::new(),
        kstack: Vec::new(),
        str_val: None,
        exit_fields: Vec::new(),
        exit_ts: None,
    }
}

#[test]
fn args0_zero_filtered_for_scx_exit_error() {
    // SCX_EXIT_ERROR (1024) fires from non-causal contexts
    // (kworker, sysrq) — the BPF side emits args[0] = 0,
    // so task_ptr is 0. The production `causal_tptr` filter
    // that `run_probe_skeleton` applies to the trigger
    // event's task_ptr must drop this (return None). Driving
    // the real `causal_tptr` (not a re-implemented copy)
    // means a regression that loosens the predicate (e.g.
    // `p >= 1`, or removing the guard entirely) surfaces here.
    let event = make_trigger_event(0, SCX_EXIT_ERROR);
    assert_eq!(
        event.task_ptr, 0,
        "SCX_EXIT_ERROR must propagate args[0]=0 into task_ptr"
    );
    assert_eq!(
        super::causal_tptr(event.task_ptr),
        None,
        "task_ptr=0 must be filtered out by the production \
         causal_tptr (no causal task → no stitch)"
    );
    assert_eq!(
        event.args[1], SCX_EXIT_ERROR,
        "args[1] must carry the exit kind for diagnostics"
    );
}

#[test]
fn args0_task_ptr_retained_for_scx_exit_error_bpf() {
    // SCX_EXIT_ERROR_BPF (1025) fires from a BPF callback in
    // the running task's context — the BPF side emits
    // args[0] = bpf_get_current_task() (a non-zero
    // task_struct pointer), so task_ptr is non-zero. The
    // production `causal_tptr` filter must retain this event
    // (return Some) so stitching can proceed. Asserting
    // against the real `causal_tptr` ties the test to the
    // code path `run_probe_skeleton` actually runs.
    const FAKE_TASK_PTR: u64 = 0xffff_8881_1234_5678; // plausible kernel VA
    let event = make_trigger_event(FAKE_TASK_PTR, SCX_EXIT_ERROR_BPF);
    assert_eq!(
        event.task_ptr, FAKE_TASK_PTR,
        "SCX_EXIT_ERROR_BPF must propagate args[0]=task_ptr into task_ptr"
    );
    assert_eq!(
        super::causal_tptr(event.task_ptr),
        Some(FAKE_TASK_PTR),
        "non-zero task_ptr must survive the production causal_tptr"
    );
    assert_eq!(
        event.args[1], SCX_EXIT_ERROR_BPF,
        "args[1] must carry the exit kind for diagnostics"
    );
}

// -- accept_kallsyms_map (kptr_restrict=2 cache poison guard) -----

#[test]
fn accept_kallsyms_map_rejects_all_zero_addresses() {
    // kernel.kptr_restrict=2 makes /proc/kallsyms readable but
    // zeros every address column. parse_kallsyms accepts the
    // file and yields a map with every value == 0. caching
    // that map would have resolve_func_ip return Some(0) for
    // every later lookup, masking the unprivileged state from
    // the retry-after-sudo path. accept_kallsyms_map must
    // collapse this case to None so the caller treats it as a
    // load failure and the retry clock keeps ticking.
    let raw = "0000000000000000 T schedule\n\
               0000000000000000 T do_exit\n\
               0000000000000000 D init_mm\n";
    let map = parse_kallsyms(raw);
    assert_eq!(map.len(), 3, "parser still records every line");
    assert!(
        map.values().all(|&a| a == 0),
        "kptr_restrict=2 must yield all-zero addresses",
    );
    assert!(
        accept_kallsyms_map(map).is_none(),
        "all-zero map must be rejected so the cache is not poisoned",
    );
}

#[test]
fn accept_kallsyms_map_accepts_when_any_nonzero() {
    // A single non-zero entry is enough to consider the file
    // genuinely populated — rate-limit pressure makes the
    // tighter test ("require ALL non-zero") wrong, since
    // legitimately exported partial dumps always contain a
    // few zero entries (NULL section markers).
    let raw = "0000000000000000 T zeroed\n\
               ffffffff81000000 T schedule\n";
    let map = parse_kallsyms(raw);
    assert_eq!(map.len(), 2);
    let accepted = accept_kallsyms_map(map).expect("mixed map must be accepted");
    assert_eq!(accepted["schedule"], 0xffffffff81000000);
    assert_eq!(accepted["zeroed"], 0);
}

#[test]
fn accept_kallsyms_map_rejects_empty_map() {
    // An empty map is vacuously all-zero — `any(|&a| a != 0)`
    // returns false on an empty iterator. Treat as a load
    // failure so the retry clock keeps ticking; otherwise an
    // empty-on-first-read race would freeze the cache at "no
    // symbols" forever.
    let map = std::collections::HashMap::<String, u64>::new();
    assert!(accept_kallsyms_map(map).is_none());
}

// -- build_task_param_idx out-of-bounds guard ---------------------

/// Construct a [`BtfFunc`] whose task_struct param sits at
/// `task_pos`. Pads earlier params with scalar `void *` entries
/// so the iterator's `.position` finds the task at the
/// requested index.
fn make_btf_with_task_at(name: &str, task_pos: usize) -> BtfFunc {
    let mut params = Vec::new();
    for i in 0..task_pos {
        params.push(super::super::btf::BtfParam {
            name: format!("a{i}"),
            struct_name: None,
            is_ptr: false,
            ..Default::default()
        });
    }
    params.push(super::super::btf::BtfParam {
        name: "p".into(),
        struct_name: Some("task_struct".into()),
        is_ptr: true,
        ..Default::default()
    });
    BtfFunc {
        name: name.to_string(),
        params,
        ..Default::default()
    }
}

#[test]
fn build_task_param_idx_drops_index_at_six() {
    // ProbeEvent::args is [u64; 6]. A task_struct param at
    // pidx=6 (i.e. arg 7) is past the captured slice — the
    // BPF probe never recorded that arg. Storing pidx=6 in
    // the stitch map would panic on `e.args[pidx]`, so the
    // builder MUST drop the entry rather than admit it.
    let func_ips = vec![(
        42u32,
        0xffff_ffff_8100_0000u64,
        "novel_callback".to_string(),
    )];
    let btf = vec![make_btf_with_task_at("novel_callback", 6)];
    let map = build_task_param_idx(&func_ips, &btf, &[]);
    assert!(
        !map.contains_key(&42),
        "pidx==6 must be dropped (args[6] is out of bounds for [u64; 6])",
    );
}

#[test]
fn build_task_param_idx_drops_index_above_six() {
    // Same boundary as pidx==6, but with a larger pidx to
    // catch a future off-by-one swap (`pidx > 6` instead of
    // `>= 6`) that would silently re-admit pidx=6.
    let func_ips = vec![(7u32, 0xffff_ffff_8100_0000u64, "wide_signature".to_string())];
    let btf = vec![make_btf_with_task_at("wide_signature", 9)];
    let map = build_task_param_idx(&func_ips, &btf, &[]);
    assert!(!map.contains_key(&7), "pidx==9 must be dropped");
}

#[test]
fn build_task_param_idx_keeps_index_at_five() {
    // pidx=5 IS the last valid slot (`args[5]`) — must be
    // kept. This is the boundary partner to the pidx==6
    // drop test: a regression that swaps `>= 6` to `>= 5`
    // would discard real, capturable callbacks.
    let func_ips = vec![(11u32, 0xffff_ffff_8100_0000u64, "tail_task".to_string())];
    let btf = vec![make_btf_with_task_at("tail_task", 5)];
    let map = build_task_param_idx(&func_ips, &btf, &[]);
    assert_eq!(map.get(&11).copied(), Some(5));
}

#[test]
fn build_task_param_idx_uses_bpf_op_callers_first() {
    // `BPF_OP_CALLERS` overrides BTF for the well-known
    // sched_ext op kernel callers — verifies the BTF fallback
    // doesn't shadow the canonical mapping. `enqueue_task_scx`
    // is registered with task_arg_idx=1 in the table; the
    // builder must return 1 even when the BTF (synthesized
    // here at task_pos=3) would say otherwise.
    let func_ips = vec![(
        0u32,
        0xffff_ffff_8100_0000u64,
        "enqueue_task_scx".to_string(),
    )];
    let btf = vec![make_btf_with_task_at("enqueue_task_scx", 3)];
    let map = build_task_param_idx(&func_ips, &btf, &[]);
    assert_eq!(
        map.get(&0).copied(),
        Some(1),
        "BPF_OP_CALLERS task_arg_idx (1) must win over BTF fallback (3)",
    );
}

#[test]
fn build_task_param_idx_phase_b_btf_chained() {
    // Phase B BTF must be searched as a fallback for funcs
    // not in the Phase A `btf_funcs` slice — the stitch map
    // must include Phase B–attached callbacks. Without this,
    // BPF callbacks discovered after the scheduler started
    // would never stitch.
    let func_ips = vec![(33u32, 0xffff_ffff_8100_0000u64, "phase_b_only".to_string())];
    let phase_b = vec![make_btf_with_task_at("phase_b_only", 2)];
    let map = build_task_param_idx(&func_ips, &[], &phase_b);
    assert_eq!(map.get(&33).copied(), Some(2));
}

#[test]
fn build_task_param_idx_skips_func_with_no_task_param() {
    // A function with no task_struct param produces no
    // entry — the stitch retain() falls back to `e.task_ptr ==
    // tptr` for those. Test the absence so a future change
    // that defaults to pidx=0 (silently mis-stitching by
    // arg[0]) is caught.
    let func_ips = vec![(99u32, 0xffff_ffff_8100_0000u64, "no_task".to_string())];
    let btf = vec![BtfFunc {
        name: "no_task".into(),
        params: vec![super::super::btf::BtfParam {
            name: "x".into(),
            struct_name: None,
            is_ptr: false,
            ..Default::default()
        }],
        ..Default::default()
    }];
    let map = build_task_param_idx(&func_ips, &btf, &[]);
    assert!(!map.contains_key(&99));
}

// -- set_probe_sched_exit_state Clean encode round-trip ------------

#[test]
fn set_probe_sched_exit_state_clean_round_trips() {
    // The Clean match arm in set_probe_sched_exit_state (encode
    // Clean -> PROBE_EXIT_STATE_CLEAN=1) is the only branch the
    // existing tests never exercise: production probe-poll code and
    // the scenario/ops tests drive only Crashed/Unknown. This pins
    // the encode side against the decode side in sched_exit_kind
    // (PROBE_EXIT_STATE_CLEAN=1 -> Clean), proving the constant
    // round-trips. PROBE_SCHED_EXIT_STATE is a process-global
    // AtomicU32; no real probe-poll thread runs in host-unit tests,
    // so this set/read pair is uncontended. Restore the mirror to
    // Unknown afterward so neighbor tests start clean, per the doc
    // on set_probe_sched_exit_state.
    super::set_probe_sched_exit_state(SchedExitKind::Clean);
    assert_eq!(
        super::sched_exit_kind(),
        SchedExitKind::Clean,
        "encode(Clean) must decode back to Clean (PROBE_EXIT_STATE_CLEAN=1 round-trip)",
    );
    super::set_probe_sched_exit_state(SchedExitKind::Unknown);
    assert_eq!(
        super::sched_exit_kind(),
        SchedExitKind::Unknown,
        "reset to Unknown must clear the mirror for neighbor tests",
    );
}

// -- build_field_keys 16-field cap (MAX_FIELDS) --------------------

#[test]
fn build_field_keys_caps_known_struct_at_sixteen() {
    // task_struct contributes 12 STRUCT_FIELDS keys per param. Two
    // task_struct params would naively yield 24 keys; the field_idx
    // >= 16 break inside the known-struct branch truncates the
    // second param after exactly 4 keys (field_idx 12..15). Pins
    // that cap so a regression loosening the guard (or dropping it)
    // is caught.
    let func = super::BtfFunc {
        name: "two_task_params".into(),
        params: vec![
            super::super::btf::BtfParam {
                name: "p1".into(),
                struct_name: Some("task_struct".into()),
                is_ptr: true,
                ..Default::default()
            },
            super::super::btf::BtfParam {
                name: "p2".into(),
                struct_name: Some("task_struct".into()),
                is_ptr: true,
                ..Default::default()
            },
        ],
        ..Default::default()
    };
    let keys = build_field_keys(&func);
    assert_eq!(
        keys.len(),
        16,
        "cap truncates at 16, not the naive 12+12=24"
    );
    assert_eq!(
        keys.iter().filter(|(k, _)| k.starts_with("p2:")).count(),
        4,
        "second task_struct param contributes exactly 4 keys before the cap",
    );
}

#[test]
fn build_field_keys_caps_auto_fields_at_sixteen() {
    // The auto_fields branch has its own field_idx >= 16 break
    // (distinct from the known-struct break). A single param with
    // 17 auto_fields must yield 16 keys, dropping the 17th ("f16").
    let func = super::BtfFunc {
        name: "wide_auto".into(),
        params: vec![super::super::btf::BtfParam {
            name: "ctx".into(),
            struct_name: None,
            is_ptr: true,
            type_name: Some("task_ctx".into()),
            auto_fields: (0..17)
                .map(|i| (format!("f{i}"), format!("->f{i}"), RenderHint::Hex))
                .collect(),
            ..Default::default()
        }],
        ..Default::default()
    };
    let keys = build_field_keys(&func);
    assert_eq!(keys.len(), 16, "17 auto_fields must be capped to 16 keys");
    assert!(
        !keys.iter().any(|(k, _)| k.contains(".f16")),
        "the 17th auto field (f16) must be dropped by the cap: {keys:?}",
    );
}

// -- set_rodata_slot out-of-range no-op + positive write ----------

#[test]
fn set_rodata_slot_ignores_slot_past_cap() {
    // Lines 858-874 (slot 0..=3) are covered by VM-running attach
    // tests; only the `_ => {}` no-op for slot > 3 is uncovered.
    // The doc states "No-op for slot indices outside 0..=3" but
    // nothing exercises that contract. Construct a host-side rodata
    // (POD, #[repr(C)], no Default derive — built via zeroed()),
    // write slot 4, and assert EVERY mutable field is unchanged.
    let mut r: crate::bpf_skel::fentry::types::rodata = unsafe { std::mem::zeroed() };
    set_rodata_slot(&mut r, 4, 0xABCD, true);
    assert_eq!(r.ktstr_fentry_func_idx_0, 0, "slot>3 must not touch idx_0");
    assert_eq!(r.ktstr_fentry_func_idx_1, 0, "slot>3 must not touch idx_1");
    assert_eq!(r.ktstr_fentry_func_idx_2, 0, "slot>3 must not touch idx_2");
    assert_eq!(r.ktstr_fentry_func_idx_3, 0, "slot>3 must not touch idx_3");
    assert_eq!(
        r.ktstr_fentry_is_kernel_0, 0,
        "slot>3 must not touch is_kernel_0"
    );
    assert_eq!(
        r.ktstr_fentry_is_kernel_1, 0,
        "slot>3 must not touch is_kernel_1"
    );
    assert_eq!(
        r.ktstr_fentry_is_kernel_2, 0,
        "slot>3 must not touch is_kernel_2"
    );
    assert_eq!(
        r.ktstr_fentry_is_kernel_3, 0,
        "slot>3 must not touch is_kernel_3"
    );
}

#[test]
fn set_rodata_slot_writes_only_target_slot() {
    // Positive partner to the no-op test: slot 2 write lands in
    // exactly slot 2's fields (idx and is_kernel) and leaves the
    // other three slots untouched. is_kernel=true must encode as 1
    // (the `as u8` cast), not a generic nonzero.
    let mut r: crate::bpf_skel::fentry::types::rodata = unsafe { std::mem::zeroed() };
    set_rodata_slot(&mut r, 2, 0xABCD, true);
    assert_eq!(
        r.ktstr_fentry_func_idx_2, 0xABCD,
        "slot 2 idx must be written"
    );
    assert_eq!(
        r.ktstr_fentry_is_kernel_2, 1u8,
        "is_kernel=true encodes as 1u8"
    );
    assert_eq!(r.ktstr_fentry_func_idx_0, 0, "slot 0 untouched");
    assert_eq!(r.ktstr_fentry_func_idx_1, 0, "slot 1 untouched");
    assert_eq!(r.ktstr_fentry_func_idx_3, 0, "slot 3 untouched");
    assert_eq!(r.ktstr_fentry_is_kernel_0, 0, "is_kernel_0 untouched");
    assert_eq!(r.ktstr_fentry_is_kernel_1, 0, "is_kernel_1 untouched");
    assert_eq!(r.ktstr_fentry_is_kernel_3, 0, "is_kernel_3 untouched");
}

#[test]
fn kernel_fexit_load_backoff_ramps_from_zero_within_a_bounded_budget() {
    // The first attempt must not wait — a quiet host should pay no
    // retry cost — and subsequent attempts ramp linearly off the base.
    assert_eq!(
        kernel_fexit_load_backoff(0),
        std::time::Duration::ZERO,
        "the first attempt must fire immediately"
    );
    assert_eq!(
        kernel_fexit_load_backoff(1),
        KERNEL_FEXIT_LOAD_BACKOFF,
        "attempt 1 waits one base interval"
    );
    assert_eq!(
        kernel_fexit_load_backoff(2),
        2 * KERNEL_FEXIT_LOAD_BACKOFF,
        "attempt 2 waits two base intervals"
    );

    // The whole retry schedule must stay a small fraction of the
    // probe's boot budget so it never pushes a slow guest past its
    // watchdog. Sum the waits skipped before every attempt.
    let total: std::time::Duration = (0..KERNEL_FEXIT_LOAD_ATTEMPTS)
        .map(kernel_fexit_load_backoff)
        .sum();
    assert!(
        total <= std::time::Duration::from_secs(1),
        "cumulative backoff {total:?} must stay well inside the boot budget"
    );
}
