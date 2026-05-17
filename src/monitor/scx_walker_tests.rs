//! Unit tests for [`super`] (the `scx_walker` module).
//! Co-located via the `tests` submodule pattern.

#![cfg(test)]

    use super::*;

    /// Pin RqScxState wire shape — every optional field skips
    /// on None, required fields land directly. Same coverage style
    /// as task_enrichment::tests::task_enrichment_serde_skip_none_fields.
    #[test]
    fn rq_scx_state_serde_skip_none() {
        let s = RqScxState {
            cpu: 3,
            nr_running: 4,
            flags: 0x10,
            cpu_released: false,
            ops_qseq: 100,
            kick_sync: Some(50),
            nr_immed: None,
            rq_clock: Some(1234567),
            curr_pid: None,
            curr_comm: None,
            runnable_task_kvas: vec![],
            runnable_truncated: false,
        };
        let json = serde_json::to_string(&s).unwrap();
        assert!(!json.contains("curr_pid"));
        assert!(!json.contains("curr_comm"));
        assert!(!json.contains("runnable_truncated"));
        // nr_immed is None — must skip via skip_serializing_if so
        // dumps from v6.14/v7.0 (no kick_sync / nr_immed fields)
        // stay tight.
        assert!(!json.contains("nr_immed"));
        // kick_sync is Some — must serialize the inner value, not
        // a `{ "Some": ... }` shape (Option<T> with default serde
        // serializes the wrapped value bare).
        assert!(json.contains("\"kick_sync\":50"));
        assert!(json.contains("\"cpu\":3"));
        assert!(json.contains("\"nr_running\":4"));
    }

    /// Roundtrip every populated field — Verdict-routed so a single
    /// regression in any field surfaces with its own labeled detail
    /// rather than a cliff-edge `assert_eq!` panic that hides the
    /// other field outcomes. Better signal when multiple serde
    /// renames land at once.
    #[test]
    fn rq_scx_state_serde_roundtrip_populated() {
        use crate::assert::Verdict;

        let s = RqScxState {
            cpu: 1,
            nr_running: 2,
            flags: 0x1,
            cpu_released: true,
            ops_qseq: 42,
            kick_sync: Some(17),
            nr_immed: Some(1),
            rq_clock: Some(999_999),
            curr_pid: Some(1234),
            curr_comm: Some("ktstr".into()),
            runnable_task_kvas: vec![0xffff_ffff_8000_1000, 0xffff_ffff_8000_2000],
            runnable_truncated: true,
        };
        let json = serde_json::to_string(&s).unwrap();
        let parsed: RqScxState = serde_json::from_str(&json).unwrap();

        let parsed_cpu = parsed.cpu;
        let parsed_nr_running = parsed.nr_running;
        let parsed_flags = parsed.flags;
        let parsed_cpu_released = parsed.cpu_released;
        let parsed_ops_qseq = parsed.ops_qseq;
        let parsed_kick_sync = parsed.kick_sync;
        let parsed_nr_immed = parsed.nr_immed;
        let parsed_rq_clock = parsed.rq_clock;
        let parsed_curr_pid = parsed.curr_pid;
        let parsed_curr_comm = parsed.curr_comm.clone();
        let parsed_runnable_kvas_len = parsed.runnable_task_kvas.len();
        let parsed_runnable_truncated = parsed.runnable_truncated;

        let mut v = Verdict::new();
        crate::claim!(v, parsed_cpu).eq(1u32);
        crate::claim!(v, parsed_nr_running).eq(2u32);
        crate::claim!(v, parsed_flags).eq(0x1u32);
        crate::claim!(v, parsed_cpu_released).eq(true);
        crate::claim!(v, parsed_ops_qseq).eq(42u64);
        // kick_sync / nr_immed are now Option<…>; the populated test
        // fixture sets both Some(…), so equality on the unwrapped
        // shape is the same correctness check the prior assertions
        // gave on the bare types.
        let kick_sync_match = parsed_kick_sync == Some(17u64);
        let nr_immed_match = parsed_nr_immed == Some(1u32);
        let rq_clock_match = parsed_rq_clock == Some(999_999u64);
        crate::claim!(v, kick_sync_match).eq(true);
        crate::claim!(v, nr_immed_match).eq(true);
        crate::claim!(v, rq_clock_match).eq(true);
        // Option<T> doesn't impl Display, so claim on the unwrapped
        // values via match-against-known-shape: bake the expected
        // outcome ("present + value matches") into a single bool.
        let curr_pid_match = parsed_curr_pid == Some(1234);
        let curr_comm_match = parsed_curr_comm.as_deref() == Some("ktstr");
        crate::claim!(v, curr_pid_match).eq(true);
        crate::claim!(v, curr_comm_match).eq(true);
        crate::claim!(v, parsed_runnable_kvas_len).eq(2usize);
        crate::claim!(v, parsed_runnable_truncated).eq(true);
        let r = v.into_result();
        assert!(
            r.passed,
            "rq_scx_state roundtrip claims must all pass: {:?}",
            r.details,
        );
    }

    #[test]
    fn dsq_state_serde_skip_truncated_when_false() {
        let d = DsqState {
            id: 0xdead_beef,
            origin: "user".into(),
            nr: 5,
            seq: 100,
            task_kvas: vec![],
            truncated: false,
        };
        let json = serde_json::to_string(&d).unwrap();
        assert!(!json.contains("truncated"));
        assert!(json.contains("\"id\":3735928559"));
        assert!(json.contains("\"nr\":5"));
        assert!(json.contains("\"seq\":100"));
    }

    #[test]
    fn dsq_state_serde_emits_truncated_when_true() {
        let d = DsqState {
            id: 1,
            origin: "global node 0".into(),
            nr: 5000,
            seq: 5001,
            task_kvas: (0..MAX_NODES_PER_LIST as u64).collect(),
            truncated: true,
        };
        let json = serde_json::to_string(&d).unwrap();
        assert!(json.contains("\"truncated\":true"));
    }

    #[test]
    fn scx_sched_state_default_empty() {
        let s = ScxSchedState::default();
        assert!(!s.aborting);
        assert_eq!(s.bypass_depth, 0);
        assert_eq!(s.exit_kind, 0);
    }

    /// Roundtrip every scalar field — Verdict-routed so a serde
    /// rename on one field doesn't mask the other two.
    #[test]
    fn scx_sched_state_serde_roundtrip() {
        use crate::assert::Verdict;

        let s = ScxSchedState {
            aborting: true,
            bypass_depth: 2,
            // SCX_EXIT_ERROR_BPF per include/linux/sched/ext.h
            exit_kind: 1027,
            ..Default::default()
        };
        let json = serde_json::to_string(&s).unwrap();
        let parsed: ScxSchedState = serde_json::from_str(&json).unwrap();

        let parsed_aborting = parsed.aborting;
        let parsed_bypass_depth = parsed.bypass_depth;
        let parsed_exit_kind = parsed.exit_kind;

        let mut v = Verdict::new();
        crate::claim!(v, parsed_aborting).eq(true);
        crate::claim!(v, parsed_bypass_depth).eq(2i32);
        crate::claim!(v, parsed_exit_kind).eq(1027u32);
        let r = v.into_result();
        assert!(
            r.passed,
            "scx_sched_state roundtrip claims must all pass: {:?}",
            r.details,
        );
    }

    /// Walk a hand-built list with two task entries — verifies
    /// the container_of subtraction and the cycle-back termination.
    #[test]
    fn walk_list_head_basic_two_tasks() {
        // Layout (PA == KVA in this test for simplicity):
        //   PA 0x100: head (next at 0, prev at 8)
        //   PA 0x200: task1's runnable_node (next at 0, prev at 8)
        //     task1 starts at PA 0x200 - runnable_node_off_in_task
        //   PA 0x300: task2's runnable_node
        //     task2 starts at PA 0x300 - runnable_node_off_in_task
        //
        // head.next = 0x200, task1.next = 0x300, task2.next = 0x100 (back to head)
        let mut buf = vec![0u8; 0x1000];
        let head = 0x100usize;
        let n1 = 0x200usize;
        let n2 = 0x300usize;
        // head.next = n1
        buf[head..head + 8].copy_from_slice(&(n1 as u64).to_le_bytes());
        // n1.next = n2
        buf[n1..n1 + 8].copy_from_slice(&(n2 as u64).to_le_bytes());
        // n2.next = head (terminator)
        buf[n2..n2 + 8].copy_from_slice(&(head as u64).to_le_bytes());

        let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };

        // Identity translation: PA == KVA for this minimal setup,
        // page_offset = 0 so kva_to_pa is identity.
        let runnable_node_off = 0x10usize;
        let (kvas, truncated) = walk_list_head_for_task_kvas(
            &mem,
            WalkContext::default(),
            head as u64,
            head as u64,
            runnable_node_off,
        );
        assert!(!truncated);
        assert_eq!(kvas.len(), 2);
        assert_eq!(kvas[0], (n1 - runnable_node_off) as u64);
        assert_eq!(kvas[1], (n2 - runnable_node_off) as u64);
    }

    /// Empty list: head.next == &head. Walker returns no kvas.
    #[test]
    fn walk_list_head_empty() {
        let mut buf = vec![0u8; 0x1000];
        let head = 0x100usize;
        // head.next = head
        buf[head..head + 8].copy_from_slice(&(head as u64).to_le_bytes());

        let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };

        let (kvas, truncated) = walk_list_head_for_task_kvas(
            &mem,
            WalkContext::default(),
            head as u64,
            head as u64,
            0x10,
        );
        assert!(!truncated);
        assert!(kvas.is_empty());
    }

    /// Zero next pointer: walker bails defensively without
    /// truncation flag (different from cycle-cap).
    #[test]
    fn walk_list_head_zero_next_bails() {
        let mut buf = vec![0u8; 0x1000];
        let head = 0x100usize;
        // head.next = 0 (uninitialized / unmapped)
        buf[head..head + 8].copy_from_slice(&0u64.to_le_bytes());

        let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };
        let (kvas, truncated) = walk_list_head_for_task_kvas(
            &mem,
            WalkContext::default(),
            head as u64,
            head as u64,
            0x10,
        );
        assert!(!truncated);
        assert!(kvas.is_empty());
    }

    /// `missing_groups()` reports every absent sub-group when offsets
    /// are constructed empty (every Option None). This is the
    /// degenerate input that surfaces every diagnostic name.
    #[test]
    fn scx_walker_offsets_missing_groups_reports_all_when_empty() {
        let offsets = ScxWalkerOffsets {
            rq: None,
            scx_rq: None,
            task: None,
            see: None,
            dsq_lnode: None,
            dsq: None,
            sched: None,
            sched_pnode: None,
            sched_pcpu: None,
            rht: None,
        };
        let missing = offsets.missing_groups();
        // 10 sub-groups, all missing.
        assert_eq!(missing.len(), 10);
        assert!(missing.contains(&"rq"));
        assert!(missing.contains(&"scx_rq"));
        assert!(missing.contains(&"task_struct"));
        assert!(missing.contains(&"sched_ext_entity"));
        assert!(missing.contains(&"scx_dsq_list_node"));
        assert!(missing.contains(&"scx_dispatch_q"));
        assert!(missing.contains(&"scx_sched"));
        assert!(missing.contains(&"scx_sched_pnode"));
        assert!(missing.contains(&"scx_sched_pcpu"));
        assert!(missing.contains(&"rhashtable/bucket_table/rhash_head"));
    }

    /// `missing_groups()` reports nothing when every sub-group is
    /// resolved — a normal, well-formed BTF parse outcome.
    #[test]
    fn scx_walker_offsets_missing_groups_reports_none_when_full() {
        use super::super::btf_offsets::{
            RhashtableOffsets, RqStructOffsets, SchedExtEntityOffsets, ScxDispatchQOffsets,
            ScxDsqListNodeOffsets, ScxRqOffsets, ScxSchedOffsets, ScxSchedPcpuOffsets,
            ScxSchedPnodeOffsets, TaskStructCoreOffsets,
        };
        let offsets = ScxWalkerOffsets {
            rq: Some(RqStructOffsets { scx: 0, curr: 8 }),
            scx_rq: Some(ScxRqOffsets {
                local_dsq: 0,
                runnable_list: 64,
                nr_running: 96,
                flags: 100,
                cpu_released: 104,
                ops_qseq: 112,
                kick_sync: Some(120),
                nr_immed: Some(128),
                clock: Some(136),
            }),
            task: Some(TaskStructCoreOffsets {
                comm: 100,
                pid: 200,
                scx: 300,
            }),
            see: Some(SchedExtEntityOffsets {
                runnable_node: 0,
                runnable_at: 16,
                weight: 24,
                slice: 32,
                dsq_vtime: 40,
                dsq: 48,
                dsq_list: 56,
                flags: 72,
                dsq_flags: 76,
                sticky_cpu: 80,
                holding_cpu: 84,
                tasks_node: 88,
            }),
            dsq_lnode: Some(ScxDsqListNodeOffsets { node: 0, flags: 16 }),
            dsq: Some(ScxDispatchQOffsets {
                list: 0,
                nr: 16,
                seq: 20,
                id: 24,
                hash_node: 32,
            }),
            sched: Some(ScxSchedOffsets {
                dsq_hash: 0,
                pnode: Some(64),
                pcpu: Some(72),
                aborting: Some(80),
                bypass_depth: Some(84),
                exit_kind: 88,
            }),
            sched_pnode: Some(ScxSchedPnodeOffsets {
                global_dsq: Some(0),
            }),
            sched_pcpu: Some(ScxSchedPcpuOffsets {
                bypass_dsq: Some(0),
            }),
            rht: Some(RhashtableOffsets {
                tbl: 0,
                nelems: 8,
                bucket_table_size: 0,
                bucket_table_buckets: 16,
                rhash_head_next: 0,
            }),
        };
        assert!(offsets.missing_groups().is_empty());
    }

    // -- Verdict API integration coverage -------------------------------
    //
    // The walker emits RqScxState / DsqState / ScxSchedState rows that
    // scenario authors will claim against via the new pointwise-claim
    // API. These tests pin the integration shape: walker output flows
    // into Verdict claims via the claim! macro, scalar fields claim
    // through ClaimBuilder, runnable_task_kvas / task_kvas claim
    // through SeqClaim. A regression that breaks the Display impls
    // those claim messages depend on, or that drops the field types
    // claim-able comparators expect, surfaces here.

    /// Author-style claim sequence over a populated RqScxState. The
    /// claims reflect what a scheduler test would actually write —
    /// nr_running ≥ 0, no truncation under healthy load, runnable
    /// task KVA list non-empty when the CPU has running work.
    /// The Verdict accumulates without relying on the legacy Expect
    /// shape; final pass/fail honors every claim.
    #[test]
    fn rq_scx_state_authorial_verdict_claims_compose() {
        use crate::assert::Verdict;

        let s = RqScxState {
            cpu: 2,
            nr_running: 3,
            flags: 0x1,
            cpu_released: false,
            ops_qseq: 4242,
            kick_sync: Some(100),
            nr_immed: Some(0),
            rq_clock: Some(999_999),
            curr_pid: Some(1234),
            curr_comm: Some("ktstr-w".into()),
            runnable_task_kvas: vec![0xffff_ffff_8000_1000, 0xffff_ffff_8000_2000],
            runnable_truncated: false,
        };

        let mut v = Verdict::new();
        // Scalar claims via the claim! macro (label = stringify of expr).
        crate::claim!(v, s.nr_running).at_least(1);
        crate::claim!(v, s.nr_running).at_most(64);
        crate::claim!(v, s.runnable_truncated).eq(false);
        // Sequence claim via claim_seq.
        v.claim_seq("runnable_task_kvas", &s.runnable_task_kvas)
            .nonempty();
        v.claim_seq("runnable_task_kvas", &s.runnable_task_kvas)
            .len_at_most(64);

        let r = v.into_result();
        assert!(
            r.passed,
            "authorial claim sequence on populated RqScxState must pass: {:?}",
            r.details,
        );
    }

    /// Failing claim path: a verdict that calls at_most on
    /// nr_running with a value BELOW the actual count must record
    /// a single kind=Other detail with the field-name label and the
    /// at-most message. Pins the integration of the walker's u32
    /// field type through ClaimBuilder<u32>::at_most's failure
    /// formatter.
    #[test]
    fn rq_scx_state_failing_at_most_records_labeled_detail() {
        use crate::assert::Verdict;

        let s = RqScxState {
            cpu: 0,
            nr_running: 100,
            flags: 0,
            cpu_released: false,
            ops_qseq: 0,
            kick_sync: None,
            nr_immed: None,
            rq_clock: None,
            curr_pid: None,
            curr_comm: None,
            runnable_task_kvas: vec![],
            runnable_truncated: false,
        };

        let mut v = Verdict::new();
        crate::claim!(v, s.nr_running).at_most(10);
        let r = v.into_result();

        assert!(!r.passed, "at_most(10) on nr_running=100 must fail");
        assert_eq!(
            r.details.len(),
            1,
            "exactly one failing detail must record: {:?}",
            r.details,
        );
        let msg = &r.details[0].message;
        assert!(
            msg.contains("s.nr_running"),
            "detail must carry the macro-stringify label: {msg}",
        );
        assert!(
            msg.contains("at most 10"),
            "detail must name the at_most threshold: {msg}",
        );
        assert!(
            msg.contains("100"),
            "detail must include the observed value: {msg}",
        );
    }

    /// DsqState.task_kvas + DsqState.truncated claims compose like
    /// RqScxState's. Pins the walker-DSQ-output shape through the
    /// Verdict surface so a scenario test can write
    /// `claim!(v, dsq.nr).at_most(LIMIT)` and
    /// `v.claim_seq("dsq.task_kvas", &dsq.task_kvas).len_at_most(LIMIT)`
    /// against a real DSQ snapshot.
    #[test]
    fn dsq_state_authorial_verdict_claims_compose() {
        use crate::assert::Verdict;

        let d = DsqState {
            id: 0xdead_beef,
            origin: "user".into(),
            nr: 5,
            seq: 100,
            task_kvas: vec![0xffff_8000_8000_1000; 5],
            truncated: false,
        };

        let mut v = Verdict::new();
        crate::claim!(v, d.nr).at_most(MAX_NODES_PER_LIST);
        crate::claim!(v, d.truncated).eq(false);
        crate::claim!(v, d.seq).at_least(d.nr);
        v.claim_seq("d.task_kvas", &d.task_kvas).len_eq(5);

        let r = v.into_result();
        assert!(
            r.passed,
            "authorial claim sequence on populated DsqState must pass: {:?}",
            r.details,
        );
    }

    /// `ScxSchedState.exit_kind == 0` is the no-error sentinel
    /// (per `enum scx_exit_kind`). Pin via Verdict + claim!(eq) so
    /// scheduler tests can write
    /// `claim!(v, sched.exit_kind).eq(0)` for the healthy-exit
    /// invariant.
    #[test]
    fn scx_sched_state_healthy_exit_kind_claim() {
        use crate::assert::Verdict;

        let healthy = ScxSchedState {
            aborting: false,
            bypass_depth: 0,
            exit_kind: 0,
            ..Default::default()
        };
        let mut v = Verdict::new();
        crate::claim!(v, healthy.aborting).eq(false);
        crate::claim!(v, healthy.bypass_depth).eq(0);
        crate::claim!(v, healthy.exit_kind).eq(0u32);
        let r = v.into_result();
        assert!(r.passed, "healthy-state claims must pass: {:?}", r.details);

        // Inverse: an aborting scheduler with non-zero exit_kind
        // must fail the same claim sequence.
        let aborted = ScxSchedState {
            aborting: true,
            bypass_depth: 4,
            // SCX_EXIT_ERROR_BPF (1027) per include/linux/sched/ext.h.
            exit_kind: 1027,
            ..Default::default()
        };
        let mut v = Verdict::new();
        crate::claim!(v, aborted.exit_kind).eq(0u32);
        let r = v.into_result();
        assert!(!r.passed, "exit_kind=1027 must fail eq(0)");
    }

    /// `walk_scx_tasks_global` returns an empty vec when the
    /// `scx_tasks` symbol KVA is 0 — kernel without sched_ext or
    /// stripped vmlinux. The walk must NOT attempt to read at PA 0
    /// (which would alias the boot-page region and surface bogus
    /// task entries).
    #[test]
    fn walk_scx_tasks_global_zero_kva_returns_empty() {
        let mut buf = vec![0u8; 0x1000];
        // Pre-populate buf at offset 0 to make the difference visible:
        // a buggy implementation that read from PA 0 would surface
        // 0xdead_beef as a task_kva (after container_of subtraction).
        buf[0..8].copy_from_slice(&0xdead_beef_u64.to_le_bytes());
        let mem = unsafe { GuestMem::new(buf.as_mut_ptr(), buf.len() as u64) };
        let kernel = crate::monitor::guest::GuestKernel::new_for_test(
            std::sync::Arc::new(mem),
            std::collections::HashMap::new(),
            0,
            0,
            false,
        );

        let kvas = walk_scx_tasks_global(&kernel, 0, 0x10, 0x60, 0x44);
        assert!(
            kvas.is_empty(),
            "scx_tasks_kva=0 must short-circuit before any read"
        );
    }

    /// `walk_scx_tasks_global` walks an empty global list (head.next
    /// points back at the head itself — kernel's empty-list
    /// invariant). Walker returns no task KVAs.
    #[test]
    fn walk_scx_tasks_global_empty_list_returns_empty() {
        // page_offset = 0 makes the GuestKernel's text_kva_to_pa
        // return KVA itself for KVAs >= __START_KERNEL_map. The KVA
        // we choose is in the text mapping range so the translation
        // lands at a sensible offset within our test buffer.
        let head_kva = crate::monitor::symbols::START_KERNEL_MAP + 0x100;
        let head_pa = head_kva.wrapping_sub(crate::monitor::symbols::START_KERNEL_MAP) as usize;
        let mut buf = vec![0u8; 0x1000];
        // head.next = head_kva (empty list invariant)
        buf[head_pa..head_pa + 8].copy_from_slice(&head_kva.to_le_bytes());
        let mem = unsafe { GuestMem::new(buf.as_mut_ptr(), buf.len() as u64) };
        let kernel = crate::monitor::guest::GuestKernel::new_for_test(
            std::sync::Arc::new(mem),
            std::collections::HashMap::new(),
            0, // page_offset = 0; kva_to_pa identity
            0,
            false,
        );

        let kvas = walk_scx_tasks_global(&kernel, head_kva, 0x10, 0x60, 0x44);
        assert!(kvas.is_empty(), "empty global list must yield no tasks");
    }

    /// `walk_scx_tasks_global` recovers task KVAs via
    /// `task_kva = node_kva - tasks_node_off_in_task`. Two-task list
    /// with the head in the kernel text mapping; the per-task
    /// see/tasks_node lives in a directly-mapped region. Verifies
    /// the container_of math against the kernel's container_of
    /// pattern.
    #[test]
    fn walk_scx_tasks_global_two_tasks_round_trip() {
        // Layout (page_offset = 0 so direct-map kva == pa for the
        // task entries; head lives in the text mapping region so
        // text_kva_to_pa_with_base reaches the buffer):
        //   head_kva = START_KERNEL_MAP + 0x100   → head_pa = 0x100
        //   t1_node_kva = 0x800                   → t1_pa = 0x800
        //   t2_node_kva = 0x900                   → t2_pa = 0x900
        // tasks_node_off_in_task = 0x40 (so task_kva = node_kva - 0x40).
        // Linkage:
        //   head.next = t1_node_kva
        //   t1.next   = t2_node_kva
        //   t2.next   = head_kva (close the list)
        let head_kva = crate::monitor::symbols::START_KERNEL_MAP + 0x100;
        let head_pa = 0x100usize;
        let t1_node_kva: u64 = 0x800;
        let t2_node_kva: u64 = 0x900;
        let tasks_node_off_in_task: usize = 0x40;
        let tasks_node_off_in_see: usize = 0x60;
        let flags_off_in_see: usize = 0x44;

        let mut buf = vec![0u8; 0x1000];
        buf[head_pa..head_pa + 8].copy_from_slice(&t1_node_kva.to_le_bytes());
        let t1_pa = t1_node_kva as usize;
        let t2_pa = t2_node_kva as usize;
        buf[t1_pa..t1_pa + 8].copy_from_slice(&t2_node_kva.to_le_bytes());
        buf[t2_pa..t2_pa + 8].copy_from_slice(&head_kva.to_le_bytes());

        // Both task entries are NOT cursors. Their see.flags slot
        // stays zero (the buf is zero-initialized) so the walker's
        // cursor-flag check passes through. The flags slot for each
        // entry sits at `see_kva + flags_off_in_see` =
        // `(node_kva - tasks_node_off_in_see) + flags_off_in_see`.
        // For t1: see_kva = 0x800 - 0x60 = 0x7a0 → flags @ 0x7a0+0x44=0x7e4.
        // For t2: see_kva = 0x900 - 0x60 = 0x8a0 → flags @ 0x8a0+0x44=0x8e4.
        // Both already 0 from buf init, so the cursor bit is unset.

        let mem = unsafe { GuestMem::new(buf.as_mut_ptr(), buf.len() as u64) };
        let kernel = crate::monitor::guest::GuestKernel::new_for_test(
            std::sync::Arc::new(mem),
            std::collections::HashMap::new(),
            0,
            0,
            false,
        );

        let kvas = walk_scx_tasks_global(
            &kernel,
            head_kva,
            tasks_node_off_in_task,
            tasks_node_off_in_see,
            flags_off_in_see,
        );
        assert_eq!(kvas.len(), 2, "two-task list must yield two task kvas");
        // container_of: task_kva = node_kva - tasks_node_off_in_task.
        assert_eq!(
            kvas[0],
            t1_node_kva.wrapping_sub(tasks_node_off_in_task as u64)
        );
        assert_eq!(
            kvas[1],
            t2_node_kva.wrapping_sub(tasks_node_off_in_task as u64)
        );
    }

    /// `walk_local_dsqs` returns `None` when any required offset
    /// sub-group is missing — the gate must NOT fabricate partial
    /// state when offsets are incomplete.
    #[test]
    fn walk_local_dsqs_none_when_offsets_missing() {
        let mut buf = vec![0u8; 0x1000];
        let mem = unsafe { GuestMem::new(buf.as_mut_ptr(), buf.len() as u64) };
        let kernel = crate::monitor::guest::GuestKernel::new_for_test(
            std::sync::Arc::new(mem),
            std::collections::HashMap::new(),
            0,
            0,
            false,
        );

        let offsets = ScxWalkerOffsets {
            rq: None, // missing → walk_local_dsqs gates to None
            scx_rq: None,
            task: None,
            see: None,
            dsq_lnode: None,
            dsq: None,
            sched: None,
            sched_pnode: None,
            sched_pcpu: None,
            rht: None,
        };

        let r = walk_local_dsqs(&kernel, &[], &[], &[], &offsets);
        assert!(r.is_none(), "missing offsets must gate to None");
    }

    /// `walk_local_dsqs` runs unconditionally — even when
    /// `*scx_root` would be 0 (no scheduler attached). With a
    /// well-formed empty per-CPU local_dsq fixture, the walker
    /// returns `Some(([DsqState{empty list}], []))` for each CPU.
    /// Confirms the new dump-path independence: the local-DSQ
    /// pass surfaces every CPU's DSQ state regardless of
    /// scheduler attachment.
    #[test]
    fn walk_local_dsqs_runs_without_scheduler() {
        // Layout: one CPU. rq fixture lives at PA 0x100 (page_offset=0,
        // identity translation). scx_rq embedded at offset 0; the
        // scx_dispatch_q within scx_rq.local_dsq lives at offset 0
        // of the rq (rq.scx + scx_rq.local_dsq = 0). The DSQ's
        // list_head sits at dsq + dsq.list = 0 + 0 = 0. An empty
        // list means head.next == head_kva.
        let rq_kva: u64 = 0x100;
        let rq_pa: u64 = 0x100;
        let mut buf = vec![0u8; 0x1000];
        // head.next = rq_kva (empty list)
        buf[rq_pa as usize..rq_pa as usize + 8].copy_from_slice(&rq_kva.to_le_bytes());
        let mem = unsafe { GuestMem::new(buf.as_mut_ptr(), buf.len() as u64) };
        let kernel = crate::monitor::guest::GuestKernel::new_for_test(
            std::sync::Arc::new(mem),
            std::collections::HashMap::new(),
            0,
            0,
            false,
        );

        let offsets = ScxWalkerOffsets {
            rq: Some(crate::monitor::btf_offsets::RqStructOffsets { scx: 0, curr: 8 }),
            scx_rq: Some(crate::monitor::btf_offsets::ScxRqOffsets {
                local_dsq: 0,
                runnable_list: 0,
                nr_running: 96,
                flags: 100,
                cpu_released: 104,
                ops_qseq: 112,
                kick_sync: None,
                nr_immed: None,
                clock: None,
            }),
            task: Some(crate::monitor::btf_offsets::TaskStructCoreOffsets {
                comm: 100,
                pid: 200,
                scx: 0,
            }),
            see: Some(crate::monitor::btf_offsets::SchedExtEntityOffsets {
                runnable_node: 0,
                runnable_at: 16,
                weight: 24,
                slice: 32,
                dsq_vtime: 40,
                dsq: 48,
                dsq_list: 56,
                flags: 72,
                dsq_flags: 76,
                sticky_cpu: 80,
                holding_cpu: 84,
                tasks_node: 88,
            }),
            dsq_lnode: Some(crate::monitor::btf_offsets::ScxDsqListNodeOffsets {
                node: 0,
                flags: 16,
            }),
            dsq: Some(crate::monitor::btf_offsets::ScxDispatchQOffsets {
                list: 0,
                nr: 16,
                seq: 20,
                id: 24,
                hash_node: 32,
            }),
            sched: None,
            sched_pnode: None,
            sched_pcpu: None,
            rht: None,
        };

        // Single-CPU per_cpu_offsets: cpu 0 has any offset (BSP can
        // legitimately be 0 — only `cpu_off == 0 && cpu > 0` triggers
        // the BSS-zero-tail skip).
        let (states, entries) = walk_local_dsqs(&kernel, &[rq_kva], &[rq_pa], &[0], &offsets)
            .expect("offsets present, should yield Some");
        assert_eq!(states.len(), 1, "one CPU → one DSQ state");
        assert_eq!(states[0].origin, "local cpu 0");
        // Empty list → no entries.
        assert!(entries.is_empty());
    }

    /// `walk_scx_tasks_global` skips cursor entries — list nodes
    /// whose enclosing `sched_ext_entity.flags` has `SCX_TASK_CURSOR`
    /// (1<<31) set. Inserts a cursor BETWEEN two real task entries
    /// and asserts the cursor's container_of result is NOT in the
    /// returned vec, but both real tasks are.
    #[test]
    fn walk_scx_tasks_global_skips_cursor_entries() {
        let head_kva = crate::monitor::symbols::START_KERNEL_MAP + 0x100;
        let head_pa = 0x100usize;
        let t1_node_kva: u64 = 0x800;
        let cursor_node_kva: u64 = 0xa00;
        let t2_node_kva: u64 = 0xc00;
        let tasks_node_off_in_task: usize = 0x40;
        let tasks_node_off_in_see: usize = 0x60;
        let flags_off_in_see: usize = 0x44;

        let mut buf = vec![0u8; 0x1000];
        buf[head_pa..head_pa + 8].copy_from_slice(&t1_node_kva.to_le_bytes());
        let t1_pa = t1_node_kva as usize;
        let cursor_pa = cursor_node_kva as usize;
        let t2_pa = t2_node_kva as usize;
        buf[t1_pa..t1_pa + 8].copy_from_slice(&cursor_node_kva.to_le_bytes());
        buf[cursor_pa..cursor_pa + 8].copy_from_slice(&t2_node_kva.to_le_bytes());
        buf[t2_pa..t2_pa + 8].copy_from_slice(&head_kva.to_le_bytes());

        // Stamp SCX_TASK_CURSOR (1<<31) into the cursor entry's
        // sched_ext_entity.flags. flags slot lives at
        // (cursor_node_kva - tasks_node_off_in_see) + flags_off_in_see.
        let cursor_see_kva = cursor_node_kva.wrapping_sub(tasks_node_off_in_see as u64);
        let cursor_flags_pa = (cursor_see_kva as usize).wrapping_add(flags_off_in_see);
        let cursor_flags: u32 = 1 << 31;
        buf[cursor_flags_pa..cursor_flags_pa + 4].copy_from_slice(&cursor_flags.to_le_bytes());

        let mem = unsafe { GuestMem::new(buf.as_mut_ptr(), buf.len() as u64) };
        let kernel = crate::monitor::guest::GuestKernel::new_for_test(
            std::sync::Arc::new(mem),
            std::collections::HashMap::new(),
            0,
            0,
            false,
        );

        let kvas = walk_scx_tasks_global(
            &kernel,
            head_kva,
            tasks_node_off_in_task,
            tasks_node_off_in_see,
            flags_off_in_see,
        );
        assert_eq!(
            kvas.len(),
            2,
            "cursor entry must be filtered; only 2 real tasks remain"
        );
        let cursor_task_kva = cursor_node_kva.wrapping_sub(tasks_node_off_in_task as u64);
        assert!(
            !kvas.contains(&cursor_task_kva),
            "cursor's container_of result must NOT appear in the task list"
        );
        assert_eq!(
            kvas[0],
            t1_node_kva.wrapping_sub(tasks_node_off_in_task as u64)
        );
        assert_eq!(
            kvas[1],
            t2_node_kva.wrapping_sub(tasks_node_off_in_task as u64)
        );
    }

    // ---------------------------------------------------------------
    // walk_dsqs partial-pass + read_scx_sched_state degradation tests
    //
    // The fix for dsq=0 / sched=absent requires that every walker
    // produces what data IT can, even when sibling walkers can't run.
    // Pre-fix, a single missing offset blinded the whole DSQ surface;
    // the contract these tests pin is "each pass independent — missing
    // offsets for one pass blind only that pass."
    // ---------------------------------------------------------------

    /// Build a fully-populated `ScxWalkerOffsets` for DSQ walker
    /// fixtures. All leaf groups present so walk_dsqs's outer
    /// short-circuit doesn't fire.
    fn dsq_test_offsets() -> ScxWalkerOffsets {
        use super::super::btf_offsets::{
            RhashtableOffsets, RqStructOffsets, SchedExtEntityOffsets, ScxDispatchQOffsets,
            ScxDsqListNodeOffsets, ScxRqOffsets, ScxSchedOffsets, ScxSchedPcpuOffsets,
            ScxSchedPnodeOffsets, TaskStructCoreOffsets,
        };
        ScxWalkerOffsets {
            rq: Some(RqStructOffsets { scx: 0, curr: 8 }),
            scx_rq: Some(ScxRqOffsets {
                local_dsq: 0,
                runnable_list: 0,
                nr_running: 96,
                flags: 100,
                cpu_released: 104,
                ops_qseq: 112,
                kick_sync: None,
                nr_immed: None,
                clock: None,
            }),
            task: Some(TaskStructCoreOffsets {
                comm: 100,
                pid: 200,
                scx: 0,
            }),
            see: Some(SchedExtEntityOffsets {
                runnable_node: 0,
                runnable_at: 16,
                weight: 24,
                slice: 32,
                dsq_vtime: 40,
                dsq: 48,
                dsq_list: 56,
                flags: 72,
                dsq_flags: 76,
                sticky_cpu: 80,
                holding_cpu: 84,
                tasks_node: 88,
            }),
            dsq_lnode: Some(ScxDsqListNodeOffsets { node: 0, flags: 16 }),
            dsq: Some(ScxDispatchQOffsets {
                list: 0,
                nr: 16,
                seq: 20,
                id: 24,
                hash_node: 32,
            }),
            sched: Some(ScxSchedOffsets {
                dsq_hash: 0x40,
                pnode: Some(0x80),
                pcpu: Some(0x88),
                aborting: Some(0x90),
                bypass_depth: Some(0x94),
                exit_kind: 0x98,
            }),
            sched_pnode: Some(ScxSchedPnodeOffsets {
                global_dsq: Some(0),
            }),
            sched_pcpu: Some(ScxSchedPcpuOffsets {
                bypass_dsq: Some(0),
            }),
            rht: Some(RhashtableOffsets {
                tbl: 0,
                nelems: 8,
                bucket_table_size: 0,
                bucket_table_buckets: 16,
                rhash_head_next: 0,
            }),
        }
    }

    /// REQ 1 / partial passes: leaves all present, sched_pcpu present
    /// (Pass 1 runs), sched_pnode None (Pass 2 skipped), rht None
    /// (Pass 3 skipped). Result must contain Pass 1's bypass DSQ
    /// entries — pinning the "each pass independent" contract.
    #[test]
    fn walk_dsqs_partial_passes_yield_partial_results() {
        // Layout (page_offset = 0; kva_to_pa identity):
        //   sched_pa = 0x100
        //   pcpu_kva = 0x300 (placed at sched_pa + sched.pcpu = 0x100 + 0x88 = 0x188)
        //   per_cpu_offsets = [0]
        //   bypass_dsq_kva for cpu 0 = pcpu_kva + 0 + bypass_dsq_off = 0x300 + 0 + 0 = 0x300
        //   bypass DSQ list head at dsq + dsq.list = 0x300 + 0 = 0x300
        //   We write head.next = head_kva to make the list empty so
        //   walk_one_dsq returns Some with task_kvas = [].
        let mut buf = vec![0u8; 0x2000];
        let sched_pa: u64 = 0x100;
        let pcpu_kva: u64 = 0x300;
        // Place pcpu_kva at sched_pa + sched.pcpu (0x88)
        buf[(sched_pa + 0x88) as usize..(sched_pa + 0x88) as usize + 8]
            .copy_from_slice(&pcpu_kva.to_le_bytes());
        // Empty DSQ at pcpu_kva: head.next = pcpu_kva (head_kva)
        buf[pcpu_kva as usize..pcpu_kva as usize + 8].copy_from_slice(&pcpu_kva.to_le_bytes());

        let mem = unsafe { GuestMem::new(buf.as_mut_ptr(), buf.len() as u64) };
        let kernel = super::super::guest::GuestKernel::new_for_test(
            std::sync::Arc::new(mem),
            std::collections::HashMap::new(),
            0,
            0,
            false,
        );

        let mut offsets = dsq_test_offsets();
        // Disable pass 2 + pass 3 by Noneing their offset groups.
        offsets.sched_pnode = None;
        offsets.rht = None;

        let (states, entries) = walk_dsqs(&kernel, sched_pa, &[0u64], 0, &offsets);
        assert_eq!(states.len(), 1, "pass 1 produces one bypass DSQ entry");
        assert_eq!(states[0].origin, "bypass cpu 0");
        assert!(entries.is_empty(), "empty bypass DSQ → no task entries");
    }

    /// REQ 4 / 6.12+ compat: leaves present, ALL three "advanced"
    /// offset groups (sched_pcpu, sched_pnode, rht) None. Result is
    /// (vec![], vec![]) — no panic, no garbage reads. This is the
    /// 6.12-kernel reality: scx_sched_pcpu didn't land until v6.18,
    /// rhashtable shape varies across kernels, and sched_pnode is
    /// dev-only. Without sched layer, walker must NOT crash.
    #[test]
    fn walk_dsqs_all_advanced_offsets_none_yields_empty() {
        let mut buf = vec![0u8; 0x1000];
        // Pre-populate buf at sched_pa to ensure a buggy walker that
        // bypassed the offset gates would surface garbage. With
        // every advanced offset None, the walker must NOT read here.
        let sched_pa: u64 = 0x100;
        buf[sched_pa as usize..sched_pa as usize + 8]
            .copy_from_slice(&0xdead_beef_dead_beef_u64.to_le_bytes());
        let mem = unsafe { GuestMem::new(buf.as_mut_ptr(), buf.len() as u64) };
        let kernel = super::super::guest::GuestKernel::new_for_test(
            std::sync::Arc::new(mem),
            std::collections::HashMap::new(),
            0,
            0,
            false,
        );

        let mut offsets = dsq_test_offsets();
        offsets.sched_pcpu = None;
        offsets.sched_pnode = None;
        offsets.rht = None;

        let (states, entries) = walk_dsqs(&kernel, sched_pa, &[0u64], 1, &offsets);
        assert!(
            states.is_empty(),
            "all advanced offsets None → no DSQ states"
        );
        assert!(entries.is_empty());
    }

    /// Regression for the PA-as-KVA bug in walk_dsqs Pass 3
    /// (user dsq_hash). With a real (non-zero) page_offset, the
    /// pre-fix code added `sched_offs.dsq_hash` to `sched_pa` and
    /// passed the result to `walk_user_dsq_hash` as a KVA. The
    /// inner translate then ran `kva_to_pa(sched_pa+off, page_offset)
    /// = (sched_pa+off).wrapping_sub(page_offset)`, which underflows
    /// for any sched_pa < page_offset and produces an out-of-range
    /// PA — `translate_any_kva` returns None, the user-DSQ list is
    /// silently empty, and the failure dump loses every user DSQ.
    ///
    /// This test fires Pass 3 with `page_offset = 0xffff_8880_0000_0000`
    /// (the x86_64 4-level direct-map base) and a single user DSQ
    /// reachable through dsq_hash. Pre-fix: the walker returns 0
    /// DSQ states for the user pass. Post-fix: 1 DSQ state with the
    /// expected scalar fields.
    #[test]
    fn walk_dsqs_user_hash_with_real_page_offset() {
        // The fixture is laid out at low PAs (small offsets into the
        // GuestMem buffer) but every KVA derived from those PAs uses
        // page_offset = X86_DIRECT_MAP. The walker must therefore
        // never treat a PA as a KVA (the bug) — every translate must
        // round-trip through `kva_to_pa(kva) = kva - page_offset`.
        const PAGE_OFFSET: u64 = 0xffff_8880_0000_0000;

        // PAs (DRAM offsets). Every "_kva" is the matching PA + PAGE_OFFSET.
        let sched_pa: u64 = 0x100;
        // sched.dsq_hash = 0x40 → rht_pa = 0x140
        let rht_pa: u64 = 0x140;
        let tbl_pa: u64 = 0x300;
        let dsq_pa: u64 = 0x500; // user-allocated scx_dispatch_q

        // KVAs returned by the rhashtable's tbl pointer, the
        // bucket-table's stored hash_head pointer, and the dsq's
        // own KVA must all live above PAGE_OFFSET so the walker's
        // translate_any_kva calls succeed.
        let tbl_kva = tbl_pa.wrapping_add(PAGE_OFFSET);
        let dsq_kva_expected = dsq_pa.wrapping_add(PAGE_OFFSET);

        // hash_node = 0 → container_of yields the dsq KVA unchanged.
        // The test asserts the dsq KVA the walker recovers, not
        // container_of math.
        // Buffer must cover up to dsq_pa + dsq.id_off + 8 = 0x500 + 24 + 8.
        let mut buf = vec![0u8; 0x1000];

        // rht.tbl = tbl_kva (offset 0 inside rhashtable).
        buf[rht_pa as usize..rht_pa as usize + 8].copy_from_slice(&tbl_kva.to_le_bytes());
        // bucket_table.size = 1 (offset 0 in bucket_table).
        buf[tbl_pa as usize..tbl_pa as usize + 4].copy_from_slice(&1u32.to_le_bytes());
        // bucket_table.buckets[0] = dsq_kva (the rhash_head node lives
        // inside scx_dispatch_q at hash_node=0). Buckets array starts
        // at offset 16.
        buf[(tbl_pa + 16) as usize..(tbl_pa + 16) as usize + 8]
            .copy_from_slice(&dsq_kva_expected.to_le_bytes());
        // rhash_head.next = 0 → bucket terminator.
        buf[dsq_pa as usize..dsq_pa as usize + 8].copy_from_slice(&0u64.to_le_bytes());
        // dsq scalars: id=0xc0ffee at offset 24.
        buf[(dsq_pa + 24) as usize..(dsq_pa + 24) as usize + 8]
            .copy_from_slice(&0xc0ffee_u64.to_le_bytes());

        let mem = unsafe { GuestMem::new(buf.as_mut_ptr(), buf.len() as u64) };
        let kernel = super::super::guest::GuestKernel::new_for_test(
            std::sync::Arc::new(mem),
            std::collections::HashMap::new(),
            PAGE_OFFSET,
            0,
            false,
        );

        let mut offsets = dsq_test_offsets();
        // Disable Pass 1 + Pass 2 so the test isolates Pass 3.
        offsets.sched_pcpu = None;
        offsets.sched_pnode = None;
        // Force `hash_node = 0` so container_of from the bucket entry
        // back to scx_dispatch_q is identity. The test pins Pass 3's
        // PA-handling fix, not container_of math.
        if let Some(dsq_offs) = offsets.dsq.as_mut() {
            dsq_offs.hash_node = 0;
        }

        let (states, _entries) = walk_dsqs(&kernel, sched_pa, &[], 0, &offsets);

        assert_eq!(
            states.len(),
            1,
            "Pass 3 must surface the user DSQ when page_offset is non-zero — \
             pre-fix the PA-as-KVA bug silently returned 0 user DSQs",
        );
        assert_eq!(states[0].origin, "user");
        assert_eq!(states[0].id, 0xc0ffee);
    }

    /// REQ 1 / not-all-or-nothing: 2 CPUs, local-DSQ pass produces
    /// one DsqState row per CPU regardless of whether that CPU's
    /// list has tasks. CPU 0 has 1 queued task; CPU 1 is empty. The
    /// result must have 2 DsqState rows — not 0, not 1. This pins
    /// the production guarantee that walk_local_dsqs surfaces every
    /// CPU's DSQ regardless of queue depth.
    #[test]
    fn walk_local_dsqs_one_cpu_empty_one_populated() {
        // Layout (page_offset = 0; identity translation):
        //   CPU 0: rq_kva = rq_pa = 0x100. local_dsq head at
        //          rq + 0 = 0x100. head.next = task1 (0x800).
        //          dsq_lnode at task1 (0x800), dsq_lnode.flags at 0x10
        //          → set to 0 (not cursor).
        //          task1.dsq_lnode.next = head_kva (0x100, terminator).
        //   CPU 1: rq_kva = rq_pa = 0x300. local_dsq head at
        //          rq + 0 = 0x300. head.next = head_kva (empty list).
        //
        // dsq.{nr,seq,id} fields read from rq_pa+{16,20,24}.
        let mut buf = vec![0u8; 0x2000];
        let cpu0_rq: u64 = 0x100;
        let cpu1_rq: u64 = 0x300;
        let task1: u64 = 0x800;

        // CPU 0 list: head.next = task1, task1.next = head_kva.
        buf[cpu0_rq as usize..cpu0_rq as usize + 8].copy_from_slice(&task1.to_le_bytes());
        buf[task1 as usize..task1 as usize + 8].copy_from_slice(&cpu0_rq.to_le_bytes());

        // Stamp DSQ scalars on CPU 0 (id=0xa, nr=1, seq=10).
        buf[(cpu0_rq + 16) as usize..(cpu0_rq + 16) as usize + 4]
            .copy_from_slice(&1u32.to_le_bytes()); // nr
        buf[(cpu0_rq + 20) as usize..(cpu0_rq + 20) as usize + 4]
            .copy_from_slice(&10u32.to_le_bytes()); // seq
        buf[(cpu0_rq + 24) as usize..(cpu0_rq + 24) as usize + 8]
            .copy_from_slice(&0xau64.to_le_bytes()); // id

        // CPU 1 list: head.next = head_kva (empty list).
        buf[cpu1_rq as usize..cpu1_rq as usize + 8].copy_from_slice(&cpu1_rq.to_le_bytes());

        let mem = unsafe { GuestMem::new(buf.as_mut_ptr(), buf.len() as u64) };
        let kernel = super::super::guest::GuestKernel::new_for_test(
            std::sync::Arc::new(mem),
            std::collections::HashMap::new(),
            0,
            0,
            false,
        );

        let offsets = dsq_test_offsets();
        // Both CPUs onlined: per_cpu_offsets non-zero for cpu>0
        // (otherwise the BSS-zero-tail guard would skip cpu 1).
        let (states, entries) = walk_local_dsqs(
            &kernel,
            &[cpu0_rq, cpu1_rq],
            &[cpu0_rq, cpu1_rq],
            &[0, 0x1000],
            &offsets,
        )
        .expect("offsets present, should yield Some");

        assert_eq!(
            states.len(),
            2,
            "two CPUs → two DSQ rows, regardless of queue depth"
        );
        let cpu0 = states.iter().find(|s| s.origin == "local cpu 0").unwrap();
        let cpu1 = states.iter().find(|s| s.origin == "local cpu 1").unwrap();
        assert_eq!(cpu0.task_kvas.len(), 1, "CPU 0 has one queued task");
        assert!(cpu1.task_kvas.is_empty(), "CPU 1 is empty");
        assert_eq!(cpu0.id, 0xa);
        assert_eq!(cpu0.nr, 1);
        assert_eq!(cpu0.seq, 10);
        // entries vec contains exactly the CPU 0 task.
        assert_eq!(entries.len(), 1);
    }

    /// BSS-zero-tail guard: `__per_cpu_offset[]` is BSS-zero for
    /// CPU slots beyond `nr_cpu_ids` because `setup_per_cpu_areas`
    /// only writes the slots in `for_each_possible_cpu`. The walker
    /// must check `per_cpu_offsets[cpu] == 0 && cpu > 0` to skip
    /// those slots; otherwise it surfaces phantom DSQ rows for
    /// un-onlined CPUs. The phantom rows would land at the bare
    /// `runqueues` symbol KVA (rq_kva = runqueues + 0), aliasing
    /// neither CPU 0's KVA (runqueues + delta on x86_64 SMP) nor
    /// each other in any well-formed way — a resolved-rq_kva
    /// comparison cannot detect the alias on x86_64 SMP because
    /// `__per_cpu_offset[0]` is non-zero (`delta = pcpu_base_addr -
    /// __per_cpu_start`, see `arch/x86/kernel/setup_percpu.c`).
    #[test]
    fn walk_local_dsqs_skips_bss_zero_tail_aliases() {
        // Layout (page_offset = 0; identity translation):
        //   CPU 0: per_cpu_offset = 0x100; rq_kva = rq_pa = 0x100.
        //          Empty list head at PA 0x100.
        //   CPU 1, 2, 3: per_cpu_offset = 0 (BSS-zero tail).
        //
        // CPU 0's rq_kva is NOT shared by the BSS-zero entries
        // (their rq_kva would be `runqueues + 0` = 0, not 0x100),
        // so the old `rq_kva == rq_kvas[0]` guard would not catch
        // them — on x86_64 SMP, the legitimate CPU 0 entry is the
        // ONLY one with rq_kva == runqueues + per_cpu_offset[0].
        // A correct walker emits one DsqState (CPU 0) using the
        // `cpu_off == 0 && cpu > 0` check; a regressed walker
        // emits four (or three, if the old guard caught some
        // accidental alias). Pinning len() == 1 makes the
        // regression visible.
        let mut buf = vec![0u8; 0x1000];
        let cpu0_rq: u64 = 0x100;
        // CPU 0 list: head.next = head_kva (empty).
        buf[cpu0_rq as usize..cpu0_rq as usize + 8].copy_from_slice(&cpu0_rq.to_le_bytes());

        let mem = unsafe { GuestMem::new(buf.as_mut_ptr(), buf.len() as u64) };
        let kernel = super::super::guest::GuestKernel::new_for_test(
            std::sync::Arc::new(mem),
            std::collections::HashMap::new(),
            0,
            0,
            false,
        );

        let offsets = dsq_test_offsets();
        // For the BSS-zero tail entries, the caller would actually
        // pass an rq_kva of `runqueues + 0`; the test mirrors the
        // production rq_kvas here (every entry equals cpu0_rq) so
        // a regressed `rq_kva == rq_kvas[0]` walker would still
        // catch the alias. The new guard ignores rq_kvas entirely
        // and gates on per_cpu_offsets — so the BSS-zero entries
        // are skipped for that reason instead.
        let (states, entries) = walk_local_dsqs(
            &kernel,
            &[cpu0_rq, cpu0_rq, cpu0_rq, cpu0_rq],
            &[cpu0_rq, cpu0_rq, cpu0_rq, cpu0_rq],
            &[0x100, 0, 0, 0], // CPU 0 onlined; CPUs 1-3 BSS-zero
            &offsets,
        )
        .expect("offsets present, should yield Some");
        assert_eq!(
            states.len(),
            1,
            "BSS-zero-tail aliases must be skipped; only CPU 0 surfaces"
        );
        assert_eq!(states[0].origin, "local cpu 0");
        assert!(entries.is_empty());
    }

    /// Regression: x86_64 SMP layout where `__per_cpu_offset[0]` is
    /// non-zero (the production case — `delta = pcpu_base_addr -
    /// __per_cpu_start` is positive when the percpu allocator places
    /// its base outside the static `.data..percpu` region, see
    /// `setup_per_cpu_areas` in `arch/x86/kernel/setup_percpu.c`).
    /// CPU 0's `rq_kva = runqueues + delta` differs from a BSS-zero
    /// tail entry's `rq_kva = runqueues + 0`, so the prior
    /// `rq_kva == rq_kvas[0]` guard would NOT catch the alias and
    /// would surface a phantom DSQ row. The new
    /// `per_cpu_offsets[cpu] == 0 && cpu > 0` guard catches it
    /// regardless of how the resolved KVAs compare.
    #[test]
    fn walk_local_dsqs_skips_bss_zero_tail_with_nonzero_cpu0_offset() {
        // Layout (page_offset = 0; identity translation):
        //   runqueues_pa  = 0x300 (a non-zero "runqueues" KVA so a
        //                          BSS-zero entry's rq_pa is also
        //                          non-zero — `walk_one_dsq` skips
        //                          dsq_pa==0, which would otherwise
        //                          mask a regressed guard).
        //   per_cpu_offset[0] = 0x100; rq_pa[0] = 0x400 (delta).
        //   per_cpu_offset[1] = 0;     rq_pa[1] = 0x300 (BSS-zero).
        // CPU 0's rq_pa (0x400) differs from CPU 1's BSS rq_pa
        // (0x300); the prior `rq_kva == rq_kvas[0]` guard would
        // NOT catch the alias because the two PAs are distinct.
        // The new guard catches it via `cpu_off == 0 && cpu > 0`.
        let runqueues_pa: u64 = 0x300;
        let cpu0_rq: u64 = runqueues_pa + 0x100; // 0x400
        let bss_rq: u64 = runqueues_pa; // 0x300
        let mut buf = vec![0u8; 0x1000];
        // Stamp empty-list heads at BOTH addresses so a regressed
        // walker would surface DsqState rows for both. The post-
        // guard list walk reads head.next at offset 0; pointing
        // it at itself terminates the list immediately.
        buf[cpu0_rq as usize..cpu0_rq as usize + 8].copy_from_slice(&cpu0_rq.to_le_bytes());
        buf[bss_rq as usize..bss_rq as usize + 8].copy_from_slice(&bss_rq.to_le_bytes());

        let mem = unsafe { GuestMem::new(buf.as_mut_ptr(), buf.len() as u64) };
        let kernel = super::super::guest::GuestKernel::new_for_test(
            std::sync::Arc::new(mem),
            std::collections::HashMap::new(),
            0,
            0,
            false,
        );

        let offsets = dsq_test_offsets();
        let (states, _entries) = walk_local_dsqs(
            &kernel,
            &[cpu0_rq, bss_rq],
            &[cpu0_rq, bss_rq],
            &[0x100, 0], // CPU 0 onlined (delta=0x100); CPU 1 BSS-zero
            &offsets,
        )
        .expect("offsets present, should yield Some");
        // The new guard skips CPU 1 because per_cpu_offset[1] == 0.
        // If the walker compared resolved KVAs (`rq_kva == rq_kvas[0]`)
        // instead, it would see cpu0_rq != bss_rq and emit a phantom
        // row for CPU 1. Pinning len() == 1 catches that regression.
        assert_eq!(
            states.len(),
            1,
            "BSS-zero entry must be skipped via cpu_off==0 guard \
             even when its rq_pa differs from rq_pas[0]"
        );
        assert_eq!(states[0].origin, "local cpu 0");
    }

    /// REQ 2: read_scx_sched_state with `offsets.sched = None` —
    /// the walker MUST short-circuit before any guest-memory read.
    /// Pre-populating sched_pa with a value that would surface as a
    /// bogus aborting/bypass_depth ensures the gate fires correctly:
    /// a regression that read despite None offsets would emit a
    /// state with the bogus values; the None-return contract pins
    /// "no fabricated state."
    #[test]
    fn read_scx_sched_state_offsets_sched_none_returns_none() {
        let mut buf = vec![0u8; 0x1000];
        // Pre-populate: a buggy walker reading at PA 0 would surface
        // the magic value as exit_kind / bypass_depth.
        buf[0..8].copy_from_slice(&0xdead_beef_u64.to_le_bytes());
        let mem = unsafe { GuestMem::new(buf.as_mut_ptr(), buf.len() as u64) };
        let kernel = super::super::guest::GuestKernel::new_for_test(
            std::sync::Arc::new(mem),
            std::collections::HashMap::new(),
            0,
            0,
            false,
        );

        let mut offsets = dsq_test_offsets();
        offsets.sched = None;
        let scx_root_kva = super::super::symbols::START_KERNEL_MAP + 0x10;
        let r = read_scx_sched_state(&kernel, scx_root_kva, &offsets);
        assert!(r.is_none(), "sched=None must short-circuit before read");
    }

    /// REQ 2 / *scx_root unset: scx_root_kva resolves but the
    /// pointer it points to reads as 0 (no scheduler attached).
    /// read_scx_sched_state must return None — pinning the
    /// "scheduler not attached" diagnosis without surfacing bogus
    /// state.
    #[test]
    fn read_scx_sched_state_scx_root_pointer_zero_returns_none() {
        // Layout: scx_root_kva is in the text mapping. We choose
        // START_KERNEL_MAP + 0x100 so kernel.text_kva_to_pa(scx_root_kva)
        // = 0x100. Stamp 0 at that PA. The walker reads sched_kva = 0
        // and returns None.
        let scx_root_kva = super::super::symbols::START_KERNEL_MAP + 0x100;
        let scx_root_pa = 0x100usize;
        let mut buf = vec![0u8; 0x1000];
        buf[scx_root_pa..scx_root_pa + 8].copy_from_slice(&0u64.to_le_bytes());

        let mem = unsafe { GuestMem::new(buf.as_mut_ptr(), buf.len() as u64) };
        let kernel = super::super::guest::GuestKernel::new_for_test(
            std::sync::Arc::new(mem),
            std::collections::HashMap::new(),
            0,
            0,
            false,
        );

        let offsets = dsq_test_offsets();
        let r = read_scx_sched_state(&kernel, scx_root_kva, &offsets);
        assert!(
            r.is_none(),
            "*scx_root == 0 (no scheduler) → None, no state surfaced"
        );
    }

    /// REQ 4 / dev-only field None: sched.aborting offset is
    /// Option<usize>. On release kernels the field is absent. The
    /// walker must NOT read at sched_pa+0 as a fallback (that would
    /// alias dsq_hash). Pinning: with aborting=None, the returned
    /// state has aborting=false.
    #[test]
    fn read_scx_sched_state_aborting_offset_none_defaults_false() {
        // Layout:
        //   scx_root_kva = START_KERNEL_MAP + 0x100
        //   *scx_root → sched_kva (we put it in direct mapping at 0x800)
        //   sched_pa = 0x800 (page_offset = 0; identity)
        //   exit_kind at sched_pa + 0x98 = 0
        //
        // Stamp a magic value at sched_pa + 0 to detect any bogus
        // fallback read for `aborting`. Without aborting=None being
        // honored, a buggy walker reading that location would
        // surface aborting=true.
        let scx_root_kva = super::super::symbols::START_KERNEL_MAP + 0x100;
        let scx_root_pa: usize = 0x100;
        let sched_pa: u64 = 0x800;
        let mut buf = vec![0u8; 0x1000];
        // *scx_root = sched_kva (direct mapping; sched_kva == sched_pa here)
        buf[scx_root_pa..scx_root_pa + 8].copy_from_slice(&sched_pa.to_le_bytes());
        // Stamp 0xff at sched_pa+0 — non-zero, would be true if read as bool.
        buf[sched_pa as usize] = 0xff;

        let mem = unsafe { GuestMem::new(buf.as_mut_ptr(), buf.len() as u64) };
        let kernel = super::super::guest::GuestKernel::new_for_test(
            std::sync::Arc::new(mem),
            std::collections::HashMap::new(),
            0,
            0,
            false,
        );

        let mut offsets = dsq_test_offsets();
        // Mark aborting offset absent — release-kernel reality.
        if let Some(s) = offsets.sched.as_mut() {
            s.aborting = None;
        }

        let (sched_kva_out, state) = read_scx_sched_state(&kernel, scx_root_kva, &offsets)
            .expect("should yield Some when sched offsets present");
        assert_eq!(sched_kva_out, sched_pa);
        assert!(
            !state.aborting,
            "aborting=None must default to false, NOT read sched_pa+0"
        );
    }

    /// REQ 4 / dev-only field None: sched.bypass_depth offset is
    /// Option<usize>. Same shape as aborting — None means the
    /// kernel doesn't have the field; walker must default to 0
    /// without reading.
    #[test]
    fn read_scx_sched_state_bypass_depth_offset_none_defaults_zero() {
        let scx_root_kva = super::super::symbols::START_KERNEL_MAP + 0x100;
        let scx_root_pa: usize = 0x100;
        let sched_pa: u64 = 0x800;
        let mut buf = vec![0u8; 0x1000];
        buf[scx_root_pa..scx_root_pa + 8].copy_from_slice(&sched_pa.to_le_bytes());
        // Stamp a magic at sched_pa+0 (would surface as bypass_depth
        // if a buggy walker read there as fallback).
        buf[sched_pa as usize..sched_pa as usize + 4]
            .copy_from_slice(&0xdead_beef_u32.to_le_bytes());

        let mem = unsafe { GuestMem::new(buf.as_mut_ptr(), buf.len() as u64) };
        let kernel = super::super::guest::GuestKernel::new_for_test(
            std::sync::Arc::new(mem),
            std::collections::HashMap::new(),
            0,
            0,
            false,
        );

        let mut offsets = dsq_test_offsets();
        if let Some(s) = offsets.sched.as_mut() {
            s.bypass_depth = None;
        }

        let (_, state) = read_scx_sched_state(&kernel, scx_root_kva, &offsets)
            .expect("should yield Some when sched offsets present");
        assert_eq!(
            state.bypass_depth, 0,
            "bypass_depth=None must default to 0, NOT read sched_pa+0"
        );
    }

    // ---------------------------------------------------------------
    // walk_user_dsq_hash truncation-cap tests
    //
    // walk_user_dsq_hash bounds the walk on three independent caps
    // (MAX_RHT_BUCKETS, MAX_RHT_NODES, PER_BUCKET_CHAIN_CAP). Each
    // cap must set the returned `truncated` flag so the failure-dump
    // consumer can distinguish "small DSQ count" from "cap silently
    // dropped tail entries."
    // ---------------------------------------------------------------

    /// Build a minimal `RhashtableOffsets` fixture whose every offset
    /// is small and consistent: tbl at 0, bucket_table.size at 0,
    /// bucket_table.buckets at 16, rhash_head.next at 0. Used by the
    /// three truncation tests below.
    fn rht_test_offsets() -> super::super::btf_offsets::RhashtableOffsets {
        super::super::btf_offsets::RhashtableOffsets {
            tbl: 0,
            nelems: 8,
            bucket_table_size: 0,
            bucket_table_buckets: 16,
            rhash_head_next: 0,
        }
    }

    /// Build a minimal `ScxDispatchQOffsets` with `hash_node = 0` so
    /// container_of yields the node KVA unchanged. Tests below assert
    /// truncation flags, not container_of math.
    fn dsq_test_offsets_for_hash() -> super::super::btf_offsets::ScxDispatchQOffsets {
        super::super::btf_offsets::ScxDispatchQOffsets {
            list: 0,
            nr: 16,
            seq: 20,
            id: 24,
            hash_node: 0,
        }
    }

    /// Per-bucket chain cap (`PER_BUCKET_CHAIN_CAP`): a single
    /// bucket's chain that doesn't terminate naturally must set
    /// `truncated = true` after exactly `PER_BUCKET_CHAIN_CAP`
    /// visits. The fixture uses a 2-node cycle so the chain has no
    /// natural terminator; the walker bails on the per-bucket cap
    /// and the post-loop check sets truncated.
    #[test]
    fn walk_user_dsq_hash_per_bucket_chain_cap_truncates() {
        // Layout (page_offset = 0; identity translation):
        //   rht_pa = 0x100  (struct rhashtable; .tbl at offset 0 = 8 bytes)
        //   tbl_pa = 0x200  (bucket_table; size at off 0, buckets at off 16)
        //   node_a = 0x300, node_b = 0x308: cycle (next-pointer at off 0 each)
        //   tbl.size = 1; buckets[0] = node_a (no LSB tag).
        //
        // Walker chases node_a → node_b → node_a → ... until
        // chain_visited reaches PER_BUCKET_CHAIN_CAP. 1024 visits,
        // cap fires, post-loop sets truncated=true.
        let mut buf = vec![0u8; 0x1000];
        let rht_pa: u64 = 0x100;
        let tbl_kva: u64 = 0x200;
        let tbl_pa: u64 = 0x200;
        let node_a: u64 = 0x300;
        let node_b: u64 = 0x308;

        // rht.tbl = tbl_kva
        buf[rht_pa as usize..rht_pa as usize + 8].copy_from_slice(&tbl_kva.to_le_bytes());
        // tbl.size = 1 (one bucket)
        buf[tbl_pa as usize..tbl_pa as usize + 4].copy_from_slice(&1u32.to_le_bytes());
        // buckets[0] = node_a
        buf[(tbl_pa + 16) as usize..(tbl_pa + 16) as usize + 8]
            .copy_from_slice(&node_a.to_le_bytes());
        // node_a.next = node_b (LSB clear → not terminator)
        buf[node_a as usize..node_a as usize + 8].copy_from_slice(&node_b.to_le_bytes());
        // node_b.next = node_a (close the cycle, LSB clear)
        buf[node_b as usize..node_b as usize + 8].copy_from_slice(&node_a.to_le_bytes());

        let mem = unsafe { GuestMem::new(buf.as_mut_ptr(), buf.len() as u64) };
        let rht_offs = rht_test_offsets();
        let dsq_offs = dsq_test_offsets_for_hash();

        let (dsq_kvas, truncated) =
            walk_user_dsq_hash(&mem, WalkContext::default(), rht_pa, &rht_offs, &dsq_offs);

        assert!(
            truncated,
            "per-bucket chain cap must set truncated=true on a non-terminating chain",
        );
        assert_eq!(
            dsq_kvas.len(),
            PER_BUCKET_CHAIN_CAP as usize,
            "PER_BUCKET_CHAIN_CAP must admit exactly {} chain visits",
            PER_BUCKET_CHAIN_CAP,
        );
    }

    /// Global node cap (`MAX_RHT_NODES`): when the cumulative node
    /// count across multiple buckets reaches `MAX_RHT_NODES`, the
    /// walker must set `truncated = true` and stop visiting further
    /// buckets. This test constructs `MAX_RHT_NODES + 1` buckets each
    /// with a single-node chain that terminates naturally; the
    /// per-bucket cap never fires (each chain has 1 entry). Truncation
    /// fires only via the global cap.
    #[test]
    fn walk_user_dsq_hash_global_node_cap_truncates() {
        // Layout (page_offset = 0; identity translation):
        //   rht_pa = 0x100         (.tbl at offset 0)
        //   tbl_pa = 0x1000        (size at 0, buckets at 16)
        //   shared_node = 0x40000  (next field at offset 0 = 0 → terminator)
        //   tbl.size = MAX_RHT_NODES + 1 = 8193 buckets, each pointing
        //   at shared_node.
        //
        // Walker enters each bucket, walks 1 node (push, total++),
        // reads next=0 → break with chain_terminated_naturally=true.
        // After bucket 8191, total=MAX_RHT_NODES (8192). Entering
        // bucket 8192, the pre-loop `total_nodes >= MAX_RHT_NODES`
        // gate returns truncated=true.
        let bucket_count: u32 = MAX_RHT_NODES + 1;
        let rht_pa: u64 = 0x100;
        let tbl_kva: u64 = 0x1000;
        let tbl_pa: u64 = 0x1000;
        let buckets_off: u64 = 16;
        let shared_node: u64 = 0x40000;

        // Buffer must cover: rht (0x100..0x108), tbl (0x1000),
        // buckets array (0x1010..0x1010 + bucket_count*8 = 0x1010 +
        // 0x10008 = 0x11018), and shared_node (0x40000..0x40008).
        let buf_size = (shared_node + 16) as usize;
        let mut buf = vec![0u8; buf_size];

        // rht.tbl = tbl_kva
        buf[rht_pa as usize..rht_pa as usize + 8].copy_from_slice(&tbl_kva.to_le_bytes());
        // tbl.size = bucket_count
        buf[tbl_pa as usize..tbl_pa as usize + 4].copy_from_slice(&bucket_count.to_le_bytes());
        // Stamp every bucket[i] = shared_node
        for i in 0..bucket_count as u64 {
            let off = (tbl_pa + buckets_off + i * 8) as usize;
            buf[off..off + 8].copy_from_slice(&shared_node.to_le_bytes());
        }
        // shared_node.next = 0 (already zero from buffer init — explicit
        // for clarity).
        buf[shared_node as usize..shared_node as usize + 8].copy_from_slice(&0u64.to_le_bytes());

        let mem = unsafe { GuestMem::new(buf.as_mut_ptr(), buf.len() as u64) };
        let rht_offs = rht_test_offsets();
        let dsq_offs = dsq_test_offsets_for_hash();

        let (dsq_kvas, truncated) =
            walk_user_dsq_hash(&mem, WalkContext::default(), rht_pa, &rht_offs, &dsq_offs);

        assert!(
            truncated,
            "global node cap (MAX_RHT_NODES) must set truncated=true",
        );
        // The walker pushes one dsq per bucket up to MAX_RHT_NODES,
        // then short-circuits — never enters bucket MAX_RHT_NODES.
        assert_eq!(
            dsq_kvas.len(),
            MAX_RHT_NODES as usize,
            "global cap halts the walk at exactly {} nodes",
            MAX_RHT_NODES,
        );
    }

    /// Bucket-table cap (`MAX_RHT_BUCKETS`): when
    /// `bucket_table.size > MAX_RHT_BUCKETS`, the walker enumerates
    /// only the first `MAX_RHT_BUCKETS` entries and sets
    /// `truncated = true` upfront — the tail of the bucket table is
    /// silently dropped. The fixture stamps `size = MAX_RHT_BUCKETS +
    /// 1`; bucket reads past the buffer return 0 (empty bucket) so
    /// the walker drains all 65536 reads with no chain work.
    #[test]
    fn walk_user_dsq_hash_bucket_table_cap_truncates() {
        // Layout (page_offset = 0; identity translation):
        //   rht_pa = 0x100   (.tbl at offset 0)
        //   tbl_pa = 0x200   (size at 0, buckets at 16)
        //   tbl.size = MAX_RHT_BUCKETS + 1 = 65537.
        //
        // The walker computes bucket_count = size.min(MAX_RHT_BUCKETS)
        // = 65536, then `truncated = size as u64 > bucket_count`
        // immediately fires. Subsequent bucket reads land outside the
        // buffer and return 0 (empty bucket); no chains walked.
        let mut buf = vec![0u8; 0x300];
        let rht_pa: u64 = 0x100;
        let tbl_kva: u64 = 0x200;
        let tbl_pa: u64 = 0x200;

        // rht.tbl = tbl_kva
        buf[rht_pa as usize..rht_pa as usize + 8].copy_from_slice(&tbl_kva.to_le_bytes());
        // tbl.size = MAX_RHT_BUCKETS + 1 → upfront truncation
        let oversize: u32 = MAX_RHT_BUCKETS + 1;
        buf[tbl_pa as usize..tbl_pa as usize + 4].copy_from_slice(&oversize.to_le_bytes());

        let mem = unsafe { GuestMem::new(buf.as_mut_ptr(), buf.len() as u64) };
        let rht_offs = rht_test_offsets();
        let dsq_offs = dsq_test_offsets_for_hash();

        let (dsq_kvas, truncated) =
            walk_user_dsq_hash(&mem, WalkContext::default(), rht_pa, &rht_offs, &dsq_offs);

        assert!(
            truncated,
            "bucket-table cap (size > MAX_RHT_BUCKETS) must set truncated=true upfront",
        );
        assert!(
            dsq_kvas.is_empty(),
            "all buckets read as 0 (out-of-buffer) → no DSQ KVAs collected",
        );
    }
