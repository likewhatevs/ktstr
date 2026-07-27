//! Failure-dump timeline support.
//!
//! Host-side consumer for the `timeline_events` BPF ringbuf
//! populated by the `ktstr_tl_*` handlers in
//! `src/bpf/probe.bpf.c`: [`parse_timeline_buf`] parses drained
//! records into [`TimelineEvent`] values for the failure dump.
//!
//! The parser and [`TimelineCapture`] are re-exported from
//! [`crate::live_host`] but no host-side drain consumes them yet —
//! the BPF producer is live (its per-CPU counters surface as
//! `bpf_timeline_count` / `bpf_timeline_drops` in the probe diag),
//! the dump-pipeline consumer is the follow-up, which is why the
//! parse path carries `#[allow(dead_code)]`.
//!
//! # Layout pinning
//!
//! [`TimelineEvent`] mirrors the on-the-wire `struct timeline_event`
//! defined in `src/bpf/intf.h`. Field order, sizes, and the type
//! constants must stay in lockstep — a unit test
//! (`tests::timeline_event_layout_pinned`) verifies the 40-byte
//! footprint and field offsets against the BPF-side layout.

use serde::{Deserialize, Serialize};

/// Type-byte values from `src/bpf/intf.h::TL_EVT_*`. Pinned here as
/// the userspace-facing identifier for each variant; the parser
/// uses these to discriminate the [`TimelineEvent`] variant.
pub mod tl_evt {
    /// `tp_btf/sched_switch` record. `prev_pid`/`next_pid`/`a` (prev_state)/`b` (preempt).
    pub const SWITCH: u32 = 1;
    /// `tp_btf/sched_migrate_task` record. `prev_pid`/`a` (dest_cpu)/`b` (orig_cpu).
    pub const MIGRATE: u32 = 2;
    /// `tp_btf/sched_wakeup` record. `prev_pid`/`a` (target_cpu).
    pub const WAKEUP: u32 = 3;
    /// `fentry/fexit` rt_mutex_setprio. PI boost record.
    pub const PI_BOOST: u32 = 4;
    /// `tp_btf/lock:contention_begin` record.
    pub const LOCK_CONTEND: u32 = 5;
}

/// Wire-format mirror of `struct timeline_event` from
/// `src/bpf/intf.h`.
///
/// Layout pinning: 40 bytes total (4 type + 4 cpu + 8 ts +
/// 4 prev_pid + 4 next_pid + 8 a + 8 b). Order matches the BPF
/// emit sites in `probe.bpf.c::ktstr_tl_switch/migrate/wakeup`.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TimelineEventRaw {
    pub type_: u32,
    pub cpu: u32,
    pub ts: u64,
    pub prev_pid: u32,
    pub next_pid: u32,
    pub a: u64,
    pub b: u64,
}

/// Parsed timeline event with variant-aware field naming.
///
/// `non_exhaustive` so future BPF event types added in `intf.h`
/// (per the TL_EVT_PI_BOOST / TL_EVT_LOCK_CONTEND sites) can land
/// without breaking existing on-disk dumps.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
#[serde(tag = "kind")]
#[allow(dead_code)] // re-exported for consumers; not yet wired into
// the dump pipeline.
pub enum TimelineEvent {
    /// `tp_btf/sched_switch`. The kernel switched from `prev_pid`
    /// to `next_pid` on `cpu` at `ts` (boot-time ns).
    Switch {
        ts: u64,
        cpu: u32,
        prev_pid: u32,
        next_pid: u32,
        /// Raw `prev_state` bitfield (TASK_RUNNING / TASK_INTERRUPTIBLE
        /// / etc., from `include/linux/sched.h`).
        prev_state: u64,
        /// True when the switch was a preemption (vs voluntary
        /// dequeue).
        preempt: bool,
    },
    /// `tp_btf/sched_migrate_task`. Task `pid` migrated from
    /// `orig_cpu` to `dest_cpu`.
    Migrate {
        ts: u64,
        cpu: u32,
        pid: u32,
        orig_cpu: u32,
        dest_cpu: u32,
    },
    /// `tp_btf/sched_wakeup`. Task `pid` woken up; scheduler
    /// chose `target_cpu` for placement.
    Wakeup {
        ts: u64,
        cpu: u32,
        pid: u32,
        target_cpu: u32,
    },
    /// PI boost. Probe-context tid `prober_tid`; boosted task
    /// `pid`. `old_prio`/`new_prio` are the boosted task's effective
    /// kernel priority (signed `int`) before and after the boost.
    /// Scheduling-class transitions are counted separately via the
    /// `pi_class_changes` counter, not carried in this record. Field
    /// layout per `src/bpf/intf.h::TL_EVT_PI_BOOST`.
    PiBoost {
        ts: u64,
        cpu: u32,
        prober_tid: u32,
        pid: u32,
        old_prio: i32,
        new_prio: i32,
    },
    /// Lock contention begin. `tid` is the waiter; `lock_kva` is
    /// the lock's kernel virtual address; `flags` carries the
    /// LCB_* class bits (F_SPIN, F_READ, F_WRITE, F_RT — see
    /// `include/trace/events/lock.h`).
    LockContend {
        ts: u64,
        cpu: u32,
        tid: u32,
        lock_kva: u64,
        flags: u32,
    },
    /// Unrecognized type byte. Library doesn't drop unknown
    /// records — surfacing them as `Unknown` lets the failure
    /// dump preserve forward-compat data the consumer can opt
    /// into rendering later.
    Unknown {
        ts: u64,
        cpu: u32,
        type_: u32,
        prev_pid: u32,
        next_pid: u32,
        a: u64,
        b: u64,
    },
}

/// Parse a single 40-byte ringbuf record.
///
/// Returns `None` when the input is shorter than the on-the-wire
/// size — the caller has truncated buffer / partial read and should
/// stop draining at this slot.
#[allow(dead_code)]
pub fn parse_timeline_record(bytes: &[u8]) -> Option<TimelineEvent> {
    if bytes.len() < std::mem::size_of::<TimelineEventRaw>() {
        return None;
    }
    // SAFETY: TimelineEventRaw is repr(C) plain-data, all fields
    // are integer types so any byte pattern is a valid value.
    // The size check above guarantees we have enough bytes.
    let raw = unsafe { std::ptr::read_unaligned(bytes.as_ptr() as *const TimelineEventRaw) };
    Some(decode_raw(&raw))
}

/// Parse a contiguous buffer of timeline records into a vec of
/// [`TimelineEvent`] values, in encounter order.
///
/// `bytes` is the concatenation of N timeline_event records.
/// Trailing bytes that don't form a full record are silently
/// dropped (a torn final record at ringbuf wrap is the typical
/// case; the consumer's next drain picks up the remainder).
#[allow(dead_code)]
pub fn parse_timeline_buf(bytes: &[u8]) -> Vec<TimelineEvent> {
    let stride = std::mem::size_of::<TimelineEventRaw>();
    let mut out = Vec::with_capacity(bytes.len() / stride);
    let mut off = 0;
    while off + stride <= bytes.len() {
        if let Some(ev) = parse_timeline_record(&bytes[off..off + stride]) {
            out.push(ev);
        }
        off += stride;
    }
    out
}

fn decode_raw(raw: &TimelineEventRaw) -> TimelineEvent {
    match raw.type_ {
        tl_evt::SWITCH => TimelineEvent::Switch {
            ts: raw.ts,
            cpu: raw.cpu,
            prev_pid: raw.prev_pid,
            next_pid: raw.next_pid,
            prev_state: raw.a,
            preempt: raw.b != 0,
        },
        tl_evt::MIGRATE => TimelineEvent::Migrate {
            ts: raw.ts,
            cpu: raw.cpu,
            pid: raw.prev_pid,
            orig_cpu: raw.b as u32,
            dest_cpu: raw.a as u32,
        },
        tl_evt::WAKEUP => TimelineEvent::Wakeup {
            ts: raw.ts,
            cpu: raw.cpu,
            pid: raw.prev_pid,
            target_cpu: raw.a as u32,
        },
        tl_evt::PI_BOOST => {
            // The producer widens the signed kernel prio to u64 via
            // (u64)(s64)prio; truncating back to i32 recovers the
            // original signed value. No class id is packed in the high
            // bits — class transitions surface via the pi_class_changes
            // counter, not this record.
            let old_prio = raw.a as i32;
            let new_prio = raw.b as i32;
            TimelineEvent::PiBoost {
                ts: raw.ts,
                cpu: raw.cpu,
                prober_tid: raw.prev_pid,
                pid: raw.next_pid,
                old_prio,
                new_prio,
            }
        }
        tl_evt::LOCK_CONTEND => TimelineEvent::LockContend {
            ts: raw.ts,
            cpu: raw.cpu,
            tid: raw.prev_pid,
            lock_kva: raw.a,
            flags: raw.b as u32,
        },
        _ => TimelineEvent::Unknown {
            ts: raw.ts,
            cpu: raw.cpu,
            type_: raw.type_,
            prev_pid: raw.prev_pid,
            next_pid: raw.next_pid,
            a: raw.a,
            b: raw.b,
        },
    }
}

/// Capture handle for the freeze coordinator's drain of the
/// `timeline_events` BPF ringbuf.
///
/// At dump time the coordinator constructs this with the drained
/// raw bytes (concatenated 40-byte records, in ringbuf order) plus
/// the BSS-side drop count. The dump consumer parses the buffer
/// into [`TimelineEvent`] values. This capture is not yet consumed
/// by the dump pipeline; the BSS-side drop count is surfaced today
/// via [`super::dump::ProbeBssCounters::timeline_drops`] (reached
/// through `super::dump::FailureDumpReport::probe_counters`).
#[derive(Debug, Clone, Default)]
#[allow(dead_code)]
pub struct TimelineCapture<'a> {
    /// Raw concatenated record bytes drained from the
    /// `timeline_events` ringbuf. Length must be a multiple of
    /// `size_of::<TimelineEventRaw>()` (40); trailing partial
    /// records are silently dropped at parse time.
    pub records: &'a [u8],
    /// `KTSTR_PCPU_TIMELINE_DROPS` per-CPU slot (summed across
    /// CPUs) at drain time. Non-zero indicates the BPF producer
    /// hit a full ringbuf and dropped the newest event(s) on
    /// submit.
    pub drops: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `TimelineEventRaw` matches the BPF-side struct timeline_event
    /// in size and field offsets. Drift here is a wire-protocol
    /// break; the test catches it at compile + run time.
    ///
    /// Verdict-routed so a multi-field layout regression (e.g.
    /// somebody re-orders the struct) surfaces every drift in one
    /// run rather than failing on the first mismatch.
    #[test]
    fn timeline_event_layout_pinned() {
        use crate::assert::Verdict;

        let total_size = std::mem::size_of::<TimelineEventRaw>();
        let off_type = std::mem::offset_of!(TimelineEventRaw, type_);
        let off_cpu = std::mem::offset_of!(TimelineEventRaw, cpu);
        let off_ts = std::mem::offset_of!(TimelineEventRaw, ts);
        let off_prev_pid = std::mem::offset_of!(TimelineEventRaw, prev_pid);
        let off_next_pid = std::mem::offset_of!(TimelineEventRaw, next_pid);
        let off_a = std::mem::offset_of!(TimelineEventRaw, a);
        let off_b = std::mem::offset_of!(TimelineEventRaw, b);

        let mut v = Verdict::new();
        // Total: 4 + 4 + 8 + 4 + 4 + 8 + 8 = 40 bytes.
        crate::claim!(v, total_size).eq(40usize);
        // Field offsets matching src/bpf/intf.h::struct timeline_event.
        crate::claim!(v, off_type).eq(0usize);
        crate::claim!(v, off_cpu).eq(4usize);
        crate::claim!(v, off_ts).eq(8usize);
        crate::claim!(v, off_prev_pid).eq(16usize);
        crate::claim!(v, off_next_pid).eq(20usize);
        crate::claim!(v, off_a).eq(24usize);
        crate::claim!(v, off_b).eq(32usize);
        let r = v.into_result();
        assert!(
            r.is_pass(),
            "timeline_event layout drift detected: {:?}",
            r.outcomes,
        );
    }

    fn raw(type_: u32, cpu: u32, ts: u64, p: u32, n: u32, a: u64, b: u64) -> Vec<u8> {
        let r = TimelineEventRaw {
            type_,
            cpu,
            ts,
            prev_pid: p,
            next_pid: n,
            a,
            b,
        };
        // Safety: r is plain-data; reading its bytes is well-defined.
        let bytes = unsafe {
            std::slice::from_raw_parts(
                &r as *const TimelineEventRaw as *const u8,
                std::mem::size_of::<TimelineEventRaw>(),
            )
        };
        bytes.to_vec()
    }

    /// Switch record decodes with prev/next pids + prev_state +
    /// preempt bool. Verdict-routed so every field surfaces its own
    /// labeled detail on regression.
    #[test]
    fn parse_switch_record() {
        use crate::assert::Verdict;

        let bytes = raw(tl_evt::SWITCH, 3, 1_000_000, 100, 200, 0x402, 1);
        let ev = parse_timeline_record(&bytes).unwrap();
        match ev {
            TimelineEvent::Switch {
                ts,
                cpu,
                prev_pid,
                next_pid,
                prev_state,
                preempt,
            } => {
                let mut v = Verdict::new();
                crate::claim!(v, ts).eq(1_000_000u64);
                crate::claim!(v, cpu).eq(3u32);
                crate::claim!(v, prev_pid).eq(100u32);
                crate::claim!(v, next_pid).eq(200u32);
                crate::claim!(v, prev_state).eq(0x402u64);
                crate::claim!(v, preempt).eq(true);
                let r = v.into_result();
                assert!(r.is_pass(), "Switch record decode drift: {:?}", r.outcomes,);
            }
            other => panic!("expected Switch, got {other:?}"),
        }
    }

    /// Migrate record decodes with pid + orig_cpu + dest_cpu.
    /// Per intf.h: a = dest_cpu, b = orig_cpu.
    #[test]
    fn parse_migrate_record() {
        let bytes = raw(tl_evt::MIGRATE, 1, 2_000_000, 555, 0, 7, 2);
        let ev = parse_timeline_record(&bytes).unwrap();
        match ev {
            TimelineEvent::Migrate {
                pid,
                orig_cpu,
                dest_cpu,
                ..
            } => {
                assert_eq!(pid, 555);
                assert_eq!(dest_cpu, 7);
                assert_eq!(orig_cpu, 2);
            }
            other => panic!("expected Migrate, got {other:?}"),
        }
    }

    /// Wakeup record decodes with pid + target_cpu.
    #[test]
    fn parse_wakeup_record() {
        let bytes = raw(tl_evt::WAKEUP, 0, 3_000_000, 777, 0, 4, 0);
        let ev = parse_timeline_record(&bytes).unwrap();
        match ev {
            TimelineEvent::Wakeup {
                pid, target_cpu, ..
            } => {
                assert_eq!(pid, 777);
                assert_eq!(target_cpu, 4);
            }
            other => panic!("expected Wakeup, got {other:?}"),
        }
    }

    /// PiBoost carries the signed kernel prio in a/b, widened to u64
    /// by the producer via `(u64)(s64)prio`. The decoder truncates
    /// back to i32 — a negative prio (task boosted into the RT band)
    /// must round-trip, which the old `(prio | class_id<<32)` split
    /// corrupted by reading the sign-extension bits as a class id.
    #[test]
    fn parse_pi_boost_record() {
        let old_a = (120i32 as i64) as u64; // prio=120 (normal band)
        let new_b = (-1i32 as i64) as u64; // prio=-1 (sign-extended)
        let bytes = raw(tl_evt::PI_BOOST, 2, 4_000_000, 10, 11, old_a, new_b);
        let ev = parse_timeline_record(&bytes).unwrap();
        match ev {
            TimelineEvent::PiBoost {
                prober_tid,
                pid,
                old_prio,
                new_prio,
                ..
            } => {
                assert_eq!(prober_tid, 10);
                assert_eq!(pid, 11);
                assert_eq!(old_prio, 120);
                assert_eq!(new_prio, -1);
            }
            other => panic!("expected PiBoost, got {other:?}"),
        }
    }

    /// LockContend record carries lock_kva + flags.
    #[test]
    fn parse_lock_contend_record() {
        let lock_kva = 0xffff_ffff_8000_1000u64;
        let flags = 0x4u64;
        let bytes = raw(tl_evt::LOCK_CONTEND, 5, 5_000_000, 99, 0, lock_kva, flags);
        let ev = parse_timeline_record(&bytes).unwrap();
        match ev {
            TimelineEvent::LockContend {
                tid,
                lock_kva: kva,
                flags: f,
                ..
            } => {
                assert_eq!(tid, 99);
                assert_eq!(kva, lock_kva);
                assert_eq!(f, 0x4);
            }
            other => panic!("expected LockContend, got {other:?}"),
        }
    }

    /// Unknown type byte surfaces as Unknown variant — preserves
    /// forward-compat data for newer kernels with TL_EVT_* values
    /// the consumer doesn't yet decode.
    #[test]
    fn parse_unknown_type_preserves_fields() {
        let bytes = raw(99, 7, 6_000_000, 1, 2, 3, 4);
        let ev = parse_timeline_record(&bytes).unwrap();
        match ev {
            TimelineEvent::Unknown {
                type_,
                prev_pid,
                a,
                b,
                ..
            } => {
                assert_eq!(type_, 99);
                assert_eq!(prev_pid, 1);
                assert_eq!(a, 3);
                assert_eq!(b, 4);
            }
            other => panic!("expected Unknown, got {other:?}"),
        }
    }

    /// Truncated record returns None — the drain loop stops
    /// parsing rather than reading past end-of-buffer.
    #[test]
    fn parse_truncated_record_returns_none() {
        let bytes = vec![0u8; 39]; // 1 byte short of 40
        assert!(parse_timeline_record(&bytes).is_none());
    }

    /// `parse_timeline_buf` parses every full record in a multi-
    /// record buffer and silently drops a partial trailing record.
    #[test]
    fn parse_timeline_buf_multi_record_with_partial_tail() {
        let mut buf: Vec<u8> = Vec::new();
        buf.extend(raw(tl_evt::SWITCH, 0, 1, 1, 2, 0, 0));
        buf.extend(raw(tl_evt::WAKEUP, 1, 2, 3, 0, 4, 0));
        // Append 20 bytes of partial record — must not parse.
        buf.extend(vec![0u8; 20]);
        let evs = parse_timeline_buf(&buf);
        assert_eq!(evs.len(), 2);
        assert!(matches!(evs[0], TimelineEvent::Switch { .. }));
        assert!(matches!(evs[1], TimelineEvent::Wakeup { .. }));
    }

    /// `TimelineEvent` round-trips through serde — every variant
    /// survives the json string.
    #[test]
    fn timeline_event_serde_roundtrip_all_variants() {
        let cases = vec![
            TimelineEvent::Switch {
                ts: 1,
                cpu: 0,
                prev_pid: 10,
                next_pid: 20,
                prev_state: 1,
                preempt: false,
            },
            TimelineEvent::Migrate {
                ts: 2,
                cpu: 1,
                pid: 30,
                orig_cpu: 1,
                dest_cpu: 2,
            },
            TimelineEvent::Wakeup {
                ts: 3,
                cpu: 2,
                pid: 40,
                target_cpu: 5,
            },
            TimelineEvent::PiBoost {
                ts: 4,
                cpu: 3,
                prober_tid: 1,
                pid: 2,
                old_prio: 120,
                new_prio: 100,
            },
            TimelineEvent::LockContend {
                ts: 5,
                cpu: 4,
                tid: 99,
                lock_kva: 0xffff_ffff,
                flags: 0x4,
            },
        ];
        for ev in cases {
            let json = serde_json::to_string(&ev).expect("serialize");
            let _: TimelineEvent = serde_json::from_str(&json).expect("deserialize");
        }
    }
}
