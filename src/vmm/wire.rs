//! Shared wire-format types for the host/guest virtio-console port-1
//! TLV stream and the multiport control protocol.
//!
//! Both [`super::guest_comms`] (guest-only senders) and
//! [`super::host_comms`] (host-only consumers) reference this module.
//! Splitting the wire format out of the transport modules keeps the
//! frame layout authoritative — a producer change here lands in both
//! the guest writer and the host parser without a hand-sync step.
//!
//! # Postcard wire-format pin inventory (contributor guide)
//!
//! Every type that crosses the in-VM postcard TLV channel MUST be
//! externally-tagged (postcard cannot decode `#[serde(untagged)]`
//! or `#[serde(tag, content)]` enums — encode raises `WontImplement`
//! at runtime and the host surfaces it as
//! `ERR_NO_TEST_FUNCTION_OUTPUT`). The compile-time
//! `#[derive(serde::Serialize, serde::Deserialize)]` itself does
//! NOT catch the shape mismatch; the contract is verified by the
//! per-type roundtrip pin tests listed below. A contributor adding
//! a new postcard payload MUST add a roundtrip pin in the
//! corresponding location and update this inventory.
//!
//! Inventory (type → test name → file):
//!   - `AssertResult` → `assert_result_postcard_roundtrip`
//!     → `src/assert/tests_serde.rs`
//!   - `KernAddrs` → `kern_addrs_roundtrip_all_present` (+ 4 sibling
//!     boundary pins) → `src/vmm/wire.rs` (this file)
//!   - `KernelOpRequestPayload` / `KernelOpReplyPayload`
//!     → 4 tests → `src/vmm/wire.rs` (this file)
//!   - `PayloadMetrics` / `RawPayloadOutput`
//!     → `payload_metrics_postcard_roundtrip` /
//!     `raw_payload_output_postcard_roundtrip`
//!     → `src/test_support/payload.rs`
//!   - `WorkloadConfig` → `payload_roundtrip`
//!     → `src/test_support/payload.rs`
//!   - `WorkerReport` → `worker_report_postcard_roundtrip` (+ 3 sibling
//!     pins covering `Vec<WorkerReport>` + all `ExitInfo` variants)
//!     → `src/workload/spawn/tests_integration.rs`
//!   - `PersistedCastAnalysis` → see `src/vmm/cast_analysis_load`
//!     module's tests
//!
//! # Frame layout
//!
//! Each guest→host bulk message is a 16-byte [`ShmMessage`] header
//! followed by `length` payload bytes. The host's
//! [`super::host_comms::parse_tlv_stream`] consumes this format. CRC32
//! covers payload bytes only, not the header.
//!
//! ```text
//! offset  size  field
//! ------  ----  ----------------------------------------------
//!   0      4    msg_type (u32 LE)  — see [`MsgType`]
//!   4      4    length   (u32 LE)  — payload bytes following
//!   8      4    crc32    (u32 LE)  — crc32fast over payload
//!  12      4    _pad     (u32 LE)  — reserved, MUST be zero
//!  16      N    payload  (N=length bytes)
//! ```
//!
//! # Control protocol
//!
//! [`VirtioConsoleControl`] mirrors the kernel uapi `struct
//! virtio_console_control` for multiport handshake messages on the
//! c_ivq / c_ovq queues (8 bytes: id u32, event u16, value u16).
//! [`ControlEvent`] enumerates the event discriminants the kernel and
//! the host VMM exchange during port enumeration.
//!
//! Many of the typed wrappers and constants in this module are part
//! of the public bulk API surface; the lib build does not yet read
//! every variant from internal call sites (the typed `MsgType` enum,
//! `ControlEvent`, `VirtioConsoleControl`, `NUM_PORTS`, `PORT1_NAME`,
//! and the `from_wire` reverse mappings are reachable via the public
//! crate path for downstream test code and wire-format tests). The
//! module-level `#[allow(dead_code)]` matches the `VmResult` field
//! pattern in `result.rs` — public surface that the in-tree readers
//! do not exercise without the unused-X lint firing.

#![allow(dead_code)]

use zerocopy::{FromBytes, IntoBytes};

// ---------------------------------------------------------------------------
// MsgType — typed message-type discriminant
// ---------------------------------------------------------------------------

/// Message-type discriminant for the bulk TLV stream.
///
/// Each variant maps to a 32-bit on-wire value via [`Self::wire_value`].
/// The values are 4-character ASCII tags chosen so the integer literal
/// itself spells the tag in hex (e.g. `0x4558_4954` reads as `"EXIT"`
/// — `45`='E', `58`='X', `49`='I', `54`='T'). Because the wire format
/// is little-endian, a raw byte-level hex dump of a captured frame
/// shows the bytes in reverse order (e.g. `54 49 58 45` for the
/// `Exit` tag, which spells `"TIXE"` byte-by-byte). The integer
/// hex value spells the tag; the on-wire bytes are reversed.
///
/// On-wire values are stable across host/guest builds — adding a new
/// variant requires picking a fresh ASCII tag and updating
/// [`Self::from_wire`] to recognise it. Existing tags must never be
/// repurposed.
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub enum MsgType {
    /// Stimulus event from the guest step executor — emitted at each
    /// step's START (the StepStart frame).
    Stimulus,
    /// Per-step END frame from the guest step executor, emitted at the
    /// end of each step's hold while its workers are still alive.
    /// Reuses the [`StimulusPayload`] body (same 24 bytes) carrying the
    /// step's coincident end-of-hold (elapsed_ms, total_iterations) and
    /// the SAME 1-indexed `step_index` as its StepStart. The host pairs
    /// `StepStart[k]` -> `StepEnd[k]` for each step's OWN throughput
    /// (step-local iteration_rate), which — unlike the cross-step
    /// `StepStart[k]` -> `StepStart[k+1]` delta — does not read ~0 for
    /// workers respawned per step.
    StepEnd,
    /// Scenario start marker. Sets a fresh watchdog deadline.
    ScenarioStart,
    /// Scenario end marker (payload: two 8-byte LE u64s — elapsed_ms
    /// then the final cumulative `total_iterations`; see
    /// [`SCENARIO_END_PAYLOAD_SIZE`] / [`parse_scenario_end`]).
    ScenarioEnd,
    /// Pause the watchdog clock. Wall time while paused doesn't
    /// count against the workload budget.
    ScenarioPause,
    /// Resume the watchdog clock after a pause. Extends the deadline
    /// by the pause duration — gives back the paused time.
    ScenarioResume,
    /// Guest exit code (payload: 4-byte LE i32).
    Exit,
    /// Test result (payload: postcard-encoded `AssertResult`).
    TestResult,
    /// Scheduler process exit (payload: 4-byte LE i32 exit code).
    SchedExit,
    /// Guest crash diagnostic (payload: UTF-8 panic + backtrace).
    /// Reserved tag — never travels on the bulk port. Panic
    /// diagnostics are written directly to COM2 (`/dev/ttyS1`)
    /// because `virtio_console` TX can block on host backpressure
    /// and blocking inside a fault handler would deadlock the
    /// guest before the diagnostic reached the host.
    Crash,
    /// Per-payload-invocation metrics (payload: postcard-encoded
    /// `PayloadMetrics`).
    PayloadMetrics,
    /// Raw stdout/stderr captured from an LlmExtract payload (payload:
    /// postcard-encoded `RawPayloadOutput`).
    RawPayloadOutput,
    /// Coverage profraw blob.
    Profraw,
    /// Guest→host stdout chunk. Payload: opaque UTF-8 bytes. Each
    /// frame carries one chunk read from the guest's stdout pipe;
    /// host concatenates chunks in arrival order to reconstruct the
    /// stream. Replaces the prior COM2 stdout redirect.
    Stdout,
    /// Guest→host stderr chunk. Payload: opaque UTF-8 bytes. Same
    /// chunked semantics as [`Self::Stdout`].
    Stderr,
    /// Guest→host scheduler-log chunk. Payload: opaque UTF-8 bytes
    /// from the scheduler child process's captured log. Replaces
    /// the prior COM2 SCHED_OUTPUT_START/END dump path. The host
    /// concatenates chunks in arrival order; the existing
    /// `SCHED_OUTPUT_START` / `SCHED_OUTPUT_END` delimiters and the
    /// embedded BPF verifier section are preserved verbatim
    /// inside the chunk bytes.
    SchedLog,
    /// Guest→host lifecycle phase event. Payload: 1-byte
    /// [`LifecyclePhase`] discriminant followed by an optional
    /// UTF-8 reason buffer (used by `SchedulerNotAttached`'s
    /// suffix detail). Replaces the prior `KTSTR_INIT_STARTED` /
    /// `KTSTR_PAYLOAD_STARTING` / `SCHEDULER_DIED` /
    /// `SCHEDULER_NOT_ATTACHED` sentinel strings on COM2.
    Lifecycle,
    /// Guest→host shell-exec exit. Payload: 4-byte LE i32 exit
    /// code from `cargo ktstr shell --exec <cmd>`. Replaces the
    /// prior COM2 `KTSTR_EXEC_EXIT=N` sentinel line.
    ExecExit,
    /// Guest→host kernel ring-buffer dump. Payload: opaque UTF-8
    /// bytes from `rmesg::logs_raw`. Sent on the
    /// initramfs-extraction failure path so the host sees the
    /// kernel OOM messages without scraping COM2.
    Dmesg,
    /// Guest→host probe-pipeline JSON output. Payload: opaque UTF-8
    /// bytes from the probe output stream. Replaces the prior
    /// COM2 ProbeDrain path so probe JSON does not interleave
    /// with sched-log dumps on the same serial port.
    ProbeOutput,
    /// Guest→host on-demand snapshot request (payload:
    /// [`SnapshotRequestPayload`]). The freeze coordinator's bulk-drain
    /// path intercepts this frame, runs the CAPTURE / WATCH dispatch,
    /// and replies with [`MsgType::SnapshotReply`] on port 1 RX.
    SnapshotRequest,
    /// Host→guest snapshot reply (payload: [`SnapshotReplyPayload`]).
    /// Sent on port 1 RX so the guest's blocking read on
    /// `/dev/vport0p1` wakes within microseconds. Reply payload
    /// carries the matching request_id, the status, and a UTF-8
    /// reason buffer for the failure path.
    SnapshotReply,
    /// Guest→host kernel-memory write/read op request (payload:
    /// postcard-encoded [`KernelOpRequestPayload`]). Carries the
    /// `Op::WriteKernel{Hot,Cold}` / `Op::ReadKernel{Hot,Cold}`
    /// invocation from the guest's step executor; variable-length
    /// payload rides this distinct MSG_TYPE rather than extending
    /// the fixed-72-byte [`SnapshotRequestPayload`].
    KernelOpRequest,
    /// Host→guest reply to [`MsgType::KernelOpRequest`] (payload:
    /// postcard-encoded [`KernelOpReplyPayload`]).
    KernelOpReply,
    /// Guest→host wprof Perfetto-format trace blob (payload: raw
    /// `.pb` bytes from `wprof -T trace.pb`). The freeze
    /// coordinator writes the payload as `wprof.pb` next to the
    /// failure-dump JSON so the operator picks it up alongside the
    /// rest of the per-test debugging artefacts.
    WprofTrace,
    /// Guest→host system-ready signal (payload: empty).
    ///
    /// Emitted by the guest's `ktstr_guest_init` after
    /// `mount_filesystems()` completes, so by the time the host
    /// observes the frame the guest's `setup_per_cpu_areas` and
    /// KASLR randomization (both kernel-boot prerequisites) are
    /// already done. The freeze coordinator's bulk-drain dispatch
    /// promotes a CRC-valid SYS_RDY frame into the monitor's
    /// boot-complete eventfd, so the monitor's pre-sample
    /// `epoll_wait` returns within microseconds rather than
    /// waiting for the 30 s fallback timeout. Replaces an earlier
    /// trigger that fired on the first port-0 TX byte (kernel
    /// printk via `/dev/hvc0`), which depended on incidental
    /// console traffic rather than an explicit readiness signal.
    SysRdy,
}

impl MsgType {
    /// 32-bit on-wire discriminant for this message type. The value is
    /// the big-endian ASCII representation of a 4-character tag.
    pub const fn wire_value(self) -> u32 {
        match self {
            MsgType::Stimulus => MSG_TYPE_STIMULUS,
            MsgType::StepEnd => MSG_TYPE_STEP_END,
            MsgType::ScenarioStart => MSG_TYPE_SCENARIO_START,
            MsgType::ScenarioEnd => MSG_TYPE_SCENARIO_END,
            MsgType::ScenarioPause => MSG_TYPE_SCENARIO_PAUSE,
            MsgType::ScenarioResume => MSG_TYPE_SCENARIO_RESUME,
            MsgType::Exit => MSG_TYPE_EXIT,
            MsgType::TestResult => MSG_TYPE_TEST_RESULT,
            MsgType::SchedExit => MSG_TYPE_SCHED_EXIT,
            MsgType::Crash => MSG_TYPE_CRASH,
            MsgType::PayloadMetrics => MSG_TYPE_PAYLOAD_METRICS,
            MsgType::RawPayloadOutput => MSG_TYPE_RAW_PAYLOAD_OUTPUT,
            MsgType::Profraw => MSG_TYPE_PROFRAW,
            MsgType::WprofTrace => MSG_TYPE_WPROF_TRACE,
            MsgType::SnapshotRequest => MSG_TYPE_SNAPSHOT_REQUEST,
            MsgType::SnapshotReply => MSG_TYPE_SNAPSHOT_REPLY,
            MsgType::KernelOpRequest => MSG_TYPE_KERNEL_OP_REQUEST,
            MsgType::KernelOpReply => MSG_TYPE_KERNEL_OP_REPLY,
            MsgType::SysRdy => MSG_TYPE_SYS_RDY,
            MsgType::Stdout => MSG_TYPE_STDOUT,
            MsgType::Stderr => MSG_TYPE_STDERR,
            MsgType::SchedLog => MSG_TYPE_SCHED_LOG,
            MsgType::Lifecycle => MSG_TYPE_LIFECYCLE,
            MsgType::ExecExit => MSG_TYPE_EXEC_EXIT,
            MsgType::Dmesg => MSG_TYPE_DMESG,
            MsgType::ProbeOutput => MSG_TYPE_PROBE_OUTPUT,
        }
    }

    /// Reverse the wire mapping. Returns `None` when `value` is not a
    /// recognised discriminant — callers can either skip the frame or
    /// surface the unknown tag for diagnostics.
    pub const fn from_wire(value: u32) -> Option<Self> {
        match value {
            MSG_TYPE_STIMULUS => Some(MsgType::Stimulus),
            MSG_TYPE_STEP_END => Some(MsgType::StepEnd),
            MSG_TYPE_SCENARIO_START => Some(MsgType::ScenarioStart),
            MSG_TYPE_SCENARIO_END => Some(MsgType::ScenarioEnd),
            MSG_TYPE_SCENARIO_PAUSE => Some(MsgType::ScenarioPause),
            MSG_TYPE_SCENARIO_RESUME => Some(MsgType::ScenarioResume),
            MSG_TYPE_EXIT => Some(MsgType::Exit),
            MSG_TYPE_TEST_RESULT => Some(MsgType::TestResult),
            MSG_TYPE_SCHED_EXIT => Some(MsgType::SchedExit),
            MSG_TYPE_CRASH => Some(MsgType::Crash),
            MSG_TYPE_PAYLOAD_METRICS => Some(MsgType::PayloadMetrics),
            MSG_TYPE_RAW_PAYLOAD_OUTPUT => Some(MsgType::RawPayloadOutput),
            MSG_TYPE_PROFRAW => Some(MsgType::Profraw),
            MSG_TYPE_WPROF_TRACE => Some(MsgType::WprofTrace),
            MSG_TYPE_SNAPSHOT_REQUEST => Some(MsgType::SnapshotRequest),
            MSG_TYPE_SNAPSHOT_REPLY => Some(MsgType::SnapshotReply),
            MSG_TYPE_KERNEL_OP_REQUEST => Some(MsgType::KernelOpRequest),
            MSG_TYPE_KERNEL_OP_REPLY => Some(MsgType::KernelOpReply),
            MSG_TYPE_SYS_RDY => Some(MsgType::SysRdy),
            MSG_TYPE_STDOUT => Some(MsgType::Stdout),
            MSG_TYPE_STDERR => Some(MsgType::Stderr),
            MSG_TYPE_SCHED_LOG => Some(MsgType::SchedLog),
            MSG_TYPE_LIFECYCLE => Some(MsgType::Lifecycle),
            MSG_TYPE_EXEC_EXIT => Some(MsgType::ExecExit),
            MSG_TYPE_DMESG => Some(MsgType::Dmesg),
            MSG_TYPE_PROBE_OUTPUT => Some(MsgType::ProbeOutput),
            _ => None,
        }
    }

    /// `true` for control frames the freeze coordinator interprets
    /// internally and that must NOT surface as test verdict entries
    /// in [`super::host_comms::BulkDrainResult`]. Both the
    /// coordinator's mid-run `bulk_messages_for_closure` filter and
    /// `collect_results`'s post-run drain key on this single
    /// classifier so the gate stays in lockstep — adding a new
    /// internal control frame is a one-line update here.
    ///
    /// The current internal set:
    ///   - [`MsgType::SnapshotRequest`] — has its matching
    ///     [`MsgType::SnapshotReply`] delivered over port-1 RX; the
    ///     request itself carries no test verdict.
    ///   - [`MsgType::SnapshotReply`] — host→guest only on port-1 RX.
    ///     A guest TX frame stamped with this tag is illegitimate
    ///     (only the host coordinator emits replies); drop it instead
    ///     of bucketing it as a phantom verdict entry. Including the
    ///     tag in the internal set keeps the dispatch and the
    ///     `collect_results` post-run drain in lockstep — both filter
    ///     the same way.
    ///   - [`MsgType::KernelOpRequest`] — paired with its
    ///     [`MsgType::KernelOpReply`] over port-1 RX (the cold-path
    ///     kernel-op roundtrip); the request carries no test verdict.
    ///   - [`MsgType::KernelOpReply`] — host→guest only on port-1 RX,
    ///     same illegitimate-guest-TX reasoning as `SnapshotReply`.
    ///   - [`MsgType::SysRdy`] — its only semantic is the eventfd
    ///     promotion that releases the monitor's pre-sample
    ///     `epoll_wait`.
    pub const fn is_coordinator_internal(self) -> bool {
        matches!(
            self,
            MsgType::SnapshotRequest
                | MsgType::SnapshotReply
                | MsgType::KernelOpRequest
                | MsgType::KernelOpReply
                | MsgType::SysRdy
        )
    }
}

/// Lifecycle phase carried in the 1-byte header of a
/// [`MsgType::Lifecycle`] payload. Replaces the prior
/// `KTSTR_INIT_STARTED` / `KTSTR_PAYLOAD_STARTING` /
/// `SCHEDULER_DIED` / `SCHEDULER_NOT_ATTACHED` COM2 sentinels.
///
/// `SchedulerNotAttached` carries an optional UTF-8 reason suffix
/// (the bytes following the 1-byte phase header in the TLV
/// payload) — every other variant has an empty suffix.
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub enum LifecyclePhase {
    /// Init started — devtmpfs mounted, initramfs verified,
    /// equivalent to the legacy `KTSTR_INIT_STARTED` sentinel.
    InitStarted,
    /// Payload starting — guest dispatch is about to invoke the
    /// `#[ktstr_test]` body. Equivalent to the legacy
    /// `KTSTR_PAYLOAD_STARTING` sentinel.
    PayloadStarting,
    /// Scheduler process exited during startup. Equivalent to the
    /// legacy `SCHEDULER_DIED` sentinel.
    SchedulerDied,
    /// Scheduler stayed alive but never attached to sched_ext (BPF
    /// verifier reject, ops mismatch, sysfs absent). Equivalent to
    /// the legacy `SCHEDULER_NOT_ATTACHED:<reason>` sentinel; the
    /// reason suffix lives in the bytes after the 1-byte phase
    /// header.
    SchedulerNotAttached,
}

impl LifecyclePhase {
    /// 1-byte on-wire discriminant. `0` is reserved as the
    /// "unknown / invalid" sentinel — host parsers reject zero
    /// rather than silently mapping it to a known phase.
    pub const fn wire_value(self) -> u8 {
        match self {
            LifecyclePhase::InitStarted => 1,
            LifecyclePhase::PayloadStarting => 2,
            LifecyclePhase::SchedulerDied => 3,
            LifecyclePhase::SchedulerNotAttached => 4,
        }
    }

    /// Reverse the wire mapping. Returns `None` for `0`
    /// (reserved sentinel) or any value not present in the variant
    /// list — host parsers skip unknown phases and log them rather
    /// than panicking.
    pub const fn from_wire(value: u8) -> Option<Self> {
        match value {
            1 => Some(LifecyclePhase::InitStarted),
            2 => Some(LifecyclePhase::PayloadStarting),
            3 => Some(LifecyclePhase::SchedulerDied),
            4 => Some(LifecyclePhase::SchedulerNotAttached),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// On-wire u32 discriminants
// ---------------------------------------------------------------------------
//
// Kept as `pub const` for callers that compare a parsed frame's
// `msg_type` field directly (e.g. the freeze coordinator's stream
// filter). [`MsgType::wire_value`] is the typed entry point; the
// constants are the same values exposed for raw-byte comparisons.

/// Stimulus event from the guest step executor (step START frame).
pub const MSG_TYPE_STIMULUS: u32 = 0x5354_494D; // "STIM"

/// Per-step END frame from the guest step executor (reuses the
/// [`StimulusPayload`] body; see [`MsgType::StepEnd`]).
pub const MSG_TYPE_STEP_END: u32 = 0x5354_454E; // "STEN"

/// Scenario start marker.
pub const MSG_TYPE_SCENARIO_START: u32 = 0x5343_5354; // "SCST"

/// Pause watchdog clock.
pub const MSG_TYPE_SCENARIO_PAUSE: u32 = 0x5343_5050; // "SCPP"

/// Resume watchdog clock after pause.
pub const MSG_TYPE_SCENARIO_RESUME: u32 = 0x5343_5252; // "SCRR"

/// Scenario end marker.
pub const MSG_TYPE_SCENARIO_END: u32 = 0x5343_454E; // "SCEN"

/// Guest exit code (payload: 4-byte i32).
pub const MSG_TYPE_EXIT: u32 = 0x4558_4954; // "EXIT"

/// Test result (payload: postcard-encoded AssertResult).
pub const MSG_TYPE_TEST_RESULT: u32 = 0x5445_5354; // "TEST"

/// Scheduler process exit (payload: 4-byte i32 exit code).
pub const MSG_TYPE_SCHED_EXIT: u32 = 0x5343_4458; // "SCDX"

/// Guest crash diagnostic (payload: UTF-8 panic + backtrace).
pub const MSG_TYPE_CRASH: u32 = 0x4352_5348; // "CRSH"

/// Per-payload-invocation metrics
/// (payload: postcard-encoded `crate::test_support::PayloadMetrics`).
pub const MSG_TYPE_PAYLOAD_METRICS: u32 = 0x504d_4554; // "PMET"

/// Raw stdout/stderr captured from an LlmExtract payload
/// (payload: postcard-encoded `crate::test_support::RawPayloadOutput`).
pub const MSG_TYPE_RAW_PAYLOAD_OUTPUT: u32 = 0x5241_574f; // "RAWO"

/// Coverage profraw blob (payload: raw `.profraw` bytes serialized by
/// `__llvm_profile_write_buffer`).
pub const MSG_TYPE_PROFRAW: u32 = 0x5052_4157; // "PRAW"

/// wprof Perfetto-format trace blob (payload: raw `.pb` bytes
/// produced by `/bin/wprof -T trace.pb` during auto-repro). The
/// host's freeze coordinator writes the payload to a sibling of
/// the failure-dump file so the operator finds it under
/// [`crate::test_support::sidecar_dir`] alongside the JSON dump.
pub const MSG_TYPE_WPROF_TRACE: u32 = 0x5750_5246; // "WPRF"

/// Guest→host on-demand snapshot request
/// (payload: [`SnapshotRequestPayload`]).
pub const MSG_TYPE_SNAPSHOT_REQUEST: u32 = 0x534e_5251; // "SNRQ"

/// Host→guest on-demand snapshot reply
/// (payload: [`SnapshotReplyPayload`]).
pub const MSG_TYPE_SNAPSHOT_REPLY: u32 = 0x534e_5250; // "SNRP"

/// Guest→host system-ready signal (payload: empty).
///
/// Tag spelled `"SRDY"` in hex digits; on-wire bytes (LE) are
/// `0x59 0x44 0x52 0x53` (`"YDRS"` byte-by-byte). The freeze
/// coordinator's bulk-drain dispatch promotes a CRC-valid
/// `MSG_TYPE_SYS_RDY` frame into the monitor's boot-complete
/// eventfd. See [`MsgType::SysRdy`] for the protocol contract.
pub const MSG_TYPE_SYS_RDY: u32 = 0x5352_4459; // "SRDY"

/// Guest→host stdout chunk (payload: opaque UTF-8 bytes).
///
/// Replaces the prior COM2 stdout redirect: the guest dups fd 1
/// onto the write-end of an internal pipe and a forwarder thread
/// chunks the pipe's read-end into TLV frames bounded by
/// [`super::bulk::MAX_BULK_FRAME_PAYLOAD`].
pub const MSG_TYPE_STDOUT: u32 = 0x534f_5554; // "SOUT"

/// Guest→host stderr chunk (payload: opaque UTF-8 bytes).
///
/// Same chunked redirect semantics as [`MSG_TYPE_STDOUT`], applied
/// to fd 2.
pub const MSG_TYPE_STDERR: u32 = 0x5345_5252; // "SERR"

/// Guest→host scheduler-log chunk (payload: opaque UTF-8 bytes).
///
/// Replaces the prior COM2 SCHED_OUTPUT_START/END dump in
/// `dump_sched_output`. The host concatenates chunks in arrival
/// order; the embedded `SCHED_OUTPUT_START` / `SCHED_OUTPUT_END`
/// markers and the BPF verifier section travel verbatim inside
/// the chunk bytes.
pub const MSG_TYPE_SCHED_LOG: u32 = 0x5343_4c47; // "SCLG"

/// Guest→host lifecycle phase event.
///
/// Payload layout: 1-byte [`LifecyclePhase`] discriminant followed
/// by an optional UTF-8 reason buffer (used by
/// `SchedulerNotAttached`'s suffix detail; empty for every other
/// phase). Replaces the COM2 `KTSTR_INIT_STARTED` /
/// `KTSTR_PAYLOAD_STARTING` / `SCHEDULER_DIED` /
/// `SCHEDULER_NOT_ATTACHED` sentinel strings.
pub const MSG_TYPE_LIFECYCLE: u32 = 0x4c49_4645; // "LIFE"

/// Guest→host kernel address parameters (payload: 16 bytes LE).
///
/// Sent BEFORE `MSG_TYPE_SYS_RDY` so the monitor has `phys_base`
/// and `page_offset_base` before its first sample iteration.
/// Payload layout: 24 bytes encoded by [`KernAddrs::to_payload`]:
///   `[phys_base + 1 : u64 LE, page_offset_base : u64 LE, kernel_text_runtime_kva + 1 : u64 LE]`
/// The guest reads these from `/proc/iomem` and `/proc/kallsyms`
/// after `mount_filesystems` — by that point `__startup_64`,
/// `kernel_randomize_memory`, `cpu_init → syscall_init`, and the
/// post-relocation kallsyms table population have all run, so the
/// values are final regardless of KASLR configuration.
pub const MSG_TYPE_KERN_ADDRS: u32 = 0x4b41_4452; // "KADR"

/// Typed payload for [`MSG_TYPE_KERN_ADDRS`].
///
/// Three u64 fields published by the guest at boot so the host can
/// translate kernel virtual addresses without walking guest page
/// tables and recover the virt-KASLR slide without a separate
/// in-VMM derivation. The wire layout uses bias-by-1 on the
/// `phys_base` and `kernel_text_runtime_kva` slots so 0 stays the
/// "not yet received / could not derive" sentinel; `page_offset_base`
/// is unbiased (today the guest always sends 0 and the host
/// re-derives via page-table walk — left in the layout for a future
/// extension that bypasses the walk).
///
/// Constructors:
///   - [`Self::new`]: bare fields, no sentinel logic. Used by
///     [`crate::vmm::guest_comms::send_kern_addrs`] on the guest
///     side.
///   - [`Self::from_payload`]: decodes a 24-byte payload, strips
///     the +1 bias on the biased slots, validates the length. Used
///     by the host dispatch arm in
///     `crate::vmm::freeze_coord::dispatch::dispatch_bulk_message`.
///
/// Field semantics:
///   - `phys_base = 0` is a legitimate KASLR-off value (the
///     payload encodes it biased as `1`, decoder strips back to
///     `0`). [`Self::has_phys_present_bit`] reports whether the
///     guest sent a non-zero biased phys_base (i.e. the payload
///     carries phys_base data at all).
///   - `kernel_text_runtime_kva` is wrapped in `Option<u64>` so
///     the decoder distinguishes "guest could not read kallsyms"
///     (`None`) from "guest read kallsyms and KASLR is off"
///     (`Some(link_kva)`). The bias-by-1 encoding handles the
///     former (biased 0 → `None`); a non-zero biased value
///     decodes to `Some(raw)`.
#[derive(Debug, Clone, Copy)]
pub struct KernAddrs {
    /// Guest-derived `phys_base` (the KASLR-physical slide), or 0
    /// when KASLR-physical is off (`__startup_64` left
    /// `phys_base = 0`). Compare with the host's expected
    /// load-address to recover the physical KASLR offset.
    pub phys_base: u64,
    /// Symbol KVA of the guest's `page_offset_base` global (NOT the
    /// runtime value the symbol points at — host dereferences via
    /// `monitor::symbols::text_kva_to_pa_with_base` + `read_u64` once
    /// it has `phys_base` resolved). Populated by
    /// `vmm::rust_init::build_kern_addrs` reading `/proc/kallsyms`.
    /// Storage class: `.data..ro_after_init` per
    /// `arch/x86/kernel/head64.c:63` — written during
    /// `kernel_randomize_memory()` in `start_kernel`, frozen after
    /// `mark_rodata_ro`. `0` means (a) arm64 (no `page_offset_base`
    /// global — `PAGE_OFFSET` is compile-time per
    /// `arch/arm64/include/asm/memory.h:43-45`), (b)
    /// CONFIG_RANDOMIZE_MEMORY=n (symbol absent), or (c) kallsyms
    /// unreadable (kptr_restrict elevated); the host falls back to
    /// `resolve_page_offset_with_tcr` (the page-table walk) in
    /// every 0 case.
    pub page_offset_base: u64,
    /// Runtime KVA of `_text` (the kernel image start symbol)
    /// from the guest's `/proc/kallsyms`, when readable. The
    /// host derives `virt_kaslr = runtime - link_text_kva` using
    /// the link-time KVA extracted from vmlinux at coordinator
    /// init. `None` when the guest could not read kallsyms
    /// (kptr_restrict masked, /proc not mountable, symbol absent).
    pub kernel_text_runtime_kva: Option<u64>,
}

impl KernAddrs {
    /// Wire-format byte length. Exact-match check on the receive
    /// side so a future payload extension trips a decoder
    /// rejection rather than silently dropping the new bytes.
    pub const WIRE_LEN: usize = 24;

    /// Construct from bare field values. Caller owns the
    /// "did I read kallsyms?" decision via the `Option` on
    /// `kernel_text_runtime_kva`.
    pub fn new(
        phys_base: u64,
        page_offset_base: u64,
        kernel_text_runtime_kva: Option<u64>,
    ) -> Self {
        Self {
            phys_base,
            page_offset_base,
            kernel_text_runtime_kva,
        }
    }

    /// Encode to a 24-byte LE payload with `+1` bias on the
    /// biased slots. Caller transmits this on the wire. Takes
    /// `self` by value since [`Self`] is `Copy` and the encoder
    /// reads each field at most once.
    pub fn to_payload(self) -> [u8; Self::WIRE_LEN] {
        let mut buf = [0u8; Self::WIRE_LEN];
        buf[..8].copy_from_slice(&(self.phys_base.wrapping_add(1)).to_le_bytes());
        buf[8..16].copy_from_slice(&self.page_offset_base.to_le_bytes());
        // bias 0 → encodes as 0 (sentinel: guest could not derive)
        let runtime_biased = match self.kernel_text_runtime_kva {
            Some(kva) => kva.wrapping_add(1),
            None => 0,
        };
        buf[16..24].copy_from_slice(&runtime_biased.to_le_bytes());
        buf
    }

    /// Decode from a wire payload. Returns `None` on length
    /// mismatch (exact match required — short payloads never
    /// publish either slot to avoid a partial-init race; longer
    /// payloads indicate a protocol extension the decoder
    /// doesn't understand).
    pub fn from_payload(payload: &[u8]) -> Option<Self> {
        if payload.len() != Self::WIRE_LEN {
            return None;
        }
        let phys_biased = u64::from_le_bytes(payload[..8].try_into().ok()?);
        let page_offset_base = u64::from_le_bytes(payload[8..16].try_into().ok()?);
        let runtime_biased = u64::from_le_bytes(payload[16..24].try_into().ok()?);
        Some(Self {
            // biased 0 means "guest didn't send" — but the
            // unbiased phys_base = 0 is legitimate (KASLR off).
            // `has_phys_present_bit` distinguishes the two on the
            // host side.
            phys_base: phys_biased.wrapping_sub(1),
            page_offset_base,
            kernel_text_runtime_kva: if runtime_biased == 0 {
                None
            } else {
                Some(runtime_biased.wrapping_sub(1))
            },
        })
    }

    /// True iff the encoded payload had a non-zero biased
    /// `phys_base` slot (i.e. the guest sent phys_base data).
    /// Distinguishes "guest sent phys_base = 0" (KASLR off, valid)
    /// from "guest didn't send phys_base at all" (truncated wire
    /// path, treat as absent). Computed from the post-decode
    /// `phys_base` field: encoded `phys_biased = phys_base + 1`
    /// is non-zero iff `phys_base != u64::MAX`. Wrap-around case
    /// (`phys_base = u64::MAX` encodes to biased 0) is impossible
    /// in practice — kernel `phys_base` is a low physical address,
    /// never the all-ones sentinel.
    pub fn has_phys_present_bit(&self) -> bool {
        self.phys_base != u64::MAX
    }
}

/// Guest→host shell-exec exit code (payload: 4-byte LE i32).
///
/// Replaces the prior COM2 `KTSTR_EXEC_EXIT=N` sentinel line
/// emitted by `cargo ktstr shell --exec <cmd>`.
pub const MSG_TYPE_EXEC_EXIT: u32 = 0x4558_4358; // "EXCX"

/// Guest→host kernel ring-buffer dump (payload: opaque UTF-8 bytes).
///
/// Sent on the initramfs-extraction failure path so the host sees
/// the kernel OOM messages without scraping COM2.
pub const MSG_TYPE_DMESG: u32 = 0x444d_5347; // "DMSG"

/// Guest→host probe-pipeline JSON output (payload: opaque UTF-8
/// bytes).
///
/// Replaces the prior COM2 ProbeDrain path so probe output and
/// scheduler-log dumps stop interleaving on the same serial port.
pub const MSG_TYPE_PROBE_OUTPUT: u32 = 0x5052_4f42; // "PROB"

/// Guest→host kernel-memory write/read op request (payload:
/// postcard-encoded [`KernelOpRequestPayload`]).
///
/// Carries an [`Op::WriteKernelHot`](crate::scenario::ops::Op::WriteKernelHot)
/// / [`Op::WriteKernelCold`](crate::scenario::ops::Op::WriteKernelCold)
/// / [`Op::ReadKernelHot`](crate::scenario::ops::Op::ReadKernelHot)
/// / [`Op::ReadKernelCold`](crate::scenario::ops::Op::ReadKernelCold)
/// request from the guest's step executor to the host coordinator.
/// Variable-length payload (target + value bytes do not fit in the
/// 72-byte [`SnapshotRequestPayload`]), so this rides a distinct
/// MSG_TYPE_* with a postcard-encoded body rather than extending the
/// fixed-size snapshot envelope.
pub const MSG_TYPE_KERNEL_OP_REQUEST: u32 = 0x4b4f_5251; // "KORQ"

/// Host→guest reply to a [`MSG_TYPE_KERNEL_OP_REQUEST`] (payload:
/// postcard-encoded [`KernelOpReplyPayload`]). Echoes the request id
/// the guest stamped, carries the status + reason + (for reads) the
/// value bytes the host coordinator read.
pub const MSG_TYPE_KERNEL_OP_REPLY: u32 = 0x4b4f_5250; // "KORP"

// ---------------------------------------------------------------------------
// ShmMessage — TLV header
// ---------------------------------------------------------------------------

/// 16-byte TLV header preceding each payload on the wire.
///
/// Used as the framing header for the bulk virtio-console port-1
/// channel; the type name `ShmMessage` is retained from the
/// predecessor SHM ring transport (now removed in favour of the
/// virtio-console port). CRC32 covers payload bytes only (not the
/// header).
///
/// SAFETY: `repr(C)` with four `u32` fields produces a 16-byte struct
/// with no padding (every field is 4-aligned). `_pad` is reserved for
/// future schema use; current writers MUST set it to 0 and current
/// readers ignore it. zerocopy derives produce no panics — every bit
/// pattern is valid for `u32`.
#[repr(C)]
#[derive(
    Clone, Copy, Default, Debug, FromBytes, IntoBytes, zerocopy::Immutable, zerocopy::KnownLayout,
)]
pub struct ShmMessage {
    pub msg_type: u32,
    pub length: u32,
    pub crc32: u32,
    pub _pad: u32,
}

const _SHM_MESSAGE_SIZE: () = assert!(std::mem::size_of::<ShmMessage>() == 16);

/// Size in bytes of the on-wire [`ShmMessage`] header.
pub const FRAME_HEADER_SIZE: usize = std::mem::size_of::<ShmMessage>();

// ---------------------------------------------------------------------------
// ShmEntry — parsed TLV entry
// ---------------------------------------------------------------------------

/// A single parsed message extracted from the bulk byte stream.
///
/// `crc_ok` is `true` when the recomputed payload CRC matched the
/// guest's stored value. CRC mismatches do not stop the walk — the
/// parser yields the entry with `crc_ok=false` and continues with the
/// next frame. Downstream consumers may filter on `crc_ok` to drop
/// corrupted entries.
#[derive(Debug, Clone)]
pub struct ShmEntry {
    pub msg_type: u32,
    pub payload: Vec<u8>,
    /// `true` when the recomputed payload CRC matched the on-wire CRC.
    pub crc_ok: bool,
}

// ---------------------------------------------------------------------------
// Stimulus payload — guest step executor → host
// ---------------------------------------------------------------------------

/// Payload for stimulus events written by the guest step executor.
///
/// Compact 24-byte struct describing the state after each step's ops
/// are applied. The host correlates these with monitor samples to map
/// scheduler telemetry to scenario phases.
#[repr(C)]
#[derive(Clone, Copy, Default, Debug, IntoBytes, zerocopy::Immutable, zerocopy::KnownLayout)]
pub struct StimulusPayload {
    /// Milliseconds since scenario start.
    pub elapsed_ms: u32,
    /// Index of the step that was just applied.
    pub step_index: u16,
    /// Number of ops applied in this step.
    pub op_count: u16,
    /// Bitmask of Op variant discriminants present in this step.
    pub op_kinds: u32,
    /// Number of live cgroups after this step: sum of step-local
    /// cgroups (from the current Step's `CgroupDef`s + `Op`s) and
    /// Backdrop-owned cgroups that persist across every Step.
    pub cgroup_count: u16,
    /// Total worker handles after this step: sum of step-local
    /// workers and Backdrop-spawned workers that persist across
    /// every Step.
    pub worker_count: u16,
    /// Sum of all workers' iteration counts at this step boundary.
    /// Read from shared MAP_SHARED counters in the step executor.
    pub total_iterations: u64,
}

const _STIMULUS_SIZE: () = assert!(std::mem::size_of::<StimulusPayload>() == 24);

/// Deserialized stimulus event.
#[derive(Debug, Clone)]
pub struct StimulusEvent {
    pub elapsed_ms: u32,
    pub step_index: u16,
    pub op_count: u16,
    pub op_kinds: u32,
    pub cgroup_count: u16,
    pub worker_count: u16,
    pub total_iterations: u64,
}

impl StimulusEvent {
    /// Deserialize from raw payload bytes. Requires EXACTLY a
    /// [`StimulusPayload`]-sized (24-byte) buffer — a shorter buffer would
    /// truncate and a longer one would carry trailing bytes the guest
    /// never frames (`send_stimulus`/`send_step_end` always write exactly
    /// 24 bytes), so both are rejected (matching [`KernAddrs::from_payload`]'s
    /// exact-length gate). A torn or hostile oversized frame is dropped
    /// rather than promoted by reading a 24-byte prefix.
    pub fn from_payload(data: &[u8]) -> Option<Self> {
        if data.len() != std::mem::size_of::<StimulusPayload>() {
            return None;
        }
        Some(StimulusEvent {
            elapsed_ms: u32::from_ne_bytes(data[0..4].try_into().ok()?),
            step_index: u16::from_ne_bytes(data[4..6].try_into().ok()?),
            op_count: u16::from_ne_bytes(data[6..8].try_into().ok()?),
            op_kinds: u32::from_ne_bytes(data[8..12].try_into().ok()?),
            cgroup_count: u16::from_ne_bytes(data[12..14].try_into().ok()?),
            worker_count: u16::from_ne_bytes(data[14..16].try_into().ok()?),
            total_iterations: u64::from_ne_bytes(data[16..24].try_into().ok()?),
        })
    }
}

/// Size in bytes of the [`MsgType::ScenarioEnd`] payload: two
/// little-endian `u64`s — scenario-relative elapsed milliseconds
/// followed by the final cumulative worker iteration count.
pub const SCENARIO_END_PAYLOAD_SIZE: usize = 16;

/// Parse the [`MsgType::ScenarioEnd`] payload written by
/// [`crate::vmm::guest_comms::send_scenario_end`]: `elapsed_ms`
/// (LE `u64`, scenario-relative) followed by `total_iterations`
/// (LE `u64`, the cumulative worker iteration count summed across
/// every live handle at the LAST step's end). The iteration count is
/// the right boundary the final step's `iteration_rate` delta needs —
/// the host folds it into a synthetic terminal
/// [`crate::timeline::StimulusEvent`] (see
/// [`crate::timeline::StimulusEvent::terminal`]). Returns `None` for a
/// short/torn payload so a CRC-bad or truncated frame is skipped
/// rather than misread.
pub fn parse_scenario_end(payload: &[u8]) -> Option<(u64, u64)> {
    if payload.len() < SCENARIO_END_PAYLOAD_SIZE {
        return None;
    }
    let elapsed_ms = u64::from_le_bytes(payload[0..8].try_into().ok()?);
    let total_iterations = u64::from_le_bytes(payload[8..16].try_into().ok()?);
    Some((elapsed_ms, total_iterations))
}

// ---------------------------------------------------------------------------
// Snapshot request/reply TLV payloads
// ---------------------------------------------------------------------------

/// Maximum length, in bytes, of a snapshot tag (capture name or
/// watchpoint symbol path) carried inside the
/// [`SnapshotRequestPayload`]. Tags longer than this bound are
/// truncated by the guest before publishing; the host treats the
/// first NUL as the boundary, or stops at this size if no NUL is
/// present.
pub const SNAPSHOT_TAG_MAX: usize = 64;

/// Maximum length, in bytes, of a host-supplied reason string carried
/// inside the [`SnapshotReplyPayload`]. Same semantics as the tag
/// buffer (NUL-terminated when shorter, truncated when longer). Sized
/// to hold typed-Err diagnostics that name the failing condition
/// (e.g. `kaslr_offset == 0`, `kern_virt_kaslr` Arc state) PLUS the
/// failing symbol + KVA PLUS the actionable remediation tip (e.g.
/// `set #[ktstr_test(kaslr = false)]`). The longest such diagnostic
/// today — Fix C's high-half/zero-offset rejection at
/// `crate::vmm::freeze_coord::snapshot::arm_user_watchpoint` — is
/// ~343 bytes when rendered with a typical symbol + KVA; 512 gives
/// ~170 bytes of headroom for future diagnostics. The original
/// 64-byte buffer and an intermediate 256-byte size both truncated
/// this message before the remediation tail.
pub const SNAPSHOT_REASON_MAX: usize = 512;

/// Snapshot request kind: no request pending. Used as the sentinel
/// value for an uninitialised request slot (this discriminant must
/// not appear on the wire — the framing of a TLV with
/// `MSG_TYPE_SNAPSHOT_REQUEST` already implies a request).
pub const SNAPSHOT_KIND_NONE: u32 = 0;

/// Snapshot request kind: capture-now. The host runs
/// `freeze_and_capture(false)` and stores the resulting
/// `FailureDumpReport` on the bridge keyed by the request tag.
pub const SNAPSHOT_KIND_CAPTURE: u32 = 1;

/// Snapshot request kind: hardware-watchpoint registration. The host
/// resolves the symbol path through the vmlinux ELF symtab,
/// allocates a free user watchpoint slot, programs the hardware
/// watchpoint via `KVM_SET_GUEST_DEBUG`, and replies. A future
/// guest write to the resolved KVA fires the corresponding debug
/// exit and synthesises a snapshot tagged by the symbol.
pub const SNAPSHOT_KIND_WATCH: u32 = 2;

/// Reply status: success — the host completed the requested action
/// (capture stored, or watchpoint armed).
pub const SNAPSHOT_STATUS_OK: u32 = 1;

/// Reply status: failure — the host rejected or could not complete
/// the request. The reason buffer carries a UTF-8 diagnostic.
pub const SNAPSHOT_STATUS_ERR: u32 = 2;

/// Outcome of a guest-driven snapshot request: ok, error with reason,
/// or transport failure (port unavailable / not in guest / timeout).
#[derive(Debug)]
pub enum SnapshotRequestResult {
    /// Host completed the request. For
    /// [`SNAPSHOT_KIND_CAPTURE`] this means the report
    /// was stored on the bridge under the supplied tag; for
    /// [`SNAPSHOT_KIND_WATCH`] this means the hardware
    /// watchpoint was armed.
    Ok,
    /// Host accepted the request but completed it as a failure. The
    /// reason carries the host-supplied diagnostic text (truncated to
    /// [`SNAPSHOT_REASON_MAX`] bytes).
    HostError { reason: String },
    /// Transport failed (called from host context, port not yet open,
    /// host did not reply within `timeout`, malformed reply frame).
    /// The supplied diagnostic names the underlying cause.
    TransportError { reason: String },
}

/// Snapshot request payload (72 bytes).
///
/// Sent guest→host as the payload of a [`MsgType::SnapshotRequest`]
/// frame on virtio-console port 1 TX. The guest fills every field
/// before publishing; the trailing zeros in `tag` form the NUL
/// terminator when the supplied tag is shorter than
/// [`SNAPSHOT_TAG_MAX`].
///
/// SAFETY: `repr(C)` with `u32 + u32 + [u8; 64]` produces a 72-byte
/// struct with no padding (every field is naturally aligned;
/// trailing array of `u8` requires no end-of-struct padding).
/// Every bit pattern is valid for `u32` and `u8`. zerocopy derives
/// produce no panics.
#[repr(C)]
#[derive(Copy, Clone, Debug, FromBytes, IntoBytes, zerocopy::Immutable, zerocopy::KnownLayout)]
pub struct SnapshotRequestPayload {
    /// Monotonic request id the guest stamped before publishing.
    /// The host echoes this value into the matching
    /// [`SnapshotReplyPayload::request_id`] so the guest's blocking
    /// reader can pair against the original request.
    pub request_id: u32,
    /// Request kind: one of [`SNAPSHOT_KIND_CAPTURE`] /
    /// [`SNAPSHOT_KIND_WATCH`]. [`SNAPSHOT_KIND_NONE`] is invalid on
    /// the wire — the host rejects it with [`SNAPSHOT_STATUS_ERR`].
    pub kind: u32,
    /// Tag — UTF-8, NUL-terminated when shorter than the buffer;
    /// truncated to [`SNAPSHOT_TAG_MAX`] when longer. For
    /// [`SNAPSHOT_KIND_CAPTURE`] the tag is the snapshot name (key
    /// the bridge stores the report under); for
    /// [`SNAPSHOT_KIND_WATCH`] the tag is the symbol path the host
    /// resolves through vmlinux ELF.
    pub tag: [u8; SNAPSHOT_TAG_MAX],
}

const _SNAPSHOT_REQUEST_PAYLOAD_SIZE: () =
    assert!(std::mem::size_of::<SnapshotRequestPayload>() == 8 + SNAPSHOT_TAG_MAX);

/// Snapshot reply payload (520 bytes: `u32 request_id + u32 status + [u8; 512] reason`).
///
/// Sent host→guest as the payload of a [`MsgType::SnapshotReply`]
/// frame on virtio-console port 1 RX. Mirrors the request layout —
/// the guest matches `request_id` against its outstanding request
/// and reads `status`/`reason` to surface the host's verdict.
///
/// SAFETY: identical layout reasoning as [`SnapshotRequestPayload`].
#[repr(C)]
#[derive(Copy, Clone, Debug, FromBytes, IntoBytes, zerocopy::Immutable, zerocopy::KnownLayout)]
pub struct SnapshotReplyPayload {
    /// Echo of the request's `request_id`. The guest's blocking
    /// reader spins until it observes this value match its
    /// outstanding request.
    pub request_id: u32,
    /// Reply status: [`SNAPSHOT_STATUS_OK`] when the host completed
    /// the request, [`SNAPSHOT_STATUS_ERR`] otherwise.
    pub status: u32,
    /// Reason — UTF-8, NUL-terminated when shorter than the buffer;
    /// truncated to [`SNAPSHOT_REASON_MAX`] when longer. Empty
    /// (all-zero) on the success path.
    pub reason: [u8; SNAPSHOT_REASON_MAX],
}

const _SNAPSHOT_REPLY_PAYLOAD_SIZE: () =
    assert!(std::mem::size_of::<SnapshotReplyPayload>() == 8 + SNAPSHOT_REASON_MAX);

// ---------------------------------------------------------------------------
// KernelOp request/reply payloads (postcard-encoded, variable-length)
// ---------------------------------------------------------------------------

/// Hot/cold orchestration discriminant for kernel-memory ops on the
/// wire. Encoded inside [`KernelOpRequestPayload`].
#[derive(Copy, Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum KernelOpMode {
    /// Hot: dispatched on a host worker thread without freeze
    /// rendezvous. Mirrors `Op::WriteKernelHot` / `Op::ReadKernelHot`
    /// orchestration. Caller is responsible for guest-side sync.
    Hot,
    /// Cold: dispatched inside a freeze rendezvous with every vCPU
    /// parked. Mirrors `Op::WriteKernelCold` / `Op::ReadKernelCold`
    /// orchestration. Coherent with respect to guest state.
    Cold,
}

/// Direction discriminant: write vs read. Inside
/// [`KernelOpRequestPayload`] the kind picks WHICH `GuestKernel::*`
/// method family the host dispatcher invokes.
#[derive(Copy, Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum KernelOpDirection {
    /// Write: `values` contains the bytes to write; reply carries
    /// success/error and the per-write byte count.
    Write,
    /// Read: `values` is empty; reply carries the bytes read into
    /// `KernelOpReplyPayload::read_values`.
    Read,
}

/// Wire-encoded [`crate::scenario::ops::KernelTarget`] variant tag.
/// Mirrors the `KernelTarget` enum variants 1:1; postcard encodes
/// the tag + the variant payload that follows in [`KernelOpTarget`].
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum KernelOpTarget {
    /// Kernel symbol (text/data/bss), resolved at dispatch via
    /// runtime kernel image base + KASLR.
    Symbol(String),
    /// Direct-mapped KVA; translated via `kva - PAGE_OFFSET`.
    Direct(u64),
    /// Vmalloc'd KVA; translated via page-table walk through CR3.
    Kva(u64),
    /// Per-CPU field of a kernel struct. Resolved at dispatch via
    /// `symbol_kva + __per_cpu_offset[cpu] + BTF byte offset of field`.
    PerCpuField {
        /// Symbol naming the per-CPU template (e.g. `"runqueues"`).
        symbol: String,
        /// Field within the symbol's struct (e.g. `"clock"`).
        field: String,
        /// CPU index whose per-CPU instance to address.
        cpu: u32,
    },
    /// Per-task field of a `struct task_struct` — SCX-managed tasks
    /// only. Resolved at dispatch time by walking `init_task.tasks`
    /// plus each leader's `signal->thread_head` to locate the
    /// `task_struct *` whose `pid` matches AND whose `start_time`
    /// matches `expected_start_time_ns` (anti-PID-reuse identity
    /// guard), then adding the BTF-resolved nested-path byte offset
    /// of `field` within `struct task_struct`.
    ///
    /// `pid` is the GUEST-side `pid_t` (positive). Both
    /// thread-group leaders AND non-leader threads are addressable:
    /// the walker iterates leaders via `for_each_process` semantics
    /// (`include/linux/sched/signal.h:639`), and for each leader
    /// also walks `leader->signal->thread_head` via
    /// `for_each_thread` semantics (same header L654-659).
    ///
    /// `expected_start_time_ns` is the value `task->start_time` had
    /// at WorkSpec spawn time. The kernel sets `start_time` once via
    /// `ktime_get_ns()` in `kernel/fork.c::copy_process`
    /// (`include/linux/sched.h:1127`); the value never changes
    /// after that. Caller records it at spawn time (e.g. via
    /// `/proc/<pid>/stat` field 22 + sysconf-to-ns conversion).
    /// The dispatcher rejects writes when the observed
    /// `task->start_time` differs — catches the PID-reuse hazard
    /// where the original worker exited and the kernel recycled
    /// the PID for an unrelated task.
    ///
    /// `field` is a dot-separated nested-member path. **SCX-only**:
    /// the dispatcher's class/policy gates accept ONLY
    /// `ext_sched_class` / `SCHED_EXT`. Recommended fields:
    /// - `"scx.dsq_vtime"` — SCX DSQ priority-queue ordering key;
    ///   preserved across dequeue/enqueue cycles
    ///   (`kernel/sched/ext.c`).
    /// - `"start_boottime"` — task fork timestamp; observable in
    ///   `/proc/<pid>/stat` field 22.
    ///
    /// **DO NOT** write `"se.vruntime"` — EEVDF's `place_entity`
    /// (`kernel/sched/fair.c:5329-5414`, since 6.6) overwrites
    /// `se->vruntime` on every enqueue via `avg_vruntime(cfs_rq) -
    /// se->vlag`. Direct vruntime writes are silently discarded for
    /// sleeping tasks (which is our validation gate). TaskField
    /// rejects non-SCX tasks before reaching this field anyway.
    ///
    /// Eight-layer task validation before any write/read lands:
    /// 1. `task->pid == requested_pid` (anti-mismatch),
    /// 2. `task->start_time == expected_start_time_ns` (anti-PID-reuse
    ///    identity),
    /// 3. `task->__state & TASK_DEAD == 0` (lifetime),
    /// 4. `task->on_rq == 0` (rb-tree / DSQ ordering safety per
    ///    `task_on_rq_queued` at `kernel/sched/sched.h:2399`),
    /// 5. `task->scx.dsq == NULL` AND `task->scx.runnable_node` is
    ///    list-empty (SCX maintains `runnable_node` linkage
    ///    independent of dsq pointer per
    ///    `include/linux/sched/ext.h:227`),
    /// 6. `task->sched_class == &ext_sched_class` (SCX-only),
    /// 7. `task->policy & ~SCHED_RESET_ON_FORK == SCHED_EXT (= 7)`
    ///    per `include/uapi/linux/sched.h:121` (belt-and-suspenders
    ///    for L6),
    /// 8. `task->start_boottime != 0` (anti-slab-recycle: a
    ///    freshly-zeroed slab page reads zero; live tasks have this
    ///    set to non-zero `ktime_get_boottime_ns()` at fork).
    TaskField {
        /// Guest-side PID of the target task. Both leaders and
        /// non-leader threads are addressable via the dispatcher's
        /// per-thread walker.
        pid: u32,
        /// `task->start_time` (`u64`, nanoseconds) recorded at
        /// WorkSpec spawn time. Used by the L2 anti-PID-reuse
        /// identity check.
        expected_start_time_ns: u64,
        /// Nested member path within `struct task_struct`. Dot-
        /// separated; first segment is a direct member of
        /// `task_struct`, subsequent segments descend through named
        /// composite members.
        field: String,
    },
}

/// Wire-encoded [`crate::scenario::ops::KernelValue`] variant tag.
/// Mirrors the four `KernelValue` enum variants 1:1.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum KernelOpValue {
    /// 32-bit unsigned, little-endian on the wire and at the
    /// resolved PA. Atomic when the resolved host PA is 4-byte
    /// aligned (see `GuestKernel::write_*_u32` doc).
    U32(u32),
    /// 64-bit unsigned, little-endian. Atomic at 8-byte alignment.
    U64(u64),
    /// Variable-length byte payload. Written non-atomically; the
    /// dispatcher emits a Release fence after the copy.
    Bytes(Vec<u8>),
    /// 32-bit unsigned read-modify-write OR mask. The cold-path
    /// dispatcher reads the live u32 at the resolved host PA,
    /// ORs the carried mask into it, and writes the new value
    /// back as two separate `read_u32` / `write_u32` calls —
    /// atomic by quiesce because the freeze rendezvous parks
    /// every guest vCPU before the RMW runs (no concurrent
    /// kernel writer can interleave). No `compare_exchange` loop
    /// in the cold path. Mirrors
    /// [`crate::scenario::ops::KernelValue::OrU32`] — see that
    /// variant's doc for the full atomicity, ordering, and
    /// width-correctness contract (the canonical
    /// `SCX_RQ_CLK_VALID` use case + the
    /// `kernel/sched/sched.h:802` u32-width citation for the
    /// `struct scx_rq.flags` field that motivated keeping the
    /// variant u32 rather than u64). Hot-path support is a
    /// future variant — it would require `AtomicU32::from_ptr`
    /// + cmpxchg + strict alignment rejection.
    OrU32(u32),
}

/// One write/read pair inside a [`KernelOpRequestPayload`] batch.
/// `value` is the bytes to write for a [`KernelOpDirection::Write`]
/// request and a placeholder ignored by the dispatcher for a
/// [`KernelOpDirection::Read`] request (the value-width discriminant
/// IS still load-bearing for reads — it picks the read method
/// family).
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct KernelOpEntry {
    /// Address to write or read.
    pub target: KernelOpTarget,
    /// Value to write (or value-width hint for a read).
    pub value: KernelOpValue,
}

/// Postcard-encoded payload for [`MsgType::KernelOpRequest`].
///
/// Carries an entire `Op::WriteKernel{Hot,Cold}` /
/// `Op::ReadKernel{Hot,Cold}` invocation including the full
/// `Vec<(KernelTarget, KernelValue)>` batch — variable-length, hence
/// the postcard encoding rather than a zerocopy fixed-size struct.
///
/// For write-direction payloads the executor's adjacent-cold-op
/// auto-merge pre-pass folds N adjacent `Op::WriteKernelCold`
/// singletons into one payload with N entries — multi-CPU seeds
/// (e.g. `with_uptime` writing per-CPU `rq.clock` on every CPU)
/// land in ONE freeze rendezvous with no inter-CPU skew. Reads
/// remain one-per-rendezvous until a follow-up batch adds
/// per-entry direction + tag.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct KernelOpRequestPayload {
    /// Monotonic request id; the host echoes it into the matching
    /// [`KernelOpReplyPayload::request_id`].
    pub request_id: u32,
    /// Hot vs cold orchestration.
    pub mode: KernelOpMode,
    /// Write vs read direction.
    pub direction: KernelOpDirection,
    /// Bridge-keyed tag for the response. For reads the tag becomes
    /// the bridge entry key; for writes the tag is informational
    /// only (the executor surfaces it in the success record).
    pub tag: String,
    /// Ordered batch entries. For [`KernelOpDirection::Write`] all
    /// entries' `value` carries the bytes to write; for
    /// [`KernelOpDirection::Read`] only `target` + the value-width
    /// discriminant are load-bearing.
    pub entries: Vec<KernelOpEntry>,
}

/// Postcard-encoded payload for `MsgType::KernelOpReply`.
///
/// Mirrors the request id so the guest's blocking reader can pair
/// against the original request. Status carries success/failure; on
/// failure `reason` describes the host-side error. For
/// `KernelOpDirection::Read` requests `read_values` carries the
/// per-entry bytes the host coordinator read; empty for writes.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct KernelOpReplyPayload {
    /// Echo of the request's `request_id`.
    pub request_id: u32,
    /// `true` when the host completed every entry in the batch;
    /// `false` when any entry failed (reason describes the first
    /// failure).
    pub success: bool,
    /// Human-readable diagnostic on the failure path; empty on
    /// success.
    pub reason: String,
    /// For a `KernelOpDirection::Read` request: one
    /// [`KernelOpValue`] per request entry in iteration order. Empty
    /// for writes.
    pub read_values: Vec<KernelOpValue>,
}

/// Upper bound on the on-wire size of a postcard-encoded
/// [`KernelOpReplyPayload`] frame the guest accepts on port-1 RX.
///
/// 1 MiB covers every realistic batch shape:
/// * `with_uptime` writing per-CPU `rq.clock` on 1024 CPUs:
///   ~9 KiB (well under cap).
/// * Bulk `read_*_bytes` of a struct page (4 KiB) per CPU on a
///   128-CPU host: ~520 KiB (within cap).
/// * Per-CPU 1 KiB `Bytes` read on 1024 CPUs: ~1 MiB (right at cap).
///
/// **Per-op entry budget**: callers that need replies larger than
/// 1 MiB must split the request across multiple ops; the cap
/// rejects forged or accidentally-huge lengths BEFORE the
/// `vec![0u8; length]` allocation in
/// [`crate::vmm::guest_comms`]'s frame reader, so a hostile or
/// buggy host cannot OOM the guest's PID 1 init.
pub const KERNEL_OP_REPLY_MAX: usize = 1024 * 1024;

/// Upper bound on [`KernelOpRequestPayload::tag`] bytes.
/// `req.tag` is a `String` and downstream formatters (the reply
/// `reason` field, tracing emits) embed it inline. Without a
/// bound, a framework bug or test-author misuse producing a
/// multi-megabyte tag would inflate the postcard-encoded reply
/// past [`KERNEL_OP_REPLY_MAX`] and the reply would silently
/// drop at the guest's RX cap, surfacing only as a 30-second
/// transport timeout. Tags longer than this cap are truncated at
/// decode time in `src/vmm/freeze_coord/dispatch.rs`'s
/// `MsgType::KernelOpRequest` arm with a UTF-8 char-boundary
/// walk-down to avoid the `String::truncate` mid-codepoint panic.
/// 256 bytes fits operator-readable test-name and scenario-phase
/// labels with headroom; framework code that benignly produces
/// longer tags loses suffix bytes from the diagnostic but the op
/// itself continues normally.
pub const KERNEL_OP_TAG_MAX: usize = 256;
/// Upper bound on [`KernelOpReplyPayload::reason`] bytes. Pairs
/// with [`KERNEL_OP_TAG_MAX`] for the reply-side bound:
/// coordinator-generated reasons embed the request tag inline and
/// otherwise format diagnostic text from typed-error payloads.
/// 256 bytes fits diagnostic messages like
/// "PA validation rejected: pa=0x... reason=wrong-half" plus the
/// request_id and the truncated tag.
pub const KERNEL_OP_REASON_MAX: usize = 256;

/// Outcome of a guest-driven kernel-memory op request: the host
/// returned a reply (caller inspects [`KernelOpReplyPayload::success`])
/// or the transport failed (port not open, timeout, malformed frame).
///
/// Distinct from a `host_error` variant the way [`SnapshotRequestResult`]
/// distinguishes — kernel-op replies are postcard-encoded with
/// arbitrary structure, so the "host completed but op failed" carrier
/// is the reply payload's `success: false` + `reason`. The
/// `TransportError` arm covers cases where the guest never receives a
/// usable reply at all.
#[derive(Debug)]
pub enum KernelOpRequestResult {
    /// Host returned a postcard-decoded reply. The caller inspects
    /// `reply.success` to distinguish op success from host-side op
    /// failure; `reply.reason` carries the failure diagnostic when
    /// `success == false`.
    Ok(KernelOpReplyPayload),
    /// Transport failed (called from host context, port not yet open,
    /// host did not reply within `timeout`, malformed reply frame).
    /// The supplied diagnostic names the underlying cause.
    TransportError { reason: String },
}

// ---------------------------------------------------------------------------
// ControlEvent — multiport control protocol discriminants
// ---------------------------------------------------------------------------

/// Multiport control-event discriminant. Mirrors the kernel uapi
/// `enum virtio_console_event` in `include/uapi/linux/virtio_console.h`.
///
/// The on-wire value is a u16. [`Self::wire_value`] returns the value
/// the kernel and the host VMM exchange on the c_ivq / c_ovq queues;
/// [`Self::from_wire`] reverses the mapping for a host-side parser.
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub enum ControlEvent {
    /// Guest-side: driver finished probing, host may begin port
    /// enumeration.
    DeviceReady,
    /// Host-side: announce a new port to the guest.
    PortAdd,
    /// Host-side: tear down a port.
    PortRemove,
    /// Guest-side: per-port driver finished setup.
    PortReady,
    /// Host-side: mark a port as the system console.
    ConsolePort,
    /// Host-side: terminal resize event.
    Resize,
    /// Bidirectional: open/close indication for a port.
    PortOpen,
    /// Host-side: PORT_NAME header followed by name bytes.
    PortName,
}

impl ControlEvent {
    /// 16-bit on-wire discriminant. Values match the kernel uapi
    /// constants `VIRTIO_CONSOLE_*`.
    pub const fn wire_value(self) -> u16 {
        match self {
            ControlEvent::DeviceReady => 0,
            ControlEvent::PortAdd => 1,
            ControlEvent::PortRemove => 2,
            ControlEvent::PortReady => 3,
            ControlEvent::ConsolePort => 4,
            ControlEvent::Resize => 5,
            ControlEvent::PortOpen => 6,
            ControlEvent::PortName => 7,
        }
    }

    /// Reverse the wire mapping. Returns `None` for unknown
    /// discriminants — the host parser is expected to log + skip such
    /// frames rather than panic.
    pub const fn from_wire(value: u16) -> Option<Self> {
        match value {
            0 => Some(ControlEvent::DeviceReady),
            1 => Some(ControlEvent::PortAdd),
            2 => Some(ControlEvent::PortRemove),
            3 => Some(ControlEvent::PortReady),
            4 => Some(ControlEvent::ConsolePort),
            5 => Some(ControlEvent::Resize),
            6 => Some(ControlEvent::PortOpen),
            7 => Some(ControlEvent::PortName),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// VirtioConsoleControl — wire-format control message
// ---------------------------------------------------------------------------

/// Wire-format control message exchanged on c_ivq / c_ovq.
///
/// Mirrors `struct virtio_console_control` in
/// `include/uapi/linux/virtio_console.h`: id (u32), event (u16),
/// value (u16). The kernel's wire format is little-endian; on the LE
/// hosts ktstr targets (x86_64, aarch64), `repr(C)` produces the
/// correct byte order via zerocopy `IntoBytes` / `FromBytes`.
///
/// SAFETY: `repr(C)` produces an 8-byte struct with no padding when
/// every field is naturally aligned (u32 at offset 0, u16 at offset
/// 4, u16 at offset 6). The `packed` qualifier is unnecessary because
/// the natural alignment matches the kernel's expected wire layout
/// and is checked by [`std::mem::size_of`] below. Every bit pattern
/// is valid for u32/u16. zerocopy derives produce no panics.
#[repr(C)]
#[derive(Copy, Clone, Debug, FromBytes, IntoBytes, zerocopy::Immutable, zerocopy::KnownLayout)]
pub struct VirtioConsoleControl {
    pub id: u32,
    pub event: u16,
    pub value: u16,
}

const _VIRTIO_CONSOLE_CONTROL_SIZE: () = assert!(std::mem::size_of::<VirtioConsoleControl>() == 8);

// ---------------------------------------------------------------------------
// Multiport device constants
// ---------------------------------------------------------------------------

/// Number of multiport ports the device exposes.
///
/// Port 0 is the kernel console (`/dev/hvc0`); port 1 is the
/// host-bound bulk TLV stream (`/dev/vport0p1`); port 2 is the
/// scheduler stats bridge (`/dev/vport0p2`) carrying raw byte
/// passthrough between the host's [`super::sched_stats::SchedStatsClient`]
/// and the guest's `scx_stats` Unix-socket relay. Three ports →
/// eight queues per virtio-v1.2 §5.3.5 (`2 + 2 * num_ports`).
pub const NUM_PORTS: u32 = 3;

/// Port-1 device-name advertised to the guest. The kernel exposes
/// this as `/sys/class/virtio-ports/vport0p1/name`; the guest init
/// reads from this path to discover the bulk channel device node.
pub const PORT1_NAME: &str = "ktstr-bulk";

/// Port-2 device-name advertised to the guest. The kernel exposes
/// this as `/sys/class/virtio-ports/vport0p2/name`; the guest init
/// reads from this path to discover the scheduler-stats relay
/// device node and connects it to the scheduler's
/// `/var/run/scx/root/stats` Unix socket.
pub const PORT2_NAME: &str = "ktstr-stats";

#[cfg(test)]
mod tests {
    use super::*;

    /// `parse_scenario_end` round-trips the two LE u64s the guest
    /// writes, and rejects a short/torn payload (returns None rather
    /// than misreading) — the host folds the parsed iteration count
    /// into the terminal StimulusEvent for the last step's rate.
    #[test]
    fn parse_scenario_end_round_trip_and_short_payload() {
        let mut payload = [0u8; SCENARIO_END_PAYLOAD_SIZE];
        payload[0..8].copy_from_slice(&12_345u64.to_le_bytes());
        payload[8..16].copy_from_slice(&987_654u64.to_le_bytes());
        assert_eq!(parse_scenario_end(&payload), Some((12_345, 987_654)));
        // A short payload (e.g. only the elapsed field) is rejected.
        assert_eq!(parse_scenario_end(&payload[..8]), None);
        assert_eq!(parse_scenario_end(&[]), None);
    }

    /// `ShmMessage` round-trips through bytes — guards against an
    /// accidental field reorder or a stray padding byte that would
    /// shift the on-wire layout for both guest writer and host
    /// reader.
    #[test]
    fn shm_message_round_trip_through_bytes() {
        let f = ShmMessage {
            msg_type: MSG_TYPE_EXIT,
            length: 4,
            crc32: 0xDEAD_BEEF,
            _pad: 0,
        };
        let bytes = f.as_bytes();
        assert_eq!(bytes.len(), FRAME_HEADER_SIZE);
        let back = ShmMessage::read_from_bytes(bytes).expect("16-byte slice deserializes");
        let msg_type = back.msg_type;
        let length = back.length;
        let crc32 = back.crc32;
        let pad = back._pad;
        assert_eq!(msg_type, MSG_TYPE_EXIT);
        assert_eq!(length, 4);
        assert_eq!(crc32, 0xDEAD_BEEF);
        assert_eq!(pad, 0);
    }

    /// Every msg_type constant is distinct — a copy/paste error
    /// that aliased two ids would silently misroute messages.
    #[test]
    fn msg_type_constants_are_unique() {
        let ids = [
            MSG_TYPE_STIMULUS,
            MSG_TYPE_STEP_END,
            MSG_TYPE_SCENARIO_START,
            MSG_TYPE_SCENARIO_END,
            MSG_TYPE_SCENARIO_PAUSE,
            MSG_TYPE_SCENARIO_RESUME,
            MSG_TYPE_EXIT,
            MSG_TYPE_TEST_RESULT,
            MSG_TYPE_SCHED_EXIT,
            MSG_TYPE_CRASH,
            MSG_TYPE_PAYLOAD_METRICS,
            MSG_TYPE_RAW_PAYLOAD_OUTPUT,
            MSG_TYPE_PROFRAW,
            MSG_TYPE_WPROF_TRACE,
            MSG_TYPE_SNAPSHOT_REQUEST,
            MSG_TYPE_SNAPSHOT_REPLY,
            MSG_TYPE_KERNEL_OP_REQUEST,
            MSG_TYPE_KERNEL_OP_REPLY,
            MSG_TYPE_SYS_RDY,
            MSG_TYPE_STDOUT,
            MSG_TYPE_STDERR,
            MSG_TYPE_SCHED_LOG,
            MSG_TYPE_LIFECYCLE,
            MSG_TYPE_EXEC_EXIT,
            MSG_TYPE_DMESG,
            MSG_TYPE_PROBE_OUTPUT,
        ];
        for (i, a) in ids.iter().enumerate() {
            for b in &ids[i + 1..] {
                assert_ne!(a, b, "duplicate MSG_TYPE id 0x{a:08x}");
            }
        }
    }

    /// Pin the on-wire byte order of `msg_type` to little-endian.
    /// The integer literal `0x4558_4954` spells `"EXIT"` in hex digits
    /// (`45`='E', `58`='X', `49`='I', `54`='T'), but the LE encoding
    /// places the least-significant byte first — so a raw byte dump
    /// of a serialized `ShmMessage` shows `[0x54, 0x49, 0x58, 0x45]`,
    /// which spells `"TIXE"` byte-by-byte. A future change that
    /// flipped the host to big-endian or switched zerocopy's
    /// serialization order would silently break the wire contract
    /// with the kernel virtio_console driver and every existing
    /// guest writer; this test fails loudly instead.
    #[test]
    fn msg_type_exit_wire_bytes_are_le() {
        let f = ShmMessage {
            msg_type: MSG_TYPE_EXIT,
            length: 0,
            crc32: 0,
            _pad: 0,
        };
        let bytes = f.as_bytes();
        // First 4 bytes of the header are msg_type as a u32 LE.
        assert_eq!(&bytes[..4], &MSG_TYPE_EXIT.to_le_bytes());
        // Spell-out check: the LE byte sequence is "TIXE", not "EXIT".
        // If the wire ever flips to BE, this assertion fails before the
        // guest driver sees the malformed frame.
        assert_eq!(&bytes[..4], b"TIXE");
    }

    /// `ShmMessage` header is exactly 16 bytes with no padding.
    #[test]
    fn shm_message_size_is_16() {
        assert_eq!(FRAME_HEADER_SIZE, 16);
        assert_eq!(std::mem::size_of::<ShmMessage>(), 16);
    }

    /// Every [`MsgType`] variant round-trips through
    /// `wire_value` → `from_wire`.
    #[test]
    fn msg_type_round_trips() {
        let all = [
            MsgType::Stimulus,
            MsgType::StepEnd,
            MsgType::ScenarioStart,
            MsgType::ScenarioEnd,
            MsgType::ScenarioPause,
            MsgType::ScenarioResume,
            MsgType::Exit,
            MsgType::TestResult,
            MsgType::SchedExit,
            MsgType::Crash,
            MsgType::PayloadMetrics,
            MsgType::RawPayloadOutput,
            MsgType::Profraw,
            MsgType::WprofTrace,
            MsgType::SnapshotRequest,
            MsgType::SnapshotReply,
            MsgType::KernelOpRequest,
            MsgType::KernelOpReply,
            MsgType::SysRdy,
            MsgType::Stdout,
            MsgType::Stderr,
            MsgType::SchedLog,
            MsgType::Lifecycle,
            MsgType::ExecExit,
            MsgType::Dmesg,
            MsgType::ProbeOutput,
        ];
        for variant in all {
            let v = variant.wire_value();
            assert_eq!(MsgType::from_wire(v), Some(variant));
        }
    }

    /// `MsgType::from_wire` returns `None` for an unrecognised
    /// discriminant — the bulk parser must surface unknown tags as
    /// errors rather than treat them as a known variant.
    #[test]
    fn msg_type_from_wire_unknown_returns_none() {
        assert_eq!(MsgType::from_wire(0xDEAD_BEEF), None);
        assert_eq!(MsgType::from_wire(0), None);
    }

    /// `MsgType::wire_value` matches the corresponding `MSG_TYPE_*`
    /// constant — guards against a typo that would diverge the typed
    /// API from the on-wire constant.
    #[test]
    fn msg_type_wire_value_matches_constants() {
        assert_eq!(MsgType::Stimulus.wire_value(), MSG_TYPE_STIMULUS);
        assert_eq!(MsgType::StepEnd.wire_value(), MSG_TYPE_STEP_END);
        assert_eq!(MsgType::ScenarioStart.wire_value(), MSG_TYPE_SCENARIO_START);
        assert_eq!(MsgType::ScenarioPause.wire_value(), MSG_TYPE_SCENARIO_PAUSE);
        assert_eq!(
            MsgType::ScenarioResume.wire_value(),
            MSG_TYPE_SCENARIO_RESUME
        );
        assert_eq!(MsgType::ScenarioEnd.wire_value(), MSG_TYPE_SCENARIO_END);
        assert_eq!(MsgType::Exit.wire_value(), MSG_TYPE_EXIT);
        assert_eq!(MsgType::TestResult.wire_value(), MSG_TYPE_TEST_RESULT);
        assert_eq!(MsgType::SchedExit.wire_value(), MSG_TYPE_SCHED_EXIT);
        assert_eq!(MsgType::Crash.wire_value(), MSG_TYPE_CRASH);
        assert_eq!(
            MsgType::PayloadMetrics.wire_value(),
            MSG_TYPE_PAYLOAD_METRICS
        );
        assert_eq!(
            MsgType::RawPayloadOutput.wire_value(),
            MSG_TYPE_RAW_PAYLOAD_OUTPUT
        );
        assert_eq!(MsgType::Profraw.wire_value(), MSG_TYPE_PROFRAW);
        assert_eq!(MsgType::WprofTrace.wire_value(), MSG_TYPE_WPROF_TRACE);
        assert_eq!(
            MsgType::SnapshotRequest.wire_value(),
            MSG_TYPE_SNAPSHOT_REQUEST
        );
        assert_eq!(MsgType::SnapshotReply.wire_value(), MSG_TYPE_SNAPSHOT_REPLY);
        assert_eq!(
            MsgType::KernelOpRequest.wire_value(),
            MSG_TYPE_KERNEL_OP_REQUEST
        );
        assert_eq!(
            MsgType::KernelOpReply.wire_value(),
            MSG_TYPE_KERNEL_OP_REPLY
        );
        assert_eq!(MsgType::SysRdy.wire_value(), MSG_TYPE_SYS_RDY);
        assert_eq!(MsgType::Stdout.wire_value(), MSG_TYPE_STDOUT);
        assert_eq!(MsgType::Stderr.wire_value(), MSG_TYPE_STDERR);
        assert_eq!(MsgType::SchedLog.wire_value(), MSG_TYPE_SCHED_LOG);
        assert_eq!(MsgType::Lifecycle.wire_value(), MSG_TYPE_LIFECYCLE);
        assert_eq!(MsgType::ExecExit.wire_value(), MSG_TYPE_EXEC_EXIT);
        assert_eq!(MsgType::Dmesg.wire_value(), MSG_TYPE_DMESG);
        assert_eq!(MsgType::ProbeOutput.wire_value(), MSG_TYPE_PROBE_OUTPUT);
    }

    /// `is_coordinator_internal` flips on for SnapshotRequest,
    /// SnapshotReply, KernelOpRequest, KernelOpReply, and SysRdy
    /// and stays off for every test-verdict-bearing variant. The
    /// Reply variants are host→guest only on port-1 RX; a guest TX
    /// frame stamped with one of those tags is illegitimate and
    /// must be dropped rather than bucketed as a phantom verdict
    /// entry. Pinning this matrix here means a future contributor
    /// adding a new control frame must explicitly opt into the
    /// gate (or explicitly opt out by adding a "verdict-bearing"
    /// entry to the test) — the freeze coord's mid-run filter and
    /// `collect_results`'s post-run drain both key on this single
    /// classifier (search for `is_coordinator_internal` in
    /// `crate::vmm::freeze_coord`).
    #[test]
    fn is_coordinator_internal_matches_filter_set() {
        let internal = [
            MsgType::SnapshotRequest,
            MsgType::SnapshotReply,
            MsgType::KernelOpRequest,
            MsgType::KernelOpReply,
            MsgType::SysRdy,
        ];
        let verdict = [
            MsgType::Stimulus,
            MsgType::StepEnd,
            MsgType::ScenarioStart,
            MsgType::ScenarioEnd,
            MsgType::ScenarioPause,
            MsgType::ScenarioResume,
            MsgType::Exit,
            MsgType::TestResult,
            MsgType::SchedExit,
            MsgType::Crash,
            MsgType::PayloadMetrics,
            MsgType::RawPayloadOutput,
            MsgType::Profraw,
            MsgType::WprofTrace,
            MsgType::Stdout,
            MsgType::Stderr,
            MsgType::SchedLog,
            MsgType::Lifecycle,
            MsgType::ExecExit,
            MsgType::Dmesg,
            MsgType::ProbeOutput,
        ];
        for v in internal {
            assert!(
                v.is_coordinator_internal(),
                "{v:?} must be classified as coordinator-internal"
            );
        }
        for v in verdict {
            assert!(
                !v.is_coordinator_internal(),
                "{v:?} carries test verdict data and must NOT be filtered out"
            );
        }
    }

    /// `LifecyclePhase` round-trips through `wire_value` →
    /// `from_wire`. Phase values are byte-stable across builds so
    /// the host never silently misclassifies a future guest's
    /// phase signal.
    #[test]
    fn lifecycle_phase_round_trips() {
        let all = [
            LifecyclePhase::InitStarted,
            LifecyclePhase::PayloadStarting,
            LifecyclePhase::SchedulerDied,
            LifecyclePhase::SchedulerNotAttached,
        ];
        for p in all {
            let v = p.wire_value();
            assert_eq!(LifecyclePhase::from_wire(v), Some(p));
        }
    }

    /// `LifecyclePhase::from_wire(0)` returns `None` — `0` is
    /// reserved as the unknown / invalid sentinel so a
    /// zero-initialised payload byte never silently maps to
    /// `InitStarted`.
    #[test]
    fn lifecycle_phase_zero_is_reserved() {
        assert_eq!(LifecyclePhase::from_wire(0), None);
        assert_eq!(LifecyclePhase::from_wire(0xFF), None);
    }

    /// Pin the `LifecyclePhase` discriminants. Wire values are part
    /// of the protocol contract — a future change that reorders
    /// the enum variants would silently shift this mapping unless
    /// pinned by an explicit assertion here.
    #[test]
    fn lifecycle_phase_wire_values_are_stable() {
        assert_eq!(LifecyclePhase::InitStarted.wire_value(), 1);
        assert_eq!(LifecyclePhase::PayloadStarting.wire_value(), 2);
        assert_eq!(LifecyclePhase::SchedulerDied.wire_value(), 3);
        assert_eq!(LifecyclePhase::SchedulerNotAttached.wire_value(), 4);
    }

    /// `SnapshotRequestPayload` round-trips through bytes — guards
    /// against an accidental field reorder or a stray padding byte
    /// that would shift the on-wire layout for both guest writer
    /// and host parser.
    #[test]
    fn snapshot_request_payload_round_trip_through_bytes() {
        let mut tag = [0u8; SNAPSHOT_TAG_MAX];
        tag[..6].copy_from_slice(b"hello!");
        let p = SnapshotRequestPayload {
            request_id: 0xDEAD_BEEF,
            kind: SNAPSHOT_KIND_CAPTURE,
            tag,
        };
        let bytes = p.as_bytes();
        assert_eq!(bytes.len(), 8 + SNAPSHOT_TAG_MAX);
        let back = SnapshotRequestPayload::read_from_bytes(bytes).expect("payload deserializes");
        let request_id = back.request_id;
        let kind = back.kind;
        assert_eq!(request_id, 0xDEAD_BEEF);
        assert_eq!(kind, SNAPSHOT_KIND_CAPTURE);
        assert_eq!(&back.tag[..6], b"hello!");
    }

    /// `SnapshotReplyPayload` round-trips through bytes.
    #[test]
    fn snapshot_reply_payload_round_trip_through_bytes() {
        let mut reason = [0u8; SNAPSHOT_REASON_MAX];
        reason[..4].copy_from_slice(b"oops");
        let p = SnapshotReplyPayload {
            request_id: 0xCAFE_BABE,
            status: SNAPSHOT_STATUS_ERR,
            reason,
        };
        let bytes = p.as_bytes();
        assert_eq!(bytes.len(), 8 + SNAPSHOT_REASON_MAX);
        let back = SnapshotReplyPayload::read_from_bytes(bytes).expect("payload deserializes");
        let request_id = back.request_id;
        let status = back.status;
        assert_eq!(request_id, 0xCAFE_BABE);
        assert_eq!(status, SNAPSHOT_STATUS_ERR);
        assert_eq!(&back.reason[..4], b"oops");
    }

    /// Snapshot kind constants are distinct.
    #[test]
    fn snapshot_kind_constants_are_unique() {
        assert_ne!(SNAPSHOT_KIND_NONE, SNAPSHOT_KIND_CAPTURE);
        assert_ne!(SNAPSHOT_KIND_NONE, SNAPSHOT_KIND_WATCH);
        assert_ne!(SNAPSHOT_KIND_CAPTURE, SNAPSHOT_KIND_WATCH);
    }

    /// Snapshot status constants are distinct.
    #[test]
    fn snapshot_status_constants_are_unique() {
        assert_ne!(SNAPSHOT_STATUS_OK, SNAPSHOT_STATUS_ERR);
    }

    /// Every [`ControlEvent`] variant round-trips through
    /// `wire_value` → `from_wire`.
    #[test]
    fn control_event_round_trips() {
        let all = [
            ControlEvent::DeviceReady,
            ControlEvent::PortAdd,
            ControlEvent::PortRemove,
            ControlEvent::PortReady,
            ControlEvent::ConsolePort,
            ControlEvent::Resize,
            ControlEvent::PortOpen,
            ControlEvent::PortName,
        ];
        for variant in all {
            let v = variant.wire_value();
            assert_eq!(ControlEvent::from_wire(v), Some(variant));
        }
    }

    /// `ControlEvent::from_wire` returns `None` for unknown values.
    #[test]
    fn control_event_from_wire_unknown_returns_none() {
        assert_eq!(ControlEvent::from_wire(8), None);
        assert_eq!(ControlEvent::from_wire(0xFFFF), None);
    }

    /// `ControlEvent` discriminants match the kernel uapi numbers
    /// (`VIRTIO_CONSOLE_*` in `include/uapi/linux/virtio_console.h`).
    #[test]
    fn control_event_discriminants_match_uapi() {
        assert_eq!(ControlEvent::DeviceReady.wire_value(), 0);
        assert_eq!(ControlEvent::PortAdd.wire_value(), 1);
        assert_eq!(ControlEvent::PortRemove.wire_value(), 2);
        assert_eq!(ControlEvent::PortReady.wire_value(), 3);
        assert_eq!(ControlEvent::ConsolePort.wire_value(), 4);
        assert_eq!(ControlEvent::Resize.wire_value(), 5);
        assert_eq!(ControlEvent::PortOpen.wire_value(), 6);
        assert_eq!(ControlEvent::PortName.wire_value(), 7);
    }

    /// `VirtioConsoleControl` is exactly 8 bytes — matches the
    /// kernel uapi struct.
    #[test]
    fn virtio_console_control_size_is_8() {
        assert_eq!(std::mem::size_of::<VirtioConsoleControl>(), 8);
    }

    /// `VirtioConsoleControl` round-trips through bytes — pins the
    /// repr(C) layout against an accidental field reorder that would
    /// produce malformed control frames on the c_ivq / c_ovq queues.
    #[test]
    fn virtio_console_control_round_trip() {
        let c = VirtioConsoleControl {
            id: 1,
            event: ControlEvent::PortOpen.wire_value(),
            value: 1,
        };
        let bytes = c.as_bytes();
        assert_eq!(bytes.len(), 8);
        let back = VirtioConsoleControl::read_from_bytes(bytes).unwrap();
        let id = back.id;
        let event = back.event;
        let value = back.value;
        assert_eq!(id, 1);
        assert_eq!(event, ControlEvent::PortOpen.wire_value());
        assert_eq!(value, 1);
    }

    /// `KernelOpRequestPayload` round-trips through postcard with
    /// every `KernelOpTarget` + `KernelOpValue` variant present —
    /// pins encode/decode against an accidental serde derive
    /// breakage on either side. The wire format the freeze coord
    /// (host) decodes is exactly what the guest's
    /// [`crate::vmm::guest_comms::request_kernel_op`] encodes, so a
    /// round-trip mismatch surfaces as a silent host-side parse
    /// failure rather than a typed error.
    #[test]
    fn kernel_op_request_payload_postcard_round_trip() {
        let payload = KernelOpRequestPayload {
            request_id: 0xCAFEBABE,
            mode: KernelOpMode::Cold,
            direction: KernelOpDirection::Write,
            tag: "with_uptime".into(),
            entries: vec![
                KernelOpEntry {
                    target: KernelOpTarget::Symbol("jiffies".into()),
                    value: KernelOpValue::U64(42),
                },
                KernelOpEntry {
                    target: KernelOpTarget::Direct(0xffff_8000_0000_1000),
                    value: KernelOpValue::U32(7),
                },
                KernelOpEntry {
                    target: KernelOpTarget::Kva(0xffff_c000_dead_beef),
                    value: KernelOpValue::Bytes(vec![1, 2, 3, 4, 5]),
                },
                KernelOpEntry {
                    target: KernelOpTarget::PerCpuField {
                        symbol: "runqueues".into(),
                        field: "clock".into(),
                        cpu: 3,
                    },
                    value: KernelOpValue::U64(0xDEAD_BEEF_CAFE_F00D),
                },
                KernelOpEntry {
                    target: KernelOpTarget::TaskField {
                        pid: 12345,
                        expected_start_time_ns: 1_700_000_000_000,
                        field: "scx.dsq_vtime".into(),
                    },
                    value: KernelOpValue::U64(30 * 86400 * 1_000_000_000),
                },
            ],
        };
        let bytes = postcard::to_allocvec(&payload).expect("encode");
        let back: KernelOpRequestPayload = postcard::from_bytes(&bytes).expect("decode");
        assert_eq!(back, payload);
    }

    /// `KernelOpReplyPayload` round-trips through postcard. The
    /// reply carries success/failure + (for reads) the per-entry
    /// values the host coordinator read — both code paths must
    /// survive encode/decode unchanged.
    #[test]
    fn kernel_op_reply_payload_postcard_round_trip() {
        let success = KernelOpReplyPayload {
            request_id: 0x1234_5678,
            success: true,
            reason: String::new(),
            read_values: vec![
                KernelOpValue::U64(100),
                KernelOpValue::U32(200),
                KernelOpValue::Bytes(vec![0xAB, 0xCD, 0xEF]),
            ],
        };
        let bytes = postcard::to_allocvec(&success).expect("encode success");
        let back: KernelOpReplyPayload = postcard::from_bytes(&bytes).expect("decode success");
        assert_eq!(back, success);

        let failure = KernelOpReplyPayload {
            request_id: 0xFEED_FACE,
            success: false,
            reason: "host: symbol 'jiffies' not found in vmlinux".into(),
            read_values: vec![],
        };
        let bytes = postcard::to_allocvec(&failure).expect("encode failure");
        let back: KernelOpReplyPayload = postcard::from_bytes(&bytes).expect("decode failure");
        assert_eq!(back, failure);
    }

    /// `KERNEL_OP_REPLY_MAX` envelope check: a representative
    /// large reply (1024-CPU per-CPU u64) fits comfortably; the
    /// cap is 1 MiB which bounds OOM exposure while accommodating
    /// realistic batch shapes. A regression that shrunk the cap
    /// below ~10 KiB would silently truncate large kernel-op
    /// replies; one that grew it beyond 1 MiB would widen the
    /// OOM-attack surface.
    #[test]
    fn kernel_op_reply_max_envelope_check() {
        // 1024 CPUs * KernelOpValue::U64 (~9 bytes each + bookkeeping)
        // is well under 1 MiB. Build a representative reply and
        // verify its encoded size sits inside the cap.
        let big = KernelOpReplyPayload {
            request_id: 1,
            success: true,
            reason: String::new(),
            read_values: (0..1024u64).map(KernelOpValue::U64).collect(),
        };
        let bytes = postcard::to_allocvec(&big).expect("encode 1024-CPU reply");
        assert!(
            bytes.len() < KERNEL_OP_REPLY_MAX,
            "1024-CPU kernel-op reply ({} bytes) must fit under \
             KERNEL_OP_REPLY_MAX ({KERNEL_OP_REPLY_MAX} bytes)",
            bytes.len(),
        );
        // The cap is exactly 1 MiB — large enough for per-CPU 1 KiB
        // Bytes reads on 1024 CPUs, small enough to keep OOM
        // exposure bounded for a forged frame.
        assert_eq!(KERNEL_OP_REPLY_MAX, 1024 * 1024);
    }

    // ----- KernAddrs wire-format pins -----
    //
    // Pin both the typed encode/decode contract AND the on-wire
    // byte layout. The byte-layout pin (last test) catches slot-
    // swap regressions that a roundtrip-equality test alone would
    // miss when encoder and decoder both flip the same way.

    #[test]
    fn kern_addrs_roundtrip_all_present() {
        let a = KernAddrs::new(
            0x12345678u64,
            0xffff_8880_0000_0000u64,
            Some(0xffff_ffff_8200_0000u64),
        );
        let payload = a.to_payload();
        assert_eq!(payload.len(), KernAddrs::WIRE_LEN);
        let b = KernAddrs::from_payload(&payload).expect("decode");
        assert_eq!(b.phys_base, a.phys_base);
        assert_eq!(b.page_offset_base, a.page_offset_base);
        assert_eq!(b.kernel_text_runtime_kva, a.kernel_text_runtime_kva);
        assert!(b.has_phys_present_bit());
    }

    #[test]
    fn kern_addrs_roundtrip_kallsyms_absent() {
        // The None branch on kernel_text_runtime_kva must decode
        // back to None (NOT Some(u64::MAX) via wrapping_sub(1) on
        // a raw-0 biased slot). The biased-0 sentinel is the
        // wire-format "guest could not derive" marker.
        let a = KernAddrs::new(0u64, 0u64, None);
        let payload = a.to_payload();
        let b = KernAddrs::from_payload(&payload).expect("decode");
        assert_eq!(
            b.kernel_text_runtime_kva, None,
            "biased-0 runtime slot must decode to None"
        );
        // phys_base = 0 IS a valid KASLR-off value; the biased
        // encoder writes 1 (non-zero) so has_phys_present_bit
        // surfaces present.
        assert!(b.has_phys_present_bit());
    }

    #[test]
    fn kern_addrs_from_payload_rejects_length_mismatch() {
        // Exact-length match required so a protocol-extension
        // partial write or truncated wire surfaces as None,
        // never as a zero-padded silent decode.
        assert!(KernAddrs::from_payload(&[]).is_none());
        assert!(KernAddrs::from_payload(&[0u8; KernAddrs::WIRE_LEN - 1]).is_none());
        assert!(KernAddrs::from_payload(&[0u8; KernAddrs::WIRE_LEN + 1]).is_none());
    }

    #[test]
    fn stimulus_from_payload_requires_exact_24_bytes() {
        // Exact-length match (24 bytes): an undersized buffer would
        // truncate and an oversized one carries bytes the guest never
        // frames (send_stimulus/send_step_end write exactly 24), so a
        // torn / hostile frame is dropped, not promoted by a prefix read.
        let n = std::mem::size_of::<StimulusPayload>();
        assert_eq!(n, 24);
        assert!(StimulusEvent::from_payload(&[0u8; 23]).is_none());
        assert!(StimulusEvent::from_payload(&[0u8; 25]).is_none());
        assert!(StimulusEvent::from_payload(&[0u8; 24]).is_some());
    }

    #[test]
    fn kern_addrs_has_phys_present_bit_distinguishes_zero_vs_absent() {
        // Pins the bias-sentinel contract on the present-bit
        // accessor. A struct constructed via KernAddrs::new with
        // phys_base=0 surfaces as present (encoded biased = 1).
        // A hand-decoded all-zero payload (the "guest never
        // sent" wire state) surfaces as absent (raw biased 0 →
        // wrapping_sub(1) = u64::MAX → has_phys_present_bit
        // returns false).
        let present = KernAddrs::new(0u64, 0u64, None);
        assert!(present.has_phys_present_bit());
        let absent = KernAddrs::from_payload(&[0u8; KernAddrs::WIRE_LEN])
            .expect("zero-length payload decodes; shape is valid");
        assert!(
            !absent.has_phys_present_bit(),
            "zero-bias slot decodes to u64::MAX; has_phys_present_bit must surface absent"
        );
    }

    #[test]
    fn kern_addrs_to_payload_byte_layout_is_le_phys_first() {
        // Pin the on-wire byte layout directly via slot offsets.
        // A reordering refactor that swapped slots would silently
        // pass the roundtrip-equality tests above (encoder and
        // decoder would flip together) — this test catches the
        // slot-swap class by asserting against fixed byte
        // positions.
        let a = KernAddrs::new(
            0x1111_2222_3333_4444u64,
            0xaaaa_bbbb_cccc_ddddu64,
            Some(0x5555_6666_7777_8888u64),
        );
        let p = a.to_payload();
        // phys_base biased: ...4444 + 1 = ...4445; LE first byte = 0x45.
        assert_eq!(p[0], 0x45, "phys_base slot is [0..8] LE biased");
        assert_eq!(
            u64::from_le_bytes(p[..8].try_into().unwrap()),
            0x1111_2222_3333_4445
        );
        assert_eq!(
            u64::from_le_bytes(p[8..16].try_into().unwrap()),
            0xaaaa_bbbb_cccc_ddddu64
        );
        // kernel_text_runtime_kva biased: ...8888 + 1 = ...8889.
        assert_eq!(
            u64::from_le_bytes(p[16..24].try_into().unwrap()),
            0x5555_6666_7777_8889u64
        );
    }

    #[test]
    fn kern_addrs_u64_max_runtime_collapses_to_absent_roundtrip() {
        // Documents the bias-encoding boundary collision: a
        // `kernel_text_runtime_kva` of `u64::MAX` wraps to biased 0
        // on encode, which the decoder reads as the "guest could
        // not derive" sentinel (None). u64::MAX is non-canonical
        // and impossible as a real `_text` KVA, AND the
        // downstream KERN_ADDRS dispatch arm in dispatch.rs
        // triple-gates derived offsets (kernel-half threshold +
        // non-negative slide + ≤1GiB max-slide-bound + link
        // canonical) which catch any synthesized variant of this
        // collision before it reaches the shared Arc. Test pins
        // the symmetric collapse so a future encoder refactor
        // that broke the "absent sentinel = biased 0" contract
        // (e.g. switched to a different sentinel value) trips
        // here loudly.
        let max_runtime = KernAddrs::new(0u64, 0u64, Some(u64::MAX));
        let payload = max_runtime.to_payload();
        // Biased slot reads 0 — collides with the absent encoding.
        assert_eq!(
            u64::from_le_bytes(payload[16..24].try_into().unwrap()),
            0,
            "Some(u64::MAX) biased-add wraps to 0; collides with absent sentinel"
        );
        // Roundtrip surfaces it as None, not Some(u64::MAX).
        let decoded = KernAddrs::from_payload(&payload).expect("decode");
        assert_eq!(
            decoded.kernel_text_runtime_kva, None,
            "u64::MAX runtime decodes to None via the bias collision"
        );
    }
}
