//! BPF verifier log parsing, cycle detection, and output formatting.
//!
//! Provides:
//! - [`VerifierStats`] / [`ProgStats`] / [`DiffRow`] — data types
//! - [`collect_verifier_output`] — boot VM, collect stats via host introspection
//! - [`format_verifier_output`] / [`format_verifier_diff`] — text formatting
//! - [`extract_verifier_log`] — extract verifier trace from libbpf log blob
//! - [`parse_verifier_stats`] — extract insn/state counts from verifier log
//! - [`normalize_verifier_line`] — strip variable register state annotations
//! - [`detect_cycle`] / [`collapse_cycles`] — loop iteration compression
//! - [`build_b_map`] / [`build_diff_rows`] — A/B comparison helpers
//! - `SCHED_OUTPUT_START` / `SCHED_OUTPUT_END` — delimiters the
//!   guest's rust_init emits over the bulk port (as `MSG_TYPE_SCHED_LOG`
//!   frames) around the scheduler log region;
//!   `parse_sched_output` extracts the enclosed block

use std::collections::HashMap;
use std::ffi::OsStr;
use std::path::{Path, PathBuf};

/// Current schema version for [`VerifierCellOwnershipManifest`].
pub const VERIFIER_CELL_OWNERSHIP_MANIFEST_VERSION: u32 = 1;

/// One parent-elected verifier-cell owner per selected scheduler identity.
///
/// Test binaries are the unit nextest lists and executes. When two warmed
/// binaries link the same exact `declare_scheduler!`, both would otherwise
/// advertise the same `verifier/<scheduler>/<kernel>/<topology>` name and
/// race to write its deterministic result record. The parent writes this
/// manifest after discovery; children use it at both listing and exact
/// dispatch boundaries.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VerifierCellOwnershipManifest {
    /// Wire schema version.
    pub version: u32,
    /// Full scheduler identity to canonical test-executable mapping.
    pub entries: Vec<VerifierCellOwnershipEntry>,
}

/// One selected scheduler declaration and the sole test executable allowed to
/// list, launch, and record its verifier cells.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VerifierCellOwnershipEntry {
    /// The complete declaration identity, not merely its cell-name component.
    pub scheduler: crate::test_support::SchedulerJson,
    /// Canonical absolute path of the elected warmed test executable.
    pub executable: PathBuf,
}

/// Validated ownership view for the current test executable.
///
/// Construction validates the entire manifest before any scheduler lookup, so
/// an ambiguous or partially corrupt manifest cannot selectively fall back to
/// duplicate cell emission.
pub(crate) struct VerifierCellOwnership {
    current_executable: PathBuf,
    entries: HashMap<String, VerifierCellOwnershipEntry>,
}

impl VerifierCellOwnership {
    fn entry_for_scheduler(
        &self,
        scheduler_name: &str,
    ) -> anyhow::Result<&VerifierCellOwnershipEntry> {
        self.entries.get(scheduler_name).ok_or_else(|| {
            anyhow::anyhow!(
                "verifier cell ownership manifest has no entry for selected scheduler {:?}",
                scheduler_name,
            )
        })
    }

    /// Whether this process is the sole owner of `scheduler`.
    ///
    /// A present manifest must contain every scheduler which reaches the
    /// child's normal emission gates, and its full declaration must match the
    /// parent-observed identity. Missing or stale entries are therefore hard
    /// errors rather than implicit ownership.
    pub(crate) fn owns_scheduler(
        &self,
        scheduler: &crate::test_support::SchedulerJson,
    ) -> anyhow::Result<bool> {
        let entry = self.entry_for_scheduler(&scheduler.name)?;
        if !entry.scheduler.eq(scheduler) {
            anyhow::bail!(
                "verifier cell ownership declaration mismatch for scheduler {:?}: \
                 parent recorded {:?}, child linked {:?}",
                scheduler.name,
                entry.scheduler,
                scheduler,
            );
        }
        Ok(entry.executable.eq(&self.current_executable))
    }

    /// Return the parent-selected full declaration when this process owns it.
    ///
    /// Exact dispatch uses this identity to select the matching local
    /// registration, rather than looking up only the scheduler name and
    /// accidentally executing an earlier same-name declaration which the
    /// parent correctly filtered as non-emitting.
    pub(crate) fn owned_scheduler(
        &self,
        scheduler_name: &str,
    ) -> anyhow::Result<Option<&crate::test_support::SchedulerJson>> {
        let entry = self.entry_for_scheduler(scheduler_name)?;
        Ok(entry
            .executable
            .eq(&self.current_executable)
            .then_some(&entry.scheduler))
    }
}

fn cell_ownership_from_manifest_path(
    manifest_path: Option<&OsStr>,
    current_executable: &Path,
) -> anyhow::Result<Option<VerifierCellOwnership>> {
    let Some(manifest_path) = manifest_path else {
        return Ok(None);
    };
    if manifest_path.is_empty() {
        anyhow::bail!(
            "{} is set but empty",
            crate::KTSTR_VERIFIER_CELL_OWNERSHIP_MANIFEST_ENV
        );
    }
    let manifest_path = PathBuf::from(manifest_path);
    let bytes = std::fs::read(&manifest_path).map_err(|error| {
        anyhow::anyhow!(
            "read verifier cell ownership manifest {}: {error}",
            manifest_path.display()
        )
    })?;
    let manifest: VerifierCellOwnershipManifest =
        serde_json::from_slice(&bytes).map_err(|error| {
            anyhow::anyhow!(
                "parse verifier cell ownership manifest {}: {error}",
                manifest_path.display()
            )
        })?;
    if manifest.version != VERIFIER_CELL_OWNERSHIP_MANIFEST_VERSION {
        anyhow::bail!(
            "unsupported verifier cell ownership manifest version {} in {} (expected {})",
            manifest.version,
            manifest_path.display(),
            VERIFIER_CELL_OWNERSHIP_MANIFEST_VERSION,
        );
    }

    let current_executable = std::fs::canonicalize(current_executable).map_err(|error| {
        anyhow::anyhow!(
            "canonicalize current verifier test executable {}: {error}",
            current_executable.display()
        )
    })?;
    let mut entries = HashMap::with_capacity(manifest.entries.len());
    for entry in manifest.entries {
        if !entry.executable.is_absolute() {
            anyhow::bail!(
                "verifier cell owner for scheduler {:?} is not absolute: {}",
                entry.scheduler.name,
                entry.executable.display(),
            );
        }
        let canonical_owner = std::fs::canonicalize(&entry.executable).map_err(|error| {
            anyhow::anyhow!(
                "canonicalize verifier cell owner {} for scheduler {:?}: {error}",
                entry.executable.display(),
                entry.scheduler.name,
            )
        })?;
        if canonical_owner.as_path() != entry.executable.as_path() {
            anyhow::bail!(
                "verifier cell owner for scheduler {:?} is not canonical: {} resolves to {}",
                entry.scheduler.name,
                entry.executable.display(),
                canonical_owner.display(),
            );
        }
        let name = entry.scheduler.name.clone();
        if entries.insert(name.clone(), entry).is_some() {
            anyhow::bail!(
                "duplicate verifier cell ownership manifest entry for scheduler {name:?}"
            );
        }
    }
    Ok(Some(VerifierCellOwnership {
        current_executable,
        entries,
    }))
}

/// Read and validate the parent-owned cell mapping for this test process.
///
/// `Ok(None)` means no dispatcher manifest was exported and preserves direct
/// test-binary behavior. Any present-manifest failure is returned to the
/// listing/dispatch boundary and must not fall back to shared ownership.
pub(crate) fn verifier_cell_ownership_from_env() -> anyhow::Result<Option<VerifierCellOwnership>> {
    let Some(manifest_path) = std::env::var_os(crate::KTSTR_VERIFIER_CELL_OWNERSHIP_MANIFEST_ENV)
    else {
        return Ok(None);
    };
    let current_executable = std::env::current_exe()
        .map_err(|error| anyhow::anyhow!("locate current verifier test executable: {error}"))?;
    cell_ownership_from_manifest_path(Some(manifest_path.as_os_str()), &current_executable)
}

/// A verifier topology which this guest kernel cannot boot without
/// changing the topology being verified.
///
/// This is intentionally verifier-specific rather than a generic host
/// insufficiency: ktstr and KVM can represent the requested topology, but
/// a distro kernel may explicitly disable x2APIC and fall back to the
/// 8-bit xAPIC limit. The cell is unsupported for that kernel, not a
/// scheduler failure, and the verifier dispatcher records it as SKIP.
#[derive(Debug)]
pub(crate) struct VerifierTopologyUnsupported {
    pub reason: String,
}

impl std::fmt::Display for VerifierTopologyUnsupported {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.reason)
    }
}

impl std::error::Error for VerifierTopologyUnsupported {}

/// Delimiter the guest's rust_init emits over the bulk port (as a
/// `MSG_TYPE_SCHED_LOG` frame) immediately before the scheduler log
/// block. Paired with [`SCHED_OUTPUT_END`].
pub(crate) const SCHED_OUTPUT_START: &str = "===SCHED_OUTPUT_START===";
/// Delimiter the guest's rust_init emits over the bulk port (as a
/// `MSG_TYPE_SCHED_LOG` frame) immediately after the scheduler log
/// block. Paired with [`SCHED_OUTPUT_START`].
pub(crate) const SCHED_OUTPUT_END: &str = "===SCHED_OUTPUT_END===";

/// Extract the scheduler log from guest output between
/// [`SCHED_OUTPUT_START`] and [`SCHED_OUTPUT_END`]. Returns `None` if
/// the delimiters are absent or the enclosed content is empty after
/// trimming.
///
/// Uses `find` on the start marker and `rfind` on the end marker: if
/// the scheduler log itself contains the end sentinel string (e.g. a
/// stack trace that quotes the marker), `rfind` anchors on the last
/// occurrence, which is the real terminator emitted by the guest's
/// post-scenario shutdown path.
pub(crate) fn parse_sched_output(output: &str) -> Option<&str> {
    let start = output.find(SCHED_OUTPUT_START)?;
    let end = output.rfind(SCHED_OUTPUT_END)?;
    let after_marker = start + SCHED_OUTPUT_START.len();
    if after_marker >= end {
        return None;
    }
    let content = output[after_marker..end].trim();
    if content.is_empty() {
        return None;
    }
    Some(content)
}

/// Concatenate every CRC-valid `MSG_TYPE_SCHED_LOG` chunk in the
/// bulk-port drain into one `String`, in arrival order.
///
/// The guest's `dump_sched_output` emits the `SCHED_OUTPUT_START`
/// and `SCHED_OUTPUT_END` markers as their own
/// [`crate::vmm::wire::MsgType::SchedLog`] frames, with the file
/// content split across one or more intermediate frames. Replaying
/// the chunks back-to-back reproduces the byte-for-byte stream the
/// prior COM2 path appended to `output`, so [`parse_sched_output`]
/// runs unchanged on the result.
///
/// Empty / `None` drain yields an empty string.
pub(crate) fn concat_sched_log_chunks(
    drain: Option<&crate::vmm::host_comms::BulkDrainResult>,
) -> String {
    concat_chunks_of(drain, crate::vmm::wire::MSG_TYPE_SCHED_LOG)
}

/// Concatenate every CRC-valid `MSG_TYPE_SCHED_STDOUT` chunk in the
/// bulk-port drain into one `String`, in arrival order.
///
/// These are the scheduler child's LIVE stdout chunks (shipped as each
/// pipe read arrives, not the teardown-only merged file), so they
/// survive a watchdog timeout that never reaches `dump_sched_output`.
/// Unlike [`concat_sched_log_chunks`] the payload carries no
/// `SCHED_OUTPUT_START/END` framing — it is the raw child stream, so
/// callers use it directly rather than through [`parse_sched_output`].
/// Empty / `None` drain yields an empty string.
pub(crate) fn concat_sched_stdout_chunks(
    drain: Option<&crate::vmm::host_comms::BulkDrainResult>,
) -> String {
    concat_chunks_of(drain, crate::vmm::wire::MSG_TYPE_SCHED_STDOUT)
}

/// Concatenate every CRC-valid `MSG_TYPE_SCHED_STDERR` chunk in the
/// bulk-port drain into one `String`, in arrival order. The scheduler
/// child's live stderr stream (where libbpf / log-crate output,
/// including the BPF verifier log region, typically lands). Same
/// semantics as [`concat_sched_stdout_chunks`].
pub(crate) fn concat_sched_stderr_chunks(
    drain: Option<&crate::vmm::host_comms::BulkDrainResult>,
) -> String {
    concat_chunks_of(drain, crate::vmm::wire::MSG_TYPE_SCHED_STDERR)
}

/// Reassemble every complete authoritative scheduler-stream transaction.
///
/// A transaction begins at offset zero and is accepted only when CRC-valid,
/// well-formed chunks cover one contiguous `0..total_len` range. Interrupted
/// replay is deliberately invisible to callers: they retain the best-effort
/// live stream instead. Multiple complete transactions are concatenated in
/// arrival order for scheduler-lifecycle runs that replace a scheduler inside
/// one VM.
fn concat_complete_sched_stream_replays(
    drain: Option<&crate::vmm::host_comms::BulkDrainResult>,
    msg_type: u32,
) -> Option<String> {
    let drain = drain?;
    let mut completed = Vec::<u8>::new();
    let mut transaction = None::<(u64, Vec<u8>)>;
    let mut saw_complete = false;

    for entry in &drain.entries {
        if entry.msg_type != msg_type {
            continue;
        }
        if !entry.crc_ok {
            transaction = None;
            continue;
        }
        let Some((total_len, offset, bytes)) =
            crate::vmm::wire::decode_sched_stream_final_chunk(&entry.payload)
        else {
            transaction = None;
            continue;
        };
        if offset == 0 {
            transaction = Some((total_len, Vec::new()));
        }
        let Some((expected_total, bytes_so_far)) = transaction.as_mut() else {
            continue;
        };
        if *expected_total != total_len
            || offset != bytes_so_far.len() as u64
            || (bytes.is_empty() && offset != total_len)
        {
            transaction = None;
            continue;
        }
        bytes_so_far.extend_from_slice(bytes);
        if bytes_so_far.len() as u64 == total_len {
            completed.extend_from_slice(bytes_so_far);
            saw_complete = true;
            transaction = None;
        }
    }

    saw_complete.then(|| String::from_utf8_lossy(&completed).into_owned())
}

/// Select the authoritative terminal replay when one arrived completely;
/// otherwise preserve the live-stream behavior used by old guests and timeout
/// paths that never reach terminal publication.
fn scheduler_stream_with_terminal_fallback(
    drain: Option<&crate::vmm::host_comms::BulkDrainResult>,
    live_msg_type: u32,
    replay_msg_type: u32,
) -> String {
    concat_complete_sched_stream_replays(drain, replay_msg_type).unwrap_or_else(|| {
        match live_msg_type {
            crate::vmm::wire::MSG_TYPE_SCHED_STDOUT => concat_sched_stdout_chunks(drain),
            crate::vmm::wire::MSG_TYPE_SCHED_STDERR => concat_sched_stderr_chunks(drain),
            _ => concat_chunks_of(drain, live_msg_type),
        }
    })
}

/// Concatenate every CRC-valid chunk of one `msg_type` in the drain,
/// in arrival order. Shared inner for the per-stream concat helpers.
fn concat_chunks_of(
    drain: Option<&crate::vmm::host_comms::BulkDrainResult>,
    msg_type: u32,
) -> String {
    let Some(drain) = drain else {
        return String::new();
    };
    let mut acc = String::new();
    for e in &drain.entries {
        if e.msg_type != msg_type || !e.crc_ok {
            continue;
        }
        acc.push_str(&String::from_utf8_lossy(&e.payload));
    }
    acc
}

/// Extract scheduler log content even when the closing delimiter is
/// absent. Tries [`parse_sched_output`] first (well-formed
/// open+close); on failure, returns the slice from
/// [`SCHED_OUTPUT_START`] to the end of `output` when only the start
/// marker is present. Returns `None` only when neither marker is
/// found or every candidate slice is empty after trimming.
///
/// Used by the auto-repro path: a scheduler that crashes mid-run
/// emits SCHED_OUTPUT_START but never reaches the post-scenario
/// shutdown that writes SCHED_OUTPUT_END. The partial content still
/// holds the stack frames the probe pipeline needs to seed kprobe
/// targets, so discarding it would lose the only crash signal.
pub(crate) fn parse_sched_output_partial(output: &str) -> Option<&str> {
    if let Some(content) = parse_sched_output(output) {
        return Some(content);
    }
    let start = output.find(SCHED_OUTPUT_START)?;
    let after_marker = start + SCHED_OUTPUT_START.len();
    let content = output[after_marker..].trim();
    if content.is_empty() {
        return None;
    }
    Some(content)
}

/// Parsed verifier stats from the kernel log line:
/// `processed N insns (limit M) max_states_per_insn X total_states Y peak_states Z mark_read W`
pub struct VerifierStats {
    /// Instructions processed during verification.
    pub processed_insns: u64,
    /// Total explored verifier states.
    pub total_states: u64,
    /// Peak concurrent explored states.
    pub peak_states: u64,
    /// Stack depth in the format `"<prog>+<subprog>+<main>"` (e.g.
    /// `"32+16+8"`) when BPF_LOG_STATS emitted a "stack depth" line.
    pub stack_depth: Option<String>,
}

/// Per-program verifier statistics collected from a VM run.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ProgStats {
    /// Program name as registered with the kernel.
    pub name: String,
    /// Instructions processed by the verifier (from host-side
    /// `bpf_prog_aux->verified_insns`).
    pub verified_insns: u32,
}

/// A single row in the A/B diff output.
pub struct DiffRow {
    /// Program name present in both A and B runs.
    pub name: String,
    /// `verified_insns` from the A run.
    pub a: u64,
    /// `verified_insns` from the B run.
    pub b: u64,
    /// Signed delta (`b - a`); positive means B's verifier cost grew
    /// relative to A.
    pub delta: i64,
}

/// Parse `raw` as `u64`, warning with `field` context on failure.
/// Returns 0 on parse error. Centralizes the
/// warn-and-default-to-zero path for the three count words
/// (`processed_insns`, `total_states`, `peak_states`) so their
/// error handling stays in lock-step.
fn parse_or_warn(raw: &str, field: &str) -> u64 {
    match raw.parse() {
        Ok(n) => n,
        Err(e) => {
            tracing::warn!(
                field,
                word = raw,
                err = %e,
                "malformed BPF verifier count; leaving 0",
            );
            0
        }
    }
}

/// Parse verifier stats from the log output.
///
/// The kernel always emits a "processed N insns ..." line. When
/// BPF_LOG_STATS is set, it may also emit a "stack depth" line.
/// Verification wall time is deliberately ignored: it is a noisy
/// environment-dependent measurement and does not belong in verifier
/// result output.
pub fn parse_verifier_stats(log: &str) -> VerifierStats {
    let mut stats = VerifierStats {
        processed_insns: 0,
        total_states: 0,
        peak_states: 0,
        stack_depth: None,
    };

    let mut found_insns = false;
    let mut found_stack = false;

    for line in log.lines().rev() {
        if !found_insns && line.starts_with("processed ") {
            found_insns = true;
            let words: Vec<&str> = line.split_whitespace().collect();
            if words.len() >= 2 {
                stats.processed_insns = parse_or_warn(words[1], "processed_insns");
            }
            for (i, &w) in words.iter().enumerate() {
                if w == "total_states"
                    && let Some(v) = words.get(i + 1)
                {
                    stats.total_states = parse_or_warn(v, "total_states");
                }
                if w == "peak_states"
                    && let Some(v) = words.get(i + 1)
                {
                    stats.peak_states = parse_or_warn(v, "peak_states");
                }
            }
        }
        if !found_stack && line.contains("stack depth") {
            found_stack = true;
            if let Some(pos) = line.find("stack depth") {
                let after = &line[pos + "stack depth".len()..];
                let depth_str = after.trim();
                if !depth_str.is_empty() {
                    stats.stack_depth = Some(depth_str.to_string());
                }
            }
        }
        if found_insns && found_stack {
            break;
        }
    }

    stats
}

/// Normalize a BPF verifier log line by stripping variable register-state
/// annotations so that lines from different loop iterations compare equal.
///
/// Handles:
/// - Instruction with `; frame` annotation: `3006: (07) r9 += 1  ; frame1: R9_w=2`
/// - Instruction with `; R` + digit annotation: `9: (15) if r7 == 0x0 goto pc+1  ; R7=scalar(...)`
/// - Branch with inline target state: `3026: (b5) if r6 <= 0x11dc0 goto pc+2 3029: frame1: R0=1`
/// - Standalone register dump with frame: `3041: frame1: R0_w=scalar()`
/// - Standalone register dump without frame: `3029: R0=1 R6=scalar()`
///
/// Preserves source comments (`; for (int j = 0; ...)`) and non-annotation
/// semicolons (`; Return value`) -- these serve as cycle anchors.
pub fn normalize_verifier_line(line: &str) -> &str {
    let trimmed = line.trim();
    if trimmed.is_empty() || !trimmed.as_bytes()[0].is_ascii_digit() {
        return trimmed;
    }
    // "3041: frame1: ..." or "3041: R0_w=scalar()" — standalone register dump.
    // State-only lines; keep just the instruction index.
    if let Some(colon) = trimmed.find(": ") {
        let after = &trimmed[colon + 2..];
        if after.starts_with("frame")
            || (after.starts_with('R')
                && after.as_bytes().get(1).is_some_and(|b| b.is_ascii_digit()))
        {
            return &trimmed[..colon + 1];
        }
    }
    // "; frame" annotation on instruction line
    if let Some(pos) = trimmed.find("; frame") {
        return trimmed[..pos].trim_end();
    }
    // "; R" followed by digit — register annotation without frame prefix
    if let Some(pos) = trimmed.find("; R")
        && trimmed
            .as_bytes()
            .get(pos + 3)
            .is_some_and(|b| b.is_ascii_digit())
    {
        return trimmed[..pos].trim_end();
    }
    // Inline branch-target state: "goto pc+2 3029: frame1: ..."
    if let Some(goto_pos) = trimmed.find("goto pc") {
        let after_goto = &trimmed[goto_pos + 7..];
        let end = after_goto
            .find(|c: char| c != '+' && c != '-' && !c.is_ascii_digit())
            .unwrap_or(after_goto.len());
        let insn_end = goto_pos + 7 + end;
        if insn_end < trimmed.len() {
            return trimmed[..insn_end].trim_end();
        }
    }
    trimmed
}

/// Normalize for cycle detection: strip register annotations (via
/// `normalize_verifier_line`) then strip the leading instruction address
/// (`NNN: `). Unrolled loops place each copy at different addresses, so
/// the address must be removed for block comparison to find repeats.
fn normalize_for_cycle_detection(line: &str) -> &str {
    let n = normalize_verifier_line(line);
    // Strip leading digits + ": " prefix (e.g. "42: (07) r1 += 8" -> "(07) r1 += 8").
    if let Some(colon) = n.find(": ") {
        let before = &n[..colon];
        if !before.is_empty() && before.bytes().all(|b| b.is_ascii_digit()) {
            return &n[colon + 2..];
        }
    }
    n
}

/// Detect a single repeating cycle in a slice of lines.
///
/// Returns `Some((start, period, count))` where the cycle begins at
/// `start`, each iteration is `period` lines, and it repeats `count` times.
pub fn detect_cycle(lines: &[&str]) -> Option<(usize, usize, usize)> {
    const MIN_PERIOD: usize = 5;
    const MIN_REPS: usize = 3;

    if lines.len() < MIN_PERIOD * MIN_REPS {
        return None;
    }

    // Two normalization levels:
    // - anchor_norms: keeps addresses, strips register annotations. Used for
    //   anchor frequency counting — prevents within-period duplicates at
    //   different addresses from inflating frequency.
    // - block_norms: also strips addresses. Used for block equality comparison
    //   so unrolled loops (same instructions at different addresses) can match.
    let anchor_norms: Vec<&str> = lines.iter().map(|l| normalize_verifier_line(l)).collect();
    let block_norms: Vec<&str> = lines
        .iter()
        .map(|l| normalize_for_cycle_detection(l))
        .collect();

    // Find most frequent non-trivial anchor-normalized line.
    let mut sorted_norms: Vec<&str> = anchor_norms
        .iter()
        .filter(|l| l.len() >= 10)
        .copied()
        .collect();
    sorted_norms.sort_unstable();

    let mut best_anchor: Option<(&str, usize)> = None;
    let mut i = 0;
    while i < sorted_norms.len() {
        let mut j = i + 1;
        while j < sorted_norms.len() && sorted_norms[j] == sorted_norms[i] {
            j += 1;
        }
        let count = j - i;
        if count >= MIN_REPS && best_anchor.is_none_or(|(_, best)| count > best) {
            best_anchor = Some((sorted_norms[i], count));
        }
        i = j;
    }

    // If address-preserving anchor search found nothing (unrolled loops
    // where every address is unique), fall back to address-stripped norms.
    let (anchor, use_block_norms_for_positions) = match best_anchor {
        Some((a, _)) => (a, false),
        None => {
            let mut sorted_block: Vec<&str> = block_norms
                .iter()
                .filter(|l| l.len() >= 10)
                .copied()
                .collect();
            sorted_block.sort_unstable();
            let mut ba: Option<(&str, usize)> = None;
            let mut bi = 0;
            while bi < sorted_block.len() {
                let mut bj = bi + 1;
                while bj < sorted_block.len() && sorted_block[bj] == sorted_block[bi] {
                    bj += 1;
                }
                let c = bj - bi;
                if c >= MIN_REPS && ba.is_none_or(|(_, best)| c > best) {
                    ba = Some((sorted_block[bi], c));
                }
                bi = bj;
            }
            match ba {
                Some((a, _)) => (a, true),
                None => return None,
            }
        }
    };

    let norms_for_pos = if use_block_norms_for_positions {
        &block_norms
    } else {
        &anchor_norms
    };
    let positions: Vec<usize> = norms_for_pos
        .iter()
        .enumerate()
        .filter(|(_, l)| **l == anchor)
        .map(|(i, _)| i)
        .collect();

    // Try strides 1..=3 to handle anchors appearing K times per cycle.
    for stride in 1..=3usize {
        if positions.len() <= stride {
            continue;
        }

        let mut gaps: Vec<usize> = positions
            .windows(stride + 1)
            .map(|w| w[stride] - w[0])
            .filter(|g| *g >= MIN_PERIOD)
            .collect();
        gaps.sort_unstable();

        let mut best_period = 0;
        let mut best_gap_count = 0;
        let mut gi = 0;
        while gi < gaps.len() {
            let mut gj = gi + 1;
            while gj < gaps.len() && gaps[gj] == gaps[gi] {
                gj += 1;
            }
            let count = gj - gi;
            if count > best_gap_count {
                best_gap_count = count;
                best_period = gaps[gi];
            }
            gi = gj;
        }
        if best_period == 0 || best_gap_count < MIN_REPS - 1 {
            continue;
        }
        let period = best_period;

        for &pos in &positions {
            if pos + 2 * period > lines.len() {
                break;
            }
            if block_norms[pos..pos + period] == block_norms[pos + period..pos + 2 * period] {
                let first_block = &block_norms[pos..pos + period];
                let mut count = 1;
                while pos + (count + 1) * period <= lines.len() {
                    if block_norms[pos + count * period..pos + (count + 1) * period] != *first_block
                    {
                        break;
                    }
                    count += 1;
                }
                // Try earlier starts to find best alignment.
                let mut best_start = pos;
                let mut best_count = count;
                for offset in 1..period {
                    let Some(cand) = pos.checked_sub(offset) else {
                        break;
                    };
                    if cand + 2 * period > lines.len() {
                        continue;
                    }
                    if block_norms[cand..cand + period]
                        != block_norms[cand + period..cand + 2 * period]
                    {
                        continue;
                    }
                    let mut c = 2;
                    while cand + (c + 1) * period <= lines.len()
                        && block_norms[cand + c * period..cand + (c + 1) * period]
                            == block_norms[cand..cand + period]
                    {
                        c += 1;
                    }
                    if c > best_count {
                        best_start = cand;
                        best_count = c;
                    }
                }
                if best_count >= MIN_REPS {
                    return Some((best_start, period, best_count));
                }
            }
        }
    }

    None
}

/// Collapse repeating cycles in a verifier log.
///
/// Runs cycle detection iteratively (up to 5 passes for nested loops).
/// Each cycle is replaced with:
/// - `--- Nx of the following M lines ---` (count header, no closing marker)
/// - first iteration (with original register annotations)
/// - `--- K identical iterations omitted ---` (omission marker)
/// - last iteration (with original register annotations)
/// - `--- end repeat ---` (closes the omission)
pub fn collapse_cycles(log: &str) -> String {
    const MAX_PASSES: usize = 5;
    let mut text = log.to_string();

    for _ in 0..MAX_PASSES {
        let lines: Vec<&str> = text.lines().collect();
        let (start, period, count) = match detect_cycle(&lines) {
            Some(c) => c,
            None => break,
        };

        let mut out = String::new();
        for line in &lines[..start] {
            out.push_str(line);
            out.push('\n');
        }
        out.push_str(&format!(
            "--- {}x of the following {} lines ---\n",
            count, period
        ));
        for line in &lines[start..start + period] {
            out.push_str(line);
            out.push('\n');
        }
        out.push_str(&format!(
            "--- {} identical iterations omitted ---\n",
            count - 2
        ));
        let last_start = start + (count - 1) * period;
        for line in &lines[last_start..last_start + period] {
            out.push_str(line);
            out.push('\n');
        }
        out.push_str("--- end repeat ---\n");
        let suffix_start = start + count * period;
        for line in &lines[suffix_start..] {
            out.push_str(line);
            out.push('\n');
        }
        text = out;
    }

    text
}

/// Build diff rows from A stats and B lookup map.
pub fn build_diff_rows(stats_a: &[ProgStats], b_map: &HashMap<String, u64>) -> Vec<DiffRow> {
    let mut rows = Vec::new();
    for ps in stats_a {
        let a = ps.verified_insns as u64;
        let b = b_map.get(&ps.name).copied().unwrap_or(0);
        rows.push(DiffRow {
            name: ps.name.clone(),
            a,
            b,
            delta: a as i64 - b as i64,
        });
    }
    rows
}

/// Build the B-side lookup map from collected stats.
pub fn build_b_map(stats_b: &[ProgStats]) -> HashMap<String, u64> {
    stats_b
        .iter()
        .map(|ps| (ps.name.clone(), ps.verified_insns as u64))
        .collect()
}

// ---------------------------------------------------------------------------
// VM-based verifier collection
// ---------------------------------------------------------------------------

/// Whether the scheduler positively confirmed it turned on during a
/// verifier VM run.
///
/// The guest init's attach gate (`poll_startup` + `poll_scx_attached`
/// in `crate::vmm::rust_init::scheduler`) already runs on every verifier
/// VM boot: it confirms the scheduler process survived BPF load AND that
/// `/sys/kernel/sched_ext/state` reached `enabled`. The kernel sets
/// `enabled` only after `ops.init`, per-task init, and switching
/// eligible tasks to the sched_ext class (`kernel/sched/ext.c`
/// `scx_root_enable_workfn`), so `enabled` proves the scheduler turned
/// on and is scheduling — not merely that its BPF loaded.
///
/// The verdict is POSITIVE-confirmation, not absence-of-failure: a
/// verifier cell PASSes only when the guest emitted the definitive
/// `SchedulerAttached` lifecycle frame. The guest emits that frame at the
/// one site where the scheduler child is alive and sched_ext reached
/// `enabled`. In particular, `PayloadStarting` is not attach evidence:
/// Phase 5 emits it for schedulerless runs too. Treating it as evidence
/// lets a schedulerless verifier workload running under the kernel's
/// fallback policy false-PASS.
///
/// On attach failure the guest emits `SchedulerDied` /
/// `SchedulerNotAttached`; a guest that vanishes without any conclusive
/// frame — e.g. a kernel panic which reboots via `panic=-1` — is
/// [`AttachOutcome::Unconfirmed`], also a FAIL. [`collect_verifier_output`]
/// consumes this verdict instead of discarding it. Scheduler-agnostic
/// (kernel sysfs state), so it holds for every declared scheduler.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AttachOutcome {
    /// The guest emitted `SchedulerAttached` with no failure frame: the
    /// scheduler loaded, stayed alive, and reached sched_ext `enabled`.
    Attached,
    /// Scheduler process exited during BPF load / startup
    /// (`LifecyclePhase::SchedulerDied`).
    Died,
    /// Scheduler stayed alive but never reached `enabled`
    /// (`LifecyclePhase::SchedulerNotAttached`); carries the guest's
    /// reason suffix when present.
    NotAttached(String),
    /// No failure frame AND no `SchedulerAttached` frame: attach was never
    /// positively confirmed. This includes an early guest kernel panic
    /// and a schedulerless run which proceeds to `PayloadStarting`.
    Unconfirmed,
}

impl AttachOutcome {
    /// Human-readable failure reason, or `None` when attached.
    pub fn failure_reason(&self) -> Option<String> {
        match self {
            AttachOutcome::Attached => None,
            AttachOutcome::Died => {
                Some("scheduler process exited during BPF load/startup".to_string())
            }
            AttachOutcome::NotAttached(reason) if reason.is_empty() => {
                Some("scheduler never reached sched_ext 'enabled'".to_string())
            }
            AttachOutcome::NotAttached(reason) => Some(format!(
                "scheduler never reached sched_ext 'enabled': {reason}"
            )),
            AttachOutcome::Unconfirmed => Some(
                "scheduler attach unconfirmed \
                 (no SchedulerAttached frame; scheduler may be absent or guest may have crashed)"
                    .to_string(),
            ),
        }
    }
}

/// Derive the [`AttachOutcome`] from a VM run's bulk-port lifecycle
/// frames using a POSITIVE-confirmation rule:
/// - a `SchedulerDied` frame ⇒ [`AttachOutcome::Died`] (wins outright —
///   a process that exited cannot have attached);
/// - else a `SchedulerNotAttached` frame ⇒ [`AttachOutcome::NotAttached`]
///   (with its reason suffix);
/// - else a `SchedulerAttached` frame ⇒ [`AttachOutcome::Attached`];
/// - else [`AttachOutcome::Unconfirmed`] — no failure AND no positive
///   frame, so attach was never confirmed (including a schedulerless run
///   which nevertheless reaches `PayloadStarting`).
///
/// Corrupt frames (`crc_ok == false`) and empty payloads are skipped. A
/// `None` `guest_messages` (no frames at all) is
/// [`AttachOutcome::Unconfirmed`].
pub(crate) fn attach_outcome_from_messages(
    guest_messages: Option<&crate::vmm::host_comms::BulkDrainResult>,
) -> AttachOutcome {
    let Some(drain) = guest_messages else {
        return AttachOutcome::Unconfirmed;
    };
    let mut not_attached: Option<String> = None;
    let mut scheduler_attached = false;
    for e in &drain.entries {
        if e.msg_type != crate::vmm::wire::MSG_TYPE_LIFECYCLE || !e.crc_ok || e.payload.is_empty() {
            continue;
        }
        match crate::vmm::wire::LifecyclePhase::from_wire(e.payload[0]) {
            Some(crate::vmm::wire::LifecyclePhase::SchedulerDied) => return AttachOutcome::Died,
            Some(crate::vmm::wire::LifecyclePhase::SchedulerNotAttached) => {
                not_attached = Some(String::from_utf8_lossy(&e.payload[1..]).into_owned());
            }
            Some(crate::vmm::wire::LifecyclePhase::SchedulerAttached) => {
                scheduler_attached = true;
            }
            _ => {}
        }
    }
    if let Some(reason) = not_attached {
        AttachOutcome::NotAttached(reason)
    } else if scheduler_attached {
        AttachOutcome::Attached
    } else {
        AttachOutcome::Unconfirmed
    }
}

/// Whether the guest confirmed workload dispatch: at least one
/// `WorkloadDispatched` lifecycle frame (crc-ok, non-empty payload) in
/// the run's bulk-port frames. Emitted by `ktstr_guest_init` Phase 5 only
/// when the injected SpinWait workload recorded a worker with non-zero
/// `iterations` under a confirmed SCHED_EXT policy (`sched_policy_error`
/// is None). Combined with a definitive `SchedulerAttached` frame, this
/// is a positive, scheduler-agnostic proof the scheduler dispatched a
/// task onto a CPU. Alone it is not attach proof: schedulerless SCHED_EXT
/// tasks may make progress through the kernel fallback.
/// Corrupt frames (`crc_ok == false`) and empty payloads are skipped. A
/// `None` `guest_messages` (no frames at all) is `false`.
fn dispatch_confirmed_from_messages(
    guest_messages: Option<&crate::vmm::host_comms::BulkDrainResult>,
) -> bool {
    let Some(drain) = guest_messages else {
        return false;
    };
    drain.entries.iter().any(|e| {
        e.msg_type == crate::vmm::wire::MSG_TYPE_LIFECYCLE
            && e.crc_ok
            && !e.payload.is_empty()
            && crate::vmm::wire::LifecyclePhase::from_wire(e.payload[0])
                == Some(crate::vmm::wire::LifecyclePhase::WorkloadDispatched)
    })
}

/// Whether the guest's scheduler-exit monitor observed the scheduler child
/// exiting unexpectedly. Unlike `SchedulerDied`, which is a startup
/// lifecycle outcome, `MSG_TYPE_SCHED_EXIT` covers a scheduler that attached
/// and then died while the verifier workload was running. A crc-invalid frame
/// is not evidence.
fn scheduler_exited_from_messages(
    guest_messages: Option<&crate::vmm::host_comms::BulkDrainResult>,
) -> bool {
    let Some(drain) = guest_messages else {
        return false;
    };
    drain
        .entries
        .iter()
        .any(|e| e.msg_type == crate::vmm::wire::MSG_TYPE_SCHED_EXIT && e.crc_ok)
}

/// Decode the verifier guest's explicit terminal exit frame.
///
/// This deliberately does not use `VmResult::exit_code`: the BSP run loop
/// assigns zero to a generic guest shutdown even when no `MSG_TYPE_EXIT`
/// arrived. Verifier success needs positive terminal evidence, so only a
/// crc-valid four-byte frame written by `guest_comms::send_exit` counts.
fn terminal_guest_exit_from_messages(
    guest_messages: Option<&crate::vmm::host_comms::BulkDrainResult>,
) -> Option<i32> {
    let drain = guest_messages?;
    drain
        .entries
        .iter()
        .rev()
        .find(|e| e.msg_type == crate::vmm::wire::MSG_TYPE_EXIT && e.crc_ok && e.payload.len() == 4)
        .map(|e| i32::from_le_bytes(e.payload[..4].try_into().unwrap()))
}

/// Result of collecting verifier output from a VM run.
pub struct VerifierVmResult {
    /// Per-program verifier statistics from host-side memory
    /// introspection (`bpf_prog_aux->verified_insns`).
    pub stats: Vec<ProgStats>,
    /// Scheduler log (merged stdout+stderr) from the VM's teardown
    /// `dump_sched_output` (the `/tmp/sched.log` file bracketed by the
    /// `SCHED_OUTPUT_START/END` markers). Contains libbpf's verifier
    /// instruction traces when BPF load fails. Present only when the run
    /// reached a dump path (teardown / scheduler-exit / boot-failure);
    /// a watchdog timeout kills the VM before any dump, leaving this
    /// empty — [`Self::scheduler_stdout`] / [`Self::scheduler_stderr`]
    /// are the live-streamed fallbacks that survive that case.
    pub scheduler_log: String,
    /// The scheduler child's stdout. Normally concatenated directly from the
    /// live `MSG_TYPE_SCHED_STDOUT` frames shipped per pipe read. When the
    /// guest detects that a live frame was suppressed or failed, a complete
    /// terminal replay replaces that partial live stream only after its
    /// offsets prove every byte arrived. Watchdog-killed and older guests keep
    /// the best-effort live behavior. Raw child stream — no
    /// `SCHED_OUTPUT_START/END` framing.
    pub scheduler_stdout: String,
    /// The scheduler child's stderr, using the same live-first,
    /// completion-proven terminal fallback semantics as
    /// [`Self::scheduler_stdout`]. Libbpf / log-crate output (including the
    /// BPF verifier log region between the `-- BEGIN/END PROG LOAD LOG --`
    /// markers) typically lands here.
    pub scheduler_stderr: String,
    /// Whether the scheduler positively confirmed attach. Derived from
    /// the guest's lifecycle frames ([`AttachOutcome`]). Attach is
    /// necessary but NOT sufficient for a cell PASS: this must be
    /// [`AttachOutcome::Attached`] (the guest emitted the definitive
    /// `SchedulerAttached` frame) AND [`Self::dispatched`] must be true.
    /// Verification alone (non-empty `stats`) is not enough — a scheduler
    /// whose BPF loads but never reaches sched_ext `enabled`, or a guest
    /// that vanishes before attach, is a real failure.
    pub attach: AttachOutcome,
    /// Whether the guest confirmed the injected verifier workload
    /// dispatched — a `WorkloadDispatched` lifecycle frame, emitted by
    /// `ktstr_guest_init` Phase 5 when the SpinWait probe recorded a
    /// worker with non-zero `iterations` after requesting SCHED_EXT
    /// policy. This progress signal is not independent attach proof:
    /// schedulerless SCHED_EXT tasks may run through the kernel fallback.
    /// A cell PASSes only when this is true AND [`Self::attach`] is
    /// [`AttachOutcome::Attached`]: a scheduler that turns on (sched_ext
    /// `enabled`) but never dispatches a runnable task is a real, distinct
    /// failure — worse than never attaching — that the attach verdict
    /// alone cannot catch. Derived from the run's lifecycle frames;
    /// scheduler-agnostic — the probe runs as SCHED_EXT, so the BPF
    /// scheduler dispatches it under any switch mode (full or
    /// `SCX_OPS_SWITCH_PARTIAL`) and non-zero worker progress proves
    /// dispatch, unlike an scx-specific `nr_dispatched` counter.
    pub dispatched: bool,
    /// The guest scheduler-exit monitor observed the scheduler process
    /// disappear after startup. This is independent of the historical
    /// `SchedulerAttached` frame: attach-at-t1 followed by exit-at-t2 is a
    /// failure even if SCHED_EXT fallback later lets a probe worker run.
    pub scheduler_exited: bool,
    /// Explicit terminal verifier guest exit code from a crc-valid
    /// `MSG_TYPE_EXIT` frame. The verifier guest emits 0 only after worker
    /// progress, a live scheduler child, and sched_ext `enabled` are observed
    /// at the same completion edge. `None` is a failure: a generic VM
    /// shutdown is not terminal verifier evidence.
    pub guest_exit_code: Option<i32>,
    /// The host watchdog fired (hard-deadline hang) before the guest
    /// exited. NOT orthogonal to [`Self::attach`] in the way the message
    /// once implied: a hang can occur EITHER after attach (a teardown
    /// wedge, leaving `attach == Attached`) OR before it (a scheduler
    /// that died / never reached `enabled` DURING BPF load and then
    /// wedged without rebooting, leaving `attach` at
    /// [`AttachOutcome::Died`] / [`AttachOutcome::NotAttached`] /
    /// [`AttachOutcome::Unconfirmed`]). An early kernel panic is the one
    /// case this flag does NOT cover: `panic=-1` reboots via an i8042
    /// reset (`ExitAction::Shutdown`, `timed_out == false`), caught by
    /// the attach gate as [`AttachOutcome::Unconfirmed`] instead. Because
    /// the hang is not attach-implying, [`Self::cell_verdict`] words the
    /// timeout message per [`Self::attach`] rather than always claiming
    /// "after attach". A verifier cell FAILs on a hang.
    pub timed_out: bool,
    /// The guest's structured crash message (a `PANIC:`-prefixed line
    /// routed through the host's `extract_panic_message` into
    /// [`crate::vmm::VmResult::crash_message`]), if any. A self-describing
    /// guest infra fault — most notably the AP-bring-up gap the boot
    /// retry exhausts — that [`Self::cell_verdict`] surfaces ABOVE the
    /// attach/dispatch gates, mirroring `test_support::eval`'s
    /// crash_message priority. `None` on the common no-crash path.
    pub crash_message: Option<String>,
    /// Host-side vCPU scheduling dilation for this cell's VM run —
    /// `D = 1 + Σrun_delay/Σon_cpu` over the vCPU host threads (see
    /// `vmm::result::HostVcpuSchedstat`). `None` on hosts
    /// without `CONFIG_SCHEDSTATS` or when no vCPU thread was sampled.
    /// Purely observational: surfaced in the per-cell output but NEVER
    /// folded into [`Self::cell_verdict`] or the exit code.
    pub dilation: Option<f64>,
}

impl VerifierVmResult {
    /// The verifier cell PASS/FAIL verdict: `Ok(())` when the scheduler
    /// verified its BPF, attached (sched_ext `enabled`), AND dispatched
    /// the injected workload; `Err(reason)` naming the first failing gate
    /// otherwise. Gate order is timeout, guest crash, observed scheduler
    /// exit, attach, dispatch, then explicit terminal guest exit. Root cause
    /// is reported first: an attach failure is named before the dispatch
    /// gate it necessarily also trips, and a terminal guest exit of 0 is
    /// required after all positive gates.
    pub fn cell_verdict(&self) -> Result<(), String> {
        // A hard-deadline hang, reported FIRST (outranks crash_message
        // and the attach/dispatch gates). The hang is NOT proof of a
        // post-attach wedge: the watchdog also fires when the scheduler
        // died / never reached `enabled` during BPF load and the guest
        // then wedged without rebooting. So keep the exact historical
        // "after attach" wording ONLY when attach was positively
        // confirmed; otherwise say the timeout carried no confirmed
        // attach and fold in the attach failure reason. Both arms keep
        // the "timed out" substring the gate-order test asserts.
        if self.timed_out {
            return Err(match self.attach {
                AttachOutcome::Attached => {
                    "VM timed out (hung after attach, before exit)".to_string()
                }
                _ => match self.attach.failure_reason() {
                    Some(reason) => {
                        format!("VM timed out (hung with no confirmed scheduler attach — {reason})")
                    }
                    None => "VM timed out (hung with no confirmed scheduler attach)".to_string(),
                },
            });
        }
        // A self-describing guest infra fault outranks the attach/dispatch
        // gates: the guest PID-1 panicked before the scheduler ran (most
        // notably the AP-bring-up gap the boot retry could not clear), so
        // "scheduler did not turn on" would misattribute an infra failure
        // to the scheduler. Surface the crash verbatim, mirroring
        // `test_support::eval`'s crash_message priority.
        if let Some(crash) = &self.crash_message {
            return Err(crash.clone());
        }
        if self.scheduler_exited {
            return Err("scheduler exited during the verifier workload".to_string());
        }
        // PASS requires the scheduler to have turned ON, not just to have
        // loaded + verified its BPF.
        if let Some(reason) = self.attach.failure_reason() {
            return Err(format!("scheduler did not turn on — {reason}"));
        }
        // PASS also requires DISPATCH: the guest injects a SpinWait
        // workload after attach and only emits WorkloadDispatched when a
        // SCHED_EXT worker makes forward progress. A scheduler that turns
        // on but never dispatches a runnable task is a distinct, worse
        // failure the attach gate can't catch.
        if !self.dispatched {
            return Err(
                "scheduler attached but did not dispatch the injected workload (0 iterations)"
                    .to_string(),
            );
        }
        match self.guest_exit_code {
            Some(0) => {}
            Some(code) => {
                return Err(format!(
                    "verifier guest did not complete successfully (exit code {code})"
                ));
            }
            None => {
                return Err("verifier guest did not publish a terminal exit status".to_string());
            }
        }
        Ok(())
    }
}

/// Complete scheduler-owned launch environment for one verifier VM.
///
/// Generated cells build this from the linked [`crate::test_support::Scheduler`]
/// declaration so verifier and ordinary test boots agree on every
/// scheduler-wide input which can affect startup or BPF verification. Direct
/// API callers retain their historical explicit-CLI-only behavior through
/// [`Self::direct`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct VerifierSchedulerLaunchPlan {
    sched_args: Vec<String>,
    cmdline_extra: String,
    include_files: Vec<(String, PathBuf)>,
    scheduler_cgroup_parent: Option<String>,
}

impl VerifierSchedulerLaunchPlan {
    fn direct(extra_sched_args: &[String]) -> Self {
        Self {
            sched_args: extra_sched_args.to_vec(),
            cmdline_extra: String::new(),
            include_files: Vec::new(),
            scheduler_cgroup_parent: None,
        }
    }

    pub(crate) fn from_scheduler(scheduler: &crate::test_support::Scheduler) -> Self {
        let mut sched_args = Vec::new();
        let mut include_files = Vec::new();
        if let Some((archive_path, host_path, guest_path)) =
            crate::test_support::runtime::scheduler_config_file_parts(scheduler)
        {
            include_files.push((archive_path, host_path));
            sched_args.push("--config".to_string());
            sched_args.push(guest_path);
        }
        sched_args.extend(scheduler.sched_args.iter().map(|arg| arg.to_string()));

        Self {
            sched_args,
            cmdline_extra: crate::test_support::runtime::scheduler_cmdline_tokens(scheduler)
                .join(" "),
            include_files,
            scheduler_cgroup_parent: scheduler
                .cgroup_parent
                .map(|parent| parent.as_str().to_string()),
        }
    }
}

/// Boot a VM and collect verifier statistics via host-side memory
/// introspection. Per-program `verified_insns` comes from
/// `bpf_prog_aux->verified_insns` read through the guest's physical
/// memory. On load failure, libbpf prints the verifier log to stderr;
/// the returned `scheduler_log` field contains the scheduler's captured
/// output from the VM.
///
/// `topology` selects the verifier VM's emulated topology via
/// [`crate::test_support::TopologyJson`] — the same named-field
/// shape carried in the per-scheduler `--ktstr-list-schedulers`
/// JSON. The named fields force callers to spell every dimension
/// at the call site, preventing position-swap misorders. The sole
/// caller is the sweep cell handler (`crate::test_support::dispatch`'s
/// `run_verifier_cell`), which passes the topology of the gauntlet
/// preset named in the cell — the verifier sweeps each scheduler across
/// every preset its constraints accept under no_perf_mode.
pub fn collect_verifier_output(
    sched_bin: &std::path::Path,
    ktstr_bin: &std::path::Path,
    kernel: &std::path::Path,
    extra_sched_args: &[String],
    topology: crate::vmm::topology::Topology,
    forced_cpu_budget: Option<u32>,
) -> anyhow::Result<VerifierVmResult> {
    let memory_min_mib = crate::test_support::runtime::cpu_scaled_memory_mib(topology.total_cpus());
    let launch = VerifierSchedulerLaunchPlan::direct(extra_sched_args);
    collect_verifier_output_with_memory_min(
        sched_bin,
        ktstr_bin,
        kernel,
        &launch,
        topology,
        memory_min_mib,
        forced_cpu_budget,
    )
}

/// Preset-sweep implementation with an explicit deferred-memory floor.
///
/// The public direct API above retains its historical CPU-scaled behavior.
/// The preset dispatcher passes the topology preset's capped floor so large
/// synthetic topologies do not inflate guest RAM independently of the
/// preset's declared budget.
pub(crate) fn collect_verifier_output_with_memory_min(
    sched_bin: &std::path::Path,
    ktstr_bin: &std::path::Path,
    kernel: &std::path::Path,
    launch: &VerifierSchedulerLaunchPlan,
    topology: crate::vmm::topology::Topology,
    memory_min_mib: u32,
    forced_cpu_budget: Option<u32>,
) -> anyhow::Result<VerifierVmResult> {
    use anyhow::Context;

    // Take the Topology directly (not the lossy TopologyJson wire shape,
    // which drops `nodes` / `distances` / `llc_cores`): a preset like
    // 192cpu-11llc-smt carries per-LLC core counts that MUST reach the guest
    // CPUID synthesis, so round-tripping through TopologyJson here would
    // silently boot a uniform shape instead. Validate directly so a bad
    // topology surfaces a clean error instead of the builder's panic.
    let validated = topology;
    validated
        .validate()
        .map_err(|e| anyhow::anyhow!("invalid topology {validated:?}: {e}"))?;

    // The verifier only loads the scheduler's BPF and reads the kernel
    // verifier's load-time `verified_insns` counts via host-side
    // introspection — a value fixed at BPF load, wholly INDEPENDENT of
    // perf-mode tuning (CPU pinning, RT priority, hugepages, NUMA mbind,
    // KVM exit suppression). So the verifier VM ALWAYS runs with
    // performance mode disabled: it needs none of that tuning. Disabling
    // perf mode also routes the run through the no-perf plan, which reserves
    // a shared (`LOCK_SH`) subset of LLCs and admission permits atomically.
    // `LOCK_SH` holders are
    // mutually compatible, so parallel verifier / no-perf cells no longer
    // starve each other on the LLC lock; a `performance_mode` peer holding
    // `LOCK_EX` on those LLCs can still defer a cell (nextest retries it),
    // which is correct — the verifier must not perturb an isolated peer's
    // pinned CPUs. Note `no_perf_mode(true)` does NOT skip the reservation
    // ENTIRELY — only `KTSTR_BYPASS_LLC_LOCKS=1` does; see
    // [`crate::vmm::KtstrVmBuilder::no_perf_mode`].
    // Pass the validated Topology directly so misorder cannot occur
    // at the builder boundary (the TryFrom above already enforces the
    // type-level invariants).
    //
    // Feed the caller-selected memory floor through the deferred path.
    // Direct API callers retain the historical vCPU-scaled floor, while
    // canned verifier cells cap it at the preset's declared budget. The
    // verifier /init is a large instrumented test
    // binary, so the deferred budget model raises the actual
    // allocation to fit the real initramfs (floor enforced at
    // `initramfs_min_memory_mib(&budget).max(self.memory_min_mib)` in
    // `vmm::setup::join_compute_memory_and_load`). Verifier cells have
    // no wprof, so no wprof floor applies here.
    // Bounded whole-boot retry on the guest AP-bring-up-gap infra fault
    // (an AP that missed its INIT-SIPI window → the guest PID-1 panics
    // pre-test). Rebuild the VM each attempt: `build` consumes the
    // builder and `run` consumes the boot. Every other failure returns
    // on the first attempt — the retry keys strictly on the marker.
    let result = crate::test_support::run_vm_with_ap_gap_retry(|| {
        let mut builder = crate::vmm::KtstrVm::builder()
            .kernel(kernel)
            // Prebuilt distro kernels ship virtio as modules; embed the
            // ordered boot-module set from the cache entry (no-op for built
            // kernels, which have no sibling modules/ dir).
            .kernel_modules(crate::cache::boot_modules_for_image(kernel))
            .initrd_compression(crate::cache::initrd_compression_for_image(kernel))
            .init_binary(ktstr_bin)
            .scheduler_binary(sched_bin)
            .sched_args(&launch.sched_args)
            .cmdline(&launch.cmdline_extra)
            // Boot the guest into the verifier dispatch probe: the sweep VM
            // has no `#[ktstr_test]` body, so Phase 5 spawns a SpinWait
            // workload and emits `WorkloadDispatched` on confirmed progress.
            // The PASS verdict requires that frame in addition to attach.
            .run_args(&[crate::test_support::VERIFIER_WORKLOAD_FLAG.to_string()])
            .topology(validated)
            .memory_deferred_min(memory_min_mib)
            // Timeout mirrors the `#[ktstr_test]` path's shape: a flat
            // lifecycle base plus vCPU-scaled dead-man boot headroom
            // (`vm_timeout_from_entry`'s split). This is only Tier-3 —
            // the progress watchdog governs wedges, and a wide cell
            // booting slowly under CI concurrency is progress-protected
            // rather than deadline-raced.
            // `workload_duration` arms the watchdog's attach reset: a
            // slow boot cannot eat the probe/teardown budget (the reset
            // is extend-only).
            .timeout(crate::test_support::runtime::verifier_vm_timeout(
                validated.total_cpus(),
            ))
            .workload_duration(crate::test_support::runtime::VERIFIER_WORKLOAD_BUDGET)
            .no_perf_mode(true);
        if !launch.include_files.is_empty() {
            builder = builder.include_files(launch.include_files.clone());
        }
        if let Some(parent) = launch.scheduler_cgroup_parent.as_ref() {
            builder = builder.scheduler_cgroup_parent(parent.clone());
        }
        // A preset with a forced CPU budget (192cpu-11llc-smt) pins the
        // no-perf mask to that many host CPUs so its vCPUs ALWAYS
        // overcommit — the deliberate, continuous time-slicing path.
        // Absent (every stock preset) leaves budget on the unchanged
        // auto-sized path: vCPU count plus service-thread headroom,
        // capped by the allowed host cpuset. no_perf_mode is already set
        // above, so cpu_budget is on its valid path.
        //
        // Clamp to the host's allowed CPUs: the forced budget exists to
        // FORCE overcommit, so on a host smaller than the budget it must
        // COLLAPSE to what is available (deeper overcommit — e.g. 192
        // vCPUs over 64 CPUs = 3x on this dev box) rather than hard-error
        // the way an explicit per-test `cpu_budget` would. That collapse
        // is the intended, storm-validated behavior. `.max(1)` guards the
        // barely-runnable host whose allowed set reads empty.
        if let Some(budget) = forced_cpu_budget {
            let allowed = crate::vmm::host_topology::host_allowed_cpus().len().max(1) as u32;
            builder = builder.cpu_budget(budget.min(allowed));
        }
        let vm = builder.build().context("build verifier VM")?;

        vm.run().context("run verifier VM")
    })?;

    // A cold retry is useful when an AP merely misses its INIT-SIPI
    // window under host contention. It is useless when the guest kernel
    // explicitly disables x2APIC and then rejects a topology APIC ID at
    // the 8-bit ceiling: another boot cannot change that kernel build.
    // Nor may the verifier compact IDs — Linux derives cache domains from
    // those topology bit fields, so compacting this shape would verify a
    // different topology under the original preset name. Surface a typed
    // unsupported-cell result; the dispatcher records SKIP, not scheduler
    // FAIL.
    #[cfg(target_arch = "x86_64")]
    if crate::test_support::guest_kernel_rejected_wide_apic(&result) {
        let max_apic_id = crate::vmm::x86_64::topology::max_apic_id(&validated);
        return Err(anyhow::Error::new(VerifierTopologyUnsupported {
            reason: format!(
                "guest kernel disabled x2APIC, but topology requires APIC IDs \
                 through {max_apic_id} (xAPIC supports 0..=254); compacting IDs \
                 would change the requested LLC topology"
            ),
        }));
    }

    // Concatenate bulk-port `MSG_TYPE_SCHED_LOG` chunks, then run
    // the marker-pair extractor on the merged stream — the
    // SCHED_OUTPUT_START/END markers travel verbatim inside chunk
    // bytes so the existing parser slices the same content the
    // prior COM2 dump produced. Falls back to `result.output` when
    // the bulk-port drain has no SchedLog frames (verifier VM
    // running on a kernel without the bulk port, for instance).
    let merged = concat_sched_log_chunks(result.guest_messages.as_ref());
    let scheduler_log = if !merged.is_empty() {
        parse_sched_output(&merged).unwrap_or("").to_string()
    } else {
        parse_sched_output(&result.output).unwrap_or("").to_string()
    };

    // Prefer a completion-proven terminal replay only when the guest observed
    // a dropped live chunk for that stream. Otherwise retain the zero-replay
    // live path. Watchdog kills and old guests never emit final frames, so
    // their existing best-effort live semantics remain unchanged.
    let scheduler_stdout = scheduler_stream_with_terminal_fallback(
        result.guest_messages.as_ref(),
        crate::vmm::wire::MSG_TYPE_SCHED_STDOUT,
        crate::vmm::wire::MSG_TYPE_SCHED_STDOUT_FINAL,
    );
    let scheduler_stderr = scheduler_stream_with_terminal_fallback(
        result.guest_messages.as_ref(),
        crate::vmm::wire::MSG_TYPE_SCHED_STDERR,
        crate::vmm::wire::MSG_TYPE_SCHED_STDERR_FINAL,
    );

    // Build ProgStats from host-side ProgVerifierStats. Each program
    // that loaded successfully is visible in prog_idr with its
    // verified_insns count.
    let stats: Vec<ProgStats> = result
        .verifier_stats
        .iter()
        .map(|pvs| ProgStats {
            name: pvs.name.clone(),
            verified_insns: pvs.verified_insns,
        })
        .collect();

    let attach = attach_outcome_from_messages(result.guest_messages.as_ref());
    let dispatched = dispatch_confirmed_from_messages(result.guest_messages.as_ref());
    let scheduler_exited = scheduler_exited_from_messages(result.guest_messages.as_ref());
    let guest_exit_code = terminal_guest_exit_from_messages(result.guest_messages.as_ref());

    Ok(VerifierVmResult {
        stats,
        scheduler_log,
        scheduler_stdout,
        scheduler_stderr,
        attach,
        dispatched,
        scheduler_exited,
        guest_exit_code,
        timed_out: result.timed_out,
        crash_message: result.crash_message.clone(),
        // Host-side vCPU dilation, derived from the run's raw schedstat
        // totals. `None` when schedstats were unavailable or no vCPU was
        // sampled — carried through untouched, never gating the verdict.
        dilation: result
            .host_vcpu_schedstat
            .as_ref()
            .and_then(|s| s.dilation()),
    })
}

/// Extract the verifier instruction trace from a scheduler log blob.
///
/// libbpf wraps the kernel verifier log between marker lines:
///   `-- BEGIN PROG LOAD LOG --`
///   `-- END PROG LOAD LOG --`
///
/// Returns the content between the first pair of markers, or `None` if
/// no markers are found (backward compat with logs that contain only
/// raw verifier output).
pub fn extract_verifier_log(scheduler_log: &str) -> Option<&str> {
    let (start, end) = verifier_log_region(scheduler_log)?;
    // `end` includes the trailing newline before the END-marker line;
    // trim it (and any earlier trailing blanks) to match the historical
    // extracted-content shape parse_verifier_stats consumes.
    Some(scheduler_log[start..end].trim_end_matches('\n'))
}

/// Locate the libbpf verifier-log region inside a scheduler log blob:
/// the byte range `[start, end)` of the content between libbpf's
///   `-- BEGIN PROG LOAD LOG --`
///   `-- END PROG LOAD LOG --`
/// markers. `start` is just past the BEGIN marker (skipping the marker's
/// own trailing newline); `end` is the byte AFTER the last content
/// newline preceding the END marker — so the region includes that
/// trailing newline but excludes the END-marker line itself (and any
/// partial `libbpf: ` prefix when the END marker sits mid-line). Returns
/// `None` when either marker is absent.
///
/// Shared marker-location logic for [`extract_verifier_log`] (which
/// trims the region into the collapsible trace) and
/// [`collapse_verifier_region`] (which replaces the region in place),
/// so the two stay consistent — notably on the mid-line-END case where
/// the partial prefix stays OUTSIDE the region and thus byte-identical
/// in the in-place transform.
fn verifier_log_region(scheduler_log: &str) -> Option<(usize, usize)> {
    const BEGIN: &str = "-- BEGIN PROG LOAD LOG --";
    const END: &str = "-- END PROG LOAD LOG --";

    let begin_pos = scheduler_log.find(BEGIN)?;
    let content_start = begin_pos + BEGIN.len();
    // Skip the newline after the BEGIN marker if present.
    let content_start = if scheduler_log.as_bytes().get(content_start) == Some(&b'\n') {
        content_start + 1
    } else {
        content_start
    };
    let rel_end = scheduler_log[content_start..].find(END)?;
    let raw_end = content_start + rel_end;
    // The END marker may appear mid-line (e.g. "libbpf: -- END ...").
    // Anchor the region end at the last content newline before the
    // marker (inclusive), so the partial "libbpf: " prefix stays in the
    // suffix — keeping it byte-identical under the in-place collapse.
    let region_end = match scheduler_log[content_start..raw_end].rfind('\n') {
        Some(p) => content_start + p + 1,
        None => raw_end,
    };
    Some((content_start, region_end))
}

/// Return `s` with the libbpf verifier-log region (between the
/// `-- BEGIN/END PROG LOAD LOG --` markers) replaced by
/// [`collapse_cycles`] of that region — collapsed IN PLACE, with every
/// byte OUTSIDE the region (the surrounding scheduler output, the BEGIN
/// and END marker lines, any mid-line-END partial prefix) preserved
/// exactly. Input WITHOUT the marker pair is returned unchanged.
///
/// This replaces the prior extract-and-drop rendering: the scheduler's
/// full output is preserved and only the (potentially huge, cyclic)
/// verifier trace is compressed, rather than discarding everything
/// outside the trace.
pub fn collapse_verifier_region(s: &str) -> String {
    let Some((start, end)) = verifier_log_region(s) else {
        return s.to_string();
    };
    let mut out = String::with_capacity(s.len());
    out.push_str(&s[..start]);
    out.push_str(&collapse_cycles(&s[start..end]));
    out.push_str(&s[end..]);
    out
}

/// Format verifier results as text: brief lines per program and collapsed
/// logs.
pub fn format_verifier_output(label: &str, result: &VerifierVmResult, raw: bool) -> String {
    let mut out = String::new();
    out.push_str(&format!("\n{label}\n"));
    if result.timed_out {
        out.push_str("  scheduler: UNKNOWN — VM timed out before exit\n");
    } else if result.scheduler_exited {
        out.push_str("  scheduler: EXITED during verifier workload\n");
    } else {
        match result.attach.failure_reason() {
            None => {
                out.push_str("  scheduler: attached (sched_ext enabled)\n");
                if result.dispatched {
                    out.push_str("  dispatch: confirmed (injected workload ran)\n");
                } else {
                    out.push_str(
                        "  dispatch: NOT CONFIRMED — attached but injected workload made no progress\n",
                    );
                }
            }
            Some(reason) => out.push_str(&format!("  scheduler: NOT ATTACHED — {reason}\n")),
        }
    }
    // Host-side vCPU scheduling dilation. `D = 1 + run_delay/on_cpu`, so
    // the fraction of the CPU the vCPUs actually received when runnable is
    // `on_cpu/(on_cpu+run_delay) == 1/D` exactly — rendered as a whole
    // percent. `None` (schedstats off / nothing sampled) is stated as
    // such, distinct from a genuine 1.00x.
    match result.dilation {
        Some(d) => {
            let pct = (100.0 / d).round() as u32;
            out.push_str(&format!(
                "  host dilation: {d:.2}x (vCPUs received {pct}% of the CPU they were runnable for)\n"
            ));
        }
        None => out.push_str("  host dilation: n/a (host schedstat unavailable)\n"),
    }
    for ps in &result.stats {
        out.push_str(&format!(
            "  {:<40} verified_insns={}\n",
            ps.name, ps.verified_insns
        ));
    }

    // verifier stats: parse from whichever stream actually carries the
    // libbpf `-- BEGIN/END PROG LOAD LOG --` markers. With the split
    // live streams the verifier log usually lands on stderr; fall back
    // to stdout, then the teardown merged-file dump. When no stream has
    // the markers, parse the first non-empty stream whole — backward
    // compat with logs that contain only raw verifier output (the same
    // fallback `extract_verifier_log` documents). Renders on the
    // timed_out branch too (the streams survive a watchdog kill).
    let streams = [
        result.scheduler_stderr.as_str(),
        result.scheduler_stdout.as_str(),
        result.scheduler_log.as_str(),
    ];
    let stats_src = streams
        .into_iter()
        .find(|s| extract_verifier_log(s).is_some())
        .or_else(|| streams.into_iter().find(|s| !s.is_empty()));
    if let Some(src) = stats_src {
        let verifier_log = extract_verifier_log(src).unwrap_or(src);
        let vs = parse_verifier_stats(verifier_log);
        if vs.processed_insns > 0 {
            out.push_str(&format!("\n{label} --- verifier stats ---\n"));
            out.push_str(&format!(
                "  processed={}  states={}/{}",
                vs.processed_insns, vs.peak_states, vs.total_states
            ));
            if let Some(ref s) = vs.stack_depth {
                out.push_str(&format!("  stack={s}"));
            }
            out.push('\n');
        }
    }

    // The scheduler-log section renders the FULL scheduler stdout — the
    // live-streamed `scheduler_stdout` (which survives a watchdog kill
    // that never reaches the teardown dump). NOT extract-and-drop: the
    // verifier trace is collapsed IN PLACE via `collapse_verifier_region`,
    // keeping everything around it. Emitted on the timed_out branch as
    // well — that is the core bug this fixes. The merged teardown dump
    // `scheduler_log` is the fallback ONLY when NEITHER live stream
    // arrived (streaming failed or the guest predates the split): when
    // live stderr did arrive, an empty live stdout means the scheduler
    // wrote nothing to stdout, and rendering the merged dump here would
    // duplicate the stderr section (the common libbpf/log-crate case —
    // schedulers that log exclusively to stderr).
    let stdout_src = if !result.scheduler_stdout.is_empty() {
        result.scheduler_stdout.as_str()
    } else if result.scheduler_stderr.is_empty() {
        result.scheduler_log.as_str()
    } else {
        ""
    };
    if !stdout_src.is_empty() {
        out.push_str(&format!("\n{label} --- scheduler log ---\n"));
        if raw {
            out.push_str(stdout_src);
        } else {
            out.push_str(&collapse_verifier_region(stdout_src));
        }
    }

    out
}

/// Format the scheduler's captured STDERR for emission to the test's
/// stderr, or the empty string when there is no stderr to show.
///
/// Best-effort: renders whatever `scheduler_stderr` streamed before the
/// VM exited or was watchdog-killed. Non-raw runs collapse the libbpf
/// verifier-log region IN PLACE via [`collapse_verifier_region`] (the
/// verifier trace usually lands on stderr with the split streams); raw
/// runs pass the bytes through unmodified. A leading `--- scheduler
/// stderr ---` header labels the section to match the stdout section's
/// style so a reader can tell the two streams apart. The caller emits
/// the result via `eprint!` so it lands on the test's real stderr.
pub fn format_verifier_stderr(label: &str, result: &VerifierVmResult, raw: bool) -> String {
    if result.scheduler_stderr.is_empty() {
        return String::new();
    }
    let body = if raw {
        result.scheduler_stderr.clone()
    } else {
        collapse_verifier_region(&result.scheduler_stderr)
    };
    format!("\n{label} --- scheduler stderr ---\n{body}")
}

/// Format an A/B diff table comparing two sets of verifier stats.
pub fn format_verifier_diff(
    label_a: &str,
    stats_a: &[ProgStats],
    label_b: &str,
    stats_b: &[ProgStats],
) -> String {
    let b_map = build_b_map(stats_b);
    let diff_rows = build_diff_rows(stats_a, &b_map);

    let mut out = String::new();
    out.push_str(&format!("\ndelta A/B diff: {label_a} vs {label_b}\n"));
    let mut table = crate::cli::new_table();
    table.set_header(vec!["program", "A", "B", "delta"]);
    for row in &diff_rows {
        table.add_row(vec![
            row.name.clone(),
            row.a.to_string(),
            row.b.to_string(),
            format!("{:+}", row.delta),
        ]);
    }
    out.push_str(&table.to_string());
    out.push('\n');
    out
}

// ---------------------------------------------------------------------------
// Per-cell PASS/FAIL result capture (for the run-summary table)
// ---------------------------------------------------------------------------

/// One `cargo ktstr verifier` cell's outcome. The cell process writes it
/// (via `write_cell_record`) into the directory named by
/// [`crate::KTSTR_VERIFIER_RESULT_DIR_ENV`]; after nextest returns the
/// dispatcher reads them back (via [`read_cell_records`]) and renders one
/// PASS/FAIL/SKIP grid per scheduler (rows = topology, cols = kernel). A cell
/// is one
/// (scheduler, kernel, topology): the verifier sweeps each declared
/// scheduler across topologies, so topology IS a result axis — a
/// scheduler can pass on one topology and fail on another.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct VerifierCellRecord {
    /// Declared scheduler name (the `<sched>` cell-name segment).
    pub scheduler: String,
    /// Sanitized kernel label (the `<kernel>` cell-name segment).
    pub kernel: String,
    /// Gauntlet topology preset (the `<preset>` cell-name segment).
    pub topology: String,
    /// Whether the cell passed. False for both FAIL and SKIP; consult
    /// [`Self::skipped`] to distinguish them.
    pub passed: bool,
    /// Whether the cell was unsupported for this guest kernel. Mutually
    /// exclusive with `passed`; rendered as `-` and excluded from both
    /// pass and fail tallies. `serde(default)` accepts records from ktstr
    /// versions which predate the explicit verifier-skip state.
    #[serde(default)]
    pub skipped: bool,
    /// Per-program stats (program name + its `verified_insns` count)
    /// captured for this cell, copied from the VM run's
    /// [`VerifierVmResult::stats`]. Empty when the cell failed before
    /// producing stats. Drives the per-scheduler `verified_insns` tables
    /// ([`render_instruction_count_tables`], rows = BPF program, cols =
    /// kernel, cell = the count summarized across topologies) that the
    /// dispatcher prints before the PASS/FAIL grids.
    pub stats: Vec<ProgStats>,
}

/// Map a cell's full name to a compact content-addressed record filename.
///
/// The fixed-seed ahash matches ktstr's other fast content-addressed
/// identities. Hashing the raw, length-delimited name avoids aliases introduced
/// by filesystem sanitization (`foo-bar` versus `foo_bar`) and keeps the
/// filename below `NAME_MAX` even when a valid scheduler name is long. A
/// nextest RETRY of the same cell resolves to the same path and overwrites its
/// prior record, so the FINAL attempt's outcome wins.
fn cell_record_filename(full_name: &str) -> String {
    use std::hash::{BuildHasher, Hasher};

    let bytes = full_name.as_bytes();
    let mut hasher = ahash::RandomState::with_seeds(0, 0, 0, 0).build_hasher();
    hasher.write(&(bytes.len() as u64).to_le_bytes());
    hasher.write(bytes);
    format!("cell-{:016x}.json", hasher.finish())
}

/// Write a cell's PASS/FAIL/SKIP record into `dir`. Parses `full_name`
/// (`verifier/<sched>/<kernel>/<preset>`); a name that does not fit that
/// shape is skipped (the cell already errored on the malformed name).
/// The record is serialized to a same-directory temporary and atomically
/// renamed over the deterministic final path. Readers therefore see either
/// the preceding retry's complete record or the current retry's complete
/// record, never a truncated JSON prefix. Best-effort: a write failure is
/// logged and swallowed — the summary table is a convenience over the
/// per-cell nextest output, so a lost record must never turn an
/// otherwise-passing cell into a failure.
///
/// `pub(crate)`: the only writer is the cell handler in
/// `test_support::dispatch` (same crate); the reader side
/// ([`read_cell_records`] / [`render_result_table`]) is `pub` for the
/// `cargo-ktstr` binary crate.
pub(crate) fn write_cell_record(
    dir: &std::path::Path,
    full_name: &str,
    passed: bool,
    skipped: bool,
    stats: &[ProgStats],
) {
    debug_assert!(
        !(passed && skipped),
        "a verifier cell cannot be both passed and skipped"
    );
    let Some(rest) = full_name.strip_prefix("verifier/") else {
        return;
    };
    let parts: Vec<&str> = rest.splitn(3, '/').collect();
    if parts.len() != 3 {
        return;
    }
    let record = VerifierCellRecord {
        scheduler: parts[0].to_string(),
        kernel: parts[1].to_string(),
        topology: parts[2].to_string(),
        passed,
        skipped,
        stats: stats.to_vec(),
    };
    let path = dir.join(cell_record_filename(full_name));
    let result = (|| -> anyhow::Result<()> {
        let mut temporary = tempfile::NamedTempFile::new_in(dir)?;
        serde_json::to_writer(temporary.as_file_mut(), &record)?;
        std::io::Write::flush(temporary.as_file_mut())?;
        temporary.persist(&path).map_err(|error| error.error)?;
        Ok(())
    })();
    if let Err(error) = result {
        eprintln!(
            "ktstr verifier: warning: could not atomically write result record {}: {error:#}",
            path.display(),
        );
    }
}

/// Read every `*.json` cell record under `dir` (non-recursive). A missing
/// dir or an unparseable record is skipped (best-effort). Returns records
/// in filesystem-iteration order; [`render_result_table`] sorts for a
/// deterministic render.
pub fn read_cell_records(dir: &std::path::Path) -> Vec<VerifierCellRecord> {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    entries
        .flatten()
        .filter(|e| {
            e.path()
                .extension()
                .is_some_and(|x| x.eq_ignore_ascii_case("json"))
        })
        .filter_map(|e| std::fs::read(e.path()).ok())
        .filter_map(|bytes| serde_json::from_slice::<VerifierCellRecord>(&bytes).ok())
        .collect()
}

/// The terminal disposition of a `cargo ktstr verifier` run, decided by
/// [`classify_run_outcome`] after nextest returns and the report grids
/// have been printed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RunOutcome {
    /// Clean success — the dispatcher returns `Ok(())`.
    Success,
    /// A diagnosable failure whose message the dispatcher routes through
    /// main's stderr error handler (exit 1). Covers a nextest nonzero
    /// exit with NO failed cell (a build / internal failure — nothing in
    /// the grid to point at), a signal exit (no numeric code), and the
    /// empty-record diagnostics.
    Failed(String),
    /// nextest exited nonzero AND the printed grid already shows at least
    /// one failing (✗) cell, so the failure is already visible on stdout.
    /// The dispatcher exits with this (nextest's own) code but does NOT emit a
    /// redundant `error: cargo nextest run exited with N` line: that line
    /// goes to stderr while the report goes to stdout, and with two
    /// unordered pipes it otherwise interleaves mid-report in CI logs.
    SilentExit(i32),
}

/// Decide the `cargo ktstr verifier` run outcome from nextest's exit
/// success, whether any per-cell records were produced, whether any of
/// those records is a FAILED cell (already shown as a ✗ in the printed
/// grid), and the optional `--scheduler` filter.
///
/// The dispatcher runs nextest with `--no-tests=pass`, so a run that
/// selects zero verifier cells exits 0 (success) with empty records
/// rather than nextest's generic "no tests to run" (exit 4). That lets
/// the dispatcher diagnose the empty case itself instead of surfacing a
/// cryptic nextest exit:
/// - `--scheduler <NAME>` set + no records: the name is not a declared
///   scheduler, or its declared topology constraints / verifier-only
///   exclusions reject every selected preset.
/// - no `--scheduler` + no records: no `declare_scheduler!` is linked into
///   a test binary the sweep sees, or every declared scheduler rejects all
///   selected presets through those declaration-level gates.
///
/// A genuine build/exec failure still fails nextest (exit non-zero, which
/// `--no-tests=pass` does not mask). When that nonzero exit is a REAL
/// cell failure — `has_failed_cell` true and nextest handed back a
/// numeric `exit_code` — the printed grid already shows the failing
/// cell(s), so the outcome is [`RunOutcome::SilentExit`] (carry nextest's
/// code, emit no stderr line). Every other nonzero case — a build /
/// internal failure that produced no failed cell, or a signal exit with
/// no numeric code — stays [`RunOutcome::Failed`] with the descriptive
/// message, as does each empty-record diagnostic.
pub fn classify_run_outcome(
    success: bool,
    records_empty: bool,
    has_failed_cell: bool,
    scheduler: Option<&str>,
    exit_code: Option<i32>,
) -> RunOutcome {
    if !success {
        // The printed grid already shows the failing cell(s): carry
        // nextest's code out silently instead of a stderr error line that
        // would interleave mid-report. Only with a numeric code — a
        // signal exit (None) keeps the descriptive message.
        if has_failed_cell && let Some(code) = exit_code {
            return RunOutcome::SilentExit(code);
        }
        let code = exit_code.map_or_else(|| "signal".to_string(), |c| c.to_string());
        return RunOutcome::Failed(format!("cargo nextest run exited with {code}"));
    }
    if records_empty {
        return RunOutcome::Failed(match scheduler {
            Some(name) => format!(
                "--scheduler {name:?}: matched no verifier cell — no declared BPF \
                 scheduler by that name, or its declared topology constraints or \
                 verifier-only exclusions rejected every selected preset. Run \
                 `cargo ktstr verifier` with no --scheduler to see the swept set."
            ),
            None => "no verifier cells ran — no scheduler is declared via \
                 declare_scheduler! in a linked test binary, or every declared \
                 scheduler's topology constraints or verifier-only exclusions \
                 rejected every selected preset."
                .to_string(),
        });
    }
    RunOutcome::Success
}

/// Build the `cargo nextest run` argument vector for the verifier sweep.
///
/// Load-bearing tokens:
/// - `--run-ignored all`: verifier cells are emitted ignore-gated, so
///   nextest skips them unless opted in.
/// - `--no-tests pass`: a zero-cell selection exits 0 (not nextest's
///   default exit-4 "no tests to run"), so [`classify_run_outcome`] can
///   emit a targeted diagnostic instead of a cryptic nextest exit.
/// - `-E 'test(/^verifier/) & !test(/^verifier::/)'`: the `verifier/...`
///   cells, excluding the verifier module's own `verifier::tests::*`.
///
/// `nextest_profile`, if set, becomes nextest's `--profile <NAME>`,
/// emitted before `forward` so a forwarded token cannot shadow it.
/// `forward` is the user's trailing cargo/nextest args, appended verbatim.
pub fn build_nextest_args(nextest_profile: Option<&str>, forward: &[String]) -> Vec<String> {
    let mut args = vec![
        "nextest".to_string(),
        "run".to_string(),
        "--run-ignored".to_string(),
        "all".to_string(),
        "--no-tests".to_string(),
        "pass".to_string(),
        "-E".to_string(),
        "test(/^verifier/) & !test(/^verifier::/)".to_string(),
    ];
    if let Some(np) = nextest_profile {
        args.push("--profile".to_string());
        args.push(np.to_string());
    }
    args.extend(forward.iter().cloned());
    args
}

/// Render the per-cell records into one PASS/FAIL/SKIP grid PER declared
/// scheduler (BTreeSet-sorted). Each scheduler's section is a tally line
/// followed by a bordered grid whose rows are the topology presets that
/// ran the scheduler (BTreeSet) and whose columns are the kernels that
/// ran it (BTreeSet); the header row's first column is `topology`. Each
/// grid cell is the (topology, kernel) result FOR THAT SCHEDULER,
/// rendered nextest-style:
/// - `✓` (bold green) — every record for the triple passed,
/// - `✗` (bold red) — any record for it failed; this also absorbs the
///   defensive duplicate-record case where one triple somehow carries
///   both a pass and a fail — any failure means failure,
/// - `-` (plain) — every record for it skipped, or no record (the
///   scheduler's constraints rejected the preset on that kernel).
///
/// Cells are single-width glyphs (U+2713 / U+2717), not emoji: emoji
/// are not exactly two monospace cells in GitHub's log viewer / code
/// blocks, so they drift the box-drawing columns. The glyph styling
/// comes from [`comfy_table::Cell::fg`] + `Attribute::Bold`, with
/// emission governed by [`crate::cli::new_bordered_table`]'s color
/// policy (styled on a TTY and in color-forcing CI, plain otherwise).
/// The tally line keeps the emoji — `{sched}: {n_pass} ✅  {n_fail} ❌`
/// — because it is prose with nothing to column-align. The counts tally
/// the grid CELLS (a `-` cell counts toward neither; a cell with any
/// failing record counts toward `n_fail`). The kernel is a grid axis
/// rather than folded into the cell, so a ✗ cell already names the
/// exact kernel that failed — the old flat `failing combinations
/// (scheduler / kernel / topology)` list it used to need is gone.
///
/// Returns `None` for an empty record set (the caller prints nothing).
/// Schedulers, rows, and columns are BTreeSet-sorted so the same run
/// renders the same output (shell-pipeline stable). Uses
/// [`crate::cli::new_bordered_table`] (box-drawing borders) so each
/// scheduler's grid is visually delimited from its tally line and from
/// the next scheduler's section.
pub fn render_result_table(records: &[VerifierCellRecord]) -> Option<String> {
    if records.is_empty() {
        return None;
    }
    use std::collections::{BTreeMap, BTreeSet};
    let mut schedulers: BTreeSet<String> = BTreeSet::new();
    // scheduler -> topology rows that ran it.
    let mut sched_rows: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    // scheduler -> kernel columns that ran it.
    let mut sched_cols: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    #[derive(Clone, Copy, Default)]
    struct CellDisposition {
        any_pass: bool,
        any_fail: bool,
        any_skip: bool,
    }
    // scheduler -> (topology, kernel) -> folded disposition. Failure
    // dominates pass, which dominates skip, for defensive duplicates.
    let mut agg: BTreeMap<String, BTreeMap<(String, String), CellDisposition>> = BTreeMap::new();
    for r in records {
        schedulers.insert(r.scheduler.clone());
        sched_rows
            .entry(r.scheduler.clone())
            .or_default()
            .insert(r.topology.clone());
        sched_cols
            .entry(r.scheduler.clone())
            .or_default()
            .insert(r.kernel.clone());
        let disposition = agg
            .entry(r.scheduler.clone())
            .or_default()
            .entry((r.topology.clone(), r.kernel.clone()))
            .or_default();
        if r.skipped {
            disposition.any_skip = true;
        } else if r.passed {
            disposition.any_pass = true;
        } else {
            disposition.any_fail = true;
        }
    }

    let mut out =
        String::from("\nverifier results (per scheduler; rows: topology, cols: kernel):\n");
    for sched in &schedulers {
        let rows = &sched_rows[sched];
        let cols = &sched_cols[sched];
        let cells = &agg[sched];
        let (mut n_pass, mut n_fail) = (0usize, 0usize);
        let mut table = crate::cli::new_bordered_table();
        let mut header: Vec<String> = vec!["topology".to_string()];
        for kernel in cols {
            header.push(kernel.clone());
        }
        table.set_header(header);
        for topo in rows {
            let mut line: Vec<comfy_table::Cell> = vec![comfy_table::Cell::new(topo)];
            for kernel in cols {
                let cell = match cells.get(&(topo.clone(), kernel.clone())).copied() {
                    None => comfy_table::Cell::new("-"),
                    Some(d) if d.any_fail => {
                        n_fail += 1;
                        comfy_table::Cell::new("✗")
                            .fg(comfy_table::Color::Red)
                            .add_attribute(comfy_table::Attribute::Bold)
                    }
                    Some(d) if d.any_pass => {
                        n_pass += 1;
                        comfy_table::Cell::new("✓")
                            .fg(comfy_table::Color::Green)
                            .add_attribute(comfy_table::Attribute::Bold)
                    }
                    Some(d) => {
                        debug_assert!(d.any_skip);
                        comfy_table::Cell::new("-")
                    }
                };
                line.push(cell);
            }
            table.add_row(line);
        }
        out.push_str(&format!("\n{sched}: {n_pass} ✅  {n_fail} ❌\n{table}\n"));
    }
    Some(out)
}

/// Render one `verified_insns` table per declared scheduler. Within each
/// scheduler's section: rows = BPF program, columns = kernel version, and
/// each cell is that program's `verified_insns` for the (scheduler,
/// kernel) summarized ACROSS the topologies that ran it — a single number
/// when topology-invariant, `lo..hi` when it varies (`-` when that program
/// reported no stats on that kernel).
///
/// `verified_insns` is the verifier's PROCESSED-instruction count
/// (`env->insn_processed`) — fixed per load, but NOT topology-invariant
/// (a scheduler whose verification path depends on topology-derived
/// `.rodata`, e.g. `nr_cpus`, processes a different count per topology).
/// So topology is folded into the cell as a range rather than shown as its
/// own (usually all-identical) axis; the axes it genuinely varies on — BPF
/// program (y) and kernel version (x) — are the table axes, sectioned per
/// declared scheduler. Program is the ROW axis because a scheduler declares
/// many programs (scx_lavd has ~26) and the count is unbounded, so rows
/// scale down the page without wrapping; kernel is the COLUMN axis because
/// a sweep runs only a handful of kernels, so the columns stay narrow
/// enough to read in a CI log. Identical-binary declarations are sectioned
/// separately on purpose (they are run separately). Returns `None` when no
/// record carries any per-program stats (the caller prints nothing).
///
/// A kernel that ran a scheduler but produced NO stats at all — e.g.
/// every cell on it died during BPF load, so no program existed to
/// introspect — still gets a column (all cells `-`) as long as the
/// scheduler has at least one program row from some OTHER kernel.
/// Without it the stats-less kernel would silently vanish from the
/// table, hiding that it ran and failed to load; the explicit all-`-`
/// column makes that absence visible. Kernel columns therefore come from
/// every kernel that ran the scheduler, not only those that produced stats.
///
/// Schedulers, kernels, and programs are BTree-sorted so the same run
/// renders the same output (shell-pipeline stable). The range drops which
/// topology produced which count; a per-topology breakdown is a separate
/// detailed view, not this summary.
pub fn render_instruction_count_tables(records: &[VerifierCellRecord]) -> Option<String> {
    use std::collections::{BTreeMap, BTreeSet};
    // scheduler -> kernel -> program -> (min, max) verified_insns across
    // the topologies that ran it. Topology is folded into the (min, max)
    // range: a flat scheduler has min == max (one number), a
    // topology-sensitive one has min < max (`lo..hi`).
    type VerifiedInsnSpans = BTreeMap<String, BTreeMap<String, BTreeMap<String, (u32, u32)>>>;
    let mut by_sched: VerifiedInsnSpans = BTreeMap::new();
    // scheduler -> the union of program names it reported (the columns).
    let mut sched_progs: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    // scheduler -> every kernel that has a record for it, whether or not
    // it contributed stats. A kernel present here but absent from
    // `by_sched[sched]` ran the scheduler yet produced no program stats,
    // so it becomes an all-`-` column rather than vanishing.
    let mut sched_all_kernels: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for r in records {
        sched_all_kernels
            .entry(r.scheduler.clone())
            .or_default()
            .insert(r.kernel.clone());
        for s in &r.stats {
            let span = by_sched
                .entry(r.scheduler.clone())
                .or_default()
                .entry(r.kernel.clone())
                .or_default()
                .entry(s.name.clone())
                .or_insert((s.verified_insns, s.verified_insns));
            span.0 = span.0.min(s.verified_insns);
            span.1 = span.1.max(s.verified_insns);
            sched_progs
                .entry(r.scheduler.clone())
                .or_default()
                .insert(s.name.clone());
        }
    }
    if by_sched.is_empty() {
        return None;
    }

    let mut out = String::from(
        "\nverifier verified_insns (per scheduler; rows: BPF program, cols: kernel, \
         cell: range across topologies):\n",
    );
    for (sched, kernels) in &by_sched {
        let progs = &sched_progs[sched];
        // EVERY kernel that ran this scheduler, not only the ones that
        // produced stats, so a load-failure kernel surfaces as an all-`-`
        // column instead of vanishing.
        let all_kernels = &sched_all_kernels[sched];
        let mut table = crate::cli::new_bordered_table();
        let mut header: Vec<String> = vec!["program".to_string()];
        for kernel in all_kernels {
            header.push(kernel.clone());
        }
        table.set_header(header);
        for p in progs {
            let mut line: Vec<String> = vec![p.clone()];
            for kernel in all_kernels {
                let text = match kernels.get(kernel).and_then(|m| m.get(p)) {
                    Some((lo, hi)) if lo == hi => lo.to_string(),
                    Some((lo, hi)) => format!("{lo}..{hi}"),
                    None => "-".to_string(),
                };
                line.push(text);
            }
            table.add_row(line);
        }
        out.push_str(&format!("\n{sched}:\n{table}\n"));
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ownership_scheduler(name: &str) -> crate::test_support::SchedulerJson {
        let mut scheduler = crate::test_support::SchedulerJson::from_scheduler(
            &crate::test_support::Scheduler::EEVDF,
        );
        scheduler.name = name.to_string();
        scheduler
    }

    fn write_ownership_manifest(dir: &Path, manifest: &VerifierCellOwnershipManifest) -> PathBuf {
        let path = dir.join("cell-ownership.json");
        std::fs::write(
            &path,
            serde_json::to_vec(manifest).expect("serialize ownership manifest"),
        )
        .expect("write ownership manifest");
        path
    }

    #[test]
    fn cell_ownership_allows_exactly_the_canonical_elected_executable() {
        let dir = tempfile::tempdir().expect("ownership tempdir");
        let owner = dir.path().join("owner");
        let non_owner = dir.path().join("non-owner");
        std::fs::write(&owner, b"owner").expect("write owner");
        std::fs::write(&non_owner, b"non-owner").expect("write non-owner");
        let owner = std::fs::canonicalize(owner).expect("canonical owner");
        let non_owner = std::fs::canonicalize(non_owner).expect("canonical non-owner");
        let owner_alias = dir.path().join("owner-alias");
        std::os::unix::fs::symlink(&owner, &owner_alias).expect("symlink owner alias");
        let scheduler = ownership_scheduler("shared");
        let manifest_path = write_ownership_manifest(
            dir.path(),
            &VerifierCellOwnershipManifest {
                version: VERIFIER_CELL_OWNERSHIP_MANIFEST_VERSION,
                entries: vec![VerifierCellOwnershipEntry {
                    scheduler: scheduler.clone(),
                    executable: owner.clone(),
                }],
            },
        );

        let elected = cell_ownership_from_manifest_path(Some(manifest_path.as_os_str()), &owner)
            .expect("load owner view")
            .expect("present manifest");
        let rejected =
            cell_ownership_from_manifest_path(Some(manifest_path.as_os_str()), &non_owner)
                .expect("load non-owner view")
                .expect("present manifest");
        let elected_through_alias =
            cell_ownership_from_manifest_path(Some(manifest_path.as_os_str()), &owner_alias)
                .expect("load owner alias view")
                .expect("present manifest");
        assert!(elected.owns_scheduler(&scheduler).expect("owner lookup"));
        assert!(
            elected_through_alias
                .owns_scheduler(&scheduler)
                .expect("owner alias lookup"),
            "current_exe aliases must compare by their canonical executable path",
        );
        assert!(
            !rejected
                .owns_scheduler(&scheduler)
                .expect("non-owner lookup")
        );
        let mut stale_child = scheduler.clone();
        stale_child.kargs.push("identity=stale-child".into());
        assert!(
            elected.owns_scheduler(&stale_child).is_err(),
            "matching only the scheduler name cannot establish ownership",
        );
    }

    #[test]
    fn present_cell_ownership_manifest_fails_closed() {
        let dir = tempfile::tempdir().expect("ownership tempdir");
        let owner = dir.path().join("owner");
        std::fs::write(&owner, b"owner").expect("write owner");
        let owner = std::fs::canonicalize(owner).expect("canonical owner");
        let scheduler = ownership_scheduler("shared");
        assert!(
            cell_ownership_from_manifest_path(None, &owner)
                .expect("absent manifest")
                .is_none(),
            "only an absent environment value retains direct invocation",
        );
        assert!(
            cell_ownership_from_manifest_path(Some(OsStr::new("")), &owner).is_err(),
            "an explicitly empty manifest path is still present and must fail",
        );

        let wrong_version_path = write_ownership_manifest(
            dir.path(),
            &VerifierCellOwnershipManifest {
                version: VERIFIER_CELL_OWNERSHIP_MANIFEST_VERSION + 1,
                entries: Vec::new(),
            },
        );
        assert!(
            cell_ownership_from_manifest_path(Some(wrong_version_path.as_os_str()), &owner)
                .is_err(),
            "unknown ownership schemas cannot fall back to shared ownership",
        );

        let empty_path = write_ownership_manifest(
            dir.path(),
            &VerifierCellOwnershipManifest {
                version: VERIFIER_CELL_OWNERSHIP_MANIFEST_VERSION,
                entries: Vec::new(),
            },
        );
        let empty = cell_ownership_from_manifest_path(Some(empty_path.as_os_str()), &owner)
            .expect("load empty manifest")
            .expect("present manifest");
        assert!(
            empty.owns_scheduler(&scheduler).is_err(),
            "a selected scheduler missing from a present manifest cannot become shared",
        );

        let duplicate_path = write_ownership_manifest(
            dir.path(),
            &VerifierCellOwnershipManifest {
                version: VERIFIER_CELL_OWNERSHIP_MANIFEST_VERSION,
                entries: vec![
                    VerifierCellOwnershipEntry {
                        scheduler: scheduler.clone(),
                        executable: owner.clone(),
                    },
                    VerifierCellOwnershipEntry {
                        scheduler: scheduler.clone(),
                        executable: owner.clone(),
                    },
                ],
            },
        );
        assert!(
            cell_ownership_from_manifest_path(Some(duplicate_path.as_os_str()), &owner).is_err(),
            "ambiguous scheduler owners cannot be accepted",
        );

        let noncanonical_owner = dir.path().join("owner-symlink");
        std::os::unix::fs::symlink(&owner, &noncanonical_owner).expect("symlink owner");
        let noncanonical_path = write_ownership_manifest(
            dir.path(),
            &VerifierCellOwnershipManifest {
                version: VERIFIER_CELL_OWNERSHIP_MANIFEST_VERSION,
                entries: vec![VerifierCellOwnershipEntry {
                    scheduler: scheduler.clone(),
                    executable: noncanonical_owner,
                }],
            },
        );
        assert!(
            cell_ownership_from_manifest_path(Some(noncanonical_path.as_os_str()), &owner).is_err(),
            "owner paths must be canonical rather than merely equivalent",
        );

        let malformed_path = dir.path().join("malformed-ownership.json");
        std::fs::write(&malformed_path, b"{").expect("write malformed manifest");
        assert!(
            cell_ownership_from_manifest_path(Some(malformed_path.as_os_str()), &owner).is_err(),
            "malformed JSON cannot fall back to shared ownership",
        );
        assert!(
            cell_ownership_from_manifest_path(
                Some(dir.path().join("missing.json").as_os_str()),
                &owner,
            )
            .is_err(),
            "a missing present-manifest path cannot fall back to shared ownership",
        );
    }

    #[test]
    fn declared_scheduler_launch_plan_preserves_boot_environment_and_order() {
        use crate::test_support::{Scheduler, Sysctl};

        const SYSCTLS: &[Sysctl] = &[
            Sysctl::new("kernel.ktstr_first", "1"),
            Sysctl::new("kernel.ktstr_second", "2"),
        ];
        const KARGS: &[&str] = &["ktstr_first_karg=one", "ktstr_second_karg=two"];
        const SCHED_ARGS: &[&str] = &["--mode", "verifier"];
        let scheduler = Scheduler::named("launch-plan")
            .sysctls(SYSCTLS)
            .kargs(KARGS)
            .cgroup_parent("/ktstr-verifier")
            .config_file("/host/configs/verifier.toml")
            .sched_args(SCHED_ARGS);

        let plan = VerifierSchedulerLaunchPlan::from_scheduler(&scheduler);
        assert_eq!(
            plan.cmdline_extra,
            "sysctl.kernel.ktstr_first=1 sysctl.kernel.ktstr_second=2 \
             ktstr_first_karg=one ktstr_second_karg=two"
        );
        assert_eq!(
            plan.include_files,
            vec![(
                "include-files/verifier.toml".to_string(),
                PathBuf::from("/host/configs/verifier.toml"),
            )]
        );
        assert_eq!(
            plan.sched_args,
            [
                "--config",
                "/include-files/verifier.toml",
                "--mode",
                "verifier"
            ]
        );
        assert_eq!(
            plan.scheduler_cgroup_parent.as_deref(),
            Some("/ktstr-verifier")
        );
    }

    #[test]
    fn inline_config_definition_without_test_content_is_not_fabricated() {
        use crate::test_support::Scheduler;

        let scheduler = Scheduler::named("inline-template")
            .config_file_def("--config={file}", "/include-files/inline.json")
            .sched_args(&["--mode", "baseline"]);
        let plan = VerifierSchedulerLaunchPlan::from_scheduler(&scheduler);
        assert!(plan.include_files.is_empty());
        assert_eq!(plan.sched_args, ["--mode", "baseline"]);
    }

    #[test]
    fn direct_verifier_api_retains_explicit_cli_only_launch_shape() {
        let plan = VerifierSchedulerLaunchPlan::direct(&["--direct".into(), "value".into()]);
        assert_eq!(plan.sched_args, ["--direct", "value"]);
        assert!(plan.cmdline_extra.is_empty());
        assert!(plan.include_files.is_empty());
        assert!(plan.scheduler_cgroup_parent.is_none());
    }

    #[test]
    fn terminal_scheduler_stream_requires_completion_and_overrides_live_atomically() {
        use crate::vmm::host_comms::BulkDrainResult;
        use crate::vmm::wire::{
            MSG_TYPE_SCHED_STDERR, MSG_TYPE_SCHED_STDERR_FINAL, ShmEntry,
            encode_sched_stream_final_chunk,
        };

        let entry = |msg_type, payload: Vec<u8>| ShmEntry {
            msg_type,
            payload,
            crc_ok: true,
        };
        let live = || {
            entry(
                MSG_TYPE_SCHED_STDERR,
                b"live prefix missing one chunk\n".to_vec(),
            )
        };

        let old_guest = BulkDrainResult {
            entries: vec![live()],
        };
        assert_eq!(
            scheduler_stream_with_terminal_fallback(
                Some(&old_guest),
                MSG_TYPE_SCHED_STDERR,
                MSG_TYPE_SCHED_STDERR_FINAL,
            ),
            "live prefix missing one chunk\n",
            "absence of new frames must preserve old-guest live semantics"
        );

        let complete = b"authoritative complete scheduler stderr\n";
        let split = 14usize;
        let recovered = BulkDrainResult {
            entries: vec![
                live(),
                entry(
                    MSG_TYPE_SCHED_STDERR_FINAL,
                    encode_sched_stream_final_chunk(complete.len() as u64, 0, &complete[..split]),
                ),
                entry(
                    MSG_TYPE_SCHED_STDERR_FINAL,
                    encode_sched_stream_final_chunk(
                        complete.len() as u64,
                        split as u64,
                        &complete[split..],
                    ),
                ),
            ],
        };
        assert_eq!(
            scheduler_stream_with_terminal_fallback(
                Some(&recovered),
                MSG_TYPE_SCHED_STDERR,
                MSG_TYPE_SCHED_STDERR_FINAL,
            ),
            String::from_utf8_lossy(complete),
            "one completion-proven replay must replace the partial live stream"
        );

        let interrupted = BulkDrainResult {
            entries: vec![
                live(),
                entry(
                    MSG_TYPE_SCHED_STDERR_FINAL,
                    encode_sched_stream_final_chunk(complete.len() as u64, 0, &complete[..split]),
                ),
            ],
        };
        assert_eq!(
            scheduler_stream_with_terminal_fallback(
                Some(&interrupted),
                MSG_TYPE_SCHED_STDERR,
                MSG_TYPE_SCHED_STDERR_FINAL,
            ),
            "live prefix missing one chunk\n",
            "an interrupted terminal transaction must never replace live output"
        );

        let explicit_empty = BulkDrainResult {
            entries: vec![
                live(),
                entry(
                    MSG_TYPE_SCHED_STDERR_FINAL,
                    encode_sched_stream_final_chunk(0, 0, &[]),
                ),
            ],
        };
        assert_eq!(
            scheduler_stream_with_terminal_fallback(
                Some(&explicit_empty),
                MSG_TYPE_SCHED_STDERR,
                MSG_TYPE_SCHED_STDERR_FINAL,
            ),
            "",
            "a complete empty transaction is distinct from no final transaction"
        );
    }

    // -----------------------------------------------------------------------
    // per-cell result capture + summary table
    // -----------------------------------------------------------------------

    /// A malformed cell name is skipped (no record); a well-formed
    /// 3-segment cell records (scheduler/kernel/topology); and a nextest
    /// RETRY of the same cell overwrites its own prior record so the
    /// FINAL outcome wins (fail-then-pass -> PASS).
    #[test]
    fn cell_record_write_read_roundtrip_and_retry_overwrites() {
        let dir = tempfile::tempdir().expect("result tempdir");
        // Malformed: no verifier/ prefix, and a 2-segment name (no
        // <preset> after the kernel) — both skipped.
        write_cell_record(dir.path(), "not_a_cell", true, false, &[]);
        write_cell_record(dir.path(), "verifier/only/two", true, false, &[]);
        // Well-formed cell: fail, then a retry passes -> overwrites.
        let name = "verifier/scx_a/kernel_6_14/4cpu-1llc-nosmt";
        write_cell_record(dir.path(), name, false, false, &[]);
        // The retry passes and carries per-program verified_insns, so the
        // final record has both the PASS outcome and the stats.
        let stats = [
            ProgStats {
                name: "ktstr_dispatch".into(),
                verified_insns: 321,
            },
            ProgStats {
                name: "ktstr_enqueue".into(),
                verified_insns: 123,
            },
        ];
        write_cell_record(dir.path(), name, true, false, &stats);
        let recs = read_cell_records(dir.path());
        assert_eq!(
            recs.len(),
            1,
            "malformed names skipped; the retry overwrote its own record (one file): {recs:?}",
        );
        assert_eq!(
            std::fs::read_dir(dir.path())
                .expect("read result directory")
                .count(),
            1,
            "atomic retry replacement must leave no temporary record behind",
        );
        assert_eq!(recs[0].scheduler, "scx_a");
        assert_eq!(recs[0].kernel, "kernel_6_14");
        assert_eq!(recs[0].topology, "4cpu-1llc-nosmt");
        assert!(
            recs[0].passed,
            "final retry outcome (PASS) wins over the earlier FAIL"
        );
        // Per-program verified_insns survive the JSON roundtrip and reflect
        // the final (retry) write, not the earlier stat-less fail.
        assert_eq!(recs[0].stats, stats, "stats roundtrip via serde");
    }

    /// Distinct valid scheduler names may differ only at punctuation which a
    /// filesystem sanitizer would collapse. Their cells can finish
    /// concurrently, so the raw-name content address must keep both records
    /// independent while preserving same-cell retry overwrite semantics.
    #[test]
    fn concurrent_cell_records_do_not_alias_sanitized_scheduler_names() {
        use std::sync::{Arc, Barrier};

        let dir = tempfile::tempdir().expect("result tempdir");
        let hyphenated = "verifier/foo-bar/kernel_6_14/4cpu-1llc-nosmt";
        let underscored = "verifier/foo_bar/kernel_6_14/4cpu-1llc-nosmt";

        assert_ne!(
            cell_record_filename(hyphenated),
            cell_record_filename(underscored),
            "the raw cell identity, not a lossy sanitized spelling, keys the record",
        );

        let barrier = Arc::new(Barrier::new(3));
        std::thread::scope(|scope| {
            for (name, verified_insns) in [(hyphenated, 111), (underscored, 222)] {
                let barrier = Arc::clone(&barrier);
                let dir = dir.path();
                scope.spawn(move || {
                    let stats = [ProgStats {
                        name: "ktstr_dispatch".into(),
                        verified_insns,
                    }];
                    barrier.wait();
                    write_cell_record(dir, name, true, false, &stats);
                });
            }
            barrier.wait();
        });

        let mut records = read_cell_records(dir.path());
        records.sort_by(|a, b| a.scheduler.cmp(&b.scheduler));
        assert_eq!(
            records.len(),
            2,
            "concurrent punctuation-distinct cells must retain two JSON records: {records:?}",
        );
        assert_eq!(records[0].scheduler, "foo-bar");
        assert_eq!(records[0].stats[0].verified_insns, 111);
        assert_eq!(records[1].scheduler, "foo_bar");
        assert_eq!(records[1].stats[0].verified_insns, 222);
    }

    /// A verifier SKIP is distinct from both PASS and FAIL in the persisted
    /// record, including across JSON round-trip.
    #[test]
    fn cell_record_skip_roundtrips() {
        let dir = std::env::temp_dir().join(format!("ktstr-verif-skip-rec-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("mk temp dir");
        write_cell_record(
            &dir,
            "verifier/scx_a/kernel_steamos/192cpu-11llc-smt",
            false,
            true,
            &[],
        );
        let recs = read_cell_records(&dir);
        assert_eq!(recs.len(), 1);
        assert!(!recs[0].passed);
        assert!(recs[0].skipped);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Per-scheduler PASS/FAIL grids: one section per declared scheduler
    /// (rows = topology, cols = kernel), each a tally line plus a bordered
    /// grid whose cells are single-width ✓/✗ glyphs (emoji only on the
    /// tally line). Covers a multi-scheduler / multi-kernel run, a `-`
    /// cell (no record for the triple), the defensive duplicate-record
    /// case (a pass AND a fail for one triple -> ✗, any failure means
    /// failure), and the box-drawing borders of the bordered table. An
    /// empty record set renders nothing.
    ///
    /// Cell assertions use `contains` on the glyph/word so they hold
    /// whether or not ANSI escapes are present — the color policy styles
    /// cells on a TTY and under `GITHUB_ACTIONS`/`FORCE_COLOR`, so this
    /// test must not assert the ABSENCE of escapes (it runs under GHA in
    /// CI). Tally lines are built with plain `format!` (never styled), so
    /// exact-match assertions on them are safe everywhere.
    #[test]
    fn render_result_table_per_scheduler_grids_tally_and_borders() {
        let recs = vec![
            // scx_a across two kernels and two topologies; the
            // (128cpu-4llc-smt, kernel_6_15) cell has no record -> `-`.
            VerifierCellRecord {
                scheduler: "scx_a".into(),
                kernel: "kernel_6_14".into(),
                topology: "4cpu-1llc-nosmt".into(),
                passed: true,
                skipped: false,
                stats: vec![],
            },
            VerifierCellRecord {
                scheduler: "scx_a".into(),
                kernel: "kernel_6_15".into(),
                topology: "4cpu-1llc-nosmt".into(),
                passed: false,
                skipped: false,
                stats: vec![],
            },
            VerifierCellRecord {
                scheduler: "scx_a".into(),
                kernel: "kernel_6_14".into(),
                topology: "128cpu-4llc-smt".into(),
                passed: true,
                skipped: false,
                stats: vec![],
            },
            // Explicit kernel/topology SKIP. It renders the same neutral
            // `-` as an inapplicable matrix intersection and contributes
            // to neither the pass nor fail tally.
            VerifierCellRecord {
                scheduler: "scx_a".into(),
                kernel: "kernel_6_15".into(),
                topology: "128cpu-4llc-smt".into(),
                passed: false,
                skipped: true,
                stats: vec![],
            },
            // scx_b: a duplicate (scheduler, kernel, topology) triple with
            // one pass AND one fail -> ✗ (any failure means failure).
            VerifierCellRecord {
                scheduler: "scx_b".into(),
                kernel: "kernel_6_14".into(),
                topology: "4cpu-1llc-nosmt".into(),
                passed: true,
                skipped: false,
                stats: vec![],
            },
            VerifierCellRecord {
                scheduler: "scx_b".into(),
                kernel: "kernel_6_14".into(),
                topology: "4cpu-1llc-nosmt".into(),
                passed: false,
                skipped: false,
                stats: vec![],
            },
        ];
        let out = render_result_table(&recs).expect("non-empty records -> Some");
        assert!(
            out.contains("verifier results (per scheduler; rows: topology, cols: kernel):"),
            "per-scheduler header: {out}"
        );
        // The grid uses the bordered (UTF8_FULL) table, so box-drawing
        // characters must appear — distinct from every other verifier
        // table, which stays borderless.
        assert!(
            ['│', '─', '┼', '├', '┤', '┬', '┴']
                .iter()
                .any(|c| out.contains(*c)),
            "bordered grid must carry box-drawing chars: {out}"
        );
        // Split into the two scheduler sections at scx_b's tally.
        let (a_sec, b_sec) = out.split_once("scx_b:").expect("scx_b section present");
        // scx_a tally: 2 ✅ + 1 ❌ (emoji live ONLY on the tally line).
        let a_tally = a_sec
            .lines()
            .find(|l| l.trim_start().starts_with("scx_a:"))
            .expect("scx_a tally line");
        assert_eq!(
            a_tally.trim(),
            "scx_a: 2 ✅  1 ❌",
            "scx_a tally counts grid cells: {a_tally}"
        );
        // Kernel is a grid axis (a column per kernel), not folded away.
        assert!(
            a_sec.contains("kernel_6_14") && a_sec.contains("kernel_6_15"),
            "scx_a kernel columns present: {a_sec}"
        );
        // The (128cpu-4llc-smt, kernel_6_15) triple explicitly SKIPped -> `-`,
        // alongside the (128cpu-4llc-smt, kernel_6_14) ✓. The space-padded
        // " - " match avoids the hyphens inside topology names.
        let a_large = a_sec
            .lines()
            .find(|l| l.contains("128cpu-4llc-smt"))
            .expect("scx_a 128cpu-4llc-smt row");
        assert!(
            a_large.contains('✓') && a_large.contains(" - "),
            "skipped cell renders `-`: {a_large}"
        );
        // 4cpu-1llc-nosmt row: kernel_6_14 ✓, kernel_6_15 ✗ — the failing
        // kernel is named by its own column, no separate failing list.
        let a_tiny = a_sec
            .lines()
            .find(|l| l.contains("4cpu-1llc-nosmt"))
            .expect("scx_a 4cpu-1llc-nosmt row");
        assert!(
            a_tiny.contains('✓') && a_tiny.contains('✗'),
            "4cpu-1llc-nosmt row shows the pass and the fail per kernel: {a_tiny}"
        );
        // scx_b: the duplicate pass+fail triple is ONE cell, counted as a
        // fail — any failure means failure, no partial state.
        assert!(
            b_sec.starts_with(" 0 ✅  1 ❌"),
            "duplicate pass+fail triple tallies as one fail: {b_sec}"
        );
        let b_tiny = b_sec
            .lines()
            .find(|l| l.contains("4cpu-1llc-nosmt"))
            .expect("scx_b 4cpu-1llc-nosmt row");
        assert!(
            b_tiny.contains('✗') && !b_tiny.contains('✓'),
            "a pass+fail duplicate-record cell renders ✗: {b_tiny}"
        );
        // Summary cells contain only their verdict glyph. Runtime
        // measurements belong in the detailed per-cell output and must
        // not widen or misalign this matrix.
        assert!(
            !out.contains('.'),
            "summary grid must not render runtime measurements: {out}"
        );
        // Emoji stay OFF the grid rows (they drift box-drawing columns in
        // GitHub's log viewer); the retired glyphs and flat list are gone.
        for row in out.lines().filter(|l| l.contains('│')) {
            assert!(
                !row.contains('✅') && !row.contains('❌'),
                "grid rows carry ✓/✗ glyphs, not emoji: {row}"
            );
        }
        assert!(
            !out.contains('❎') && !out.contains('🟡') && !out.contains("failing combinations"),
            "retired ❎ / 🟡 / failing-combinations list must not appear: {out}"
        );
        assert!(render_result_table(&[]).is_none(), "empty -> None");
    }

    /// Result records from v0.40 carried a `dilation` field. Serde ignores
    /// that now-unknown field during a rolling upgrade, and the summary
    /// still renders a glyph-only verdict cell.
    #[test]
    fn legacy_record_dilation_is_ignored_by_summary() {
        let record: VerifierCellRecord = serde_json::from_str(
            r#"{
                "scheduler": "scx_a",
                "kernel": "kernel_6_14",
                "topology": "4cpu-1llc-nosmt",
                "passed": true,
                "skipped": false,
                "stats": [],
                "dilation": 11.62
            }"#,
        )
        .expect("v0.40 record with dilation still deserializes");

        let out = render_result_table(&[record]).expect("record renders a summary");
        assert!(out.contains('✓'), "pass glyph is present: {out}");
        assert!(
            !out.contains("11.62"),
            "legacy dilation must not enter the summary cell: {out}"
        );
    }

    /// Per-scheduler verified_insns tables: one section per declared
    /// scheduler; within it rows = BPF program, columns = kernel version,
    /// each cell that program's verified_insns across the topologies that
    /// ran it — a single number when topology-invariant, `lo..hi` when it
    /// varies. A (kernel, program) that reported no stats shows `-`; a
    /// kernel that ran a scheduler but produced NO stats at all still gets
    /// an all-`-` column; an empty record set renders nothing.
    #[test]
    fn instruction_count_tables_per_scheduler_kernel_program_range() {
        let recs = vec![
            // scx_a / kernel_6_14 on two topologies with IDENTICAL counts
            // -> the cell collapses to a single number (topology-flat).
            VerifierCellRecord {
                scheduler: "scx_a".into(),
                kernel: "kernel_6_14".into(),
                topology: "4cpu-1llc-nosmt".into(),
                passed: true,
                skipped: false,
                stats: vec![
                    ProgStats {
                        name: "ktstr_dispatch".into(),
                        verified_insns: 128,
                    },
                    ProgStats {
                        name: "ktstr_enqueue".into(),
                        verified_insns: 64,
                    },
                ],
            },
            VerifierCellRecord {
                scheduler: "scx_a".into(),
                kernel: "kernel_6_14".into(),
                topology: "128cpu-4llc-smt".into(),
                passed: true,
                skipped: false,
                stats: vec![
                    ProgStats {
                        name: "ktstr_dispatch".into(),
                        verified_insns: 128,
                    },
                    ProgStats {
                        name: "ktstr_enqueue".into(),
                        verified_insns: 64,
                    },
                ],
            },
            // scx_a / kernel_6_15: ktstr_dispatch DIFFERS across topologies
            // -> `lo..hi` range; ktstr_enqueue is absent on this kernel
            // -> `-` in the ktstr_enqueue row's kernel_6_15 column.
            VerifierCellRecord {
                scheduler: "scx_a".into(),
                kernel: "kernel_6_15".into(),
                topology: "4cpu-1llc-nosmt".into(),
                passed: true,
                skipped: false,
                stats: vec![ProgStats {
                    name: "ktstr_dispatch".into(),
                    verified_insns: 130,
                }],
            },
            VerifierCellRecord {
                scheduler: "scx_a".into(),
                kernel: "kernel_6_15".into(),
                topology: "128cpu-4llc-smt".into(),
                passed: true,
                skipped: false,
                stats: vec![ProgStats {
                    name: "ktstr_dispatch".into(),
                    verified_insns: 150,
                }],
            },
            // scx_b: a separate section (its own declaration).
            VerifierCellRecord {
                scheduler: "scx_b".into(),
                kernel: "kernel_6_14".into(),
                topology: "4cpu-1llc-nosmt".into(),
                passed: true,
                skipped: false,
                stats: vec![ProgStats {
                    name: "ktstr_dispatch".into(),
                    verified_insns: 200,
                }],
            },
            // scx_a / kernel_6_16: a record but NO stats — every cell died
            // during BPF load, so no program existed to introspect. It must
            // still surface as an all-`-` column, not vanish from the table.
            VerifierCellRecord {
                scheduler: "scx_a".into(),
                kernel: "kernel_6_16".into(),
                topology: "4cpu-1llc-nosmt".into(),
                passed: false,
                skipped: false,
                stats: vec![],
            },
        ];
        let out = render_instruction_count_tables(&recs).expect("stats present -> Some");
        // One section per declared scheduler.
        assert!(
            out.contains("scx_a:") && out.contains("scx_b:"),
            "one section per declared scheduler: {out}"
        );
        // Rows = BPF programs; columns = kernel version.
        assert!(
            out.contains("ktstr_dispatch") && out.contains("ktstr_enqueue"),
            "BPF-program rows: {out}"
        );
        assert!(
            out.contains("kernel_6_14") && out.contains("kernel_6_15"),
            "kernel-version columns: {out}"
        );
        // Topology folded into the cell as a range: flat -> "128",
        // varies across topologies -> "130..150".
        assert!(
            out.contains("128"),
            "topology-flat cell is a single number: {out}"
        );
        assert!(
            out.contains("130..150"),
            "topology-varying cell is a lo..hi range: {out}"
        );
        assert!(
            out.contains("64") && out.contains("200"),
            "other counts render: {out}"
        );
        // ktstr_enqueue reported no stats on kernel_6_15 -> `-`.
        assert!(
            out.contains('-'),
            "a (kernel, program) with no stats renders '-': {out}"
        );
        // kernel_6_16 ran scx_a but produced NO stats -> it stays as a
        // column (named in the header) rather than vanishing. Its all-`-`
        // column shows up as a `-` in every program row: the ktstr_enqueue
        // row is absent on both kernel_6_15 and kernel_6_16, so it carries
        // exactly two `-` cells (the box-drawing borders use `─`/`│`, not
        // ASCII `-`, so they do not count).
        let header = out
            .lines()
            .find(|l| l.contains("program") && l.contains("kernel_6_16"))
            .expect("stats-less kernel kept as a column in the header");
        assert!(
            header.contains("kernel_6_14") && header.contains("kernel_6_15"),
            "every kernel that ran is a column: {header}"
        );
        let enqueue_row = out
            .lines()
            .find(|l| l.contains("ktstr_enqueue"))
            .expect("ktstr_enqueue row present");
        assert_eq!(
            enqueue_row.matches('-').count(),
            2,
            "ktstr_enqueue is absent on kernel_6_15 and kernel_6_16: {enqueue_row}"
        );
        // Topology is NOT a table axis (folded into the range), so no
        // topology label appears in the output.
        assert!(
            !out.contains("4cpu-1llc-nosmt") && !out.contains("128cpu-4llc-smt"),
            "topology is not a table axis: {out}"
        );

        // No record carries stats -> nothing to render.
        let bare = vec![VerifierCellRecord {
            scheduler: "scx_a".into(),
            kernel: "kernel_6_14".into(),
            topology: "4cpu-1llc-nosmt".into(),
            passed: false,
            skipped: false,
            stats: vec![],
        }];
        assert!(
            render_instruction_count_tables(&bare).is_none(),
            "no stats -> None"
        );
    }

    /// classify_run_outcome maps nextest's exit + the records into a
    /// [`RunOutcome`]: a successful run with records is `Success`; a
    /// successful-but-empty run is a diagnosed `Failed` (friendly
    /// no-such-scheduler when --scheduler was set, no-cells-ran otherwise);
    /// a nonzero exit whose failed cell the grid ALREADY shows is a
    /// message-less `SilentExit` carrying nextest's code, while a nonzero
    /// exit with NO failed cell (build/internal failure) or a signal exit
    /// stays a descriptive `Failed`.
    #[test]
    fn classify_run_outcome_cases() {
        use RunOutcome::{Failed, SilentExit, Success};

        // Records present + success -> Success regardless of --scheduler.
        assert_eq!(
            classify_run_outcome(true, false, false, None, Some(0)),
            Success
        );
        assert_eq!(
            classify_run_outcome(true, false, false, Some("ktstr_sched"), Some(0)),
            Success
        );

        // Success + empty + --scheduler -> friendly "no such scheduler".
        // Reachable ONLY because --no-tests=pass turns a 0-cell match into
        // exit 0; under the old `auto` default a 0-match exited 4 -> the
        // failure arm, leaving this message dead.
        let Failed(e) = classify_run_outcome(true, true, false, Some("nope"), Some(0)) else {
            panic!("expected Failed");
        };
        assert!(
            e.contains("--scheduler \"nope\"") && e.contains("matched no verifier cell"),
            "scheduler-empty diagnostic: {e}"
        );
        assert!(
            e.contains("declared topology constraints or verifier-only exclusions")
                && !e.contains("host"),
            "scheduler-empty diagnostic must describe declaration gates, not host capacity: {e}"
        );

        // Success + empty + no --scheduler -> "no cells ran" diagnosis
        // (must NOT silently succeed under --no-tests=pass).
        let Failed(e) = classify_run_outcome(true, true, false, None, Some(0)) else {
            panic!("expected Failed");
        };
        assert!(
            e.contains("no verifier cells ran") && e.contains("declare_scheduler!"),
            "no-cells diagnostic: {e}"
        );
        assert!(
            e.contains("topology constraints or verifier-only exclusions") && !e.contains("host"),
            "no-cells diagnostic must describe declaration gates, not host capacity: {e}"
        );

        // Nonzero exit WITH a failed cell + numeric code -> SilentExit: the
        // printed grid already shows the ✗, so carry nextest's code out
        // without a redundant stderr line that would interleave mid-report.
        assert_eq!(
            classify_run_outcome(false, false, true, None, Some(100)),
            SilentExit(100)
        );

        // Nonzero exit with NO failed cell (build/internal failure) keeps
        // the descriptive message — nothing in the grid to point at.
        let Failed(e) = classify_run_outcome(false, true, false, None, Some(101)) else {
            panic!("expected Failed");
        };
        assert_eq!(e, "cargo nextest run exited with 101");

        // A signal exit (no numeric code) keeps the message EVEN with a
        // failed cell present — there is no code to carry out silently.
        let Failed(e) = classify_run_outcome(false, false, true, None, None) else {
            panic!("expected Failed");
        };
        assert_eq!(e, "cargo nextest run exited with signal");
    }

    /// build_nextest_args carries the flags that make the friendly
    /// diagnostic reachable: `--run-ignored all` (cells are ignore-gated)
    /// and `--no-tests pass` (a 0-cell selection exits 0 so
    /// classify_run_outcome runs instead of nextest's exit-4). Guards
    /// against a future edit silently dropping either, plus the
    /// profile-before-forwarded-args ordering.
    #[test]
    fn build_nextest_args_carries_load_bearing_flags() {
        let args = build_nextest_args(None, &[]);
        let ri = args
            .iter()
            .position(|a| a == "--run-ignored")
            .expect("--run-ignored present");
        assert_eq!(args[ri + 1], "all", "--run-ignored all");
        let nt = args
            .iter()
            .position(|a| a == "--no-tests")
            .expect("--no-tests present");
        assert_eq!(args[nt + 1], "pass", "--no-tests pass");
        assert!(
            args.iter()
                .any(|a| a == "test(/^verifier/) & !test(/^verifier::/)"),
            "verifier-cell filter present: {args:?}"
        );

        // --profile <NAME> is emitted before forwarded args so a forwarded
        // token cannot shadow it.
        let args = build_nextest_args(Some("ci"), &["--features".to_string(), "wprof".to_string()]);
        let p = args
            .iter()
            .position(|a| a == "--profile")
            .expect("--profile present");
        assert_eq!(args[p + 1], "ci");
        let f = args
            .iter()
            .position(|a| a == "--features")
            .expect("forwarded --features present");
        assert!(p < f, "profile emitted before forwarded args: {args:?}");
    }

    // -----------------------------------------------------------------------
    // scheduler attach verdict
    // -----------------------------------------------------------------------

    /// attach_outcome_from_messages positive-confirmation rule:
    /// Died > NotAttached > SchedulerAttached > Unconfirmed. A
    /// PayloadStarting frame alone is Unconfirmed (FAIL), because the
    /// guest emits it for schedulerless runs. Corrupt / empty /
    /// non-LIFECYCLE / unknown frames are skipped.
    #[test]
    fn attach_outcome_from_lifecycle_frames() {
        use crate::vmm::host_comms::BulkDrainResult;
        use crate::vmm::wire::{LifecyclePhase, MSG_TYPE_LIFECYCLE, ShmEntry};

        let frame = |phase: LifecyclePhase, reason: &str| -> ShmEntry {
            let mut payload = vec![phase.wire_value()];
            payload.extend_from_slice(reason.as_bytes());
            ShmEntry {
                msg_type: MSG_TYPE_LIFECYCLE,
                payload,
                crc_ok: true,
            }
        };
        let drain = |entries: Vec<ShmEntry>| BulkDrainResult { entries };

        // No frames at all -> Unconfirmed (guest vanished before any
        // phase; e.g. an early kernel panic that reboots via panic=-1).
        assert_eq!(
            attach_outcome_from_messages(None),
            AttachOutcome::Unconfirmed,
        );

        // Reached init but no definitive attach -> Unconfirmed.
        let init_only = drain(vec![frame(LifecyclePhase::InitStarted, "")]);
        assert_eq!(
            attach_outcome_from_messages(Some(&init_only)),
            AttachOutcome::Unconfirmed,
        );

        // Schedulerless runs also reach PayloadStarting. That is not
        // positive attach evidence.
        let schedulerless = drain(vec![
            frame(LifecyclePhase::InitStarted, ""),
            frame(LifecyclePhase::PayloadStarting, ""),
        ]);
        assert_eq!(
            attach_outcome_from_messages(Some(&schedulerless)),
            AttachOutcome::Unconfirmed,
        );

        // Only the explicit SchedulerAttached frame confirms attach.
        let attached = drain(vec![
            frame(LifecyclePhase::InitStarted, ""),
            frame(LifecyclePhase::SchedulerAttached, ""),
            frame(LifecyclePhase::PayloadStarting, ""),
        ]);
        assert_eq!(
            attach_outcome_from_messages(Some(&attached)),
            AttachOutcome::Attached,
        );

        // SchedulerNotAttached carries its reason suffix verbatim.
        let not_attached = drain(vec![frame(LifecyclePhase::SchedulerNotAttached, "timeout")]);
        assert_eq!(
            attach_outcome_from_messages(Some(&not_attached)),
            AttachOutcome::NotAttached("timeout".to_string()),
        );

        // A failure frame wins over a (defensively) co-present
        // SchedulerAttached.
        let fail_beats_positive = drain(vec![
            frame(LifecyclePhase::SchedulerAttached, ""),
            frame(LifecyclePhase::SchedulerNotAttached, "sysfs absent"),
        ]);
        assert_eq!(
            attach_outcome_from_messages(Some(&fail_beats_positive)),
            AttachOutcome::NotAttached("sysfs absent".to_string()),
        );

        // SchedulerDied wins over NotAttached, in BOTH orders.
        for entries in [
            vec![
                frame(LifecyclePhase::SchedulerNotAttached, "timeout"),
                frame(LifecyclePhase::SchedulerDied, ""),
            ],
            vec![
                frame(LifecyclePhase::SchedulerDied, ""),
                frame(LifecyclePhase::SchedulerNotAttached, "timeout"),
            ],
        ] {
            let d = drain(entries);
            assert_eq!(attach_outcome_from_messages(Some(&d)), AttachOutcome::Died);
        }

        // Died wins even over SchedulerAttached.
        let died_beats_positive = drain(vec![
            frame(LifecyclePhase::SchedulerAttached, ""),
            frame(LifecyclePhase::SchedulerDied, ""),
        ]);
        assert_eq!(
            attach_outcome_from_messages(Some(&died_beats_positive)),
            AttachOutcome::Died,
        );

        // Skipped frames (corrupt crc / empty payload / non-LIFECYCLE /
        // unknown discriminant) must NOT suppress a real
        // SchedulerAttached: pairing each with a valid SchedulerAttached
        // must still resolve Attached — proving the frame was skipped,
        // not acted on (a corrupt/non-LIFECYCLE Died byte would otherwise
        // force Died).
        let corrupt_died = ShmEntry {
            msg_type: MSG_TYPE_LIFECYCLE,
            payload: vec![LifecyclePhase::SchedulerDied.wire_value()],
            crc_ok: false,
        };
        let empty = ShmEntry {
            msg_type: MSG_TYPE_LIFECYCLE,
            payload: Vec::new(),
            crc_ok: true,
        };
        let non_lifecycle_died = ShmEntry {
            msg_type: MSG_TYPE_LIFECYCLE + 1,
            payload: vec![LifecyclePhase::SchedulerDied.wire_value()],
            crc_ok: true,
        };
        let unknown_phase = ShmEntry {
            msg_type: MSG_TYPE_LIFECYCLE,
            payload: vec![250],
            crc_ok: true,
        };
        for skipped in [corrupt_died, empty, non_lifecycle_died, unknown_phase] {
            let d = drain(vec![skipped, frame(LifecyclePhase::SchedulerAttached, "")]);
            assert_eq!(
                attach_outcome_from_messages(Some(&d)),
                AttachOutcome::Attached,
                "a skipped frame must not suppress a valid SchedulerAttached",
            );
        }
    }

    /// A schedulerless verifier run can reach Phase 5 and make progress
    /// under the kernel fallback policy. PayloadStarting plus
    /// WorkloadDispatched therefore must not be enough to PASS without
    /// the definitive SchedulerAttached frame.
    #[test]
    fn schedulerless_payload_progress_cannot_pass_verification() {
        use crate::vmm::host_comms::BulkDrainResult;
        use crate::vmm::wire::{LifecyclePhase, MSG_TYPE_LIFECYCLE, ShmEntry};

        let messages = BulkDrainResult {
            entries: [
                LifecyclePhase::PayloadStarting,
                LifecyclePhase::WorkloadDispatched,
            ]
            .into_iter()
            .map(|phase| ShmEntry {
                msg_type: MSG_TYPE_LIFECYCLE,
                payload: vec![phase.wire_value()],
                crc_ok: true,
            })
            .collect(),
        };
        let result = VerifierVmResult {
            stats: Vec::new(),
            scheduler_log: String::new(),
            scheduler_stdout: String::new(),
            scheduler_stderr: String::new(),
            attach: attach_outcome_from_messages(Some(&messages)),
            dispatched: dispatch_confirmed_from_messages(Some(&messages)),
            scheduler_exited: scheduler_exited_from_messages(Some(&messages)),
            guest_exit_code: Some(0),
            timed_out: false,
            crash_message: None,
            dilation: None,
        };

        assert_eq!(result.attach, AttachOutcome::Unconfirmed);
        assert!(
            result.dispatched,
            "the fallback-running probe made progress"
        );
        assert!(
            result
                .cell_verdict()
                .expect_err("no scheduler attached, so the cell must fail")
                .contains("no SchedulerAttached frame"),
        );
    }

    /// A scheduler that attached and then died cannot PASS by combining
    /// its historical attach frame with probe progress made later through
    /// the kernel's SCHED_EXT fallback.
    #[test]
    fn attached_then_scheduler_exit_cannot_pass_verification() {
        use crate::vmm::host_comms::BulkDrainResult;
        use crate::vmm::wire::{LifecyclePhase, MSG_TYPE_LIFECYCLE, MSG_TYPE_SCHED_EXIT, ShmEntry};

        let lifecycle = |phase: LifecyclePhase| ShmEntry {
            msg_type: MSG_TYPE_LIFECYCLE,
            payload: vec![phase.wire_value()],
            crc_ok: true,
        };
        let messages = BulkDrainResult {
            entries: vec![
                lifecycle(LifecyclePhase::SchedulerAttached),
                lifecycle(LifecyclePhase::WorkloadDispatched),
                ShmEntry {
                    msg_type: MSG_TYPE_SCHED_EXIT,
                    payload: 1i32.to_le_bytes().to_vec(),
                    crc_ok: true,
                },
            ],
        };
        let result = VerifierVmResult {
            stats: Vec::new(),
            scheduler_log: String::new(),
            scheduler_stdout: String::new(),
            scheduler_stderr: String::new(),
            attach: attach_outcome_from_messages(Some(&messages)),
            dispatched: dispatch_confirmed_from_messages(Some(&messages)),
            scheduler_exited: scheduler_exited_from_messages(Some(&messages)),
            guest_exit_code: Some(0),
            timed_out: false,
            crash_message: None,
            dilation: None,
        };

        assert_eq!(result.attach, AttachOutcome::Attached);
        assert!(result.dispatched);
        assert!(result.scheduler_exited);
        assert!(
            result
                .cell_verdict()
                .expect_err("post-attach scheduler exit must fail")
                .contains("scheduler exited"),
        );

        let mut torn = messages;
        torn.entries.last_mut().unwrap().crc_ok = false;
        assert!(
            !scheduler_exited_from_messages(Some(&torn)),
            "crc-invalid scheduler-exit frame is not evidence",
        );
    }

    #[test]
    fn terminal_guest_exit_requires_explicit_valid_exit_frame() {
        use crate::vmm::host_comms::BulkDrainResult;
        use crate::vmm::wire::{LifecyclePhase, MSG_TYPE_EXIT, MSG_TYPE_LIFECYCLE, ShmEntry};

        assert_eq!(terminal_guest_exit_from_messages(None), None);

        let lifecycle_only = BulkDrainResult {
            entries: vec![ShmEntry {
                msg_type: MSG_TYPE_LIFECYCLE,
                payload: vec![LifecyclePhase::WorkloadDispatched.wire_value()],
                crc_ok: true,
            }],
        };
        assert_eq!(
            terminal_guest_exit_from_messages(Some(&lifecycle_only)),
            None,
            "a generic shutdown plus lifecycle transcript is not terminal evidence",
        );

        for invalid in [
            ShmEntry {
                msg_type: MSG_TYPE_EXIT,
                payload: 0i32.to_le_bytes().to_vec(),
                crc_ok: false,
            },
            ShmEntry {
                msg_type: MSG_TYPE_EXIT,
                payload: vec![0, 0, 0],
                crc_ok: true,
            },
        ] {
            let drain = BulkDrainResult {
                entries: vec![invalid],
            };
            assert_eq!(terminal_guest_exit_from_messages(Some(&drain)), None);
        }

        let explicit = BulkDrainResult {
            entries: vec![ShmEntry {
                msg_type: MSG_TYPE_EXIT,
                payload: 7i32.to_le_bytes().to_vec(),
                crc_ok: true,
            }],
        };
        assert_eq!(terminal_guest_exit_from_messages(Some(&explicit)), Some(7));
    }

    /// dispatch_confirmed_from_messages: true only when a crc-ok,
    /// non-empty WorkloadDispatched frame is present; false for None / no
    /// such frame / corrupt / empty / non-LIFECYCLE.
    #[test]
    fn dispatch_confirmed_from_lifecycle_frames() {
        use crate::vmm::host_comms::BulkDrainResult;
        use crate::vmm::wire::{LifecyclePhase, MSG_TYPE_LIFECYCLE, ShmEntry};

        let frame = |phase: LifecyclePhase| -> ShmEntry {
            ShmEntry {
                msg_type: MSG_TYPE_LIFECYCLE,
                payload: vec![phase.wire_value()],
                crc_ok: true,
            }
        };
        let drain = |entries: Vec<ShmEntry>| BulkDrainResult { entries };

        // No frames at all -> false.
        assert!(!dispatch_confirmed_from_messages(None));

        // SchedulerAttached but no WorkloadDispatched -> false.
        let attached_only = drain(vec![frame(LifecyclePhase::SchedulerAttached)]);
        assert!(!dispatch_confirmed_from_messages(Some(&attached_only)));

        // WorkloadDispatched present -> true.
        let dispatched = drain(vec![
            frame(LifecyclePhase::SchedulerAttached),
            frame(LifecyclePhase::WorkloadDispatched),
        ]);
        assert!(dispatch_confirmed_from_messages(Some(&dispatched)));

        // Corrupt crc / empty payload / non-LIFECYCLE WorkloadDispatched
        // frames are skipped and must not confirm dispatch.
        let corrupt = ShmEntry {
            msg_type: MSG_TYPE_LIFECYCLE,
            payload: vec![LifecyclePhase::WorkloadDispatched.wire_value()],
            crc_ok: false,
        };
        let empty = ShmEntry {
            msg_type: MSG_TYPE_LIFECYCLE,
            payload: Vec::new(),
            crc_ok: true,
        };
        let non_lifecycle = ShmEntry {
            msg_type: MSG_TYPE_LIFECYCLE + 1,
            payload: vec![LifecyclePhase::WorkloadDispatched.wire_value()],
            crc_ok: true,
        };
        for skipped in [corrupt, empty, non_lifecycle] {
            let d = drain(vec![skipped]);
            assert!(
                !dispatch_confirmed_from_messages(Some(&d)),
                "a corrupt/empty/non-LIFECYCLE frame must not confirm dispatch",
            );
        }
    }

    /// VerifierVmResult::cell_verdict gate order + messages: timed_out >
    /// scheduler exit > attach failure > dispatch failure > terminal exit
    /// > PASS.
    #[test]
    fn cell_verdict_gate_order_and_messages() {
        let base = |attach: AttachOutcome, dispatched: bool, timed_out: bool| VerifierVmResult {
            stats: Vec::new(),
            scheduler_log: String::new(),
            scheduler_stdout: String::new(),
            scheduler_stderr: String::new(),
            attach,
            dispatched,
            scheduler_exited: false,
            guest_exit_code: Some(0),
            timed_out,
            crash_message: None,
            dilation: None,
        };

        // Verified + attached + dispatched, no scheduler-exit evidence,
        // and explicit terminal exit 0 -> PASS.
        assert_eq!(
            base(AttachOutcome::Attached, true, false).cell_verdict(),
            Ok(()),
        );

        // Attached but 0-dispatch -> FAIL naming the dispatch gate.
        let no_dispatch = base(AttachOutcome::Attached, false, false).cell_verdict();
        assert!(
            no_dispatch
                .as_ref()
                .unwrap_err()
                .contains("did not dispatch"),
            "dispatch gate must name the failure: {no_dispatch:?}",
        );

        let mut exited = base(AttachOutcome::Attached, true, false);
        exited.scheduler_exited = true;
        assert!(
            exited
                .cell_verdict()
                .expect_err("scheduler exit must fail")
                .contains("scheduler exited"),
        );

        let mut guest_failed = base(AttachOutcome::Attached, true, false);
        guest_failed.guest_exit_code = Some(1);
        assert!(
            guest_failed
                .cell_verdict()
                .expect_err("non-zero terminal exit must fail")
                .contains("exit code 1"),
        );

        let mut no_terminal_exit = base(AttachOutcome::Attached, true, false);
        no_terminal_exit.guest_exit_code = None;
        assert!(
            no_terminal_exit
                .cell_verdict()
                .expect_err("missing terminal frame must fail")
                .contains("did not publish"),
        );

        // Attach failure -> FAIL naming attach, even if dispatched is
        // (defensively) true.
        let attach_fail = base(AttachOutcome::Died, true, false).cell_verdict();
        assert!(
            attach_fail
                .as_ref()
                .unwrap_err()
                .contains("did not turn on"),
            "attach gate must win over dispatch: {attach_fail:?}",
        );

        // timed_out wins over everything, even a clean attach + dispatch.
        // With a CONFIRMED attach the historical post-attach wording is
        // preserved verbatim.
        let hung = base(AttachOutcome::Attached, true, true).cell_verdict();
        assert_eq!(
            hung,
            Err("VM timed out (hung after attach, before exit)".to_string()),
            "timed_out + Attached keeps the exact post-attach message: {hung:?}",
        );

        // A hang WITHOUT a confirmed attach (scheduler died / never
        // reached `enabled` / unconfirmed during BPF load, then wedged)
        // must NOT claim "after attach": it says the timeout carried no
        // confirmed attach and folds in the attach failure reason, while
        // still containing the "timed out" substring the gate-order
        // machinery keys on.
        for attach in [
            AttachOutcome::Died,
            AttachOutcome::NotAttached(String::new()),
            AttachOutcome::NotAttached("sysfs absent".to_string()),
            AttachOutcome::Unconfirmed,
        ] {
            let verdict = base(attach.clone(), false, true).cell_verdict();
            let msg = verdict.as_ref().unwrap_err();
            assert!(
                msg.contains("timed out"),
                "timeout must keep the 'timed out' substring for {attach:?}: {verdict:?}",
            );
            assert!(
                msg.contains("no confirmed scheduler attach"),
                "un-attached timeout must say so for {attach:?}: {verdict:?}",
            );
            assert!(
                !msg.contains("hung after attach"),
                "un-attached timeout must not claim post-attach for {attach:?}: {verdict:?}",
            );
        }

        // timed_out still outranks the crash_message gate.
        let mut hung_crash = base(AttachOutcome::Died, false, true);
        hung_crash.crash_message = Some("PANIC: guest infra fault".to_string());
        let verdict = hung_crash.cell_verdict();
        assert!(
            verdict.as_ref().unwrap_err().contains("timed out"),
            "timed_out must outrank crash_message: {verdict:?}",
        );

        // Attach failure outranks a co-present dispatch failure (root
        // cause reported first).
        let both = base(AttachOutcome::Died, false, false).cell_verdict();
        assert!(
            both.as_ref().unwrap_err().contains("did not turn on"),
            "attach failure reported before dispatch failure: {both:?}",
        );

        // A guest crash message (a self-describing infra fault — e.g. the
        // AP-bring-up gap the boot retry exhausted) is surfaced VERBATIM
        // and outranks the attach gate: the scheduler never ran, so
        // "did not turn on" would misattribute the infra failure. Sits
        // below timed_out (a hang has no crash message) but above attach.
        let mut crashed = base(AttachOutcome::NotAttached(String::new()), false, false);
        crashed.crash_message =
            Some("CPUs [4] failed to come online (AP bring-up failed; 127/128 online)".to_string());
        let verdict = crashed.cell_verdict();
        assert_eq!(
            verdict,
            Err("CPUs [4] failed to come online (AP bring-up failed; 127/128 online)".to_string()),
            "a crash message must be surfaced verbatim, above the attach gate",
        );
        assert!(
            !verdict.as_ref().unwrap_err().contains("did not turn on"),
            "crash message must not be masked by the attach gate: {verdict:?}",
        );
    }

    /// AttachOutcome::failure_reason surfaces (None when attached, the
    /// distinct Died / NotAttached reasons otherwise).
    #[test]
    fn attach_outcome_failure_reason() {
        assert_eq!(AttachOutcome::Attached.failure_reason(), None);
        assert!(
            AttachOutcome::Died
                .failure_reason()
                .unwrap()
                .contains("exited during BPF load"),
        );
        assert!(
            AttachOutcome::NotAttached(String::new())
                .failure_reason()
                .unwrap()
                .contains("never reached sched_ext 'enabled'"),
        );
        assert_eq!(
            AttachOutcome::NotAttached("sysfs absent".to_string()).failure_reason(),
            Some("scheduler never reached sched_ext 'enabled': sysfs absent".to_string()),
        );
        assert!(
            AttachOutcome::Unconfirmed
                .failure_reason()
                .unwrap()
                .contains("attach unconfirmed"),
        );
    }

    /// A timed-out run shows UNKNOWN (not attached), overriding the
    /// attach line — catches a post-attach teardown hang (the frame scan
    /// + Unconfirmed already handle a guest that vanishes before attach).
    #[test]
    fn format_verifier_output_timed_out_shows_unknown() {
        let result = VerifierVmResult {
            stats: Vec::new(),
            scheduler_log: String::new(),
            scheduler_stdout: String::new(),
            scheduler_stderr: String::new(),
            attach: AttachOutcome::Attached,
            dispatched: false,
            scheduler_exited: false,
            guest_exit_code: None,
            timed_out: true,
            crash_message: None,
            dilation: None,
        };
        let out = format_verifier_output("verifier", &result, false);
        assert!(
            out.contains("scheduler: UNKNOWN — VM timed out"),
            "timed-out run must show UNKNOWN: {out}",
        );
        assert!(
            !out.contains("scheduler: attached"),
            "timed-out run must not claim attached: {out}",
        );
    }

    /// An attached-but-not-dispatched run shows the attach line AND a
    /// "dispatch: NOT CONFIRMED" line — the signal that the scheduler
    /// turned on but never dispatched the injected workload.
    /// Guards the `format_verifier_output` dispatched==false render branch,
    /// which the snapshot tests (dispatched==true / Died) do not reach.
    #[test]
    fn format_verifier_output_attached_not_dispatched_shows_not_confirmed() {
        let result = VerifierVmResult {
            stats: Vec::new(),
            scheduler_log: String::new(),
            scheduler_stdout: String::new(),
            scheduler_stderr: String::new(),
            attach: AttachOutcome::Attached,
            dispatched: false,
            scheduler_exited: false,
            guest_exit_code: Some(1),
            timed_out: false,
            crash_message: None,
            dilation: None,
        };
        let out = format_verifier_output("verifier", &result, false);
        assert!(
            out.contains("scheduler: attached"),
            "attached run must show the attach line: {out}",
        );
        assert!(
            out.contains("dispatch: NOT CONFIRMED"),
            "attached-but-not-dispatched must render the NOT CONFIRMED signal: {out}",
        );
    }

    #[test]
    fn format_verifier_output_omits_verification_time() {
        let result = VerifierVmResult {
            stats: Vec::new(),
            scheduler_log: String::new(),
            scheduler_stdout: String::new(),
            scheduler_stderr: "processed 1234 insns (limit 1000000) \
                max_states_per_insn 5 total_states 200 peak_states 50 mark_read 10\n\
                verification time 424242 usec\n\
                stack depth 32+0\n"
                .to_string(),
            attach: AttachOutcome::Attached,
            dispatched: true,
            scheduler_exited: false,
            guest_exit_code: Some(0),
            timed_out: false,
            crash_message: None,
            dilation: None,
        };
        let out = format_verifier_output("verifier", &result, false);
        assert!(out.contains("processed=1234  states=50/200  stack=32+0"));
        assert!(
            !out.contains("time=") && !out.contains("424242"),
            "derived verifier output must not contain environment-dependent time: {out}",
        );
    }

    // -----------------------------------------------------------------------
    // parse_verifier_stats
    // -----------------------------------------------------------------------

    #[test]
    fn parse_verifier_stats_full_line() {
        let log = "processed 1234 insns (limit 1000000) max_states_per_insn 5 total_states 200 peak_states 50 mark_read 10\nverification time 42 usec\nstack depth 32+0\n";
        let vs = parse_verifier_stats(log);
        assert_eq!(vs.processed_insns, 1234);
        assert_eq!(vs.total_states, 200);
        assert_eq!(vs.peak_states, 50);
        assert_eq!(vs.stack_depth.as_deref(), Some("32+0"));
    }

    #[test]
    fn parse_verifier_stats_insns_only() {
        let log = "processed 500 insns (limit 1000000) max_states_per_insn 1 total_states 10 peak_states 3 mark_read 0\n";
        let vs = parse_verifier_stats(log);
        assert_eq!(vs.processed_insns, 500);
        assert_eq!(vs.total_states, 10);
        assert_eq!(vs.peak_states, 3);
        assert!(vs.stack_depth.is_none());
    }

    #[test]
    fn parse_verifier_stats_empty() {
        let vs = parse_verifier_stats("");
        assert_eq!(vs.processed_insns, 0);
        assert_eq!(vs.total_states, 0);
        assert_eq!(vs.peak_states, 0);
        assert!(vs.stack_depth.is_none());
    }

    #[test]
    fn parse_verifier_stats_garbage_lines() {
        let log = "some random output\nnot a stats line\n";
        let vs = parse_verifier_stats(log);
        assert_eq!(vs.processed_insns, 0);
        assert_eq!(vs.total_states, 0);
    }

    #[test]
    fn parse_verifier_stats_ignores_time_without_insns() {
        let log = "verification time 100 usec\nstack depth 64\n";
        let vs = parse_verifier_stats(log);
        assert_eq!(vs.processed_insns, 0);
        assert_eq!(vs.stack_depth.as_deref(), Some("64"));
    }

    #[test]
    fn parse_verifier_stats_multi_subprogram_stack() {
        let log = "processed 42 insns (limit 1000000) max_states_per_insn 1 total_states 5 peak_states 2 mark_read 0\nstack depth 32+16+8\n";
        let vs = parse_verifier_stats(log);
        assert_eq!(vs.processed_insns, 42);
        assert_eq!(vs.stack_depth.as_deref(), Some("32+16+8"));
    }

    #[test]
    fn parse_verifier_stats_noise_between_lines() {
        let log = "\
libbpf: loading something
processed 999 insns (limit 1000000) max_states_per_insn 3 total_states 77 peak_states 20 mark_read 5
libbpf: prog 'dispatch': attached
verification time 7 usec
stack depth 48+0
";
        let vs = parse_verifier_stats(log);
        assert_eq!(vs.processed_insns, 999);
        assert_eq!(vs.total_states, 77);
        assert_eq!(vs.peak_states, 20);
        assert_eq!(vs.stack_depth.as_deref(), Some("48+0"));
    }

    #[test]
    fn parse_verifier_stats_partial_insns_line() {
        let log = "processed 123\n";
        let vs = parse_verifier_stats(log);
        assert_eq!(vs.processed_insns, 123);
        assert_eq!(vs.total_states, 0);
        assert_eq!(vs.peak_states, 0);
    }

    #[test]
    fn parse_verifier_stats_only_stack_depth() {
        let log = "stack depth 128\n";
        let vs = parse_verifier_stats(log);
        assert_eq!(vs.stack_depth.as_deref(), Some("128"));
        assert_eq!(vs.processed_insns, 0);
    }

    #[test]
    fn parse_verifier_stats_zero_insns() {
        let log = "processed 0 insns (limit 1000000) max_states_per_insn 0 total_states 0 peak_states 0 mark_read 0\n";
        let vs = parse_verifier_stats(log);
        assert_eq!(vs.processed_insns, 0);
        assert_eq!(vs.total_states, 0);
        assert_eq!(vs.peak_states, 0);
    }

    #[test]
    fn parse_verifier_stats_large_values() {
        let log = "processed 999999 insns (limit 1000000) max_states_per_insn 100 total_states 50000 peak_states 12345 mark_read 9999\nverification time 123456 usec\n";
        let vs = parse_verifier_stats(log);
        assert_eq!(vs.processed_insns, 999999);
        assert_eq!(vs.total_states, 50000);
        assert_eq!(vs.peak_states, 12345);
    }

    #[test]
    fn parse_verifier_stats_stack_depth_single() {
        let log = "stack depth 64\n";
        let vs = parse_verifier_stats(log);
        assert_eq!(vs.stack_depth.as_deref(), Some("64"));
    }

    #[test]
    fn parse_verifier_stats_stack_depth_many_subprograms() {
        let log = "stack depth 32+16+8+0+0\n";
        let vs = parse_verifier_stats(log);
        assert_eq!(vs.stack_depth.as_deref(), Some("32+16+8+0+0"));
    }

    #[test]
    fn parse_verifier_stats_multiple_processed_lines_takes_last() {
        let log = "processed 100 insns (limit 1000000) max_states_per_insn 1 total_states 5 peak_states 2 mark_read 0\nprocessed 200 insns (limit 1000000) max_states_per_insn 2 total_states 10 peak_states 4 mark_read 0\n";
        let vs = parse_verifier_stats(log);
        assert_eq!(vs.processed_insns, 200);
        assert_eq!(vs.total_states, 10);
    }

    #[test]
    fn parse_verifier_stats_complexity_error_with_stats() {
        let log = "\
func#0 @0
0: R1=ctx() R10=fp0
1: (bf) r6 = r1                       ; R1=ctx() R6_w=ctx()
back-edge from insn 42 to 10
BPF program is too complex
processed 131071 insns (limit 131072) max_states_per_insn 12 total_states 9999 peak_states 5000 mark_read 800
verification time 250000 usec
stack depth 96+32
";
        let vs = parse_verifier_stats(log);
        assert_eq!(vs.processed_insns, 131071);
        assert_eq!(vs.total_states, 9999);
        assert_eq!(vs.peak_states, 5000);
        assert_eq!(vs.stack_depth.as_deref(), Some("96+32"));
    }

    #[test]
    fn parse_verifier_stats_complexity_error_no_stats() {
        let log = "\
func#0 @0
0: R1=ctx() R10=fp0
R1 type=ctx expected=fp
";
        let vs = parse_verifier_stats(log);
        assert_eq!(vs.processed_insns, 0);
        assert_eq!(vs.total_states, 0);
        assert!(vs.stack_depth.is_none());
    }

    #[test]
    fn parse_verifier_stats_loop_warning_with_stats() {
        let log = "\
infinite loop detected at insn 15
back-edge from insn 30 to 15
processed 500 insns (limit 1000000) max_states_per_insn 3 total_states 40 peak_states 15 mark_read 5
verification time 100 usec
";
        let vs = parse_verifier_stats(log);
        assert_eq!(vs.processed_insns, 500);
        assert_eq!(vs.total_states, 40);
        assert_eq!(vs.peak_states, 15);
    }

    #[test]
    fn parse_verifier_stats_processed_no_number() {
        let log = "processed\n";
        let vs = parse_verifier_stats(log);
        assert_eq!(vs.processed_insns, 0);
    }

    #[test]
    fn parse_verifier_stats_keyword_at_end_no_value() {
        let log = "processed 100 insns (limit 1000000) max_states_per_insn 1 total_states\n";
        let vs = parse_verifier_stats(log);
        assert_eq!(vs.processed_insns, 100);
        assert_eq!(vs.total_states, 0);
    }

    #[test]
    fn parse_verifier_stats_non_numeric_values() {
        let log = "processed 100 insns (limit 1000000) max_states_per_insn 1 total_states abc peak_states xyz mark_read 0\n";
        let vs = parse_verifier_stats(log);
        assert_eq!(vs.processed_insns, 100);
        assert_eq!(vs.total_states, 0);
        assert_eq!(vs.peak_states, 0);
    }

    #[test]
    fn parse_verifier_stats_verification_time_is_ignored() {
        let log = "verification time unknown usec\n";
        let vs = parse_verifier_stats(log);
        assert_eq!(vs.processed_insns, 0);
        assert!(vs.stack_depth.is_none());
    }

    #[test]
    fn parse_verifier_stats_stack_depth_empty() {
        let log = "stack depth   \n";
        let vs = parse_verifier_stats(log);
        assert!(vs.stack_depth.is_none());
    }

    #[test]
    fn parse_verifier_stats_peak_states_at_end() {
        let log = "processed 50 insns (limit 1000000) max_states_per_insn 1 total_states 10 peak_states\n";
        let vs = parse_verifier_stats(log);
        assert_eq!(vs.processed_insns, 50);
        assert_eq!(vs.total_states, 10);
        assert_eq!(vs.peak_states, 0);
    }

    #[test]
    fn parse_verifier_stats_windows_line_endings() {
        let log = "processed 42 insns (limit 1000000) max_states_per_insn 1 total_states 5 peak_states 2 mark_read 0\r\nverification time 10 usec\r\nstack depth 16\r\n";
        let vs = parse_verifier_stats(log);
        assert_eq!(vs.processed_insns, 42);
        assert!(vs.stack_depth.is_some());
    }

    // -----------------------------------------------------------------------
    // normalize_verifier_line
    // -----------------------------------------------------------------------

    #[test]
    fn normalize_plain_instruction() {
        assert_eq!(
            normalize_verifier_line("100: (07) r1 += 8"),
            "100: (07) r1 += 8"
        );
    }

    #[test]
    fn normalize_strips_frame_annotation() {
        assert_eq!(
            normalize_verifier_line("3006: (07) r9 += 1  ; frame1: R9_w=2"),
            "3006: (07) r9 += 1"
        );
    }

    #[test]
    fn normalize_strips_register_annotation() {
        assert_eq!(
            normalize_verifier_line("42: (bf) r6 = r1 ; R1=ctx() R6_w=ctx()"),
            "42: (bf) r6 = r1"
        );
    }

    #[test]
    fn normalize_standalone_register_dump() {
        assert_eq!(
            normalize_verifier_line("3041: frame1: R0_w=scalar()"),
            "3041:"
        );
    }

    #[test]
    fn normalize_goto_inline_state() {
        assert_eq!(
            normalize_verifier_line(
                "3026: (b5) if r6 <= 0x11dc0 goto pc+2 3029: frame1: R0=1 R6=scalar()"
            ),
            "3026: (b5) if r6 <= 0x11dc0 goto pc+2"
        );
    }

    #[test]
    fn normalize_goto_no_inline_state() {
        assert_eq!(
            normalize_verifier_line("50: (05) goto pc+10"),
            "50: (05) goto pc+10"
        );
    }

    #[test]
    fn normalize_non_instruction_line() {
        assert_eq!(normalize_verifier_line("func#0 @0"), "func#0 @0");
    }

    #[test]
    fn normalize_empty() {
        assert_eq!(normalize_verifier_line(""), "");
    }

    #[test]
    fn normalize_goto_negative_offset() {
        assert_eq!(
            normalize_verifier_line("50: (05) goto pc-10 60: frame1: R0=1"),
            "50: (05) goto pc-10"
        );
    }

    #[test]
    fn normalize_semicolon_source_comment() {
        let line = "100: (07) r1 += 8 ; for (int j = 0; j < n; j++)";
        assert_eq!(normalize_verifier_line(line), line);
    }

    #[test]
    fn normalize_semicolon_return_value_comment() {
        let line = "200: (b7) r0 = 0 ; Return value";
        assert_eq!(normalize_verifier_line(line), line);
    }

    #[test]
    fn normalize_standalone_bare_register_dump() {
        assert_eq!(
            normalize_verifier_line("3029: R0=1 R6=scalar(id=1)"),
            "3029:"
        );
    }

    #[test]
    fn normalize_standalone_r10_dump() {
        assert_eq!(normalize_verifier_line("42: R10=fp0"), "42:");
    }

    // -----------------------------------------------------------------------
    // detect_cycle / collapse_cycles
    // -----------------------------------------------------------------------

    fn repeating_log(prefix: usize, period: usize, reps: usize, suffix: usize) -> String {
        let mut lines = Vec::new();
        for i in 0..prefix {
            lines.push(format!("{}: (07) r1 += {i}", 1000 + i));
        }
        for rep in 0..reps {
            for j in 0..period {
                let insn = 100 + j;
                lines.push(format!(
                    "{insn}: (bf) r{} = r{} ; frame1: R{}_w={}",
                    j % 10,
                    (j + 1) % 10,
                    j % 10,
                    rep * 100 + j
                ));
            }
        }
        for i in 0..suffix {
            lines.push(format!("{}: (95) exit_{i}", 2000 + i));
        }
        lines.join("\n")
    }

    #[test]
    fn detect_cycle_basic() {
        let log = repeating_log(0, 10, 8, 0);
        let lines: Vec<&str> = log.lines().collect();
        let result = detect_cycle(&lines);
        assert!(result.is_some(), "should detect cycle");
        let (start, period, count) = result.unwrap();
        assert_eq!(period, 10);
        assert!(count >= 6, "count={count}");
        assert_eq!(start, 0);
    }

    #[test]
    fn detect_cycle_with_prefix_suffix() {
        let log = repeating_log(5, 10, 8, 5);
        let lines: Vec<&str> = log.lines().collect();
        let result = detect_cycle(&lines);
        assert!(result.is_some(), "should detect cycle with prefix/suffix");
        let (_start, period, count) = result.unwrap();
        assert_eq!(period, 10);
        assert!(count >= 6);
    }

    #[test]
    fn detect_cycle_too_few_reps() {
        let log = repeating_log(0, 10, 2, 0);
        let lines: Vec<&str> = log.lines().collect();
        assert!(detect_cycle(&lines).is_none());
    }

    #[test]
    fn detect_cycle_too_few_lines() {
        let lines: Vec<String> = (0..20)
            .map(|i| format!("{}: (07) r1 += {i}", 100 + i % 3))
            .collect();
        let refs: Vec<&str> = lines.iter().map(|s| s.as_str()).collect();
        assert!(detect_cycle(&refs).is_none());
    }

    #[test]
    fn detect_cycle_no_cycle() {
        let lines: Vec<String> = (0..100).map(|i| format!("{i}: unique_insn_{i}")).collect();
        let refs: Vec<&str> = lines.iter().map(|s| s.as_str()).collect();
        assert!(detect_cycle(&refs).is_none());
    }

    #[test]
    fn detect_cycle_empty() {
        let empty: Vec<&str> = vec![];
        assert!(detect_cycle(&empty).is_none());
    }

    #[test]
    fn detect_cycle_exact_boundary() {
        let log = repeating_log(0, 5, 6, 0);
        let lines: Vec<&str> = log.lines().collect();
        assert_eq!(lines.len(), 30);
        let result = detect_cycle(&lines);
        assert!(result.is_some(), "boundary case should detect cycle");
        let (_start, period, count) = result.unwrap();
        assert_eq!(period, 5);
        assert_eq!(count, 6);
    }

    #[test]
    fn collapse_cycles_empty_string() {
        assert_eq!(collapse_cycles(""), "");
    }

    #[test]
    fn collapse_cycles_basic() {
        let log = repeating_log(2, 10, 8, 2);
        let collapsed = collapse_cycles(&log);
        assert!(collapsed.contains("identical iterations omitted"));
        assert!(collapsed.contains("8x of the following 10 lines"));
        assert!(collapsed.contains("end repeat"));
        assert!(collapsed.lines().count() < log.lines().count());
    }

    #[test]
    fn collapse_cycles_no_cycle() {
        let log = "line 1\nline 2\nline 3\n";
        let collapsed = collapse_cycles(log);
        assert_eq!(collapsed, log);
    }

    #[test]
    fn collapse_cycles_preserves_stats() {
        let mut log = repeating_log(0, 10, 8, 0);
        log.push_str("\nprocessed 1000 insns (limit 1000000) max_states_per_insn 5 total_states 100 peak_states 30 mark_read 10\n");
        let collapsed = collapse_cycles(&log);
        assert!(collapsed.contains("processed 1000 insns"));
    }

    #[test]
    fn collapse_cycles_with_register_annotations() {
        let mut lines = Vec::new();
        lines.push("0: (07) r1 += 1".to_string());
        for rep in 0..8 {
            for j in 0..6 {
                let insn = 100 + j;
                lines.push(format!(
                    "{insn}: (bf) r{} = r{} ; frame1: R{}_w={}",
                    j % 10,
                    (j + 1) % 10,
                    j % 10,
                    rep * 100 + j
                ));
            }
        }
        lines.push("200: (95) exit".to_string());
        let log = lines.join("\n");
        let collapsed = collapse_cycles(&log);
        assert!(collapsed.contains("identical iterations omitted"));
    }

    // -----------------------------------------------------------------------
    // build_b_map / build_diff_rows
    // -----------------------------------------------------------------------

    fn prog(name: &str, verified_insns: u32) -> ProgStats {
        ProgStats {
            name: name.to_string(),
            verified_insns,
        }
    }

    #[test]
    fn build_b_map_basic() {
        let stats_b = vec![prog("dispatch", 500)];
        let map = build_b_map(&stats_b);
        assert_eq!(map.get("dispatch"), Some(&500));
    }

    #[test]
    fn build_b_map_empty() {
        let map = build_b_map(&[]);
        assert!(map.is_empty());
    }

    #[test]
    fn build_diff_rows_matching_programs() {
        let stats_a = vec![prog("dispatch", 500)];
        let mut b_map = HashMap::new();
        b_map.insert("dispatch".to_string(), 300u64);
        let rows = build_diff_rows(&stats_a, &b_map);
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].name, "dispatch");
        assert_eq!(rows[0].a, 500);
        assert_eq!(rows[0].b, 300);
        assert_eq!(rows[0].delta, 200);
    }

    #[test]
    fn build_diff_rows_program_missing_from_b() {
        let stats_a = vec![prog("new_prog", 100)];
        let b_map = HashMap::new();
        let rows = build_diff_rows(&stats_a, &b_map);
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].a, 100);
        assert_eq!(rows[0].b, 0);
        assert_eq!(rows[0].delta, 100);
    }

    #[test]
    fn build_diff_rows_negative_delta() {
        let stats_a = vec![prog("dispatch", 200)];
        let mut b_map = HashMap::new();
        b_map.insert("dispatch".to_string(), 500u64);
        let rows = build_diff_rows(&stats_a, &b_map);
        assert_eq!(rows[0].delta, -300);
    }

    #[test]
    fn build_diff_rows_empty_a() {
        let b_map = HashMap::new();
        let rows = build_diff_rows(&[], &b_map);
        assert!(rows.is_empty());
    }

    /// Simulates the verifier trace produced by #pragma unroll loops.
    /// Each copy is at a different base address but has the same
    /// instruction sequence. After normalize_for_cycle_detection strips
    /// addresses and register annotations, all copies look identical.
    fn unrolled_verifier_log(copies: usize, body_len: usize) -> String {
        let ops = [
            "(85) call bpf_ktime_get_ns#5",
            "(bf) r2 = r0",
            "(77) r0 >>= 16",
            "(af) r1 ^= r0",
            "(77) r2 >>= 32",
            "(0f) r1 += r2",
            "(24) w1 *= 7",
            "(04) w1 += 1",
        ];
        let mut lines = Vec::new();
        lines.push("func#0 @0".to_string());
        lines.push("0: R1=ctx() R10=fp0".to_string());
        let mut addr = 10;
        for copy in 0..copies {
            for (j, op) in ops.iter().enumerate().take(body_len) {
                lines.push(format!(
                    "{}: {op} ; R0_w=scalar(id={})",
                    addr,
                    copy * 100 + j
                ));
                addr += 1;
            }
        }
        lines.push(format!("{addr}: (05) goto pc-1"));
        lines.push(
            "processed 1000 insns (limit 1000000) max_states_per_insn 3 \
             total_states 50 peak_states 20 mark_read 5"
                .to_string(),
        );
        lines.join("\n")
    }

    #[test]
    fn detect_cycle_unrolled_loop() {
        let log = unrolled_verifier_log(8, 6);
        let lines: Vec<&str> = log.lines().collect();
        let result = detect_cycle(&lines);
        assert!(result.is_some(), "should detect cycle in unrolled loop");
        let (_start, period, count) = result.unwrap();
        assert_eq!(period, 6);
        assert!(count >= 6, "count={count}");
    }

    #[test]
    fn collapse_cycles_unrolled_loop() {
        let log = unrolled_verifier_log(8, 6);
        let collapsed = collapse_cycles(&log);
        assert!(
            collapsed.contains("identical iterations omitted"),
            "should collapse unrolled loop"
        );
        assert!(collapsed.lines().count() < log.lines().count());
    }

    // -----------------------------------------------------------------------
    // extract_verifier_log
    // -----------------------------------------------------------------------

    #[test]
    fn extract_verifier_log_basic() {
        let log = "\
libbpf: prog 'dispatch': BPF program load failed: -22
-- BEGIN PROG LOAD LOG --
func#0 @0
0: R1=ctx() R10=fp0
processed 100 insns (limit 1000000) max_states_per_insn 1 total_states 5 peak_states 2 mark_read 0
-- END PROG LOAD LOG --
libbpf: failed to load object 'ktstr_ops'
";
        let extracted = extract_verifier_log(log);
        assert!(extracted.is_some());
        let v = extracted.unwrap();
        assert!(v.starts_with("func#0 @0"));
        assert!(v.contains("processed 100 insns"));
        assert!(!v.contains("BEGIN PROG LOAD LOG"));
        assert!(!v.contains("END PROG LOAD LOG"));
        assert!(!v.contains("libbpf:"));
    }

    #[test]
    fn extract_verifier_log_none_without_markers() {
        let log = "func#0 @0\n0: R1=ctx()\nprocessed 50 insns\n";
        assert!(extract_verifier_log(log).is_none());
    }

    #[test]
    fn extract_verifier_log_empty() {
        assert!(extract_verifier_log("").is_none());
    }

    /// Attack 1: libbpf wraps verifier output with "libbpf: " prefix lines.
    /// `parse_verifier_stats` looks for `starts_with("processed ")` which
    /// won't match `libbpf: processed ...`. Without extraction, stats
    /// parsing fails on blobs where the `processed` line is only inside
    /// the markers.
    #[test]
    fn extract_verifier_log_attack1_stats_parse() {
        let blob = "\
libbpf: prog 'ktstr_ops_dispatch': BPF program load failed: -22
libbpf: -- BEGIN PROG LOAD LOG --
func#0 @0
0: R1=ctx() R10=fp0
1: (bf) r6 = r1 ; R1=ctx() R6_w=ctx()
back-edge from insn 42 to 10
BPF program is too complex
processed 131071 insns (limit 131072) max_states_per_insn 12 total_states 9999 peak_states 5000 mark_read 800
verification time 250000 usec
stack depth 96+32
libbpf: -- END PROG LOAD LOG --
libbpf: failed to load BPF skeleton 'ktstr_ops': -22
";
        let extracted = extract_verifier_log(blob);
        assert!(extracted.is_some(), "should find markers");
        let v = extracted.unwrap();
        let vs = parse_verifier_stats(v);
        assert_eq!(vs.processed_insns, 131071);
        assert_eq!(vs.total_states, 9999);
        assert_eq!(vs.peak_states, 5000);
        assert_eq!(vs.stack_depth.as_deref(), Some("96+32"));

        // Without extraction, parsing the full blob must also work
        // because the "processed" line doesn't have a "libbpf: " prefix
        // inside the markers. But verify extraction gives cleaner input.
        let vs_raw = parse_verifier_stats(blob);
        assert_eq!(vs_raw.processed_insns, 131071);
    }

    /// Attack 3: three distinct program load logs in a single blob.
    /// Each has different instructions. `collapse_cycles` must NOT treat
    /// them as a repeating cycle.
    #[test]
    fn extract_verifier_log_attack3_no_false_collapse() {
        let blob = "\
libbpf: prog 'init': BPF program load failed: -22
libbpf: -- BEGIN PROG LOAD LOG --
func#0 @0
0: R1=ctx() R10=fp0
1: (bf) r6 = r1
2: (07) r6 += 8
3: (61) r0 = *(u32 *)(r6 + 0)
4: (95) exit
processed 5 insns (limit 1000000) max_states_per_insn 1 total_states 3 peak_states 1 mark_read 0
libbpf: -- END PROG LOAD LOG --
libbpf: prog 'dispatch': BPF program load failed: -22
libbpf: -- BEGIN PROG LOAD LOG --
func#1 @10
10: R1=ctx() R10=fp0
11: (bf) r7 = r1
12: (85) call bpf_ktime_get_ns#5
13: (77) r0 >>= 32
14: (95) exit
processed 5 insns (limit 1000000) max_states_per_insn 1 total_states 3 peak_states 1 mark_read 0
libbpf: -- END PROG LOAD LOG --
libbpf: prog 'enqueue': BPF program load failed: -22
libbpf: -- BEGIN PROG LOAD LOG --
func#2 @20
20: R1=ctx() R10=fp0
21: (b7) r0 = 0
22: (63) *(u32 *)(r10 - 4) = r0
23: (61) r1 = *(u32 *)(r10 - 4)
24: (95) exit
processed 5 insns (limit 1000000) max_states_per_insn 1 total_states 3 peak_states 1 mark_read 0
libbpf: -- END PROG LOAD LOG --
libbpf: failed to load BPF skeleton 'ktstr_ops': -22
";
        // extract_verifier_log returns the FIRST log section.
        let extracted = extract_verifier_log(blob);
        assert!(extracted.is_some());
        let v = extracted.unwrap();
        assert!(v.contains("func#0 @0"), "should get first program's log");
        assert!(!v.contains("func#1"), "should not include second program");

        // collapse_cycles on the extracted first section must not
        // collapse — it's only 7 lines total.
        let collapsed = collapse_cycles(v);
        assert!(
            !collapsed.contains("identical iterations omitted"),
            "must not false-collapse distinct program logs"
        );
    }

    // -----------------------------------------------------------------------
    // collapse_verifier_region — in-place collapse of the verifier trace,
    // preserving everything outside the BEGIN/END markers.
    // -----------------------------------------------------------------------

    /// Markers present: the region between them is collapsed while the
    /// prefix (surrounding scheduler chatter + BEGIN marker) and suffix
    /// (END marker + trailing chatter) survive byte-for-byte. Built with
    /// a genuine cycle (a >=5-line block repeated >=3x) so the collapse
    /// fires and we can assert the omission marker lands INSIDE the
    /// region without disturbing the surrounding text.
    #[test]
    fn collapse_verifier_region_collapses_in_place() {
        // 6-line repeating block × 4 → detect_cycle fires (>=5 period,
        // >=3 reps). Distinct addresses per copy so normalization maps
        // them together.
        let mut trace = String::new();
        for i in 0..24 {
            trace.push_str(&format!("{}: (07) r1 += 8 ; op{}\n", i, i % 6));
        }
        let prefix = "scheduler: starting up\nlibbpf: loading\n";
        let suffix = "libbpf: load failed: -22\nscheduler exiting\n";
        let input =
            format!("{prefix}-- BEGIN PROG LOAD LOG --\n{trace}-- END PROG LOAD LOG --\n{suffix}",);

        let out = collapse_verifier_region(&input);

        // Everything before the BEGIN marker's content and everything
        // from the END marker onward is byte-identical.
        assert!(
            out.starts_with(&format!("{prefix}-- BEGIN PROG LOAD LOG --\n")),
            "prefix + BEGIN marker must be preserved verbatim: {out}",
        );
        assert!(
            out.contains(&format!("-- END PROG LOAD LOG --\n{suffix}")),
            "END marker + suffix must be preserved verbatim: {out}",
        );
        // The collapse fired inside the region.
        assert!(
            out.contains("identical iterations omitted"),
            "the cyclic region must be collapsed in place: {out}",
        );
    }

    /// No markers: identity (returned unchanged), even when the input
    /// contains a collapsible cycle — without the markers there is no
    /// region to collapse.
    #[test]
    fn collapse_verifier_region_identity_without_markers() {
        let mut input = String::from("scheduler output with no verifier markers\n");
        for i in 0..24 {
            input.push_str(&format!("{}: (07) r1 += 8 ; op{}\n", i, i % 6));
        }
        assert_eq!(
            collapse_verifier_region(&input),
            input,
            "input without BEGIN/END markers must be returned unchanged",
        );
    }

    /// Mid-line END marker (`libbpf: -- END ...`): the partial `libbpf: `
    /// prefix line stays OUTSIDE the collapsed region (in the suffix),
    /// consistent with `extract_verifier_log`'s trim-back-to-newline
    /// behavior — so the region boundary the two functions compute agree.
    #[test]
    fn collapse_verifier_region_midline_end_matches_extract() {
        let input = "\
head\n\
-- BEGIN PROG LOAD LOG --\n\
0: R1=ctx()\n\
processed 5 insns (limit 1) total_states 3 peak_states 1\n\
libbpf: -- END PROG LOAD LOG --\n\
tail\n";
        // Short region → no collapse; collapse_verifier_region is an
        // identity transform on the bytes here, so the whole input is
        // preserved AND the `libbpf: -- END` line is intact.
        let out = collapse_verifier_region(input);
        assert_eq!(
            out, input,
            "short region must round-trip byte-for-byte: {out}"
        );
        // The extracted region must NOT include the partial libbpf END
        // line — cross-check that the shared region logic agrees.
        let extracted = extract_verifier_log(input).unwrap();
        assert!(
            extracted.contains("processed 5 insns"),
            "extract must include the last real content line: {extracted}",
        );
        assert!(
            !extracted.contains("-- END"),
            "extract must not include the END-marker line: {extracted}",
        );
    }

    /// A timed-out run STILL renders the live-streamed scheduler stdout:
    /// the core bug fix — a watchdog kill never reaches the teardown
    /// merged-file dump (`scheduler_log` empty), but whatever streamed
    /// over `scheduler_stdout` before the kill must appear.
    #[test]
    fn format_verifier_output_timed_out_prints_live_stdout() {
        let result = VerifierVmResult {
            stats: Vec::new(),
            scheduler_log: String::new(),
            scheduler_stdout: "scheduler: entered main loop\ndispatching tasks...\n".to_string(),
            scheduler_stderr: String::new(),
            attach: AttachOutcome::Attached,
            dispatched: false,
            scheduler_exited: false,
            guest_exit_code: None,
            timed_out: true,
            crash_message: None,
            dilation: None,
        };
        let out = format_verifier_output("verifier", &result, false);
        assert!(
            out.contains("scheduler: UNKNOWN — VM timed out"),
            "timed-out run must show UNKNOWN: {out}",
        );
        assert!(
            out.contains("--- scheduler log ---"),
            "timed-out run must still render the scheduler-log section: {out}",
        );
        assert!(
            out.contains("scheduler: entered main loop") && out.contains("dispatching tasks..."),
            "the live-streamed stdout captured before the watchdog kill must be printed: {out}",
        );
    }

    /// Merged-dump fallback selection: with live stderr present and live
    /// stdout empty, the stdout section renders NOTHING — the merged
    /// teardown dump would duplicate the stderr section (schedulers that
    /// log exclusively to stderr, the common libbpf/log-crate case). The
    /// dump only renders when NEITHER live stream arrived.
    #[test]
    fn format_verifier_output_no_dump_duplication_with_live_stderr() {
        let base = |stdout: &str, stderr: &str, log: &str| VerifierVmResult {
            stats: Vec::new(),
            scheduler_log: log.to_string(),
            scheduler_stdout: stdout.to_string(),
            scheduler_stderr: stderr.to_string(),
            attach: AttachOutcome::Died,
            dispatched: false,
            scheduler_exited: false,
            guest_exit_code: Some(1),
            timed_out: false,
            crash_message: None,
            dilation: None,
        };
        // Live stderr arrived, live stdout empty: the merged dump (same
        // bytes as stderr here) must NOT render on stdout.
        let with_stderr = base("", "load failed: EINVAL\n", "load failed: EINVAL\n");
        let out = format_verifier_output("verifier", &with_stderr, false);
        assert!(
            !out.contains("--- scheduler log ---") && !out.contains("load failed"),
            "live stderr present -> no merged-dump duplication on stdout: {out}",
        );
        // No live stream at all: the merged dump is the fallback.
        let dump_only = base("", "", "load failed: EINVAL\n");
        let out = format_verifier_output("verifier", &dump_only, false);
        assert!(
            out.contains("--- scheduler log ---") && out.contains("load failed"),
            "no live streams -> merged dump renders: {out}",
        );
    }

    /// The stderr formatter labels the section and collapses the verifier
    /// region in place (non-raw); empty stderr yields the empty string so
    /// the caller emits nothing.
    #[test]
    fn format_verifier_stderr_labels_and_collapses() {
        let empty = VerifierVmResult {
            stats: Vec::new(),
            scheduler_log: String::new(),
            scheduler_stdout: String::new(),
            scheduler_stderr: String::new(),
            attach: AttachOutcome::Attached,
            dispatched: true,
            scheduler_exited: false,
            guest_exit_code: Some(0),
            timed_out: false,
            crash_message: None,
            dilation: None,
        };
        assert_eq!(
            format_verifier_stderr("verifier", &empty, false),
            "",
            "empty stderr must format to the empty string",
        );

        let mut trace = String::new();
        for i in 0..24 {
            trace.push_str(&format!("{}: (07) r1 += 8 ; op{}\n", i, i % 6));
        }
        let with_err = VerifierVmResult {
            stats: Vec::new(),
            scheduler_log: String::new(),
            scheduler_stdout: String::new(),
            scheduler_stderr: format!(
                "libbpf: -- BEGIN PROG LOAD LOG --\n{trace}-- END PROG LOAD LOG --\nload failed\n",
            ),
            attach: AttachOutcome::Died,
            dispatched: false,
            scheduler_exited: false,
            guest_exit_code: Some(1),
            timed_out: false,
            crash_message: None,
            dilation: None,
        };
        let s = format_verifier_stderr("verifier", &with_err, false);
        assert!(
            s.contains("--- scheduler stderr ---"),
            "stderr section must carry a labeled header: {s}",
        );
        assert!(
            s.contains("identical iterations omitted"),
            "non-raw stderr must collapse the verifier region: {s}",
        );
        assert!(
            s.contains("load failed"),
            "text outside the verifier region must be preserved: {s}",
        );
        // Raw mode: no collapsing.
        let raw = format_verifier_stderr("verifier", &with_err, true);
        assert!(
            !raw.contains("identical iterations omitted"),
            "raw mode must not collapse: {raw}",
        );
    }

    // -- insta snapshot tests --

    #[test]
    fn snapshot_format_verifier_output_no_log() {
        let result = VerifierVmResult {
            stats: vec![
                ProgStats {
                    name: "enqueue".into(),
                    verified_insns: 500,
                },
                ProgStats {
                    name: "dispatch".into(),
                    verified_insns: 1200,
                },
                ProgStats {
                    name: "init".into(),
                    verified_insns: 300,
                },
            ],
            scheduler_log: String::new(),
            scheduler_stdout: String::new(),
            scheduler_stderr: String::new(),
            attach: AttachOutcome::Attached,
            dispatched: true,
            scheduler_exited: false,
            guest_exit_code: Some(0),
            timed_out: false,
            crash_message: None,
            dilation: None,
        };
        insta::assert_snapshot!(format_verifier_output("default", &result, false));
    }

    #[test]
    fn snapshot_format_verifier_output_with_log() {
        let log = "\
-- BEGIN PROG LOAD LOG --\n\
func#0 @0\n\
0: R1=ctx() R10=fp0\n\
processed 42 insns (limit 1000000) max_states_per_insn 1 total_states 10 peak_states 8 mark_read 5\n\
-- END PROG LOAD LOG --";
        let result = VerifierVmResult {
            stats: vec![ProgStats {
                name: "enqueue".into(),
                verified_insns: 42,
            }],
            scheduler_log: log.into(),
            scheduler_stdout: String::new(),
            scheduler_stderr: String::new(),
            // A load log present means the scheduler printed a verifier
            // trace then exited — the SchedulerDied failure path.
            attach: AttachOutcome::Died,
            dispatched: false,
            scheduler_exited: false,
            guest_exit_code: Some(1),
            timed_out: false,
            crash_message: None,
            dilation: None,
        };
        insta::assert_snapshot!(format_verifier_output("llc+steal", &result, false));
    }

    #[test]
    fn snapshot_format_verifier_diff() {
        let stats_a = vec![
            ProgStats {
                name: "enqueue".into(),
                verified_insns: 500,
            },
            ProgStats {
                name: "dispatch".into(),
                verified_insns: 1200,
            },
            ProgStats {
                name: "init".into(),
                verified_insns: 300,
            },
        ];
        let stats_b = vec![
            ProgStats {
                name: "enqueue".into(),
                verified_insns: 480,
            },
            ProgStats {
                name: "dispatch".into(),
                verified_insns: 1350,
            },
            ProgStats {
                name: "init".into(),
                verified_insns: 300,
            },
        ];
        insta::assert_snapshot!(format_verifier_diff("default", &stats_a, "llc", &stats_b));
    }

    #[test]
    fn snapshot_format_verifier_diff_missing_program() {
        let stats_a = vec![
            ProgStats {
                name: "enqueue".into(),
                verified_insns: 500,
            },
            ProgStats {
                name: "new_prog".into(),
                verified_insns: 100,
            },
        ];
        let stats_b = vec![ProgStats {
            name: "enqueue".into(),
            verified_insns: 500,
        }];
        insta::assert_snapshot!(format_verifier_diff("A", &stats_a, "B", &stats_b));
    }

    // -----------------------------------------------------------------------
    // extract_verifier_log — log extraction + cross-check against
    // parse_sched_output so the two slicers stay consistent on shared input.
    // -----------------------------------------------------------------------

    #[test]
    fn extract_verifier_log_between_begin_end_markers() {
        // libbpf wraps the verifier log between explicit marker lines;
        // the extractor returns the content between them, trimmed of
        // the BEGIN newline and the trailing libbpf END prefix.
        let blob = "\
            unrelated preamble\n\
            libbpf: -- BEGIN PROG LOAD LOG --\n\
            processed 1234 insns (limit 1000000) max_states_per_insn 5 total_states 200 peak_states 50 mark_read 10\n\
            libbpf: -- END PROG LOAD LOG --\n\
            trailing diagnostics\n";
        let log = extract_verifier_log(blob).expect("markers present");
        assert!(log.contains("processed 1234 insns"));
        assert!(!log.contains("BEGIN PROG LOAD LOG"));
        assert!(!log.contains("END PROG LOAD LOG"));
    }

    #[test]
    fn extract_verifier_log_returns_none_when_markers_absent() {
        // Backward compat: logs without the libbpf markers are treated
        // as "no markers" — the caller falls back to using the raw blob.
        assert!(extract_verifier_log("no markers in here").is_none());
        assert!(extract_verifier_log("only BEGIN marker -- BEGIN PROG LOAD LOG --").is_none());
    }

    #[test]
    fn extract_verifier_log_consistent_with_parse_sched_output() {
        // `collect_verifier_output` chains parse_sched_output →
        // extract_verifier_log on the VM stdout blob. Both slicers
        // operate on the same input without duplicating work, so a
        // single SCHED_OUTPUT block that wraps a libbpf-marked verifier
        // log must produce the same verifier text when extracted in
        // that order.
        let sched_inner = "\
            libbpf: -- BEGIN PROG LOAD LOG --\n\
            processed 7 insns (limit 1000000) max_states_per_insn 1 total_states 1 peak_states 1 mark_read 0\n\
            libbpf: -- END PROG LOAD LOG --\n";
        let vm_output = format!(
            "kernel boot junk\n{SCHED_OUTPUT_START}\n{sched_inner}{SCHED_OUTPUT_END}\nafterward\n",
        );
        let sched = parse_sched_output(&vm_output).expect("SCHED_OUTPUT block");
        let verifier_log = extract_verifier_log(sched).expect("verifier markers");
        assert!(verifier_log.contains("processed 7 insns"));
        assert!(!verifier_log.contains("SCHED_OUTPUT"));
        assert!(!verifier_log.contains("BEGIN PROG LOAD LOG"));
    }

    #[test]
    fn parse_sched_output_valid() {
        let output = format!(
            "noise\n{SCHED_OUTPUT_START}\nscheduler log line 1\nline 2\n{SCHED_OUTPUT_END}\nmore"
        );
        let parsed = parse_sched_output(&output);
        assert!(parsed.is_some());
        let content = parsed.unwrap();
        assert!(content.contains("scheduler log line 1"));
        assert!(content.contains("line 2"));
    }

    #[test]
    fn parse_sched_output_missing_start() {
        let output = format!("no start\n{SCHED_OUTPUT_END}\n");
        assert!(parse_sched_output(&output).is_none());
    }

    #[test]
    fn parse_sched_output_missing_end() {
        let output = format!("{SCHED_OUTPUT_START}\nsome content");
        assert!(parse_sched_output(&output).is_none());
    }

    #[test]
    fn parse_sched_output_empty_content() {
        let output = format!("{SCHED_OUTPUT_START}\n\n{SCHED_OUTPUT_END}");
        assert!(parse_sched_output(&output).is_none());
    }

    #[test]
    fn parse_sched_output_with_stack_traces() {
        let stack = "do_enqueue_task+0x1a0/0x380\nbalance_one+0x50/0x100\n";
        let output = format!("{SCHED_OUTPUT_START}\n{stack}\n{SCHED_OUTPUT_END}");
        let parsed = parse_sched_output(&output).unwrap();
        assert!(parsed.contains("do_enqueue_task"));
        assert!(parsed.contains("balance_one"));
    }

    #[test]
    fn parse_sched_output_rfind_survives_end_marker_in_content() {
        // Regression: if the scheduler log echoes the END marker
        // inside its own content (e.g. a shell heredoc, a diagnostic
        // that quotes the sentinel), `find` truncated the section at
        // the first occurrence — which was inside the content, not
        // at the terminator. `rfind` anchors on the last occurrence,
        // which is the real terminator.
        let content = format!("line1\nfake {SCHED_OUTPUT_END} inside\nline3");
        let output = format!("{SCHED_OUTPUT_START}\n{content}\n{SCHED_OUTPUT_END}\n");
        let parsed = parse_sched_output(&output).unwrap();
        assert!(
            parsed.contains("line3"),
            "rfind must keep content after an embedded END marker: {parsed:?}"
        );
        assert!(
            parsed.contains("fake"),
            "content before the embedded marker must also survive: {parsed:?}"
        );
    }

    // -- parse_sched_output_partial --

    #[test]
    fn parse_sched_output_partial_well_formed_matches_strict() {
        // When both delimiters are present, the partial parser
        // returns the same content as the strict parser.
        let output = format!(
            "noise\n{SCHED_OUTPUT_START}\nscheduler log line 1\nline 2\n{SCHED_OUTPUT_END}\nmore"
        );
        assert_eq!(
            parse_sched_output_partial(&output),
            parse_sched_output(&output),
        );
    }

    #[test]
    fn parse_sched_output_partial_missing_end_returns_partial() {
        // When SCHED_OUTPUT_END is absent (scheduler crashed mid-run
        // before writing the closing delimiter), the partial parser
        // returns content from after SCHED_OUTPUT_START to end of
        // buffer. The strict parser returns None for the same input.
        let output = format!("{SCHED_OUTPUT_START}\nstack frame 1\nstack frame 2");
        assert!(parse_sched_output(&output).is_none());
        let partial = parse_sched_output_partial(&output).unwrap();
        assert!(partial.contains("stack frame 1"));
        assert!(partial.contains("stack frame 2"));
    }

    #[test]
    fn parse_sched_output_partial_missing_start_returns_none() {
        // No start marker → no content recoverable. The end-marker-
        // only case is unrecoverable: we cannot infer where the log
        // begins.
        let output = format!("garbage\n{SCHED_OUTPUT_END}\n");
        assert!(parse_sched_output_partial(&output).is_none());
    }

    #[test]
    fn parse_sched_output_partial_empty_content_returns_none() {
        // Start marker present but no payload after it.
        let output = format!("{SCHED_OUTPUT_START}\n");
        assert!(parse_sched_output_partial(&output).is_none());
    }
}
