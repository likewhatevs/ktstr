//! Shared host-side failure-dump readers for the dump-assertion e2e
//! tests (failure_dump, cast_analysis, vm_integration, silent_drop).
//!
//! Each test boots a stall scenario and asserts on the freeze
//! coordinator's host-written `FailureDumpReport` from a
//! `post_vm_unconditional` callback. The guest is a separate process
//! and cannot read the host-side dump, so the read must happen on the
//! host via `VmResult` — never from inside the in-guest scenario body
//! (an in-guest read fails, and under `expect_err` that failure is
//! inverted to PASS, making the assertion decorative).
//!
//! Two readers, differing only in what they treat as INCONCLUSIVE
//! (a `post_vm_skip`, which the eval gate turns into a test SKIP) vs
//! what they leave for the caller to assert (a REGRESSION → hard FAIL):
//!
//! - [`read_dump_skip_placeholder`] skips ONLY a PLACEHOLDER dump (the
//!   freeze captured no real state). A real dump — even a partial one
//!   with empty `maps` — is returned; the caller asserts its
//!   invariants, and an empty `maps` in a real dump is a silent-drop
//!   REGRESSION the caller must catch, not skip. Use this for tests
//!   whose purpose is exactly to catch empty/dropped capture
//!   (dump-level invariants: `maps` non-empty, `vcpu_regs` non-empty,
//!   the schema discriminant, the scx walker / perf / event-counter
//!   facets).
//! - [`read_failure_dump`] also skips a PARTIAL dump (empty `maps`):
//!   the owned_accessor / dump_btf was not ready before the freeze, so
//!   no BPF map rendered. Use this for tests that assert on rendered
//!   map CONTENT (a specific `.bss` field, arena chase, TASK_STORAGE
//!   entry) — a partial dump cannot carry that content, so the render
//!   under test is unverifiable (inconclusive). The "map enumeration
//!   produced nothing at all" regression is caught by the
//!   `read_dump_skip_placeholder` invariant tests, not here.
#![allow(dead_code)]

use anyhow::{Context, Result};
use ktstr::prelude::{VmResult, post_vm_skip};

/// A bounded diagnostic that points at the authoritative JSON artifact.
///
/// Failure dumps can contain captured arena pages and other large byte arrays.
/// Embedding the parsed top-level value in an assertion error duplicates that
/// artifact onto nextest's stdout and can emit hundreds of megabytes during a
/// storm. Keep assertion messages actionable without copying the payload.
pub fn failure_dump_artifact(result: &VmResult) -> String {
    match result.failure_dump_path() {
        Ok(path) => format!("full dump preserved at {}", path.display()),
        Err(error) => format!("full dump path unavailable: {error:#}"),
    }
}

/// Reads + parses the freeze coordinator's host-side failure dump for
/// `result`, skipping (via [`post_vm_skip`]) only when the dump is a
/// PLACEHOLDER — the freeze captured no real state (rendezvous timeout
/// or the coordinator exited before walking). A real dump is returned
/// even if `maps` is empty: an empty `maps` in a non-placeholder dump is
/// a silent-drop REGRESSION the caller asserts on, not an inconclusive
/// skip. A genuinely-unreadable or non-JSON dump is a hard error.
pub fn read_dump_skip_placeholder(result: &VmResult) -> Result<serde_json::Value> {
    let dump_path = result.failure_dump_path()?;
    let json = std::fs::read_to_string(&dump_path).with_context(|| {
        format!(
            "failure dump file missing at {} — freeze coordinator did not \
             write the JSON dump (SCX_EXIT_ERROR_STALL latch did not fire or \
             the write failed silently)",
            dump_path.display()
        )
    })?;
    let value: serde_json::Value = serde_json::from_str(&json)
        .map_err(|e| anyhow::anyhow!("dump file is not valid JSON: {e}"))?;
    let is_placeholder = value
        .get("is_placeholder")
        .and_then(|p| p.as_bool())
        .unwrap_or(false);
    if is_placeholder {
        return Err(post_vm_skip(
            "failure dump is a placeholder — the freeze captured no real state \
             (rendezvous timeout, or the coordinator exited before walking), so \
             the assertion under test cannot be verified",
        ));
    }
    Ok(value)
}

/// Like [`read_dump_skip_placeholder`], but ALSO skips when the dump has
/// no rendered BPF `maps`: the owned_accessor / dump_btf was not ready
/// before the freeze (a PARTIAL dump under VM starvation, with
/// `is_placeholder=false`). For tests that assert on rendered map
/// CONTENT — a partial dump cannot carry it, so the render under test is
/// unverifiable (inconclusive).
pub fn read_failure_dump(result: &VmResult) -> Result<serde_json::Value> {
    let value = read_dump_skip_placeholder(result)?;
    let no_maps = value
        .get("maps")
        .and_then(|m| m.as_array())
        .is_none_or(|m| m.is_empty());
    if no_maps {
        return Err(post_vm_skip(
            "failure dump has no rendered BPF maps — the owned_accessor / \
             dump_btf was not ready before the freeze (a partial dump under VM \
             starvation), so the map render under test cannot be verified",
        ));
    }
    Ok(value)
}
