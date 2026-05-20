//! Diagnostic snapshot capture and traversal.
//!
//! Test scenarios use [`Op::CaptureSnapshot`](crate::scenario::ops::Op::CaptureSnapshot)
//! to request a host-side diagnostic capture mid-run. The capture
//! result — a `crate::monitor::dump::FailureDumpReport` — is keyed by the `name` argument
//! and stored on the scenario's [`SnapshotBridge`], where downstream
//! test code reaches it via [`Snapshot`] for typed traversal of
//! BTF-rendered map values, per-CPU entries, and scalar variables.
//!
//! # Lifecycle
//!
//! 1. **Wire-up.** Before [`execute_steps`](crate::scenario::ops::execute_steps)
//!    runs, host orchestration installs a [`SnapshotBridge`] in the
//!    current thread via [`SnapshotBridge::set_thread_local`]. The
//!    bridge owns the storage map and a callable that performs the
//!    capture.
//!
//! 2. **Capture.** When the executor reaches `Op::CaptureSnapshot { name }`,
//!    it invokes [`SnapshotBridge::capture`] with the name. The
//!    closure performs the freeze rendezvous (request/reply with
//!    the freeze coordinator), builds a `crate::monitor::dump::FailureDumpReport`, and
//!    returns it; the bridge stores it under the name.
//!
//! 3. **Inspection.** After the scenario completes, the test author
//!    pulls captured reports out via [`SnapshotBridge::drain`] and
//!    constructs [`Snapshot`] views to assert against rendered
//!    values:
//!    `snapshot.var("nr_cpus_onln").as_u64()? > 0`,
//!    `snapshot.map("scx_per_task")?.find(|e| e.get("tid").as_i64()? == pid)?`.
//!
//! # On-demand vs error-trigger captures
//!
//! `Op::CaptureSnapshot` requests are orthogonal to the error-class freeze
//! path. The freeze coordinator's existing state machine for
//! `SCX_EXIT_ERROR` triggers (Idle → TookEarly → Done) governs the
//! *unsolicited* capture pipeline; on-demand captures funnel
//! through a separate request/reply channel and never touch the
//! error-trigger state. The coordinator services on-demand requests
//! even after Done so post-failure scenarios can still snapshot
//! state for context. The serialisation rule: at most one capture in
//! flight at a time — the on-demand path waits for the previous
//! capture's vCPUs to fully return to `parked == false` before
//! issuing the next freeze request, mirroring the rendezvous
//! invariants the error-trigger path already obeys.
//!
//! # Guest → host wire: ioeventfd doorbell (locked)
//!
//! The guest-driven capture trigger uses an in-kernel ioeventfd
//! doorbell, NOT a synchronous MMIO `BusDevice` arm. Per user
//! direction:
//!
//! 1. Host registers an ioeventfd at a dedicated MMIO GPA inside
//!    the existing MMIO gap (e.g. `MMIO_GAP_START + 0x3000`) via
//!    `KVM_IOEVENTFD`. The exact GPA is arch-dependent —
//!    `MMIO_GAP_START + 0x3000` on x86_64,
//!    `VIRTIO_NET_MMIO_BASE + VIRTIO_MMIO_SIZE` on aarch64. The
//!    fd is owned by the freeze coordinator and polled alongside
//!    its existing wake sources.
//! 2. Guest [`Op::CaptureSnapshot`](crate::scenario::ops::Op::CaptureSnapshot)
//!    handler `mmap`s `/dev/mem` to reach the doorbell GPA (same
//!    pattern the SHM ring already uses) and writes the tag value
//!    plus a serial counter into a small per-call slot, then
//!    writes the doorbell. KVM dispatches the write in-kernel and
//!    raises the eventfd; the vCPU thread does NOT exit to
//!    userspace for the doorbell write itself.
//! 3. The freeze coordinator wakes on `eventfd_signal`, reads the
//!    tag from the slot, runs `freeze_and_capture`, builds the
//!    `crate::monitor::dump::FailureDumpReport`, and stores it on the bridge keyed by
//!    that tag. Reply to the guest is implicit — the
//!    [`SnapshotBridge::capture`] callback installed in the
//!    executor's thread-local blocks on a per-request reply
//!    eventfd / completion channel paired with the doorbell.
//!
//! This shape keeps the capture trigger off the vCPU userspace
//! exit path (cleaner — no MMIO `BusDevice` round-trip) and is
//! extensible to higher-rate triggers without redesigning the
//! wire. The [`SnapshotBridge`] surface defined below is the
//! integration point; `ioeventfd` is the wake mechanism that
//! drives the `CaptureCallback` from the guest side. The guest
//! [`Op::WatchSnapshot`](crate::scenario::ops::Op::WatchSnapshot)
//! registration uses the same doorbell at scenario setup
//! (separate tag namespace) so symbol resolution + user
//! watchpoint slot allocation happen on the host without a vCPU
//! userspace exit.
//!
//! # No-bridge fallback
//!
//! When `Op::CaptureSnapshot` runs in a context with no installed bridge
//! (e.g. unit tests that exercise the executor without spinning up
//! a VM), the op is a no-op with a `tracing::warn!`. Existing
//! scenarios that do not declare snapshot ops keep working
//! unchanged.
//!
//! # Field accessor traversal
//!
//! [`SnapshotMap`], [`SnapshotEntry`], and [`SnapshotField`] form a
//! lazy borrow chain over the report. Dotted-path lookups (e.g.
//! `entry.get("ctx.weight.value")`) walk
//! `RenderedValue::Struct` members by name and follow
//! `RenderedValue::Ptr` dereferences transparently — the test
//! author writes the dotted path the BTF source would suggest;
//! pointer chasing is invisible.
//!
//! Missing fields land in [`SnapshotField::Missing`] with an
//! actionable error string identifying the path component that
//! could not be resolved AND the available alternatives at that
//! level. Terminal accessors (`as_u64`, `as_i64`, `as_bool`,
//! `as_str`) return `Result<T, SnapshotError>` so an absent /
//! type-mismatched field bubbles up as a recoverable error rather
//! than panicking.
//!
//! # Cross-surface accessor vocabulary
//!
//! [`SnapshotField`], [`JsonField`], and
//! `crate::monitor::btf_render::RenderedValue` share a uniform
//! method vocabulary so a test author moves between the
//! BTF-rendered (BPF maps + globals), JSON-rendered (scheduler
//! stats), and raw-tree surfaces without re-learning syntax:
//!
//! | Method                | What it does                                                     |
//! |-----------------------|------------------------------------------------------------------|
//! | `.as_u64()`/`.as_i64()`/`.as_f64()`/`.as_bool()` | Typed scalar extract.                  |
//! | `.as_str()`           | UTF-8 string extract (Enum variant / JSON string).               |
//! | `.as_u64_array()` / `.as_u32_array()` / `.as_i64_array()` / `.as_f64_array()` / `.as_bool_array()` | Element-typed array extract. |
//! | `.get(path)`          | Dotted-path walk (`"a.b.c"`); returns a typed sub-view.          |
//! | `.member(name)`       | Single-step struct-member walk (RenderedValue only; no dots).    |
//! | `.index(i)`           | Array element by 0-indexed position (RenderedValue only).        |
//! | `.raw()`              | Drop into the underlying RenderedValue for raw Option-returning navigation. |
//!
//! The wrapper types ([`SnapshotField`], [`JsonField`]) return
//! `Result` with rich [`SnapshotError`] context; the raw
//! `RenderedValue` layer returns `Option` (the caller has already
//! pattern-matched into a known variant, so absence is a
//! programming-error class handled locally). Convert between
//! layers with `SnapshotField::raw()`.
//!
//! For multi-scheduler scenarios (after
//! [`crate::scenario::ops::Op::ReplaceScheduler`] or two
//! [`crate::scenario::ops::Op::AttachScheduler`] calls), use
//! [`Snapshot::active`] to project the view to the currently-
//! attached scheduler's maps and chain the standard accessors
//! against it. [`Snapshot::live_var`] is the shorthand for
//! `self.active()?.var(name)`; [`Snapshot::vars`] iterates every
//! captured copy when the framework cannot determine "active"
//! automatically.

/// Maximum number of rendered keys captured into
/// [`SnapshotError::NoMatch::available_keys`] during a failed
/// `find` / `max_by` traversal. Three is a balance between
/// disambiguation power (enough to suggest the keyspace shape) and
/// failure-message readability (does not overrun a terminal line).
pub(super) const NO_MATCH_KEY_SAMPLE: usize = 3;

/// Maximum number of characters each rendered key in
/// [`SnapshotError::NoMatch::available_keys`] retains before being
/// truncated with a trailing `…`. Wide struct keys (e.g. a
/// 50-field `task_ctx`) would otherwise produce kilobytes of
/// failure text per sampled key.
pub(super) const NO_MATCH_KEY_CHAR_CAP: usize = 80;

/// Discriminator that `render_entry_key`'s fallback path prepends
/// to the raw `key_hex` bytes when an entry's BTF-rendered key was
/// missing at capture time. [`SnapshotError::NoMatch`]'s `Display`
/// impl uses the same prefix as the gate for its BTF-missing hint
/// (when every sampled key starts with this string, BTF was
/// uniformly absent for the map's key type and the hint points the
/// operator at `CONFIG_DEBUG_INFO_BTF=y`). Naming the producer +
/// consumer contract once here keeps a future rename of one side
/// from silently desynchronising the other. Test sites in this
/// module intentionally retain the literal `"hex:"` so they pin the
/// value separately from the const that synchronises production.
pub(super) const HEX_KEY_PREFIX: &str = "hex:";

mod error;

pub use error::{DrainedSnapshotEntry, MissingStatsReason, SnapshotError, SnapshotResult};

pub mod bridge;

pub use bridge::{
    BridgeGuard, CaptureCallback, KernelOpCallback, MAX_STORED_EVENTS, MAX_STORED_SNAPSHOTS,
    MAX_WATCH_SNAPSHOTS, SnapshotBridge, SnapshotBridgeEvent, WatchRegisterCallback,
    with_active_bridge,
};

mod entry;
mod field;
mod json;
mod view;

pub use entry::SnapshotEntry;
pub use field::SnapshotField;
pub(crate) use field::walk_dotted_path;
pub use json::{JsonField, stats_path};
pub use view::{Snapshot, SnapshotMap};

// ---------------------------------------------------------------------------
// Snapshot view over a captured FailureDumpReport
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// SnapshotEntry
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests;
