//! Cgroup v2 filesystem operations for test cgroup management.
//!
//! Creates, configures, and removes cgroups under a parent path
//! (default `/sys/fs/cgroup/ktstr`). Provides cpuset assignment,
//! task migration, and cleanup.
//!
//! # Walk root (cgroup-v2 delegation)
//!
//! [`CgroupManager`] carries a `walk_root` that bounds two operations:
//! - [`CgroupManager::setup`] walks every ancestor's
//!   `cgroup.subtree_control` from `walk_root` down to `parent`;
//! - [`CgroupManager::drain_tasks`] / `cleanup_recursive` drain pids
//!   into `{walk_root}/cgroup.procs` (a writable root that is exempt
//!   from the kernel's no-internal-process constraint).
//!
//! `walk_root` defaults to `/sys/fs/cgroup` (Mode A: root-owned cgroup
//! tree). [`CgroupManager::with_walk_root`] retargets it for Mode B/C
//! delegation (systemd `Delegate=yes`, container `nsdelegate`) where
//! the operator owns `subtree_control` writes only inside a delegated
//! subtree. The constructor enforces that `parent` is at or below
//! `walk_root` so the strip-prefix walk cannot escape.
//!
//! # Controller surface
//!
//! [`CgroupManager`] enables a fixed controller set in
//! `cgroup.subtree_control` at `Self::setup` time so every method
//! that writes a controller knob succeeds without per-call lazy
//! enablement (which would race against concurrent sibling cgroup
//! creation). The enabled controllers and the knobs each one exposes
//! map to:
//!
//! | Controller | `setup` writes | Methods that touch the controller's files |
//! |------------|----------------|-------------------------------------------|
//! | `cpuset`   | when `Controller::Cpuset` in the set passed to `setup` (runtime adds it when a `CgroupDef` declares `cpuset`/`cpuset_mems`) | `Self::set_cpuset`, `Self::set_cpuset_mems`, `Self::clear_cpuset`, `Self::clear_cpuset_mems` |
//! | `cpu`      | when `Controller::Cpu` in the set passed to `setup` (runtime adds it when a `CgroupDef` declares `cpu`) | `Self::set_cpu_max`, `Self::set_cpu_weight` |
//! | `memory`   | when `Controller::Memory` in the set passed to `setup` (runtime adds it when a `CgroupDef` declares `memory`) | `Self::set_memory_max`, `Self::set_memory_high`, `Self::set_memory_low`, `Self::set_memory_swap_max` |
//! | `pids`     | when `Controller::Pids` in the set passed to `setup` (runtime adds it when a `CgroupDef` declares `pids`) | `Self::set_pids_max` |
//! | `io`       | when `Controller::Io` in the set passed to `setup` (runtime adds it when a `CgroupDef` declares `io`) | `Self::set_io_weight` |
//! | (cgroup-core) | not gated   | `Self::set_freeze`, `Self::move_task`, `Self::move_tasks` |
//!
//! `cgroup.freeze` and `cgroup.procs` are cgroup-core files exposed on
//! every non-root cgroup automatically; they do not require a
//! controller in `subtree_control`. `memory.swap.max` only exists when
//! the kernel was built with `CONFIG_SWAP=y` — the file is absent on
//! swap-disabled kernels and a write returns ENOENT (callers route
//! through the wire-time error chain).
//!
//! # Untrusted-name validation
//!
//! Cgroup names flow into [`Path::join`] under `parent` to address
//! files inside cgroupfs. `validate_cgroup_name` rejects shapes that
//! would escape that parent (`..`, absolute leading `/`, `NUL`) or
//! that produce invisible cgroupfs entries (leading `.`); other ASCII
//! is passed through to the kernel which is the final authority on
//! per-component validity. Every public method that takes a `name`
//! validates it before any filesystem write.

use crate::topology::TestTopology;
use anyhow::{Context, Result, anyhow, bail};
use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc;
use std::time::Duration;

/// Cgroup v2 controllers that [`CgroupManager::setup`] can enable in
/// `cgroup.subtree_control`.
///
/// Each variant maps to a literal token the kernel parses in
/// `cgroup_subtree_control_write`. The enum is exhaustive over the
/// controllers the framework's [`CgroupOps`] surface actually writes
/// to (cpuset, cpu, memory, pids, io); cgroup-core knobs
/// (`cgroup.freeze`, `cgroup.procs`) are not gated by any controller
/// and never appear here.
///
/// Callers pass a `BTreeSet<Controller>` to `setup` — sets compose
/// naturally across nested CgroupDef declarations and the deterministic
/// `BTreeSet` iteration order keeps the rendered subtree_control write
/// stable between runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Controller {
    /// `+cpuset` — gates `cpuset.cpus`, `cpuset.cpus.effective`,
    /// `cpuset.mems`, `cpuset.mems.effective` files on every child.
    Cpuset,
    /// `+cpu` — gates `cpu.max`, `cpu.weight`, `cpu.weight.nice`,
    /// `cpu.stat`, `cpu.pressure` files on every child.
    Cpu,
    /// `+memory` — gates `memory.max`, `memory.high`, `memory.low`,
    /// `memory.min`, `memory.current`, `memory.swap.max`,
    /// `memory.events`, `memory.stat`, `memory.pressure` files.
    Memory,
    /// `+pids` — gates `pids.max`, `pids.current`, `pids.events` files.
    Pids,
    /// `+io` — gates `io.max`, `io.weight`, `io.bfq.weight`,
    /// `io.stat`, `io.pressure` files.
    Io,
}

impl Controller {
    /// Kernel token written to `cgroup.subtree_control` (the bare name
    /// without the `+`/`-` prefix; see `Self::as_subtree_control_add`
    /// for the full token).
    pub fn name(self) -> &'static str {
        match self {
            Controller::Cpuset => "cpuset",
            Controller::Cpu => "cpu",
            Controller::Memory => "memory",
            Controller::Pids => "pids",
            Controller::Io => "io",
        }
    }
}

/// Default timeout for cgroup filesystem writes. Normally <1ms; 2s catches
/// real hangs without waiting so long the test result is meaningless.
const CGROUP_WRITE_TIMEOUT: Duration = Duration::from_secs(2);

/// Write `data` to `path` with a timeout. Spawns a thread for the blocking
/// `fs::write` and waits on a channel. If the write does not complete within
/// `timeout`, returns an error (the spawned thread may still be blocked in
/// the kernel but will not prevent the caller from making progress).
///
/// # Stranded-writer thread semantics
///
/// On timeout the helper returns `Err` while the spawned thread stays
/// blocked in the kernel inside `fs::write` — typically inside the
/// cgroupfs `cgroup_kn_lock_live` / `cgroup_mutex` lock acquisition or
/// the per-file `kn->active` semaphore. The host-side fd to `path` is
/// owned by the spawned thread, so:
///
/// - **Per-file lock retention.** While the writer is blocked, the
///   target cgroupfs file's `kn->active` (kernfs's per-knob writer
///   semaphore) remains held by the stranded thread. Concurrent
///   writes to the SAME file from any thread in the same process —
///   including this same caller's retry — will queue behind the
///   stranded write inside the kernel. Writes to OTHER files in the
///   same cgroup are unaffected (kernfs holds `kn->active`
///   per-knob, not per-cgroup).
/// - **Thread-handle drop.** The `JoinHandle` returned by
///   `thread::spawn` is dropped when the helper returns. Rust's
///   `JoinHandle::Drop` implementation detaches the thread without
///   waiting; the thread continues to run and is implicitly joined
///   when the kernel write eventually unblocks (or when the process
///   exits).
/// - **Bounded leak under wedged cgroupfs.** A genuinely-wedged
///   cgroupfs (e.g. a stuck filesystem driver in the kernel) would
///   accumulate threads at a rate of one per timed-out write site.
///   The 2s per-write timeout caps the per-site stall to 2s; the
///   total accumulation is driven by how many distinct write sites
///   the scenario hits, not by elapsed wall-clock time alone.
///   Operators noticing stranded `<defunct>` cgroupfs writers in
///   `ps` should investigate whether the underlying kernel cgroup
///   subsystem is hung; the framework's own teardown does not
///   block on these stranded threads.
///
/// Each stranded thread holds the file's `kn->active` until the
/// kernel write returns. The OS-level memory cost per stranded
/// thread is the default Rust thread stack (8 MiB on Linux, mostly
/// virtual until touched).
fn write_with_timeout(path: &Path, data: &str, timeout: Duration) -> Result<()> {
    let display = path.display().to_string();
    let path = path.to_owned();
    let data = data.to_owned();
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let result = fs::write(&path, &data);
        let _ = tx.send(result);
    });
    match rx.recv_timeout(timeout) {
        Ok(Ok(())) => Ok(()),
        Ok(Err(e)) => {
            let errno_suffix = e
                .raw_os_error()
                .and_then(crate::errno_name)
                .map(|name| format!(" ({name})"))
                .unwrap_or_default();
            Err(e).with_context(|| format!("write {display}{errno_suffix}"))
        }
        Err(_) => bail!(
            "cgroup write to {display} timed out after {}ms",
            timeout.as_millis()
        ),
    }
}

/// Validate a cgroup name before joining it onto the parent path.
///
/// Rejects shapes that would either escape the parent directory
/// (`..` component, absolute leading `/`, embedded NUL) or produce
/// a hidden / invisible cgroupfs entry (leading `.`). Empty names
/// are also rejected — `parent.join("")` returns `parent`, which
/// would let a caller accidentally clobber the parent's own
/// `cpuset.cpus` / `cgroup.subtree_control` files via a method
/// that expected to address a child.
///
/// Permits `/` only as a path separator between non-empty
/// components (nested cgroups like `"cg_0/narrow"`); a leading
/// `/` is rejected because `Path::join` would replace `parent`
/// entirely with the absolute path.
///
/// Beyond these structural checks the kernel is the final authority
/// on per-component validity: cgroupfs rejects names containing
/// newlines or names colliding with reserved knobs (`cgroup.procs`,
/// `cpuset.cpus`, etc.) at `mkdir` time with EINVAL / EEXIST. Those
/// failures surface through the regular `fs::create_dir_all` /
/// `fs::write` error chain.
pub(crate) fn validate_cgroup_name(name: &str) -> Result<()> {
    if name.is_empty() {
        bail!("cgroup name must not be empty");
    }
    if name.starts_with('/') {
        bail!(
            "cgroup name '{name}' starts with '/' — would escape the \
             managed parent via Path::join (absolute paths replace the \
             join base)"
        );
    }
    if name.contains('\0') {
        bail!("cgroup name '{name}' contains a NUL byte");
    }
    // Per-component checks run before the whole-name leading-dot
    // check so a component like `..` matches the more specific
    // path-traversal diagnostic instead of the generic hidden-entry
    // one. The ordering matters for error messages — `'..' component`
    // is what callers grep for.
    for component in name.split('/') {
        if component.is_empty() {
            bail!(
                "cgroup name '{name}' contains an empty path component \
                 (consecutive '/') — Path::join would emit a malformed path"
            );
        }
        if component == ".." {
            bail!(
                "cgroup name '{name}' contains a '..' component — \
                 would escape the managed parent via Path::join"
            );
        }
        if component == "." {
            bail!(
                "cgroup name '{name}' contains a '.' component — \
                 ambiguous self-reference, refuse before fs writes"
            );
        }
        if component.starts_with('.') {
            bail!(
                "cgroup name '{name}' contains a leading-dot component \
                 ('{component}') — produces a hidden cgroupfs entry"
            );
        }
    }
    Ok(())
}

/// Walk an `anyhow::Error` chain and return the first
/// `std::io::Error`'s raw errno, if any. Shared helper for errno
/// classification across cgroup orchestration — both this module's
/// ESRCH/EBUSY checks and [`crate::vmm::cgroup_sandbox`]'s
/// EACCES/EPERM/EBUSY branches walk the same chain shape.
pub(crate) fn anyhow_first_io_errno(err: &anyhow::Error) -> Option<i32> {
    err.chain()
        .find_map(|cause| cause.downcast_ref::<std::io::Error>())
        .and_then(|io| io.raw_os_error())
}

/// ESRCH: task exited between listing and migration
/// (`cgroup_procs_write_start` -> `find_task_by_vpid` returns NULL).
fn is_esrch(err: &anyhow::Error) -> bool {
    anyhow_first_io_errno(err) == Some(libc::ESRCH)
}

/// EBUSY: either the cgroup v2 no-internal-process constraint
/// (`cgroup_migrate_vet_dst` when `subtree_control` is set) or a
/// transient rejection from a sched_ext BPF `cgroup_prep_move`
/// callback (`scx_cgroup_can_attach`).
fn is_ebusy(err: &anyhow::Error) -> bool {
    anyhow_first_io_errno(err) == Some(libc::EBUSY)
}

/// Snapshot the cgroup-tree state at the moment a cpuset.cpus
/// write fails, for diagnostic attachment to the returned error.
///
/// Captures (per the diagnostic contract on
/// [`CgroupManager::set_cpuset`]):
/// - the parent's `cgroup.controllers` (controllers AVAILABLE for
///   children — confirms whether subtree_control already
///   propagated to this child)
/// - the parent's `cgroup.subtree_control` (controllers ENABLED
///   for children — what setup() last wrote)
/// - the child's `cgroup.controllers` (the set children of the
///   CHILD inherit — useful for nested cgroups)
/// - whether `cpuset.cpus` exists at the child (distinguishes a
///   "controller never propagated" failure mode from a
///   "kernel rejected this specific value" failure mode)
/// - the child's directory listing (so an unexpected presence/
///   absence of any cgroupfs knob is visible)
///
/// Read failures inside the snapshot are folded into the snapshot
/// string as `<read failed: {err}>` rather than propagating —
/// the caller's error path is what the caller cares about; the
/// snapshot is best-effort instrumentation.
fn capture_cpuset_state(parent: &Path, name: &str) -> String {
    let child = parent.join(name);
    let parent_controllers = read_or_label(&parent.join("cgroup.controllers"));
    let parent_subtree_control = read_or_label(&parent.join("cgroup.subtree_control"));
    let child_controllers = read_or_label(&child.join("cgroup.controllers"));
    let cpuset_cpus_exists = child.join("cpuset.cpus").exists();
    let child_listing = match fs::read_dir(&child) {
        Ok(entries) => {
            let mut names: Vec<String> = entries
                .filter_map(|e| e.ok())
                .map(|e| e.file_name().to_string_lossy().into_owned())
                .collect();
            names.sort_unstable();
            format!("[{}]", names.join(", "))
        }
        Err(e) => format!("<read_dir failed: {e}>"),
    };
    format!(
        "cgroup-state-snapshot: \
         parent={} name={} \
         parent.cgroup.controllers={:?} \
         parent.cgroup.subtree_control={:?} \
         child.cgroup.controllers={:?} \
         child.cpuset.cpus.exists={} \
         child.listing={}",
        parent.display(),
        name,
        parent_controllers,
        parent_subtree_control,
        child_controllers,
        cpuset_cpus_exists,
        child_listing,
    )
}

/// Read `path` to a string for snapshotting, returning a
/// `<...>` placeholder if the read fails. Used by
/// [`capture_cpuset_state`] so a missing or permission-denied
/// snapshot field shows up as a labeled placeholder rather than
/// killing the whole snapshot.
fn read_or_label(path: &Path) -> String {
    match fs::read_to_string(path) {
        Ok(s) => s.trim().to_string(),
        Err(e) => format!("<read failed: {e}>"),
    }
}

/// Cap on the number of successive [`CgroupManager::remove_cgroup`]
/// failures the manager will tolerate before bailing further removes.
///
/// A churn workload (rapid create→remove cycles) may legitimately
/// race the freeze/drain path and see EBUSY/ENOENT on individual
/// remove calls — those are absorbed and the un-removed cgroup is
/// counted toward `outstanding_removes`. When the counter exceeds
/// this cap, subsequent [`CgroupManager::remove_cgroup`] calls
/// return Err immediately so the loop driving the churn (e.g.
/// `custom_cgroup_rapid_churn` in scenario/dynamic.rs) can bail
/// instead of accumulating cgroupfs entries unboundedly. Successful
/// removes decrement the counter, so a transient stall that
/// eventually clears does not strand the manager in the bailed
/// state.
const MAX_OUTSTANDING_REMOVES: usize = 10;

/// RAII manager for cgroup v2 filesystem operations.
///
/// Creates, configures, and removes cgroups under a parent directory.
/// Provides cpuset assignment and task migration.
///
/// # Outstanding-remove tracking
///
/// `outstanding_removes` counts cgroups whose
/// [`Self::remove_cgroup`] call failed (the directory still exists
/// in the cgroupfs tree). It increments on every removal failure,
/// decrements on every removal success, and gates further calls:
/// once the count exceeds `MAX_OUTSTANDING_REMOVES`,
/// [`Self::remove_cgroup`] returns Err without attempting the
/// underlying writes. The counter is `AtomicUsize` because
/// scenario code holds the manager behind `&dyn CgroupOps` and
/// shares it across threads via `&self` borrows.
///
/// # Walk root
///
/// `walk_root` bounds the cgroup-fs walk for two operations:
/// 1. [`Self::setup`] walks every ancestor's `cgroup.subtree_control`
///    between `walk_root` and `parent`.
/// 2. [`Self::drain_tasks`] and `cleanup_recursive` drain pids into
///    `{walk_root}/cgroup.procs` (the writable root exempt from the
///    no-internal-process constraint).
///
/// Defaults to `/sys/fs/cgroup` in [`Self::new`] for Mode A (root-owned
/// cgroup tree). Override via [`Self::with_walk_root`] for cgroup-v2
/// user delegation (Mode B/C: systemd `Delegate=yes`, container
/// `nsdelegate`). The override is validated against `parent` at
/// construction — if `parent` is not at or below `walk_root`, the
/// chained call returns an error rather than letting the strip-prefix
/// walk fall through to an opaque cgroupfs EACCES at the delegation
/// boundary.
#[derive(Debug)]
pub struct CgroupManager {
    parent: PathBuf,
    walk_root: PathBuf,
    outstanding_removes: AtomicUsize,
}

/// Free-function inner of [`CgroupManager::move_tasks`] —
/// extracted so the per-pid migration loop + ESRCH tolerance +
/// all-vanished bail can be unit-tested without a real
/// cgroupfs (which is what surfaces the kernel-side ESRCH that
/// the bail guards against). The per-pid write closure is
/// caller-supplied: production callers route through
/// [`CgroupManager::move_task_with_retry`] (which talks to
/// real `cgroup.procs` files); unit tests pass a closure that
/// synthesises [`std::io::Error::from_raw_os_error`]`(libc::ESRCH)`
/// for selected pids so the partial-vanish (allowed) and
/// all-vanished (bail) paths are both directly observable.
///
/// The empty-slice exemption (`pids.is_empty() -> Ok`) is
/// preserved here so the documented "no move requested" form
/// (post-Drop diagnostic, post-mortem capture) stays a clean
/// no-op rather than tripping the all-vanished gate.
fn move_tasks_inner<W>(name: &str, pids: &[libc::pid_t], mut write_one: W) -> Result<()>
where
    W: FnMut(&str, libc::pid_t) -> Result<()>,
{
    let mut vanished = 0usize;
    for &pid in pids {
        if let Err(e) = write_one(name, pid) {
            if is_esrch(&e) {
                tracing::warn!(pid, cgroup = name, "task vanished during migration");
                vanished += 1;
                continue;
            }
            return Err(e);
        }
    }
    if !pids.is_empty() && vanished == pids.len() {
        anyhow::bail!(
            "move_tasks to '{name}': ALL {n} pid(s) ESRCH'd before \
             migration completed (pids: {pids:?}). Likely causes: \
             (a) `WorkloadHandle::spawn` child pre_exec init-panic \
             cascade (uid/gid/mempolicy/cgroup-handshake failure \
             between fork and the start-pipe read — the parent is \
             blocked on the start-pipe waiting for the child to \
             reach work-ready and only observes the child's death \
             via SIGCHLD reap, by which point the pid has already \
             vanished from any cgroup it was placed in); (b) \
             scheduler-attach-time cgroup-pull (sched_ext init may \
             move existing tasks out of test-created cgroups); \
             (c) external signal (SIGKILL from operator OR \
             OOM-killer). The silent-Ok path this bail replaces \
             was a no-silent-drops violation: a downstream \
             `cgroup.procs` read would see 0 pids with no signal \
             that ANY migration was even attempted. If the caller \
             LEGITIMATELY moves an already-vanished cohort \
             (post-Drop diagnostic), pass an empty pids slice \
             instead — the empty-slice path returns Ok cleanly \
             without bailing.",
            n = pids.len(),
        );
    }
    Ok(())
}

impl CgroupManager {
    /// Default cgroup-fs root used by [`Self::new`]. Override per
    /// instance via [`Self::with_walk_root`] for cgroup-v2 user
    /// delegation.
    const DEFAULT_WALK_ROOT: &'static str = "/sys/fs/cgroup";

    /// Create a manager rooted at the given cgroup v2 path.
    ///
    /// The walk root defaults to `/sys/fs/cgroup` (Mode A: root-owned
    /// cgroup tree). For cgroup-v2 user delegation (Mode B/C), chain
    /// [`Self::with_walk_root`] before any [`Self::setup`] call.
    pub fn new(parent: &str) -> Self {
        Self {
            parent: PathBuf::from(parent),
            walk_root: PathBuf::from(Self::DEFAULT_WALK_ROOT),
            outstanding_removes: AtomicUsize::new(0),
        }
    }

    /// Retarget the cgroup-fs walk root used by [`Self::setup`] and
    /// [`Self::drain_tasks`].
    ///
    /// `root` becomes the upper bound of the
    /// `cgroup.subtree_control` enable walk and the destination
    /// `{root}/cgroup.procs` for pid drains. Use for cgroup-v2 user
    /// delegation (Mode B/C) where the operator owns
    /// `subtree_control` writes only inside the delegated subtree and
    /// a blind walk from `/sys/fs/cgroup` would EACCES at the
    /// `user.slice` / container-root boundary.
    ///
    /// Returns an error when:
    /// - **Either `parent` or `root` contains a `..` component** —
    ///   [`Path::starts_with`](std::path::Path::starts_with) is component-based and treats `..`
    ///   as a literal segment, so `/sys/fs/cgroup/op/../escape` would
    ///   component-prefix `/sys/fs/cgroup/op` while the kernel
    ///   resolves the path to `/sys/fs/cgroup/escape` (outside the
    ///   delegation root). Rejecting `..` upfront keeps the prefix
    ///   invariant honest against canonical-vs-component drift.
    /// - **The manager's `parent` is not at or below `root`** —
    ///   without the prefix invariant the `Self::setup_under_root`
    ///   strip-prefix gate would silently skip the subtree_control
    ///   walk and the caller would see downstream EACCES on the
    ///   first `set_*` write. Surfaces the misconfiguration upfront
    ///   with both paths in the error message.
    pub fn with_walk_root(mut self, root: impl Into<PathBuf>) -> Result<Self> {
        let root = root.into();
        // Reject `..` components on either side. `PathBuf::starts_with`
        // is component-based and treats `..` as a literal segment, so
        // `/sys/fs/cgroup/operator/../escape` would pass the prefix
        // check below while the kernel resolves the path to
        // `/sys/fs/cgroup/escape` (outside walk_root). Either side
        // carrying `..` is a misconfiguration; bail upfront before the
        // canonical-vs-component mismatch becomes a downstream EACCES.
        for (path, label) in [
            (self.parent.as_path(), "parent"),
            (root.as_path(), "walk_root"),
        ] {
            if path
                .components()
                .any(|c| matches!(c, std::path::Component::ParentDir))
            {
                bail!(
                    "CgroupManager::with_walk_root: {label} {path:?} contains `..` components; \
                     parent and walk_root must be normalized absolute paths because \
                     PathBuf::starts_with is component-based and `/a/b/../c` is treated as \
                     starting with `/a/b/..` not the kernel-resolved `/a/c` — the prefix \
                     invariant would be silently violated",
                );
            }
        }
        if !self.parent.starts_with(&root) {
            bail!(
                "CgroupManager::with_walk_root: parent {:?} is not below walk_root {:?}; \
                 the subtree_control walk must originate at a root that contains the parent — \
                 either lower walk_root to a prefix of parent or raise parent to a descendant of \
                 walk_root",
                self.parent,
                root,
            );
        }
        self.walk_root = root;
        Ok(self)
    }

    /// Path to the parent cgroup directory.
    pub fn parent_path(&self) -> &std::path::Path {
        &self.parent
    }

    /// Path to the cgroup-fs root [`Self::setup`] walks down from and
    /// [`Self::drain_tasks`] drains pids to. See [`Self::with_walk_root`].
    pub fn walk_root(&self) -> &std::path::Path {
        &self.walk_root
    }

    /// Count of un-removed cgroups currently tracked by this
    /// manager — incremented when [`Self::remove_cgroup`] fails,
    /// decremented when it succeeds. Exposed for tests and for
    /// callers that want to inspect the budget without forcing a
    /// remove attempt.
    pub fn outstanding_removes(&self) -> usize {
        self.outstanding_removes.load(Ordering::Relaxed)
    }

    /// Create the parent directory and enable the requested cgroup
    /// controllers in every ancestor `cgroup.subtree_control` between
    /// `self.walk_root` (default `/sys/fs/cgroup`) and `self.parent`.
    ///
    /// Pass the controllers the test actually needs — empty set means
    /// "create the parent dir, write nothing to subtree_control". The
    /// scenario runtime computes the controller union from
    /// [`CgroupDef`](crate::scenario::ops::CgroupDef) declarations
    /// (cpuset/cpuset_mems → [`Controller::Cpuset`], cpu →
    /// [`Controller::Cpu`], memory → [`Controller::Memory`], pids →
    /// [`Controller::Pids`], io → [`Controller::Io`]) so a test
    /// that never sets a memory limit never enables `+memory` and
    /// vice versa. `cgroup.freeze` and `cgroup.procs` are
    /// cgroup-core, ungated by any controller, and need no entry.
    ///
    /// # Walk root
    ///
    /// The ancestor walk stops at `self.walk_root` so cgroup-v2 user
    /// delegation (Mode B/C) does not attempt subtree_control writes
    /// above the delegation boundary. [`Self::with_walk_root`]
    /// retargets the walk; the constructor validates that
    /// `self.parent` is below `walk_root`.
    ///
    /// # Availability check
    ///
    /// Each requested controller is verified against
    /// `{walk_root}/cgroup.controllers` before any write. A
    /// requested controller missing from the kernel's available set
    /// surfaces as `controller {ctrl} not available; cgroup.controllers
    /// = {available:?}` rather than the bare ENOENT/EACCES the
    /// downstream `set_*` write would otherwise emit.
    ///
    /// # Error propagation
    ///
    /// All filesystem writes propagate via `?`. A user inspecting
    /// `RUST_BACKTRACE=1` output sees the exact subtree_control path
    /// that failed and the underlying errno, instead of a swallowed
    /// `tracing::warn!` followed by a downstream EACCES at the
    /// controller-knob write site.
    pub fn setup(&self, controllers: &BTreeSet<Controller>) -> Result<()> {
        self.setup_under_root(controllers, &self.walk_root)
    }

    /// Does managing cgroups require root privileges for this
    /// `(root, parent, euid)`? True only when `root` is the kernel-owned
    /// default walk root (`/sys/fs/cgroup`), `parent` is actually under
    /// that root (a real cgroupfs operation — create_dir_all of the
    /// parent, or the subtree_control walk, that EACCESes for a non-root
    /// euid), AND the euid is non-root. A `parent` OUTSIDE the root (e.g.
    /// a tmpdir — the non-cgroup-path early-bail that creates a dir and
    /// skips the walk) touches no cgroupfs and needs no root. A delegated
    /// walk root (set via [`Self::with_walk_root`]) is exempt: cgroup-v2
    /// delegation grants the delegatee write access to
    /// `cgroup.subtree_control` inside the delegated subtree, so a
    /// non-root euid can manage it (Documentation/admin-guide/cgroup-v2.rst,
    /// Delegation). Pure + takes `parent`/`euid` explicitly so the
    /// privilege gate is unit-tested regardless of the test runner's own
    /// euid and working directory.
    fn default_root_requires_root(root: &Path, parent: &Path, euid: u32) -> bool {
        root == Path::new(Self::DEFAULT_WALK_ROOT) && parent.starts_with(root) && euid != 0
    }

    /// Inner setup that takes the cgroup-fs root as an explicit
    /// argument so tests can drive the controller-enable path against
    /// a tmpdir without touching `/sys/fs/cgroup`. Production
    /// [`Self::setup`] threads `self.walk_root` (defaults to
    /// `/sys/fs/cgroup` via [`Self::new`], overridable via
    /// [`Self::with_walk_root`]). The strip-prefix gate stays — if
    /// the parent is outside the supplied root, directory creation
    /// still happens but no subtree_control walk fires (matches the
    /// existing "non-cgroup-mount" early-bail).
    fn setup_under_root(&self, controllers: &BTreeSet<Controller>, root: &Path) -> Result<()> {
        // Managing cgroups under the kernel-owned default walk root
        // (/sys/fs/cgroup, Mode A) requires root: create_dir_all of a
        // parent UNDER /sys/fs/cgroup, or the subtree_control walk below,
        // would EACCES for a non-root caller with an errno that buries
        // the cause. Fail fast here so the message names the fix. Gated
        // on the parent being under the root: a parent OUTSIDE it (the
        // non-cgroup-path early-bail — create a dir, skip the walk)
        // touches no cgroupfs and needs no root. Checked at setup (first
        // real cgroup use), NOT at manager construction: host_only tests
        // that never create a cgroup (macro-attribute fixtures,
        // host-topology reads, nested-VM verifier orchestration) must not
        // fail for a resource they never touch. A delegated walk root
        // (Mode B/C via with_walk_root) is exempt — the operator owns
        // subtree_control inside the delegated subtree.
        let euid = unsafe { libc::geteuid() };
        if Self::default_root_requires_root(root, &self.parent, euid) {
            return Err(anyhow!(
                "CgroupManager::setup: cannot manage cgroups under the \
                 kernel-owned default walk root {root:?} as a non-root \
                 process (euid {euid}); run as root, or for cgroup-v2 \
                 user delegation set a delegated walk root via \
                 CgroupManager::with_walk_root (a systemd Delegate=yes \
                 subtree or a container nsdelegate root) — when driven by \
                 cargo-ktstr, set the {walk_env} env var to that delegated \
                 root",
                walk_env = crate::KTSTR_CGROUP_WALK_ROOT_ENV,
            ));
        }
        // No controllers to enable means no subtree_control walk, and the
        // parent cgroup is only needed when the scenario actually creates
        // child cgroups -- which `create_cgroup`'s `create_dir_all` makes
        // lazily -- or enables controllers. Return BEFORE the eager parent
        // mkdir so a cgroup-free scenario (no CgroupDefs, no workloads --
        // e.g. snapshot-bridge tests, host-topology reads, macro-attribute
        // fixtures) runs without root or a cgroup fs. Previously this mkdir
        // fired unconditionally and EACCES'd a non-root caller (or a
        // deliberately-unwritable dummy parent like `/nonexistent/...`).
        if controllers.is_empty() {
            return Ok(());
        }
        if !self.parent.exists() {
            fs::create_dir_all(&self.parent)
                .with_context(|| format!("mkdir {}", self.parent.display()))?;
        }
        if let Ok(rel) = self.parent.strip_prefix(root) {
            let available_path = root.join("cgroup.controllers");
            if available_path.exists() {
                let raw = fs::read_to_string(&available_path).with_context(|| {
                    format!("read cgroup.controllers: {}", available_path.display())
                })?;
                let available: BTreeSet<&str> = raw.split_whitespace().collect();
                for c in controllers {
                    if !available.contains(c.name()) {
                        return Err(anyhow!(
                            "cgroup controller '{}' not available at {}; \
                             cgroup.controllers reports {:?}. CONFIG_{}_CONTROLLER \
                             may be unset, or the controller is masked at this \
                             level of the hierarchy",
                            c.name(),
                            available_path.display(),
                            available,
                            c.name().to_uppercase(),
                        ));
                    }
                }
            }
            let line: String = controllers
                .iter()
                .map(|c| format!("+{}", c.name()))
                .collect::<Vec<_>>()
                .join(" ");
            let mut cur = root.to_path_buf();
            for c in rel.components() {
                let sc = cur.join("cgroup.subtree_control");
                if sc.exists() {
                    write_with_timeout(&sc, &line, CGROUP_WRITE_TIMEOUT).with_context(|| {
                        format!("enable controllers '{line}' at {}", sc.display())
                    })?;
                }
                cur = cur.join(c);
            }
            let sc = self.parent.join("cgroup.subtree_control");
            if sc.exists() {
                write_with_timeout(&sc, &line, CGROUP_WRITE_TIMEOUT)
                    .with_context(|| format!("enable controllers '{line}' at {}", sc.display()))?;
            }
        }
        Ok(())
    }

    /// Create a child cgroup directory.
    ///
    /// For nested paths (e.g. `"cg_0/narrow"`), enables `+cpuset` on
    /// each intermediate cgroup's `subtree_control` so the leaf has
    /// `cpuset.cpus` / `cpuset.mems` files available. The kernel
    /// requires each parent to have the controller in
    /// `subtree_control` for its children to have the corresponding
    /// files (`cgroup_control()` returns `parent->subtree_control`).
    ///
    /// # Limitation: only `+cpuset` is propagated through nested
    /// intermediates
    ///
    /// `Self::enable_subtree_cpuset` writes ONLY `+cpuset` to each
    /// intermediate's `cgroup.subtree_control`; the `+cpu` /
    /// `+memory` / `+pids` / `+io` controllers enabled by
    /// [`Self::setup`] cover only the manager's parent cgroup, not
    /// arbitrary intermediate cgroups created via nested
    /// `create_cgroup` calls. As a result, a nested leaf like
    /// `"cg_0/narrow"` exposes `cpuset.*` knobs but NOT
    /// `memory.max` / `pids.max` / `io.weight`. If a future
    /// [`CgroupDef`](crate::scenario::ops::CgroupDef) addresses such
    /// a leaf with a memory/pids/io knob, the corresponding
    /// `set_*` write will return ENOENT.
    ///
    /// Today's in-tree consumers (host topology cpuset locks,
    /// `BuildSandbox`, scenario ops) only nest cgroups for cpuset
    /// scoping, so this matches the actual surface the framework
    /// exercises. Extending `Self::enable_subtree_cpuset` to
    /// propagate the remaining controllers across intermediates is
    /// straightforward (write the same controller list as
    /// [`Self::setup`] uses) but is deferred until a use case
    /// concretely needs it; without one, the wider write would
    /// race against concurrent sibling cgroup creation under the
    /// same intermediate without buying anything.
    pub fn create_cgroup(&self, name: &str) -> Result<()> {
        validate_cgroup_name(name)?;
        let p = self.parent.join(name);
        if !p.exists() {
            fs::create_dir_all(&p).with_context(|| format!("mkdir {}", p.display()))?;
        }
        self.enable_subtree_cpuset(name);
        Ok(())
    }

    /// Enable a controller on the parent cgroup's `cgroup.subtree_control`.
    ///
    /// Writes `+{controller}` to `{parent}/cgroup.subtree_control` so
    /// children created under the parent inherit the controller and
    /// expose the corresponding `*.cpus`, `*.mems`, etc. files. No-op
    /// (returns `Ok`) when the subtree_control file does not exist —
    /// callers treat that as "parent is not a cgroup v2 node" and
    /// degrade elsewhere.
    ///
    /// Unlike [`Self::setup`] and `Self::enable_subtree_cpuset`,
    /// which swallow write failures via `tracing::warn!`, this method
    /// propagates the underlying [`std::io::Error`] so callers can
    /// classify errnos (EACCES/EPERM for permission, EBUSY for a
    /// peer holding the subtree) via `anyhow_first_io_errno` and
    /// map them to operator-facing degrade variants. Used by
    /// `crate::vmm::cgroup_sandbox::BuildSandbox::try_create` under
    /// the `--cpu-cap` hard-error contract.
    pub fn add_parent_subtree_controller(&self, controller: &str) -> Result<()> {
        let p = self.parent.join("cgroup.subtree_control");
        if !p.exists() {
            return Ok(());
        }
        write_with_timeout(&p, &format!("+{controller}"), CGROUP_WRITE_TIMEOUT)
    }

    /// Drain tasks from a child cgroup and remove it.
    ///
    /// Auto-unfreezes the cgroup before draining: a frozen cgroup that
    /// reaches teardown (e.g. a step body issues `Op::FreezeCgroup` and
    /// never pairs it with `Op::UnfreezeCgroup`) would migrate its
    /// frozen tasks to the cgroup root via `drain_tasks` and rely on
    /// the kernel's `cgroup_freezer_migrate_task` to clear the JOBCTL
    /// freeze bit when the destination cgroup is unfrozen. The kernel
    /// path is correct, but writing `cgroup.freeze=0` first makes the
    /// teardown deterministic regardless of who froze the cgroup and
    /// when. Tolerates ENOENT on the freeze file (cgroup directory
    /// already gone, or `CONFIG_CGROUP_FREEZE` absent on legacy
    /// kernels) silently — only non-ENOENT failures warn.
    ///
    /// # Post-drain settle window
    ///
    /// Between [`Self::drain_tasks`] and `rmdir`,
    /// `remove_cgroup_inner` calls `wait_for_cgroup_unpopulated` with
    /// a 1s budget. Writes to `cgroup.procs` queue the task move but
    /// the source cgroup's populated state only clears once the
    /// per-task css_set switch completes — `rmdir` returns EBUSY
    /// while the cgroup is still populated. Rather than a blind
    /// sleep, the wait is event-driven: it blocks on an
    /// inotify(IN_MODIFY) watch of the cgroup's `cgroup.events` file
    /// and returns as soon as that file reports `populated 0`, so it
    /// wakes on the actual kernel state-transition write.
    ///
    /// The wait falls through to `rmdir` on deadline (or when
    /// `cgroup.events` is absent / inotify setup fails), so a
    /// genuinely stuck-populated cgroup still surfaces the same
    /// EBUSY error from the subsequent `rmdir`.
    ///
    /// # Outstanding-remove cap
    ///
    /// A churn workload (rapid create→remove cycles) may legitimately
    /// race freeze/drain and see EBUSY/ENOENT on individual remove
    /// calls. Each failed remove increments
    /// [`Self::outstanding_removes`]; once the counter exceeds
    /// `MAX_OUTSTANDING_REMOVES`, the next call returns Err
    /// without attempting any filesystem writes — bounding the peak
    /// resident cgroup leak to that cap regardless of how long the
    /// scenario runs. Successful removes decrement the counter, so a
    /// transient stall that eventually clears (e.g. RCU drain
    /// catches up between iterations) does not strand the manager
    /// in the bailed state.
    ///
    /// A `name` whose directory does not exist returns `Ok(())`
    /// without touching the counter — the cgroup was already
    /// reaped (e.g. by [`Self::cleanup_all`] or a prior remove),
    /// so it is not "outstanding".
    pub fn remove_cgroup(&self, name: &str) -> Result<()> {
        validate_cgroup_name(name)?;
        let outstanding = self.outstanding_removes.load(Ordering::Relaxed);
        if outstanding > MAX_OUTSTANDING_REMOVES {
            bail!(
                "remove_cgroup '{name}' refused: {outstanding} cgroups outstanding \
                 (cap {MAX_OUTSTANDING_REMOVES}); cgroup.procs draining wedged or \
                 churn loop outpacing the kernel's RCU grace period — bailing to \
                 avoid unbounded cgroupfs accumulation"
            );
        }
        let p = self.parent.join(name);
        if !p.exists() {
            return Ok(());
        }
        match self.remove_cgroup_inner(name, &p) {
            Ok(()) => {
                // Successful remove: decrement (saturating at 0 so a
                // remove of a cgroup we never failed-to-remove does
                // not underflow the counter into usize::MAX).
                self.outstanding_removes
                    .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| {
                        Some(n.saturating_sub(1))
                    })
                    .ok();
                Ok(())
            }
            Err(err) => {
                self.outstanding_removes.fetch_add(1, Ordering::Relaxed);
                Err(err)
            }
        }
    }

    /// Inner body of [`Self::remove_cgroup`] — exists so the public
    /// method can wrap the unfreeze/drain/rmdir result in the
    /// outstanding-counter bookkeeping without duplicating the
    /// sequence in success and failure arms.
    ///
    /// Gates the pre-drain unfreeze on `cgroup.freeze` existence to
    /// match [`cleanup_recursive`]'s same-file gate. `set_freeze`
    /// goes through `fs::write` which CREATES the file when it does
    /// not exist (open(O_WRONLY | O_CREAT | O_TRUNC)), so an
    /// unconditional call would plant a stray 1-byte file under any
    /// non-cgroupfs directory and cause the subsequent
    /// `fs::remove_dir(p)` to fail with ENOTEMPTY. On a real cgroup
    /// v2 tree the file is always present (cgroup-core, ungated by
    /// controllers); on a legacy kernel without `CONFIG_CGROUP_FREEZE`
    /// or on a non-cgroup directory entry the file is absent and the
    /// unfreeze step is a no-op.
    fn remove_cgroup_inner(&self, name: &str, p: &Path) -> Result<()> {
        if p.join("cgroup.freeze").exists()
            && let Err(err) = self.set_freeze(name, false)
            && anyhow_first_io_errno(&err) != Some(libc::ENOENT)
        {
            tracing::warn!(
                cgroup = name,
                err = %format!("{err:#}"),
                "remove_cgroup: pre-drain unfreeze failed; drain may strand frozen tasks at root"
            );
        }
        self.drain_tasks(name)?;
        // Wait for the kernel to reflect the empty state via
        // cgroup.events `populated 0` (event-driven via inotify on
        // the events file) before attempting rmdir. The legacy
        // 50 ms blind sleep was a hopeful settle: too short under
        // load (rmdir EBUSY) and too long on a quiet host (wasted
        // tens of ms × every cgroup teardown). Falls through to
        // rmdir on deadline so the caller still sees the same
        // EBUSY error if the cgroup is genuinely stuck-populated;
        // 1 s ceiling matches the prior pessimistic upper bound on
        // a settling cgroup.
        wait_for_cgroup_unpopulated(p, std::time::Duration::from_secs(1));
        fs::remove_dir(p).with_context(|| format!("rmdir {}", p.display()))
    }

    /// Write `cpuset.cpus` for a child cgroup.
    ///
    /// On write failure, captures and emits a snapshot of the
    /// cgroup-tree state at the moment of failure: the parent's
    /// `cgroup.controllers` (controllers AVAILABLE to children),
    /// the parent's `cgroup.subtree_control` (controllers ENABLED
    /// for children), the child's `cgroup.controllers` (the
    /// inheritance ROOT for children of the child), the
    /// `cpuset.cpus` file's existence, and a directory listing of
    /// the child cgroup's knob files. The capture lets a kernel /
    /// hierarchy-state bug surface as a focused diagnostic instead
    /// of a bare `EACCES` at the write site.
    pub fn set_cpuset(&self, name: &str, cpus: &BTreeSet<usize>) -> Result<()> {
        validate_cgroup_name(name)?;
        let p = self.parent.join(name).join("cpuset.cpus");
        match write_with_timeout(&p, &TestTopology::cpuset_string(cpus), CGROUP_WRITE_TIMEOUT) {
            Ok(()) => Ok(()),
            Err(e) => {
                let snapshot = capture_cpuset_state(&self.parent, name);
                Err(e.context(snapshot))
            }
        }
    }

    /// Enable `+cpuset` on `cgroup.subtree_control` for each ancestor
    /// of the leaf in a nested cgroup path. For `"cg_0/narrow"`, writes
    /// `+cpuset` to `{parent}/cgroup.subtree_control` and
    /// `{parent}/cg_0/cgroup.subtree_control`. No-op for
    /// single-component paths.
    fn enable_subtree_cpuset(&self, name: &str) {
        let components: Vec<&str> = name.split('/').collect();
        if components.len() < 2 {
            return;
        }
        let mut cur = self.parent.clone();
        for c in &components[..components.len() - 1] {
            let sc = cur.join("cgroup.subtree_control");
            if sc.exists()
                && let Err(e) = write_with_timeout(&sc, "+cpuset", CGROUP_WRITE_TIMEOUT)
            {
                tracing::warn!(path = %sc.display(), err = %e, "failed to enable cpuset");
            }
            cur = cur.join(c);
        }
        // Write at the last intermediate (direct parent of the leaf).
        let sc = cur.join("cgroup.subtree_control");
        if sc.exists()
            && let Err(e) = write_with_timeout(&sc, "+cpuset", CGROUP_WRITE_TIMEOUT)
        {
            tracing::warn!(path = %sc.display(), err = %e, "failed to enable cpuset");
        }
    }

    /// Clear `cpuset.cpus` for a child cgroup (empty string = inherit parent).
    pub fn clear_cpuset(&self, name: &str) -> Result<()> {
        validate_cgroup_name(name)?;
        let p = self.parent.join(name).join("cpuset.cpus");
        write_with_timeout(&p, "", CGROUP_WRITE_TIMEOUT).with_context(|| {
            format!("cgroup '{name}': clear cpuset.cpus (write empty string for inherit-parent)")
        })
    }

    /// Write `cpuset.mems` for a child cgroup. Constrains which NUMA
    /// nodes the cgroup's tasks can allocate memory on.
    ///
    /// Shape mirrors `set_cpuset` exactly — [`TestTopology::cpuset_string`]
    /// range-compact-formats the node set, `write_with_timeout` bounds
    /// the filesystem-write at 2s. Used by `BuildSandbox` under the
    /// `--cpu-cap` flow to bind build memory to the NUMA nodes hosting
    /// the locked LLCs, avoiding cross-socket DRAM latency for gcc's
    /// symbol tables and linker working sets.
    ///
    /// # Ordering contract
    ///
    /// Caller MUST have already called [`Self::set_cpuset`] (or
    /// equivalent direct write to `cpuset.cpus`) and — when running
    /// under a parent that may narrow the set — MUST have read back
    /// `cpuset.cpus.effective` to detect kernel-side narrowing
    /// BEFORE invoking this method. The per-knob ordering is
    /// load-bearing: `crate::vmm::cgroup_sandbox::BuildSandbox`
    /// interleaves `cpuset.cpus.effective` readback between the
    /// `cpuset.cpus` and `cpuset.mems` writes to abort on narrowing
    /// under the `--cpu-cap` hard-error contract; folding the two
    /// writes into a single helper would erase that gate.
    ///
    /// A cgroup whose `cpuset.cpus` is set should also have a
    /// non-empty `cpuset.mems.effective` before any task is migrated
    /// into it: the half-configured shape (cpus set locally, no
    /// nodemask anywhere up the hierarchy) is suspicious enough that
    /// the framework refuses it. The kernel itself does NOT
    /// SIGKILL on first allocation — `guarantee_online_mems`
    /// (`kernel/cgroup/cpuset.c`) walks UP via `parent_cs(cs)` until
    /// `effective_mems` intersects `node_states[N_MEMORY]`, and the
    /// top cpuset always has online memory, so the walk always finds
    /// a non-empty mask. The actual kernel behavior under a fully
    /// empty hierarchy is path-dependent (parent-walk fallback
    /// generally succeeds; degenerate states without any online
    /// memory may OOM). cgroup v2's `cpuset_can_attach_check` only
    /// rejects empty `effective_cpus`, not empty `effective_mems`.
    /// In cgroup v2, the local `cpuset.mems` file is normally empty
    /// (the cgroup inherits from its parent via `effective_mems`),
    /// so reading the local file alone would falsely flag every
    /// inheriting child. [`Self::move_task`] enforces the gate at
    /// runtime by reading the cgroup's `cpuset.cpus` and
    /// `cpuset.mems.effective` files before each migration and
    /// refusing the write if `cpuset.cpus` is non-empty while
    /// `cpuset.mems.effective` is empty — surfacing a focused
    /// error rather than letting a half-configured cgroup through
    /// to the kernel's path-dependent behavior.
    pub fn set_cpuset_mems(&self, name: &str, nodes: &BTreeSet<usize>) -> Result<()> {
        validate_cgroup_name(name)?;
        let p = self.parent.join(name).join("cpuset.mems");
        let nodes_str = TestTopology::cpuset_string(nodes);
        write_with_timeout(&p, &nodes_str, CGROUP_WRITE_TIMEOUT).with_context(|| {
            format!(
                "cgroup '{name}': set cpuset.mems='{nodes_str}' (requires +cpuset in parent cgroup.subtree_control)"
            )
        })
    }

    /// Clear `cpuset.mems` for a child cgroup (empty string = inherit parent).
    /// Parallels `clear_cpuset`; callers use it only when tearing
    /// down a cpuset-restricted cgroup that needs to accept a
    /// fresh task binding with a different NUMA budget.
    pub fn clear_cpuset_mems(&self, name: &str) -> Result<()> {
        validate_cgroup_name(name)?;
        let p = self.parent.join(name).join("cpuset.mems");
        write_with_timeout(&p, "", CGROUP_WRITE_TIMEOUT).with_context(|| {
            format!("cgroup '{name}': clear cpuset.mems (write empty string for inherit-parent)")
        })
    }

    /// Write `cpu.max` for a child cgroup. `quota_us = None` writes
    /// `"max <period_us>"` (no upper bound — same as a freshly
    /// created cgroup); `Some(q)` writes `"<q> <period_us>"`.
    ///
    /// Per the kernel's cgroup v2 docs ("Documentation/admin-guide/
    /// cgroup-v2.rst", "CPU Interface Files"): each period the
    /// cgroup gets `quota` microseconds of CPU time across its
    /// CPUs, and is throttled until the next period boundary once
    /// the quota is exhausted. `quota` MAY exceed `period` to let
    /// the cgroup use multiple CPUs concurrently (e.g. quota
    /// 200_000 / period 100_000 = up to 2 CPUs of throughput).
    ///
    /// Requires `+cpu` in the parent's `cgroup.subtree_control`;
    /// missing controller surfaces as ENOENT on the file (handled
    /// generically by `write_with_timeout`'s error path with the
    /// errno suffix).
    pub fn set_cpu_max(&self, name: &str, quota_us: Option<u64>, period_us: u64) -> Result<()> {
        validate_cgroup_name(name)?;
        let p = self.parent.join(name).join("cpu.max");
        let line = match quota_us {
            Some(q) => format!("{q} {period_us}"),
            None => format!("max {period_us}"),
        };
        write_with_timeout(&p, &line, CGROUP_WRITE_TIMEOUT).with_context(|| {
            format!(
                "cgroup '{name}': set cpu.max='{line}' (requires +cpu in parent cgroup.subtree_control)"
            )
        })
    }

    /// Write `cpu.weight` for a child cgroup (cgroup v2 weight,
    /// range 1..=10000, default 100). Used together with sibling
    /// cgroups to bias relative CPU share inside the parent's
    /// quota. Independent from `cpu.max` — weights govern share
    /// when CPU is contended, max enforces an absolute ceiling.
    ///
    /// Per "Documentation/admin-guide/cgroup-v2.rst" the legacy
    /// "shares" knob is `cpu.weight.nice` (mapped from nice value);
    /// this method targets the canonical `cpu.weight` knob.
    pub fn set_cpu_weight(&self, name: &str, weight: u32) -> Result<()> {
        validate_cgroup_name(name)?;
        let p = self.parent.join(name).join("cpu.weight");
        write_with_timeout(&p, &weight.to_string(), CGROUP_WRITE_TIMEOUT).with_context(|| {
            format!(
                "cgroup '{name}': set cpu.weight={weight} (requires +cpu in parent cgroup.subtree_control)"
            )
        })
    }

    /// Write `memory.max` for a child cgroup. `bytes = None` writes
    /// `"max"` (no hard limit). When the cgroup's RSS exceeds the
    /// limit, the kernel OOM-kills tasks per the documented
    /// `memory.max` semantics. Requires `+memory` in the parent's
    /// `cgroup.subtree_control`.
    pub fn set_memory_max(&self, name: &str, bytes: Option<u64>) -> Result<()> {
        validate_cgroup_name(name)?;
        let p = self.parent.join(name).join("memory.max");
        let line = match bytes {
            Some(b) => b.to_string(),
            None => "max".to_string(),
        };
        write_with_timeout(&p, &line, CGROUP_WRITE_TIMEOUT).with_context(|| {
            format!(
                "cgroup '{name}': set memory.max='{line}' (requires +memory in parent cgroup.subtree_control)"
            )
        })
    }

    /// Write `memory.high` for a child cgroup. `bytes = None`
    /// writes `"max"` (no high-water mark). Crossing the high
    /// threshold triggers reclaim throttling but NOT OOM-kill,
    /// distinguishing it from `memory.max`.
    pub fn set_memory_high(&self, name: &str, bytes: Option<u64>) -> Result<()> {
        validate_cgroup_name(name)?;
        let p = self.parent.join(name).join("memory.high");
        let line = match bytes {
            Some(b) => b.to_string(),
            None => "max".to_string(),
        };
        write_with_timeout(&p, &line, CGROUP_WRITE_TIMEOUT).with_context(|| {
            format!(
                "cgroup '{name}': set memory.high='{line}' (requires +memory in parent cgroup.subtree_control)"
            )
        })
    }

    /// Write `memory.low` for a child cgroup. `bytes = None` writes
    /// `"0"` (no low-water protection). The kernel preferentially
    /// reclaims FROM other cgroups before reclaiming this cgroup's
    /// memory below `memory.low`; not a hard reservation.
    pub fn set_memory_low(&self, name: &str, bytes: Option<u64>) -> Result<()> {
        validate_cgroup_name(name)?;
        let p = self.parent.join(name).join("memory.low");
        let line = match bytes {
            Some(b) => b.to_string(),
            None => "0".to_string(),
        };
        write_with_timeout(&p, &line, CGROUP_WRITE_TIMEOUT).with_context(|| {
            format!(
                "cgroup '{name}': set memory.low='{line}' (requires +memory in parent cgroup.subtree_control)"
            )
        })
    }

    /// Write `io.weight` for a child cgroup (cgroup v2 weight,
    /// range 1..=10000, default 100). Biases relative IO share
    /// across sibling cgroups when the io controller is enabled
    /// in the parent's `cgroup.subtree_control`. The kernel's BFQ
    /// or io.cost backend (whichever is active) applies the
    /// weight when contending devices are saturated.
    ///
    /// `io.max` (per-device throughput cap) is intentionally NOT
    /// surfaced here — the per-device interface needs major:minor
    /// device-id lookup which has no in-tree consumer; surface it
    /// when a concrete use case lands.
    pub fn set_io_weight(&self, name: &str, weight: u16) -> Result<()> {
        validate_cgroup_name(name)?;
        let p = self.parent.join(name).join("io.weight");
        write_with_timeout(&p, &weight.to_string(), CGROUP_WRITE_TIMEOUT).with_context(|| {
            format!(
                "cgroup '{name}': set io.weight={weight} (requires +io in parent cgroup.subtree_control)"
            )
        })
    }

    /// Write `cgroup.freeze` for a child cgroup. `frozen = true` writes
    /// `"1"`, `frozen = false` writes `"0"`.
    ///
    /// `cgroup.freeze` is a cgroup-core file exposed on every non-root
    /// cgroup automatically — it is NOT gated by `cgroup.subtree_control`.
    /// The kernel's `cgroup_freeze_write` parses the value via
    /// `kstrtoint`, rejects anything outside `{0, 1}` with `-ERANGE`,
    /// and dispatches `cgroup_freeze(cgrp, freeze)`. Writing `1` to a
    /// cgroup containing tasks transitions every task in the subtree to
    /// the frozen state; writing `0` releases. The transition is
    /// asynchronous — `cgroup.events`'s `frozen` field reaches `1` once
    /// every task has parked.
    pub fn set_freeze(&self, name: &str, frozen: bool) -> Result<()> {
        validate_cgroup_name(name)?;
        let p = self.parent.join(name).join("cgroup.freeze");
        let line = if frozen { "1" } else { "0" };
        write_with_timeout(&p, line, CGROUP_WRITE_TIMEOUT).with_context(|| {
            format!("cgroup '{name}': set cgroup.freeze='{line}' (cgroup-core file, no controller required)")
        })
    }

    /// Write `pids.max` for a child cgroup. `max = None` writes `"max"`
    /// (the kernel's `PIDS_MAX_STR` sentinel for unlimited);
    /// `Some(n)` writes the decimal `n`.
    ///
    /// Per the kernel's `pids_max_write`: the parser short-circuits to
    /// the unlimited limit when `buf == PIDS_MAX_STR`; otherwise
    /// `kstrtoll(buf, 0, &limit)` parses a signed integer and rejects
    /// `< 0` or `>= PIDS_MAX` with `-EINVAL`. The update is atomic
    /// (`atomic64_set(&pids->limit, limit)`); existing tasks are NOT
    /// killed when the limit lands below the current task count — only
    /// future `fork()` / `clone()` calls are blocked.
    ///
    /// Requires `+pids` in the parent's `cgroup.subtree_control`;
    /// [`Self::setup`] enables it unconditionally so this write
    /// succeeds on every ktstr-managed cgroup tree.
    pub fn set_pids_max(&self, name: &str, max: Option<u64>) -> Result<()> {
        validate_cgroup_name(name)?;
        let p = self.parent.join(name).join("pids.max");
        let line = match max {
            Some(n) => n.to_string(),
            None => "max".to_string(),
        };
        write_with_timeout(&p, &line, CGROUP_WRITE_TIMEOUT).with_context(|| {
            format!(
                "cgroup '{name}': set pids.max='{line}' (requires +pids in parent cgroup.subtree_control)"
            )
        })
    }

    /// Write `memory.swap.max` for a child cgroup. `bytes = None` writes
    /// `"max"` (no swap cap); `Some(b)` writes the decimal byte count.
    ///
    /// Per the kernel's `swap_max_write`: the value is parsed via
    /// `page_counter_memparse(buf, "max", &max)`, which accepts the
    /// literal `"max"` token for unlimited or a numeric byte count.
    /// The store is `xchg(&memcg->swap.max, max)` — atomic, with no
    /// failure path beyond the parse.
    ///
    /// Requires `+memory` in the parent's `cgroup.subtree_control`;
    /// [`Self::setup`] enables it unconditionally.
    ///
    /// Requires CONFIG_SWAP=y in the test kernel. The file does not
    /// exist on swapless builds; the write returns ENOENT.
    pub fn set_memory_swap_max(&self, name: &str, bytes: Option<u64>) -> Result<()> {
        validate_cgroup_name(name)?;
        let p = self.parent.join(name).join("memory.swap.max");
        let line = match bytes {
            Some(b) => b.to_string(),
            None => "max".to_string(),
        };
        write_with_timeout(&p, &line, CGROUP_WRITE_TIMEOUT).with_context(|| {
            format!(
                "cgroup '{name}': set memory.swap.max='{line}' (requires +memory in parent cgroup.subtree_control; file absent on CONFIG_SWAP=n kernels)"
            )
        })
    }

    /// Move a single task into a child cgroup via `cgroup.procs`.
    ///
    /// `move_task` is host-side scenario orchestration, never
    /// invoked from a vCPU thread, so the bare `fs::read_to_string`
    /// reads in `Self::check_cpuset_ordering` are not bounded by
    /// the freeze-rendezvous timeout. A wedged cgroupfs read here
    /// would stall the orchestrator thread, not a vCPU.
    ///
    /// # cpuset ordering gate
    ///
    /// Before issuing the `cgroup.procs` write, the method reads the
    /// destination's `cpuset.cpus` (the local-write knob the caller
    /// either set or did not) and `cpuset.mems.effective` (the
    /// kernel's effective view, inheritance-aware). The gate
    /// refuses migrations into a cgroup whose `cpuset.cpus` is set
    /// but `cpuset.mems.effective` reads empty — a half-configured
    /// state we surface as a focused error rather than letting it
    /// through to the kernel.
    ///
    /// The kernel's behavior on the half-configured shape is
    /// path-dependent: `guarantee_online_mems`
    /// (`kernel/cgroup/cpuset.c`) walks UP via `parent_cs(cs)`
    /// until `effective_mems` intersects `node_states[N_MEMORY]`,
    /// and the top cpuset always has online memory, so the walk
    /// generally succeeds; the empty-nodemask OOM path is reachable
    /// only in degenerate hierarchies. cgroup v2's
    /// `cpuset_can_attach_check` rejects only empty `effective_cpus`
    /// (not empty `effective_mems`), so a v2 attach into a cgroup
    /// with empty `effective_mems` is not a hard kernel error
    /// either. The framework refuses the migration anyway because
    /// the half-configured shape almost always reflects a missing
    /// [`Self::set_cpuset_mems`] call; surfacing it directly is
    /// more debuggable than letting it become whatever the kernel
    /// happens to do on this particular hierarchy.
    ///
    /// # Why `cpuset.mems.effective`, not `cpuset.mems`
    ///
    /// In cgroup v2, the local `cpuset.mems` file echoes
    /// `cs->mems_allowed` — the LOCAL nodemask, which is empty by
    /// default until the caller explicitly writes it. The kernel's
    /// allocation path uses `cs->effective_mems` instead, which
    /// inherits from the parent when the local mask is empty (per
    /// `cpuset_common_seq_show`'s FILE_EFFECTIVE_MEMLIST branch and
    /// `guarantee_online_mems`'s `parent_cs(cs)` walk). A gate that
    /// reads the local file would falsely flag every inheriting
    /// child as half-configured even though the kernel sees a
    /// perfectly valid `effective_mems` from the parent. The
    /// effective view captures both "this cgroup wrote `cpuset.mems`
    /// directly" and "this cgroup inherits a non-empty mask from
    /// its parent" without false positives.
    ///
    /// Both reads are best-effort — a cgroup without cpuset
    /// controllers (`cpuset.cpus` does not exist) bypasses the
    /// gate, matching the kernel's "no cpuset constraints to
    /// enforce" path. Read errors on either knob are absorbed: the
    /// gate exists to catch the configured-but-half-configured
    /// shape, not to fight cgroupfs read failures. If
    /// `cpuset.mems.effective` cannot be read for any reason, the
    /// gate degrades to "accept" — it cannot make a sound decision
    /// without the kernel's effective view.
    pub fn move_task(&self, name: &str, pid: libc::pid_t) -> Result<()> {
        validate_cgroup_name(name)?;
        self.check_cpuset_ordering(name)?;
        let p = self.parent.join(name).join("cgroup.procs");
        write_with_timeout(&p, &pid.to_string(), CGROUP_WRITE_TIMEOUT)
    }

    /// Verify that a cgroup's `cpuset.cpus` /
    /// `cpuset.mems.effective` are in a consistent state before
    /// admitting a task migration into it.
    ///
    /// Returns `Err` only when the destination has `cpuset.cpus`
    /// non-empty AND `cpuset.mems.effective` reads empty — a
    /// half-configured shape we surface as a focused error rather
    /// than letting through. The kernel's behavior in this state is
    /// path-dependent: `guarantee_online_mems` (`kernel/cgroup/
    /// cpuset.c`) walks UP via `parent_cs(cs)` until effective_mems
    /// intersects `node_states[N_MEMORY]` and the top cpuset always
    /// has online memory, so the parent-walk fallback usually
    /// succeeds; degenerate hierarchies may OOM. cgroup v2's
    /// `cpuset_can_attach_check` rejects only empty `effective_cpus`,
    /// not empty `effective_mems`. All other shapes (no cpuset
    /// controller, local cpus empty, effective mems non-empty
    /// whether locally written or parent-inherited) are accepted.
    ///
    /// Read failures on either knob are absorbed (the gate degrades
    /// to "accept" rather than blocking on any cgroupfs read
    /// error). The effective-view file is the source of truth
    /// because in cgroup v2 the local `cpuset.mems` is normally
    /// empty (the cgroup inherits from its parent via
    /// `effective_mems`); reading the local file would emit false
    /// positives for every child that inherits a parent's NUMA
    /// budget without writing its own.
    fn check_cpuset_ordering(&self, name: &str) -> Result<()> {
        let cpus_path = self.parent.join(name).join("cpuset.cpus");
        let mems_effective_path = self.parent.join(name).join("cpuset.mems.effective");
        let cpus = match fs::read_to_string(&cpus_path) {
            Ok(s) => s,
            Err(_) => return Ok(()),
        };
        // `cpuset.cpus` is empty when the cgroup inherits from its
        // parent — no constraint imposed locally, so the
        // `cpuset.mems` invariant doesn't apply.
        if cpus.trim().is_empty() {
            return Ok(());
        }
        let mems_effective = match fs::read_to_string(&mems_effective_path) {
            Ok(s) => s,
            Err(_) => return Ok(()),
        };
        if mems_effective.trim().is_empty() {
            bail!(
                "move_task into '{name}' refused: cpuset.cpus is set ({}) \
                 but cpuset.mems.effective reads empty — half-configured \
                 cgroup. The kernel's behavior here is path-dependent \
                 (guarantee_online_mems walks up to find a non-empty \
                 ancestor mask; the empty-nodemask OOM path is reachable \
                 only in degenerate hierarchies), but the framework \
                 surfaces a focused error rather than letting the \
                 migration through. Call set_cpuset_mems on this cgroup \
                 or widen an ancestor's cpuset.mems before move_task",
                cpus.trim(),
            );
        }
        Ok(())
    }

    /// Write `child_pid` to `<cgroup_name>/cgroup.procs` during the
    /// payload-spawn cgroup-sync handshake.
    ///
    /// Distinct from [`Self::move_task`]: this is the
    /// placement-before-exec write that runs while the child is
    /// paused in pre_exec between `fork(2)` and `execve(2)`. The
    /// `move_task` cpuset-ordering gate does NOT apply here —
    /// placement runs before cpuset is finalised at scenario setup
    /// time, and the gate would reject otherwise-valid spawn
    /// requests. Callers that need the gate (post-spawn migration)
    /// invoke [`Self::move_task`] / [`Self::move_tasks`] instead.
    ///
    /// Uses the same `write_with_timeout` shape as the other
    /// `cgroup.procs` write sites so a wedged cgroupfs is bounded
    /// to `CGROUP_WRITE_TIMEOUT` rather than blocking the parent
    /// indefinitely.
    pub fn place_task_during_handshake(
        &self,
        cgroup_name: &str,
        child_pid: libc::pid_t,
    ) -> Result<()> {
        validate_cgroup_name(cgroup_name)?;
        let cgroup_procs_path = self.parent.join(cgroup_name).join("cgroup.procs");
        let line = format!("{child_pid}\n");
        write_with_timeout(&cgroup_procs_path, &line, CGROUP_WRITE_TIMEOUT).with_context(|| {
            format!(
                "place pid {child_pid} into cgroup '{cgroup_name}' via {} during cgroup-sync handshake",
                cgroup_procs_path.display(),
            )
        })
    }

    /// Move multiple tasks into a child cgroup by PID.
    ///
    /// Tolerates per-pid ESRCH (a task that exited between the listing
    /// snapshot and the migration write) and logs a warn for each
    /// vanished pid — partial migration is a legitimate outcome when
    /// one of N workers has voluntarily exited. Retries EBUSY up to
    /// 3 times with 100ms backoff for transient rejections from
    /// sched_ext BPF `cgroup_prep_move` callbacks
    /// (`scx_cgroup_can_attach`). Propagates EBUSY after retries
    /// exhausted. Propagates all other errors immediately.
    ///
    /// # All-vanished bail
    ///
    /// When `pids` is non-empty AND every supplied pid ESRCH'd, this
    /// fn bails with an actionable diagnostic rather than silently
    /// returning Ok. The silent-Ok path violates the project's
    /// no-silent-drops rule (any data loss must fail loudly):
    /// a downstream consumer reading the destination
    /// `cgroup.procs` would see 0 pids and have no idea whether
    /// the migration was supposed to move 0 or N — masking a real
    /// test-setup regression (e.g. `WorkloadHandle::spawn` child
    /// pre_exec init-panic cascade that killed every paused worker
    /// before move_tasks ran) behind a downstream-state empty-read.
    ///
    /// A test that LEGITIMATELY moves only already-exited workers
    /// (post-Drop diagnostic, post-mortem capture) should pass an
    /// empty `pids` slice rather than calling with non-empty + all
    /// pre-vanished — the empty-slice path is the documented "no
    /// move requested" form that returns Ok cleanly.
    pub fn move_tasks(&self, name: &str, pids: &[libc::pid_t]) -> Result<()> {
        validate_cgroup_name(name)?;
        move_tasks_inner(name, pids, |n, pid| self.move_task_with_retry(n, pid))
    }

    /// Move a single task with bounded EBUSY retry.
    fn move_task_with_retry(&self, name: &str, pid: libc::pid_t) -> Result<()> {
        const MAX_RETRIES: u32 = 3;
        const RETRY_DELAY: Duration = Duration::from_millis(100);

        for attempt in 0..MAX_RETRIES {
            match self.move_task(name, pid) {
                Ok(()) => return Ok(()),
                Err(e) if is_ebusy(&e) && attempt + 1 < MAX_RETRIES => {
                    tracing::debug!(
                        pid,
                        cgroup = name,
                        attempt = attempt + 1,
                        "EBUSY on cgroup.procs write, retrying"
                    );
                    std::thread::sleep(RETRY_DELAY);
                }
                Err(e) => return Err(e),
            }
        }
        unreachable!()
    }

    /// Clear `subtree_control` on a child cgroup by writing an empty
    /// string. Disables all controllers for the cgroup's children.
    ///
    /// Required before moving tasks into a cgroup that has
    /// `subtree_control` set: the kernel's no-internal-process
    /// constraint (`cgroup_migrate_vet_dst`) returns EBUSY when
    /// tasks are written to `cgroup.procs` of a cgroup with
    /// controllers in `subtree_control`.
    pub fn clear_subtree_control(&self, name: &str) -> Result<()> {
        validate_cgroup_name(name)?;
        let p = self.parent.join(name).join("cgroup.subtree_control");
        if !p.exists() {
            return Ok(());
        }
        // Read current controllers and disable each one.
        let content = fs::read_to_string(&p).with_context(|| format!("read {}", p.display()))?;
        let content = content.trim();
        if content.is_empty() {
            return Ok(());
        }
        // Each controller name needs a "-" prefix to disable.
        let disable: Vec<String> = content
            .split_whitespace()
            .map(|c| format!("-{c}"))
            .collect();
        let disable_str = disable.join(" ");
        write_with_timeout(&p, &disable_str, CGROUP_WRITE_TIMEOUT)
            .with_context(|| format!("clear subtree_control on {name}"))
    }

    /// Move all tasks from a child cgroup to the walk-root cgroup.
    ///
    /// Drains to `{self.walk_root}/cgroup.procs` instead of the
    /// parent because the parent has `subtree_control` set (enabling
    /// cpuset for children), and the kernel's no-internal-process
    /// constraint rejects writes to `cgroup.procs` when
    /// `subtree_control` is active. The walk-root cgroup is the
    /// uppermost cgroup the operator can write to without crossing
    /// the delegation boundary; under Mode A it is the canonical
    /// `/sys/fs/cgroup` root (exempt from the no-internal-process
    /// constraint), under Mode B/C it is the delegated subtree root
    /// (which also has procs-writability inside the delegation).
    pub fn drain_tasks(&self, name: &str) -> Result<()> {
        validate_cgroup_name(name)?;
        let src = self.parent.join(name).join("cgroup.procs");
        if !src.exists() {
            return Ok(());
        }
        let dst = self.walk_root.join("cgroup.procs");
        drain_pids_to_root(&src, &dst, name);
        Ok(())
    }

    /// Read `cgroup.procs` of `name`, returning the thread-group
    /// leaders (PIDs) currently in the cgroup.
    ///
    /// Distinct from [`Self::drain_tasks`]:
    /// - `drain_tasks` MIGRATES tasks to the walk-root and treats a
    ///   missing `cgroup.procs` file as a no-op (`Ok(())`) so
    ///   best-effort teardown of an already-rmdir'd cgroup is safe.
    /// - `read_procs` is a READ accessor for assertions
    ///   ([`Op::CaptureCgroupProcs`](crate::scenario::ops::Op::CaptureCgroupProcs)
    ///   and direct callers). A missing `cgroup.procs` file is a
    ///   real error (cgroup doesn't exist, typo'd name, race with
    ///   teardown) — propagating it lets the caller distinguish
    ///   "empty cgroup" from "no such cgroup."
    ///
    /// # Semantics
    ///
    /// - Returns thread-group leaders (PIDs / TGIDs) as the kernel
    ///   exposes them via `cgroup_procs_show` in `kernel/cgroup/cgroup.c`.
    ///   For per-thread TIDs the kernel exposes `cgroup.threads`; this
    ///   method reads ONLY `cgroup.procs`.
    /// - Non-atomic snapshot as exposed by the kernel's pidlist
    ///   iteration (`cgroup_procs_show` / `css_task_iter_next` in
    ///   `kernel/cgroup/cgroup.c`): the kernel walks the css_set's
    ///   task list one entry at a time, so a task that joins or exits
    ///   mid-read can appear in the next read but not this one (or
    ///   vice versa). The userspace `fs::read_to_string` here returns
    ///   when seq_file signals EOF; the per-pid atomicity is a kernel
    ///   property, not an impl one. Callers asserting on membership
    ///   of a stable task set (e.g. SpinWait workers spawned in the
    ///   prior op) are unaffected.
    /// - Empty cgroup: returns `Ok(Vec::new())` (kernel emits an
    ///   empty file, not an error). Lets callers distinguish "no
    ///   tasks" from "no such cgroup."
    /// - Malformed pid lines: skipped with a `tracing::warn!`
    ///   naming the offending line, matching
    ///   `drain_pids_to_root`'s tolerance. The kernel never emits
    ///   such lines today; the tolerance exists so a future kernel
    ///   gaining a header or comment line surfaces as a warn
    ///   instead of an opaque parse error.
    pub fn read_procs(&self, name: &str) -> Result<Vec<libc::pid_t>> {
        validate_cgroup_name(name)?;
        let procs_path = self.parent.join(name).join("cgroup.procs");
        let content = fs::read_to_string(&procs_path).with_context(|| {
            format!(
                "read cgroup.procs from '{}' (cgroup name '{name}'); the cgroup may not \
                 exist or may have been removed (check that `Op::AddCgroup(name)` or a \
                 `CgroupDef` covers this name, and that the test's `workload_root_cgroup` \
                 is correct)",
                procs_path.display(),
            )
        })?;
        let mut pids = Vec::new();
        for line in content.lines() {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            match trimmed.parse::<libc::pid_t>() {
                Ok(pid) => pids.push(pid),
                Err(e) => {
                    tracing::warn!(
                        path = %procs_path.display(),
                        cgroup = name,
                        line = trimmed,
                        err = %e,
                        "read_procs: malformed pid line; skipping",
                    );
                }
            }
        }
        Ok(pids)
    }

    /// Remove all child cgroups under the parent (keeps the parent itself).
    ///
    /// Returns `Ok` even when individual filesystem probes fail; callers
    /// treat cleanup as best-effort teardown (see the runner's warn-
    /// and-continue in `src/runner.rs`). Per-entry `read_dir` /
    /// `DirEntry` / `file_type` errors are surfaced via
    /// `tracing::warn!` — mirrors `CgroupGroup::drop` so a failure
    /// shows up in logs instead of silently leaving children behind.
    ///
    /// # Outer-read_dir failure semantic
    ///
    /// When `read_dir(self.parent)` itself fails — e.g. the parent
    /// directory is unreadable, the cgroup mount has been unmounted
    /// out from under us, or a stat-side IO error fires — the
    /// failure is surfaced via `tracing::warn!` and the function
    /// still returns `Ok(())`. The deliberate semantic here is
    /// "teardown that observes a hostile filesystem state must
    /// not block scenario completion": a hard `Err` would propagate
    /// up through the runner's teardown and abort the whole test
    /// run on a transient cgroupfs failure that the operator can
    /// follow up on by reading the warn line.
    ///
    /// Production callers (the runner's drop path, scenario teardown)
    /// already log-and-continue on `cleanup_all` errors, so the
    /// always-Ok return is consistent with how every consumer
    /// already treats the result. Operators who need to detect
    /// teardown leakage should grep `tracing` output for
    /// `"cleanup_all: read_dir failed"` rather than relying on a
    /// non-zero exit; the warn includes both the offending path and
    /// the underlying io::Error.
    pub fn cleanup_all(&self) -> Result<()> {
        if !self.parent.exists() {
            return Ok(());
        }
        let walk_root = self.walk_root.clone();
        if let Err(err) = for_each_child_dir(&self.parent, "cleanup_all", |p| {
            cleanup_recursive(p, &walk_root)
        }) {
            tracing::warn!(
                parent = %self.parent.display(),
                err = %err,
                "cleanup_all: read_dir failed; child cgroups may remain under parent",
            );
        }
        Ok(())
    }
}

/// Abstraction over the cgroup v2 filesystem surface used by the
/// scenario runtime. The production implementation is [`CgroupManager`],
/// which translates each method into real writes under `/sys/fs/cgroup`.
///
/// Extracted so `scenario::ops::apply_setup` and related orchestration
/// code can be unit-tested against an in-memory double: tests construct
/// a recording or failure-injecting implementor, drive `apply_setup`
/// against it, and assert on the recorded call sequence without
/// touching the host cgroup hierarchy.
///
/// Object-safe by design — scenario code holds the trait object behind
/// `&dyn CgroupOps` rather than being generic. Callers keep writing
/// `ctx.cgroups.set_cpuset(...)` with no syntactic change; dynamic
/// dispatch resolves to `CgroupManager` in production and to the
/// test double under `#[cfg(test)]`. The per-call indirect-call cost
/// is dominated by the filesystem I/O the trait abstracts over.
pub trait CgroupOps {
    /// Path to the parent cgroup directory. See
    /// [`CgroupManager::parent_path`].
    fn parent_path(&self) -> &Path;
    /// Create the parent directory and enable controllers. See
    /// [`CgroupManager::setup`].
    fn setup(&self, controllers: &BTreeSet<Controller>) -> Result<()>;
    /// Create a child cgroup. See [`CgroupManager::create_cgroup`].
    fn create_cgroup(&self, name: &str) -> Result<()>;
    /// Drain and remove a child cgroup. See
    /// [`CgroupManager::remove_cgroup`].
    fn remove_cgroup(&self, name: &str) -> Result<()>;
    /// Write `cpuset.cpus`. See [`CgroupManager::set_cpuset`].
    fn set_cpuset(&self, name: &str, cpus: &BTreeSet<usize>) -> Result<()>;
    /// Clear `cpuset.cpus` (inherit from parent). See
    /// [`CgroupManager::clear_cpuset`].
    fn clear_cpuset(&self, name: &str) -> Result<()>;
    /// Write `cpuset.mems`. See [`CgroupManager::set_cpuset_mems`].
    fn set_cpuset_mems(&self, name: &str, nodes: &BTreeSet<usize>) -> Result<()>;
    /// Clear `cpuset.mems` (inherit from parent). See
    /// [`CgroupManager::clear_cpuset_mems`].
    fn clear_cpuset_mems(&self, name: &str) -> Result<()>;
    /// Write `cpu.max`. See [`CgroupManager::set_cpu_max`].
    fn set_cpu_max(&self, name: &str, quota_us: Option<u64>, period_us: u64) -> Result<()>;
    /// Write `cpu.weight`. See [`CgroupManager::set_cpu_weight`].
    fn set_cpu_weight(&self, name: &str, weight: u32) -> Result<()>;
    /// Write `memory.max`. See [`CgroupManager::set_memory_max`].
    fn set_memory_max(&self, name: &str, bytes: Option<u64>) -> Result<()>;
    /// Write `memory.high`. See [`CgroupManager::set_memory_high`].
    fn set_memory_high(&self, name: &str, bytes: Option<u64>) -> Result<()>;
    /// Write `memory.low`. See [`CgroupManager::set_memory_low`].
    fn set_memory_low(&self, name: &str, bytes: Option<u64>) -> Result<()>;
    /// Write `io.weight`. See [`CgroupManager::set_io_weight`].
    fn set_io_weight(&self, name: &str, weight: u16) -> Result<()>;
    /// Write `cgroup.freeze`. See [`CgroupManager::set_freeze`].
    fn set_freeze(&self, name: &str, frozen: bool) -> Result<()>;
    /// Write `pids.max`. See [`CgroupManager::set_pids_max`].
    fn set_pids_max(&self, name: &str, max: Option<u64>) -> Result<()>;
    /// Write `memory.swap.max`. See
    /// [`CgroupManager::set_memory_swap_max`].
    fn set_memory_swap_max(&self, name: &str, bytes: Option<u64>) -> Result<()>;
    /// Move a single task via `cgroup.procs`. See
    /// [`CgroupManager::move_task`].
    fn move_task(&self, name: &str, pid: libc::pid_t) -> Result<()>;
    /// Move multiple tasks (tolerates ESRCH, retries EBUSY). See
    /// [`CgroupManager::move_tasks`].
    fn move_tasks(&self, name: &str, pids: &[libc::pid_t]) -> Result<()>;
    /// Place a single task into a child cgroup's `cgroup.procs`
    /// during the payload-spawn cgroup-sync handshake.
    ///
    /// Distinct from [`Self::move_task`] / [`Self::move_tasks`]:
    /// those run post-spawn for synthetic workers whose pids are
    /// already in their final cgroup-permissive state. This method
    /// runs INSIDE the two-pipe handshake between the child's
    /// pre_exec pid-notify and the parent's release-signal write,
    /// when the child is paused between `fork(2)` and `execve(2)`.
    /// The write MUST land BEFORE the release byte so the child's
    /// `execve` lands in the destination cgroup — this is the
    /// placement-before-exec invariant required to keep tasks like
    /// `Op::RunPayload { cgroup: Some(name), ... }` from briefly
    /// inheriting the parent's cgroup at exec time.
    ///
    /// # Caller contract
    ///
    /// - MUST be invoked exactly once during the handshake between
    ///   pid-notify and release-signal.
    /// - Failure MUST propagate to the caller, which is responsible
    ///   for dropping the release pipe to unblock the child with
    ///   EOF so it bails out of pre_exec rather than execve'ing
    ///   into an unspecified cgroup.
    /// - The `cgroup_name` argument is the user-facing name the
    ///   test author passed in `Op::RunPayload { cgroup: Some(name),
    ///   ... }` or `PayloadRun::in_cgroup(name)` — NOT a derived
    ///   absolute path. The implementation derives the
    ///   `cgroup.procs` path from this name plus its own
    ///   parent-path knowledge.
    ///
    /// See [`CgroupManager::place_task_during_handshake`].
    fn place_task_during_handshake(&self, cgroup_name: &str, child_pid: libc::pid_t) -> Result<()>;
    /// Clear `cgroup.subtree_control` on a child. See
    /// [`CgroupManager::clear_subtree_control`].
    fn clear_subtree_control(&self, name: &str) -> Result<()>;
    /// Drain tasks from a child to the cgroup root. See
    /// [`CgroupManager::drain_tasks`].
    fn drain_tasks(&self, name: &str) -> Result<()>;
    /// Read `cgroup.procs` of a child, returning thread-group leaders.
    /// See [`CgroupManager::read_procs`].
    fn read_procs(&self, name: &str) -> Result<Vec<libc::pid_t>>;
    /// Remove all child cgroups under the parent. See
    /// [`CgroupManager::cleanup_all`].
    fn cleanup_all(&self) -> Result<()>;
}

// Thin forwarding trait impl: inherent `CgroupManager` methods hold the
// real bodies; this trait impl exists so scenario code can hold
// `&dyn CgroupOps` for test-double injection without threading a generic
// through every caller. Trait default methods cannot access the private
// fields, and macro-generated delegation would lose Go-To-Definition.
impl CgroupOps for CgroupManager {
    fn parent_path(&self) -> &Path {
        CgroupManager::parent_path(self)
    }
    fn setup(&self, controllers: &BTreeSet<Controller>) -> Result<()> {
        CgroupManager::setup(self, controllers)
    }
    fn create_cgroup(&self, name: &str) -> Result<()> {
        CgroupManager::create_cgroup(self, name)
    }
    fn remove_cgroup(&self, name: &str) -> Result<()> {
        CgroupManager::remove_cgroup(self, name)
    }
    fn set_cpuset(&self, name: &str, cpus: &BTreeSet<usize>) -> Result<()> {
        CgroupManager::set_cpuset(self, name, cpus)
    }
    fn clear_cpuset(&self, name: &str) -> Result<()> {
        CgroupManager::clear_cpuset(self, name)
    }
    fn set_cpuset_mems(&self, name: &str, nodes: &BTreeSet<usize>) -> Result<()> {
        CgroupManager::set_cpuset_mems(self, name, nodes)
    }
    fn clear_cpuset_mems(&self, name: &str) -> Result<()> {
        CgroupManager::clear_cpuset_mems(self, name)
    }
    fn set_cpu_max(&self, name: &str, quota_us: Option<u64>, period_us: u64) -> Result<()> {
        CgroupManager::set_cpu_max(self, name, quota_us, period_us)
    }
    fn set_cpu_weight(&self, name: &str, weight: u32) -> Result<()> {
        CgroupManager::set_cpu_weight(self, name, weight)
    }
    fn set_memory_max(&self, name: &str, bytes: Option<u64>) -> Result<()> {
        CgroupManager::set_memory_max(self, name, bytes)
    }
    fn set_memory_high(&self, name: &str, bytes: Option<u64>) -> Result<()> {
        CgroupManager::set_memory_high(self, name, bytes)
    }
    fn set_memory_low(&self, name: &str, bytes: Option<u64>) -> Result<()> {
        CgroupManager::set_memory_low(self, name, bytes)
    }
    fn set_io_weight(&self, name: &str, weight: u16) -> Result<()> {
        CgroupManager::set_io_weight(self, name, weight)
    }
    fn set_freeze(&self, name: &str, frozen: bool) -> Result<()> {
        CgroupManager::set_freeze(self, name, frozen)
    }
    fn set_pids_max(&self, name: &str, max: Option<u64>) -> Result<()> {
        CgroupManager::set_pids_max(self, name, max)
    }
    fn set_memory_swap_max(&self, name: &str, bytes: Option<u64>) -> Result<()> {
        CgroupManager::set_memory_swap_max(self, name, bytes)
    }
    fn move_task(&self, name: &str, pid: libc::pid_t) -> Result<()> {
        CgroupManager::move_task(self, name, pid)
    }
    fn move_tasks(&self, name: &str, pids: &[libc::pid_t]) -> Result<()> {
        CgroupManager::move_tasks(self, name, pids)
    }
    fn place_task_during_handshake(&self, cgroup_name: &str, child_pid: libc::pid_t) -> Result<()> {
        CgroupManager::place_task_during_handshake(self, cgroup_name, child_pid)
    }
    fn clear_subtree_control(&self, name: &str) -> Result<()> {
        CgroupManager::clear_subtree_control(self, name)
    }
    fn drain_tasks(&self, name: &str) -> Result<()> {
        CgroupManager::drain_tasks(self, name)
    }
    fn read_procs(&self, name: &str) -> Result<Vec<libc::pid_t>> {
        CgroupManager::read_procs(self, name)
    }
    fn cleanup_all(&self) -> Result<()> {
        CgroupManager::cleanup_all(self)
    }
}

/// Block until the cgroup at `cgroup_dir` reports `populated 0` via
/// its `cgroup.events` file, or until `budget` elapses. Event-driven
/// via inotify(IN_MODIFY) on the events file so the wait wakes on
/// the actual kernel state-transition write rather than a blind
/// sleep. Callers use the return value to decide whether to proceed
/// (cgroup empty — rmdir will succeed) or to fall through and let
/// the subsequent rmdir surface EBUSY for a genuinely-stuck cgroup.
///
/// Best-effort: a missing `cgroup.events` file (legacy kernels
/// without cgroup v2 events, non-cgroupfs paths threaded into this
/// helper by a test fixture, races where the parent dir was already
/// removed) returns `false` without waiting — the caller falls
/// through to its rmdir attempt which will surface the actual
/// error. inotify_init / add_watch failures degrade silently to a
/// short blocking sleep for the remaining budget.
fn wait_for_cgroup_unpopulated(cgroup_dir: &Path, budget: std::time::Duration) -> bool {
    use nix::poll::{PollFd, PollFlags, PollTimeout, poll};
    use nix::sys::inotify::{AddWatchFlags, InitFlags, Inotify};
    use std::os::unix::io::AsFd;

    let events_path = cgroup_dir.join("cgroup.events");
    // Tight initial check so a cgroup that's already empty
    // (extremely common — most drain_tasks call sites finish
    // synchronously) returns immediately without setting up inotify
    // or sleeping.
    if cgroup_events_reports_unpopulated(&events_path) {
        return true;
    }
    let deadline = std::time::Instant::now() + budget;
    // Inotify on the events file. IN_MODIFY fires every time the
    // kernel updates the populated count (1 → 0 transition included).
    // IN_NONBLOCK so read_events returns EAGAIN when empty — we
    // drive wake-vs-timeout via poll(2).
    let inotify_result =
        Inotify::init(InitFlags::IN_CLOEXEC | InitFlags::IN_NONBLOCK).and_then(|i| {
            i.add_watch(&events_path, AddWatchFlags::IN_MODIFY)?;
            Ok(i)
        });
    loop {
        if cgroup_events_reports_unpopulated(&events_path) {
            return true;
        }
        let now = std::time::Instant::now();
        if now >= deadline {
            return false;
        }
        let remaining_ms = deadline
            .duration_since(now)
            .as_millis()
            .min(u16::MAX as u128) as u16;
        match inotify_result.as_ref() {
            Ok(inotify) => {
                let fd = inotify.as_fd();
                let mut pollfds = [PollFd::new(fd, PollFlags::POLLIN)];
                let _ = poll(&mut pollfds, PollTimeout::from(remaining_ms));
                let _ = inotify.read_events();
            }
            Err(_) => {
                // Inotify unavailable on this path (legacy kernel,
                // missing events file, transient race). Fall back
                // to a brief blocking sleep so the loop still makes
                // progress under the deadline.
                std::thread::sleep(
                    std::time::Duration::from_millis(10).min(deadline.duration_since(now)),
                );
            }
        }
    }
}

/// Read `cgroup.events` and return `true` iff it contains a
/// `populated 0` line. Returns `false` for any read error or for
/// `populated 1` so the caller can keep waiting. The events file
/// is a small (~50 byte) flat key/value listing; full read each
/// poll iteration is cheap and avoids stateful parsing edge cases.
fn cgroup_events_reports_unpopulated(events_path: &Path) -> bool {
    match fs::read_to_string(events_path) {
        Ok(s) => s
            .lines()
            .any(|line| line.split_whitespace().eq(["populated", "0"])),
        Err(_) => false,
    }
}

/// Drain all tasks from `procs_path` to `dst` (the walk-root
/// `cgroup.procs`).
///
/// `dst` must be the `cgroup.procs` file at the cgroup-fs root the
/// caller is permitted to write to (under Mode A: `/sys/fs/cgroup`;
/// under Mode B/C: the delegated subtree root the operator owns).
/// The walk-root cgroup is exempt from (or above) the
/// no-internal-process constraint inside its delegation, so writes
/// to its `cgroup.procs` succeed even when intermediate cgroups have
/// `subtree_control` set.
///
/// ESRCH (task exited) is silently tolerated; other errors are
/// logged. A `read_to_string` failure or a malformed pid line is
/// surfaced via `tracing::warn!` — silently dropping either would
/// hide a cgroup that still contains tasks and send it into cleanup,
/// which then fails with EBUSY and compounds the confusion.
fn drain_pids_to_root(procs_path: &Path, dst: &Path, context: &str) {
    let content = match fs::read_to_string(procs_path) {
        Ok(c) => c,
        Err(e) => {
            tracing::warn!(
                path = %procs_path.display(),
                cgroup = context,
                err = %e,
                "drain_pids_to_root: read_to_string failed; tasks may remain in cgroup",
            );
            return;
        }
    };
    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let pid: u32 = match trimmed.parse() {
            Ok(p) => p,
            Err(e) => {
                tracing::warn!(
                    path = %procs_path.display(),
                    cgroup = context,
                    line = trimmed,
                    err = %e,
                    "drain_pids_to_root: malformed pid line; skipping",
                );
                continue;
            }
        };
        if let Err(e) = write_with_timeout(dst, &pid.to_string(), CGROUP_WRITE_TIMEOUT)
            && !is_esrch(&e)
        {
            tracing::warn!(pid, cgroup = context, err = %e, "failed to drain task");
        }
    }
}

/// Iterate the direct child directories of `path`, calling `f` on each.
///
/// `context` is a short caller name (e.g. `"cleanup_all"`,
/// `"cleanup_recursive"`) that is prefixed into every per-entry
/// `tracing::warn!` message so operators grepping logs for
/// `"cleanup_all: "` still see both the outer read_dir failure (which
/// stays with the caller) and the per-entry `DirEntry` / `file_type`
/// warnings emitted here.
///
/// `read_dir` failure is surfaced to the caller via `Err`; the caller
/// owns the top-level warn message. Non-directory entries are skipped.
/// Per-entry errors are logged and the iteration continues.
///
/// The structured log field key is normalized to `path =` at this
/// boundary; `cleanup_all`'s outer warn still uses `parent =` for the
/// top-level read_dir failure since that warn is emitted by the
/// caller, not here.
fn for_each_child_dir(path: &Path, context: &str, mut f: impl FnMut(&Path)) -> std::io::Result<()> {
    for entry in fs::read_dir(path)? {
        let entry = match entry {
            Ok(e) => e,
            Err(err) => {
                tracing::warn!(
                    path = %path.display(),
                    err = %err,
                    "{context}: dir entry read failed; skipping",
                );
                continue;
            }
        };
        match entry.file_type() {
            Ok(t) if t.is_dir() => f(&entry.path()),
            Ok(_) => {}
            Err(err) => tracing::warn!(
                path = %entry.path().display(),
                err = %err,
                "{context}: file_type read failed; skipping entry",
            ),
        }
    }
    Ok(())
}

/// Depth-first removal of `path` and every descendant cgroup
/// directory. Drains each cgroup's pids to `{walk_root}/cgroup.procs`
/// before rmdir.
///
/// `walk_root` mirrors [`CgroupManager::walk_root`]: under Mode A it
/// is `/sys/fs/cgroup` (the canonical cgroup-v2 mount); under Mode
/// B/C it is the delegated subtree root the operator owns. Threaded
/// through the recursion so every descendant drain targets the
/// caller's writable root and never the canonical
/// `/sys/fs/cgroup/cgroup.procs` (which would EACCES under
/// delegation).
fn cleanup_recursive(path: &std::path::Path, walk_root: &Path) {
    // Depth-first: clean children before parent
    if let Err(err) = for_each_child_dir(path, "cleanup_recursive", |child| {
        cleanup_recursive(child, walk_root)
    }) {
        tracing::warn!(
            path = %path.display(),
            err = %err,
            "cleanup_recursive: read_dir failed; child cgroups may remain",
        );
    }
    // Auto-unfreeze before draining tasks. Mirrors
    // `CgroupManager::remove_cgroup`'s pre-drain unfreeze, but for
    // defense-in-depth and source-cgroup state hygiene rather than
    // for correctness: the kernel's `cgroup_freezer_migrate_task`
    // path DOES unfreeze tasks when they migrate to an unfrozen
    // destination (the cgroup root is always unfrozen), so frozen
    // tasks would not actually strand at the root. The explicit
    // pre-drain `cgroup.freeze=0` write is still worthwhile because
    // it (a) makes the source cgroup's transient state visible in
    // tracing / `cgroup.events` before the directory disappears,
    // (b) avoids a brief frozen-counter churn while migration
    // batches advance, and (c) makes the teardown path symmetric
    // with `remove_cgroup` so operators reading either function
    // see the same auto-unfreeze step.
    //
    // Gate on existence: `fs::write` on a regular filesystem
    // CREATES the file when it doesn't exist (open(O_WRONLY |
    // O_CREAT | O_TRUNC)), so unconditionally writing
    // `cgroup.freeze` would plant a stray 1-byte file under any
    // non-cgroupfs directory and cause the subsequent
    // `fs::remove_dir(path)` to fail with ENOTEMPTY. On a real
    // cgroup v2 tree the file is always present (cgroup-core,
    // ungated by controllers); on a legacy kernel without
    // `CONFIG_CGROUP_FREEZE` or on a non-cgroup directory entry
    // the file is absent and the unfreeze step is a no-op.
    let freeze_path = path.join("cgroup.freeze");
    if freeze_path.exists()
        && let Err(err) = write_with_timeout(&freeze_path, "0", CGROUP_WRITE_TIMEOUT)
    {
        tracing::warn!(
            path = %path.display(),
            err = %format!("{err:#}"),
            "cleanup_recursive: pre-drain unfreeze failed; source-cgroup state-hygiene step skipped",
        );
    }
    drain_pids_to_root(
        &path.join("cgroup.procs"),
        &walk_root.join("cgroup.procs"),
        &path.display().to_string(),
    );
    // Wait event-driven on cgroup.events `populated 0` rather than
    // a blind 10 ms sleep — see `wait_for_cgroup_unpopulated`'s doc
    // for the rationale. 1 s deadline matches `remove_cgroup_inner`.
    wait_for_cgroup_unpopulated(path, std::time::Duration::from_secs(1));
    if let Err(err) = fs::remove_dir(path) {
        tracing::warn!(
            path = %path.display(),
            err = %err,
            "cleanup_recursive: remove_dir failed; cgroup directory may remain",
        );
    }
}

#[cfg(test)]
#[path = "cgroup_tests.rs"]
mod tests;
