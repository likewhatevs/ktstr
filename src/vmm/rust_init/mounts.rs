//! Guest filesystem mounts and cgroup-v2 hierarchy setup.
//!
//! Split from rust_init.rs; the shared consts/statics/imports live in the
//! parent module (`super`), reached via the glob below.
use super::*;

/// Mount essential filesystems.
pub(crate) fn mount_filesystems() {
    let mounts: &[(&str, &str, &str, bool)] = &[
        ("/proc", "proc", "proc", true),
        ("/sys", "sys", "sysfs", true),
        ("/dev", "dev", "devtmpfs", true),
        ("/sys/kernel/debug", "debugfs", "debugfs", false),
        ("/sys/kernel/tracing", "tracefs", "tracefs", false),
        ("/sys/fs/bpf", "bpffs", "bpf", false),
        ("/sys/fs/cgroup", "none", "cgroup2", false),
        ("/tmp", "tmpfs", "tmpfs", true),
        ("/dev/shm", "tmpfs", "tmpfs", false),
        ("/run", "tmpfs", "tmpfs", false),
    ];

    for &(target, source, fstype, required) in mounts {
        mkdir_p(target);
        let result = mount(
            Some(source),
            target,
            Some(fstype),
            MsFlags::empty(),
            None::<&str>,
        );
        if let Err(e) = result
            && required
        {
            // mount_filesystems() runs BEFORE the tracing subscriber
            // is installed (the subscriber needs /proc mounted to read
            // RUST_LOG from /proc/cmdline, so subscriber init follows
            // this call). Until that point fd 2 still routes to the
            // kernel console, but a `tracing::error!` event is dropped
            // because no subscriber is installed yet — this is the
            // tradeoff for installing the subscriber as early as
            // possible. A failed required-mount this early is itself
            // diagnosed downstream when /proc, /sys, or /dev are
            // missing for subsequent guest init steps.
            tracing::error!(fstype, target, err = %e, "ktstr-init: mount failed");
        }
    }

    // Standard /dev/fd symlinks. Needed by bpftrace and shell
    // process substitution (e.g. <(cmd)).
    let _ = std::os::unix::fs::symlink("/proc/self/fd", "/dev/fd");
    let _ = std::os::unix::fs::symlink("/proc/self/fd/0", "/dev/stdin");
    let _ = std::os::unix::fs::symlink("/proc/self/fd/1", "/dev/stdout");
    let _ = std::os::unix::fs::symlink("/proc/self/fd/2", "/dev/stderr");
}

/// Auto-mount the user-configured data disk at `/mnt/disk0` if the
/// host pre-formatted it. Driven by two kernel cmdline tokens
/// emitted by the host's
/// [`crate::vmm::KtstrVmBuilder::build`] cmdline assembly:
///
/// * `KTSTR_DISK0_FS=<tag>` — selects the on-disk filesystem to
///   pass to `mount(2)` (`btrfs` for the only non-Raw variant
///   today). Absence short-circuits this whole function: a `Raw`
///   disk has nothing to mount, and a config with no disk attached
///   never sees a `KTSTR_DISK0_FS` token at all.
/// * `KTSTR_DISK0_RO=1` — set when the host configured the disk
///   `read_only`. The virtio_blk device advertises
///   `VIRTIO_BLK_F_RO` for that case so the guest's gendisk is
///   read-only at the block layer; mounting RW would fail with
///   `-EROFS` (kernel `do_mount` sets the superblock RO from the
///   bdev). Setting `MS_RDONLY` proactively avoids that error path
///   entirely.
///
/// Failure modes are non-fatal: if the mount syscall returns an
/// error (unrecognized fstype tag, kernel `CONFIG_BTRFS_FS=n`,
/// device probe race, ENOMEM), the function logs to COM2 and
/// returns. The test still gets a usable VM; a subsequent test
/// step that depends on `/mnt/disk0` surfaces as a clean
/// userspace filesystem error rather than a confusing init abort.
///
/// Skips entirely when `KTSTR_DISK0_FS` is absent. The cmdline
/// emission on the host side is gated on
/// `disk.filesystem != Filesystem::Raw && !disk.no_auto_mount`, so
/// this branch matches the host-side opt-in: a `Raw` disk or a disk
/// built with `no_auto_mount` emits no `KTSTR_DISK0_FS` token and is
/// not auto-mounted here, while every other config that requests an
/// on-disk filesystem gets the auto-mount.
pub(crate) fn auto_mount_data_disks() {
    let Some(fstype) = cmdline_val("KTSTR_DISK0_FS") else {
        return;
    };
    // Validate the fstype against the known set. Today only
    // `btrfs` is wired (mirroring `Filesystem::Btrfs::cache_tag`);
    // unknown values warn-and-skip rather than handing arbitrary
    // strings to `mount(2)`. A future `Filesystem` variant must
    // add its tag here AND in the disk_config.rs `cache_tag`
    // match — keeping both lists in lockstep is the on-disk-format
    // / cmdline contract.
    let recognized = matches!(fstype.as_str(), "btrfs");
    if !recognized {
        let msg = format!(
            "ktstr-init: KTSTR_DISK0_FS={fstype} not recognized; \
             skipping auto-mount of /dev/vda"
        );
        let _ = fs::write(COM2, &msg);
        tracing::warn!("{msg}");
        return;
    }
    // RO bit. Absent or any value other than "1" means RW.
    // Strict-`==` rather than truthy-string parsing keeps the
    // contract simple and aligned with the host-side emission
    // (`KTSTR_DISK0_RO=1`).
    let ro = cmdline_val("KTSTR_DISK0_RO").as_deref() == Some("1");
    // Mount path. The host emits `KTSTR_DISK0_MOUNT=<path>` based
    // on `DiskConfig.name` — `/mnt/<name>` when set, `/mnt/disk0`
    // otherwise. Fall back to the default if the host-side value
    // is absent so a future host that emits FS but not MOUNT
    // (e.g. an older binary against a newer kernel) still mounts
    // somewhere sane rather than failing.
    let mount_point_owned =
        cmdline_val("KTSTR_DISK0_MOUNT").unwrap_or_else(|| "/mnt/disk0".to_string());
    let mount_point = mount_point_owned.as_str();
    mkdir_p(mount_point);
    let flags = if ro {
        MsFlags::MS_RDONLY
    } else {
        MsFlags::empty()
    };
    let result = mount(
        Some("/dev/vda"),
        mount_point,
        Some(fstype.as_str()),
        flags,
        None::<&str>,
    );
    if let Err(e) = result {
        let msg = format!(
            "ktstr-init: mount {fstype} on {mount_point} \
             (ro={ro}): {e}"
        );
        let _ = fs::write(COM2, &msg);
        tracing::warn!("{msg}");
    }
}

/// Recursive mkdir -p equivalent. `DirBuilder::recursive(true)` is
/// idempotent (returns Ok when the path already exists as a
/// directory) and walks parents internally, so the hand-rolled
/// recursion this replaced was redundant. Errors are swallowed to
/// match the previous behavior — the early guest init best-effort
/// creates each mount point and continues regardless, since any
/// real failure surfaces downstream when `mount()` itself fails.
///
/// Directory mode is pinned explicitly at 0o755 via
/// `DirBuilder::mode`. Relying on the default (0o777 & !umask) is
/// fragile: the guest init's umask is process state inherited from
/// the kernel/caller, and a caller that sets umask=0 before exec
/// would produce world-writable mount points. Pinning the mode in
/// the mkdir syscall itself keeps the traversal bit stable
/// regardless of umask.
pub(crate) fn mkdir_p(path: &str) {
    use std::os::unix::fs::DirBuilderExt;
    let _ = fs::DirBuilder::new()
        .recursive(true)
        .mode(0o755)
        .create(path);
}

/// Write a line to COM2 (the application serial port).
/// Falls back to the tracing subscriber (writing to stderr) if COM2
/// is not available.
pub(crate) fn write_com2(msg: &str) {
    if let Ok(mut f) = fs::OpenOptions::new().write(true).open(COM2) {
        let _ = writeln!(f, "{msg}");
    } else {
        // COM2 unavailable (devtmpfs mount failed or device missing).
        // Surface via the tracing subscriber so the host sees
        // something on the COM1 fallback path.
        tracing::warn!(target: "com1_fallback", "ktstr-init: {msg}");
    }
}

/// Materialise the per-test workload-cgroup root declared via
/// `#[ktstr_test(workload_root_cgroup = "/path")]`. Reads
/// `/workload_root_cgroup` (written by
/// [`crate::vmm::initramfs::build_dynamic_tail`] when
/// [`crate::vmm::initramfs::SuffixParams::workload_root_cgroup`] is
/// `Some`), validates the absolute-path shape, mkdir's
/// `/sys/fs/cgroup{path}`, and enables `+cpuset +cpu` controllers
/// along every ancestor so the workload cgroups the test author
/// creates beneath this root inherit the controllers they need.
///
/// Distinct from [`create_cgroup_parent_from_sched_args`]: that one
/// services the `--cell-parent-cgroup` scheduler-argv knob (only
/// present when the scheduler declaration explicitly carries the
/// flag); this one services the framework's per-test workload root
/// (created unconditionally when the test sets the field). Both
/// run in Phase 3 before `start_scheduler`; ordering between the
/// two is `workload_root_cgroup` first so it's visible when a
/// scheduler that does carry `--cell-parent-cgroup` walks the
/// cgroup tree at startup.
#[tracing::instrument]
pub(crate) fn create_workload_root_cgroup_from_file() {
    create_cgroup_from_file("/workload_root_cgroup");
}

/// Materialise the per-scheduler cgroup the scheduler process is
/// placed in. Reads `/scheduler_cgroup_parent` (written by
/// [`crate::vmm::initramfs::build_dynamic_tail`] when
/// [`crate::vmm::initramfs::SuffixParams::scheduler_cgroup_parent`]
/// is `Some` — sourced from
/// [`crate::test_support::Scheduler::cgroup_parent`]), validates
/// the absolute-path shape, mkdir's `/sys/fs/cgroup{path}`, and
/// enables `+cpuset +cpu` controllers along every ancestor so the
/// scheduler's later cgroup operations find the controllers
/// already available.
///
/// Distinct from [`create_workload_root_cgroup_from_file`] (per-
/// test workload tree) and from
/// [`create_cgroup_parent_from_sched_args`] (which fires only
/// when `--cell-parent-cgroup` is present in `/sched_args` for
/// cell-aware schedulers).
#[tracing::instrument]
pub(crate) fn create_scheduler_cgroup_parent_from_file() {
    create_cgroup_from_file("/scheduler_cgroup_parent");
}

/// Shared mkdir + subtree-controller setup for any
/// framework-stamped cgroup-path file. Centralises the file-read,
/// path-validation, mkdir, and `enable_subtree_controllers_to`
/// sequence so future cgroup-path slots reuse the same flow
/// without duplicating the guard logic.
fn create_cgroup_from_file(file: &str) {
    let raw = match fs::read_to_string(file) {
        Ok(s) => s,
        Err(_) => return,
    };
    let path = raw.trim();
    if !crate::test_support::cell_parent_path_is_valid(path) {
        if !path.is_empty() {
            write_com2(&format!(
                "ktstr-init: ignoring malformed `{file}` value {path:?}; \
                 skipping cgroup creation (host-side `CgroupPath::new` \
                 gate normally rejects this at compile time)",
            ));
        }
        return;
    }
    let cgroup_dir = format!("/sys/fs/cgroup{path}");
    mkdir_p(&cgroup_dir);
    enable_subtree_controllers_to(&cgroup_dir);
}

/// Create the cgroup parent directory specified by `--cell-parent-cgroup`
/// (two-token or `=`-combined form) in `/sched_args`. The directory must
/// exist before the scheduler starts because the scheduler expects it at
/// startup.
///
/// In cgroup v2, a controller is only visible inside a cgroup when its
/// parent's `cgroup.subtree_control` enables it. The kernel enforces
/// this in `cgroup_subtree_control_write` via `cgroup_control(cgrp)`,
/// which returns `parent->subtree_control` for non-root cgroups. To
/// make `cpuset` and `cpu` available in the leaf, every ancestor from
/// the cgroup root down to (and including) the leaf's immediate parent
/// must enable both controllers. Writes are applied root-to-leaf so
/// each level's prerequisite is already in place when its child is
/// written.
#[tracing::instrument]
pub(crate) fn create_cgroup_parent_from_sched_args() {
    let sched_args = match fs::read_to_string("/sched_args") {
        Ok(s) => s,
        Err(_) => return,
    };
    // Defense-in-depth filter: the host-side gate in
    // `runtime::append_base_sched_args` panics on malformed values
    // (non-absolute, bare `/`, missing) before `/sched_args` is
    // written. Reaching this path with a bad value means the gate
    // was bypassed (operator hand-edited an exported `.run` script,
    // ad-hoc argv injection); log to COM2 and skip the cgroup-tree
    // setup rather than mkdir on the host cgroup root.
    let path = match crate::test_support::parse_cell_parent_cgroup(sched_args.split_whitespace()) {
        crate::test_support::CellParentCgroupArg::Value(p)
            if crate::test_support::cell_parent_path_is_valid(p) =>
        {
            p
        }
        crate::test_support::CellParentCgroupArg::Value(bad) => {
            write_com2(&format!(
                "ktstr-init: ignoring malformed `--cell-parent-cgroup` value \
                 {bad:?} in /sched_args; skipping per-test cgroup creation \
                 (host-side gate normally panics on this)",
            ));
            return;
        }
        crate::test_support::CellParentCgroupArg::MissingValue => {
            write_com2(
                "ktstr-init: ignoring bare `--cell-parent-cgroup` (no value) \
                 in /sched_args; skipping per-test cgroup creation",
            );
            return;
        }
        crate::test_support::CellParentCgroupArg::Absent => return,
    };
    let cgroup_dir = format!("/sys/fs/cgroup{path}");
    mkdir_p(&cgroup_dir);
    enable_subtree_controllers_to(&cgroup_dir);
}

/// Enable `+cpuset +cpu` in `cgroup.subtree_control` at every ancestor
/// from `/sys/fs/cgroup` (inclusive) down to (and including) the
/// immediate parent of `leaf`. Writes are ordered root-first so each
/// level's parent already advertises the controllers when its child is
/// written — without that ordering the kernel rejects the write with
/// `-ENOENT` (see `cgroup_subtree_control_write` /
/// `cgroup_control` in `kernel/cgroup/cgroup.c`).
///
/// `leaf` is expected to live under `/sys/fs/cgroup/...` (the format
/// emitted at the call site). The leaf itself is NOT written: enabling
/// controllers in a cgroup means they are visible inside that cgroup's
/// CHILDREN, so the leaf's own `subtree_control` only matters if the
/// scheduler ever creates sub-cgroups under it. The scheduler attaches
/// tasks to the leaf, so what it needs is `cpuset`/`cpu` enabled IN
/// the leaf — which is achieved by writing to the leaf's parent.
///
/// Failures on individual writes are logged via [`write_com2`] and do
/// not abort the walk: a single intermediate level that already has
/// both controllers enabled returns `0` from kernel side, so most
/// failures observed here will surface a real misconfiguration that
/// the scheduler's own `cgroup_attach` will then re-report with
/// scheduler-specific context.
fn enable_subtree_controllers_to(leaf: &str) {
    let cgroup_root = Path::new("/sys/fs/cgroup");
    let leaf_path = Path::new(leaf);
    // Verify leaf is under the cgroup root before touching anything.
    // A malformed `--cell-parent-cgroup` argument that produces a path
    // outside `/sys/fs/cgroup` (e.g. an empty or missing-leading-slash
    // value) would otherwise walk into `/sys/fs`, `/sys`, or `/`.
    if !leaf_path.starts_with(cgroup_root) || leaf_path == cgroup_root {
        return;
    }
    // `Path::ancestors` yields leaf-first; collect the strict ancestors
    // (skip the leaf itself) up to and including the cgroup root.
    let mut ancestors: Vec<&Path> = leaf_path
        .ancestors()
        .skip(1)
        .take_while(|p| p.starts_with(cgroup_root))
        .collect();
    // Apply root-to-leaf-parent: each level's parent must already
    // enable the controller before the child write is accepted.
    ancestors.reverse();
    for level in ancestors {
        let control = level.join("cgroup.subtree_control");
        if let Err(e) = fs::write(&control, "+cpuset +cpu") {
            write_com2(&format!(
                "ktstr-init: write {} +cpuset +cpu: {}",
                control.display(),
                e
            ));
        }
    }
}
