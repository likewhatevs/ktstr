//! Unit tests for [`super`] (the `cgroup` module).
//! Co-located via the `tests` submodule pattern.

#![cfg(test)]

    use super::*;

    #[test]
    fn cgroup_manager_path() {
        let cg = CgroupManager::new("/sys/fs/cgroup/test");
        assert_eq!(
            cg.parent_path(),
            std::path::Path::new("/sys/fs/cgroup/test")
        );
    }

    #[test]
    fn create_cgroup_in_tmpdir() {
        let _tempdir_keep_alive = make_inline_tempdir("create-in-tmpdir");
        let dir = _tempdir_keep_alive.path();
        let cg = CgroupManager::new(dir.to_str().unwrap());
        cg.create_cgroup("test_cg").unwrap();
        assert!(dir.join("test_cg").exists());
        cg.create_cgroup("nested/deep").unwrap();
        assert!(dir.join("nested/deep").exists());
    }

    #[test]
    fn create_cgroup_idempotent() {
        let _tempdir_keep_alive = make_inline_tempdir("idem");
        let dir = _tempdir_keep_alive.path();
        let cg = CgroupManager::new(dir.to_str().unwrap());
        cg.create_cgroup("cg_0").unwrap();
        cg.create_cgroup("cg_0").unwrap(); // should not error
        assert!(dir.join("cg_0").exists());
    }

    #[test]
    fn cleanup_all_on_nonexistent() {
        let cg = CgroupManager::new("/nonexistent/ktstr-test-path");
        assert!(cg.cleanup_all().is_ok());
    }

    #[test]
    fn remove_cgroup_nonexistent() {
        let cg = CgroupManager::new("/nonexistent/ktstr-test-path");
        assert!(cg.remove_cgroup("no_such_cgroup").is_ok());
    }

    #[test]
    fn cleanup_removes_child_dirs() {
        let _tempdir_keep_alive = make_inline_tempdir("clean");
        let dir = _tempdir_keep_alive.path();
        let cg = CgroupManager::new(dir.to_str().unwrap());
        cg.create_cgroup("a").unwrap();
        cg.create_cgroup("b").unwrap();
        cg.create_cgroup("c/deep").unwrap();
        assert!(dir.join("a").exists());
        assert!(dir.join("c/deep").exists());
        // cleanup_all removes child dirs (not real cgroups, so drain_tasks is a no-op)
        cg.cleanup_all().unwrap();
        assert!(!dir.join("a").exists());
        assert!(!dir.join("b").exists());
        assert!(!dir.join("c").exists());
    }

    #[test]
    fn drain_tasks_nonexistent_source() {
        let cg = CgroupManager::new("/nonexistent/ktstr-drain-test");
        assert!(cg.drain_tasks("missing_cgroup").is_ok());
    }

    /// `cleanup_all` must skip non-directory entries rather than
    /// recurse into them. Plants a regular file alongside a child
    /// cgroup directory and verifies: (a) the child dir is removed,
    /// (b) the file is left in place. Pins the `Ok(t) if t.is_dir()`
    /// branch in [`for_each_child_dir`] so a future refactor that
    /// drops the `is_dir` guard fails this test instead of silently
    /// deleting arbitrary files under the cgroup parent.
    #[test]
    fn cleanup_all_skips_non_dir_entries() {
        let _tempdir_keep_alive = make_inline_tempdir("nondir");
        let dir = _tempdir_keep_alive.path();
        let cg = CgroupManager::new(dir.to_str().unwrap());
        cg.create_cgroup("cg_child").unwrap();
        let stray_file = dir.join("stray.txt");
        fs::write(&stray_file, b"do not descend").unwrap();
        assert!(dir.join("cg_child").exists());
        assert!(stray_file.exists());
        cg.cleanup_all().unwrap();
        assert!(
            !dir.join("cg_child").exists(),
            "cleanup_all should remove the child directory",
        );
        assert!(
            stray_file.exists(),
            "cleanup_all must not descend into or remove regular files",
        );
        assert_eq!(fs::read_to_string(&stray_file).unwrap(), "do not descend");
    }

    /// `cleanup_recursive` on a 2-level nested directory structure must
    /// remove leaves before their parents (depth-first). Plants
    /// `root/mid/leaf/` plus `root/sibling/`, invokes
    /// [`cleanup_recursive`] directly on `root`, and verifies every
    /// directory is gone. Exercises the recursive call inside
    /// [`for_each_child_dir`] that item 7's `cleanup_recursive`
    /// function-pointer arg drives.
    #[test]
    fn cleanup_recursive_removes_nested_dirs_depth_first() {
        let _tempdir_keep_alive = make_inline_tempdir("nested");
        let base = _tempdir_keep_alive.path();
        let root = base.join("root");
        fs::create_dir_all(root.join("mid").join("leaf")).unwrap();
        fs::create_dir_all(root.join("sibling")).unwrap();
        assert!(root.join("mid/leaf").exists());
        assert!(root.join("sibling").exists());
        cleanup_recursive(&root);
        assert!(
            !root.exists(),
            "cleanup_recursive should remove root and every descendant",
        );
    }

    #[test]
    fn setup_non_cgroup_path() {
        // setup() on a non-cgroup path should still create the dir.
        // Empty controller set skips the subtree_control walk entirely.
        let _tempdir_keep_alive = make_inline_tempdir("setup");
        let dir = _tempdir_keep_alive.path();
        let cg = CgroupManager::new(dir.to_str().unwrap());
        cg.setup(&BTreeSet::new()).unwrap();
        assert!(dir.exists());
    }

    /// `setup` writes only the controllers the caller requested to
    /// `cgroup.subtree_control`. Pinning a focused minimum
    /// (cpuset + memory) catches regressions where the rendered
    /// `+token` list grows past what the caller asked for.
    ///
    /// Path: drives [`setup_under_root`] against a tmpdir-rooted
    /// "cgroup tree" (parent dir + pre-created
    /// `cgroup.controllers` advertising both controllers +
    /// `cgroup.subtree_control` at root and leaf), then reads the
    /// leaf back and asserts both requested tokens land while
    /// non-requested controllers do not.
    #[test]
    fn setup_writes_requested_controllers_only() {
        let _tempdir_keep_alive = make_inline_tempdir("setup-controllers");
        let root = _tempdir_keep_alive.path();
        let parent = root.join("ktstr");
        fs::create_dir_all(&parent).unwrap();
        // Pre-create cgroup.controllers at root so the availability
        // check in setup_under_root passes for the requested
        // controllers. Production cgroup v2 mount populates this file.
        fs::write(root.join("cgroup.controllers"), "cpuset cpu memory pids io").unwrap();
        // Pre-create the subtree_control file at root and leaf so
        // the strip-prefix walk's exists() gate sees them.
        fs::write(root.join("cgroup.subtree_control"), "").unwrap();
        fs::write(parent.join("cgroup.subtree_control"), "").unwrap();

        let cg = CgroupManager::new(parent.to_str().unwrap());
        let mut requested = BTreeSet::new();
        requested.insert(Controller::Cpuset);
        requested.insert(Controller::Memory);
        cg.setup_under_root(&requested, &root).unwrap();

        let written = fs::read_to_string(parent.join("cgroup.subtree_control")).unwrap();
        assert!(
            written.contains("+cpuset"),
            "subtree_control must contain +cpuset; got: {written:?}",
        );
        assert!(
            written.contains("+memory"),
            "subtree_control must contain +memory; got: {written:?}",
        );
        // Non-requested controllers must NOT appear.
        assert!(
            !written.contains("+pids"),
            "+pids must be absent when not requested; got: {written:?}",
        );
        assert!(
            !written.contains("+io"),
            "+io must be absent when not requested; got: {written:?}",
        );
        // +cpu must be absent. Distinguish from +cpuset by walking
        // every +cpu* match position and asserting it's the +cpuset
        // prefix.
        let cpu_positions: Vec<usize> = written.match_indices("+cpu").map(|(i, _)| i).collect();
        for pos in cpu_positions {
            let suffix = &written[pos..];
            assert!(
                suffix.starts_with("+cpuset"),
                "+cpu must be absent when not requested (only +cpuset allowed); \
                 got '{suffix}' at pos {pos} in {written:?}",
            );
        }
    }

    /// `setup` rejects an unavailable controller with a clear error
    /// citing both the requested controller name and the kernel's
    /// advertised set. Without the gate, the downstream
    /// `set_*` write would fail with bare ENOENT/EACCES — much
    /// harder to diagnose than "controller X not available".
    #[test]
    fn setup_rejects_unavailable_controller() {
        let _tempdir_keep_alive = make_inline_tempdir("setup-unavail");
        let root = _tempdir_keep_alive.path();
        let parent = root.join("ktstr");
        fs::create_dir_all(&parent).unwrap();
        // Advertise only memory; request cpuset.
        fs::write(root.join("cgroup.controllers"), "memory").unwrap();
        fs::write(root.join("cgroup.subtree_control"), "").unwrap();
        fs::write(parent.join("cgroup.subtree_control"), "").unwrap();

        let cg = CgroupManager::new(parent.to_str().unwrap());
        let mut requested = BTreeSet::new();
        requested.insert(Controller::Cpuset);
        let err = cg.setup_under_root(&requested, &root).unwrap_err();
        let msg = format!("{err:#}");
        assert!(
            msg.contains("cpuset") && msg.contains("not available"),
            "error must cite missing 'cpuset' and 'not available'; got {msg:?}",
        );
    }

    #[test]
    fn write_with_timeout_success() {
        let _tempdir_keep_alive = make_inline_tempdir("write-timeout");
        let dir = _tempdir_keep_alive.path();
        let f = dir.join("test_write");
        write_with_timeout(&f, "hello", Duration::from_secs(5)).unwrap();
        assert_eq!(fs::read_to_string(&f).unwrap(), "hello");
    }

    #[test]
    fn write_with_timeout_bad_path() {
        let f = Path::new("/nonexistent/dir/file");
        assert!(write_with_timeout(f, "data", Duration::from_secs(5)).is_err());
    }

    #[test]
    fn move_task_nonexistent_cgroup() {
        let cg = CgroupManager::new("/nonexistent/ktstr-move-test");
        assert!(cg.move_task("no_cgroup", 1).is_err());
    }

    #[test]
    fn set_cpuset_empty() {
        let _tempdir_keep_alive = make_inline_tempdir("cpuset");
        let dir = _tempdir_keep_alive.path();
        let dir_a = dir.join("cg_a");
        fs::create_dir_all(&dir_a).unwrap();
        let cg = CgroupManager::new(dir.to_str().unwrap());
        // Empty BTreeSet → writes empty string via cpuset_string
        cg.set_cpuset("cg_a", &BTreeSet::new()).unwrap();
        assert_eq!(fs::read_to_string(dir_a.join("cpuset.cpus")).unwrap(), "");
    }

    #[test]
    fn move_tasks_partial_failure() {
        // move_tasks propagates non-ESRCH errors immediately
        let cg = CgroupManager::new("/nonexistent/ktstr-partial");
        let err = cg.move_tasks("cg", &[1, 2, 3]).unwrap_err();
        // The error comes from the first pid (write to nonexistent path)
        let msg = format!("{err:#}");
        assert!(msg.contains("cgroup.procs"), "unexpected error: {msg}");
    }

    #[test]
    fn drain_tasks_empty_cgroup() {
        let _tempdir_keep_alive = make_inline_tempdir("drain");
        let dir = _tempdir_keep_alive.path();
        let dir_d = dir.join("cg_d");
        fs::create_dir_all(&dir_d).unwrap();
        // Create an empty cgroup.procs file
        fs::write(dir_d.join("cgroup.procs"), "").unwrap();
        // Parent also needs cgroup.procs
        fs::write(dir.join("cgroup.procs"), "").unwrap();
        let cg = CgroupManager::new(dir.to_str().unwrap());
        // drain_tasks on a cgroup with empty procs file should succeed
        assert!(cg.drain_tasks("cg_d").is_ok());
    }

    #[test]
    fn is_esrch_detects_esrch_in_chain() {
        let io_err = std::io::Error::from_raw_os_error(libc::ESRCH);
        let anyhow_err = anyhow::Error::new(io_err).context("write cgroup.procs");
        assert!(is_esrch(&anyhow_err));
    }

    #[test]
    fn is_esrch_rejects_enoent() {
        let io_err = std::io::Error::from_raw_os_error(libc::ENOENT);
        let anyhow_err = anyhow::Error::new(io_err).context("write cgroup.procs");
        assert!(!is_esrch(&anyhow_err));
    }

    #[test]
    fn is_ebusy_detects_ebusy_in_chain() {
        let io_err = std::io::Error::from_raw_os_error(libc::EBUSY);
        let anyhow_err = anyhow::Error::new(io_err).context("write cgroup.procs");
        assert!(is_ebusy(&anyhow_err));
    }

    #[test]
    fn is_ebusy_rejects_esrch() {
        let io_err = std::io::Error::from_raw_os_error(libc::ESRCH);
        let anyhow_err = anyhow::Error::new(io_err).context("write cgroup.procs");
        assert!(!is_ebusy(&anyhow_err));
    }

    #[test]
    fn clear_subtree_control_nonexistent() {
        let cg = CgroupManager::new("/nonexistent/ktstr-clear-sc");
        // No subtree_control file → no-op success.
        assert!(cg.clear_subtree_control("cg_0").is_ok());
    }

    #[test]
    fn clear_subtree_control_empty() {
        let _tempdir_keep_alive = make_inline_tempdir("subtree-control");
        let dir = _tempdir_keep_alive.path();
        let dir_a = dir.join("cg_a");
        fs::create_dir_all(&dir_a).unwrap();
        fs::write(dir_a.join("cgroup.subtree_control"), "").unwrap();
        let cg = CgroupManager::new(dir.to_str().unwrap());
        // Empty subtree_control → no-op success.
        assert!(cg.clear_subtree_control("cg_a").is_ok());
    }

    #[test]
    fn write_with_timeout_blocks_on_fifo() {
        use std::ffi::CString;
        let _tempdir_keep_alive = make_inline_tempdir("fifo");
        let dir = _tempdir_keep_alive.path();
        let fifo_path = dir.join("blocked_write");
        let c_path = CString::new(fifo_path.to_str().unwrap()).unwrap();
        let rc = unsafe { libc::mkfifo(c_path.as_ptr(), 0o700) };
        assert_eq!(rc, 0, "mkfifo failed: {}", std::io::Error::last_os_error());
        // Very short timeout — write blocks until a reader opens the FIFO
        let err = write_with_timeout(&fifo_path, "data", Duration::from_millis(50)).unwrap_err();
        let msg = format!("{err:#}");
        assert!(msg.contains("timed out"), "unexpected error: {msg}");
    }

    #[test]
    fn anyhow_first_io_errno_extracts_raw_errno() {
        let io = std::io::Error::from_raw_os_error(libc::EBUSY);
        let err = anyhow::Error::new(io);
        assert_eq!(anyhow_first_io_errno(&err), Some(libc::EBUSY));
    }

    #[test]
    fn anyhow_first_io_errno_through_context() {
        let io = std::io::Error::from_raw_os_error(libc::ESRCH);
        let err = anyhow::Error::new(io).context("wrapping context");
        assert_eq!(anyhow_first_io_errno(&err), Some(libc::ESRCH));
    }

    #[test]
    fn anyhow_first_io_errno_no_io_returns_none() {
        let err = anyhow::anyhow!("plain text error");
        assert_eq!(anyhow_first_io_errno(&err), None);
    }

    #[test]
    fn add_parent_subtree_controller_missing_file_noop() {
        let cg = CgroupManager::new("/nonexistent/ktstr-add-parent-sc");
        assert!(cg.add_parent_subtree_controller("cpuset").is_ok());
    }

    #[test]
    fn add_parent_subtree_controller_writes_plus_prefixed_token() {
        let _tempdir_keep_alive = make_inline_tempdir("addparent");
        let dir = _tempdir_keep_alive.path();
        // The subtree_control file in a real cgroup v2 tree echoes the
        // currently-enabled controllers (no `+` prefix) when read back;
        // here we just observe that our write landed verbatim.
        let sc = dir.join("cgroup.subtree_control");
        fs::write(&sc, "").unwrap();
        let cg = CgroupManager::new(dir.to_str().unwrap());
        cg.add_parent_subtree_controller("cpuset").unwrap();
        assert_eq!(fs::read_to_string(&sc).unwrap(), "+cpuset");
    }

    // -- Cgroup v2 resource control writes ----------------------------
    //
    // Each new CgroupOps method writes a single cgroupfs file. The
    // tests below stand up a tmpdir representing the parent cgroup,
    // pre-create the child + the target file (real cgroupfs creates
    // these on directory creation; tmpfs needs them touched), invoke
    // the method, and assert on the resulting file contents.

    /// Construct a `tempfile::TempDir` with the canonical
    /// `ktstr-{label}-` prefix. Returns the `TempDir` by value so
    /// callers own the RAII handle; bind it as
    /// `_tempdir_keep_alive` (or `_<role>_keep_alive`) per the
    /// convention spelled out on [`make_test_cgroup`].
    ///
    /// `label` should describe what the test exercises (e.g.,
    /// `nested`, `cpuset`, `subtree-control`) — not the file the
    /// test lives in. The `ktstr-` prefix already namespaces
    /// against other on-host tempdirs; an additional `cg-` prefix
    /// is automatically injected by [`make_test_cgroup`] for tests
    /// built around the `CgroupManager` fixture, so direct callers
    /// of `make_inline_tempdir` should NOT add `cg-` themselves.
    /// Use hyphens (not underscores) for multi-word labels to
    /// keep tempdir names readable.
    fn make_inline_tempdir(label: &str) -> tempfile::TempDir {
        tempfile::Builder::new()
            .prefix(&format!("ktstr-{label}-"))
            .tempdir()
            .expect("tempdir")
    }

    /// Spin up an isolated tempdir + pre-populated `cg_x` subdir +
    /// `CgroupManager` for a single `#[test]` body. Wraps
    /// [`make_inline_tempdir`] with a `cg-{label}` prefix and adds
    /// the `cg_x` child directory that production code paths expect.
    /// The returned `TempDir` is the RAII teardown handle; bind it as
    /// `_tempdir_keep_alive` (or any underscore-prefix identifier
    /// other than bare `_`) to keep the tempdir alive for the
    /// duration of the test and let `Drop` recursively remove it
    /// on test exit (success OR panic).
    ///
    /// DO NOT rename the binding to bare `_` — bare `_` discards
    /// the value immediately, so `Drop` would fire BEFORE the test
    /// body runs and every subsequent `fs::write(&target, ...)`
    /// would fail with `ENOENT`. The failure mode is loud (the
    /// test panics on the missing path), but the trap is real: any
    /// IDE rename or clippy fix that mistakes the underscore-prefix
    /// identifier for a true discard binding will introduce the
    /// regression. The `_tempdir_keep_alive` name is deliberately
    /// verbose to discourage that rewrite. The same convention
    /// applies to direct [`make_inline_tempdir`] callers.
    ///
    /// The second tuple element is the tempdir root path (a
    /// `PathBuf`) for test bodies that use `dir.join("cg_x").join(...)`
    /// to build target file paths.
    ///
    /// See [`make_test_cgroup_with_procs`] for tests that need an
    /// empty `cgroup.procs` pre-seeded, or
    /// [`make_test_cgroup_with_seeded_file`] for tests that need a
    /// specific knob file (e.g. `cpu.max`) pre-seeded. Note that
    /// those helpers return the same `(TempDir, PathBuf, CgroupManager)`
    /// shape but the second element's SEMANTIC differs: this helper
    /// returns the tempdir root, `_with_procs` returns the `cg_x`
    /// subdir, `_with_seeded_file` returns the seeded file path.
    /// Rename the binding (`dir` / `inner` / `target`) to match
    /// when migrating between helpers.
    fn make_test_cgroup(label: &str) -> (tempfile::TempDir, PathBuf, CgroupManager) {
        let tmp = make_inline_tempdir(&format!("cg-{label}"));
        let dir = tmp.path().to_path_buf();
        fs::create_dir_all(dir.join("cg_x")).unwrap();
        let cg = CgroupManager::new(dir.to_str().unwrap());
        (tmp, dir, cg)
    }

    /// Like [`make_test_cgroup`] but additionally pre-creates an
    /// empty `cgroup.procs` file inside the `cg_x` subdir. The
    /// second tuple element is the `cg_x` subdir path (NOT the
    /// tempdir root and NOT a file path — distinct from the second
    /// tuple element returned by [`make_test_cgroup`] and
    /// [`make_test_cgroup_with_seeded_file`]) so callers can
    /// `inner.join("FILE.name")` against it directly without
    /// re-deriving the path from the tempdir.
    ///
    /// See [`make_test_cgroup_with_seeded_file`] when the test needs
    /// a specific knob file pre-seeded instead of `cgroup.procs`.
    fn make_test_cgroup_with_procs(label: &str) -> (tempfile::TempDir, PathBuf, CgroupManager) {
        let (tmp, dir, cg) = make_test_cgroup(label);
        let inner = dir.join("cg_x");
        fs::write(inner.join("cgroup.procs"), "").unwrap();
        (tmp, inner, cg)
    }

    /// Like [`make_test_cgroup`] but additionally pre-creates an
    /// empty file named `filename` inside the `cg_x` subdir. The
    /// second tuple element is the seeded file path (NOT the
    /// tempdir root and NOT the `cg_x` subdir — distinct from the
    /// second tuple element returned by [`make_test_cgroup`] and
    /// [`make_test_cgroup_with_procs`]) so callers can
    /// `fs::read_to_string(&target)` it directly without re-joining.
    ///
    /// See [`make_test_cgroup_with_procs`] when the test needs
    /// `cgroup.procs` pre-seeded instead of a specific knob file.
    fn make_test_cgroup_with_seeded_file(
        label: &str,
        filename: &str,
    ) -> (tempfile::TempDir, PathBuf, CgroupManager) {
        let (tmp, dir, cg) = make_test_cgroup(label);
        let target = dir.join("cg_x").join(filename);
        fs::write(&target, "").unwrap();
        (tmp, target, cg)
    }

    #[test]
    fn set_cpu_max_writes_quota_and_period_when_some() {
        let (_tempdir_keep_alive, target, cg) = make_test_cgroup_with_seeded_file("cpu-max-some", "cpu.max");
        cg.set_cpu_max("cg_x", Some(50_000), 100_000).unwrap();
        assert_eq!(fs::read_to_string(&target).unwrap(), "50000 100000");
    }

    #[test]
    fn set_cpu_max_writes_max_keyword_when_none() {
        let (_tempdir_keep_alive, target, cg) = make_test_cgroup_with_seeded_file("cpu-max-none", "cpu.max");
        cg.set_cpu_max("cg_x", None, 100_000).unwrap();
        assert_eq!(fs::read_to_string(&target).unwrap(), "max 100000");
    }

    #[test]
    fn set_cpu_weight_writes_decimal_value() {
        let (_tempdir_keep_alive, target, cg) = make_test_cgroup_with_seeded_file("cpu-weight", "cpu.weight");
        cg.set_cpu_weight("cg_x", 250).unwrap();
        assert_eq!(fs::read_to_string(&target).unwrap(), "250");
    }

    #[test]
    fn set_memory_max_writes_bytes_or_max_keyword() {
        let (_tempdir_keep_alive, target, cg) = make_test_cgroup_with_seeded_file("mem-max", "memory.max");
        cg.set_memory_max("cg_x", Some(1_048_576)).unwrap();
        assert_eq!(fs::read_to_string(&target).unwrap(), "1048576");
        cg.set_memory_max("cg_x", None).unwrap();
        assert_eq!(fs::read_to_string(&target).unwrap(), "max");
    }

    #[test]
    fn set_memory_high_writes_bytes_or_max_keyword() {
        let (_tempdir_keep_alive, target, cg) = make_test_cgroup_with_seeded_file("mem-high", "memory.high");
        cg.set_memory_high("cg_x", Some(524_288)).unwrap();
        assert_eq!(fs::read_to_string(&target).unwrap(), "524288");
        cg.set_memory_high("cg_x", None).unwrap();
        assert_eq!(fs::read_to_string(&target).unwrap(), "max");
    }

    /// `memory.low`'s "no protection" wire value is `"0"`, NOT
    /// `"max"` — the kernel treats `max` as a syntax error on
    /// `memory.low`. Pin both the bytes-set and the cleared paths.
    #[test]
    fn set_memory_low_writes_bytes_or_zero() {
        let (_tempdir_keep_alive, target, cg) = make_test_cgroup_with_seeded_file("mem-low", "memory.low");
        cg.set_memory_low("cg_x", Some(2_048)).unwrap();
        assert_eq!(fs::read_to_string(&target).unwrap(), "2048");
        cg.set_memory_low("cg_x", None).unwrap();
        assert_eq!(fs::read_to_string(&target).unwrap(), "0");
    }

    #[test]
    fn set_io_weight_writes_decimal_value() {
        let (_tempdir_keep_alive, target, cg) = make_test_cgroup_with_seeded_file("io-weight", "io.weight");
        cg.set_io_weight("cg_x", 500).unwrap();
        assert_eq!(fs::read_to_string(&target).unwrap(), "500");
    }

    /// `set_freeze(true)` writes the literal `"1"`; `false` writes
    /// `"0"`. Pinned because the kernel's `cgroup_freeze_write` rejects
    /// any other value with `-ERANGE` — a regression that emits "true"
    /// or "frozen" would surface as a syscall failure on real cgroupfs.
    #[test]
    fn set_freeze_writes_zero_or_one() {
        let (_tempdir_keep_alive, target, cg) = make_test_cgroup_with_seeded_file("freeze", "cgroup.freeze");
        cg.set_freeze("cg_x", true).unwrap();
        assert_eq!(fs::read_to_string(&target).unwrap(), "1");
        cg.set_freeze("cg_x", false).unwrap();
        assert_eq!(fs::read_to_string(&target).unwrap(), "0");
    }

    /// `set_pids_max(Some(n))` writes the decimal `n`;
    /// `set_pids_max(None)` writes `"max"` — the kernel's
    /// `PIDS_MAX_STR` sentinel that selects the unlimited path in
    /// `pids_max_write`.
    #[test]
    fn set_pids_max_writes_decimal_or_max_keyword() {
        let (_tempdir_keep_alive, target, cg) = make_test_cgroup_with_seeded_file("pids-max", "pids.max");
        cg.set_pids_max("cg_x", Some(1024)).unwrap();
        assert_eq!(fs::read_to_string(&target).unwrap(), "1024");
        cg.set_pids_max("cg_x", None).unwrap();
        assert_eq!(fs::read_to_string(&target).unwrap(), "max");
    }

    /// `set_memory_swap_max(Some(b))` writes the decimal byte count;
    /// `None` writes `"max"` — the unlimited sentinel
    /// `page_counter_memparse` recognises in `swap_max_write`.
    #[test]
    fn set_memory_swap_max_writes_bytes_or_max_keyword() {
        let (_tempdir_keep_alive, target, cg) = make_test_cgroup_with_seeded_file("mem-swap-max", "memory.swap.max");
        cg.set_memory_swap_max("cg_x", Some(2 * 1024 * 1024))
            .unwrap();
        assert_eq!(fs::read_to_string(&target).unwrap(), "2097152");
        cg.set_memory_swap_max("cg_x", None).unwrap();
        assert_eq!(fs::read_to_string(&target).unwrap(), "max");
    }

    // -- validate_cgroup_name ----------------------------------------

    /// Reject the path-escape and hidden-entry shapes that
    /// [`validate_cgroup_name`] guards against. Each branch is named
    /// in the assertion so a future regression that drops a check
    /// surfaces with the specific shape that slipped through.
    #[test]
    fn validate_cgroup_name_rejects_unsafe_shapes() {
        for (name, reason) in [
            ("", "empty"),
            ("/abs", "starts with '/'"),
            ("nul\0byte", "NUL byte"),
            (".hidden", "leading-dot component"),
            ("..", "'..' component"),
            ("a/..", "'..' component"),
            ("../escape", "'..' component"),
            (".", "'.' component"),
            ("a//b", "empty path component"),
            ("ok/.dotfile", "leading-dot component"),
        ] {
            let err =
                validate_cgroup_name(name).expect_err(&format!("must reject {name:?} ({reason})"));
            assert!(
                err.to_string().contains(reason),
                "error for {name:?} must mention {reason:?}; got: {err:#}"
            );
        }
    }

    /// Names the validator accepts: simple identifiers, nested paths
    /// with non-leading dots, plain numeric suffixes. Pinned so a
    /// future tightening that breaks legitimate `cg_0/narrow` shapes
    /// is caught at test time.
    #[test]
    fn validate_cgroup_name_accepts_valid_shapes() {
        for name in [
            "cg_0",
            "cg-1",
            "cg.0",
            "cg_0/narrow",
            "level1/level2/level3",
            "a.b.c",
            "x",
        ] {
            validate_cgroup_name(name).unwrap_or_else(|e| {
                panic!("must accept legitimate name {name:?}; got: {e:#}");
            });
        }
    }

    /// Public methods that take a `name` must run name validation
    /// before any filesystem write so a hostile name never reaches
    /// `Path::join`. Pin one representative method per knob type.
    #[test]
    fn cgroup_methods_reject_bad_names_before_fs_writes() {
        let _tempdir_keep_alive = make_inline_tempdir("badname");
        let dir = _tempdir_keep_alive.path();
        let cg = CgroupManager::new(dir.to_str().unwrap());
        let bad = "../escape";
        // Each call must fail at validation, not at the fs write.
        // The shared error fragment ('..' component) appears in
        // every diagnostic so callers see the same shape regardless
        // of which method tripped.
        let err = cg.create_cgroup(bad).unwrap_err();
        assert!(err.to_string().contains("'..' component"));
        let err = cg.set_freeze(bad, true).unwrap_err();
        assert!(err.to_string().contains("'..' component"));
        let err = cg.set_pids_max(bad, Some(10)).unwrap_err();
        assert!(err.to_string().contains("'..' component"));
        let err = cg.set_memory_swap_max(bad, Some(1024)).unwrap_err();
        assert!(err.to_string().contains("'..' component"));
        let err = cg.set_cpuset_mems(bad, &BTreeSet::new()).unwrap_err();
        assert!(err.to_string().contains("'..' component"));
        let err = cg.move_task(bad, 1).unwrap_err();
        assert!(err.to_string().contains("'..' component"));
        let err = cg.drain_tasks(bad).unwrap_err();
        assert!(err.to_string().contains("'..' component"));
        let err = cg.remove_cgroup(bad).unwrap_err();
        assert!(err.to_string().contains("'..' component"));
        // No directory under `dir` should have been created from any
        // of these calls — the validator bails before fs writes.
        let escape_marker = dir.join("escape");
        assert!(
            !escape_marker.exists(),
            "validator must bail before fs writes; saw {escape_marker:?}"
        );
    }

    // -- setup_under_root strip-prefix-fail branch -------------------

    /// When `parent` does not lie under the supplied `root`, the
    /// `strip_prefix` call returns `Err` and `setup_under_root` skips
    /// the subtree-control walk entirely. The function must still
    /// create the parent directory and return Ok — the early-bail
    /// matches the production "non-cgroup-mount" path described on
    /// [`Self::setup_under_root`]. Pin both: the parent dir exists,
    /// no subtree_control was written.
    ///
    /// `outside` uses a raw `std::env::temp_dir().join(...)` path
    /// (not `tempfile::TempDir`) because the production code under
    /// test must create the directory itself via `mkdir` — the
    /// `assert!(outside.exists(), ...)` below verifies that
    /// production behavior. `tempfile::TempDir` would create
    /// `outside` eagerly on construction, making the assertion
    /// vacuously true and defeating the test's intent. `outside`
    /// is cleaned up manually at the end of the test (best-effort;
    /// a panic between construction and cleanup leaks the dir, but
    /// the path is process-id-suffixed so collisions only matter
    /// for the same in-process test re-run).
    ///
    /// `unrelated_root` does NOT have that constraint — the test
    /// pre-creates it manually before the call — so it uses
    /// `tempfile::TempDir` RAII for panic-safe cleanup.
    #[test]
    fn setup_under_root_outside_root_creates_dir_and_skips_walk() {
        let outside = std::env::temp_dir().join(format!("ktstr-out-{}", std::process::id()));
        let _unrelated_root_keep_alive = make_inline_tempdir("unrelated-root");
        let unrelated_root = _unrelated_root_keep_alive.path();
        // `unrelated_root` is created by `tempfile::TempDir` itself;
        // `outside` must NOT exist pre-call — `setup_under_root`
        // creates it as part of its parent-directory step, after
        // which `outside.strip_prefix(unrelated_root)` returns Err
        // and the subtree-control walk is skipped.
        let cg = CgroupManager::new(outside.to_str().unwrap());
        let mut requested = BTreeSet::new();
        requested.insert(Controller::Cpuset);
        cg.setup_under_root(&requested, unrelated_root).unwrap();
        assert!(outside.exists(), "setup must create the parent directory");
        assert!(
            !outside.join("cgroup.subtree_control").exists(),
            "no subtree_control walk should fire when the parent is not under root"
        );
        let _ = fs::remove_dir_all(&outside);
    }

    // -- set_freeze idempotency --------------------------------------

    /// Freezing an already-frozen cgroup must not error — the kernel
    /// short-circuits on the duplicate write. Pinned because the
    /// `remove_cgroup` auto-unfreeze path depends on the inverse
    /// idempotency (unfreezing an unfrozen cgroup), so the symmetric
    /// case is checked here.
    #[test]
    fn set_freeze_is_idempotent_when_already_in_target_state() {
        let (_tempdir_keep_alive, target, cg) = make_test_cgroup_with_seeded_file("freeze-idem", "cgroup.freeze");
        cg.set_freeze("cg_x", true).unwrap();
        assert_eq!(fs::read_to_string(&target).unwrap(), "1");
        // Second freeze: no error, file content unchanged.
        cg.set_freeze("cg_x", true).unwrap();
        assert_eq!(fs::read_to_string(&target).unwrap(), "1");
        // Inverse: idempotent unfreeze on already-unfrozen.
        cg.set_freeze("cg_x", false).unwrap();
        assert_eq!(fs::read_to_string(&target).unwrap(), "0");
        cg.set_freeze("cg_x", false).unwrap();
        assert_eq!(fs::read_to_string(&target).unwrap(), "0");
    }

    // -- pids.max / memory.swap.max overflow boundary ---------------

    /// `set_pids_max(Some(u64::MAX))` writes the decimal representation
    /// verbatim. The kernel rejects values `>= PIDS_MAX` with EINVAL,
    /// but the framework wire layer is responsible only for byte-exact
    /// stringification — pinning u64::MAX guards against accidental
    /// narrowing to i64 (which would turn the value into "-1") or to
    /// u32 (which would silently saturate).
    #[test]
    fn set_pids_max_writes_u64_max_verbatim() {
        let (_tempdir_keep_alive, target, cg) = make_test_cgroup_with_seeded_file("pids-overflow", "pids.max");
        cg.set_pids_max("cg_x", Some(u64::MAX)).unwrap();
        assert_eq!(
            fs::read_to_string(&target).unwrap(),
            u64::MAX.to_string(),
            "u64::MAX must round-trip without narrowing or sign change"
        );
    }

    /// `set_memory_swap_max(Some(u64::MAX))` mirrors the pids.max
    /// boundary check. Catches the same narrowing-regression class.
    #[test]
    fn set_memory_swap_max_writes_u64_max_verbatim() {
        let (_tempdir_keep_alive, target, cg) = make_test_cgroup_with_seeded_file("swap-overflow", "memory.swap.max");
        cg.set_memory_swap_max("cg_x", Some(u64::MAX)).unwrap();
        assert_eq!(
            fs::read_to_string(&target).unwrap(),
            u64::MAX.to_string(),
            "u64::MAX must round-trip without narrowing or sign change"
        );
    }

    // -- write_with_timeout failure paths for new methods ------------
    //
    // [`write_with_timeout`] surfaces the per-method "knob file does
    // not exist" path as an error chain that callers (e.g.
    // `apply_setup`) propagate up. Pin the failure surface for every
    // recently-added method so a regression that swallows the error
    // (or returns Ok despite a missing file) trips here.

    /// `set_pids_max` against a missing parent directory returns an
    /// error whose chain walks back to the missing path. The cgroup
    /// directory has to be missing — `fs::write` to a nonexistent
    /// file inside an existing directory just creates the file —
    /// so the test exercises the realistic "cgroup never created"
    /// path through ENOENT on `parent.join(name).join("pids.max")`.
    #[test]
    fn set_pids_max_returns_err_when_pids_max_file_missing() {
        let cg = CgroupManager::new("/nonexistent/ktstr-pids-test");
        let err = cg
            .set_pids_max("cg_x", Some(1024))
            .expect_err("missing pids.max must surface as Err");
        let msg = format!("{err:#}");
        assert!(
            msg.contains("pids.max"),
            "error chain must name the missing file: {msg}"
        );
    }

    /// Mirror of the pids.max test for memory.swap.max.
    #[test]
    fn set_memory_swap_max_returns_err_when_file_missing() {
        let cg = CgroupManager::new("/nonexistent/ktstr-swap-test");
        let err = cg
            .set_memory_swap_max("cg_x", Some(2_000_000))
            .expect_err("missing memory.swap.max must surface as Err");
        let msg = format!("{err:#}");
        assert!(
            msg.contains("memory.swap.max"),
            "error chain must name the missing file: {msg}"
        );
    }

    /// `set_freeze` against a missing `cgroup.freeze` file surfaces
    /// an ENOENT errno reachable from the error chain — what
    /// `remove_cgroup`'s auto-unfreeze branch uses to suppress the
    /// warn. Pin the errno so a regression that wraps the underlying
    /// IO error in a way that loses the raw_os_error trips here.
    #[test]
    fn set_freeze_returns_err_with_enoent_when_freeze_file_missing() {
        let cg = CgroupManager::new("/nonexistent/ktstr-freeze-test");
        let err = cg
            .set_freeze("cg_x", true)
            .expect_err("missing cgroup.freeze must surface as Err");
        assert_eq!(
            anyhow_first_io_errno(&err),
            Some(libc::ENOENT),
            "ENOENT errno must be reachable from the error chain so \
             remove_cgroup's auto-unfreeze can suppress it; got: {err:#}"
        );
    }

    // -- setter Err-chain context wraps (#128 MEDIUM coverage) ------
    //
    // Each setter's `with_context` closure encodes the cgroup name,
    // the failed value, and a controller hint pointing at the
    // `+<controller>` line the operator needs in
    // `cgroup.subtree_control`. A regression that dropped any of the
    // three components from the wrap (e.g. switched to bare
    // `?`-propagation) would silently degrade the operator
    // diagnostic. One test per setter pins all three.
    //
    // Value assertions use the KNOB-PREFIXED form
    // (e.g. `"set cpu.weight=250"`, `"set io.weight=500"`,
    // `"set cgroup.freeze='1'"`) — bare digit substrings like `"250"`
    // would collide with errno numbers, line refs, or other 3-digit
    // tokens in the chain. The knob-prefixed form mirrors the
    // exact `with_context` format string and is collision-safe.

    #[test]
    fn set_cpu_max_err_contains_cgroup_name_value_and_controller_hint() {
        let cg = CgroupManager::new("/nonexistent/ktstr-set-cpu-max-test");
        let err = cg
            .set_cpu_max("cg_alpha", Some(50_000), 100_000)
            .expect_err("missing cgroup must surface as Err");
        let msg = format!("{err:#}");
        assert!(msg.contains("cg_alpha"), "missing cgroup name: {msg}");
        assert!(
            msg.contains("set cpu.max='50000 100000'"),
            "missing knob-prefixed value: {msg}"
        );
        assert!(msg.contains("+cpu"), "missing controller hint: {msg}");
    }

    #[test]
    fn set_cpu_weight_err_contains_cgroup_name_value_and_controller_hint() {
        let cg = CgroupManager::new("/nonexistent/ktstr-set-cpu-weight-test");
        let err = cg
            .set_cpu_weight("cg_beta", 250)
            .expect_err("missing cgroup must surface as Err");
        let msg = format!("{err:#}");
        assert!(msg.contains("cg_beta"), "missing cgroup name: {msg}");
        assert!(
            msg.contains("set cpu.weight=250"),
            "missing knob-prefixed value: {msg}"
        );
        assert!(msg.contains("+cpu"), "missing controller hint: {msg}");
    }

    #[test]
    fn set_memory_max_err_contains_cgroup_name_value_and_controller_hint() {
        let cg = CgroupManager::new("/nonexistent/ktstr-set-mem-max-test");
        let err = cg
            .set_memory_max("cg_gamma", Some(1_048_576))
            .expect_err("missing cgroup must surface as Err");
        let msg = format!("{err:#}");
        assert!(msg.contains("cg_gamma"), "missing cgroup name: {msg}");
        assert!(
            msg.contains("set memory.max='1048576'"),
            "missing knob-prefixed value: {msg}"
        );
        assert!(msg.contains("+memory"), "missing controller hint: {msg}");
    }

    #[test]
    fn set_memory_high_err_contains_cgroup_name_value_and_controller_hint() {
        let cg = CgroupManager::new("/nonexistent/ktstr-set-mem-high-test");
        let err = cg
            .set_memory_high("cg_delta", Some(524_288))
            .expect_err("missing cgroup must surface as Err");
        let msg = format!("{err:#}");
        assert!(msg.contains("cg_delta"), "missing cgroup name: {msg}");
        assert!(
            msg.contains("set memory.high='524288'"),
            "missing knob-prefixed value: {msg}"
        );
        assert!(msg.contains("+memory"), "missing controller hint: {msg}");
    }

    #[test]
    fn set_memory_low_err_contains_cgroup_name_value_and_controller_hint() {
        let cg = CgroupManager::new("/nonexistent/ktstr-set-mem-low-test");
        let err = cg
            .set_memory_low("cg_epsilon", Some(262_144))
            .expect_err("missing cgroup must surface as Err");
        let msg = format!("{err:#}");
        assert!(msg.contains("cg_epsilon"), "missing cgroup name: {msg}");
        assert!(
            msg.contains("set memory.low='262144'"),
            "missing knob-prefixed value: {msg}"
        );
        assert!(msg.contains("+memory"), "missing controller hint: {msg}");
    }

    #[test]
    fn set_io_weight_err_contains_cgroup_name_value_and_controller_hint() {
        let cg = CgroupManager::new("/nonexistent/ktstr-set-io-weight-test");
        let err = cg
            .set_io_weight("cg_zeta", 500)
            .expect_err("missing cgroup must surface as Err");
        let msg = format!("{err:#}");
        assert!(msg.contains("cg_zeta"), "missing cgroup name: {msg}");
        assert!(
            msg.contains("set io.weight=500"),
            "missing knob-prefixed value: {msg}"
        );
        assert!(msg.contains("+io"), "missing controller hint: {msg}");
    }

    #[test]
    fn set_freeze_err_contains_cgroup_name_value_and_core_hint() {
        let cg = CgroupManager::new("/nonexistent/ktstr-set-freeze-test");
        let err = cg
            .set_freeze("cg_eta", true)
            .expect_err("missing cgroup must surface as Err");
        let msg = format!("{err:#}");
        assert!(msg.contains("cg_eta"), "missing cgroup name: {msg}");
        assert!(
            msg.contains("set cgroup.freeze='1'"),
            "missing knob-prefixed value: {msg}"
        );
        // cgroup.freeze is a cgroup-core file (no `+freeze`
        // controller exists); the wrap calls this out explicitly so
        // operators don't go hunting subtree_control.
        assert!(
            msg.contains("cgroup-core"),
            "missing cgroup-core hint: {msg}"
        );
    }

    #[test]
    fn set_pids_max_err_contains_cgroup_name_value_and_controller_hint() {
        let cg = CgroupManager::new("/nonexistent/ktstr-set-pids-max-test");
        let err = cg
            .set_pids_max("cg_theta", Some(4096))
            .expect_err("missing cgroup must surface as Err");
        let msg = format!("{err:#}");
        assert!(msg.contains("cg_theta"), "missing cgroup name: {msg}");
        assert!(
            msg.contains("set pids.max='4096'"),
            "missing knob-prefixed value: {msg}"
        );
        assert!(msg.contains("+pids"), "missing controller hint: {msg}");
    }

    #[test]
    fn set_memory_swap_max_err_contains_cgroup_name_value_and_controller_hint() {
        let cg = CgroupManager::new("/nonexistent/ktstr-set-swap-max-test");
        let err = cg
            .set_memory_swap_max("cg_iota", Some(8_388_608))
            .expect_err("missing cgroup must surface as Err");
        let msg = format!("{err:#}");
        assert!(msg.contains("cg_iota"), "missing cgroup name: {msg}");
        assert!(
            msg.contains("set memory.swap.max='8388608'"),
            "missing knob-prefixed value: {msg}"
        );
        assert!(msg.contains("+memory"), "missing controller hint: {msg}");
    }

    // -- remove_cgroup auto-unfreeze --------------------------------

    /// `remove_cgroup` writes `0` to `cgroup.freeze` before draining
    /// tasks — pin the side effect so a regression that drops the
    /// auto-unfreeze surfaces here. The test observes the freeze
    /// file's post-call contents instead of asserting on the rmdir
    /// outcome: real cgroupfs auto-removes child files during a
    /// directory rmdir, but tmpfs requires the files to be unlinked
    /// first. The unfreeze-before-drain ordering is the invariant
    /// under test, not the rmdir success.
    #[test]
    fn remove_cgroup_auto_unfreezes_before_drain() {
        let (_tempdir_keep_alive, inner, cg) = make_test_cgroup_with_procs("autounf");
        // Seed `cgroup.freeze` with "1" so the test can observe
        // the unfreeze write.
        let freeze_path = inner.join("cgroup.freeze");
        fs::write(&freeze_path, "1").unwrap();
        // The rmdir at the end fails on tmpfs because cgroup.freeze
        // / cgroup.procs are leftover non-cgroupfs files; we don't
        // care — the assertion is on the freeze-file content.
        let _ = cg.remove_cgroup("cg_x");
        // The auto-unfreeze must have written "0" to cgroup.freeze
        // before the drain.
        assert_eq!(
            fs::read_to_string(&freeze_path).unwrap(),
            "0",
            "remove_cgroup must write '0' to cgroup.freeze before draining"
        );
    }

    /// `remove_cgroup` swallows ENOENT on the unfreeze write so a
    /// cgroup without `cgroup.freeze` (legacy kernel without
    /// CONFIG_CGROUP_FREEZE) still drains cleanly. The drain reaches
    /// `cgroup.procs` instead of returning early because of the
    /// missing freeze file.
    #[test]
    fn remove_cgroup_tolerates_missing_freeze_file() {
        // Helper pre-creates cgroup.procs (drain_tasks needs it).
        // Deliberately omit cgroup.freeze.
        let (_tempdir_keep_alive, _inner, cg) = make_test_cgroup_with_procs("nofrz");
        // The rmdir at the end fails on tmpfs (cgroup.procs left
        // over) — we only care that no error propagates from the
        // pre-drain unfreeze branch. The test would catch a
        // regression where the missing freeze file produces a hard
        // error before the drain runs.
        let _ = cg.remove_cgroup("cg_x");
        // No assertion on the freeze file — it never existed. The
        // test passes when the call body runs to completion without
        // panicking on the tolerated ENOENT branch.
    }

    /// `cleanup_recursive` writes `0` to `cgroup.freeze` before
    /// draining tasks — mirrors
    /// [`remove_cgroup_auto_unfreezes_before_drain`]. The pre-drain
    /// unfreeze is a state-hygiene step (the kernel would unfreeze
    /// migrated tasks at the unfrozen root anyway), but the write
    /// itself is observable in `cgroup.events` and tracing, so a
    /// regression that drops the unfreeze step would silently lose
    /// that visibility. Pin the side effect by seeding
    /// `cgroup.freeze="1"` and asserting the post-call contents
    /// are `"0"`. Test mirrors the `remove_cgroup` shape: rmdir at
    /// the end fails on tmpfs (leftover non-cgroupfs files), but
    /// the freeze-file content is what the assertion targets.
    #[test]
    fn cleanup_recursive_auto_unfreezes_before_drain() {
        let _tempdir_keep_alive = make_inline_tempdir("cleanup-rec-autounf");
        let dir = _tempdir_keep_alive.path();
        // Seed cgroup.freeze="1" + empty cgroup.procs so the test
        // observes the unfreeze write the same way
        // remove_cgroup_auto_unfreezes_before_drain does.
        let freeze_path = dir.join("cgroup.freeze");
        fs::write(&freeze_path, "1").unwrap();
        fs::write(dir.join("cgroup.procs"), "").unwrap();
        // `cleanup_recursive` is the free fn the cleanup_all walk
        // dispatches per directory; call it directly.
        cleanup_recursive(dir);
        // The auto-unfreeze must have written "0" to cgroup.freeze
        // before the drain — pinned identically to the
        // remove_cgroup test so a regression on either path
        // surfaces with the same diagnostic shape.
        assert_eq!(
            fs::read_to_string(&freeze_path).unwrap(),
            "0",
            "cleanup_recursive must write '0' to cgroup.freeze before draining \
             (mirrors remove_cgroup auto-unfreeze for state hygiene)",
        );
    }

    // -- outstanding-removes cap ------------------------------------

    /// `remove_cgroup` increments `outstanding_removes` on every
    /// failure. Drives a non-existent parent path so the underlying
    /// `set_freeze` / `drain_tasks` / `rmdir` chain reaches the
    /// rmdir step and fails (the parent directory is created by the
    /// test, but the child does not exist on every iteration after
    /// the first because nothing creates it). Verifies the counter
    /// rises monotonically as failures accumulate.
    #[test]
    fn remove_cgroup_increments_outstanding_on_failure() {
        // Helper pre-creates cgroup.procs (drain_tasks needs it;
        // rmdir then fails on tmpfs because cgroup.procs is a
        // regular file tmpfs won't auto-unlink).
        let (_tempdir_keep_alive, inner, cg) = make_test_cgroup_with_procs("outstanding");
        assert_eq!(cg.outstanding_removes(), 0);
        // First call: rmdir fails with ENOTEMPTY → counter goes to 1.
        let _ = cg.remove_cgroup("cg_x");
        assert_eq!(
            cg.outstanding_removes(),
            1,
            "outstanding_removes must increment when rmdir fails"
        );
        // Re-create the seed state (drain_tasks may have left the dir
        // intact since rmdir failed) so the next remove can fail again.
        fs::create_dir_all(&inner).unwrap();
        fs::write(inner.join("cgroup.procs"), "").unwrap();
        let _ = cg.remove_cgroup("cg_x");
        assert_eq!(
            cg.outstanding_removes(),
            2,
            "outstanding_removes must increment monotonically on repeat failures"
        );
    }

    /// `remove_cgroup` decrements `outstanding_removes` on success
    /// (saturating at 0 so a remove of a never-failed cgroup does
    /// not underflow into usize::MAX). Drives a successful remove
    /// path through a fully-clean tmpdir (no leftover non-cgroupfs
    /// files) and verifies the counter drops back from a seeded
    /// non-zero state.
    #[test]
    fn remove_cgroup_decrements_outstanding_on_success() {
        let _tempdir_keep_alive = make_inline_tempdir("decrement");
        let dir = _tempdir_keep_alive.path();
        let cg = CgroupManager::new(dir.to_str().unwrap());
        // Seed the counter to simulate a prior failure.
        cg.outstanding_removes.store(3, Ordering::Relaxed);
        // Create + remove a cgroup that has nothing inside but the
        // directory itself, so rmdir succeeds on tmpfs.
        let inner = dir.join("cg_clean");
        fs::create_dir_all(&inner).unwrap();
        cg.remove_cgroup("cg_clean").unwrap();
        assert_eq!(
            cg.outstanding_removes(),
            2,
            "successful remove must decrement outstanding_removes by 1"
        );
    }

    /// Once `outstanding_removes` exceeds [`MAX_OUTSTANDING_REMOVES`],
    /// further [`CgroupManager::remove_cgroup`] calls bail without
    /// touching the filesystem. Pin the message shape so a regression
    /// that silences the bail or rephrases the diagnostic surfaces
    /// here.
    #[test]
    fn remove_cgroup_bails_when_cap_exceeded() {
        let _tempdir_keep_alive = make_inline_tempdir("cap");
        let dir = _tempdir_keep_alive.path();
        let inner = dir.join("cg_x");
        fs::create_dir_all(&inner).unwrap();
        let cg = CgroupManager::new(dir.to_str().unwrap());
        cg.outstanding_removes
            .store(MAX_OUTSTANDING_REMOVES + 1, Ordering::Relaxed);
        let err = cg
            .remove_cgroup("cg_x")
            .expect_err("cap-exceeded remove must surface as Err");
        let msg = format!("{err:#}");
        assert!(
            msg.contains("outstanding") && msg.contains("cap"),
            "error must cite the cap; got: {msg}"
        );
        // The cgroup directory must still exist — the bail short-
        // circuited before any rmdir attempt.
        assert!(
            inner.exists(),
            "cap-exceeded bail must not touch the filesystem"
        );
    }

    /// `remove_cgroup` on a missing directory returns Ok and does
    /// NOT touch the outstanding counter — the cgroup was already
    /// reaped (e.g. by an earlier remove or `cleanup_all`), so it
    /// is not "outstanding" and should not skew the budget.
    #[test]
    fn remove_cgroup_missing_dir_does_not_touch_counter() {
        let cg = CgroupManager::new("/nonexistent/ktstr-missing-counter");
        cg.outstanding_removes.store(5, Ordering::Relaxed);
        cg.remove_cgroup("no_such_cgroup").unwrap();
        assert_eq!(
            cg.outstanding_removes(),
            5,
            "missing-dir early return must not decrement the counter"
        );
    }

    // -- move_task cpuset ordering gate -----------------------------

    /// [`CgroupManager::move_task`] refuses migrations into a cgroup
    /// whose `cpuset.cpus` is set but `cpuset.mems.effective` reads
    /// empty — a half-configured shape the framework surfaces as a
    /// focused error rather than letting through to the kernel's
    /// path-dependent behavior (parent-walk fallback in
    /// `guarantee_online_mems`, or OOM in degenerate hierarchies).
    /// The gate keys on `cpuset.mems.effective` (the kernel's
    /// inheritance-aware view) rather than the local `cpuset.mems`
    /// because in cgroup v2 the local file is normally empty even
    /// when the cgroup inherits a valid `effective_mems` from its
    /// parent. Pin the runtime gate so a regression that drops the
    /// readback surfaces here.
    #[test]
    fn move_task_refuses_when_cpuset_cpus_set_but_effective_mems_empty() {
        let _tempdir_keep_alive = make_inline_tempdir("cpuset-gate");
        let dir = _tempdir_keep_alive.path();
        let inner = dir.join("cg_x");
        fs::create_dir_all(&inner).unwrap();
        // Configured: cpuset.cpus has bits; cpuset.mems.effective
        // empty (no inherited nodemask either).
        fs::write(inner.join("cpuset.cpus"), "0-1").unwrap();
        fs::write(inner.join("cpuset.mems.effective"), "").unwrap();
        let cg = CgroupManager::new(dir.to_str().unwrap());
        let err = cg
            .move_task("cg_x", 1)
            .expect_err("half-configured cpuset must refuse move_task");
        let msg = format!("{err:#}");
        assert!(
            msg.contains("cpuset.mems.effective") && msg.contains("set_cpuset_mems"),
            "error must cite cpuset.mems.effective and direct caller to set_cpuset_mems; got: {msg}"
        );
        // No write to cgroup.procs should have landed.
        let procs_path = inner.join("cgroup.procs");
        assert!(
            !procs_path.exists(),
            "gate must bail before any cgroup.procs write; cgroup.procs exists at {procs_path:?}"
        );
    }

    /// [`CgroupManager::move_task`] admits migrations when
    /// `cpuset.cpus` is set locally and `cpuset.mems.effective` is
    /// non-empty (whether populated by a local `cpuset.mems` write
    /// or by parent inheritance). Test observes that the gate
    /// accepts the call by completing the underlying tmpfs write to
    /// `cgroup.procs` without bailing.
    #[test]
    fn move_task_admits_when_cpus_set_and_effective_mems_non_empty() {
        // Helper pre-creates cgroup.procs so the underlying
        // tmpfs write succeeds (not a real cgroupfs, so the write
        // just appends to a regular file — the assertion is that
        // the gate did NOT bail).
        let (_tempdir_keep_alive, inner, cg) = make_test_cgroup_with_procs("cpuset-ok");
        fs::write(inner.join("cpuset.cpus"), "0-1").unwrap();
        fs::write(inner.join("cpuset.mems.effective"), "0").unwrap();
        cg.move_task("cg_x", 1)
            .expect("non-empty effective mems must admit move_task");
    }

    /// [`CgroupManager::move_task`] admits migrations when the local
    /// `cpuset.mems` is empty (the v2 default for an inheriting
    /// child) but `cpuset.mems.effective` is non-empty because the
    /// parent supplies the nodemask. This is the canonical case the
    /// previous `cpuset.mems`-based gate would have wrongly refused;
    /// pin the inheritance-aware path so a regression that reverts
    /// to reading the local file fails this test.
    #[test]
    fn move_task_admits_when_local_mems_empty_but_effective_inherited() {
        // Helper pre-creates cgroup.procs.
        let (_tempdir_keep_alive, inner, cg) = make_test_cgroup_with_procs("cpuset-inherit-mems");
        fs::write(inner.join("cpuset.cpus"), "0-1").unwrap();
        // Local cpuset.mems empty: the v2 default for an inheriting
        // child. The kernel's effective view shows the inherited
        // nodemask "0" — the gate must use the effective view.
        fs::write(inner.join("cpuset.mems"), "").unwrap();
        fs::write(inner.join("cpuset.mems.effective"), "0").unwrap();
        cg.move_task("cg_x", 1)
            .expect("inherited effective mems must admit move_task");
    }

    /// [`CgroupManager::move_task`] admits migrations when
    /// `cpuset.cpus` is empty (cgroup inherits parent's cpuset —
    /// no local constraint, so the `cpuset.mems.effective`
    /// invariant doesn't apply). The kernel allows attach into
    /// such a cgroup without a local cpus mask.
    #[test]
    fn move_task_admits_when_cpuset_cpus_empty() {
        // Helper pre-creates cgroup.procs.
        let (_tempdir_keep_alive, inner, cg) = make_test_cgroup_with_procs("cpuset-inherit");
        fs::write(inner.join("cpuset.cpus"), "").unwrap();
        // Whether cpuset.mems.effective is set or not is irrelevant
        // when local cpus is empty — the gate short-circuits before
        // reading the effective file.
        fs::write(inner.join("cpuset.mems.effective"), "").unwrap();
        cg.move_task("cg_x", 1)
            .expect("inherit-cpuset cgroup must admit move_task");
    }

    /// [`CgroupManager::move_task`] admits migrations when
    /// `cpuset.cpus` does not exist at all (cgroup has no cpuset
    /// controller — the gate has nothing to enforce). Models a
    /// cgroup tree without `+cpuset` in `subtree_control`.
    #[test]
    fn move_task_admits_when_cpuset_files_absent() {
        // Deliberately omit cpuset.cpus / cpuset.mems.effective.
        // Helper pre-creates cgroup.procs.
        let (_tempdir_keep_alive, _inner, cg) = make_test_cgroup_with_procs("no-cpuset");
        cg.move_task("cg_x", 1)
            .expect("no-cpuset cgroup must admit move_task");
    }

    /// [`CgroupManager::move_task`] admits migrations when
    /// `cpuset.cpus` is set but `cpuset.mems.effective` cannot be
    /// read — the gate absorbs read failures on either knob and
    /// degrades to "accept" rather than blocking on a transient
    /// cgroupfs read error. Matches the general "read failures are
    /// absorbed" contract documented on
    /// [`CgroupManager::move_task`].
    #[test]
    fn move_task_admits_when_effective_mems_file_absent() {
        // Helper pre-creates cgroup.procs.
        let (_tempdir_keep_alive, inner, cg) = make_test_cgroup_with_procs("no-effective-mems");
        fs::write(inner.join("cpuset.cpus"), "0-1").unwrap();
        // Deliberately omit cpuset.mems.effective: read fails,
        // gate degrades to accept.
        cg.move_task("cg_x", 1)
            .expect("missing cpuset.mems.effective must admit move_task (read-failure absorb)");
    }
