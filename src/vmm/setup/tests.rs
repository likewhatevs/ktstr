use super::*;

/// Regression for the PVTIME/initrd overlap: the initrd top
/// must stay below pvtime_base, never entering the steal-time carve
/// `[pvtime_base, fdt_addr)`. Otherwise the host clobbers the initramfs
/// before the guest unpacks it — kvm_update_stolen_time writes the
/// 8-byte stolen_time field (steal_base+8) from check_vcpu_requests on
/// the FIRST KVM_RUN, before guest code executes — and, independently,
/// the carve is outside advertised /memory so the guest never reserves
/// those pages; either way /init never starts. The earlier PVTIME tests only
/// checked the /memory math, never the initrd-vs-carve relationship.
#[cfg(target_arch = "aarch64")]
#[test]
fn aarch64_initrd_stays_below_pvtime_carve() {
    use crate::vmm::{aarch64::fdt::pvtime_base, kvm::DRAM_START};
    for &(mem, cpus) in &[(512u32, 2u32), (512, 8), (2048, 256), (4096, 512)] {
        let pvt = pvtime_base(mem, cpus);
        // A near-max initrd that still fits below the carve.
        let max = pvt - DRAM_START - (1 << 20);
        let load = aarch64_initrd_addr(mem, cpus, max).expect("near-max initrd must fit");
        assert!(
            load >= DRAM_START,
            "initrd underflows DRAM_START (mem={mem} cpus={cpus} load={load:#x})"
        );
        assert!(
            load + max <= pvt,
            "initrd top {:#x} entered the PVTIME carve at {pvt:#x} (mem={mem} cpus={cpus})",
            load + max
        );
    }
}

/// An initramfs whose compressed size exceeds the advertised RAM span
/// `[DRAM_START, pvtime_base)` must produce a clean `Err`, never a
/// panic and never a wrapped (near-`u64::MAX`) load address that would
/// silently pass the `>= DRAM_START` check. The initrd must reside
/// entirely within advertised RAM (above pvtime_base it is outside the
/// advertised /memory and the guest never memblock-reserves it);
/// firecracker (`initrd_load_addr`) and cloud-hypervisor
/// (`initramfs_load_addr`) likewise return an error/None in this case
/// instead of panicking or returning a bogus address. The min-memory
/// budget sizes RAM for the tmpfs/init constraint, not for
/// 'initrd fits below the PVTIME carve', so this bound is reachable on
/// a payload-controlled (large) initramfs.
#[cfg(target_arch = "aarch64")]
#[test]
fn aarch64_initrd_oversized_returns_err_not_panic_or_wrap() {
    use crate::vmm::{aarch64::fdt::pvtime_base, kvm::DRAM_START};
    for &(mem, cpus) in &[(512u32, 2u32), (512, 8), (2048, 256), (4096, 512)] {
        let pvt = pvtime_base(mem, cpus);
        // One MiB larger than the ENTIRE advertised span below the
        // carve: ceiling.checked_sub(oversized) cannot land at or above
        // DRAM_START, so the function must report an error rather than
        // wrap. On the pre-fix code this input either panics (debug,
        // unchecked sub) or wraps to a huge value that passes the
        // assert (release) — both wrong.
        let oversized = (pvt - DRAM_START) + (1 << 20);
        let result = aarch64_initrd_addr(mem, cpus, oversized);
        assert!(
            result.is_err(),
            "oversized initrd must Err (mem={mem} cpus={cpus} \
             oversized={oversized:#x} pvt={pvt:#x}), got {result:?}"
        );
    }
    // Total underflow: an initrd larger than pvtime_base itself drives
    // the raw subtraction past zero. checked_sub must catch it as Err,
    // never panic, never wrap.
    let (mem, cpus) = (512u32, 2u32);
    let pvt = pvtime_base(mem, cpus);
    let huge = pvt + (1 << 20);
    assert!(
        aarch64_initrd_addr(mem, cpus, huge).is_err(),
        "initrd larger than pvtime_base must Err (huge={huge:#x} pvt={pvt:#x})"
    );
}

/// `Filesystem::Raw` disks emit no auto-mount cmdline tokens.
/// The host has nothing to advertise: no on-disk fs to mount,
/// the guest sees an unformatted `/dev/vda` and the
/// `auto_mount_data_disks` short-circuits at the absent
/// `KTSTR_DISK0_FS` check. Pin the empty-string contract so a
/// future regression that emits Raw-disk tokens (e.g. for a
/// "mount as raw block device" feature) surfaces here loudly.
#[test]
fn disk_auto_mount_cmdline_tokens_raw_emits_nothing() {
    let disk = disk_config::DiskConfig::default();
    assert_eq!(disk.filesystem, disk_config::Filesystem::Raw);
    assert_eq!(disk_auto_mount_cmdline_tokens(&disk), "");
}

/// `Filesystem::Btrfs` with no name and no read_only emits the
/// FS + MOUNT pair only — no RO token. Default mount path is
/// `/mnt/disk0` (driven by `auto_mount_path()` returning the
/// disk0 fallback when `name` is `None`). The leading space
/// is the cmdline-concatenation contract: callers paste the
/// returned string directly.
#[test]
fn disk_auto_mount_cmdline_tokens_btrfs_default() {
    let disk = disk_config::DiskConfig::default().filesystem(disk_config::Filesystem::Btrfs);
    assert_eq!(
        disk_auto_mount_cmdline_tokens(&disk),
        " KTSTR_DISK0_FS=btrfs KTSTR_DISK0_MOUNT=/mnt/disk0",
    );
}

/// Named `Filesystem::Btrfs` disk emits the name-driven mount
/// path `/mnt/<name>` instead of `/mnt/disk0`. Pin the name
/// → mount-path translation so a future `auto_mount_path`
/// regression (e.g. dropping the name and reverting to fixed
/// /mnt/disk0) surfaces here.
#[test]
fn disk_auto_mount_cmdline_tokens_btrfs_named() {
    let disk = disk_config::DiskConfig::default()
        .filesystem(disk_config::Filesystem::Btrfs)
        .with_name("data");
    assert_eq!(
        disk_auto_mount_cmdline_tokens(&disk),
        " KTSTR_DISK0_FS=btrfs KTSTR_DISK0_MOUNT=/mnt/data",
    );
}

/// Read-only Btrfs disk emits the RO token in addition to FS
/// + MOUNT. The guest's `auto_mount_data_disks` checks
///   `KTSTR_DISK0_RO == "1"` and sets `MS_RDONLY` to avoid the
///   kernel-side -EROFS path on RW mount of a F_RO bdev.
#[test]
fn disk_auto_mount_cmdline_tokens_btrfs_read_only() {
    let disk = disk_config::DiskConfig::default()
        .filesystem(disk_config::Filesystem::Btrfs)
        .read_only();
    assert_eq!(
        disk_auto_mount_cmdline_tokens(&disk),
        " KTSTR_DISK0_FS=btrfs KTSTR_DISK0_MOUNT=/mnt/disk0 KTSTR_DISK0_RO=1",
    );
}

/// `no_auto_mount` opt-out suppresses every auto-mount token,
/// even for a Btrfs disk that would otherwise emit them. The
/// host-side mkfs still happens (Filesystem::Btrfs drives the
/// template-cache lifecycle); only the guest auto-mount is
/// skipped, leaving raw `/dev/vda` access to the test author.
#[test]
fn disk_auto_mount_cmdline_tokens_no_auto_mount_suppresses() {
    let disk = disk_config::DiskConfig::default()
        .filesystem(disk_config::Filesystem::Btrfs)
        .no_auto_mount();
    assert_eq!(disk_auto_mount_cmdline_tokens(&disk), "");

    // RO + named + no_auto_mount: still empty. The opt-out
    // dominates every other config dimension.
    let disk = disk_config::DiskConfig::default()
        .filesystem(disk_config::Filesystem::Btrfs)
        .with_name("data")
        .read_only()
        .no_auto_mount();
    assert_eq!(disk_auto_mount_cmdline_tokens(&disk), "");
}

/// Raw disk + no_auto_mount: still empty. The Raw branch is
/// the gate; no_auto_mount is only meaningful for non-Raw
/// filesystems but the function tolerates the redundant
/// combination.
#[test]
fn disk_auto_mount_cmdline_tokens_raw_with_no_auto_mount() {
    let disk = disk_config::DiskConfig::default().no_auto_mount();
    assert_eq!(disk.filesystem, disk_config::Filesystem::Raw);
    assert_eq!(disk_auto_mount_cmdline_tokens(&disk), "");
}

/// Pin the leading-space cmdline-concatenation contract. The
/// returned tokens MUST start with a space when non-empty so
/// they can be appended directly to the cmdline buffer in
/// `setup_memory`. A regression that drops the leading space
/// would create a glued-together token like
/// `virtio_mmio.device=...KTSTR_DISK0_FS=btrfs` which the
/// kernel cmdline parser would mis-classify as a single token.
#[test]
fn disk_auto_mount_cmdline_tokens_starts_with_space() {
    let disk = disk_config::DiskConfig::default().filesystem(disk_config::Filesystem::Btrfs);
    let s = disk_auto_mount_cmdline_tokens(&disk);
    assert!(
        s.starts_with(' '),
        "non-empty tokens must start with a space for safe \
         cmdline concatenation; got {s:?}",
    );
}

/// Helper: build a temp dir with a payload binary + N staged-
/// scheduler binaries. Returns the tempdir guard (keep alive)
/// plus the payload path and a Vec<StagedScheduler> the test
/// can feed to `assemble_extras_and_key`.
fn build_synthetic_staged_set(
    names: &[&str],
) -> (
    tempfile::TempDir,
    PathBuf,
    Vec<crate::vmm::builder::StagedScheduler>,
) {
    let dir = tempfile::Builder::new()
        .prefix("ktstr-assemble-test-")
        .tempdir()
        .unwrap();
    let payload = dir.path().join("payload");
    std::fs::write(&payload, b"payload-content").unwrap();
    let staged: Vec<crate::vmm::builder::StagedScheduler> = names
        .iter()
        .map(|name| {
            let bin = dir.path().join(format!("staged_bin_{name}"));
            std::fs::write(&bin, format!("staged-content-{name}").as_bytes()).unwrap();
            crate::vmm::builder::StagedScheduler {
                name: (*name).to_string(),
                binary: bin,
                sched_args: vec![format!("--variant={name}")],
            }
        })
        .collect();
    (dir, payload, staged)
}

/// Helper: pre-compute staged_extras_names the same way
/// spawn_initramfs_resolve does.
fn staged_extras_names_for(staged: &[crate::vmm::builder::StagedScheduler]) -> Vec<String> {
    staged
        .iter()
        .map(|s| {
            format!(
                "{}/scheduler",
                crate::test_support::staged::staged_scheduler_archive_dir(&s.name),
            )
        })
        .collect()
}

/// Each staged scheduler must land in `extras` under the
/// canonical `staging/schedulers/<name>/scheduler` archive path.
/// Pins the wire-up against a refactor that synthesizes the
/// archive path inline without going through
/// `staged_scheduler_archive_dir` — a drift would silently
/// desynchronize from the runtime resolver path.
#[test]
fn assemble_extras_and_key_emits_staged_binary_under_correct_archive_path() {
    let (_tmp, payload, staged) = build_synthetic_staged_set(&["scx_foo", "scx_bar"]);
    let names = staged_extras_names_for(&staged);
    let (extras, _key) = assemble_extras_and_key(
        payload.as_path(),
        None,
        None,
        None,
        &staged,
        &names,
        &[],
        None,
        false,
    )
    .unwrap();
    let extras_names: Vec<&str> = extras.iter().map(|(n, _)| *n).collect();
    assert!(
        extras_names.contains(&"staging/schedulers/scx_foo/scheduler"),
        "missing scx_foo at canonical archive path; got {extras_names:?}",
    );
    assert!(
        extras_names.contains(&"staging/schedulers/scx_bar/scheduler"),
        "missing scx_bar at canonical archive path; got {extras_names:?}",
    );
}

/// staged_schedulers iteration order must align with the
/// extras-push order so `staged_extras_names[idx]` matches
/// `staged_schedulers[idx].binary`. Misalignment would silently
/// point name A at binary B's content — disastrous regression
/// where tests boot with wrong scheduler binaries under
/// correct-looking names.
#[test]
fn assemble_extras_and_key_preserves_staged_iteration_order_in_extras() {
    let (_tmp, payload, staged) = build_synthetic_staged_set(&["alpha", "beta", "gamma"]);
    let names = staged_extras_names_for(&staged);
    let (extras, _key) = assemble_extras_and_key(
        payload.as_path(),
        None,
        None,
        None,
        &staged,
        &names,
        &[],
        None,
        false,
    )
    .unwrap();
    // Staged entries start after any of scheduler/probe (none
    // here), so they occupy extras[0..3].
    for (i, name) in ["alpha", "beta", "gamma"].iter().enumerate() {
        let (entry_name, entry_path) = extras[i];
        let expected_name = format!("staging/schedulers/{name}/scheduler");
        assert_eq!(
            entry_name, expected_name,
            "extras[{i}] expected name '{expected_name}', got '{entry_name}'",
        );
        // The binary file is named staged_bin_<name> in the
        // helper; verify the extras entry points at the matching
        // binary path (binary owns the content for that name).
        assert!(
            entry_path
                .to_string_lossy()
                .ends_with(&format!("staged_bin_{name}")),
            "extras[{i}] binary path '{}' does not match expected staged_bin_{name}",
            entry_path.display(),
        );
    }
}

/// Staged binaries must contribute to BaseKey in BOTH
/// shell-mode and non-shell-mode dispatch arms. A regression
/// dropping staged_for_key from one arm would silently un-
/// invalidate the cache for that mode, contaminating tests
/// across staged-set differences. Compares each mode's
/// "with-staged" key against an "empty-staged" baseline to
/// confirm the staged inputs participate in the digest.
#[test]
fn assemble_extras_and_key_threads_staged_into_basekey_in_both_modes() {
    let (_tmp, payload, staged) = build_synthetic_staged_set(&["mitosis_a"]);
    let names = staged_extras_names_for(&staged);
    let empty: Vec<crate::vmm::builder::StagedScheduler> = vec![];
    let empty_names: Vec<String> = vec![];

    // Non-shell-mode arm (no busybox, no includes, no
    // jemalloc extras).
    let (_, key_with_staged_nonshell) = assemble_extras_and_key(
        payload.as_path(),
        None,
        None,
        None,
        &staged,
        &names,
        &[],
        None,
        false,
    )
    .unwrap();
    let (_, key_empty_nonshell) = assemble_extras_and_key(
        payload.as_path(),
        None,
        None,
        None,
        &empty,
        &empty_names,
        &[],
        None,
        false,
    )
    .unwrap();
    assert_ne!(
        key_with_staged_nonshell, key_empty_nonshell,
        "non-shell-mode BaseKey must reflect staged contribution",
    );

    // Shell-mode arm (Some(bytes) forces shell mode without
    // requiring any include_files / jemalloc extras).
    let stub_busybox: &[u8] = b"#!/bin/sh\n";
    let (_, key_with_staged_shell) = assemble_extras_and_key(
        payload.as_path(),
        None,
        None,
        None,
        &staged,
        &names,
        &[],
        Some(stub_busybox),
        false,
    )
    .unwrap();
    let (_, key_empty_shell) = assemble_extras_and_key(
        payload.as_path(),
        None,
        None,
        None,
        &empty,
        &empty_names,
        &[],
        Some(stub_busybox),
        false,
    )
    .unwrap();
    assert_ne!(
        key_with_staged_shell, key_empty_shell,
        "shell-mode BaseKey must reflect staged contribution",
    );

    // Belt-and-suspenders: shell-mode and non-shell-mode keys
    // for the SAME staged set must differ (shell-mode keys mix
    // a "ktstr-shell" sentinel — verify the shell-mode arm
    // didn't accidentally call BaseKey::new).
    assert_ne!(
        key_with_staged_nonshell, key_with_staged_shell,
        "shell-mode and non-shell-mode keys for same staged set \
         must differ — confirms each arm calls its respective \
         BaseKey constructor",
    );
}

/// Drive the REAL `try_cow_overlay` against a live LZ4 SHM segment and a
/// two-region `GuestMemoryMmap`. The overlay must succeed (return
/// `Some`), map the SHM bytes into region A, and leave the adjacent
/// marker region B byte-for-byte untouched. Exercises the full prod
/// path — `shm_open_lz4`, the LZ4-magic pread validation, the rounded-
/// length bounds check, and the `MAP_FIXED` overlay via `cow_overlay` —
/// not a re-implementation of the bounds check.
#[test]
fn try_cow_overlay_maps_segment_and_preserves_adjacent_region() {
    use vm_memory::{Bytes, GuestAddress};

    let page = host_page_size() as usize;
    // Region A holds the overlay target; region B holds a marker that
    // must survive. Each is several host pages so the rounded-up overlay
    // length fits comfortably inside region A.
    let region_a_size = page * 4;
    let region_b_size = page * 4;
    let region_a_start: u64 = 0;
    let region_b_start: u64 = (region_a_size as u64) + (1 << 20); // 1 MiB gap
    let mem = GuestMemoryMmap::<()>::from_ranges(&[
        (GuestAddress(region_a_start), region_a_size),
        (GuestAddress(region_b_start), region_b_size),
    ])
    .unwrap();

    // Plant a detectable marker across the whole of region B.
    let marker: Vec<u8> = (0..region_b_size).map(|i| (i & 0xff) as u8).collect();
    mem.write_slice(&marker, GuestAddress(region_b_start))
        .unwrap();

    // Store a real LZ4-magic SHM segment (one host page of content) and
    // key the overlay off its content hash. Use a hash unlikely to
    // collide with any concurrent test's segment.
    let hash = 0xC0FF_EE00_DEAD_F00Du64;
    let _ = rustix::shm::unlink(initramfs::shm_lz4_segment_name(hash).as_str());
    let mut segment = initramfs::LZ4_LEGACY_MAGIC.to_vec();
    segment.extend((segment.len()..page).map(|i| (i & 0xff) as u8));
    assert_eq!(segment.len(), page, "segment sized to one host page");
    initramfs::shm_store_lz4(hash, &segment).unwrap();

    let key = BaseKey(hash);
    let guard = KtstrVm::try_cow_overlay(&mem, &key, segment.len(), region_a_start);
    assert!(
        guard.is_some(),
        "overlay of a valid, in-bounds, page-aligned segment must succeed",
    );

    // Region A now reflects the SHM segment content (MAP_PRIVATE reads
    // see the backing bytes until first write).
    let mut a_readback = vec![0u8; segment.len()];
    mem.read_slice(&mut a_readback, GuestAddress(region_a_start))
        .unwrap();
    assert_eq!(
        a_readback, segment,
        "region A must reflect the COW-mapped segment bytes",
    );

    // Region B is byte-for-byte untouched — the overlay never reached it.
    let mut b_readback = vec![0u8; region_b_size];
    mem.read_slice(&mut b_readback, GuestAddress(region_b_start))
        .unwrap();
    assert_eq!(
        b_readback, marker,
        "adjacent region B must be untouched by the overlay",
    );

    // Drop the guard (releases LOCK_SH + closes fd) BEFORE the guest
    // memory unmaps the MAP_FIXED region, matching the prod drop order.
    drop(guard);
    drop(mem);
    let _ = rustix::shm::unlink(initramfs::shm_lz4_segment_name(hash).as_str());
}

/// Drive the REAL `try_cow_overlay` with a request whose rounded length
/// overruns region A into the inter-region gap. The prod bounds check
/// (`get_slice` on the rounded length) must reject it: `try_cow_overlay`
/// returns `None`, never invokes `MAP_FIXED`, and the adjacent marker
/// region survives. Unlike the dependency-contract pin in
/// `initramfs_tests.rs` (which calls `get_slice` directly), this routes
/// through the production function, so dropping the guard or bounds-
/// checking `len` instead of `rounded_len` would fail here.
#[test]
fn try_cow_overlay_rejects_oversized_request_and_preserves_region() {
    use vm_memory::{Bytes, GuestAddress};

    let page = host_page_size() as usize;
    let region_a_size = page * 2;
    let region_b_size = page * 2;
    let region_a_start: u64 = 0;
    let region_b_start: u64 = (region_a_size as u64) + (1 << 20); // 1 MiB gap
    let mem = GuestMemoryMmap::<()>::from_ranges(&[
        (GuestAddress(region_a_start), region_a_size),
        (GuestAddress(region_b_start), region_b_size),
    ])
    .unwrap();

    let marker: Vec<u8> = (0..region_b_size).map(|i| (i & 0xff) as u8).collect();
    mem.write_slice(&marker, GuestAddress(region_b_start))
        .unwrap();

    // Segment one host page LARGER than region A: the rounded overlay
    // length cannot fit, so the bounds check must reject it.
    let hash = 0xBADC_0DE0_0BAD_F00Du64;
    let _ = rustix::shm::unlink(initramfs::shm_lz4_segment_name(hash).as_str());
    let oversized_len = region_a_size + page;
    let mut segment = initramfs::LZ4_LEGACY_MAGIC.to_vec();
    segment.extend((segment.len()..oversized_len).map(|i| (i & 0xff) as u8));
    assert_eq!(segment.len(), oversized_len);
    initramfs::shm_store_lz4(hash, &segment).unwrap();

    let key = BaseKey(hash);
    let guard = KtstrVm::try_cow_overlay(&mem, &key, segment.len(), region_a_start);
    assert!(
        guard.is_none(),
        "an overlay whose rounded length overruns region A must be rejected",
    );

    // Region B is untouched: MAP_FIXED was never invoked.
    let mut b_readback = vec![0u8; region_b_size];
    mem.read_slice(&mut b_readback, GuestAddress(region_b_start))
        .unwrap();
    assert_eq!(
        b_readback, marker,
        "region B must survive a rejected overlay",
    );

    drop(mem);
    let _ = rustix::shm::unlink(initramfs::shm_lz4_segment_name(hash).as_str());
}
