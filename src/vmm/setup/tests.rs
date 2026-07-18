use super::*;

#[test]
fn prepared_host_address_accepts_page_aligned_non_hugepage_va() {
    let host_page = host_page_size() as usize;
    assert!(
        host_page < PREPARED_MAPPING_GRANULE,
        "test requires the supported host page size to be below 2 MiB"
    );
    let address = host_page;
    assert!(address.is_multiple_of(host_page));
    assert_ne!(
        address % PREPARED_MAPPING_GRANULE,
        0,
        "fixture must be host-page aligned but not 2 MiB aligned"
    );
    validate_prepared_host_address(address as *mut u8)
        .expect("ordinary host-page-aligned GuestMemory VA must be accepted");
    assert!(
        validate_prepared_host_address((address + 1) as *mut u8).is_err(),
        "a truly host-page-misaligned destination must still be rejected"
    );
}

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

    // Match the prod teardown order (x86_64/kvm.rs, aarch64 mirror):
    // guest memory unmaps the MAP_FIXED COW region FIRST, then the guard
    // releases LOCK_SH + closes the fd.
    drop(mem);
    drop(guard);
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

/// Drive the REAL `try_cow_overlay` against a stored SHM segment whose
/// length matches `expected_len` but whose first 4 bytes are NOT the
/// LZ4 legacy magic. The magic-validation arm (`if magic !=
/// initramfs::LZ4_LEGACY_MAGIC`) must reject it: `try_cow_overlay`
/// closes the fd and returns `None`, never reaching the `MAP_FIXED`
/// overlay, so the adjacent marker region stays byte-identical. The two
/// existing overlay tests only store segments whose header IS the magic,
/// so this stale-format rejection arm was never executed.
#[test]
fn try_cow_overlay_rejects_stale_non_lz4_magic_segment() {
    use vm_memory::{Bytes, GuestAddress};

    let page = host_page_size() as usize;
    // Same two-region fixture as the success path: region A is the
    // overlay target, region B holds a marker that must survive.
    let region_a_size = page * 4;
    let region_b_size = page * 4;
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

    // One host page of content whose first 4 bytes are 0xAB.. — never
    // the LZ4 legacy magic (0x184C2102 little-endian). `expected_len`
    // equals the stored length so the len check passes and execution
    // reaches the magic pread.
    let hash = 0x5741_4C45_F00D_BEEFu64;
    let _ = rustix::shm::unlink(initramfs::shm_lz4_segment_name(hash).as_str());
    let segment: Vec<u8> = vec![0xABu8; page];
    assert_ne!(
        segment[..4],
        initramfs::LZ4_LEGACY_MAGIC,
        "fixture header must NOT be the LZ4 legacy magic",
    );
    initramfs::shm_store_lz4(hash, &segment).unwrap();

    let key = BaseKey(hash);
    let guard = KtstrVm::try_cow_overlay(&mem, &key, segment.len(), region_a_start);
    assert!(
        guard.is_none(),
        "a segment without the LZ4 legacy magic must be rejected",
    );

    // Region B is byte-for-byte untouched — MAP_FIXED never ran.
    let mut b_readback = vec![0u8; region_b_size];
    mem.read_slice(&mut b_readback, GuestAddress(region_b_start))
        .unwrap();
    assert_eq!(
        b_readback, marker,
        "region B must survive a magic-rejected overlay",
    );

    drop(mem);
    let _ = rustix::shm::unlink(initramfs::shm_lz4_segment_name(hash).as_str());
}

/// Drive the REAL `try_cow_overlay` with a non-host-page-aligned
/// `load_addr`. The alignment gate (`if load_addr & (host_page - 1) !=
/// 0`) must reject it: `try_cow_overlay` returns `None` and never
/// invokes `MAP_FIXED` (mmap would return `EINVAL` on a mid-page
/// target). The segment carries a VALID LZ4 magic so execution passes
/// the magic check and reaches the alignment gate; `load_addr = 1` sits
/// inside region A's bounds yet fails the page-alignment test on every
/// supported host page size. The two existing overlay tests pass
/// `load_addr = 0` (page-aligned), so this arm was never executed.
#[test]
fn try_cow_overlay_rejects_unaligned_load_addr() {
    use vm_memory::{Bytes, GuestAddress};

    let page = host_page_size() as usize;
    let region_a_size = page * 4;
    let region_b_size = page * 4;
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

    // Valid one-host-page LZ4 segment so the magic + len + bounds checks
    // all pass; only the alignment gate must trip.
    let hash = 0x0FF5_E700_A11A_BEEFu64;
    let _ = rustix::shm::unlink(initramfs::shm_lz4_segment_name(hash).as_str());
    let mut segment = initramfs::LZ4_LEGACY_MAGIC.to_vec();
    segment.extend((segment.len()..page).map(|i| (i & 0xff) as u8));
    assert_eq!(segment.len(), page, "segment sized to one host page");
    initramfs::shm_store_lz4(hash, &segment).unwrap();

    // load_addr = 1: inside region A [0, region_a_size) but not aligned
    // to any host page size (4 KiB or 16 KiB).
    let unaligned_addr: u64 = 1;
    let key = BaseKey(hash);
    let guard = KtstrVm::try_cow_overlay(&mem, &key, segment.len(), unaligned_addr);
    assert!(
        guard.is_none(),
        "a non-host-page-aligned load_addr must be rejected",
    );

    // No overlay touched memory: region B (and region A's marker-free
    // start) survive. Assert region B against its planted marker.
    let mut b_readback = vec![0u8; region_b_size];
    mem.read_slice(&mut b_readback, GuestAddress(region_b_start))
        .unwrap();
    assert_eq!(
        b_readback, marker,
        "region B must survive an alignment-rejected overlay",
    );

    drop(mem);
    let _ = rustix::shm::unlink(initramfs::shm_lz4_segment_name(hash).as_str());
}

/// `aarch64_initrd_addr` must return a host-page-aligned load address —
/// the exact invariant the COW `MAP_FIXED` overlay relies on (the
/// function header cites `EINVAL` on a mid-host-page target). The
/// existing aarch64 tests bound the address (`>= DRAM_START`,
/// `load + max <= pvtime_base`) and check the oversized `Err` path but
/// never assert host-page alignment. Each fixture size is chosen so the
/// pre-mask `pvtime_base - size` is NOT page-aligned, proving the mask
/// rounds it down rather than passing trivially.
#[cfg(target_arch = "aarch64")]
#[test]
fn aarch64_initrd_addr_returns_host_page_aligned_address() {
    use crate::vmm::{aarch64::fdt::pvtime_base, kvm::DRAM_START};
    let page = host_page_size();
    for &(mem, cpus) in &[(512u32, 2u32), (512, 8), (2048, 256), (4096, 512)] {
        let pvt = pvtime_base(mem, cpus);
        // Size chosen so `pvt - size = DRAM_START + 12345`, whose low
        // bits (12345) are not page-aligned: the mask must round down.
        let size = pvt - DRAM_START - 12345;
        let load = aarch64_initrd_addr(mem, cpus, size)
            .expect("non-oversized initrd must produce a load address");
        assert_eq!(
            load & (page - 1),
            0,
            "initrd load addr {load:#x} not host-page-aligned \
             (mem={mem} cpus={cpus} size={size:#x} page={page:#x})",
        );
        // The pre-mask top was deliberately unaligned, so the mask did
        // real work: the aligned result must be strictly below it.
        let pre_mask_top = pvt - size;
        assert_ne!(
            pre_mask_top & (page - 1),
            0,
            "fixture must present a non-page-aligned pre-mask top so the \
             mask is exercised (mem={mem} cpus={cpus})",
        );
        assert!(
            load < pre_mask_top,
            "masked load {load:#x} must round DOWN from unaligned top \
             {pre_mask_top:#x} (mem={mem} cpus={cpus})",
        );
    }
}

/// Pin the EXACT `aarch64_initrd_addr` arithmetic against an
/// independently-computed reference: `(pvtime_base(mem,cpus) - size) &
/// !(host_page_size() - 1)`. The existing aarch64 tests only bound the
/// result, so a drift that swapped the ceiling (`pvtime_base` vs
/// `fdt_address`) or changed the alignment granule would still satisfy
/// their inequalities but fail this exact-equality pin.
#[cfg(target_arch = "aarch64")]
#[test]
fn aarch64_initrd_addr_exact_value_for_aligned_fit() {
    use crate::vmm::aarch64::fdt::pvtime_base;
    let page = host_page_size();
    for &(mem, cpus, size) in &[(512u32, 2u32, 1u64 << 20), (2048, 256, 7_000_000)] {
        let expected = (pvtime_base(mem, cpus) - size) & !(page - 1);
        assert_eq!(
            aarch64_initrd_addr(mem, cpus, size).unwrap(),
            expected,
            "exact load addr drift (mem={mem} cpus={cpus} size={size:#x})",
        );
    }
}

/// `base_guest_cmdline` must splice the arch-specific tail in and pin
/// the cross-arch common flags. The free fn was extracted so a flag
/// added once applies to BOTH arches — the doc cites a past per-arch
/// drift that left `sysctl.vm.overcommit_memory=1` on x86 only and
/// OOM-ed the aarch64 guest /init. Neither caller (`build_guest_cmdline`,
/// `finish_aarch64_setup`) is host-testable, so this directly pins the
/// assembled string: the cross-arch invariant flags, the spliced arch
/// tail, and the `console=ttyS0` / `KTSTR_GUEST=1` anchors.
#[test]
fn base_guest_cmdline_splices_arch_tail_and_pins_common_flags() {
    let s = base_guest_cmdline("KFENCE_TAIL_MARKER");
    // Cross-arch common flags (the drift this fn exists to prevent).
    assert!(
        s.contains("sysctl.vm.overcommit_memory=1"),
        "missing overcommit_memory=1 (the OOM-prevention flag); got {s:?}",
    );
    assert!(
        s.contains("sysctl.kernel.sched_schedstats=1"),
        "missing sched_schedstats=1; got {s:?}",
    );
    assert!(s.contains("delayacct"), "missing delayacct; got {s:?}");
    // The arch tail is spliced in verbatim.
    assert!(
        s.contains("KFENCE_TAIL_MARKER"),
        "arch_extra tail not spliced in; got {s:?}",
    );
    // Start/end anchors: cmdline opens with console=ttyS0 and the
    // KTSTR_GUEST=1 trailer is the final token.
    assert!(
        s.starts_with("console=ttyS0"),
        "cmdline must open with console=ttyS0; got {s:?}",
    );
    assert!(
        s.ends_with("KTSTR_GUEST=1"),
        "cmdline must end with the KTSTR_GUEST=1 trailer; got {s:?}",
    );
}

#[test]
fn numa_balancing_token_uses_kernel_accepted_spellings() {
    use crate::vmm::topology::{NumaNode, Topology};
    // Uniform topology has no memory-only nodes -> disable. The token
    // MUST be the kernel-accepted "disable" string: setup_numabalancing
    // (mm/mempolicy.c) strcmp-rejects everything but "enable"/"disable",
    // so the old "numa_balancing=0" was silently ignored, leaving NUMA
    // balancing at its CONFIG default instead of off.
    let uniform = Topology::new(1, 1, 2, 1);
    assert!(!uniform.has_memory_only_nodes());
    assert_eq!(
        numa_balancing_cmdline_token(&uniform),
        " numa_balancing=disable",
        "disable token must be the kernel-accepted 'disable' string, not '0'",
    );
    // Memory-only (CXL) topology -> enable (migrate pages toward
    // CPU-bearing nodes).
    static NODES: [NumaNode; 3] = [
        NumaNode::new(2, 512),
        NumaNode::new(2, 512),
        NumaNode::new(0, 1024),
    ];
    let cxl = Topology::with_nodes(4, 1, &NODES);
    assert!(cxl.has_memory_only_nodes());
    assert_eq!(numa_balancing_cmdline_token(&cxl), " numa_balancing=enable");
}

/// Truth table for [`halt_poll_policy`]: the run-lock outcome and mode
/// flags map to the halt-poll interval (or `None` = leave the module
/// default). Mirrors the W2f policy documented on the fn.
#[test]
fn halt_poll_policy_truth_table() {
    // (no_perf_mode, performance_mode, overcommit) -> expected
    let cases = [
        // no_perf_mode always disables polling (shared host CPUs).
        ((true, false, false), Some(0u64)),
        ((true, false, true), Some(0)),
        // performance_mode leaves the module default (guest haltpoll drives it).
        ((false, true, false), None),
        // default mode, 1:1 pin -> leave the module default.
        ((false, false, false), None),
        // default mode, overcommit fallback -> disable polling.
        ((false, false, true), Some(0)),
    ];
    for ((no_perf, perf, over), expected) in cases {
        assert_eq!(
            halt_poll_policy(no_perf, perf, over),
            expected,
            "halt_poll_policy(no_perf={no_perf}, perf={perf}, overcommit={over})",
        );
    }
}
