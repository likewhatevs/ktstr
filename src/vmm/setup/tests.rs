use super::*;
use std::os::unix::fs::FileExt as _;

fn test_prepared_mapping(magic: &[u8], guest_offset: u64, map_len: usize) -> PreparedMapping {
    use std::io::Write as _;

    let mut file = tempfile::tempfile().unwrap();
    file.set_len(map_len as u64).unwrap();
    file.write_all(magic).unwrap();
    PreparedMapping {
        fd: file.into(),
        file_offset: 0,
        guest_offset,
        map_len,
        overlays: Vec::new(),
    }
}

fn test_prepared_overlay(bytes: &[u8], guest_offset: u64, map_len: usize) -> PreparedOverlay {
    use std::io::Write as _;

    let mut file = tempfile::tempfile().unwrap();
    file.set_len(map_len as u64).unwrap();
    file.write_all(bytes).unwrap();
    PreparedOverlay {
        fd: file.into(),
        file_offset: 0,
        guest_offset,
        map_len,
    }
}

#[test]
fn prepared_load_rejects_gap_overlap_and_reordering_before_mapping() {
    let page = PREPARED_MAPPING_GRANULE;
    let host_page = host_page_size() as usize;
    let compressed_len = page + 1;
    let magic = initramfs::LZ4_LEGACY_MAGIC;
    for (name, guest_offsets) in [
        ("gap", [0, (2 * page) as u64]),
        ("overlap", [0, 0]),
        ("reordering", [page as u64, 0]),
    ] {
        let ranges =
            guest_offsets.map(|guest_offset| test_prepared_mapping(&magic, guest_offset, page));
        let error = validate_prepared_load(
            compressed_len,
            initramfs::InitrdCompression::Lz4,
            page,
            host_page,
            0,
            &ranges,
        )
        .unwrap_err();
        assert!(
            format!("{error:#}").contains("gap, overlap, or reordering"),
            "{name} reached mapping instead of failing geometry validation: {error:#}"
        );
    }
}

#[cfg(target_pointer_width = "64")]
#[test]
fn prepared_load_rejects_u32_overflow_before_reading_or_mapping() {
    let error = validate_prepared_load(
        u32::MAX as usize + 1,
        initramfs::InitrdCompression::Lz4,
        host_page_size() as usize,
        host_page_size() as usize,
        0,
        &[],
    )
    .unwrap_err();
    assert!(
        format!("{error:#}").contains("exceeds u32 boot-size field"),
        "boot protocol size conversion must be the first validation: {error:#}"
    );
}

#[test]
fn prepared_load_validates_magic_for_every_compression() {
    let page = PREPARED_MAPPING_GRANULE;
    let host_page = host_page_size() as usize;
    for (compression, magic) in [
        (
            initramfs::InitrdCompression::Lz4,
            initramfs::LZ4_LEGACY_MAGIC.as_slice(),
        ),
        (
            initramfs::InitrdCompression::Zstd,
            b"\x28\xb5\x2f\xfd".as_slice(),
        ),
        (initramfs::InitrdCompression::Gzip, b"\x1f\x8b".as_slice()),
        (
            initramfs::InitrdCompression::Uncompressed,
            b"070701".as_slice(),
        ),
    ] {
        let ranges = [test_prepared_mapping(magic, 0, page)];
        assert_eq!(
            validate_prepared_load(magic.len(), compression, page, host_page, 0, &ranges).unwrap(),
            magic.len() as u32
        );
    }

    let ranges = [test_prepared_mapping(b"nope", 0, page)];
    let error = validate_prepared_load(
        4,
        initramfs::InitrdCompression::Lz4,
        page,
        host_page,
        0,
        &ranges,
    )
    .unwrap_err();
    assert!(
        format!("{error:#}").contains("invalid LZ4 legacy magic"),
        "compression metadata must select the validation magic: {error:#}"
    );
}

#[test]
fn prepared_load_rejects_invalid_backing_extents_before_mapping() {
    let page = PREPARED_MAPPING_GRANULE;
    let host_page = host_page_size() as usize;
    let mut ranges = vec![
        test_prepared_mapping(&initramfs::LZ4_LEGACY_MAGIC, 0, page),
        test_prepared_mapping(&[], page as u64, page),
    ];
    rustix::fs::ftruncate(&ranges[1].fd, (page - 1) as u64).unwrap();
    let error = validate_prepared_load(
        page + 1,
        initramfs::InitrdCompression::Lz4,
        page,
        host_page,
        0,
        &ranges,
    )
    .unwrap_err();
    assert!(
        format!("{error:#}").contains("mapping exceeds its backing file"),
        "a malformed later range must fail before an earlier range can be mapped: {error:#}"
    );

    ranges.truncate(1);
    ranges[0].file_offset = libc::off_t::MAX as u64 + 1;
    let error = validate_prepared_load(
        1,
        initramfs::InitrdCompression::Lz4,
        page,
        host_page,
        0,
        &ranges,
    )
    .unwrap_err();
    assert!(
        format!("{error:#}").contains("file offset exceeds off_t"),
        "every mmap offset must be representable before MAP_FIXED: {error:#}"
    );
}

#[test]
fn prepared_load_rejects_malformed_nested_overlays_before_mapping() {
    let page = PREPARED_MAPPING_GRANULE;
    let host_page = host_page_size() as usize;
    let valid_overlay = |guest_offset, map_len| {
        test_prepared_overlay(&initramfs::LZ4_LEGACY_MAGIC, guest_offset, map_len)
    };
    let validate = |overlays| {
        let mut range = test_prepared_mapping(&initramfs::LZ4_LEGACY_MAGIC, 0, page);
        range.overlays = overlays;
        validate_prepared_load(
            1,
            initramfs::InitrdCompression::Lz4,
            page,
            host_page,
            0,
            &[range],
        )
        .unwrap_err()
    };

    for (name, overlay, expected) in [
        (
            "zero length",
            valid_overlay(0, 0),
            "overlay length is not aligned",
        ),
        (
            "misaligned length",
            valid_overlay(0, host_page + 1),
            "overlay length is not aligned",
        ),
        (
            "misaligned guest offset",
            valid_overlay(1, host_page),
            "overlay guest offset is not aligned",
        ),
        (
            "escapes primary mapping",
            valid_overlay((page - host_page) as u64, 2 * host_page),
            "overlay escapes its primary mapping",
        ),
    ] {
        let error = validate(vec![overlay]);
        assert!(
            format!("{error:#}").contains(expected),
            "{name} reached mapping instead of failing overlay validation: {error:#}"
        );
    }

    let mut misaligned_file_offset = valid_overlay(0, host_page);
    misaligned_file_offset.file_offset = 1;
    let error = validate(vec![misaligned_file_offset]);
    assert!(
        format!("{error:#}").contains("file offset is not aligned"),
        "a misaligned overlay file offset reached mapping: {error:#}"
    );

    for (name, overlays) in [
        (
            "overlap",
            vec![
                valid_overlay(host_page as u64, 2 * host_page),
                valid_overlay((2 * host_page) as u64, host_page),
            ],
        ),
        (
            "reordering",
            vec![
                valid_overlay((2 * host_page) as u64, host_page),
                valid_overlay(host_page as u64, host_page),
            ],
        ),
    ] {
        let error = validate(overlays);
        assert!(
            format!("{error:#}").contains("overlays overlap or are reordered"),
            "{name} reached mapping instead of failing overlay ordering: {error:#}"
        );
    }

    let truncated = valid_overlay(0, host_page);
    rustix::fs::ftruncate(&truncated.fd, (host_page - 1) as u64).unwrap();
    let error = validate(vec![truncated]);
    assert!(
        format!("{error:#}").contains("overlay exceeds its backing file"),
        "a truncated overlay reached mapping: {error:#}"
    );
}

#[test]
fn prepared_load_validates_magic_from_offset_zero_overlay() {
    let page = PREPARED_MAPPING_GRANULE;
    let host_page = host_page_size() as usize;

    let mut visible_magic = test_prepared_mapping(b"nope", 0, page);
    visible_magic.overlays.push(test_prepared_overlay(
        &initramfs::LZ4_LEGACY_MAGIC,
        0,
        host_page,
    ));
    assert_eq!(
        validate_prepared_load(
            initramfs::LZ4_LEGACY_MAGIC.len(),
            initramfs::InitrdCompression::Lz4,
            page,
            host_page,
            0,
            &[visible_magic],
        )
        .unwrap(),
        initramfs::LZ4_LEGACY_MAGIC.len() as u32,
        "compression validation must read the offset-zero overlay visible to the guest"
    );

    let mut hidden_magic = test_prepared_mapping(&initramfs::LZ4_LEGACY_MAGIC, 0, page);
    hidden_magic
        .overlays
        .push(test_prepared_overlay(b"nope", 0, host_page));
    let error = validate_prepared_load(
        initramfs::LZ4_LEGACY_MAGIC.len(),
        initramfs::InitrdCompression::Lz4,
        page,
        host_page,
        0,
        &[hidden_magic],
    )
    .unwrap_err();
    assert!(
        format!("{error:#}").contains("invalid LZ4 legacy magic"),
        "a valid but hidden primary header must not mask invalid visible overlay bytes: {error:#}"
    );
}

#[test]
fn prepared_subrange_validation_covers_adjacent_regions_and_rejects_holes() {
    let page = host_page_size() as usize;
    let adjacent = GuestMemoryMmap::<()>::from_ranges(&[
        (GuestAddress(0), page),
        (GuestAddress(page as u64), page),
    ])
    .unwrap();
    let validated = validate_prepared_subranges(
        &adjacent,
        vec![test_prepared_mapping(
            &initramfs::LZ4_LEGACY_MAGIC,
            0,
            2 * page,
        )],
        0,
        page,
        page,
    )
    .unwrap();
    assert_eq!(validated.len(), 1);
    assert_eq!(validated[0].subranges.len(), 2);
    assert_eq!(validated[0].subranges[0].guest_addr, 0);
    assert_eq!(validated[0].subranges[0].file_offset, 0);
    assert_eq!(validated[0].subranges[0].len, page);
    assert_eq!(validated[0].subranges[1].guest_addr, page as u64);
    assert_eq!(validated[0].subranges[1].file_offset, page as u64);
    assert_eq!(validated[0].subranges[1].len, page);

    let with_hole = GuestMemoryMmap::<()>::from_ranges(&[
        (GuestAddress(0), page),
        (GuestAddress((2 * page) as u64), page),
    ])
    .unwrap();
    let error = validate_prepared_subranges(
        &with_hole,
        vec![test_prepared_mapping(
            &initramfs::LZ4_LEGACY_MAGIC,
            0,
            3 * page,
        )],
        0,
        page,
        page,
    )
    .unwrap_err();
    assert!(
        format!("{error:#}").contains("guest-memory hole"),
        "the exact production split must reject uncovered guest ranges: {error:#}"
    );
}

#[test]
fn prepared_multi_region_direct_cow_maps_primary_then_overlay_and_isolates_both() {
    use std::io::Write as _;
    use std::os::unix::fs::PermissionsExt as _;
    use vm_memory::mmap::{GuestRegionMmap, MmapRegion};

    struct Reservation {
        base: *mut libc::c_void,
        len: usize,
    }

    impl Drop for Reservation {
        fn drop(&mut self) {
            // SAFETY: `base` and `len` are the unchanged result and length of
            // the successful reservation mmap below. MAP_FIXED replacements
            // preserve that complete virtual-address extent.
            let _ = unsafe { libc::munmap(self.base, self.len) };
        }
    }

    let granule = PREPARED_MAPPING_GRANULE;
    let host_page = host_page_size() as usize;
    let reservation_len = 4 * granule;

    // Declare the backing-fd guards first so the VA reservation is always
    // torn down before they unlock and close, including during unwinding.
    let mut cow_guards = Vec::new();

    // SAFETY: this creates a fresh inaccessible arena owned by `reservation`.
    let base = unsafe {
        libc::mmap(
            std::ptr::null_mut(),
            reservation_len,
            libc::PROT_NONE,
            libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
            -1,
            0,
        )
    };
    assert_ne!(
        base,
        libc::MAP_FAILED,
        "reserve test address space: {}",
        std::io::Error::last_os_error()
    );
    let reservation = Reservation {
        base,
        len: reservation_len,
    };

    let first_host = base.cast::<u8>();
    // Adjacent guest regions deliberately live at unrelated host VMAs. Two
    // whole prepared granules remain PROT_NONE between them, so treating the
    // first HVA as a contiguous base deterministically faults.
    // SAFETY: the offset remains inside the reserved four-granule arena.
    let second_host = unsafe { first_host.add(3 * granule) };
    for host in [first_host, second_host] {
        // SAFETY: each target is an aligned, non-overlapping granule wholly
        // inside our PROT_NONE reservation. MAP_FIXED replaces only that slot.
        let mapped = unsafe {
            libc::mmap(
                host.cast(),
                granule,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_FIXED,
                -1,
                0,
            )
        };
        assert_eq!(
            mapped,
            host.cast(),
            "map test guest region at {host:p}: {}",
            std::io::Error::last_os_error()
        );
    }

    let mut underlay_named = tempfile::NamedTempFile::new().unwrap();
    let mut underlay_bytes = vec![0x31; 2 * granule];
    underlay_bytes[..initramfs::LZ4_LEGACY_MAGIC.len()]
        .copy_from_slice(&initramfs::LZ4_LEGACY_MAGIC);
    underlay_bytes[granule..].fill(0x72);
    underlay_named.write_all(&underlay_bytes).unwrap();
    underlay_named.as_file().sync_all().unwrap();
    let underlay_path = underlay_named.into_temp_path();
    std::fs::set_permissions(&underlay_path, std::fs::Permissions::from_mode(0o444)).unwrap();
    let underlay = std::fs::OpenOptions::new()
        .read(true)
        .open(&underlay_path)
        .unwrap();
    let underlay_observer = underlay.try_clone().unwrap();
    rustix::fs::flock(&underlay, rustix::fs::FlockOperation::LockShared).unwrap();

    let overlay_guest_offset = granule + host_page;
    let overlay_len = 2 * host_page;
    let overlay_file_offset = host_page;
    let mut overlay_named = tempfile::NamedTempFile::new().unwrap();
    let mut overlay_bytes = vec![0x19; overlay_file_offset + overlay_len];
    overlay_bytes[overlay_file_offset..].fill(0xb6);
    overlay_named.write_all(&overlay_bytes).unwrap();
    overlay_named.as_file().sync_all().unwrap();
    let overlay_path = overlay_named.into_temp_path();
    std::fs::set_permissions(&overlay_path, std::fs::Permissions::from_mode(0o444)).unwrap();
    let overlay = std::fs::OpenOptions::new()
        .read(true)
        .open(&overlay_path)
        .unwrap();
    let overlay_observer = overlay.try_clone().unwrap();
    rustix::fs::flock(&overlay, rustix::fs::FlockOperation::LockShared).unwrap();
    let underlay_ino = rustix::fs::fstat(&underlay).unwrap().st_ino;
    let overlay_ino = rustix::fs::fstat(&overlay).unwrap().st_ino;

    // SAFETY: both raw regions describe the complete live anonymous VMAs
    // installed above. `Reservation` owns and outlives these non-owning
    // vm-memory wrappers.
    let first_region = unsafe {
        MmapRegion::build_raw(
            first_host,
            granule,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
        )
        .unwrap()
    };
    // SAFETY: same invariant as `first_region`, for the discontiguous slot.
    let second_region = unsafe {
        MmapRegion::build_raw(
            second_host,
            granule,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
        )
        .unwrap()
    };
    let guest_mem = GuestMemoryMmap::from_regions(vec![
        GuestRegionMmap::new(first_region, GuestAddress(0)).unwrap(),
        GuestRegionMmap::new(second_region, GuestAddress(granule as u64)).unwrap(),
    ])
    .unwrap();

    let ranges = vec![PreparedMapping {
        fd: underlay.into(),
        file_offset: 0,
        guest_offset: 0,
        map_len: underlay_bytes.len(),
        overlays: vec![PreparedOverlay {
            fd: overlay.into(),
            file_offset: overlay_file_offset as u64,
            guest_offset: overlay_guest_offset as u64,
            map_len: overlay_len,
        }],
    }];
    assert_eq!(
        validate_prepared_load(
            underlay_bytes.len(),
            initramfs::InitrdCompression::Lz4,
            granule,
            host_page,
            0,
            &ranges,
        )
        .unwrap(),
        underlay_bytes.len() as u32
    );
    let validated =
        validate_prepared_subranges(&guest_mem, ranges, 0, host_page, host_page).unwrap();
    assert_eq!(validated.len(), 1);
    assert_eq!(validated[0].subranges.len(), 2);
    assert_eq!(validated[0].subranges[0].guest_addr, 0);
    assert_eq!(validated[0].subranges[0].host_addr, first_host);
    assert_eq!(validated[0].subranges[0].file_offset, 0);
    assert_eq!(validated[0].subranges[0].len, granule);
    assert_eq!(validated[0].subranges[1].guest_addr, granule as u64);
    assert_eq!(validated[0].subranges[1].host_addr, second_host);
    assert_eq!(validated[0].subranges[1].file_offset, granule as u64);
    assert_eq!(validated[0].subranges[1].len, granule);
    assert_eq!(validated[0].overlays.len(), 1);
    assert_eq!(validated[0].overlays[0].subranges.len(), 1);
    assert_eq!(
        validated[0].overlays[0].subranges[0].guest_addr,
        overlay_guest_offset as u64
    );
    assert_eq!(
        validated[0].overlays[0].subranges[0].file_offset,
        overlay_file_offset as u64
    );
    assert_eq!(validated[0].overlays[0].subranges[0].len, overlay_len);
    assert_eq!(second_host as usize, first_host as usize + 3 * granule);

    let mut map_order = Vec::new();
    map_validated_prepared_ranges(
        &mut cow_guards,
        validated,
        |subrange, fd| {
            map_order.push(rustix::fs::fstat(fd).unwrap().st_ino);
            // SAFETY: production validation above proved that each complete
            // destination subrange lies inside one live guest-memory region.
            unsafe {
                initramfs::cow_overlay_file_borrowed(
                    subrange.host_addr,
                    subrange.len,
                    fd,
                    subrange.file_offset,
                )
            }
        },
        |_| Ok(()),
    )
    .unwrap();
    assert_eq!(
        cow_guards.len(),
        2,
        "the split primary and its overlay must retain both locked backing fds"
    );
    assert_eq!(
        map_order,
        [underlay_ino, underlay_ino, overlay_ino],
        "every primary subrange must be installed before its nested overlay"
    );

    let mut magic = [0; 4];
    guest_mem.read_slice(&mut magic, GuestAddress(0)).unwrap();
    assert_eq!(magic, initramfs::LZ4_LEGACY_MAGIC);
    let mut first_byte = [0];
    guest_mem
        .read_slice(&mut first_byte, GuestAddress(host_page as u64))
        .unwrap();
    assert_eq!(first_byte, [0x31]);
    let mut second_byte = [0];
    guest_mem
        .read_slice(&mut second_byte, GuestAddress(granule as u64))
        .unwrap();
    assert_eq!(
        second_byte,
        [0x72],
        "the second guest region must map from the second file offset"
    );
    let mut visible_overlay = vec![0; overlay_len];
    guest_mem
        .read_slice(
            &mut visible_overlay,
            GuestAddress(overlay_guest_offset as u64),
        )
        .unwrap();
    assert_eq!(
        visible_overlay,
        overlay_bytes[overlay_file_offset..],
        "the nested overlay must replace exactly its declared guest bytes"
    );
    guest_mem
        .read_slice(
            &mut second_byte,
            GuestAddress((overlay_guest_offset + overlay_len) as u64),
        )
        .unwrap();
    assert_eq!(
        second_byte,
        [0x72],
        "bytes immediately after the overlay must remain mapped from the primary"
    );

    let private_underlay_offset = host_page;
    guest_mem
        .write_slice(&[0xe5], GuestAddress(private_underlay_offset as u64))
        .unwrap();
    guest_mem
        .read_slice(
            &mut second_byte,
            GuestAddress(private_underlay_offset as u64),
        )
        .unwrap();
    assert_eq!(second_byte, [0xe5]);
    assert_eq!(
        underlay_observer
            .read_at(&mut second_byte, private_underlay_offset as u64)
            .unwrap(),
        1
    );
    assert_eq!(
        second_byte,
        [0x31],
        "MAP_PRIVATE writes to primary-visible bytes must not modify the underlay"
    );

    let private_overlay_offset = overlay_guest_offset + host_page;
    guest_mem
        .write_slice(&[0xd7], GuestAddress(private_overlay_offset as u64))
        .unwrap();
    guest_mem
        .read_slice(
            &mut second_byte,
            GuestAddress(private_overlay_offset as u64),
        )
        .unwrap();
    assert_eq!(second_byte, [0xd7]);
    assert_eq!(
        overlay_observer
            .read_at(&mut second_byte, (overlay_file_offset + host_page) as u64,)
            .unwrap(),
        1
    );
    assert_eq!(
        second_byte,
        [0xb6],
        "MAP_PRIVATE writes to overlay-visible bytes must not modify the overlay object"
    );
    assert_eq!(
        underlay_observer
            .read_at(&mut second_byte, private_overlay_offset as u64)
            .unwrap(),
        1
    );
    assert_eq!(
        second_byte,
        [0x72],
        "overlay-visible writes must not reach the hidden underlay object"
    );

    // Preserve the production lifetime order explicitly: wrappers first,
    // mapped reservation second, and the locked backing-fd guard last.
    drop(guest_mem);
    drop(reservation);
    drop(cow_guards);
}

const PREPARED_MULTIRANGE_CHILD_TEST: &str =
    "vmm::setup::tests::prepared_multirange_cross_process_child";
const PREPARED_MULTIRANGE_ROOT_ENV: &str = "KTSTR_PREPARED_MULTIRANGE_ROOT";
const PREPARED_MULTIRANGE_INDEX_ENV: &str = "KTSTR_PREPARED_MULTIRANGE_INDEX";
const PREPARED_MULTIRANGE_CHILDREN: usize = 2;

struct PreparedMultirangeChildren(Vec<Option<std::process::Child>>);

impl PreparedMultirangeChildren {
    fn spawn(root: &std::path::Path) -> Self {
        let mut children = Vec::with_capacity(PREPARED_MULTIRANGE_CHILDREN);
        for index in 0..PREPARED_MULTIRANGE_CHILDREN {
            let child = std::process::Command::new(std::env::current_exe().unwrap())
                .arg("--exact")
                .arg(PREPARED_MULTIRANGE_CHILD_TEST)
                .arg("--nocapture")
                .arg("--test-threads=1")
                .env(PREPARED_MULTIRANGE_ROOT_ENV, root)
                .env(PREPARED_MULTIRANGE_INDEX_ENV, index.to_string())
                .env(crate::KTSTR_CACHE_DIR_ENV, root.join("cache"))
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::inherit())
                .spawn()
                .unwrap();
            children.push(Some(child));
        }
        Self(children)
    }

    fn assert_running(&mut self, stage: &str) {
        for child in &mut self.0 {
            if let Some(process) = child
                && let Some(status) = process.try_wait().unwrap()
            {
                panic!("prepared multirange child exited during {stage} with {status}");
            }
        }
    }

    fn wait_all_success(&mut self, timeout: std::time::Duration) {
        let deadline = std::time::Instant::now() + timeout;
        while self.0.iter().any(Option::is_some) {
            for child in &mut self.0 {
                let Some(process) = child.as_mut() else {
                    continue;
                };
                if let Some(status) = process.try_wait().unwrap() {
                    assert!(
                        status.success(),
                        "prepared multirange child exited with {status}"
                    );
                    *child = None;
                }
            }
            assert!(
                std::time::Instant::now() < deadline,
                "prepared multirange children did not finish before timeout"
            );
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
    }
}

impl Drop for PreparedMultirangeChildren {
    fn drop(&mut self) {
        for process in self.0.iter_mut().flatten() {
            let _ = process.kill();
            let _ = process.wait();
        }
    }
}

fn prepared_multirange_wait_for_markers(
    directory: &std::path::Path,
    expected: usize,
    children: &mut PreparedMultirangeChildren,
    stage: &str,
) {
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
    loop {
        let count = std::fs::read_dir(directory).unwrap().count();
        if count == expected {
            return;
        }
        children.assert_running(stage);
        assert!(
            std::time::Instant::now() < deadline,
            "only {count}/{expected} prepared multirange children reached {stage}"
        );
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
}

fn prepared_multirange_wait_for_path(path: &std::path::Path, stage: &str) {
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
    while !path.exists() {
        assert!(
            std::time::Instant::now() < deadline,
            "prepared multirange child timed out waiting for {stage}: {}",
            path.display()
        );
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
}

fn prepared_multirange_wait_on_lock(path: &std::path::Path) {
    let lock = std::fs::File::open(path).unwrap();
    rustix::fs::flock(&lock, rustix::fs::FlockOperation::LockShared).unwrap();
}

fn prepared_multirange_pattern(len: usize, seed: usize) -> Vec<u8> {
    (0..len)
        .map(|index| ((index.wrapping_mul(131).wrapping_add(seed) % 251).wrapping_add(1)) as u8)
        .collect()
}

fn prepared_multirange_read_exact_at(file: &std::fs::File, mut offset: u64, mut buffer: &mut [u8]) {
    while !buffer.is_empty() {
        match file.read_at(buffer, offset) {
            Ok(0) => panic!("prepared multirange backing was truncated at {offset:#x}"),
            Ok(read) => {
                offset += read as u64;
                buffer = &mut buffer[read..];
            }
            Err(error) if error.kind() == std::io::ErrorKind::Interrupted => {}
            Err(error) => panic!("read prepared multirange backing at {offset:#x}: {error}"),
        }
    }
}

fn prepared_multirange_hash_update(mut hash: u64, bytes: &[u8]) -> u64 {
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100_0000_01b3);
    }
    hash
}

struct PreparedMultirangeBackingObservation {
    label: String,
    dev: u64,
    ino: u64,
    file_offset: u64,
    guest_offset: u64,
    map_len: usize,
    sample_file_offset: u64,
    sample_guest_offset: u64,
    original: u8,
    private: u8,
    observer: std::fs::File,
}

struct PreparedMultirangeObservation {
    index: usize,
    primary: PreparedMultirangeBackingObservation,
    overlays: Vec<PreparedMultirangeBackingObservation>,
}

fn prepared_multirange_backing_observation(
    label: String,
    fd: &OwnedFd,
    file_offset: u64,
    guest_offset: u64,
    map_len: usize,
    sample_relative_offset: usize,
    child_index: usize,
) -> PreparedMultirangeBackingObservation {
    assert!(
        sample_relative_offset < map_len,
        "{label} sample lies outside its mapping"
    );
    let stat = rustix::fs::fstat(fd).unwrap();
    let observer = std::fs::File::from(fd.try_clone().unwrap());
    let sample_file_offset = file_offset + sample_relative_offset as u64;
    let sample_guest_offset = guest_offset + sample_relative_offset as u64;
    let mut original = [0u8; 1];
    prepared_multirange_read_exact_at(&observer, sample_file_offset, &mut original);
    PreparedMultirangeBackingObservation {
        label,
        dev: stat.st_dev,
        ino: stat.st_ino,
        file_offset,
        guest_offset,
        map_len,
        sample_file_offset,
        sample_guest_offset,
        original: original[0],
        private: original[0].wrapping_add(child_index as u8 + 1),
        observer,
    }
}

fn prepared_multirange_observations(
    ranges: &[PreparedMapping],
    child_index: usize,
    host_page: usize,
) -> Vec<PreparedMultirangeObservation> {
    ranges
        .iter()
        .enumerate()
        .map(|(index, range)| {
            let primary_sample = (0..range.map_len)
                .step_by(host_page)
                .find(|relative| {
                    let guest = range.guest_offset + *relative as u64;
                    range.overlays.iter().all(|overlay| {
                        guest < overlay.guest_offset
                            || guest >= overlay.guest_offset + overlay.map_len as u64
                    })
                })
                .unwrap_or_else(|| {
                    panic!("prepared range {index} has no primary-visible page to sample")
                });
            let primary = prepared_multirange_backing_observation(
                format!("range {index} primary"),
                &range.fd,
                range.file_offset,
                range.guest_offset,
                range.map_len,
                primary_sample,
                child_index,
            );
            let overlays = range
                .overlays
                .iter()
                .enumerate()
                .map(|(overlay_index, overlay)| {
                    prepared_multirange_backing_observation(
                        format!("range {index} overlay {overlay_index}"),
                        &overlay.fd,
                        overlay.file_offset,
                        overlay.guest_offset,
                        overlay.map_len,
                        0,
                        child_index,
                    )
                })
                .collect();
            PreparedMultirangeObservation {
                index,
                primary,
                overlays,
            }
        })
        .collect()
}

fn prepared_multirange_assert_complete_mapping(
    guest_mem: &GuestMemoryMmap,
    observation: &PreparedMultirangeObservation,
) -> u64 {
    let primary = &observation.primary;
    let mut expected = vec![0u8; primary.map_len];
    prepared_multirange_read_exact_at(&primary.observer, primary.file_offset, &mut expected);
    for overlay in &observation.overlays {
        let destination = usize::try_from(overlay.guest_offset - primary.guest_offset).unwrap();
        prepared_multirange_read_exact_at(
            &overlay.observer,
            overlay.file_offset,
            &mut expected[destination..destination + overlay.map_len],
        );
    }
    let mut actual = vec![0u8; primary.map_len];
    guest_mem
        .read_slice(&mut actual, GuestAddress(primary.guest_offset))
        .unwrap();
    if expected != actual {
        let mismatch = expected
            .iter()
            .zip(&actual)
            .position(|(expected, actual)| expected != actual)
            .unwrap();
        panic!(
            "prepared range {} differs from its primary-plus-overlay composition at \
             logical offset {mismatch:#x}: expected {:#04x}, got {:#04x}",
            observation.index, expected[mismatch], actual[mismatch]
        );
    }
    prepared_multirange_hash_update(0xcbf2_9ce4_8422_2325, &expected)
}

fn prepared_multirange_assert_guest_samples(
    guest_mem: &GuestMemoryMmap,
    observations: &[PreparedMultirangeObservation],
    private: bool,
) {
    for observation in observations {
        for backing in std::iter::once(&observation.primary).chain(&observation.overlays) {
            let mut actual = [0u8; 1];
            guest_mem
                .read_slice(&mut actual, GuestAddress(backing.sample_guest_offset))
                .unwrap();
            let expected = if private {
                backing.private
            } else {
                backing.original
            };
            assert_eq!(
                actual[0],
                expected,
                "{} has the wrong {} sample",
                backing.label,
                if private { "private" } else { "shared" }
            );
        }
    }
}

fn prepared_multirange_assert_backing_samples(observations: &[PreparedMultirangeObservation]) {
    for observation in observations {
        for backing in std::iter::once(&observation.primary).chain(&observation.overlays) {
            let mut actual = [0u8; 1];
            prepared_multirange_read_exact_at(
                &backing.observer,
                backing.sample_file_offset,
                &mut actual,
            );
            assert_eq!(
                actual[0], backing.original,
                "MAP_PRIVATE write changed {}",
                backing.label
            );
        }
    }
}

fn prepared_multirange_write_private_samples(
    guest_mem: &GuestMemoryMmap,
    observations: &[PreparedMultirangeObservation],
) {
    for observation in observations {
        for backing in std::iter::once(&observation.primary).chain(&observation.overlays) {
            guest_mem
                .write_slice(
                    &[backing.private],
                    GuestAddress(backing.sample_guest_offset),
                )
                .unwrap();
        }
    }
}

fn prepared_multirange_child_result(
    plan: super::super::initramfs_cache::PreparedRangePlan,
    observations: &[PreparedMultirangeObservation],
    hashes: &[u64],
) -> String {
    use std::fmt::Write as _;

    let backing_count: usize = observations
        .iter()
        .map(|observation| 1 + observation.overlays.len())
        .sum();
    let mut result = format!(
        "plan {} {} {} {} {}\n",
        plan.part_count,
        plan.direct_ranges,
        plan.stitch_pages,
        observations.len(),
        backing_count,
    );
    for (observation, hash) in observations.iter().zip(hashes) {
        let primary = &observation.primary;
        writeln!(
            result,
            "range {} primary {} {} {} {} {} {hash:016x}",
            observation.index,
            primary.dev,
            primary.ino,
            primary.file_offset,
            primary.guest_offset,
            primary.map_len,
        )
        .unwrap();
        for (overlay_index, overlay) in observation.overlays.iter().enumerate() {
            writeln!(
                result,
                "range {} overlay {} {} {} {} {} {}",
                observation.index,
                overlay_index,
                overlay.dev,
                overlay.ino,
                overlay.file_offset,
                overlay.guest_offset,
                overlay.map_len,
            )
            .unwrap();
        }
    }
    result
}

#[test]
fn prepared_multirange_cross_process_child() {
    let Some(root) = std::env::var_os(PREPARED_MULTIRANGE_ROOT_ENV).map(PathBuf::from) else {
        return;
    };
    let child_index: usize = std::env::var(PREPARED_MULTIRANGE_INDEX_ENV)
        .unwrap()
        .parse()
        .unwrap();
    assert!(child_index < PREPARED_MULTIRANGE_CHILDREN);

    std::fs::write(root.join("ready").join(child_index.to_string()), b"ready").unwrap();
    prepared_multirange_wait_on_lock(&root.join("build-start"));

    let payload = root.join("inputs/payload");
    let include = root.join("inputs/base-content");
    let module = root.join("inputs/test-module.ko");
    let includes = [("fixture/base-content".to_string(), include)];
    let inputs = prepare_base_inputs(&payload, &[], &includes, None).unwrap();
    let prepared_base =
        get_or_prepare_base(inputs, initramfs::InitrdCompression::Uncompressed).unwrap();
    let modules = [module];
    let params = initramfs::SuffixParams {
        payload: Some(&payload),
        kernel_modules: &modules,
        ..Default::default()
    };
    let prepared = complete_prepared_initrd(prepared_base, &params).unwrap();
    let plan = prepared.plan();
    assert_eq!(plan.part_count, 4, "base, payload, modules, and tail");
    assert!(
        plan.direct_ranges >= 2,
        "large base and module parts must each produce a direct range: {plan:?}"
    );
    assert!(
        plan.stitch_pages >= 2,
        "both immutable-part boundaries must produce stitches: {plan:?}"
    );

    let compressed_len = prepared.compressed_len();
    let mapping_granule = prepared.mapping_granule();
    let compression = prepared.compression();
    let ranges = prepared.into_ranges();
    assert_eq!(
        ranges.len(),
        plan.direct_ranges + plan.stitch_pages,
        "every direct range and stitch page must reach the loader"
    );
    assert!(
        ranges.len() >= 4,
        "fixture must exercise multiple direct and stitched mappings"
    );
    let mapped_len: usize = ranges.iter().map(|range| range.map_len).sum();
    let host_page = host_page_size() as usize;
    assert_eq!(
        validate_prepared_load(
            compressed_len,
            compression,
            mapping_granule,
            host_page,
            0,
            &ranges,
        )
        .unwrap(),
        compressed_len as u32
    );
    let observations = prepared_multirange_observations(&ranges, child_index, host_page);
    let overlay_count: usize = observations
        .iter()
        .map(|observation| observation.overlays.len())
        .sum();
    assert!(
        overlay_count > 0,
        "the cross-process fixture must exercise a nested boundary overlay"
    );
    let backing_count = observations.len() + overlay_count;

    // Guards are declared before guest memory so the MAP_FIXED VMAs are
    // unmapped before their shared CAS locks are released.
    let mut cow_guards = Vec::with_capacity(backing_count);
    let guest_mem = GuestMemoryMmap::<()>::from_ranges(&[(GuestAddress(0), mapped_len)]).unwrap();
    let validated =
        validate_prepared_subranges(&guest_mem, ranges, 0, host_page, host_page).unwrap();
    assert_eq!(validated.len(), observations.len());
    assert!(
        validated.iter().all(|range| range.subranges.len() == 1),
        "one contiguous base-page guest region must not split logical ranges"
    );
    assert!(
        validated
            .iter()
            .flat_map(|range| &range.overlays)
            .all(|overlay| overlay.subranges.len() == 1),
        "one contiguous base-page guest region must not split nested overlays"
    );
    map_validated_prepared_ranges(
        &mut cow_guards,
        validated,
        |subrange, fd| unsafe {
            initramfs::cow_overlay_file_borrowed(
                subrange.host_addr,
                subrange.len,
                fd,
                subrange.file_offset,
            )
        },
        |_| Ok(()),
    )
    .unwrap();
    assert_eq!(
        cow_guards.len(),
        backing_count,
        "every primary and nested overlay must retain its own locked CAS fd"
    );

    let hashes: Vec<u64> = observations
        .iter()
        .map(|observation| prepared_multirange_assert_complete_mapping(&guest_mem, observation))
        .collect();
    prepared_multirange_assert_guest_samples(&guest_mem, &observations, false);
    prepared_multirange_assert_backing_samples(&observations);

    // This is the second barrier, after preparation and after every CAS range
    // has been mapped and faulted. Both independent processes therefore keep
    // the same clean file pages live while the ordered COW exchange runs.
    std::fs::write(root.join("mapped").join(child_index.to_string()), b"mapped").unwrap();
    prepared_multirange_wait_on_lock(&root.join("cow-start"));

    let writer_done = root.join("writer-done");
    let observer_written = root.join("observer-written");
    let writer_confirmed = root.join("writer-confirmed");
    if child_index == 0 {
        prepared_multirange_write_private_samples(&guest_mem, &observations);
        prepared_multirange_assert_guest_samples(&guest_mem, &observations, true);
        prepared_multirange_assert_backing_samples(&observations);
        std::fs::write(&writer_done, b"writer-done").unwrap();

        prepared_multirange_wait_for_path(&observer_written, "observer COW writes");
        prepared_multirange_assert_guest_samples(&guest_mem, &observations, true);
        prepared_multirange_assert_backing_samples(&observations);
        std::fs::write(&writer_confirmed, b"writer-confirmed").unwrap();
    } else {
        prepared_multirange_wait_for_path(&writer_done, "writer COW writes");
        prepared_multirange_assert_guest_samples(&guest_mem, &observations, false);
        prepared_multirange_assert_backing_samples(&observations);

        prepared_multirange_write_private_samples(&guest_mem, &observations);
        prepared_multirange_assert_guest_samples(&guest_mem, &observations, true);
        prepared_multirange_assert_backing_samples(&observations);
        std::fs::write(&observer_written, b"observer-written").unwrap();

        prepared_multirange_wait_for_path(&writer_confirmed, "writer isolation check");
        prepared_multirange_assert_guest_samples(&guest_mem, &observations, true);
        prepared_multirange_assert_backing_samples(&observations);
    }

    let result = prepared_multirange_child_result(plan, &observations, &hashes);
    std::fs::write(root.join("results").join(child_index.to_string()), result).unwrap();

    // Retain the production lifetime order explicitly.
    drop(guest_mem);
    drop(cow_guards);
}

#[test]
fn prepared_multirange_cross_process_reuses_every_cas_range_and_cow_isolates() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path();
    for directory in ["inputs", "ready", "mapped", "results"] {
        std::fs::create_dir(root.join(directory)).unwrap();
    }

    let fixture_len = 2 * PREPARED_MAPPING_GRANULE + (host_page_size() as usize);
    std::fs::write(
        root.join("inputs/payload"),
        b"non-ELF payload used only to shape a prepared archive",
    )
    .unwrap();
    std::fs::write(
        root.join("inputs/base-content"),
        prepared_multirange_pattern(fixture_len, 17),
    )
    .unwrap();
    std::fs::write(
        root.join("inputs/test-module.ko"),
        prepared_multirange_pattern(fixture_len, 83),
    )
    .unwrap();

    let build_start = std::fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(root.join("build-start"))
        .unwrap();
    let cow_start = std::fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(root.join("cow-start"))
        .unwrap();
    rustix::fs::flock(&build_start, rustix::fs::FlockOperation::LockExclusive).unwrap();
    rustix::fs::flock(&cow_start, rustix::fs::FlockOperation::LockExclusive).unwrap();

    let mut children = PreparedMultirangeChildren::spawn(root);
    prepared_multirange_wait_for_markers(
        &root.join("ready"),
        PREPARED_MULTIRANGE_CHILDREN,
        &mut children,
        "build-start barrier",
    );
    rustix::fs::flock(&build_start, rustix::fs::FlockOperation::Unlock).unwrap();

    prepared_multirange_wait_for_markers(
        &root.join("mapped"),
        PREPARED_MULTIRANGE_CHILDREN,
        &mut children,
        "live-mapping barrier",
    );
    rustix::fs::flock(&cow_start, rustix::fs::FlockOperation::Unlock).unwrap();
    children.wait_all_success(std::time::Duration::from_secs(60));

    let writer = std::fs::read_to_string(root.join("results/0")).unwrap();
    let observer = std::fs::read_to_string(root.join("results/1")).unwrap();
    assert_eq!(
        writer, observer,
        "independent processes must map every range from the same CAS \
         inode, file offset, guest offset, length, and content"
    );
    assert!(
        writer.lines().any(|line| line.contains(" overlay ")),
        "fixture must report at least one cross-process-reused overlay CAS object:\n{writer}"
    );
    assert!(
        writer.lines().count() >= 6,
        "fixture must report one plan, at least four primary CAS ranges, and an overlay:\n{writer}"
    );
}

#[test]
fn prepared_mapper_keeps_primary_and_overlay_guards_on_overlay_failures() {
    let page = host_page_size() as usize;
    let validated_fixture = || {
        let primary = test_prepared_mapping(&initramfs::LZ4_LEGACY_MAGIC, 0, 2 * page);
        let overlay = test_prepared_overlay(&[], 0, 2 * page);
        vec![ValidatedPreparedRange {
            range: primary,
            subranges: vec![ValidatedPreparedSubrange {
                guest_addr: 0,
                host_addr: page as *mut u8,
                file_offset: 0,
                len: 2 * page,
            }],
            overlays: vec![ValidatedPreparedOverlay {
                overlay,
                subranges: vec![
                    ValidatedPreparedSubrange {
                        guest_addr: 0,
                        host_addr: page as *mut u8,
                        file_offset: 0,
                        len: page,
                    },
                    ValidatedPreparedSubrange {
                        guest_addr: page as u64,
                        host_addr: (2 * page) as *mut u8,
                        file_offset: page as u64,
                        len: page,
                    },
                ],
            }],
        }]
    };

    let mut guards = Vec::new();
    let mut map_calls = 0usize;
    let error = map_validated_prepared_ranges(
        &mut guards,
        validated_fixture(),
        |_, _| {
            map_calls += 1;
            anyhow::ensure!(map_calls < 3, "injected overlay MAP_FIXED failure");
            Ok(())
        },
        |_| Ok(()),
    )
    .unwrap_err();
    assert_eq!(
        map_calls, 3,
        "the mapper must install the primary and first overlay subrange before failing"
    );
    assert_eq!(
        guards.len(),
        2,
        "the completed primary and partially live overlay must retain both backing fds"
    );
    assert!(
        format!("{error:#}").contains("injected overlay MAP_FIXED failure"),
        "the strict mapper must return the overlay mapping failure without a copy fallback: \
         {error:#}"
    );
    assert!(
        format!("{error:#}").contains("prepared initrd overlay subrange"),
        "the partial-map error must identify the nested overlay: {error:#}"
    );

    let mut guards = Vec::new();
    let mut map_calls = 0usize;
    let mut numa_calls = 0usize;
    let error = map_validated_prepared_ranges(
        &mut guards,
        validated_fixture(),
        |_, _| {
            map_calls += 1;
            Ok(())
        },
        |_| {
            numa_calls += 1;
            anyhow::ensure!(numa_calls < 2, "injected overlay NUMA restore failure");
            Ok(())
        },
    )
    .unwrap_err();
    assert_eq!(
        map_calls, 2,
        "NUMA failure must occur after the primary and first overlay mapping"
    );
    assert_eq!(numa_calls, 2);
    assert_eq!(
        guards.len(),
        2,
        "the completed primary and NUMA-failed overlay must retain both backing fds"
    );
    assert!(
        format!("{error:#}").contains("injected overlay NUMA restore failure"),
        "the mapper must return the overlay NUMA failure: {error:#}"
    );
    assert!(
        format!("{error:#}").contains("prepared initrd overlay subrange"),
        "the NUMA error must identify the nested overlay: {error:#}"
    );
}

#[test]
fn prepared_mapper_keeps_guard_on_partial_primary_map() {
    let page = host_page_size() as usize;
    let subranges = vec![
        ValidatedPreparedSubrange {
            guest_addr: 0,
            host_addr: page as *mut u8,
            file_offset: 0,
            len: page,
        },
        ValidatedPreparedSubrange {
            guest_addr: page as u64,
            host_addr: (2 * page) as *mut u8,
            file_offset: page as u64,
            len: page,
        },
    ];
    let validated = vec![ValidatedPreparedRange {
        range: test_prepared_mapping(&initramfs::LZ4_LEGACY_MAGIC, 0, 2 * page),
        subranges,
        overlays: Vec::new(),
    }];
    let mut guards = Vec::new();
    let mut map_calls = 0usize;
    let error = map_validated_prepared_ranges(
        &mut guards,
        validated,
        |_, _| {
            map_calls += 1;
            anyhow::ensure!(map_calls < 2, "injected MAP_FIXED failure");
            Ok(())
        },
        |_| Ok(()),
    )
    .unwrap_err();
    assert_eq!(map_calls, 2, "the mapper must stop at the first failure");
    assert_eq!(
        guards.len(),
        1,
        "a partially live range must retain its backing fd and shared lock"
    );
    assert!(
        format!("{error:#}").contains("injected MAP_FIXED failure"),
        "the strict mapper must return the mapping failure without a copy fallback: {error:#}"
    );
}

#[test]
fn prepared_split_alignment_accepts_257mib_custom_node_boundary_on_base_pages() {
    let host_page = host_page_size() as usize;
    let first_slice = 129usize << 20;
    assert_eq!(
        prepared_region_split_alignment(
            MemoryBacking::BasePages,
            host_page,
            PREPARED_MAPPING_GRANULE,
        ),
        host_page
    );
    assert!(first_slice.is_multiple_of(host_page));
    assert!(!first_slice.is_multiple_of(PREPARED_MAPPING_GRANULE));
    assert_eq!(
        prepared_region_split_alignment(
            MemoryBacking::HugeTlb2M,
            host_page,
            PREPARED_MAPPING_GRANULE,
        ),
        PREPARED_MAPPING_GRANULE
    );
}

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

#[test]
fn prepared_hugetlb_split_requires_huge_aligned_destination_not_file_offset() {
    let host_page = host_page_size() as usize;
    let huge = PREPARED_MAPPING_GRANULE;
    assert!(huge.is_multiple_of(host_page));
    assert!(
        validate_prepared_split_host_address(huge as *mut u8, huge).is_ok(),
        "a 2 MiB-aligned hugetlb replacement destination must be accepted"
    );
    if host_page < huge {
        assert!(
            validate_prepared_split_host_address(host_page as *mut u8, huge).is_err(),
            "base-page alignment alone is insufficient for a hugetlb-backed destination"
        );
        assert!(
            validate_prepared_file_offset(host_page as u64, host_page).is_ok(),
            "the ordinary-file CAS source requires only base-page offset alignment"
        );
        assert_ne!(
            host_page % huge,
            0,
            "fixture must prove that a non-2MiB file offset remains valid"
        );
    }
    assert!(
        validate_prepared_file_offset((host_page + 1) as u64, host_page).is_err(),
        "an offset that is not aligned to the runtime base page must be rejected"
    );
}

#[test]
fn prepared_nested_overlay_replaces_real_hugetlb_vma_when_available() {
    use std::io::Write as _;
    use vm_memory::mmap::{GuestRegionMmap, MmapRegion};

    struct Reservation {
        address: *mut libc::c_void,
        len: usize,
    }

    impl Drop for Reservation {
        fn drop(&mut self) {
            let _ = unsafe { libc::munmap(self.address, self.len) };
        }
    }

    let granule = PREPARED_MAPPING_GRANULE;
    let host_page = host_page_size() as usize;
    let address = unsafe {
        libc::mmap(
            std::ptr::null_mut(),
            granule,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_HUGETLB | libc::MAP_HUGE_2MB,
            -1,
            0,
        )
    };
    if address == libc::MAP_FAILED {
        skip!(
            "2 MiB MAP_HUGETLB reservation unavailable; nested replacement not applicable: {}",
            std::io::Error::last_os_error()
        );
    }

    let mut guards = Vec::new();
    let reservation = Reservation {
        address,
        len: granule,
    };
    let host_addr = address.cast::<u8>();
    assert_eq!((host_addr as usize) % granule, 0);

    let mut primary_file = tempfile::tempfile().unwrap();
    let mut primary_bytes = vec![0x4a; granule];
    primary_bytes[..initramfs::LZ4_LEGACY_MAGIC.len()]
        .copy_from_slice(&initramfs::LZ4_LEGACY_MAGIC);
    primary_file.write_all(&primary_bytes).unwrap();
    rustix::fs::flock(&primary_file, rustix::fs::FlockOperation::LockShared).unwrap();

    let overlay_guest_offset = granule - host_page;
    let mut overlay_file = tempfile::tempfile().unwrap();
    let overlay_bytes = vec![0xc7; host_page];
    overlay_file.write_all(&overlay_bytes).unwrap();
    rustix::fs::flock(&overlay_file, rustix::fs::FlockOperation::LockShared).unwrap();

    let range = PreparedMapping {
        fd: primary_file.into(),
        file_offset: 0,
        guest_offset: 0,
        map_len: granule,
        overlays: vec![PreparedOverlay {
            fd: overlay_file.into(),
            file_offset: 0,
            guest_offset: overlay_guest_offset as u64,
            map_len: host_page,
        }],
    };
    validate_prepared_load(
        granule,
        initramfs::InitrdCompression::Lz4,
        granule,
        host_page,
        0,
        std::slice::from_ref(&range),
    )
    .unwrap();

    let region = unsafe {
        MmapRegion::build_raw(
            host_addr,
            granule,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_HUGETLB | libc::MAP_HUGE_2MB,
        )
        .unwrap()
    };
    let guest_mem =
        GuestMemoryMmap::from_regions(vec![GuestRegionMmap::new(region, GuestAddress(0)).unwrap()])
            .unwrap();
    let validated =
        validate_prepared_subranges(&guest_mem, vec![range], 0, granule, host_page).unwrap();
    map_validated_prepared_ranges(
        &mut guards,
        validated,
        |subrange, fd| unsafe {
            initramfs::cow_overlay_file_borrowed(
                subrange.host_addr,
                subrange.len,
                fd,
                subrange.file_offset,
            )
        },
        |_| Ok(()),
    )
    .expect("replace the complete hugetlb VMA before its host-page overlay");

    assert_eq!(guards.len(), 2);
    let mut actual = [0u8; 1];
    guest_mem.read_slice(&mut actual, GuestAddress(0)).unwrap();
    assert_eq!(actual, [initramfs::LZ4_LEGACY_MAGIC[0]]);
    guest_mem
        .read_slice(&mut actual, GuestAddress(overlay_guest_offset as u64))
        .unwrap();
    assert_eq!(
        actual,
        [0xc7],
        "the host-page overlay must remain visible after replacing the hugetlb VMA"
    );

    drop(guest_mem);
    drop(reservation);
    drop(guards);
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
        // Leave two mapping granules of slack: both the ceiling and mapped
        // extent round on 2 MiB boundaries.
        let max = pvt - DRAM_START - (2 * PREPARED_MAPPING_GRANULE as u64);
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

/// Drive the legacy test-only SHM compatibility helper against a live LZ4
/// segment and a two-region `GuestMemoryMmap`. The overlay must succeed (return
/// `Some`), map the SHM bytes into region A, and leave the adjacent
/// marker region B byte-for-byte untouched. Exercises that compatibility
/// path — `shm_open_lz4`, the LZ4-magic pread validation, the rounded-
/// length bounds check, and the `MAP_FIXED` overlay via `cow_overlay` —
/// not the production prepared-CAS loader.
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

/// Drive the legacy test-only SHM compatibility helper with a request whose
/// rounded length overruns region A into the inter-region gap. Its bounds check
/// (`get_slice` on the rounded length) must reject it: `try_cow_overlay`
/// returns `None`, never invokes `MAP_FIXED`, and the adjacent marker
/// region survives. Unlike the dependency-contract pin in
/// `initramfs_tests.rs` (which calls `get_slice` directly), this routes
/// through the compatibility function, so dropping the guard or bounds-
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

/// Drive the legacy test-only SHM compatibility helper against a stored segment
/// whose length matches `expected_len` but whose first 4 bytes are NOT the
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

/// Drive the legacy test-only SHM compatibility helper with a
/// non-host-page-aligned `load_addr`. The alignment gate
/// (`if load_addr & (host_page - 1) !=
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

/// `aarch64_initrd_addr` aligns both the ceiling and mapped extent to the
/// uniform 2 MiB prepared-object granule. This keeps every range/file offset
/// directly mappable and valid when the underlying guest VMA is hugetlb.
#[cfg(target_arch = "aarch64")]
#[test]
fn aarch64_initrd_addr_returns_mapping_granule_aligned_address() {
    use crate::vmm::aarch64::fdt::pvtime_base;
    let granule = PREPARED_MAPPING_GRANULE as u64;
    for &(mem, cpus) in &[(512u32, 2u32), (512, 8), (2048, 256), (4096, 512)] {
        let pvt = pvtime_base(mem, cpus);
        let size = 7_000_123;
        let load = aarch64_initrd_addr(mem, cpus, size)
            .expect("non-oversized initrd must produce a load address");
        assert_eq!(
            load & (granule - 1),
            0,
            "initrd load addr {load:#x} not prepared-granule-aligned \
             (mem={mem} cpus={cpus} size={size:#x} granule={granule:#x})",
        );
        let mapped = (size + granule - 1) & !(granule - 1);
        assert_eq!(load + mapped, pvt & !(granule - 1));
    }
}

/// Pin the EXACT `aarch64_initrd_addr` arithmetic against an
/// independently-computed reference:
/// `align_down(pvtime_base, 2MiB) - align_up(size, 2MiB)`.
#[cfg(target_arch = "aarch64")]
#[test]
fn aarch64_initrd_addr_exact_value_for_aligned_fit() {
    use crate::vmm::aarch64::fdt::pvtime_base;
    let granule = PREPARED_MAPPING_GRANULE as u64;
    for &(mem, cpus, size) in &[(512u32, 2u32, 1u64 << 20), (2048, 256, 7_000_000)] {
        let aligned_ceiling = pvtime_base(mem, cpus) & !(granule - 1);
        let mapped = (size + granule - 1) & !(granule - 1);
        let expected = aligned_ceiling - mapped;
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
