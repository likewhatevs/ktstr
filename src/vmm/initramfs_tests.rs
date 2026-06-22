//! Unit tests for [`super`] (the `initramfs` module).
//! Co-located via the `tests` submodule pattern.

#![cfg(test)]

use super::*;

/// Thin test wrapper over [`build_suffix`] that takes only the
/// args tests commonly vary. The remaining [`SuffixParams`] fields
/// default to empty, so assertions stay focused on the args- and
/// sched-args-only suffix shape.
fn build_suffix_args(base_len: usize, args: &[String], sched_args: &[String]) -> Result<Vec<u8>> {
    build_suffix(
        base_len,
        &SuffixParams {
            args,
            sched_args,
            ..Default::default()
        },
    )
}

/// Thin test wrapper that produces a complete cpio newc archive by
/// concatenating [`build_initramfs_base`] and [`build_suffix`] output.
/// Passes `payload` to the suffix so the assembled archive carries
/// `/init` (the base no longer does). Production callers build base and
/// suffix separately so they can stream the parts into guest memory
/// without an intermediate `Vec`; the monolithic form is only needed for
/// round-trip archive-shape assertions in tests.
fn build_initramfs(
    payload: &Path,
    extra_binaries: &[(&str, &Path)],
    args: &[String],
) -> Result<Vec<u8>> {
    let base = build_initramfs_base(payload, extra_binaries, &[], None)?;
    let suffix = build_suffix(
        base.len(),
        &SuffixParams {
            payload: Some(payload),
            args,
            ..Default::default()
        },
    )?;
    let mut archive = Vec::with_capacity(base.len() + suffix.len());
    archive.extend_from_slice(&base);
    archive.extend_from_slice(&suffix);
    Ok(archive)
}

/// Extract cpio entry names from a newc archive for test assertions.
fn cpio_entry_names(archive: &[u8]) -> Vec<String> {
    let mut names = Vec::new();
    let mut remaining: &[u8] = archive;
    while let Ok(reader) = cpio::newc::Reader::new(remaining) {
        let name = reader.entry().name().to_string();
        if reader.entry().is_trailer() {
            break;
        }
        names.push(name);
        remaining = reader.finish().unwrap();
    }
    names
}

/// Extract cpio entries with name, size, mode, and inode for diagnostics.
fn cpio_entries(archive: &[u8]) -> Vec<(String, u32, u32, u32)> {
    let mut entries = Vec::new();
    let mut remaining: &[u8] = archive;
    while let Ok(reader) = cpio::newc::Reader::new(remaining) {
        if reader.entry().is_trailer() {
            break;
        }
        let name = reader.entry().name().to_string();
        let size = reader.entry().file_size();
        let mode = reader.entry().mode();
        let ino = reader.entry().ino();
        entries.push((name, size, mode, ino));
        remaining = reader.finish().unwrap();
    }
    entries
}

/// Extract a single cpio entry's data bytes by name (None if absent).
fn cpio_entry_data(archive: &[u8], target: &str) -> Option<Vec<u8>> {
    let mut remaining: &[u8] = archive;
    while let Ok(mut reader) = cpio::newc::Reader::new(remaining) {
        if reader.entry().is_trailer() {
            break;
        }
        if reader.entry().name() == target {
            let mut data = Vec::new();
            std::io::Read::read_to_end(&mut reader, &mut data).ok()?;
            return Some(data);
        }
        remaining = reader.finish().unwrap();
    }
    None
}

#[test]
fn cpio_header_format() {
    let mut archive = Vec::new();
    write_entry(&mut archive, "test", b"hello", 0o100644).unwrap();
    assert_eq!(&archive[..6], b"070701");
}

#[test]
fn cpio_trailer() {
    let mut archive = Vec::new();
    write_entry(&mut archive, "test", b"data", 0o100755).unwrap();
    cpio::newc::trailer(&mut archive as &mut dyn std::io::Write).unwrap();
    let s = String::from_utf8_lossy(&archive);
    assert!(s.contains("TRAILER!!!"));
}

#[test]
fn build_initramfs_has_init() {
    let exe = crate::resolve_current_exe().unwrap();
    let initrd = build_initramfs(&exe, &[], &[]).unwrap();
    let s = String::from_utf8_lossy(&initrd);
    assert!(s.contains("init"), "should contain init entry");
    assert!(s.contains("TRAILER!!!"));
}

#[test]
fn build_initramfs_base_is_valid_cpio() {
    let exe = crate::resolve_current_exe().unwrap();
    let initrd = build_initramfs_base(&exe, &[], &[], None).unwrap();
    assert_eq!(&initrd[..6], b"070701");
    // Base is NOT 512-aligned on its own; only base+suffix is.
    let full = build_initramfs(&exe, &[], &[]).unwrap();
    assert!(initrd.len() <= full.len());
}

#[test]
fn build_initramfs_padded() {
    let exe = crate::resolve_current_exe().unwrap();
    let initrd = build_initramfs(&exe, &[], &[]).unwrap();
    assert_eq!(initrd.len() % 512, 0);
}

#[test]
fn initramfs_nonexistent_file() {
    let result = build_initramfs(Path::new("/nonexistent"), &[], &[]);
    assert!(result.is_err());
}

#[test]
fn initramfs_nonexistent_extra_binary() {
    let exe = crate::resolve_current_exe().unwrap();
    let result = build_initramfs(&exe, &[("bad", Path::new("/nonexistent"))], &[]);
    assert!(result.is_err());
}

#[test]
fn initramfs_with_args() {
    let exe = crate::resolve_current_exe().unwrap();
    let args = vec!["run".into(), "--json".into(), "scenario".into()];
    let initrd = build_initramfs(&exe, &[], &args).unwrap();
    let s = String::from_utf8_lossy(&initrd);
    assert!(s.contains("args"));
}

#[test]
fn initramfs_empty_args() {
    let exe = crate::resolve_current_exe().unwrap();
    let initrd = build_initramfs(&exe, &[], &[]).unwrap();
    assert_eq!(initrd.len() % 512, 0);
}

// -- base + suffix split tests --

#[test]
fn suffix_adds_args_and_trailer() {
    let exe = crate::resolve_current_exe().unwrap();
    let base = build_initramfs_base(&exe, &[], &[], None).unwrap();
    let args = vec!["run".into(), "--json".into()];
    let suffix = build_suffix_args(base.len(), &args, &[]).unwrap();
    let s = String::from_utf8_lossy(&suffix);
    assert!(s.contains("args"), "suffix should contain args entry");
    assert!(s.contains("TRAILER!!!"), "suffix should contain trailer");
    assert_eq!(
        (base.len() + suffix.len()) % 512,
        0,
        "base+suffix should be 512-byte aligned"
    );
}

#[test]
fn split_matches_monolithic() {
    let exe = crate::resolve_current_exe().unwrap();
    let args = vec!["run".into(), "--json".into(), "scenario".into()];
    let monolithic = build_initramfs(&exe, &[], &args).unwrap();
    let base = build_initramfs_base(&exe, &[], &[], None).unwrap();
    let suffix = build_suffix(
        base.len(),
        &SuffixParams {
            payload: Some(&exe),
            args: &args,
            ..Default::default()
        },
    )
    .unwrap();
    let mut split = Vec::with_capacity(base.len() + suffix.len());
    split.extend_from_slice(&base);
    split.extend_from_slice(&suffix);
    assert_eq!(
        monolithic, split,
        "split path should produce identical output"
    );
}

#[test]
fn suffix_different_args_differ() {
    let exe = crate::resolve_current_exe().unwrap();
    let base = build_initramfs_base(&exe, &[], &[], None).unwrap();
    let a = build_suffix_args(base.len(), &["a".into()], &[]).unwrap();
    let b = build_suffix_args(base.len(), &["b".into()], &[]).unwrap();
    assert_ne!(a, b, "different args should produce different suffixes");
}

#[test]
fn suffix_empty_args() {
    let exe = crate::resolve_current_exe().unwrap();
    let base = build_initramfs_base(&exe, &[], &[], None).unwrap();
    let suffix = build_suffix_args(base.len(), &[], &[]).unwrap();
    assert_eq!((base.len() + suffix.len()) % 512, 0);
    let s = String::from_utf8_lossy(&suffix);
    assert!(s.contains("TRAILER!!!"));
}

#[test]
fn suffix_with_sched_enable() {
    let exe = crate::resolve_current_exe().unwrap();
    let base = build_initramfs_base(&exe, &[], &[], None).unwrap();
    let sched_enable = vec!["echo 1 > /sys/kernel/sched_ext/enable".to_string()];
    let suffix = build_suffix(
        base.len(),
        &SuffixParams {
            sched_enable: &sched_enable,
            ..Default::default()
        },
    )
    .unwrap();
    let mut archive = Vec::with_capacity(base.len() + suffix.len());
    archive.extend_from_slice(&base);
    archive.extend_from_slice(&suffix);
    let entries = cpio_entries(&archive);
    let entry = entries
        .iter()
        .find(|(name, ..)| name == "sched_enable")
        .expect("sched_enable entry missing");
    assert_eq!(
        entry.1 as usize,
        sched_enable[0].len(),
        "sched_enable size should match joined content length",
    );
    // 0o100755 = S_IFREG | 0o755 (executable — it's a shell script).
    assert_eq!(entry.2, 0o100755, "sched_enable must be executable");
}

#[test]
fn suffix_with_sched_disable() {
    let exe = crate::resolve_current_exe().unwrap();
    let base = build_initramfs_base(&exe, &[], &[], None).unwrap();
    let sched_disable = vec!["echo 0 > /sys/kernel/sched_ext/enable".to_string()];
    let suffix = build_suffix(
        base.len(),
        &SuffixParams {
            sched_disable: &sched_disable,
            ..Default::default()
        },
    )
    .unwrap();
    let mut archive = Vec::with_capacity(base.len() + suffix.len());
    archive.extend_from_slice(&base);
    archive.extend_from_slice(&suffix);
    let entries = cpio_entries(&archive);
    let entry = entries
        .iter()
        .find(|(name, ..)| name == "sched_disable")
        .expect("sched_disable entry missing");
    assert_eq!(entry.1 as usize, sched_disable[0].len());
    assert_eq!(entry.2, 0o100755, "sched_disable must be executable");
}

#[test]
fn suffix_with_exec_cmd() {
    let exe = crate::resolve_current_exe().unwrap();
    let base = build_initramfs_base(&exe, &[], &[], None).unwrap();
    let cmd = "/usr/bin/stress-ng --cpu 1 --timeout 5s";
    let suffix = build_suffix(
        base.len(),
        &SuffixParams {
            exec_cmd: Some(cmd),
            ..Default::default()
        },
    )
    .unwrap();
    let mut archive = Vec::with_capacity(base.len() + suffix.len());
    archive.extend_from_slice(&base);
    archive.extend_from_slice(&suffix);
    let entries = cpio_entries(&archive);
    let entry = entries
        .iter()
        .find(|(name, ..)| name == "exec_cmd")
        .expect("exec_cmd entry missing");
    assert_eq!(entry.1 as usize, cmd.len());
    // 0o100644 = S_IFREG | 0o644 (non-executable data file — read by init, not exec'd).
    assert_eq!(entry.2, 0o100644, "exec_cmd must be a plain data file");
}

#[test]
fn suffix_omits_empty_optional_entries() {
    // Confirms the is_empty() / Option::None guards in build_suffix —
    // empty sched_enable, empty sched_disable, and None exec_cmd must
    // not leave zero-length cpio entries in the archive.
    let exe = crate::resolve_current_exe().unwrap();
    let base = build_initramfs_base(&exe, &[], &[], None).unwrap();
    let suffix = build_suffix(base.len(), &SuffixParams::default()).unwrap();
    let mut archive = Vec::with_capacity(base.len() + suffix.len());
    archive.extend_from_slice(&base);
    archive.extend_from_slice(&suffix);
    let names = cpio_entry_names(&archive);
    assert!(!names.iter().any(|n| n == "sched_enable"));
    assert!(!names.iter().any(|n| n == "sched_disable"));
    assert!(!names.iter().any(|n| n == "exec_cmd"));
    // Same guard for the staged-scheduler path: empty
    // `staged_sched_args` must NOT emit `staging/...` entries.
    assert!(!names.iter().any(|n| n.starts_with("staging/")));
}

/// Each non-empty `staged_sched_args` entry must materialize as a
/// `staging/schedulers/<name>/sched_args` cpio entry carrying the
/// joined argv. The boot-time `/sched_args` parser will read this
/// at scheduler-spawn time inside the guest — wrong path or
/// missing entry would silently spawn the staged scheduler with
/// the WRONG arg vector, masking the very mid-experiment behavior
/// the lifecycle Ops are designed to expose.
#[test]
fn suffix_emits_per_staged_scheduler_args_entries() {
    let exe = crate::resolve_current_exe().unwrap();
    let base = build_initramfs_base(&exe, &[], &[], None).unwrap();
    let staged = vec![
        (
            "mitosis_args_a".to_string(),
            vec!["--slice-us".to_string(), "5000".to_string()],
        ),
        (
            "mitosis_args_b".to_string(),
            vec!["--slice-us".to_string(), "20000".to_string()],
        ),
    ];
    let suffix = build_suffix(
        base.len(),
        &SuffixParams {
            staged_sched_args: &staged,
            ..Default::default()
        },
    )
    .unwrap();
    let mut archive = Vec::with_capacity(base.len() + suffix.len());
    archive.extend_from_slice(&base);
    archive.extend_from_slice(&suffix);
    let entries = cpio_entries(&archive);

    for (name, args) in &staged {
        let archive_path = format!("staging/schedulers/{name}/sched_args");
        let entry = entries
            .iter()
            .find(|(n, ..)| n == &archive_path)
            .unwrap_or_else(|| panic!("staged args entry missing for {archive_path}"));
        let expected = args.join("\n");
        assert_eq!(
            entry.1 as usize,
            expected.len(),
            "{archive_path} size mismatch",
        );
        assert_eq!(
            entry.2, 0o100644,
            "{archive_path} must be a plain data file (parser reads, init does not exec)",
        );
    }
}

/// A staged entry with empty `args` must be skipped — same is_empty()
/// guard shape as `sched_args` / `sched_enable`. Catches a future
/// regression that emits a zero-byte `sched_args` file the boot
/// parser would then interpret as "no args" silently overriding
/// whatever the operator intended.
#[test]
fn suffix_skips_staged_entries_with_empty_args() {
    let exe = crate::resolve_current_exe().unwrap();
    let base = build_initramfs_base(&exe, &[], &[], None).unwrap();
    let staged = vec![
        ("populated".to_string(), vec!["--flag".to_string()]),
        ("empty".to_string(), vec![]),
    ];
    let suffix = build_suffix(
        base.len(),
        &SuffixParams {
            staged_sched_args: &staged,
            ..Default::default()
        },
    )
    .unwrap();
    let mut archive = Vec::with_capacity(base.len() + suffix.len());
    archive.extend_from_slice(&base);
    archive.extend_from_slice(&suffix);
    let names = cpio_entry_names(&archive);
    assert!(
        names
            .iter()
            .any(|n| n == "staging/schedulers/populated/sched_args"),
        "populated args must emit an entry: {names:?}"
    );
    assert!(
        !names
            .iter()
            .any(|n| n == "staging/schedulers/empty/sched_args"),
        "empty args must NOT emit a zero-byte entry: {names:?}"
    );
}

#[test]
fn try_cow_overlay_rejects_cross_region_span() {
    // The bounds check in try_cow_overlay relies on
    // GuestMemoryMmap::get_slice failing when a range would cross
    // a region boundary. This test locks that semantic in: two
    // non-contiguous regions; a range that starts in region A but
    // extends past its end must be rejected. If this ever passes
    // (e.g. vm-memory swaps in multi-region get_slices semantics
    // here), try_cow_overlay's MAP_FIXED would silently clobber
    // whatever host mapping sits between the regions.
    use vm_memory::{GuestAddress, GuestMemory};
    let region_a_size: usize = 64 * 1024;
    let region_b_size: usize = 64 * 1024;
    let region_a_start: u64 = 0;
    let region_b_start: u64 = 1 << 20; // 1 MiB gap
    let mem = vm_memory::GuestMemoryMmap::<()>::from_ranges(&[
        (GuestAddress(region_a_start), region_a_size),
        (GuestAddress(region_b_start), region_b_size),
    ])
    .unwrap();

    // Range fully inside region A: must succeed.
    assert!(
        mem.get_slice(GuestAddress(region_a_start), region_a_size)
            .is_ok(),
        "full-region slice must succeed"
    );

    // Range starting mid-region-A and extending past region A's
    // end: must fail. This is the exact shape of the hazardous
    // cow_overlay case.
    let overrun_start = region_a_start + (region_a_size as u64 / 2);
    let overrun_len = region_a_size; // well past the region's end
    assert!(
        mem.get_slice(GuestAddress(overrun_start), overrun_len)
            .is_err(),
        "cross-boundary slice must fail"
    );

    // Range starting at a GPA inside the gap between regions:
    // also fails (no region covers the start address).
    let gap_addr = (region_a_start + region_a_size as u64) + 0x1000;
    assert!(
        mem.get_slice(GuestAddress(gap_addr), 4).is_err(),
        "gap-start slice must fail"
    );
}

// NOTE: the former `try_cow_overlay_preserves_adjacent_region_bytes`
// test lived here but never invoked the production `try_cow_overlay`
// (it re-implemented the bounds check inline and asserted that NOT
// writing leaves memory unchanged — a tautology). It is replaced by
// `try_cow_overlay_maps_segment_and_preserves_adjacent_region` and
// `try_cow_overlay_rejects_oversized_request_and_preserves_region`
// in `src/vmm/setup/tests.rs`, which drive the real function against
// a live LZ4 SHM segment. The `try_cow_overlay_rejects_cross_region_span`
// test below remains as the vm_memory `get_slice` dependency-contract pin.

#[test]
fn load_initramfs_parts_sequential() {
    let part1 = vec![0xAAu8; 4096];
    let part2 = vec![0xBBu8; 512];
    let mem =
        vm_memory::GuestMemoryMmap::<()>::from_ranges(&[(vm_memory::GuestAddress(0), 16 << 20)])
            .unwrap();
    let (addr, size) = load_initramfs_parts(&mem, &[&part1, &part2], 0x200000).unwrap();
    assert_eq!(addr, 0x200000);
    assert_eq!(size, 4608);
    let mut buf = vec![0u8; 4608];
    use vm_memory::{Bytes, GuestAddress};
    mem.read_slice(&mut buf, GuestAddress(0x200000)).unwrap();
    assert_eq!(&buf[..4096], &part1[..]);
    assert_eq!(&buf[4096..], &part2[..]);
}

// -- shared lib resolution tests --

#[test]
fn resolve_shared_libs_nonexistent_returns_error() {
    let result = resolve_shared_libs(Path::new("/nonexistent/binary"));
    // Nonexistent file cannot be read.
    assert!(result.is_err());
}

#[test]
fn resolve_shared_libs_non_elf_returns_empty() {
    let _tempfile_keep_alive = tempfile::Builder::new()
        .prefix("ktstr-test-resolve-nonelf-")
        .tempfile()
        .unwrap();
    let tmp = _tempfile_keep_alive.path();
    std::fs::write(tmp, b"not an elf").unwrap();
    let result = resolve_shared_libs(tmp).unwrap();
    assert!(result.found.is_empty());
    assert!(result.missing.is_empty());
}

#[test]
fn resolve_shared_libs_dynamic_binary() {
    let sh = Path::new("/bin/sh");
    if sh.exists() {
        let shared = resolve_shared_libs(sh).unwrap();
        if !shared.found.is_empty() {
            assert!(
                shared.found.iter().any(|(g, _)| g.contains("libc")),
                "dynamic binary should depend on libc: {:?}",
                shared.found
            );
            for (g, _) in &shared.found {
                assert!(!g.starts_with('/'), "guest path should be relative: {g}");
            }
        }
    }
}

#[test]
fn resolve_interpreter_deps_walks_nonstandard_interp_deps() {
    // The cache key (BaseKey::hash_shared_libs) and the base packer
    // (build_initramfs_base) both fold a non-standard interpreter's OWN
    // shared-lib deps in via resolve_interpreter_deps. A standard ld.so is
    // statically linked (no deps) and must resolve to an EMPTY set so the
    // common case adds nothing. A custom-toolchain linker can itself be
    // dynamically linked; its dep chain must be resolved so the base packs
    // it and the key hashes it (else an interp-dep change serves a stale
    // base). Model the non-standard interp with a copy of a real dynamic
    // binary at a non-standard path — it presents a DT_NEEDED chain (libc)
    // exactly like a toolchain linker would.
    let sh = Path::new("/bin/sh");
    if !sh.exists() || !is_elf(sh) {
        skip!("/bin/sh not an ELF");
    }
    let sh_libs = resolve_shared_libs(sh).unwrap();
    if sh_libs.found.is_empty() {
        skip!("/bin/sh statically linked on this host");
    }

    // The host's own (standard) ld.so is statically linked, so its dep set
    // is empty — the common case adds nothing to the hashed set.
    if let Some(std_interp) = &sh_libs.interpreter {
        let deps = resolve_interpreter_deps(std_interp).unwrap();
        assert!(
            deps.found.is_empty(),
            "standard interp {std_interp} (statically linked) must resolve to \
             no deps, got {:?}",
            deps.found
        );
    }

    // A non-standard interp (a copy of the dynamic /bin/sh at a
    // toolchain-like path) -> its own DT_NEEDED chain is walked.
    let tmp = tempfile::TempDir::new().unwrap();
    let fake_ld = tmp.path().join("ld-fake-toolchain.so");
    std::fs::copy(sh, &fake_ld).unwrap();
    let deps = resolve_interpreter_deps(fake_ld.to_str().unwrap()).unwrap();
    assert!(
        deps.found.iter().any(|(g, _)| g.contains("libc")),
        "a non-standard interpreter's own deps (libc) must be resolved so the \
         base packs them and the cache key hashes them; got {:?}",
        deps.found
    );
}

#[test]
fn elf_dynamic_needed_extracts_sonames() {
    let sh = Path::new("/bin/sh");
    if !sh.exists() || !is_elf(sh) {
        skip!("/bin/sh not ELF");
    }
    let data = std::fs::read(sh).unwrap();
    let elf = goblin::elf::Elf::parse(&data).unwrap();
    let needed: Vec<&str> = elf.libraries.clone();
    assert!(
        needed.iter().any(|n| n.contains("libc")),
        "/bin/sh should need libc: {:?}",
        needed
    );
}

#[test]
fn resolve_soname_finds_libc() {
    let result = resolve_soname("libc.so.6", &ElfSearchPaths::default(), &[]);
    assert!(
        result.is_some(),
        "should resolve libc.so.6 via default paths"
    );
    assert!(result.unwrap().is_file());
}

/// Regression for glibc-divergence bug in [`resolve_soname`]: DT_RPATH
/// must be consulted BEFORE DT_RUNPATH (via the LD_LIBRARY_PATH step),
/// and DT_RUNPATH must come BEFORE interp-relative hints.
///
/// The test populates both a unique `rpath` dir and a unique `runpath`
/// dir, each containing a distinct "library" file with the same
/// soname. The file picked must be the one from `rpath` — matching
/// glibc's "DT_RPATH before LD_LIBRARY_PATH, LD_LIBRARY_PATH before
/// DT_RUNPATH" ordering.
#[test]
fn resolve_soname_rpath_beats_runpath_when_both_present() {
    let tmp = tempfile::TempDir::new().unwrap();
    let rpath_dir = tmp.path().join("rpath");
    let runpath_dir = tmp.path().join("runpath");
    std::fs::create_dir_all(&rpath_dir).unwrap();
    std::fs::create_dir_all(&runpath_dir).unwrap();
    let soname = "libktstrfake-rpath-beats-runpath.so.1";
    std::fs::write(rpath_dir.join(soname), b"rpath-copy").unwrap();
    std::fs::write(runpath_dir.join(soname), b"runpath-copy").unwrap();

    let paths = ElfSearchPaths {
        rpath: vec![rpath_dir.clone()],
        runpath: vec![runpath_dir.clone()],
    };
    let got = resolve_soname(soname, &paths, &[]).expect("should resolve");
    assert_eq!(
        got,
        rpath_dir.join(soname),
        "DT_RPATH must be preferred over DT_RUNPATH when both are \
             populated (the LD_LIBRARY_PATH step separates them)"
    );
}

/// Regression: DT_RUNPATH must beat the interp-hint fallback,
/// otherwise a binary with a perfectly good DT_RUNPATH could pick
/// up a wrong copy from the dynamic linker's directory.
#[test]
fn resolve_soname_runpath_beats_interp_hints() {
    let tmp = tempfile::TempDir::new().unwrap();
    let runpath_dir = tmp.path().join("runpath");
    let interp_dir = tmp.path().join("interp");
    std::fs::create_dir_all(&runpath_dir).unwrap();
    std::fs::create_dir_all(&interp_dir).unwrap();
    let soname = "libktstrfake-runpath-beats-interp.so.1";
    std::fs::write(runpath_dir.join(soname), b"runpath-copy").unwrap();
    std::fs::write(interp_dir.join(soname), b"interp-copy").unwrap();

    let paths = ElfSearchPaths {
        rpath: Vec::new(),
        runpath: vec![runpath_dir.clone()],
    };
    let got =
        resolve_soname(soname, &paths, std::slice::from_ref(&interp_dir)).expect("should resolve");
    assert_eq!(
        got,
        runpath_dir.join(soname),
        "DT_RUNPATH must be searched before interp-relative hints"
    );
}

/// Legacy-binary path: when [`elf_search_paths`] populates `rpath`
/// with `runpath` empty (binary has DT_RPATH and no DT_RUNPATH),
/// DT_RPATH must resolve the soname before interp-relative hints
/// get a chance. This guards the "rpath precedes interp_hints"
/// ordering in [`resolve_soname`] for the legacy-only-RPATH case.
#[test]
fn resolve_soname_rpath_only_wins_when_runpath_empty() {
    let tmp = tempfile::TempDir::new().unwrap();
    let rpath_dir = tmp.path().join("rpath-legacy");
    let interp_dir = tmp.path().join("interp");
    std::fs::create_dir_all(&rpath_dir).unwrap();
    std::fs::create_dir_all(&interp_dir).unwrap();
    let soname = "libktstrfake-rpath-legacy.so.1";
    std::fs::write(rpath_dir.join(soname), b"rpath-copy").unwrap();
    std::fs::write(interp_dir.join(soname), b"interp-copy").unwrap();

    let paths = ElfSearchPaths {
        rpath: vec![rpath_dir.clone()],
        runpath: Vec::new(),
    };
    let got =
        resolve_soname(soname, &paths, std::slice::from_ref(&interp_dir)).expect("should resolve");
    assert_eq!(
        got,
        rpath_dir.join(soname),
        "legacy binary with DT_RPATH (no DT_RUNPATH) must resolve \
             via DT_RPATH, not interp hints"
    );
}

#[test]
fn suffix_with_sched_args() {
    let exe = crate::resolve_current_exe().unwrap();
    let base = build_initramfs_base(&exe, &[], &[], None).unwrap();
    let sched_args = vec!["--enable-borrow".into(), "--llc".into()];
    let suffix = build_suffix_args(base.len(), &[], &sched_args).unwrap();
    let s = String::from_utf8_lossy(&suffix);
    assert!(
        s.contains("sched_args"),
        "suffix should contain sched_args entry"
    );
    assert!(s.contains("TRAILER!!!"));
    assert_eq!((base.len() + suffix.len()) % 512, 0);
}

#[test]
fn suffix_without_sched_args_omits_entry() {
    let exe = crate::resolve_current_exe().unwrap();
    let base = build_initramfs_base(&exe, &[], &[], None).unwrap();
    let suffix = build_suffix_args(base.len(), &[], &[]).unwrap();
    let s = String::from_utf8_lossy(&suffix);
    assert!(
        !s.contains("sched_args"),
        "empty sched_args should not produce entry"
    );
}

#[test]
fn shm_segment_name_format() {
    let name = shm_segment_name(0xDEADBEEF);
    assert!(name.starts_with("/ktstr-base-"));
    assert!(name.contains("deadbeef"));
}

#[test]
fn is_deleted_self_returns_false_for_nonexistent() {
    assert!(!is_deleted_self(Path::new("/nonexistent/binary")));
}

#[test]
fn is_deleted_self_returns_false_for_current() {
    let exe = crate::resolve_current_exe().unwrap();
    // Current binary is not deleted.
    assert!(!is_deleted_self(&exe));
}

#[test]
fn shm_store_load_unlink_roundtrip() {
    let hash = 0xABCD_EF01_2345_6789u64;
    let data = vec![0x42u8; 1024];
    shm_store_base(hash, &data).unwrap();
    let loaded = shm_load_base(hash);
    assert!(loaded.is_some());
    assert_eq!(loaded.unwrap().as_ref(), &data[..]);
    shm_unlink_base(hash);
    // After unlink, load should return None.
    assert!(shm_load_base(hash).is_none());
}

#[test]
fn shm_load_nonexistent_returns_none() {
    let hash = 0xFFFF_FFFF_FFFF_FFFFu64;
    shm_unlink_base(hash); // ensure clean
    assert!(shm_load_base(hash).is_none());
}

#[test]
fn shm_store_last_writer_wins_even_with_size_change() {
    // Documents actual semantics: shm_store reuses the segment name,
    // so a second write with different size overwrites the first.
    // Idempotent writes (same content_hash → same contents) rely on
    // callers to derive the hash from the actual content — this test
    // deliberately uses differently-sized payloads to prove the
    // writer does NOT assume the old name's size is still valid.
    let hash = 0x1234_5678_9ABC_DEF0u64;
    let d1 = vec![0x11u8; 64];
    let d2 = vec![0x22u8; 128];
    shm_store_base(hash, &d1).unwrap();
    shm_store_base(hash, &d2).unwrap();
    let loaded = shm_load_base(hash);
    assert!(loaded.is_some());
    assert_eq!(loaded.unwrap().as_ref(), &d2[..]);
    shm_unlink_base(hash);
}

#[test]
fn shm_segment_name_unique_per_hash() {
    let n1 = shm_segment_name(0);
    let n2 = shm_segment_name(1);
    assert_ne!(n1, n2);
    assert!(n1.starts_with("/ktstr-base-"));
    assert!(n2.starts_with("/ktstr-base-"));
}

#[test]
fn shm_unlink_nonexistent_is_noop() {
    // Should not panic.
    shm_unlink_base(0xDEAD_DEAD_DEAD_DEADu64);
}

#[test]
fn mapped_shm_send_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<MappedShm>();
}

#[test]
fn shm_load_base_holds_lock_until_drop() {
    // Invariant: as long as a MappedShm is live, the SHM
    // segment's flock is held in LOCK_SH. A concurrent writer
    // calling LOCK_EX | LOCK_NB must fail with EWOULDBLOCK. Once
    // the MappedShm is dropped, the lock releases and a subsequent
    // LOCK_EX | LOCK_NB must succeed.
    //
    // This is the core invariant — if it regresses, shm_store's
    // ftruncate can race with a live reader and cause SIGBUS on
    // the mapped pages.
    let hash = 0xD0D0_BEEF_F00D_BA5Eu64;
    shm_unlink_base(hash); // clean any stale segment
    shm_store_base(hash, &vec![0x55u8; 256]).unwrap();
    let loaded = shm_load_base(hash).expect("load must succeed");

    // Open a second fd and attempt LOCK_EX|LOCK_NB. Should fail
    // with EWOULDBLOCK because the MappedShm holds LOCK_SH.
    let name = shm_segment_name(hash);
    let fd2 = rustix::shm::open(
        name.as_str(),
        rustix::shm::OFlags::RDONLY,
        rustix::fs::Mode::empty(),
    )
    .expect("second shm_open must succeed");
    let err = rustix::fs::flock(&fd2, rustix::fs::FlockOperation::NonBlockingLockExclusive);
    assert!(
        matches!(err, Err(e) if e == rustix::io::Errno::WOULDBLOCK),
        "LOCK_EX|LOCK_NB must be blocked by the live reader's LOCK_SH (got {err:?})",
    );
    drop(fd2);

    // Drop the mapping; lock releases.
    drop(loaded);

    // Now LOCK_EX|LOCK_NB must succeed on a fresh fd.
    let fd3 = rustix::shm::open(
        name.as_str(),
        rustix::shm::OFlags::RDONLY,
        rustix::fs::Mode::empty(),
    )
    .expect("third shm_open must succeed");
    rustix::fs::flock(&fd3, rustix::fs::FlockOperation::NonBlockingLockExclusive)
        .expect("LOCK_EX|LOCK_NB must succeed after the MappedShm is dropped");
    rustix::fs::flock(&fd3, rustix::fs::FlockOperation::Unlock).ok();
    drop(fd3);
    shm_unlink_base(hash);
}

#[test]
fn shm_store_skips_write_when_reader_holds_lock_sh() {
    // Regression (x86 CI wedge): shm_store takes a NON-BLOCKING
    // LOCK_EX and SKIPS the write when a reader holds LOCK_SH (a live
    // MappedShm / COW overlay). A blocking LOCK_EX starved indefinitely
    // under sustained host concurrency: Linux flock grants no writer
    // preference, so VM-lifetime LOCK_SH readers perpetually jumped the
    // queue and wedged the suite. The skip also preserves the SIGBUS
    // invariant: it never ftruncates a segment a reader is mapping.
    //
    // Same-process flock conflict: the reader's fd holds LOCK_SH and
    // shm_store opens a second fd for LOCK_EX; flock(2) treats the two
    // fds independently, so the writer genuinely contends.
    let hash = 0x5117_F10C_5117_F10Cu64;
    shm_unlink_base(hash); // clean any stale segment

    let original = [0x55u8; 256];
    shm_store_base(hash, &original).unwrap();
    let reader = shm_load_base(hash).expect("load must succeed");

    // Writer with DIFFERENT content while the reader holds LOCK_SH:
    // must return Ok WITHOUT blocking (a blocking LOCK_EX would
    // self-deadlock this thread and hang) and WITHOUT rewriting the
    // live segment.
    shm_store_base(hash, &[0xAAu8; 512]).expect("shm_store must skip (Ok), not block");

    // The reader's mapped view is untouched: the write was skipped.
    assert!(
        reader.as_ref().iter().all(|&b| b == 0x55),
        "reader bytes must be unchanged: shm_store skipped, did not rewrite/truncate",
    );
    drop(reader);

    // Once the reader releases LOCK_SH, the cache is not wedged: a
    // writer acquires LOCK_EX and the write takes effect.
    shm_store_base(hash, &[0xCCu8; 128]).expect("shm_store must write after reader drops");
    let reloaded = shm_load_base(hash).expect("reload after write");
    assert!(
        reloaded.as_ref().len() == 128 && reloaded.as_ref().iter().all(|&b| b == 0xCC),
        "post-drop write must take effect",
    );
    drop(reloaded);
    shm_unlink_base(hash);
}

#[test]
fn shm_load_lz4_roundtrips_and_rejects_non_lz4() {
    // shm_load_lz4 (the #2 shared loader): round-trips an LZ4-magic
    // segment, and rejects a segment whose first bytes are not the LZ4
    // legacy magic (a stale zstd/gzip segment or a truncated write).
    let hash = 0x0CA5_E1F2_0CA5_E1F2u64;
    let lz4_name = shm_lz4_segment_name(hash);
    let _ = rustix::shm::unlink(lz4_name.as_str()); // clean any stale segment

    // LZ4-magic-prefixed bytes round-trip store -> load.
    let mut data = LZ4_LEGACY_MAGIC.to_vec();
    data.extend_from_slice(&[0x11u8; 64]);
    shm_store_lz4(hash, &data).unwrap();
    assert_eq!(
        shm_load_lz4(hash).as_deref(),
        Some(data.as_slice()),
        "LZ4-magic segment must round-trip",
    );

    // A segment lacking the LZ4 legacy magic is rejected as stale.
    let _ = rustix::shm::unlink(lz4_name.as_str());
    shm_store_lz4(hash, &[0xDEu8; 64]).unwrap();
    assert!(
        shm_load_lz4(hash).is_none(),
        "segment lacking the LZ4 legacy magic must be rejected",
    );

    let _ = rustix::shm::unlink(lz4_name.as_str());
}

#[test]
fn strip_debug_current_exe() {
    let exe = crate::resolve_current_exe().unwrap();
    let data = strip_debug(&exe).unwrap();
    assert!(!data.is_empty());
    // Stripped binary should be an ELF (first 4 bytes = 0x7f ELF).
    assert_eq!(&data[..4], b"\x7fELF");
}

#[test]
fn strip_debug_nonexistent_fails() {
    let result = strip_debug(Path::new("/nonexistent/binary"));
    assert!(result.is_err());
}

#[test]
fn init_lives_in_suffix_not_base() {
    let exe = crate::resolve_current_exe().unwrap();
    let base = build_initramfs_base(&exe, &[], &[], None).unwrap();

    // The base must NOT carry /init or the sentinel — keeping the
    // payload's bytes out of the base is what makes it cacheable across
    // payload recompiles.
    let base_names = cpio_entry_names(&base);
    assert!(
        !base_names.iter().any(|n| n == "init"),
        "base must not contain an init entry (got {base_names:?})"
    );
    assert!(
        !base_names.iter().any(|n| n == ".ktstr_init_ok"),
        "sentinel must be in the suffix, not the base (got {base_names:?})"
    );

    // The assembled archive (base ++ suffix) carries exactly one /init,
    // mode 0755, content == strip_debug(payload).
    let suffix = build_suffix(
        base.len(),
        &SuffixParams {
            payload: Some(&exe),
            ..Default::default()
        },
    )
    .unwrap();
    let mut archive = base.clone();
    archive.extend_from_slice(&suffix);

    let inits: Vec<_> = cpio_entries(&archive)
        .into_iter()
        .filter(|(name, ..)| name == "init")
        .collect();
    assert_eq!(inits.len(), 1, "exactly one init entry; got {inits:?}");
    assert_eq!(inits[0].2 & 0o777, 0o755, "/init must be mode 0755");

    let want = strip_debug(&exe).unwrap();
    let got = cpio_entry_data(&archive, "init").expect("init entry data");
    assert_eq!(got, want, "/init content must equal strip_debug(payload)");

    // The sentinel is the LAST non-trailer entry, so its presence proves
    // the whole archive extracted without truncation.
    let names = cpio_entry_names(&archive);
    assert_eq!(
        names.last().map(String::as_str),
        Some(".ktstr_init_ok"),
        "sentinel must be the last entry (got {names:?})"
    );
}

#[test]
fn build_initramfs_base_includes_extra_shared_libs() {
    let exe = crate::resolve_current_exe().unwrap();
    let sched = crate::test_support::require_binary("scx-ktstr");
    let extras: Vec<(&str, &Path)> = vec![("scheduler", sched.as_path())];
    let base = build_initramfs_base(&exe, &extras, &[], None).unwrap();
    let s = String::from_utf8_lossy(&base);

    // Every shared lib the extra resolves to must be packed into the base.
    // Assert the resolved SET rather than a hardcoded soname: whether the
    // scheduler static- or dynamic-links libbpf/libelf is a property of
    // the scx-ktstr build, not of build_initramfs_base's lib-packing
    // (which is what this test guards).
    let resolved = resolve_shared_libs(sched.as_path()).unwrap();
    assert!(
        !resolved.found.is_empty(),
        "scx-ktstr should resolve at least its C-runtime libs"
    );
    for (guest_path, _host) in &resolved.found {
        assert!(
            s.contains(guest_path.as_str()),
            "base must contain extra's resolved lib {guest_path}; \
             resolved set: {:?}",
            resolved.found
        );
    }
}

#[test]
fn load_initramfs_to_memory() {
    let data = vec![0xAA; 4096];
    let mem =
        vm_memory::GuestMemoryMmap::<()>::from_ranges(&[(vm_memory::GuestAddress(0), 16 << 20)])
            .unwrap();
    let (addr, size) = load_initramfs_parts(&mem, &[&data], 0x200000).unwrap();
    assert_eq!(addr, 0x200000);
    assert_eq!(size, 4096);
    let mut buf = vec![0u8; 4096];
    use vm_memory::{Bytes, GuestAddress};
    mem.read_slice(&mut buf, GuestAddress(0x200000)).unwrap();
    assert_eq!(buf, data);
}

// -- include_files and busybox tests --

#[test]
fn busybox_with_include_files() {
    let exe = crate::resolve_current_exe().unwrap();
    // Per-test tempdir: TempDir generates a unique directory name
    // per invocation and cleans up on Drop. Replaces a fixed
    // `/tmp/ktstr-test-include-busybox` path that collided across
    // parallel nextest runs and left orphans after test panics.
    let tmp_dir = tempfile::TempDir::new().unwrap();
    let tmp = tmp_dir.path().join("included");
    std::fs::write(&tmp, b"hello").unwrap();
    let includes: Vec<(&str, &Path)> = vec![("include-files/test.txt", tmp.as_path())];
    let base = build_initramfs_base(
        &exe,
        &[],
        &includes,
        Some(
            b"#!/bin/sh
",
        ),
    )
    .unwrap();
    let names = cpio_entry_names(&base);
    assert!(
        names.iter().any(|n| n == "bin/busybox"),
        "busybox=true should have bin/busybox entry: {:?}",
        names
    );
}

#[test]
fn include_files_no_busybox_when_empty() {
    let exe = crate::resolve_current_exe().unwrap();
    let base = build_initramfs_base(&exe, &[], &[], None).unwrap();
    let names = cpio_entry_names(&base);
    assert!(
        !names.iter().any(|n| n == "bin/busybox"),
        "busybox=false should not have bin/busybox entry: {:?}",
        names
    );
}

#[test]
fn include_files_preserves_mode() {
    let tmp_dir = tempfile::TempDir::new().unwrap();
    let tmp = tmp_dir.path().join("script");
    std::fs::write(&tmp, b"script content").unwrap();
    // Set executable mode.
    std::fs::set_permissions(&tmp, std::fs::Permissions::from_mode(0o100755)).unwrap();

    let exe = crate::resolve_current_exe().unwrap();
    let includes: Vec<(&str, &Path)> = vec![("include-files/run.sh", tmp.as_path())];
    let base = build_initramfs_base(
        &exe,
        &[],
        &includes,
        Some(
            b"#!/bin/sh
",
        ),
    )
    .unwrap();
    // build_initramfs_base captures the host file's real mode
    // (meta.permissions().mode()) and writes the include entry with it,
    // so the cpio entry's mode must be exactly the 0o100755 we chmod'd —
    // a regression forcing every include to 0o644 (or dropping the
    // executable bit) would change this value. The name-presence is
    // covered by include_files_adds_directory_entries.
    let entries = cpio_entries(&base);
    let entry = entries
        .iter()
        .find(|(name, ..)| name == "include-files/run.sh")
        .expect("include-files/run.sh entry missing");
    assert_eq!(
        entry.2, 0o100755,
        "include file mode must be preserved (S_IFREG | 0o755)",
    );
}

#[test]
fn include_files_elf_gets_shared_libs() {
    // /bin/sh is a dynamic ELF on most systems.
    let sh = Path::new("/bin/sh");
    if !sh.exists() {
        skip!("/bin/sh not found");
    }
    if !is_elf(sh) {
        skip!("/bin/sh is not ELF");
    }
    let exe = crate::resolve_current_exe().unwrap();
    let includes: Vec<(&str, &Path)> = vec![("include-files/sh", sh)];
    let base = build_initramfs_base(
        &exe,
        &[],
        &includes,
        Some(
            b"#!/bin/sh
",
        ),
    )
    .unwrap();
    let s = String::from_utf8_lossy(&base);
    // Dynamic ELF should pull in libc shared libs.
    let shared = resolve_shared_libs(sh).unwrap();
    if !shared.found.is_empty() {
        assert!(
            shared.found.iter().any(|(g, _)| s.contains(g.as_str())),
            "include ELF shared libs should appear in archive: {:?}",
            shared.found
        );
    }
}

#[test]
fn include_files_non_elf_no_shared_libs() {
    let tmp_dir = tempfile::TempDir::new().unwrap();
    let tmp = tmp_dir.path().join("hello.sh");
    std::fs::write(&tmp, b"#!/bin/sh\necho hello\n").unwrap();
    let exe = crate::resolve_current_exe().unwrap();
    let includes: Vec<(&str, &Path)> = vec![("include-files/hello.sh", tmp.as_path())];
    // Should not fail (ELF parsing skipped for non-ELF).
    let base = build_initramfs_base(
        &exe,
        &[],
        &includes,
        Some(
            b"#!/bin/sh
",
        ),
    )
    .unwrap();
    let s = String::from_utf8_lossy(&base);
    assert!(s.contains("include-files/hello.sh"));
}

#[test]
fn include_files_adds_directory_entries() {
    let tmp_dir = tempfile::TempDir::new().unwrap();
    let tmp = tmp_dir.path().join("file.txt");
    std::fs::write(&tmp, b"data").unwrap();
    let exe = crate::resolve_current_exe().unwrap();
    let includes: Vec<(&str, &Path)> =
        vec![("include-files/subdir/nested/file.txt", tmp.as_path())];
    let base = build_initramfs_base(
        &exe,
        &[],
        &includes,
        Some(
            b"#!/bin/sh
",
        ),
    )
    .unwrap();
    let s = String::from_utf8_lossy(&base);
    assert!(s.contains("include-files"), "should have include-files dir");
    assert!(
        s.contains("include-files/subdir"),
        "should have subdir entry"
    );
    assert!(
        s.contains("include-files/subdir/nested"),
        "should have nested subdir entry"
    );
    assert!(s.contains("bin"), "should have bin dir for busybox");
}

#[test]
fn is_elf_detects_elf_binary() {
    let exe = crate::resolve_current_exe().unwrap();
    assert!(is_elf(&exe), "test binary should be ELF");
}

#[test]
fn is_elf_rejects_non_elf() {
    let tmp_dir = tempfile::TempDir::new().unwrap();
    let tmp = tmp_dir.path().join("not-elf");
    std::fs::write(&tmp, b"not an elf file").unwrap();
    assert!(!is_elf(&tmp));
}

#[test]
fn is_elf_rejects_short_file() {
    let tmp_dir = tempfile::TempDir::new().unwrap();
    let tmp = tmp_dir.path().join("short-elf");
    std::fs::write(&tmp, b"ab").unwrap();
    assert!(!is_elf(&tmp));
}

#[test]
fn is_elf_nonexistent_returns_false() {
    assert!(!is_elf(Path::new("/nonexistent/file")));
}

#[test]
fn include_files_rejects_path_traversal() {
    let tmp_dir = tempfile::TempDir::new().unwrap();
    let tmp = tmp_dir.path().join("payload");
    std::fs::write(&tmp, b"data").unwrap();
    let exe = crate::resolve_current_exe().unwrap();
    let includes: Vec<(&str, &Path)> = vec![("include-files/../etc/passwd", tmp.as_path())];
    let result = build_initramfs_base(
        &exe,
        &[],
        &includes,
        Some(
            b"#!/bin/sh
",
        ),
    );
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains(".."),
        "error should mention path traversal: {err}"
    );
}

#[test]
fn include_files_rejects_fifo() {
    let tmp_dir = tempfile::TempDir::new().unwrap();
    let fifo_path = tmp_dir.path().join("fifo");
    // Create a FIFO.
    let c_path = std::ffi::CString::new(fifo_path.to_str().unwrap()).unwrap();
    let rc = unsafe { libc::mkfifo(c_path.as_ptr(), 0o644) };
    assert_eq!(
        rc,
        0,
        "ktstr: mkfifo({}) failed -- test infrastructure broken",
        fifo_path.display(),
    );
    let exe = crate::resolve_current_exe().unwrap();
    let includes: Vec<(&str, &Path)> = vec![("include-files/pipe", fifo_path.as_path())];
    let result = build_initramfs_base(
        &exe,
        &[],
        &includes,
        Some(
            b"#!/bin/sh
",
        ),
    );
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("not a regular file"),
        "error should reject FIFO: {err}"
    );
}

#[test]
fn include_files_rejects_directory() {
    let tmp_dir = tempfile::TempDir::new().unwrap();
    let dir_path = tmp_dir.path().join("mydir");
    std::fs::create_dir(&dir_path).unwrap();
    let exe = crate::resolve_current_exe().unwrap();
    let includes: Vec<(&str, &Path)> = vec![("include-files/mydir", dir_path.as_path())];
    let result = build_initramfs_base(
        &exe,
        &[],
        &includes,
        Some(
            b"#!/bin/sh
",
        ),
    );
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("not a regular file"),
        "error should reject directory: {err}"
    );
}

#[test]
fn busybox_independent_of_include_files() {
    let exe = crate::resolve_current_exe().unwrap();
    // busybox=true but no include_files.
    let base = build_initramfs_base(
        &exe,
        &[],
        &[],
        Some(
            b"#!/bin/sh
",
        ),
    )
    .unwrap();
    let names = cpio_entry_names(&base);
    assert!(
        names.iter().any(|n| n == "bin/busybox"),
        "busybox=true should have bin/busybox entry even without includes: {:?}",
        names
    );
}

// -- ld.so.cache parsing tests --

#[test]
fn parse_ld_so_cache_finds_libc() {
    let cache = parse_ld_so_cache(Path::new("/etc/ld.so.cache"));
    // libc.so.6 is in every glibc system's ld.so.cache.
    assert!(
        cache.contains_key("libc.so.6"),
        "ld.so.cache should contain libc.so.6: found {} entries",
        cache.len(),
    );
    let path = &cache["libc.so.6"];
    assert!(
        path.is_file(),
        "cached libc path should exist: {}",
        path.display()
    );
}

#[test]
fn parse_ld_so_cache_nonexistent_returns_empty() {
    let cache = parse_ld_so_cache(Path::new("/nonexistent/ld.so.cache"));
    assert!(cache.is_empty());
}

#[test]
fn parse_ld_so_cache_bad_magic_returns_empty() {
    let tmp_dir = tempfile::TempDir::new().unwrap();
    let tmp = tmp_dir.path().join("ldcache");
    std::fs::write(&tmp, b"not a valid cache file").unwrap();
    let cache = parse_ld_so_cache(&tmp);
    assert!(cache.is_empty());
}

#[test]
fn parse_ld_so_cache_truncated_returns_empty() {
    let tmp_dir = tempfile::TempDir::new().unwrap();
    let tmp = tmp_dir.path().join("ldcache");
    // Valid magic but truncated header.
    let mut data = LD_CACHE_MAGIC.to_vec();
    data.extend_from_slice(&[0u8; 10]); // not enough for full header
    std::fs::write(&tmp, &data).unwrap();
    let cache = parse_ld_so_cache(&tmp);
    assert!(cache.is_empty());
}

#[test]
fn ld_so_cache_consistent_with_resolve_soname() {
    // If libc.so.6 is in the cache, resolve_soname should find it.
    let result = resolve_soname("libc.so.6", &ElfSearchPaths::default(), &[]);
    assert!(
        result.is_some(),
        "resolve_soname should find libc.so.6 (cache or paths)"
    );
    assert!(result.unwrap().is_file());
}

#[test]
fn no_duplicate_cpio_entries() {
    let exe = crate::resolve_current_exe().unwrap();
    let base = build_initramfs_base(&exe, &[], &[], None).unwrap();
    let entries = cpio_entries(&base);
    let mut seen = std::collections::HashSet::new();
    let mut duplicates = Vec::new();
    for (name, size, mode, ino) in &entries {
        if !seen.insert(name.clone()) {
            duplicates.push((name.clone(), *size, *mode, *ino));
        }
    }
    assert!(
        duplicates.is_empty(),
        "archive contains duplicate entries: {:?}",
        duplicates
    );
}

#[test]
fn no_duplicate_entries_with_include_files() {
    let exe = crate::resolve_current_exe().unwrap();
    // Create include files in a deeply nested path mimicking custom
    // linker library directories.
    let tmp_dir_guard = tempfile::TempDir::new().unwrap();
    let tmp_dir = tmp_dir_guard.path();
    let lib_data = vec![0xCCu8; 4096];
    let f1 = tmp_dir.join("libcustom1.so");
    let f2 = tmp_dir.join("libcustom2.so");
    let f3 = tmp_dir.join("libcustom3.so");
    std::fs::write(&f1, &lib_data).unwrap();
    std::fs::write(&f2, &lib_data).unwrap();
    std::fs::write(&f3, &lib_data).unwrap();

    let includes: Vec<(&str, &Path)> = vec![
        ("usr/local/custom/platform/lib/libcustom1.so", f1.as_path()),
        ("usr/local/custom/platform/lib/libcustom2.so", f2.as_path()),
        ("usr/local/custom/platform/lib/libcustom3.so", f3.as_path()),
    ];

    let base = build_initramfs_base(&exe, &[], &includes, None).unwrap();
    let entries = cpio_entries(&base);
    let entry_names: Vec<&str> = entries.iter().map(|(n, _, _, _)| n.as_str()).collect();

    // Check that all include files are present.
    for (archive_path, _) in &includes {
        assert!(
            entry_names.contains(archive_path),
            "missing include file entry '{}'; archive entries: {:?}",
            archive_path,
            entry_names
        );
    }

    // Check that all include files have correct size.
    for (archive_path, _) in &includes {
        let entry = entries.iter().find(|(n, _, _, _)| n == archive_path);
        assert!(
            entry.is_some_and(|(_, size, _, _)| *size == lib_data.len() as u32),
            "include file '{}' has wrong size: {:?}",
            archive_path,
            entry
        );
    }

    // Check that directory entries exist for the nested path.
    assert!(entry_names.contains(&"usr"), "missing 'usr' dir entry");
    assert!(
        entry_names.contains(&"usr/local"),
        "missing 'usr/local' dir entry"
    );
    assert!(
        entry_names.contains(&"usr/local/custom"),
        "missing 'usr/local/custom' dir entry"
    );
    assert!(
        entry_names.contains(&"usr/local/custom/platform"),
        "missing 'usr/local/custom/platform' dir entry"
    );
    assert!(
        entry_names.contains(&"usr/local/custom/platform/lib"),
        "missing 'usr/local/custom/platform/lib' dir entry"
    );

    // Check that directories come before files they contain.
    let dir_pos = entries
        .iter()
        .position(|(n, _, _, _)| n == "usr/local/custom/platform/lib")
        .unwrap();
    for (archive_path, _) in &includes {
        let file_pos = entries
            .iter()
            .position(|(n, _, _, _)| n == *archive_path)
            .unwrap();
        assert!(
            dir_pos < file_pos,
            "directory entry must precede file '{}': dir at {}, file at {}",
            archive_path,
            dir_pos,
            file_pos
        );
    }

    // No duplicate entries.
    let mut seen = std::collections::HashSet::new();
    let mut duplicates = Vec::new();
    for (name, _, _, _) in &entries {
        if !seen.insert(name.clone()) {
            duplicates.push(name.clone());
        }
    }
    assert!(
        duplicates.is_empty(),
        "duplicate entries in archive: {:?}",
        duplicates
    );
}

#[test]
fn include_elf_shared_libs_all_present_in_archive() {
    // Use /bin/sh as an include file — its shared libs must all
    // appear in the archive with non-zero sizes.
    let sh_path = Path::new("/bin/sh");
    let sh_resolved = std::fs::canonicalize(sh_path).unwrap_or_else(|_| sh_path.to_path_buf());
    let sh = sh_resolved.as_path();
    if !sh.exists() || !is_elf(sh) {
        skip!("/bin/sh not available or not ELF");
    }
    let exe = crate::resolve_current_exe().unwrap();
    let includes: Vec<(&str, &Path)> = vec![("include-files/sh", sh)];
    let base = build_initramfs_base(&exe, &[], &includes, None).unwrap();
    let entries = cpio_entries(&base);
    let entry_map: std::collections::HashMap<&str, (u32, u32, u32)> = entries
        .iter()
        .map(|(n, s, m, i)| (n.as_str(), (*s, *m, *i)))
        .collect();

    let shared = resolve_shared_libs(sh).unwrap();
    for (guest_path, _host_path) in &shared.found {
        assert!(
            entry_map.contains_key(guest_path.as_str()),
            "shared lib '{}' missing from archive; entries: {:?}",
            guest_path,
            entries
                .iter()
                .map(|(n, _, _, _)| n.as_str())
                .collect::<Vec<_>>()
        );
        let (size, _, _) = entry_map[guest_path.as_str()];
        assert!(
            size > 0,
            "shared lib '{}' has zero size in archive",
            guest_path
        );
    }

    // Check the include file itself is present.
    assert!(
        entry_map.contains_key("include-files/sh"),
        "include file itself missing from archive"
    );
}

#[test]
fn all_inode_zero_entries_have_nlink_one() {
    // Check that all entries use ino=0 and nlink=1, so the kernel
    // initramfs unpacker never enters the hardlink path.
    let exe = crate::resolve_current_exe().unwrap();
    let base = build_initramfs_base(&exe, &[], &[], None).unwrap();
    let mut remaining: &[u8] = base.as_slice();
    while let Ok(reader) = cpio::newc::Reader::new(remaining) {
        if reader.entry().is_trailer() {
            break;
        }
        let name = reader.entry().name().to_string();
        let ino = reader.entry().ino();
        let nlink = reader.entry().nlink();
        assert_eq!(
            ino, 0,
            "entry '{}' has non-zero inode {}: risk of kernel hardlink confusion",
            name, ino
        );
        assert_eq!(
            nlink, 1,
            "entry '{}' has nlink {}: kernel only hardlinks when nlink >= 2",
            name, nlink
        );
        remaining = reader.finish().unwrap();
    }
}

#[test]
fn lz4_legacy_compress_format() {
    let data = vec![0xAAu8; 4096];
    let compressed = lz4_legacy_compress(&data);
    // Must start with LZ4 legacy magic.
    assert_eq!(
        &compressed[..4],
        &LZ4_LEGACY_MAGIC,
        "output must start with LZ4 legacy magic 0x184C2102"
    );
    // First chunk: 4-byte compressed size follows magic.
    let chunk_size = u32::from_le_bytes(compressed[4..8].try_into().unwrap()) as usize;
    assert!(
        chunk_size > 0 && chunk_size < data.len(),
        "compressed chunk should be non-empty and smaller than input: {}",
        chunk_size
    );
    // Decompress and check roundtrip.
    let decompressed = lz4_flex::block::decompress(&compressed[8..8 + chunk_size], data.len())
        .expect("lz4 block decompress failed");
    assert_eq!(decompressed, data);
}

#[test]
fn lz4_legacy_compress_large_input_splits_chunks() {
    // Input larger than LZ4_CHUNK_SIZE (8MB) must produce multiple chunks.
    let data = vec![0xBBu8; LZ4_CHUNK_SIZE + 1024];
    let compressed = lz4_legacy_compress(&data);
    assert_eq!(&compressed[..4], &LZ4_LEGACY_MAGIC);
    // Parse chunks: should be at least 2.
    let mut pos = 4;
    let mut chunk_count = 0;
    let mut total_decompressed = Vec::new();
    while pos + 4 <= compressed.len() {
        let chunk_size = u32::from_le_bytes(compressed[pos..pos + 4].try_into().unwrap()) as usize;
        if chunk_size == 0 {
            break;
        }
        pos += 4;
        let remaining_uncompressed = data.len() - total_decompressed.len();
        let expected_chunk_len = remaining_uncompressed.min(LZ4_CHUNK_SIZE);
        let decompressed =
            lz4_flex::block::decompress(&compressed[pos..pos + chunk_size], expected_chunk_len)
                .expect("lz4 block decompress failed");
        total_decompressed.extend_from_slice(&decompressed);
        pos += chunk_size;
        chunk_count += 1;
    }
    assert!(
        chunk_count >= 2,
        "input > 8MB should produce >= 2 chunks, got {}",
        chunk_count
    );
    assert_eq!(total_decompressed, data);
}

#[test]
fn lz4_legacy_compress_empty_input() {
    let compressed = lz4_legacy_compress(&[]);
    // Empty input: just the magic, no chunks.
    assert_eq!(compressed, LZ4_LEGACY_MAGIC);
}

/// Build a synthetic cpio archive from generated data for LZ4 tests.
/// Uses generic paths to avoid banned terms.
fn build_synthetic_cpio(total_size: usize) -> Vec<u8> {
    let mut archive = Vec::new();
    // Directory entries.
    write_entry(&mut archive, "lib", &[], 0o40755).unwrap();
    write_entry(&mut archive, "data", &[], 0o40755).unwrap();

    // Fill with generated binary data to reach target size.
    // Use a simple PRNG for reproducible high-entropy content.
    let mut rng_state = 0x12345678u64;
    let entry_size = 256 * 1024; // 256KB per entry
    let mut entry_num = 0;
    while archive.len() + entry_size < total_size {
        let mut payload = vec![0u8; entry_size];
        for byte in &mut payload {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            *byte = (rng_state >> 33) as u8;
        }
        let name = format!("lib/test_{entry_num:04}.so");
        write_entry(&mut archive, &name, &payload, 0o100755).unwrap();
        entry_num += 1;
    }

    // Pad remaining space with a data file.
    if archive.len() < total_size {
        let remaining = total_size - archive.len() - 200; // room for header
        let remaining = remaining.min(total_size);
        let mut payload = vec![0u8; remaining];
        for byte in &mut payload {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            *byte = (rng_state >> 33) as u8;
        }
        write_entry(&mut archive, "data/fill.bin", &payload, 0o100644).unwrap();
    }

    // Trailer and padding.
    cpio::newc::trailer(&mut archive as &mut dyn std::io::Write).unwrap();
    let pad = (512 - (archive.len() % 512)) % 512;
    archive.extend(std::iter::repeat_n(0u8, pad));
    archive
}

/// Simulate the kernel's unlz4() decompression loop (non-fill path).
/// This mirrors lib/decompress_unlz4.c behavior:
///   1. Read and validate 4-byte magic (0x184C2102)
///   2. Loop: read 4-byte LE chunk size, decompress chunk, advance
///   3. Handle concatenated magic (re-encounter mid-stream)
///   4. Terminate on size < 4 or size == 0
fn simulate_kernel_unlz4(input: &[u8]) -> Result<Vec<u8>, String> {
    const UNCOMP_CHUNK_SIZE: usize = 8 << 20; // LZ4_DEFAULT_UNCOMPRESSED_CHUNK_SIZE

    if input.len() < 4 {
        return Err("input too short for magic".into());
    }

    let mut inp = 0usize; // current position
    let mut size = input.len() as isize; // remaining bytes

    // Read and validate magic.
    let magic = u32::from_le_bytes(input[inp..inp + 4].try_into().unwrap());
    if magic != 0x184C2102 {
        return Err(format!("invalid header: 0x{magic:08X}"));
    }
    inp += 4;
    size -= 4;

    let mut output = Vec::new();

    loop {
        if size < 4 {
            // End of input — clean exit.
            break;
        }

        let chunksize = u32::from_le_bytes(input[inp..inp + 4].try_into().unwrap()) as usize;

        // Handle concatenated magic mid-stream.
        if chunksize == 0x184C2102 {
            inp += 4;
            size -= 4;
            continue;
        }

        // Zero chunk size — end of stream.
        if chunksize == 0 {
            break;
        }

        inp += 4;
        size -= 4;

        // Kernel: LZ4_decompress_safe(inp, outp, chunksize, dest_len)
        // dest_len = uncomp_chunksize (8MB max output)
        let chunk_data = &input[inp..inp + chunksize];
        let decompressed = lz4_flex::block::decompress(chunk_data, UNCOMP_CHUNK_SIZE)
            .map_err(|e| format!("LZ4_decompress_safe failed: {e}"))?;

        output.extend_from_slice(&decompressed);

        size -= chunksize as isize;
        if size == 0 {
            break;
        } else if size < 0 {
            return Err("data corrupted: size went negative".into());
        }
        inp += chunksize;
    }

    Ok(output)
}

/// Roundtrip test with synthetic cpio data through the kernel's
/// unlz4() decompression logic. Uses generated test data with
/// generic paths.
#[test]
fn lz4_legacy_kernel_unlz4_roundtrip() {
    // Single chunk (< 8MB).
    let small = build_synthetic_cpio(1 << 20); // ~1MB
    let compressed = lz4_legacy_compress(&small);
    let decompressed =
        simulate_kernel_unlz4(&compressed).expect("kernel unlz4 simulation failed on small input");
    assert_eq!(decompressed, small);

    // Multi-chunk (> 8MB, forces chunk splitting).
    let large = build_synthetic_cpio(10 << 20); // ~10MB
    let compressed = lz4_legacy_compress(&large);
    let decompressed = simulate_kernel_unlz4(&compressed)
        .expect("kernel unlz4 simulation failed on multi-chunk input");
    assert_eq!(decompressed, large);
}

/// Test concatenated LZ4 legacy streams (base + suffix) through
/// the kernel unlz4 simulation. This is the format used when
/// base and suffix are compressed separately.
#[test]
fn lz4_legacy_kernel_unlz4_concatenated() {
    let base = build_synthetic_cpio(2 << 20); // ~2MB
    let suffix_data = b"arg1\narg2\narg3\n";

    let lz4_base = lz4_legacy_compress(&base);
    let lz4_suffix = lz4_legacy_compress(suffix_data);

    // Concatenate the two streams.
    let mut combined = Vec::with_capacity(lz4_base.len() + lz4_suffix.len());
    combined.extend_from_slice(&lz4_base);
    combined.extend_from_slice(&lz4_suffix);

    let decompressed = simulate_kernel_unlz4(&combined)
        .expect("kernel unlz4 simulation failed on concatenated streams");

    let mut expected = Vec::with_capacity(base.len() + suffix_data.len());
    expected.extend_from_slice(&base);
    expected.extend_from_slice(suffix_data);
    assert_eq!(decompressed, expected);
}

/// Check lz4_flex block output is decompressible by the C lz4
/// library (same decompressor as the kernel's LZ4_decompress_safe).
/// Uses synthetic cpio data with generic paths.
#[test]
fn lz4_legacy_compress_c_compat() {
    let lz4_check = std::process::Command::new("lz4").arg("--version").output();
    if lz4_check.is_err() {
        skip!("lz4 CLI not found");
    }

    let data = build_synthetic_cpio(2 << 20); // ~2MB
    let compressed = lz4_legacy_compress(&data);
    let _compressed_keep_alive = tempfile::Builder::new()
        .prefix("ktstr-test-lz4-compat-compressed-")
        .suffix(".lz4")
        .tempfile()
        .unwrap();
    let _decompressed_keep_alive = tempfile::Builder::new()
        .prefix("ktstr-test-lz4-compat-decompressed-")
        .suffix(".bin")
        .tempfile()
        .unwrap();
    let compressed_path = _compressed_keep_alive.path();
    let decompressed_path = _decompressed_keep_alive.path();
    std::fs::write(compressed_path, &compressed).unwrap();

    let output = std::process::Command::new("lz4")
        .args(["-d", "-f", "--no-frame-crc"])
        .arg(compressed_path)
        .arg(decompressed_path)
        .output()
        .expect("lz4 -d failed to execute");

    assert!(
        output.status.success(),
        "lz4 -d failed: stderr={}",
        String::from_utf8_lossy(&output.stderr),
    );

    let result = std::fs::read(decompressed_path).unwrap();
    assert_eq!(result.len(), data.len(), "decompressed size mismatch");
    assert_eq!(&result[..], &data[..], "decompressed content mismatch");
}

/// Check our output can be decompressed by `lz4 -d` when compressed
/// with `lz4 -l` as reference. Tests cross-compatibility of our
/// legacy format framing with the reference implementation.
#[test]
fn lz4_legacy_reference_cross_compat() {
    let lz4_check = std::process::Command::new("lz4").arg("--version").output();
    if lz4_check.is_err() {
        skip!("lz4 CLI not found");
    }

    let data = build_synthetic_cpio(2 << 20);

    // Compress with `lz4 -l` (reference legacy mode).
    let _input_keep_alive = tempfile::Builder::new()
        .prefix("ktstr-test-lz4-ref-input-")
        .suffix(".bin")
        .tempfile()
        .unwrap();
    let _ref_keep_alive = tempfile::Builder::new()
        .prefix("ktstr-test-lz4-ref-")
        .suffix(".lz4")
        .tempfile()
        .unwrap();
    let input_path = _input_keep_alive.path();
    let ref_path = _ref_keep_alive.path();
    std::fs::write(input_path, &data).unwrap();

    let ref_output = std::process::Command::new("lz4")
        .args(["-l", "-f"])
        .arg(input_path)
        .arg(ref_path)
        .output()
        .expect("lz4 -l failed to execute");

    assert!(
        ref_output.status.success(),
        "lz4 -l failed: stderr={}",
        String::from_utf8_lossy(&ref_output.stderr),
    );

    // Decompress reference output through our kernel simulation.
    let ref_compressed = std::fs::read(ref_path).unwrap();

    let ref_decompressed = simulate_kernel_unlz4(&ref_compressed)
        .expect("kernel unlz4 simulation failed on lz4 -l output");
    assert_eq!(
        ref_decompressed, data,
        "reference lz4 -l roundtrip mismatch"
    );

    // Also compress with our encoder, decompress with lz4 -d.
    let our_compressed = lz4_legacy_compress(&data);
    let _our_lz4_keep_alive = tempfile::Builder::new()
        .prefix("ktstr-test-lz4-ref-ours-")
        .suffix(".lz4")
        .tempfile()
        .unwrap();
    let _our_decompressed_keep_alive = tempfile::Builder::new()
        .prefix("ktstr-test-lz4-ref-ours-decompressed-")
        .suffix(".bin")
        .tempfile()
        .unwrap();
    let our_lz4_path = _our_lz4_keep_alive.path();
    let our_decompressed_path = _our_decompressed_keep_alive.path();
    std::fs::write(our_lz4_path, &our_compressed).unwrap();

    let our_output = std::process::Command::new("lz4")
        .args(["-d", "-f", "--no-frame-crc"])
        .arg(our_lz4_path)
        .arg(our_decompressed_path)
        .output()
        .expect("lz4 -d on our output failed to execute");

    assert!(
        our_output.status.success(),
        "lz4 -d on our output failed: stderr={}",
        String::from_utf8_lossy(&our_output.stderr),
    );

    let our_result = std::fs::read(our_decompressed_path).unwrap();
    assert_eq!(our_result, data, "our lz4 output cross-compat mismatch");
}

/// [`write_symlink_entry`] must emit a cpio entry with the symlink mode
/// `0o120777` (`S_IFLNK | 0777`) and store the link target verbatim as
/// the entry's data payload — the on-wire shape the kernel's cpio
/// extractor reads to materialize a symlink. No existing test packs a
/// symlink entry, so the whole function is otherwise uncovered.
#[test]
fn write_symlink_entry_emits_symlink_mode_and_target() {
    let mut archive = Vec::new();
    write_symlink_entry(&mut archive, "usr/lib64/libc.so.6", "/lib64/libc.so.6").unwrap();
    let entries = cpio_entries(&archive);
    let entry = entries
        .iter()
        .find(|(name, ..)| name == "usr/lib64/libc.so.6")
        .expect("symlink entry missing");
    // 0o120777 = S_IFLNK | 0777 — the cpio newc mode for a symlink.
    assert_eq!(entry.2, 0o120777, "symlink entry must carry S_IFLNK|0777");
    // The link target is stored verbatim as the entry's data bytes.
    assert_eq!(
        cpio_entry_data(&archive, "usr/lib64/libc.so.6").unwrap(),
        b"/lib64/libc.so.6".to_vec(),
        "symlink target must be the entry's data payload verbatim",
    );
}

/// [`register_parent_dirs`] walks `guest_path`'s parent into every
/// intermediate directory component and inserts each — but NOT the file
/// itself. A top-level file (no parent components) registers nothing.
/// The function is otherwise only exercised transitively through
/// [`build_initramfs_base`]; this pins its exact produced set.
#[test]
fn register_parent_dirs_walks_all_ancestors() {
    let mut dirs = BTreeSet::new();
    register_parent_dirs(&mut dirs, "a/b/c/f");
    assert_eq!(
        dirs,
        BTreeSet::from(["a".to_string(), "a/b".to_string(), "a/b/c".to_string(),]),
        "every intermediate dir registered, the file itself excluded",
    );

    // Top-level file: Path::new("init").parent() is Some("") whose
    // components() iterator is empty → the walk loop never inserts.
    let mut top = BTreeSet::new();
    register_parent_dirs(&mut top, "init");
    assert!(
        top.is_empty(),
        "a top-level file registers no parent dirs: {top:?}",
    );

    // Empty path: Path::new("").parent() is None → the `let Some(parent)
    // = ... else { return }` early-return arm fires; nothing inserted.
    let mut empty = BTreeSet::new();
    register_parent_dirs(&mut empty, "");
    assert!(
        empty.is_empty(),
        "an empty guest_path hits the no-parent early return: {empty:?}",
    );
}

/// [`read_cstr`] returns the UTF-8 string up to the first NUL, or `None`
/// for both failure arms: (a) no NUL terminator in `data[offset..]`, and
/// (b) bytes that are valid-terminated but not UTF-8. The
/// [`parse_ld_so_cache`] tests never reach a populated string table, so
/// these error paths are otherwise uncovered.
#[test]
fn read_cstr_rejects_invalid_utf8_and_missing_terminator() {
    // Success: stop at the NUL, ignore trailing bytes.
    assert_eq!(read_cstr(b"libc.so.6\0extra", 0), Some("libc.so.6"));
    // Missing terminator: position(NUL) is None → ?.
    assert_eq!(read_cstr(b"no-nul-here", 0), None);
    // Invalid UTF-8 before the NUL: from_utf8(...).ok() is None.
    assert_eq!(read_cstr(&[0xFF, 0xFE, 0x00], 0), None);
}

/// Build a valid glibc new-format `ld.so.cache` blob and drive it
/// through [`parse_ld_so_cache`]. Exercises the per-entry loop body that
/// existing tests skip (they hit only nonexistent/bad-magic/truncated
/// early returns): a successful soname→PathBuf insert gated on
/// `path_str.starts_with('/') && p.is_file()`, the OOB
/// `val_off >= data.len()` `continue`, and the `or_insert` first-wins
/// dedup when two entries share a soname.
#[test]
fn parse_ld_so_cache_parses_valid_entries_and_skips_oob_and_nonabsolute() {
    let tmp_dir = tempfile::TempDir::new().unwrap();
    // Two real regular files so the `p.is_file()` gate passes for BOTH
    // libfoo entries — that makes `or_insert` (not the is_file filter)
    // the sole reason the second mapping is dropped.
    let lib_path = tmp_dir.path().join("libfoo.so");
    std::fs::write(&lib_path, b"\x7fELF-stub").unwrap();
    let lib_abs = lib_path.to_str().unwrap();
    let lib2_path = tmp_dir.path().join("libfoo-second.so");
    std::fs::write(&lib2_path, b"\x7fELF-stub2").unwrap();
    let lib2_abs = lib2_path.to_str().unwrap();

    // 4 entries:
    //   A: soname "libfoo.so" -> lib_abs   (existing file, accepted)
    //   B: soname "libfoo.so" -> lib2_abs  (also existing → or_insert
    //                                        keeps A, not is_file filter)
    //   C: soname "liboob.so", val_off == data.len()  (OOB continue)
    //   D: soname "librel.so" -> "relative/libbar.so" (no leading
    //                            slash → rejected by the starts_with('/')
    //                            gate in parse_ld_so_cache before is_file)
    let nlibs: usize = 4;
    let header_end = LD_CACHE_HEADER_SIZE + nlibs * LD_CACHE_ENTRY_SIZE;

    // Lay out the string table after the entry array; record absolute
    // offsets (from file start) for each key/value.
    let mut strtab: Vec<u8> = Vec::new();
    let push_str = |s: &str, strtab: &mut Vec<u8>| -> u32 {
        let off = (header_end + strtab.len()) as u32;
        strtab.extend_from_slice(s.as_bytes());
        strtab.push(0);
        off
    };
    let a_key = push_str("libfoo.so", &mut strtab);
    let a_val = push_str(lib_abs, &mut strtab);
    let b_key = push_str("libfoo.so", &mut strtab);
    let b_val = push_str(lib2_abs, &mut strtab);
    let c_key = push_str("liboob.so", &mut strtab);
    let d_key = push_str("librel.so", &mut strtab);
    // Non-absolute value: no leading slash → the starts_with('/') gate
    // rejects it before is_file is ever called.
    let d_val = push_str("relative/libbar.so", &mut strtab);

    let mut data = Vec::new();
    // Header: magic[20] + nlibs[4] + len_strings[4] + flags[4] + unused[16].
    data.extend_from_slice(LD_CACHE_MAGIC);
    data.extend_from_slice(&(nlibs as u32).to_le_bytes());
    data.extend_from_slice(&(strtab.len() as u32).to_le_bytes());
    data.extend_from_slice(&0u32.to_le_bytes()); // flags
    data.extend_from_slice(&[0u8; 16]); // unused
    assert_eq!(data.len(), LD_CACHE_HEADER_SIZE);

    // Entry layout: flags[4] + key[4] + value[4] + osversion[4] + hwcap[8].
    let total_len = header_end + strtab.len();
    let push_entry = |key_off: u32, val_off: u32, data: &mut Vec<u8>| {
        data.extend_from_slice(&0u32.to_le_bytes()); // flags
        data.extend_from_slice(&key_off.to_le_bytes());
        data.extend_from_slice(&val_off.to_le_bytes());
        data.extend_from_slice(&0u32.to_le_bytes()); // osversion
        data.extend_from_slice(&0u64.to_le_bytes()); // hwcap
    };
    push_entry(a_key, a_val, &mut data);
    push_entry(b_key, b_val, &mut data);
    // C's val_off points exactly at data.len() → OOB `continue`.
    push_entry(c_key, total_len as u32, &mut data);
    push_entry(d_key, d_val, &mut data);
    assert_eq!(data.len(), header_end);
    data.extend_from_slice(&strtab);
    assert_eq!(data.len(), total_len);

    let cache_path = tmp_dir.path().join("ld.so.cache");
    std::fs::write(&cache_path, &data).unwrap();
    let map = parse_ld_so_cache(&cache_path);

    // Both libfoo entries pass is_file(); or_insert keeps the FIRST.
    assert_eq!(
        map.get("libfoo.so"),
        Some(&PathBuf::from(lib_abs)),
        "or_insert first-wins: the second existing mapping is ignored",
    );
    // C is skipped by the OOB guard — never inserted.
    assert!(
        !map.contains_key("liboob.so"),
        "out-of-bounds val_off entry must be skipped, not panic",
    );
    // D's value lacks a leading slash → rejected by the
    // starts_with('/') gate in parse_ld_so_cache before is_file.
    assert!(
        !map.contains_key("librel.so"),
        "non-absolute path must be rejected by the starts_with('/') gate",
    );
}

/// A header claiming more entries than the file holds (hostile/corrupt
/// cache) must yield an empty map without panicking on the entry-offset
/// slices. The `nlibs * ENTRY` min-size guard (distinct from the
/// truncated-header check that stops at HEADER_SIZE) is otherwise
/// uncovered.
#[test]
fn parse_ld_so_cache_rejects_nlibs_overflow() {
    let tmp_dir = tempfile::TempDir::new().unwrap();
    let cache_path = tmp_dir.path().join("ld.so.cache");
    // Valid magic + full 48-byte header with an inflated nlibs and no
    // entry/string bytes following.
    let mut data = Vec::new();
    data.extend_from_slice(LD_CACHE_MAGIC);
    data.extend_from_slice(&1000u32.to_le_bytes()); // nlibs at offset 20
    data.extend_from_slice(&0u32.to_le_bytes()); // len_strings
    data.extend_from_slice(&0u32.to_le_bytes()); // flags
    data.extend_from_slice(&[0u8; 16]); // unused
    assert_eq!(data.len(), LD_CACHE_HEADER_SIZE);
    std::fs::write(&cache_path, &data).unwrap();
    assert!(
        parse_ld_so_cache(&cache_path).is_empty(),
        "an nlibs count larger than the file holds must return an empty map",
    );
}

/// [`is_standard_interpreter`] direct-match fast path (a literal known
/// linker, no syscall) and the canonicalize-failure `false` arm (a
/// nonexistent custom path). Otherwise only exercised indirectly via the
/// standard-vs-custom interpreter dispatch.
#[test]
fn is_standard_interpreter_matches_known_and_rejects_custom() {
    // Direct STANDARD_INTERPRETERS membership — no FS access required.
    assert!(
        is_standard_interpreter("/lib64/ld-linux-x86-64.so.2"),
        "a literal known linker matches the fast path",
    );
    // canonicalize() of a nonexistent path fails → false.
    assert!(
        !is_standard_interpreter("/opt/custom-toolchain/ld-fake.so"),
        "a nonexistent custom interpreter path is not standard",
    );
}

/// The `Some`-arms for `workload_root_cgroup` and
/// `scheduler_cgroup_parent` in [`build_suffix`] emit a cpio file
/// carrying the path verbatim with mode `0o100644` (a plain data file
/// the guest reads, not execs). No existing test sets these fields, so
/// both Some-arms are uncovered (the None path is covered by
/// `suffix_omits_empty_optional_entries`).
#[test]
fn suffix_emits_workload_root_cgroup_and_scheduler_cgroup_parent() {
    let exe = crate::resolve_current_exe().unwrap();
    let base = build_initramfs_base(&exe, &[], &[], None).unwrap();
    let suffix = build_suffix(
        base.len(),
        &SuffixParams {
            workload_root_cgroup: Some("/ktstr/workload"),
            scheduler_cgroup_parent: Some("/ktstr/sched"),
            ..Default::default()
        },
    )
    .unwrap();
    let mut archive = Vec::with_capacity(base.len() + suffix.len());
    archive.extend_from_slice(&base);
    archive.extend_from_slice(&suffix);
    let entries = cpio_entries(&archive);

    for (name, expected) in [
        ("workload_root_cgroup", "/ktstr/workload"),
        ("scheduler_cgroup_parent", "/ktstr/sched"),
    ] {
        let entry = entries
            .iter()
            .find(|(n, ..)| n == name)
            .unwrap_or_else(|| panic!("{name} entry missing"));
        assert_eq!(
            entry.1 as usize,
            expected.len(),
            "{name} size must match the verbatim path length",
        );
        // 0o100644 = S_IFREG | 0o644 — a plain data file (read, not exec'd).
        assert_eq!(entry.2, 0o100644, "{name} must be a plain data file");
        assert_eq!(
            cpio_entry_data(&archive, name).unwrap(),
            expected.as_bytes().to_vec(),
            "{name} payload must be the path written verbatim (no transform)",
        );
    }
}
