//! Unit tests for [`super`] (the `host_thread_probe` module).
//! Co-located via the `tests` submodule pattern.

#![cfg(test)]

    use super::*;

    #[test]
    fn round_up_pow2_zero_align_clamps_to_one() {
        // Degenerate align=0 must clamp to 1, not crash on `align - 1`.
        assert_eq!(round_up_pow2(7, 0), Some(7));
    }

    #[test]
    fn round_up_pow2_handles_overflow() {
        assert_eq!(round_up_pow2(u64::MAX, 8), None);
    }

    #[test]
    fn round_up_pow2_basic() {
        assert_eq!(round_up_pow2(7, 8), Some(8));
        assert_eq!(round_up_pow2(8, 8), Some(8));
        assert_eq!(round_up_pow2(9, 8), Some(16));
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn variant_ii_basic() {
        // fs_base=0x10000, image_size=0x100, st_value=0x10, field=8
        // → image_base=0xff00, addr=0xff18.
        let addr = compute_tls_address_variant_ii(0x10000, 0x100, 0x10, 8).unwrap();
        assert_eq!(addr, 0xff18);
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn variant_ii_underflow() {
        // fs_base less than image_size → underflow surface as Err.
        assert!(compute_tls_address_variant_ii(0x100, 0x200, 0, 0).is_err());
    }

    #[test]
    fn counter_offsets_rejects_reversed_pair() {
        // thread_allocated must precede thread_deallocated.
        assert!(CounterOffsets::new(64, 16).is_err());
        assert!(CounterOffsets::new(16, 16).is_err());
        assert!(CounterOffsets::new(16, 32).is_ok());
    }

    #[test]
    fn counter_offsets_combined_span() {
        let off = CounterOffsets::new(16, 32).unwrap();
        // span covers allocated + intermediate u64 + deallocated.
        // dealloc_offset=32, +8 = 40, -16 = 24.
        assert_eq!(off.combined_read_span(), 24);
    }

    #[test]
    fn parse_maps_elf_path_keeps_executable_only() {
        let exe_line = "5583e6f7a000-5583e6f7b000 r-xp 00000000 fe:00 12345 /usr/bin/example";
        assert_eq!(
            parse_maps_elf_path(exe_line),
            Some(PathBuf::from("/usr/bin/example"))
        );
    }

    #[test]
    fn parse_maps_elf_path_drops_non_executable() {
        let data_line = "5583e6f7a000-5583e6f7b000 r--p 00000000 fe:00 12345 /usr/bin/example";
        assert_eq!(parse_maps_elf_path(data_line), None);
    }

    #[test]
    fn parse_maps_elf_path_drops_anonymous() {
        let anon = "7f0000000000-7f0000001000 r-xp 00000000 00:00 0";
        assert_eq!(parse_maps_elf_path(anon), None);
    }

    #[test]
    fn parse_maps_elf_path_drops_special_brackets() {
        let stack = "7fff00000000-7fff00001000 r-xp 00000000 00:00 0 [stack]";
        // Path doesn't start with /
        assert_eq!(parse_maps_elf_path(stack), None);
    }

    #[test]
    fn attach_error_tags_are_unique() {
        // Stable-token guarantee: every variant maps to a distinct
        // tag string. A renamed token should regress here loudly.
        let pairs: Vec<(&'static str, AttachError)> = vec![
            ("pid-missing", AttachError::PidMissing(anyhow!("x"))),
            (
                "readlink-failure",
                AttachError::ReadlinkFailure(anyhow!("x")),
            ),
            (
                "maps-read-failure",
                AttachError::MapsReadFailure(anyhow!("x")),
            ),
            (
                "jemalloc-not-found",
                AttachError::JemallocNotFound(anyhow!("x")),
            ),
            ("jemalloc-in-dso", AttachError::JemallocInDso(anyhow!("x"))),
            ("arch-mismatch", AttachError::ArchMismatch(anyhow!("x"))),
            (
                "dwarf-parse-failure",
                AttachError::DwarfParseFailure(anyhow!("x")),
            ),
        ];
        let mut seen: std::collections::BTreeSet<&'static str> = std::collections::BTreeSet::new();
        for (expected, err) in &pairs {
            assert_eq!(*expected, err.tag());
            assert!(seen.insert(err.tag()), "duplicate tag {}", err.tag());
        }
    }

    #[test]
    fn probe_error_tags_are_unique() {
        // Stable-token guarantee: every variant maps to a distinct
        // tag string. A renamed token should regress here loudly.
        let pairs: Vec<(&'static str, ProbeError)> = vec![
            ("ptrace-seize", ProbeError::PtraceSeize(anyhow!("x"))),
            (
                "ptrace-interrupt",
                ProbeError::PtraceInterrupt(anyhow!("x")),
            ),
            ("waitpid", ProbeError::Waitpid(anyhow!("x"))),
            ("get-regset", ProbeError::GetRegset(anyhow!("x"))),
            ("process-vm-readv", ProbeError::ProcessVmReadv(anyhow!("x"))),
            ("tls-arithmetic", ProbeError::TlsArithmetic(anyhow!("x"))),
        ];
        let mut seen: std::collections::BTreeSet<&'static str> = std::collections::BTreeSet::new();
        for (expected, err) in &pairs {
            assert_eq!(*expected, err.tag());
            assert!(seen.insert(err.tag()), "duplicate tag {}", err.tag());
        }
    }

    #[test]
    fn attach_pid_missing_returns_pid_missing_error() {
        // Try a pid we know doesn't exist — pid 0 is reserved, kernel
        // never assigns it. The attach path's existence check should
        // surface PidMissing, not crash on /proc/0 inspection.
        match attach_jemalloc(0) {
            Err(AttachError::PidMissing(_)) => {}
            other => panic!("expected PidMissing for pid=0, got {other:?}"),
        }
    }

    /// A regular non-existent pid (not the special-case `0`)
    /// also surfaces `PidMissing`. The kernel's
    /// `PID_MAX_LIMIT` caps live pids at 2^22 (4 Mi), so
    /// `i32::MAX` (≈ 2^31) is guaranteed never to be a live
    /// tgid on any host this code runs against. Pins the
    /// `Path::new(&format!("/proc/{pid}")).exists()` guard
    /// against the "live-pid-now-dead" race separately from
    /// the reserved `0` special case.
    #[test]
    fn attach_returns_pid_missing_for_regular_dead_pid() {
        match attach_jemalloc(i32::MAX) {
            Err(AttachError::PidMissing(_)) => {}
            other => panic!("expected PidMissing for pid=i32::MAX, got {other:?}"),
        }
    }

    // ------------------------------------------------------------
    // Synthetic-procfs tests for `attach_jemalloc_at`.
    //
    // `attach_jemalloc` is the sole detection gate. Drives it
    // against a tempdir-staged `<tmp>/<pid>/{exe,maps}` shape so
    // the failure classification stays pinned regardless of what
    // the host's real `/proc` looks like.
    // ------------------------------------------------------------

    /// Stage a synthetic `<tmp>/<pid>/maps` referencing a NON-
    /// jemalloc binary (`/bin/sleep`) with a matching `exe`
    /// symlink. The gate must return `JemallocNotFound` because
    /// /bin/sleep has no `tsd_tls` symbol — every non-jemalloc
    /// target flows through the precise ELF/DWARF walk, never a
    /// string match.
    #[test]
    fn attach_at_returns_jemalloc_not_found_on_maps_without_jemalloc() {
        // /bin/sleep is a coreutils binary not linked against
        // jemalloc. If absent on this host, skip — the test's
        // value is observing `JemallocNotFound` against a real
        // non-jemalloc ELF, not constructing fake bytes.
        let sleep = PathBuf::from("/bin/sleep");
        if !sleep.exists() {
            eprintln!("skipping — /bin/sleep unavailable");
            return;
        }
        let tmp = tempfile::TempDir::new().expect("tempdir");
        let pid: i32 = 4242;
        let pid_dir = tmp.path().join(pid.to_string());
        std::fs::create_dir_all(&pid_dir).expect("mkdir pid_dir");

        // exe symlink → /bin/sleep, satisfying the readlink read
        // and the static-TLS path-equality guard.
        std::os::unix::fs::symlink(&sleep, pid_dir.join("exe")).expect("symlink exe");

        // maps line with an r-x mapping for /bin/sleep — the
        // gate's only mapping to inspect.
        let maps = format!(
            "5583e6f7a000-5583e6f7b000 r-xp 00000000 fe:00 12345 {}\n",
            sleep.display(),
        );
        std::fs::write(pid_dir.join("maps"), maps).expect("write maps");

        match attach_jemalloc_at(tmp.path(), pid) {
            Err(AttachError::JemallocNotFound(_)) => {}
            other => panic!("expected JemallocNotFound for non-jemalloc maps, got {other:?}",),
        }
    }

    /// Stage a synthetic `<tmp>/<pid>/` directory with a maps
    /// file referencing some r-x mapping but with the
    /// `<tmp>/<pid>/exe` symlink ABSENT. The gate must return
    /// `ReadlinkFailure` — proving the exe pre-check engages
    /// BEFORE the maps walk so a target with a vanished exe
    /// symlink (race-with-exit, container teardown, …) surfaces
    /// a precise classifier rather than degrading to a
    /// downstream parse error.
    ///
    /// The maps content is irrelevant because the readlink
    /// happens first; we still write a plausible line so a
    /// future refactor that re-orders the operations would
    /// surface here as a different (downstream) error variant
    /// rather than passing.
    #[test]
    fn attach_at_returns_readlink_failure_when_exe_symlink_missing() {
        let tmp = tempfile::TempDir::new().expect("tempdir");
        let pid: i32 = 4243;
        let pid_dir = tmp.path().join(pid.to_string());
        std::fs::create_dir_all(&pid_dir).expect("mkdir pid_dir");

        // DELIBERATELY NO `exe` symlink — the gate's readlink
        // step must fail.
        let maps = "5583e6f7a000-5583e6f7b000 r-xp 00000000 fe:00 12345 /usr/bin/anything\n";
        std::fs::write(pid_dir.join("maps"), maps).expect("write maps");

        match attach_jemalloc_at(tmp.path(), pid) {
            Err(AttachError::ReadlinkFailure(_)) => {}
            other => panic!("expected ReadlinkFailure when exe symlink is absent, got {other:?}",),
        }
    }

    // ---------------------------------------------------------------
    // Engine helper unit tests
    // ---------------------------------------------------------------

    /// Variant II TLS TP math: fs_base - aligned_tls_size + st_value
    /// + field_offset.  Worked example pins the arithmetic against a
    ///   hand-checked case.
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn variant_ii_worked_example() {
        let fs_base = 0x7f12_3456_7000;
        let aligned = 512;
        let st_value = 0x100;
        let field = 264;
        let addr = compute_tls_address_variant_ii(fs_base, aligned, st_value, field).unwrap();
        // 0x7f1234567000 - 0x200 + 0x100 + 264 = 0x7f1234567008
        assert_eq!(addr, 0x7f12_3456_7008);
    }

    /// Thread pointer equal to aligned image size is the minimum
    /// valid configuration — subtraction lands at zero rather than
    /// underflowing.
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn variant_ii_boundary_tp_equals_image_size() {
        let addr = compute_tls_address_variant_ii(4096, 4096, 0, 0).unwrap();
        assert_eq!(addr, 0);
    }

    /// Variant I (aarch64) worked example: `TP +
    /// round_up(TCB_SIZE=16, p_align) + st_value + field_offset`
    /// with `p_align=16` giving `image_offset=16`. Pure arithmetic
    /// guarded behind `cfg(target_arch = "aarch64")` because
    /// `compute_tls_address_variant_i` itself is only compiled on
    /// aarch64 builds.
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn variant_i_worked_example() {
        let tpidr = 0x7f12_3456_7000;
        let p_align = 16;
        let st_value = 0x100;
        let field = 264;
        let addr = compute_tls_address_variant_i(tpidr, p_align, st_value, field).unwrap();
        // 0x7f1234567000 + 0x10 + 0x100 + 264 = 0x7f1234567218
        assert_eq!(addr, 0x7f12_3456_7218);
    }

    /// Variant I with `p_align > TCB_SIZE_AARCH64`: image base
    /// rounded up to `p_align`, not pinned at 16.
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn variant_i_high_alignment() {
        // TP + round_up(16, 64) + 0 + 0 = TP + 64
        let addr = compute_tls_address_variant_i(0x1000, 64, 0, 0).unwrap();
        assert_eq!(addr, 0x1040);
    }

    /// Variant I `p_align == TCB_SIZE_AARCH64`: exact fit, no
    /// padding past the reserved TCB words.
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn variant_i_tcb_sized_alignment() {
        let addr = compute_tls_address_variant_i(0x1000, TCB_SIZE_AARCH64, 0, 0).unwrap();
        assert_eq!(addr, 0x1010);
    }

    /// Variant I with `p_align < TCB_SIZE_AARCH64`: the reserved
    /// TCB size is the minimum — sub-TCB alignments do NOT shrink
    /// the image-base offset. `round_up(16, 8) = 16`.
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn variant_i_sub_tcb_alignment() {
        let addr = compute_tls_address_variant_i(0x1000, 8, 0, 0).unwrap();
        assert_eq!(addr, 0x1010);
    }

    /// Variant I degenerate-align fallback: `p_align = 0` in a
    /// malformed ELF must not divide-by-zero. The implementation's
    /// `.max(1)` coerces to `align = 1`, giving `round_up(16, 1) =
    /// 16`.
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn variant_i_zero_align_clamped() {
        let addr = compute_tls_address_variant_i(0x1000, 0, 0, 0).unwrap();
        assert_eq!(addr, 0x1010);
    }

    /// Variant I overflow: `TP + image_offset + st_value +
    /// field_offset` near `u64::MAX` errors rather than wrapping
    /// into the low address space.
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn variant_i_overflow_errors() {
        let err = compute_tls_address_variant_i(u64::MAX - 10, 16, 0x100, 0).unwrap_err();
        assert!(
            format!("{err}").contains("TLS address arithmetic overflow"),
            "got: {err}",
        );
    }

    /// Variant I image-offset overflow: `p_align` near `u64::MAX`
    /// makes `round_up(TCB_SIZE, p_align)` overflow the
    /// `checked_add` BEFORE the TP addition runs. The error must be
    /// the image-offset variant, not the address-arithmetic
    /// variant.
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn variant_i_image_offset_overflow_errors() {
        let err = compute_tls_address_variant_i(0x1000, u64::MAX, 0, 0).unwrap_err();
        assert!(
            format!("{err}").contains("TLS image offset overflow"),
            "expected image-offset overflow, got: {err}",
        );
    }

    /// Arch dispatcher routes to the right Variant based on
    /// `target_arch`. Inputs produce distinct answers under each
    /// formula so a cfg-dispatch regression would surface here.
    #[test]
    fn compute_tls_address_dispatches_by_target_arch() {
        // Variant II: 4096 - 4096 + 0 + 0 = 0
        // Variant I:  4096 + round_up(16, 16) + 0 + 0 = 4112
        let got = compute_tls_address(4096, 4096, 16, 0, 0).unwrap();
        #[cfg(target_arch = "x86_64")]
        assert_eq!(got, 0, "x86_64 must dispatch to Variant II");
        #[cfg(target_arch = "aarch64")]
        assert_eq!(got, 4112, "aarch64 must dispatch to Variant I");
    }

    /// Position-distinct dispatcher test — every arg is a distinct
    /// prime so an argument-position swap shifts the result by an
    /// identifiable amount.
    ///
    /// Variant II: 13_000_009 - 1009 + 307 + 83 = 12_999_390
    /// Variant I:  13_000_009 + round_up(16, 64) + 307 + 83
    ///           = 13_000_009 + 64 + 307 + 83
    ///           = 13_000_463
    #[test]
    fn compute_tls_address_dispatches_positionally_distinct() {
        let got = compute_tls_address(13_000_009, 1009, 64, 307, 83).unwrap();
        #[cfg(target_arch = "x86_64")]
        assert_eq!(got, 12_999_390, "x86_64 Variant II formula");
        #[cfg(target_arch = "aarch64")]
        assert_eq!(got, 13_000_463, "aarch64 Variant I formula");
    }

    /// `extract_pt_tls_layout` against the test binary's own ELF.
    /// The lib's containing crate links `tikv_jemallocator` as the
    /// global allocator, so the compiled test binary carries
    /// jemalloc's `tsd_tls` in a real `PT_TLS` segment. Parsing it
    /// exercises the actual extraction function end-to-end and
    /// pins the toolchain-emitted invariants (power-of-two
    /// alignment; aligned size is a multiple of align) against a
    /// real program header.
    #[test]
    fn extract_pt_tls_layout_on_real_elf() {
        let exe = std::env::current_exe().expect("current_exe");
        let data = std::fs::read(&exe).expect("read current_exe");
        let elf = goblin::elf::Elf::parse(&data).expect("parse current_exe");
        let (rounded, align) = extract_pt_tls_layout(&elf).expect("test binary must carry PT_TLS");
        assert!(
            align.is_power_of_two(),
            "p_align {align} must be a power of two",
        );
        assert!(
            rounded >= align,
            "aligned_size {rounded} must be >= align {align}"
        );
        assert!(
            rounded % align == 0,
            "aligned_size {rounded} must be a multiple of align {align}",
        );
    }

    /// Adjacency variant of `combined_read_span`: a hypothetical
    /// future jemalloc that drops the fast_event field and places
    /// deallocated immediately after allocated produces a 16-byte
    /// span. Pins that the helper would not over-read in that
    /// case.
    #[test]
    fn counter_offsets_combined_span_adjacent() {
        let o = CounterOffsets::new(100, 108).unwrap();
        let span = o.combined_read_span();
        assert_eq!(span, 16);
    }

    /// `read_build_id` against the test binary's own ELF must
    /// surface the `NT_GNU_BUILD_ID` note descriptor as a lowercase
    /// hex string when the toolchain emits one. Skips on a
    /// linker / RUSTFLAGS combination that elides the note (e.g.
    /// `-Wl,--build-id=none`); the negative path is covered by the
    /// `candidate_debuginfo_paths_*` tests below.
    #[test]
    fn read_build_id_on_real_elf_is_lowercase_hex() {
        let exe = std::env::current_exe().expect("current_exe");
        let data = std::fs::read(&exe).expect("read current_exe");
        let elf = goblin::elf::Elf::parse(&data).expect("parse current_exe");
        let Some(hex) = read_build_id(&elf, &data) else {
            eprintln!("skip: current_exe carries no NT_GNU_BUILD_ID; toolchain elided it",);
            return;
        };
        assert!(!hex.is_empty(), "build-id hex must be non-empty");
        assert_eq!(
            hex,
            hex.to_ascii_lowercase(),
            "build-id must be rendered in lowercase hex",
        );
        assert!(
            hex.chars()
                .all(|c| c.is_ascii_hexdigit() && (c.is_ascii_digit() || c.is_ascii_lowercase())),
            "build-id must contain only ASCII hex digits [0-9a-f]; got {hex:?}",
        );
        assert!(
            build_id_hex_is_safe(&hex),
            "read_build_id output must pass build_id_hex_is_safe",
        );
    }

    /// `read_gnu_debuglink` on the test binary's own ELF returns
    /// `None` — the binary carries inline `.debug_info` rather than
    /// a debuglink to an external file.
    #[test]
    fn read_gnu_debuglink_on_inline_debug_elf_returns_none() {
        let exe = std::env::current_exe().expect("current_exe");
        let data = std::fs::read(&exe).expect("read current_exe");
        let elf = goblin::elf::Elf::parse(&data).expect("parse current_exe");
        assert!(
            read_gnu_debuglink(&elf, &data).is_none(),
            "test binary has inline .debug_info; .gnu_debuglink must be absent",
        );
    }

    /// `candidate_debuginfo_paths` is a pure function. Pin the
    /// path-construction rules: build-id first (most-discriminating),
    /// then parent-relative debuglink, then parent/.debug, then
    /// `/usr/lib/debug`-rooted on absolute targets.
    #[test]
    fn candidate_debuginfo_paths_full_layout() {
        let target = Path::new("/usr/bin/example");
        let paths =
            candidate_debuginfo_paths(target, Some("example.debug"), Some("abcdef0123456789"));
        assert_eq!(paths.len(), 4);
        assert_eq!(
            paths[0],
            PathBuf::from("/usr/lib/debug/.build-id/ab/cdef0123456789.debug"),
        );
        assert_eq!(paths[1], PathBuf::from("/usr/bin/example.debug"));
        assert_eq!(paths[2], PathBuf::from("/usr/bin/.debug/example.debug"));
        assert_eq!(
            paths[3],
            PathBuf::from("/usr/lib/debug/usr/bin/example.debug"),
        );
    }

    /// No build-id and no debuglink → no candidates.
    #[test]
    fn candidate_debuginfo_paths_returns_empty_when_no_hints() {
        let target = Path::new("/usr/bin/example");
        let paths = candidate_debuginfo_paths(target, None, None);
        assert!(paths.is_empty());
    }

    /// Build-id shorter than 2 hex chars → build-id path skipped
    /// (cannot do the `split_at(2)` prefix/rest split). Other
    /// candidates (debuglink-based) still emit.
    #[test]
    fn candidate_debuginfo_paths_skips_short_build_id() {
        let target = Path::new("/usr/bin/example");
        let paths = candidate_debuginfo_paths(target, Some("example.debug"), Some("a"));
        assert_eq!(paths.len(), 3);
        assert!(
            !paths[0].to_string_lossy().contains("/.build-id/"),
            "first candidate must be a debuglink path; got {:?}",
            paths[0],
        );
    }

    /// Empty-string build-id: `Some("")` fails the `hex.len() >= 2`
    /// gate and the build-id branch is skipped. Distinct from
    /// `None` (which skips the whole branch before the gate). Pins
    /// the zero-length boundary so a future tightening of the gate
    /// would not silently shift the cutoff.
    #[test]
    fn candidate_debuginfo_paths_empty_build_id_skipped() {
        let target = Path::new("/usr/bin/example");
        let paths = candidate_debuginfo_paths(target, Some("example.debug"), Some(""));
        assert_eq!(paths.len(), 3);
        assert!(
            !paths
                .iter()
                .any(|p| p.to_string_lossy().contains(".build-id")),
        );
    }

    /// Relative target path: the absolute-path-rooted
    /// `/usr/lib/debug/<...>` fallback is skipped because the
    /// debuglink convention only meaningfully applies to absolute
    /// targets — the rpm/deb layout that populates that tree keys
    /// off the install path, not a relative one. The build-id
    /// candidate (which roots in `/usr/lib/debug/.build-id/...`,
    /// independent of the target's own path) still emits, plus
    /// the two parent-relative debuglink candidates.
    #[test]
    fn candidate_debuginfo_paths_relative_target_skips_lib_debug_root() {
        let target = Path::new("./example");
        let paths =
            candidate_debuginfo_paths(target, Some("example.debug"), Some("deadbeef12345678"));
        // Three: build-id + parent-relative debuglink + parent/.debug
        // debuglink. The /usr/lib/debug/<abs_dir>/<name> candidate
        // is dropped because the target is relative.
        assert_eq!(paths.len(), 3);
        assert!(
            !paths
                .iter()
                .any(|p| p.starts_with("/usr/lib/debug")
                    && !p.to_string_lossy().contains(".build-id")),
            "no /usr/lib/debug-rooted debuglink candidate may emit \
             for a relative target; got {:?}",
            paths,
        );
    }

    /// Build-id with exactly 2 hex chars: boundary of the `>= 2`
    /// gate. `"ab".split_at(2)` yields `("ab", "")`, producing
    /// `/usr/lib/debug/.build-id/ab/.debug`. Pins boundary
    /// behavior.
    #[test]
    fn candidate_debuginfo_paths_build_id_exactly_two_chars() {
        let target = Path::new("/usr/bin/example");
        let paths = candidate_debuginfo_paths(target, None, Some("ab"));
        assert_eq!(paths.len(), 1);
        assert_eq!(
            paths[0],
            PathBuf::from("/usr/lib/debug/.build-id/ab/.debug"),
        );
    }

    /// Build-id alone, no debuglink: only the build-id candidate
    /// emits.
    #[test]
    fn candidate_debuginfo_paths_build_id_only() {
        let target = Path::new("/usr/bin/example");
        let paths = candidate_debuginfo_paths(target, None, Some("abcdef0123456789"));
        assert_eq!(paths.len(), 1);
        assert_eq!(
            paths[0],
            PathBuf::from("/usr/lib/debug/.build-id/ab/cdef0123456789.debug"),
        );
    }

    /// Debuglink alone, no build-id: three debuglink candidates
    /// emit on an absolute target.
    #[test]
    fn candidate_debuginfo_paths_debuglink_only() {
        let target = Path::new("/usr/bin/example");
        let paths = candidate_debuginfo_paths(target, Some("example.debug"), None);
        assert_eq!(paths.len(), 3);
        assert_eq!(paths[0], PathBuf::from("/usr/bin/example.debug"));
        assert_eq!(paths[1], PathBuf::from("/usr/bin/.debug/example.debug"));
        assert_eq!(
            paths[2],
            PathBuf::from("/usr/lib/debug/usr/bin/example.debug"),
        );
    }

    /// Target path with no parent directory (`/`): debuglink branch
    /// emits zero candidates (parent = None); build-id is
    /// orthogonal and still emits.
    #[test]
    fn candidate_debuginfo_paths_no_parent_skips_debuglink() {
        let target = Path::new("/");
        let paths =
            candidate_debuginfo_paths(target, Some("orphan.debug"), Some("abcdef0123456789"));
        assert_eq!(paths.len(), 1);
        assert_eq!(
            paths[0],
            PathBuf::from("/usr/lib/debug/.build-id/ab/cdef0123456789.debug"),
        );
        let paths = candidate_debuginfo_paths(target, Some("orphan.debug"), None);
        assert!(paths.is_empty());
    }

    /// Root-relative target (`/example` — direct child of `/`):
    /// `parent` is `/`, so `parent.join(name)` is `/<name>` and
    /// `parent.join(".debug").join(name)` is `/.debug/<name>`. The
    /// `strip_prefix("/")` succeeds and yields an empty relative
    /// path, so the `/usr/lib/debug`-rooted candidate is
    /// `/usr/lib/debug/<name>` (no intermediate directory).
    /// Pins this corner of the path-construction matrix because
    /// `/example` is a perfectly legitimate executable location
    /// (busybox-style minimal images, ktstr's own initramfs) and
    /// the candidate set must remain coherent for it.
    #[test]
    fn candidate_debuginfo_paths_root_relative_target() {
        let target = Path::new("/example");
        let paths = candidate_debuginfo_paths(target, Some("example.debug"), None);
        assert_eq!(paths.len(), 3);
        assert_eq!(paths[0], PathBuf::from("/example.debug"));
        assert_eq!(paths[1], PathBuf::from("/.debug/example.debug"));
        assert_eq!(paths[2], PathBuf::from("/usr/lib/debug/example.debug"));
    }

    /// Bare-basename target (`example` — no directory components):
    /// `parent` is the empty path, which is non-absolute, so the
    /// `strip_prefix("/")` gate rejects the `/usr/lib/debug`-rooted
    /// candidate. Only the parent-relative debuglink candidates
    /// emit. Pins this against accidentally falling back to the
    /// `/usr/lib/debug/example.debug` shape — that path collides
    /// with an absolute `/example` target's lib-debug-rooted
    /// candidate (see the `_root_relative_target` test above), so
    /// the gate is load-bearing for distinguishing the two cases.
    #[test]
    fn candidate_debuginfo_paths_bare_basename_target() {
        let target = Path::new("example");
        let paths = candidate_debuginfo_paths(target, Some("example.debug"), None);
        assert_eq!(paths.len(), 2);
        assert_eq!(paths[0], PathBuf::from("example.debug"));
        assert_eq!(paths[1], PathBuf::from(".debug/example.debug"));
        // No /usr/lib/debug rooted candidate.
        assert!(
            !paths.iter().any(|p| p.starts_with("/usr/lib/debug")),
            "bare-basename target must not produce /usr/lib/debug-rooted \
             debuglink candidate; got {:?}",
            paths,
        );
    }

    /// `debuglink_name_is_safe` rejects every shape that would
    /// allow path traversal or an absolute-path replacement when
    /// joined into a candidate path. Per binutils
    /// `bfd/opncls.c::bfd_get_debug_link_info`, the on-disk
    /// `.gnu_debuglink` section is supposed to carry only a bare
    /// filename, but a hostile or corrupt ELF could embed any
    /// byte string up to the section bound.
    ///
    /// Defenses pinned here:
    /// - `/foo` → reject (would replace receiver in `PathBuf::join`)
    /// - `../etc/passwd` → reject (path traversal)
    /// - `subdir/foo` → reject (escapes `parent.join` semantics)
    /// - `""` → reject (empty filename)
    /// - `"."` / `".."` → reject (literal traversal forms)
    /// - `"\0..."` → reject (leading NUL byte)
    /// - `"foo\0..."` → reject (embedded NUL byte)
    ///
    /// Accepts every shape a well-formed `.gnu_debuglink` could
    /// emit: dotted basenames, multi-component basenames with
    /// dashes / digits / lowercase, the canonical
    /// `<binary>.debug` form.
    #[test]
    fn debuglink_name_rejects_path_traversal_and_absolute_paths() {
        // Reject — path separators / absolute / traversal / NUL / empty.
        assert!(!debuglink_name_is_safe(""));
        assert!(!debuglink_name_is_safe("/"));
        assert!(!debuglink_name_is_safe("/etc/passwd"));
        assert!(!debuglink_name_is_safe("/etc/shadow"));
        assert!(!debuglink_name_is_safe("../etc/passwd"));
        assert!(!debuglink_name_is_safe("../../etc/passwd"));
        assert!(!debuglink_name_is_safe("subdir/foo.debug"));
        assert!(!debuglink_name_is_safe("./foo.debug"));
        assert!(!debuglink_name_is_safe("."));
        assert!(!debuglink_name_is_safe(".."));
        assert!(!debuglink_name_is_safe("\0"));
        assert!(!debuglink_name_is_safe("foo\0bar.debug"));

        // Accept — every shape a real `.gnu_debuglink` produces.
        assert!(debuglink_name_is_safe("example.debug"));
        assert!(debuglink_name_is_safe("ktstr.debug"));
        assert!(debuglink_name_is_safe("libfoo-1.2.3.so.debug"));
        assert!(debuglink_name_is_safe(".hidden.debug"));
        assert!(debuglink_name_is_safe("a"));
    }

    /// End-to-end: a `candidate_debuginfo_paths` call with a
    /// hostile debuglink name produces ZERO debuglink candidates,
    /// regardless of the elf_path or build-id state. The build-id
    /// path is orthogonal and still emits when present (build-id
    /// is hex-validated upstream, so it's a separate trust domain).
    #[test]
    fn candidate_debuginfo_paths_drops_unsafe_debuglink_name() {
        // Absolute debuglink name — would replace receiver in
        // `parent.join(name)` and produce `/etc/passwd` directly.
        let target = Path::new("/usr/bin/example");
        let paths = candidate_debuginfo_paths(target, Some("/etc/passwd"), None);
        assert!(
            paths.is_empty(),
            "unsafe debuglink name (absolute path) must produce zero \
             candidates; got {:?}",
            paths,
        );

        // Path-traversal debuglink name.
        let paths = candidate_debuginfo_paths(target, Some("../../etc/passwd"), None);
        assert!(paths.is_empty());

        // Build-id is orthogonal — survives a poisoned debuglink name.
        let paths =
            candidate_debuginfo_paths(target, Some("/etc/passwd"), Some("abcdef0123456789"));
        assert_eq!(paths.len(), 1);
        assert_eq!(
            paths[0],
            PathBuf::from("/usr/lib/debug/.build-id/ab/cdef0123456789.debug"),
        );
    }

    /// `build_id_hex_is_safe` rejects every shape that could
    /// traverse out of the `.build-id` candidate-path tree or
    /// otherwise deviate from the `read_build_id` output format.
    /// The format-match check is strict enough that
    /// `/usr/lib/debug/.build-id/<head>/<tail>.debug` is bounded
    /// to `[0-9a-f]/` segments — no `/`, `..`, NUL, or uppercase
    /// / non-hex byte that could alter the path tree.
    ///
    /// Defenses pinned here:
    /// - `/.passwd` → reject (path separator)
    /// - `../etc` → reject (path traversal — `..` resolves
    ///   directly when split into `<head>/<tail>` in the
    ///   candidate path)
    /// - `a/b` → reject (mid-string path separator)
    /// - `..` → reject (literal traversal as hex pair would split
    ///   into `/usr/lib/debug/.build-id/../<tail>.debug`)
    /// - `"\0\0"` → reject (NUL byte)
    /// - `ABCD` → reject (uppercase — `read_build_id` only emits
    ///   lowercase, so uppercase implies the upstream parser was
    ///   bypassed or regressed)
    /// - `xx` → reject (non-hex chars)
    /// - `abc` → reject (odd length — `read_build_id` always emits
    ///   even-length output since each byte produces two chars)
    /// - `""` → reject (empty)
    #[test]
    fn build_id_hex_rejects_non_hex_inputs() {
        assert!(!build_id_hex_is_safe(""));
        assert!(!build_id_hex_is_safe("/"));
        assert!(!build_id_hex_is_safe("/."));
        assert!(!build_id_hex_is_safe("/.passwd"));
        assert!(!build_id_hex_is_safe("../etc"));
        assert!(!build_id_hex_is_safe("a/b"));
        assert!(!build_id_hex_is_safe(".."));
        assert!(!build_id_hex_is_safe("\0\0"));
        assert!(!build_id_hex_is_safe("AB"));
        assert!(!build_id_hex_is_safe("ABCD"));
        assert!(!build_id_hex_is_safe("ABCDEF0123456789"));
        assert!(!build_id_hex_is_safe("xx"));
        assert!(!build_id_hex_is_safe("xy"));
        assert!(!build_id_hex_is_safe("zz"));
        assert!(!build_id_hex_is_safe("abc")); // odd length
        assert!(!build_id_hex_is_safe("ab cd")); // whitespace
        assert!(!build_id_hex_is_safe("ab-cd")); // dash
    }

    /// `build_id_hex_is_safe` accepts every shape `read_build_id`
    /// actually emits — the `{:02x}` byte-by-byte hex encoding,
    /// always lowercase, always even length. Pinned shapes cover
    /// the byte-count range of every build-id format the
    /// toolchain emits today (md5 → 16 chars, sha1 → 40, sha256
    /// → 64 with `--build-id=sha256`).
    #[test]
    fn build_id_hex_accepts_lowercase_hex() {
        assert!(build_id_hex_is_safe("ab"));
        assert!(build_id_hex_is_safe("abcd"));
        assert!(build_id_hex_is_safe("0123456789abcdef"));
        // SHA-1 build-id (20 bytes → 40 chars).
        assert!(build_id_hex_is_safe(
            "abcdef0123456789abcdef0123456789abcdef01"
        ));
        // SHA-256 build-id (32 bytes → 64 chars), seen on some
        // toolchains via `--build-id=sha256`.
        assert!(build_id_hex_is_safe(
            "0011223344556677889900112233445566778899001122334455667788990011"
        ));
    }

    /// End-to-end: a `candidate_debuginfo_paths` call with a
    /// hostile `build_id_hex` produces ZERO candidates from the
    /// build-id branch. The debuglink branch is orthogonal and
    /// still emits when present (debuglink is name-validated by
    /// `debuglink_name_is_safe` — separate trust boundary).
    #[test]
    fn candidate_debuginfo_paths_drops_unsafe_build_id_hex() {
        let target = Path::new("/usr/bin/example");

        // Path-traversal hex — `/`, `..`, NUL.
        let paths = candidate_debuginfo_paths(target, None, Some("/.passwd"));
        assert!(
            paths.is_empty(),
            "unsafe build-id hex (path separator) must produce zero \
             candidates; got {:?}",
            paths,
        );
        let paths = candidate_debuginfo_paths(target, None, Some(".."));
        assert!(paths.is_empty());

        // Uppercase hex — `read_build_id` never emits this; if a
        // caller surfaces it, treat as bypass and reject.
        let paths = candidate_debuginfo_paths(target, None, Some("ABCDEF0123456789"));
        assert!(paths.is_empty());

        // Non-hex chars.
        let paths = candidate_debuginfo_paths(target, None, Some("xyzzy012345"));
        assert!(paths.is_empty());

        // Odd-length hex.
        let paths = candidate_debuginfo_paths(target, None, Some("abc"));
        assert!(paths.is_empty());

        // Debuglink is orthogonal — survives a poisoned build-id.
        let paths = candidate_debuginfo_paths(target, Some("example.debug"), Some("/.passwd"));
        assert_eq!(paths.len(), 3);
        assert_eq!(paths[0], PathBuf::from("/usr/bin/example.debug"));
        assert_eq!(paths[1], PathBuf::from("/usr/bin/.debug/example.debug"));
        assert_eq!(
            paths[2],
            PathBuf::from("/usr/lib/debug/usr/bin/example.debug"),
        );
    }

    /// Fixture-health pin: the test binary must carry both a
    /// populated `.debug_info` AND at least one `STT_FUNC` symbol
    /// in `.symtab`. Without those, a future strip-debug or
    /// link-stripping change would silently invalidate the
    /// debuginfo-discovery tests above.
    #[test]
    fn test_elf_has_populated_debug_info_section_and_stt_func_symbols() {
        use goblin::elf::sym;
        let exe = std::env::current_exe().expect("current_exe");
        let data = std::fs::read(&exe).expect("read current_exe");
        let elf = goblin::elf::Elf::parse(&data).expect("parse current_exe");

        assert!(
            section_is_populated(&elf, &data, ".debug_info"),
            "test binary must carry a populated .debug_info section",
        );
        let func_count = elf
            .syms
            .iter()
            .filter(|s| s.st_type() == sym::STT_FUNC)
            .count();
        assert!(
            func_count > 0,
            "test binary must carry at least one STT_FUNC symbol in .symtab",
        );
    }

    /// `round_up_pow2` boundary matrix: degenerate-align, zero,
    /// max-value overflow, and the 8-byte-align rounding triad.
    #[test]
    fn round_up_pow2_boundary_matrix() {
        assert_eq!(round_up_pow2(0, 0), Some(0));
        assert_eq!(round_up_pow2(0, 1), Some(0));
        assert_eq!(round_up_pow2(u64::MAX, 1), Some(u64::MAX));
        assert_eq!(round_up_pow2(u64::MAX, 2), None);
        assert_eq!(round_up_pow2(7, 8), Some(8));
        assert_eq!(round_up_pow2(8, 8), Some(8));
        assert_eq!(round_up_pow2(9, 8), Some(16));
    }

    /// `find_jemalloc_tsd_tls_in_table` against an empty symbol
    /// table + strtab returns None — must not panic on degenerate
    /// inputs.
    #[test]
    fn find_jemalloc_tsd_tls_in_table_empty_returns_none() {
        let tab: goblin::elf::Symtab<'_> = Default::default();
        let strs = goblin::strtab::Strtab::default();
        assert!(find_jemalloc_tsd_tls_in_table(&tab, &strs).is_none());
    }

    /// `is_jemalloc_tsd_tls_symbol` accepts the bare `tsd_tls`
    /// (unprefixed jemalloc builds, older versions) and rejects
    /// every superficially-similar name that is NOT a clean
    /// `<prefix>_tsd_tls` form.
    #[test]
    fn is_jemalloc_tsd_tls_symbol_accepts_bare_form() {
        assert!(is_jemalloc_tsd_tls_symbol("tsd_tls"));
    }

    /// Every observed `--with-jemalloc-prefix=…` build matches:
    /// default `je_`, tikv-jemallocator-sys's `_rjem_je_`, the
    /// large-binary `jemalloc_je_` variant. The suffix-match
    /// predicate makes the registry self-extending — a future
    /// custom prefix will Just Work without code changes.
    #[test]
    fn is_jemalloc_tsd_tls_symbol_accepts_known_prefixes() {
        assert!(is_jemalloc_tsd_tls_symbol("je_tsd_tls"));
        assert!(is_jemalloc_tsd_tls_symbol("_rjem_je_tsd_tls"));
        assert!(is_jemalloc_tsd_tls_symbol("jemalloc_je_tsd_tls"));
        // Hypothetical future custom prefix — must match without a
        // registry edit. Pins the open-set semantic of the
        // suffix-match contract.
        assert!(is_jemalloc_tsd_tls_symbol("custom_prefix_tsd_tls"));
    }

    /// Negative cases — names that look adjacent but lack the
    /// trailing `_tsd_tls` separator must not match. This is the
    /// key safety property of the predicate: `mytsd_tls` (no
    /// underscore separator) is not a jemalloc symbol; treating it
    /// as one would let any non-jemalloc TLS variable whose name
    /// happens to end in `tsd_tls` falsely satisfy the gate. The
    /// downstream DWARF walk would catch most false positives, but
    /// rejecting them at the name-match stage keeps the diagnostic
    /// path crisp (a `JemallocNotFound` rather than a
    /// `DwarfParseFailure` for a non-jemalloc target).
    #[test]
    fn is_jemalloc_tsd_tls_symbol_rejects_lookalikes() {
        // No leading underscore separator — not a clean prefix.
        assert!(!is_jemalloc_tsd_tls_symbol("mytsd_tls"));
        // Trailing extra suffix breaks the suffix match.
        assert!(!is_jemalloc_tsd_tls_symbol("tsd_tls_v2"));
        // Substring inside a longer name — must not match.
        assert!(!is_jemalloc_tsd_tls_symbol("je_tsd_tls_extra"));
        // Truncated forms — must reject.
        assert!(!is_jemalloc_tsd_tls_symbol("tsd"));
        assert!(!is_jemalloc_tsd_tls_symbol("je_tsd"));
        assert!(!is_jemalloc_tsd_tls_symbol("_tsd_tls")); // empty prefix → bare separator only, no actual prefix
        assert!(!is_jemalloc_tsd_tls_symbol(""));
        assert!(!is_jemalloc_tsd_tls_symbol("tls"));
    }
