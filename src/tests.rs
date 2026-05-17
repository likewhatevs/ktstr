//! Unit tests for the ktstr crate root. Co-located via the `tests`
//! submodule pattern.

#![cfg(test)]

    use super::*;

    #[test]
    fn resolve_current_exe_happy_path() {
        let exe = resolve_current_exe().unwrap();
        // The test binary is running, so current_exe() returns a path that
        // exists on disk. resolve_current_exe should return that same path
        // (the exe.exists() early-return branch).
        let std_exe = std::env::current_exe().unwrap();
        if std_exe.exists() {
            // Happy path: binary not deleted, should return std::env::current_exe().
            assert_eq!(exe, std_exe);
        } else {
            // Fallback: binary deleted (llvm-cov), should return /proc/self/exe.
            assert_eq!(exe, std::path::PathBuf::from("/proc/self/exe"));
        }
    }

    // -- errno_name --

    #[test]
    fn errno_name_known_values() {
        assert_eq!(errno_name(libc::EPERM), Some("EPERM"));
        assert_eq!(errno_name(libc::ENOENT), Some("ENOENT"));
        assert_eq!(errno_name(libc::EINVAL), Some("EINVAL"));
        assert_eq!(errno_name(libc::ENOMEM), Some("ENOMEM"));
        assert_eq!(errno_name(libc::EBUSY), Some("EBUSY"));
        assert_eq!(errno_name(libc::EACCES), Some("EACCES"));
        assert_eq!(errno_name(libc::EAGAIN), Some("EAGAIN"));
        assert_eq!(errno_name(libc::ENOSYS), Some("ENOSYS"));
        assert_eq!(errno_name(libc::ETIMEDOUT), Some("ETIMEDOUT"));
    }

    #[test]
    fn errno_name_unknown() {
        assert_eq!(errno_name(9999), None);
        assert_eq!(errno_name(0), None);
        assert_eq!(errno_name(-1), None);
    }

    // -- find_kernel cache filter --

    /// `find_kernel`'s cache-scan step (step 2) must keep cache
    /// entries whose `ktstr_kconfig_hash` is `None` (pre-tracking
    /// format → [`cache::KconfigStatus::Untracked`]). The filter
    /// checks for `KconfigStatus::Stale` specifically; `Untracked`
    /// falls through to the return.
    ///
    /// A regression that tightened the filter to "anything not
    /// Matches" (e.g. `kconfig_status(&kc) != Matches`) would quietly
    /// drop every legacy cache entry built before ktstr tracked the
    /// kconfig fingerprint, forcing users to rebuild kernels whose
    /// only defect is the absence of a recorded hash. This test
    /// materializes exactly that shape — one valid image, no
    /// recorded hash — and asserts `find_kernel` returns it.
    #[test]
    fn find_kernel_preserves_untracked_cache_entries() {
        use crate::cache::{CacheArtifacts, CacheDir, KernelMetadata, KernelSource};
        use crate::test_support::test_helpers::{EnvVarGuard, lock_env};

        // `find_kernel` reads KTSTR_KERNEL and KTSTR_CACHE_DIR. Both
        // are process-wide, so other tests in this binary must not be
        // mutating them in parallel while this test owns them.
        let _env_lock = lock_env();
        // KTSTR_KERNEL: unset so find_kernel skips step 1 and falls
        // straight into the cache scan (step 2), which is the branch
        // this test targets.
        let _kernel_guard = EnvVarGuard::remove("KTSTR_KERNEL");

        // KTSTR_CACHE_DIR: point at an isolated temp dir so the
        // test sees only the Untracked entry we stage below — and
        // the host's real cache (if any) does not influence the
        // result.
        let tmp = tempfile::TempDir::new().unwrap();
        let cache_root = tmp.path().join("cache");
        let _cache_guard = EnvVarGuard::set("KTSTR_CACHE_DIR", &cache_root);

        // Stage one valid image with `ktstr_kconfig_hash = None`
        // (Untracked). `has_vmlinux` stays false so find_kernel's
        // vmlinux-symbol guard at lib.rs (only reached when
        // vmlinux_path() is Some) does not fire.
        let cache = CacheDir::with_root(cache_root.clone());
        let src_dir = tempfile::TempDir::new().unwrap();
        let image = src_dir.path().join("bzImage");
        std::fs::write(&image, b"fake kernel image").unwrap();
        let meta = KernelMetadata::new(
            KernelSource::Tarball,
            "x86_64",
            "bzImage",
            "2026-04-12T10:00:00Z",
        )
        .with_version("6.14.2");
        // `ktstr_kconfig_hash` defaults to None in `KernelMetadata::new`,
        // which is exactly the Untracked shape this test needs.
        assert!(
            meta.ktstr_kconfig_hash.is_none(),
            "test fixture must have no recorded kconfig hash to exercise the \
             Untracked branch of kconfig_status"
        );
        let entry = cache
            .store("untracked-entry", &CacheArtifacts::new(&image), &meta)
            .unwrap();
        let expected_image = entry.image_path();
        assert!(
            expected_image.exists(),
            "fixture image must exist on disk so find_kernel's image.exists() \
             check passes — got {expected_image:?}"
        );

        // find_kernel must return the Untracked entry's image.
        let resolved = find_kernel().unwrap();
        assert_eq!(
            resolved,
            Some(expected_image),
            "find_kernel dropped an Untracked cache entry — the kconfig-hash \
             filter at lib.rs must treat `Untracked` as keep, not stale"
        );
    }

    /// [`find_kernel`]'s cache scan (step 2) must skip entries where
    /// `matches!(entry.kconfig_status(&hash), KconfigStatus::Stale { .. })`
    /// and fall through to the next viable candidate.
    ///
    /// Two valid entries are seeded:
    ///
    /// * a current-hash entry built at `2026-04-01T00:00:00Z`, and
    /// * a stale-hash entry built at `2026-04-20T00:00:00Z`.
    ///
    /// [`cache::CacheDir::list`] sorts by `built_at` descending, so
    /// the stale entry is yielded first. If the `KconfigStatus::Stale`
    /// skip branch at the top of [`find_kernel`]'s cache-scan loop
    /// regressed to a no-op, `find_kernel` would return the stale
    /// entry's image. Asserting it returns the current entry's image
    /// proves the filter actually engages — strictly stronger than a
    /// single-entry `assert_ne!` on the stale path.
    #[test]
    fn find_kernel_skips_stale_cache_entry() {
        use crate::cache::{CacheArtifacts, CacheDir, KernelMetadata, KernelSource};
        use crate::test_support::test_helpers::{EnvVarGuard, lock_env};

        let _env_lock = lock_env();
        let _kernel_guard = EnvVarGuard::remove("KTSTR_KERNEL");

        let tmp = tempfile::TempDir::new().unwrap();
        let cache_root = tmp.path().join("cache");
        let _cache_guard = EnvVarGuard::set("KTSTR_CACHE_DIR", &cache_root);

        let current_hash = crate::kconfig_hash();
        let stale_hash = format!("{current_hash}-stale");

        let cache = CacheDir::with_root(cache_root.clone());
        let src_dir = tempfile::TempDir::new().unwrap();

        // Current-hash entry: older built_at. cache.list() sorts
        // newest-first, so this entry lands AFTER the stale entry — if
        // find_kernel failed to skip stale, it would return the stale
        // image instead of this one.
        let current_image = src_dir.path().join("current.bzImage");
        std::fs::write(&current_image, b"current kernel image").unwrap();
        let current_meta = KernelMetadata::new(
            KernelSource::Tarball,
            "x86_64",
            "current.bzImage",
            "2026-04-01T00:00:00Z",
        )
        .with_version("6.14.2")
        .with_ktstr_kconfig_hash(current_hash.clone());
        let current_entry = cache
            .store(
                "current-entry",
                &CacheArtifacts::new(&current_image),
                &current_meta,
            )
            .unwrap();

        // Stale entry: newer built_at, so list() yields it first.
        let stale_image = src_dir.path().join("stale.bzImage");
        std::fs::write(&stale_image, b"stale kernel image").unwrap();
        let stale_meta = KernelMetadata::new(
            KernelSource::Tarball,
            "x86_64",
            "stale.bzImage",
            "2026-04-20T00:00:00Z",
        )
        .with_version("6.14.3")
        .with_ktstr_kconfig_hash(stale_hash);
        cache
            .store(
                "stale-entry",
                &CacheArtifacts::new(&stale_image),
                &stale_meta,
            )
            .unwrap();

        let resolved = find_kernel().unwrap();
        assert_eq!(
            resolved,
            Some(current_entry.image_path()),
            "find_kernel must skip the newer stale entry and return the \
             current-hash entry — regression of the KconfigStatus::Stale \
             skip branch in find_kernel's cache-scan loop",
        );
    }

    // -- worker_ready marker path format --

    /// Pin the path format produced by
    /// [`crate::worker_ready::worker_ready_marker_path`].
    /// Downstream callers never spell the marker path as a literal —
    /// the worker writes it via `worker_ready_marker_path(pid)` and
    /// the test side polls via the same function — so a rename of
    /// the prefix does not surface at any caller's build time. This
    /// test's literal assertions are where the drift lands: changing
    /// the prefix breaks these equalities, catching the serialized-
    /// form change that would otherwise go silent.
    ///
    /// Lives in lib.rs (not in `worker_ready.rs`) because that file
    /// is dual-compiled via `#[path]` into the worker bin; a
    /// `#[cfg(test)] mod tests` inside it would duplicate the test
    /// into both the lib test binary and the bin test binary.
    #[test]
    fn worker_ready_marker_path_format_is_stable() {
        use crate::worker_ready::{WORKER_READY_MARKER_PREFIX, worker_ready_marker_path};
        assert_eq!(WORKER_READY_MARKER_PREFIX, "/tmp/ktstr-worker-ready-");
        assert_eq!(worker_ready_marker_path(0), "/tmp/ktstr-worker-ready-0");
        assert_eq!(
            worker_ready_marker_path(12345),
            "/tmp/ktstr-worker-ready-12345"
        );
        assert_eq!(
            worker_ready_marker_path(u32::MAX),
            "/tmp/ktstr-worker-ready-4294967295"
        );
    }

    // -- ktstr_kernel_env + KTSTR_KERNEL round-trip --
    //
    // KTSTR_KERNEL is the canonical cross-process hand-off for kernel
    // selection: `cargo ktstr test` resolves `--kernel` in the parent,
    // writes the resolved path back through `KTSTR_KERNEL_ENV`, and
    // every reader in the child (`find_kernel`, `detect_kernel_version`,
    // `find_test_vmlinux`) pulls it via `ktstr_kernel_env`. A
    // normalization drift between writer and reader would produce
    // silent resolution errors on the child side. Pin the round-trip
    // so such a drift is caught at test time.

    /// `ktstr_kernel_env` must return exactly the unmodified interior
    /// string when the env holds a plain absolute path. This is the
    /// happy-path channel between the parent's
    /// `std::fs::canonicalize(&p)` → `cmd.env(KTSTR_KERNEL_ENV, dir)`
    /// and the child's `ktstr_kernel_env` read. A regression that
    /// changed the normalization (adding a trailing slash, resolving
    /// symlinks, URL-encoding, etc.) would break the hand-off.
    #[test]
    fn ktstr_kernel_env_round_trips_absolute_path() {
        use crate::test_support::test_helpers::{EnvVarGuard, lock_env};
        let _lock = lock_env();
        let tmp = tempfile::TempDir::new().unwrap();
        let canonical = std::fs::canonicalize(tmp.path()).unwrap();
        let _guard = EnvVarGuard::set(KTSTR_KERNEL_ENV, &canonical);
        let read_back = ktstr_kernel_env().expect("env is set");
        assert_eq!(
            std::path::PathBuf::from(&read_back),
            canonical,
            "writer-reader round-trip must preserve the exact path; \
             drift between parent's canonicalize output and child's \
             ktstr_kernel_env read breaks every downstream resolver",
        );
    }

    /// Unset env reads as `None` — the default-resolution branch
    /// that `find_kernel`'s cache scan and `find_test_vmlinux`'s
    /// local-tree fallback both depend on.
    #[test]
    fn ktstr_kernel_env_unset_is_none() {
        use crate::test_support::test_helpers::{EnvVarGuard, lock_env};
        let _lock = lock_env();
        let _guard = EnvVarGuard::remove(KTSTR_KERNEL_ENV);
        assert!(
            ktstr_kernel_env().is_none(),
            "unset KTSTR_KERNEL must read as None so fallback resolvers activate",
        );
    }

    /// Empty-string env reads as `None`. Every reader should treat
    /// `KTSTR_KERNEL=""` the same as "unset" rather than erroring
    /// on an empty path — shells and Makefiles routinely emit empty
    /// values for unused variables, and failing on them would break
    /// unrelated CI flows.
    #[test]
    fn ktstr_kernel_env_empty_is_none() {
        use crate::test_support::test_helpers::{EnvVarGuard, lock_env};
        let _lock = lock_env();
        let _guard = EnvVarGuard::set(KTSTR_KERNEL_ENV, "");
        assert!(
            ktstr_kernel_env().is_none(),
            "empty KTSTR_KERNEL must collapse to None; CI flows routinely \
             pass empty strings for unused variables",
        );
    }

    /// Whitespace-only env reads as `None`. A shell-quoted
    /// `KTSTR_KERNEL="   "` is semantically the empty case even
    /// though `std::env::var` sees a non-empty string — the reader
    /// must trim before the empty check.
    #[test]
    fn ktstr_kernel_env_whitespace_is_none() {
        use crate::test_support::test_helpers::{EnvVarGuard, lock_env};
        let _lock = lock_env();
        let _guard = EnvVarGuard::set(KTSTR_KERNEL_ENV, "   \t\n  ");
        assert!(
            ktstr_kernel_env().is_none(),
            "whitespace-only KTSTR_KERNEL must collapse to None via trim \
             + empty-filter; no caller parses a whitespace-only value \
             meaningfully",
        );
    }

    /// A valid path with surrounding whitespace is trimmed and
    /// returned as the interior token. Pins the contract that the
    /// reader tolerates shell-quoting quirks without distorting the
    /// underlying value.
    #[test]
    fn ktstr_kernel_env_trims_surrounding_whitespace() {
        use crate::test_support::test_helpers::{EnvVarGuard, lock_env};
        let _lock = lock_env();
        let _guard = EnvVarGuard::set(KTSTR_KERNEL_ENV, "  ../linux  ");
        let read_back = ktstr_kernel_env().expect("env is set");
        assert_eq!(
            read_back, "../linux",
            "surrounding whitespace must be trimmed but the interior \
             preserved verbatim",
        );
    }

    /// `find_kernel` must call `KernelId::validate()` BEFORE the
    /// generic "multi-kernel specs are not supported in env-var form"
    /// bail when the env value parses as a Range or Git spec, so an
    /// inverted range like `KTSTR_KERNEL=6.16..6.12` surfaces the
    /// actionable "swap the endpoints" diagnostic. A future
    /// regression that drops the validate() call (or reorders it
    /// after the generic bail) would flip the error from the
    /// specific form to the generic redirect, landing here.
    ///
    /// `KTSTR_CACHE_DIR` is pointed at a fresh tempdir so the cache
    /// scan can't preempt the env-reader (the reader runs first
    /// regardless, but pinning the cache state prevents host noise
    /// from changing the assertion shape).
    #[test]
    fn find_kernel_inverted_range_env_surfaces_swap_diagnostic() {
        use crate::test_support::test_helpers::{EnvVarGuard, lock_env};
        let _env_lock = lock_env();
        let tmp = tempfile::TempDir::new().unwrap();
        let cache_root = tmp.path().join("cache");
        let _cache_guard = EnvVarGuard::set("KTSTR_CACHE_DIR", &cache_root);
        let _kernel_guard = EnvVarGuard::set(KTSTR_KERNEL_ENV, "6.16..6.12");

        let err = find_kernel().expect_err("inverted range must error");
        let msg = format!("{err:#}");
        assert!(
            msg.contains("inverted kernel range"),
            "validate() diagnostic must surface ahead of the generic \
             env-form bail; got: {msg}",
        );
        assert!(
            msg.contains("6.12..6.16"),
            "swap suggestion must appear in the error; got: {msg}",
        );
        assert!(
            !msg.contains("not supported in env-var form"),
            "validate() must short-circuit before the generic bail; got: {msg}",
        );
    }

    /// `KTSTR_KERNEL_ENV` must match the literal spelling `"KTSTR_KERNEL"`.
    /// Trivial pin, but load-bearing: readers that bypass the helper
    /// (e.g. hand-rolled `std::env::var("KTSTR_KERNEL")` in an ad-hoc
    /// script) match this string. A typo here would silently divorce
    /// the crate's canonical reader from every external tool.
    #[test]
    fn ktstr_kernel_env_constant_is_literal() {
        assert_eq!(KTSTR_KERNEL_ENV, "KTSTR_KERNEL");
    }

    // -- extra_kconfig_hash + cache_key_suffix_with_extra --
    //
    // The two-segment cache-key suffix underpins cargo-ktstr's
    // `--extra-kconfig` behavior: an extra-kconfig build must land
    // at a distinct cache slot from a vanilla build (different
    // content = miss), the same extra content must hit the same
    // slot on re-run (same content = hit), and `None` (no
    // `--extra-kconfig`) must produce the byte-identical suffix to
    // `cache_key_suffix()` so paths that don't expose the flag
    // continue resolving the existing keyspace.
    //
    // Suffix shape is `kc{baked_hash}` (no extra) or
    // `kc{baked_hash}-xkc{extra_hash}` (with extra). The two-segment
    // form makes `kernel list` self-describing — a reader can see at
    // a glance which entries carry user extras.

    /// `cache_key_suffix_with_extra(None)` must equal
    /// `cache_key_suffix()` byte-for-byte. Pins the
    /// no-`--extra-kconfig` path's compatibility contract: the
    /// test/coverage/shell/verifier resolution paths (which never
    /// pass `Some(extra)`) keep producing the pre-flag cache keys so
    /// existing cache entries remain addressable across the addition
    /// of this feature.
    #[test]
    fn cache_key_suffix_with_extra_none_matches_bare_suffix() {
        assert_eq!(cache_key_suffix_with_extra(None), cache_key_suffix());
    }

    /// `cache_key_suffix_with_extra(Some(content))` must contain
    /// the bare baked-in hash AS A PREFIX followed by `-xkc{...}`.
    /// Pins the two-segment shape:
    /// the leading segment is the existing `kconfig_hash()` so
    /// `kernel list` can decompose the suffix into baked-in vs
    /// user-extras components.
    #[test]
    fn cache_key_suffix_with_extra_some_has_two_segment_shape() {
        let suffix = cache_key_suffix_with_extra(Some("CONFIG_FOO=y\n"));
        let baked = kconfig_hash();
        assert!(
            suffix.starts_with(&baked),
            "Some suffix must start with bare baked-in hash {baked:?}, got {suffix:?}"
        );
        let after = &suffix[baked.len()..];
        assert!(
            after.starts_with("-xkc"),
            "after the baked-in segment, the next bytes must be `-xkc`, got {after:?}"
        );
        let extra_segment = &after["-xkc".len()..];
        assert_eq!(
            extra_segment.len(),
            8,
            "extra-hash segment must be 8 hex chars, got {extra_segment:?}"
        );
        assert!(
            extra_segment.chars().all(|c| c.is_ascii_hexdigit()),
            "extra-hash segment must be lowercase hex, got {extra_segment:?}"
        );
    }

    /// `cache_key_suffix_with_extra(Some(...))` must DIFFER from
    /// `cache_key_suffix()` for any non-empty user fragment. Pins
    /// the cache-discrimination contract: a build with a user
    /// fragment lands at a different cache key from a vanilla
    /// build. Also asserts the 8-hex-char shape of the appended
    /// xkc segment so a regression that changed the hash width
    /// (e.g. switched to a longer/shorter hex digest) doesn't
    /// silently slip past this test.
    #[test]
    fn cache_key_suffix_with_extra_some_differs_from_bare_suffix() {
        let suffix = cache_key_suffix_with_extra(Some("CONFIG_FOO=y\n"));
        assert_ne!(suffix, cache_key_suffix());
        // Width pin: the suffix-shape contract embeds the same
        // 8-hex-char width on both segments.
        let baked = kconfig_hash();
        let after = &suffix[baked.len()..];
        assert_eq!(
            after.len(),
            "-xkc".len() + 8,
            "suffix tail must be `-xkc{{8 hex chars}}`, got {after:?}"
        );
    }

    /// Production format-string shape assertion: the suffix
    /// `cache_key_suffix_with_extra(Some(...))` produces must be
    /// exactly the literal `format!("{baked}-xkc{extra_hash}")`
    /// cargo-ktstr.rs uses to build its tarball cache key. Pins
    /// the structural mirror so a refactor that changed the
    /// helper's format would surface here as a divergence from
    /// the production call site.
    #[test]
    fn cache_key_suffix_with_extra_matches_production_format_string() {
        let extra = "CONFIG_FOO=y\n";
        let baked = kconfig_hash();
        let extra_h = extra_kconfig_hash(extra);
        let helper = cache_key_suffix_with_extra(Some(extra));
        let expected = format!("{baked}-xkc{extra_h}");
        assert_eq!(
            helper, expected,
            "helper output must match production format `{{baked}}-xkc{{extra}}` \
             (cargo-ktstr.rs builds the tarball cache key with the same shape \
             via `{{ver}}-tarball-{{arch}}-kc{{cache_key_suffix_with_extra(...)}}`)"
        );
    }

    /// An empty user fragment ALSO differs from `None`: the
    /// `Some("")` branch always emits `-xkc{empty_hash}` as a
    /// distinct second segment, while `None` produces only the
    /// bare baked-in hash. Pins that `--extra-kconfig /empty/file`
    /// is a deliberate signal — even when no symbols are added,
    /// the build lands in a distinct cache slot from a no-flag
    /// build.
    #[test]
    fn cache_key_suffix_with_extra_empty_differs_from_none() {
        let with_empty = cache_key_suffix_with_extra(Some(""));
        let without = cache_key_suffix_with_extra(None);
        assert_ne!(with_empty, without);
    }

    /// Same user fragment must produce the SAME suffix across
    /// invocations. Pins cache-hit determinism for the
    /// `--extra-kconfig` repeat-invocation case.
    #[test]
    fn cache_key_suffix_with_extra_same_content_same_suffix() {
        let extra = "CONFIG_FOO=y\nCONFIG_BAR=n\n";
        let a = cache_key_suffix_with_extra(Some(extra));
        let b = cache_key_suffix_with_extra(Some(extra));
        assert_eq!(a, b, "same fragment must produce same suffix");
    }

    /// Different user fragments must produce DIFFERENT suffixes.
    /// Pins cache-miss discrimination across distinct extra files.
    /// The `xkc` segment carries the discriminator while the
    /// baked-in `kc` segment stays constant.
    #[test]
    fn cache_key_suffix_with_extra_different_content_different_suffix() {
        let a = cache_key_suffix_with_extra(Some("CONFIG_FOO=y\n"));
        let b = cache_key_suffix_with_extra(Some("CONFIG_FOO=n\n"));
        assert_ne!(a, b, "distinct fragments must produce distinct suffixes");
        // The baked-in prefix must match across both — only the
        // `-xkc{...}` tail differs.
        let baked = kconfig_hash();
        assert!(a.starts_with(&baked) && b.starts_with(&baked));
    }

    /// `extra_kconfig_hash` is 8 hex chars (CRC32 in lowercase
    /// hex), matching the existing `kconfig_hash` shape so the
    /// `xkc{...}` segment width is consistent with the baked-in
    /// `kc{...}` segment.
    #[test]
    fn extra_kconfig_hash_is_8_hex_chars() {
        for content in ["", "CONFIG_X=y\n", "# CONFIG_BPF is not set\n"] {
            let h = extra_kconfig_hash(content);
            assert_eq!(h.len(), 8, "expected 8 hex chars, got {h}");
            assert!(
                h.chars().all(|c| c.is_ascii_hexdigit()),
                "expected lowercase hex, got {h}",
            );
        }
    }

    /// `extra_kconfig_hash` hashes raw bytes — no comment stripping,
    /// no CRLF canonicalization. Two semantically-equivalent inputs
    /// with different comments or line endings produce different
    /// hashes and therefore land at distinct cache slots.
    #[test]
    fn extra_kconfig_hash_is_byte_sensitive() {
        let lf = "CONFIG_FOO=y\n";
        let crlf = "CONFIG_FOO=y\r\n";
        assert_ne!(
            extra_kconfig_hash(lf),
            extra_kconfig_hash(crlf),
            "CRLF and LF must hash differently — raw-byte hashing is intentional for byte-deterministic discrimination"
        );

        let with_comment = "# user note\nCONFIG_FOO=y\n";
        let without_comment = "CONFIG_FOO=y\n";
        assert_ne!(
            extra_kconfig_hash(with_comment),
            extra_kconfig_hash(without_comment),
            "comments must affect the hash — raw-byte hashing is intentional"
        );
    }

    /// CRLF cache-key discrimination. A user who edits their
    /// fragment on a Windows host and saves with CRLF line endings
    /// must land at a different cache slot from a Unix-LF fragment
    /// with otherwise-identical content. No canonicalization is
    /// performed — the disk waste is the price of byte-deterministic
    /// discrimination.
    #[test]
    fn cache_key_suffix_with_extra_crlf_differs_from_lf() {
        let lf = "CONFIG_FOO=y\n";
        let crlf = "CONFIG_FOO=y\r\n";
        let lf_suffix = cache_key_suffix_with_extra(Some(lf));
        let crlf_suffix = cache_key_suffix_with_extra(Some(crlf));
        assert_ne!(
            lf_suffix, crlf_suffix,
            "LF and CRLF user fragments must produce distinct cache \
             keys (no CRLF canonicalization). A Windows operator and \
             a Unix operator who supplied 'the same' fragment land at \
             distinct cache slots; this is the documented \
             byte-deterministic cache contract."
        );
    }

    /// Legacy cache entry must NOT be served when `--extra-kconfig`
    /// is passed. The cache key shape
    /// `kc{baked}` (legacy, no extras) and `kc{baked}-xkc{...}`
    /// (with extras) are STRUCTURALLY distinct strings, so any
    /// cache lookup that builds its key via
    /// `cache_key_suffix_with_extra(Some(...))` cannot collide with
    /// an entry stored under `cache_key_suffix()` only.
    ///
    /// Pins this invariant at the suffix level: an extras-aware
    /// suffix is strictly LONGER than the bare suffix, and the
    /// bare suffix is a proper prefix of the extras suffix. A
    /// substring lookup (cache lookup is exact-match by key, not
    /// substring) therefore cannot serve the legacy slot when the
    /// caller asks for an extras key. End-to-end roundtrip is
    /// covered at the integration level; this unit test pins the
    /// structural property the integration relies on.
    #[test]
    fn legacy_bare_suffix_is_proper_prefix_of_extras_suffix() {
        let bare = cache_key_suffix();
        let extras = cache_key_suffix_with_extra(Some("CONFIG_FOO=y\n"));
        assert!(
            extras.starts_with(&bare),
            "extras suffix must extend bare suffix — bare={bare:?} extras={extras:?}",
        );
        assert!(
            extras.len() > bare.len(),
            "extras suffix must be strictly longer than bare suffix",
        );
        assert_ne!(
            bare, extras,
            "structural distinction: legacy entries (key ending in `kc{{bare}}`) cannot \
             collide with extras entries (key ending in `kc{{bare}}-xkc{{...}}`) on \
             exact-match cache lookup."
        );
    }

    /// Pin that `extra_kconfig_hash("")` is the legitimate CRC32 of zero
    /// bytes (`00000000`), NOT a sentinel meaning "no extras".
    /// `None` is the no-extras signal — `Some("")` means "operator
    /// supplied an empty fragment file". An empty Some still
    /// produces a distinct cache key from None, so cache-key
    /// readers must distinguish "extras absent" (None →
    /// `kc{baked}` only) from "extras empty" (Some("") →
    /// `kc{baked}-xkc00000000`) by the presence/absence of the
    /// `-xkc` segment, NOT by hash content.
    #[test]
    fn extra_kconfig_hash_empty_is_crc32_zero_not_sentinel() {
        let empty = extra_kconfig_hash("");
        assert_eq!(
            empty, "00000000",
            "CRC32 of zero bytes is 0x00000000 by spec. This value is \
             a legitimate hash output, not a sentinel — readers that \
             want to detect 'no extras' must check the metadata's \
             `extra_kconfig_hash: Option<String>` for None, not for \
             this string."
        );
    }

    /// MERGE-SEMANTICS pin: when the user fragment overrides a
    /// baked-in symbol, the merged content fed to
    /// `make olddefconfig` must place the user line LAST so
    /// kbuild's last-wins rule
    /// (`scripts/kconfig/confdata.c::conf_read_simple` — "If
    /// conflicting CONFIG options are given from an input file,
    /// the last one wins.") makes the user value take precedence.
    ///
    /// Calls [`merge_kconfig_fragments`] DIRECTLY — the same
    /// helper [`crate::cli::kernel_build_pipeline`] uses to build
    /// the configure-pass content. Pinning the production helper
    /// (rather than reproducing its `format!` shape inline) means
    /// a regression that swapped the merge order would break this
    /// test in lock-step with the production path it represents.
    ///
    /// `EMBEDDED_KCONFIG` carries `CONFIG_BPF=y` (see
    /// `ktstr.kconfig`). A user fragment that flips it to
    /// `# CONFIG_BPF is not set` must appear AFTER the baked-in
    /// `CONFIG_BPF=y` line in the merged content. We can't drive
    /// `make olddefconfig` from a pure unit test (no in-tree
    /// kbuild fixture), so the test directly inspects the merged-
    /// fragment construction the pipeline uses and verifies the
    /// ordering invariant kbuild's last-wins rule depends on.
    #[test]
    fn merge_user_extra_appears_after_baked_in_for_conflict_resolution() {
        let user = "# CONFIG_BPF is not set\n";
        let merged = merge_kconfig_fragments(EMBEDDED_KCONFIG, Some(user));
        let baked_pos = merged
            .find("CONFIG_BPF=y")
            .expect("baked-in CONFIG_BPF=y must be present in merged fragment");
        let user_pos = merged
            .find("# CONFIG_BPF is not set")
            .expect("user override must be present in merged fragment");
        assert!(
            baked_pos < user_pos,
            "baked-in line must appear BEFORE user override so kbuild's \
             last-wins rule (confdata.c::conf_read_simple) keeps the user \
             value (baked_pos={baked_pos}, user_pos={user_pos})",
        );
        // Pin that the LAST occurrence of the symbol in
        // the merged content is the user's override. Walks every
        // line, tracks the last hit, and asserts it is the user's
        // disable directive — the kbuild parser walks lines top to
        // bottom and the last assignment wins, so the LAST
        // occurrence determines the final value.
        let mut last = None;
        for line in merged.lines() {
            let trimmed = line.trim();
            if trimmed == "CONFIG_BPF=y" || trimmed == "# CONFIG_BPF is not set" {
                last = Some(trimmed.to_string());
            }
        }
        assert_eq!(
            last.as_deref(),
            Some("# CONFIG_BPF is not set"),
            "the LAST occurrence of CONFIG_BPF in the merged content must \
             be the user override; kbuild's `conf_read_simple` walks lines \
             top-to-bottom and keeps the last assignment, so the user line \
             determines the final config value",
        );
    }

    /// NON-CONFLICT pin: when the user fragment adds a symbol the
    /// baked-in fragment doesn't
    /// mention, the merged content carries BOTH lines verbatim.
    /// `make olddefconfig` then sees two distinct symbols (one
    /// from each origin) and produces a `.config` containing
    /// both. Calls [`merge_kconfig_fragments`] directly so the
    /// production path's combine semantics are pinned.
    #[test]
    fn merge_user_extra_combines_with_baked_in_for_disjoint_symbols() {
        // Pick a CONFIG_X name not present in EMBEDDED_KCONFIG.
        let novel = "CONFIG_KTSTR_TEST_NOVEL_SYMBOL_FOR_MERGE_TEST=y\n";
        assert!(
            !EMBEDDED_KCONFIG.contains("CONFIG_KTSTR_TEST_NOVEL_SYMBOL_FOR_MERGE_TEST"),
            "test fixture must use a symbol absent from EMBEDDED_KCONFIG"
        );
        let merged = merge_kconfig_fragments(EMBEDDED_KCONFIG, Some(novel));
        // Both the user-novel line and at least one canonical
        // baked-in line must appear in the merged content.
        assert!(
            merged.contains("CONFIG_KTSTR_TEST_NOVEL_SYMBOL_FOR_MERGE_TEST=y"),
            "user-novel line must appear in merged fragment",
        );
        assert!(
            merged.contains("CONFIG_BPF=y"),
            "baked-in CONFIG_BPF=y must still appear in merged fragment",
        );
    }

    /// `merge_kconfig_fragments(baked, None)` returns the baked
    /// string unchanged — pinning the no-extras short-circuit so
    /// callers that pass `None` always observe the original
    /// fragment byte-for-byte.
    #[test]
    fn merge_kconfig_fragments_none_returns_baked_unchanged() {
        let merged = merge_kconfig_fragments(EMBEDDED_KCONFIG, None);
        assert_eq!(
            merged, EMBEDDED_KCONFIG,
            "merge with None must return the baked fragment unchanged"
        );
    }

    /// `merge_kconfig_fragments(baked, Some(""))` returns
    /// `{baked}\n` — the empty string still triggers the
    /// `format!` branch which appends a separator newline. Pins
    /// that the helper's branch boundary is `Option::is_some`,
    /// not `Option::is_some && !is_empty`. Operators reading the
    /// merged content under an empty fragment see the baked-in
    /// content followed by a single trailing newline (which kbuild
    /// ignores).
    #[test]
    fn merge_kconfig_fragments_some_empty_appends_separator_newline() {
        let merged = merge_kconfig_fragments(EMBEDDED_KCONFIG, Some(""));
        let expected = format!("{EMBEDDED_KCONFIG}\n");
        assert_eq!(merged, expected);
    }

    /// Last-wins ordering invariant: when the user fragment overrides
    /// a baked-in symbol, the user line MUST appear AFTER the baked-in
    /// line in the merged content. kbuild's
    /// `scripts/kconfig/confdata.c::conf_read_simple` keeps the
    /// last-occurring assignment per symbol, so a regression that
    /// flipped the order would silently make user values lose. Pinning
    /// at the merge-helper level catches the byte sequence kbuild
    /// operates on without spinning up a kernel build.
    #[test]
    fn merge_kconfig_fragments_user_line_appears_after_baked_for_overrides() {
        let baked = "CONFIG_FOO=y\nCONFIG_BAR=m\n";
        let user = "CONFIG_FOO=n\n";
        let merged = merge_kconfig_fragments(baked, Some(user)).into_owned();
        let baked_idx = merged
            .find("CONFIG_FOO=y")
            .expect("baked CONFIG_FOO=y must be present");
        let user_idx = merged
            .find("CONFIG_FOO=n")
            .expect("user CONFIG_FOO=n override must be present");
        assert!(
            baked_idx < user_idx,
            "baked-in CONFIG_FOO=y must precede user override CONFIG_FOO=n so \
             kbuild's last-wins rule picks the user value: {merged}"
        );
    }

    /// Disjoint fragments (user adds a symbol the baked-in fragment
    /// doesn't mention) combine verbatim — both lines reach kbuild
    /// untouched. Pins the non-conflict path so a refactor that
    /// dedupes or reorders user lines doesn't drop additive
    /// configuration.
    #[test]
    fn merge_kconfig_fragments_disjoint_symbols_both_present() {
        let baked = "CONFIG_FOO=y\n";
        let user = "CONFIG_DISJOINT_TEST_SYMBOL=m\n";
        let merged = merge_kconfig_fragments(baked, Some(user)).into_owned();
        assert!(
            merged.contains("CONFIG_FOO=y"),
            "baked symbol must survive merge: {merged}"
        );
        assert!(
            merged.contains("CONFIG_DISJOINT_TEST_SYMBOL=m"),
            "user-added disjoint symbol must survive merge: {merged}"
        );
    }

    // -- --extra-kconfig cache roundtrip / discrimination --
    //
    // These integration tests pin the cache-key behavior the
    // `--extra-kconfig` plumbing depends on. They use
    // `CacheDir::with_root` to plant fixture entries against an
    // isolated tempdir so the host's real cache (if any) does not
    // interfere, and exercise the production `cache.lookup` (the
    // same call site `cli::cache_lookup` consumes) directly. No
    // network, no kernel build — the bug surface is the cache-key
    // suffix machinery, and the tests target that surface
    // directly.

    /// Two consecutive lookups with the SAME `--extra-kconfig`
    /// content must hit the same cache slot. Pins the cache-hit
    /// branch of the roundtrip: identical extras content →
    /// identical `cache_key_suffix_with_extra` → identical cache
    /// key → planted entry retrieved.
    #[test]
    fn cache_lookup_same_extras_hits_planted_entry() {
        use crate::cache::{CacheArtifacts, CacheDir, KernelMetadata, KernelSource};
        use crate::test_support::test_helpers::{EnvVarGuard, lock_env};

        let _env_lock = lock_env();
        let _kernel_guard = EnvVarGuard::remove("KTSTR_KERNEL");
        let tmp = tempfile::TempDir::new().unwrap();
        let cache_root = tmp.path().join("cache");
        let _cache_guard = EnvVarGuard::set("KTSTR_CACHE_DIR", &cache_root);

        let extra = "CONFIG_KTSTR_CACHE_ROUNDTRIP_TEST_A=y\n";
        let extra_hash = extra_kconfig_hash(extra);
        let cache_key = format!("test-roundtrip-{}-xkc{}", kconfig_hash(), extra_hash);

        // Plant a fixture entry whose key matches what a build
        // with this extras content would produce.
        let cache = CacheDir::with_root(cache_root.clone());
        let src_dir = tempfile::TempDir::new().unwrap();
        let image = src_dir.path().join("bzImage");
        std::fs::write(&image, b"fake kernel image").unwrap();
        let meta = KernelMetadata::new(
            KernelSource::Tarball,
            "x86_64",
            "bzImage",
            "2026-04-12T10:00:00Z",
        )
        .with_extra_kconfig_hash(extra_hash.clone());
        cache
            .store(&cache_key, &CacheArtifacts::new(&image), &meta)
            .unwrap();

        // Look up the same key — must hit.
        let hit = cache.lookup(&cache_key);
        assert!(
            hit.is_some(),
            "cache lookup with same extras must return planted entry; \
             cache_key={cache_key}"
        );
        assert_eq!(
            hit.as_ref().unwrap().metadata.extra_kconfig_hash.as_deref(),
            Some(extra_hash.as_str()),
            "retrieved entry must carry the planted extra_kconfig_hash"
        );
    }

    /// Different `--extra-kconfig` contents must land at distinct
    /// cache slots — a build with extras=A must NOT serve a cached
    /// entry produced with extras=B. Pins the cache-discrimination
    /// branch: distinct extras → distinct hashes → distinct keys.
    #[test]
    fn cache_lookup_different_extras_misses_planted_entry() {
        use crate::cache::{CacheArtifacts, CacheDir, KernelMetadata, KernelSource};
        use crate::test_support::test_helpers::{EnvVarGuard, lock_env};

        let _env_lock = lock_env();
        let _kernel_guard = EnvVarGuard::remove("KTSTR_KERNEL");
        let tmp = tempfile::TempDir::new().unwrap();
        let cache_root = tmp.path().join("cache");
        let _cache_guard = EnvVarGuard::set("KTSTR_CACHE_DIR", &cache_root);

        let extra_a = "CONFIG_KTSTR_CACHE_DISCRIMINATE_A=y\n";
        let extra_b = "CONFIG_KTSTR_CACHE_DISCRIMINATE_B=y\n";
        let key_a = format!(
            "test-disc-{}-xkc{}",
            kconfig_hash(),
            extra_kconfig_hash(extra_a)
        );
        let key_b = format!(
            "test-disc-{}-xkc{}",
            kconfig_hash(),
            extra_kconfig_hash(extra_b)
        );
        assert_ne!(
            key_a, key_b,
            "extras A and B must produce distinct cache keys (precondition)"
        );

        // Plant entry under key_a only.
        let cache = CacheDir::with_root(cache_root.clone());
        let src_dir = tempfile::TempDir::new().unwrap();
        let image = src_dir.path().join("bzImage");
        std::fs::write(&image, b"fake kernel image A").unwrap();
        let meta = KernelMetadata::new(
            KernelSource::Tarball,
            "x86_64",
            "bzImage",
            "2026-04-12T10:00:00Z",
        )
        .with_extra_kconfig_hash(extra_kconfig_hash(extra_a));
        cache
            .store(&key_a, &CacheArtifacts::new(&image), &meta)
            .unwrap();

        // Lookup with key_b must MISS — extras=B's slot is
        // different from extras=A's slot.
        let hit_b = cache.lookup(&key_b);
        assert!(
            hit_b.is_none(),
            "lookup with extras=B's key must miss when only extras=A is planted; \
             key_a={key_a} key_b={key_b}"
        );

        // Sanity: the planted entry IS reachable via key_a.
        assert!(
            cache.lookup(&key_a).is_some(),
            "planted entry must be reachable via its own key"
        );
    }

    /// A bare-suffix entry (built without `--extra-kconfig`) must
    /// NOT be served when an extras lookup runs — and conversely,
    /// an extras-suffix entry must NOT be served to a bare lookup.
    /// Both halves of the segregation are pinned because either
    /// regression silently mis-serves a kernel built against a
    /// different configuration to a build that asked for a
    /// distinct one.
    #[test]
    fn cache_lookup_bare_and_extras_keys_segregated() {
        use crate::cache::{CacheArtifacts, CacheDir, KernelMetadata, KernelSource};
        use crate::test_support::test_helpers::{EnvVarGuard, lock_env};

        let _env_lock = lock_env();
        let _kernel_guard = EnvVarGuard::remove("KTSTR_KERNEL");
        let tmp = tempfile::TempDir::new().unwrap();
        let cache_root = tmp.path().join("cache");
        let _cache_guard = EnvVarGuard::set("KTSTR_CACHE_DIR", &cache_root);

        let baked = kconfig_hash();
        let extra = "CONFIG_KTSTR_CACHE_SEGREGATE=y\n";
        let bare_key = format!("test-seg-{baked}");
        let extras_key = format!("test-seg-{baked}-xkc{}", extra_kconfig_hash(extra));
        assert_ne!(
            bare_key, extras_key,
            "bare and extras-suffix keys must be distinct (precondition)"
        );

        // Plant only the bare entry.
        let cache = CacheDir::with_root(cache_root.clone());
        let src_dir = tempfile::TempDir::new().unwrap();
        let image = src_dir.path().join("bzImage");
        std::fs::write(&image, b"bare kernel").unwrap();
        let bare_meta = KernelMetadata::new(
            KernelSource::Tarball,
            "x86_64",
            "bzImage",
            "2026-04-12T10:00:00Z",
        );
        // bare_meta has extra_kconfig_hash=None by default.
        assert!(
            bare_meta.extra_kconfig_hash.is_none(),
            "bare entry fixture must not carry extras hash"
        );
        cache
            .store(&bare_key, &CacheArtifacts::new(&image), &bare_meta)
            .unwrap();

        // An extras lookup must not reach the bare entry.
        assert!(
            cache.lookup(&extras_key).is_none(),
            "extras lookup must NOT serve the bare entry — operator built with \
             --extra-kconfig and would silently get a kernel without their \
             user symbols if this regressed"
        );

        // Now plant an extras entry.
        let extras_image = src_dir.path().join("bzImage-extras");
        std::fs::write(&extras_image, b"extras kernel").unwrap();
        let extras_meta = KernelMetadata::new(
            KernelSource::Tarball,
            "x86_64",
            "bzImage",
            "2026-04-13T10:00:00Z",
        )
        .with_extra_kconfig_hash(extra_kconfig_hash(extra));
        cache
            .store(
                &extras_key,
                &CacheArtifacts::new(&extras_image),
                &extras_meta,
            )
            .unwrap();

        // Both keys must now hit their own slot independently.
        let bare_hit = cache.lookup(&bare_key).expect("bare entry");
        let extras_hit = cache.lookup(&extras_key).expect("extras entry");
        assert!(
            bare_hit.metadata.extra_kconfig_hash.is_none(),
            "bare entry must report None extras hash"
        );
        assert!(
            extras_hit.metadata.extra_kconfig_hash.is_some(),
            "extras entry must report Some(hash)"
        );
    }

    /// `CacheEntry::has_extra_kconfig()` must return true for an
    /// entry built with `--extra-kconfig` and false for a bare
    /// entry. Pins the metadata-readback contract that drives the
    /// `(extra kconfig)` tag in `kernel list` output — operators
    /// inspecting their cache need to distinguish at-a-glance
    /// which entries carry user modifications.
    #[test]
    fn cache_entry_has_extra_kconfig_reflects_metadata() {
        use crate::cache::{CacheArtifacts, CacheDir, KernelMetadata, KernelSource};
        use crate::test_support::test_helpers::{EnvVarGuard, lock_env};

        let _env_lock = lock_env();
        let _kernel_guard = EnvVarGuard::remove("KTSTR_KERNEL");
        let tmp = tempfile::TempDir::new().unwrap();
        let cache_root = tmp.path().join("cache");
        let _cache_guard = EnvVarGuard::set("KTSTR_CACHE_DIR", &cache_root);

        let cache = CacheDir::with_root(cache_root.clone());
        let src_dir = tempfile::TempDir::new().unwrap();
        let image = src_dir.path().join("bzImage");
        std::fs::write(&image, b"img").unwrap();

        // Bare entry: extra_kconfig_hash = None.
        let bare_meta = KernelMetadata::new(
            KernelSource::Tarball,
            "x86_64",
            "bzImage",
            "2026-04-12T10:00:00Z",
        );
        let bare = cache
            .store("test-has-bare", &CacheArtifacts::new(&image), &bare_meta)
            .unwrap();
        assert!(
            !bare.has_extra_kconfig(),
            "bare entry (extra_kconfig_hash = None) must report has_extra_kconfig() = false"
        );

        // Extras entry: extra_kconfig_hash = Some(hash).
        let extras_meta = KernelMetadata::new(
            KernelSource::Tarball,
            "x86_64",
            "bzImage",
            "2026-04-13T10:00:00Z",
        )
        .with_extra_kconfig_hash("deadbeef");
        let extras = cache
            .store(
                "test-has-extras",
                &CacheArtifacts::new(&image),
                &extras_meta,
            )
            .unwrap();
        assert!(
            extras.has_extra_kconfig(),
            "entry with extra_kconfig_hash = Some(...) must report has_extra_kconfig() = true"
        );
    }
