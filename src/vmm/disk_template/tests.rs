//! Unit tests for [`super`] (the `disk_template` module).
//! Co-located via the `tests` submodule pattern.

#![cfg(test)]

use super::*;

#[test]
fn cache_key_renders_capacity_in_mib_and_version_fp() {
    let key = template_cache_key(Filesystem::Btrfs, 256 * 1024 * 1024, "deadbeef");
    assert_eq!(key, "btrfs-256m-deadbeef");
    let key = template_cache_key(Filesystem::Raw, 1024 * 1024 * 1024, NOVERSION_FP);
    assert_eq!(key, "raw-1024m-noversion");
}

#[test]
fn cache_key_truncates_sub_mib_capacity_to_zero() {
    // Capacity less than 1 MiB rounds down to 0m. This is
    // intentional — DiskConfig's capacity is u32 mebibytes (see
    // capacity_mib), so the only way to hit this is constructing
    // capacity_bytes by hand below 2^20. Pinning the rendering
    // for that corner so a future bug that rounds up silently
    // is caught.
    let key = template_cache_key(Filesystem::Btrfs, 1024, "deadbeef");
    assert_eq!(key, "btrfs-0m-deadbeef");
}

#[test]
fn cache_key_rotates_with_version_fp() {
    // Two different mkfs versions produce two different keys for
    // the same (fs, capacity) pair. Pins the cache-key
    // self-invalidation on mkfs upgrade — without this property
    // the cache would silently reuse stale templates whose
    // internal format the new kernel may reject.
    let v1 = template_cache_key(Filesystem::Btrfs, 256 * 1024 * 1024, "fp_v1");
    let v2 = template_cache_key(Filesystem::Btrfs, 256 * 1024 * 1024, "fp_v2");
    assert_ne!(v1, v2, "cache key must rotate when version_fp changes");
    assert_eq!(v1, "btrfs-256m-fp_v1");
    assert_eq!(v2, "btrfs-256m-fp_v2");
}

#[test]
fn template_path_includes_filename_constant() {
    // Isolate from operator state: KTSTR_CACHE_DIR / XDG_CACHE_HOME
    // / $HOME bleed into template_path_for_key via cache_root().
    let tmp = tempfile::tempdir().expect("create tempdir");
    let _guard =
        crate::test_support::test_helpers::EnvVarGuard::set(crate::KTSTR_CACHE_DIR_ENV, tmp.path());
    let path = template_path_for_key("btrfs-256m").expect("resolve template path");
    assert!(path.ends_with(format!("btrfs-256m/{TEMPLATE_FILENAME}")));
}

#[test]
fn lookup_missing_returns_none() {
    // Use a tempdir as cache root so we don't pollute the
    // operator's real cache. The cache_root() helper reads
    // KTSTR_CACHE_DIR; setting it for the lifetime of the test
    // via EnvVarGuard isolates per-test state.
    let tmp = tempfile::tempdir().expect("create tempdir");
    let _guard =
        crate::test_support::test_helpers::EnvVarGuard::set(crate::KTSTR_CACHE_DIR_ENV, tmp.path());
    let result = lookup(Filesystem::Raw, "missing-key").expect("lookup must not error on miss");
    assert!(result.is_none());
}

#[test]
fn store_atomic_publishes_then_lookup_finds() {
    let tmp = tempfile::tempdir().expect("create tempdir");
    let _guard =
        crate::test_support::test_helpers::EnvVarGuard::set(crate::KTSTR_CACHE_DIR_ENV, tmp.path());
    // Stage a fake template under the cache root so the rename
    // is on the same filesystem.
    let cache_root_path = cache_root().unwrap();
    std::fs::create_dir_all(&cache_root_path).unwrap();
    let staged = cache_root_path.join("staged.img");
    std::fs::write(&staged, b"FAKE_TEMPLATE_BODY").unwrap();
    let key = "test-key";
    let installed = store_atomic(Filesystem::Raw, key, &staged).expect("store_atomic publishes");
    assert!(installed.ends_with(format!("{key}/{TEMPLATE_FILENAME}")));
    // Now lookup must find it.
    let found = lookup(Filesystem::Raw, key)
        .expect("lookup ok")
        .expect("lookup must hit");
    assert_eq!(found, installed);
    // And content survived the rename.
    let body = std::fs::read(&found).unwrap();
    assert_eq!(body, b"FAKE_TEMPLATE_BODY");
}

#[test]
fn lookup_btrfs_rejects_magicless_template() {
    // Layer A self-heal: a cached template lacking the btrfs superblock
    // magic (a stale all-zero image a prior build published, or a torn
    // write) must be reported as a MISS so ensure_template rebuilds —
    // otherwise the guest mount fails -EINVAL on the missing magic.
    // store_atomic does not validate content, so a magic-less body
    // publishes; lookup(Btrfs, ...) must reject it while lookup(Raw,
    // ...) (no content-validation) still finds it.
    let tmp = tempfile::tempdir().expect("create tempdir");
    let _guard =
        crate::test_support::test_helpers::EnvVarGuard::set(crate::KTSTR_CACHE_DIR_ENV, tmp.path());
    let cache_root_path = cache_root().unwrap();
    std::fs::create_dir_all(&cache_root_path).unwrap();
    let staged = cache_root_path.join("staged.img");
    // All-zero image larger than the magic offset — the stale-empty
    // template shape that motivated the fix.
    std::fs::write(&staged, vec![0u8; 0x1_0048]).unwrap();
    let key = "btrfs-256m";
    store_atomic(Filesystem::Raw, key, &staged).expect("store_atomic publishes");
    assert!(
        lookup(Filesystem::Raw, key)
            .expect("raw lookup ok")
            .is_some(),
        "Raw lookup skips content-validation and finds the magic-less file",
    );
    assert!(
        lookup(Filesystem::Btrfs, key)
            .expect("btrfs lookup ok")
            .is_none(),
        "Btrfs lookup must reject a template lacking the superblock magic",
    );
}

#[test]
fn lookup_btrfs_accepts_magic_stamped_template() {
    // Positive half: a template carrying the btrfs superblock magic at
    // offset 0x10040 is a valid hit under Btrfs content-validation.
    let tmp = tempfile::tempdir().expect("create tempdir");
    let _guard =
        crate::test_support::test_helpers::EnvVarGuard::set(crate::KTSTR_CACHE_DIR_ENV, tmp.path());
    let cache_root_path = cache_root().unwrap();
    std::fs::create_dir_all(&cache_root_path).unwrap();
    let staged = cache_root_path.join("staged.img");
    let (offset, magic) = Filesystem::Btrfs
        .superblock_magic()
        .expect("btrfs declares a superblock magic");
    let mut body = vec![0u8; offset as usize + 8];
    body[offset as usize..offset as usize + 8].copy_from_slice(&magic.to_le_bytes());
    std::fs::write(&staged, body).unwrap();
    let key = "btrfs-256m";
    store_atomic(Filesystem::Raw, key, &staged).expect("store_atomic publishes");
    assert!(
        lookup(Filesystem::Btrfs, key)
            .expect("btrfs lookup ok")
            .is_some(),
        "Btrfs lookup must accept a template carrying the superblock magic",
    );
}

#[test]
fn store_atomic_idempotent_on_existing_entry() {
    // If a peer published between lookup() and store_atomic(),
    // the second store_atomic returns the existing path rather
    // than raising. This test uses Filesystem::Raw, whose
    // superblock_magic() is None, so the content re-validate is
    // skipped and the legacy discard-ours early return holds (the
    // existing entry wins, ours is discarded). On Filesystem::Btrfs,
    // store_atomic re-validates the existing magic and REPLACES a stale entry
    // instead — see store_atomic_replaces_stale_invalid_btrfs_entry.
    let tmp = tempfile::tempdir().expect("create tempdir");
    let _guard =
        crate::test_support::test_helpers::EnvVarGuard::set(crate::KTSTR_CACHE_DIR_ENV, tmp.path());
    let cache_root_path = cache_root().unwrap();
    std::fs::create_dir_all(&cache_root_path).unwrap();
    let staged1 = cache_root_path.join("staged1.img");
    std::fs::write(&staged1, b"FIRST").unwrap();
    let key = "idem-key";
    let installed1 = store_atomic(Filesystem::Raw, key, &staged1).unwrap();
    // Second call with a different staging file must return the
    // already-installed path without overwriting it.
    let staged2 = cache_root_path.join("staged2.img");
    std::fs::write(&staged2, b"SECOND").unwrap();
    let installed2 = store_atomic(Filesystem::Raw, key, &staged2).unwrap();
    assert_eq!(installed1, installed2);
    // Content must remain "FIRST" — store_atomic on an existing
    // entry is a no-op publish.
    let body = std::fs::read(&installed2).unwrap();
    assert_eq!(body, b"FIRST");
}

/// Early-return cleanup contract: when `store_atomic` discovers
/// the cache entry is already published (peer raced us between
/// lookup and store), the now-obsolete staging image at
/// `src_path` MUST be unlinked before returning. Otherwise the
/// staging image leaks in the cache root forever — no other
/// code path GCs an unattached staging image at this name (the
/// debris sweep targets `template.img.in-flight.<key>.<pid>` and
/// `<key>.tmp.<pid>` patterns, not the in-flight name the caller
/// chose for `src_path`).
///
/// Uses `Filesystem::Raw` (superblock_magic() == None) so the content
/// re-validate is skipped and the discard-ours-with-unlink path runs;
/// on `Filesystem::Btrfs` a stale existing entry would be replaced
/// instead (see store_atomic_replaces_stale_invalid_btrfs_entry).
#[test]
fn store_atomic_unlinks_src_on_idempotent_early_return() {
    let tmp = tempfile::tempdir().expect("create tempdir");
    let _guard =
        crate::test_support::test_helpers::EnvVarGuard::set(crate::KTSTR_CACHE_DIR_ENV, tmp.path());
    let cache_root_path = cache_root().unwrap();
    std::fs::create_dir_all(&cache_root_path).unwrap();
    // First publish populates the cache entry.
    let staged1 = cache_root_path.join("staged1.img");
    std::fs::write(&staged1, b"FIRST").unwrap();
    let key = "early-return-key";
    store_atomic(Filesystem::Raw, key, &staged1).unwrap();
    // Second call must observe the existing entry, return the
    // already-installed path, AND unlink staged2 so it does not
    // leak.
    let staged2 = cache_root_path.join("staged2.img");
    std::fs::write(&staged2, b"SECOND").unwrap();
    store_atomic(Filesystem::Raw, key, &staged2).unwrap();
    assert!(
        !staged2.exists(),
        "early-return path must unlink the obsolete staging image \
             at {staged2:?}; without this cleanup the cache root \
             accumulates orphan staging files across every concurrent \
             peer that loses the publish race",
    );
}

#[test]
fn store_atomic_replaces_stale_invalid_btrfs_entry() {
    // When a Btrfs cache entry already exists but is STALE (no
    // valid superblock magic), store_atomic must REPLACE it with the
    // freshly-built image — NOT discard the rebuild and re-bless the
    // stale entry. Pins the exact failure mode fixed here (a 2-day-old
    // all-zero template survived every rebuild because the discard-ours
    // early return kept it). FAILS on the pre-fix code.
    let tmp = tempfile::tempdir().expect("create tempdir");
    let _guard =
        crate::test_support::test_helpers::EnvVarGuard::set(crate::KTSTR_CACHE_DIR_ENV, tmp.path());
    let cache_root_path = cache_root().unwrap();
    std::fs::create_dir_all(&cache_root_path).unwrap();
    let (offset, magic) = Filesystem::Btrfs
        .superblock_magic()
        .expect("btrfs declares a superblock magic");
    let key = "btrfs-256m-stale-replace";
    // Pre-seed a STALE cache dir: template.img is all-zero (no magic).
    let final_dir = cache_root_path.join(key);
    std::fs::create_dir_all(&final_dir).unwrap();
    std::fs::write(
        final_dir.join(TEMPLATE_FILENAME),
        vec![0u8; offset as usize + 8],
    )
    .unwrap();
    // A fresh staging image carrying the correct btrfs magic.
    let staged = cache_root_path.join("staged.img");
    let mut fresh = vec![0u8; offset as usize + 8];
    fresh[offset as usize..offset as usize + 8].copy_from_slice(&magic.to_le_bytes());
    std::fs::write(&staged, &fresh).unwrap();
    // store_atomic with Btrfs MUST replace the stale entry.
    let installed = store_atomic(Filesystem::Btrfs, key, &staged).expect("replace stale entry");
    let installed_bytes = std::fs::read(&installed).unwrap();
    assert_eq!(
        &installed_bytes[offset as usize..offset as usize + 8],
        &magic.to_le_bytes(),
        "F1 must replace the stale cache entry with the fresh rebuild's magic, \
         not keep the stale zeros",
    );
    assert!(
        !staged.exists(),
        "the fresh staging image must be consumed by the install rename",
    );
}

#[test]
fn locate_host_binary_actionable_error_when_missing() {
    // Override PATH to a single empty dir so the host binary is
    // guaranteed to be missing.
    let tmp = tempfile::tempdir().expect("create tempdir");
    let _guard = crate::test_support::test_helpers::EnvVarGuard::set("PATH", tmp.path());
    let err = locate_host_binary("nonexistent-binary-9242", "imagined-package")
        .expect_err("must error when binary absent");
    let msg = err.to_string();
    assert!(
        msg.contains("nonexistent-binary-9242"),
        "error names the binary: {msg}",
    );
    assert!(
        msg.contains("imagined-package"),
        "error names the package hint: {msg}",
    );
}

/// `locate_host_mkfs(Filesystem::Raw)` returns `Ok(None)` without
/// touching `PATH`. Pin the short-circuit branch so a regression
/// that always falls through to [`locate_host_binary`] for `Raw`
/// surfaces here — that regression would either bail spuriously
/// (no `mkfs.raw` on PATH) or, worse, locate an unrelated binary
/// named `<empty>` and pack it into the template-VM initramfs.
/// This test exercises the `Raw` arm of
/// [`Filesystem::mkfs_binary_name`]'s `match` via the
/// [`locate_host_mkfs`] entry point.
///
/// PATH is forced to an empty tempdir so a `Some(_)` result
/// would have to come from a phantom PATH walk that ignores the
/// `None` short-circuit; the empty-tempdir override removes the
/// possibility that the test passes for the wrong reason.
#[test]
fn locate_host_mkfs_raw_returns_none() {
    let tmp = tempfile::tempdir().expect("create tempdir");
    let _path_guard = crate::test_support::test_helpers::EnvVarGuard::set("PATH", tmp.path());
    let result =
        locate_host_mkfs(Filesystem::Raw).expect("Raw must short-circuit before any PATH walk");
    assert!(
        result.is_none(),
        "Filesystem::Raw has no userspace formatter; \
             locate_host_mkfs must return Ok(None) without consulting \
             PATH. Got: {result:?}",
    );
}

/// [`mkfs_version_fingerprint`] is deterministic for the same
/// binary: two invocations against the same path produce
/// byte-identical fingerprints. Pin the determinism contract so
/// a regression that includes a timestamp / random nonce in the
/// fingerprint would surface here. Without this property the
/// cache key would rotate on every call and defeat caching
/// entirely.
///
/// Searches `PATH` for a series of binaries known to emit a
/// stable `--version` banner (coreutils `cat`, `ls`, `true`).
/// At least one of these is on every Linux distro ktstr
/// supports; the first to produce non-empty output for
/// `--version` wins. We don't care WHAT the fingerprint says,
/// only that it's stable across two invocations.
///
/// Skips when none of the candidate binaries produces output
/// for `--version` (extremely rare — would require a
/// busybox-only system that strips `--version` from every
/// candidate).
#[test]
fn mkfs_version_fingerprint_is_deterministic() {
    let path_var = match std::env::var_os("PATH") {
        Some(p) => p,
        None => return,
    };
    // Try several candidates; the first to produce non-empty
    // `--version` output wins. `cat`/`ls` are GNU coreutils
    // mainstays that emit a multi-line banner on `--version`;
    // even on busybox, `cat --version` typically emits a
    // banner-shaped one-liner.
    let mut working_binary: Option<PathBuf> = None;
    for name in &["cat", "ls", "true"] {
        for dir in std::env::split_paths(&path_var) {
            let candidate = dir.join(name);
            if !std::fs::metadata(&candidate)
                .map(|m| m.is_file())
                .unwrap_or(false)
            {
                continue;
            }
            // Probe: does `--version` produce any output?
            let probe = std::process::Command::new(&candidate)
                .arg("--version")
                .output();
            let Ok(output) = probe else {
                continue;
            };
            if !output.stdout.is_empty() || !output.stderr.is_empty() {
                working_binary = Some(candidate);
                break;
            }
        }
        if working_binary.is_some() {
            break;
        }
    }
    let Some(binary_path) = working_binary else {
        return;
    };
    let fp1 =
        mkfs_version_fingerprint(&binary_path).expect("first --version invocation must succeed");
    let fp2 =
        mkfs_version_fingerprint(&binary_path).expect("second --version invocation must succeed");
    assert_eq!(
        fp1, fp2,
        "fingerprint must be deterministic across repeated \
             invocations of the same binary"
    );
    assert_eq!(
        fp1.len(),
        16,
        "fingerprint must render as 16 hex chars (64 bits): {fp1}",
    );
    assert!(
        fp1.chars().all(|c| c.is_ascii_hexdigit()),
        "fingerprint must be hex-only: {fp1}",
    );
    // The first call must have populated the per-process cache.
    // Pin the cache write so a regression that drops the
    // memoization (and re-execs `--version` on every call)
    // surfaces here.
    let cached = mkfs_version_fingerprint_cache()
        .lock()
        .expect("cache mutex")
        .get(&binary_path)
        .cloned();
    assert_eq!(
        cached.as_deref(),
        Some(fp1.as_str()),
        "first call must populate the per-process fingerprint cache; \
             without the cache, ensure_template re-execs `--version` on \
             every VM boot",
    );
}

#[test]
fn build_template_via_vm_rejects_raw_filesystem() {
    // [`build_template_via_vm`] is only supposed to be invoked
    // from filesystem variants that require pre-formatting. A
    // `Filesystem::Raw` argument means a caller bypassed the
    // gate in [`crate::vmm::KtstrVm::init_virtio_blk`] and would
    // produce a no-op template (Raw disks have no on-disk
    // format). Pin the rejection so that bypass surfaces as a
    // bail with a hint at the offending caller rather than as a
    // silent empty template.
    let tmp = tempfile::tempdir().expect("create tempdir");
    let _guard =
        crate::test_support::test_helpers::EnvVarGuard::set(crate::KTSTR_CACHE_DIR_ENV, tmp.path());
    let err = build_template_via_vm(Filesystem::Raw, 256 * 1024 * 1024, tmp.path(), "raw-256m")
        .expect_err("Raw must be rejected");
    let msg = err.to_string();
    assert!(
        msg.contains("Filesystem::Raw"),
        "error must name the rejected variant: {msg}",
    );
    assert!(
        msg.contains("init_virtio_blk"),
        "error must name the gate location for the operator: {msg}",
    );
}

#[test]
fn verify_cache_dir_walks_up_to_existing_ancestor() {
    // A non-existent cache root must still resolve a usable statfs
    // target by walking up to its nearest existing ancestor. Anchor
    // the missing path under a per-test tempdir so parallel runs do
    // not collide on a shared system path. Only `<tempdir>` itself
    // exists; `<tempdir>/nonexistent`, `.../sub`, and `.../dir` do
    // not, so the walk-up MUST land on `<tempdir>` exactly.
    let tmp = tempfile::tempdir().expect("create tempdir");
    let nonexistent = tmp.path().join("nonexistent/sub/dir");

    // POSITIVE contract: the walk-up reached the existing ancestor
    // (the tempdir), not the missing leaf and not a higher ancestor.
    // This is the "walked up to an existing ancestor" behaviour the
    // test name promises — without this assertion a regression that
    // returned `dir` unchanged (skipping the walk) or over-ascended
    // to `/` would pass on a green outer verify.
    let resolved = resolve_existing_ancestor(&nonexistent)
        .expect("an existing ancestor (the tempdir) must be found");
    assert_eq!(
        resolved,
        tmp.path(),
        "walk-up must stop at the nearest existing ancestor (the \
         tempdir), got {resolved:?}",
    );

    // And the full verify drives that same resolved ancestor into
    // statfs: it either returns Ok (tempdir on btrfs/xfs) or a
    // fs-magic-named error (any other filesystem). Either outcome is
    // valid; the load-bearing assertion is the walk-up above.
    match verify_cache_dir_supports_reflink(&nonexistent) {
        Ok(()) => { /* tempdir lives on btrfs/xfs */ }
        Err(e) => {
            let msg = e.to_string();
            assert!(
                msg.contains("statfs.f_type") || msg.contains("FICLONE"),
                "unexpected error wording: {msg}",
            );
        }
    }
}

/// When the walk-up lands on an ancestor (`probe != dir`), the
/// bail diagnostic appends a `probe_note` that names the probed
/// ancestor explicitly so the operator can tell the f_type came
/// from an ancestor rather than `dir` itself. Pins the
/// conditional interpolation: a regression that drops
/// `{probe_note}` from the bail string would silently strip the
/// "(no part of {dir:?} exists yet; ... ancestor {probe:?} ...)"
/// guidance, leaving operators with the misleading
/// "cache directory X lives on f_type Y" wording even when Y
/// came from a probed ancestor.
///
/// Skipped when the tempdir lives on btrfs/xfs — the helper
/// returns Ok and there is no diagnostic to inspect. Most
/// CI runners use tmpfs or ext4 for `TMPDIR`, so the
/// assertion fires there.
#[test]
fn verify_cache_dir_probe_note_fires_when_probe_differs_from_dir() {
    let tmp = tempfile::tempdir().expect("create tempdir");
    let nonexistent = tmp.path().join("nonexistent/sub/dir");
    match verify_cache_dir_supports_reflink(&nonexistent) {
        Ok(()) => {
            // tempdir lives on btrfs/xfs — no diagnostic emitted,
            // skip the probe_note assertion.
        }
        Err(e) => {
            let msg = e.to_string();
            assert!(
                msg.contains("ancestor") && msg.contains("no part of"),
                "walk-up diagnostic must surface the probed \
                     ancestor when probe != dir; got: {msg}",
            );
        }
    }
}

/// When `dir` itself exists (`probe == dir`), the bail diagnostic
/// MUST NOT include the probe_note text — that text is
/// conditional on the walk-up landing on an ancestor. Pins the
/// `probe == dir` branch of the conditional interpolation: a
/// regression that always emits the probe_note (e.g. drops the
/// `if probe == dir` guard) would leak the misleading "no part
/// of dir exists yet" wording on every non-btrfs/xfs probe.
///
/// Skipped when the tempdir lives on btrfs/xfs — the helper
/// returns Ok and there is no diagnostic to inspect.
#[test]
fn verify_cache_dir_probe_note_absent_when_probe_equals_dir() {
    let tmp = tempfile::tempdir().expect("create tempdir");
    match verify_cache_dir_supports_reflink(tmp.path()) {
        Ok(()) => {
            // tempdir lives on btrfs/xfs — no diagnostic emitted.
        }
        Err(e) => {
            let msg = e.to_string();
            assert!(
                !msg.contains("ancestor") && !msg.contains("no part of"),
                "probe == dir branch must NOT emit the probe_note \
                     text; got: {msg}",
            );
            // Sanity: the rest of the diagnostic still names the
            // f_type so the operator gets actionable guidance.
            assert!(
                msg.contains("statfs.f_type") || msg.contains("FICLONE"),
                "diagnostic must still name the f_type; got: {msg}",
            );
        }
    }
}

/// `Path::exists` follows symlinks, so a dangling symlink
/// probes as missing and the walk-up moves to the symlink
/// container's parent rather than the (nonexistent) target's
/// parent. Pin the documented behaviour at
/// `verify_cache_dir_supports_reflink`'s "Symlink behaviour"
/// paragraph: the diagnostic must reference the tempdir's
/// f_type (the container, which exists) rather than failing on
/// the broken symlink.
///
/// A regression that switches `Path::exists` to
/// `Path::try_exists` would surface here: try_exists returns
/// `Err` on a broken symlink, breaking the walk-up loop
/// invariant.
///
/// Linux-only: requires `std::os::unix::fs::symlink`. Skipped
/// when the tempdir lives on btrfs/xfs (helper returns Ok by
/// walking up to a reflink-capable filesystem, which is the
/// correct outcome).
#[cfg(target_os = "linux")]
#[test]
fn verify_cache_dir_walks_through_dangling_symlink() {
    let tmp = tempfile::tempdir().expect("create tempdir");
    let symlink_path = tmp.path().join("dangling");
    // Target does not exist; dangling symlink lands in the
    // tempdir.
    std::os::unix::fs::symlink("/nonexistent-symlink-target-9242", &symlink_path)
        .expect("create dangling symlink");
    // Probing a path under the dangling symlink: walk-up
    // ascends to symlink_path → tmp.path() (the symlink's
    // container). The symlink target's parent is never
    // consulted.
    let probe_path = symlink_path.join("sub");
    match verify_cache_dir_supports_reflink(&probe_path) {
        Ok(()) => {
            // tempdir lives on btrfs/xfs — helper returned Ok
            // by walking up to a reflink-capable filesystem,
            // which is the correct outcome.
        }
        Err(e) => {
            let msg = e.to_string();
            // The diagnostic must reference the f_type of the
            // walked-up ancestor (tempdir's filesystem) rather
            // than failing on the dangling symlink. The error
            // wording always names the f_type magic, regardless
            // of whether the probed ancestor is the original
            // dir or an ancestor.
            assert!(
                msg.contains("statfs.f_type") || msg.contains("FICLONE"),
                "symlink walk-up must produce an f_type-named \
                     diagnostic, not a symlink-resolution error; got: {msg}",
            );
        }
    }
}

/// Cross-key concurrency invariant: two distinct cache keys held
/// by the same pid produce distinct staging-image paths. Without
/// the cache_key qualifier in the filename, the same process
/// concurrently building `btrfs-256m` and `btrfs-1024m` would
/// collide on `template.img.in-flight.<pid>` — the second open
/// would truncate the first's image while it boots, corrupting
/// the template the first build is formatting. Pin the
/// uniqueness contract here so a regression that drops the
/// cache_key from [`staging_image_path`] surfaces immediately
/// rather than as a flaky cross-key test.
#[test]
fn staging_image_path_is_unique_per_key_and_pid() {
    let cache_root = std::path::Path::new("/tmp/ktstr-fake-cache-root");
    let pid = 12_345u32;
    let p_256 = staging_image_path(cache_root, "btrfs-256m", pid);
    let p_1024 = staging_image_path(cache_root, "btrfs-1024m", pid);
    // Same pid, different keys → different paths.
    assert_ne!(
        p_256, p_1024,
        "cache_key qualifier missing from staging-image path: \
             distinct keys collided",
    );
    // Both paths embed the cache_key and the pid verbatim.
    assert!(
        p_256
            .to_string_lossy()
            .contains("template.img.in-flight.btrfs-256m.12345"),
        "256m staging path missing key/pid token: {p_256:?}",
    );
    assert!(
        p_1024
            .to_string_lossy()
            .contains("template.img.in-flight.btrfs-1024m.12345"),
        "1024m staging path missing key/pid token: {p_1024:?}",
    );
    // Same key, different pids → different paths (per-pid debris
    // never collides with a live peer's staging file).
    let p_256_other_pid = staging_image_path(cache_root, "btrfs-256m", 67_890);
    assert_ne!(p_256, p_256_other_pid);

    // Idempotence: same input → same output. Defends against a
    // future regression that introduces nondeterminism (e.g.
    // reads `process::id()` internally instead of taking pid as
    // an argument, or appends a randomised suffix). The function
    // must be a pure mapping from `(cache_root, key, pid)` to
    // `PathBuf` so the per-key flock and the staging-image path
    // can coordinate without surprise.
    assert_eq!(
        p_256,
        staging_image_path(cache_root, "btrfs-256m", pid),
        "staging_image_path must be a pure function of its inputs",
    );
}

/// Cleanup contract for the [`create_and_size_staging_image`]
/// helper: when `set_len` fails (ENOSPC, EFBIG, EINVAL, etc.)
/// the just-created empty file must be unlinked before
/// propagating the error, so the cache root does not accumulate
/// 0-byte staging images across retries.
///
/// Drives the failure via `set_len(u64::MAX)`:
/// [`std::fs::File::set_len`] internally `try_into::<i64>()`-s
/// its `u64` argument and returns an `io::Error` of kind
/// `InvalidInput` ("out of range integral type conversion
/// attempted") for any value above `i64::MAX`, BEFORE issuing
/// the `ftruncate(2)` syscall. That gives a deterministic,
/// process-local, signal-free failure path — no `RLIMIT_FSIZE`
/// manipulation, no SIGXFSZ disposition juggling, no parallel-
/// test cross-talk. The cleanup arm semantics are identical
/// regardless of whether the failure originates in the std
/// pre-syscall guard or in the kernel itself, so this exercises
/// the same drop-fd-then-unlink path that ENOSPC / EFBIG / EINVAL
/// in production hit.
///
/// Without the cleanup, the just-created 0-byte file would
/// persist (the open succeeded; only the size enlargement
/// failed). The post-condition asserts ENOENT at the staging
/// path after the helper returns Err.
#[test]
fn create_and_size_staging_image_cleans_up_on_set_len_failure() {
    let tmp = tempfile::tempdir().expect("create tempdir");
    let staging_path = tmp.path().join("template.img.in-flight.btrfs-256m.0");

    // u64::MAX > i64::MAX → File::set_len returns InvalidInput
    // before any ftruncate syscall is issued. Sentinel choice
    // pins to this Rust-side guard rather than to a kernel
    // errno that varies across filesystems.
    let err = create_and_size_staging_image(&staging_path, u64::MAX)
        .expect_err("set_len(u64::MAX) must fail at the i64 cast");
    let msg = err.to_string();
    assert!(
        msg.contains("set staging image length"),
        "error must surface the set_len-failed context: {msg}",
    );

    // The cleanup arm must have unlinked the 0-byte file.
    // Verify by stat'ing the path: ENOENT is the success
    // criterion. Distinguishes the cleanup-fired success case
    // from the cleanup-skipped regression where the empty file
    // still sits on disk waiting to leak across retries.
    match std::fs::metadata(&staging_path) {
        Err(e) if e.kind() == io::ErrorKind::NotFound => { /* ok */ }
        Ok(m) => panic!(
            "staging image not cleaned up after set_len failure: \
                 still exists at {staging_path:?} ({} bytes)",
            m.len(),
        ),
        Err(e) => panic!("unexpected stat error: {e}"),
    }
}

/// Determinism contract for [`fsid_bytes`]: two `statfs` calls
/// against the same path must produce byte-identical
/// `fsid_bytes` outputs. The bytewise `f_fsid` read in
/// [`fsid_bytes`] sidesteps the private `__val` field on
/// `libc::fsid_t`; this test pins the same-input → same-output
/// property through the actual host libc. A regression that,
/// for instance, mis-sizes the read or includes uninitialised
/// padding would surface here as flaky byte mismatches across
/// the pair of statfs calls.
///
/// Uses a tempdir so the test does not depend on operator
/// state — `tempfile::tempdir()` resolves under `TMPDIR` /
/// `$XDG_RUNTIME_DIR` / `/tmp`, all real filesystems with a
/// stable `f_fsid` for the duration of the test.
#[test]
fn fsid_bytes_is_deterministic_for_same_path() {
    let tmp = tempfile::tempdir().expect("create tempdir");
    let buf1 = statfs_path(tmp.path()).expect("first statfs");
    let buf2 = statfs_path(tmp.path()).expect("second statfs");
    assert_eq!(
        fsid_bytes(&buf1),
        fsid_bytes(&buf2),
        "fsid_bytes must be deterministic across repeated statfs \
             calls against the same path; a mismatch would indicate \
             the bytewise f_fsid read produces different output for \
             the same input on this host",
    );
}

/// Cross-filesystem distinguishability for [`fsid_bytes`]: two
/// paths that live on distinct filesystems must produce
/// different `fsid_bytes` outputs. This is the property
/// [`store_atomic`] relies on at the cross-fs gate (`f_fsid`
/// inequality across two distinct btrfs subvolumes is the
/// reason `f_fsid` is compared in addition to `f_type`).
///
/// Probes `tempfile::tempdir()` against a list of standard
/// pseudo filesystems (`/proc`, `/sys`, `/dev`, `/`) ordered
/// most-likely-distinct first. The first candidate whose
/// statfs differs from the tempdir's exercises the
/// distinguishability invariant; the test asserts inequality
/// loudly and returns. If NO candidate produces a different
/// f_type-or-fsid, the test fails LOUDLY because silent-skip
/// would falsely report green when the cross-fs property at
/// `store_atomic` was never exercised. Probe outcomes
/// (per-candidate "same fs" / statfs error reasons) are
/// surfaced in the panic message so the operator can see WHY
/// no candidate distinguished — e.g. a minimal container with
/// every probe collapsed onto the rootfs.
#[test]
fn fsid_bytes_distinguishes_different_filesystems() {
    let tmp = tempfile::tempdir().expect("create tempdir");
    let tmp_buf = statfs_path(tmp.path()).expect("statfs tempdir");
    let tmp_fsid = fsid_bytes(&tmp_buf);

    // Most-likely-distinct first; rootfs `/` last (collapses on
    // minimal containers).
    let candidates: &[&str] = &["/proc", "/sys", "/dev", "/"];
    let mut probe_outcomes: Vec<String> = Vec::with_capacity(candidates.len());
    for cand in candidates {
        let path = std::path::Path::new(cand);
        match statfs_path(path) {
            Ok(buf) => {
                let fsid = fsid_bytes(&buf);
                if buf.f_type != tmp_buf.f_type || fsid != tmp_fsid {
                    assert_ne!(
                        tmp_fsid, fsid,
                        "fsid_bytes must differ across distinct filesystems \
                             (tempdir f_type=0x{:x}, {cand} f_type=0x{:x}); a match \
                             would indicate the bytewise f_fsid read is producing a \
                             constant byte pattern instead of the real fsid_t — \
                             e.g. reading from a wrong offset within libc::statfs",
                        tmp_buf.f_type, buf.f_type,
                    );
                    return;
                }
                probe_outcomes.push(format!(
                    "{cand}: same fs (f_type=0x{:x}, fsid==tempdir)",
                    buf.f_type,
                ));
            }
            Err(e) => {
                probe_outcomes.push(format!("{cand}: statfs error ({e})"));
            }
        }
    }
    panic!(
        "fsid_bytes_distinguishes_different_filesystems found no candidate path \
             that resolves to a different filesystem from tempdir (f_type=0x{:x}). \
             At least one of the standard pseudo filesystems should mount \
             independently of /tmp; the absence of any distinguishing path is \
             anomalous — the cross-fs property at store_atomic depends on \
             distinguishability, so silent-skip would falsely report green. \
             Probe outcomes: {probe_outcomes:?}",
        tmp_buf.f_type,
    );
}

// -- clean_orphaned_tmp_dirs / clean_all coverage ------------

/// `clean_orphaned_tmp_dirs` returns `Ok(0)` and does not
/// error when the cache root does not exist. Mirrors the
/// early-return contract that lets `clean_all` invoke this on
/// a never-materialised root without bailing.
#[test]
fn clean_orphaned_tmp_dirs_handles_missing_root() {
    let tmp = tempfile::tempdir().expect("create tempdir");
    let nonexistent = tmp.path().join("never-created");
    let count = clean_orphaned_tmp_dirs(&nonexistent).expect("missing root must not error");
    assert_eq!(count, 0, "missing root sweeps zero entries");
}

/// `clean_orphaned_tmp_dirs` removes a stale staging image
/// (`template.img.in-flight.<key>.<pid>`) when the embedded
/// pid is dead. Uses pid=1 with a sentinel suffix that
/// distinguishes the "dead" path from a real pid: pid=1 is
/// reserved for init and exists; instead we use the highest
/// possible pid value (`i32::MAX`) which is guaranteed not
/// to be allocated on Linux — `kernel/pid.c` caps at
/// `PID_MAX_LIMIT = 4194304` (2^22), well below i32::MAX.
#[test]
fn clean_orphaned_tmp_dirs_removes_dead_pid_staging_image() {
    let tmp = tempfile::tempdir().expect("create tempdir");
    let cache_root = tmp.path();
    // i32::MAX > PID_MAX_LIMIT (2^22); guaranteed-dead.
    let dead_pid = i32::MAX;
    let leaked = cache_root.join(format!("template.img.in-flight.btrfs-256m.{dead_pid}",));
    std::fs::write(&leaked, b"FAKE_STAGING_IMG").unwrap();
    let count = clean_orphaned_tmp_dirs(cache_root).expect("sweep must succeed");
    assert_eq!(count, 1, "exactly one debris entry removed");
    assert!(!leaked.exists(), "dead-pid staging image must be unlinked",);
}

/// `clean_orphaned_tmp_dirs` removes a stale staging directory
/// (`<key>.tmp.<pid>`) when the embedded pid is dead. Mirrors
/// the previous test for the second debris shape.
#[test]
fn clean_orphaned_tmp_dirs_removes_dead_pid_staging_directory() {
    let tmp = tempfile::tempdir().expect("create tempdir");
    let cache_root = tmp.path();
    let dead_pid = i32::MAX;
    let leaked = cache_root.join(format!("btrfs-256m.tmp.{dead_pid}"));
    std::fs::create_dir_all(&leaked).unwrap();
    std::fs::write(leaked.join("template.img"), b"PARTIAL").unwrap();
    let count = clean_orphaned_tmp_dirs(cache_root).expect("sweep must succeed");
    assert_eq!(count, 1, "exactly one debris entry removed");
    assert!(
        !leaked.exists(),
        "dead-pid staging directory must be removed",
    );
}

/// `clean_orphaned_tmp_dirs` removes a stale per-test FICLONE
/// backing file (`.per-test-<pid>-<ns>-<rnd>.img`) when the
/// embedded pid is dead. Pin the third debris shape contract:
/// without sweeping these, every crashed test leaks one such
/// file in the cache root permanently — the in-process unlink
/// at [`crate::vmm::KtstrVm::init_virtio_blk`] is best-effort
/// (warn-only on failure) and skipped entirely when SIGKILL
/// fires between FICLONE and the unlink.
#[test]
fn clean_orphaned_tmp_dirs_removes_dead_pid_per_test_image() {
    let tmp = tempfile::tempdir().expect("create tempdir");
    let cache_root = tmp.path();
    let dead_pid = i32::MAX;
    let leaked = cache_root.join(format!(".per-test-{dead_pid}-deadbeef-cafe.img"));
    std::fs::write(&leaked, b"FAKE_PER_TEST_IMG").unwrap();
    let count = clean_orphaned_tmp_dirs(cache_root).expect("sweep must succeed");
    assert_eq!(count, 1, "exactly one debris entry removed");
    assert!(
        !leaked.exists(),
        "dead-pid per-test backing file must be unlinked",
    );
}

/// `clean_orphaned_tmp_dirs` PRESERVES a per-test backing file
/// owned by the current process — the in-process unlink path
/// at [`crate::vmm::KtstrVm::init_virtio_blk`] runs after
/// FICLONE returns; if the sweep ran concurrently with a live
/// test that just FICLONE'd but hasn't yet unlinked, the
/// sweep MUST NOT yank the file out from under the live
/// device.
#[test]
fn clean_orphaned_tmp_dirs_preserves_live_pid_per_test_image() {
    let tmp = tempfile::tempdir().expect("create tempdir");
    let cache_root = tmp.path();
    let live_pid = std::process::id();
    let live_file = cache_root.join(format!(".per-test-{live_pid}-deadbeef-cafe.img"));
    std::fs::write(&live_file, b"LIVE_PER_TEST_BACKING").unwrap();
    let count = clean_orphaned_tmp_dirs(cache_root).expect("sweep must succeed");
    assert_eq!(
        count, 0,
        "live-pid per-test backing must not be removed by sweep",
    );
    assert!(
        live_file.exists(),
        "live-pid per-test backing must survive the sweep",
    );
}

/// `clean_orphaned_tmp_dirs` PRESERVES debris owned by a live
/// peer pid. The current process's own pid is the obvious
/// "live" sentinel: as long as this test is running,
/// `kill(getpid(), None)` returns `Ok(())`, NOT `Err(ESRCH)`.
/// Without this skip, a multi-process ktstr operator running
/// `cargo ktstr disk-template clean` while a sibling test is
/// in flight would yank the sibling's staging file mid-build.
#[test]
fn clean_orphaned_tmp_dirs_preserves_live_pid_debris() {
    let tmp = tempfile::tempdir().expect("create tempdir");
    let cache_root = tmp.path();
    let live_pid = std::process::id();
    let live_image = cache_root.join(format!("template.img.in-flight.btrfs-256m.{live_pid}",));
    std::fs::write(&live_image, b"LIVE_PEER_DEBRIS").unwrap();
    let count = clean_orphaned_tmp_dirs(cache_root).expect("sweep must succeed");
    assert_eq!(
        count, 0,
        "no entries removed when only live-pid debris exists",
    );
    assert!(
        live_image.exists(),
        "live-pid debris must be preserved across sweep",
    );
}

/// `clean_orphaned_tmp_dirs` does NOT touch published cache
/// entries (`<cache_key>/`) — those have no pid suffix and
/// don't match either debris pattern. Pin the
/// non-removal contract for published entries; a regression
/// that broadened the prefix filter would silently delete
/// healthy templates.
#[test]
fn clean_orphaned_tmp_dirs_preserves_published_entries() {
    let tmp = tempfile::tempdir().expect("create tempdir");
    let cache_root = tmp.path();
    // Published entry: directory whose name matches a cache
    // key (no `.tmp.` infix, no `template.img.in-flight.`
    // prefix) containing a `template.img`.
    let published = cache_root.join("btrfs-256m");
    std::fs::create_dir_all(&published).unwrap();
    std::fs::write(published.join(TEMPLATE_FILENAME), b"GOOD").unwrap();
    let count = clean_orphaned_tmp_dirs(cache_root).expect("sweep must succeed");
    assert_eq!(
        count, 0,
        "published cache entries must not be swept by debris GC",
    );
    assert!(published.is_dir(), "published entry must survive");
    assert!(
        published.join(TEMPLATE_FILENAME).is_file(),
        "published template.img must survive",
    );
}

/// `clean_orphaned_tmp_dirs` skips the `.locks/` subdirectory
/// — it's not debris, it's the lockfile namespace. Pin the
/// skip so a regression that broadened the prefix filter
/// (e.g. adding `.locks` to a generic dotfile bucket) does
/// not shatter the lockfile inodes that live peers may have
/// open.
#[test]
fn clean_orphaned_tmp_dirs_preserves_lock_subdirectory() {
    let tmp = tempfile::tempdir().expect("create tempdir");
    let cache_root = tmp.path();
    let locks = cache_root.join(LOCK_DIR_NAME);
    std::fs::create_dir_all(&locks).unwrap();
    std::fs::write(locks.join("btrfs-256m.lock"), b"").unwrap();
    let count = clean_orphaned_tmp_dirs(cache_root).expect("sweep must succeed");
    assert_eq!(count, 0, ".locks/ must be invisible to the debris sweep",);
    assert!(locks.is_dir(), ".locks/ subdirectory must survive");
    assert!(
        locks.join("btrfs-256m.lock").is_file(),
        "individual lockfiles must survive",
    );
}

/// `clean_all` removes a published entry and reports the
/// count. Stages a fake template via `store_atomic`, then
/// calls `clean_all` and asserts the entry is gone and the
/// returned count is 1.
#[test]
fn clean_all_removes_published_entry() {
    let tmp = tempfile::tempdir().expect("create tempdir");
    let _guard =
        crate::test_support::test_helpers::EnvVarGuard::set(crate::KTSTR_CACHE_DIR_ENV, tmp.path());
    let cache_root_path = cache_root().unwrap();
    std::fs::create_dir_all(&cache_root_path).unwrap();
    let staged = cache_root_path.join("staged.img");
    std::fs::write(&staged, b"FAKE_TEMPLATE").unwrap();
    let installed =
        store_atomic(Filesystem::Raw, "btrfs-256m", &staged).expect("store_atomic publishes");
    assert!(installed.is_file());
    let count = clean_all().expect("clean_all must succeed");
    assert_eq!(count, 1, "exactly one published entry removed");
    // The published entry directory is gone.
    assert!(
        lookup(Filesystem::Raw, "btrfs-256m")
            .expect("lookup ok")
            .is_none(),
        "published entry must be gone after clean_all",
    );
    // But the lockfile inode survives.
    let lock_path = lock_path_for_key("btrfs-256m").unwrap();
    if lock_path.exists() {
        // Lock dir/file may or may not exist depending on
        // whether store_atomic touched it (this code path
        // doesn't); but if it does exist, it must NOT have
        // been removed by clean_all.
        assert!(lock_path.is_file(), "lockfile inode must survive clean_all",);
    }
}

/// `clean_all` reports 0 for an empty cache root. Pin the
/// "no entries" return value so a regression that double-
/// counts (e.g. counts the `.locks/` subdirectory) trips here.
#[test]
fn clean_all_reports_zero_on_empty_cache() {
    let tmp = tempfile::tempdir().expect("create tempdir");
    let _guard =
        crate::test_support::test_helpers::EnvVarGuard::set(crate::KTSTR_CACHE_DIR_ENV, tmp.path());
    let count = clean_all().expect("clean_all must succeed on empty");
    assert_eq!(count, 0);
}

/// `clean_all` returns 0 (not Err) on a never-materialised
/// cache root. Lets operator-driven runs against a fresh host
/// (where the cache directory has not been created yet)
/// succeed silently rather than bail.
#[test]
fn clean_all_handles_missing_cache_root() {
    let tmp = tempfile::tempdir().expect("create tempdir");
    // KTSTR_CACHE_DIR points at a path that does NOT exist
    // (no create_dir_all, no store_atomic call). cache_root()
    // resolves the path string but the directory is absent.
    let nonexistent = tmp.path().join("never-created");
    let _guard = crate::test_support::test_helpers::EnvVarGuard::set(
        crate::KTSTR_CACHE_DIR_ENV,
        &nonexistent,
    );
    let count = clean_all().expect("missing cache root must not error");
    assert_eq!(count, 0);
}

/// `clean_all` SKIPS an entry whose lockfile is currently
/// held by a live peer — even when run inside the same
/// process. Acquire the lock via `acquire_template_lock`
/// before calling `clean_all` and assert the entry survives.
/// This covers the most operationally important contract:
/// a `cargo ktstr disk-template clean` invoked while another
/// ktstr process holds the lock for an in-flight test must
/// NOT remove that entry.
///
/// We hold the lock from the SAME process to avoid spawning
/// a child; flock is per-open-file-description, so an
/// independent open in the same process produces a distinct
/// fd that is observed as a separate holder by `try_flock`
/// on a third open from `clean_all`.
#[test]
fn clean_all_skips_entry_locked_by_live_peer() {
    let tmp = tempfile::tempdir().expect("create tempdir");
    let _guard =
        crate::test_support::test_helpers::EnvVarGuard::set(crate::KTSTR_CACHE_DIR_ENV, tmp.path());
    // Stage a published entry so there's something to skip.
    let cache_root_path = cache_root().unwrap();
    std::fs::create_dir_all(&cache_root_path).unwrap();
    let staged = cache_root_path.join("staged.img");
    std::fs::write(&staged, b"FAKE_TEMPLATE").unwrap();
    let installed =
        store_atomic(Filesystem::Raw, "btrfs-256m", &staged).expect("store_atomic publishes");
    assert!(installed.is_file());
    // Hold the per-key flock from this process. `clean_all`'s
    // `try_flock(LOCK_EX|LOCK_NB)` against the same file
    // returns `Ok(None)` because EX is exclusive — even our
    // own process's prior fd blocks the second acquire (flock
    // semantics: fd-scoped, not process-scoped).
    let _hold = acquire_template_lock("btrfs-256m").expect("acquire template lock");
    let count = clean_all().expect("clean_all must succeed");
    assert_eq!(count, 0, "locked entry must not be removed by clean_all",);
    // And the entry directory must still be on disk.
    assert!(
        lookup(Filesystem::Raw, "btrfs-256m")
            .expect("lookup ok")
            .is_some(),
        "locked entry must survive clean_all",
    );
}

/// `clean_all` invokes `clean_orphaned_tmp_dirs` before
/// walking published entries. Stage a dead-pid staging image
/// alongside a published entry, run `clean_all`, and assert
/// BOTH are removed. The published entry counts toward the
/// returned value; the debris does not (per the doc
/// "`clean_all` reports published-entry removals only").
#[test]
fn clean_all_sweeps_debris_alongside_published_entries() {
    let tmp = tempfile::tempdir().expect("create tempdir");
    let _guard =
        crate::test_support::test_helpers::EnvVarGuard::set(crate::KTSTR_CACHE_DIR_ENV, tmp.path());
    let cache_root_path = cache_root().unwrap();
    std::fs::create_dir_all(&cache_root_path).unwrap();
    // Published entry.
    let staged = cache_root_path.join("staged.img");
    std::fs::write(&staged, b"FAKE_TEMPLATE").unwrap();
    store_atomic(Filesystem::Raw, "btrfs-256m", &staged).unwrap();
    // Dead-pid staging image debris.
    let dead_pid = i32::MAX;
    let debris = cache_root_path.join(format!("template.img.in-flight.btrfs-1024m.{dead_pid}",));
    std::fs::write(&debris, b"DEBRIS").unwrap();
    // Sanity: both exist before clean_all.
    assert!(debris.is_file());
    assert!(lookup(Filesystem::Raw, "btrfs-256m").unwrap().is_some());
    let count = clean_all().expect("clean_all must succeed");
    // The returned count covers published entries only (1).
    // The debris removal is documented in clean_all's body
    // but not folded into the count.
    assert_eq!(count, 1, "one published entry removed");
    // Both should be gone on disk regardless of count
    // accounting.
    assert!(
        !debris.exists(),
        "debris must be removed by the embedded sweep",
    );
    assert!(
        lookup(Filesystem::Raw, "btrfs-256m").unwrap().is_none(),
        "published entry must be removed by clean_all",
    );
}

// -- tail_lines coverage -------------------------------------

/// `tail_lines` returns the last `n` lines joined by `\n` when the
/// input has more than `n` lines. Pins the saturating-sub windowing
/// at [`tail_lines`] — the function picks `start =
/// lines.len().saturating_sub(n)` and joins from there, so a 5-line
/// input with `n=2` yields exactly the trailing two lines.
#[test]
fn tail_lines_returns_last_n_lines() {
    let text = "a\nb\nc\nd\ne";
    assert_eq!(tail_lines(text, 2), "d\ne");
    assert_eq!(tail_lines(text, 3), "c\nd\ne");
}

/// `tail_lines` returns the whole input (joined by `\n`) when `n`
/// meets or exceeds the line count — `saturating_sub` floors `start`
/// at 0 so no lines are dropped. Covers the `n >= len` branch that
/// the trailing-stderr surface at [`build_template_via_vm`] relies on
/// when the guest emitted fewer than `n` lines.
#[test]
fn tail_lines_returns_all_when_n_exceeds_line_count() {
    let text = "one\ntwo\nthree";
    // n equal to the line count.
    assert_eq!(tail_lines(text, 3), "one\ntwo\nthree");
    // n far larger than the line count — saturating_sub floors at 0.
    assert_eq!(tail_lines(text, 100), "one\ntwo\nthree");
}

/// `tail_lines` on the empty string yields the empty string:
/// `"".lines()` produces zero lines, `saturating_sub` floors `start`
/// at 0, and `[].join("\n")` is `""`. Pins the no-output corner so an
/// empty guest transcript renders cleanly in the bail message rather
/// than panicking on an out-of-range slice.
#[test]
fn tail_lines_empty_input_yields_empty_string() {
    assert_eq!(tail_lines("", 20), "");
    assert_eq!(tail_lines("", 0), "");
}

/// `tail_lines` with `n == 0` selects zero trailing lines:
/// `start = len.saturating_sub(0) == len`, so `lines[len..]` is an
/// empty slice and the join is `""`. Pins the zero-window edge — a
/// regression that used `len - n` without saturating, or off-by-one'd
/// the slice bound, would panic or return a stray line here.
#[test]
fn tail_lines_zero_n_yields_empty_string() {
    assert_eq!(tail_lines("x\ny\nz", 0), "");
}

/// `tail_lines` does not preserve a trailing newline: `str::lines`
/// drops the final line terminator, so a `"a\nb\n"` input has two
/// lines (`a`, `b`) and `tail_lines(.., 2)` rejoins them as `"a\nb"`
/// (no trailing `\n`). Pins the `lines()`-then-`join("\n")` semantics
/// so the rendered tail never carries a dangling blank line.
#[test]
fn tail_lines_strips_trailing_newline() {
    assert_eq!(tail_lines("a\nb\n", 2), "a\nb");
    // A single trailing newline = one line, no empty tail line.
    assert_eq!(tail_lines("solo\n", 5), "solo");
}

// -- mkfs_package_hint coverage ------------------------------

/// `mkfs_package_hint` returns the concrete distro-package name for
/// [`Filesystem::Btrfs`] — `btrfs-progs`, the install target surfaced
/// in [`locate_host_binary`]'s "binary not found" diagnostic. Pins the
/// Btrfs arm of the exhaustive match so a regression that renames the
/// package hint (and breaks the operator's actionable install
/// guidance) surfaces here.
#[test]
fn mkfs_package_hint_btrfs_names_btrfs_progs() {
    assert_eq!(mkfs_package_hint(Filesystem::Btrfs), "btrfs-progs");
}

/// `mkfs_package_hint` returns the documented "<none — Raw needs no
/// formatter>" sentinel for [`Filesystem::Raw`]. The arm is
/// unreachable in production (callers gate on
/// `mkfs_binary_name().is_some()` first) but is retained so the match
/// stays exhaustive at the type level; pin the sentinel text so the
/// retained arm cannot silently drift.
#[test]
fn mkfs_package_hint_raw_returns_no_formatter_sentinel() {
    assert_eq!(
        mkfs_package_hint(Filesystem::Raw),
        "<none — Raw needs no formatter>",
    );
}

// -- ensure_template cache-hit fast path ---------------------

/// `ensure_template(Filesystem::Raw, ..)` returns a pre-seeded cache
/// entry WITHOUT booting a template VM. The `Raw` variant's
/// `mkfs_binary_name()` is `None`, so `locate_host_mkfs` short-circuits
/// to `Ok(None)` and the version fingerprint falls back to the
/// [`NOVERSION_FP`] sentinel — no `--version` fork+exec, no PATH walk.
/// The computed key matches [`template_cache_key`], so a same-key
/// entry pre-published via [`store_atomic`] satisfies the first
/// `lookup` and `ensure_template` returns at the cache-hit fast path
/// before any `verify_cache_dir_supports_reflink` / VM-boot work.
///
/// This drives the cache-hit return arm of `ensure_template` (the
/// `if let Some(hit) = lookup(...)` early return) entirely in-process:
/// the heavy build path past the lookup is not reached because the hit
/// short-circuits first. The empty PATH override guarantees a `Some`
/// fingerprint result would have to be a phantom — `Raw` must not
/// consult PATH at all.
#[test]
fn ensure_template_raw_returns_pre_seeded_cache_hit() {
    let _lock = crate::test_support::test_helpers::lock_env();
    let tmp = tempfile::tempdir().expect("create tempdir");
    let _cache_guard =
        crate::test_support::test_helpers::EnvVarGuard::set(crate::KTSTR_CACHE_DIR_ENV, tmp.path());
    // Empty PATH so any (incorrect) attempt to locate a Raw formatter
    // would fail loudly rather than silently locating a stray binary.
    let _path_guard = crate::test_support::test_helpers::EnvVarGuard::set("PATH", tmp.path());

    let capacity_bytes: u64 = 256 * 1024 * 1024;
    // The key ensure_template computes for Raw: noversion sentinel.
    let expected_key = template_cache_key(Filesystem::Raw, capacity_bytes, NOVERSION_FP);

    // Pre-publish a fake template under that exact key so the first
    // lookup inside ensure_template hits.
    let cache_root_path = cache_root().unwrap();
    std::fs::create_dir_all(&cache_root_path).unwrap();
    let staged = cache_root_path.join("staged.img");
    std::fs::write(&staged, b"PRE_SEEDED_RAW_TEMPLATE").unwrap();
    let installed =
        store_atomic(Filesystem::Raw, &expected_key, &staged).expect("pre-seed publish");

    let returned = ensure_template(Filesystem::Raw, capacity_bytes)
        .expect("ensure_template must hit the pre-seeded cache entry");
    assert_eq!(
        returned, installed,
        "ensure_template must return the pre-seeded entry path via the \
         cache-hit fast path, not rebuild",
    );
    // Content must be the pre-seeded body — a rebuild (which can't
    // happen here without a VM) would have replaced it.
    assert_eq!(
        std::fs::read(&returned).unwrap(),
        b"PRE_SEEDED_RAW_TEMPLATE",
    );
}

// -- clone_to_per_test coverage ------------------------------

/// `clone_to_per_test` FICLONE-clones the source into a fresh dest and
/// returns an open RW `File` whose contents equal the source. Requires
/// a reflink-capable cache filesystem; skipped (Ok-path asserted only)
/// when the tempdir is not btrfs/xfs — the FICLONE ioctl returns
/// `EOPNOTSUPP` there and the dedicated failure test
/// (`clone_to_per_test_ficlone_failure_cleans_up_dest`) covers that
/// branch instead.
///
/// On a reflink-capable host this drives the full happy path: open
/// src (read), `create_new` the dest (RW), the FICLONE ioctl
/// (`rc == 0`), and the returned `File`. The returned fd is opened
/// `O_RDWR` so the assertion can read the cloned bytes straight back.
#[test]
fn clone_to_per_test_reflinks_source_contents_on_capable_fs() {
    let tmp = tempfile::tempdir().expect("create tempdir");
    let src = tmp.path().join("template.img");
    let body = b"FICLONE_SOURCE_BODY_0123456789";
    std::fs::write(&src, body).unwrap();
    let dest = tmp.path().join("per-test.img");

    match clone_to_per_test(&src, &dest) {
        Ok(mut file) => {
            use std::io::{Read, Seek, SeekFrom};
            // The dest path now exists on disk.
            assert!(dest.exists(), "FICLONE dest must exist after clone");
            // The returned fd is RW and positioned at start; cloned
            // bytes must equal the source verbatim.
            file.seek(SeekFrom::Start(0)).expect("rewind cloned fd");
            let mut got = Vec::new();
            file.read_to_end(&mut got).expect("read cloned fd");
            assert_eq!(
                got, body,
                "FICLONE clone must reproduce the source bytes exactly",
            );
        }
        Err(e) => {
            // Non-reflink tempdir (tmpfs/ext4): the ioctl fails with
            // EOPNOTSUPP/EXDEV/EINVAL and the dest is cleaned up. The
            // failure-path test asserts the cleanup contract; here we
            // only confirm the error wording points the operator at a
            // reflink-capable cache dir.
            let msg = e.to_string();
            assert!(
                msg.contains("FICLONE") && msg.contains("KTSTR_CACHE_DIR"),
                "FICLONE failure on a non-reflink fs must name the ioctl \
                 and the cache-dir override: {msg}",
            );
            // Cleanup contract still holds on the failure path.
            assert!(
                !dest.exists(),
                "FICLONE failure must unlink the half-written dest",
            );
        }
    }
}

/// `clone_to_per_test` surfaces a pre-existing dest as a hard error
/// via the `O_CREAT | O_EXCL` (`create_new`) open — it must NOT
/// silently overwrite leftover per-test debris. Pins the EEXIST
/// surface documented at the fn's "Stale per-test debris" section:
/// the open of an already-present dest fails before the FICLONE ioctl
/// runs, and the pre-existing file's bytes are left untouched.
#[test]
fn clone_to_per_test_rejects_existing_dest() {
    let tmp = tempfile::tempdir().expect("create tempdir");
    let src = tmp.path().join("template.img");
    std::fs::write(&src, b"SRC").unwrap();
    // Pre-create the dest so create_new (O_EXCL) trips.
    let dest = tmp.path().join("per-test.img");
    std::fs::write(&dest, b"LEFTOVER_DEBRIS").unwrap();

    let err =
        clone_to_per_test(&src, &dest).expect_err("create_new must reject a pre-existing dest");
    let msg = err.to_string();
    assert!(
        msg.contains("dest path") && msg.contains("FICLONE"),
        "EEXIST error must name the dest-open-for-FICLONE context: {msg}",
    );
    // The leftover file must be untouched — create_new failing must
    // not overwrite or unlink the pre-existing debris.
    assert_eq!(
        std::fs::read(&dest).unwrap(),
        b"LEFTOVER_DEBRIS",
        "a rejected create_new must leave the pre-existing dest intact",
    );
}

/// `clone_to_per_test` bails with the source-open context when the
/// template source does not exist — the very first `OpenOptions::open`
/// fails with ENOENT before any dest is created. Pins the open-src
/// error arm: no dest file must be created when the source is missing.
#[test]
fn clone_to_per_test_missing_source_bails_without_creating_dest() {
    let tmp = tempfile::tempdir().expect("create tempdir");
    let missing_src = tmp.path().join("does-not-exist.img");
    let dest = tmp.path().join("per-test.img");

    let err = clone_to_per_test(&missing_src, &dest).expect_err("missing source must bail");
    let msg = err.to_string();
    assert!(
        msg.contains("open template source"),
        "missing-source error must name the source-open context: {msg}",
    );
    // The dest must never be created when the source open fails first.
    assert!(
        !dest.exists(),
        "a missing-source bail must not create the dest file",
    );
}

#[test]
fn build_template_via_vm_overflow_does_not_stage_image() {
    // A capacity whose MiB value exceeds u32::MAX must be rejected
    // BEFORE any staging image is created, so an oversized capacity
    // never leaks an orphan staging file into the cache root. The
    // overflow check now runs as the first statement of
    // build_template_via_vm (ahead of create_and_size_staging_image),
    // so this pins validation-before-consumption: a retry never trips
    // on a leaked staging file.
    let tmp = tempfile::tempdir().expect("create tempdir");
    let _guard =
        crate::test_support::test_helpers::EnvVarGuard::set(crate::KTSTR_CACHE_DIR_ENV, tmp.path());
    let cache_key = "btrfs-overflow";
    // (u32::MAX + 1) MiB expressed in bytes — the MiB value overflows u32.
    let capacity_bytes = ((u32::MAX as u64) + 1) * 1024 * 1024;
    let err = build_template_via_vm(Filesystem::Btrfs, capacity_bytes, tmp.path(), cache_key)
        .expect_err("oversized capacity must be rejected");
    let msg = err.to_string();
    assert!(
        msg.contains("capacity_mib overflow"),
        "error must name the overflow: {msg}",
    );
    // No staging image must exist at the deterministic staging path.
    let staging = staging_image_path(tmp.path(), cache_key, std::process::id());
    assert!(
        !staging.exists(),
        "oversized capacity must not stage (and then leak) an image: {staging:?}",
    );
}
