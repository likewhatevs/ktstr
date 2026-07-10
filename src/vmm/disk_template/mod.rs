//! Disk-template cache and per-test fan-out.
//!
//! This module ships the cache and clone primitives — the
//! `(Filesystem, capacity, mkfs version)` keyed lookup,
//! atomic-rename publish, per-key flock coordination, statfs-based
//! btrfs/xfs gate at the cache root, FICLONE per-test fan-out,
//! host mkfs locator (see [`Filesystem::mkfs_binary_name`] and
//! [`locate_host_mkfs`]), AND the host-side template-VM driver in
//! [`build_template_via_vm`] that boots a one-shot guest to run
//! the variant's `mkfs.<fstype>` against a sparse staging image
//! (`mkfs.btrfs /dev/vda` for `Filesystem::Btrfs`). The
//! guest-side dispatch lives in [`crate::vmm::rust_init`] and is
//! gated on `KTSTR_MODE=disk_template`.
//!
//! Design: the framework caches a guest-formatted backing image on
//! the host and per-test reflink-clones it via the `FICLONE` ioctl.
//! The host never execs `mkfs.btrfs` against a real backing file —
//! the kernel inside a one-off template VM is the on-disk-format
//! authority. Driving the actual formatting through a guest kernel
//! keeps the produced layout aligned with the kernel under test
//! (so a btrfs feature regression in the guest kernel surfaces as
//! a test failure, not as a host/guest mkfs disagreement).
//!
//! # Lifecycle
//!
//! 1. **Cache lookup.** [`ensure_template`] is called by
//!    `KtstrVm::init_virtio_blk` (or callers that
//!    pre-warm the cache). The lookup keys off
//!    `(Filesystem::cache_tag, capacity_mib, mkfs_fingerprint)`.
//!    Hit → return the template path. The mkfs fingerprint
//!    component (see [`mkfs_fingerprint`]) ensures an mkfs
//!    upgrade rotates the key and forces a fresh template build.
//! 2. **Lockfile.** Miss → acquire an exclusive flock under
//!    `<cache>/disk_templates/.locks/<key>.lock`. If a peer process is
//!    already populating the cache, this blocks until they finish (or
//!    the timeout fires). After acquire, re-check the cache for
//!    publish-while-waiting.
//! 3. **Template VM boot.** [`build_template_via_vm`] materialises
//!    a sparse `template.img.in-flight.<cache_key>.<pid>` of the
//!    requested capacity under the cache root (so `rename(2)` into
//!    place is same-filesystem; the `<cache_key>` qualifier
//!    disambiguates cross-key concurrent builds in the same pid —
//!    see [`staging_image_path`]), packs the host's mkfs binary
//!    (resolved via [`locate_host_mkfs`]) into the template-VM
//!    initramfs at `bin/<mkfs_name>`, and boots a one-shot guest
//!    with `KTSTR_MODE=disk_template` on the kernel cmdline. The
//!    disk attaches via
//!    [`crate::vmm::KtstrVmBuilder::template_staging_image`], which
//!    bypasses both the per-test `Raw` tempfile branch AND the
//!    `Btrfs` ensure_template branch in
//!    `KtstrVm::init_virtio_blk` — the template-build
//!    VM cannot recursively re-enter the cache it is itself
//!    populating. Guest dispatch
//!    (`crate::vmm::rust_init::run_disk_template_mode`) execs
//!    `/bin/<mkfs_binary_name>` against `/dev/vda` (currently
//!    `mkfs.btrfs` for `Filesystem::Btrfs` per
//!    [`Filesystem::mkfs_binary_name`]) and reboots cleanly; on
//!    non-zero exit / timeout the staging image is unlinked and
//!    the build bails with the trailing guest stderr.
//! 4. **Atomic install.** The formatted image is moved into
//!    `<cache>/disk_templates/<key>/template.img` via tempdir +
//!    `rename(2)` ([`store_atomic`]). Partial failures leave no
//!    entry behind.
//! 5. **Per-test fan-out.** [`clone_to_per_test`] FICLONE-clones the
//!    template into a tempfile on the same cache filesystem.
//!    `FICLONE` is O(metadata) — independent of capacity — and copy-
//!    on-write at the extent level so per-test writes do not touch
//!    the template.
//!
//! # Filesystem requirements
//!
//! `FICLONE` is implemented only on btrfs and xfs (kernel
//! `fs/remap_range.c:vfs_clone_file_range`; the VFS gates on the
//! `remap_file_range` superblock op which neither tmpfs nor ext4
//! provide). [`verify_cache_dir_supports_reflink`] checks the cache
//! filesystem's `statfs.f_type` and bails fast on non-supporting
//! filesystems with an actionable error.
//!
//! # Why not the `reflink` crate
//!
//! The `reflink` crate (v0.1.3) hardcodes
//! `IOCTL_FICLONE = 0x40049409` with a TODO questioning cross-arch
//! validity. The Linux generic ioctl encoding makes this number the
//! same on x86_64 and aarch64 (both use `<asm-generic/ioctl.h>`),
//! but `reflink::reflink` also opens the destination via
//! `OpenOptions::create_new`, which obscures the tempfile pattern
//! the cache fan-out wants (caller already controls dest creation
//! to apply mode bits and chown atomically). A direct `libc::ioctl`
//! call lets the cache module own dest semantics and produce
//! errno-precise diagnostics.

use std::fs::{File, OpenOptions};
use std::io;
use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{Context, Result, anyhow, bail};

use crate::flock::{FlockMode, acquire_flock_with_timeout, try_flock};
use crate::vmm::disk_config::Filesystem;

/// Cache subdirectory suffix passed to
/// [`crate::cache::resolve_cache_root_with_suffix`]. Distinct from
/// `"kernels"` (kernel image cache) and `"cast_analysis"` (cast-analysis
/// cache) so the three flavors share a parent root via `KTSTR_CACHE_DIR` /
/// `XDG_CACHE_HOME` without colliding on filesystem paths.
const CACHE_SUFFIX: &str = "disk_templates";

/// Filename used for the template image inside each cache entry.
const TEMPLATE_FILENAME: &str = "template.img";

/// Lockfile subdirectory name (per-key serialization).
const LOCK_DIR_NAME: &str = ".locks";

/// Maximum wall-clock duration to wait for a peer process holding
/// the cache lockfile while it builds the template.
///
/// # Budget breakdown
///
/// 600s = 10 minutes. The template build inside the lock holder
/// covers, in order:
///
/// - **Kernel boot** (~2-30s on a cold page cache, sub-second when
///   the kernel image is already mapped from a prior test).
///   First-run on a host without the kernel image cached can stall
///   on disk reads of the kernel + initramfs.
/// - **`mkfs.<fstype>` execution against `/dev/vda`** (~1-30s for a
///   256 MiB-1 GiB device on tmpfs/btrfs/xfs; 1-3 minutes on slow
///   spinning storage when the cache directory points at HDD-backed
///   storage). `mkfs.btrfs` does extent-tree initialisation plus
///   metadata block allocation — bound by storage IOPS, not CPU.
/// - **VM teardown** (sub-second).
///
/// The 10-minute ceiling absorbs the worst plausible host: a cold
/// HDD-backed `KTSTR_CACHE_DIR` running its first ever `mkfs.btrfs`
/// against a multi-GiB capacity. Below 10 minutes, a CI runner with
/// a cold cache and contentious IO would surface flaky-template
/// timeouts. Above 10 minutes, an interactive run against a
/// genuinely-stuck peer would hang the developer's terminal beyond
/// their patience threshold.
///
/// Operators who hit the timeout see a holder list parsed from
/// `/proc/locks` so they can kill a stuck peer (`kill <pid>`) or
/// wait by hand. The lockfile path is also surfaced so manual
/// cleanup is always available.
const TEMPLATE_LOCK_TIMEOUT: Duration = Duration::from_secs(600);

// Reject 32-bit targets at compile time. `statfs.f_type` is
// `__fsword_t` — `i64` on 64-bit Linux (LP64) and `i32` on 32-bit
// Linux. Bit 31 of `BTRFS_SUPER_MAGIC` (`0x9123_683e`) is set, so
// on 32-bit `__fsword_t` is a negative `i32` value. A subsequent
// `as u64` cast sign-extends the negative bit pattern into the high
// 32 bits (`0xFFFFFFFF_9123_683E`) and silently breaks the magic
// comparison — a btrfs cache directory would be rejected as
// "wrong filesystem". `XFS_SUPER_MAGIC` (`0x5846_5342`) has bit 31
// clear and would survive a 32-bit port, so the failure mode is
// asymmetric (btrfs always fails, xfs always passes). Reject the
// 32-bit build at compile time rather than ship a silently-wrong
// magic comparison.
#[cfg(not(target_pointer_width = "64"))]
compile_error!(
    "ktstr's disk-template f_type comparison requires a 64-bit \
     target. On 32-bit Linux `__fsword_t` is `i32`; sign-extension \
     of `BTRFS_SUPER_MAGIC` (bit 31 set) into u64 silently breaks \
     the magic comparison and rejects valid btrfs cache directories. \
     Porting to 32-bit requires casting through u32 to clear the \
     high bits before widening to u64."
);

/// btrfs `statfs.f_type` magic per `linux/magic.h`. `libc::BTRFS_SUPER_MAGIC`
/// covers GNU but is gated on Linux; pinning the constant defends
/// against a future libc minor release that drops/renames it.
///
/// Stored as `u64` so the comparison expression has matching unsigned
/// types. `statfs.f_type` is `__fsword_t` — `i64` on 64-bit Linux
/// (LP64), and ktstr only targets 64-bit Linux (`x86_64-unknown-linux-*`
/// and `aarch64-unknown-linux-*`); the `compile_error!` above rejects
/// 32-bit builds before they reach the cast. The call-site `as u64`
/// cast preserves the bit pattern of an `i64` source, so the
/// comparison against `0x9123_683e` matches the on-disk magic
/// correctly on every supported target.
const BTRFS_SUPER_MAGIC: u64 = 0x9123_683e;
/// xfs `statfs.f_type` magic per `linux/magic.h`. Same reasoning as
/// `BTRFS_SUPER_MAGIC`.
const XFS_SUPER_MAGIC: u64 = 0x5846_5342;

/// Run `statfs(2)` against an existing path and return the populated
/// `libc::statfs` buffer. Used by [`verify_cache_dir_supports_reflink`]
/// and [`store_atomic`] (the latter compares two `f_type`s and the
/// `f_fsid` pair to detect cross-filesystem renames before they fail
/// with a less-obvious `EXDEV`).
fn statfs_path(path: &Path) -> Result<libc::statfs> {
    let cstr = std::ffi::CString::new(path.as_os_str().as_encoded_bytes())
        .with_context(|| format!("path contains nul bytes: {path:?}"))?;
    // SAFETY: cstr is a NUL-terminated C string, statfs writes into
    // a stack-allocated zero-initialized buffer of the correct
    // layout. The kernel returns 0 on success and -1 with errno set
    // on failure.
    let mut buf: libc::statfs = unsafe { std::mem::zeroed() };
    let rc = unsafe { libc::statfs(cstr.as_ptr(), &mut buf) };
    if rc != 0 {
        let err = io::Error::last_os_error();
        return Err(anyhow!("statfs({path:?}) failed: {err}"));
    }
    Ok(buf)
}

/// Resolve the cache root directory for disk templates.
///
/// Reuses the global `KTSTR_CACHE_DIR` / `XDG_CACHE_HOME` / `$HOME`
/// cascade documented at
/// [`crate::cache::resolve_cache_root_with_suffix`]. Does not
/// create the directory; callers materialize on demand via
/// [`std::fs::create_dir_all`].
pub(crate) fn cache_root() -> Result<PathBuf> {
    crate::cache::resolve_cache_root_with_suffix(CACHE_SUFFIX)
}

/// Walk `dir`'s path tree upward and return the first ancestor (or
/// `dir` itself) that exists on disk, so its filesystem can be
/// `statfs`-probed. The cache root is created lazily and `statfs` on
/// a not-yet-created path returns `ENOENT`, so [`Path::exists`] is
/// the loop's termination gate; `Path::exists` follows symlinks, so a
/// dangling symlink probes as missing and the walk ascends to the
/// symlink's container rather than the (nonexistent) target's parent.
///
/// Returns the resolved ancestor on success, or an error when no
/// component of `dir` exists (the path has no existing ancestor at
/// all — only possible for a relative path with no current-directory
/// anchor). Split out of [`verify_cache_dir_supports_reflink`] so the
/// walk-up itself is unit-testable: a test can assert the resolved
/// ancestor equals the existing prefix it expects, rather than only
/// observing that the outer verify did not panic.
fn resolve_existing_ancestor(dir: &Path) -> Result<PathBuf> {
    let mut probe: PathBuf = dir.to_path_buf();
    loop {
        if probe.exists() {
            return Ok(probe);
        }
        match probe.parent() {
            Some(p) => probe = p.to_path_buf(),
            None => bail!(
                "no existing ancestor of {dir:?} found while probing \
                 cache filesystem; cannot verify FICLONE support",
            ),
        }
    }
}

/// Verify that `dir` lives on a filesystem that supports `FICLONE`.
///
/// Returns `Ok(())` for btrfs and xfs. Other filesystems (tmpfs,
/// ext4, fuse, …) bail with an actionable error naming the
/// filesystem magic and pointing the operator at
/// `KTSTR_CACHE_DIR` / `XDG_CACHE_HOME` for an override.
///
/// Walks up the path tree until a real component exists — the cache
/// root is created lazily, and `statfs` on a path that does not
/// exist yet returns `ENOENT`. Walking up reaches the parent
/// `XDG_CACHE_HOME` (or `$HOME/.cache`) and probes that filesystem
/// instead, which is the correct answer because filesystem boundaries
/// only show up at mount points and the cache root inherits its
/// parent's filesystem unless an operator mounted something custom
/// on top.
///
/// When the walk-up lands on an ancestor that is not `dir` itself —
/// because no leaf component of `dir` exists yet — the bail
/// diagnostic names both `dir` and the probed ancestor so the
/// operator can tell the f_type they see came from an ancestor, not
/// from `dir`. This matters when `dir` would, once created, mount on
/// a different filesystem than the ancestor (e.g. `KTSTR_CACHE_DIR`
/// points at a not-yet-mounted btrfs subvolume): the diagnostic does
/// not silently mislead about which filesystem was probed.
///
/// Symlink behaviour: `Path::exists` follows symlinks, so a
/// dangling symlink probes as missing and the walk-up moves to the
/// symlink's parent (the directory containing the symlink), not the
/// symlink target's parent. Operators who set `KTSTR_CACHE_DIR` to a
/// dangling symlink see the diagnostic name the symlink container's
/// filesystem rather than the (nonexistent) target's. Resolving the
/// symlink target before probing is intentionally NOT done — the
/// missing target is a configuration error, not a filesystem-type
/// question.
pub(crate) fn verify_cache_dir_supports_reflink(dir: &Path) -> Result<()> {
    let probe = resolve_existing_ancestor(dir)?;
    let buf = statfs_path(&probe).with_context(|| {
        format!(
            "cannot verify FICLONE support for cache directory {dir:?} \
             (probed ancestor {probe:?})"
        )
    })?;
    let fs_type = buf.f_type as u64;
    if fs_type == BTRFS_SUPER_MAGIC || fs_type == XFS_SUPER_MAGIC {
        return Ok(());
    }
    // Surface the probed ancestor in the diagnostic when it differs
    // from `dir`: the f_type we read came from `probe`, not from
    // `dir`, and an operator who reads only "dir lives on f_type X"
    // can be misled when X is the root filesystem's magic and the
    // intended cache mount simply does not exist yet.
    let probe_note = if probe == dir {
        String::new()
    } else {
        format!(
            " (no part of {dir:?} exists yet; the f_type was read from \
             ancestor {probe:?} — once {dir:?} is created on that same \
             filesystem the cache will inherit f_type=0x{fs_type:x}, \
             so create the intermediate mount first if you intended a \
             different filesystem)"
        )
    };
    bail!(
        "ktstr disk-template cache requires a btrfs or xfs filesystem \
         for FICLONE-based per-test fan-out; cache directory {dir:?} \
         lives on a filesystem whose statfs.f_type=0x{fs_type:x} (not \
         btrfs=0x{btrfs:x}, not xfs=0x{xfs:x}).{probe_note} Set \
         KTSTR_CACHE_DIR to a directory on a btrfs/xfs mount, or use \
         Filesystem::Raw which does not need a reflink-capable cache.",
        btrfs = BTRFS_SUPER_MAGIC,
        xfs = XFS_SUPER_MAGIC,
    );
}

/// Cache key for one template flavor (filesystem variant +
/// capacity + mkfs version fingerprint).
///
/// Renders as `"{tag}-{capacity_mib}m-{version_fp}"`, e.g.
/// `"btrfs-256m-a1b2c3d4e5f6a7b8"`. The components:
///
/// - `tag` is the [`Filesystem::cache_tag`] short identifier.
/// - `capacity_mib` forces the capacity into MiB (rather than raw
///   bytes) so every entry has the same magnitude regardless of
///   compiler-side rounding; the `m` suffix disambiguates from any
///   future GiB/sector-count keying.
/// - `version_fp` is a 16-hex-char SHA-256 prefix of the host
///   `mkfs.<fstype>` binary's contents (see [`mkfs_fingerprint`]).
///   It captures the on-disk format the host's mkfs binary produces;
///   an mkfs upgrade that changes the binary rotates the fingerprint
///   and forces a fresh template build. Without this component the
///   cache would silently reuse stale templates whose internal
///   format the new kernel may reject ([`clean_all`] is the
///   operator-driven escape hatch when the fingerprint somehow
///   misses a relevant change). Variants
///   whose [`Filesystem::mkfs_binary_name`] returns `None` (today
///   only `Raw`) pass `version_fp = "noversion"` because there is no
///   formatter to fingerprint.
///
/// The rendering is stable across rebuilds for a given
/// `(fs, capacity, mkfs version)` triple. New `Filesystem` variants
/// must pick a new `cache_tag` (see the `cache_tag` doc).
pub(crate) fn template_cache_key(fs: Filesystem, capacity_bytes: u64, version_fp: &str) -> String {
    let mib = capacity_bytes / (1024 * 1024);
    let tag = fs.cache_tag();
    format!("{tag}-{mib}m-{version_fp}")
}

/// Sentinel `version_fp` for filesystem variants that have no
/// userspace formatter ([`Filesystem::mkfs_binary_name`] returns
/// `None`). [`Filesystem::Raw`] is the only such variant today;
/// the production cache only ever sees this sentinel through unit
/// tests that call [`template_cache_key`] with `Filesystem::Raw`
/// (no real path computes a `Raw` template). Pinning the sentinel
/// as a named constant keeps the test fixture in lockstep with the
/// production fallback in [`ensure_template`].
const NOVERSION_FP: &str = "noversion";

/// Per-process cache for [`mkfs_fingerprint`] keyed by `mkfs_path`.
/// The fingerprint is invariant for a given binary file (the
/// production case for `mkfs.btrfs` / `mkfs.xfs`), so paying the
/// read+hash cost once per process is sufficient. Without this cache
/// every `ensure_template` call — i.e. every VM boot in the parallel
/// test run — re-reads and rehashes the same binary, adding a file
/// read on the hot path of test startup.
///
/// Keyed by [`PathBuf`] (not the resolved canonical path) because
/// the caller is [`locate_host_mkfs`], which already returns the
/// canonical path; storing the same canonicalized form here means a
/// repeat call with the same caller-side path hits without
/// recanonicalising.
///
/// `std::sync::Mutex` is sufficient — contention is bounded to
/// first-use per binary path (after which every subsequent call is
/// a `HashMap::get` under the lock), and the critical section never
/// runs the read+hash while holding the lock (see
/// [`mkfs_fingerprint`] for the read-then-insert shape).
fn mkfs_fingerprint_cache() -> &'static std::sync::Mutex<std::collections::HashMap<PathBuf, String>>
{
    static CACHE: std::sync::OnceLock<
        std::sync::Mutex<std::collections::HashMap<PathBuf, String>>,
    > = std::sync::OnceLock::new();
    CACHE.get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()))
}

/// Compute a 16-hex-char SHA-256 prefix of the `mkfs.<fstype>`
/// binary's contents, memoized per process by `mkfs_path`.
///
/// Used by [`template_cache_key`]: the fingerprint participates in
/// the cache key so an mkfs upgrade rotates the key and forces a
/// fresh template build. Without the fingerprint, an upgraded mkfs
/// (e.g. `btrfs-progs v6.5 → v6.10` introducing a new on-disk
/// feature flag default) would silently reuse the stale template
/// whose internal format the new kernel may reject.
///
/// The fingerprint is the SHA-256 hash of the binary file's bytes —
/// the same bytes the template-build VM packs into its initramfs and
/// execs against `/dev/vda` — truncated to the first 16 hex
/// characters. Hashing the binary itself rather than its `--version`
/// banner rotates the key on any change to the formatter, including a
/// downstream rebuild that alters on-disk behaviour without touching
/// the version string. 16 hex chars (~64 bits) is well below the
/// birthday-collision threshold for the dozens-to-hundreds of
/// binaries a single host will see across its lifetime.
///
/// The bytes are read directly from `mkfs_path` (the canonicalized
/// path returned by [`locate_host_mkfs`]). A read failure surfaces as
/// a bail message naming the binary path so an operator can rerun by
/// hand.
///
/// # Process-lifetime caching
///
/// Results are cached in a per-process map keyed by `mkfs_path`
/// (see [`mkfs_fingerprint_cache`]). The first call for a given path
/// reads the binary and hashes it; subsequent calls (in the same
/// process) return the cached string without touching the disk. This
/// matters because `ensure_template` runs on every VM boot — without
/// the cache, parallel-test runs re-read and rehash the same binary
/// for N tests against a file that hasn't changed across the run.
///
/// The cache is never invalidated. An mkfs upgrade between calls in
/// the same process would not be observed, but mkfs binaries do not
/// hot-swap during a test run — and even if one did, the prior
/// fingerprint still captures the binary that built any cached
/// template the run already produced, so reusing the prior key is
/// correct.
///
/// # Fingerprint stability
///
/// The hash covers the binary file's bytes, which are fixed for a
/// given install, so the fingerprint is stable across runs of the
/// same binary and rotates only when the file itself changes (a
/// reinstall or upgrade) — verified by the
/// `mkfs_fingerprint_is_deterministic` unit test.
fn mkfs_fingerprint(mkfs_path: &Path) -> Result<String> {
    // Hot path: cached. The lock is held only for the map lookup
    // and (on miss) for the insertion; the read+hash runs after the
    // first lookup so concurrent first-use against different paths
    // does not serialize.
    if let Some(cached) = mkfs_fingerprint_cache()
        .lock()
        .expect("mkfs_fingerprint cache mutex poisoned")
        .get(mkfs_path)
    {
        return Ok(cached.clone());
    }
    use sha2::Digest;
    let bytes = std::fs::read(mkfs_path)
        .with_context(|| format!("read {mkfs_path:?} for cache-key fingerprint"))?;
    if bytes.is_empty() {
        bail!(
            "{mkfs_path:?} is empty. Cannot fingerprint the binary for \
             the disk-template cache key — the binary may be a stub or \
             corrupted."
        );
    }
    let mut hasher = sha2::Sha256::new();
    hasher.update(&bytes);
    let digest = hasher.finalize();
    // 16 hex chars = 64 bits. Birthday collision around ~2^32
    // distinct binaries; vastly more than any host will ever see.
    let fp = hex::encode(&digest[..8]);
    // Memoize for the rest of this process. A concurrent first-use
    // against the same path would compute the fingerprint twice
    // (the lookup-then-insert is not atomic), but both reads hash the
    // same bytes and produce the same string, so the map's eventual
    // value is deterministic regardless of which insertion wins. The
    // redundant read is bounded by the number of concurrent
    // first-callers — a one-time cost paid before the cache is warm.
    mkfs_fingerprint_cache()
        .lock()
        .expect("mkfs_fingerprint cache mutex poisoned")
        .insert(mkfs_path.to_path_buf(), fp.clone());
    Ok(fp)
}

/// Path to the template image for the given key, relative to the
/// cache root. Does not check existence — use [`lookup`] for that.
pub(crate) fn template_path_for_key(key: &str) -> Result<PathBuf> {
    let root = cache_root()?;
    Ok(root.join(key).join(TEMPLATE_FILENAME))
}

/// Path to the per-key lockfile, relative to the cache root.
fn lock_path_for_key(key: &str) -> Result<PathBuf> {
    let root = cache_root()?;
    Ok(root.join(LOCK_DIR_NAME).join(format!("{key}.lock")))
}

/// Look up a cached template by key.
///
/// Returns `Some(path)` when a readable template image exists AND
/// carries the filesystem's on-disk superblock magic. Returns `None`
/// on a cache miss, a partial install, a removed-by-hand entry, OR a
/// content-invalid image (wrong/missing magic — a 0-byte or torn
/// template): a content-invalid hit is treated as a miss so
/// [`ensure_template`] rebuilds and self-heals the cache. Propagates
/// `Err` on an unexpected `stat(2)` failure (e.g. `EACCES`, `EIO`) — a
/// broken cache root surfaces as a hard error rather than a silent
/// miss. Callers materialize a miss via [`ensure_template`].
pub(crate) fn lookup(fs: Filesystem, key: &str) -> Result<Option<PathBuf>> {
    let path = template_path_for_key(key)?;
    match std::fs::metadata(&path) {
        Ok(meta) if meta.is_file() => {
            // Content-validate: confirm the cached image carries the
            // filesystem's on-disk superblock magic. A 0-byte / all-zero
            // template — an unformatted image a prior build published
            // (the publish gate in build_template_via_vm now blocks
            // that, but pre-fix caches and torn writes persist) — passes
            // is_file() yet fails to mount: the guest kernel's superblock
            // validator returns -EINVAL on the missing magic. Treat a
            // mismatch as a MISS so ensure_template rebuilds,
            // self-healing a stale cache without operator action.
            let Some((offset, magic)) = fs.superblock_magic() else {
                return Ok(Some(path));
            };
            match read_superblock_magic(&path, offset) {
                Ok(found) if found == magic => Ok(Some(path)),
                Ok(found) => {
                    tracing::warn!(
                        "cached disk template {} lacks the expected filesystem \
                         superblock magic at offset {:#x} (found {:#018x}, expected \
                         {:#018x}); treating as a cache miss and rebuilding",
                        path.display(),
                        offset,
                        found,
                        magic,
                    );
                    Ok(None)
                }
                Err(e) => {
                    tracing::warn!(
                        "could not read cached disk template {} superblock magic \
                         ({e:#}); treating as a cache miss and rebuilding",
                        path.display(),
                    );
                    Ok(None)
                }
            }
        }
        Ok(_) => Ok(None),
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(e).with_context(|| format!("stat cached template {path:?}")),
    }
}

/// Read the 8-byte little-endian superblock magic at `offset` from
/// `path`. Shared by [`lookup`] (cache content validation) and
/// [`build_template_via_vm`] (publish gate). A read that runs off the
/// end of a short image surfaces as an `Err` (`UnexpectedEof`), which
/// both callers treat as "not a valid template".
fn read_superblock_magic(path: &Path, offset: u64) -> Result<u64> {
    use std::io::{Read, Seek, SeekFrom};
    let mut f = std::fs::File::open(path)
        .with_context(|| format!("open template {path:?} for superblock magic check"))?;
    f.seek(SeekFrom::Start(offset))
        .with_context(|| format!("seek to superblock magic offset {offset:#x} in {path:?}"))?;
    let mut buf = [0u8; 8];
    f.read_exact(&mut buf)
        .with_context(|| format!("read superblock magic at {offset:#x} in {path:?}"))?;
    Ok(u64::from_le_bytes(buf))
}

/// Atomically install the file at `src_path` as the template for
/// `key`.
///
/// Stages under `<cache>/<key>.tmp.<pid>/template.img` and then
/// `rename(2)`'s the staging directory into place. Concurrent
/// installs serialize via the per-key lockfile (callers acquire
/// the lock before staging — see [`ensure_template`]); this
/// function trusts the caller already holds the lock.
///
/// The atomic-rename pattern matches [`crate::cache::CacheDir::store`]:
/// on partial failure the staging directory is removed by the
/// caller (best-effort), and the live cache always sees either no
/// entry or a complete entry — never a half-written one.
///
/// # Existing entry
///
/// When `<cache>/<key>` already exists the function re-validates the
/// installed image against [`Filesystem::superblock_magic`] before
/// publishing. A valid existing entry means a concurrent peer already
/// published (or `lookup` accepted it): `src_path` is discarded and the
/// existing path returned. An invalid existing entry — the stale
/// zeros/torn image `lookup` rejected as a miss, which prompted this
/// rebuild — is removed so the staging rename below installs the fresh
/// image. Variants with no on-disk magic ([`Filesystem::Raw`]) skip the
/// re-validation and always treat an existing entry as valid.
///
/// # Failure cleanup
///
/// Two failure points after the staging directory is created can
/// strand intermediate state:
///
/// - The first `fs::rename(src_path, &staging_image)` failure
///   leaves an empty staging directory (`src_path` is untouched —
///   `rename(2)` does not modify the source on failure). The
///   staging dir is removed best-effort before propagating.
/// - The second `fs::rename(&staging, &final_dir)` failure leaves
///   the populated staging dir on disk (the first rename moved
///   `src_path` into `staging_image` and is irreversible). The
///   staging dir AND its contained image are removed best-effort
///   before propagating; without this cleanup the staging tree
///   would accumulate across retries inside the cache root, where
///   neither the per-key flock nor a future `ensure_template` peer
///   would garbage-collect it.
///
/// Cleanup errors are best-effort because the original error is the
/// dominant signal; a `remove_dir_all` failure on top of an already-
/// failing publish adds no actionable diagnostic for the caller.
pub(crate) fn store_atomic(fs: Filesystem, key: &str, src_path: &Path) -> Result<PathBuf> {
    let root = cache_root()?;
    std::fs::create_dir_all(&root)
        .with_context(|| format!("create disk-template cache root {root:?}"))?;
    let final_dir = root.join(key);
    if final_dir.exists() {
        // A pre-existing cache dir is EITHER a concurrent peer's valid
        // publish (the race this originally guarded) OR a stale/invalid
        // entry that [`lookup`]'s read-side validation rejected and the
        // caller just rebuilt. Re-validate the existing template's
        // superblock magic before honoring the discard-ours early
        // return: if it is valid, the peer won and our rebuild is
        // redundant (discard ours); if it is stale, OUR freshly-built
        // image is the valid one and must replace the stale dir.
        // Without this check the discard-ours path keeps a zeros/torn
        // template alive forever, defeating lookup's self-heal.
        //
        // Race-safety: the caller holds the per-key flock
        // (`ensure_template`), so any concurrent peer's rename has
        // already completed before this read — `unwrap_or(false)` on a
        // read error therefore covers only non-peer torn images
        // (operator intervention, or an OS crash mid-rebuild by a prior
        // process), never a peer mid-publish.
        let existing = final_dir.join(TEMPLATE_FILENAME);
        let existing_valid = match fs.superblock_magic() {
            Some((offset, magic)) => read_superblock_magic(&existing, offset)
                .map(|found| found == magic)
                .unwrap_or(false),
            None => true,
        };
        if existing_valid {
            // Unlink our now-obsolete staging image before returning so
            // it does not leak in the cache root (the success path below
            // renames it into the staging dir; this early return skips
            // that, and no other code path GCs it at this name).
            let _ = std::fs::remove_file(src_path);
            return Ok(existing);
        }
        // Stale/invalid existing entry — remove it and fall through to
        // the rename below, which installs our valid rebuild. If the
        // subsequent publish rename fails, `final_dir` is already gone
        // and the next `ensure_template` observes the cache miss and
        // rebuilds — the removal self-heals rather than stranding a
        // half-state.
        std::fs::remove_dir_all(&final_dir)
            .with_context(|| format!("remove stale cache dir {final_dir:?} before replacing"))?;
    }
    // Pre-flight cross-filesystem check. `rename(2)` returns EXDEV
    // when src and dest live on different filesystems; the caller
    // should have staged src_path on the cache filesystem, but a
    // bug in caller logic (e.g. staging via `tempfile::tempfile()`
    // which honors `TMPDIR`) would surface as a less-obvious EXDEV
    // from `fs::rename` below. statfs both paths up front and bail
    // with a precise diagnostic naming the f_type magics. f_fsid is
    // also compared because two distinct btrfs subvolumes share an
    // f_type but differ in f_fsid, and rename(2) treats them as
    // different filesystems on most kernels.
    let src_buf = statfs_path(src_path)
        .with_context(|| format!("statfs source {src_path:?} for cross-fs check"))?;
    let dest_buf = statfs_path(&root)
        .with_context(|| format!("statfs cache root {root:?} for cross-fs check"))?;
    if src_buf.f_type != dest_buf.f_type || fsid_bytes(&src_buf) != fsid_bytes(&dest_buf) {
        bail!(
            "disk-template store_atomic: source {src_path:?} \
             (f_type=0x{src_type:x}) and cache root {root:?} \
             (f_type=0x{dest_type:x}) live on different filesystems. \
             rename(2) would return EXDEV. Stage the template image \
             on the cache filesystem before calling store_atomic.",
            src_type = src_buf.f_type as u64,
            dest_type = dest_buf.f_type as u64,
        );
    }
    let staging = root.join(format!("{key}.tmp.{pid}", pid = std::process::id()));
    if staging.exists() {
        std::fs::remove_dir_all(&staging)
            .with_context(|| format!("remove stale staging directory {staging:?}"))?;
    }
    std::fs::create_dir_all(&staging)
        .with_context(|| format!("create staging directory {staging:?}"))?;
    let staging_image = staging.join(TEMPLATE_FILENAME);
    // Move src_path into the staging dir. `fs::rename` is atomic on
    // the same filesystem; the cross-fs gate above guarantees that.
    // On failure src_path is unchanged (rename(2) is atomic), but
    // the empty staging directory is left behind — clean it up
    // before propagating so the cache root does not accumulate
    // empty `.tmp.<pid>` directories across retries.
    if let Err(e) = std::fs::rename(src_path, &staging_image) {
        let _ = std::fs::remove_dir_all(&staging);
        return Err(e).with_context(|| format!("rename {src_path:?} -> {staging_image:?}"));
    }
    // Durability: fsync the staged image data + the staging directory
    // before the publish rename so a host crash is less likely to leave
    // the cache pointing at a torn image. Best-effort — lookup re-checks
    // the 8-byte superblock magic on every read, so a zeroed/truncated
    // image re-detects as a cache miss and rebuilds. The residual it
    // does NOT catch is a crash leaving the magic bytes durable but
    // later blocks torn: that image passes the magic check and is
    // reflinked, but the corruption surfaces at guest mount (kernel
    // superblock/mount validation + fs metadata checksums) — a loud
    // test failure on a per-test scratch disk, never silent (the host
    // never parses the image content).
    if let Err(e) = crate::cache::fsync_staging_dir(&staging) {
        tracing::warn!(
            err = %e,
            staging = %staging.display(),
            "fsync of staged template before publish failed; entry will be \
             published without the durability barrier (validate-on-read \
             remains the correctness backstop)",
        );
    }
    // Final atomic publish. On failure the staging directory now
    // contains `staging_image` (the first rename moved src_path into
    // it and is not reversible). Without the cleanup arm below the
    // populated staging dir would persist across retries — the
    // per-key flock prevents a peer from racing on the same key,
    // but the in-flight staging tree is not garbage-collected by
    // any other code path.
    if let Err(e) = std::fs::rename(&staging, &final_dir) {
        let _ = std::fs::remove_dir_all(&staging);
        return Err(e).with_context(|| {
            format!("publish staging {staging:?} -> {final_dir:?} (cache key {key})",)
        });
    }
    // Persist the publish rename itself: fsync the cache root so the new
    // entry name survives a crash. Best-effort, same rationale.
    if let Err(e) = crate::cache::fsync_parent(&final_dir) {
        tracing::warn!(
            err = %e,
            final_dir = %final_dir.display(),
            "fsync of cache root after publish failed; the rename may not \
             survive a crash (validate-on-read backstop)",
        );
    }
    Ok(final_dir.join(TEMPLATE_FILENAME))
}

/// Extract `f_fsid` as a fixed-size byte tuple for equality
/// comparisons between two `statfs` results. `libc::fsid_t` is
/// `__val: [c_int; 2]` across glibc, musl, and uClibc, but `__val`
/// is a private field — direct field access does not compile. The
/// bytewise read via `ptr::copy_nonoverlapping` is layout-opaque
/// and does not depend on which libc backend the build links
/// against. `fsid_t` also does not implement `PartialEq`, so the
/// fixed-width byte read also serves as the equality primitive
/// [`store_atomic`]'s cross-fs gate uses.
fn fsid_bytes(buf: &libc::statfs) -> [u8; std::mem::size_of::<libc::fsid_t>()] {
    let mut out = [0u8; std::mem::size_of::<libc::fsid_t>()];
    // SAFETY: we read exactly size_of::<fsid_t>() bytes out of an
    // initialized statfs struct. Both source and destination cover
    // the same byte range, no aliasing, no out-of-bounds.
    unsafe {
        std::ptr::copy_nonoverlapping(
            &buf.f_fsid as *const libc::fsid_t as *const u8,
            out.as_mut_ptr(),
            std::mem::size_of::<libc::fsid_t>(),
        );
    }
    out
}

/// Acquire an exclusive flock on the per-key cache lockfile.
///
/// **Held ONLY around the mkfs+publish branch of [`ensure_template`].**
/// The pre-lock [`crate::cache::CacheDir::lookup`] at the top of `ensure_template`
/// runs WITHOUT a flock — manifest read is atomic on the read side
/// and the published template is read-only thereafter, so concurrent
/// readers (including the per-test fan-out path via
/// [`clone_to_per_test`]) coexist with each other and with an
/// in-flight builder via Unix open-file semantics (an EX-holder's
/// rename publishes a new inode; existing open fds keep the old
/// inode alive until closed).
///
/// Read-only callers MUST NOT call this function. The per-key flock
/// exists exclusively to serialize concurrent BUILDERS — two peers
/// both observing the same cache miss must not both run mkfs and
/// race their atomic renames. Calling from a read-only path would
/// reintroduce the wasted-wait pathology this design avoids.
///
/// Held for the timeout window [`TEMPLATE_LOCK_TIMEOUT`]; bails with
/// a holder list (PIDs, comms) on timeout so operators can triage a
/// stuck peer. Lockfile lives under the cache root's `.locks/`
/// subdirectory so the cache enumeration code skips it.
///
/// Future writes that mutate the published template inode in place
/// (e.g. `ftruncate`, `fallocate(PUNCH_HOLE)`) would invalidate the
/// open-fd safety property concurrent readers rely on — any such
/// path MUST acquire this lock and MUST be documented as the third
/// caller of this function. Today the only writers are atomic-rename
/// publishes (safe by inode-swap) and `clean_all`'s remove-tree
/// walk (which uses a separate non-blocking EX probe to skip live
/// peers — see [`clean_all`]).
pub(crate) fn acquire_template_lock(key: &str) -> Result<std::os::fd::OwnedFd> {
    let lock_path = lock_path_for_key(key)?;
    acquire_flock_with_timeout(
        &lock_path,
        FlockMode::Exclusive,
        TEMPLATE_LOCK_TIMEOUT,
        &format!("disk-template cache entry {key}"),
        Some(
            "A peer ktstr process is currently building this template. \
             Wait for it to finish, kill the peer with the listed PID, \
             or remove the lockfile if you are sure it is stale.",
        ),
    )
}

/// FICLONE-clone `src_path` into `dest_path`.
///
/// Both paths must reside on the same filesystem AND that filesystem
/// must implement `remap_file_range` (btrfs or xfs).
/// [`verify_cache_dir_supports_reflink`] gates on this for the cache
/// root; per-test fan-out callers must arrange for `dest_path` to
/// live under the cache root or another filesystem-validated path.
///
/// Returns the open `File` for `dest_path` ready for the device to
/// use. Caller is responsible for `unlink`-ing `dest_path` after
/// use. Failures with `EOPNOTSUPP` / `EXDEV` / `EINVAL` indicate a
/// reflink-incapable filesystem or cross-fs attempt and bail with a
/// hint at the operator's KTSTR_CACHE_DIR.
///
/// # Stale per-test debris and `EEXIST` diagnostics
///
/// `dest_path` is opened with `O_CREAT | O_EXCL` (via
/// [`OpenOptions::create_new`]), so the open returns `EEXIST` when
/// a regular file already sits at that path. Operators reading an
/// `EEXIST` here should NOT look at [`acquire_template_lock`] —
/// the per-key flock guards the cache *template* (read-only after
/// publish), not the per-test fan-out *dest*. The `EEXIST` surfaces
/// at the dest open, NOT at lock acquisition.
///
/// The realistic source of an `EEXIST` here is leftover staging
/// debris from a previous run that crashed before unlinking its
/// per-test fan-out file. The caller's dest name embeds a pid plus
/// a timestamp-and-random suffix (`.per-test-<pid>-<ns>-<rnd>.img`);
/// a prior ktstr peer that crashed mid-test (SIGKILL,
/// host reboot, OOM kill, panic before the per-test cleanup ran)
/// can leave its dest file in place. If the operating system later
/// reuses the same pid for a new ktstr process and that process
/// happens to generate a tempfile name colliding with the leaked
/// file's name, the `O_EXCL` open trips on the leftover. PID reuse
/// alone does not collide — the timestamp+random suffix disambiguates
/// most cases — but the check is `O_EXCL` precisely to surface the
/// rare collision as a hard error rather than a silent overwrite.
///
/// **Triage checklist for an `EEXIST`-shaped failure here**:
///
/// 1. List the cache directory for orphan per-test files matching
///    the dest tempfile pattern. They are unlinked by ktstr after
///    each test; survivors indicate a crashed predecessor.
/// 2. Verify no live ktstr peer holds the file open
///    (`fuser`/`lsof`-equivalent against the path); a live owner
///    means the collision is real and the tempfile generator is the
///    bug, not the leftover.
/// 3. If no live owner, remove the leftover by hand and retry. The
///    cache template (under [`acquire_template_lock`]) is unaffected
///    by per-test fan-out failures — only the per-test dest file
///    needs cleanup.
///
/// The flock itself is irrelevant to this failure mode: a stale
/// flock on the per-key lockfile would cause [`ensure_template`] to
/// time out at [`acquire_template_lock`] long before any per-test
/// fan-out runs, surfacing as a holder-list bail with the lockfile
/// path — a visibly different diagnostic than the `EEXIST` here.
///
/// # Distinct from `store_atomic`'s EEXIST surface
///
/// [`store_atomic`] also has a "destination already exists" surface
/// — its `final_dir.exists()` check on the published cache entry —
/// but that surface is **absorbed**, not propagated: when the
/// `<cache>/<key>/` directory already exists, `store_atomic`
/// returns the existing template path as `Ok(...)` (idempotent
/// no-op publish, because two concurrent peers building the same
/// `(fs, capacity, mkfs version)` key produce byte-identical
/// templates by construction). Operators do NOT see an `EEXIST`
/// error from `store_atomic` in the steady state.
///
/// The `EEXIST` surface in `clone_to_per_test` here is fundamentally
/// different: it is **propagated** as a hard error because two
/// per-test fan-out files at the same path are NOT byte-identical
/// (each test writes its own per-test mutations on top of the
/// reflink clone). Silently overwriting would lose the leftover
/// peer's data; absorbing as a no-op would hand the new test a
/// stale per-test image. Hard error is the only correct disposition.
///
/// In short: `store_atomic` EEXIST = "two peers raced and that's
/// fine, the template is the same"; `clone_to_per_test` EEXIST =
/// "leftover debris, investigate the predecessor". Never confuse
/// the two when triaging.
pub(crate) fn clone_to_per_test(src_path: &Path, dest_path: &Path) -> Result<File> {
    let src = OpenOptions::new()
        .read(true)
        .open(src_path)
        .with_context(|| format!("open template source {src_path:?}"))?;
    // O_RDWR (not O_WRONLY): the returned fd becomes virtio-blk's
    // backing store, which pread()s for guest READ requests — an
    // O_WRONLY fd returns EBADF on every read, so the guest sees the
    // disk as all-I/O-error and a btrfs mount fails EIO (silently
    // falling back to the initramfs tmpfs). `.read(true)` is
    // load-bearing; do not drop it. O_CREAT | O_EXCL (create_new)
    // surfaces stale leftover debris as EEXIST instead of silently
    // overwriting. See "Stale per-test debris and EEXIST diagnostics"
    // on this fn's doc comment.
    let dest = OpenOptions::new()
        .read(true)
        .write(true)
        .create_new(true)
        .open(dest_path)
        .with_context(|| format!("open dest path {dest_path:?} for FICLONE"))?;
    // Reflink the template into the per-test dest via the shared FICLONE
    // inner. Unlike reflink_or_copy, the per-test backing MUST be a
    // copy-on-write clone (a byte copy would defeat the fan-out and blow up
    // per-test disk usage), so ANY error hard-bails rather than falling back
    // — verify_cache_dir_supports_reflink already pre-checked the fs.
    if let Err(err) = crate::reflink::ficlone(&dest, &src) {
        // Best-effort cleanup of the half-written dest file.
        let _ = std::fs::remove_file(dest_path);
        return Err(anyhow!(
            "FICLONE {src_path:?} -> {dest_path:?} failed: {err}. \
             This usually means the destination filesystem does not \
             support reflinks (btrfs/xfs only) or the source and \
             destination live on different filesystems. Set \
             KTSTR_CACHE_DIR to a directory on a btrfs/xfs mount.",
        ));
    }
    Ok(dest)
}

/// Locate the host mkfs binary for `fs` so it can be packed into
/// the template-VM initramfs.
///
/// Resolves the userspace formatter name via
/// [`Filesystem::mkfs_binary_name`] and walks `PATH` (split on `:`)
/// for the first directory containing an executable of that name.
/// Returns `Ok(None)` when the variant requires no formatter
/// (`Filesystem::Raw`). Bails with an actionable error when a
/// formatter-requiring variant's binary is absent — the operator's
/// signal to install the corresponding distro package (e.g.
/// `btrfs-progs` for `Btrfs`) before using that filesystem.
///
/// The returned tuple carries BOTH the canonicalized binary path
/// AND the `mkfs.<fstype>` name. Callers that pack the binary into
/// the template-VM initramfs need both: the path to read the bytes
/// off disk, the name to compose the in-archive path
/// (`bin/<name>`). Returning both in a single call lets the caller
/// avoid a redundant [`Filesystem::mkfs_binary_name`] dispatch — a
/// caller that already has the path always has the matching name
/// without going back to the typed accessor.
///
/// The host binary is NOT exec'd at template-build time for
/// formatting — it is embedded into the template-VM initramfs and
/// exec'd by guest init inside the VM. The kernel inside the VM is
/// the on-disk-format authority; the host binary just provides the
/// `mkfs.<fstype>` userspace driver to drive the kernel into
/// formatting.
pub(crate) fn locate_host_mkfs(fs: Filesystem) -> Result<Option<(PathBuf, &'static str)>> {
    let Some(name) = fs.mkfs_binary_name() else {
        return Ok(None);
    };
    let path = locate_host_binary(name, mkfs_package_hint(fs))?;
    Ok(Some((path, name)))
}

/// Distro package hint for the formatter binary returned by
/// [`Filesystem::mkfs_binary_name`]. Surfaced in
/// [`locate_host_binary`]'s "binary not found" diagnostic so an
/// operator hitting the missing-formatter case sees a concrete
/// install target.
///
/// The match is exhaustive on `Filesystem` so a future variant
/// that ships a `mkfs_binary_name` Some(_) without picking a
/// package hint here surfaces as a non-exhaustive-match build
/// error. The `Raw` arm is unreachable in practice — callers gate
/// on `mkfs_binary_name().is_some()` first — but the arm is
/// retained so the match stays exhaustive at the type level.
fn mkfs_package_hint(fs: Filesystem) -> &'static str {
    match fs {
        Filesystem::Btrfs => "btrfs-progs",
        Filesystem::Raw => "<none — Raw needs no formatter>",
    }
}

/// Locate a binary by name on the host `PATH`. Used by
/// [`locate_host_mkfs`] today; future filesystem variants
/// ([`Filesystem`] extensions) reuse the same machinery via
/// [`Filesystem::mkfs_binary_name`] for their respective mkfs
/// binaries.
fn locate_host_binary(name: &str, package_hint: &str) -> Result<PathBuf> {
    let path_var = std::env::var_os("PATH")
        .ok_or_else(|| anyhow!("PATH environment variable is unset; cannot locate {name}"))?;
    for dir in std::env::split_paths(&path_var) {
        let candidate = dir.join(name);
        // `metadata` follows symlinks (`stat`, not `lstat`), so a
        // PATH entry like `/usr/sbin/mkfs.btrfs -> /usr/bin/btrfs`
        // resolves to the target's regular-file metadata. After
        // confirming we have a non-empty regular file, canonicalize
        // the candidate so the embedded copy in the template-VM
        // initramfs points at the real binary instead of a symlink
        // that the guest cannot follow once unpacked.
        let Ok(meta) = std::fs::metadata(&candidate) else {
            continue;
        };
        if !meta.is_file() {
            continue;
        }
        // Reject zero-byte stand-ins (a directory entry that exists
        // but contains no executable code). A bare `touch` or a
        // failed-install leftover lands here. Without this gate the
        // template-VM initramfs would pack a 0-byte binary that
        // exec(2) returns ENOEXEC on, with no clear hint at the
        // host-side root cause.
        if meta.len() == 0 {
            continue;
        }
        // Don't filter by mode bits — different distros have
        // different group/world execute settings on
        // /usr/sbin/mkfs.* binaries; the `exec` syscall checks
        // permissions correctly when the guest runs the binary, so
        // the host-side resolver only verifies "regular non-empty
        // file exists at this path."
        let canonical = std::fs::canonicalize(&candidate)
            .with_context(|| format!("canonicalize {candidate:?}"))?;
        return Ok(canonical);
    }
    bail!(
        "{name} not found on PATH. \
         Install the {package_hint} package (or your distro's \
         equivalent) so the disk-template VM can format the requested \
         filesystem. PATH={path:?}",
        path = path_var,
    )
}

/// Ensure a template exists for `(fs, capacity_bytes)` and return
/// the cached image path.
///
/// Cache hits return immediately (no lock acquisition, no boot).
/// Misses acquire the per-key flock, re-check, then build the
/// template via [`build_template_via_vm`] and atomically install it.
///
/// The cache key includes a fingerprint of the `mkfs.<fstype>`
/// binary's contents (see [`mkfs_fingerprint`]) so an mkfs upgrade
/// rotates the key and forces a fresh template build. The binary is
/// read once per [`ensure_template`] call; the cache lookup
/// short-circuits on hit before any further work.
///
/// # Tradeoff: hit path needs the formatter present (formatter-dependent variants only)
///
/// This tradeoff applies ONLY to filesystem variants that have a
/// userspace formatter — variants whose
/// [`Filesystem::mkfs_binary_name`] returns `Some(_)` (today
/// `Filesystem::Btrfs`). For those variants, the fingerprint is
/// required to construct the cache key, so every call to
/// `ensure_template` (cache hit or miss) must locate the host
/// formatter and query its version. If the formatter binary is
/// removed from PATH after the cache is populated,
/// `ensure_template` bails even on cache hits — the lookup cannot
/// run without a key, and the key cannot be built without the
/// fingerprint. The bail surfaces from
/// [`locate_host_mkfs`]'s "binary not found" diagnostic with the
/// distro-package install hint.
///
/// `Filesystem::Raw` is **exempt** from this tradeoff: its
/// [`Filesystem::mkfs_binary_name`] returns `None`,
/// [`locate_host_mkfs`] returns `None` without consulting PATH,
/// and the `version_fp` falls back to the [`NOVERSION_FP`]
/// sentinel. There is no PATH dependency at all for `Raw`. (In
/// practice the production path never reaches `ensure_template`
/// for `Raw` — the gate at
/// `KtstrVm::init_virtio_blk` short-circuits first —
/// but the fallback exists for defensive/test invocations.)
///
/// Operators hitting the formatter-removed bail on a
/// formatter-dependent variant must reinstall the formatter (e.g.
/// `apt install btrfs-progs` for `Filesystem::Btrfs`) OR run
/// [`clean_all`] and switch the test config to `Filesystem::Raw`,
/// which bypasses the template lifecycle entirely (no formatter
/// required, no FICLONE clone, fresh sparse tempfile per test).
/// The framework does NOT silently fall back to a stale-key
/// lookup when the formatter is missing — the cache key would be
/// ambiguous, so refusal is the correct disposition.
///
/// Callers (typically `KtstrVm::init_virtio_blk`)
/// then pass the returned path to [`clone_to_per_test`] for the
/// per-test reflink clone.
pub(crate) fn ensure_template(fs: Filesystem, capacity_bytes: u64) -> Result<PathBuf> {
    // Reclaim dead-pid staging / per-test debris from crashed prior runs,
    // once per process, BEFORE the cache-hit early-return below so a run
    // that only hits the template cache (but still creates per-test backing
    // files) also GCs. See sweep_cache_debris_once.
    sweep_cache_debris_once();
    // Resolve the host mkfs binary up front and query its version
    // fingerprint so the cache key reflects which mkfs would build
    // the template if we miss. The PATH lookup here is cheap (one
    // stat per PATH entry until found); the `--version` invocation
    // is one fork+exec per ensure_template call. Running it on
    // every call (including hits) is the price of a key that
    // self-invalidates on mkfs upgrade. Variants whose
    // [`Filesystem::mkfs_binary_name`] returns `None` (today only
    // [`Filesystem::Raw`]) skip the fingerprint and use the
    // `noversion` sentinel; the production path never builds a
    // template for `Raw` (the gate at
    // `KtstrVm::init_virtio_blk` short-circuits
    // first), so this branch is defensive.
    let version_fp = match locate_host_mkfs(fs)? {
        Some((mkfs_path, _name)) => mkfs_fingerprint(&mkfs_path)?,
        None => NOVERSION_FP.to_string(),
    };
    let key = template_cache_key(fs, capacity_bytes, &version_fp);
    if let Some(hit) = lookup(fs, &key)? {
        return Ok(hit);
    }
    let root = cache_root()?;
    // First-pass walk-up check: catches the common case (operator
    // pointed KTSTR_CACHE_DIR at a non-reflink fs) before we
    // create_dir_all on a doomed path.
    verify_cache_dir_supports_reflink(&root)?;
    std::fs::create_dir_all(&root)
        .with_context(|| format!("create disk-template cache root {root:?}"))?;
    // Re-verify against the now-existing cache root. Closes the
    // case where the walk-up landed on an ancestor that lives on a
    // different mount than the eventual cache directory (e.g. the
    // operator created a fresh sub-mount under HOME between probe
    // and now, or `~/.cache` is itself a separate mountpoint that
    // is not reflink-capable while `$HOME` is).
    verify_cache_dir_supports_reflink(&root)?;
    let _lock = acquire_template_lock(&key)?;
    // Re-check after acquire — a peer may have published while we
    // waited.
    if let Some(hit) = lookup(fs, &key)? {
        return Ok(hit);
    }
    let staged = build_template_via_vm(fs, capacity_bytes, &root, &key)
        .with_context(|| format!("build disk template for {key}"))?;
    // store_atomic either renames `staged` into the cache (a miss, or
    // replacing a stale/invalid existing entry) or discards it (a valid
    // peer already published — store_atomic unlinks `staged` itself on
    // that path). On failure (cross-fs detection, staging-dir creation,
    // the rename itself) `staged` is stranded: the per-key flock
    // prevents a peer from observing a partial cache entry, but the
    // in-flight file persists in the cache root until the next build.
    // Unlink before propagating so retries find a clean root.
    // Best-effort because the store_atomic error is the dominant
    // signal — a remove_file failure here adds no actionable
    // diagnostic.
    let final_path = match store_atomic(fs, &key, &staged) {
        Ok(p) => p,
        Err(e) => {
            let _ = std::fs::remove_file(&staged);
            return Err(e).with_context(|| format!("install disk template {key}"));
        }
    };
    Ok(final_path)
}

/// Reclaim disk-template cache debris orphaned by dead-pid prior runs
/// (staging images + per-test FICLONE backing files), ONCE per process.
/// Runs at the top of [`ensure_template`] — before its cache-hit
/// early-return — so even a run that only hits the template cache (but
/// creates per-test backing files) still GCs. Best-effort: a cache-root
/// resolution failure or sweep error is ignored (a later run reclaims the
/// debris, and `clean_orphaned_tmp_dirs` is dead-pid gated so it never
/// removes a live run's in-flight file).
fn sweep_cache_debris_once() {
    static SWEEP_ONCE: std::sync::Once = std::sync::Once::new();
    SWEEP_ONCE.call_once(|| {
        if let Ok(root) = cache_root() {
            let _ = cleanup::clean_orphaned_tmp_dirs(&root);
        }
    });
}

/// Compose the staging-image path for a `(cache_key, pid)` pair.
///
/// The filename includes BOTH the cache key and the pid because the
/// per-key flock only serialises peers within a single key — the
/// same process holds different per-key flocks concurrently across
/// distinct `(fs, capacity, mkfs version)` triples (cross-key
/// concurrency is permitted). Without the key in the filename, two
/// simultaneous in-flight builds for `btrfs-256m-<fp>` and
/// `btrfs-1024m-<fp>` from the same pid would collide on
/// `template.img.in-flight.<pid>` — the second open would truncate
/// the first's image while it boots, corrupting the template the
/// first build is formatting. Including the key makes the filename
/// unique per `(key, pid)`.
///
/// Pulled out as a free fn so the uniqueness invariant has a
/// dedicated test (`staging_image_path_is_unique_per_key_and_pid`).
fn staging_image_path(cache_root: &Path, cache_key: &str, pid: u32) -> PathBuf {
    cache_root.join(format!("template.img.in-flight.{cache_key}.{pid}"))
}

/// Materialise an empty sparse image at `staging_path` of exactly
/// `capacity_bytes`.
///
/// Removes any same-path leftover from a prior crashed run (the
/// per-key flock guarantees no live peer holds it; same-pid debris
/// is the only realistic source). On `set_len` failure (the
/// specific errno depends on the cache filesystem — common
/// examples include ENOSPC and EFBIG) the empty file is
/// unlinked best-effort before propagating; without that cleanup
/// a 0-byte staging image would accumulate in the cache root
/// across retries, mirroring the leak-cleanup behaviour at the
/// VM-boot/run failure sites farther down. The file descriptor is
/// dropped before the unlink as defense-in-depth: local
/// filesystems (btrfs/ext4/xfs) propagate truncate synchronously
/// but FUSE/NFS backings can delay until close.
///
/// Pulled out as a free fn so the cleanup arm has a dedicated
/// test (`create_and_size_staging_image_cleans_up_on_set_len_failure`)
/// that does not require booting a VM. Production callsites in
/// [`build_template_via_vm`] reach this helper via the standard
/// resource-bootstrap path.
fn create_and_size_staging_image(staging_path: &Path, capacity_bytes: u64) -> Result<()> {
    if staging_path.exists() {
        std::fs::remove_file(staging_path).with_context(|| {
            format!("remove leftover staging image {staging_path:?} before rebuild")
        })?;
    }
    let staging_file = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(true)
        .open(staging_path)
        .with_context(|| format!("create staging image {staging_path:?}"))?;
    if let Err(e) = staging_file.set_len(capacity_bytes) {
        drop(staging_file);
        let _ = std::fs::remove_file(staging_path);
        return Err(e).with_context(|| {
            format!(
                "set staging image length to {capacity_bytes} bytes \
                 ({staging_path:?})"
            )
        });
    }
    // Drop the host-side fd before returning; the VM opens its own
    // RW fd via `template_staging_image`, and host writes through a
    // stale fd would race the guest's mkfs.
    drop(staging_file);
    Ok(())
}

/// Build a fresh template image by booting a one-shot template VM.
///
/// Steps:
/// 1. Materialise a sparse `template.img.in-flight.<key>.<pid>`
///    of `capacity_bytes` under `cache_root` so the file shares
///    the cache filesystem ([`store_atomic`]'s rename requires
///    same-fs source/dest). The `<key>` qualifier disambiguates
///    cross-key concurrent builds in the same process; the per-key
///    flock already serialises within a single key.
/// 2. Locate `mkfs.<fstype>` on the host PATH and pack it into the
///    template-VM initramfs at `bin/mkfs.<fstype>`. The kernel
///    inside the VM is the on-disk-format authority — the host's
///    `mkfs` binary just provides the userspace driver that runs
///    against `/dev/vda` inside the guest.
/// 3. Boot a [`crate::vmm::KtstrVm`] with the sparse image attached
///    via [`crate::vmm::KtstrVmBuilder::template_staging_image`],
///    which short-circuits the per-test backing-file branches in
///    `KtstrVm::init_virtio_blk` so the template-build
///    VM cannot recursively re-enter [`ensure_template`] for its
///    own `(fs, capacity_bytes)` key. Cmdline carries
///    `KTSTR_MODE=disk_template`; the guest dispatch at
///    `crate::vmm::rust_init::run_disk_template_mode` execs the
///    embedded `bin/<mkfs_binary_name>` against `/dev/vda`
///    (currently `mkfs.btrfs` for `Filesystem::Btrfs` per
///    [`Filesystem::mkfs_binary_name`]) and reboots.
/// 4. After clean exit (`VmResult::success` and `exit_code == 0`),
///    return the staging path for [`store_atomic`] to rename into
///    the cache. Non-zero exit, timeout, or run failure unlinks
///    the staging file and bails.
///
/// Filesystem variants whose [`Filesystem::mkfs_binary_name`]
/// returns `None` (currently `Filesystem::Raw`) are unreachable on
/// this path: [`ensure_template`] only invokes this driver from the
/// gated formatting arm in `KtstrVm::init_virtio_blk`.
/// Such an argument means a caller bypassed that gate; bail with an
/// actionable error rather than build an unformatted template
/// (which would be a no-op).
fn build_template_via_vm(
    fs: Filesystem,
    capacity_bytes: u64,
    cache_root: &Path,
    cache_key: &str,
) -> Result<PathBuf> {
    // Validate the u32 `capacity_mib` BEFORE staging any file (or even
    // resolving mkfs/kernel): the overflow is a pure function of
    // `capacity_bytes` and needs no on-disk resource, so an oversized
    // capacity must reject without ever creating — and then leaking —
    // a staging image. `capacity_mib` is u32; an `as u32` cast would
    // silently truncate above u32::MAX MiB, so `try_from` surfaces the
    // overflow as an actionable error. Threaded into `build_template_vm`
    // so the disk config is built from the already-validated value.
    let capacity_mib = u32::try_from(capacity_bytes / (1024 * 1024)).with_context(|| {
        format!(
            "capacity_mib overflow: capacity_bytes={capacity_bytes} \
             yields {} MiB which exceeds u32::MAX. DiskConfig::capacity_mib \
             is u32; use a smaller capacity.",
            capacity_bytes / (1024 * 1024),
        )
    })?;

    // Resolve the mkfs binary for `fs` via the typed accessor so
    // the exhaustive match forces a future `Filesystem` variant to
    // declare its formatter at compile time. `locate_host_mkfs`
    // returns `Ok(None)` when the variant has no formatter
    // (currently only `Filesystem::Raw`); that case is unreachable
    // on this path because [`ensure_template`] gates on
    // [`Filesystem::mkfs_binary_name`] before calling here. A
    // `None` result means a caller bypassed the gate in
    // `KtstrVm::init_virtio_blk`; reject with an
    // actionable diagnostic.
    //
    // The returned tuple carries both the canonicalized path AND
    // the in-archive name — the consolidated return shape avoids
    // calling [`Filesystem::mkfs_binary_name`] twice (once via
    // `locate_host_mkfs`, once for the archive path). One typed
    // dispatch, one match arm.
    let (mkfs, mkfs_name) = locate_host_mkfs(fs)?.ok_or_else(|| {
        anyhow!(
            "build_template_via_vm called with Filesystem::{fs:?} — \
             this filesystem variant has no userspace formatter \
             (mkfs_binary_name() returned None) so there is no \
             template image to build. ensure_template should only \
             invoke this path for filesystem variants that require \
             pre-formatting; this call indicates a bypass of the gate \
             in init_virtio_blk."
        )
    })?;

    // Resolve a kernel image so the template-build VM can boot.
    // Reuses the same KTSTR_KERNEL / cache / sysroot cascade the
    // test framework uses, so an operator who set KTSTR_KERNEL for
    // tests gets the same kernel for the template build.
    let kernel = crate::find_kernel()
        .context("locate kernel image for template-build VM")?
        .ok_or_else(|| {
            anyhow!(
                "no kernel image found for template-build VM. {}",
                crate::KTSTR_KERNEL_HINT,
            )
        })?;

    // Stage the sparse image under the cache root so the eventual
    // rename(2) into place is on the same filesystem.
    std::fs::create_dir_all(cache_root)
        .with_context(|| format!("create cache root {cache_root:?} for staging image"))?;
    // Re-verify reflink support against the materialized cache root.
    // [`ensure_template`] performs this check too, but
    // `build_template_via_vm` is also reachable from direct callers
    // (tests, future operator-driven flows). Without this check, a
    // direct caller that staged a non-reflink-capable cache_root would
    // produce a template image whose subsequent
    // [`clone_to_per_test`] fan-out would fail at FICLONE time —
    // wasting the whole template-VM boot cost on a doomed run.
    // Re-verifying here closes the gate at the earliest point that
    // can detect the mismatch.
    verify_cache_dir_supports_reflink(cache_root)?;
    let staging_path = staging_image_path(cache_root, cache_key, std::process::id());
    create_and_size_staging_image(&staging_path, capacity_bytes)?;

    // Build the template VM. `build_template_vm` uses
    // `template_staging_image` (rather than a normal Btrfs disk) to
    // break the recursion that would otherwise occur: a Btrfs disk
    // inside a build-time VM would re-call ensure_template on the same
    // key whose flock the calling ensure_template already holds.
    //
    // `mkfs_name` came from the same [`locate_host_mkfs`] tuple
    // that produced the canonicalized binary path above; the
    // host-PATH lookup name and the in-archive path stay in
    // lockstep without a parallel match arm to drift.
    let vm = build_template_vm(
        fs,
        capacity_bytes,
        capacity_mib,
        kernel,
        &staging_path,
        mkfs,
        mkfs_name,
    )?;
    let result = vm.run().with_context(|| {
        format!("run template-build VM for {fs:?} capacity_bytes={capacity_bytes}")
    });
    let result = match result {
        Ok(r) => r,
        Err(e) => {
            // Best-effort cleanup of the staging image. The
            // template-build error itself is the dominant signal
            // and any remove_file error here is a tertiary
            // problem the caller cannot meaningfully act on.
            let _ = std::fs::remove_file(&staging_path);
            return Err(e);
        }
    };
    if result.timed_out || result.exit_code != 0 || !result.success {
        let _ = std::fs::remove_file(&staging_path);
        bail!(
            "template-build VM did not complete cleanly \
             (timed_out={}, exit_code={}, success={}). \
             Tail of guest output: {}",
            result.timed_out,
            result.exit_code,
            result.success,
            tail_lines(&result.output, 20),
        );
    }
    verify_template_superblock(fs, &staging_path, &result)?;
    Ok(staging_path)
}

/// Construct and boot the template-build VM that formats the staging
/// image, returning the booted [`crate::vmm::KtstrVm`] ready to run.
///
/// The disk attaches `staging_path` directly via
/// `template_staging_image`, which makes `init_virtio_blk` open it as
/// `Filesystem::Raw` (an unformatted device for the guest to format)
/// and bypasses the per-test and `ensure_template` branches. The host
/// `mkfs` binary rides through `include_files` packed at `bin/<name>`.
/// On a `.build()` failure (host-resource errors after the staging
/// image is already on disk) the staging file is unlinked best-effort
/// before propagating, mirroring the later `.run()` cleanup.
fn build_template_vm(
    fs: Filesystem,
    capacity_bytes: u64,
    capacity_mib: u32,
    kernel: PathBuf,
    staging_path: &Path,
    mkfs: PathBuf,
    mkfs_name: &'static str,
) -> Result<crate::vmm::KtstrVm> {
    let mkfs_archive_path = format!("bin/{mkfs_name}");
    // `capacity_mib` is validated in `build_template_via_vm` before any
    // staging image is created (see the overflow check there); it is
    // passed in already-bounded to u32.
    let disk = crate::vmm::disk_config::DiskConfig::default()
        .capacity_mib(capacity_mib)
        .filesystem(Filesystem::Raw);
    // VM-level timeout for the template build. 120s = 2 minutes,
    // chosen as the inner bound that lets the outer
    // [`TEMPLATE_LOCK_TIMEOUT`] (10 minutes) catch stuck peers
    // without firing on the legitimate worst-case build:
    //
    // - Kernel boot inside the VM: ~1-15s once the kernel image is
    //   already cached on the host (first-run cold-page-cache boot
    //   can stretch toward 30s on slow storage but is dominated by
    //   host-side disk reads, NOT this in-VM timeout).
    // - `mkfs.<fstype>` against `/dev/vda` inside the guest:
    //   ~1-60s for 256 MiB-2 GiB capacities on a backing image that
    //   itself lives on tmpfs/btrfs/xfs. The host backing-file IO
    //   cost (sparse-file zero-fill on first write) is included in
    //   this budget.
    // - VM shutdown: sub-second.
    //
    // 120s sits above the expected worst-case build cost
    // (kernel boot + mkfs + shutdown summed at the upper end of
    // the per-stage ranges above), which lets `mkfs` finish even
    // when KVM contention or a briefly-loaded host slows the
    // guest. If a build genuinely hangs (e.g. mkfs deadlocked,
    // kernel oops), the 120s VM timeout fires inside `vm.run()`,
    // the caller unlinks the staging image, and `ensure_template`
    // propagates the failure up — no peer holds the per-key flock
    // past this point.
    let busybox_bytes = crate::vmm::blobs::load_busybox_bytes()
        .context("load busybox blob for disk-template build VM")?;
    let build_result = crate::vmm::KtstrVm::builder()
        // Prebuilt distro kernels ship virtio as modules; embed the
        // ordered boot-module set from the cache entry (no-op for built
        // kernels). A distro kernel used for a disk-template build would
        // otherwise hang the same way a test-runner boot would. Computed
        // before `.kernel(kernel)` moves the path.
        .kernel_modules(crate::cache::boot_modules_for_image(&kernel))
        .kernel(kernel)
        // The build VM boots THIS binary as its guest /init: the ctor's
        // PID==1 branch dispatches to ktstr_guest_init, which sees
        // `KTSTR_MODE=disk_template` and runs mkfs against /dev/vda.
        // Without an init_binary the builder loads NO initramfs at all
        // (setup.rs gates both `rdinit=/init` and the initramfs blob on
        // `init_binary.is_some()`), so the guest never reaches userspace,
        // mkfs never runs, and the staging image stays zero — the root
        // cause of the empty-template failure this whole path guards.
        .init_binary(
            crate::resolve_current_exe()
                .context("resolve current exe as template-build VM /init")?,
        )
        .topology(crate::vmm::Topology::new(1, 1, 1, 1))
        // Defer + auto-size memory from the initramfs: with the test
        // binary as /init (~47MiB) the initramfs is ~88MiB, which a
        // fixed 256MiB VM cannot hold (the builder bails "insufficient
        // for initramfs"). memory_deferred_min sizes to the payload
        // (floor 256), matching the canonical test-VM path (runtime.rs).
        .memory_deferred_min(256)
        .timeout(std::time::Duration::from_secs(120))
        .cmdline("KTSTR_MODE=disk_template")
        .disk(disk)
        .template_staging_image(staging_path.to_path_buf())
        .include_files(vec![(mkfs_archive_path, mkfs)])
        .busybox(Some(busybox_bytes))
        .build();
    // .build() can fail for host-resource reasons (KVM ioctl
    // ENOMEM, sysfs unreadable, hugepage planning) AFTER the
    // staging image is already on disk. Without the cleanup arm
    // below, those failures leak the staging file across retries
    // — same pattern as the .run() error handler farther down,
    // but earlier in the lifecycle.
    let vm = match build_result {
        Ok(vm) => vm,
        Err(e) => {
            let _ = std::fs::remove_file(staging_path);
            return Err(e).with_context(|| {
                format!("build template-VM for {fs:?} capacity_bytes={capacity_bytes}")
            });
        }
    };
    Ok(vm)
}

/// Validate the freshly-built staging image carries the expected
/// on-disk superblock magic before it can be published to the cache.
///
/// A clean VM exit does NOT prove the in-guest mkfs wrote a valid
/// superblock — a silent mkfs failure, or a build path that never
/// reached mkfs, leaves the staging image unformatted. Caching an
/// unformatted image strands every future per-test clone with a
/// `-EINVAL` mount (the guest kernel's superblock validator rejects
/// the missing magic). Both the read-error and magic-mismatch arms
/// unlink the staging image best-effort before propagating, so a
/// bad build never reaches the cache and a retry finds a clean cache
/// root. Filesystem variants whose `superblock_magic()` returns
/// `None` skip the check (no magic to validate).
fn verify_template_superblock(
    fs: Filesystem,
    staging_path: &Path,
    result: &crate::vmm::VmResult,
) -> Result<()> {
    if let Some((offset, magic)) = fs.superblock_magic() {
        let found = match read_superblock_magic(staging_path, offset) {
            Ok(found) => found,
            Err(e) => {
                // A read error here (e.g. a short staging image from a
                // half-completed mkfs) would otherwise leak the staging
                // file across retries. Layer A treats a read error as a
                // cache miss, but Layer B must propagate the unusual
                // case — so remove the staging image before propagating
                // so a retry finds a clean cache root.
                let _ = std::fs::remove_file(staging_path);
                return Err(e).with_context(|| {
                    format!("read superblock magic from freshly-built template {staging_path:?}")
                });
            }
        };
        if found != magic {
            let _ = std::fs::remove_file(staging_path);
            bail!(
                "template-build VM exited cleanly but the staging image at \
                 {staging_path:?} lacks the {} superblock magic at offset {offset:#x} \
                 (found {found:#018x}, expected {magic:#018x}) — the in-guest mkfs \
                 reported success without writing a valid filesystem. Refusing to \
                 cache an unformatted template.\nTail of build-VM guest output:\n{}",
                fs.cache_tag(),
                tail_lines(&result.output, 30),
            );
        }
    }
    Ok(())
}

mod cleanup;
// Re-export the cleanup entry points so disk_template_tests.rs reaches them
// by their crate::vmm::disk_template path; the lib build's only non-test
// caller is the in-cleanup self-call, so the glob looks unused there.
#[allow(unused_imports)]
pub use cleanup::{clean_all, clean_orphaned_tmp_dirs};

/// Extract the last `n` lines of `text` for an error context.
/// Used by [`build_template_via_vm`] to surface the trailing guest
/// stderr — typically the `mkfs` failure message — without
/// dumping the whole transcript into the bail message.
fn tail_lines(text: &str, n: usize) -> String {
    let lines: Vec<&str> = text.lines().collect();
    let start = lines.len().saturating_sub(n);
    lines[start..].join("\n")
}

#[cfg(test)]
mod tests;
