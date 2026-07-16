//! Kernel-image resolution and KVM preflight: /dev/kvm accessibility
//! check, KernelUnavailable error, resolve_test_kernel discovery, and
//! the cache reader-lock acquisition. Split out of eval/mod.rs to keep
//! the module under the size ceiling.

use super::*;

/// Check that `/dev/kvm` is accessible for read+write.
///
/// Pre-flight check for VM-booting test runs: every ktstr test needs
/// a KVM fd, and failing fast here yields an actionable error
/// ("add your user to the kvm group") before the VM builder starts
/// allocating memory / fetching kernels.
///
/// Errno classification on open failure (two branches):
/// - Transient host pressure (`ENOMEM` / `EBUSY` / `EMFILE` / `ENFILE`
///   / `EAGAIN`, mirroring the `TRANSIENT_HOST_ERRNOS` set used by
///   [`crate::vmm::map_transient_to_contention`]): kernel memory
///   allocator under load, the kvm misc-device's per-CPU init
///   contended, the calling process exhausting its `RLIMIT_NOFILE`
///   (`EMFILE`), the system fd table full (`ENFILE`), or a kernel
///   subsystem signalling "try again" (`EAGAIN`). Routed through
///   [`crate::vmm::host_topology::ResourceContention`] so the
///   `#[ktstr_test]` macro routes the run to the retryable-contention
///   verdict (nextest re-runs it) instead of an unclassified fault. The
///   `EMFILE` / `ENFILE` arms specifically prevent fd-table pressure
///   on `/dev/kvm` open from surfacing as a hard error with a
///   misleading "kvm group" hint.
/// - Everything else (`EACCES` / `ENOENT` / `EINVAL` / etc.):
///   infrastructure misconfiguration or a real fault — the device is
///   missing, the user lacks permission, or the kernel returned an
///   unexpected errno. Surfaced as a hard error with the actionable
///   "kvm group" hint; contention-classifying these would hide a
///   misconfigured runner behind the retry budget.
pub(crate) fn ensure_kvm() -> Result<()> {
    match std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open("/dev/kvm")
    {
        Ok(_) => Ok(()),
        Err(e) => {
            let errno = e.raw_os_error();
            if matches!(
                errno,
                Some(libc::ENOMEM)
                    | Some(libc::EBUSY)
                    | Some(libc::EMFILE)
                    | Some(libc::ENFILE)
                    | Some(libc::EAGAIN)
            ) {
                let snapshot = vmm::host_resource_snapshot();
                let errno_label = match errno {
                    Some(libc::ENOMEM) => "ENOMEM",
                    Some(libc::EBUSY) => "EBUSY",
                    Some(libc::EMFILE) => "EMFILE",
                    Some(libc::ENFILE) => "ENFILE",
                    Some(libc::EAGAIN) => "EAGAIN",
                    _ => unreachable!(),
                };
                Err(anyhow::Error::new(
                    crate::vmm::host_topology::ResourceContention {
                        reason: format!(
                            "/dev/kvm open: transient host errno {errno_label}: \
                             host resources: {snapshot}\n  \
                             hint: KVM device open failed with a host-resource \
                             errno; another peer may be holding the budget. \
                             nextest retries the cell; if the pressure has \
                             cleared, the retry passes.",
                        ),
                    },
                ))
            } else {
                Err(anyhow::Error::new(e).context(
                    "/dev/kvm not accessible — KVM is required for ktstr_test. \
                     Check that KVM is enabled and your user is in the kvm group.",
                ))
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Scheduler resolution
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Kernel resolution
// ---------------------------------------------------------------------------

/// Marker error for "the test harness can't find a kernel image to
/// boot the VM against". Wraps the actionable diagnostic that
/// [`resolve_test_kernel`] emits when neither
/// `KTSTR_TEST_KERNEL` nor any standard cache / sysroot location
/// produced a bootable image.
///
/// Distinct from a generic `anyhow::bail!` so the
/// `#[ktstr_test]` macro's wrapper can downcast and emit a SKIP
/// banner instead of panicking — the canonical "running under
/// `cargo nextest run` instead of `cargo ktstr test`" symptom.
/// Routes through [`crate::test_support::is_kernel_unavailable`]
/// for the macro's predicate; downcast directly when adding new
/// SKIP arms.
#[derive(Debug)]
pub struct KernelUnavailable {
    pub diagnostic: String,
}

impl std::fmt::Display for KernelUnavailable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.diagnostic)
    }
}

impl std::error::Error for KernelUnavailable {}

/// Find a kernel image for running tests.
///
/// Checks `KTSTR_TEST_KERNEL` env var first (direct image path),
/// then delegates to [`crate::find_kernel()`] for cache and
/// filesystem discovery. Returns a typed [`KernelUnavailable`] on
/// failure so the `#[ktstr_test]` macro wrapper can map it onto a
/// clean SKIP banner — generic `anyhow` errors propagate to the
/// panic arm and surface as confusing test failures when the
/// binary runs outside `cargo ktstr test`.
pub fn resolve_test_kernel() -> Result<PathBuf> {
    // Check environment variable first. A set-but-missing
    // `KTSTR_TEST_KERNEL` is an OPERATOR mistake (they pointed at
    // a path that doesn't exist), not a "harness not configured"
    // situation — surface it as a regular anyhow error so the
    // panic arm catches it. Skipping on a typo would silently mask
    // the bad path.
    if let Ok(path) = std::env::var(crate::KTSTR_TEST_KERNEL_ENV) {
        let p = PathBuf::from(&path);
        anyhow::ensure!(p.exists(), "KTSTR_TEST_KERNEL not found: {path}");
        return Ok(p);
    }

    // Standard locations.
    if let Some(p) = crate::find_kernel()? {
        return Ok(p);
    }

    let image_name = if cfg!(target_arch = "aarch64") {
        "Image"
    } else {
        "bzImage"
    };
    Err(anyhow::Error::new(KernelUnavailable {
        diagnostic: format!(
            "no kernel found — the test harness was likely invoked \
             outside `cargo ktstr test` (which builds and injects a \
             kernel automatically).\n  \
             hint: run `cargo ktstr test --kernel <path-or-version>` \
             to drive this test, or set KTSTR_TEST_KERNEL=/path/to/{image_name} \
             to point at a pre-built bootable image directly.\n  \
             hint: {kernel_hint}",
            kernel_hint = crate::KTSTR_KERNEL_HINT,
        ),
    }))
}

/// Detection seam for the [`crate::flock`] helper's timeout-bail
/// message shape.
///
/// Returns `true` iff `rendered` contains BOTH `"timed out after"` and
/// `"flock LOCK_"`. The two substrings together are the helper's
/// internal contract for a flock-acquisition timeout — see
/// `flock/acquire.rs`'s bail format
/// `"flock {LOCK_EX|LOCK_SH} on {context} timed out after ..."`.
///
/// Pinned via the unit test
/// `flock_timeout_substring_classification_pins_seam` so a
/// rewording of the bail message that drops either substring is
/// caught at test time before
/// [`acquire_test_kernel_lock_if_cached`] starts misclassifying
/// timeouts as plain anyhow errors.
pub(crate) fn is_flock_timeout_message(rendered: &str) -> bool {
    rendered.contains("timed out after") && rendered.contains("flock LOCK_")
}

/// If `kernel_path` resolves to an image inside a cache entry, hold a
/// `LOCK_SH` on that entry's coordination lockfile for the duration of
/// the returned guard. Prevents a concurrent
/// `cargo ktstr kernel build` from swapping the entry's directory
/// (see [`crate::cache::CacheDir::store`]) under the VM while the test
/// reads from it.
///
/// Returns `Ok(None)` when `kernel_path` is not shaped like a cache
/// entry — explicit `KTSTR_TEST_KERNEL=/path/to/bzImage`,
/// `/lib/modules/.../vmlinuz`, `/boot/vmlinuz-*`, or any path whose
/// two-level parent does not match the resolved cache root. Such
/// paths do not need coordination because the build pipeline never
/// touches them.
///
/// Detection: the image is expected at `{root}/{key}/{image_name}`.
/// Walk `kernel_path` up by two components (image_name, key) to
/// produce a candidate root and canonicalize both sides before
/// comparing — symlinks, redundant `./` segments, and `..` traversals
/// must all reduce to the same inode path or the entry is treated as
/// non-cache.
pub(crate) fn acquire_test_kernel_lock_if_cached(
    kernel_path: &Path,
) -> Result<Option<crate::cache::SharedLockGuard>> {
    // Peel the image filename. Fail → not a cache entry.
    let Some(entry_dir) = kernel_path.parent() else {
        return Ok(None);
    };
    // Peel the entry directory name (this is the candidate cache
    // key). Fail → not a cache entry.
    let Some(key_os) = entry_dir.file_name() else {
        return Ok(None);
    };
    let Some(cache_key) = key_os.to_str() else {
        return Ok(None);
    };
    // The directory above the entry is the candidate cache root.
    let Some(candidate_root) = entry_dir.parent() else {
        return Ok(None);
    };

    // Canonicalize both the candidate root and the resolved cache
    // root so symlinks / `.` / `..` reduce to the same inode path
    // before comparing. A non-cache path (e.g. /lib/modules/...)
    // simply canonicalizes to itself and will not match.
    let candidate_root_canon = match candidate_root.canonicalize() {
        Ok(p) => p,
        Err(_) => return Ok(None),
    };
    let resolved_root = match crate::cache::CacheDir::default_root() {
        Ok(p) => p,
        // Cache root unresolvable (no HOME and no XDG_CACHE_HOME, or
        // non-UTF-8 KTSTR_CACHE_DIR): no cache exists, so `kernel_path`
        // cannot be an entry. (A cache root that resolves but is absent
        // on disk is handled by the canonicalize() arm below.)
        Err(_) => return Ok(None),
    };
    let resolved_root_canon = match resolved_root.canonicalize() {
        Ok(p) => p,
        // Cache root resolves but does not exist on disk yet (fresh
        // developer checkout). `kernel_path` is not inside a cache
        // entry, so no lock needed.
        Err(_) => return Ok(None),
    };

    if candidate_root_canon != resolved_root_canon {
        return Ok(None);
    }

    // The path is shaped as a cache entry under the resolved root.
    // Acquire the reader lock. The flock helper polls on
    // `EAGAIN`/`EWOULDBLOCK` until either the lock is granted or its
    // wall-clock timeout elapses. A timeout means a peer (concurrent
    // `cargo ktstr kernel build` or another reader-blocking writer)
    // is holding the lock — that is host-resource contention, not a
    // kernel fault, so route it through
    // [`crate::vmm::host_topology::ResourceContention`] so the
    // `#[ktstr_test]` macro fires the retryable-contention verdict
    // (nextest re-runs after the holder finishes). Non-timeout failures
    // (parent-directory creation failure, an unexpected `try_flock`
    // errno other than `EAGAIN`/`EWOULDBLOCK`) propagate as hard
    // errors — they indicate filesystem corruption or a programming
    // fault a contention verdict would obscure.
    //
    // Detection seam: the flock helper's bail format starts with
    // `flock LOCK_SH on` (or `LOCK_EX`) and contains `timed out
    // after`. Both substrings are pinned by the helper's internal
    // contract and embedded in the rendered message together; the
    // message also contains the lockfile path and the holder PID
    // list parsed from `/proc/locks`, which we forward verbatim into
    // the `ResourceContention` reason so the operator sees the
    // identical triage information either way.
    let cache = crate::cache::CacheDir::with_root(resolved_root_canon);
    match cache.acquire_shared_lock(cache_key) {
        Ok(guard) => Ok(Some(guard)),
        Err(e) => {
            let rendered = format!("{e:#}");
            if is_flock_timeout_message(&rendered) {
                let snapshot = crate::vmm::host_resource_snapshot();
                Err(anyhow::Error::new(
                    crate::vmm::host_topology::ResourceContention {
                        reason: format!(
                            "test kernel cache lock: {rendered}. host resources: \
                             {snapshot}\n  \
                             hint: a concurrent `cargo ktstr kernel build` or \
                             another lockholder is preventing the test VM from \
                             reading the cached kernel image. nextest retries \
                             the cell after the holder finishes. If the failure \
                             persists, wait for the holder PIDs above to \
                             finish, or kill them, then retry.",
                        ),
                    },
                ))
            } else {
                Err(e)
            }
        }
    }
}
