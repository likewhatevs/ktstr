//! Compute the minimum guest memory required to boot, extract the
//! initramfs, and run the post-boot test workload.
//!
//! Used by the deferred-memory path in [`KtstrVm`](super::KtstrVm) to
//! size guest memory from observed initramfs sizes instead of a static
//! caller estimate.

use anyhow::{Context, Result};
use std::path::Path;

/// The fraction of `totalram_pages` the guest rootfs tmpfs is sized for
/// during initramfs extraction.
///
/// `shmem_default_max_blocks` (`mm/shmem.c`) returns `totalram_pages() /
/// 2`, so the rootfs tmpfs admits at most 50% of RAM by default. The
/// `initramfs_options=size=90%` cmdline token (emitted unconditionally
/// when `init_binary.is_some()`) raises that to 90% — but only on
/// kernels carrying mainline commit 278033a225e1 ("fs: Add
/// 'initramfs_options'"), first tagged v6.18-rc1 and backported to the
/// stable series (see [`Self::for_kernel_version`] for the per-series
/// floors). On kernels without it the token is silently ignored and the
/// tmpfs stays at the 50% default.
///
/// The asymmetry is one-directional: sizing RAM for 50% when the kernel
/// honors 90% wastes a little RAM (the tmpfs is bigger than we sized
/// for — safe); sizing RAM for 90% when the kernel only gives 50% means
/// extraction overruns the tmpfs and the guest panics. So [`Self::Half`]
/// is the conservative default for ANY uncertainty, and
/// [`Self::NinetyPercent`] is selected only when the kernel is
/// POSITIVELY known to honor the token (mainline >= 6.18, or a stable
/// series at or above its backport floor — see
/// [`Self::for_kernel_version`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TmpfsFraction {
    /// 50% of `totalram_pages` — the `shmem_default_max_blocks` default
    /// and the universally-safe floor.
    Half,
    /// 90% of `totalram_pages` — the ceiling `initramfs_options=size=90%`
    /// raises the tmpfs to, on honoring kernels (mainline 6.18+ or a
    /// backported stable series — see [`Self::for_kernel_version`]).
    /// Reclaims RAM proportional to the uncompressed payload — roughly a
    /// third of the boot budget at the instrumented-payload shape (the
    /// `ninety_percent_fraction_sizes_less_ram_than_half` test measures
    /// the 3030 -> 1947 MiB drop).
    NinetyPercent,
}

impl TmpfsFraction {
    /// Numerator/denominator of the fraction, as a fixed-point pair the
    /// budget formula multiplies through with integer saturating
    /// arithmetic (no floats — the floor must round the same on every
    /// host). `Half` -> 1/2, `NinetyPercent` -> 9/10.
    fn ratio(self) -> (u64, u64) {
        match self {
            TmpfsFraction::Half => (1, 2),
            TmpfsFraction::NinetyPercent => (9, 10),
        }
    }

    /// Select the tmpfs fraction for a guest kernel whose
    /// `(major, minor, patch)` version is `version`.
    ///
    /// [`Self::NinetyPercent`] iff the kernel is positively known to
    /// honor `initramfs_options=size=90%` (upstream commit 278033a225e1):
    /// mainline `(major, minor) >= (6, 18)` (first tagged v6.18-rc1,
    /// regardless of patch), or a stable series at or above the patch
    /// level where the backport first shipped.
    ///
    /// The per-series backport floors, verified via `git tag --contains`
    /// of each series' backport commit against the linux-stable trees,
    /// are 5.4.301, 5.10.246, 5.15.195, 6.1.157, 6.6.113, 6.12.54, and
    /// 6.17.4.
    ///
    /// `None` (no version established), a series absent from the table
    /// (EOL / no backport, e.g. 6.7-6.11 or 6.13-6.16 — 6.12 IS in the
    /// table), or a version below its series floor yields the
    /// conservative [`Self::Half`]. The table is a
    /// verified snapshot: a series that gains a backport later still
    /// falls to `Half` until added here — safe (no reclaim), never a
    /// panic.
    pub(crate) fn for_kernel_version(version: Option<(u16, u16, u16)>) -> Self {
        let Some((major, minor, patch)) = version else {
            return TmpfsFraction::Half;
        };
        // Mainline 6.18+ honors the token regardless of patch level.
        if (major, minor) >= (6, 18) {
            return TmpfsFraction::NinetyPercent;
        }
        // Per-stable-series patch floor where the initramfs_options
        // backport first shipped. A series not listed (EOL / no
        // backport) falls through to the safe Half.
        let floor = match (major, minor) {
            (5, 4) => 301,
            (5, 10) => 246,
            (5, 15) => 195,
            (6, 1) => 157,
            (6, 6) => 113,
            (6, 12) => 54,
            (6, 17) => 4,
            _ => return TmpfsFraction::Half,
        };
        if patch >= floor {
            TmpfsFraction::NinetyPercent
        } else {
            TmpfsFraction::Half
        }
    }
}

/// Parameters for computing minimum guest memory.
pub(crate) struct MemoryBudget {
    /// Uncompressed initramfs size (base + suffix cpio) in bytes.
    pub uncompressed_initramfs_bytes: u64,
    /// LZ4-compressed initrd size in bytes. The compressed initrd
    /// is memblock-reserved in guest physical memory from load until
    /// free_initrd_mem() releases it after extraction.
    pub compressed_initrd_bytes: u64,
    /// Kernel `init_size` from bzImage setup_header (offset 0x260).
    /// The kernel's declared contiguous memory requirement during
    /// boot decompression. Includes compressed payload, decompressed
    /// kernel, and decompression workspace. Overestimates resident
    /// kernel (init sections and workspace are freed post-boot),
    /// absorbing percpu and misc boot allocations.
    pub kernel_init_size: u64,
    /// `true` when the `/init` payload (the guest's PID 1) is built
    /// with `-C instrument-coverage`. An instrumented `/init` holds
    /// the LLVM profile-counter sections resident AND, at flush time,
    /// `__llvm_profile_write_buffer` serializes them into a heap
    /// buffer (`crate::test_support::try_flush_profraw`) — neither of
    /// which a non-instrumented payload pays for. When set,
    /// [`initramfs_min_memory_mib`] adds
    /// `instrumented_reserve_bytes` to the workload term so the
    /// instrumented `/init` does not OOM during boot.
    pub init_coverage_instrumented: bool,
    /// Extra resident bytes to reserve when
    /// `init_coverage_instrumented` is set: the summed sizes of the
    /// payload's `__llvm_prf_cnts` + `__llvm_prf_data` sections (the
    /// live coverage-counter and profile-metadata arrays the
    /// instrumented binary keeps resident). `0` when the payload is not
    /// instrumented or the sections are absent.
    ///
    /// This is a STEADY-STATE floor, not the flush-time peak: at flush
    /// `__llvm_profile_write_buffer` allocates a
    /// `__llvm_profile_get_size_for_buffer()` heap buffer (cnts + data +
    /// `__llvm_prf_names`) that briefly coexists with the resident
    /// sections, so the true peak is ~2x(cnts+data)+names. `WORKLOAD_MIB`
    /// slack absorbs the second copy at current binary sizes; a much
    /// larger instrumented `/init` may want a peak-aware (2x) reserve.
    pub instrumented_reserve_bytes: u64,
    /// Fraction of `totalram_pages` the guest rootfs tmpfs is sized for
    /// during initramfs extraction; see [`TmpfsFraction`]. A larger
    /// fraction sizes LESS total RAM for the same payload. Selected via
    /// [`TmpfsFraction::for_kernel_version`].
    pub tmpfs_fraction: TmpfsFraction,
}

/// Read the kernel's declared memory footprint from the image file.
///
/// x86_64 bzImage: reads `init_size` from setup_header at file offset
/// 0x260 (setup_header starts at 0x1F1, `init_size` is at byte 111
/// within it). This is the kernel's declared contiguous memory
/// requirement during boot decompression. For a bootable ELF, returns
/// the highest end address (`p_paddr + p_memsz`) among PT_LOAD program
/// headers so the memory budget always contains every loaded segment.
///
/// aarch64 Image: reads `image_size` from the arm64 image header at
/// file offset 16 (after code0 + code1 + text_offset). For gzip-
/// compressed vmlinuz, falls back to file size * 4 as a conservative
/// estimate of the decompressed Image size.
pub(crate) fn read_kernel_init_size(kernel_path: &Path) -> Result<u64> {
    use std::io::{Read, Seek, SeekFrom};
    let mut f = std::fs::File::open(kernel_path)
        .with_context(|| format!("open kernel for init_size: {}", kernel_path.display()))?;

    #[cfg(target_arch = "x86_64")]
    {
        let mut magic = [0u8; 4];
        f.read_exact(&mut magic).context("read kernel magic")?;
        if magic == *b"\x7fELF" {
            // ELF64_Ehdr layout (little-endian): e_phoff at 0x20,
            // e_phentsize at 0x36, and e_phnum at 0x38.
            let mut ehdr = [0u8; 64];
            ehdr[..4].copy_from_slice(&magic);
            f.read_exact(&mut ehdr[4..]).context("read ELF64 header")?;
            anyhow::ensure!(ehdr[4] == 2, "kernel ELF is not ELF64");
            anyhow::ensure!(ehdr[5] == 1, "kernel ELF is not little-endian");
            let phoff = u64::from_le_bytes(ehdr[32..40].try_into().unwrap());
            let phentsize = u16::from_le_bytes(ehdr[54..56].try_into().unwrap());
            let phnum = u16::from_le_bytes(ehdr[56..58].try_into().unwrap());
            anyhow::ensure!(
                phentsize as usize >= 56,
                "kernel ELF program header is too small: {phentsize}"
            );
            anyhow::ensure!(phnum > 0, "kernel ELF has no program headers");

            let mut max_end = 0u64;
            let mut phdr = vec![0u8; phentsize as usize];
            for index in 0..phnum {
                let offset = phoff
                    .checked_add(u64::from(index) * u64::from(phentsize))
                    .context("kernel ELF program-header offset overflow")?;
                f.seek(SeekFrom::Start(offset))
                    .context("seek to ELF program header")?;
                f.read_exact(&mut phdr).context("read ELF program header")?;
                let p_type = u32::from_le_bytes(phdr[0..4].try_into().unwrap());
                if p_type != 1 {
                    continue;
                }
                let p_paddr = u64::from_le_bytes(phdr[24..32].try_into().unwrap());
                let p_memsz = u64::from_le_bytes(phdr[40..48].try_into().unwrap());
                let end = p_paddr
                    .checked_add(p_memsz)
                    .context("kernel ELF PT_LOAD address overflow")?;
                max_end = max_end.max(end);
            }
            anyhow::ensure!(max_end > 0, "kernel ELF has no non-empty PT_LOAD segments");
            return Ok(max_end);
        }

        // setup_header starts at 0x1F1, init_size at offset 111.
        f.seek(SeekFrom::Start(0x260))
            .context("seek to init_size in bzImage")?;
        let mut buf = [0u8; 4];
        f.read_exact(&mut buf)
            .context("read init_size from bzImage")?;
        Ok(u32::from_le_bytes(buf) as u64)
    }

    #[cfg(target_arch = "aarch64")]
    {
        // Check for gzip magic (0x1f 0x8b).
        let mut magic = [0u8; 2];
        f.read_exact(&mut magic).context("read kernel magic")?;
        if magic == [0x1f, 0x8b] {
            // Compressed vmlinuz — decompress header to read image_size.
            f.seek(SeekFrom::Start(0))
                .context("seek vmlinuz to start")?;
            let mut decoder = flate2::read::GzDecoder::new(&mut f);
            let mut header = [0u8; 24];
            decoder
                .read_exact(&mut header)
                .context("decompress arm64 vmlinuz header for image_size")?;
            return Ok(u64::from_le_bytes(header[16..24].try_into().unwrap()));
        }
        // Raw PE Image: image_size is a little-endian u64 at offset 16.
        f.seek(SeekFrom::Start(16))
            .context("seek to image_size in arm64 Image")?;
        let mut buf = [0u8; 8];
        f.read_exact(&mut buf)
            .context("read image_size from arm64 Image")?;
        Ok(u64::from_le_bytes(buf))
    }
}

/// Read the guest kernel's `(major, minor, patch)` version from the
/// kernel image, for the `initramfs_options=size=90%` honoring gate (see
/// [`TmpfsFraction::for_kernel_version`]).
///
/// x86_64 bzImage: the setup_header `kernel_version` field (a `u16` at
/// file offset 0x20E) is "a pointer to a NUL-terminated version string,
/// less 0x200" (`Documentation/arch/x86/boot.rst`). When nonzero, the
/// string lives at file offset `0x200 + value` and begins with
/// `UTS_RELEASE` (`arch/x86/boot/version.c`), i.e.
/// `MAJOR.MINOR.PATCH[-extra]` (e.g. `6.18.0-rc1`). This parses the
/// leading `MAJOR.MINOR.PATCH`.
///
/// aarch64 Image: the arm64 image header carries no version string —
/// returns `None` here. The caller `tmpfs_fraction` falls back to the
/// cache `metadata.json` sidecar
/// (`read_kernel_version_from_metadata_sidecar`) to recover the aarch64
/// version, so aarch64 is not unconditionally sized at 50%.
///
/// Returns `None` (-> conservative 50% sizing) on ANY uncertainty:
/// unreadable image, `kernel_version` field zero (pre-2.00 protocol or
/// stripped), the string offset out of range, non-UTF-8, or a malformed
/// leading `MAJOR.MINOR`. NEVER errors — a missing version is a normal,
/// safe outcome, not a boot-blocking failure.
pub(crate) fn read_kernel_version(kernel_path: &Path) -> Option<(u16, u16, u16)> {
    #[cfg(target_arch = "x86_64")]
    {
        use std::io::{Read, Seek, SeekFrom};
        let mut f = std::fs::File::open(kernel_path).ok()?;
        // Validate this is a real bzImage before trusting the version
        // field: the setup_header "HdrS" magic is a 4-byte field at file
        // offset 0x202 (boot protocol >= 2.00; the same check
        // `linux_loader::BzImage::load` makes). Without it, a
        // non-bzImage / pre-2.00 / corrupt image's arbitrary bytes at
        // 0x20E could parse as a honoring version and select the 90%
        // tmpfs fraction on a kernel that only gives 50% — the forbidden
        // panic direction. On any mismatch, return None (=> the safe
        // Half), matching the hostile-input doctrine: 90% is taken only
        // for a positively-confirmed bzImage.
        f.seek(SeekFrom::Start(0x202)).ok()?;
        let mut magic = [0u8; 4];
        f.read_exact(&mut magic).ok()?;
        if &magic != b"HdrS" {
            return None;
        }
        // setup_header `kernel_version` is a u16 at file offset 0x20E.
        f.seek(SeekFrom::Start(0x20E)).ok()?;
        let mut vbuf = [0u8; 2];
        f.read_exact(&mut vbuf).ok()?;
        let ver_ptr = u16::from_le_bytes(vbuf);
        // Zero => no version string (boot protocol < 2.00, or absent).
        if ver_ptr == 0 {
            return None;
        }
        // String is at file offset 0x200 + ver_ptr, NUL-terminated. Read
        // a bounded 256-byte window so a corrupt pointer can't drive an
        // unbounded read.
        f.seek(SeekFrom::Start(0x200u64 + ver_ptr as u64)).ok()?;
        let mut window = [0u8; 256];
        let n = f.read(&mut window).ok()?;
        let bytes = &window[..n];
        // The string is `RELEASE (compile-by@host) ...`; a NUL or space
        // bounds the RELEASE token.
        let end = bytes
            .iter()
            .position(|&b| b == 0 || b == b' ')
            .unwrap_or(bytes.len());
        let release = std::str::from_utf8(&bytes[..end]).ok()?;
        parse_kernel_version(release)
    }
    #[cfg(target_arch = "aarch64")]
    {
        // arm64 Image header carries no version string; the cache
        // metadata.json sidecar is the aarch64 version source (see
        // read_kernel_version_from_metadata_sidecar).
        let _ = kernel_path;
        None
    }
}

/// Parse the leading `MAJOR.MINOR.PATCH` from a kernel release string
/// such as `6.6.113`, `6.18.0-rc1`, or `7.1.0-rc7-gc80ba8d32ec3`.
/// Returns `None` if either of the first two dot-separated components is
/// absent or non-numeric. The patch component is optional — mainline rc
/// tags (`6.18-rc1`) carry none, and mainline >= 6.18 honors regardless
/// of patch; an absent or non-numeric patch is reported as `0` (only the
/// stable-backport floors consult patch, and every stable release is
/// `MAJOR.MINOR.PATCH`). Free fn so the host unit tests pin the parse
/// against real release-string shapes without constructing a bzImage.
fn parse_kernel_version(release: &str) -> Option<(u16, u16, u16)> {
    let mut parts = release.split('.');
    let major: u16 = parts.next()?.parse().ok()?;
    // Minor may carry a trailing `-rcN` only when there is no PATCH
    // component (e.g. `6.18-rc1`); strip any non-digit suffix.
    let minor_raw = parts.next()?;
    let minor_digits: String = minor_raw
        .chars()
        .take_while(|c| c.is_ascii_digit())
        .collect();
    if minor_digits.is_empty() {
        return None;
    }
    let minor: u16 = minor_digits.parse().ok()?;
    // Patch is optional; strip any trailing non-digit suffix (e.g.
    // `0-rc7-g...`). Absent, empty, or non-numeric => 0.
    let patch: u16 = parts
        .next()
        .map(|p| {
            p.chars()
                .take_while(|c| c.is_ascii_digit())
                .collect::<String>()
        })
        .and_then(|d| d.parse().ok())
        .unwrap_or(0);
    Some((major, minor, patch))
}

/// Recover the guest kernel `(major, minor, patch)` version from the
/// cache `metadata.json` sidecar next to the boot image.
///
/// The aarch64 `Image` carries no embedded version string (so
/// [`read_kernel_version`] returns `None` there), but a
/// cached kernel records its version in `metadata.json` alongside the
/// image: `crate::cache::CacheDir` stores the boot image at
/// `<entry>/<image_name>` and its metadata at `<entry>/metadata.json`,
/// so the sidecar is the image's sibling. This recovers the 90% reclaim
/// for cache-resident aarch64 kernels (whose version is recorded into
/// `metadata.json` by the acquisition: the kernel.org tarball download,
/// or — via the source-tree Makefile — a local source tree or git
/// clone).
///
/// Returns `None` — falling to the safe [`TmpfsFraction::Half`] — for
/// every case the version can't be positively established: a non-cache
/// image path (raw `--kernel`, no sibling sidecar), an acquisition that
/// recorded no version (`version` absent/`null` — e.g. a source build
/// whose `Makefile` was unparsable), an unreadable or malformed
/// `metadata.json`,
/// or an unparsable version string. Only `version` is read (a minimal
/// probe struct ignores the rest of the schema), keeping this decoupled
/// from `crate::cache::KernelMetadata`'s other fields. The sidecar is
/// host-authored cache infrastructure, not guest input, so trusting its
/// version for the fraction decision is consistent with the threat model
/// (the guest never writes it).
pub(crate) fn read_kernel_version_from_metadata_sidecar(
    kernel_path: &Path,
) -> Option<(u16, u16, u16)> {
    #[derive(serde::Deserialize)]
    struct VersionProbe {
        version: Option<String>,
    }
    let sidecar = kernel_path.parent()?.join("metadata.json");
    let json = std::fs::read_to_string(sidecar).ok()?;
    let probe: VersionProbe = serde_json::from_str(&json).ok()?;
    parse_kernel_version(&probe.version?)
}

/// Minimum guest memory (in MiB) needed to boot, extract the initramfs,
/// and run the test workload.
///
/// ```text
/// total = computed_boot_requirement + WORKLOAD_MIB + shm
/// ```
///
/// ## Computed boot requirement
///
/// Every term is derived from values known at allocation time. The model
/// follows the kernel's boot memory layout.
///
/// **memblock-reserved regions** (excluded from `totalram_pages`):
///
/// - `kernel_init_size`: bzImage setup_header `init_size` field (offset
///   0x260) — the kernel's declared contiguous memory requirement during
///   boot decompression. Includes compressed payload, decompressed
///   vmlinux, and decompression workspace. Overestimates resident kernel
///   since init sections (`free_initmem`, `init/main.c`) and the
///   decompression workspace are freed post-boot. The slack absorbs
///   percpu allocations (`pcpu_embed_first_chunk` in `mm/percpu.c`
///   reserves `static_size + reserved_size + dyn_size` per CPU via
///   memblock, ~220KB/CPU with ktstr's kconfig which disables LOCKDEP)
///   and misc boot allocations (page tables, slab bootstrap, hash tables).
///
/// - `compressed_initrd`: memblock-reserved by `reserve_initrd_mem()`
///   (`init/initramfs.c:640`: `memblock_reserve(start, size)`) until
///   `free_initrd_mem()` after `unpack_to_rootfs` completes.
///
/// - struct page array: `P / 64` bytes. Each 4KB page requires a
///   `struct page` descriptor. Base size = 56 bytes (flags:8 + 5-word
///   union:40 + _mapcount:4 + _refcount:4), reaching 64 either by
///   `CONFIG_HAVE_ALIGNED_STRUCT_PAGE` 16-byte alignment padding
///   (`CONFIG_MEMCG=n`) or by the extra `memcg_data:8` field
///   (`CONFIG_MEMCG=y`) — same `/64` either way
///   (`include/linux/mm_types.h`). `CONFIG_KMSAN` (off here) would add
///   two pointers (→ 80 bytes); excluded.
///
/// **tmpfs constraint** (the binding limit for initramfs extraction):
///
/// The rootfs tmpfs is mounted by `init_mount_tree()` (`fs/namespace.c`)
/// via `vfs_kern_mount(&rootfs_fs_type, 0, ...)` — flags=0, NOT
/// `SB_KERNMOUNT`. `alloc_super` (`fs/super.c`) sets `s->s_flags = flags`,
/// so `SB_KERNMOUNT` is not set. In `shmem_fill_super` (`mm/shmem.c`),
/// the `!(sb->s_flags & SB_KERNMOUNT)` branch runs, and since no
/// `size=` mount option was parsed (`SHMEM_SEEN_BLOCKS` unset), it
/// falls through to `ctx->blocks = shmem_default_max_blocks()` =
/// `totalram_pages() / 2` (`mm/shmem.c:146`).
///
/// `initramfs_options=size=90%` on the cmdline is consumed by
/// `init_mount_tree()` (`fs/namespace.c`, via `initramfs_options_setup`)
/// when mounting the rootfs tmpfs — but only on kernels carrying mainline
/// commit 278033a225e1 ("fs: Add 'initramfs_options' to set initramfs
/// mount options"), first tagged v6.18-rc1. On kernels without it the
/// parameter is silently ignored ("Unknown kernel command line
/// parameters …, will be passed to user space") and the tmpfs uses its
/// 50% default. (The commit was also backported to the stable series —
/// v5.4.301, v5.10.246, v5.15.195, v6.1.157, v6.6.113, v6.12.54, v6.17.4
/// — which ktstr recognizes via `TmpfsFraction::for_kernel_version`.)
/// Initramfs unpacking then fails
/// with `write error` partway through if the uncompressed payload
/// exceeds the live tmpfs limit, leaving `/init` packed but its
/// dynamic-linker dep missing → `Failed to execute /init (error -2)`
/// → kernel panic.
///
/// The formula below sizes for `budget.tmpfs_fraction`: 90% only when
/// the guest kernel is positively known to honor the hint (mainline
/// 6.18+ or a stable series at or above its backport floor, via
/// `TmpfsFraction::for_kernel_version`), else the 50% default. Sizing
/// for 50% on a kernel that honors 90% is safe — the tmpfs is bigger
/// than we sized for; sizing for 90% on a kernel that only gives 50%
/// panics the guest mid-extraction, so the 90% path is taken ONLY on a
/// confirmed-honoring version (every uncertainty — unknown version, an
/// image lacking both an embedded version and a honoring sidecar, a
/// series absent from the table, or below its series floor — falls to
/// the safe 50%).
///
/// Note: `rootflags=size=90%` would set `root_mount_data` (assigned by
/// `root_data_setup` via `__setup("rootflags=", ...)` in
/// `init/do_mounts.c`), consumed only by `do_mount_root()` via
/// `prepare_namespace()`. With `rdinit=`, `kernel_init_freeable`
/// (`init/main.c`) skips `prepare_namespace()` when `init_eaccess`
/// succeeds, so `rootflags=` is never applied to the rootfs.
///
/// The `SB_KERNMOUNT` (unlimited) tmpfs is the separate `shm_mnt`
/// created by `shmem_init()` via `kern_mount()` — used for anonymous
/// shared memory (`shmem_file_setup`), not the rootfs.
///
/// ```text
/// totalram_pages(P) = (P - init_size - compressed - P/64) / 4096
/// tmpfs_max_pages = totalram_pages / 2
/// constraint: tmpfs_max_pages >= uncompressed / 4096
///
/// Solving for P:
/// (P - init_size - compressed - P/64) / 2 >= uncompressed
/// P * 63/64 >= 2 * uncompressed + init_size + compressed
/// P >= (2 * uncompressed + init_size + compressed) * 64/63
/// ```
///
/// ## Workload budget
///
/// 256 MiB for scheduler execution, test scenarios, and runtime
/// allocations (cgroup memory, BPF maps, process stacks, slab caches).
/// This is a deliberate budget for post-boot workload, not a guess at
/// kernel overhead.
///
/// Workload budget (MiB): scheduler execution, test scenarios, cgroup
/// memory, BPF maps, and runtime allocations.
///
/// ## Coverage-instrumented `/init` reserve
///
/// A `-C instrument-coverage` `/init` payload (the
/// [`MemoryBudget::init_coverage_instrumented`] flag) needs more than
/// the base workload budget: it holds the LLVM profile-counter
/// sections resident, and at flush time
/// `__llvm_profile_write_buffer`
/// (`crate::test_support::try_flush_profraw`) serializes them into a
/// heap buffer whose size `__llvm_profile_get_size_for_buffer`
/// reports. `WORKLOAD_MIB` is sized for a non-instrumented payload,
/// so the instrumented case adds
/// [`MemoryBudget::instrumented_reserve_bytes`] (the payload's
/// `__llvm_prf_cnts` + `__llvm_prf_data` section sizes, MiB-ceil) to
/// the workload term. Without it a ~600 MiB-stripped instrumented
/// `/init` OOMs during boot (the empirically-confirmed
/// `memory_deferred_min(4096)` workaround the reserve replaces with a
/// right-sized figure derived from the actual section sizes).
const WORKLOAD_MIB: u64 = 256;

/// Per-vCPU host-residency reserve (MiB) added to the touch ceiling
/// ([`touch_ceiling_mib`]).
///
/// A running guest faults in per-vCPU kernel state — kernel stacks,
/// percpu allocations, and per-CPU page-table / slab pages — that scales
/// with vCPU count rather than with the guest's advertised RAM.
///
/// Calibration: a 224-vCPU cell shows ~1.5 GiB steady host residency. The
/// image/kernel base (`uncompressed + compressed + init_size`, ~440 MiB on
/// the production image) and the post-boot workload budget
/// ([`WORKLOAD_MIB`], 256 MiB) account for ~700 MiB, leaving
/// ~800 MiB / 224 ≈ 3.6 MiB/vCPU of per-vCPU kernel state. The constant is
/// set to 6 MiB/vCPU — measured-with-margin — so the permit never
/// under-charges a genuinely resident cell. Over-charging is the safe
/// direction (it costs a little admission headroom); under-charging risks
/// host OOM under a VM storm.
const PER_VCPU_KERNEL_TOUCH_MIB: u64 = 6;

pub(crate) fn initramfs_min_memory_mib(budget: &MemoryBudget) -> u32 {
    let ceil_mib = |bytes: u64| -> u64 { bytes.saturating_add((1 << 20) - 1) >> 20 };

    let init_size_mib = ceil_mib(budget.kernel_init_size);
    let compressed_mib = ceil_mib(budget.compressed_initrd_bytes);
    let uncompressed_mib = ceil_mib(budget.uncompressed_initramfs_bytes);

    // Boot requirement: the rootfs tmpfs block limit is a fraction F of
    // totalram_pages. F = 50% (`shmem_default_max_blocks`) on every
    // kernel that does not positively honor `initramfs_options=size=90%`;
    // F = 90% only on a positively-honoring kernel (mainline 6.18+ or a
    // stable series at/above its backport floor; see `TmpfsFraction` /
    // `TmpfsFraction::for_kernel_version`). A larger F sizes LESS RAM.
    //
    // Constraint (F = frac_num/frac_den): F * totalram_pages >= uncompressed_pages.
    // totalram_pages = (P - reserved) / PAGE_SIZE.
    // reserved = init_size + compressed + struct_page(P) = init_size + compressed + P/64.
    //
    // Solving:
    //   (frac_num/frac_den) * (P - init_size - compressed - P/64) >= uncompressed
    //   P * 63/64 >= (frac_den/frac_num) * uncompressed + init_size + compressed
    //   P >= ((frac_den/frac_num)*uncompressed + init_size + compressed) * 64/63
    // At F=1/2 this is the prior `2 * uncompressed`; at F=9/10 it scales
    // uncompressed by 10/9. div_ceil never rounds DOWN (rounding down
    // would under-size RAM and risk a mid-boot tmpfs overrun).
    let (frac_num, frac_den) = budget.tmpfs_fraction.ratio();
    let uncompressed_scaled = uncompressed_mib.saturating_mul(frac_den).div_ceil(frac_num);
    let content_mib = uncompressed_scaled
        .saturating_add(init_size_mib)
        .saturating_add(compressed_mib);

    // struct page overhead: P/64 is part of reserved, creating a
    // circular dependency. Solve: P = content * 64/63.
    let boot_mib = content_mib.saturating_mul(64).div_ceil(63);

    // Coverage-instrumented `/init` reserve: add the live profile
    // sections (cnts + data) on top of the workload budget. Non-zero
    // only when the payload is instrumented (see the const-level doc).
    let coverage_reserve_mib = if budget.init_coverage_instrumented {
        ceil_mib(budget.instrumented_reserve_bytes)
    } else {
        0
    };

    // total = computed boot requirement + workload budget + coverage
    // reserve. All arithmetic above is saturating, so a pathological
    // (hostile or buggy) input — a corrupt kernel Image `init_size`, a
    // malformed `/init` ELF section size — saturates toward `u64::MAX`
    // rather than wrapping to a too-small value; the `u32::try_from`
    // below then fails LOUDLY (panic) instead of silently truncating
    // the floor and OOMing the guest mid-boot.
    let total_mib = boot_mib
        .saturating_add(WORKLOAD_MIB)
        .saturating_add(coverage_reserve_mib);
    u32::try_from(total_mib).unwrap_or_else(|_| {
        panic!(
            "initramfs_min_memory_mib: computed floor {total_mib}MiB exceeds u32 \
             (boot={boot_mib}MiB, workload={WORKLOAD_MIB}MiB, \
             coverage_reserve={coverage_reserve_mib}MiB)"
        )
    })
}

/// Host physical memory (MiB) a *trusted deferred* guest is expected to
/// make resident — the memory-PERMIT ceiling, deliberately distinct from
/// the guest's sized RAM ([`initramfs_min_memory_mib`]).
///
/// The admission permit reserves what the workload actually touches, not
/// the guest's advertised size. Default/no-perf guest RAM is
/// `MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE` demand-paged (see
/// `super::numa_mem::anonymous_node_map_flags` and its oversubscription
/// note): untouched pages cost no host memory and are free to
/// oversubscribe, so charging the full sized RAM would reject useful
/// concurrency for memory the guest never faults in. On a wide cell the
/// sized RAM is the `64 MiB/vCPU` floor (e.g. 14.3 GiB at 224 vCPU) while
/// the resident set is a small fraction of it.
///
/// The ceiling sums:
/// - the image/kernel base — `uncompressed + compressed + init_size` —
///   faulted in to load and unpack the initramfs. The unpack peak holds
///   the compressed archive and its uncompressed expansion together (the
///   archive is freed only after `unpack_to_rootfs` completes), and
///   `init_size` covers kernel decompression + workspace. The
///   tmpfs-fraction multiplier from [`initramfs_min_memory_mib`] is
///   deliberately EXCLUDED here: it sizes the guest's rootfs tmpfs
///   *limit*, not host residency.
/// - [`WORKLOAD_MIB`] — the same post-boot working-set budget the SIZE
///   formula reserves (single source of truth).
/// - a per-vCPU kernel-state reserve ([`PER_VCPU_KERNEL_TOUCH_MIB`] per
///   vCPU).
/// - the coverage-instrumented reserve when present: an instrumented
///   `/init` keeps its profile-counter sections resident, so those bytes
///   are genuinely touched and must be charged (mirrors
///   [`initramfs_min_memory_mib`]).
///
/// The caller charges `required_chunks(max(touch_ceiling, declared))`, so a
/// test that legitimately touches more than this must declare it via
/// `.memory_mib(...)`; the only OOM exposure is an undeclared
/// over-allocating test.
pub(crate) fn touch_ceiling_mib(budget: &MemoryBudget, vcpus: u32) -> u32 {
    let ceil_mib = |bytes: u64| -> u64 { bytes.saturating_add((1 << 20) - 1) >> 20 };

    // Image/kernel base actually faulted in to boot and unpack. Unlike the
    // SIZE formula, uncompressed is charged once (host residency, not the
    // tmpfs limit) and the *64/63 struct-page factor is omitted (that
    // overhead scales with sized RAM, which the touch ceiling is not).
    let image_base_mib = ceil_mib(budget.uncompressed_initramfs_bytes)
        .saturating_add(ceil_mib(budget.compressed_initrd_bytes))
        .saturating_add(ceil_mib(budget.kernel_init_size));

    let per_vcpu_mib = u64::from(vcpus).saturating_mul(PER_VCPU_KERNEL_TOUCH_MIB);

    let coverage_reserve_mib = if budget.init_coverage_instrumented {
        ceil_mib(budget.instrumented_reserve_bytes)
    } else {
        0
    };

    let total_mib = image_base_mib
        .saturating_add(WORKLOAD_MIB)
        .saturating_add(per_vcpu_mib)
        .saturating_add(coverage_reserve_mib);
    u32::try_from(total_mib).unwrap_or_else(|_| {
        panic!(
            "touch_ceiling_mib: computed ceiling {total_mib}MiB exceeds u32 \
             (image_base={image_base_mib}MiB, workload={WORKLOAD_MIB}MiB, \
             per_vcpu={per_vcpu_mib}MiB, coverage_reserve={coverage_reserve_mib}MiB, \
             vcpus={vcpus})"
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Pin the workload-budget constant. Bumping the value
    /// (`WORKLOAD_MIB`) changes the floor for every deferred-memory
    /// VM boot; this test fails any change so the bump goes through
    /// review rather than slipping in unnoticed.
    #[test]
    fn workload_mib_is_256() {
        assert_eq!(WORKLOAD_MIB, 256);
    }

    /// All-zero inputs collapse to just the workload budget — no
    /// kernel, no initramfs. Pins the lower bound the deferred-memory
    /// path always allocates.
    #[test]
    fn initramfs_min_memory_mib_zeros_returns_workload_budget() {
        let budget = MemoryBudget {
            uncompressed_initramfs_bytes: 0,
            compressed_initrd_bytes: 0,
            kernel_init_size: 0,
            init_coverage_instrumented: false,
            instrumented_reserve_bytes: 0,
            tmpfs_fraction: TmpfsFraction::Half,
        };
        assert_eq!(initramfs_min_memory_mib(&budget), WORKLOAD_MIB as u32);
    }

    /// `kernel_init_size` and `compressed_initrd_bytes` flow into
    /// `content_mib` additively, then through the `*64/63` struct-page
    /// circular-dependency factor. Verify the math against a
    /// hand-computed reference. Inputs:
    ///   uncompressed=10 MiB, init_size=5 MiB, compressed=2 MiB.
    /// Hand trace per `initramfs_min_memory_mib`:
    ///   uncompressed_scaled = 10 * 2 = 20
    ///   content_mib         = 20 + 5 + 2 = 27
    ///   boot_mib            = ceil(27*64/63) = ceil(27.428) = 28
    ///   total              = 28 + 256 (WORKLOAD_MIB) = 284
    #[test]
    fn initramfs_min_memory_mib_known_input() {
        let budget = MemoryBudget {
            uncompressed_initramfs_bytes: 10 * (1 << 20),
            compressed_initrd_bytes: 2 * (1 << 20),
            kernel_init_size: 5 * (1 << 20),
            init_coverage_instrumented: false,
            instrumented_reserve_bytes: 0,
            tmpfs_fraction: TmpfsFraction::Half,
        };
        assert_eq!(initramfs_min_memory_mib(&budget), 284);
    }

    /// Sub-MiB inputs round up to 1 MiB before participating in the
    /// math. A 1-byte initramfs (degenerate but reachable when test
    /// fixtures construct empty payloads) must not silently round
    /// down to zero and bypass the tmpfs-50% safety factor. With
    /// uncompressed=1 byte, init=0, compressed=0:
    ///   uncompressed_scaled = 1 * 2 = 2
    ///   content_mib         = 2 + 0 + 0 = 2
    ///   boot_mib            = ceil(2*64/63) = ceil(2.031) = 3
    ///   total              = 3 + 256 = 259
    #[test]
    fn initramfs_min_memory_mib_subbyte_uncompressed_rounds_up() {
        let budget = MemoryBudget {
            uncompressed_initramfs_bytes: 1,
            compressed_initrd_bytes: 0,
            kernel_init_size: 0,
            init_coverage_instrumented: false,
            instrumented_reserve_bytes: 0,
            tmpfs_fraction: TmpfsFraction::Half,
        };
        assert_eq!(initramfs_min_memory_mib(&budget), 259);
    }

    /// Larger realistic-shape inputs: uncompressed=200 MiB,
    /// compressed=50 MiB, init_size=30 MiB.
    /// Verifies the math holds at integration-realistic scales (the
    /// production callers in vmm/mod.rs feed values of this order).
    /// Trace:
    ///   uncompressed_scaled = 200 * 2 = 400
    ///   content_mib         = 400 + 30 + 50 = 480
    ///   boot_mib            = ceil(480*64/63) = ceil(487.619) = 488
    ///   total              = 488 + 256 = 744
    #[test]
    fn initramfs_min_memory_mib_larger_input() {
        let budget = MemoryBudget {
            uncompressed_initramfs_bytes: 200 * (1 << 20),
            compressed_initrd_bytes: 50 * (1 << 20),
            kernel_init_size: 30 * (1 << 20),
            init_coverage_instrumented: false,
            instrumented_reserve_bytes: 0,
            tmpfs_fraction: TmpfsFraction::Half,
        };
        let memory_mib = initramfs_min_memory_mib(&budget);
        assert_eq!(memory_mib, 744);
        assert!(
            memory_mib < 2048,
            "deferred sizing must be allowed to choose a right-sized sub-2-GiB VM"
        );
    }

    /// Coverage-instrumented shape: a GiB-scale instrumented `/init`
    /// with a multi-hundred-MiB profile-counter reserve must produce a
    /// floor well above the non-instrumented case AND above the
    /// previously-used `memory_deferred_min(4096)` workaround.
    ///
    /// Inputs model a ~600 MiB-stripped instrumented `/init`:
    ///   uncompressed=1200 MiB (binary + suffix), compressed=300 MiB,
    ///   init_size=30 MiB, reserve=3500 MiB (cnts + data sections of a
    ///   heavily-instrumented binary).
    /// Trace:
    ///   uncompressed_scaled = 1200 * 2 = 2400
    ///   content_mib         = 2400 + 30 + 300 = 2730
    ///   boot_mib            = ceil(2730*64/63) = ceil(2773.33) = 2774
    ///   coverage_reserve    = 3500
    ///   total              = 2774 + 256 + 3500 = 6530
    ///
    /// Two assertions pin the contract: (1) the instrumented floor is
    /// strictly larger than the SAME budget with the flag off (the
    /// reserve is actually added), and (2) the floor clears 4096 so an
    /// instrumented `/init` of this shape boots without the manual
    /// `memory_deferred_min(4096)` override.
    #[test]
    fn initramfs_min_memory_mib_instrumented_reserve_raises_floor() {
        let base = MemoryBudget {
            uncompressed_initramfs_bytes: 1200 * (1 << 20),
            compressed_initrd_bytes: 300 * (1 << 20),
            kernel_init_size: 30 * (1 << 20),
            init_coverage_instrumented: false,
            instrumented_reserve_bytes: 3500 * (1 << 20),
            tmpfs_fraction: TmpfsFraction::Half,
        };
        let instrumented = MemoryBudget {
            init_coverage_instrumented: true,
            ..base
        };

        let base_floor = initramfs_min_memory_mib(&base);
        let instrumented_floor = initramfs_min_memory_mib(&instrumented);

        // Flag off: reserve bytes are ignored entirely.
        assert_eq!(
            base_floor, 3030,
            "non-instrumented floor must NOT include the reserve \
             (2774 boot + 256 workload)"
        );
        // Flag on: reserve is added on top of the workload term.
        assert_eq!(
            instrumented_floor, 6530,
            "instrumented floor = 2774 boot + 256 workload + 3500 reserve"
        );
        assert!(
            instrumented_floor > base_floor,
            "instrumented reserve must raise the floor ({instrumented_floor} \
             vs {base_floor})"
        );
        assert!(
            instrumented_floor > 4096,
            "instrumented floor must clear the old memory_deferred_min(4096) \
             workaround (got {instrumented_floor})"
        );
    }

    /// `TmpfsFraction::for_kernel_version` gates the 90% bump on a
    /// positively-honoring kernel: mainline 6.18+ (regardless of patch)
    /// OR a stable series at/above its verified backport floor. Every
    /// uncertain, absent-series, or below-floor case falls to the safe
    /// 50%.
    #[test]
    fn tmpfs_fraction_gates_on_honoring_versions() {
        use TmpfsFraction::{Half, NinetyPercent};
        let frac = TmpfsFraction::for_kernel_version;

        // Mainline 6.18+ honors regardless of patch.
        assert_eq!(frac(Some((6, 18, 0))), NinetyPercent);
        assert_eq!(frac(Some((6, 19, 0))), NinetyPercent);
        assert_eq!(frac(Some((7, 0, 5))), NinetyPercent);
        // Mainline below 6.18 in a series with no backport at all (6.16 is
        // absent from the floor table) -> Half. (6.17 below-floor is
        // covered in the per-series block below.)
        assert_eq!(frac(Some((6, 16, 0))), Half);

        // Stable-backport series: AT the floor -> NinetyPercent; one
        // BELOW the floor -> Half. Floors verified via git tag --contains.
        assert_eq!(frac(Some((6, 17, 4))), NinetyPercent);
        assert_eq!(frac(Some((6, 17, 3))), Half);
        assert_eq!(frac(Some((6, 12, 54))), NinetyPercent);
        assert_eq!(frac(Some((6, 12, 53))), Half);
        assert_eq!(frac(Some((6, 6, 113))), NinetyPercent);
        assert_eq!(frac(Some((6, 6, 112))), Half);
        assert_eq!(frac(Some((6, 1, 157))), NinetyPercent);
        assert_eq!(frac(Some((6, 1, 156))), Half);
        assert_eq!(frac(Some((5, 15, 195))), NinetyPercent);
        assert_eq!(frac(Some((5, 15, 194))), Half);
        assert_eq!(frac(Some((5, 10, 246))), NinetyPercent);
        assert_eq!(frac(Some((5, 10, 245))), Half);
        assert_eq!(frac(Some((5, 4, 301))), NinetyPercent);
        assert_eq!(frac(Some((5, 4, 300))), Half);

        // A series absent from the table (EOL / no backport) -> Half,
        // even at a high patch level.
        assert_eq!(frac(Some((6, 9, 999))), Half);
        assert_eq!(frac(Some((6, 13, 999))), Half);
        // No version established -> Half.
        assert_eq!(frac(None), Half);
    }

    /// `parse_kernel_version` extracts the leading MAJOR.MINOR.PATCH from
    /// real kernel release-string shapes; the patch is optional (absent
    /// or rc-only => 0); a malformed major/minor yields `None`.
    #[test]
    fn parse_kernel_version_shapes() {
        assert_eq!(parse_kernel_version("6.18.0-rc1"), Some((6, 18, 0)));
        assert_eq!(
            parse_kernel_version("7.1.0-rc7-gc80ba8d32ec3"),
            Some((7, 1, 0))
        );
        assert_eq!(parse_kernel_version("6.12.54"), Some((6, 12, 54)));
        assert_eq!(parse_kernel_version("6.6.113"), Some((6, 6, 113)));
        // Minor-only rc tag: no patch component => patch 0.
        assert_eq!(parse_kernel_version("6.18-rc1"), Some((6, 18, 0)));
        // Non-numeric patch => 0 (major.minor still parse).
        assert_eq!(parse_kernel_version("6.6.x"), Some((6, 6, 0)));
        assert_eq!(parse_kernel_version(""), None);
        assert_eq!(parse_kernel_version("garbage"), None);
        assert_eq!(parse_kernel_version("6"), None);
        assert_eq!(parse_kernel_version("6."), None);
        assert_eq!(parse_kernel_version("x.18"), None);
    }

    /// At the SAME payload, the 90% fraction sizes strictly LESS total
    /// RAM than the 50% fraction — the entire point of the reclaim. A
    /// change inverting the constraint (sizing MORE RAM for 90%) would
    /// silently lose the reclaim AND risk under-sizing.
    #[test]
    fn ninety_percent_fraction_sizes_less_ram_than_half() {
        let make = |frac: TmpfsFraction| MemoryBudget {
            uncompressed_initramfs_bytes: 1200 * (1 << 20),
            compressed_initrd_bytes: 300 * (1 << 20),
            kernel_init_size: 30 * (1 << 20),
            init_coverage_instrumented: false,
            instrumented_reserve_bytes: 0,
            tmpfs_fraction: frac,
        };
        let half = initramfs_min_memory_mib(&make(TmpfsFraction::Half));
        let ninety = initramfs_min_memory_mib(&make(TmpfsFraction::NinetyPercent));
        assert!(
            ninety < half,
            "90% tmpfs fraction must size less RAM than 50% \
             (ninety={ninety}MiB, half={half}MiB)"
        );
        // half: uncompressed_scaled = 1200*2 = 2400; content = 2730;
        //   boot = ceil(2730*64/63) = 2774; total = 2774 + 256 = 3030.
        assert_eq!(half, 3030, "50% floor: 2774 boot + 256 workload");
        // ninety: uncompressed_scaled = ceil(1200*10/9) = 1334;
        //   content = 1664; boot = ceil(1664*64/63) = 1691;
        //   total = 1691 + 256 = 1947.
        assert_eq!(ninety, 1947, "90% floor: 1691 boot + 256 workload");
    }

    /// `touch_ceiling_mib` sums the image/kernel base (uncompressed +
    /// compressed + init_size, each once), the workload budget, and the
    /// per-vCPU reserve. Production wide-cell shape: uncompressed=280 MiB,
    /// compressed=121 MiB, init_size=40 MiB, 224 vCPUs.
    /// Trace:
    ///   image_base = 280 + 121 + 40 = 441
    ///   per_vcpu   = 224 * 6         = 1344
    ///   total     = 441 + 256 (WORKLOAD_MIB) + 1344 = 2041
    #[test]
    fn touch_ceiling_mib_wide_cell_shape() {
        let budget = MemoryBudget {
            uncompressed_initramfs_bytes: 280 * (1 << 20),
            compressed_initrd_bytes: 121 * (1 << 20),
            kernel_init_size: 40 * (1 << 20),
            init_coverage_instrumented: false,
            instrumented_reserve_bytes: 0,
            tmpfs_fraction: TmpfsFraction::Half,
        };
        assert_eq!(touch_ceiling_mib(&budget, 224), 2041);
    }

    /// The whole point of the SIZE/PERMIT split: on a wide cell the sized
    /// RAM is the 64 MiB/vCPU floor (14.3 GiB) but the touch ceiling is a
    /// small fraction of it. Charged in 256 MiB permit chunks, a 224-vCPU
    /// cell drops from 56 chunks (sized) to 8 (touch), lifting projected
    /// pool concurrency on the 384 GB host (~1380 usable chunks) from ~24
    /// to ~170. This test pins the chunk arithmetic that
    /// `MemoryPermitPool::required_chunks` performs at admission.
    #[test]
    fn touch_ceiling_mib_wide_cell_permit_chunk_drop() {
        const CHUNK_MIB: u32 = 256; // MEMORY_PERMIT_CHUNK_MIB
        const USABLE_CHUNKS: u32 = 1380; // 384 GB host after the host reserve

        let budget = MemoryBudget {
            uncompressed_initramfs_bytes: 280 * (1 << 20),
            compressed_initrd_bytes: 121 * (1 << 20),
            kernel_init_size: 40 * (1 << 20),
            init_coverage_instrumented: false,
            instrumented_reserve_bytes: 0,
            tmpfs_fraction: TmpfsFraction::Half,
        };
        let vcpus = 224u32;

        // Sized RAM = max(initramfs_min, 64 MiB/vCPU floor) = 14336 MiB.
        let initramfs_min = initramfs_min_memory_mib(&budget);
        let cpu_scaled = vcpus * 64;
        let sized = initramfs_min.max(cpu_scaled);
        assert_eq!(sized, 14336, "wide cell sizes to the 64 MiB/vCPU floor");
        assert_eq!(sized.div_ceil(CHUNK_MIB), 56, "sized RAM charges 56 chunks");

        // Touch ceiling = 2041 MiB -> 8 chunks.
        let touch = touch_ceiling_mib(&budget, vcpus);
        assert_eq!(touch, 2041);
        assert_eq!(
            touch.div_ceil(CHUNK_MIB),
            8,
            "touch ceiling charges 8 chunks"
        );

        // Projected concurrency on the wide host.
        assert_eq!(USABLE_CHUNKS / sized.div_ceil(CHUNK_MIB), 24);
        assert_eq!(USABLE_CHUNKS / touch.div_ceil(CHUNK_MIB), 172);
    }

    /// The touch ceiling is host-residency, so it must NOT depend on the
    /// tmpfs fraction (that fraction sizes the guest's rootfs tmpfs limit,
    /// not what the host makes resident). Same payload, both fractions ->
    /// identical ceiling. This is the SIZE/PERMIT distinction the split
    /// exists for: `initramfs_min_memory_mib` differs across fractions (see
    /// `ninety_percent_fraction_sizes_less_ram_than_half`), the ceiling does
    /// not.
    #[test]
    fn touch_ceiling_mib_ignores_tmpfs_fraction() {
        let make = |frac: TmpfsFraction| MemoryBudget {
            uncompressed_initramfs_bytes: 280 * (1 << 20),
            compressed_initrd_bytes: 121 * (1 << 20),
            kernel_init_size: 40 * (1 << 20),
            init_coverage_instrumented: false,
            instrumented_reserve_bytes: 0,
            tmpfs_fraction: frac,
        };
        assert_eq!(
            touch_ceiling_mib(&make(TmpfsFraction::Half), 224),
            touch_ceiling_mib(&make(TmpfsFraction::NinetyPercent), 224),
        );
    }

    /// A coverage-instrumented `/init` keeps its profile-counter sections
    /// resident, so those bytes are genuinely touched and raise the ceiling
    /// (mirrors `initramfs_min_memory_mib`). Flag off ignores the reserve;
    /// flag on adds it (MiB-ceil).
    #[test]
    fn touch_ceiling_mib_charges_instrumented_reserve() {
        let base = MemoryBudget {
            uncompressed_initramfs_bytes: 280 * (1 << 20),
            compressed_initrd_bytes: 121 * (1 << 20),
            kernel_init_size: 40 * (1 << 20),
            init_coverage_instrumented: false,
            instrumented_reserve_bytes: 500 * (1 << 20),
            tmpfs_fraction: TmpfsFraction::Half,
        };
        let instrumented = MemoryBudget {
            init_coverage_instrumented: true,
            ..base
        };
        // Flag off: reserve ignored -> the 224-vCPU ceiling from above.
        assert_eq!(touch_ceiling_mib(&base, 224), 2041);
        // Flag on: + 500 MiB resident profile sections.
        assert_eq!(touch_ceiling_mib(&instrumented, 224), 2541);
    }

    /// Degenerate all-zero image at zero vCPUs collapses to just the
    /// workload budget; the per-vCPU term scales linearly from there.
    #[test]
    fn touch_ceiling_mib_zeros_and_per_vcpu_scaling() {
        let budget = MemoryBudget {
            uncompressed_initramfs_bytes: 0,
            compressed_initrd_bytes: 0,
            kernel_init_size: 0,
            init_coverage_instrumented: false,
            instrumented_reserve_bytes: 0,
            tmpfs_fraction: TmpfsFraction::Half,
        };
        assert_eq!(touch_ceiling_mib(&budget, 0), WORKLOAD_MIB as u32);
        // 4 vCPUs add 4 * 6 = 24 MiB.
        assert_eq!(touch_ceiling_mib(&budget, 4), WORKLOAD_MIB as u32 + 24);
    }

    /// `read_kernel_init_size` on x86_64 reads 4 little-endian bytes
    /// at file offset 0x260. Construct a tempfile padded to that
    /// offset with a known init_size value and assert the function
    /// returns it as u64. Pins the exact byte-offset and width
    /// against a future drift in the bzImage setup_header layout.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn read_kernel_init_size_x86_64_reads_offset_0x260() {
        use std::io::Write;

        let mut f = tempfile::NamedTempFile::new().expect("tempfile");
        // Pad up to 0x260 with zeros, then write 4 bytes of init_size.
        let pad = vec![0u8; 0x260];
        f.write_all(&pad).expect("write pad");
        // Distinct value, large enough that wrong-offset reads would
        // yield zero (the surrounding pad).
        let init_size: u32 = 0x1234_5678;
        f.write_all(&init_size.to_le_bytes())
            .expect("write init_size");
        f.flush().expect("flush");

        let got = read_kernel_init_size(f.path()).expect("read init_size");
        assert_eq!(got, init_size as u64);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn read_kernel_init_size_x86_64_reads_elf_load_extent() {
        use std::io::Write;

        let mut image = vec![0u8; 64 + 2 * 56];
        image[0..4].copy_from_slice(b"\x7fELF");
        image[4] = 2; // ELFCLASS64
        image[5] = 1; // ELFDATA2LSB
        image[32..40].copy_from_slice(&64u64.to_le_bytes()); // e_phoff
        image[54..56].copy_from_slice(&56u16.to_le_bytes()); // e_phentsize
        image[56..58].copy_from_slice(&2u16.to_le_bytes()); // e_phnum

        for (index, (paddr, memsz)) in
            [(0x0100_0000u64, 0x0020_0000u64), (0x0340_0000, 0x00d0_0000)]
                .into_iter()
                .enumerate()
        {
            let ph = 64 + index * 56;
            image[ph..ph + 4].copy_from_slice(&1u32.to_le_bytes()); // PT_LOAD
            image[ph + 24..ph + 32].copy_from_slice(&paddr.to_le_bytes());
            image[ph + 40..ph + 48].copy_from_slice(&memsz.to_le_bytes());
        }

        let mut f = tempfile::NamedTempFile::new().expect("tempfile");
        f.write_all(&image).expect("write ELF");
        f.flush().expect("flush ELF");

        let got = read_kernel_init_size(f.path()).expect("read ELF load extent");
        assert_eq!(got, 0x0410_0000);
    }

    /// Reading a file shorter than 0x264 bytes (the high end of the
    /// init_size field on x86_64) must surface an error rather than
    /// silently returning 0. Pin the failure shape so a future
    /// "graceful-fallback" refactor that swallows truncated-bzImage
    /// errors can't slip past review.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn read_kernel_init_size_x86_64_short_file_errors() {
        use std::io::Write;

        let mut f = tempfile::NamedTempFile::new().expect("tempfile");
        // Only 0x100 bytes — well short of the 0x264 needed.
        let truncated = vec![0u8; 0x100];
        f.write_all(&truncated).expect("write truncated");
        f.flush().expect("flush");

        let result = read_kernel_init_size(f.path());
        assert!(result.is_err(), "truncated file must fail; got: {result:?}",);
    }

    /// `read_kernel_init_size` on aarch64 reads a raw PE Image: 8 bytes
    /// of `image_size` at file offset 16 (after `code0` at 0 and
    /// `text_offset` at 8). Construct a tempfile that does NOT begin
    /// with the gzip magic (0x1f 0x8b) so the function takes the raw
    /// PE Image branch, then assert the function returns the value at
    /// offset 16 as u64 little-endian. Pins the byte-offset and width
    /// against a future drift in the arm64 image header layout
    /// (Documentation/arch/arm64/booting.rst, struct arm64_image_header
    /// in arch/arm64/include/asm/image.h).
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn read_kernel_init_size_aarch64_reads_offset_16() {
        use std::io::Write;

        let mut f = tempfile::NamedTempFile::new().expect("tempfile");
        // First 16 bytes: code0 + text_offset, neither of which is the
        // gzip magic 0x1f 0x8b. Use a recognizable non-gzip prefix so
        // a wrong-branch read (decompressing as gzip) would error
        // immediately.
        let prefix = [0u8; 16];
        f.write_all(&prefix).expect("write prefix");
        // Distinct value, large enough that a wrong-offset read would
        // yield zero (the surrounding zero pad).
        let image_size: u64 = 0x1234_5678_9abc_def0;
        f.write_all(&image_size.to_le_bytes())
            .expect("write image_size");
        f.flush().expect("flush");

        let got = read_kernel_init_size(f.path()).expect("read image_size");
        assert_eq!(got, image_size);
    }

    /// Reading a file shorter than 24 bytes (offset 16 + 8 bytes of
    /// image_size) on aarch64 must surface an error rather than
    /// silently returning 0. Mirror of the x86_64 short-file test:
    /// pin the failure shape so a future "graceful-fallback" refactor
    /// that swallows truncated-Image errors can't slip past review.
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn read_kernel_init_size_aarch64_short_file_errors() {
        use std::io::Write;

        let mut f = tempfile::NamedTempFile::new().expect("tempfile");
        // Only 8 bytes — well short of the 24 needed (offset 16 + 8).
        // Also avoids the gzip magic so the raw-Image branch fires.
        let truncated = vec![0u8; 8];
        f.write_all(&truncated).expect("write truncated");
        f.flush().expect("flush");

        let result = read_kernel_init_size(f.path());
        assert!(result.is_err(), "truncated file must fail; got: {result:?}",);
    }

    /// `read_kernel_version` on x86_64 walks the bzImage
    /// setup_header: the "HdrS" magic at 0x202, the kernel_version u16
    /// at 0x20E, and the version string at 0x200 + ptr. Pins the
    /// offset-chase AND the HdrS-magic gate (a non-bzImage returns None
    /// — the safe-50% direction, the panic-direction guard).
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn read_kernel_version_x86_64_offset_chase_and_magic_gate() {
        use std::io::{Seek, SeekFrom, Write};

        let ver_ptr: u16 = 0x0100; // string at file offset 0x200 + 0x100 = 0x300
        let string_off = 0x200u64 + ver_ptr as u64;

        let write_image = |magic: &[u8; 4]| {
            let mut f = tempfile::NamedTempFile::new().expect("tempfile");
            // Pad past the version-string region.
            f.write_all(&vec![0u8; (string_off as usize) + 64])
                .expect("pad");
            // setup_header "HdrS" magic at 0x202.
            f.seek(SeekFrom::Start(0x202)).expect("seek magic");
            f.write_all(magic).expect("write magic");
            // kernel_version pointer at 0x20E.
            f.seek(SeekFrom::Start(0x20E)).expect("seek ver_ptr");
            f.write_all(&ver_ptr.to_le_bytes()).expect("write ver_ptr");
            // Version string at 0x200 + ptr: "RELEASE (builder@host)" + NUL,
            // as the kernel writes it (a space bounds the RELEASE token).
            f.seek(SeekFrom::Start(string_off)).expect("seek string");
            f.write_all(b"6.18.0-rc1 (builder@host)\0")
                .expect("write string");
            f.flush().expect("flush");
            f
        };

        // Valid bzImage: HdrS magic present -> the version-chase parses
        // (6, 18, 0) from the "6.18.0-rc1" release string.
        let good = write_image(b"HdrS");
        assert_eq!(
            read_kernel_version(good.path()),
            Some((6, 18, 0)),
            "valid bzImage offset-chase must parse (6, 18, 0)",
        );

        // Wrong HdrS magic (not a bzImage / corrupt) -> None, even though
        // the 0x20E/0x300 bytes would parse as 6.18. The panic-direction
        // guard: an unvalidated image must NOT select the 90% fraction.
        let bad = write_image(b"XXXX");
        assert_eq!(
            read_kernel_version(bad.path()),
            None,
            "wrong HdrS magic must return None (the safe-50% direction)",
        );
    }

    /// `read_kernel_version_from_metadata_sidecar` reads `version` from a
    /// `metadata.json` sibling of the image path and parses its
    /// MAJOR.MINOR.PATCH — the aarch64 reclaim source. Pins: a 6.18.2
    /// version parses (and unrelated schema keys are ignored); an absent
    /// `version` key, an explicit `null`, an unparsable string, malformed
    /// JSON, and a missing sidecar all return None — the safe-50%
    /// direction.
    #[test]
    fn read_kernel_version_from_metadata_sidecar_parses_and_guards() {
        use std::io::Write;

        let dir = tempfile::tempdir().expect("tempdir");
        // The reader only needs the image path's parent; the image file
        // itself is never read, so it need not exist.
        let image = dir.path().join("Image");
        let sidecar = dir.path().join("metadata.json");
        let write_sidecar = |json: &str| {
            let mut f = std::fs::File::create(&sidecar).expect("create sidecar");
            f.write_all(json.as_bytes()).expect("write sidecar");
            f.flush().expect("flush");
        };

        // Honoring version present, with an unrelated key -> parses
        // (6, 18, 2) (extra schema fields ignored by the minimal probe).
        write_sidecar(r#"{"version":"6.18.2","arch":"aarch64"}"#);
        assert_eq!(
            read_kernel_version_from_metadata_sidecar(&image),
            Some((6, 18, 2)),
            "sidecar version must parse to (6, 18, 2)",
        );

        // version key absent -> None (an acquisition that recorded no
        // version, e.g. a source build whose Makefile was unparsable).
        write_sidecar(r#"{"arch":"aarch64"}"#);
        assert_eq!(
            read_kernel_version_from_metadata_sidecar(&image),
            None,
            "absent version key must return None",
        );

        // Explicit null version -> None.
        write_sidecar(r#"{"version":null}"#);
        assert_eq!(read_kernel_version_from_metadata_sidecar(&image), None);

        // Unparsable version string -> None.
        write_sidecar(r#"{"version":"not-a-version"}"#);
        assert_eq!(read_kernel_version_from_metadata_sidecar(&image), None);

        // Malformed JSON -> None.
        write_sidecar("{not json");
        assert_eq!(read_kernel_version_from_metadata_sidecar(&image), None);

        // No sidecar at all (raw --kernel path) -> None.
        std::fs::remove_file(&sidecar).expect("remove sidecar");
        assert_eq!(
            read_kernel_version_from_metadata_sidecar(&image),
            None,
            "missing sidecar must return None (raw --kernel path)",
        );

        // A path with no parent (root "/") exercises the
        // `kernel_path.parent()?` guard: it early-returns None before any
        // file read, deterministically (no cwd dependency).
        assert_eq!(
            read_kernel_version_from_metadata_sidecar(std::path::Path::new("/")),
            None,
            "root path (no parent) must return None",
        );
    }
}
