use std::io::Seek;

use anyhow::{Context, Result};
use kvm_bindings::{KVM_REG_ARM_CORE, KVM_REG_ARM64, KVM_REG_SIZE_U64};
use kvm_ioctls::VcpuFd;
use vm_memory::{Address, GuestAddress, GuestMemoryMmap};

use crate::vmm::kvm::{CMDLINE_MAX, KERNEL_LOAD_ADDR};

/// Result of loading a kernel image.
pub struct KernelLoadResult {
    /// Entry point address (kernel_load from PE loader).
    pub entry: u64,
}

/// Gzip magic bytes.
const GZIP_MAGIC: [u8; 2] = [0x1f, 0x8b];

/// Load an aarch64 kernel into guest memory.
///
/// Accepts both raw PE Image files and gzip-compressed vmlinuz files.
/// Compressed kernels (identified by gzip magic `1f 8b`) are decompressed
/// in memory before loading via the PE loader.
pub fn load_kernel(
    guest_mem: &GuestMemoryMmap,
    kernel_path: &std::path::Path,
) -> Result<KernelLoadResult> {
    use linux_loader::loader::{KernelLoader, pe::PE};
    use std::fs::File;
    use std::io::Read;

    let mut kernel_file = File::open(kernel_path)
        .with_context(|| format!("open kernel: {}", kernel_path.display()))?;

    // Read the first 2 bytes to detect gzip compression.
    let mut magic = [0u8; 2];
    kernel_file
        .read_exact(&mut magic)
        .context("read kernel magic")?;
    kernel_file
        .seek(std::io::SeekFrom::Start(0))
        .context("seek kernel to start")?;

    if magic == GZIP_MAGIC {
        // Content-cache the decompressed Image next to the cached kernel as a
        // `<kernel>.decompressed` sidecar. nextest runs each cell in its own
        // process, so without this every cell re-inflates the same
        // multi-MB vmlinuz. Canonicalize first (symlink-safe, mirroring the
        // `.btf` sidecar's canonicalize-at-top) so the sidecar path and its
        // mtime freshness track the real cached file; a failure leaves us on
        // the plain-path fallback, which the cache-membership gate below
        // rejects anyway.
        let canon =
            std::fs::canonicalize(kernel_path).unwrap_or_else(|_| kernel_path.to_path_buf());
        // Gate reads and writes on cache-root membership so ktstr never
        // deposits a sibling artifact in a directory it does not own (a source
        // tree, a distro path) — identical policy to the vmlinux sidecars.
        let cache_ok = crate::cache::path_inside_cache_root(&canon);
        let sidecar = decompressed_sidecar_path(&canon);

        // HIT: a fresh sidecar inside the cache. mmap it read-only — the point
        // is page-cache sharing across concurrent cells loading the same
        // kernel, and PE::load reads volatile straight into guest memory with
        // no heap copy of the whole image. No version tag: the sidecar is the
        // RAW decompressed Image (not a versioned struct like `.artifacts`),
        // so the mtime freshness rule alone suffices — the same reasoning the
        // raw-bytes `.btf` sidecar relies on.
        if cache_ok && crate::monitor::btf_offsets::sidecar_fresh(&sidecar, &canon) {
            match load_pe_from_sidecar(guest_mem, &sidecar) {
                Ok(entry) => return Ok(KernelLoadResult { entry }),
                Err(e) => {
                    // Defensive: a fresh-looking sidecar that fails PE::load is
                    // truncated/corrupt. Remove it and fall through to inflate;
                    // the miss path below rewrites a good one.
                    tracing::warn!(
                        path = %sidecar.display(),
                        err = %e,
                        "decompressed-Image sidecar failed PE::load; removing and re-inflating",
                    );
                    let _ = std::fs::remove_file(&sidecar);
                }
            }
        }

        // MISS (or the defensive fallback above): inflate as before.
        let mut decoder = flate2::read::GzDecoder::new(kernel_file);
        let mut decompressed = Vec::new();
        decoder
            .read_to_end(&mut decompressed)
            .context("decompress gzip kernel")?;
        // Borrow the bytes for PE::load (Cursor<&[u8]> is ReadVolatile) so the
        // Vec is still owned afterwards for the sidecar write.
        let mut cursor = std::io::Cursor::new(&decompressed[..]);
        let result = PE::load(
            guest_mem,
            Some(GuestAddress(KERNEL_LOAD_ADDR)),
            &mut cursor,
            None,
        )
        .context("load decompressed aarch64 Image")?;
        // PE::load accepted the bytes -> cache them for sibling cells. Written
        // only after validation so a corrupt inflate is never persisted;
        // best-effort (the load already succeeded) and suppressed outside the
        // cache root.
        if cache_ok && let Err(e) = write_decompressed_sidecar(&sidecar, &decompressed) {
            tracing::warn!(
                path = %sidecar.display(),
                err = %e,
                "decompressed-Image sidecar write failed; re-inflated on next load",
            );
        }
        Ok(KernelLoadResult {
            entry: result.kernel_load.raw_value(),
        })
    } else {
        let result = PE::load(
            guest_mem,
            Some(GuestAddress(KERNEL_LOAD_ADDR)),
            &mut kernel_file,
            None,
        )
        .context("load aarch64 Image")?;
        Ok(KernelLoadResult {
            entry: result.kernel_load.raw_value(),
        })
    }
}

/// Sidecar path for a cached kernel image: append `.decompressed` so it sits
/// next to the kernel in the same cache entry (`<entry>/vmlinuz` →
/// `<entry>/vmlinuz.decompressed`). Append-suffix (not `with_extension`)
/// preserves any existing extension, matching `btf_sidecar_path`.
fn decompressed_sidecar_path(path: &std::path::Path) -> std::path::PathBuf {
    let mut name = path.as_os_str().to_os_string();
    name.push(".decompressed");
    std::path::PathBuf::from(name)
}

/// Load a raw (already-decompressed) arm64 Image from the sidecar into guest
/// memory via a read-only mmap, returning the PE loader's entry address.
///
/// The mmap (not `fs::read`) is deliberate: concurrent cells loading the same
/// cached kernel share the sidecar's page cache, and `PE::load` reads volatile
/// straight into guest memory without a heap copy of the whole image.
fn load_pe_from_sidecar(
    guest_mem: &GuestMemoryMmap,
    sidecar: &std::path::Path,
) -> Result<u64> {
    use linux_loader::loader::{KernelLoader, pe::PE};
    let file = std::fs::File::open(sidecar)
        .with_context(|| format!("open decompressed sidecar: {}", sidecar.display()))?;
    // SAFETY: the sidecar is only ever replaced atomically (tempfile +
    // rename in `write_decompressed_sidecar`), never modified in place, so the
    // mapped pages are stable for this map's lifetime even if a concurrent
    // cell rewrites the sidecar (which swaps the inode, leaving our pages).
    let mmap = unsafe { memmap2::Mmap::map(&file) }
        .with_context(|| format!("mmap decompressed sidecar: {}", sidecar.display()))?;
    let mut cursor = std::io::Cursor::new(mmap);
    let result = PE::load(guest_mem, Some(GuestAddress(KERNEL_LOAD_ADDR)), &mut cursor, None)
        .context("load decompressed Image from sidecar")?;
    Ok(result.kernel_load.raw_value())
}

/// Atomically write `bytes` to the `.decompressed` sidecar via a tempfile in
/// the same directory + fsync + rename, so a concurrent reader sees either the
/// old sidecar or the new one, never a partial write. Mirrors the vmlinux
/// `atomic_write_sidecar`.
fn write_decompressed_sidecar(sidecar: &std::path::Path, bytes: &[u8]) -> Result<()> {
    use std::io::Write;
    let parent = sidecar
        .parent()
        .context("decompressed sidecar path has no parent directory")?;
    let mut tmp = tempfile::NamedTempFile::new_in(parent)
        .context("create tempfile for decompressed sidecar")?;
    tmp.write_all(bytes)
        .context("write decompressed sidecar contents")?;
    tmp.as_file()
        .sync_all()
        .context("fsync decompressed sidecar before rename")?;
    tmp.persist(sidecar)
        .map_err(|e| anyhow::anyhow!("persist decompressed sidecar: {}", e.error))?;
    Ok(())
}

/// Validate that a kernel command line fits within the maximum length.
///
/// On aarch64 the kernel reads the command line from the FDT /chosen
/// node's bootargs property. No separate memory write is needed.
pub fn validate_cmdline(cmdline: &str) -> Result<()> {
    anyhow::ensure!(
        cmdline.len() < CMDLINE_MAX,
        "cmdline too long ({} > {})",
        cmdline.len(),
        CMDLINE_MAX
    );
    Ok(())
}

/// KVM register IDs for aarch64 core registers.
///
/// The encoding follows the KVM_REG_ARM_CORE format:
///   KVM_REG_ARM64 | KVM_REG_SIZE_U64 | KVM_REG_ARM_CORE | (offset / 4)
///
/// The offset is into struct kvm_regs (user_pt_regs) defined in
/// arch/arm64/include/uapi/asm/kvm.h.
const REG_CORE_BASE: u64 = KVM_REG_ARM64 | KVM_REG_SIZE_U64 | KVM_REG_ARM_CORE as u64;

/// Register ID for x0 (`regs.regs[0]`).
const REG_X0: u64 = REG_CORE_BASE;

/// Register ID for PC (regs.pc, offset = 256/4 = 64 in u32 units).
/// In user_pt_regs: regs[0..31] at offsets 0..248, sp at 248, pc at
/// 256, pstate at 264. 256 bytes / 4 = 64 u32-offset.
const REG_PC: u64 = REG_CORE_BASE | (256 / 4);

/// Register ID for pstate (regs.pstate, offset = 264/4 = 66).
const REG_PSTATE: u64 = REG_CORE_BASE | (264 / 4);

/// PSR mode bits for EL1h (EL1, SP_EL1).
const PSTATE_MODE_EL1H: u64 = 0x5;

/// PSR D/A/I/F mask bits — mask debug, SError, IRQ, FIQ exceptions.
const PSTATE_DAIF_MASK: u64 = 0x3C0;

/// Set up vCPU registers for the BSP.
///
/// Per the arm64 boot protocol (Documentation/arch/arm64/booting.rst):
/// - x0 = physical address of the FDT
/// - PC = kernel entry point
/// - pstate = EL1h with DAIF masked
/// - x1 = x2 = x3 = 0 (reserved for future use); satisfied because
///   KVM_ARM_VCPU_INIT zero-initializes the GP registers and we set
///   only PC/x0/pstate below. All remaining GP registers are undefined.
pub fn setup_regs(vcpu: &VcpuFd, entry: u64, fdt_addr: u64) -> Result<()> {
    // Set PC to kernel entry point.
    vcpu.set_one_reg(REG_PC, &entry.to_le_bytes())
        .context("set PC")?;

    // Set x0 to FDT address.
    vcpu.set_one_reg(REG_X0, &fdt_addr.to_le_bytes())
        .context("set x0 (FDT address)")?;

    // Set pstate to EL1h with all exceptions masked.
    let pstate: u64 = PSTATE_MODE_EL1H | PSTATE_DAIF_MASK;
    vcpu.set_one_reg(REG_PSTATE, &pstate.to_le_bytes())
        .context("set pstate")?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimal 64-byte arm64 Image the PE loader accepts
    /// (Documentation/arch/arm64/booting.rst): the "ARM\x64" magic
    /// (0x644d_5241) at byte offset 56, plus a nonzero `image_size` at offset
    /// 16 so `PE::load` honors `text_offset` (offset 8), which we leave 0 so
    /// the image loads at `KERNEL_LOAD_ADDR` exactly (keeps the test guest
    /// memory a single page).
    fn minimal_arm64_image() -> Vec<u8> {
        let mut img = vec![0u8; 64];
        img[16..24].copy_from_slice(&0x1_0000u64.to_le_bytes()); // image_size
        img[56..60].copy_from_slice(&0x644d_5241u32.to_le_bytes()); // magic
        img
    }

    fn gzip(bytes: &[u8]) -> Vec<u8> {
        use flate2::{Compression, write::GzEncoder};
        use std::io::Write;
        let mut e = GzEncoder::new(Vec::new(), Compression::fast());
        e.write_all(bytes).unwrap();
        e.finish().unwrap()
    }

    /// text_offset=0 loads at `KERNEL_LOAD_ADDR` (2 MiB-aligned: it is
    /// DRAM_START = 1 GiB), so a single guest page holds the 64-byte image.
    fn tiny_guest_mem() -> GuestMemoryMmap {
        GuestMemoryMmap::from_ranges(&[(GuestAddress(KERNEL_LOAD_ADDR), 0x1000)]).unwrap()
    }

    // The four W2c tests exercise the aarch64 gzip-kernel sidecar cache. They
    // compile and run only on aarch64 (this whole module is
    // `#[cfg(target_arch = "aarch64")]` via `vmm/mod.rs`), i.e. on arm64 CI.

    /// MISS: no sidecar -> inflate, load, and write the raw Image sidecar.
    #[test]
    fn gzip_miss_inflates_and_writes_sidecar() {
        use crate::test_support::test_helpers::{EnvVarGuard, lock_env};
        let _lock = lock_env();
        let tmp = tempfile::TempDir::new().unwrap();
        let _g = EnvVarGuard::set(crate::KTSTR_CACHE_DIR_ENV, tmp.path());
        let entry = tmp.path().join("kentry");
        std::fs::create_dir_all(&entry).unwrap();
        let kernel = entry.join("Image");
        let raw = minimal_arm64_image();
        std::fs::write(&kernel, gzip(&raw)).unwrap();

        let gm = tiny_guest_mem();
        let r = load_kernel(&gm, &kernel).expect("first load must inflate + succeed");
        assert_eq!(r.entry, KERNEL_LOAD_ADDR);

        let canon = std::fs::canonicalize(&kernel).unwrap();
        let sidecar = decompressed_sidecar_path(&canon);
        assert!(sidecar.exists(), "miss must write the decompressed sidecar");
        assert_eq!(
            std::fs::read(&sidecar).unwrap(),
            raw,
            "sidecar must hold the raw decompressed Image bytes",
        );
    }

    /// HIT: a fresh in-cache sidecar is used even when the ORIGINAL gzip body
    /// is corrupt. The original keeps the gzip magic (so the gzip branch is
    /// taken) but a garbage body that would fail to inflate — so a successful
    /// load proves the bytes came from the sidecar, not a re-inflate.
    #[test]
    fn gzip_hit_loads_from_sidecar_ignoring_original() {
        use crate::test_support::test_helpers::{EnvVarGuard, lock_env};
        let _lock = lock_env();
        let tmp = tempfile::TempDir::new().unwrap();
        let _g = EnvVarGuard::set(crate::KTSTR_CACHE_DIR_ENV, tmp.path());
        let entry = tmp.path().join("kentry");
        std::fs::create_dir_all(&entry).unwrap();
        let kernel = entry.join("Image");
        std::fs::write(&kernel, [0x1f, 0x8b, 0xde, 0xad, 0xbe, 0xef]).unwrap();
        let canon = std::fs::canonicalize(&kernel).unwrap();
        let sidecar = decompressed_sidecar_path(&canon);
        // Written after the original, so its mtime is >= the original's and
        // `sidecar_fresh` (a `>=` compare) treats it as a hit.
        std::fs::write(&sidecar, minimal_arm64_image()).unwrap();
        assert!(
            crate::monitor::btf_offsets::sidecar_fresh(&sidecar, &canon),
            "precondition: sidecar must be fresh vs the original",
        );

        let gm = tiny_guest_mem();
        let r = load_kernel(&gm, &kernel).expect("hit must load from the sidecar");
        assert_eq!(r.entry, KERNEL_LOAD_ADDR);
    }

    /// STALE: a sidecar older than its kernel is ignored; the kernel is
    /// re-inflated and the sidecar rewritten fresh. The planted sidecar is
    /// garbage, so if it were (wrongly) used PE::load would fail.
    #[test]
    fn gzip_stale_sidecar_ignored_and_rewritten() {
        use crate::test_support::test_helpers::{EnvVarGuard, lock_env};
        let _lock = lock_env();
        let tmp = tempfile::TempDir::new().unwrap();
        let _g = EnvVarGuard::set(crate::KTSTR_CACHE_DIR_ENV, tmp.path());
        let entry = tmp.path().join("kentry");
        std::fs::create_dir_all(&entry).unwrap();
        let kernel = entry.join("Image");
        let raw = minimal_arm64_image();
        std::fs::write(&kernel, gzip(&raw)).unwrap();
        let canon = std::fs::canonicalize(&kernel).unwrap();
        let sidecar = decompressed_sidecar_path(&canon);
        std::fs::write(&sidecar, b"garbage-not-an-image").unwrap();
        let past = std::time::SystemTime::now() - std::time::Duration::from_secs(3600);
        let f = std::fs::File::options().write(true).open(&sidecar).unwrap();
        f.set_modified(past).unwrap();
        drop(f);
        assert!(
            !crate::monitor::btf_offsets::sidecar_fresh(&sidecar, &canon),
            "precondition: planted sidecar must be stale",
        );

        let gm = tiny_guest_mem();
        let r = load_kernel(&gm, &kernel).expect("stale sidecar ignored; inflate succeeds");
        assert_eq!(r.entry, KERNEL_LOAD_ADDR);
        assert_eq!(
            std::fs::read(&sidecar).unwrap(),
            raw,
            "stale sidecar must be rewritten with the real Image",
        );
    }

    /// A kernel OUTSIDE the cache root gets no sidecar (ktstr never deposits
    /// artifacts in directories it does not own).
    #[test]
    fn gzip_outside_cache_root_writes_no_sidecar() {
        use crate::test_support::test_helpers::{EnvVarGuard, lock_env};
        let _lock = lock_env();
        let cache = tempfile::TempDir::new().unwrap();
        let _g = EnvVarGuard::set(crate::KTSTR_CACHE_DIR_ENV, cache.path());
        let outside = tempfile::TempDir::new().unwrap();
        let kernel = outside.path().join("Image");
        std::fs::write(&kernel, gzip(&minimal_arm64_image())).unwrap();

        let gm = tiny_guest_mem();
        let r = load_kernel(&gm, &kernel).expect("outside-cache load still succeeds");
        assert_eq!(r.entry, KERNEL_LOAD_ADDR);
        let sidecar = decompressed_sidecar_path(&std::fs::canonicalize(&kernel).unwrap());
        assert!(
            !sidecar.exists(),
            "no sidecar may be written outside the cache root",
        );
    }

    #[test]
    fn kernel_load_result_fields() {
        let r = KernelLoadResult { entry: 0x28_0000 };
        assert_eq!(r.entry, 0x28_0000);
    }

    #[test]
    fn write_cmdline_basic() {
        validate_cmdline("console=ttyAMA0").unwrap();
    }

    #[test]
    fn write_cmdline_too_long() {
        let long = "x".repeat(CMDLINE_MAX + 1);
        assert!(validate_cmdline(&long).is_err());
    }

    #[test]
    fn reg_ids_follow_encoding() {
        // Verify register ID encoding matches the KVM ABI.
        // x0 is at offset 0 in user_pt_regs.
        assert_eq!(REG_X0 & 0xFFFF, 0);
        // PC is at byte offset 256 -> u32 offset 64.
        assert_eq!(REG_PC & 0xFFFF, 64);
        // pstate is at byte offset 264 -> u32 offset 66.
        assert_eq!(REG_PSTATE & 0xFFFF, 66);
    }

    #[test]
    fn pstate_el1h_value() {
        let pstate = PSTATE_MODE_EL1H | PSTATE_DAIF_MASK;
        // EL1h = 0x5, DAIF = 0x3C0 -> 0x3C5
        assert_eq!(pstate, 0x3C5);
    }

    #[test]
    fn setup_regs_on_real_vcpu() {
        use crate::vmm::topology::Topology;
        let topo = Topology {
            llcs: 1,
            cores_per_llc: 1,
            threads_per_core: 1,
            numa_nodes: 1,
            nodes: None,
            distances: None,
        };
        let vm = crate::vmm::kvm::KtstrKvm::new(topo, 64, false).unwrap();
        let result = setup_regs(&vm.vcpus[0], 0x28_0000, 0x4000_0000);
        assert!(result.is_ok(), "setup_regs failed: {:?}", result.err());

        // Verify registers were set.
        let mut pc_buf = [0u8; 8];
        vm.vcpus[0].get_one_reg(REG_PC, &mut pc_buf).unwrap();
        assert_eq!(u64::from_le_bytes(pc_buf), 0x28_0000);

        let mut x0_buf = [0u8; 8];
        vm.vcpus[0].get_one_reg(REG_X0, &mut x0_buf).unwrap();
        assert_eq!(u64::from_le_bytes(x0_buf), 0x4000_0000);

        let mut pstate_buf = [0u8; 8];
        vm.vcpus[0]
            .get_one_reg(REG_PSTATE, &mut pstate_buf)
            .unwrap();
        assert_eq!(u64::from_le_bytes(pstate_buf), 0x3C5);
    }
}
