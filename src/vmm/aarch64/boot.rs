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

/// Machine-wide derivation cache for gzip-compressed arm64 kernels.
///
/// The directory version is the recipe version: changing the decompressor or
/// the bytes persisted as an object must move to a new namespace.
const DECOMPRESSED_CACHE_DIR: &str = "arm64-kernel-decompressed-v1";
const DECOMPRESSED_OBJECTS_DIR: &str = "objects";
const DECOMPRESSED_LOCKS_DIR: &str = ".locks";
const DECOMPRESSED_NAMESPACE_GATE: &str = "namespace.lock";

struct DecompressedCachePaths {
    object: std::path::PathBuf,
    lock: std::path::PathBuf,
    namespace_gate: std::path::PathBuf,
}

/// Load an aarch64 kernel into guest memory.
///
/// Accepts both raw PE Image files and gzip-compressed vmlinuz files.
/// Compressed kernels (identified by gzip magic `1f 8b`) are decompressed
/// once per compressed-content hash across processes, then loaded via the PE
/// loader.
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
        let source_identity = crate::cache::content::StableFileIdentity::from_file(&kernel_file)
            .with_context(|| format!("stat gzip kernel: {}", kernel_path.display()))?;
        let content_hash =
            crate::cache::content::cached_file_digest(&kernel_file, source_identity)
                .with_context(|| format!("digest gzip kernel: {}", kernel_path.display()))?;
        let paths = decompressed_cache_paths(content_hash)?;

        anyhow::ensure!(
            crate::cache::content::StableFileIdentity::from_file(&kernel_file)?
                == source_identity,
            "gzip kernel changed before decompressed-cache lookup: {}",
            kernel_path.display()
        );
        let entry = crate::cache::content::load_or_build(
            &paths.namespace_gate,
            &paths.lock,
            &format!("decompressed arm64 kernel {content_hash:016x}"),
            || try_load_decompressed_object(guest_mem, &paths.object),
            || {
                build_decompressed_object(
                    guest_mem,
                    &mut kernel_file,
                    source_identity,
                    content_hash,
                    &paths.object,
                )
            },
        )?;
        anyhow::ensure!(
            crate::cache::content::StableFileIdentity::from_file(&kernel_file)?
                == source_identity,
            "gzip kernel changed during decompressed-cache lookup: {}",
            kernel_path.display()
        );
        Ok(KernelLoadResult { entry })
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

fn decompressed_cache_paths(content_hash: u64) -> Result<DecompressedCachePaths> {
    // `KTSTR_CACHE_DIR` intentionally resolves to the override verbatim, so
    // retain a component-specific directory below it. This also keeps the
    // derived-object locks separate from the shared input-digest namespace.
    let root = crate::cache::resolve_cache_root_with_suffix("vmm-derived")?
        .join(DECOMPRESSED_CACHE_DIR);
    let objects = root.join(DECOMPRESSED_OBJECTS_DIR);
    let locks = root.join(DECOMPRESSED_LOCKS_DIR);
    std::fs::create_dir_all(&objects)
        .with_context(|| format!("create decompressed-kernel CAS {}", objects.display()))?;
    std::fs::create_dir_all(&locks)
        .with_context(|| format!("create decompressed-kernel lock dir {}", locks.display()))?;
    Ok(DecompressedCachePaths {
        object: objects.join(format!("{content_hash:016x}.image")),
        lock: locks.join(format!("object-{content_hash:016x}.lock")),
        namespace_gate: locks.join(DECOMPRESSED_NAMESPACE_GATE),
    })
}

/// Load a raw arm64 Image from one pinned immutable CAS inode.
fn load_pe_from_object(
    guest_mem: &GuestMemoryMmap,
    file: std::fs::File,
    object: &std::path::Path,
) -> Result<u64> {
    use linux_loader::loader::{KernelLoader, pe::PE};
    // SAFETY: published objects are read-only and replaced only by atomic
    // rename during corruption recovery. The open inode remains stable even
    // if another process replaces its pathname.
    let mmap = unsafe { memmap2::Mmap::map(&file) }
        .with_context(|| format!("mmap decompressed kernel object: {}", object.display()))?;
    let mut cursor = std::io::Cursor::new(mmap);
    let result = PE::load(
        guest_mem,
        Some(GuestAddress(KERNEL_LOAD_ADDR)),
        &mut cursor,
        None,
    )
    .context("load decompressed Image from content cache")?;
    Ok(result.kernel_load.raw_value())
}

fn remove_decompressed_object(object: &std::path::Path) -> Result<()> {
    match std::fs::remove_file(object) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error)
            .with_context(|| format!("remove decompressed kernel object {}", object.display())),
    }
}

/// Return a checked cache hit. A malformed or writable object is invalidated
/// and becomes a coordinated miss; no caller privately inflates around a cache
/// failure.
fn try_load_decompressed_object(
    guest_mem: &GuestMemoryMmap,
    object: &std::path::Path,
) -> Result<Option<u64>> {
    use std::os::unix::fs::PermissionsExt as _;

    let Some((file, identity)) =
        crate::cache::content::open_cache_record(object, "decompressed kernel object")?
    else {
        return Ok(None);
    };
    let mode = file
        .metadata()
        .with_context(|| format!("stat decompressed kernel object {}", object.display()))?
        .permissions()
        .mode();
    if identity.size == 0 || mode & 0o222 != 0 {
        drop(file);
        remove_decompressed_object(object)?;
        return Ok(None);
    }
    match load_pe_from_object(guest_mem, file, object) {
        Ok(entry) => Ok(Some(entry)),
        Err(error) => {
            tracing::warn!(
                path = %object.display(),
                %error,
                "decompressed kernel cache object failed PE validation; rebuilding",
            );
            remove_decompressed_object(object)?;
            Ok(None)
        }
    }
}

/// Atomically publish bytes already accepted by `PE::load`.
fn publish_decompressed_object(object: &std::path::Path, bytes: &[u8]) -> Result<()> {
    use std::io::Write as _;
    use std::os::unix::fs::PermissionsExt as _;

    let parent = object
        .parent()
        .context("decompressed kernel object has no parent directory")?;
    let mut tmp = tempfile::Builder::new()
        .prefix(".tmp-decompressed-kernel-")
        .tempfile_in(parent)
        .context("create decompressed kernel object temp")?;
    tmp.write_all(bytes)
        .context("write decompressed kernel object")?;
    tmp.as_file()
        .set_permissions(std::fs::Permissions::from_mode(0o444))
        .context("mark decompressed kernel object read-only")?;
    tmp.persist(object)
        .map_err(|error| error.error)
        .with_context(|| format!("publish decompressed kernel object {}", object.display()))?;
    Ok(())
}

fn build_decompressed_object(
    guest_mem: &GuestMemoryMmap,
    source: &mut std::fs::File,
    source_identity: crate::cache::content::StableFileIdentity,
    _content_hash: u64,
    object: &std::path::Path,
) -> Result<u64> {
    use linux_loader::loader::{KernelLoader, pe::PE};
    use std::io::Read as _;

    anyhow::ensure!(
        crate::cache::content::StableFileIdentity::from_file(source)? == source_identity,
        "gzip kernel changed before decompression"
    );
    source
        .seek(std::io::SeekFrom::Start(0))
        .context("seek gzip kernel for decompression")?;
    let mut decompressed = Vec::new();
    {
        let mut decoder = flate2::read::GzDecoder::new(&mut *source);
        decoder
            .read_to_end(&mut decompressed)
            .context("decompress gzip kernel")?;
    }
    anyhow::ensure!(
        crate::cache::content::StableFileIdentity::from_file(source)? == source_identity,
        "gzip kernel changed during decompression"
    );

    // Validate before publication, preserving the old sidecar invariant that a
    // bad gzip payload never becomes a reusable cache object.
    let mut cursor = std::io::Cursor::new(&decompressed[..]);
    let result = PE::load(
        guest_mem,
        Some(GuestAddress(KERNEL_LOAD_ADDR)),
        &mut cursor,
        None,
    )
    .context("load decompressed aarch64 Image")?;

    #[cfg(test)]
    record_decompressed_builder_claim_for_test(_content_hash)?;
    publish_decompressed_object(object, &decompressed)?;
    Ok(result.kernel_load.raw_value())
}

#[cfg(test)]
fn record_decompressed_builder_claim_for_test(content_hash: u64) -> Result<()> {
    use std::io::Write as _;

    let Some(path) = std::env::var_os("KTSTR_ARM64_DECOMPRESSED_BUILDER_CLAIMS") else {
        return Ok(());
    };
    let mut claims = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
        .with_context(|| {
            format!(
                "open decompressed-kernel builder claims {}",
                std::path::Path::new(&path).display()
            )
        })?;
    writeln!(claims, "{content_hash:016x} {}", std::process::id())
        .context("record decompressed-kernel builder claim")?;
    claims
        .sync_data()
        .context("sync decompressed-kernel builder claim")?;
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

    // These tests compile and run only on aarch64 (this whole module is
    // `#[cfg(target_arch = "aarch64")]` via `vmm/mod.rs`), i.e. on arm64 CI.

    fn object_path_for(kernel: &std::path::Path) -> std::path::PathBuf {
        let file = std::fs::File::open(kernel).unwrap();
        let identity = crate::cache::content::StableFileIdentity::from_file(&file).unwrap();
        let hash = crate::cache::content::cached_file_digest(&file, identity).unwrap();
        decompressed_cache_paths(hash).unwrap().object
    }

    #[test]
    fn gzip_content_aliases_share_one_machine_object() {
        use crate::test_support::test_helpers::{EnvVarGuard, lock_env};
        use std::os::unix::fs::PermissionsExt as _;

        let _lock = lock_env();
        let cache = tempfile::TempDir::new().unwrap();
        let _g = EnvVarGuard::set(crate::KTSTR_CACHE_DIR_ENV, cache.path());
        let sources = tempfile::TempDir::new().unwrap();
        let first = sources.path().join("first-vmlinuz");
        let alias = sources.path().join("byte-identical-alias-vmlinuz");
        let raw = minimal_arm64_image();
        let compressed = gzip(&raw);
        std::fs::write(&first, &compressed).unwrap();
        // A distinct inode with identical bytes proves the derived key is
        // content-addressed rather than path/inode-addressed.
        std::fs::write(&alias, &compressed).unwrap();

        let first_object = object_path_for(&first);
        let alias_object = object_path_for(&alias);
        assert_eq!(first_object, alias_object);
        assert!(!first_object.exists());

        for kernel in [&first, &alias] {
            let result = load_kernel(&tiny_guest_mem(), kernel).unwrap();
            assert_eq!(result.entry, KERNEL_LOAD_ADDR);
        }
        assert_eq!(std::fs::read(&first_object).unwrap(), raw);
        assert_eq!(
            std::fs::metadata(&first_object)
                .unwrap()
                .permissions()
                .mode()
                & 0o222,
            0,
            "published decompressed objects must be immutable",
        );
        let mut old_sidecar = first.as_os_str().to_os_string();
        old_sidecar.push(".decompressed");
        assert!(
            !std::path::Path::new(&old_sidecar).exists(),
            "content-CAS loading must not recreate pathname sidecars",
        );
    }

    #[test]
    fn gzip_corrupt_content_object_is_checked_and_rebuilt() {
        use crate::test_support::test_helpers::{EnvVarGuard, lock_env};
        use std::os::unix::fs::PermissionsExt as _;

        let _lock = lock_env();
        let cache = tempfile::TempDir::new().unwrap();
        let _g = EnvVarGuard::set(crate::KTSTR_CACHE_DIR_ENV, cache.path());
        let sources = tempfile::TempDir::new().unwrap();
        let kernel = sources.path().join("vmlinuz");
        let raw = minimal_arm64_image();
        std::fs::write(&kernel, gzip(&raw)).unwrap();
        let object = object_path_for(&kernel);
        std::fs::write(&object, b"not-an-arm64-Image").unwrap();
        std::fs::set_permissions(&object, std::fs::Permissions::from_mode(0o444)).unwrap();

        let result = load_kernel(&tiny_guest_mem(), &kernel).unwrap();
        assert_eq!(result.entry, KERNEL_LOAD_ADDR);
        assert_eq!(
            std::fs::read(&object).unwrap(),
            raw,
            "a cache object rejected by PE::load must be replaced atomically",
        );
    }

    #[test]
    fn gzip_cold_process_storm_elects_one_inflater() {
        const CHILD_KERNEL: &str = "KTSTR_ARM64_DECOMPRESSED_CHILD_KERNEL";
        const CLAIMS: &str = "KTSTR_ARM64_DECOMPRESSED_BUILDER_CLAIMS";

        if let Some(kernel) = std::env::var_os(CHILD_KERNEL) {
            let result = load_kernel(&tiny_guest_mem(), std::path::Path::new(&kernel)).unwrap();
            assert_eq!(result.entry, KERNEL_LOAD_ADDR);
            return;
        }

        use crate::test_support::test_helpers::{EnvVarGuard, lock_env};
        let _lock = lock_env();
        let cache = tempfile::TempDir::new().unwrap();
        let _g = EnvVarGuard::set(crate::KTSTR_CACHE_DIR_ENV, cache.path());
        let sources = tempfile::TempDir::new().unwrap();
        let kernel = sources.path().join("vmlinuz");
        let raw = minimal_arm64_image();
        std::fs::write(&kernel, gzip(&raw)).unwrap();
        let object = object_path_for(&kernel);
        let claims = sources.path().join("builder-claims");
        let test_name = std::thread::current()
            .name()
            .expect("test harness named this thread")
            .to_owned();

        let mut children = Vec::new();
        for _ in 0..8 {
            children.push(
                std::process::Command::new(std::env::current_exe().unwrap())
                    .arg("--exact")
                    .arg(&test_name)
                    .arg("--nocapture")
                    .env(CHILD_KERNEL, &kernel)
                    .env(CLAIMS, &claims)
                    .stdout(std::process::Stdio::piped())
                    .stderr(std::process::Stdio::piped())
                    .spawn()
                    .unwrap(),
            );
        }
        for child in children {
            let output = child.wait_with_output().unwrap();
            assert!(
                output.status.success(),
                "cold-storm child failed:\nstdout:\n{}\nstderr:\n{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr),
            );
        }

        let claims = std::fs::read_to_string(&claims).unwrap();
        assert_eq!(
            claims.lines().count(),
            1,
            "one content key must elect exactly one cross-process inflater",
        );
        assert_eq!(std::fs::read(object).unwrap(), raw);
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
            llc_cores: None,
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
