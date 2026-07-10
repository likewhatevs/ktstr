//! Load initramfs-embedded kernel modules before first device access.
//!
//! Prebuilt distro kernels (Fedora / Ubuntu / AL2023) ship virtio
//! (blk / console / net / pci) as modules rather than `=y`, so the
//! guest's virtio `/dev` nodes do not exist until the drivers are
//! loaded. The host packs the required `.ko` images into the initramfs
//! (`initramfs::build_suffix`), and this loads them — in the
//! caller-chosen order — right after devtmpfs is mounted and BEFORE
//! init opens `/dev/vport0p1`, `/dev/vda`, or `/dev/hvc0`.
//!
//! Split from rust_init.rs; the shared consts/imports live in the
//! parent module (`super`), reached via the glob below.
use super::*;

use std::ffi::CString;

use nix::kmod::{ModuleInitFlags, finit_module};

/// Directory the host writes initramfs-embedded modules into. Files are
/// named `NNN-<filename>` (zero-padded index) so a lexical sort of the
/// directory reproduces the host's chosen load order (see
/// `initramfs::build_suffix`).
const MODULES_DIR: &str = "/modules";

/// Load every module under [`MODULES_DIR`] via `finit_module(2)`
/// (fd-based, flags 0) in filename order.
///
/// No-op when the directory is absent: the ktstr-built kernels pin
/// virtio `=y` and ship no modules, so `build_suffix` emits no
/// `modules/` entries and this returns immediately — the byte-identical
/// path for the default kernel.
///
/// A module that fails with `EEXIST` is treated as success — the driver
/// is already built in or loaded (a benign race, or a duplicate entry).
/// Any other error is fatal: init cannot reach its console, disk, or the
/// host control port without the virtio drivers, so there is nowhere to
/// degrade to. The failure is reported (naming the module) via
/// [`fatal_module_error`] and the guest reboots.
pub(crate) fn load_kernel_modules() {
    let entries = match fs::read_dir(MODULES_DIR) {
        Ok(e) => e,
        // Absent dir ⇒ no modules shipped (ktstr-built kernel). Any
        // other read error is fatal the same way a load failure is: a
        // present-but-unreadable module dir means the virtio drivers
        // init needs are unreachable.
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return,
        Err(e) => fatal_module_error(&format!("read {MODULES_DIR}: {e}")),
    };

    // Sort by full path: the `NNN-` numeric prefix makes the lexical
    // order match the host's load order.
    let mut modules: Vec<std::path::PathBuf> =
        entries.filter_map(|e| e.ok().map(|e| e.path())).collect();
    modules.sort();

    // The kernel applies `modname.param=value` command-line params only
    // to BUILTIN modules; one loaded via `finit_module` gets only the
    // params passed as its load args. ktstr describes the virtio-MMIO
    // console/block/net devices via `virtio_mmio.device=...@...:...`
    // cmdline tokens (see `vmm::setup`), so for a modular virtio_mmio
    // those tokens must be forwarded as load args or the driver binds no
    // device and `/dev/hvc0` / `/dev/vport0p1` never appear (init then
    // hangs waiting for the control port). Reproduce the builtin
    // behavior: pass each module the matching `modname.*` cmdline params.
    let cmdline = fs::read_to_string("/proc/cmdline").unwrap_or_default();

    for module in &modules {
        let file = match fs::File::open(module) {
            Ok(f) => f,
            Err(e) => fatal_module_error(&format!("open {}: {e}", module.display())),
        };
        let args = module_load_args(module, &cmdline);
        match finit_module(&file, &args, ModuleInitFlags::empty()) {
            Ok(()) => {}
            Err(nix::errno::Errno::EEXIST) => {}
            Err(e) => fatal_module_error(&format!("finit_module {}: {e}", module.display())),
        }
    }
}

/// Build the `finit_module(2)` param string for `module` from the
/// kernel `cmdline`: every `modname.param=value` token, with the
/// `modname.` prefix stripped, space-joined — the same set the kernel
/// would have applied had the module been builtin.
///
/// The module name is the archive filename with its `NNN-` load-order
/// prefix and `.ko` suffix removed, dashes folded to underscores (the
/// kernel normalizes both the module name and the cmdline param prefix
/// this way). A module with no matching cmdline params yields an empty
/// string (equivalent to the previous no-args load).
fn module_load_args(module: &std::path::Path, cmdline: &str) -> CString {
    let file_name = module.file_name().and_then(|n| n.to_str()).unwrap_or("");
    // Strip the `.ko` suffix and every leading `NNN-` load-order prefix.
    // Two layers add one each — the cache entry (`000-virtio_mmio.ko`)
    // and the initramfs packer (`000-000-virtio_mmio.ko`) — so strip all
    // of them, then fold dashes to underscores (the kernel normalizes
    // both the module name and the cmdline param prefix this way).
    let mut stem = file_name.strip_suffix(".ko").unwrap_or(file_name);
    while let Some(pos) = stem.find('-') {
        let (head, tail) = stem.split_at(pos);
        if !head.is_empty() && head.bytes().all(|b| b.is_ascii_digit()) {
            stem = &tail[1..];
        } else {
            break;
        }
    }
    let modname = stem.replace('-', "_");
    let prefix = format!("{modname}.");

    let mut parts: Vec<&str> = Vec::new();
    for tok in cmdline.split_whitespace() {
        if let Some(param) = tok.strip_prefix(&prefix)
            && param.contains('=')
        {
            parts.push(param);
        }
    }
    // NUL can't appear in a whitespace-split cmdline token, so the
    // CString build cannot fail; default to empty on the impossible case.
    CString::new(parts.join(" ")).unwrap_or_default()
}

/// Report a fatal module-load failure to every pre-console diagnostic
/// sink, then reboot.
///
/// `/dev/kmsg` is written FIRST and is the load-bearing sink: when the
/// guest console is itself a module that failed to load, the kernel log
/// (char 1:11, created by devtmpfs) is the only path whose output still
/// reaches the host — the kernel's `console=ttyS0` drains printk to the
/// emulated 16550 UART the host captures. COM2/COM1 cover the case where
/// the console driver did load. This runs before the tracing subscriber
/// is installed, so it writes raw rather than via `tracing::error!`.
fn fatal_module_error(msg: &str) -> ! {
    let line = format!("FATAL: ktstr-init: kernel module load failed: {msg}\n");
    let _ = fs::write("/dev/kmsg", &line);
    let _ = fs::write(COM2, &line);
    let _ = fs::write(COM1, &line);
    force_reboot()
}

#[cfg(test)]
mod tests {
    use super::module_load_args;
    use std::path::Path;

    const CMDLINE: &str = "console=ttyS0 rdinit=/init virtio_mmio.device=0x1000@0xc0000000:5 \
                           KTSTR_MODE=shell";

    /// The doubled `NNN-` prefix (cache entry + initramfs packer) must be
    /// stripped so the derived module name matches the cmdline param
    /// prefix and the `device=` token is forwarded — the fix for the
    /// modular-virtio_mmio boot hang.
    #[test]
    fn forwards_virtio_mmio_device_through_double_prefix() {
        let args = module_load_args(Path::new("/modules/000-000-virtio_mmio.ko"), CMDLINE);
        assert_eq!(args.to_str().unwrap(), "device=0x1000@0xc0000000:5");
    }

    #[test]
    fn single_prefix_also_resolves() {
        let args = module_load_args(Path::new("/modules/000-virtio_mmio.ko"), CMDLINE);
        assert_eq!(args.to_str().unwrap(), "device=0x1000@0xc0000000:5");
    }

    /// A module with no matching cmdline params loads with empty args,
    /// and a module name containing an underscore is preserved.
    #[test]
    fn no_matching_params_yields_empty() {
        let args = module_load_args(Path::new("/modules/002-002-net_failover.ko"), CMDLINE);
        assert_eq!(args.to_str().unwrap(), "");
    }
}
