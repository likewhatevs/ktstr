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

    for module in &modules {
        let file = match fs::File::open(module) {
            Ok(f) => f,
            Err(e) => fatal_module_error(&format!("open {}: {e}", module.display())),
        };
        match finit_module(&file, c"", ModuleInitFlags::empty()) {
            Ok(()) => {}
            Err(nix::errno::Errno::EEXIST) => {}
            Err(e) => fatal_module_error(&format!("finit_module {}: {e}", module.display())),
        }
    }
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
