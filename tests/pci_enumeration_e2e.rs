//! VM-backed integration test for the virtio-PCI transport's
//! foundational surface: an empty PCI segment (`PCI0`) whose only
//! function is the host bridge at `00:00.0`.
//!
//! Boots a KVM guest via the `#[ktstr_test]` harness with `pci =
//! true`, then, INSIDE the guest, asserts the guest kernel
//! enumerated the host bridge and that config-space access works
//! from a non-boot vCPU.
//!
//! What this exercises end-to-end:
//!
//! - The MCFG ACPI table + the `_SB.PCI0` DSDT object: the guest's
//!   PCI subsystem only enumerates `PCI0` if it accepted both.
//! - Type-1 CAM dispatch (`0xCF8`/`0xCFC` PIO): on x86_64 the kernel
//!   reads base config (vendor/device/class) via `pci_direct_conf1`
//!   (CAM), so a successful enumeration proves the CAM PIO arms in
//!   `exit_dispatch` decode and serve the host bridge.
//! - The AP-threaded `pci_bus`: the final check pins a thread to a
//!   non-boot CPU and issues a FRESH config read there. The
//!   `/sys/.../config` read is a real config access (not the cached
//!   `struct pci_dev` copy), so the resulting VMEXIT lands on the AP
//!   vCPU thread — proving the `pci_bus` handle is shared with and
//!   dispatched from every vCPU, not just the BSP.
//!
//! Extended config space (offset >= 0x100, the ECAM/MMIO dispatch
//! path) IS reachable here and is exercised. A host-bridge-class
//! device takes the extended-size path in `pci_cfg_space_size`
//! (drivers/pci/probe.c: `class == PCI_CLASS_BRIDGE_HOST` ->
//! `pci_cfg_space_size_ext`) BEFORE the PCIe-capability check, reads
//! reg 0x100 (our zeroed, non-aliased extended config — not the
//! 0xffffffff "absent" sentinel), and sizes the sysfs `config`
//! attribute at 4096 bytes, NOT the 256-byte default. So a
//! guest-userspace read of `config` at offset 0x100 routes via ECAM
//! (reg >= 256 -> `raw_pci_ext_ops`/MMCONFIG) to the ECAM MMIO
//! dispatch. The final check reads offset 0x100 from a non-boot CPU
//! to exercise that ECAM MMIO path end-to-end (it returns 0 — the
//! host bridge has no extended capabilities). The ECAM/CAM decode is
//! also unit-tested in `src/vmm/pci/mod.rs` (`ecam_*`,
//! `cam_decodes_extended_register_not_aliased_to_base`,
//! `ecam_decodes_extended_register_not_aliased_to_base`).
//!
//! No scheduler is attached: the in-kernel default (EEVDF) is
//! sufficient — PCI enumeration does not depend on a sched_ext
//! scheduler, and skipping the attach keeps the boot short.

use anyhow::Result;
use ktstr::assert::{AssertDetail, AssertResult, DetailKind};
use ktstr::ktstr_test;
use ktstr::scenario::Ctx;

/// The host bridge function lives at segment 0, bus 0, device 0,
/// function 0 — the address every PCI enumerator probes first.
const HOST_BRIDGE_DEV: &str = "/sys/bus/pci/devices/0000:00:00.0";

/// Expected identity of the host bridge, matching the constants in
/// `src/vmm/pci/mod.rs` (`HOST_BRIDGE_VENDOR_ID` / `_DEVICE_ID`) and
/// the host-bridge class (base 0x06 / subclass 0x00 / prog-if 0x00,
/// which Linux renders as `0x060000`).
const EXPECT_VENDOR: u32 = 0x8086;
const EXPECT_DEVICE: u32 = 0x0d57;
const EXPECT_CLASS: u32 = 0x060000;

/// Read a sysfs attribute that holds a single `0x`-prefixed hex
/// integer (e.g. `vendor`, `device`, `class`) and parse it.
fn read_sysfs_hex(path: &str) -> Result<u32, String> {
    let raw = std::fs::read_to_string(path).map_err(|e| format!("read {path}: {e}"))?;
    let trimmed = raw.trim();
    let digits = trimmed.strip_prefix("0x").unwrap_or(trimmed);
    u32::from_str_radix(digits, 16).map_err(|e| format!("parse {path}={trimmed:?}: {e}"))
}

/// Pin a fresh thread to `cpu`, then read a 4-byte little-endian
/// dword from the host bridge's raw `config` file at `offset`. The
/// read runs on `cpu`'s vCPU, so its config-access VMEXIT is served
/// by that vCPU's `pci_bus` dispatch.
fn read_config_dword_on_cpu(cpu: usize, offset: u64) -> Result<u32, String> {
    let handle = std::thread::spawn(move || -> Result<u32, String> {
        // Force this thread onto `cpu`. A single-CPU mask migrates
        // the running thread during the syscall; yield-spin (no
        // sleep) until sched_getcpu() confirms the move, so the
        // subsequent config read genuinely executes on the AP vCPU.
        unsafe {
            let mut set: libc::cpu_set_t = std::mem::zeroed();
            libc::CPU_ZERO(&mut set);
            libc::CPU_SET(cpu, &mut set);
            if libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &set) != 0 {
                return Err(format!(
                    "sched_setaffinity(cpu={cpu}) failed: {}",
                    std::io::Error::last_os_error()
                ));
            }
            let mut on_cpu = libc::sched_getcpu() == cpu as i32;
            for _ in 0..10_000 {
                if on_cpu {
                    break;
                }
                libc::sched_yield();
                on_cpu = libc::sched_getcpu() == cpu as i32;
            }
            if !on_cpu {
                return Err(format!(
                    "pinned to cpu {cpu} but sched_getcpu() never reported it"
                ));
            }
        }

        use std::io::{Read, Seek, SeekFrom};
        let config_path = format!("{HOST_BRIDGE_DEV}/config");
        let mut f =
            std::fs::File::open(&config_path).map_err(|e| format!("open {config_path}: {e}"))?;
        if offset != 0 {
            f.seek(SeekFrom::Start(offset))
                .map_err(|e| format!("seek {config_path}@{offset:#x}: {e}"))?;
        }
        let mut buf = [0u8; 4];
        f.read_exact(&mut buf)
            .map_err(|e| format!("read {config_path}@{offset:#x}: {e}"))?;
        Ok(u32::from_le_bytes(buf))
    });
    handle
        .join()
        .map_err(|_| "config-read thread panicked".to_string())?
}

/// Enumerate the host bridge and prove config access works from a
/// non-boot vCPU.
///
/// Topology: 1 LLC / 2 cores / 1 thread = 2 vCPUs, so CPU 1 is a
/// non-boot CPU and is online after SMP bringup. Duration is short:
/// the assertions read post-boot enumeration state, not behavior
/// over time.
#[ktstr_test(pci = true, llcs = 1, cores = 2, threads = 1, duration_s = 5, watchdog_timeout_s = 60)]
fn pci_host_bridge_enumerates_and_dispatches_from_ap_cpu(_ctx: &Ctx) -> Result<AssertResult> {
    // 1. The host bridge must be present. If it isn't, the guest
    //    either rejected MCFG/DSDT or the CAM dispatch never served
    //    a valid vendor id at 00:00.0 (an absent function reads as
    //    all-ones, which the kernel skips). Dump what DID enumerate.
    if !std::path::Path::new(HOST_BRIDGE_DEV).exists() {
        let listing = std::fs::read_dir("/sys/bus/pci/devices")
            .map(|rd| {
                rd.filter_map(|e| e.ok().map(|e| e.file_name().to_string_lossy().into_owned()))
                    .collect::<Vec<_>>()
            })
            .unwrap_or_else(|e| vec![format!("(read /sys/bus/pci/devices: {e})")]);
        return Ok(AssertResult::fail(AssertDetail::new(
            DetailKind::Other,
            format!(
                "host bridge not enumerated: {HOST_BRIDGE_DEV} is absent. \
                 The guest kernel did not find a PCI function at \
                 00:00.0 — it either rejected the MCFG table / the \
                 _SB.PCI0 DSDT object, or the CAM config-access \
                 dispatch (0xCF8/0xCFC) did not return a valid vendor \
                 id. Devices that DID enumerate under \
                 /sys/bus/pci/devices: {listing:?}"
            ),
        )));
    }

    // 2. Identity, via the kernel-parsed sysfs attributes (populated
    //    from base config read over CAM during enumeration).
    for (attr, expect, label) in [
        ("vendor", EXPECT_VENDOR, "vendor id"),
        ("device", EXPECT_DEVICE, "device id"),
        ("class", EXPECT_CLASS, "class code"),
    ] {
        let path = format!("{HOST_BRIDGE_DEV}/{attr}");
        let got = match read_sysfs_hex(&path) {
            Ok(v) => v,
            Err(e) => {
                return Ok(AssertResult::fail(AssertDetail::new(
                    DetailKind::Other,
                    format!("reading host bridge {label}: {e}"),
                )));
            }
        };
        if got != expect {
            return Ok(AssertResult::fail(AssertDetail::new(
                DetailKind::Other,
                format!(
                    "host bridge {label} mismatch: sysfs {attr} = \
                     {got:#06x}, expected {expect:#06x}. The guest \
                     read a different value over CAM than the \
                     HostBridge config space advertises",
                ),
            )));
        }
    }

    // 3. AP-threaded dispatch: a FRESH config read pinned to a
    //    non-boot CPU. The dword at offset 0 is
    //    (device << 16) | vendor; reading it over CAM from CPU 1's
    //    vCPU proves the pci_bus handle is shared with and served
    //    by every vCPU, not just the BSP.
    let vendor_device = match read_config_dword_on_cpu(1, 0) {
        Ok(v) => v,
        Err(e) => {
            return Ok(AssertResult::fail(AssertDetail::new(
                DetailKind::Other,
                format!(
                    "config read from non-boot CPU 1 failed: {e}. The \
                     pci_bus must be dispatched from AP vCPU threads, \
                     not only the BSP",
                ),
            )));
        }
    };
    let got_vendor = vendor_device & 0xFFFF;
    let got_device = vendor_device >> 16;
    if got_vendor != EXPECT_VENDOR || got_device != EXPECT_DEVICE {
        return Ok(AssertResult::fail(AssertDetail::new(
            DetailKind::Other,
            format!(
                "host bridge vendor/device read from non-boot CPU 1 \
                 mismatch: vendor = {got_vendor:#06x} (expected \
                 {EXPECT_VENDOR:#06x}), device = {got_device:#06x} \
                 (expected {EXPECT_DEVICE:#06x}). The AP vCPU's \
                 config-access dispatch returned the wrong bytes",
            ),
        )));
    }

    // 4. EXTENDED config via ECAM from the non-boot CPU. The host
    //    bridge's sysfs `config` is 4096 bytes (host-bridge class takes
    //    pci_cfg_space_size_ext), so a read at offset 0x100 routes
    //    through the ECAM MMIO dispatch (reg >= 256) on CPU 1's vCPU —
    //    distinct from the CAM PIO path that step 3 exercised. The host
    //    bridge has no extended capabilities, so its extended config is
    //    zero; a non-zero or failed read means the ECAM extended path is
    //    broken (e.g. reg masked to base, or config wrongly sized 256B).
    let ext_dword = match read_config_dword_on_cpu(1, 0x100) {
        Ok(v) => v,
        Err(e) => {
            return Ok(AssertResult::fail(AssertDetail::new(
                DetailKind::Other,
                format!(
                    "extended config read at offset 0x100 from non-boot \
                     CPU 1 failed: {e}. The host bridge's config space \
                     should be 4096 bytes (host-bridge class takes the \
                     extended-size path in pci_cfg_space_size), making \
                     the ECAM MMIO dispatch reachable from guest \
                     userspace",
                ),
            )));
        }
    };
    if ext_dword != 0 {
        return Ok(AssertResult::fail(AssertDetail::new(
            DetailKind::Other,
            format!(
                "extended config dword at offset 0x100 (read via ECAM \
                 from non-boot CPU 1) = {ext_dword:#010x}, expected 0. \
                 The host bridge has no extended capabilities, so its \
                 extended config region is zero — a non-zero value means \
                 the ECAM extended-register decode is wrong",
            ),
        )));
    }

    Ok(AssertResult::pass())
}
