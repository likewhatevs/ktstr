//! Emulated PCI Express transport, additive to the virtio-MMIO transport.
//!
//! ktstr keeps virtio-MMIO for the console on x86_64 (aarch64 also keeps
//! block+net on MMIO), discovered by the guest via `virtio_mmio.device=`
//! kernel-cmdline tokens, and adds this virtio-PCI transport so virtio-net
//! NICs and virtio-blk attach as PCI functions on x86_64 — one function per
//! NIC (enabling multi-NIC topologies with per-NIC IRQ steering) plus the
//! single block function, none of which emit an MMIO cmdline token. The two
//! transports coexist in a single guest: the MMIO devices are advertised on
//! the kernel cmdline, the PCI devices via ACPI (MCFG + a `_SB.PCI0` host
//! bridge in the DSDT). No reference VMM runs both transports in one guest, so
//! the coexistence (disjoint GSI pools, orthogonal discovery) is ktstr's own.
//!
//! Scope of this module (the arch-neutral PCI core):
//! - a single PCI segment with a single bus (bus 0),
//! - per-function 4 KiB configuration space with a per-byte write mask,
//! - type-1 CAM (I/O ports 0xCF8/0xCFC) and ECAM (Enhanced Configuration
//!   Access Mechanism) decode — BOTH cover the full 4 KiB configuration
//!   space (CAM reaches extended config via the type-1 address bits \[27:24\],
//!   matching the kernel's `PCI_CONF1_ADDRESS`),
//! - the host bridge function at 00:00.0.
//!
//! Both access mechanisms are mandatory on x86_64: the guest's PCI core routes
//! base config (the entire enumeration-critical region — vendor/device id,
//! BARs, the capability chain, the MSI-X capability) through type-1 CAM
//! (`raw_pci_ops` = `pci_direct_conf1`, arch/x86/pci/direct.c) and only
//! extended config (reg >= 256) through ECAM (`raw_pci_ext_ops`,
//! arch/x86/pci/mmconfig_64.c); `pci_check_type1` unconditionally probes
//! 0xCF8/0xCFC at boot. The two mechanisms share one configuration-space
//! backing per function — alternate windows onto the same registers. The CAM
//! I/O ports and the ECAM window are wired by the arch layer; this module owns
//! the decode and the CAM address latch.
//!
//! The ECAM window base/size, the ACPI MCFG/DSDT emission (x86_64) and the FDT
//! `pci` node (aarch64) are arch-specific and live under `x86_64`/`aarch64`;
//! this module takes the ECAM base as a constructor parameter so it stays
//! arch-neutral. The host bridge occupies slot 0; one virtio-net function
//! attaches per configured NIC at slots 1..=MAX_VIRTIO_NICS and the single
//! virtio-blk function at slot MAX_VIRTIO_NICS+1 (each with a memory BAR,
//! INTx, and per-queue MSI-X).
//!
//! Hostile-guest defense: every config access is bounds-checked. An access to
//! an absent bus/device/function, or one that would read past the 4 KiB
//! configuration space, returns all-ones (the PCI "no device" / unimplemented
//! convention the guest's enumerator relies on, drivers/pci/probe.c
//! `pci_bus_generic_read_dev_vendor_id`); a write to an absent function or
//! past the config space is silently dropped. No access panics or indexes out
//! of the backing store.

/// Size of one PCI(e) function's configuration space (4 KiB extended config).
const CONFIG_SPACE_SIZE: usize = 4096;

/// Devices per bus the host bridge backs. PCI allows 32 device slots per bus
/// (`PCI_MAX_NR_DEVS`, drivers/pci/pci.h); ktstr exposes a single bus, which is
/// ample for the NIC counts these tests use.
const MAX_DEVICES: usize = 32;

/// ECAM address decode (PCIe Base Spec / `arch/x86/pci/mmconfig_64.c`):
/// `config_offset = (bus << 20) | (devfn << 12) | reg`. `reg` is the 12-bit
/// (4 KiB) offset within a function's configuration space.
const ECAM_BUS_SHIFT: u64 = 20;
const ECAM_DEVFN_SHIFT: u64 = 12;
const ECAM_REG_MASK: u64 = 0xFFF;

/// ECAM bytes per bus: 256 functions × 4 KiB = 1 MiB.
pub(crate) const ECAM_BYTES_PER_BUS: u64 = 1 << ECAM_BUS_SHIFT;

/// Type-1 CAM CONFIG_ADDRESS enable bit (bit 31). A CONFIG_DATA (0xCFC)
/// access is only decoded when the latched address has this set; otherwise
/// the data window reads all-ones and drops writes (qemu `pci_host.c`).
const CAM_ENABLE_BIT: u32 = 0x8000_0000;

// Host bridge identity. Both cloud-hypervisor and firecracker expose the host
// bridge as a real function at 00:00.0 with the Intel vendor id and a
// host-bridge class; the guest's enumerator expects 00:00.0 present (a
// truly-empty bus 0 is unusual). The guest binds the bus via the ACPI
// PNP0A08 object, not via this function's id, so the exact device id is
// informational (matches the references' 0x0d57).
const HOST_BRIDGE_VENDOR_ID: u16 = 0x8086;
const HOST_BRIDGE_DEVICE_ID: u16 = 0x0D57;
/// PCI class code, big-end byte = base class. Base 0x06 (bridge),
/// subclass 0x00 (host bridge), prog-if 0x00.
const CLASS_BRIDGE: u8 = 0x06;
const SUBCLASS_HOST: u8 = 0x00;

// Standard configuration-space register offsets (include/uapi/linux/pci_regs.h).
// `pub(crate)` so device-function modules (e.g. virtio-net) building their own
// config space reuse the canonical offsets rather than redefining them.
pub(crate) const REG_VENDOR_ID: u16 = 0x00;
pub(crate) const REG_DEVICE_ID: u16 = 0x02;
pub(crate) const REG_COMMAND: u16 = 0x04;
pub(crate) const REG_STATUS: u16 = 0x06;
pub(crate) const REG_REVISION_ID: u16 = 0x08;
pub(crate) const REG_PROG_IF: u16 = 0x09;
pub(crate) const REG_SUBCLASS: u16 = 0x0A;
pub(crate) const REG_CLASS: u16 = 0x0B;
pub(crate) const REG_HEADER_TYPE: u16 = 0x0E;
/// First memory BAR (BAR0), a 32-bit memory BAR
/// (`include/uapi/linux/pci_regs.h` PCI_BASE_ADDRESS_0). Register 0x14 is
/// BAR1 (unimplemented for the virtio-net function — a 32-bit BAR0 occupies
/// only 0x10).
pub(crate) const REG_BAR0: u16 = 0x10;
/// Capabilities-list pointer (PCI_CAPABILITY_LIST) — the config offset of the
/// first capability; valid only when `PCI_STATUS_CAP_LIST` is set in STATUS.
pub(crate) const REG_CAP_PTR: u16 = 0x34;
pub(crate) const REG_INTERRUPT_LINE: u16 = 0x3C;
pub(crate) const REG_INTERRUPT_PIN: u16 = 0x3D;

/// `PCI_STATUS` capabilities-list bit (PCI_STATUS_CAP_LIST): tells the guest a
/// capability chain is present at `REG_CAP_PTR`.
pub(crate) const PCI_STATUS_CAP_LIST: u16 = 0x0010;
/// `PCI_COMMAND` memory-space-enable bit (PCI_COMMAND_MEMORY): the guest sets
/// it to enable a function's memory BAR decode.
pub(crate) const PCI_COMMAND_MEMORY: u16 = 0x0002;

/// COMMAND register writable bits: I/O space, memory space, bus master, SERR
/// enable, INTx disable (`PCI_COMMAND_*`) — the bits a guest meaningfully
/// toggles. This is NARROWER than cloud-hypervisor, which makes the whole low 16
/// bits writable; restricting to the guest-meaningful bits is a safe subset (it
/// can only reject bits the guest has no reason to set). The host bridge has no
/// BARs, so these are inert there, but a writable COMMAND is what the guest's
/// PCI core expects.
pub(crate) const PCI_COMMAND_WMASK: u16 = 0x0001 | 0x0002 | 0x0004 | 0x0100 | 0x0400;

/// A single PCI function's view: its configuration space plus, for
/// functions with a memory BAR, the MMIO behind it.
///
/// The bus dispatches decoded config accesses (read/write at a byte register
/// within \[0, 4096)) to the function. Implementors back this with a
/// [`ConfigSpace`]. A function that exposes a memory BAR (e.g. virtio-net,
/// whose modern transport regions live behind BAR0) overrides
/// [`Self::bar_window`] to publish the programmed window and
/// [`Self::bar_read`]/[`Self::bar_write`] to serve MMIO within it; the
/// host bridge has no BAR and uses the no-op defaults.
pub(crate) trait PciFunction: Send {
    /// Read `data.len()` bytes of configuration space starting at byte `reg`.
    fn config_read(&self, reg: u16, data: &mut [u8]);
    /// Write `data.len()` bytes of configuration space starting at byte `reg`.
    fn config_write(&mut self, reg: u16, data: &[u8]);

    /// The function's memory-BAR window `[base, base + len)` once the
    /// guest has programmed the BAR and enabled memory-space decode
    /// (PCI_COMMAND memory-space bit). `None` for a function with no BAR
    /// (the host bridge), or while the BAR is unprogrammed / memory
    /// decode is disabled — the bus then routes no BAR MMIO here. The
    /// bus uses this to decide which function (if any) owns a given
    /// guest-physical MMIO address.
    fn bar_window(&self) -> Option<(u64, u64)> {
        None
    }

    /// Read `data.len()` bytes at byte `offset` within this function's
    /// BAR window. `&mut self` because a BAR region can be read-to-clear
    /// (the virtio-pci ISR register: the read returns the pending bits
    /// AND clears them — there is no separate ACK register). Only invoked
    /// when [`Self::bar_window`] is `Some` and contains the access; the
    /// default (all-ones) is never reached for a no-BAR function because
    /// [`Self::bar_window`] returns `None`.
    fn bar_read(&mut self, offset: u64, data: &mut [u8]) {
        let _ = offset;
        data.fill(0xFF);
    }

    /// Write `data.len()` bytes at byte `offset` within this function's
    /// BAR window. Only invoked when [`Self::bar_window`] is `Some` and
    /// contains the access. The default (drop) is never reached for a
    /// no-BAR function because [`Self::bar_window`] returns `None`.
    fn bar_write(&mut self, offset: u64, data: &[u8]) {
        let _ = (offset, data);
    }
}

/// A 4 KiB PCI configuration space with a per-byte write mask.
///
/// Modeled on qemu's `pci.c`: a write applies `new = (cur & !wmask) | (val &
/// wmask)` per byte. Read-only identity fields (vendor/device/class) get
/// `wmask = 0`; the BAR-size probe (write all-ones, read back `!(size-1) |
/// type`) and writable registers (COMMAND) fall out of the mask with no
/// stateful "sizing mode". Initialization writes (`set_*`) bypass the mask.
pub(crate) struct ConfigSpace {
    space: Box<[u8; CONFIG_SPACE_SIZE]>,
    wmask: Box<[u8; CONFIG_SPACE_SIZE]>,
}

impl ConfigSpace {
    pub(crate) fn new() -> Self {
        Self {
            space: Box::new([0u8; CONFIG_SPACE_SIZE]),
            wmask: Box::new([0u8; CONFIG_SPACE_SIZE]),
        }
    }

    /// Read into `data` from byte `reg`. An access that would read past the
    /// 4 KiB space returns all-ones for the ENTIRE access rather than panicking.
    /// For an access wholly past the end this matches qemu
    /// `pci_host_config_read_common` (`limit <= addr` → `~0`); for a STRADDLING
    /// access (start in-range, end past) it diverges from qemu, which serves the
    /// in-range bytes (`MIN(len, limit - addr)`). The divergence is unobservable
    /// and safe: no conformant guest issues a config access crossing the 4 KiB
    /// boundary (config accesses are naturally aligned, ≤4 bytes), and the
    /// all-ones fill is non-OOB (the `checked_add` + range guard below).
    pub(crate) fn read(&self, reg: u16, data: &mut [u8]) {
        let start = reg as usize;
        match start.checked_add(data.len()) {
            Some(end) if end <= CONFIG_SPACE_SIZE => data.copy_from_slice(&self.space[start..end]),
            _ => data.fill(0xFF),
        }
    }

    /// Write `data` at byte `reg`, honoring the per-byte write mask. An access
    /// that would write past the 4 KiB space is dropped.
    pub(crate) fn write(&mut self, reg: u16, data: &[u8]) {
        let start = reg as usize;
        let end = match start.checked_add(data.len()) {
            Some(end) if end <= CONFIG_SPACE_SIZE => end,
            _ => return,
        };
        for (i, &b) in (start..end).zip(data.iter()) {
            let w = self.wmask[i];
            self.space[i] = (self.space[i] & !w) | (b & w);
        }
    }

    // Init-time setters (bypass the write mask). All callers use compile-time
    // constant offsets within the 4 KiB space, so an out-of-range `reg` is a
    // caller bug; each asserts `reg + width <= CONFIG_SPACE_SIZE` so it surfaces
    // as a clear debug-build assertion rather than an opaque slice-index panic.
    pub(crate) fn set_u8(&mut self, reg: u16, val: u8) {
        debug_assert!(
            (reg as usize) < CONFIG_SPACE_SIZE,
            "config-space set_u8 out of range"
        );
        self.space[reg as usize] = val;
    }

    pub(crate) fn set_u16(&mut self, reg: u16, val: u16) {
        let reg = reg as usize;
        debug_assert!(
            reg + 2 <= CONFIG_SPACE_SIZE,
            "config-space set_u16 out of range"
        );
        self.space[reg..reg + 2].copy_from_slice(&val.to_le_bytes());
    }

    pub(crate) fn set_u32(&mut self, reg: u16, val: u32) {
        let reg = reg as usize;
        debug_assert!(
            reg + 4 <= CONFIG_SPACE_SIZE,
            "config-space set_u32 out of range"
        );
        self.space[reg..reg + 4].copy_from_slice(&val.to_le_bytes());
    }

    /// Make the 16-bit register at `reg` writable with the given mask.
    pub(crate) fn set_wmask_u16(&mut self, reg: u16, mask: u16) {
        let reg = reg as usize;
        debug_assert!(
            reg + 2 <= CONFIG_SPACE_SIZE,
            "config-space set_wmask_u16 out of range"
        );
        self.wmask[reg..reg + 2].copy_from_slice(&mask.to_le_bytes());
    }

    /// Make the 32-bit register at `reg` writable with the given mask —
    /// used for a memory BAR: the base-address bits are writable, the
    /// low type/prefetch bits stay read-only so the guest's BAR-size
    /// probe (write all-ones, read back) recovers the region size.
    pub(crate) fn set_wmask_u32(&mut self, reg: u16, mask: u32) {
        let reg = reg as usize;
        debug_assert!(
            reg + 4 <= CONFIG_SPACE_SIZE,
            "config-space set_wmask_u32 out of range"
        );
        self.wmask[reg..reg + 4].copy_from_slice(&mask.to_le_bytes());
    }
}

/// The host bridge function at 00:00.0: class "host bridge", no BARs, no
/// capabilities. The guest's PCI host-bridge driver binds the bus via the ACPI
/// `_SB.PCI0` (PNP0A08) object; this function is the conventional bus-0
/// device-0 presence the enumerator expects.
pub(crate) struct HostBridge {
    cfg: ConfigSpace,
}

impl HostBridge {
    fn new() -> Self {
        let mut cfg = ConfigSpace::new();
        cfg.set_u16(REG_VENDOR_ID, HOST_BRIDGE_VENDOR_ID);
        cfg.set_u16(REG_DEVICE_ID, HOST_BRIDGE_DEVICE_ID);
        cfg.set_u8(REG_REVISION_ID, 0);
        cfg.set_u8(REG_PROG_IF, 0);
        cfg.set_u8(REG_SUBCLASS, SUBCLASS_HOST);
        cfg.set_u8(REG_CLASS, CLASS_BRIDGE);
        // Header type 0 (single-function general device); REG_HEADER_TYPE
        // already zero. The reference is named to document the layout.
        let _ = REG_HEADER_TYPE;
        cfg.set_wmask_u16(REG_COMMAND, PCI_COMMAND_WMASK);
        Self { cfg }
    }
}

impl PciFunction for HostBridge {
    fn config_read(&self, reg: u16, data: &mut [u8]) {
        self.cfg.read(reg, data);
    }

    fn config_write(&mut self, reg: u16, data: &[u8]) {
        self.cfg.write(reg, data);
    }
}

/// A single PCI segment's bus 0: the device slots plus ECAM decode.
///
/// Holds up to [`MAX_DEVICES`] single-function devices indexed by device slot
/// (devfn `>> 3`); function 0 only. Slot 0 is the [`HostBridge`]. The run loop
/// owns the bus (threaded into the MMIO dispatch alongside the other device
/// handles) and routes ECAM-window accesses here.
pub(crate) struct PciBus {
    ecam_base: u64,
    ecam_size: u64,
    funcs: [Option<Box<dyn PciFunction>>; MAX_DEVICES],
    /// Latched type-1 CAM CONFIG_ADDRESS (last 0xCF8 write).
    config_address: u32,
}

impl PciBus {
    /// Create a single-bus segment whose ECAM window is `[ecam_base,
    /// ecam_base + ecam_size)`, with the host bridge at 00:00.0. `ecam_size`
    /// must be a whole number of buses ([`ECAM_BYTES_PER_BUS`] each); a
    /// single-bus segment uses exactly one.
    pub(crate) fn new(ecam_base: u64, ecam_size: u64) -> Self {
        let mut funcs: [Option<Box<dyn PciFunction>>; MAX_DEVICES] = std::array::from_fn(|_| None);
        funcs[0] = Some(Box::new(HostBridge::new()));
        Self {
            ecam_base,
            ecam_size,
            funcs,
            config_address: 0,
        }
    }

    /// Install `func` at device `slot` (1..[`MAX_DEVICES`); slot 0 is the
    /// [`HostBridge`]). Setup uses this to attach virtio-net PCI functions.
    /// A slot-0 or out-of-range install is dropped — a debug assertion catches
    /// the caller bug in tests, and a `tracing::error!` makes the drop loud in
    /// release so a NIC silently failing to enumerate (no eth0) is diagnosable
    /// rather than a mystery. Setup installs one function per configured NIC
    /// at a per-NIC slot from `kvm::virtio_net_pci_slot(index)`, so the slot
    /// is dynamic (one call per NIC, slots 1..N).
    pub(crate) fn add_function(&mut self, slot: usize, func: Box<dyn PciFunction>) {
        debug_assert!(
            slot != 0 && slot < MAX_DEVICES,
            "slot 0 is the host bridge; slot must be in 1..{MAX_DEVICES}"
        );
        if slot != 0 && slot < MAX_DEVICES {
            self.funcs[slot] = Some(func);
        } else {
            tracing::error!(
                slot,
                max = MAX_DEVICES,
                "PCI function install dropped: slot 0 is the host bridge and \
                 slot must be in 1..MAX_DEVICES — the device will not enumerate"
            );
        }
    }

    /// True when `gpa` falls within this segment's ECAM window — the guard the
    /// MMIO dispatch uses to route an access here.
    pub(crate) fn ecam_contains(&self, gpa: u64) -> bool {
        gpa >= self.ecam_base && gpa < self.ecam_base + self.ecam_size
    }

    /// Service an ECAM read at guest-physical `gpa`. Decodes (bus, devfn, reg)
    /// and reads the target function's config space; an absent function or an
    /// out-of-window address yields all-ones.
    pub(crate) fn ecam_read(&self, gpa: u64, data: &mut [u8]) {
        let offset = gpa.wrapping_sub(self.ecam_base);
        if offset >= self.ecam_size {
            data.fill(0xFF);
            return;
        }
        let (bus, devfn, reg) = decode_ecam(offset);
        self.read_config(bus, devfn, reg, data);
    }

    /// Service an ECAM write at guest-physical `gpa`. An absent function or an
    /// out-of-window address drops the write.
    pub(crate) fn ecam_write(&mut self, gpa: u64, data: &[u8]) {
        let offset = gpa.wrapping_sub(self.ecam_base);
        if offset >= self.ecam_size {
            return;
        }
        let (bus, devfn, reg) = decode_ecam(offset);
        self.write_config(bus, devfn, reg, data);
    }

    /// True when `gpa` falls within some present function's programmed
    /// BAR window — the guard the MMIO dispatch uses to route a BAR
    /// access here (the BAR analog of [`Self::ecam_contains`]). A
    /// function publishes its window via [`PciFunction::bar_window`],
    /// which returns `None` until the guest programs the BAR and enables
    /// memory decode, so an unprogrammed device claims no MMIO.
    pub(crate) fn bar_mmio_contains(&self, gpa: u64) -> bool {
        self.bar_owner(gpa).is_some()
    }

    /// Service a BAR MMIO read: dispatch to the function whose BAR
    /// window contains `gpa`, at the window-relative offset. `&mut self`
    /// because a BAR read can be read-to-clear (the virtio-pci ISR). An
    /// address in no function's window yields all-ones (the PCI
    /// unmapped-MMIO convention), never a panic.
    pub(crate) fn bar_mmio_read(&mut self, gpa: u64, data: &mut [u8]) {
        match self.bar_owner(gpa) {
            Some((slot, base)) => {
                self.funcs[slot]
                    .as_mut()
                    .unwrap()
                    .bar_read(gpa - base, data);
            }
            None => data.fill(0xFF),
        }
    }

    /// Service a BAR MMIO write: dispatch to the owning function at the
    /// window-relative offset. An address in no function's window is
    /// dropped.
    pub(crate) fn bar_mmio_write(&mut self, gpa: u64, data: &[u8]) {
        if let Some((slot, base)) = self.bar_owner(gpa) {
            self.funcs[slot]
                .as_mut()
                .unwrap()
                .bar_write(gpa - base, data);
        }
    }

    /// Resolve `gpa` to the `(slot, bar_base)` of the present function
    /// whose programmed BAR window contains it, or `None`. Windows are
    /// disjoint (each function is assigned a distinct BAR range), so the
    /// first match is unique.
    fn bar_owner(&self, gpa: u64) -> Option<(usize, u64)> {
        self.funcs.iter().enumerate().find_map(|(slot, f)| {
            let (base, len) = f.as_ref()?.bar_window()?;
            (gpa >= base && gpa < base + len).then_some((slot, base))
        })
    }

    /// Latch a type-1 CAM CONFIG_ADDRESS (an I/O write to port 0xCF8).
    pub(crate) fn cam_set_address(&mut self, val: u32) {
        self.config_address = val;
    }

    /// Read back the latched CAM CONFIG_ADDRESS (an I/O read of port 0xCF8).
    pub(crate) fn cam_get_address(&self) -> u32 {
        self.config_address
    }

    /// Service a type-1 CAM CONFIG_DATA read (I/O read of 0xCFC + `byte_off`,
    /// `byte_off` in 0..4) of the latched (bus, devfn). Covers the full
    /// 0..4095 register range via the extended type-1 address bits (see
    /// [`Self::cam_decode`]); a disabled latch or absent function yields
    /// all-ones.
    pub(crate) fn cam_data_read(&self, byte_off: u8, data: &mut [u8]) {
        match self.cam_decode(byte_off) {
            Some((bus, devfn, reg)) => {
                self.read_config(bus, devfn, reg, data);
            }
            None => data.fill(0xFF),
        }
    }

    /// Service a type-1 CAM CONFIG_DATA write (I/O write of 0xCFC + `byte_off`).
    /// A disabled latch or absent function drops the write.
    pub(crate) fn cam_data_write(&mut self, byte_off: u8, data: &[u8]) {
        if let Some((bus, devfn, reg)) = self.cam_decode(byte_off) {
            self.write_config(bus, devfn, reg, data);
        }
    }

    /// Decode the latched CONFIG_ADDRESS plus the CONFIG_DATA byte offset into
    /// (bus, devfn, reg). `None` when the enable bit is clear. Per the kernel's
    /// type-1 address format (`PCI_CONF1_ADDRESS`, arch/x86/pci/direct.c:17),
    /// the register number is the dword bits \[7:2\] of CONFIG_ADDRESS PLUS the
    /// extended bits \[11:8\] held in CONFIG_ADDRESS \[27:24\]; CAM can thus
    /// address the full 0..4095 register range (the extended-capable type-1
    /// fallback the guest's `pci_conf1_*` uses when ECAM/MMCONFIG is absent),
    /// not only base config. The CONFIG_DATA byte offset selects the byte
    /// within the dword. [`ConfigSpace::read`] bounds-checks against the 4 KiB
    /// backing, so a multi-byte access spanning the end returns all-ones, never
    /// an OOB.
    fn cam_decode(&self, byte_off: u8) -> Option<(u8, u8, u16)> {
        if self.config_address & CAM_ENABLE_BIT == 0 {
            return None;
        }
        let bus = ((self.config_address >> 16) & 0xFF) as u8;
        let devfn = ((self.config_address >> 8) & 0xFF) as u8;
        // reg[7:2] from CONFIG_ADDRESS[7:2]; reg[11:8] from CONFIG_ADDRESS
        // [27:24] (the extended-config bits, PCI_CONF1_ADDRESS); plus the
        // data-port byte offset. Dropping [11:8] would silently alias an
        // extended-config access onto base config.
        let reg_dword = (self.config_address & 0xFC) | ((self.config_address >> 16) & 0xF00);
        let reg = (reg_dword + byte_off as u32) as u16;
        Some((bus, devfn, reg))
    }

    /// Resolve (bus, devfn) to a present single-function device slot, or `None`
    /// for an absent bus / non-zero function / empty or out-of-range slot.
    fn slot_of(&self, bus: u8, devfn: u8) -> Option<usize> {
        let slot = (devfn >> 3) as usize;
        let function = devfn & 0x7;
        if bus != 0 || function != 0 || slot >= MAX_DEVICES {
            return None;
        }
        self.funcs[slot].as_ref().map(|_| slot)
    }

    fn read_config(&self, bus: u8, devfn: u8, reg: u16, data: &mut [u8]) {
        match self.slot_of(bus, devfn) {
            Some(slot) => self.funcs[slot].as_ref().unwrap().config_read(reg, data),
            None => data.fill(0xFF),
        }
    }

    fn write_config(&mut self, bus: u8, devfn: u8, reg: u16, data: &[u8]) {
        if let Some(slot) = self.slot_of(bus, devfn) {
            self.funcs[slot].as_mut().unwrap().config_write(reg, data);
        }
    }
}

/// Decode an ECAM window offset into (bus, devfn, reg).
fn decode_ecam(offset: u64) -> (u8, u8, u16) {
    let bus = (offset >> ECAM_BUS_SHIFT) as u8;
    let devfn = ((offset >> ECAM_DEVFN_SHIFT) & 0xFF) as u8;
    let reg = (offset & ECAM_REG_MASK) as u16;
    (bus, devfn, reg)
}

#[cfg(test)]
mod tests {
    use super::*;

    const BASE: u64 = 0xE000_0000;
    const SIZE: u64 = ECAM_BYTES_PER_BUS;

    fn read_u32(bus: &PciBus, devfn: u8, reg: u16) -> u32 {
        let gpa = BASE + ((devfn as u64) << ECAM_DEVFN_SHIFT) + reg as u64;
        let mut data = [0u8; 4];
        bus.ecam_read(gpa, &mut data);
        u32::from_le_bytes(data)
    }

    #[test]
    fn host_bridge_present_at_00_00_0() {
        let bus = PciBus::new(BASE, SIZE);
        let vendor_device = read_u32(&bus, 0, REG_VENDOR_ID);
        assert_eq!(vendor_device & 0xFFFF, HOST_BRIDGE_VENDOR_ID as u32);
        assert_eq!(vendor_device >> 16, HOST_BRIDGE_DEVICE_ID as u32);
    }

    #[test]
    fn host_bridge_class_is_host_bridge() {
        let bus = PciBus::new(BASE, SIZE);
        // Class code occupies bytes 0x09 (prog-if), 0x0a (subclass),
        // 0x0b (base). Read the dword at 0x08 and extract the top 3 bytes.
        let class_dword = read_u32(&bus, 0, 0x08);
        assert_eq!((class_dword >> 24) as u8, CLASS_BRIDGE);
        assert_eq!((class_dword >> 16) as u8, SUBCLASS_HOST);
    }

    #[test]
    fn absent_device_reads_all_ones() {
        let bus = PciBus::new(BASE, SIZE);
        // Slot 1 is empty: the guest's enumerator reads 0xffffffff and skips.
        assert_eq!(read_u32(&bus, 1 << 3, REG_VENDOR_ID), 0xFFFF_FFFF);
        // Every empty slot on the bus reads absent.
        for slot in 1..MAX_DEVICES as u8 {
            assert_eq!(read_u32(&bus, slot << 3, REG_VENDOR_ID), 0xFFFF_FFFF);
        }
    }

    #[test]
    fn nonzero_bus_reads_all_ones() {
        let bus = PciBus::new(BASE, ECAM_BYTES_PER_BUS * 2);
        // bus 1, device 0 — single-segment exposes only bus 0.
        let gpa = BASE + (1u64 << ECAM_BUS_SHIFT);
        let mut data = [0u8; 4];
        bus.ecam_read(gpa, &mut data);
        assert_eq!(u32::from_le_bytes(data), 0xFFFF_FFFF);
    }

    #[test]
    fn nonzero_function_reads_all_ones() {
        let bus = PciBus::new(BASE, SIZE);
        // device 0, function 1 — host bridge is single-function.
        assert_eq!(read_u32(&bus, 1, REG_VENDOR_ID), 0xFFFF_FFFF);
    }

    #[test]
    fn out_of_window_reads_all_ones() {
        let bus = PciBus::new(BASE, SIZE);
        let mut data = [0u8; 4];
        // Below the window.
        bus.ecam_read(BASE - 4, &mut data);
        assert_eq!(u32::from_le_bytes(data), 0xFFFF_FFFF);
        // At/after the window end.
        bus.ecam_read(BASE + SIZE, &mut data);
        assert_eq!(u32::from_le_bytes(data), 0xFFFF_FFFF);
        assert!(!bus.ecam_contains(BASE - 1));
        assert!(bus.ecam_contains(BASE));
        assert!(bus.ecam_contains(BASE + SIZE - 1));
        assert!(!bus.ecam_contains(BASE + SIZE));
    }

    #[test]
    fn overread_past_config_space_is_all_ones_not_panic() {
        let bus = PciBus::new(BASE, SIZE);
        // Read a dword straddling the 4 KiB config-space end of function 0.
        let gpa = BASE + (ECAM_REG_MASK - 1); // reg 0xFFE, 4 bytes -> past 0x1000
        let mut data = [0u8; 4];
        bus.ecam_read(gpa, &mut data);
        assert_eq!(u32::from_le_bytes(data), 0xFFFF_FFFF);
    }

    #[test]
    fn command_register_is_writable_identity_is_read_only() {
        let mut bus = PciBus::new(BASE, SIZE);
        let cmd_gpa = BASE + REG_COMMAND as u64;
        // Write the bus-master + memory-space bits to COMMAND.
        bus.ecam_write(cmd_gpa, &(0x0006u16).to_le_bytes());
        let mut cmd = [0u8; 2];
        bus.ecam_read(cmd_gpa, &mut cmd);
        assert_eq!(u16::from_le_bytes(cmd) & 0x0006, 0x0006);

        // Vendor id is read-only: a write must not change it.
        bus.ecam_write(BASE + REG_VENDOR_ID as u64, &(0x1234u16).to_le_bytes());
        assert_eq!(
            read_u32(&bus, 0, REG_VENDOR_ID) & 0xFFFF,
            HOST_BRIDGE_VENDOR_ID as u32
        );
    }

    #[test]
    fn command_write_rejects_unwritable_bits() {
        let mut bus = PciBus::new(BASE, SIZE);
        let cmd_gpa = BASE + REG_COMMAND as u64;
        // Bit 3 (special cycles) is not in PCI_COMMAND_WMASK -> must stay 0.
        bus.ecam_write(cmd_gpa, &(0x0008u16).to_le_bytes());
        let mut cmd = [0u8; 2];
        bus.ecam_read(cmd_gpa, &mut cmd);
        assert_eq!(u16::from_le_bytes(cmd) & 0x0008, 0);
    }

    #[test]
    fn decode_ecam_extracts_bus_devfn_reg() {
        let offset = (3u64 << ECAM_BUS_SHIFT) | (0x1Au64 << ECAM_DEVFN_SHIFT) | 0x0AC;
        assert_eq!(decode_ecam(offset), (3, 0x1A, 0x0AC));
    }

    /// Build a type-1 CONFIG_ADDRESS for (bus, devfn, dword-aligned reg).
    fn cam_addr(bus: u8, devfn: u8, reg: u8) -> u32 {
        CAM_ENABLE_BIT | ((bus as u32) << 16) | ((devfn as u32) << 8) | (reg as u32 & 0xFC)
    }

    #[test]
    fn cam_reads_host_bridge_vendor() {
        let mut bus = PciBus::new(BASE, SIZE);
        bus.cam_set_address(cam_addr(0, 0, REG_VENDOR_ID as u8));
        assert_eq!(bus.cam_get_address() & CAM_ENABLE_BIT, CAM_ENABLE_BIT);
        let mut data = [0u8; 4];
        bus.cam_data_read(0, &mut data);
        let vendor_device = u32::from_le_bytes(data);
        assert_eq!(vendor_device & 0xFFFF, HOST_BRIDGE_VENDOR_ID as u32);
        assert_eq!(vendor_device >> 16, HOST_BRIDGE_DEVICE_ID as u32);
    }

    #[test]
    fn cam_disabled_latch_reads_all_ones() {
        let bus = PciBus::new(BASE, SIZE);
        // No address latched (enable bit clear) -> data window reads all-ones.
        let mut data = [0u8; 4];
        bus.cam_data_read(0, &mut data);
        assert_eq!(u32::from_le_bytes(data), 0xFFFF_FFFF);
    }

    #[test]
    fn cam_absent_function_reads_all_ones() {
        let mut bus = PciBus::new(BASE, SIZE);
        bus.cam_set_address(cam_addr(0, 1 << 3, REG_VENDOR_ID as u8));
        let mut data = [0u8; 4];
        bus.cam_data_read(0, &mut data);
        assert_eq!(u32::from_le_bytes(data), 0xFFFF_FFFF);
    }

    #[test]
    fn cam_byte_offset_selects_within_dword() {
        let mut bus = PciBus::new(BASE, SIZE);
        // Latch the class-code dword (reg 0x08); byte offset 3 selects the
        // base-class byte (0x0b) = bridge.
        bus.cam_set_address(cam_addr(0, 0, 0x08));
        let mut byte = [0u8; 1];
        bus.cam_data_read(3, &mut byte);
        assert_eq!(byte[0], CLASS_BRIDGE);
    }

    #[test]
    fn cam_hostile_misaligned_read_no_panic() {
        let mut bus = PciBus::new(BASE, SIZE);
        // Latch the last base-config dword (reg 0xFC) and read 4 bytes at byte
        // offset 3 — a misaligned multi-byte CONFIG_DATA access the kernel
        // never issues (reg = 0xFC + 3 = 0xFF, len 4 -> spans 0xFF..0x103).
        // Must not panic; ConfigSpace::read bounds-checks within the 4 KiB
        // backing and returns adjacent in-bounds bytes (NOT all-ones — this
        // access stays inside the 4 KiB space, unlike the straddle-past-end
        // case pinned by overread_past_config_space_is_all_ones_not_panic).
        bus.cam_set_address(cam_addr(0, 0, 0xFC));
        let mut data = [0u8; 4];
        bus.cam_data_read(3, &mut data);
        // Bytes 0xFF..0x103 of the host bridge config are unpopulated, so the
        // misaligned in-bounds read serves the real (zero) bytes, pinning that
        // the behavior is in-bounds-serve, not all-ones and not garbage.
        assert_eq!(u32::from_le_bytes(data), 0);
    }

    #[test]
    fn cam_decodes_extended_register_not_aliased_to_base() {
        let mut bus = PciBus::new(BASE, SIZE);
        // CONFIG_ADDRESS for extended register 0x100: the kernel encodes
        // reg[11:8] in bits [27:24] (PCI_CONF1_ADDRESS), so reg 0x100 sets
        // bit 24. bus 0, devfn 0, dword bits [7:2] = 0.
        let addr = CAM_ENABLE_BIT | ((0x100u32 & 0xF00) << 16);
        bus.cam_set_address(addr);
        let mut data = [0u8; 4];
        bus.cam_data_read(0, &mut data);
        // The host bridge's extended config (reg 0x100) is zero. A decode that
        // dropped bits [11:8] would alias this to base reg 0x00 and return the
        // vendor id 0x8086 — assert it does NOT.
        assert_eq!(u32::from_le_bytes(data), 0);
    }

    #[test]
    fn ecam_decodes_extended_register_not_aliased_to_base() {
        let bus = PciBus::new(BASE, SIZE);
        // ECAM read of extended register 0x100 of 00:00.0. decode_ecam
        // masks reg to the full 12 bits (ECAM_REG_MASK = 0xFFF), so 0x100
        // is a distinct register; the host bridge's extended config is
        // zero. A decode that truncated reg to the low 8 bits would alias
        // this to base reg 0x00 and return vendor 0x8086 — assert it does
        // NOT (the ECAM analog of cam_decodes_extended_register_*).
        assert_eq!(read_u32(&bus, 0, 0x100), 0);
        // Sanity: base reg 0 still returns the vendor id, proving 0x100
        // is not a wrapped/aliased base access.
        assert_eq!(
            read_u32(&bus, 0, REG_VENDOR_ID) & 0xFFFF,
            HOST_BRIDGE_VENDOR_ID as u32
        );
    }

    #[test]
    fn cam_and_ecam_share_config_space() {
        let mut bus = PciBus::new(BASE, SIZE);
        // Enable bus-master + memory via CAM, observe it via ECAM.
        bus.cam_set_address(cam_addr(0, 0, REG_COMMAND as u8));
        bus.cam_data_write(0, &(0x0006u32).to_le_bytes());
        let mut cmd = [0u8; 2];
        bus.ecam_read(BASE + REG_COMMAND as u64, &mut cmd);
        assert_eq!(u16::from_le_bytes(cmd) & 0x0006, 0x0006);
    }
}
