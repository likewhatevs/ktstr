//! Userspace IOAPIC (Intel 82093AA model) for the split-irqchip / >255-vCPU path.
//!
//! When a guest has more than 254 vCPUs, KVM runs in split-irqchip mode
//! (see [`super::kvm`]): the LAPIC stays in-kernel but the IOAPIC does not
//! exist in the kernel, because the in-kernel IOAPIC's redirection-entry
//! destination field is `u8` and cannot address an APIC ID > 255
//! (arch/x86/kvm/ioapic.h `kvm_ioapic_redirect_entry.fields.dest_id`).
//! Device interrupts must therefore be delivered as KVM MSI routes rather
//! than IOAPIC GSI pins — the MSI path carries a 32-bit destination
//! (arch/x86/kernel/apic/apic.c `x86_msi_msg_get_destid`).
//!
//! This type emulates the IOAPIC MMIO window the guest programs (at GPA
//! [`IOAPIC_BASE`], the address the MADT advertises). Each redirection-table
//! entry the guest writes is translated into an [`MsiRoute`] the caller
//! installs via `KVM_SET_GSI_ROUTING`, so an irqfd kick on the entry's GSI
//! is delivered as an MSI to the (possibly > 255) destination APIC.
//!
//! It is a **pure device**: it owns the register file and the RTE→MSI
//! translation but performs no KVM ioctls itself — the run loop drives
//! `set_gsi_routing` when [`Ioapic::mmio_write`] reports a routing change,
//! and services [`Ioapic::end_of_interrupt`] on a `KVM_EXIT_IOAPIC_EOI`
//! exit. Keeping the translation ioctl-free makes it unit-testable in
//! isolation. Reference model: cloud-hypervisor `devices/src/ioapic.rs`;
//! field/bit layout grounded in the guest's `struct IO_APIC_route_entry`
//! (arch/x86/include/asm/io_apic.h).
//!
//! v0 targets edge-triggered device pins (ktstr's virtio-MMIO + 8250 serial
//! IRQs). Edge pins raise NO `KVM_EXIT_IOAPIC_EOI`: KVM intercepts EOI only
//! for vectors in `ioapic_handled_vectors`, and `kvm_scan_ioapic_routes` adds
//! a vector there only when the route is level (`if (!irq.trig_mode)
//! continue`) — edge vectors are never EOI-intercepted (independent of
//! remote-IRR). The exit is still serviced defensively for any level entry
//! the guest may program: `end_of_interrupt` clears that entry's remote-IRR
//! (edge entries never have it set, so it is a no-op for them). Servicing the
//! exit matters because an unhandled `KVM_EXIT_IOAPIC_EOI` kills the vCPU (the
//! bug libkrun's IOAPIC has). Level-triggered re-injection (re-asserting a
//! still-pending line after EOI) is a follow-up if a level device is added.

/// IOAPIC MMIO window base (GPA). Matches the MADT IOAPIC address and the
/// guest kernel's `IO_APIC_DEFAULT_PHYS_BASE`.
pub(crate) const IOAPIC_BASE: u64 = 0xFEC0_0000;
/// Decoded MMIO window length. The IOAPIC occupies a full guest page; the
/// only live registers are the index latch (0x00), the data window (0x10)
/// and the EOI register (0x40).
pub(crate) const IOAPIC_SIZE: u64 = 0x1000;

/// Number of redirection-table entries (input pins) the 82093AA exposes.
pub(crate) const NUM_PINS: usize = 24;

// MMIO register offsets within the window (indirect-access scheme).
const IOREGSEL: u64 = 0x00; // index latch (selects the register IOWIN accesses)
const IOWIN: u64 = 0x10; // data window (reads/writes the selected register)
const IOEOI: u64 = 0x40; // write-only EOI register (valid when version >= 0x20)

// Register indices selectable through IOREGSEL.
const REG_ID: u32 = 0x00;
const REG_VER: u32 = 0x01;
const REG_REDTBL_BASE: u32 = 0x10; // entry i: low dword 0x10+2i, high dword 0x11+2i

/// IOAPIC version. 0x20 advertises the EOI register; the high byte of the
/// VER register reports the max redirection entry (NUM_PINS - 1).
const IOAPIC_VERSION: u32 = 0x20;

// Redirection-table entry (64-bit) bit fields, per the guest's
// `struct IO_APIC_route_entry` (arch/x86/include/asm/io_apic.h):
//   vector[7:0] delivery_mode[10:8] dest_mode_logical[11] delivery_status[12]
//   active_low[13] irr(remote_IRR)[14] is_level[15] masked[16]
//   virt_destid_8_14[55:49] destid_0_7[63:56].
const RTE_VECTOR_MASK: u64 = 0xff;
const RTE_DELIV_MODE_SHIFT: u64 = 8;
const RTE_DELIV_MODE_MASK: u64 = 0x7;
const RTE_DEST_MODE_BIT: u64 = 1 << 11;
const RTE_DELIV_STATUS_BIT: u64 = 1 << 12; // read-only
const RTE_REMOTE_IRR_BIT: u64 = 1 << 14; // read-only
const RTE_TRIGGER_LEVEL_BIT: u64 = 1 << 15; // is_level
const RTE_MASKED_BIT: u64 = 1 << 16;
const RTE_DESTID_0_7_SHIFT: u64 = 56;
const RTE_VIRT_DESTID_8_14_SHIFT: u64 = 49;
const RTE_VIRT_DESTID_8_14_MASK: u64 = 0x7f;
/// Bits the guest must not write directly — delivery_status and remote_IRR
/// are status bits the IOAPIC owns.
const RTE_RO_BITS: u64 = RTE_DELIV_STATUS_BIT | RTE_REMOTE_IRR_BIT;

/// LAPIC MSI address base (`APIC_DEFAULT_PHYS_BASE`). The 15-bit destination's
/// low 8 bits go to address_lo\[19:12\]; bits \[14:8\] go to address_hi\[31:8\] (the
/// `destid_8_31` field), which is the extended-destination-ID path KVM decodes
/// when `KVM_FEATURE_MSI_EXT_DEST_ID` + x2APIC-API 32-bit IDs are enabled.
const MSI_ADDR_BASE: u32 = 0xFEE0_0000;

/// An MSI message derived from a redirection-table entry, ready to be placed
/// in a `kvm_irq_routing_msi` entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct MsiRoute {
    pub address_lo: u32,
    pub address_hi: u32,
    pub data: u32,
}

/// Emulated IOAPIC register file.
pub(crate) struct Ioapic {
    /// The register index latched by a write to IOREGSEL.
    ioregsel: u8,
    /// IOAPIC ID (REG_ID bits \[27:24\]).
    id: u32,
    /// The 24 redirection-table entries.
    ioredtbl: [u64; NUM_PINS],
}

impl Ioapic {
    /// Reset state: every entry masked, matching the hardware power-on
    /// state. The guest unmasks an entry when it programs an IRQ.
    pub(crate) fn new() -> Self {
        Ioapic {
            ioregsel: 0,
            id: 0,
            ioredtbl: [RTE_MASKED_BIT; NUM_PINS],
        }
    }

    /// Handle a guest MMIO read from the IOAPIC window. `offset` is relative
    /// to [`IOAPIC_BASE`]. Fills `data` little-endian; unknown registers and
    /// out-of-range entries read back 0 (matching qemu).
    pub(crate) fn mmio_read(&self, offset: u64, data: &mut [u8]) {
        let val = match offset {
            IOREGSEL => self.ioregsel as u32,
            IOWIN => self.read_indirect(self.ioregsel as u32),
            // The EOI register is write-only; reads return 0.
            _ => 0,
        };
        write_u32_le(data, val);
    }

    /// Handle a guest MMIO write to the IOAPIC window. Returns `true` if a
    /// redirection-table entry changed and the caller must rebuild the KVM
    /// MSI routing table.
    pub(crate) fn mmio_write(&mut self, offset: u64, data: &[u8]) -> bool {
        let val = read_u32_le(data);
        match offset {
            IOREGSEL => {
                self.ioregsel = val as u8;
                false
            }
            IOWIN => self.write_indirect(self.ioregsel as u32, val),
            IOEOI => {
                // Erratum / CPU-offline EOI path: the guest writes the
                // acked vector here when the IOAPIC version is >= 0x20.
                self.end_of_interrupt(val as u8);
                false
            }
            _ => false,
        }
    }

    fn read_indirect(&self, reg: u32) -> u32 {
        match reg {
            REG_ID => self.id << 24,
            REG_VER => IOAPIC_VERSION | (((NUM_PINS as u32) - 1) << 16),
            _ => {
                // Redirection-table low/high dwords.
                let (index, high) = redtbl_index(reg);
                match self.ioredtbl.get(index) {
                    Some(&entry) if high => (entry >> 32) as u32,
                    Some(&entry) => entry as u32,
                    None => 0,
                }
            }
        }
    }

    fn write_indirect(&mut self, reg: u32, val: u32) -> bool {
        match reg {
            REG_ID => {
                self.id = (val >> 24) & 0xf;
                false
            }
            REG_VER => false, // version/arb are read-only
            _ => {
                let (index, high) = redtbl_index(reg);
                let Some(entry) = self.ioredtbl.get_mut(index) else {
                    return false;
                };
                // Merge the written dword, preserving the IOAPIC-owned
                // read-only status bits (delivery_status, remote_IRR).
                let ro = *entry & RTE_RO_BITS;
                if high {
                    *entry = (*entry & 0x0000_0000_ffff_ffff) | ((val as u64) << 32);
                } else {
                    *entry = (*entry & 0xffff_ffff_0000_0000) | (val as u64);
                }
                *entry = (*entry & !RTE_RO_BITS) | ro;
                true
            }
        }
    }

    /// Translate redirection-table entry `pin` into its MSI message. The
    /// destination is reconstructed from the guest's split
    /// `destid_0_7`/`virt_destid_8_14` fields and re-split into the MSI
    /// `address_lo`/`address_hi` extended-destination encoding KVM decodes.
    pub(crate) fn redtbl_to_msi(&self, pin: usize) -> MsiRoute {
        let entry = self.ioredtbl[pin];
        let vector = (entry & RTE_VECTOR_MASK) as u32;
        let delivery_mode = ((entry >> RTE_DELIV_MODE_SHIFT) & RTE_DELIV_MODE_MASK) as u32;
        let dest_mode = ((entry & RTE_DEST_MODE_BIT) != 0) as u32;
        let trigger_level = ((entry & RTE_TRIGGER_LEVEL_BIT) != 0) as u32;
        let dest = self.destination(pin);

        // redirect_hint (address_lo bit 3) is left 0: it only steers
        // lowest-priority delivery to the least-loaded CPU. We forward the
        // guest's delivery_mode verbatim, so fixed/physical edge device IRQs
        // (v0) never use it.
        let address_lo =
            MSI_ADDR_BASE | ((dest & 0xff) << 12) | (dest_mode << 2);
        let address_hi = (dest >> 8) << 8;
        let data = vector | (delivery_mode << 8) | (trigger_level << 15);
        MsiRoute {
            address_lo,
            address_hi,
            data,
        }
    }

    /// The 15-bit destination APIC ID programmed in entry `pin`, recombined
    /// from the guest's `destid_0_7` (bits 7:0) and `virt_destid_8_14`
    /// (bits 14:8) fields.
    pub(crate) fn destination(&self, pin: usize) -> u32 {
        let entry = self.ioredtbl[pin];
        let destid_0_7 = ((entry >> RTE_DESTID_0_7_SHIFT) & 0xff) as u32;
        let virt_destid_8_14 =
            ((entry >> RTE_VIRT_DESTID_8_14_SHIFT) & RTE_VIRT_DESTID_8_14_MASK) as u32;
        destid_0_7 | (virt_destid_8_14 << 8)
    }

    /// Whether entry `pin` is masked (the guest has not enabled it).
    pub(crate) fn is_masked(&self, pin: usize) -> bool {
        self.ioredtbl[pin] & RTE_MASKED_BIT != 0
    }

    /// Service an end-of-interrupt for `vector` (delivered as a
    /// `KVM_EXIT_IOAPIC_EOI` exit, or via a write to the EOI register).
    /// Clears remote-IRR on every level-triggered entry carrying that vector
    /// so the guest's level IRQ is not wedged. Edge entries never set
    /// remote-IRR, so this is a no-op for them. Returns the pins whose line
    /// may need re-injection (none in the edge-only v0; reserved for level
    /// support).
    pub(crate) fn end_of_interrupt(&mut self, vector: u8) -> Vec<usize> {
        let mut pending = Vec::new();
        for (pin, entry) in self.ioredtbl.iter_mut().enumerate() {
            let matches_vector = (*entry & RTE_VECTOR_MASK) as u8 == vector;
            let level = *entry & RTE_TRIGGER_LEVEL_BIT != 0;
            if matches_vector && level && (*entry & RTE_REMOTE_IRR_BIT != 0) {
                *entry &= !RTE_REMOTE_IRR_BIT;
                if *entry & RTE_MASKED_BIT == 0 {
                    pending.push(pin);
                }
            }
        }
        pending
    }

    /// `(gsi, MsiRoute)` for every UNMASKED redirection entry (gsi == pin).
    /// `KVM_SET_GSI_ROUTING` replaces the whole table, so the caller installs
    /// this full set on each routing change; masked pins are omitted (no
    /// route -> an irqfd kick on that pin is dropped, matching the mask).
    pub(crate) fn gsi_routes(&self) -> Vec<(u32, MsiRoute)> {
        (0..NUM_PINS)
            .filter(|&pin| !self.is_masked(pin))
            .map(|pin| (pin as u32, self.redtbl_to_msi(pin)))
            .collect()
    }
}

/// Decode a redirection-table register index into (entry, is_high_dword).
fn redtbl_index(reg: u32) -> (usize, bool) {
    let off = reg.wrapping_sub(REG_REDTBL_BASE);
    ((off >> 1) as usize, off & 1 == 1)
}

/// Read a little-endian u32 from the front of `data` (zero-extended if the
/// guest issued a narrower access).
fn read_u32_le(data: &[u8]) -> u32 {
    let mut buf = [0u8; 4];
    let n = data.len().min(4);
    buf[..n].copy_from_slice(&data[..n]);
    u32::from_le_bytes(buf)
}

/// Write a little-endian u32 into the front of `data` (truncated if the
/// guest issued a narrower access).
fn write_u32_le(data: &mut [u8], val: u32) {
    let bytes = val.to_le_bytes();
    let n = data.len().min(4);
    data[..n].copy_from_slice(&bytes[..n]);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Program redirection-table entry `pin` via the indirect MMIO window,
    /// returning whether the write reported a routing change.
    fn write_rte(io: &mut Ioapic, pin: usize, entry: u64) -> bool {
        let lo_reg = (REG_REDTBL_BASE + 2 * pin as u32) as u8;
        let hi_reg = lo_reg + 1;
        io.mmio_write(IOREGSEL, &[lo_reg]);
        let dirty_lo = io.mmio_write(IOWIN, &(entry as u32).to_le_bytes());
        io.mmio_write(IOREGSEL, &[hi_reg]);
        let dirty_hi = io.mmio_write(IOWIN, &((entry >> 32) as u32).to_le_bytes());
        dirty_lo || dirty_hi
    }

    #[test]
    fn version_register_reports_24_entries() {
        let mut io = Ioapic::new();
        io.mmio_write(IOREGSEL, &[REG_VER as u8]);
        let mut buf = [0u8; 4];
        io.mmio_read(IOWIN, &mut buf);
        let ver = u32::from_le_bytes(buf);
        assert_eq!(ver & 0xff, IOAPIC_VERSION, "version byte");
        assert_eq!((ver >> 16) & 0xff, (NUM_PINS as u32) - 1, "max redir entry");
    }

    #[test]
    fn entries_reset_masked() {
        let io = Ioapic::new();
        for pin in 0..NUM_PINS {
            assert!(io.is_masked(pin), "pin {pin} must reset masked");
        }
    }

    #[test]
    fn gsi_routes_includes_unmasked_omits_masked() {
        let mut io = Ioapic::new();
        // All pins reset masked, so the route set starts empty.
        assert!(
            io.gsi_routes().is_empty(),
            "all-masked IOAPIC must yield no routes"
        );
        // Unmask pin 4 (vector 0x30, dest 1) and pin 6 (vector 0x40, dest 2);
        // masked bit (16) clear, dest in destid_0_7 ([63:56]).
        let rte4 = 0x30u64 | (1u64 << RTE_DESTID_0_7_SHIFT);
        let rte6 = 0x40u64 | (2u64 << RTE_DESTID_0_7_SHIFT);
        write_rte(&mut io, 4, rte4);
        write_rte(&mut io, 6, rte6);
        // Re-mask pin 4 (set bit 16); pin 6 stays unmasked.
        write_rte(&mut io, 4, rte4 | RTE_MASKED_BIT);
        let routes = io.gsi_routes();
        let gsis: Vec<u32> = routes.iter().map(|(g, _)| *g).collect();
        assert_eq!(routes.len(), 1, "only the unmasked pin is routed; got {gsis:?}");
        assert!(gsis.contains(&6), "unmasked pin 6 must be routed; got {gsis:?}");
        assert!(!gsis.contains(&4), "re-masked pin 4 must be omitted; got {gsis:?}");
        let (_, msi) = routes.iter().find(|(g, _)| *g == 6).expect("pin 6 route");
        assert_eq!(msi.data & 0xff, 0x40, "pin 6 MSI carries vector 0x40");
    }

    #[test]
    fn ioregsel_roundtrips() {
        let mut io = Ioapic::new();
        io.mmio_write(IOREGSEL, &[0x12]);
        let mut buf = [0u8; 4];
        io.mmio_read(IOREGSEL, &mut buf);
        assert_eq!(u32::from_le_bytes(buf) & 0xff, 0x12);
    }

    #[test]
    fn rte_write_is_routing_dirty_and_reads_back() {
        let mut io = Ioapic::new();
        // vector 0x33, fixed delivery, physical dest, edge, unmasked, dest 0.
        let entry: u64 = 0x33;
        let dirty = write_rte(&mut io, 6, entry);
        assert!(dirty, "RTE write must report a routing change");
        // Read back the low dword.
        io.mmio_write(IOREGSEL, &[(REG_REDTBL_BASE + 12) as u8]);
        let mut buf = [0u8; 4];
        io.mmio_read(IOWIN, &mut buf);
        assert_eq!(u32::from_le_bytes(buf) & 0xff, 0x33);
    }

    #[test]
    fn ioregsel_write_is_not_routing_dirty() {
        let mut io = Ioapic::new();
        assert!(!io.mmio_write(IOREGSEL, &[0x10]), "selecting a reg is not a route change");
    }

    #[test]
    fn translate_vector_delivery_trigger() {
        let mut io = Ioapic::new();
        // vector 0x42, delivery_mode 0b001 (lowest-pri), level-triggered.
        let entry: u64 = 0x42 | (0b001 << 8) | RTE_TRIGGER_LEVEL_BIT;
        write_rte(&mut io, 5, entry);
        let msi = io.redtbl_to_msi(5);
        assert_eq!(msi.data & 0xff, 0x42, "vector");
        assert_eq!((msi.data >> 8) & 0x7, 0b001, "delivery mode");
        assert_eq!((msi.data >> 15) & 1, 1, "trigger level");
    }

    #[test]
    fn translate_low_destination() {
        let mut io = Ioapic::new();
        // dest 0x07 in destid_0_7, physical mode, vector 0x20.
        let entry: u64 = 0x20 | (0x07u64 << RTE_DESTID_0_7_SHIFT);
        write_rte(&mut io, 7, entry);
        assert_eq!(io.destination(7), 0x07);
        let msi = io.redtbl_to_msi(7);
        // dest low 8 bits -> address_lo[19:12]; physical mode -> bit2 clear.
        assert_eq!((msi.address_lo >> 12) & 0xff, 0x07);
        assert_eq!(msi.address_lo & MSI_ADDR_BASE, MSI_ADDR_BASE);
        assert_eq!(msi.address_lo & (1 << 2), 0, "physical dest mode");
        assert_eq!(msi.address_hi, 0, "no high dest bits for dest<256");
    }

    #[test]
    fn translate_wide_destination_above_255() {
        let mut io = Ioapic::new();
        // dest = 300 (0x12C): destid_0_7 = 0x2C, virt_destid_8_14 = 0x01.
        let dest: u32 = 300;
        let destid_0_7 = (dest & 0xff) as u64;
        let virt_destid_8_14 = ((dest >> 8) & 0x7f) as u64;
        let entry: u64 = 0x20
            | (destid_0_7 << RTE_DESTID_0_7_SHIFT)
            | (virt_destid_8_14 << RTE_VIRT_DESTID_8_14_SHIFT);
        write_rte(&mut io, 6, entry);
        assert_eq!(io.destination(6), 300, "reconstructed dest");
        let msi = io.redtbl_to_msi(6);
        // KVM decode: dest = addr_lo[19:12] | (addr_hi[31:8] << 8).
        let decoded = ((msi.address_lo >> 12) & 0xff) | (((msi.address_hi >> 8) & 0xff_ffff) << 8);
        assert_eq!(decoded, 300, "MSI must round-trip the >255 destination");
        assert_eq!(msi.address_hi & 0xff, 0, "addr_hi low byte must be zero (KVM requires it)");
    }

    #[test]
    fn logical_dest_mode_sets_address_bit() {
        let mut io = Ioapic::new();
        let entry: u64 = 0x20 | RTE_DEST_MODE_BIT;
        write_rte(&mut io, 5, entry);
        let msi = io.redtbl_to_msi(5);
        assert_ne!(msi.address_lo & (1 << 2), 0, "logical dest mode -> address_lo bit 2");
    }

    #[test]
    fn remote_irr_is_read_only_to_guest() {
        let mut io = Ioapic::new();
        // Guest attempts to set remote_IRR (bit 14) + delivery_status (12).
        let entry: u64 = 0x20 | RTE_REMOTE_IRR_BIT | RTE_DELIV_STATUS_BIT;
        write_rte(&mut io, 6, entry);
        assert_eq!(io.ioredtbl[6] & RTE_REMOTE_IRR_BIT, 0, "remote_IRR must not be guest-writable");
        assert_eq!(io.ioredtbl[6] & RTE_DELIV_STATUS_BIT, 0, "delivery_status must not be guest-writable");
    }

    #[test]
    fn eoi_clears_remote_irr_on_matching_level_entry() {
        let mut io = Ioapic::new();
        // Level entry, vector 0x50, unmasked.
        let entry: u64 = 0x50 | RTE_TRIGGER_LEVEL_BIT;
        write_rte(&mut io, 6, entry);
        // Simulate the IOAPIC having set remote_IRR on level delivery.
        io.ioredtbl[6] |= RTE_REMOTE_IRR_BIT;
        let pending = io.end_of_interrupt(0x50);
        assert_eq!(io.ioredtbl[6] & RTE_REMOTE_IRR_BIT, 0, "EOI clears remote_IRR");
        assert_eq!(pending, vec![6], "unmasked still-level pin reported for re-injection");
    }

    #[test]
    fn eoi_for_edge_entry_is_noop() {
        let mut io = Ioapic::new();
        let entry: u64 = 0x60; // edge, vector 0x60
        write_rte(&mut io, 7, entry);
        let pending = io.end_of_interrupt(0x60);
        assert!(pending.is_empty(), "edge EOI re-injects nothing");
    }

    #[test]
    fn eoi_unknown_vector_is_harmless() {
        let mut io = Ioapic::new();
        // No entry carries vector 0xAB; must not panic or report work.
        let pending = io.end_of_interrupt(0xAB);
        assert!(pending.is_empty());
    }

    #[test]
    fn out_of_range_rte_read_is_zero_not_panic() {
        let mut io = Ioapic::new();
        // Select a register past the redirection table.
        io.mmio_write(IOREGSEL, &[0xfe]);
        let mut buf = [0u8; 4];
        io.mmio_read(IOWIN, &mut buf);
        assert_eq!(u32::from_le_bytes(buf), 0);
    }

    #[test]
    fn narrow_and_wide_access_helpers_are_total() {
        // Byte access must not panic; u32 read zero-extends.
        assert_eq!(read_u32_le(&[0xaa]), 0x0000_00aa);
        let mut one = [0u8; 1];
        write_u32_le(&mut one, 0x1122_3344);
        assert_eq!(one[0], 0x44, "LE low byte");
    }

    use proptest::prelude::*;

    proptest! {
        /// Arbitrary guest RTE writes must never panic, must never let the
        /// guest set the read-only status bits, and must round-trip the
        /// destination through the MSI encoding KVM decodes
        /// (x86_msi_msg_get_destid).
        #[test]
        fn arbitrary_rte_is_safe(raw in any::<u64>(), pin in 0usize..NUM_PINS) {
            let mut io = Ioapic::new();
            write_rte(&mut io, pin, raw);
            prop_assert_eq!(io.ioredtbl[pin] & RTE_RO_BITS, 0, "guest must not set RO bits");
            let msi = io.redtbl_to_msi(pin);
            let dest = io.destination(pin);
            let decoded =
                ((msi.address_lo >> 12) & 0xff) | (((msi.address_hi >> 8) & 0x00ff_ffff) << 8);
            prop_assert_eq!(decoded, dest, "MSI must round-trip the destination");
            prop_assert_eq!(msi.address_hi & 0xff, 0, "addr_hi low byte must be zero (KVM requires it)");
            // EOI for any vector must never panic.
            let _ = io.end_of_interrupt((raw & 0xff) as u8);
        }

        /// Arbitrary MMIO accesses (any offset in the window, any data) must
        /// never panic.
        #[test]
        fn arbitrary_mmio_is_safe(offset in 0u64..IOAPIC_SIZE, val in any::<u32>()) {
            let mut io = Ioapic::new();
            let _ = io.mmio_write(offset, &val.to_le_bytes());
            let mut buf = [0u8; 4];
            io.mmio_read(offset, &mut buf);
        }
    }
}
