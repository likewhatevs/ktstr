//! Minimal ACPI AML byte encoder for the DSDT PCI host bridge node.
//!
//! ktstr hand-rolls all ACPI tables (no `aml` crate), so this module encodes
//! exactly the AML the guest's interpreter needs to bind a single-segment,
//! single-bus PCIe host bridge: a `_SB.PCI0` device (`_HID` PNP0A08 / `_CID`
//! PNP0A03) advertising bus range \[0,0\] and the 32-bit PCI MMIO window, plus
//! a PNP0C02 motherboard-resources device reserving the ECAM window so the OS
//! does not reuse it. Byte layouts follow the ACPI 6.x spec and qemu
//! hw/acpi/aml-build.c. All multi-byte integers are little-endian.
//!
//! Encoding is inner-first: a package's PkgLength depends on its body length,
//! so each combinator builds the body to a `Vec<u8>` then prepends the
//! opcode(s) + PkgLength.

/// Encode a 4-byte NameSeg, `_`-padded (e.g. "_HID" -> 5F 48 49 44, "PCI0").
fn nameseg(s: &str) -> [u8; 4] {
    let mut b = [b'_'; 4];
    let bytes = s.as_bytes();
    b[..bytes.len()].copy_from_slice(bytes);
    b
}

/// Encode an AML PkgLength for a package whose CONTENT (the bytes following
/// the PkgLength field) is `content_len` bytes. The encoded length value
/// includes the PkgLength bytes themselves (ACPI 6.4 §20.2.4; qemu
/// `build_prepend_package_length` with `incl_self`).
fn pkg_length(content_len: usize) -> Vec<u8> {
    if content_len + 1 < 0x40 {
        // 1-byte form: total in the low 6 bits (top 2 bits = 0b00).
        vec![(content_len + 1) as u8]
    } else if content_len + 2 < 0x1000 {
        // 2-byte form (top 2 bits = 0b01).
        let total = content_len + 2;
        vec![0x40 | (total & 0x0F) as u8, (total >> 4) as u8]
    } else {
        // 3-byte form (top 2 bits = 0b10). The DSDT body here is far under
        // 1 MiB, so the 4-byte form is never needed.
        let total = content_len + 3;
        vec![
            0x80 | (total & 0x0F) as u8,
            (total >> 4) as u8,
            (total >> 12) as u8,
        ]
    }
}

/// Encode an AML integer constant (ZeroOp/OneOp/Byte/Word/DWord/QWord).
fn int_obj(v: u64) -> Vec<u8> {
    match v {
        0 => vec![0x00], // ZeroOp
        1 => vec![0x01], // OneOp
        x if x <= 0xFF => vec![0x0A, x as u8],
        x if x <= 0xFFFF => {
            let mut o = vec![0x0B];
            o.extend_from_slice(&(x as u16).to_le_bytes());
            o
        }
        x if x <= 0xFFFF_FFFF => {
            let mut o = vec![0x0C];
            o.extend_from_slice(&(x as u32).to_le_bytes());
            o
        }
        x => {
            let mut o = vec![0x0E];
            o.extend_from_slice(&x.to_le_bytes());
            o
        }
    }
}

/// Encode `Name(seg, <int>)`: NameOp (0x08) + NameSeg + integer object.
fn name_int(name: &str, v: u64) -> Vec<u8> {
    let mut o = vec![0x08];
    o.extend_from_slice(&nameseg(name));
    o.extend(int_obj(v));
    o
}

/// Encode `Name(seg, <data object>)`: NameOp (0x08) + NameSeg + raw object.
fn name_data(name: &str, data: &[u8]) -> Vec<u8> {
    let mut o = vec![0x08];
    o.extend_from_slice(&nameseg(name));
    o.extend_from_slice(data);
    o
}

/// Encode an EisaId DWord object for a 7-char PNP id (e.g. "PNP0A08").
///
/// The 7 characters pack into a 32-bit value: 3 compressed-ASCII letters
/// (5 bits each, char − 0x40) in bits \[30:16\], then 4 hex nibbles in
/// \[15:0\] — "PNP0A08" packs to 0x41D00A08. The DWord constant is emitted
/// BIG-ENDIAN (the packed value high-to-low: `0C 41 D0 0A 08`). This is
/// load-bearing: the AML interpreter reads the DWord little-endian (0x080AD041),
/// then ACPICA byte-swaps it back to 0x41D00A08
/// (acpi_ex_eisa_id_to_string -> acpi_ut_dword_byte_swap, drivers/acpi/acpica/
/// exutils.c + utmisc.c) and unpacks that to "PNP0A08". Emitting little-endian
/// instead would make ACPICA decode a garbage id and `_HID` would never match,
/// so the guest would not bind the PCI host bridge.
fn eisaid(id: &str) -> Vec<u8> {
    let b = id.as_bytes();
    debug_assert!(b.len() >= 7, "EISA id must be 7 chars, got {id:?}");
    let hex = |c: u8| if c.is_ascii_digit() { c - b'0' } else { c - b'A' + 10 };
    let packed: u32 = ((b[0] - 0x40) as u32) << 26
        | ((b[1] - 0x40) as u32) << 21
        | ((b[2] - 0x40) as u32) << 16
        | (hex(b[3]) as u32) << 12
        | (hex(b[4]) as u32) << 8
        | (hex(b[5]) as u32) << 4
        | (hex(b[6]) as u32);
    let mut o = vec![0x0C]; // DWordPrefix — EisaId is always a DWord constant.
    // Big-endian: ACPICA byte-swaps the LE-read DWord before unpacking, so the
    // on-wire bytes are the packed value high-to-low.
    o.extend_from_slice(&packed.to_be_bytes());
    o
}

/// Wrap `body` in `Device(seg) { body }`: ExtOpPrefix(0x5B) DeviceOp(0x82)
/// PkgLength NameSeg body. The PkgLength counts NameSeg + body (the 0x5B 0x82
/// prefix is outside it).
fn device(name: &str, body: &[u8]) -> Vec<u8> {
    let mut inner = Vec::with_capacity(4 + body.len());
    inner.extend_from_slice(&nameseg(name));
    inner.extend_from_slice(body);
    let mut o = vec![0x5B, 0x82];
    o.extend(pkg_length(inner.len()));
    o.extend(inner);
    o
}

/// Wrap `body` in `Scope(\seg) { body }`: ScopeOp(0x10) PkgLength NameString
/// body, where the NameString is root-anchored single-seg (e.g. "\_SB").
fn root_scope(name: &str, body: &[u8]) -> Vec<u8> {
    let mut inner = vec![0x5C]; // RootChar '\'
    inner.extend_from_slice(&nameseg(name));
    inner.extend_from_slice(body);
    let mut o = vec![0x10];
    o.extend(pkg_length(inner.len()));
    o.extend(inner);
    o
}

/// Wrap resource descriptors in a `ResourceTemplate() { ... }` buffer:
/// BufferOp(0x11) PkgLength <bufsize int> <descriptors> EndTag(0x79) 0x00.
/// The trailing 0x00 is the EndTag checksum byte (a zero checksum is accepted
/// per ACPI 6.4 §6.4.2.9); omitting it shifts every following byte and the
/// guest's _CRS parse fails silently.
fn resource_template(descriptors: &[u8]) -> Vec<u8> {
    let mut buf = Vec::with_capacity(descriptors.len() + 2);
    buf.extend_from_slice(descriptors);
    buf.push(0x79); // EndTag
    buf.push(0x00); // EndTag checksum (zero accepted)
    let mut inner = int_obj(buf.len() as u64);
    inner.extend_from_slice(&buf);
    let mut o = vec![0x11];
    o.extend(pkg_length(inner.len()));
    o.extend(inner);
    o
}

/// WordBusNumber address-space descriptor for the producer bus range
/// \[min, max\] (ACPI 6.4 §6.4.3.5.3).
fn word_bus_number(min: u16, max: u16) -> Vec<u8> {
    debug_assert!(max >= min, "bus range max {max} < min {min}");
    let mut o = vec![0x88]; // Word Address Space Descriptor
    o.extend_from_slice(&0x000Du16.to_le_bytes()); // descriptor length = 13
    o.push(0x02); // resource type: bus number range
    o.push(0x0C); // flags: MIN_FIXED | MAX_FIXED | POS_DECODE
    o.push(0x00); // type-specific flags
    o.extend_from_slice(&0u16.to_le_bytes()); // granularity
    o.extend_from_slice(&min.to_le_bytes()); // min
    o.extend_from_slice(&max.to_le_bytes()); // max
    o.extend_from_slice(&0u16.to_le_bytes()); // translation offset
    o.extend_from_slice(&(max - min + 1).to_le_bytes()); // length
    o
}

/// DWordMemory address-space descriptor for a fixed, read/write,
/// non-cacheable 32-bit memory window \[min, max\] (ACPI 6.4 §6.4.3.5.2).
fn dword_memory(min: u32, max: u32) -> Vec<u8> {
    debug_assert!(max >= min, "memory window max {max:#x} < min {min:#x}");
    let mut o = vec![0x87]; // DWord Address Space Descriptor
    o.extend_from_slice(&0x0017u16.to_le_bytes()); // descriptor length = 23
    o.push(0x00); // resource type: memory range
    o.push(0x0C); // flags: MIN_FIXED | MAX_FIXED | POS_DECODE
    o.push(0x01); // type flags: read/write, non-cacheable
    o.extend_from_slice(&0u32.to_le_bytes()); // granularity
    o.extend_from_slice(&min.to_le_bytes()); // min
    o.extend_from_slice(&max.to_le_bytes()); // max
    o.extend_from_slice(&0u32.to_le_bytes()); // translation offset
    o.extend_from_slice(&(max - min + 1).to_le_bytes()); // length
    o
}

/// Memory32Fixed descriptor (read/write) for `[base, base+len)` — the simplest
/// fixed 32-bit memory reservation (ACPI 6.4 §6.4.3.4).
fn memory32_fixed(base: u32, len: u32) -> Vec<u8> {
    let mut o = vec![0x86]; // 32-bit Fixed Memory Range Descriptor
    o.extend_from_slice(&0x0009u16.to_le_bytes()); // descriptor length = 9
    o.push(0x01); // read/write
    o.extend_from_slice(&base.to_le_bytes());
    o.extend_from_slice(&len.to_le_bytes());
    o
}

/// Build the DSDT body (everything after the 36-byte SDT header) for a
/// single-segment, single-bus, empty PCIe host bridge:
/// `Scope(\_SB) { Device(PCI0){...}  Device(PCIR){PNP0C02 ECAM reservation} }`.
///
/// - `ecam_base`/`ecam_size`: the ECAM window reserved by the PNP0C02 device.
/// - `bar_base`/`bar_last`: the inclusive 32-bit PCI MMIO BAR window granted in
///   PCI0 `_CRS` (where the guest assigns device BARs).
///
/// Intentional divergences from cloud-hypervisor (documented per the
/// reference-impl-divergence rule; none are bugs):
/// - ECAM is reserved via a separate PNP0C02 motherboard-resources device (the
///   qemu idiom) rather than a Memory32Fixed inside PCI0 `_CRS`; both keep the
///   OS from re-using the ECAM range.
/// - `_CRS` grants no I/O-port window: virtio-pci NICs use MMIO BARs only. A
///   future PCI device needing a PIO BAR would add an I/O window here
///   (excluding 0xCF8-0xCFF, the type-1 CAM ports).
/// - `_CRS` grants only a 32-bit MMIO window: phase-1 virtio-net BARs fit below
///   4 GiB. A >32-bit or prefetchable-64-bit BAR would need a QWordMemory
///   window above RAM (a later increment).
pub(super) fn pci_host_bridge_dsdt_body(
    ecam_base: u32,
    ecam_size: u32,
    bar_base: u32,
    bar_last: u32,
) -> Vec<u8> {
    // Device(PCI0) — the PCIe host bridge.
    let mut pci0 = Vec::new();
    pci0.extend(name_data("_HID", &eisaid("PNP0A08"))); // PCIe ECAM host bridge
    pci0.extend(name_data("_CID", &eisaid("PNP0A03"))); // legacy PCI fallback
    pci0.extend(name_int("_SEG", 0));
    pci0.extend(name_int("_UID", 0));
    pci0.extend(name_int("_CCA", 1)); // DMA cache-coherent (required on ARM, benign on x86)
    pci0.extend(name_int("_BBN", 0)); // base bus number 0
    let mut crs = Vec::new();
    crs.extend(word_bus_number(0, 0)); // single bus 0
    crs.extend(dword_memory(bar_base, bar_last)); // 32-bit BAR window
    pci0.extend(name_data("_CRS", &resource_template(&crs)));
    let pci0_dev = device("PCI0", &pci0);

    // Device(PCIR) — PNP0C02 motherboard resources reserving the ECAM window
    // so the OS does not hand it out as general MMIO.
    let mut pcir = Vec::new();
    pcir.extend(name_data("_HID", &eisaid("PNP0C02")));
    pcir.extend(name_data(
        "_CRS",
        &resource_template(&memory32_fixed(ecam_base, ecam_size)),
    ));
    let pcir_dev = device("PCIR", &pcir);

    let mut sb = pci0_dev;
    sb.extend(pcir_dev);
    root_scope("_SB", &sb)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eisaid_canonical_bytes() {
        // EisaId = the packed value emitted big-endian (ACPICA byte-swaps the
        // LE-read DWord before unpacking). PNP0A08 packs to 0x41D00A08 ->
        // on-wire bytes 41 D0 0A 08.
        assert_eq!(eisaid("PNP0A08"), [0x0C, 0x41, 0xD0, 0x0A, 0x08]);
        assert_eq!(eisaid("PNP0A03"), [0x0C, 0x41, 0xD0, 0x0A, 0x03]);
        assert_eq!(eisaid("PNP0C02"), [0x0C, 0x41, 0xD0, 0x0C, 0x02]);
    }

    /// Decode an EisaId DWord the way ACPICA does
    /// (acpi_ex_eisa_id_to_string): read the 4 bytes little-endian, byte-swap,
    /// then unpack 3 letters (5 bits each from bits 26/21/16) + 4 hex nibbles.
    fn decode_eisaid(dword_bytes: [u8; 4]) -> String {
        let swapped = u32::from_le_bytes(dword_bytes).swap_bytes();
        let letter = |shift: u32| (b'@' + ((swapped >> shift) & 0x1F) as u8) as char;
        let nibble = |shift: u32| {
            char::from_digit((swapped >> shift) & 0xF, 16)
                .unwrap()
                .to_ascii_uppercase()
        };
        format!(
            "{}{}{}{}{}{}{}",
            letter(26),
            letter(21),
            letter(16),
            nibble(12),
            nibble(8),
            nibble(4),
            nibble(0),
        )
    }

    #[test]
    fn eisaid_decodes_to_pnp_id() {
        // Pins MEANING, not just bytes: mirror ACPICA's decode and assert the
        // emitted DWord round-trips to the PNP id. A byte-order bug (the kind a
        // byte-pin test silently repeats) fails here.
        for id in ["PNP0A08", "PNP0A03", "PNP0C02"] {
            let aml = eisaid(id);
            let bytes: [u8; 4] = aml[1..5].try_into().unwrap();
            assert_eq!(decode_eisaid(bytes), id, "eisaid({id}) must round-trip");
        }
    }

    #[test]
    fn pkg_length_widths() {
        // <0x40 content -> 1 byte (value = content+1).
        assert_eq!(pkg_length(10), vec![11]);
        assert_eq!(pkg_length(0x3E), vec![0x3F]); // 0x3E+1 = 0x3F fits 1 byte
        // content needing 2 bytes: 0x3F+1 = 0x40 not < 0x40 -> 2-byte form.
        assert_eq!(pkg_length(0x3F), vec![0x40 | (0x41 & 0x0F), 0x41 >> 4]);
        // larger 2-byte case.
        let pl = pkg_length(200);
        let total = 202usize; // 200 + 2
        assert_eq!(pl, vec![0x40 | (total & 0x0F) as u8, (total >> 4) as u8]);
    }

    #[test]
    fn int_obj_forms() {
        assert_eq!(int_obj(0), vec![0x00]);
        assert_eq!(int_obj(1), vec![0x01]);
        assert_eq!(int_obj(0x42), vec![0x0A, 0x42]);
        assert_eq!(int_obj(0x1234), vec![0x0B, 0x34, 0x12]);
        assert_eq!(int_obj(0xDEAD_BEEF), vec![0x0C, 0xEF, 0xBE, 0xAD, 0xDE]);
    }

    #[test]
    fn word_bus_number_single_bus() {
        let d = word_bus_number(0, 0);
        assert_eq!(d.len(), 16); // tag(1) + len(2) + 13 body
        assert_eq!(d[0], 0x88);
        assert_eq!(&d[1..3], &0x000Du16.to_le_bytes());
        assert_eq!(d[3], 0x02); // bus number range
        // length field (last 2 bytes) = max-min+1 = 1.
        assert_eq!(&d[14..16], &1u16.to_le_bytes());
    }

    #[test]
    fn dword_memory_window_length_consistent() {
        let d = dword_memory(0xE010_0000, 0xFEBF_FFFF);
        assert_eq!(d.len(), 26); // tag(1) + len(2) + 23 body
        assert_eq!(d[0], 0x87);
        // length field at the end = max-min+1.
        let len_field = u32::from_le_bytes([d[22], d[23], d[24], d[25]]);
        assert_eq!(len_field, 0xFEBF_FFFF - 0xE010_0000 + 1);
    }

    #[test]
    fn resource_template_has_endtag_and_checksum() {
        let rt = resource_template(&memory32_fixed(0xE000_0000, 0x10_0000));
        assert_eq!(rt[0], 0x11); // BufferOp
        // Last two bytes are EndTag + zero checksum.
        assert_eq!(rt[rt.len() - 2], 0x79);
        assert_eq!(rt[rt.len() - 1], 0x00);
    }

    #[test]
    fn dsdt_body_structure() {
        let body = pci_host_bridge_dsdt_body(0xE000_0000, 0x10_0000, 0xE010_0000, 0xFEBF_FFFF);
        // Scope(\_SB) opcode + root-prefixed name.
        assert_eq!(body[0], 0x10); // ScopeOp
        // The body must contain both EisaIds (PCI0 _HID PNP0A08, PCIR PNP0C02),
        // the bus-number + memory descriptors, and an EndTag.
        let contains = |needle: &[u8]| body.windows(needle.len()).any(|w| w == needle);
        assert!(contains(&[0x0C, 0x41, 0xD0, 0x0A, 0x08]), "PNP0A08 _HID");
        assert!(contains(&[0x0C, 0x41, 0xD0, 0x0A, 0x03]), "PNP0A03 _CID");
        assert!(contains(&[0x0C, 0x41, 0xD0, 0x0C, 0x02]), "PNP0C02 reservation");
        assert!(contains(&[0x88, 0x0D, 0x00]), "WordBusNumber descriptor");
        assert!(contains(&[0x87, 0x17, 0x00]), "DWordMemory descriptor");
        assert!(contains(&[0x86, 0x09, 0x00]), "Memory32Fixed descriptor");
        assert!(contains(&[0x79, 0x00]), "ResourceTemplate EndTag + checksum");
        // Device opcode (0x5B 0x82) appears for PCI0 and PCIR.
        assert!(contains(&[0x5B, 0x82]), "Device opcode");
    }
}
