//! AF_PACKET traffic sender for
//! [`WorkType::NetTraffic`](crate::workload::WorkType::NetTraffic).
//!
//! Replicates the proven send recipe from the wide-SMP net-IRQ e2e
//! (`tests/wide_smp_net_irq_e2e.rs`): an `AF_PACKET` / `SOCK_RAW` socket
//! bound to the single non-loopback (virtio-net) interface, brought
//! administratively up, sending self-addressed L2 frames. Each `sendto` is a
//! virtio TX kick the v0 in-VMM loopback echoes straight into RX, raising the
//! guest's RX hardirq + NAPI softirq (the full kernel path is in the variant
//! doc). Pure sysfs + ioctl + raw socket — no `ip`/`ifconfig` shell-out,
//! matching ktstr's no-shell-out convention. The interface-up ioctl needs
//! `CAP_NET_ADMIN`; ktstr always runs as root, so it is present.

use std::fs;
use std::io;
use std::mem;

/// Send timeout (µs) so a `sendto` into a full TX ring does not block the
/// worker past one `SO_SNDTIMEO` window — the worker re-checks the stop flag
/// and retries rather than blocking unboundedly (the vCPU-thread loopback
/// drains the ring continuously, so this only triggers under severe
/// back-pressure). 100 ms.
const SEND_TIMEOUT_US: i64 = 100_000;

/// Local-experimental EtherType (matches the e2e). The loopback echoes the
/// frame verbatim, so the value only needs to be benign.
const ETHERTYPE_LOCAL_EXPERIMENTAL: u16 = 0x88B5;

/// A configured AF_PACKET sender bound to the virtio-net NIC. Owns the raw fd
/// (closed on [`Drop`]), the pre-built frame, and the destination sockaddr.
pub(super) struct NetTrafficSender {
    fd: i32,
    frame: Vec<u8>,
    dst: libc::sockaddr_ll,
}

impl NetTrafficSender {
    /// Discover the NIC, open + bind an `AF_PACKET` raw socket, bring the
    /// interface up, set the send timeout, and build the `frame_bytes`-sized
    /// self-addressed frame. Returns `Err` (with context in the message) when
    /// no non-loopback NIC is present or any setup syscall fails — the caller
    /// turns that into a LOUD no-op. `frame_bytes` is validated to `[60, 1514]`
    /// at spawn (`validate_workload_admission`).
    pub(super) fn setup(frame_bytes: u16) -> io::Result<Self> {
        let iface = virtio_net_iface()?;
        let ifindex = iface_ifindex(&iface)?;
        let mac = iface_mac(&iface)?;
        // AF_PACKET / SOCK_RAW over ETH_P_ALL (big-endian on the wire).
        let proto = (libc::ETH_P_ALL as u16).to_be() as libc::c_int;
        // SAFETY: standard AF_PACKET raw socket creation; the returned fd is
        // checked below and owned by `Self` (closed on Drop) from then on.
        let fd = unsafe { libc::socket(libc::AF_PACKET, libc::SOCK_RAW, proto) };
        if fd < 0 {
            return Err(io::Error::last_os_error());
        }
        // Own the fd immediately so every early return below closes it.
        let sender = NetTrafficSender {
            fd,
            frame: build_frame(mac, frame_bytes),
            dst: dst_sockaddr(ifindex, mac),
        };
        sender.bring_up(&iface)?;
        sender.bind_to(ifindex)?;
        sender.set_send_timeout()?;
        Ok(sender)
    }

    /// Send one frame. `true` on a successful TX (one virtio kick → one
    /// RX-completion IRQ via the loopback). A timed-out / failed send (e.g.
    /// `SO_SNDTIMEO` on a full TX ring) returns `false`; the worker re-checks
    /// the stop flag and retries next iteration.
    pub(super) fn send_one(&self) -> bool {
        // SAFETY: `fd` is a bound AF_PACKET socket; `frame` and `dst` are
        // valid for the lengths passed.
        let sent = unsafe {
            libc::sendto(
                self.fd,
                self.frame.as_ptr() as *const libc::c_void,
                self.frame.len(),
                0,
                &self.dst as *const _ as *const libc::sockaddr,
                mem::size_of::<libc::sockaddr_ll>() as libc::socklen_t,
            )
        };
        // A real frame TX returns the bytes sent (raw sockets are
        // all-or-nothing); `> 0` excludes a degenerate 0-byte "success" so
        // work_units counts only genuine sends.
        sent > 0
    }

    /// Bring the interface administratively up (`IFF_UP | IFF_RUNNING` via
    /// `SIOCSIFFLAGS`). Mandatory before TX: `packet_snd` rejects a down
    /// device with `-ENETDOWN` (`!(dev->flags & IFF_UP)`), and
    /// `dev_queue_xmit`'s `netif_running` check is the deeper gate.
    fn bring_up(&self, iface: &str) -> io::Result<()> {
        let mut ifr: libc::ifreq = unsafe { mem::zeroed() };
        copy_ifname(&mut ifr.ifr_name, iface)?;
        // SAFETY: ifr_name is set; SIOCGIFFLAGS reads current flags, then
        // SIOCSIFFLAGS writes them back with the up bits set.
        let rc = unsafe { libc::ioctl(self.fd, libc::SIOCGIFFLAGS, &mut ifr) };
        if rc != 0 {
            return Err(io::Error::last_os_error());
        }
        unsafe {
            ifr.ifr_ifru.ifru_flags |= (libc::IFF_UP | libc::IFF_RUNNING) as libc::c_short;
        }
        let rc = unsafe { libc::ioctl(self.fd, libc::SIOCSIFFLAGS, &ifr) };
        if rc != 0 {
            return Err(io::Error::last_os_error());
        }
        Ok(())
    }

    /// Bind the socket to the NIC by ifindex (associates TX/RX with the
    /// device).
    fn bind_to(&self, ifindex: i32) -> io::Result<()> {
        let mut sll: libc::sockaddr_ll = unsafe { mem::zeroed() };
        sll.sll_family = libc::AF_PACKET as u16;
        sll.sll_protocol = (libc::ETH_P_ALL as u16).to_be();
        sll.sll_ifindex = ifindex;
        // SAFETY: sll is a fully-initialized sockaddr_ll; len matches.
        let rc = unsafe {
            libc::bind(
                self.fd,
                &sll as *const _ as *const libc::sockaddr,
                mem::size_of::<libc::sockaddr_ll>() as libc::socklen_t,
            )
        };
        if rc != 0 {
            return Err(io::Error::last_os_error());
        }
        Ok(())
    }

    /// Set `SO_SNDTIMEO` so a full-ring `sendto` is bounded (see
    /// [`SEND_TIMEOUT_US`]).
    fn set_send_timeout(&self) -> io::Result<()> {
        let tv = libc::timeval {
            tv_sec: SEND_TIMEOUT_US / 1_000_000,
            tv_usec: (SEND_TIMEOUT_US % 1_000_000) as libc::suseconds_t,
        };
        // SAFETY: fd is valid; tv is a valid timeval for SO_SNDTIMEO.
        let rc = unsafe {
            libc::setsockopt(
                self.fd,
                libc::SOL_SOCKET,
                libc::SO_SNDTIMEO,
                &tv as *const _ as *const libc::c_void,
                mem::size_of::<libc::timeval>() as libc::socklen_t,
            )
        };
        if rc != 0 {
            return Err(io::Error::last_os_error());
        }
        Ok(())
    }
}

impl Drop for NetTrafficSender {
    fn drop(&mut self) {
        // SAFETY: `fd` is owned by this struct and not closed elsewhere.
        unsafe {
            libc::close(self.fd);
        }
    }
}

/// The single non-loopback network interface (the virtio-net NIC). Skips
/// `lo`; the NIC is the one interface backed by a device (a `device` symlink
/// under its sysfs node). `NotFound` when none is present — the no-NIC case
/// the caller turns into a LOUD no-op.
fn virtio_net_iface() -> io::Result<String> {
    virtio_net_iface_in(std::path::Path::new("/sys/class/net"))
}

/// [`virtio_net_iface`] scoped to a sysfs-net `root`, so the no-NIC
/// (`NotFound`) discovery path is unit-testable against a tempdir without a
/// real interface.
fn virtio_net_iface_in(root: &std::path::Path) -> io::Result<String> {
    for ent in fs::read_dir(root)? {
        let ent = ent?;
        let name = ent.file_name().to_string_lossy().into_owned();
        if name == "lo" {
            continue;
        }
        if ent.path().join("device").exists() {
            return Ok(name);
        }
    }
    Err(io::Error::new(
        io::ErrorKind::NotFound,
        "no non-loopback network interface with a device under /sys/class/net \
         (attach a NIC with #[ktstr_test(network = ...)])",
    ))
}

/// Read `/sys/class/net/<iface>/ifindex`.
fn iface_ifindex(iface: &str) -> io::Result<i32> {
    let s = fs::read_to_string(format!("/sys/class/net/{iface}/ifindex"))?;
    s.trim()
        .parse::<i32>()
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, format!("ifindex for {iface}: {e}")))
}

/// Read `/sys/class/net/<iface>/address` into a 6-byte MAC.
fn iface_mac(iface: &str) -> io::Result<[u8; 6]> {
    let s = fs::read_to_string(format!("/sys/class/net/{iface}/address"))?;
    parse_mac(s.trim())
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, format!("bad MAC for {iface}: {s:?}")))
}

/// Parse a colon-separated 6-octet MAC (e.g. `52:54:00:12:34:56`).
fn parse_mac(s: &str) -> Option<[u8; 6]> {
    let mut mac = [0u8; 6];
    let mut n = 0;
    for (i, octet) in s.split(':').enumerate() {
        if i >= 6 {
            return None;
        }
        mac[i] = u8::from_str_radix(octet, 16).ok()?;
        n = i + 1;
    }
    (n == 6).then_some(mac)
}

/// Build a `frame_bytes`-sized Ethernet frame: dst == src == our MAC (the
/// loopback echoes verbatim, no MAC swap), a local-experimental EtherType,
/// and a `KTST` payload marker; the remainder is zero-padded. `frame_bytes`
/// is `>= 60` post-validation, so the 18-byte header + marker always fits.
fn build_frame(mac: [u8; 6], frame_bytes: u16) -> Vec<u8> {
    debug_assert!(
        frame_bytes as usize >= 18,
        "frame_bytes {frame_bytes} < 18 leaves no room for the L2 header + marker; \
         validate_workload_admission enforces >= 60"
    );
    let mut frame = vec![0u8; (frame_bytes as usize).max(18)];
    frame[0..6].copy_from_slice(&mac);
    frame[6..12].copy_from_slice(&mac);
    frame[12..14].copy_from_slice(&ETHERTYPE_LOCAL_EXPERIMENTAL.to_be_bytes());
    frame[14..18].copy_from_slice(b"KTST");
    frame
}

/// The destination `sockaddr_ll` for `sendto`: our own NIC (ifindex + MAC).
/// v0 is pure in-VMM loopback, so the NIC is its own peer.
fn dst_sockaddr(ifindex: i32, mac: [u8; 6]) -> libc::sockaddr_ll {
    let mut dst: libc::sockaddr_ll = unsafe { mem::zeroed() };
    dst.sll_family = libc::AF_PACKET as u16;
    dst.sll_protocol = (libc::ETH_P_ALL as u16).to_be();
    dst.sll_ifindex = ifindex;
    dst.sll_halen = 6;
    dst.sll_addr[0..6].copy_from_slice(&mac);
    dst
}

/// Copy an interface name into a fixed `ifr_name` buffer, erroring if it is
/// too long for `IFNAMSIZ` (leaving room for the NUL terminator).
fn copy_ifname(buf: &mut [libc::c_char; libc::IFNAMSIZ], iface: &str) -> io::Result<()> {
    let bytes = iface.as_bytes();
    if bytes.len() >= libc::IFNAMSIZ {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("interface name {iface} exceeds IFNAMSIZ"),
        ));
    }
    for (i, &b) in bytes.iter().enumerate() {
        buf[i] = b as libc::c_char;
    }
    Ok(())
}

/// Warn ONCE per process that [`NetTrafficSender::setup`] failed (no NIC or a
/// setup syscall error), so a `NetTraffic` workload with no attached NIC is a
/// LOUD no-op rather than silently doing nothing. Mirrors the
/// `warn_custom_iterations_zero_once` `std::sync::Once` idiom.
pub(super) fn warn_net_traffic_setup_failed_once(err: &io::Error) {
    static WARNED: std::sync::Once = std::sync::Once::new();
    WARNED.call_once(|| {
        eprintln!(
            "workload: WorkType::NetTraffic could not open an AF_PACKET sender \
             ({err}); the worker is a no-op (work_units=0). Attach a NIC with \
             #[ktstr_test(network = ...)]. See the WorkType::NetTraffic variant doc."
        );
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_frame_is_self_addressed_and_sized() {
        let mac = [0x52, 0x54, 0x00, 0x12, 0x34, 0x56];
        let f = build_frame(mac, 60);
        assert_eq!(f.len(), 60, "frame is the requested size");
        assert_eq!(&f[0..6], &mac, "dst MAC = our MAC (loopback echoes verbatim)");
        assert_eq!(&f[6..12], &mac, "src MAC = our MAC");
        assert_eq!(
            &f[12..14],
            &ETHERTYPE_LOCAL_EXPERIMENTAL.to_be_bytes(),
            "ethertype is the local-experimental marker, big-endian"
        );
        assert_eq!(&f[14..18], b"KTST", "payload marker present");
        assert!(f[18..].iter().all(|&b| b == 0), "remainder is zero-padded");
    }

    #[test]
    fn build_frame_larger_size_zero_pads_tail() {
        let mac = [1, 2, 3, 4, 5, 6];
        let f = build_frame(mac, 1514);
        assert_eq!(f.len(), 1514);
        assert_eq!(&f[14..18], b"KTST");
        assert!(f[18..].iter().all(|&b| b == 0), "the tail past the marker stays zero");
    }

    #[test]
    fn parse_mac_roundtrips_and_rejects_malformed() {
        assert_eq!(
            parse_mac("52:54:00:12:34:56"),
            Some([0x52, 0x54, 0x00, 0x12, 0x34, 0x56])
        );
        assert_eq!(parse_mac("01:02:03:04:05"), None, "5 octets rejected");
        assert_eq!(parse_mac("01:02:03:04:05:06:07"), None, ">6 octets rejected");
        assert_eq!(parse_mac("zz:02:03:04:05:06"), None, "non-hex rejected");
    }

    #[test]
    fn dst_sockaddr_carries_ifindex_and_mac() {
        let mac = [0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff];
        let d = dst_sockaddr(7, mac);
        assert_eq!(d.sll_family, libc::AF_PACKET as u16);
        assert_eq!(d.sll_ifindex, 7);
        assert_eq!(d.sll_halen, 6);
        assert_eq!(&d.sll_addr[0..6], &mac);
    }

    #[test]
    fn virtio_net_iface_in_lo_only_is_not_found() {
        // A net root with only `lo` (no device-backed NIC) is the no-NIC
        // case the dispatch arm turns into a LOUD no-op — discovery must
        // return NotFound, the Err that triggers it.
        let tmp = tempfile::TempDir::new().expect("tempdir");
        std::fs::create_dir(tmp.path().join("lo")).unwrap();
        let err = virtio_net_iface_in(tmp.path()).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::NotFound);
    }

    #[test]
    fn virtio_net_iface_in_picks_the_device_backed_iface() {
        // Skips `lo`, returns the one interface with a `device` node.
        let tmp = tempfile::TempDir::new().expect("tempdir");
        std::fs::create_dir(tmp.path().join("lo")).unwrap();
        let eth = tmp.path().join("eth0");
        std::fs::create_dir(&eth).unwrap();
        std::fs::create_dir(eth.join("device")).unwrap();
        assert_eq!(virtio_net_iface_in(tmp.path()).unwrap(), "eth0");
    }
}
