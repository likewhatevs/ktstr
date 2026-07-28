//! Same-PID `execve(2)` transfer for a PENDING admission identity.
//!
//! The target runner registers a lightweight arrival record before mapping the
//! test binary. It transfers that record's liveness fd, bounded weighted
//! preparation permits, and the physical CPU-SH owner for the process affinity
//! mask. No exact VM topology exists until the test finishes preparing
//! immutable artifacts and activates the record.

use super::{AdmissionFlock, PendingAdmission, registry};
use anyhow::{Context, Result};
use std::fs::File;
use std::io::Write;
use std::marker::PhantomData;
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd, RawFd};
use std::os::unix::fs::FileExt;
use std::process::Command;
use std::sync::atomic::{AtomicBool, Ordering};

pub(crate) const EXEC_HANDOFF_ENV: &str = "KTSTR_PENDING_ADMISSION_FD_V3";

const MAGIC: [u8; 8] = *b"KTSTRP03";
const VERSION: u32 = 3;
const HEADER_LEN: usize = 64;
const MAX_METADATA_LEN: usize = 4 << 20;
const MAX_PREPARATION_FDS: usize = 256;
const MAX_AFFINITY_CPUS: usize = 1 << 16;
const REQUIRED_SEALS: libc::c_int =
    libc::F_SEAL_SEAL | libc::F_SEAL_SHRINK | libc::F_SEAL_GROW | libc::F_SEAL_WRITE;
static HANDOFF_CONSUMED: AtomicBool = AtomicBool::new(false);

const H_MAGIC: usize = 0;
const H_VERSION: usize = 8;
const H_HEADER_LEN: usize = 12;
const H_TOTAL_LEN: usize = 16;
const H_PID: usize = 24;
const H_ORIGINAL_AFFINITY_COUNT: usize = 28;
const H_SLOT: usize = 32;
const H_TICKET: usize = 40;
const H_LIVENESS_FD: usize = 48;
const H_METADATA_LEN: usize = 52;
const H_PREPARATION_FD_COUNT: usize = 56;
const H_PREPARATION_INDEX: usize = 60;

/// Sealed one-shot transfer descriptor. If exec fails, Drop restores CLOEXEC
/// on every inherited descriptor before the pending owner is released.
pub(crate) struct PreparedPendingExecHandoff<'a> {
    descriptor: OwnedFd,
    inherited: Vec<(RawFd, libc::c_int)>,
    _pending: PhantomData<&'a PendingAdmission>,
}

impl PreparedPendingExecHandoff<'_> {
    pub(crate) fn configure_exec(&self, command: &mut Command) {
        command.env(
            EXEC_HANDOFF_ENV,
            format!("{}:{}", std::process::id(), self.descriptor.as_raw_fd()),
        );
    }

    #[cfg(test)]
    pub(crate) fn descriptor_fd_for_tests(&self) -> RawFd {
        self.descriptor.as_raw_fd()
    }
}

impl Drop for PreparedPendingExecHandoff<'_> {
    fn drop(&mut self) {
        for &(fd, flags) in self.inherited.iter().rev() {
            if unsafe { libc::fcntl(fd, libc::F_SETFD, flags) } == -1 {
                tracing::warn!(
                    fd,
                    error = %std::io::Error::last_os_error(),
                    "failed to restore descriptor flags after pending-admission exec handoff"
                );
            }
        }
    }
}

pub(crate) struct ImportedPendingExecHandoff {
    pub(crate) pending: PendingAdmission,
    pub(crate) metadata: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Header {
    pid: u32,
    slot: u64,
    ticket: u64,
    liveness_fd: RawFd,
    preparation_fds: Vec<(usize, RawFd)>,
    preparation_index: usize,
    affinity_cpu: usize,
    affinity_fd: RawFd,
    original_affinity: Vec<usize>,
    metadata_len: usize,
    total_len: usize,
}

pub(crate) fn prepare_pending_exec_handoff<'a>(
    pending: &'a PendingAdmission,
    metadata: &[u8],
) -> Result<PreparedPendingExecHandoff<'a>> {
    anyhow::ensure!(
        metadata.len() <= MAX_METADATA_LEN,
        "pending admission handoff metadata is {} bytes; maximum is {MAX_METADATA_LEN}",
        metadata.len(),
    );
    let (slot, ticket, liveness_fd) = pending.exec_handoff_parts()?;
    let (preparation_index, preparation_fds) = pending.preparation_handoff_parts()?;
    let (affinity_cpu, affinity_fd, original_affinity) =
        pending.preparation_affinity_handoff_parts()?;
    anyhow::ensure!(
        !preparation_fds.is_empty() && preparation_fds.len() <= MAX_PREPARATION_FDS,
        "pending admission has invalid preparation descriptor count {}",
        preparation_fds.len(),
    );
    validate_distinct_fds(
        std::iter::once(liveness_fd)
            .chain(std::iter::once(affinity_fd))
            .chain(preparation_fds.iter().map(|(_, fd)| *fd)),
        None,
    )?;
    anyhow::ensure!(
        !original_affinity.is_empty() && original_affinity.len() <= MAX_AFFINITY_CPUS,
        "pending admission has invalid original affinity width {}",
        original_affinity.len(),
    );
    let fd_bytes = preparation_fds
        .len()
        .checked_mul(2 * std::mem::size_of::<u32>())
        .ok_or_else(|| anyhow::anyhow!("preparation descriptor table length overflow"))?;
    let affinity_bytes = original_affinity
        .len()
        .checked_mul(std::mem::size_of::<u32>())
        .and_then(|bytes| bytes.checked_add(2 * std::mem::size_of::<u32>()))
        .ok_or_else(|| anyhow::anyhow!("preparation affinity table length overflow"))?;
    let bytes = encode(
        Header {
            pid: std::process::id(),
            slot,
            ticket,
            liveness_fd,
            preparation_fds: preparation_fds.clone(),
            preparation_index,
            affinity_cpu,
            affinity_fd,
            original_affinity: original_affinity.to_vec(),
            metadata_len: metadata.len(),
            total_len: HEADER_LEN + fd_bytes + affinity_bytes + metadata.len(),
        },
        metadata,
    )?;
    let descriptor = create_sealed_memfd(&bytes)?;
    validate_distinct_fds(
        std::iter::once(liveness_fd)
            .chain(std::iter::once(affinity_fd))
            .chain(preparation_fds.iter().map(|(_, fd)| *fd)),
        Some(descriptor.as_raw_fd()),
    )?;

    let mut inherited = Vec::with_capacity(preparation_fds.len() + 3);
    for fd in std::iter::once(liveness_fd)
        .chain(std::iter::once(affinity_fd))
        .chain(preparation_fds.iter().map(|(_, fd)| *fd))
        .chain(std::iter::once(descriptor.as_raw_fd()))
    {
        match make_inheritable(fd) {
            Ok(flags) => inherited.push((fd, flags)),
            Err(error) => {
                for &(changed, flags) in inherited.iter().rev() {
                    unsafe { libc::fcntl(changed, libc::F_SETFD, flags) };
                }
                return Err(error);
            }
        }
    }
    Ok(PreparedPendingExecHandoff {
        descriptor,
        inherited,
        _pending: PhantomData,
    })
}

/// Consume and validate the new-format pending handoff, if present. The value
/// includes the pre-exec PID, so a later child which inherits the immutable
/// environment cannot mistake a reused descriptor number for this handoff.
/// A process-local atomic provides one-shot consumption without mutating the
/// environment after libtest may have started threads.
pub(crate) fn take_pending_exec_handoff() -> Result<Option<ImportedPendingExecHandoff>> {
    let Some(raw_value) = std::env::var_os(EXEC_HANDOFF_ENV) else {
        return Ok(None);
    };
    let value = raw_value
        .into_string()
        .map_err(|_| anyhow::anyhow!("pending admission handoff fd is not UTF-8"))?;
    let (pid, descriptor_fd) = value
        .split_once(':')
        .ok_or_else(|| anyhow::anyhow!("malformed pending admission handoff identity"))?;
    let pid: u32 = pid
        .parse()
        .with_context(|| format!("parse pending admission handoff PID {pid:?}"))?;
    if pid != std::process::id() {
        return Ok(None);
    }
    if HANDOFF_CONSUMED.swap(true, Ordering::AcqRel) {
        return Ok(None);
    }
    let descriptor_fd: RawFd = descriptor_fd
        .parse()
        .with_context(|| format!("parse pending admission handoff fd {value:?}"))?;
    validate_distinct_fds([], Some(descriptor_fd))?;
    get_fd_flags(descriptor_fd)?;
    // SAFETY: the one-shot env value transfers sole ownership to this call.
    let descriptor = unsafe { OwnedFd::from_raw_fd(descriptor_fd) };
    set_fd_flags(
        descriptor_fd,
        get_fd_flags(descriptor_fd)? | libc::FD_CLOEXEC,
    )?;
    validate_seals(descriptor_fd)?;
    let bytes = read_descriptor(&descriptor)?;
    let header = decode(&bytes)?;
    anyhow::ensure!(
        header.pid == std::process::id(),
        "pending admission handoff belongs to PID {}, current PID is {}",
        header.pid,
        std::process::id(),
    );
    validate_distinct_fds(
        std::iter::once(header.liveness_fd)
            .chain(std::iter::once(header.affinity_fd))
            .chain(header.preparation_fds.iter().map(|(_, fd)| *fd)),
        Some(descriptor_fd),
    )?;
    for fd in std::iter::once(header.liveness_fd)
        .chain(std::iter::once(header.affinity_fd))
        .chain(header.preparation_fds.iter().map(|(_, fd)| *fd))
    {
        get_fd_flags(fd)?;
    }
    // All raw descriptors have been proved open and distinct. Take ownership
    // of the complete set before any later fallible validation, so every error
    // closes every inherited owner rather than leaking a partial handoff.
    let liveness = unsafe { OwnedFd::from_raw_fd(header.liveness_fd) };
    let affinity_lock =
        AdmissionFlock::from_acquired(unsafe { OwnedFd::from_raw_fd(header.affinity_fd) });
    let preparation_fds = header
        .preparation_fds
        .iter()
        .map(|(permit, fd)| {
            // SAFETY: the sealed one-shot descriptor transfers sole ownership
            // of every distinct inherited preparation descriptor.
            (
                *permit,
                AdmissionFlock::from_acquired(unsafe { OwnedFd::from_raw_fd(*fd) }),
            )
        })
        .collect::<Vec<_>>();
    set_fd_flags(
        liveness.as_raw_fd(),
        get_fd_flags(liveness.as_raw_fd())? | libc::FD_CLOEXEC,
    )?;
    set_fd_flags(
        affinity_lock.as_raw_fd(),
        get_fd_flags(affinity_lock.as_raw_fd())? | libc::FD_CLOEXEC,
    )?;
    for (_, fd) in &preparation_fds {
        set_fd_flags(
            fd.as_raw_fd(),
            get_fd_flags(fd.as_raw_fd())? | libc::FD_CLOEXEC,
        )?;
    }
    let preparation = super::super::validate_preparation_permit(
        header.preparation_index,
        preparation_fds,
        header.affinity_cpu,
        affinity_lock,
        header.original_affinity.clone(),
    )?;
    let (ticket, pending_claim) = registry::import_pending_exec_handoff(
        header.slot,
        header.ticket,
        &liveness,
        &preparation.claim(),
    )?;
    // Establish the physical-first Drop ordering before any further fallible
    // work. The registry ticket owns a duplicate liveness fd, so the inherited
    // descriptor can now close without withdrawing the PENDING publication.
    let pending = PendingAdmission::from_imported_ticket(ticket, preparation, pending_claim);
    drop(liveness);
    let metadata_offset = HEADER_LEN
        + header.preparation_fds.len() * 2 * std::mem::size_of::<u32>()
        + 2 * std::mem::size_of::<u32>()
        + header.original_affinity.len() * std::mem::size_of::<u32>();
    let metadata = bytes[metadata_offset..header.total_len].to_vec();
    anyhow::ensure!(
        metadata.len() == header.metadata_len,
        "pending admission metadata length changed during decode",
    );
    Ok(Some(ImportedPendingExecHandoff { pending, metadata }))
}

fn encode(header: Header, metadata: &[u8]) -> Result<Vec<u8>> {
    let fd_bytes = header.preparation_fds.len() * 2 * std::mem::size_of::<u32>();
    let affinity_bytes = 2 * std::mem::size_of::<u32>()
        + header.original_affinity.len() * std::mem::size_of::<u32>();
    anyhow::ensure!(
        header.total_len == HEADER_LEN + fd_bytes + affinity_bytes + metadata.len(),
        "invalid handoff length"
    );
    let mut bytes = vec![0u8; header.total_len];
    bytes[H_MAGIC..H_MAGIC + MAGIC.len()].copy_from_slice(&MAGIC);
    write_u32(&mut bytes, H_VERSION, VERSION);
    write_u32(&mut bytes, H_HEADER_LEN, HEADER_LEN as u32);
    write_u64(&mut bytes, H_TOTAL_LEN, header.total_len as u64);
    write_u32(&mut bytes, H_PID, header.pid);
    write_u32(
        &mut bytes,
        H_ORIGINAL_AFFINITY_COUNT,
        u32::try_from(header.original_affinity.len())
            .context("original affinity width does not fit u32")?,
    );
    write_u64(&mut bytes, H_SLOT, header.slot);
    write_u64(&mut bytes, H_TICKET, header.ticket);
    write_u32(
        &mut bytes,
        H_LIVENESS_FD,
        u32::try_from(header.liveness_fd).context("pending liveness fd does not fit u32")?,
    );
    write_u32(
        &mut bytes,
        H_METADATA_LEN,
        u32::try_from(metadata.len()).context("pending metadata length does not fit u32")?,
    );
    write_u32(
        &mut bytes,
        H_PREPARATION_FD_COUNT,
        u32::try_from(header.preparation_fds.len())
            .context("preparation descriptor count does not fit u32")?,
    );
    write_u32(
        &mut bytes,
        H_PREPARATION_INDEX,
        u32::try_from(header.preparation_index)
            .context("preparation permit index does not fit u32")?,
    );
    for (offset, (permit, fd)) in header.preparation_fds.iter().enumerate() {
        let entry = HEADER_LEN + offset * 2 * std::mem::size_of::<u32>();
        write_u32(
            &mut bytes,
            entry,
            u32::try_from(*permit).context("preparation permit id does not fit u32")?,
        );
        write_u32(
            &mut bytes,
            entry + std::mem::size_of::<u32>(),
            u32::try_from(*fd).context("preparation permit fd does not fit u32")?,
        );
    }
    let affinity_offset = HEADER_LEN + fd_bytes;
    write_u32(
        &mut bytes,
        affinity_offset,
        u32::try_from(header.affinity_cpu).context("preparation affinity CPU does not fit u32")?,
    );
    write_u32(
        &mut bytes,
        affinity_offset + std::mem::size_of::<u32>(),
        u32::try_from(header.affinity_fd).context("preparation affinity fd does not fit u32")?,
    );
    for (offset, cpu) in header.original_affinity.iter().enumerate() {
        write_u32(
            &mut bytes,
            affinity_offset + 2 * std::mem::size_of::<u32>() + offset * std::mem::size_of::<u32>(),
            u32::try_from(*cpu).context("original affinity CPU does not fit u32")?,
        );
    }
    bytes[HEADER_LEN + fd_bytes + affinity_bytes..].copy_from_slice(metadata);
    Ok(bytes)
}

fn decode(bytes: &[u8]) -> Result<Header> {
    anyhow::ensure!(
        bytes.len() >= HEADER_LEN,
        "pending admission handoff is truncated"
    );
    anyhow::ensure!(
        bytes[H_MAGIC..H_MAGIC + MAGIC.len()] == MAGIC,
        "unsupported pending admission handoff magic"
    );
    anyhow::ensure!(
        read_u32(bytes, H_VERSION) == VERSION,
        "unsupported pending admission handoff version"
    );
    anyhow::ensure!(
        read_u32(bytes, H_HEADER_LEN) as usize == HEADER_LEN,
        "malformed pending admission header length"
    );
    let original_affinity_count = read_u32(bytes, H_ORIGINAL_AFFINITY_COUNT) as usize;
    anyhow::ensure!(
        (1..=MAX_AFFINITY_CPUS).contains(&original_affinity_count),
        "pending admission has invalid original affinity width {original_affinity_count}",
    );
    let total_len = usize::try_from(read_u64(bytes, H_TOTAL_LEN))
        .context("pending admission total length does not fit this process")?;
    let metadata_len = read_u32(bytes, H_METADATA_LEN) as usize;
    let preparation_fd_count = read_u32(bytes, H_PREPARATION_FD_COUNT) as usize;
    anyhow::ensure!(
        (1..=MAX_PREPARATION_FDS).contains(&preparation_fd_count),
        "pending admission has invalid preparation descriptor count {preparation_fd_count}"
    );
    let fd_bytes = preparation_fd_count
        .checked_mul(2 * std::mem::size_of::<u32>())
        .ok_or_else(|| anyhow::anyhow!("preparation descriptor table length overflow"))?;
    let affinity_bytes = original_affinity_count
        .checked_mul(std::mem::size_of::<u32>())
        .and_then(|bytes| bytes.checked_add(2 * std::mem::size_of::<u32>()))
        .ok_or_else(|| anyhow::anyhow!("preparation affinity table length overflow"))?;
    anyhow::ensure!(
        metadata_len <= MAX_METADATA_LEN,
        "pending admission metadata is too large"
    );
    anyhow::ensure!(
        total_len == bytes.len()
            && total_len == HEADER_LEN + fd_bytes + affinity_bytes + metadata_len,
        "pending admission handoff length fields are inconsistent"
    );
    let preparation_fds = (0..preparation_fd_count)
        .map(|offset| {
            let entry = HEADER_LEN + offset * 2 * std::mem::size_of::<u32>();
            let permit = usize::try_from(read_u32(bytes, entry))
                .context("preparation permit id does not fit usize")?;
            let fd = RawFd::try_from(read_u32(bytes, entry + std::mem::size_of::<u32>()))
                .context("preparation permit fd does not fit RawFd")?;
            Ok((permit, fd))
        })
        .collect::<Result<Vec<_>>>()?;
    let affinity_offset = HEADER_LEN + fd_bytes;
    let affinity_cpu = usize::try_from(read_u32(bytes, affinity_offset))
        .context("preparation affinity CPU does not fit usize")?;
    let affinity_fd = RawFd::try_from(read_u32(
        bytes,
        affinity_offset + std::mem::size_of::<u32>(),
    ))
    .context("preparation affinity fd does not fit RawFd")?;
    let original_affinity = (0..original_affinity_count)
        .map(|offset| {
            usize::try_from(read_u32(
                bytes,
                affinity_offset
                    + 2 * std::mem::size_of::<u32>()
                    + offset * std::mem::size_of::<u32>(),
            ))
            .context("original affinity CPU does not fit usize")
        })
        .collect::<Result<Vec<_>>>()?;
    anyhow::ensure!(
        original_affinity.windows(2).all(|pair| pair[0] < pair[1]),
        "pending admission original affinity is not sorted and unique",
    );
    anyhow::ensure!(
        original_affinity.contains(&affinity_cpu),
        "preparation affinity CPU is outside the original mask",
    );
    Ok(Header {
        pid: read_u32(bytes, H_PID),
        slot: read_u64(bytes, H_SLOT),
        ticket: read_u64(bytes, H_TICKET),
        liveness_fd: RawFd::try_from(read_u32(bytes, H_LIVENESS_FD))
            .context("pending liveness fd does not fit RawFd")?,
        preparation_fds,
        preparation_index: read_u32(bytes, H_PREPARATION_INDEX) as usize,
        affinity_cpu,
        affinity_fd,
        original_affinity,
        metadata_len,
        total_len,
    })
}

fn create_sealed_memfd(bytes: &[u8]) -> Result<OwnedFd> {
    let name = c"ktstr-pending-admission";
    let raw =
        unsafe { libc::memfd_create(name.as_ptr(), libc::MFD_CLOEXEC | libc::MFD_ALLOW_SEALING) };
    if raw == -1 {
        return Err(std::io::Error::last_os_error()).context("create pending handoff memfd");
    }
    let fd = unsafe { OwnedFd::from_raw_fd(raw) };
    let mut file = File::from(fd);
    file.write_all(bytes)
        .context("write pending handoff descriptor")?;
    let raw = file.as_raw_fd();
    if unsafe { libc::fcntl(raw, libc::F_ADD_SEALS, REQUIRED_SEALS) } == -1 {
        return Err(std::io::Error::last_os_error()).context("seal pending handoff descriptor");
    }
    validate_seals(raw)?;
    Ok(file.into())
}

fn read_descriptor(descriptor: &OwnedFd) -> Result<Vec<u8>> {
    let file = File::from(
        descriptor
            .try_clone()
            .context("duplicate pending handoff descriptor")?,
    );
    let length = usize::try_from(file.metadata()?.len())
        .context("pending handoff descriptor length does not fit this process")?;
    anyhow::ensure!(
        (HEADER_LEN + 5 * std::mem::size_of::<u32>()
            ..=HEADER_LEN
                + MAX_PREPARATION_FDS * 2 * std::mem::size_of::<u32>()
                + MAX_AFFINITY_CPUS * std::mem::size_of::<u32>()
                + 2 * std::mem::size_of::<u32>()
                + MAX_METADATA_LEN)
            .contains(&length),
        "pending handoff descriptor length {length} is invalid"
    );
    let mut bytes = vec![0u8; length];
    file.read_exact_at(&mut bytes, 0)
        .context("read pending handoff descriptor")?;
    Ok(bytes)
}

fn validate_seals(fd: RawFd) -> Result<()> {
    let seals = unsafe { libc::fcntl(fd, libc::F_GET_SEALS) };
    if seals == -1 {
        return Err(std::io::Error::last_os_error()).context("read pending handoff seals");
    }
    anyhow::ensure!(
        seals & REQUIRED_SEALS == REQUIRED_SEALS,
        "pending handoff descriptor is not immutable"
    );
    Ok(())
}

fn validate_distinct_fds(
    fds: impl IntoIterator<Item = RawFd>,
    descriptor: Option<RawFd>,
) -> Result<()> {
    let mut seen = std::collections::BTreeSet::new();
    for fd in fds.into_iter().chain(descriptor) {
        anyhow::ensure!(
            fd > libc::STDERR_FILENO,
            "pending admission fd {fd} aliases standard I/O"
        );
        anyhow::ensure!(seen.insert(fd), "pending admission repeats descriptor {fd}");
    }
    Ok(())
}

fn get_fd_flags(fd: RawFd) -> Result<libc::c_int> {
    let flags = unsafe { libc::fcntl(fd, libc::F_GETFD) };
    if flags == -1 {
        Err(std::io::Error::last_os_error())
            .with_context(|| format!("read flags for pending admission descriptor {fd}"))
    } else {
        Ok(flags)
    }
}

fn set_fd_flags(fd: RawFd, flags: libc::c_int) -> Result<()> {
    if unsafe { libc::fcntl(fd, libc::F_SETFD, flags) } == -1 {
        Err(std::io::Error::last_os_error())
            .with_context(|| format!("set flags for pending admission descriptor {fd}"))
    } else {
        Ok(())
    }
}

fn make_inheritable(fd: RawFd) -> Result<libc::c_int> {
    let flags = get_fd_flags(fd)?;
    set_fd_flags(fd, flags & !libc::FD_CLOEXEC)?;
    Ok(flags)
}

fn write_u32(bytes: &mut [u8], offset: usize, value: u32) {
    bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}

fn write_u64(bytes: &mut [u8], offset: usize, value: u64) {
    bytes[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
}

fn read_u32(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap())
}

fn read_u64(bytes: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn descriptor_header() -> Header {
        let preparation_fds = vec![(17, 40), (23, 41), (29, 42)];
        let original_affinity = vec![2, 7, 65, 4_097];
        let table = preparation_fds.len() * 2 * std::mem::size_of::<u32>()
            + 2 * std::mem::size_of::<u32>()
            + original_affinity.len() * std::mem::size_of::<u32>();
        Header {
            pid: std::process::id(),
            slot: 71,
            ticket: 113,
            liveness_fd: 39,
            preparation_fds,
            preparation_index: 5,
            affinity_cpu: 65,
            affinity_fd: 43,
            original_affinity,
            metadata_len: 9,
            total_len: HEADER_LEN + table + 9,
        }
    }

    #[test]
    fn v3_descriptor_round_trips_physical_affinity_and_every_preparation_fd() {
        let header = descriptor_header();
        let bytes = encode(header.clone(), b"cell-meta").expect("encode v3 handoff");
        assert_eq!(decode(&bytes).expect("decode v3 handoff"), header);
        assert_eq!(&bytes[bytes.len() - 9..], b"cell-meta");
    }

    #[test]
    fn v3_descriptor_rejects_noncanonical_original_affinity() {
        let header = descriptor_header();
        let mut bytes = encode(header.clone(), b"cell-meta").expect("encode v3 handoff");
        let affinity_offset =
            HEADER_LEN + header.preparation_fds.len() * 2 * std::mem::size_of::<u32>();
        write_u32(
            &mut bytes,
            affinity_offset + 3 * std::mem::size_of::<u32>(),
            header.original_affinity[0] as u32,
        );
        let error = decode(&bytes).expect_err("duplicate affinity CPU must fail closed");
        assert!(
            error.to_string().contains("sorted and unique"),
            "malformed affinity diagnostic: {error:#}",
        );
    }
}
