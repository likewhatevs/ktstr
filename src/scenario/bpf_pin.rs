//! Guest-side BPF map fd pinning. See
//! [`crate::scenario::ops::Op::PinBpfMap`] for the full
//! motivation; in short, the same-binary `Op::ReplaceScheduler`
//! swap window's multi-bss case (two `<obj>.bss` copies coexisting
//! while the dying scheduler's BPF object is being torn down) only
//! fires when both copies are still alive at freeze time, and the
//! kernel frees the dying instance's maps as soon as libbpf drops
//! their fds. Holding an extra refcount via this helper keeps the
//! dying scheduler's map alive long enough for at least one
//! post-swap freeze to observe both copies, which is what the
//! framework's [`crate::scenario::snapshot::Snapshot::active`]
//! plus walker disambiguation chain exists to handle.

use anyhow::{Result, bail};
use libbpf_rs::libbpf_sys;
use libbpf_rs::query::MapInfoIter;
use std::io;
use std::os::fd::{FromRawFd, OwnedFd};

/// Walk the kernel's BPF map ID space, find the first map whose
/// `bpf_map_info.name` matches `name`, return its [`OwnedFd`]. The
/// caller holds the returned fd to keep the map alive (the kernel
/// refcount only drops to zero once every fd holder releases).
///
/// `name` is matched against the kernel-side map name by full-string
/// equality. BPF map names are NUL-terminated and capped at
/// `BPF_OBJ_NAME_LEN = 16` bytes (including the trailing NUL — so 15
/// usable chars max) per `kernel/bpf/syscall.c`'s `bpf_obj_name_cpy`.
/// Pass the kernel-visible name (typically `<obj>.bss` / `<obj>.data`
/// / `<obj>.rodata`); libbpf truncates long object prefixes to fit
/// the 15-char cap, so for a scheduler whose libbpf-source obj name
/// exceeds the limit, the kernel-visible name is the FIRST-15-chars
/// form. Reading a previous [`crate::monitor::dump::FailureDumpReport`]'s
/// `maps[].name` or running `bpftool map list` outside the test is
/// the safe way to discover the exact string the kernel sees.
///
/// **Order matters at the test layer**: this helper must run AFTER
/// the target scheduler's BPF object is loaded. The companion
/// [`crate::scenario::ops::Op::PinBpfMap`] doc documents the "place
/// after a hold long enough for the scheduler to be ready" pattern;
/// this helper itself does not block or retry.
///
/// **ID-order tiebreaker**: the underlying
/// [`libbpf_rs::query::MapInfoIter`] walks in monotonically-
/// increasing map-id order, so when multiple maps share the same
/// name (the same-binary swap window's multi-bss case), the lowest-
/// id (oldest) map is returned. For the swap-window scenario this
/// means: call BEFORE `Op::ReplaceScheduler` so the captured fd is
/// on the OUTGOING scheduler's map; the new scheduler's load will
/// then create a SECOND copy that's also kept alive because the
/// old refcount blocks the kernel from freeing the id.
///
/// **Error on miss**: returns Err naming every map name the walk
/// observed, so the caller can sanity-check what's actually loaded
/// (vs typo'd name vs scheduler-not-attached-yet vs map-already-freed).
///
/// **Privilege**: requires `CAP_SYS_ADMIN`. The kernel gates
/// `BPF_*_GET_NEXT_ID` and `BPF_MAP_GET_FD_BY_ID` on CAP_SYS_ADMIN
/// unconditionally (`kernel/bpf/syscall.c:4741` and `:4849`),
/// independent of `CAP_BPF` (which only governs prog/map creation).
/// ktstr always runs as root inside the guest VM so this is satisfied.
pub fn open_bpf_map_fd_by_name(name: &str) -> Result<OwnedFd> {
    let mut observed_names: Vec<String> = Vec::new();
    for info in MapInfoIter::default() {
        let map_name = info.name.to_string_lossy().into_owned();
        if map_name == name {
            // MapInfoIter consumes its per-iteration enumeration fd
            // at the end of `next()` (the OwnedFd it built drops),
            // so we re-open via id to obtain a caller-owned fd.
            // TOCTOU window: the map may have been freed between the
            // enumeration step and this call (e.g. the dying-side
            // BPF object's last fd just dropped); surface that as a
            // usable error rather than a silent test misfire.
            //
            // SAFETY: `bpf_map_get_fd_by_id` is a syscall wrapper
            // with no preconditions beyond a valid `u32` id; on
            // success it returns a kernel-owned file descriptor
            // that we take ownership of, on failure it returns -1
            // and sets errno.
            let fd = unsafe { libbpf_sys::bpf_map_get_fd_by_id(info.id) };
            if fd < 0 {
                bail!(
                    "BPF map '{name}' (id={}) disappeared between enumeration and \
                     fd-open: {}",
                    info.id,
                    io::Error::last_os_error(),
                );
            }
            // SAFETY: `fd` came from a successful kernel syscall on
            // the line above and has not been exposed to any other
            // code path, so we are the sole owner; transferring it
            // into `OwnedFd` makes the Drop close it at the right
            // time (when the caller drops the returned value).
            return Ok(unsafe { OwnedFd::from_raw_fd(fd) });
        }
        observed_names.push(map_name);
    }
    bail!(
        "BPF map '{name}' not found in any currently-attached BPF object — \
         scanned {} maps; observed names: {observed_names:?}. \
         Common causes: (a) the target scheduler's BPF object hasn't \
         finished loading yet (place this op AFTER a hold long enough for \
         the scheduler to be ready); (b) the requested name exceeds the \
         15-char usable cap of `BPF_OBJ_NAME_LEN` and was truncated by \
         libbpf when loaded — compare against the observed names above; \
         (c) the map has already been freed (no fd holders left).",
        observed_names.len(),
    );
}
