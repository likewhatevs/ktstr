//! Live-host BPF map accessor backed by the `bpf(2)` syscall.
//!
//! Companion to [`super::bpf_map::GuestMemMapAccessor`]: same trait
//! ([`super::bpf_map::BpfMapAccessor`]), different data path. Where
//! GuestMemMapAccessor walks frozen guest physical memory via PTE
//! resolution against `init_mm`, this backend talks directly to the
//! running host kernel through the `bpf()` syscall — KASLR is fully
//! abstracted, no symbol resolution required, no page-walk math.
//!
//! # Backend differences vs. guest-memory path
//!
//! | concern        | GuestMemMapAccessor                                            | BpfSyscallAccessor                                                           |
//! |----------------|----------------------------------------------------------------|------------------------------------------------------------------------------|
//! | discovery      | walk `map_idr` xarray in guest memory                          | `BPF_MAP_GET_NEXT_ID` + `BPF_MAP_GET_FD_BY_ID` loop                          |
//! | array values   | follow `bpf_array.value` flex array via PTE walks              | `BPF_MAP_LOOKUP_ELEM(fd, &key=0, buf)` returns the inline value bytes        |
//! | hash iteration | walk `bpf_htab.buckets` directly (freeze rendezvous = sync)    | `BPF_MAP_GET_NEXT_KEY` + `BPF_MAP_LOOKUP_ELEM` per key (kernel RCU read-side) |
//! | per-CPU array  | read each CPU's slot via `__per_cpu_offset[cpu]`               | one `BPF_MAP_LOOKUP_ELEM` returns `nr_possible_cpus * value_size` bytes      |
//! | arena          | walk `bpf_arena -> kern_vm -> vm_struct.addr` PTE-by-PTE        | `mmap(arena_fd, ...)` — `lookup_elem` returns `-EINVAL` on arena             |
//! | program BTF    | read split-BTF blob from guest memory                          | `BPF_BTF_GET_FD_BY_ID` + `BPF_OBJ_GET_INFO_BY_FD` to extract BTF bytes       |
//!
//! # Map fd pinning
//!
//! Every map discovered at construction time has its fd held open for
//! the lifetime of the accessor. The kernel's
//! `bpf_map_put`/`atomic64_dec_and_test` (`kernel/bpf/syscall.c`) only
//! frees a map when its refcount reaches zero, and userspace fds count
//! as references. This means the scheduler can exit and tear down its
//! struct_ops link while the accessor is still iterating maps — the
//! underlying memory stays valid.
//!
//! # Required capabilities
//!
//! `BPF_MAP_GET_NEXT_ID` and `BPF_MAP_GET_FD_BY_ID` require
//! `CAP_SYS_ADMIN` (or, since 5.16, `CAP_BPF` for some commands;
//! `..._GET_NEXT_ID` still requires `CAP_SYS_ADMIN`). ktstr always runs
//! as root in the test environment, so this is a non-issue for the
//! library's primary consumer; the `from_running_kernel` constructor
//! surfaces the kernel's `EPERM` directly so live-host CLI use cases
//! can produce a clear error.
//!
//! # Lock-free reads
//!
//! Without a freeze rendezvous, the kernel's per-element atomicity is
//! the only ordering primitive. Per-element u64-aligned fields are
//! atomic on x86_64; multi-element transactions the scheduler intended
//! to commit atomically may surface as torn views relative to the
//! walker. This is identical to the guest-memory backend's torn-read
//! behavior, just for a different reason. Two-snapshot in-BPF capture
//! (bpf_timer + tp_btf) is the recommended remedy and lives outside
//! this backend.

use std::os::fd::{AsRawFd, FromRawFd, OwnedFd, RawFd};
use std::ptr;

use anyhow::{Context, Result, anyhow};
use btf_rs::Btf;

use super::arena::{ArenaPage, ArenaSnapshot, BpfArenaOffsets};
use super::bpf_map::{
    BPF_MAP_TYPE_ARENA, BPF_MAP_TYPE_ARRAY, BPF_MAP_TYPE_HASH, BPF_MAP_TYPE_LRU_HASH,
    BPF_MAP_TYPE_LRU_PERCPU_HASH, BPF_MAP_TYPE_PERCPU_ARRAY, BPF_MAP_TYPE_PERCPU_HASH,
    BPF_MAP_TYPE_STRUCT_OPS, BpfMapAccessor, BpfMapInfo, MAP_MATERIALIZE_MAX,
};

/// `BPF_MAP_LOOKUP_ELEM` — read one map value into a userspace buffer.
const BPF_MAP_LOOKUP_ELEM: u32 = 1;
/// `BPF_MAP_GET_NEXT_KEY` — advance hash iteration cursor.
const BPF_MAP_GET_NEXT_KEY: u32 = 4;
/// `BPF_MAP_GET_NEXT_ID` — advance the kernel's map id walk.
const BPF_MAP_GET_NEXT_ID: u32 = 0xc;
/// `BPF_MAP_GET_FD_BY_ID` — pin a map by id.
const BPF_MAP_GET_FD_BY_ID: u32 = 0xe;
/// `BPF_OBJ_GET_INFO_BY_FD` — fetch map/btf metadata from an open fd.
const BPF_OBJ_GET_INFO_BY_FD: u32 = 0xf;
/// `BPF_BTF_GET_FD_BY_ID` — pin a BTF object by id.
/// Per `include/uapi/linux/bpf.h::enum bpf_cmd`: 19 (0x13). Counting
/// from `BPF_MAP_CREATE = 0` through `BPF_BTF_LOAD = 18` makes the
/// next entry `BPF_BTF_GET_FD_BY_ID = 19`.
const BPF_BTF_GET_FD_BY_ID: u32 = 0x13;

/// `BPF_OBJ_NAME_LEN` from `include/uapi/linux/bpf.h`.
const BPF_OBJ_NAME_LEN: usize = 16;

/// 4 KiB — page size for arena mmap. Matches `arena.c`'s
/// `PAGE_SIZE` (the kernel arena allocator works at 4 KiB granularity
/// regardless of host THP/hugetlb config).
const ARENA_PAGE_SIZE: usize = 4096;

/// Maximum total bytes the arena snapshot reads via mmap, mirroring the
/// guest-memory backend's `MAX_VM_RANGE_BYTES`. Keeps a runaway
/// `max_entries` from inducing a multi-GiB read.
const MAX_ARENA_BYTES: u64 = 4 * 1024 * 1024 * 1024;

/// Maximum number of arena pages enumerated sequentially before the
/// walker switches to a stride-probe sweep. Mirrors the
/// `MAX_ARENA_PAGES` cap on the guest-memory side so both backends
/// produce comparable snapshot extents.
const MAX_ARENA_PAGES: u64 = 16 * 1024;

// `bpf_attr` is a uapi union with many command-specific shapes. Rather
// than declare a full union we lay out per-command structs that match
// the relevant union arm exactly (uapi-stable size + field order). The
// kernel verifies the size passed to the syscall matches one of the
// recognized arm lengths; we pass `size_of::<arm>()` for each.

/// `bpf_attr` arm for `BPF_MAP_*_ELEM` and `BPF_MAP_GET_NEXT_KEY`.
/// Source: `include/uapi/linux/bpf.h::union bpf_attr` (the
/// MAP_ELEM_OPS arm).
#[repr(C)]
#[derive(Default)]
struct BpfAttrMapElem {
    map_fd: u32,
    _pad0: u32,
    key: u64,
    value_or_next_key: u64,
    flags: u64,
}

/// `bpf_attr` arm for `BPF_MAP_GET_NEXT_ID`, `BPF_BTF_GET_NEXT_ID`,
/// and the corresponding `*_GET_FD_BY_ID` commands.
#[repr(C)]
#[derive(Default)]
struct BpfAttrGetId {
    /// `start_id` for `*_GET_NEXT_ID`; `id` for `*_GET_FD_BY_ID`.
    id_or_start_id: u32,
    next_id: u32,
    open_flags: u32,
}

/// `bpf_attr` arm for `BPF_OBJ_GET_INFO_BY_FD`.
#[repr(C)]
#[derive(Default)]
struct BpfAttrInfoByFd {
    bpf_fd: u32,
    info_len: u32,
    info: u64,
}

/// `struct bpf_map_info` from `include/uapi/linux/bpf.h`. The kernel
/// has grown this struct over time; we pass our struct size as
/// `info_len` and the kernel zero-fills any tail it doesn't fill in.
/// All fields are documented in the kernel header.
#[repr(C)]
#[derive(Default)]
struct BpfMapInfoUapi {
    map_type: u32,
    id: u32,
    key_size: u32,
    value_size: u32,
    max_entries: u32,
    map_flags: u32,
    name: [u8; BPF_OBJ_NAME_LEN],
    ifindex: u32,
    btf_vmlinux_value_type_id: u32,
    netns_dev: u64,
    netns_ino: u64,
    btf_id: u32,
    btf_key_type_id: u32,
    btf_value_type_id: u32,
    /// Kernel field `btf_vmlinux_id` per
    /// `include/uapi/linux/bpf.h::struct bpf_map_info`. Unused by the
    /// caller; named `_pad` here because the value is currently
    /// discarded by the BPF accessor — rename without binding the
    /// field to a public consumer that can rot.
    _pad: u32,
    map_extra: u64,
}

/// `struct bpf_btf_info` from `include/uapi/linux/bpf.h`. Used to
/// extract a BTF blob's bytes given an open BTF fd.
#[repr(C)]
#[derive(Default)]
struct BpfBtfInfoUapi {
    btf: u64,
    btf_size: u32,
    id: u32,
    name: u64,
    name_len: u32,
    kernel_btf: u32,
}

/// Raw `bpf(2)` syscall wrapper. Returns the kernel's return value as
/// `i64` so callers can check for `< 0` and inspect `errno`. The
/// kernel's `__sys_bpf` (`kernel/bpf/syscall.c`) verifies the `size`
/// argument matches a recognized `bpf_attr` arm length.
///
/// SAFETY: `attr_ptr` must be a valid pointer to `attr_size` bytes of
/// the appropriate `bpf_attr` arm. The kernel reads the union by size,
/// so passing a smaller-than-required arm causes -EINVAL; passing a
/// larger one is rejected as well.
unsafe fn bpf_syscall(cmd: u32, attr_ptr: *const u8, attr_size: usize) -> i64 {
    // SAFETY: caller must ensure attr_ptr/attr_size validity. The
    // syscall itself is signal-safe and reentrant.
    unsafe { libc::syscall(libc::SYS_bpf, cmd as i64, attr_ptr, attr_size) as i64 }
}

/// Wrap a `bpf()` syscall result in a `Result<RawFd>` for commands
/// that return an fd. Negative returns are converted to errno-bearing
/// errors; non-negative returns become the fd.
fn bpf_call_fd(cmd: u32, attr_ptr: *const u8, attr_size: usize) -> Result<RawFd> {
    // SAFETY: caller has built attr_ptr/attr_size correctly per the
    // command's bpf_attr arm.
    let ret = unsafe { bpf_syscall(cmd, attr_ptr, attr_size) };
    if ret < 0 {
        let err = std::io::Error::last_os_error();
        Err(anyhow!("bpf({cmd}) failed: {err}"))
    } else {
        Ok(ret as RawFd)
    }
}

/// Wrap a `bpf()` syscall result for commands that return 0 on
/// success, `< 0` on error.
fn bpf_call_status(cmd: u32, attr_ptr: *const u8, attr_size: usize) -> Result<()> {
    // SAFETY: caller has built attr_ptr/attr_size correctly.
    let ret = unsafe { bpf_syscall(cmd, attr_ptr, attr_size) };
    if ret < 0 {
        let err = std::io::Error::last_os_error();
        Err(anyhow!("bpf({cmd}) failed: {err}"))
    } else {
        Ok(())
    }
}

/// One discovered map together with its pinned fd. The `OwnedFd`
/// guarantees the map's refcount stays >0 for the accessor's
/// lifetime — even if the scheduler exits and userspace tear-down
/// runs, `bpf_map_put` only frees when every fd is dropped (see
/// `kernel/bpf/syscall.c` `bpf_map_put`).
struct PinnedMap {
    info: BpfMapInfo,
    fd: OwnedFd,
    /// Raw `map_extra` from the kernel info struct. Arena maps
    /// hardcode this to a deterministic mmap target address (x86:
    /// `1<<44`, aarch64: `1<<32`) per `lib/arena_map.h`. Surfaced
    /// here so the arena mmap path can use `MAP_FIXED_NOREPLACE` at
    /// the kernel-blessed address rather than letting `mmap` pick
    /// one — which would diverge from what BPF programs see.
    map_extra: u64,
}

/// Live-host BPF map accessor.
///
/// Construction enumerates every map id reachable via
/// `BPF_MAP_GET_NEXT_ID`, opens an fd for each via
/// `BPF_MAP_GET_FD_BY_ID`, and caches the metadata. The fd vector is
/// held for the accessor's lifetime so the maps cannot be freed
/// underneath us — even if the scheduler exits and tears down its
/// struct_ops link mid-walk.
///
/// Selectively populating the cache is intentional: the same trait
/// surface accepts a `BpfMapInfo` argument on every method, so an
/// accessor that holds only the maps a particular failure dump cares
/// about (filtered by name suffix at construction time) is just as
/// valid as one that holds every map on the system. The
/// `from_running_kernel_filtered` constructor exposes that knob.
#[allow(dead_code)]
pub struct BpfSyscallAccessor {
    maps: Vec<PinnedMap>,
}

impl BpfSyscallAccessor {
    /// Discover and pin every BPF map currently visible to the
    /// running kernel.
    ///
    /// Walks the kernel's id space via `BPF_MAP_GET_NEXT_ID` (starting
    /// from id 0), pinning each map with `BPF_MAP_GET_FD_BY_ID` and
    /// fetching its metadata via `BPF_OBJ_GET_INFO_BY_FD`. Maps that
    /// disappear between the `NEXT_ID` and `GET_FD_BY_ID` calls (a
    /// concurrent scheduler unload, for instance) are silently
    /// skipped — that race is inherent to live-host enumeration and
    /// is not an error.
    ///
    /// Requires `CAP_SYS_ADMIN`. ktstr always runs as root in the
    /// test environment so this is a non-issue for the primary
    /// consumer; live-host CLI users that hit `EPERM` will see it
    /// in the returned error.
    #[allow(dead_code)]
    pub fn from_running_kernel() -> Result<Self> {
        Self::from_running_kernel_filtered(|_info: &BpfMapInfo| true)
    }

    /// Discover and pin every BPF map for which `predicate` returns
    /// `true`. Maps that fail the predicate are closed (their fds
    /// drop) so the kernel can free them as usual.
    ///
    /// Useful when the caller knows which maps the failure dump will
    /// touch — typically the scheduler's named maps that match a
    /// specific suffix — and wants to avoid pinning hundreds of
    /// unrelated maps that happen to be alive (cilium, systemd,
    /// other workloads).
    #[allow(dead_code)]
    pub fn from_running_kernel_filtered<F>(mut predicate: F) -> Result<Self>
    where
        F: FnMut(&BpfMapInfo) -> bool,
    {
        let mut maps: Vec<PinnedMap> = Vec::new();
        let mut start_id: u32 = 0;

        loop {
            // The kernel writes `next_id` via the syscall's raw pointer
            // path, but Rust's borrow checker doesn't see that — it
            // sees the struct as never mutated through a Rust binding.
            // Declare mut anyway so the compiler treats `attr.next_id`
            // as written, then read it back through a raw read after
            // the syscall returns.
            let mut attr = BpfAttrGetId {
                id_or_start_id: start_id,
                next_id: 0,
                open_flags: 0,
            };
            // SAFETY: BpfAttrGetId is repr(C) with the exact layout the
            // kernel expects for *_GET_NEXT_ID.
            let res = unsafe {
                bpf_syscall(
                    BPF_MAP_GET_NEXT_ID,
                    &raw mut attr as *const u8,
                    std::mem::size_of::<BpfAttrGetId>(),
                )
            };
            if res < 0 {
                let err = std::io::Error::last_os_error();
                if err.raw_os_error() == Some(libc::ENOENT) {
                    break;
                }
                return Err(anyhow!("BPF_MAP_GET_NEXT_ID failed: {err}"));
            }

            let next_id = attr.next_id;
            // Defensive: kernel returned 0 for `next_id` somehow.
            // Shouldn't happen on success, but guard against an
            // infinite loop.
            if next_id == 0 {
                break;
            }
            // Advance start_id for the next iteration BEFORE the
            // get-fd-by-id call so a transient EPERM/ENOENT on a
            // single id doesn't wedge the walk.
            start_id = next_id;

            // Try to pin the map. ENOENT here means the map was
            // freed between the NEXT_ID and GET_FD_BY_ID calls. The
            // kernel doesn't write to this attr (GET_FD_BY_ID is
            // input-only), so the binding is plain (no mut).
            let fd_attr = BpfAttrGetId {
                id_or_start_id: next_id,
                next_id: 0,
                open_flags: 0,
            };
            let fd_ret = unsafe {
                bpf_syscall(
                    BPF_MAP_GET_FD_BY_ID,
                    &raw const fd_attr as *const u8,
                    std::mem::size_of::<BpfAttrGetId>(),
                )
            };
            if fd_ret < 0 {
                // A failed `BPF_MAP_GET_FD_BY_ID` skips this map and
                // keeps walking — a single bad map must not abort
                // enumeration. The error categories matter for
                // diagnostics, so surface non-ENOENT cases via
                // tracing rather than silently dropping them:
                //
                // - `ENOENT`: the map was freed between
                //   `GET_NEXT_ID` and `GET_FD_BY_ID`. Routine
                //   under churn; suppressed at `debug` level so the
                //   normal log stays quiet.
                // - `EPERM`: missing CAP_SYS_ADMIN / CAP_BPF for
                //   this map (e.g. a kernel-internal map a less-
                //   privileged caller can't pin). Logged at `warn`
                //   so an operator who expects to see the map knows
                //   why it's missing.
                // - `EBADF` / others: a kernel-side state error.
                //   Logged at `warn` with the errno so the operator
                //   can correlate against `dmesg`.
                let err = std::io::Error::last_os_error();
                let raw = err.raw_os_error().unwrap_or(0);
                if raw == libc::ENOENT {
                    tracing::debug!(
                        map_id = next_id,
                        "BPF_MAP_GET_FD_BY_ID: map vanished mid-walk (ENOENT); skipping"
                    );
                } else {
                    tracing::warn!(
                        map_id = next_id,
                        errno = raw,
                        error = %err,
                        "BPF_MAP_GET_FD_BY_ID failed; skipping this map but continuing the walk"
                    );
                }
                continue;
            }
            // SAFETY: fd_ret >= 0; the kernel guarantees a valid fd
            // for non-negative returns.
            let fd = unsafe { OwnedFd::from_raw_fd(fd_ret as RawFd) };

            // Fetch info to populate BpfMapInfo + decide whether to
            // keep the fd. A failure here means the map's metadata
            // can't be read (kernel-side state error or fd was
            // closed mid-walk); surface it via tracing so the
            // operator sees the correlation rather than a silently
            // dropped map.
            let (info, map_extra) = match obj_get_info_map(fd.as_raw_fd()) {
                Ok(pair) => pair,
                Err(e) => {
                    tracing::warn!(
                        map_id = next_id,
                        error = %e,
                        "BPF_OBJ_GET_INFO_BY_FD failed for pinned map; skipping"
                    );
                    continue;
                }
            };

            // Hand the predicate a BpfMapInfo for the keep/discard
            // decision. Discarded fds drop here.
            if !predicate(&info) {
                continue;
            }

            maps.push(PinnedMap {
                info,
                fd,
                map_extra,
            });
        }

        Ok(Self { maps })
    }

    /// Number of pinned maps currently held. Test helper.
    #[cfg(test)]
    #[allow(dead_code)]
    pub(crate) fn pinned_count(&self) -> usize {
        self.maps.len()
    }

    /// Look up the pinned fd for a map identified by its
    /// `BpfMapInfo`. Returns `None` when no pinned map matches.
    ///
    /// Match key: `name` field (via [`info_name_matches`]). Map ids
    /// would be more precise but they're not part of `BpfMapInfo`
    /// today (a known follow-up if the live-host backend grows other
    /// consumers); within a single scheduler instance, names are
    /// unique and stable for the duration of the run.
    fn pinned_for(&self, target: &BpfMapInfo) -> Option<&PinnedMap> {
        self.maps
            .iter()
            .find(|p| info_name_matches(&p.info, target))
    }
}

/// Match key for [`BpfSyscallAccessor::pinned_for`] and the
/// construction-time predicate filter: two `BpfMapInfo`s identify the
/// same map iff their active name bytes
/// ([`BpfMapInfo::name_bytes_active`]) are byte-equal. Extracted as a
/// free fn so the keep/discard semantics are exercisable over a
/// hand-built `&[BpfMapInfo]` fixture without the live-kernel walk.
fn info_name_matches(a: &BpfMapInfo, b: &BpfMapInfo) -> bool {
    a.name_bytes_active() == b.name_bytes_active()
}

/// Pure mirror of the construction-time keep/discard step in
/// [`BpfSyscallAccessor::from_running_kernel_filtered`]: returns the
/// subset of `infos` for which `predicate` returns `true`, preserving
/// order. The production constructor applies the same
/// `if !predicate(&info) { continue; }` gate inline against each map
/// freshly fetched from the kernel; this fn lets a test pin the
/// filter's keep/discard contract over a deterministic fixture.
#[cfg(test)]
fn select_keeping<F>(infos: &[BpfMapInfo], mut predicate: F) -> Vec<&BpfMapInfo>
where
    F: FnMut(&BpfMapInfo) -> bool,
{
    infos.iter().filter(|info| predicate(info)).collect()
}

/// Fetch `bpf_map_info` for an open map fd via
/// `BPF_OBJ_GET_INFO_BY_FD`. Returns the populated [`BpfMapInfo`]
/// alongside the raw `map_extra` field — the latter is needed by the
/// arena mmap path but doesn't fit on the cross-backend
/// [`BpfMapInfo`] surface (the guest-memory path doesn't use it).
fn obj_get_info_map(fd: RawFd) -> Result<(BpfMapInfo, u64)> {
    let mut info = BpfMapInfoUapi::default();
    let attr = BpfAttrInfoByFd {
        bpf_fd: fd as u32,
        info_len: std::mem::size_of::<BpfMapInfoUapi>() as u32,
        info: &raw mut info as u64,
    };
    bpf_call_status(
        BPF_OBJ_GET_INFO_BY_FD,
        &raw const attr as *const u8,
        std::mem::size_of::<BpfAttrInfoByFd>(),
    )
    .context("BPF_OBJ_GET_INFO_BY_FD on map fd")?;

    let nul = info
        .name
        .iter()
        .position(|&b| b == 0)
        .unwrap_or(BPF_OBJ_NAME_LEN);
    let mut name_bytes = [0u8; BPF_OBJ_NAME_LEN];
    name_bytes.copy_from_slice(&info.name);

    Ok((
        BpfMapInfo {
            // map_pa / map_kva / value_kva are guest-memory concepts
            // that don't apply on the live host. Populating with 0
            // is fine — the live-host backend's read paths route
            // through the pinned fd, not these fields.
            map_pa: 0,
            map_kva: 0,
            name_bytes,
            name_len: nul as u8,
            map_type: info.map_type,
            map_flags: info.map_flags,
            key_size: info.key_size,
            value_size: info.value_size,
            max_entries: info.max_entries,
            value_kva: None,
            // btf_kva is similarly a guest-memory locator. Live-host
            // BTF resolution goes through `btf_id` →
            // `BPF_BTF_GET_FD_BY_ID` instead.
            btf_kva: u64::from(info.btf_id),
            btf_value_type_id: info.btf_value_type_id,
            // bpf(2) `BPF_OBJ_GET_INFO_BY_FD` does not surface
            // `btf_vmlinux_value_type_id` directly; the live-host
            // backend would need a parallel resolution path
            // (BPF_BTF_GET_INFO_BY_ID + walk the wrapper) to expose
            // it. Until that lands, leave 0 — the dump's STRUCT_OPS
            // arm falls through to hex on a zero id, matching the
            // behavior on guest-memory maps without struct_ops
            // BTF support.
            btf_vmlinux_value_type_id: 0,
            btf_key_type_id: info.btf_key_type_id,
        },
        info.map_extra,
    ))
}

impl BpfMapAccessor for BpfSyscallAccessor {
    fn maps(&self) -> Vec<BpfMapInfo> {
        self.maps.iter().map(|p| p.info.clone()).collect()
    }

    fn read_value(&self, map: &BpfMapInfo, offset: usize, len: usize) -> Option<Vec<u8>> {
        let pinned = self.pinned_for(map)?;

        // The live-host backend supports single-buffer value reads on
        // ARRAY (key=0 returns inline value bytes) and STRUCT_OPS
        // (key=0 returns the populated `bpf_struct_ops_value`). HASH
        // goes through `iter_hash_map`; PERCPU_ARRAY through
        // `read_percpu_array`; ARENA through `read_arena_pages`. Any
        // other type falls through to None so the dump renderer can
        // surface a specific reason.
        //
        // STRUCT_OPS quirk: the in-kernel
        // `bpf_struct_ops_map_lookup_elem` returns -EINVAL
        // (`kernel/bpf/bpf_struct_ops.c:518`), but the syscall path
        // `bpf_struct_ops_map_sys_lookup_elem`
        // (`kernel/bpf/bpf_struct_ops.c::struct_ops_map_sys_lookup_elem`)
        // implements its own lookup, copying the kernel's
        // `bpf_struct_ops_value` (refcnt + state + the registered
        // kernel struct) into the userspace buffer. The kernel-only
        // `lookup_elem` call is the in-program path; userspace
        // syscalls reach the sys variant.
        if map.map_type != BPF_MAP_TYPE_ARRAY && map.map_type != BPF_MAP_TYPE_STRUCT_OPS {
            return None;
        }

        // Build the lookup. ARRAY and STRUCT_OPS both use a u32 key;
        // STRUCT_OPS only ever has one entry (key=0).
        let mut key: u32 = 0;
        let mut buf = vec![0u8; map.value_size as usize];
        let attr = BpfAttrMapElem {
            map_fd: pinned.fd.as_raw_fd() as u32,
            _pad0: 0,
            key: &raw mut key as u64,
            value_or_next_key: buf.as_mut_ptr() as u64,
            flags: 0,
        };
        bpf_call_status(
            BPF_MAP_LOOKUP_ELEM,
            &raw const attr as *const u8,
            std::mem::size_of::<BpfAttrMapElem>(),
        )
        .ok()?;

        // Slice into the requested window. Out-of-bounds offsets
        // return None to mirror the guest-memory backend's behavior
        // when a value-region read straddles an unmapped page.
        let end = offset.checked_add(len)?;
        if end > buf.len() {
            return None;
        }
        Some(buf[offset..end].to_vec())
    }

    fn read_array(&self, map: &BpfMapInfo, key: u32) -> Option<Vec<u8>> {
        let pinned = self.pinned_for(map)?;
        // ARRAY only. STRUCT_OPS and single-entry global-section maps
        // go through read_value (key 0); HASH/PERCPU/ARENA have their
        // own methods. Replicate array_map_lookup_elem's pre-mask
        // `index >= max_entries` rejection (the kernel's index_mask is
        // a Spectre bound, not a range check).
        if map.map_type != BPF_MAP_TYPE_ARRAY {
            return None;
        }
        if key >= map.max_entries {
            return None;
        }
        // BPF_MAP_LOOKUP_ELEM copies value_size bytes for a plain
        // ARRAY (copy_map_value) — no per-entry stride padding, unlike
        // the PERCPU_ARRAY path which returns nr_cpus * round_up_8.
        // Pass the entry index as the key.
        let mut k: u32 = key;
        // No MAX_VALUE_SIZE cap here (unlike the guest-memory
        // `read_bpf_map_array_value`): value_size is sourced from
        // BPF_OBJ_GET_INFO_BY_FD (kernel-validated metadata), not
        // corruptible guest DRAM, so the kernel's own value_size
        // validation guards this allocation.
        let mut buf = vec![0u8; map.value_size as usize];
        let attr = BpfAttrMapElem {
            map_fd: pinned.fd.as_raw_fd() as u32,
            _pad0: 0,
            key: &raw mut k as u64,
            value_or_next_key: buf.as_mut_ptr() as u64,
            flags: 0,
        };
        bpf_call_status(
            BPF_MAP_LOOKUP_ELEM,
            &raw const attr as *const u8,
            std::mem::size_of::<BpfAttrMapElem>(),
        )
        .ok()?;
        Some(buf)
    }

    fn iter_hash_map(&self, map: &BpfMapInfo) -> Vec<(Vec<u8>, Vec<u8>)> {
        let Some(pinned) = self.pinned_for(map) else {
            return Vec::new();
        };
        // HASH and LRU_HASH share the inline-value `htab_elem` layout
        // (`kernel/bpf/hashtab.c::htab_elem_value`), and the kernel
        // syscall path returns the value bytes directly for both —
        // `bpf_map_copy_value` falls into the generic `map_lookup_elem`
        // arm for them. Reject other map types so callers route
        // PERCPU_HASH/LRU_PERCPU_HASH to `iter_percpu_hash_map` instead.
        if map.map_type != BPF_MAP_TYPE_HASH && map.map_type != BPF_MAP_TYPE_LRU_HASH {
            return Vec::new();
        }

        let key_sz = map.key_size as usize;
        let val_sz = map.value_size as usize;
        let mut out: Vec<(Vec<u8>, Vec<u8>)> = Vec::new();

        // First key: pass NULL for the input key per `bpf(2)` man
        // page — kernel returns the first key in the table.
        let mut cur_key = vec![0u8; key_sz];
        let mut next_key = vec![0u8; key_sz];

        // Cap iterations at max_entries * 2 to bound a pathological
        // walk on a torn table. RCU-protected reads on the kernel
        // side are best-effort across concurrent updates.
        let cap = (map.max_entries as usize).saturating_mul(2).max(1);
        let mut got_first = false;
        for _ in 0..cap {
            // Get next key.
            let attr = BpfAttrMapElem {
                map_fd: pinned.fd.as_raw_fd() as u32,
                _pad0: 0,
                key: if got_first {
                    cur_key.as_ptr() as u64
                } else {
                    0 // first call: NULL means "first key"
                },
                value_or_next_key: next_key.as_mut_ptr() as u64,
                flags: 0,
            };
            let ret = unsafe {
                bpf_syscall(
                    BPF_MAP_GET_NEXT_KEY,
                    &raw const attr as *const u8,
                    std::mem::size_of::<BpfAttrMapElem>(),
                )
            };
            if ret < 0 {
                // ENOENT marks end of iteration; anything else
                // ends the walk silently with whatever was
                // collected so far.
                break;
            }
            got_first = true;

            // Look up the value for next_key.
            let mut value = vec![0u8; val_sz];
            let lookup_attr = BpfAttrMapElem {
                map_fd: pinned.fd.as_raw_fd() as u32,
                _pad0: 0,
                key: next_key.as_ptr() as u64,
                value_or_next_key: value.as_mut_ptr() as u64,
                flags: 0,
            };
            let lret = unsafe {
                bpf_syscall(
                    BPF_MAP_LOOKUP_ELEM,
                    &raw const lookup_attr as *const u8,
                    std::mem::size_of::<BpfAttrMapElem>(),
                )
            };
            if lret >= 0 {
                out.push((next_key.clone(), value));
            }
            // Bound materialization at the renderer's cap (one-past so
            // render's truncation check fires); see MAP_MATERIALIZE_MAX.
            if out.len() > MAP_MATERIALIZE_MAX {
                break;
            }
            // Advance cursor — even when lookup failed (the key
            // disappeared between get_next_key and lookup_elem; a
            // concurrent delete is inherent to live-host walking).
            cur_key.copy_from_slice(&next_key);
        }

        out
    }

    fn read_percpu_array(&self, map: &BpfMapInfo, key: u32, num_cpus: u32) -> Vec<Option<Vec<u8>>> {
        let Some(pinned) = self.pinned_for(map) else {
            return Vec::new();
        };
        if map.map_type != BPF_MAP_TYPE_PERCPU_ARRAY {
            return Vec::new();
        }
        if key >= map.max_entries {
            return Vec::new();
        }

        let val_sz = map.value_size as usize;
        let total = (num_cpus as usize).saturating_mul(val_sz);
        let mut buf = vec![0u8; total];
        let mut k: u32 = key;
        let attr = BpfAttrMapElem {
            map_fd: pinned.fd.as_raw_fd() as u32,
            _pad0: 0,
            key: &raw mut k as u64,
            value_or_next_key: buf.as_mut_ptr() as u64,
            flags: 0,
        };
        if bpf_call_status(
            BPF_MAP_LOOKUP_ELEM,
            &raw const attr as *const u8,
            std::mem::size_of::<BpfAttrMapElem>(),
        )
        .is_err()
        {
            return vec![None; num_cpus as usize];
        }

        // Kernel rounds each CPU's slot up to 8 bytes internally
        // (see `kernel/bpf/syscall.c` bpf_map_value_size for the
        // PERCPU_ARRAY arm calling round_up_8). The returned buffer
        // is `nr_cpus * round_up_8(value_size)` bytes; we slice at
        // the rounded stride to extract each CPU's bytes and then
        // truncate to value_size.
        let stride = (val_sz + 7) & !7;
        let mut out = Vec::with_capacity(num_cpus as usize);
        for cpu in 0..num_cpus as usize {
            let start = cpu * stride;
            let end = start + val_sz;
            if end > buf.len() {
                out.push(None);
            } else {
                out.push(Some(buf[start..end].to_vec()));
            }
        }
        out
    }

    fn iter_percpu_hash_map(
        &self,
        map: &BpfMapInfo,
        num_cpus: u32,
    ) -> super::bpf_map::PerCpuHashEntries {
        let Some(pinned) = self.pinned_for(map) else {
            return Vec::new();
        };
        if map.map_type != BPF_MAP_TYPE_PERCPU_HASH && map.map_type != BPF_MAP_TYPE_LRU_PERCPU_HASH
        {
            return Vec::new();
        }

        let key_sz = map.key_size as usize;
        let val_sz = map.value_size as usize;
        // Kernel returns nr_cpus * round_up_8(value_size) bytes per
        // lookup (`bpf_percpu_hash_copy` -> `pcpu_copy_value`); same
        // 8-byte stride as PERCPU_ARRAY. The buffer must be sized to
        // the full stride or the kernel writes past it.
        let stride = (val_sz + 7) & !7;
        let buf_total = (num_cpus as usize).saturating_mul(stride);
        let mut out: super::bpf_map::PerCpuHashEntries = Vec::new();

        let mut cur_key = vec![0u8; key_sz];
        let mut next_key = vec![0u8; key_sz];

        let cap = (map.max_entries as usize).saturating_mul(2).max(1);
        let mut got_first = false;
        for _ in 0..cap {
            let attr = BpfAttrMapElem {
                map_fd: pinned.fd.as_raw_fd() as u32,
                _pad0: 0,
                key: if got_first {
                    cur_key.as_ptr() as u64
                } else {
                    0
                },
                value_or_next_key: next_key.as_mut_ptr() as u64,
                flags: 0,
            };
            let ret = unsafe {
                bpf_syscall(
                    BPF_MAP_GET_NEXT_KEY,
                    &raw const attr as *const u8,
                    std::mem::size_of::<BpfAttrMapElem>(),
                )
            };
            if ret < 0 {
                break;
            }
            got_first = true;

            let mut value_buf = vec![0u8; buf_total];
            let lookup_attr = BpfAttrMapElem {
                map_fd: pinned.fd.as_raw_fd() as u32,
                _pad0: 0,
                key: next_key.as_ptr() as u64,
                value_or_next_key: value_buf.as_mut_ptr() as u64,
                flags: 0,
            };
            let lret = unsafe {
                bpf_syscall(
                    BPF_MAP_LOOKUP_ELEM,
                    &raw const lookup_attr as *const u8,
                    std::mem::size_of::<BpfAttrMapElem>(),
                )
            };
            if lret >= 0 {
                let mut per_cpu = Vec::with_capacity(num_cpus as usize);
                for cpu in 0..num_cpus as usize {
                    let start = cpu * stride;
                    let end = start + val_sz;
                    if end > value_buf.len() {
                        per_cpu.push(None);
                    } else {
                        per_cpu.push(Some(value_buf[start..end].to_vec()));
                    }
                }
                out.push((next_key.clone(), per_cpu));
            }
            // Bound materialization at the renderer's cap (one-past so
            // render's truncation check fires); see MAP_MATERIALIZE_MAX.
            if out.len() > MAP_MATERIALIZE_MAX {
                break;
            }
            cur_key.copy_from_slice(&next_key);
        }

        out
    }

    fn read_arena_pages(
        &self,
        map: &BpfMapInfo,
        _arena_offsets: &BpfArenaOffsets,
    ) -> ArenaSnapshot {
        let Some(pinned) = self.pinned_for(map) else {
            return ArenaSnapshot::default();
        };
        if map.map_type != BPF_MAP_TYPE_ARENA {
            return ArenaSnapshot::default();
        }

        // Compute declared span. Same caps as the guest-memory side
        // for cross-backend parity.
        let declared_bytes_raw = (map.max_entries as u64).saturating_mul(ARENA_PAGE_SIZE as u64);
        let span_capped = declared_bytes_raw > MAX_ARENA_BYTES;
        let declared_bytes = declared_bytes_raw.min(MAX_ARENA_BYTES);
        let declared_pages = declared_bytes / ARENA_PAGE_SIZE as u64;

        // Use map_extra as the user_vm_start anchor. BPF programs
        // see arena addresses at this base (lib/arena_map.h hardcodes
        // it: x86 `1<<44`, aarch64 `1<<32`). Operators correlating
        // arena pointers want the same base in the snapshot.
        // Lifted above the early returns so every snapshot — empty
        // or populated — carries the anchor in `user_vm_start`; the
        // pointer-chasing reader needs it to classify arena addresses
        // even when the page set is empty.
        let user_vm_start = pinned.map_extra;

        if declared_pages == 0 {
            return ArenaSnapshot {
                pages: Vec::new(),
                truncated: false,
                declared_pages: 0,
                span_capped,
                user_vm_start,
                ..Default::default()
            };
        }

        // mmap the arena fd at offset 0 over the declared span. The
        // kernel's arena_vm_fault populates pages on first access;
        // unmapped pgoffs return SIGBUS — we use MAP_POPULATE so the
        // kernel walks every present page eagerly (without
        // allocating new ones). Sparse pgoffs that have never been
        // touched by the BPF program raise SIGBUS on first read —
        // we install a sigbus handler that longjmps out, marking
        // those pages as unmapped.
        //
        // Capping the mmap span at MAX_ARENA_PAGES * 4 KiB matches
        // the sequential-prefix cap on the guest-memory side; the
        // tail isn't stride-probed here because mmap covers the
        // whole window already. The `truncated` flag on
        // ArenaSnapshot still fires when the cap kicks in, for
        // operator visibility.
        let walk_pages = declared_pages.min(MAX_ARENA_PAGES);
        let walk_bytes = (walk_pages as usize) * ARENA_PAGE_SIZE;
        let truncated = declared_pages > walk_pages;

        // SAFETY: mmap with PROT_READ + MAP_SHARED on the arena fd
        // is exactly what the kernel exports for arena maps. The
        // `arena_map_mmap` op (`kernel/bpf/arena.c::arena_map_mmap`)
        // is the userspace mmap entry point; passing length =
        // walk_bytes and offset 0 maps the prefix.
        let addr = unsafe {
            libc::mmap(
                ptr::null_mut(),
                walk_bytes,
                libc::PROT_READ,
                libc::MAP_SHARED,
                pinned.fd.as_raw_fd(),
                0,
            )
        };
        if addr == libc::MAP_FAILED {
            // mmap rejected — return an empty snapshot with the
            // declared/truncation hints filled in for visibility.
            return ArenaSnapshot {
                pages: Vec::new(),
                truncated,
                declared_pages,
                span_capped,
                user_vm_start,
                ..Default::default()
            };
        }

        let mut pages: Vec<ArenaPage> = Vec::new();
        // Read every page out of the mmap. Pages whose pgoff was
        // never populated by the BPF program will raise SIGBUS;
        // install a sigaction with a setjmp longjmp to recover.
        // We do NOT install the handler here — instead we use
        // mincore() to filter out pages that aren't present, then
        // read only the present ones. mincore returns 0 for
        // resident pages, < 0 on error.
        let mut residency = vec![0u8; walk_pages as usize];
        let mincore_ret = unsafe { libc::mincore(addr, walk_bytes, residency.as_mut_ptr()) };
        if mincore_ret == 0 {
            for (idx, &resident) in residency.iter().enumerate() {
                if resident & 1 == 0 {
                    // Page not in core — sparse arena, never
                    // populated by the BPF program. Skip.
                    continue;
                }
                let page_addr = (addr as usize) + idx * ARENA_PAGE_SIZE;
                // SAFETY: page is resident per mincore; reading
                // ARENA_PAGE_SIZE bytes is in-bounds.
                let mut buf = vec![0u8; ARENA_PAGE_SIZE];
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        page_addr as *const u8,
                        buf.as_mut_ptr(),
                        ARENA_PAGE_SIZE,
                    );
                }
                // user_vm_start comes from the BPF map's map_extra
                // field — a guest-controllable value. A hostile or
                // corrupt map metadata could push the page identifier
                // past u64::MAX. Skip the page rather than emit a
                // wrapped address that consumers would treat as
                // legitimate.
                let Some(idx_offset) = (idx as u64).checked_mul(ARENA_PAGE_SIZE as u64) else {
                    continue;
                };
                let Some(user_addr) = user_vm_start.checked_add(idx_offset) else {
                    continue;
                };
                pages.push(ArenaPage {
                    user_addr,
                    bytes: buf,
                });
            }
        }

        // SAFETY: we created this mapping above and aren't using it
        // after this point.
        unsafe {
            libc::munmap(addr, walk_bytes);
        }

        ArenaSnapshot {
            pages,
            truncated,
            declared_pages,
            span_capped,
            user_vm_start,
            ..Default::default()
        }
    }

    fn load_program_btf(&self, map: &BpfMapInfo, base_btf: &Btf) -> Option<Btf> {
        // map.btf_kva on the live-host backend stores the kernel's
        // btf_id (u32) — see obj_get_info_map. 0 means no BTF.
        let btf_id = map.btf_kva as u32;
        if btf_id == 0 {
            return None;
        }

        // Pin the BTF object by id.
        let attr = BpfAttrGetId {
            id_or_start_id: btf_id,
            next_id: 0,
            open_flags: 0,
        };
        let btf_fd = bpf_call_fd(
            BPF_BTF_GET_FD_BY_ID,
            &raw const attr as *const u8,
            std::mem::size_of::<BpfAttrGetId>(),
        )
        .ok()?;
        // SAFETY: btf_fd >= 0 from a successful bpf_call_fd.
        let btf_owned = unsafe { OwnedFd::from_raw_fd(btf_fd) };

        // Two-pass info fetch: first call to learn btf_size, then
        // allocate a buffer and refetch with `btf` populated to
        // pull the BTF blob bytes.
        let mut info = BpfBtfInfoUapi::default();
        let info_attr = BpfAttrInfoByFd {
            bpf_fd: btf_owned.as_raw_fd() as u32,
            info_len: std::mem::size_of::<BpfBtfInfoUapi>() as u32,
            info: &raw mut info as u64,
        };
        bpf_call_status(
            BPF_OBJ_GET_INFO_BY_FD,
            &raw const info_attr as *const u8,
            std::mem::size_of::<BpfAttrInfoByFd>(),
        )
        .ok()?;
        if info.btf_size == 0 {
            return None;
        }

        // Second pass with a real buffer.
        let mut buf = vec![0u8; info.btf_size as usize];
        info.btf = buf.as_mut_ptr() as u64;
        let info_attr2 = BpfAttrInfoByFd {
            bpf_fd: btf_owned.as_raw_fd() as u32,
            info_len: std::mem::size_of::<BpfBtfInfoUapi>() as u32,
            info: &raw mut info as u64,
        };
        bpf_call_status(
            BPF_OBJ_GET_INFO_BY_FD,
            &raw const info_attr2 as *const u8,
            std::mem::size_of::<BpfAttrInfoByFd>(),
        )
        .ok()?;

        if info.kernel_btf != 0 {
            Btf::from_split_bytes(&buf, base_btf).ok()
        } else {
            Btf::from_bytes(&buf).ok()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify the bpf_attr arms have the exact UAPI layout the
    /// kernel expects. Wrong sizes or field offsets cause -EINVAL
    /// on every syscall — this test catches the layout drift before
    /// it produces silent failures at runtime.
    #[test]
    fn bpf_attr_map_elem_size() {
        // include/uapi/linux/bpf.h: the MAP_ELEM_OPS arm is exactly
        // 32 bytes (4 + 4 pad + 8 + 8 + 8).
        assert_eq!(std::mem::size_of::<BpfAttrMapElem>(), 32);
    }

    #[test]
    fn bpf_attr_get_id_size() {
        // GET_NEXT_ID / GET_FD_BY_ID arm: 12 bytes (4 + 4 + 4)
        // — the kernel doesn't pad this struct to 8 bytes; size
        // matches the union arm exactly.
        assert_eq!(std::mem::size_of::<BpfAttrGetId>(), 12);
    }

    #[test]
    fn bpf_attr_info_by_fd_size() {
        // OBJ_GET_INFO_BY_FD arm: 16 bytes (4 + 4 + 8).
        assert_eq!(std::mem::size_of::<BpfAttrInfoByFd>(), 16);
    }

    /// `bpf_map_info` must be at least the historical minimum
    /// (the kernel rejects info_len smaller than its known floor).
    /// Modern kernels accept the full struct including map_extra.
    ///
    /// Verdict-routed so a multi-field uapi-shape regression
    /// surfaces every drift in one run rather than failing on
    /// the first mismatch.
    #[test]
    fn bpf_map_info_uapi_layout() {
        use crate::assert::Verdict;

        let off_map_type = std::mem::offset_of!(BpfMapInfoUapi, map_type);
        let off_name = std::mem::offset_of!(BpfMapInfoUapi, name);
        let total_size = std::mem::size_of::<BpfMapInfoUapi>();
        let off_map_extra = std::mem::offset_of!(BpfMapInfoUapi, map_extra);
        let map_extra_tail = off_map_extra + 8;

        let mut v = Verdict::new();
        // map_type at offset 0 per uapi.
        crate::claim!(v, off_map_type).eq(0usize);
        // name at offset 24 per uapi.
        crate::claim!(v, off_name).eq(24usize);
        // map_extra is the trailing field — its offset + 8 should
        // equal total struct size.
        crate::claim!(v, map_extra_tail).eq(total_size);
        let r = v.into_result();
        assert!(
            r.is_pass(),
            "bpf_map_info uapi layout drift: {:?}",
            r.outcomes,
        );
    }

    /// Round-up arithmetic for percpu stride matches the kernel's
    /// `round_up(value_size, 8)`.
    #[test]
    fn percpu_stride_round_up() {
        let cases = [
            (0usize, 0),
            (1, 8),
            (7, 8),
            (8, 8),
            (9, 16),
            (15, 16),
            (16, 16),
        ];
        for (val_sz, expected) in cases {
            let stride = (val_sz + 7) & !7;
            assert_eq!(stride, expected, "value_size {val_sz} → stride {stride}");
        }
    }

    /// Build a `BpfMapInfo` whose only populated field is the active
    /// name — the key both the construction-time predicate filter
    /// and `pinned_for` match on.
    fn info_named(name: &str) -> BpfMapInfo {
        let bytes = name.as_bytes();
        assert!(bytes.len() <= BPF_OBJ_NAME_LEN, "test name too long");
        let mut name_bytes = [0u8; BPF_OBJ_NAME_LEN];
        name_bytes[..bytes.len()].copy_from_slice(bytes);
        BpfMapInfo {
            name_bytes,
            name_len: bytes.len() as u8,
            ..Default::default()
        }
    }

    /// The construction-time keep/discard filter
    /// ([`select_keeping`], the pure mirror of
    /// `from_running_kernel_filtered`'s inline predicate gate) keeps
    /// exactly the maps the predicate accepts: a predicate matching
    /// no names yields the empty set; a name-suffix predicate yields
    /// exactly the matching subset, in order.
    #[test]
    fn predicate_filters_pinned_set() {
        let infos = vec![
            info_named("scx_central"),
            info_named("central_dsq"),
            info_named("cilium_lb"),
            info_named("central_data"),
        ];

        // Match-nothing predicate ⇒ empty kept set.
        let none = select_keeping(&infos, |_| false);
        assert!(none.is_empty(), "false predicate must discard every map");

        // Match-everything predicate ⇒ full set, order preserved.
        let all = select_keeping(&infos, |_| true);
        assert_eq!(all.len(), 4, "true predicate must keep every map");
        assert_eq!(all[0].name(), "scx_central");
        assert_eq!(all[3].name(), "central_data");

        // Name-suffix predicate ⇒ exactly the "central"-bearing subset.
        let kept = select_keeping(&infos, |i| i.name().contains("central"));
        let kept_names: Vec<String> = kept.iter().map(|i| i.name().to_string()).collect();
        assert_eq!(
            kept_names,
            vec!["scx_central", "central_dsq", "central_data"],
            "suffix predicate must keep exactly the matching subset, in order",
        );

        // The same name-match key drives pinned_for: a target sharing
        // active name bytes matches; a differing name does not.
        assert!(
            info_name_matches(&info_named("central_dsq"), &info_named("central_dsq")),
            "identical active name bytes must match",
        );
        assert!(
            !info_name_matches(&info_named("central_dsq"), &info_named("central_data")),
            "differing active name bytes must NOT match",
        );
        // name_len bounds the compared region: a longer NUL-padded
        // buffer with a shorter name_len compares only the active
        // prefix, so "scx" (len 3) does not match "scx_central".
        assert!(
            !info_name_matches(&info_named("scx"), &info_named("scx_central")),
            "name_len must bound the match to the active prefix",
        );
    }

    /// A cheap real fd for a test [`PinnedMap`]. `pinned_for`
    /// (lines 446-450) only compares names and never dereferences
    /// `.fd`, and every accessor read path under test returns at an
    /// early guard before the fd reaches a `bpf()` syscall — so any
    /// open fd suffices to satisfy the `OwnedFd` field. `/dev/null`
    /// is always present on Linux and `File` -> `OwnedFd` is the
    /// std-only conversion (no extra libc unsafe).
    fn dummy_fd() -> OwnedFd {
        OwnedFd::from(
            std::fs::File::open("/dev/null").expect("/dev/null must open on Linux test host"),
        )
    }

    /// Build a [`PinnedMap`] from `info` + `map_extra`, carrying the
    /// dummy fd. The fields are private to the parent module; the
    /// `tests` child module can name and construct them directly,
    /// which is the inject seam the blueprint requires (no live
    /// kernel walk, no `from_running_kernel*` syscall path).
    fn pinned(info: BpfMapInfo, map_extra: u64) -> PinnedMap {
        PinnedMap {
            info,
            fd: dummy_fd(),
            map_extra,
        }
    }

    /// Build a [`BpfSyscallAccessor`] holding exactly `maps`. The
    /// production constructors (`from_running_kernel*`) only ever
    /// populate `maps` via the live bpf(2) id walk; this literal is
    /// the host-test inject seam that lets the early-guard branches
    /// run without a kernel.
    fn accessor(maps: Vec<PinnedMap>) -> BpfSyscallAccessor {
        BpfSyscallAccessor { maps }
    }

    /// `read_array` rejects before touching the fd on three guards:
    /// no name match (`pinned_for` -> None, lines 600), wrong map
    /// type (lines 606-608), and `key >= max_entries` (lines
    /// 609-611). All three return `None` and run ahead of the
    /// `bpf()` lookup at line 630, so they are host-isolable through
    /// the inject seam. The `key < max_entries` success path is NOT
    /// asserted here — it issues a real `BPF_MAP_LOOKUP_ELEM`.
    #[test]
    fn read_array_pre_lookup_guards_reject() {
        let arr = BpfMapInfo {
            map_type: BPF_MAP_TYPE_ARRAY,
            max_entries: 4,
            value_size: 8,
            ..info_named("arr")
        };
        let acc = accessor(vec![pinned(arr.clone(), 0)]);

        // No pinned map matches the target name -> pinned_for None.
        assert_eq!(acc.read_array(&info_named("missing"), 0), None);

        // Name matches but map_type is HASH, not ARRAY -> type-reject.
        let hash = BpfMapInfo {
            map_type: BPF_MAP_TYPE_HASH,
            ..info_named("arr")
        };
        assert_eq!(acc.read_array(&hash, 0), None);

        // key == max_entries and key > max_entries both reject before
        // the lookup (the kernel index_mask is a Spectre bound, the
        // explicit `key >= max_entries` is the range check).
        assert_eq!(acc.read_array(&arr, 4), None);
        assert_eq!(acc.read_array(&arr, 99), None);
    }

    /// `read_value` rejects before touching the fd when the target
    /// name has no pinned match (`pinned_for` -> None, line 547) or
    /// the map type is neither ARRAY nor STRUCT_OPS (lines 567-569,
    /// ahead of the `bpf()` lookup at line 582). The post-lookup
    /// window-bounds / `checked_add` guards (lines 592-595) sit past
    /// the live lookup and are NOT asserted here.
    #[test]
    fn read_value_pre_lookup_type_reject() {
        // PERCPU_HASH is neither ARRAY nor STRUCT_OPS.
        let percpu_hash = BpfMapInfo {
            map_type: BPF_MAP_TYPE_PERCPU_HASH,
            value_size: 8,
            ..info_named("v")
        };
        let acc = accessor(vec![pinned(percpu_hash.clone(), 0)]);

        assert_eq!(acc.read_value(&info_named("nomatch"), 0, 4), None);
        assert_eq!(acc.read_value(&percpu_hash, 0, 4), None);
    }

    /// `iter_hash_map` returns an empty `Vec` before any fd use on
    /// the no-name-match let-else (lines 640-642) and the type-reject
    /// (lines 649-651: only HASH and LRU_HASH proceed). The
    /// iteration loop at line 667 needs a live hash map.
    #[test]
    fn iter_hash_map_pre_walk_guards_empty() {
        let arr = BpfMapInfo {
            map_type: BPF_MAP_TYPE_ARRAY,
            ..info_named("h")
        };
        let acc = accessor(vec![pinned(arr.clone(), 0)]);

        // No pinned match -> let-else returns Vec::new().
        assert_eq!(acc.iter_hash_map(&info_named("none")).len(), 0);
        // Name matches but map_type is ARRAY, not HASH/LRU_HASH.
        assert_eq!(acc.iter_hash_map(&arr).len(), 0);
    }

    /// `read_percpu_array` returns an empty `Vec` (length 0) on the
    /// three pre-lookup guards: no name match (lines 729-731), wrong
    /// map type (lines 732-734), and `key >= max_entries` (lines
    /// 735-737). The length distinguishes these from the
    /// post-lookup-failure branch at line 757 which returns
    /// `vec![None; num_cpus]` (length `num_cpus`), so the assertions
    /// pin LENGTH 0, not just emptiness-of-content.
    #[test]
    fn read_percpu_array_pre_lookup_guards_empty() {
        let pa = BpfMapInfo {
            map_type: BPF_MAP_TYPE_PERCPU_ARRAY,
            max_entries: 2,
            value_size: 8,
            ..info_named("pa")
        };
        let acc = accessor(vec![pinned(pa.clone(), 0)]);

        // No pinned match.
        assert_eq!(acc.read_percpu_array(&info_named("x"), 0, 4).len(), 0);
        // Name matches but map_type is ARRAY, not PERCPU_ARRAY.
        let arr = BpfMapInfo {
            map_type: BPF_MAP_TYPE_ARRAY,
            ..info_named("pa")
        };
        assert_eq!(acc.read_percpu_array(&arr, 0, 4).len(), 0);
        // key == max_entries rejects with length 0 (distinct from the
        // num_cpus-length lookup-failure vector).
        assert_eq!(acc.read_percpu_array(&pa, 2, 4).len(), 0);
    }

    /// `iter_percpu_hash_map` returns an empty `PerCpuHashEntries`
    /// before any fd use on the no-name-match let-else (lines
    /// 785-787) and the type-reject (lines 788-791: only PERCPU_HASH
    /// and LRU_PERCPU_HASH proceed). The walk loop at line 808 needs
    /// a live map.
    #[test]
    fn iter_percpu_hash_map_pre_walk_guards_empty() {
        let hash = BpfMapInfo {
            map_type: BPF_MAP_TYPE_HASH,
            ..info_named("ph")
        };
        let acc = accessor(vec![pinned(hash.clone(), 0)]);

        // No pinned match.
        assert_eq!(acc.iter_percpu_hash_map(&info_named("none"), 4).len(), 0);
        // Name matches but map_type is HASH, not PERCPU_HASH/LRU_PERCPU_HASH.
        assert_eq!(acc.iter_percpu_hash_map(&hash, 4).len(), 0);
    }

    /// `read_arena_pages` has three isolable, fd-free blocks ahead of
    /// `mmap` (line 935): no name match (lines 876-878 ->
    /// `ArenaSnapshot::default()`), wrong map type (lines 879-881 ->
    /// default), and the declared-span math + `declared_pages == 0`
    /// early return (lines 885-909). The span math is pure:
    /// `declared_bytes_raw = max_entries * 4096` (saturating),
    /// `span_capped = declared_bytes_raw > MAX_ARENA_BYTES` (4 GiB),
    /// and the zero-page snapshot carries `user_vm_start = map_extra`.
    /// The populated mmap/mincore path needs a live arena fd.
    #[test]
    fn read_arena_pages_pre_mmap_paths() {
        // A 3-field literal: BpfArenaOffsets derives only Debug+Clone
        // (no Default), and the value is unused on every path under
        // test (the fn parameter `_arena_offsets` is ignored), so the
        // concrete offsets are arbitrary.
        let offsets = BpfArenaOffsets {
            arena_kern_vm: 0,
            arena_user_vm_start: 0,
            vm_struct_addr: 0,
        };

        // max_entries == 0 -> declared_pages == 0 early return,
        // carrying user_vm_start = map_extra.
        let arena0 = BpfMapInfo {
            map_type: BPF_MAP_TYPE_ARENA,
            max_entries: 0,
            ..info_named("a")
        };
        let acc = accessor(vec![pinned(arena0.clone(), 0x1000)]);

        // No name match -> ArenaSnapshot::default() (all-zero).
        let no_match = acc.read_arena_pages(&info_named("no"), &offsets);
        assert!(no_match.pages.is_empty());
        assert_eq!(no_match.declared_pages, 0);
        assert_eq!(no_match.user_vm_start, 0);
        assert!(!no_match.span_capped);
        assert!(!no_match.truncated);

        // Name matches but map_type is ARRAY, not ARENA -> default.
        let arr = BpfMapInfo {
            map_type: BPF_MAP_TYPE_ARRAY,
            ..info_named("a")
        };
        let type_reject = acc.read_arena_pages(&arr, &offsets);
        assert!(type_reject.pages.is_empty());
        assert_eq!(type_reject.declared_pages, 0);
        assert_eq!(type_reject.user_vm_start, 0);
        assert!(!type_reject.span_capped);

        // declared_pages == 0 path: empty pages, span not capped,
        // user_vm_start carried through from map_extra.
        let zero = acc.read_arena_pages(&arena0, &offsets);
        assert_eq!(zero.pages.len(), 0);
        assert_eq!(zero.declared_pages, 0);
        assert!(!zero.span_capped);
        assert_eq!(zero.user_vm_start, 0x1000);
        assert!(!zero.truncated);

        // max_entries == u32::MAX -> declared_bytes_raw =
        // 0xFFFF_FFFF * 4096 > 4 GiB, so span_capped is set. With the
        // span capped to MAX_ARENA_BYTES, declared_pages > 0, so the
        // span-math result is only observable on this sub-case
        // through the MAP_FAILED snapshot or a populated walk — both
        // need a live fd. The dummy /dev/null fd makes mmap fail
        // (MAP_FAILED), exercising the lines 945-956 early return,
        // which carries span_capped + user_vm_start. Assert exactly
        // those two carry-through fields, which the blueprint marks
        // host-assertable.
        let arena_max = BpfMapInfo {
            map_type: BPF_MAP_TYPE_ARENA,
            max_entries: u32::MAX,
            ..info_named("a")
        };
        let acc_max = accessor(vec![pinned(arena_max.clone(), 0x2000)]);
        let capped = acc_max.read_arena_pages(&arena_max, &offsets);
        assert!(capped.span_capped, "u32::MAX max_entries must cap the span");
        assert_eq!(capped.user_vm_start, 0x2000);
    }

    /// `load_program_btf` returns `None` immediately when
    /// `btf_id == 0` (lines 1024-1027, `btf_id = map.btf_kva as u32`).
    /// `info_named` leaves `btf_kva` at its `Default` (0), so the
    /// guard fires before `BPF_BTF_GET_FD_BY_ID` (line 1035) and the
    /// `base_btf` argument is never dereferenced (it is only used on
    /// the post-fetch arms at lines 1079/1081). Only this guard is
    /// host-assertable; the BTF-fetch path needs a live kernel BTF
    /// object.
    #[test]
    fn load_program_btf_btf_id_zero_returns_none() {
        // info_named leaves btf_kva == 0 (Default).
        let prog = info_named("prog");
        assert_eq!(prog.btf_kva, 0, "info_named must default btf_kva to 0");
        let acc = accessor(vec![pinned(prog.clone(), 0)]);

        // base_btf: a minimal valid BTF blob — magic 0xEB9F, version 1,
        // 24-byte header, one Int type (id 1) so the type section is
        // non-empty, strtab leading with NUL. Mirrors the
        // `cast_analysis` tests' `build_btf` minimal layout. Never
        // dereferenced on the btf_id==0 path; built only to satisfy
        // the `&Btf` parameter.
        let base = minimal_btf();
        // `btf_rs::Btf` derives neither `PartialEq` nor `Debug`, so
        // `assert_eq!(.., None)` cannot be used on `Option<Btf>`;
        // `is_none()` is the exact discriminant check here (the
        // btf_id==0 guard returns the `None` variant outright, with
        // no `Btf` value to compare).
        assert!(
            acc.load_program_btf(&prog, &base).is_none(),
            "btf_id==0 must short-circuit to None before any bpf() call",
        );
    }

    /// Hand-build a minimal parseable BTF blob: a single `int` type
    /// (id 1, named "u64", 8 bytes) plus a NUL-led string table,
    /// wrapped in the 24-byte BTF header. Layout verified against the
    /// in-tree `src/monitor/cast_analysis/tests/mod.rs::build_btf`
    /// minimal path (the `empty_btf_no_panic` test proves a
    /// single-Int blob parses via `Btf::from_bytes`).
    fn minimal_btf() -> Btf {
        // String table: leading NUL (offset 0 = anonymous) + "u64\0".
        let mut strings: Vec<u8> = vec![0];
        let n_u64 = strings.len() as u32;
        strings.extend_from_slice(b"u64\0");

        // Type section: one BTF_KIND_INT (kind 1).
        let mut type_section: Vec<u8> = Vec::new();
        const BTF_KIND_INT: u32 = 1;
        type_section.extend_from_slice(&n_u64.to_le_bytes()); // name_off
        let info = (BTF_KIND_INT << 24) & 0x1f00_0000; // vlen 0
        type_section.extend_from_slice(&info.to_le_bytes());
        type_section.extend_from_slice(&8u32.to_le_bytes()); // size = 8
        // btf_int data word: encoding 0, offset 0, bits 64.
        let int_data: u32 = 64;
        type_section.extend_from_slice(&int_data.to_le_bytes());

        let type_len = type_section.len() as u32;
        let str_len = strings.len() as u32;

        let mut blob: Vec<u8> = Vec::new();
        blob.extend_from_slice(&0xEB9F_u16.to_le_bytes()); // magic
        blob.push(1); // version
        blob.push(0); // flags
        blob.extend_from_slice(&24u32.to_le_bytes()); // hdr_len
        blob.extend_from_slice(&0u32.to_le_bytes()); // type_off
        blob.extend_from_slice(&type_len.to_le_bytes()); // type_len
        blob.extend_from_slice(&type_len.to_le_bytes()); // str_off = type_len
        blob.extend_from_slice(&str_len.to_le_bytes()); // str_len
        blob.extend_from_slice(&type_section);
        blob.extend_from_slice(&strings);

        Btf::from_bytes(&blob).expect("minimal synthetic BTF must parse")
    }
}
