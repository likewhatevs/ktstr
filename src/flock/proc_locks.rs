//! `/proc/locks` scanner — enumerate processes holding a given flock.
//!
//! Path → needle → /proc/locks scan → [`super::HolderInfo`] list.
//! The needle (`{major:02x}:{minor:02x}:{inode}`) is derived from the
//! lockfile path via [`super::mountinfo`]; this module owns the
//! /proc/locks side of the pipeline.
//!
//! [`read_holders`] is the one-shot entry point: it reads
//! `/proc/self/mountinfo` and `/proc/locks`, then returns the [`HolderInfo`]
//! list for one lockfile. Callers with only a path use it (the `ktstr locks`
//! observational scan and the cache EWOULDBLOCK peer-holder lookup); admission
//! placement does NOT — it derives holder counts from the registry aggregate.
//!
//! The pure parser seam [`parse_flock_pids_for_needle`] is exposed so tests
//! can feed synthetic `/proc/locks` fixtures (POSIX / OFDLCK / FLOCK
//! interleavings, malformed lines) without touching the real filesystem.

use anyhow::{Context, Result};
use std::path::Path;

use super::HolderInfo;
use super::holder::holder_info_for_pid;
use super::mountinfo::needle_from_path;

fn read_proc_locks(context: &'static str) -> Result<String> {
    #[cfg(test)]
    PROC_LOCKS_READS.with(|count| count.set(count.get().saturating_add(1)));
    std::fs::read_to_string("/proc/locks").with_context(|| context)
}

#[cfg(test)]
thread_local! {
    static PROC_LOCKS_READS: std::cell::Cell<usize> =
        const { std::cell::Cell::new(0) };
}

#[cfg(test)]
pub(crate) fn proc_locks_read_count_for_tests() -> usize {
    PROC_LOCKS_READS.with(std::cell::Cell::get)
}

/// Parse `/proc/locks` and return [`HolderInfo`] entries for every
/// process holding an advisory `FLOCK` matching `needle`.
///
/// `needle` must be the `{major:02x}:{minor:02x}:{inode}` triple in
/// /proc/locks' own formatting, produced by
/// [`super::mountinfo::needle_from_path`]:
/// `(major, minor)` via `/proc/self/mountinfo` and `inode` via
/// `stat().st_ino`. Used by path-only callers ([`read_holders`],
/// the `ktstr locks` observational scan, and the EWOULDBLOCK-branch
/// peer-holder lookup in `src/cache/cache_dir.rs`). `acquire_llc_plan`
/// derives holder occupancy from the registry aggregate, not this scan.
///
/// Best-effort: returns `Ok(vec![])` when no /proc/locks entry
/// matches the needle, and propagates only the hard `/proc/locks`
/// read failure.
///
/// For each matching PID, reads `/proc/{pid}/cmdline`, decodes as
/// lossy UTF-8, replaces `\0` with ` `, and truncates to
/// `super::holder::CMDLINE_MAX_CHARS` with a `…` suffix on
/// overflow. A cmdline read failure is non-fatal — the entry
/// carries `"<cmdline unavailable>"` so the pid still surfaces.
pub(super) fn read_holders_for_needle(needle: &str) -> Result<Vec<HolderInfo>> {
    let contents = read_proc_locks("read /proc/locks for lockfile holder lookup")?;
    Ok(read_holders_from_contents(&contents, needle))
}

/// Content-based seam behind [`read_holders_for_needle`]. Takes
/// already-read `/proc/locks` `contents` plus one match `needle` and
/// returns the [`HolderInfo`] vector.
///
/// Thin shell over [`parse_flock_pids_for_needle`]: the latter
/// filters `/proc/locks` lines to the matching FLOCK PIDs; this
/// function adds the per-PID cmdline lookup via
/// [`super::holder::holder_info_for_pid`] that the
/// [`read_holders_for_needle`] caller expects.
pub(super) fn read_holders_from_contents(contents: &str, needle: &str) -> Vec<HolderInfo> {
    let pids = parse_flock_pids_for_needle(contents, needle);
    pids.into_iter().map(holder_info_for_pid).collect()
}

/// Pure parser seam behind [`read_holders_for_needle`]. Takes
/// already-read `/proc/locks` `contents` and the match `needle`, walks
/// every line, and returns the PIDs of processes holding a FLOCK
/// whose `{major:02x}:{minor:02x}:{inode}` triple byte-equals the
/// needle. POSIX-byte-range locks (`POSIX`) and open-file-description
/// locks (`OFDLCK`) are skipped — ktstr coordinates exclusively
/// through `flock(2)`, and misclassifying a POSIX range-lock as a
/// ktstr holder would confuse the holder-enumeration diagnostic.
///
/// Exposed as `pub(crate)` so tests can feed synthetic `/proc/locks`
/// fixtures (POSIX + OFDLCK + FLOCK interleavings, malformed lines,
/// empty input) without touching the real filesystem. The production
/// wrapper above reads `/proc/locks` and calls this seam; everything
/// below is pure text processing.
pub(crate) fn parse_flock_pids_for_needle(contents: &str, needle: &str) -> Vec<u32> {
    let mut pids: Vec<u32> = Vec::new();
    for line in contents.lines() {
        let Some((pid, dev_inode)) = parse_held_flock(line) else {
            continue;
        };
        if dev_inode == needle && !pids.contains(&pid) {
            pids.push(pid);
        }
    }
    pids
}

/// Return `(pid, dev:inode)` for a held FLOCK line.
///
/// Waiting flock entries contain an extra `->` field after the lock
/// id, so their second token is not `FLOCK` and they are intentionally
/// ignored: holder diagnostics and placement must describe current
/// holders, not queued contenders.
fn parse_held_flock(line: &str) -> Option<(u32, &str)> {
    parse_held_flock_with_mode(line).map(|(pid, _mode, dev_inode)| (pid, dev_inode))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum HeldFlockMode {
    Shared,
    Exclusive,
    /// A future kernel mode token is conservatively incompatible with both
    /// SH and EX requesters rather than being misreported as free.
    Unknown,
}

/// Return `(pid, mode, dev:inode)` for one held FLOCK line.
fn parse_held_flock_with_mode(line: &str) -> Option<(u32, HeldFlockMode, &str)> {
    // Expected format:
    //   "1: FLOCK ADVISORY WRITE 12345 08:02:1234 0 EOF"
    // POSIX / OFDLCK lines have the same pid + dev_inode slots but a
    // different lock_type keyword in the second field.
    let mut fields = line.split_whitespace();
    let _id = fields.next()?;
    if fields.next()? != "FLOCK" {
        return None;
    }
    let _advisory = fields.next()?;
    let mode = match fields.next()? {
        "READ" => HeldFlockMode::Shared,
        "WRITE" => HeldFlockMode::Exclusive,
        _ => HeldFlockMode::Unknown,
    };
    let pid = fields.next()?.parse::<u32>().ok()?;
    let dev_inode = fields.next()?;
    Some((pid, mode, dev_inode))
}

/// Path-only adapter over [`read_holders_for_needle`]. Computes the
/// needle via [`super::mountinfo::needle_from_path`] and forwards.
/// This is the stable entry point for callers that only have a
/// lockfile path — cache EWOULDBLOCK diagnostics and `ktstr locks`.
/// `acquire_llc_plan` derives holder occupancy from the registry
/// aggregate instead of this `/proc/locks` scan.
///
/// Propagates stat failures on the path (context: "stat lockfile …
/// for holder lookup") and mountinfo failures ("resolve kernel
/// major:minor …").
pub(crate) fn read_holders(path: &Path) -> Result<Vec<HolderInfo>> {
    let needle = needle_from_path(path)?;
    read_holders_for_needle(&needle)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// [`parse_flock_pids_for_needle`] skips `POSIX` and `OFDLCK`
    /// lines and matches only `FLOCK` lines whose dev:inode triple
    /// byte-equals the needle.
    ///
    /// Feeds a synthetic `/proc/locks` fixture containing one POSIX,
    /// one OFDLCK, and one FLOCK line — all with the same dev:inode
    /// triple — and asserts only the FLOCK PID is returned. This
    /// pins the lock_type filter at the second-field check: without
    /// it, POSIX-byte-range locks would be misclassified as ktstr
    /// flock holders and the holder-enumeration diagnostic would
    /// name the wrong peers.
    #[test]
    fn parse_flock_pids_for_needle_skips_posix_and_ofdlck() {
        let needle = "08:02:1234";
        let contents = "\
1: POSIX  ADVISORY  WRITE 11111 08:02:1234 0 EOF
2: OFDLCK ADVISORY  READ  22222 08:02:1234 0 EOF
3: FLOCK  ADVISORY  WRITE 33333 08:02:1234 0 EOF
4: FLOCK  ADVISORY  READ  44444 08:02:5678 0 EOF
";
        let pids = parse_flock_pids_for_needle(contents, needle);
        assert_eq!(
            pids,
            vec![33333],
            "only the FLOCK line at the matching triple must contribute a PID; \
             POSIX/OFDLCK must be filtered",
        );
    }

    /// [`parse_flock_pids_for_needle`] deduplicates PIDs when a single
    /// process holds multiple FLOCK entries on the same lockfile
    /// (each `/proc/locks` line is prefixed with a numeric lock id +
    /// colon like `N: FLOCK ...`, not a `lock:` token; a process gets
    /// multiple FLOCK entries on one inode by open(2)ing the lockfile
    /// more than once — each open is a distinct struct file, and
    /// flock keys on the struct file, so dup/fork share one entry).
    /// One PID per holder, regardless of how many entries.
    #[test]
    fn parse_flock_pids_for_needle_deduplicates_pids() {
        let needle = "08:02:1234";
        let contents = "\
1: FLOCK  ADVISORY  WRITE 55555 08:02:1234 0 EOF
2: FLOCK  ADVISORY  READ  55555 08:02:1234 0 EOF
3: FLOCK  ADVISORY  WRITE 66666 08:02:1234 0 EOF
";
        let pids = parse_flock_pids_for_needle(contents, needle);
        assert_eq!(pids, vec![55555, 66666], "PIDs must dedupe");
    }

    /// [`parse_flock_pids_for_needle`] with empty contents returns an
    /// empty Vec — degenerate case.
    #[test]
    fn parse_flock_pids_for_needle_empty_contents_returns_empty() {
        let pids = parse_flock_pids_for_needle("", "08:02:1234");
        assert!(pids.is_empty());
    }

    /// [`parse_flock_pids_for_needle`] skips malformed lines (missing
    /// fields, non-numeric PIDs) without failing the whole parse.
    /// Pins the graceful-degradation contract for corrupt
    /// `/proc/locks` (unlikely but possible).
    #[test]
    fn parse_flock_pids_for_needle_skips_malformed_lines() {
        let needle = "08:02:1234";
        let contents = "\
1: FLOCK
2: FLOCK ADVISORY WRITE notanumber 08:02:1234 0 EOF
3: FLOCK ADVISORY WRITE 77777 08:02:1234 0 EOF
";
        let pids = parse_flock_pids_for_needle(contents, needle);
        assert_eq!(
            pids,
            vec![77777],
            "only the well-formed matching line contributes",
        );
    }

    /// [`read_holders_from_contents`] preserves the HolderInfo shape
    /// for a single matching needle. `holder_info_for_pid` reads our
    /// own cmdline so we can assert the PID half deterministically on
    /// any host.
    #[test]
    fn read_holders_from_contents_returns_holder_info_per_matching_pid() {
        let our_pid = std::process::id();
        let needle = "08:02:1234";
        let contents = format!(
            "1: FLOCK  ADVISORY  WRITE {our_pid} 08:02:1234 0 EOF\n\
             2: POSIX  ADVISORY  WRITE 11111 08:02:1234 0 EOF\n",
        );
        let holders = read_holders_from_contents(&contents, needle);
        assert_eq!(
            holders.len(),
            1,
            "only the FLOCK line at the matching triple produces a holder; \
             POSIX must be filtered: {holders:?}",
        );
        assert_eq!(holders[0].pid, our_pid);
        // cmdline comes from our own /proc/self/cmdline — must be non-empty
        // and distinct from the unavailable sentinel.
        assert_ne!(holders[0].cmdline, "<cmdline unavailable>");
    }

    /// [`read_holders_from_contents`] with contents empty (no
    /// `/proc/locks` lines at all) returns an empty Vec. Degenerate
    /// case — ensures the batched seam never errors on a clean pool.
    #[test]
    fn read_holders_from_contents_empty_returns_empty() {
        let holders = read_holders_from_contents("", "08:02:1234");
        assert!(holders.is_empty());
    }

    /// [`read_holders_from_contents`] is deterministic across the same
    /// contents — feeding the same contents+needle twice produces
    /// identical output (no hidden iteration-order dependency). Pins
    /// the batched-call-site invariant: callers that loop `N` needles
    /// over one `contents` must see the same result as `N` per-call
    /// reads of the same snapshot.
    #[test]
    fn read_holders_from_contents_deterministic_for_same_input() {
        let contents = format!(
            "1: FLOCK  ADVISORY  WRITE {pid} 08:02:1234 0 EOF\n",
            pid = std::process::id(),
        );
        let a = read_holders_from_contents(&contents, "08:02:1234");
        let b = read_holders_from_contents(&contents, "08:02:1234");
        assert_eq!(a.len(), b.len());
        assert_eq!(a.len(), 1);
        assert_eq!(a[0].pid, b[0].pid);
        assert_eq!(a[0].cmdline, b[0].cmdline);
    }
}
