//! `/proc/locks` scanner — enumerate processes holding a given flock.
//!
//! Path → needle → /proc/locks scan → [`super::HolderInfo`] list.
//! The needle (`{major:02x}:{minor:02x}:{inode}`) is derived from the
//! lockfile path via [`super::mountinfo`]; this module owns the
//! /proc/locks side of the pipeline.
//!
//! Two API tiers for batched callers:
//!
//!  - [`read_holders`] — one-shot, reads both `/proc/self/mountinfo`
//!    and `/proc/locks` itself. Use when looking up one lockfile.
//!  - [`read_holders_with_mountinfo`] — accepts pre-read mountinfo.
//!    Use when scanning N lockfiles in one pass (e.g.
//!    `acquire_llc_plan`'s DISCOVER phase) so mountinfo is read once
//!    per batch.
//!
//! Both ultimately call [`read_holders_for_needle`], which scans
//! `/proc/locks` exactly once and returns one [`HolderInfo`] per
//! matching PID. The pure parser seam
//! [`parse_flock_pids_for_needle`] is exposed so tests can feed
//! synthetic `/proc/locks` fixtures (POSIX / OFDLCK / FLOCK
//! interleavings, malformed lines) without touching the real
//! filesystem.

use anyhow::Result;
use std::path::Path;

use super::HolderInfo;
use super::holder::holder_info_for_pid;
use super::mountinfo::{needle_from_path, needle_from_path_with_mountinfo};

/// Parse `/proc/locks` and return [`HolderInfo`] entries for every
/// process holding an advisory `FLOCK` matching `needle`.
///
/// `needle` must be the `{major:02x}:{minor:02x}:{inode}` triple in
/// /proc/locks' own formatting, produced by
/// [`super::mountinfo::needle_from_path`]:
/// `(major, minor)` via `/proc/self/mountinfo` and `inode` via
/// `stat().st_ino`. Used by path-only callers ([`read_holders`],
/// the `ktstr locks` observational scan, and the EWOULDBLOCK-branch
/// peer-holder lookup in `cache.rs`). `acquire_llc_plan`'s DISCOVER
/// phase uses [`super::mountinfo::needle_from_path_with_mountinfo`]
/// instead so the mountinfo read amortizes across every LLC in one
/// invocation.
///
/// Best-effort: returns `Ok(vec![])` when no /proc/locks entry
/// matches the needle, and propagates only the hard `/proc/locks`
/// read failure.
///
/// For each matching PID, reads `/proc/{pid}/cmdline`, decodes as
/// lossy UTF-8, replaces `\0` with ` `, and truncates to
/// [`super::holder::CMDLINE_MAX_CHARS`] with a `…` suffix on
/// overflow. A cmdline read failure is non-fatal — the entry
/// carries `"<cmdline unavailable>"` so the pid still surfaces.
pub(crate) fn read_holders_for_needle(needle: &str) -> Result<Vec<HolderInfo>> {
    use anyhow::Context;
    use std::fs;

    let contents = fs::read_to_string("/proc/locks")
        .with_context(|| "read /proc/locks for lockfile holder lookup")?;
    Ok(read_holders_from_contents(&contents, needle))
}

/// Content-based seam behind [`read_holders_for_needle`]. Takes
/// already-read `/proc/locks` `contents` plus the match `needle` and
/// returns the [`HolderInfo`] vector. Skips the `/proc/locks` read so
/// a caller with N needles (e.g. `acquire_llc_plan`'s DISCOVER phase,
/// which visits every host LLC's lockfile) can read `/proc/locks`
/// ONCE and call this N times instead of re-reading the same file
/// per iteration — the per-LLC scan was O(N) file reads against a
/// kernel-synthesized text source that is already consistent across
/// the whole batch.
///
/// Thin shell over [`parse_flock_pids_for_needle`]: the latter
/// filters `/proc/locks` lines to the matching FLOCK PIDs; this
/// function adds the per-PID cmdline lookup via
/// [`super::holder::holder_info_for_pid`] that the
/// [`read_holders_for_needle`] caller expects. Extracted so batched
/// callers and the per-needle wrapper both key against the same seam
/// rather than duplicating the `.into_iter().map()` plumbing.
pub(crate) fn read_holders_from_contents(contents: &str, needle: &str) -> Vec<HolderInfo> {
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
        // Expected format (after the id colon):
        //   "1: FLOCK ADVISORY WRITE 12345 08:02:1234 0 EOF"
        // POSIX / OFDLCK lines have the same pid + dev_inode slot
        // shape but a different lock_type keyword in the second
        // field — filter them out here.
        let mut fields = line.split_whitespace();
        // Skip the "N:" id.
        let _id = fields.next();
        let lock_type = fields.next();
        if lock_type != Some("FLOCK") {
            continue;
        }
        // advisory/mandatory
        let _adv = fields.next();
        // READ/WRITE
        let _mode = fields.next();
        let pid = match fields.next().and_then(|s| s.parse::<u32>().ok()) {
            Some(p) => p,
            None => continue,
        };
        let dev_inode = match fields.next() {
            Some(s) => s,
            None => continue,
        };
        if dev_inode == needle && !pids.contains(&pid) {
            pids.push(pid);
        }
    }
    pids
}

/// Path-only adapter over [`read_holders_for_needle`]. Computes the
/// needle via [`super::mountinfo::needle_from_path`] and forwards.
/// This is the stable entry point for callers that only have a
/// lockfile path — cache EWOULDBLOCK diagnostics and `ktstr locks`.
///
/// `acquire_llc_plan`'s DISCOVER phase does NOT call this adapter —
/// it threads a pre-read `/proc/self/mountinfo` through
/// [`read_holders_with_mountinfo`] so the whole per-LLC walk reads
/// mountinfo exactly once per plan invocation. See
/// [`super::mountinfo::needle_from_path_with_mountinfo`] for the seam.
///
/// Propagates stat failures on the path (context: "stat lockfile …
/// for holder lookup") and mountinfo failures ("resolve kernel
/// major:minor …").
pub(crate) fn read_holders(path: &Path) -> Result<Vec<HolderInfo>> {
    let needle = needle_from_path(path)?;
    read_holders_for_needle(&needle)
}

/// Variant of [`read_holders`] that accepts pre-read
/// `/proc/self/mountinfo` contents. Used by callers that walk a
/// batch of lockfiles in one invocation (e.g.
/// `acquire_llc_plan`'s DISCOVER phase, which visits every LLC's
/// lockfile) and want to amortize the mountinfo read across the
/// whole batch instead of re-reading per lockfile.
///
/// Semantically identical to [`read_holders`] — the same needle
/// format, the same /proc/locks scan, the same HolderInfo shape —
/// just with the mountinfo text supplied by the caller rather than
/// read inside this function.
pub(crate) fn read_holders_with_mountinfo(path: &Path, mountinfo: &str) -> Result<Vec<HolderInfo>> {
    let needle = needle_from_path_with_mountinfo(path, mountinfo)?;
    read_holders_for_needle(&needle)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flock::FlockMode;
    use crate::flock::mountinfo::read_mountinfo;
    use crate::flock::primitives::try_flock;

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
    /// (e.g. the kernel emits one `lock:` line per OFD, and a
    /// process that dup'd its fd has multiple OFDs on the same
    /// inode). One PID per holder, regardless of how many entries.
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
    /// for matching FLOCK lines. This is the batched-read seam used
    /// by callers that have N needles and want to read `/proc/locks`
    /// exactly once across the batch — the function must return one
    /// [`HolderInfo`] per matching PID in the same order the parser
    /// produces them. `holder_info_for_pid` reads our own cmdline so
    /// we can assert the PID half deterministically on any host.
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

    /// [`read_holders_for_needle`] with an impossible needle returns
    /// an empty Vec. Exercises the /proc/locks read path on any
    /// Linux host without requiring specific lockfile state. The
    /// needle format is `{major:02x}:{minor:02x}:{inode}`; pick
    /// values guaranteed-not-to-exist (major=ff, minor=ff, inode
    /// larger than any real inode at test time).
    #[test]
    fn read_holders_for_needle_no_match_returns_empty() {
        // u64 max inode, max 8-bit major:minor pair. No real
        // /proc/locks entry will match this.
        let needle = "ff:ff:18446744073709551615";
        let holders = read_holders_for_needle(needle)
            .expect("/proc/locks read must succeed on any Linux host");
        assert!(
            holders.is_empty(),
            "impossible needle must not match any holder: {holders:?}"
        );
    }

    /// Holder-list equivalence under a live flock.
    ///
    /// Beyond "both needles are equal strings," the full
    /// `/proc/locks` scan must surface the same [`HolderInfo`]
    /// set via both the cached batch API
    /// ([`read_holders_with_mountinfo`]) and the one-shot API
    /// ([`read_holders`]) for a lockfile we actually hold. A
    /// regression where the cached path e.g. canonicalizes
    /// differently (altering the mount-point prefix match) would
    /// surface here: the needles would still be valid triples but
    /// point at different (major, minor) for the same path, and
    /// exactly one of the two scans would find our pid.
    #[test]
    fn read_holders_cached_mountinfo_equals_uncached() {
        use tempfile::TempDir;

        let tmp = TempDir::new().expect("tempdir");
        let path = tmp.path().join("cache-holder-equivalence.lock");

        let fd = try_flock(&path, FlockMode::Exclusive)
            .expect("try_flock must succeed on fresh tempfile")
            .expect("EX must acquire on clean pool");

        // Uncached: inline mountinfo read per call.
        let uncached = read_holders(&path).expect("uncached holders");

        // Cached: read mountinfo once, pass through.
        let mountinfo = read_mountinfo().expect("read mountinfo");
        let cached = read_holders_with_mountinfo(&path, &mountinfo).expect("cached holders");

        // /proc/locks race-safety: holder sets can drift between two
        // scans on a loaded host (peer exits, a separate test flock
        // created/released). Pin the invariant we actually care
        // about: OUR pid appears in BOTH sets.
        let our_pid = std::process::id();
        assert!(
            uncached.iter().any(|h| h.pid == our_pid),
            "our pid {our_pid} must appear in uncached holders {uncached:?}",
        );
        assert!(
            cached.iter().any(|h| h.pid == our_pid),
            "our pid {our_pid} must appear in cached holders {cached:?}",
        );

        drop(fd);
    }
}
