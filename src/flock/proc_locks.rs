//! `/proc/locks` scanner — enumerate processes holding a given flock.
//!
//! Path → needle → /proc/locks scan → [`super::HolderInfo`] list.
//! The needle (`{major:02x}:{minor:02x}:{inode}`) is derived from the
//! lockfile path via [`super::mountinfo`]; this module owns the
//! /proc/locks side of the pipeline.
//!
//! Three API tiers:
//!
//!  - [`read_holders`] — one-shot, reads both `/proc/self/mountinfo`
//!    and `/proc/locks` itself. Use when looking up one lockfile.
//!  - `read_holders_with_mountinfo` — a test seam that accepts pre-read
//!    mountinfo for one lockfile.
//!  - [`read_flock_mode_summaries`],
//!    [`read_holder_pids_batch_with_mountinfo`], and
//!    [`read_holders_batch_with_mountinfo`] — accept N lockfile paths
//!    or already-derived needles and scan `/proc/locks` once. Admission
//!    uses the mode-summary form; placement callers use the PID-only
//!    form; diagnostics use the enriched form, which resolves each
//!    distinct PID's [`HolderInfo`] once for the whole batch.
//!
//! The pure parser seams [`parse_flock_pids_for_needle`] and
//! [`parse_flock_pids_for_needles`] are exposed so tests can feed
//! synthetic `/proc/locks` fixtures (POSIX / OFDLCK / FLOCK
//! interleavings, malformed lines) without touching the real
//! filesystem.

use anyhow::{Context, Result};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::path::Path;

use super::HolderInfo;
use super::holder::holder_info_for_pid;
use super::mountinfo::{needle_from_path, needle_from_path_with_mountinfo};

/// Compatibility-relevant holder state for one flock inode.
///
/// `any_holder` is true for either a held READ (`LOCK_SH`) or WRITE
/// (`LOCK_EX`) flock. `exclusive_holder` is true only for a held WRITE flock.
/// Thus an LLC shared requester is compatible exactly when
/// `exclusive_holder` is false, while an LLC exclusive requester (and every
/// CPU requester) is compatible exactly when `any_holder` is false.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct FlockModeSummary {
    pub(crate) any_holder: bool,
    pub(crate) exclusive_holder: bool,
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
/// peer-holder lookup in `src/cache/cache_dir.rs`).
/// `acquire_llc_plan`'s DISCOVER
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
/// `super::holder::CMDLINE_MAX_CHARS` with a `…` suffix on
/// overflow. A cmdline read failure is non-fatal — the entry
/// carries `"<cmdline unavailable>"` so the pid still surfaces.
pub(super) fn read_holders_for_needle(needle: &str) -> Result<Vec<HolderInfo>> {
    use std::fs;

    let contents = fs::read_to_string("/proc/locks")
        .with_context(|| "read /proc/locks for lockfile holder lookup")?;
    Ok(read_holders_from_contents(&contents, needle))
}

/// Content-based seam behind [`read_holders_for_needle`]. Takes
/// already-read `/proc/locks` `contents` plus one match `needle` and
/// returns the [`HolderInfo`] vector. Batched production callers use
/// [`read_holders_batch_with_mountinfo`] instead so both the text scan
/// and per-PID cmdline resolution are shared across all needles.
///
/// Thin shell over [`parse_flock_pids_for_needle`]: the latter
/// filters `/proc/locks` lines to the matching FLOCK PIDs; this
/// function adds the per-PID cmdline lookup via
/// [`super::holder::holder_info_for_pid`] that the
/// [`read_holders_for_needle`] caller expects. Extracted so batched
/// callers and the per-needle wrapper both key against the same seam
/// rather than duplicating the `.into_iter().map()` plumbing.
pub(super) fn read_holders_from_contents(contents: &str, needle: &str) -> Vec<HolderInfo> {
    let pids = parse_flock_pids_for_needle(contents, needle);
    pids.into_iter().map(holder_info_for_pid).collect()
}

/// Resolver-injected seam behind
/// [`read_holders_batch_with_mountinfo`]. Keeping PID resolution
/// separate from parsing makes the once-per-distinct-PID guarantee
/// directly testable without depending on live `/proc/{pid}` state.
fn read_holders_for_needles_from_contents_with(
    contents: &str,
    needles: &[String],
    resolve: impl FnMut(u32) -> HolderInfo,
) -> Vec<Vec<HolderInfo>> {
    let pid_sets = parse_flock_pids_for_needles(contents, needles);
    resolve_holder_pid_sets_with(pid_sets, resolve)
}

fn resolve_holder_pid_sets_with(
    pid_sets: Vec<Vec<u32>>,
    mut resolve: impl FnMut(u32) -> HolderInfo,
) -> Vec<Vec<HolderInfo>> {
    let mut holder_cache: HashMap<u32, HolderInfo> = HashMap::new();
    pid_sets
        .into_iter()
        .map(|pids| {
            pids.into_iter()
                .map(|pid| {
                    holder_cache
                        .entry(pid)
                        .or_insert_with(|| resolve(pid))
                        .clone()
                })
                .collect()
        })
        .collect()
}

fn resolve_batched_holder_info(pid: u32) -> HolderInfo {
    #[cfg(test)]
    BATCH_HOLDER_INFO_RESOLUTIONS.with(|count| count.set(count.get().saturating_add(1)));
    holder_info_for_pid(pid)
}

#[cfg(test)]
thread_local! {
    static BATCH_HOLDER_INFO_RESOLUTIONS: std::cell::Cell<usize> =
        const { std::cell::Cell::new(0) };
}

#[cfg(test)]
pub(crate) fn batch_holder_info_resolution_count_for_tests() -> usize {
    BATCH_HOLDER_INFO_RESOLUTIONS.with(std::cell::Cell::get)
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

/// Parse one `/proc/locks` image for N needles in a single line scan.
///
/// The output is positional and preserves first-seen PID order for
/// each needle. PIDs are deduplicated per needle, including when the
/// input contains multiple FLOCK entries for the same process and
/// inode. Duplicate input needles share the same parsed PID set.
pub(crate) fn parse_flock_pids_for_needles(contents: &str, needles: &[String]) -> Vec<Vec<u32>> {
    let mut unique_indices: HashMap<&str, usize> = HashMap::new();
    let mut output_indices = Vec::with_capacity(needles.len());
    let mut unique_pids: Vec<Vec<u32>> = Vec::new();
    let mut unique_seen: Vec<HashSet<u32>> = Vec::new();

    for needle in needles {
        let index = match unique_indices.get(needle.as_str()).copied() {
            Some(index) => index,
            None => {
                let index = unique_pids.len();
                unique_indices.insert(needle, index);
                unique_pids.push(Vec::new());
                unique_seen.push(HashSet::new());
                index
            }
        };
        output_indices.push(index);
    }

    for line in contents.lines() {
        let Some((pid, dev_inode)) = parse_held_flock(line) else {
            continue;
        };
        let Some(&index) = unique_indices.get(dev_inode) else {
            continue;
        };
        if unique_seen[index].insert(pid) {
            unique_pids[index].push(pid);
        }
    }

    output_indices
        .into_iter()
        .map(|index| unique_pids[index].clone())
        .collect()
}

/// Parse one `/proc/locks` image into compatibility summaries for the
/// requested inode needles.
///
/// The input and output are keyed by the exact
/// `{major:02x}:{minor:02x}:{inode}` strings emitted by `/proc/locks`.
/// Every requested needle is present in the output, including needles with no
/// held flock. Waiting (`->`) entries and non-FLOCK lock classes are ignored.
pub(crate) fn parse_flock_mode_summaries(
    contents: &str,
    needles: &BTreeSet<String>,
) -> BTreeMap<String, FlockModeSummary> {
    let mut summaries: BTreeMap<String, FlockModeSummary> = needles
        .iter()
        .cloned()
        .map(|needle| (needle, FlockModeSummary::default()))
        .collect();

    for line in contents.lines() {
        let Some((_pid, mode, dev_inode)) = parse_held_flock_with_mode(line) else {
            continue;
        };
        let Some(summary) = summaries.get_mut(dev_inode) else {
            continue;
        };
        summary.any_holder = true;
        summary.exclusive_holder |= mode != HeldFlockMode::Shared;
    }
    summaries
}

/// Read `/proc/locks` once and return compatibility summaries for every
/// requested inode needle.
pub(crate) fn read_flock_mode_summaries(
    needles: &BTreeSet<String>,
) -> Result<BTreeMap<String, FlockModeSummary>> {
    let contents = std::fs::read_to_string("/proc/locks")
        .with_context(|| "read /proc/locks for batched flock-mode observation")?;
    Ok(parse_flock_mode_summaries(&contents, needles))
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
///
/// `acquire_llc_plan`'s DISCOVER phase does NOT call this adapter —
/// it threads a pre-read `/proc/self/mountinfo` through
/// [`read_holders_batch_with_mountinfo`] so the whole per-LLC walk
/// reads both mountinfo and `/proc/locks` once per plan invocation. See
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
/// `/proc/self/mountinfo` contents. This saves the mountinfo read for
/// a single lookup but still reads `/proc/locks`; callers looking up
/// several paths must use [`read_holders_batch_with_mountinfo`] to
/// amortize both operations.
///
/// Semantically identical to [`read_holders`] — the same needle
/// format, the same /proc/locks scan, the same HolderInfo shape —
/// just with the mountinfo text supplied by the caller rather than
/// read inside this function.
#[cfg(test)]
pub(crate) fn read_holders_with_mountinfo(path: &Path, mountinfo: &str) -> Result<Vec<HolderInfo>> {
    let needle = needle_from_path_with_mountinfo(path, mountinfo)?;
    read_holders_for_needle(&needle)
}

/// Batch variant of the test-only `read_holders_with_mountinfo` seam.
///
/// Derives one `/proc/locks` needle per input path, reads
/// `/proc/locks` exactly once, scans that text exactly once for all
/// needles, and caches [`HolderInfo`] by PID across the entire batch.
/// The returned vector is positional: entry `i` corresponds to input
/// path `i`.
pub(crate) fn read_holders_batch_with_mountinfo<'a>(
    paths: impl IntoIterator<Item = &'a Path>,
    mountinfo: &str,
) -> Result<Vec<Vec<HolderInfo>>> {
    let pid_sets = read_holder_pids_batch_with_mountinfo(paths, mountinfo)?;
    Ok(resolve_holder_pid_sets_with(
        pid_sets,
        resolve_batched_holder_info,
    ))
}

/// Read holder PIDs for N lockfile paths without cmdline enrichment.
///
/// This is the hot-path placement API. It derives every needle from
/// the supplied mountinfo, reads and scans `/proc/locks` exactly once,
/// deduplicates PIDs per path, and returns positional PID rows. It
/// deliberately performs no `/proc/{pid}/cmdline` reads.
pub(crate) fn read_holder_pids_batch_with_mountinfo<'a>(
    paths: impl IntoIterator<Item = &'a Path>,
    mountinfo: &str,
) -> Result<Vec<Vec<u32>>> {
    let needles = paths
        .into_iter()
        .map(|path| needle_from_path_with_mountinfo(path, mountinfo))
        .collect::<Result<Vec<_>>>()?;
    let contents = std::fs::read_to_string("/proc/locks")
        .with_context(|| "read /proc/locks for batched lockfile holder lookup")?;
    Ok(parse_flock_pids_for_needles(&contents, &needles))
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

    #[test]
    fn parse_flock_pids_for_needles_matches_all_needles_in_one_pass() {
        let needles = vec![
            "08:02:100".to_owned(),
            "08:02:200".to_owned(),
            "08:02:100".to_owned(),
            "08:02:999".to_owned(),
        ];
        let contents = "\
1: FLOCK  ADVISORY  WRITE 101 08:02:100 0 EOF
2: FLOCK  ADVISORY  READ  202 08:02:200 0 EOF
3: FLOCK  ADVISORY  WRITE 101 08:02:100 0 EOF
4: FLOCK  ADVISORY  WRITE 303 08:02:100 0 EOF
5: POSIX  ADVISORY  WRITE 404 08:02:200 0 EOF
6: -> FLOCK ADVISORY WRITE 505 08:02:200 0 EOF
7: FLOCK  ADVISORY  WRITE 606 08:02:300 0 EOF
";

        assert_eq!(
            parse_flock_pids_for_needles(contents, &needles),
            vec![vec![101, 303], vec![202], vec![101, 303], vec![]],
            "one scan must route each held FLOCK to every requested needle, \
             deduplicate repeated PID entries, preserve first-seen order, and \
             reuse the parsed set for duplicate needles",
        );
    }

    #[test]
    fn parse_flock_mode_summaries_distinguishes_shared_and_exclusive_holders() {
        let needles = BTreeSet::from([
            "08:02:100".to_owned(),
            "08:02:200".to_owned(),
            "08:02:300".to_owned(),
            "08:02:400".to_owned(),
        ]);
        let contents = "\
1: FLOCK  ADVISORY  READ  101 08:02:100 0 EOF
2: FLOCK  ADVISORY  READ  102 08:02:100 0 EOF
3: FLOCK  ADVISORY  WRITE 201 08:02:200 0 EOF
4: POSIX  ADVISORY  WRITE 301 08:02:300 0 EOF
5: OFDLCK ADVISORY  WRITE 302 08:02:300 0 EOF
6: -> FLOCK ADVISORY WRITE 303 08:02:300 0 EOF
7: FLOCK  ADVISORY  UNKNOWN 304 08:02:300 0 EOF
8: FLOCK  ADVISORY  WRITE 999 08:02:999 0 EOF
malformed
";

        assert_eq!(
            parse_flock_mode_summaries(contents, &needles),
            BTreeMap::from([
                (
                    "08:02:100".to_owned(),
                    FlockModeSummary {
                        any_holder: true,
                        exclusive_holder: false,
                    },
                ),
                (
                    "08:02:200".to_owned(),
                    FlockModeSummary {
                        any_holder: true,
                        exclusive_holder: true,
                    },
                ),
                (
                    "08:02:300".to_owned(),
                    FlockModeSummary {
                        any_holder: true,
                        exclusive_holder: true,
                    },
                ),
                ("08:02:400".to_owned(), FlockModeSummary::default()),
            ]),
            "READ must block EX only, WRITE and unknown held modes must \
             conservatively block SH and EX, and queued, non-FLOCK, malformed, \
             and unrequested entries must not contribute",
        );
    }

    #[test]
    fn batched_holder_resolution_runs_once_per_distinct_pid() {
        let needles = vec![
            "08:02:100".to_owned(),
            "08:02:200".to_owned(),
            "08:02:100".to_owned(),
        ];
        let contents = "\
1: FLOCK ADVISORY WRITE 101 08:02:100 0 EOF
2: FLOCK ADVISORY READ  202 08:02:100 0 EOF
3: FLOCK ADVISORY WRITE 202 08:02:200 0 EOF
4: FLOCK ADVISORY WRITE 303 08:02:200 0 EOF
";
        let calls = std::cell::RefCell::new(std::collections::BTreeMap::<u32, usize>::new());

        let holders = read_holders_for_needles_from_contents_with(contents, &needles, |pid| {
            *calls.borrow_mut().entry(pid).or_default() += 1;
            HolderInfo {
                pid,
                cmdline: format!("pid-{pid}"),
            }
        });

        let holder_pids: Vec<Vec<u32>> = holders
            .iter()
            .map(|holders| holders.iter().map(|holder| holder.pid).collect())
            .collect();
        assert_eq!(
            holder_pids,
            vec![vec![101, 202], vec![202, 303], vec![101, 202]],
        );
        assert_eq!(
            *calls.borrow(),
            std::collections::BTreeMap::from([(101, 1), (202, 1), (303, 1)]),
            "a PID shared by several needles or duplicate input entries must \
             incur one /proc/<pid>/cmdline lookup for the whole batch",
        );
    }

    #[test]
    fn empty_batched_needle_set_does_not_resolve_holders() {
        let mut calls = 0usize;
        let holders = read_holders_for_needles_from_contents_with("malformed input", &[], |_| {
            calls += 1;
            unreachable!("an empty batch must never resolve a PID")
        });
        assert!(holders.is_empty());
        assert_eq!(calls, 0);
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
    /// set via both the cached-mountinfo API
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

    #[test]
    fn pid_only_batch_skips_cmdlines_and_enriched_batch_caches_by_pid() {
        let tmp = tempfile::TempDir::new().expect("tempdir");
        let first_path = tmp.path().join("batch-first.lock");
        let second_path = tmp.path().join("batch-second.lock");
        let first = try_flock(&first_path, FlockMode::Exclusive)
            .expect("open first lock")
            .expect("hold first lock");
        let second = try_flock(&second_path, FlockMode::Exclusive)
            .expect("open second lock")
            .expect("hold second lock");
        let paths = [first_path.as_path(), second_path.as_path()];
        let mountinfo = read_mountinfo().expect("read mountinfo");
        let our_pid = std::process::id();
        let before = batch_holder_info_resolution_count_for_tests();

        let pid_rows =
            read_holder_pids_batch_with_mountinfo(paths, &mountinfo).expect("read PID-only batch");
        assert!(
            pid_rows
                .iter()
                .all(|row| row.iter().any(|pid| *pid == our_pid)),
            "our PID must appear for both held lockfiles: {pid_rows:?}",
        );
        assert_eq!(
            batch_holder_info_resolution_count_for_tests(),
            before,
            "the placement batch must not resolve any cmdlines",
        );

        let holder_rows = read_holders_batch_with_mountinfo(paths, &mountinfo)
            .expect("read enriched holder batch");
        assert!(
            holder_rows
                .iter()
                .all(|row| row.iter().any(|holder| holder.pid == our_pid)),
            "our HolderInfo must appear for both held lockfiles: {holder_rows:?}",
        );
        assert_eq!(
            batch_holder_info_resolution_count_for_tests(),
            before + 1,
            "one PID holding several requested lockfiles must be enriched once",
        );

        drop(second);
        drop(first);
    }
}
