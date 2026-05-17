//! `/proc/self/mountinfo` parser and `/proc/locks`-match needle derivation.
//!
//! `/proc/locks` keys each held flock by `i_sb->s_dev` (the
//! superblock device id) plus inode. For most filesystems
//! `i_sb->s_dev` matches what `stat()` reports via `st_dev`, but on
//! btrfs, overlayfs, and bind mounts the kernel installs a custom
//! `getattr` that returns an anonymous device id distinct from
//! `s_dev`. A stat-derived needle would silently never match in
//! those cases — see the module-level docs in [`super`].
//!
//! This module reads the kernel's own value from
//! `/proc/self/mountinfo` (longest-prefix match on `mount_point`),
//! then composes the
//! `{major:02x}:{minor:02x}:{inode}` needle in /proc/locks' own
//! formatting so byte-equality on the wire suffices downstream.
//!
//! Two API tiers:
//!
//!  - One-shot: [`needle_from_path`] reads mountinfo inline. Use
//!    when computing a single needle.
//!  - Batched: [`needle_from_path_with_mountinfo`] accepts pre-read
//!    contents. Use when computing N needles in one pass (e.g.
//!    `acquire_llc_plan`'s DISCOVER phase visits every host LLC's
//!    lockfile) so mountinfo is read once per batch instead of N
//!    times.
//!
//! The pure-text parser seam
//! [`mount_major_minor_for_path_from_contents`] is exposed as
//! `pub(crate)` so tests can feed synthetic mountinfo fixtures
//! (bind mounts stacked over tmpfs, btrfs subvolumes, mount points
//! with whitespace) without reproducing those states on the host
//! filesystem.

use anyhow::Result;
use std::path::{Path, PathBuf};

/// Read `/proc/self/mountinfo` once. Callers that need to derive
/// needles for multiple lockfiles in a single pass (e.g.
/// `acquire_llc_plan`'s DISCOVER phase, which visits every host
/// LLC's lockfile on every DISCOVER attempt) read mountinfo via
/// this helper once per batch and hand the resulting `String` to
/// [`super::proc_locks::read_holders_with_mountinfo`] /
/// [`needle_from_path_with_mountinfo`].
///
/// One-shot callers ([`needle_from_path`],
/// [`super::proc_locks::read_holders`]) also route through this
/// helper so every /proc/self/mountinfo read in the crate shares
/// the same error context and any future retry / instrumentation
/// has a single place to land.
pub(crate) fn read_mountinfo() -> Result<String> {
    use anyhow::Context;
    std::fs::read_to_string("/proc/self/mountinfo").context("read /proc/self/mountinfo")
}

/// Build a /proc/locks match needle for `path` using
/// `/proc/self/mountinfo` (for `i_sb->s_dev`) and `stat().st_ino`
/// (for the inode). Format: `{major:02x}:{minor:02x}:{inode}` —
/// kernel's own /proc/locks formatting, so a byte-equality check
/// suffices downstream.
///
/// Refuses to derive a needle from `stat().st_dev`: on btrfs,
/// overlayfs, and bind-mounts that dev diverges from the
/// superblock dev that /proc/locks emits, and a stat-derived
/// needle would silently never match. See module-level rationale.
pub(crate) fn needle_from_path(path: &Path) -> Result<String> {
    let mountinfo = read_mountinfo()?;
    needle_from_path_with_mountinfo(path, &mountinfo)
}

/// Variant of [`needle_from_path`] that accepts pre-read
/// `/proc/self/mountinfo` contents. Both functions produce
/// byte-identical needles for the same `path` — this one just
/// skips the mountinfo read.
///
/// Used by [`super::proc_locks::read_holders_with_mountinfo`] so a
/// caller walking N lockfiles pays for exactly one mountinfo read
/// instead of N.
pub(crate) fn needle_from_path_with_mountinfo(path: &Path, mountinfo: &str) -> Result<String> {
    use anyhow::Context;
    use std::fs;
    use std::os::unix::fs::MetadataExt;

    let meta = fs::metadata(path)
        .with_context(|| format!("stat lockfile {} for holder lookup", path.display()))?;
    let inode = meta.ino();
    let (major, minor) =
        mount_major_minor_for_path_with_contents(path, mountinfo).with_context(|| {
            format!(
                "resolve kernel major:minor for {} via /proc/self/mountinfo",
                path.display()
            )
        })?;
    Ok(format!("{major:02x}:{minor:02x}:{inode}"))
}

/// Resolve `path` to its containing mount point's kernel major:minor
/// from pre-read `/proc/self/mountinfo` text.
///
/// Format per `Documentation/filesystems/proc.rst` §3.5:
/// ```text
/// {mount_id} {parent_id} {major:minor} {root} {mount_point} {options} ...
/// ```
/// We canonicalize `path` (fall back to lexical absolute form when
/// canonicalize fails — the lockfile may not yet exist), enumerate
/// every mountinfo line, find the longest-prefix match on the
/// `mount_point` field, and return that entry's `{major:minor}`
/// decoded as `(u32, u32)`.
///
/// Longest-prefix is load-bearing: a bind mount of `/tmp/ktstr-cache`
/// onto `/tmp/ktstr-cache` stacked over tmpfs `/tmp` must match the
/// bind's mountinfo entry (more specific), not tmpfs's (less
/// specific). The lockfile lives on the bind-backing filesystem;
/// /proc/locks emits the bind's s_dev.
///
/// Production callers obtain `contents` from [`read_mountinfo`] —
/// either once per batch (acquire_llc_plan DISCOVER) or per-call
/// (one-shot needle derivation via [`needle_from_path`]).
fn mount_major_minor_for_path_with_contents(path: &Path, contents: &str) -> Result<(u32, u32)> {
    use std::fs;

    // Canonicalize the query path. When the lockfile doesn't yet
    // exist (first-call create path), canonicalize fails; fall back
    // to the caller's path verbatim, which is already absolute for
    // every ktstr call site (`{lock_dir}/ktstr-llc-{N}.lock`,
    // `{cache_root}/.locks/{key}.lock`, …).
    let canon: PathBuf = fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());

    mount_major_minor_for_path_from_contents(contents, &canon)
}

/// Pure mountinfo-parser seam behind
/// [`mount_major_minor_for_path_with_contents`].
/// Takes the already-read mountinfo `contents` and the
/// already-canonicalized `path`, walks the lines, longest-prefix-
/// matches `mount_point` against `path`, and returns the matching
/// entry's `(major, minor)` decoded from the `{major:minor}` field.
///
/// Exposed as `pub(crate)` so tests can feed synthetic mountinfo
/// text (bind mounts stacked over tmpfs, btrfs subvolume mounts,
/// mount points with whitespace) without having to reproduce those
/// states in the host filesystem. The production wrapper above
/// canonicalizes before calling this seam; everything below is pure
/// text processing.
pub(crate) fn mount_major_minor_for_path_from_contents(
    contents: &str,
    path: &Path,
) -> Result<(u32, u32)> {
    let mut best: Option<(usize, u32, u32)> = None;
    for line in contents.lines() {
        // Split on whitespace once, walk fields by index. Field 2 is
        // `major:minor`, field 4 is the mount point. A single pass
        // collects both without re-splitting. Whitespace inside a
        // mount_point is safe to whitespace-split on: the kernel
        // octal-escapes space/tab/newline in the mount_point field
        // (fs/proc_namespace.c: seq_path_root(..., " \t\n\\") →
        // fs/seq_file.c: mangle_path()), so a literal space never
        // appears inline — it arrives as the 4-byte sequence `\040`,
        // which splitting preserves. [`unescape_mountinfo_field`]
        // restores the original bytes below before the prefix match.
        let mut fields = line.split_whitespace();
        let _mount_id = fields.next();
        let _parent_id = fields.next();
        let major_minor = match fields.next() {
            Some(s) => s,
            None => continue,
        };
        let _root = fields.next();
        let mount_point_raw = match fields.next() {
            Some(s) => s,
            None => continue,
        };
        // Optional fields, then `-`, then fs_type — we don't consume
        // them; `fields` is discarded after this line.

        // Kernel escapes space (`\040`), tab (`\011`), newline
        // (`\012`), and backslash (`\134`) in the mount_point field
        // via fs/seq_file.c: mangle_path(). `path` arrives from the
        // caller with literal bytes (a tempdir named "my dir" has a
        // real space, not `\040`), so we must octal-unescape the
        // mountinfo field before the prefix check or a path with any
        // of those bytes would silently miss its covering mount —
        // producing "no mountinfo entry covers {path}" on otherwise
        // valid hosts that happened to place `/tmp` or a cache root
        // under a mount point containing whitespace.
        let mount_point = unescape_mountinfo_field(mount_point_raw);

        // Prefix match: `mount_point` must be a prefix of `path`
        // on a path-component boundary. A pure string prefix check
        // would accept `/tmp/foo` against `/tmp/foobar`, so anchor
        // the comparison on components.
        if !path_starts_with(path, Path::new(mount_point.as_ref())) {
            continue;
        }
        let (major, minor) = match parse_major_minor(major_minor) {
            Some(mm) => mm,
            None => continue,
        };
        let len = mount_point.len();
        if best.is_none_or(|(best_len, _, _)| len > best_len) {
            best = Some((len, major, minor));
        }
    }
    match best {
        Some((_, major, minor)) => Ok((major, minor)),
        None => anyhow::bail!(
            "no mountinfo entry covers {} — is /proc mounted?",
            path.display()
        ),
    }
}

/// Decode the kernel's `\NNN` octal escape sequences in a mountinfo
/// text field back to the original bytes. The kernel's mountinfo
/// writer (`fs/proc_namespace.c:show_mountinfo`) passes the escape
/// set `" \t\n\\"` to `fs/seq_file.c:seq_path_root`, which then calls
/// `mangle_path()` with 3-digit-octal for each matched character:
///
/// - space (0x20) → `\040`
/// - tab   (0x09) → `\011`
/// - LF    (0x0A) → `\012`
/// - `\`   (0x5C) → `\134`
///
/// Bytes outside that set are copied verbatim. This decoder handles
/// the general form `\NNN` (3 octal digits), not just the 4
/// characters above — the kernel's escape logic is parameterized,
/// and a future kernel could extend the escape set without changing
/// the wire format; a generic decoder matches whatever the kernel
/// emits. Non-`\NNN` backslashes (none in practice, but defensive)
/// are kept as literal bytes, so malformed input cannot produce a
/// shorter string that silently matches a different mount point
/// than the caller intended.
///
/// Returns `Cow::Borrowed(raw)` when no `\` appears — avoids an
/// allocation for the overwhelmingly common "no escape needed" case
/// (`/tmp`, `/home`, `/var`, …).
fn unescape_mountinfo_field(raw: &str) -> std::borrow::Cow<'_, str> {
    if !raw.contains('\\') {
        return std::borrow::Cow::Borrowed(raw);
    }
    // `\` was found — switch to the owned path. Byte-level walk so
    // `\NNN` with non-ASCII octal-decoded bytes (e.g. `\200`)
    // produces the exact kernel-emitted byte sequence; pushing via
    // `push_str` would require UTF-8 validation the kernel does not
    // itself apply to path components.
    let bytes = raw.as_bytes();
    let mut out: Vec<u8> = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'\\'
            && i + 3 < bytes.len()
            && is_octal_digit(bytes[i + 1])
            && is_octal_digit(bytes[i + 2])
            && is_octal_digit(bytes[i + 3])
        {
            // Compose three octal digits in u16 to avoid u8 shift
            // overflow on `\400` and above. The kernel's mangle_path
            // only emits values ≤ `\377` (the escape set is " \t\n\\"
            // and the ASCII byte for each is < 0x80), but malformed
            // input must not panic — reject out-of-range values and
            // copy the original `\NNN` bytes verbatim instead.
            let val = ((bytes[i + 1] - b'0') as u16) << 6
                | ((bytes[i + 2] - b'0') as u16) << 3
                | (bytes[i + 3] - b'0') as u16;
            if val <= 0xff {
                out.push(val as u8);
                i += 4;
            } else {
                out.push(bytes[i]);
                i += 1;
            }
        } else {
            out.push(bytes[i]);
            i += 1;
        }
    }
    // The kernel's mangle_path never produces invalid UTF-8 for the
    // escaped set (`" \t\n\\"` are all ASCII). Lossy decode matches
    // the same contract [`super::holder::holder_info_for_pid`] applies
    // to argv bytes: a malformed mountinfo line produces U+FFFD
    // substitutions rather than aborting the whole parse.
    std::borrow::Cow::Owned(String::from_utf8_lossy(&out).into_owned())
}

/// True when `b` is one of `b'0'..=b'7'` — the valid digits of a
/// `\NNN` octal escape. Inlined so the hot mountinfo parse loop
/// stays branch-light; the intent is obvious enough that a named
/// helper documents itself.
#[inline]
fn is_octal_digit(b: u8) -> bool {
    (b'0'..=b'7').contains(&b)
}

/// True when `path` begins with `prefix` on a path-component
/// boundary. Distinct from byte-level `String::starts_with`:
/// `/tmp/foo` does NOT start with `/tmp/foobar`. `Path::starts_with`
/// already handles this correctly; wrap it for readability at the
/// mountinfo call site.
fn path_starts_with(path: &Path, prefix: &Path) -> bool {
    path.starts_with(prefix)
}

/// Parse a mountinfo `major:minor` field (e.g. `"259:3"`) into a
/// `(u32, u32)` tuple. Decimal — the kernel emits these in base 10,
/// unlike /proc/locks which uses hex for the same pair.
fn parse_major_minor(s: &str) -> Option<(u32, u32)> {
    let (maj, min) = s.split_once(':')?;
    Some((maj.parse().ok()?, min.parse().ok()?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flock::primitives::materialize;

    // ---------------------------------------------------------------
    // mountinfo parsing — mount_major_minor_for_path_from_contents
    // ---------------------------------------------------------------

    /// Single-mount synthetic mountinfo: path on tmpfs-mounted
    /// `/tmp` resolves to the tmpfs entry's (major, minor).
    /// Values from `man 5 proc` mountinfo format example.
    #[test]
    fn mountinfo_single_mount_hits_right_major_minor() {
        let mountinfo = "\
22 28 0:21 / /tmp rw,nosuid,nodev shared:5 - tmpfs tmpfs rw,size=8g
";
        let (major, minor) =
            mount_major_minor_for_path_from_contents(mountinfo, Path::new("/tmp/ktstr-llc-0.lock"))
                .expect("tmp mount covers the lockfile path");
        assert_eq!((major, minor), (0, 21));
    }

    /// Longest-prefix wins: a bind mount at `/tmp/ktstr-cache`
    /// stacked over tmpfs `/tmp` must resolve to the BIND's
    /// major:minor, not tmpfs's. `/proc/locks` emits the bind's
    /// `s_dev`, so the lookup must match the more-specific mount.
    #[test]
    fn mountinfo_longest_prefix_wins_for_bind_over_tmpfs() {
        let mountinfo = "\
22 28 0:21 / /tmp rw,nosuid,nodev shared:5 - tmpfs tmpfs rw,size=8g
35 22 0:99 / /tmp/ktstr-cache rw,nosuid - tmpfs tmpfs rw,size=1g
";
        let (major, minor) = mount_major_minor_for_path_from_contents(
            mountinfo,
            Path::new("/tmp/ktstr-cache/entry.lock"),
        )
        .expect("bind mount wins longest-prefix match");
        assert_eq!((major, minor), (0, 99), "bind's major:minor expected");
    }

    /// A path not covered by any mount errors out with an
    /// actionable message. The production wrapper always pre-
    /// populates `/proc/self/mountinfo`, which always contains at
    /// minimum the root `/` entry, so this failure is effectively
    /// "path has no leading slash" territory.
    #[test]
    fn mountinfo_uncovered_path_errors() {
        let mountinfo = "\
22 28 0:21 / /tmp rw - tmpfs tmpfs rw
";
        let err = mount_major_minor_for_path_from_contents(
            mountinfo,
            Path::new("/var/log/unrelated.lock"),
        )
        .expect_err("no mountinfo entry covers /var/log/...");
        let msg = format!("{err:#}");
        assert!(msg.contains("no mountinfo entry covers"), "msg={msg}");
    }

    /// Component-boundary prefix check: a path at `/tmp/foo` must
    /// NOT match a mount at `/tmp/foobar`. Byte-level string
    /// prefix would incorrectly accept this; `Path::starts_with`
    /// anchors on components and rejects it. This is the correctness
    /// test for `path_starts_with`.
    ///
    /// Covers both directions on the same mountinfo fixture:
    ///   - `/tmp/foo/entry.lock`      → (0, 21) — the /tmp mount
    ///     (NOT /tmp/foobar, despite string-prefix overlap).
    ///   - `/tmp/foobar/entry.lock`   → (0, 99) — the /tmp/foobar
    ///     mount wins longest-prefix because it IS a component-
    ///     boundary prefix of the query path.
    #[test]
    fn mountinfo_respects_component_boundary() {
        let mountinfo = "\
22 28 0:21 / /tmp rw - tmpfs tmpfs rw
35 22 0:99 / /tmp/foobar rw - tmpfs tmpfs rw
";
        // /tmp/foo/ is NOT under /tmp/foobar/ on a component
        // boundary — only /tmp matches.
        let (major, minor) =
            mount_major_minor_for_path_from_contents(mountinfo, Path::new("/tmp/foo/entry.lock"))
                .expect("path under /tmp (not /tmp/foobar) resolves to the tmp mount");
        assert_eq!(
            (major, minor),
            (0, 21),
            "/tmp/foo must NOT match the /tmp/foobar mount",
        );

        // Reverse: /tmp/foobar/ IS under /tmp/foobar on a component
        // boundary — the more-specific mount wins longest-prefix.
        let (major, minor) = mount_major_minor_for_path_from_contents(
            mountinfo,
            Path::new("/tmp/foobar/entry.lock"),
        )
        .expect("path under /tmp/foobar resolves to the /tmp/foobar mount");
        assert_eq!(
            (major, minor),
            (0, 99),
            "/tmp/foobar/ must match the /tmp/foobar mount, not the /tmp one",
        );
    }

    /// Malformed major:minor on one line doesn't prevent a later
    /// valid line from matching. Graceful degradation — a corrupt
    /// mountinfo line (unlikely but possible on exotic hosts)
    /// must not kill the whole lookup.
    #[test]
    fn mountinfo_skips_malformed_major_minor() {
        let mountinfo = "\
22 28 BAD:NUMBER / /tmp rw - tmpfs tmpfs rw
35 28 0:42 / /tmp rw - tmpfs tmpfs rw
";
        let (major, minor) =
            mount_major_minor_for_path_from_contents(mountinfo, Path::new("/tmp/entry.lock"))
                .expect("second (valid) line still matches after malformed first");
        assert_eq!((major, minor), (0, 42));
    }

    /// Short line (missing fields) is skipped without error.
    /// Real mountinfo always has ≥5 fields before the `-`, but
    /// defensive coding protects against proc-fs corruption.
    #[test]
    fn mountinfo_skips_truncated_lines() {
        let mountinfo = "\
22 28 0:21
35 28 0:42 / /tmp rw - tmpfs tmpfs rw
";
        let (major, minor) =
            mount_major_minor_for_path_from_contents(mountinfo, Path::new("/tmp/entry.lock"))
                .expect("truncated line skipped; second line matches");
        assert_eq!((major, minor), (0, 42));
    }

    /// Mount point containing a literal space. The kernel emits it
    /// as `\040` in the mountinfo `mount_point` field via
    /// `fs/seq_file.c:mangle_path`; without unescaping, a path like
    /// `/mnt/my dir/cache.lock` would byte-split into `/mnt/my` and
    /// `dir/cache.lock`, the parser would see the mount_point as
    /// `/mnt/my\040dir`, and `path_starts_with` would compare
    /// against the escaped form — a silent miss on any host whose
    /// cache root or `/tmp` happens to sit under a whitespace mount.
    /// Pins the fix: the parser unescapes before comparing.
    #[test]
    fn mountinfo_unescapes_space_in_mount_point() {
        let mountinfo = "\
22 28 0:77 / /mnt/my\\040dir rw,nosuid - tmpfs tmpfs rw
";
        let (major, minor) = mount_major_minor_for_path_from_contents(
            mountinfo,
            Path::new("/mnt/my dir/cache.lock"),
        )
        .expect(
            "mount point with `\\040`-escaped space must unescape to real \
             space and match the query path's literal space",
        );
        assert_eq!((major, minor), (0, 77));
    }

    /// Mount point containing a literal tab (`\011`) — same fix
    /// surface as `\040`, different escape byte. Tabs in mount
    /// points are vanishingly rare but the kernel escapes them
    /// alongside spaces; testing the class broadly pins the
    /// general-octal contract rather than just the space-specific
    /// one.
    #[test]
    fn mountinfo_unescapes_tab_in_mount_point() {
        let mountinfo = "\
22 28 0:78 / /mnt/tab\\011dir rw,nosuid - tmpfs tmpfs rw
";
        let (major, minor) = mount_major_minor_for_path_from_contents(
            mountinfo,
            Path::new("/mnt/tab\tdir/cache.lock"),
        )
        .expect("mount point with `\\011` must unescape to real tab");
        assert_eq!((major, minor), (0, 78));
    }

    /// Mount point containing a literal backslash — escaped as
    /// `\134` per the kernel's `mangle_path(..., " \\t\\n\\\\")`
    /// escape set. A caller's path bytes include the literal
    /// backslash; the parser must match after unescaping.
    #[test]
    fn mountinfo_unescapes_backslash_in_mount_point() {
        // Rust source: `\\134` → the four bytes `\`, `1`, `3`, `4`
        // (the on-wire octal escape). The query path holds a
        // literal backslash (Rust source: `\\` → `\`).
        let mountinfo = "\
22 28 0:79 / /mnt/bs\\134dir rw,nosuid - tmpfs tmpfs rw
";
        let (major, minor) = mount_major_minor_for_path_from_contents(
            mountinfo,
            Path::new("/mnt/bs\\dir/cache.lock"),
        )
        .expect("mount point with `\\134` must unescape to real backslash");
        assert_eq!((major, minor), (0, 79));
    }

    /// `unescape_mountinfo_field` returns a borrowed `Cow` when
    /// the input contains no `\` — the common case on every Linux
    /// host. Pins the zero-allocation contract so a future refactor
    /// that always allocates regresses through this test.
    #[test]
    fn unescape_mountinfo_field_borrows_when_no_escapes() {
        let raw = "/tmp";
        let decoded = unescape_mountinfo_field(raw);
        match decoded {
            std::borrow::Cow::Borrowed(b) => assert_eq!(b, raw),
            std::borrow::Cow::Owned(_) => {
                panic!("unescape must return Cow::Borrowed when input has no `\\`")
            }
        }
    }

    /// `unescape_mountinfo_field` decodes multi-escape inputs —
    /// `/a b\tc` encodes as `/a\040b\011c`, and all three bytes
    /// must decode simultaneously in one pass. Pins the loop
    /// correctness against a future refactor that only handles
    /// a single escape per field.
    #[test]
    fn unescape_mountinfo_field_handles_multiple_escapes() {
        let raw = "/a\\040b\\011c";
        let decoded = unescape_mountinfo_field(raw);
        assert_eq!(decoded.as_ref(), "/a b\tc");
    }

    /// Non-`\NNN` backslash — defensive: the kernel never emits
    /// this form, but if corrupt /proc/self/mountinfo has a bare
    /// `\` or a partial escape (e.g. `\4` with < 3 following
    /// digits), we must keep the byte literal rather than advance
    /// past it. Silently consuming would produce a shorter
    /// mount_point that could match a different mount than the
    /// caller intended.
    #[test]
    fn unescape_mountinfo_field_preserves_non_octal_backslash() {
        // `\9` — 9 is not an octal digit.
        let raw = "/bad\\9suffix";
        let decoded = unescape_mountinfo_field(raw);
        assert_eq!(decoded.as_ref(), "/bad\\9suffix");

        // Trailing `\` with < 3 bytes after — defensive.
        let raw = "/trunc\\04";
        let decoded = unescape_mountinfo_field(raw);
        assert_eq!(decoded.as_ref(), "/trunc\\04");
    }

    /// `\NNN` octal-overflow guard. The kernel's `mangle_path` only
    /// emits values ≤ `\377` (escape set is `" \t\n\\"`, all ASCII),
    /// but malformed `/proc/self/mountinfo` input — corrupt file,
    /// kernel bug, in-process tampering — must not panic the parser
    /// nor silently truncate via u8 wraparound. Pins three points:
    ///
    ///  - `\377` (val = 0xFF) is the in-range upper bound and decodes
    ///    to the single byte `0xFF`. The lossy UTF-8 layer turns that
    ///    isolated continuation byte into `U+FFFD` — the established
    ///    contract for non-UTF-8 mountinfo bytes.
    ///  - `\400` (val = 0x100) is the first out-of-range value and
    ///    must round-trip as the literal 4-byte sequence `\400`.
    ///    Without this guard, `(b'4' - b'0') << 6` overflows u8 to
    ///    `0` and the parser silently emits `NUL`, masking the
    ///    malformed input.
    ///  - `\777` (val = 0x1FF) is the upper bound of the 3-octal-digit
    ///    grammar — a future change that compared the decoded byte
    ///    against a `>= 0x100` threshold using a wider type but
    ///    forgot to widen the shift would still trip on `\777`.
    #[test]
    fn unescape_mountinfo_field_preserves_out_of_range_octal() {
        // \377 → 0xFF (in range), then lossy UTF-8 → U+FFFD.
        let decoded = unescape_mountinfo_field("\\377");
        assert_eq!(decoded.as_ref(), "\u{FFFD}");

        // \400 (val == 256, > 0xff) — must preserve the literal
        // 4-byte sequence rather than wrap to a NUL byte.
        let decoded = unescape_mountinfo_field("\\400");
        assert_eq!(decoded.as_ref(), "\\400");

        // \777 (val == 511) — upper bound of the 3-octal-digit
        // grammar; same preserve-as-literal contract.
        let decoded = unescape_mountinfo_field("\\777");
        assert_eq!(decoded.as_ref(), "\\777");
    }

    /// `is_octal_digit` accepts exactly the 8 valid digits of a
    /// `\NNN` escape and rejects everything else. Pins the
    /// boundary so a future refactor that uses
    /// `char::is_ascii_digit` (which accepts 8 and 9) regresses
    /// through this test — those two bytes would silently admit
    /// corrupt input as valid octal and produce the wrong
    /// decoded byte.
    #[test]
    fn is_octal_digit_rejects_8_and_9() {
        for b in b'0'..=b'7' {
            assert!(is_octal_digit(b), "byte 0x{b:02x} must be octal");
        }
        assert!(!is_octal_digit(b'8'), "byte 0x38 must NOT be octal");
        assert!(!is_octal_digit(b'9'), "byte 0x39 must NOT be octal");
        assert!(!is_octal_digit(b'a'), "non-digit must NOT be octal");
        assert!(!is_octal_digit(b'/'), "byte before '0' must NOT be octal");
    }

    /// Component-boundary semantics are the correctness property of
    /// `path_starts_with`. Pins the wrapper's contract against a
    /// future refactor that inlines the call but forgets the
    /// `Path::starts_with` semantics.
    #[test]
    fn path_starts_with_respects_component_boundary() {
        assert!(
            path_starts_with(Path::new("/tmp/foo"), Path::new("/tmp")),
            "/tmp/foo must start with /tmp",
        );
        assert!(
            path_starts_with(Path::new("/tmp/foo/bar"), Path::new("/tmp/foo")),
            "/tmp/foo/bar must start with /tmp/foo (deeper component path)",
        );
        assert!(
            !path_starts_with(Path::new("/tmp/foobar"), Path::new("/tmp/foo")),
            "/tmp/foobar must NOT start with /tmp/foo (component boundary)",
        );
        assert!(
            path_starts_with(Path::new("/tmp"), Path::new("/tmp")),
            "/tmp must start with itself (identity)",
        );
        assert!(
            !path_starts_with(Path::new("/"), Path::new("/tmp")),
            "/ is a parent of /tmp, not a child — must NOT match",
        );
    }

    /// `parse_major_minor` happy path — kernel's decimal
    /// `{major}:{minor}` (NOT the hex form /proc/locks emits).
    #[test]
    fn parse_major_minor_happy_path() {
        assert_eq!(parse_major_minor("0:21"), Some((0, 21)));
        assert_eq!(parse_major_minor("259:3"), Some((259, 3)));
    }

    /// Missing colon — invalid format.
    #[test]
    fn parse_major_minor_missing_colon() {
        assert_eq!(parse_major_minor("notvalid"), None);
        assert_eq!(parse_major_minor(""), None);
    }

    /// Non-numeric major or minor.
    #[test]
    fn parse_major_minor_non_numeric() {
        assert_eq!(parse_major_minor("abc:21"), None);
        assert_eq!(parse_major_minor("0:xyz"), None);
        assert_eq!(parse_major_minor(":"), None);
    }

    /// Negative integers. `parse_major_minor` uses `parse::<u32>()`,
    /// which rejects the leading `-`. Pins the unsigned contract —
    /// the kernel never emits negative major:minor, but a corrupt
    /// /proc/self/mountinfo or hand-crafted synthetic must not be
    /// accepted silently (would otherwise miscompare /proc/locks).
    #[test]
    fn parse_major_minor_negative_numbers() {
        assert_eq!(parse_major_minor("-1:0"), None);
        assert_eq!(parse_major_minor("0:-1"), None);
    }

    /// Equivalence between the cached-mountinfo and one-shot
    /// needle-derivation paths.
    ///
    /// `acquire_llc_plan`'s DISCOVER phase reads
    /// `/proc/self/mountinfo` once at the plan level and threads it
    /// through [`needle_from_path_with_mountinfo`] for every LLC
    /// lockfile in the host. The one-shot path
    /// [`needle_from_path`] reads mountinfo inline for each call.
    /// Both must produce byte-identical needles for the same path —
    /// if they diverge, `/proc/locks` byte-equality would fail and
    /// the cached DISCOVER walk would misreport holders.
    ///
    /// Pins equivalence on a real tempfile: the cached path reads
    /// mountinfo once via [`read_mountinfo`] and hands that text to
    /// the contents-seam; the uncached path walks its own internal
    /// read. For the same path, the needles must be equal.
    #[test]
    fn needle_cached_mountinfo_equals_uncached() {
        use tempfile::TempDir;

        let tmp = TempDir::new().expect("tempdir");
        let path = tmp.path().join("cache-equivalence.lock");

        // Materialize a lockfile inode so both paths stat the same
        // underlying file. We use `materialize` because it's the
        // same entry point DISCOVER uses in production, so a
        // divergence between materialize+stat and bare-open would
        // also regress through this test.
        materialize(&path).expect("materialize lockfile");

        // Uncached: inline mountinfo read.
        let uncached = needle_from_path(&path).expect("uncached needle");

        // Cached: read mountinfo once, pass the text through.
        let mountinfo = read_mountinfo().expect("read mountinfo");
        let cached = needle_from_path_with_mountinfo(&path, &mountinfo).expect("cached needle");

        assert_eq!(
            cached, uncached,
            "cached and uncached paths must produce byte-identical needles \
             for the same lockfile. Divergence means DISCOVER's /proc/locks \
             lookup would miss holders the one-shot path would see. \
             uncached={uncached} cached={cached}",
        );
    }

    /// Contents-seam parity: identical synthetic mountinfo text
    /// must produce identical `(major, minor)` tuples via the
    /// cached-wrapper API and the raw parser seam. Catches a
    /// regression where the wrapper's canonicalize-or-fallback
    /// step differs from what the parser expects.
    ///
    /// Uses a tmpfs-covered path so canonicalize succeeds; the
    /// mountinfo fixture covers `/tmp` so both calls hit the same
    /// mount entry.
    #[test]
    fn mount_major_minor_wrapper_matches_parser_seam() {
        let mountinfo = "\
22 28 0:21 / /tmp rw,nosuid,nodev shared:5 - tmpfs tmpfs rw,size=8g
";
        // A tmpfs-covered path the wrapper can canonicalize. `/tmp`
        // itself works on every Linux host nextest runs on.
        let path = Path::new("/tmp");
        let (wrapper_major, wrapper_minor) =
            mount_major_minor_for_path_with_contents(path, mountinfo)
                .expect("wrapper must resolve /tmp under synthetic mountinfo");
        let (parser_major, parser_minor) =
            mount_major_minor_for_path_from_contents(mountinfo, path)
                .expect("parser seam must resolve /tmp");
        assert_eq!(
            (wrapper_major, wrapper_minor),
            (parser_major, parser_minor),
            "wrapper + parser must produce the same (major, minor) for the \
             same (path, mountinfo). Divergence means the cached DISCOVER \
             path is reading different mount state than the uncached \
             one-shot path would.",
        );
        assert_eq!((wrapper_major, wrapper_minor), (0, 21));
    }
}
