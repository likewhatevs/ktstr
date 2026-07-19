// Build-script helpers that are also lib-testable.
//
// Following kernel_path.rs's pattern: this file is `include!`d
// into build.rs (so the helper is available during the build
// script) AND `#[cfg(test)] mod`d into lib.rs (so the `#[cfg(test)]`
// blocks run via `cargo nextest`/`cargo ktstr test`). The
// `#[cfg(test)]` items below are elided from the build.rs include
// because `cargo build` doesn't define `cfg(test)`.

/// Retry `attempt` up to `max_attempts` times with exponential
/// backoff (2s, 4s, 8s before the 2nd / 3rd / 4th tries; no sleep
/// after the final attempt). Emits `cargo:warning=...` log lines
/// at each attempt start and on each failure so build output stays
/// diagnosable.
///
/// `attempt` receives the 1-indexed attempt number; the closure
/// owns its own per-attempt cleanup if the work leaves partial
/// state behind that would block the next try (the helper does no
/// inter-attempt mutation — fully transactional work doesn't need
/// per-attempt cleanup; partial-state work like git clone into a
/// non-empty dir does, and handles it inside the closure conditional
/// on `i > 1`).
///
/// Retry sites in build.rs (busybox tarball download, and wprof
/// git clone when the `wprof` feature is enabled) route through
/// this helper so backoff timing, attempt
/// counting, and log wording stay in lockstep; a change to the
/// retry strategy (e.g. jittered backoff, max-attempts bump) only
/// edits this one function.
///
/// # Panics
///
/// Panics when `max_attempts == 0` — caller bug; the helper has
/// no work to retry. Call sites use `const MAX_*_ATTEMPTS: u32 = 4`
/// literals; the assert guards future callers.
fn retry_with_backoff<F, T>(label: &str, max_attempts: u32, mut attempt: F) -> Result<T, String>
where
    F: FnMut(u32) -> Result<T, String>,
{
    assert!(
        max_attempts > 0,
        "retry_with_backoff requires max_attempts >= 1; got 0 for label {label:?}",
    );
    let mut last_err: Option<String> = None;
    for i in 1..=max_attempts {
        println!("cargo:warning={label}: attempt {i}/{max_attempts}");
        match attempt(i) {
            Ok(v) => return Ok(v),
            Err(e) => {
                println!("cargo:warning={label}: attempt {i} failed: {e}");
                last_err = Some(e);
                if i < max_attempts {
                    // Exponential backoff: 2s, 4s, 8s before the next try.
                    let backoff = 1u64 << i;
                    std::thread::sleep(std::time::Duration::from_secs(backoff));
                }
            }
        }
    }
    Err(last_err.expect(
        "max_attempts > 0 guarded above; loop ran at least once; last_err set on every Err arm",
    ))
}

const PREBUILT_BLOB_STAMP_SCHEMA: &str = "ktstr-prebuilt-blob-v1";

/// Result of considering a `KTSTR_{BUSYBOX,WPROF}_BIN` handoff.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PrebuiltBlobStatus {
    /// The environment did not request a prebuilt handoff.
    NotRequested,
    /// A path was requested but was not a non-empty regular file.
    Rejected,
    /// The fixed `$OUT_DIR` path already contains the handed-over bytes.
    Reused,
    /// Changed bytes were atomically installed at the fixed path.
    Refreshed,
}

fn is_nonempty_regular_file(path: &std::path::Path) -> bool {
    std::fs::metadata(path).is_ok_and(|metadata| metadata.is_file() && metadata.len() > 0)
}

#[derive(Debug)]
struct PrebuiltBlobSnapshot {
    content_key: String,
    len: u64,
    permissions: std::fs::Permissions,
}

/// A same-directory temporary that removes itself unless `publish()` wins.
///
/// Keeping the temporary beside the target makes the final rename atomic and
/// prevents a build-script crash from exposing a partially copied executable
/// through the fixed path consumed by `include_bytes!`.
struct AtomicSibling {
    path: Option<std::path::PathBuf>,
}

impl AtomicSibling {
    fn create(target: &std::path::Path, purpose: &str) -> std::io::Result<(Self, std::fs::File)> {
        use std::fs::OpenOptions;
        use std::sync::atomic::{AtomicU64, Ordering};

        static NEXT_TEMP: AtomicU64 = AtomicU64::new(0);
        let parent = target.parent().unwrap_or_else(|| std::path::Path::new("."));
        let name = target
            .file_name()
            .unwrap_or_else(|| std::ffi::OsStr::new("blob"))
            .to_string_lossy();
        for _ in 0..128 {
            let nonce = NEXT_TEMP.fetch_add(1, Ordering::Relaxed);
            let path = parent.join(format!(
                ".{name}.ktstr-{purpose}-{}-{nonce}",
                std::process::id()
            ));
            match OpenOptions::new().write(true).create_new(true).open(&path) {
                Ok(file) => {
                    return Ok((Self { path: Some(path) }, file));
                }
                Err(err) if err.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(err) => return Err(err),
            }
        }
        Err(std::io::Error::new(
            std::io::ErrorKind::AlreadyExists,
            format!(
                "could not allocate a unique temporary beside {}",
                target.display()
            ),
        ))
    }

    fn path(&self) -> &std::path::Path {
        self.path
            .as_deref()
            .expect("atomic sibling path exists until publication")
    }

    fn publish(mut self, target: &std::path::Path) -> std::io::Result<()> {
        let path = self
            .path
            .as_deref()
            .expect("atomic sibling path exists until publication");
        std::fs::rename(path, target)?;
        self.path = None;
        Ok(())
    }
}

impl Drop for AtomicSibling {
    fn drop(&mut self) {
        if let Some(path) = self.path.take() {
            let _ = std::fs::remove_file(path);
        }
    }
}

fn same_prebuilt_source(left: &std::fs::Metadata, right: &std::fs::Metadata) -> bool {
    if !left.is_file()
        || !right.is_file()
        || left.len() != right.len()
        || left.modified().ok() != right.modified().ok()
    {
        return false;
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        left.dev() == right.dev()
            && left.ino() == right.ino()
            && left.mtime() == right.mtime()
            && left.mtime_nsec() == right.mtime_nsec()
            && left.ctime() == right.ctime()
            && left.ctime_nsec() == right.ctime_nsec()
    }
    #[cfg(not(unix))]
    {
        true
    }
}

/// Hash one stable snapshot of a handed-over blob and optionally copy the same
/// bytes into `sink`.
///
/// Fixed seeds match the repository's other build-time content addressing.
/// The source is checked through both its open descriptor and its pathname
/// before and after the read. A concurrently replaced or rewritten handoff
/// therefore fails closed instead of publishing a mixture of two versions.
fn snapshot_prebuilt_blob(
    src: &std::path::Path,
    blob_name: &str,
    mut sink: Option<&mut std::fs::File>,
) -> std::io::Result<PrebuiltBlobSnapshot> {
    use std::hash::{BuildHasher, Hasher};
    use std::io::{Read, Write};

    let mut source = std::fs::File::open(src)?;
    let before = source.metadata()?;
    let path_before = std::fs::metadata(src)?;
    if !same_prebuilt_source(&before, &path_before) || before.len() == 0 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "prebuilt {blob_name} at {} is not a stable non-empty regular file",
                src.display()
            ),
        ));
    }

    let state = ahash::RandomState::with_seeds(
        0x4b54_5354_522d_4341,
        0x532d_4749_582d_5631,
        0xa076_1d64_78bd_642f,
        0xe703_7ed1_a0b4_28db,
    );
    let mut hasher = state.build_hasher();
    hasher.write_u64(PREBUILT_BLOB_STAMP_SCHEMA.len() as u64);
    hasher.write(PREBUILT_BLOB_STAMP_SCHEMA.as_bytes());
    hasher.write_u64(before.len());

    let mut copied = 0u64;
    let mut buffer = [0u8; 128 * 1024];
    loop {
        let count = source.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        hasher.write(&buffer[..count]);
        if let Some(output) = sink.as_mut() {
            output.write_all(&buffer[..count])?;
        }
        copied = copied
            .checked_add(count as u64)
            .ok_or_else(|| std::io::Error::other("prebuilt blob length overflow"))?;
    }

    let after = source.metadata()?;
    let path_after = std::fs::metadata(src)?;
    if copied != before.len()
        || !same_prebuilt_source(&before, &after)
        || !same_prebuilt_source(&after, &path_after)
    {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "prebuilt {blob_name} at {} changed while it was being read",
                src.display()
            ),
        ));
    }

    Ok(PrebuiltBlobSnapshot {
        content_key: format!("{:016x}", hasher.finish()),
        len: copied,
        permissions: before.permissions(),
    })
}

fn prebuilt_destination_identity(metadata: &std::fs::Metadata) -> String {
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        format!(
            "{}:{}:{}:{}:{}:{}:{}",
            metadata.dev(),
            metadata.ino(),
            metadata.len(),
            metadata.mtime(),
            metadata.mtime_nsec(),
            metadata.ctime(),
            metadata.ctime_nsec(),
        )
    }
    #[cfg(not(unix))]
    {
        let modified = metadata
            .modified()
            .ok()
            .and_then(|time| time.duration_since(std::time::UNIX_EPOCH).ok());
        format!("{}:{modified:?}", metadata.len())
    }
}

fn prebuilt_blob_stamp(
    blob_name: &str,
    source_revision: &str,
    snapshot: &PrebuiltBlobSnapshot,
    destination: &std::fs::Metadata,
) -> String {
    format!(
        "{PREBUILT_BLOB_STAMP_SCHEMA}\n{blob_name}\n{source_revision}\n{}\n{}\n{}\n",
        snapshot.content_key,
        snapshot.len,
        prebuilt_destination_identity(destination),
    )
}

fn parsed_prebuilt_blob_stamp<'a>(
    stamp: &'a str,
    blob_name: &str,
) -> Option<(&'a str, &'a str, u64, &'a str)> {
    let mut lines = stamp.lines();
    if lines.next()? != PREBUILT_BLOB_STAMP_SCHEMA || lines.next()? != blob_name {
        return None;
    }
    let source_revision = lines.next()?;
    let content_key = lines.next()?;
    let len = lines.next()?.parse().ok()?;
    let identity = lines.next()?;
    if source_revision.is_empty()
        || content_key.is_empty()
        || identity.is_empty()
        || lines.next().is_some()
    {
        return None;
    }
    Some((source_revision, content_key, len, identity))
}

fn reuse_prebuilt_blob_without_source(
    stamp_path: &std::path::Path,
    dest: &std::path::Path,
    blob_name: &str,
    source_revision: &str,
) -> bool {
    let Ok(stamp) = std::fs::read_to_string(stamp_path) else {
        return false;
    };
    let Some((revision, content_key, len, stamped_identity)) =
        parsed_prebuilt_blob_stamp(&stamp, blob_name)
    else {
        return false;
    };
    if revision != source_revision {
        return false;
    }
    let Ok(destination) = std::fs::metadata(dest) else {
        return false;
    };
    if !destination.is_file() || destination.len() == 0 || destination.len() != len {
        return false;
    }
    let destination_identity = prebuilt_destination_identity(&destination);
    if stamped_identity == destination_identity {
        return true;
    }

    // An inode/mtime/ctime change is not necessarily a content change: cache
    // restore and atomic replacement commonly publish the same bytes under a
    // new identity. Hash only on that cold path, then bind the sidecar to the
    // replacement without rewriting the executable itself.
    let Ok(snapshot) = snapshot_prebuilt_blob(dest, blob_name, None) else {
        return false;
    };
    if snapshot.len != len || snapshot.content_key != content_key {
        return false;
    }
    let refreshed = prebuilt_blob_stamp(blob_name, source_revision, &snapshot, &destination);
    atomic_write_prebuilt_stamp(stamp_path, &refreshed).is_ok()
}

fn atomic_write_blob_file(
    path: &std::path::Path,
    contents: &[u8],
    purpose: &str,
) -> std::io::Result<()> {
    use std::io::Write;

    let (temp, mut file) = AtomicSibling::create(path, purpose)?;
    file.write_all(contents)?;
    file.flush()?;
    drop(file);
    temp.publish(path)
}

fn atomic_write_prebuilt_stamp(path: &std::path::Path, contents: &str) -> std::io::Result<()> {
    atomic_write_blob_file(path, contents.as_bytes(), "stamp")
}

/// Make a build opt-out authoritative over any real blob left in a reused
/// `OUT_DIR`. The fixed binary path is replaced atomically, so a pre-existing
/// symlink is replaced rather than followed and no reader can observe a
/// partially truncated executable.
fn install_skipped_blob(dest: &std::path::Path, stamp_path: &std::path::Path, blob_name: &str) {
    atomic_write_blob_file(dest, b"", "skipped").unwrap_or_else(|err| {
        panic!(
            "write 0-byte {blob_name} placeholder {}: {err}",
            dest.display()
        )
    });
    atomic_write_prebuilt_stamp(stamp_path, &format!("skipped:{blob_name}")).unwrap_or_else(
        |err| {
            panic!(
                "stamp skipped {blob_name} placeholder {}: {err}",
                stamp_path.display()
            )
        },
    );
}

/// Reuse a pre-built blob binary that `cargo-ktstr` already embedded and
/// extracted, instead of fetching + compiling another copy.
///
/// The handoff is content-addressed rather than keyed only by its environment
/// path or by the existence of `$OUT_DIR/{blob_name}`. This matters when Cargo
/// reuses one `OUT_DIR` and `KTSTR_{BUSYBOX,WPROF}_BIN` is overwritten or
/// redirected to new bytes: the new content key forces an atomic refresh.
/// Unchanged bytes reuse the fixed output without rewriting it, even when the
/// source path changes. A compact destination identity in the sidecar stamp
/// makes that common path metadata-only; if the identity drifted, the helper
/// hashes the destination and re-stamps identical bytes before deciding that a
/// copy is necessary. The stamp also retains the caller's logical source
/// revision, so content addressing supplements pin invalidation rather than
/// making a later wprof revision look current merely because an old handoff
/// still exists.
///
/// A set source is also emitted as `cargo:rerun-if-changed`, complementing the
/// caller's `rerun-if-env-changed` so an in-place source update invokes this
/// comparison. Missing, empty, directory, and zero-byte handoffs are rejected
/// for the caller to fall through to its normal acquisition path.
fn install_prebuilt_blob(
    src: Option<&std::ffi::OsStr>,
    dest: &std::path::Path,
    stamp_path: &std::path::Path,
    blob_name: &str,
    source_revision: &str,
) -> PrebuiltBlobStatus {
    use std::io::Write;

    let src = match src {
        Some(src) if !src.is_empty() => std::path::Path::new(src),
        _ => return PrebuiltBlobStatus::NotRequested,
    };
    println!("cargo:rerun-if-changed={}", src.display());
    if !is_nonempty_regular_file(src) {
        println!(
            "cargo:warning=prebuilt {blob_name} at {} is missing, not a \
             regular file, or empty — building {blob_name} from source instead",
            src.display()
        );
        return PrebuiltBlobStatus::Rejected;
    }

    let source = snapshot_prebuilt_blob(src, blob_name, None).unwrap_or_else(|err| {
        panic!(
            "fingerprint prebuilt {blob_name} at {} before copying: {err}",
            src.display()
        )
    });
    let existing_stamp = std::fs::read_to_string(stamp_path).unwrap_or_default();
    if let Ok(destination) = std::fs::metadata(dest)
        && destination.is_file()
        && destination.len() > 0
    {
        let destination_identity = prebuilt_destination_identity(&destination);
        let trusted_key = parsed_prebuilt_blob_stamp(&existing_stamp, blob_name)
            .filter(|(revision, _, len, identity)| {
                *revision == source_revision
                    && *len == destination.len()
                    && *identity == destination_identity.as_str()
            })
            .map(|(_, key, _, _)| key);
        match trusted_key {
            Some(key) if key == source.content_key => {
                return PrebuiltBlobStatus::Reused;
            }
            Some(_) => {
                // The stamp still names this exact published file, so its
                // differing content key proves a refresh is needed without
                // reading the old destination again.
            }
            None => {
                // A missing/legacy stamp or changed destination identity is
                // not proof that the bytes differ. Hash once before rewriting
                // so same-content handoffs remain true no-ops.
                let current = snapshot_prebuilt_blob(dest, blob_name, None).unwrap_or_else(|err| {
                    panic!(
                        "fingerprint existing prebuilt {blob_name} at {}: {err}",
                        dest.display()
                    )
                });
                if current.content_key == source.content_key && current.len == source.len {
                    let stamp =
                        prebuilt_blob_stamp(blob_name, source_revision, &source, &destination);
                    atomic_write_prebuilt_stamp(stamp_path, &stamp).unwrap_or_else(|err| {
                        panic!(
                            "stamp reused prebuilt {blob_name} at {}: {err}",
                            stamp_path.display()
                        )
                    });
                    return PrebuiltBlobStatus::Reused;
                }
            }
        }
    }

    let (temp, mut output) = AtomicSibling::create(dest, "blob").unwrap_or_else(|err| {
        panic!(
            "create temporary for prebuilt {blob_name} at {}: {err}",
            dest.display()
        )
    });
    let copied = snapshot_prebuilt_blob(src, blob_name, Some(&mut output)).unwrap_or_else(|err| {
        panic!(
            "copy prebuilt {blob_name} from {} to {}: {err}",
            src.display(),
            temp.path().display()
        )
    });
    output.flush().unwrap_or_else(|err| {
        panic!(
            "flush copied prebuilt {blob_name} at {}: {err}",
            temp.path().display()
        )
    });
    std::fs::set_permissions(temp.path(), copied.permissions.clone()).unwrap_or_else(|err| {
        panic!(
            "preserve prebuilt {blob_name} permissions at {}: {err}",
            temp.path().display()
        )
    });
    drop(output);
    assert_eq!(
        (copied.content_key.as_str(), copied.len),
        (source.content_key.as_str(), source.len),
        "prebuilt {blob_name} at {} changed between fingerprint and copy; \
         refusing to publish inconsistent bytes",
        src.display(),
    );
    temp.publish(dest)
        .unwrap_or_else(|err| panic!("publish prebuilt {blob_name} at {}: {err}", dest.display()));
    let destination = std::fs::metadata(dest).unwrap_or_else(|err| {
        panic!(
            "validate published prebuilt {blob_name} at {}: {err}",
            dest.display()
        )
    });
    let stamp = prebuilt_blob_stamp(blob_name, source_revision, &copied, &destination);
    atomic_write_prebuilt_stamp(stamp_path, &stamp).unwrap_or_else(|err| {
        panic!(
            "stamp published prebuilt {blob_name} at {}: {err}",
            stamp_path.display()
        )
    });
    println!("cargo:warning=using embedded {blob_name} from cargo-ktstr (skipped fetch + compile)");
    PrebuiltBlobStatus::Refreshed
}

/// Does `wprof_src` hold a complete recursive git clone? Requires
/// both `.git/HEAD` (init reached) AND `src/Makefile` (working tree
/// populated) — catches "init succeeded, checkout failed" partial
/// clones that the prior Makefile-only check missed.
#[cfg(feature = "wprof")]
fn is_wprof_clone_complete(wprof_src: &std::path::Path) -> bool {
    wprof_src.join(".git").join("HEAD").exists() && wprof_src.join("src").join("Makefile").exists()
}

/// Append an empty `[workspace]` sentinel to every wprof `src/`
/// sub-crate manifest that lacks one, so cargo stops its upward
/// workspace walk at the sub-crate instead of reaching ktstr's
/// workspace via OUT_DIR (the clone lives under `target/`). The wprof
/// Makefile runs a standalone `cargo build` for each sub-crate
/// (`demangle`, `wpb`, `wrust` → `lib*_c.a`); without the sentinel
/// each fails with "current package believes it's in a workspace when
/// it's not." This generalizes the former demangle-only patch so a
/// wprof rev that ships additional sub-crates needs no build.rs edit.
///
/// Scope: immediate child dirs of `src/`. A child whose manifest
/// already declares `[workspace]` is skipped — that covers both the
/// idempotent re-run case and a hypothetical sub-workspace ROOT under
/// `src/` (patching it would be wrong; its members are never immediate
/// `src/` children, so they are never visited). The blazesym submodule
/// lives outside `src/` and is its own workspace, so it is untouched.
///
/// The exact-line check (`l.trim() == "[workspace]"`, not substring)
/// avoids matching `[workspace.lints]` or a commented `# [workspace]`,
/// either of which would trick a substring check into skipping a
/// manifest that lacks the real sentinel table.
#[cfg(feature = "wprof")]
fn isolate_wprof_subcrate_workspaces(wprof_src: &std::path::Path) {
    let src = wprof_src.join("src");
    let entries = match std::fs::read_dir(&src) {
        Ok(e) => e,
        // No `src/` dir means the clone layout changed out from under
        // us; the wprof build (make in `src/`) will fail loudly next,
        // so there is nothing to isolate here.
        Err(_) => return,
    };
    for entry in entries.flatten() {
        if !entry.file_type().map(|t| t.is_dir()).unwrap_or(false) {
            continue;
        }
        let manifest = entry.path().join("Cargo.toml");
        if !manifest.exists() {
            continue;
        }
        let existing = std::fs::read_to_string(&manifest)
            .unwrap_or_else(|e| panic!("read {}: {e}", manifest.display()));
        let is_package = existing.lines().any(|l| l.trim() == "[package]");
        let has_workspace = existing.lines().any(|l| l.trim() == "[workspace]");
        if is_package && !has_workspace {
            use std::io::Write;
            let mut f = std::fs::OpenOptions::new()
                .append(true)
                .open(&manifest)
                .unwrap_or_else(|e| panic!("open {} for append: {e}", manifest.display()));
            f.write_all(b"\n[workspace]\n")
                .unwrap_or_else(|e| panic!("append [workspace] to {}: {e}", manifest.display()));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;
    use std::time::Instant;

    #[test]
    fn succeeds_on_first_try_no_sleep() {
        let calls = Cell::new(0u32);
        let started = Instant::now();
        let r: Result<u32, String> = retry_with_backoff("succeeds-first", 4, |_| {
            calls.set(calls.get() + 1);
            Ok(42)
        });
        assert_eq!(r.unwrap(), 42);
        assert_eq!(calls.get(), 1, "must not retry on success");
        assert!(
            started.elapsed().as_secs() < 1,
            "no sleep on first-try success",
        );
    }

    #[test]
    fn returns_last_err_after_max_attempts() {
        let calls = Cell::new(0u32);
        let r: Result<(), String> = retry_with_backoff("returns-last", 2, |i| {
            calls.set(calls.get() + 1);
            Err(format!("attempt {i} failed"))
        });
        assert_eq!(calls.get(), 2);
        assert!(
            r.unwrap_err().contains("attempt 2 failed"),
            "returns the LAST err, not the first",
        );
    }

    #[test]
    #[should_panic(expected = "retry_with_backoff requires max_attempts >= 1")]
    fn max_zero_panics_with_actionable_message() {
        let _: Result<(), String> = retry_with_backoff("max-zero", 0, |_| Ok(()));
    }

    #[test]
    fn install_prebuilt_blob_copies_nonempty_source() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let src = tmp.path().join("src-busybox");
        std::fs::write(&src, b"BUSYBOX-BINARY-BYTES").expect("write src");
        let dest = tmp.path().join("out-busybox");
        let stamp = tmp.path().join(".busybox-content-key");
        assert_eq!(
            install_prebuilt_blob(
                Some(src.as_os_str()),
                &dest,
                &stamp,
                "busybox",
                "test-revision",
            ),
            PrebuiltBlobStatus::Refreshed,
            "a non-empty source must be atomically installed",
        );
        assert_eq!(
            std::fs::read(&dest).expect("read dest"),
            b"BUSYBOX-BINARY-BYTES",
            "dest must hold the source bytes verbatim",
        );
        let installed_stamp = std::fs::read_to_string(stamp).expect("read content stamp");
        assert!(
            parsed_prebuilt_blob_stamp(&installed_stamp, "busybox")
                .is_some_and(|(revision, _, _, _)| revision == "test-revision"),
            "installed bytes must carry the content-addressed handoff stamp",
        );
    }

    #[test]
    fn prebuilt_handoffs_refresh_changed_bytes_in_one_out_dir() {
        // Exercise the exact reused-OUT_DIR failure mode for both callers.
        // The second blob deliberately has the same length as the first so
        // existence/length/mtime-only schemes cannot make this pass.
        let tmp = tempfile::tempdir().expect("tempdir");
        for blob_name in ["busybox", "wprof"] {
            let source = tmp.path().join(format!("{blob_name}-handoff"));
            let destination = tmp.path().join(blob_name);
            let stamp = tmp.path().join(format!(".{blob_name}-content-key"));

            std::fs::write(&source, b"FIRST-PREBUILT!").expect("write first blob");
            assert_eq!(
                install_prebuilt_blob(
                    Some(source.as_os_str()),
                    &destination,
                    &stamp,
                    blob_name,
                    "test-revision",
                ),
                PrebuiltBlobStatus::Refreshed,
                "{blob_name}: first preparation installs the handoff",
            );
            let first_stamp = std::fs::read_to_string(&stamp).expect("read first stamp");

            std::fs::write(&source, b"SECOND-PREBUILT").expect("replace handoff bytes");
            assert_eq!(
                install_prebuilt_blob(
                    Some(source.as_os_str()),
                    &destination,
                    &stamp,
                    blob_name,
                    "test-revision",
                ),
                PrebuiltBlobStatus::Refreshed,
                "{blob_name}: changed handoff bytes must refresh a reused OUT_DIR",
            );
            assert_eq!(
                std::fs::read(&destination).expect("read refreshed destination"),
                b"SECOND-PREBUILT",
                "{blob_name}: the second handed-over blob must win",
            );
            assert_ne!(
                std::fs::read_to_string(&stamp).expect("read second stamp"),
                first_stamp,
                "{blob_name}: changed content must replace its content key",
            );
            #[cfg(unix)]
            let refreshed_inode = {
                use std::os::unix::fs::MetadataExt;
                std::fs::metadata(&destination)
                    .expect("metadata after refresh")
                    .ino()
            };

            assert_eq!(
                install_prebuilt_blob(
                    Some(source.as_os_str()),
                    &destination,
                    &stamp,
                    blob_name,
                    "test-revision",
                ),
                PrebuiltBlobStatus::Reused,
                "{blob_name}: identical handed-over bytes must not be rewritten",
            );
            #[cfg(unix)]
            {
                use std::os::unix::fs::MetadataExt;
                assert_eq!(
                    std::fs::metadata(&destination)
                        .expect("metadata after reuse")
                        .ino(),
                    refreshed_inode,
                    "{blob_name}: reuse must preserve the published file itself",
                );
            }

            assert_eq!(
                install_prebuilt_blob(
                    Some(source.as_os_str()),
                    &destination,
                    &stamp,
                    blob_name,
                    "next-test-revision",
                ),
                PrebuiltBlobStatus::Reused,
                "{blob_name}: a pin update with identical bytes only needs a re-stamp",
            );
            let rekeyed_stamp = std::fs::read_to_string(&stamp).expect("read re-keyed stamp");
            assert!(
                parsed_prebuilt_blob_stamp(&rekeyed_stamp, blob_name)
                    .is_some_and(|(revision, _, _, _)| revision == "next-test-revision"),
                "{blob_name}: the sidecar must carry the current logical source revision",
            );
            #[cfg(unix)]
            {
                use std::os::unix::fs::MetadataExt;
                assert_eq!(
                    std::fs::metadata(&destination)
                        .expect("metadata after revision re-key")
                        .ino(),
                    refreshed_inode,
                    "{blob_name}: a revision-only stamp change must not rewrite identical bytes",
                );
            }
        }
    }

    #[test]
    fn no_source_prebuilt_reuse_validates_and_rekeys_destination_identity() {
        let tmp = tempfile::tempdir().expect("tempdir");
        for blob_name in ["busybox", "wprof"] {
            let source = tmp.path().join(format!("{blob_name}-source"));
            let destination = tmp.path().join(format!("{blob_name}-destination"));
            let stamp = tmp.path().join(format!(".{blob_name}-content-key"));
            std::fs::write(&source, b"PREBUILT-CONTENT").expect("write source");
            assert_eq!(
                install_prebuilt_blob(
                    Some(source.as_os_str()),
                    &destination,
                    &stamp,
                    blob_name,
                    "test-revision",
                ),
                PrebuiltBlobStatus::Refreshed,
            );

            assert!(
                reuse_prebuilt_blob_without_source(
                    &stamp,
                    &destination,
                    blob_name,
                    "test-revision",
                ),
                "{blob_name}: an untouched published handoff remains reusable after its env disappears",
            );

            // Cache restore / atomic republish can change the inode while
            // preserving content. Re-hash that cold path, refresh only the
            // sidecar identity, and retain the published blob itself.
            let identical = tmp.path().join(format!("{blob_name}-identical"));
            std::fs::write(&identical, b"PREBUILT-CONTENT").expect("write identical replacement");
            std::fs::rename(&identical, &destination).expect("publish identical replacement");
            #[cfg(unix)]
            let identical_inode = {
                use std::os::unix::fs::MetadataExt;
                std::fs::metadata(&destination)
                    .expect("identical replacement metadata")
                    .ino()
            };
            assert!(
                reuse_prebuilt_blob_without_source(
                    &stamp,
                    &destination,
                    blob_name,
                    "test-revision",
                ),
                "{blob_name}: identical bytes under a new identity are re-keyed without source access",
            );
            #[cfg(unix)]
            {
                use std::os::unix::fs::MetadataExt;
                assert_eq!(
                    std::fs::metadata(&destination)
                        .expect("metadata after identity re-key")
                        .ino(),
                    identical_inode,
                    "{blob_name}: re-keying must not rewrite identical destination bytes",
                );
            }

            let changed = tmp.path().join(format!("{blob_name}-changed"));
            std::fs::write(&changed, b"CHANGED!-CONTENT").expect("write changed replacement");
            assert_eq!(
                std::fs::metadata(&changed).expect("changed metadata").len(),
                std::fs::metadata(&destination)
                    .expect("destination metadata")
                    .len(),
                "fixture replacement must keep the same length",
            );
            std::fs::rename(&changed, &destination).expect("publish changed replacement");
            assert!(
                !reuse_prebuilt_blob_without_source(
                    &stamp,
                    &destination,
                    blob_name,
                    "test-revision",
                ),
                "{blob_name}: same-length different bytes cannot reuse the sidecar",
            );

            std::fs::write(&stamp, b"embedded:test-revision").expect("write legacy stamp");
            assert!(
                !reuse_prebuilt_blob_without_source(
                    &stamp,
                    &destination,
                    blob_name,
                    "test-revision",
                ),
                "{blob_name}: a legacy revision-only stamp cannot prove content identity",
            );
        }
    }

    #[test]
    fn skipped_blob_replaces_a_real_reused_out_dir_blob() {
        let tmp = tempfile::tempdir().expect("tempdir");
        for blob_name in ["busybox", "wprof"] {
            let destination = tmp.path().join(blob_name);
            let stamp = tmp.path().join(format!(".{blob_name}-content-key"));
            std::fs::write(&destination, b"REAL-EXECUTABLE").expect("write prior real blob");
            std::fs::write(&stamp, b"prior-content-stamp").expect("write prior stamp");

            install_skipped_blob(&destination, &stamp, blob_name);

            assert_eq!(
                std::fs::metadata(&destination)
                    .expect("placeholder metadata")
                    .len(),
                0,
                "{blob_name}: SKIP must replace a real blob even in reused OUT_DIR",
            );
            assert!(
                !is_nonempty_regular_file(&destination),
                "{blob_name}: a skipped placeholder must not satisfy the build cache gate",
            );
            assert_eq!(
                std::fs::read_to_string(&stamp).expect("skipped stamp"),
                format!("skipped:{blob_name}"),
            );
        }
    }

    #[test]
    fn install_prebuilt_blob_rejects_none_and_empty_string() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let dest = tmp.path().join("out");
        let stamp = tmp.path().join(".content-key");
        assert_eq!(
            install_prebuilt_blob(None, &dest, &stamp, "busybox", "test-revision"),
            PrebuiltBlobStatus::NotRequested,
            "None means no handoff was requested",
        );
        assert_eq!(
            install_prebuilt_blob(
                Some(std::ffi::OsStr::new("")),
                &dest,
                &stamp,
                "busybox",
                "test-revision",
            ),
            PrebuiltBlobStatus::NotRequested,
            "an empty path means no handoff was requested",
        );
        assert!(
            !dest.exists(),
            "dest must not be created when there is no usable source",
        );
    }

    #[test]
    fn install_prebuilt_blob_rejects_missing_source() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let missing = tmp.path().join("does-not-exist");
        let dest = tmp.path().join("out");
        let stamp = tmp.path().join(".content-key");
        assert_eq!(
            install_prebuilt_blob(
                Some(missing.as_os_str()),
                &dest,
                &stamp,
                "busybox",
                "test-revision",
            ),
            PrebuiltBlobStatus::Rejected,
            "a missing source file falls through to acquisition",
        );
        assert!(!dest.exists());
    }

    #[test]
    fn install_prebuilt_blob_rejects_zero_byte_source() {
        // A KTSTR_SKIP_BUSYBOX_BUILD-built cargo-ktstr embeds a 0-byte
        // placeholder; copying it would bake a broken busybox into the
        // guest initramfs. Must reject and fall through to a real build.
        let tmp = tempfile::tempdir().expect("tempdir");
        let empty = tmp.path().join("empty-busybox");
        std::fs::write(&empty, b"").expect("write empty");
        let dest = tmp.path().join("out");
        let stamp = tmp.path().join(".content-key");
        assert_eq!(
            install_prebuilt_blob(
                Some(empty.as_os_str()),
                &dest,
                &stamp,
                "busybox",
                "test-revision",
            ),
            PrebuiltBlobStatus::Rejected,
            "a 0-byte source must fall through to acquisition",
        );
        assert!(!dest.exists(), "must not copy a 0-byte placeholder");
    }

    #[test]
    fn install_prebuilt_blob_rejects_directory_source() {
        // A directory reports metadata().len() > 0 on common
        // filesystems; without the is_file() guard it would pass the
        // non-empty check and then panic in fs::copy. Must be rejected
        // (return false) cleanly so the caller falls through to build.
        let tmp = tempfile::tempdir().expect("tempdir");
        let dir = tmp.path().join("a-directory");
        std::fs::create_dir(&dir).expect("create dir");
        let dest = tmp.path().join("out");
        let stamp = tmp.path().join(".content-key");
        assert_eq!(
            install_prebuilt_blob(
                Some(dir.as_os_str()),
                &dest,
                &stamp,
                "busybox",
                "test-revision",
            ),
            PrebuiltBlobStatus::Rejected,
            "a directory source must fall through to acquisition",
        );
        assert!(!dest.exists());
    }

    #[cfg(feature = "wprof")]
    #[test]
    fn is_wprof_clone_complete_rejects_missing_git_head() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let src = tmp.path();
        std::fs::create_dir_all(src.join("src")).expect("create src/");
        std::fs::write(src.join("src/Makefile"), "").expect("write Makefile");
        // .git/HEAD missing → clone interrupted before init reached:
        // init writes .git/HEAD as its very first step, so absence
        // here is the strongest "no clone happened" signal.
        assert!(
            !is_wprof_clone_complete(src),
            "Makefile alone is not enough; .git/HEAD must also exist",
        );
    }

    #[cfg(feature = "wprof")]
    #[test]
    fn is_wprof_clone_complete_rejects_missing_src_makefile() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let src = tmp.path();
        std::fs::create_dir_all(src.join(".git")).expect("create .git/");
        std::fs::write(src.join(".git/HEAD"), "ref: refs/heads/main\n").expect("write .git/HEAD");
        // .git/HEAD present + Makefile missing → init reached but
        // working tree wasn't populated (fetch / checkout / submodule
        // step failed mid-clone).
        assert!(
            !is_wprof_clone_complete(src),
            ".git/HEAD alone is not enough; src/Makefile must also exist",
        );
    }

    #[cfg(feature = "wprof")]
    #[test]
    fn is_wprof_clone_complete_accepts_both_present() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let src = tmp.path();
        std::fs::create_dir_all(src.join(".git")).expect("create .git/");
        std::fs::write(src.join(".git/HEAD"), "ref: refs/heads/main\n").expect("write .git/HEAD");
        std::fs::create_dir_all(src.join("src")).expect("create src/");
        std::fs::write(src.join("src/Makefile"), "").expect("write Makefile");
        assert!(
            is_wprof_clone_complete(src),
            "both files present → clone considered complete",
        );
    }

    #[cfg(feature = "wprof")]
    #[test]
    fn isolate_patches_subcrate_lacking_workspace() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let ws = tmp.path();
        let crate_dir = ws.join("src").join("wpb");
        std::fs::create_dir_all(&crate_dir).expect("create src/wpb");
        let manifest = crate_dir.join("Cargo.toml");
        std::fs::write(&manifest, "[package]\nname = \"wpb\"\n").expect("write manifest");
        isolate_wprof_subcrate_workspaces(ws);
        let patched = std::fs::read_to_string(&manifest).expect("read manifest");
        assert!(
            patched.lines().any(|l| l.trim() == "[workspace]"),
            "a [package] sub-crate with no [workspace] must get the sentinel",
        );
    }

    #[cfg(feature = "wprof")]
    #[test]
    fn isolate_skips_manifest_already_carrying_workspace() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let ws = tmp.path();
        let crate_dir = ws.join("src").join("demangle");
        std::fs::create_dir_all(&crate_dir).expect("create src/demangle");
        let manifest = crate_dir.join("Cargo.toml");
        std::fs::write(&manifest, "[package]\nname = \"demangle\"\n\n[workspace]\n")
            .expect("write manifest");
        isolate_wprof_subcrate_workspaces(ws);
        let after = std::fs::read_to_string(&manifest).expect("read manifest");
        assert_eq!(
            after.lines().filter(|l| l.trim() == "[workspace]").count(),
            1,
            "an already-patched manifest must not gain a second [workspace] (idempotent)",
        );
    }

    #[cfg(feature = "wprof")]
    #[test]
    fn isolate_patches_every_src_subcrate() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let ws = tmp.path();
        for name in ["demangle", "wpb", "wrust"] {
            let d = ws.join("src").join(name);
            std::fs::create_dir_all(&d).expect("create src/<name>");
            std::fs::write(
                d.join("Cargo.toml"),
                format!("[package]\nname = \"{name}\"\n"),
            )
            .expect("write manifest");
        }
        isolate_wprof_subcrate_workspaces(ws);
        for name in ["demangle", "wpb", "wrust"] {
            let m = std::fs::read_to_string(ws.join("src").join(name).join("Cargo.toml"))
                .expect("read manifest");
            assert!(
                m.lines().any(|l| l.trim() == "[workspace]"),
                "src/{name} must be isolated",
            );
        }
    }

    #[cfg(feature = "wprof")]
    #[test]
    fn isolate_leaves_crates_outside_src_untouched() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let ws = tmp.path();
        // A src/ sub-crate so the isolation loop actually runs.
        let inner = ws.join("src").join("wpb");
        std::fs::create_dir_all(&inner).expect("create src/wpb");
        std::fs::write(inner.join("Cargo.toml"), "[package]\nname = \"wpb\"\n").expect("write");
        // blazesym is a submodule at <wprof>/blazesym (sibling of src/)
        // and is its OWN workspace — patching it would break that
        // workspace, so it must stay untouched.
        let blaze = ws.join("blazesym");
        std::fs::create_dir_all(&blaze).expect("create blazesym");
        let blaze_manifest = blaze.join("Cargo.toml");
        std::fs::write(&blaze_manifest, "[package]\nname = \"blazesym\"\n").expect("write");
        isolate_wprof_subcrate_workspaces(ws);
        assert!(
            std::fs::read_to_string(&blaze_manifest)
                .expect("read")
                .lines()
                .all(|l| l.trim() != "[workspace]"),
            "a crate outside src/ (blazesym submodule) must not be patched",
        );
        assert!(
            std::fs::read_to_string(inner.join("Cargo.toml"))
                .expect("read")
                .lines()
                .any(|l| l.trim() == "[workspace]"),
            "sanity: the src/ sub-crate WAS isolated (loop ran)",
        );
    }
}
