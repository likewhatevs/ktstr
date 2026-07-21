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

const BUILD_BLOB_CACHE_SCHEMA: &str = "ktstr-build-blob-v1";
const BUILD_BLOB_INPUT_SENTINEL: &str = ".ktstr-input-key";
const BUILD_BLOB_OUTPUT_SENTINEL: &str = ".ktstr-blob-key";
const BUILD_BLOB_LOCK_POLL: std::time::Duration = std::time::Duration::from_millis(100);
const BUILD_BLOB_HEARTBEAT: std::time::Duration = std::time::Duration::from_secs(10);

/// Resolve the shared build-blob cache under the same override cascade as the
/// runtime caches. `KTSTR_CACHE_DIR` names the common root directly; the
/// XDG/HOME fallbacks add ktstr's conventional directory first.
///
/// The build script cannot call `crate::cache` (it is a separate crate), so
/// this deliberately mirrors that small, std-only resolution seam. Requiring
/// an absolute path prevents the cross-process namespace from moving with a
/// runner's current directory.
fn build_blob_cache_root(namespace: &str) -> Result<std::path::PathBuf, String> {
    fn absolute(path: std::path::PathBuf, variable: &str) -> Result<std::path::PathBuf, String> {
        if path.is_absolute() {
            Ok(path)
        } else {
            Err(format!(
                "{variable}={:?} is not absolute; the shared build-blob cache must have a stable machine path",
                path
            ))
        }
    }

    if let Some(cache) = std::env::var_os("KTSTR_CACHE_DIR").filter(|value| !value.is_empty()) {
        return absolute(std::path::PathBuf::from(cache), "KTSTR_CACHE_DIR")
            .map(|root| root.join("build-blobs-v1").join(namespace));
    }
    if let Some(xdg) = std::env::var_os("XDG_CACHE_HOME").filter(|value| !value.is_empty()) {
        return absolute(std::path::PathBuf::from(xdg), "XDG_CACHE_HOME")
            .map(|root| root.join("ktstr/build-blobs-v1").join(namespace));
    }
    let home = std::env::var_os("HOME")
        .filter(|value| !value.is_empty())
        .ok_or_else(|| {
            "HOME is unset or empty and neither KTSTR_CACHE_DIR nor XDG_CACHE_HOME names the shared build-blob cache"
                .to_string()
        })?;
    absolute(std::path::PathBuf::from(home), "HOME")
        .map(|root| root.join(".cache/ktstr/build-blobs-v1").join(namespace))
}

fn executable_file(path: &std::path::Path) -> bool {
    let Ok(metadata) = std::fs::metadata(path) else {
        return false;
    };
    if !metadata.is_file() {
        return false;
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt as _;
        metadata.permissions().mode() & 0o111 != 0
    }
    #[cfg(not(unix))]
    {
        true
    }
}

/// Resolve one program exactly as an ordinary `Command::new(program)` lookup
/// would on the Linux hosts supported by ktstr. The returned pathname is used
/// only to fingerprint the executable bytes; it is deliberately not retained
/// in the cache identity because ghars installs identical toolchains below
/// runner-specific home prefixes.
fn resolve_tool_program(program: &str) -> Result<std::path::PathBuf, String> {
    if program.is_empty() {
        return Err("tool program is empty".to_string());
    }
    let direct = std::path::Path::new(program);
    if direct.components().count() > 1 || direct.is_absolute() {
        return executable_file(direct)
            .then(|| direct.to_path_buf())
            .ok_or_else(|| format!("tool program {program:?} is not an executable regular file"));
    }
    let path = std::env::var_os("PATH")
        .ok_or_else(|| format!("PATH is unset while resolving tool program {program:?}"))?;
    std::env::split_paths(&path)
        .map(|directory| directory.join(program))
        .find(|candidate| executable_file(candidate))
        .ok_or_else(|| format!("tool program {program:?} was not found on PATH"))
}

/// Stable identity for an executable independent of the directory spelling
/// used to reach it. The basename is retained because multicall tools select
/// behavior from argv[0]; the executable bytes and mode distinguish patched
/// tools that report the same version.
fn stable_tool_program_identity(program: &str) -> Result<String, String> {
    let path = resolve_tool_program(program)?;
    let snapshot = snapshot_prebuilt_blob(&path, "tool executable").map_err(|error| {
        format!(
            "fingerprint tool program {program:?} at {}: {error}",
            path.display()
        )
    })?;
    let permissions = std::fs::metadata(&path)
        .map_err(|error| format!("stat tool program {}: {error}", path.display()))?
        .permissions();
    let basename = std::path::Path::new(program)
        .file_name()
        .or_else(|| path.file_name())
        .and_then(std::ffi::OsStr::to_str)
        .ok_or_else(|| format!("tool program {program:?} has no UTF-8 basename"))?;
    Ok(format!(
        "exec:{}:{basename}:{}:{}:{}",
        basename.len(),
        snapshot.len,
        snapshot.content_key,
        build_blob_permission_identity(&permissions),
    ))
}

/// Parse the argv-like subset accepted for cached native-build tool
/// assignments. Shell control flow, expansion, and globbing are rejected: they
/// cannot be normalized without executing arbitrary discovery-time commands,
/// and retaining their raw spelling would reintroduce runner-path-specific
/// cache keys. Quotes and backslash escaping remain supported.
fn split_tool_command(command: &str) -> Result<Vec<String>, String> {
    #[derive(Clone, Copy, Eq, PartialEq)]
    enum Quote {
        None,
        Single,
        Double,
    }

    let mut words = Vec::new();
    let mut word = String::new();
    let mut started = false;
    let mut quote = Quote::None;
    let mut chars = command.chars();
    while let Some(character) = chars.next() {
        match quote {
            Quote::Single => {
                if character == '\'' {
                    quote = Quote::None;
                } else {
                    word.push(character);
                }
            }
            Quote::Double => match character {
                '"' => quote = Quote::None,
                '\\' => {
                    let escaped = chars.next().ok_or_else(|| {
                        format!("tool command {command:?} ends in an incomplete escape")
                    })?;
                    word.push(escaped);
                    started = true;
                }
                '$' | '`' => {
                    return Err(format!(
                        "tool command {command:?} uses shell expansion, which has no stable build-cache identity"
                    ));
                }
                _ => word.push(character),
            },
            Quote::None => match character {
                character if character.is_whitespace() => {
                    if started {
                        words.push(std::mem::take(&mut word));
                        started = false;
                    }
                }
                '\'' => {
                    quote = Quote::Single;
                    started = true;
                }
                '"' => {
                    quote = Quote::Double;
                    started = true;
                }
                '\\' => {
                    let escaped = chars.next().ok_or_else(|| {
                        format!("tool command {command:?} ends in an incomplete escape")
                    })?;
                    word.push(escaped);
                    started = true;
                }
                '|' | '&' | ';' | '<' | '>' | '(' | ')' | '$' | '`' | '*' | '?' | '[' => {
                    return Err(format!(
                        "tool command {command:?} uses shell control or expansion syntax, which has no stable build-cache identity"
                    ));
                }
                '~' if !started => {
                    return Err(format!(
                        "tool command {command:?} uses tilde expansion, which has no stable build-cache identity"
                    ));
                }
                '#' if !started => {
                    return Err(format!(
                        "tool command {command:?} uses a shell comment, which has no stable build-cache identity"
                    ));
                }
                _ => {
                    word.push(character);
                    started = true;
                }
            },
        }
    }
    if quote != Quote::None {
        return Err(format!(
            "tool command {command:?} has an unterminated quote"
        ));
    }
    if started {
        words.push(word);
    }
    if words.is_empty() {
        return Err("tool command is empty".to_string());
    }
    Ok(words)
}

/// Normalize a make tool assignment to semantic argv. Any word that resolves
/// to an executable is replaced by basename + content/mode identity; all other
/// arguments remain length-prefixed verbatim. This lets `/runner-a/.../rustc`
/// and `/runner-b/.../rustc` share a cache entry when their actual executable
/// bytes match without conflating different flags or wrapper/compiler chains.
fn stable_tool_command_identity(command: &str) -> Result<String, String> {
    let words = split_tool_command(command)?;
    let mut identity = String::new();
    for word in words {
        let normalized = stable_tool_program_identity(&word)
            .unwrap_or_else(|_| format!("arg:{}:{word}", word.len()));
        identity.push_str(&normalized.len().to_string());
        identity.push(':');
        identity.push_str(&normalized);
        identity.push('\n');
    }
    Ok(identity)
}

/// Fixed-seed, non-cryptographic identity for one exact build input tuple.
/// Length-prefixing prevents tuple ambiguity and keeps the key stable across
/// processes without paying for a cryptographic digest.
fn build_blob_content_id(parts: &[&str]) -> String {
    use std::hash::{BuildHasher as _, Hasher as _};

    let state = ahash::RandomState::with_seeds(
        0x4b54_5354_522d_4341,
        0x532d_424c_4f42_5631,
        0xa076_1d64_78bd_642f,
        0xe703_7ed1_a0b4_28db,
    );
    let mut hasher = state.build_hasher();
    hasher.write_u64(BUILD_BLOB_CACHE_SCHEMA.len() as u64);
    hasher.write(BUILD_BLOB_CACHE_SCHEMA.as_bytes());
    for part in parts {
        hasher.write_u64(part.len() as u64);
        hasher.write(part.as_bytes());
    }
    format!("{:016x}", hasher.finish())
}

fn build_blob_input_manifest(parts: &[&str]) -> String {
    let mut manifest = String::from(BUILD_BLOB_CACHE_SCHEMA);
    for part in parts {
        manifest.push('\n');
        manifest.push_str(&part.len().to_string());
        manifest.push(':');
        manifest.push_str(part);
    }
    manifest
}

fn validate_build_blob_name(blob_name: &str) -> Result<(), String> {
    let mut components = std::path::Path::new(blob_name).components();
    if !matches!(components.next(), Some(std::path::Component::Normal(_)))
        || components.next().is_some()
    {
        return Err(format!(
            "build-blob name must be one normal path component, got {blob_name:?}"
        ));
    }
    Ok(())
}

fn build_blob_output_manifest(path: &std::path::Path, blob_name: &str) -> Result<String, String> {
    let snapshot = snapshot_prebuilt_blob(path, blob_name).map_err(|error| {
        format!(
            "fingerprint built blob {} before publication: {error}",
            path.display()
        )
    })?;
    let permissions = std::fs::metadata(path)
        .map_err(|error| format!("stat built blob {}: {error}", path.display()))?
        .permissions();
    Ok(format!(
        "{BUILD_BLOB_CACHE_SCHEMA}\n{blob_name}\n{}\n{}\n{}\n",
        snapshot.len,
        snapshot.content_key,
        build_blob_permission_identity(&permissions)
    ))
}

fn build_blob_permission_identity(permissions: &std::fs::Permissions) -> String {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt as _;
        format!("unix:{:o}", permissions.mode())
    }
    #[cfg(not(unix))]
    {
        format!("readonly:{}", permissions.readonly())
    }
}

fn parsed_build_blob_output_manifest(value: &str) -> Option<(&str, u64, &str, &str)> {
    let mut lines = value.lines();
    if lines.next()? != BUILD_BLOB_CACHE_SCHEMA {
        return None;
    }
    let blob_name = lines.next()?;
    let len = lines.next()?.parse().ok()?;
    let content_key = lines.next()?;
    let permissions = lines.next()?;
    if blob_name.is_empty()
        || len == 0
        || content_key.is_empty()
        || permissions.is_empty()
        || lines.next().is_some()
    {
        return None;
    }
    Some((blob_name, len, content_key, permissions))
}

fn cached_build_blob_complete(
    entry: &std::path::Path,
    input_manifest: &str,
    blob_name: &str,
) -> bool {
    if !std::fs::read_to_string(entry.join(BUILD_BLOB_INPUT_SENTINEL))
        .is_ok_and(|value| value == input_manifest)
    {
        return false;
    }
    let Ok(output_manifest) = std::fs::read_to_string(entry.join(BUILD_BLOB_OUTPUT_SENTINEL))
    else {
        return false;
    };
    let Some((manifest_name, manifest_len, manifest_key, manifest_permissions)) =
        parsed_build_blob_output_manifest(&output_manifest)
    else {
        return false;
    };
    if manifest_name != blob_name {
        return false;
    }
    let path = entry.join(blob_name);
    let Ok(metadata) = std::fs::symlink_metadata(&path) else {
        return false;
    };
    if !metadata.file_type().is_file()
        || metadata.len() != manifest_len
        || build_blob_permission_identity(&metadata.permissions()) != manifest_permissions
    {
        return false;
    }
    snapshot_prebuilt_blob(&path, blob_name)
        .is_ok_and(|snapshot| snapshot.len == manifest_len && snapshot.content_key == manifest_key)
}

struct RemoveBuildBlobStage(Option<std::path::PathBuf>);

impl RemoveBuildBlobStage {
    fn disarm(mut self) {
        self.0 = None;
    }
}

impl Drop for RemoveBuildBlobStage {
    fn drop(&mut self) {
        if let Some(path) = self.0.take() {
            let _ = std::fs::remove_dir_all(path);
        }
    }
}

fn remove_build_blob_path(path: &std::path::Path) -> Result<(), String> {
    match std::fs::symlink_metadata(path) {
        Ok(metadata) if metadata.file_type().is_dir() => std::fs::remove_dir_all(path)
            .map_err(|error| format!("remove directory {}: {error}", path.display())),
        Ok(_) => std::fs::remove_file(path)
            .map_err(|error| format!("remove file {}: {error}", path.display())),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(format!(
            "inspect {} before removal: {error}",
            path.display()
        )),
    }
}

fn remove_stale_build_blob_stages(root: &std::path::Path, id: &str) -> Result<(), String> {
    let prefix = format!(".{id}.work-");
    for entry in std::fs::read_dir(root)
        .map_err(|error| format!("scan build-blob cache {}: {error}", root.display()))?
    {
        let entry = entry.map_err(|error| format!("read build-blob cache entry: {error}"))?;
        if entry.file_name().to_string_lossy().starts_with(&prefix) {
            remove_build_blob_path(&entry.path())?;
        }
    }
    Ok(())
}

/// Elect one cross-process builder for an exact input tuple and atomically
/// publish one immutable executable blob. A crash leaves only a private stage;
/// the next elected builder removes it. Deleted or incomplete final entries
/// fail validation and are rebuilt under the same lock.
fn ensure_cached_build_blob<Build>(
    root: &std::path::Path,
    parts: &[&str],
    label: &str,
    blob_name: &str,
    build: Build,
) -> Result<std::path::PathBuf, String>
where
    Build: FnOnce(&std::path::Path) -> Result<(), String>,
{
    validate_build_blob_name(blob_name)?;
    let id = build_blob_content_id(parts);
    let input_manifest = build_blob_input_manifest(parts);
    let entry = root.join(&id);
    if cached_build_blob_complete(&entry, &input_manifest, blob_name) {
        return Ok(entry.join(blob_name));
    }

    std::fs::create_dir_all(root)
        .map_err(|error| format!("create build-blob cache {}: {error}", root.display()))?;
    let lock_path = root.join(format!(".{id}.lock"));
    let lock = std::fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(&lock_path)
        .map_err(|error| format!("open build-blob lock {}: {error}", lock_path.display()))?;
    let started = std::time::Instant::now();
    let mut next_heartbeat = BUILD_BLOB_HEARTBEAT;
    loop {
        match fs2::FileExt::try_lock_exclusive(&lock) {
            Ok(()) => break,
            Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                if started.elapsed() >= next_heartbeat {
                    println!(
                        "cargo:warning={label}: waiting for shared builder; elapsed={:.1}s",
                        started.elapsed().as_secs_f64()
                    );
                    next_heartbeat += BUILD_BLOB_HEARTBEAT;
                }
                std::thread::sleep(BUILD_BLOB_LOCK_POLL);
            }
            Err(error) => {
                return Err(format!(
                    "acquire build-blob lock {}: {error}",
                    lock_path.display()
                ));
            }
        }
    }

    if cached_build_blob_complete(&entry, &input_manifest, blob_name) {
        return Ok(entry.join(blob_name));
    }
    remove_build_blob_path(&entry)?;
    remove_stale_build_blob_stages(root, &id)?;

    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let stage = root.join(format!(".{id}.work-{}-{nonce}", std::process::id()));
    remove_build_blob_path(&stage)?;
    std::fs::create_dir_all(&stage)
        .map_err(|error| format!("create build-blob stage {}: {error}", stage.display()))?;
    let stage_guard = RemoveBuildBlobStage(Some(stage.clone()));

    println!("cargo:warning={label}: elected shared builder");
    build(&stage)?;
    let blob = stage.join(blob_name);
    let output_manifest = build_blob_output_manifest(&blob, blob_name)?;
    atomic_write_blob_file(
        &stage.join(BUILD_BLOB_OUTPUT_SENTINEL),
        output_manifest.as_bytes(),
        "output-key",
    )
    .map_err(|error| format!("write built-blob output sentinel: {error}"))?;
    atomic_write_blob_file(
        &stage.join(BUILD_BLOB_INPUT_SENTINEL),
        input_manifest.as_bytes(),
        "input-key",
    )
    .map_err(|error| format!("write built-blob input sentinel: {error}"))?;
    if !cached_build_blob_complete(&stage, &input_manifest, blob_name) {
        return Err(format!(
            "{label}: builder returned without a complete {blob_name} blob"
        ));
    }
    std::fs::File::open(&blob)
        .and_then(|file| file.sync_all())
        .map_err(|error| format!("sync built blob {}: {error}", blob.display()))?;
    std::fs::File::open(&stage)
        .and_then(|directory| directory.sync_all())
        .map_err(|error| format!("sync build-blob stage {}: {error}", stage.display()))?;
    std::fs::rename(&stage, &entry).map_err(|error| {
        format!(
            "atomically publish build-blob cache {} -> {}: {error}",
            stage.display(),
            entry.display()
        )
    })?;
    stage_guard.disarm();
    std::fs::File::open(root)
        .and_then(|directory| directory.sync_all())
        .map_err(|error| format!("sync build-blob cache {}: {error}", root.display()))?;
    println!(
        "cargo:warning={label}: published shared blob in {:.1}s",
        started.elapsed().as_secs_f64()
    );
    Ok(entry.join(blob_name))
}

#[cfg(target_os = "linux")]
fn ficlone_build_blob(destination: &std::fs::File, source: &std::fs::File) -> std::io::Result<()> {
    use std::os::fd::AsRawFd as _;

    const FICLONE_IOCTL: std::os::raw::c_ulong = 0x4004_9409;
    unsafe extern "C" {
        fn ioctl(
            fd: std::os::raw::c_int,
            request: std::os::raw::c_ulong,
            ...
        ) -> std::os::raw::c_int;
    }
    // SAFETY: FICLONE consumes two valid descriptors for the duration of the
    // call and does not retain them. The destination is open read/write.
    let result = unsafe { ioctl(destination.as_raw_fd(), FICLONE_IOCTL, source.as_raw_fd()) };
    if result == 0 {
        Ok(())
    } else {
        Err(std::io::Error::last_os_error())
    }
}

#[cfg(not(target_os = "linux"))]
fn ficlone_build_blob(
    _destination: &std::fs::File,
    _source: &std::fs::File,
) -> std::io::Result<()> {
    Err(std::io::Error::new(
        std::io::ErrorKind::Unsupported,
        "FICLONE is only available on Linux",
    ))
}

fn materialize_build_blob_with<Clone>(
    source_path: &std::path::Path,
    destination_path: &std::path::Path,
    clone: Clone,
) -> std::io::Result<()>
where
    Clone: FnOnce(&std::fs::File, &std::fs::File) -> std::io::Result<()>,
{
    let source = std::fs::File::open(source_path)?;
    let metadata = source.metadata()?;
    if !metadata.is_file() || metadata.len() == 0 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "cached build blob {} is not a non-empty regular file",
                source_path.display()
            ),
        ));
    }
    let (temporary, destination) = AtomicSibling::create(destination_path, "reflink")?;
    clone(&destination, &source).map_err(|error| {
        std::io::Error::new(
            error.kind(),
            format!(
                "FICLONE {} -> {} failed ({error}); place KTSTR_CACHE_DIR and Cargo's target directory on the same reflink-capable filesystem",
                source_path.display(),
                temporary.path().display()
            ),
        )
    })?;
    destination.set_permissions(metadata.permissions())?;
    destination.sync_all()?;
    drop(destination);
    temporary.publish(destination_path)
}

/// Materialize an immutable cached blob into one private OUT_DIR inode. This
/// is intentionally strict: a byte-copy fallback would multiply every large
/// embedded blob across concurrent Cargo runners and hide a broken COW setup.
fn materialize_build_blob(
    source_path: &std::path::Path,
    destination_path: &std::path::Path,
) -> std::io::Result<()> {
    materialize_build_blob_with(source_path, destination_path, |destination, source| {
        ficlone_build_blob(destination, source)
    })
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

#[derive(Clone, Debug, Eq, PartialEq)]
struct PrebuiltBlobSnapshot {
    content_key: String,
    len: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct ValidatedBuiltBlob {
    snapshot: PrebuiltBlobSnapshot,
    permissions: String,
}

/// A same-directory temporary that removes itself unless `publish()` wins.
///
/// Keeping the temporary beside the target makes the final rename atomic and
/// prevents a build-script crash from exposing a partially materialized executable
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
            match OpenOptions::new()
                .read(true)
                .write(true)
                .create_new(true)
                .open(&path)
            {
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

/// Hash one stable snapshot of a handed-over blob.
///
/// Fixed seeds match the repository's other build-time content addressing.
/// The source is checked through both its open descriptor and its pathname
/// before and after the read. A concurrently replaced or rewritten handoff
/// therefore fails closed instead of publishing a mixture of two versions.
fn snapshot_prebuilt_blob(
    src: &std::path::Path,
    blob_name: &str,
) -> std::io::Result<PrebuiltBlobSnapshot> {
    use std::hash::{BuildHasher, Hasher};
    use std::io::Read;

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

const BUILT_BLOB_STAMP_SCHEMA: &str = "ktstr-built-blob-v1";

fn built_blob_stamp(
    blob_name: &str,
    input_id: &str,
    snapshot: &PrebuiltBlobSnapshot,
    destination: &std::fs::Metadata,
) -> String {
    format!(
        "{BUILT_BLOB_STAMP_SCHEMA}\n{blob_name}\n{input_id}\n{}\n{}\n{}\n{}\n",
        snapshot.content_key,
        snapshot.len,
        build_blob_permission_identity(&destination.permissions()),
        prebuilt_destination_identity(destination),
    )
}

fn parsed_built_blob_stamp<'a>(
    stamp: &'a str,
    blob_name: &str,
    input_id: &str,
) -> Option<(&'a str, u64, &'a str, &'a str)> {
    let mut lines = stamp.lines();
    if lines.next()? != BUILT_BLOB_STAMP_SCHEMA
        || lines.next()? != blob_name
        || lines.next()? != input_id
    {
        return None;
    }
    let content_key = lines.next()?;
    let len = lines.next()?.parse().ok()?;
    let permissions = lines.next()?;
    let identity = lines.next()?;
    if content_key.is_empty()
        || len == 0
        || permissions.is_empty()
        || identity.is_empty()
        || lines.next().is_some()
    {
        return None;
    }
    Some((content_key, len, permissions, identity))
}

/// Validate a private OUT_DIR materialization against a content-bound stamp.
///
/// The inode identity makes the common path metadata-only. Any replacement or
/// in-place write changes that identity and forces a complete content hash;
/// identical cache restores are then re-stamped, while corruption fails
/// closed and can be repaired from the shared CAS.
fn validated_built_blob(
    stamp_path: &std::path::Path,
    blob_path: &std::path::Path,
    blob_name: &str,
    input_id: &str,
) -> Option<ValidatedBuiltBlob> {
    let stamp = std::fs::read_to_string(stamp_path).ok()?;
    let (content_key, len, permissions, stamped_identity) =
        parsed_built_blob_stamp(&stamp, blob_name, input_id)?;
    let metadata = std::fs::symlink_metadata(blob_path).ok()?;
    if !metadata.file_type().is_file()
        || metadata.len() != len
        || build_blob_permission_identity(&metadata.permissions()) != permissions
    {
        return None;
    }
    let expected = ValidatedBuiltBlob {
        snapshot: PrebuiltBlobSnapshot {
            content_key: content_key.to_string(),
            len,
        },
        permissions: permissions.to_string(),
    };
    if prebuilt_destination_identity(&metadata) == stamped_identity {
        return Some(expected);
    }

    let actual = snapshot_prebuilt_blob(blob_path, blob_name).ok()?;
    if actual != expected.snapshot {
        return None;
    }
    let refreshed = built_blob_stamp(blob_name, input_id, &actual, &metadata);
    atomic_write_prebuilt_stamp(stamp_path, &refreshed).ok()?;
    Some(expected)
}

/// Ensure both sides of one build-blob handoff: a validated shared CAS entry
/// and its private OUT_DIR COW inode.
///
/// A valid local inode can reseed a manually deleted or corrupt shared entry,
/// avoiding a repeated source build. A corrupt local inode is never trusted:
/// an intact CAS repairs it, and only loss of both copies invokes `build`.
fn ensure_materialized_build_blob<Build>(
    root: &std::path::Path,
    parts: &[&str],
    label: &str,
    blob_name: &str,
    destination: &std::path::Path,
    stamp_path: &std::path::Path,
    build: Build,
) -> Result<std::path::PathBuf, String>
where
    Build: FnOnce(&std::path::Path) -> Result<(), String>,
{
    let input_id = build_blob_content_id(parts);
    let local = validated_built_blob(stamp_path, destination, blob_name, &input_id);
    let local_seed = local.clone();
    let cached = ensure_cached_build_blob(root, parts, label, blob_name, |stage| {
        if let Some(expected) = local_seed {
            let staged = stage.join(blob_name);
            materialize_build_blob(destination, &staged).map_err(|error| {
                format!(
                    "COW-reseed {label} from {} into shared cache: {error}",
                    destination.display()
                )
            })?;
            let actual = snapshot_prebuilt_blob(&staged, blob_name)
                .map_err(|error| format!("verify COW-reseeded {label}: {error}"))?;
            let permissions = std::fs::metadata(&staged)
                .map_err(|error| format!("stat COW-reseeded {label}: {error}"))?
                .permissions();
            if actual != expected.snapshot
                || build_blob_permission_identity(&permissions) != expected.permissions
            {
                return Err(format!(
                    "{label}: local blob changed while reseeding the shared cache"
                ));
            }
            Ok(())
        } else {
            build(stage)
        }
    })?;

    let cached_snapshot = snapshot_prebuilt_blob(&cached, blob_name)
        .map_err(|error| format!("fingerprint cached {label}: {error}"))?;
    let cached_permissions = std::fs::metadata(&cached)
        .map_err(|error| format!("stat cached {label}: {error}"))?
        .permissions();
    let cached_permission_id = build_blob_permission_identity(&cached_permissions);
    let local_matches = local.is_some_and(|local| {
        local.snapshot == cached_snapshot
            && local.permissions == cached_permission_id
            && std::fs::symlink_metadata(destination).is_ok_and(|metadata| {
                metadata.file_type().is_file()
                    && build_blob_permission_identity(&metadata.permissions())
                        == cached_permission_id
            })
    });
    if !local_matches {
        materialize_build_blob(&cached, destination).map_err(|error| {
            format!(
                "COW-materialize cached {label} {} -> {}: {error}",
                cached.display(),
                destination.display()
            )
        })?;
    }

    let published = snapshot_prebuilt_blob(destination, blob_name)
        .map_err(|error| format!("fingerprint materialized {label}: {error}"))?;
    if published != cached_snapshot {
        return Err(format!(
            "{label}: materialized output differs from the validated shared cache"
        ));
    }
    let destination_metadata = std::fs::metadata(destination)
        .map_err(|error| format!("stat materialized {label}: {error}"))?;
    if build_blob_permission_identity(&destination_metadata.permissions()) != cached_permission_id {
        return Err(format!(
            "{label}: materialized output permissions differ from the validated shared cache"
        ));
    }
    let stamp = built_blob_stamp(blob_name, &input_id, &published, &destination_metadata);
    atomic_write_prebuilt_stamp(stamp_path, &stamp)
        .map_err(|error| format!("stamp materialized {label}: {error}"))?;
    Ok(cached)
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
    let Ok(snapshot) = snapshot_prebuilt_blob(dest, blob_name) else {
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
/// extracted, instead of fetching + compiling another instance.
///
/// The handoff is content-addressed rather than keyed only by its environment
/// path or by the existence of `$OUT_DIR/{blob_name}`. This matters when Cargo
/// reuses one `OUT_DIR` and `KTSTR_{BUSYBOX,WPROF}_BIN` is overwritten or
/// redirected to new bytes: the new content key forces an atomic refresh.
/// Unchanged bytes reuse the fixed output without rewriting it, even when the
/// source path changes. A compact destination identity in the sidecar stamp
/// makes that common path metadata-only; if the identity drifted, the helper
/// hashes the destination and re-stamps identical bytes before deciding that a
/// COW refresh is necessary. The stamp also retains the caller's logical source
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

    let source = snapshot_prebuilt_blob(src, blob_name).unwrap_or_else(|err| {
        panic!(
            "fingerprint prebuilt {blob_name} at {} before COW materialization: {err}",
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
                let current = snapshot_prebuilt_blob(dest, blob_name).unwrap_or_else(|err| {
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

    materialize_build_blob(src, dest).unwrap_or_else(|err| {
        panic!(
            "COW-materialize prebuilt {blob_name} from {} to {}: {err}",
            src.display(),
            dest.display()
        )
    });
    let copied = snapshot_prebuilt_blob(dest, blob_name).unwrap_or_else(|err| {
        panic!(
            "fingerprint COW-materialized prebuilt {blob_name} at {}: {err}",
            dest.display()
        )
    });
    assert_eq!(
        (copied.content_key.as_str(), copied.len),
        (source.content_key.as_str(), source.len),
        "prebuilt {blob_name} at {} changed between fingerprint and COW materialization; \
         refusing to publish inconsistent bytes",
        src.display(),
    );
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

#[cfg(feature = "vendored")]
const BUSYBOX_HERMETIC_MAKE_ASSIGNMENTS: &[(&str, &str)] =
    &[("KBUILD_OUTPUT", ""), ("KCONFIG_CONFIG", ".config")];

#[cfg(feature = "vendored")]
const BUSYBOX_KEYED_BUILD_ENVIRONMENT: &[&str] = &[
    "ARCH",
    "CFLAGS",
    "CPPFLAGS",
    "HOSTCFLAGS",
    "HOSTLDFLAGS",
    "KBUILD_BUILD_HOST",
    "KBUILD_BUILD_TIMESTAMP",
    "KBUILD_BUILD_USER",
    "KCFLAGS",
    "KCONFIG_ALLCONFIG",
    "KCPPFLAGS",
    "LDFLAGS",
    "SOURCE_DATE_EPOCH",
];

/// Keep the shared BusyBox builder inside its elected CAS stage.
///
/// BusyBox's kbuild accepts both variables from the ambient environment. An
/// operator's kernel-build setup could otherwise send the output to a shared
/// `KBUILD_OUTPUT` directory or make our config edits target a different file
/// than kconfig reads. Command-line assignments are stronger than inherited
/// environment and propagate to recursive makes; removing the environment
/// entries as well makes the child contract explicit in diagnostics.
#[cfg(feature = "vendored")]
fn configure_hermetic_busybox_make(command: &mut std::process::Command) {
    for (name, value) in BUSYBOX_HERMETIC_MAKE_ASSIGNMENTS {
        command.env_remove(name).arg(format!("{name}={value}"));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;
    use std::time::Instant;

    #[test]
    fn build_blob_cache_root_is_absolute_and_namespaced() {
        let root = build_blob_cache_root("fixture").expect("resolve build-blob cache root");
        assert!(root.is_absolute(), "cache root must be absolute: {root:?}");
        assert!(
            root.ends_with("build-blobs-v1/fixture"),
            "cache root must carry schema and namespace: {root:?}",
        );
    }

    #[cfg(feature = "vendored")]
    #[test]
    fn busybox_make_forces_source_local_output_and_config() {
        let mut command = std::process::Command::new("make");
        configure_hermetic_busybox_make(&mut command);

        let args = command
            .get_args()
            .map(|arg| arg.to_string_lossy().into_owned())
            .collect::<Vec<_>>();
        assert_eq!(
            args,
            ["KBUILD_OUTPUT=", "KCONFIG_CONFIG=.config"],
            "command-line make assignments must override both environment and recursive make state"
        );
        let removals = command
            .get_envs()
            .filter(|(_, value)| value.is_none())
            .map(|(name, _)| name.to_string_lossy().into_owned())
            .collect::<Vec<_>>();
        assert_eq!(
            removals,
            ["KBUILD_OUTPUT", "KCONFIG_CONFIG"],
            "ambient redirect variables must not reach BusyBox make"
        );
        for (name, _) in BUSYBOX_HERMETIC_MAKE_ASSIGNMENTS {
            assert!(
                !BUSYBOX_KEYED_BUILD_ENVIRONMENT.contains(name),
                "ignored ambient redirect {name} must not create useless shared-cache splits"
            );
        }
    }

    #[test]
    fn tool_identity_ignores_runner_directory_but_not_bytes_basename_or_argv() {
        let temp = tempfile::tempdir().expect("tempdir");
        let left_dir = temp.path().join("ghars-ktstr-x64-1/toolchain/bin");
        let right_dir = temp.path().join("ghars-ktstr-x64-9/toolchain/bin");
        std::fs::create_dir_all(&left_dir).expect("create left tool directory");
        std::fs::create_dir_all(&right_dir).expect("create right tool directory");
        let left = left_dir.join("rustc");
        let right = right_dir.join("rustc");
        std::fs::write(&left, b"identical-tool-payload").expect("write left tool");
        std::fs::write(&right, b"identical-tool-payload").expect("write right tool");
        #[cfg(unix)]
        for path in [&left, &right] {
            use std::os::unix::fs::PermissionsExt as _;
            std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o755))
                .expect("make fixture executable");
        }

        let left_command = format!("{} --crate-name fixture", left.display());
        let right_command = format!("{} --crate-name fixture", right.display());
        assert_eq!(
            stable_tool_command_identity(&left_command).expect("left identity"),
            stable_tool_command_identity(&right_command).expect("right identity"),
            "runner-local parent directories must not split identical tool payloads"
        );
        assert_ne!(
            stable_tool_command_identity(&left_command).expect("left identity"),
            stable_tool_command_identity(&format!("{} --crate-name changed", right.display()))
                .expect("changed-argv identity"),
            "command arguments remain part of exact build semantics"
        );

        std::fs::write(&right, b"different-tool-payload").expect("replace right tool bytes");
        assert_ne!(
            stable_tool_command_identity(&left_command).expect("left identity"),
            stable_tool_command_identity(&right_command).expect("changed-content identity"),
            "different executable bytes must split cache entries"
        );

        let alias = right_dir.join("cargo");
        std::fs::write(&alias, b"identical-tool-payload").expect("write basename alias");
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt as _;
            std::fs::set_permissions(&alias, std::fs::Permissions::from_mode(0o755))
                .expect("make alias executable");
        }
        assert_ne!(
            stable_tool_program_identity(left.to_str().expect("UTF-8 fixture path"))
                .expect("rustc identity"),
            stable_tool_program_identity(alias.to_str().expect("UTF-8 fixture path"))
                .expect("cargo identity"),
            "argv[0] basename remains semantic for multicall executables"
        );
    }

    #[test]
    fn tool_command_parser_supports_argv_quoting_and_rejects_expansion() {
        assert_eq!(
            split_tool_command("ccache 'clang compiler' \"-DVALUE=two words\" escaped\\ value")
                .expect("parse argv-like command"),
            [
                "ccache",
                "clang compiler",
                "-DVALUE=two words",
                "escaped value"
            ]
        );
        for command in [
            "gcc $CFLAGS",
            "gcc $(hostname)",
            "gcc *.c",
            "gcc | tee output",
            "~/bin/gcc",
        ] {
            assert!(
                split_tool_command(command).is_err(),
                "unstable shell expression must be rejected: {command:?}"
            );
        }
    }

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

    const BUILD_BLOB_HELPER_MODE: &str = "KTSTR_BUILD_BLOB_HELPER_MODE";
    const BUILD_BLOB_HELPER_ROOT: &str = "KTSTR_BUILD_BLOB_HELPER_ROOT";
    const BUILD_BLOB_HELPER_GATE: &str = "KTSTR_BUILD_BLOB_HELPER_GATE";
    const BUILD_BLOB_HELPER_COUNT: &str = "KTSTR_BUILD_BLOB_HELPER_COUNT";
    const BUILD_BLOB_HELPER_PARTS: &[&str] = &[
        "fixture-recipe",
        "fixture-revision",
        "x86_64",
        "fixture-toolchain",
    ];

    fn spawn_build_blob_helper(
        root: &std::path::Path,
        gate: &std::path::Path,
        count: &std::path::Path,
        mode: &str,
    ) -> std::process::Child {
        std::process::Command::new(std::env::current_exe().expect("current test executable"))
            .args([
                "--exact",
                "build_helpers::tests::build_blob_cache_process_helper",
                "--ignored",
                "--nocapture",
                "--test-threads=1",
            ])
            .env(BUILD_BLOB_HELPER_MODE, mode)
            .env(BUILD_BLOB_HELPER_ROOT, root)
            .env(BUILD_BLOB_HELPER_GATE, gate)
            .env(BUILD_BLOB_HELPER_COUNT, count)
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .expect("spawn build-blob cache helper")
    }

    fn wait_build_blob_helper(mut child: std::process::Child, expected_code: i32) {
        use std::io::Read as _;

        let deadline = Instant::now() + std::time::Duration::from_secs(30);
        let (status, timed_out) = loop {
            match child.try_wait().expect("poll build-blob helper") {
                Some(status) => break (status, false),
                None if Instant::now() < deadline => {
                    std::thread::sleep(std::time::Duration::from_millis(10));
                }
                None => {
                    let _ = child.kill();
                    break (child.wait().expect("reap build-blob helper"), true);
                }
            }
        };
        let mut stdout = Vec::new();
        let mut stderr = Vec::new();
        child
            .stdout
            .take()
            .expect("helper stdout")
            .read_to_end(&mut stdout)
            .expect("read helper stdout");
        child
            .stderr
            .take()
            .expect("helper stderr")
            .read_to_end(&mut stderr)
            .expect("read helper stderr");
        assert!(
            !timed_out,
            "build-blob helper timed out\nstdout:\n{}\nstderr:\n{}",
            String::from_utf8_lossy(&stdout),
            String::from_utf8_lossy(&stderr)
        );
        assert_eq!(
            status.code(),
            Some(expected_code),
            "build-blob helper exited {status}\nstdout:\n{}\nstderr:\n{}",
            String::from_utf8_lossy(&stdout),
            String::from_utf8_lossy(&stderr)
        );
    }

    #[test]
    #[ignore = "invoked by cross-process build-blob cache tests"]
    fn build_blob_cache_process_helper() {
        use std::io::Write as _;

        let mode = std::env::var(BUILD_BLOB_HELPER_MODE).expect("helper mode");
        let root = std::path::PathBuf::from(
            std::env::var_os(BUILD_BLOB_HELPER_ROOT).expect("helper cache root"),
        );
        let gate = std::path::PathBuf::from(
            std::env::var_os(BUILD_BLOB_HELPER_GATE).expect("helper gate"),
        );
        let count = std::path::PathBuf::from(
            std::env::var_os(BUILD_BLOB_HELPER_COUNT).expect("helper count"),
        );
        let deadline = Instant::now() + std::time::Duration::from_secs(10);
        while !gate.exists() {
            assert!(Instant::now() < deadline, "parent did not open helper gate");
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        ensure_cached_build_blob(
            &root,
            BUILD_BLOB_HELPER_PARTS,
            "build-blob process fixture",
            "fixture",
            |stage| {
                let mut builders = std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(&count)
                    .map_err(|error| error.to_string())?;
                writeln!(builders, "{mode}:{}", std::process::id())
                    .map_err(|error| error.to_string())?;
                builders.sync_all().map_err(|error| error.to_string())?;
                match mode.as_str() {
                    "waiter" | "takeover" => {
                        std::thread::sleep(std::time::Duration::from_millis(150));
                        std::fs::write(stage.join("fixture"), b"published-build-blob")
                            .map_err(|error| error.to_string())
                    }
                    "crash" => {
                        std::fs::write(stage.join("partial"), b"unpublished")
                            .map_err(|error| error.to_string())?;
                        std::process::exit(97);
                    }
                    other => Err(format!("unknown helper mode {other}")),
                }
            },
        )
        .expect("helper obtains cached build blob");
    }

    #[test]
    fn build_blob_cache_elects_one_builder_across_processes() {
        let temp = tempfile::tempdir().expect("tempdir");
        let root = temp.path().join("cache");
        let gate = temp.path().join("gate");
        let count = temp.path().join("builders");
        let children: Vec<_> = (0..8)
            .map(|_| spawn_build_blob_helper(&root, &gate, &count, "waiter"))
            .collect();
        std::fs::write(&gate, b"go").expect("open helper gate");
        for child in children {
            wait_build_blob_helper(child, 0);
        }
        let builders = std::fs::read_to_string(&count).expect("read builder record");
        assert_eq!(
            builders.lines().count(),
            1,
            "only one process may perform expensive work; records:\n{builders}"
        );
        let entry = root.join(build_blob_content_id(BUILD_BLOB_HELPER_PARTS));
        assert_eq!(
            std::fs::read(entry.join("fixture")).expect("read shared result"),
            b"published-build-blob"
        );
    }

    #[test]
    fn build_blob_cache_recovers_after_crashed_builder() {
        let temp = tempfile::tempdir().expect("tempdir");
        let root = temp.path().join("cache");
        let gate = temp.path().join("gate");
        let count = temp.path().join("builders");
        let crashed = spawn_build_blob_helper(&root, &gate, &count, "crash");
        std::fs::write(&gate, b"go").expect("open helper gate");
        wait_build_blob_helper(crashed, 97);

        let id = build_blob_content_id(BUILD_BLOB_HELPER_PARTS);
        let stage_prefix = format!(".{id}.work-");
        assert!(
            std::fs::read_dir(&root)
                .expect("scan crash-leftover cache")
                .any(|entry| entry
                    .expect("cache entry")
                    .file_name()
                    .to_string_lossy()
                    .starts_with(&stage_prefix)),
            "a process exit must model a realistic unpublished stage"
        );

        let takeover = spawn_build_blob_helper(&root, &gate, &count, "takeover");
        wait_build_blob_helper(takeover, 0);
        assert_eq!(
            std::fs::read(root.join(id).join("fixture")).expect("read takeover result"),
            b"published-build-blob"
        );
        assert!(
            std::fs::read_dir(&root)
                .expect("scan recovered cache")
                .all(|entry| !entry
                    .expect("cache entry")
                    .file_name()
                    .to_string_lossy()
                    .starts_with(&stage_prefix)),
            "the takeover must remove crash-leftover stages"
        );
    }

    #[test]
    fn build_blob_cache_rebuilds_deleted_incomplete_and_corrupt_entries() {
        let temp = tempfile::tempdir().expect("tempdir");
        let root = temp.path().join("cache");
        let parts = ["recipe", "revision", "aarch64", "toolchain"];
        let entry = root.join(build_blob_content_id(&parts));
        let builds = std::cell::Cell::new(0usize);
        let build = |stage: &std::path::Path| {
            builds.set(builds.get() + 1);
            std::fs::write(stage.join("artifact"), b"correct-content")
                .map_err(|error| error.to_string())
        };

        ensure_cached_build_blob(&root, &parts, "fixture", "artifact", build)
            .expect("initial build");
        std::fs::remove_dir_all(&entry).expect("delete complete cache entry");
        ensure_cached_build_blob(&root, &parts, "fixture", "artifact", build)
            .expect("rebuild deleted entry");

        std::fs::remove_file(entry.join(BUILD_BLOB_OUTPUT_SENTINEL))
            .expect("make entry incomplete");
        ensure_cached_build_blob(&root, &parts, "fixture", "artifact", build)
            .expect("rebuild incomplete entry");

        std::fs::write(entry.join("artifact"), b"corrupt-content")
            .expect("corrupt cached artifact at same length");
        ensure_cached_build_blob(&root, &parts, "fixture", "artifact", build)
            .expect("rebuild corrupt entry");
        assert_eq!(builds.get(), 4, "each invalid cache state elects a rebuild");
        assert_eq!(
            std::fs::read(entry.join("artifact")).expect("read repaired artifact"),
            b"correct-content"
        );
    }

    #[test]
    fn materialized_build_blob_heals_deleted_cache_and_local_corruption() {
        let temp = tempfile::tempdir().expect("tempdir");
        let root = temp.path().join("cache");
        let destination = temp.path().join("out-artifact");
        let stamp = temp.path().join("out-artifact.stamp");
        let parts = ["recipe", "revision", "x86_64", "exact-tools"];
        let entry = root.join(build_blob_content_id(&parts));
        let builds = std::cell::Cell::new(0usize);
        let build = |stage: &std::path::Path| {
            builds.set(builds.get() + 1);
            std::fs::write(stage.join("artifact"), b"correct-content")
                .map_err(|error| error.to_string())
        };

        ensure_materialized_build_blob(
            &root,
            &parts,
            "materialized fixture",
            "artifact",
            &destination,
            &stamp,
            build,
        )
        .expect("initial build and materialization");
        assert_eq!(builds.get(), 1);
        assert!(
            std::fs::read_to_string(&stamp)
                .expect("read content-bound stamp")
                .starts_with(BUILT_BLOB_STAMP_SCHEMA),
            "the local fast path must carry output content, not only an input id"
        );

        std::fs::remove_dir_all(&entry).expect("delete shared CAS entry");
        ensure_materialized_build_blob(
            &root,
            &parts,
            "materialized fixture",
            "artifact",
            &destination,
            &stamp,
            build,
        )
        .expect("reseed deleted CAS from validated local COW inode");
        assert_eq!(builds.get(), 1, "CAS deletion must not repeat source work");
        assert_eq!(
            std::fs::read(entry.join("artifact")).expect("read reseeded CAS"),
            b"correct-content"
        );

        let corrupt = temp.path().join("corrupt-local");
        std::fs::write(&corrupt, b"corrupt-content").expect("write corrupt replacement");
        std::fs::rename(&corrupt, &destination).expect("replace local inode with corruption");
        ensure_materialized_build_blob(
            &root,
            &parts,
            "materialized fixture",
            "artifact",
            &destination,
            &stamp,
            build,
        )
        .expect("repair corrupt local output from intact CAS");
        assert_eq!(builds.get(), 1, "local corruption must reuse an intact CAS");
        assert_eq!(
            std::fs::read(&destination).expect("read repaired local output"),
            b"correct-content"
        );

        std::fs::write(entry.join("artifact"), b"corrupt-content").expect("corrupt shared CAS");
        ensure_materialized_build_blob(
            &root,
            &parts,
            "materialized fixture",
            "artifact",
            &destination,
            &stamp,
            build,
        )
        .expect("repair corrupt CAS from intact local output");
        assert_eq!(
            builds.get(),
            1,
            "an intact local copy must reseed corrupt CAS"
        );

        let corrupt = temp.path().join("corrupt-local-again");
        std::fs::write(&corrupt, b"corrupt-content").expect("write second corruption");
        std::fs::rename(&corrupt, &destination).expect("replace local output again");
        std::fs::remove_dir_all(&entry).expect("delete CAS while local is corrupt");
        ensure_materialized_build_blob(
            &root,
            &parts,
            "materialized fixture",
            "artifact",
            &destination,
            &stamp,
            build,
        )
        .expect("rebuild after both copies are lost");
        assert_eq!(
            builds.get(),
            2,
            "loss of both validated copies must rebuild"
        );
    }

    #[test]
    fn build_blob_identity_separates_source_revision_arch_recipe_and_toolchain() {
        let base =
            build_blob_content_id(&["recipe-v1", "rev-a", "source-sha-a", "x86_64", "gcc-a"]);
        for changed in [
            ["recipe-v2", "rev-a", "source-sha-a", "x86_64", "gcc-a"],
            ["recipe-v1", "rev-b", "source-sha-a", "x86_64", "gcc-a"],
            ["recipe-v1", "rev-a", "source-sha-b", "x86_64", "gcc-a"],
            ["recipe-v1", "rev-a", "source-sha-a", "aarch64", "gcc-a"],
            ["recipe-v1", "rev-a", "source-sha-a", "x86_64", "gcc-b"],
        ] {
            assert_ne!(base, build_blob_content_id(&changed));
        }
    }

    #[test]
    fn build_blob_materialization_never_byte_copies_on_ficlone_failure() {
        let temp = tempfile::tempdir().expect("tempdir");
        let source = temp.path().join("source");
        let destination = temp.path().join("destination");
        std::fs::write(&source, b"new-cached-content").expect("write source");
        std::fs::write(&destination, b"old-destination").expect("write old destination");
        let error = materialize_build_blob_with(&source, &destination, |_destination, _source| {
            Err(std::io::Error::from_raw_os_error(libc::EOPNOTSUPP))
        })
        .expect_err("unsupported FICLONE must fail rather than copying bytes");
        assert!(error.to_string().contains("FICLONE"));
        assert_eq!(
            std::fs::read(&destination).expect("read preserved destination"),
            b"old-destination",
            "atomic failure must preserve the previous fixed-path blob"
        );
        assert_eq!(
            std::fs::read_dir(temp.path())
                .expect("scan materialization directory")
                .count(),
            2,
            "failed clone temporary must be removed"
        );
    }

    #[test]
    fn build_blob_materialization_uses_real_cross_inode_cow() {
        let temp = tempfile::tempdir().expect("tempdir");
        let source = temp.path().join("source");
        let destination = temp.path().join("destination");
        std::fs::write(&source, b"direct-ficlone-content").expect("write source");
        materialize_build_blob(&source, &destination).expect("direct FICLONE materialization");
        assert_eq!(
            std::fs::read(&destination).expect("read materialized blob"),
            b"direct-ficlone-content"
        );
        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt as _;
            assert_ne!(
                std::fs::metadata(&source).expect("source metadata").ino(),
                std::fs::metadata(&destination)
                    .expect("destination metadata")
                    .ino(),
                "materialization must create a private COW inode, not a hard link"
            );
        }
    }

    #[test]
    fn install_prebuilt_blob_cow_materializes_nonempty_source() {
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
