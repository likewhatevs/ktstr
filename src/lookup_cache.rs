//! Persistent last-known-good cache for typed metadata lookups.
//!
//! Metadata discovery is deliberately refresh-first: a successful, validated
//! lookup is always returned even when the cache root cannot be resolved or
//! written. When refresh fails, a previously validated typed value may be used
//! without a TTL. The stale value remains useful until the caller's validator
//! rejects it, while the warning records both its age and the live refresh
//! failure.
//!
//! Each logical key maps to one bounded JSON envelope under
//! `<CacheDir::default_root()>/.metadata-lookups`. The filename is a
//! fixed-seed ahash of the schema version plus the exact logical key; the exact
//! key is also embedded in the envelope, so a hash collision is a rejected
//! cache entry rather than an alias. Writers publish a synced unique temporary
//! through an atomic rename. Concurrent readers therefore observe either the
//! previous complete envelope or the next complete envelope, never a torn
//! intermediate write.

use std::fs::{self, File};
use std::hash::{BuildHasher, Hasher};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::cache::CacheDir;

const LOOKUP_DIR_NAME: &str = ".metadata-lookups";
const ENVELOPE_SCHEMA: u32 = 1;
const ENVELOPE_DOMAIN: &[u8] = b"ktstr-metadata-lookup-envelope";
const FILENAME_DOMAIN: &[u8] = b"ktstr-metadata-lookup-filename";

/// Hard upper bound for one complete persisted envelope.
///
/// Lookup results are metadata, not downloaded artifacts. Sixteen MiB leaves
/// ample room for package/version catalogs while bounding allocation and parse
/// work on corrupt or hostile cache files. Both producer and consumer enforce
/// the same limit.
const MAX_ENVELOPE_BYTES: usize = 16 * 1024 * 1024;

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Envelope {
    schema: u32,
    logical_key: String,
    fetched_unix_secs: u64,
    payload: Value,
    integrity_ahash: String,
}

#[derive(Debug)]
struct Cached<T> {
    value: T,
    fetched_unix_secs: u64,
}

/// Refresh a typed metadata lookup and retain its last-known-good value.
///
/// `key` must be a stable, exact, namespaced description of every input that
/// affects the result, including a caller-owned semantic version. A typical key
/// looks like `distro-resolution-v1/ubuntu/latest/aarch64`.
///
/// A refreshed value and a cached value both have to pass `validate`. Only a
/// refreshed value that passes validation is published. Cache resolution,
/// serialization, and persistence errors never turn a valid fresh result into
/// a failure.
///
/// When refresh (including refreshed-value validation) fails, an intact cached
/// value is returned regardless of age. Process interruption suppresses that
/// fallback so Ctrl-C/SIGTERM cannot appear to complete successfully from stale
/// state.
pub(crate) fn last_known_good<T, F, V>(key: &str, refresh: F, validate: V) -> Result<T>
where
    T: Serialize + DeserializeOwned,
    F: FnOnce() -> Result<T>,
    V: Fn(&T) -> Result<()>,
{
    last_known_good_with(
        key,
        refresh,
        validate,
        CacheDir::default_root,
        system_unix_secs,
        crate::fetch::git_operation_interrupted,
    )
}

fn last_known_good_with<T, F, V, R, N, I>(
    key: &str,
    refresh: F,
    validate: V,
    cache_root: R,
    now_unix_secs: N,
    interrupted: I,
) -> Result<T>
where
    T: Serialize + DeserializeOwned,
    F: FnOnce() -> Result<T>,
    V: Fn(&T) -> Result<()>,
    R: Fn() -> Result<PathBuf>,
    N: Fn() -> u64,
    I: Fn() -> bool,
{
    let refreshed = refresh().and_then(|value| {
        validate(&value)?;
        Ok(value)
    });

    match refreshed {
        Ok(value) => {
            let cache_result = cache_root().and_then(|root| {
                store_cached(&root, key, now_unix_secs(), &value)
                    .with_context(|| format!("publish metadata lookup cache for {key:?}"))
            });
            if let Err(cache_error) = cache_result {
                tracing::warn!(
                    logical_key = key,
                    error = %cache_error,
                    "metadata lookup succeeded but its last-known-good cache could not be updated",
                );
            }
            Ok(value)
        }
        Err(refresh_error) => {
            if interrupted() {
                return Err(refresh_error);
            }

            let cached = cache_root()
                .and_then(|root| read_cached::<T>(&root, key))
                .and_then(|cached| {
                    validate(&cached.value)?;
                    Ok(cached)
                });
            let cached = match cached {
                Ok(cached) => cached,
                Err(cache_error) => {
                    tracing::warn!(
                        logical_key = key,
                        refresh_error = %refresh_error,
                        cache_error = %cache_error,
                        "metadata lookup refresh failed and no valid last-known-good value was available",
                    );
                    return Err(refresh_error);
                }
            };

            // A signal may arrive while the envelope is being read and
            // validated. Recheck immediately before converting the failed
            // operation into a stale success.
            if interrupted() {
                return Err(refresh_error);
            }

            let now = now_unix_secs();
            let age_secs = now.saturating_sub(cached.fetched_unix_secs);
            let clock_skew_secs = cached.fetched_unix_secs.saturating_sub(now);
            if clock_skew_secs == 0 {
                tracing::warn!(
                    logical_key = key,
                    stale_age = %humantime::format_duration(Duration::from_secs(age_secs)),
                    stale_age_secs = age_secs,
                    refresh_error = %format!("{refresh_error:#}"),
                    "metadata lookup refresh failed; using last-known-good cached value",
                );
            } else {
                tracing::warn!(
                    logical_key = key,
                    stale_age = %humantime::format_duration(Duration::ZERO),
                    stale_age_secs = 0,
                    stale_clock_skew_secs = clock_skew_secs,
                    refresh_error = %format!("{refresh_error:#}"),
                    "metadata lookup refresh failed; using last-known-good cached value whose \
                     timestamp is ahead of the local clock",
                );
            }
            Ok(cached.value)
        }
    }
}

fn store_cached<T: Serialize>(
    cache_root: &Path,
    key: &str,
    fetched_unix_secs: u64,
    value: &T,
) -> Result<()> {
    let payload = serde_json::to_value(value).context("serialize typed lookup payload")?;
    let envelope = Envelope {
        schema: ENVELOPE_SCHEMA,
        logical_key: key.to_owned(),
        fetched_unix_secs,
        integrity_ahash: envelope_integrity(ENVELOPE_SCHEMA, key, fetched_unix_secs, &payload)?,
        payload,
    };
    let bytes = serde_json::to_vec(&envelope).context("serialize lookup cache envelope")?;
    anyhow::ensure!(
        bytes.len() <= MAX_ENVELOPE_BYTES,
        "lookup cache envelope is {} bytes, exceeding the {} byte limit",
        bytes.len(),
        MAX_ENVELOPE_BYTES,
    );

    let dir = lookup_dir(cache_root);
    fs::create_dir_all(&dir)
        .with_context(|| format!("create metadata lookup cache directory {}", dir.display()))?;
    // Persist creation of the hidden lookup directory itself before publishing
    // a child. This is redundant after the first write and cheap compared with
    // the network lookup that led here.
    File::open(cache_root)
        .and_then(|directory| directory.sync_all())
        .with_context(|| format!("sync metadata lookup cache root {}", cache_root.display()))?;

    if let Err(error) = sweep_orphaned_temporary_files(&dir) {
        tracing::warn!(
            path = %dir.display(),
            error = %format!("{error:#}"),
            "failed to sweep crashed metadata lookup writers; continuing cache publish",
        );
    }

    let final_path = cache_file_path(cache_root, key);
    let filename_hash = filename_hash(key);
    let mut temporary = tempfile::Builder::new()
        .prefix(&format!(
            ".v{ENVELOPE_SCHEMA}-{filename_hash:016x}.tmp-{}-",
            std::process::id(),
        ))
        .suffix(".json")
        .tempfile_in(&dir)
        .with_context(|| {
            format!(
                "create unique metadata lookup cache temporary in {}",
                dir.display()
            )
        })?;
    temporary
        .write_all(&bytes)
        .context("write metadata lookup cache temporary")?;
    temporary
        .as_file()
        .sync_all()
        .context("sync metadata lookup cache temporary")?;
    temporary
        .persist(&final_path)
        .map_err(|error| error.error)
        .with_context(|| {
            format!(
                "atomically publish metadata lookup cache {}",
                final_path.display()
            )
        })?;
    File::open(&dir)
        .and_then(|directory| directory.sync_all())
        .with_context(|| format!("sync metadata lookup cache directory {}", dir.display()))?;
    Ok(())
}

/// Remove temporary envelopes whose writer PID is verifiably gone.
///
/// Normal error paths are covered by `NamedTempFile`'s drop cleanup. This
/// sweep handles SIGKILL and host-crash debris without racing a live writer:
/// only `kill(pid, 0) == ESRCH` authorizes unlinking. EPERM and every other
/// result preserve the file. PID reuse can delay cleanup, but can never remove
/// a file owned by a process that is still publishing it.
fn sweep_orphaned_temporary_files(dir: &Path) -> Result<()> {
    let entries = match fs::read_dir(dir) {
        Ok(entries) => entries,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(error) => {
            return Err(error)
                .with_context(|| format!("read metadata lookup cache {}", dir.display()));
        }
    };
    for entry in entries {
        let entry = match entry {
            Ok(entry) => entry,
            Err(error) => {
                tracing::warn!(
                    path = %dir.display(),
                    error = %error,
                    "skip unreadable metadata lookup cache entry during orphan sweep",
                );
                continue;
            }
        };
        let Some(name) = entry.file_name().to_str().map(str::to_owned) else {
            continue;
        };
        let Some(pid) = temporary_owner_pid(&name) else {
            continue;
        };
        let dead = matches!(
            nix::sys::signal::kill(nix::unistd::Pid::from_raw(pid), None),
            Err(nix::errno::Errno::ESRCH),
        );
        if !dead {
            continue;
        }
        let path = entry.path();
        match fs::remove_file(&path) {
            Ok(()) => tracing::info!(
                path = %path.display(),
                orphan_pid = pid,
                "removed metadata lookup temporary left by a crashed writer",
            ),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => tracing::warn!(
                path = %path.display(),
                orphan_pid = pid,
                error = %error,
                "failed to remove orphaned metadata lookup temporary",
            ),
        }
    }
    Ok(())
}

fn temporary_owner_pid(name: &str) -> Option<i32> {
    let versioned = name.strip_prefix(&format!(".v{ENVELOPE_SCHEMA}-"))?;
    let (hash, owner_and_random) = versioned.split_once(".tmp-")?;
    if hash.len() != 16 || !hash.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return None;
    }
    let (pid, random) = owner_and_random.split_once('-')?;
    if random.is_empty() || !random.ends_with(".json") {
        return None;
    }
    let pid = pid.parse::<i32>().ok()?;
    (pid > 0).then_some(pid)
}

fn read_cached<T: DeserializeOwned>(cache_root: &Path, key: &str) -> Result<Cached<T>> {
    let path = cache_file_path(cache_root, key);
    let file = File::open(&path)
        .with_context(|| format!("open metadata lookup cache {}", path.display()))?;
    let metadata = file
        .metadata()
        .with_context(|| format!("stat metadata lookup cache {}", path.display()))?;
    anyhow::ensure!(
        metadata.len() <= MAX_ENVELOPE_BYTES as u64,
        "metadata lookup cache {} is {} bytes, exceeding the {} byte limit",
        path.display(),
        metadata.len(),
        MAX_ENVELOPE_BYTES,
    );

    // `take(limit + 1)` retains the bound even if a concurrently malicious
    // writer grows the already-open inode after the metadata check. Normal
    // writers never mutate a published inode; they rename a new one over it.
    let mut bytes = Vec::with_capacity(metadata.len() as usize);
    file.take(MAX_ENVELOPE_BYTES as u64 + 1)
        .read_to_end(&mut bytes)
        .with_context(|| format!("read metadata lookup cache {}", path.display()))?;
    anyhow::ensure!(
        bytes.len() <= MAX_ENVELOPE_BYTES,
        "metadata lookup cache {} grew beyond the {} byte limit while reading",
        path.display(),
        MAX_ENVELOPE_BYTES,
    );

    let envelope: Envelope = serde_json::from_slice(&bytes)
        .with_context(|| format!("parse metadata lookup cache {}", path.display()))?;
    validate_envelope(&envelope, key)?;
    let value = serde_json::from_value(envelope.payload)
        .context("deserialize typed cached lookup value")?;
    Ok(Cached {
        value,
        fetched_unix_secs: envelope.fetched_unix_secs,
    })
}

fn validate_envelope(envelope: &Envelope, expected_key: &str) -> Result<()> {
    anyhow::ensure!(
        envelope.schema == ENVELOPE_SCHEMA,
        "metadata lookup cache schema {} is unsupported (expected {})",
        envelope.schema,
        ENVELOPE_SCHEMA,
    );
    anyhow::ensure!(
        envelope.logical_key == expected_key,
        "metadata lookup cache logical key mismatch: found {:?}, expected {:?}",
        envelope.logical_key,
        expected_key,
    );
    let expected_integrity = envelope_integrity(
        envelope.schema,
        &envelope.logical_key,
        envelope.fetched_unix_secs,
        &envelope.payload,
    )?;
    anyhow::ensure!(
        envelope.integrity_ahash == expected_integrity,
        "metadata lookup cache integrity mismatch",
    );
    Ok(())
}

fn envelope_integrity(
    schema: u32,
    key: &str,
    fetched_unix_secs: u64,
    payload: &Value,
) -> Result<String> {
    let payload_bytes =
        serde_json::to_vec(payload).context("serialize payload for lookup cache integrity")?;
    let mut hasher = fixed_hasher();
    write_len_prefixed(&mut hasher, ENVELOPE_DOMAIN);
    hasher.write(&schema.to_le_bytes());
    write_len_prefixed(&mut hasher, key.as_bytes());
    hasher.write(&fetched_unix_secs.to_le_bytes());
    write_len_prefixed(&mut hasher, &payload_bytes);
    Ok(format!("{:016x}", hasher.finish()))
}

fn filename_hash(key: &str) -> u64 {
    let mut hasher = fixed_hasher();
    write_len_prefixed(&mut hasher, FILENAME_DOMAIN);
    hasher.write(&ENVELOPE_SCHEMA.to_le_bytes());
    write_len_prefixed(&mut hasher, key.as_bytes());
    hasher.finish()
}

fn fixed_hasher() -> ahash::AHasher {
    ahash::RandomState::with_seeds(0, 0, 0, 0).build_hasher()
}

fn write_len_prefixed(hasher: &mut impl Hasher, bytes: &[u8]) {
    hasher.write(&(bytes.len() as u64).to_le_bytes());
    hasher.write(bytes);
}

fn lookup_dir(cache_root: &Path) -> PathBuf {
    cache_root.join(LOOKUP_DIR_NAME)
}

fn cache_file_path(cache_root: &Path, key: &str) -> PathBuf {
    lookup_dir(cache_root).join(format!(
        "v{ENVELOPE_SCHEMA}-{:016x}.json",
        filename_hash(key)
    ))
}

fn system_unix_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or(Duration::ZERO)
        .as_secs()
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Barrier};
    use std::thread;

    use anyhow::anyhow;
    use serde::ser::Error as _;
    use tempfile::TempDir;

    use super::*;

    #[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
    struct Fixture {
        generation: u64,
        body: String,
    }

    fn fixture(generation: u64) -> Fixture {
        Fixture {
            generation,
            body: format!("metadata-generation-{generation}"),
        }
    }

    fn valid(value: &Fixture) -> Result<()> {
        anyhow::ensure!(!value.body.is_empty(), "fixture body is empty");
        Ok(())
    }

    fn root_fn(path: &Path) -> impl Fn() -> Result<PathBuf> + '_ {
        move || Ok(path.to_owned())
    }

    fn seed(root: &Path, key: &str, fetched: u64, value: &Fixture) {
        store_cached(root, key, fetched, value).expect("seed last-known-good cache");
    }

    #[test]
    fn fresh_valid_value_is_returned_and_published_under_hidden_root() {
        let temp = TempDir::new().unwrap();
        let expected = fixture(7);
        let actual = last_known_good_with(
            "fixture/v1/arm64",
            || Ok(expected.clone()),
            valid,
            root_fn(temp.path()),
            || 1234,
            || false,
        )
        .unwrap();
        assert_eq!(actual, expected);

        let path = cache_file_path(temp.path(), "fixture/v1/arm64");
        assert_eq!(
            path.parent(),
            Some(temp.path().join(LOOKUP_DIR_NAME).as_path())
        );
        let cached: Cached<Fixture> = read_cached(temp.path(), "fixture/v1/arm64").unwrap();
        assert_eq!(cached.value, expected);
        assert_eq!(cached.fetched_unix_secs, 1234);
    }

    #[test]
    fn fresh_success_survives_root_resolution_and_serialization_failures() {
        #[derive(Debug, PartialEq, Deserialize)]
        struct FailsSerialization {
            generation: u64,
        }

        impl Serialize for FailsSerialization {
            fn serialize<S>(&self, _serializer: S) -> std::result::Result<S::Ok, S::Error>
            where
                S: serde::Serializer,
            {
                Err(S::Error::custom("intentional serializer failure"))
            }
        }

        let value = last_known_good_with(
            "fixture/v1/root-error",
            || Ok(FailsSerialization { generation: 11 }),
            |_| Ok(()),
            || Err(anyhow!("cache root unavailable")),
            || 1,
            || false,
        )
        .unwrap();
        assert_eq!(value.generation, 11);

        let temp = TempDir::new().unwrap();
        let value = last_known_good_with(
            "fixture/v1/serialization-error",
            || Ok(FailsSerialization { generation: 12 }),
            |_| Ok(()),
            root_fn(temp.path()),
            || 1,
            || false,
        )
        .unwrap();
        assert_eq!(value.generation, 12);
        assert!(!cache_file_path(temp.path(), "fixture/v1/serialization-error").exists());
    }

    #[test]
    #[tracing_test::traced_test]
    fn failed_refresh_uses_arbitrarily_old_valid_cache_and_warns_with_age_and_error() {
        let temp = TempDir::new().unwrap();
        let key = "fixture/v1/stale";
        seed(temp.path(), key, 100, &fixture(3));

        let actual = last_known_good_with(
            key,
            || Err(anyhow!("upstream returned 503")),
            valid,
            root_fn(temp.path()),
            || 10_100,
            || false,
        )
        .unwrap();
        assert_eq!(actual, fixture(3));
        assert!(logs_contain("using last-known-good cached value"));
        assert!(logs_contain("stale_age_secs=10000"));
        assert!(logs_contain("upstream returned 503"));
    }

    #[test]
    #[tracing_test::traced_test]
    fn future_cached_timestamp_reports_clock_skew_instead_of_false_age() {
        let temp = TempDir::new().unwrap();
        let key = "fixture/v1/future-timestamp";
        seed(temp.path(), key, 150, &fixture(3));

        let actual = last_known_good_with(
            key,
            || Err(anyhow!("upstream unavailable")),
            valid,
            root_fn(temp.path()),
            || 100,
            || false,
        )
        .unwrap();
        assert_eq!(actual, fixture(3));
        assert!(logs_contain("timestamp is ahead of the local clock"));
        assert!(logs_contain("stale_clock_skew_secs=50"));
        assert!(logs_contain("stale_age_secs=0"));
    }

    #[test]
    fn failed_refresh_never_falls_back_during_either_interruption_check() {
        let temp = TempDir::new().unwrap();
        let key = "fixture/v1/interrupted";
        seed(temp.path(), key, 100, &fixture(3));

        let err = last_known_good_with(
            key,
            || Err(anyhow!("refresh interrupted")),
            valid,
            root_fn(temp.path()),
            || 200,
            || true,
        )
        .unwrap_err();
        assert_eq!(err.to_string(), "refresh interrupted");

        let checks = std::cell::Cell::new(0);
        let err = last_known_good_with(
            key,
            || Err(anyhow!("signal raced with cache read")),
            valid,
            root_fn(temp.path()),
            || 200,
            || {
                checks.set(checks.get() + 1);
                checks.get() == 2
            },
        )
        .unwrap_err();
        assert_eq!(err.to_string(), "signal raced with cache read");
        assert_eq!(checks.get(), 2);
    }

    #[test]
    fn invalid_refreshed_value_is_not_published_and_cannot_replace_good_cache() {
        let temp = TempDir::new().unwrap();
        let key = "fixture/v1/validation";
        seed(temp.path(), key, 10, &fixture(1));
        let before = fs::read(cache_file_path(temp.path(), key)).unwrap();

        let actual = last_known_good_with(
            key,
            || {
                Ok(Fixture {
                    generation: 2,
                    body: String::new(),
                })
            },
            valid,
            root_fn(temp.path()),
            || 20,
            || false,
        )
        .unwrap();
        assert_eq!(actual, fixture(1));
        assert_eq!(
            fs::read(cache_file_path(temp.path(), key)).unwrap(),
            before,
            "failed validation must not overwrite the validated envelope",
        );
    }

    #[test]
    fn cached_typed_value_must_pass_the_current_validator() {
        let temp = TempDir::new().unwrap();
        let key = "fixture/v1/current-validator";
        seed(temp.path(), key, 10, &fixture(1));

        let err = last_known_good_with(
            key,
            || Err(anyhow!("refresh sentinel")),
            |value: &Fixture| {
                anyhow::ensure!(value.generation >= 2, "generation is obsolete");
                Ok(())
            },
            root_fn(temp.path()),
            || 20,
            || false,
        )
        .unwrap_err();
        assert_eq!(err.to_string(), "refresh sentinel");
    }

    fn write_envelope(root: &Path, requested_key: &str, envelope: &Envelope) {
        let dir = lookup_dir(root);
        fs::create_dir_all(&dir).unwrap();
        fs::write(
            cache_file_path(root, requested_key),
            serde_json::to_vec(envelope).unwrap(),
        )
        .unwrap();
    }

    fn envelope(key: &str, schema: u32, fetched: u64, payload: Value) -> Envelope {
        Envelope {
            schema,
            logical_key: key.to_owned(),
            fetched_unix_secs: fetched,
            integrity_ahash: envelope_integrity(schema, key, fetched, &payload).unwrap(),
            payload,
        }
    }

    fn assert_original_refresh_error(root: &Path, key: &str) {
        let err = last_known_good_with::<Fixture, _, _, _, _, _>(
            key,
            || Err(anyhow!("refresh root cause").context("refresh sentinel")),
            valid,
            root_fn(root),
            || 999,
            || false,
        )
        .unwrap_err();
        assert_eq!(
            format!("{err:#}"),
            "refresh sentinel: refresh root cause",
            "cache rejection must return the original refresh error and chain",
        );
    }

    #[test]
    fn malformed_wrong_schema_wrong_key_and_bad_integrity_preserve_refresh_error() {
        let key = "fixture/v1/rejection";

        let malformed = TempDir::new().unwrap();
        fs::create_dir_all(lookup_dir(malformed.path())).unwrap();
        fs::write(cache_file_path(malformed.path(), key), b"{").unwrap();
        assert_original_refresh_error(malformed.path(), key);

        let wrong_schema = TempDir::new().unwrap();
        let wrong = envelope(
            key,
            ENVELOPE_SCHEMA + 1,
            10,
            serde_json::to_value(fixture(1)).unwrap(),
        );
        write_envelope(wrong_schema.path(), key, &wrong);
        assert_original_refresh_error(wrong_schema.path(), key);

        let wrong_key = TempDir::new().unwrap();
        let wrong = envelope(
            "fixture/v1/different-exact-key",
            ENVELOPE_SCHEMA,
            10,
            serde_json::to_value(fixture(1)).unwrap(),
        );
        write_envelope(wrong_key.path(), key, &wrong);
        assert_original_refresh_error(wrong_key.path(), key);

        let bad_hash = TempDir::new().unwrap();
        let mut wrong = envelope(
            key,
            ENVELOPE_SCHEMA,
            10,
            serde_json::to_value(fixture(1)).unwrap(),
        );
        wrong.integrity_ahash = "0000000000000000".to_owned();
        write_envelope(bad_hash.path(), key, &wrong);
        assert_original_refresh_error(bad_hash.path(), key);
    }

    #[test]
    fn oversized_and_wrong_typed_payloads_preserve_refresh_error() {
        let key = "fixture/v1/bounds";
        let oversized = TempDir::new().unwrap();
        fs::create_dir_all(lookup_dir(oversized.path())).unwrap();
        fs::write(
            cache_file_path(oversized.path(), key),
            vec![b' '; MAX_ENVELOPE_BYTES + 1],
        )
        .unwrap();
        assert_original_refresh_error(oversized.path(), key);

        let wrong_type = TempDir::new().unwrap();
        let wrong = envelope(
            key,
            ENVELOPE_SCHEMA,
            10,
            Value::String("not a Fixture".to_owned()),
        );
        write_envelope(wrong_type.path(), key, &wrong);
        assert_original_refresh_error(wrong_type.path(), key);
    }

    #[test]
    fn filename_is_versioned_fixed_hash_and_never_contains_untrusted_key_text() {
        let root = Path::new("/cache-root");
        let key = "../../../noble/arm64?query=yes";
        let path_a = cache_file_path(root, key);
        let path_b = cache_file_path(root, key);
        assert_eq!(path_a, path_b);
        assert_eq!(path_a.parent(), Some(root.join(LOOKUP_DIR_NAME).as_path()));
        let name = path_a.file_name().unwrap().to_string_lossy();
        assert!(name.starts_with("v1-"));
        assert!(name.ends_with(".json"));
        assert!(!name.contains("noble"));
        assert_ne!(path_a, cache_file_path(root, "different-key"));
    }

    #[test]
    fn temporary_parser_and_sweep_remove_only_verifiably_dead_writers() {
        let temp = TempDir::new().unwrap();
        let dir = lookup_dir(temp.path());
        fs::create_dir_all(&dir).unwrap();
        let hash = format!("{:016x}", filename_hash("fixture/v1/orphans"));
        let dead_pid = libc::pid_t::MAX;
        let live_pid = std::process::id();
        let dead = dir.join(format!(
            ".v{ENVELOPE_SCHEMA}-{hash}.tmp-{dead_pid}-dead.json"
        ));
        let live = dir.join(format!(
            ".v{ENVELOPE_SCHEMA}-{hash}.tmp-{live_pid}-live.json"
        ));
        let unrelated = dir.join("operator-note.json");
        let malformed = dir.join(format!(
            ".v{ENVELOPE_SCHEMA}-{hash}.tmp-not-a-pid-random.json"
        ));
        for path in [&dead, &live, &unrelated, &malformed] {
            fs::write(path, b"fixture").unwrap();
        }

        assert_eq!(
            temporary_owner_pid(dead.file_name().unwrap().to_str().unwrap()),
            Some(dead_pid),
        );
        assert_eq!(
            temporary_owner_pid(live.file_name().unwrap().to_str().unwrap()),
            Some(live_pid as i32),
        );
        assert_eq!(temporary_owner_pid("operator-note.json"), None);
        assert_eq!(
            temporary_owner_pid(malformed.file_name().unwrap().to_str().unwrap()),
            None,
        );

        sweep_orphaned_temporary_files(&dir).unwrap();
        assert!(!dead.exists(), "dead writer's temporary must be reclaimed");
        assert!(live.exists(), "live writer's temporary must be preserved");
        assert!(
            unrelated.exists(),
            "unrelated cache files must be preserved"
        );
        assert!(
            malformed.exists(),
            "unowned malformed files must be preserved"
        );
    }

    #[test]
    fn concurrent_readers_and_writers_only_observe_complete_valid_envelopes() {
        let temp = TempDir::new().unwrap();
        let root = Arc::new(temp.path().to_owned());
        let key = "fixture/v1/concurrent";
        let large_body = "x".repeat(64 * 1024);
        let make_value = |generation| Fixture {
            generation,
            body: format!("{generation}:{large_body}"),
        };
        store_cached(root.as_path(), key, 1, &make_value(1)).unwrap();

        let barrier = Arc::new(Barrier::new(6));
        let mut threads = Vec::new();
        for writer_id in 0..2 {
            let root = Arc::clone(&root);
            let barrier = Arc::clone(&barrier);
            let large_body = large_body.clone();
            threads.push(thread::spawn(move || {
                barrier.wait();
                for iteration in 0..30 {
                    let generation = 2 + writer_id * 30 + iteration;
                    let value = Fixture {
                        generation,
                        body: format!("{generation}:{large_body}"),
                    };
                    store_cached(root.as_path(), key, generation, &value).unwrap();
                }
            }));
        }
        for _ in 0..4 {
            let root = Arc::clone(&root);
            let barrier = Arc::clone(&barrier);
            let large_body = large_body.clone();
            threads.push(thread::spawn(move || {
                barrier.wait();
                for _ in 0..100 {
                    let cached: Cached<Fixture> = read_cached(root.as_path(), key).unwrap();
                    assert_eq!(
                        cached.value.body,
                        format!("{}:{large_body}", cached.value.generation),
                        "reader observed a torn or cross-generation envelope",
                    );
                }
            }));
        }
        for thread in threads {
            thread.join().unwrap();
        }

        let temporary_files = fs::read_dir(lookup_dir(root.as_path()))
            .unwrap()
            .filter_map(std::result::Result::ok)
            .filter(|entry| entry.file_name().to_string_lossy().contains(".tmp-"))
            .count();
        assert_eq!(temporary_files, 0);
    }
}
