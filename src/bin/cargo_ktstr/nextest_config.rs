//! Low-priority nextest defaults shipped with `cargo-ktstr`.
//!
//! Nextest's `--tool-config-file TOOL:ABS_PATH` interface is specifically
//! intended for wrappers that need defaults without overriding repository or
//! CLI policy. The embedded config gives ktstr's resource-taking generated
//! cells a wide admission budget while bounding ordinary host tests to the
//! host CPU count.

use std::hash::{BuildHasher, Hasher};
use std::io::Write as _;
use std::path::{Path, PathBuf};

const TOOL_NAME: &str = "ktstr";
const TOOL_CONFIG_SCHEMA: u64 = 1;
const TOOL_CONFIG_DOMAIN: &[u8] = b"ktstr-nextest-tool-config";
const TOOL_CONFIG: &str = include_str!("nextest-tool.toml");

/// Add ktstr's low-priority tool config to a nextest argument vector.
///
/// A user-supplied ktstr tool config is authoritative and suppresses the
/// built-in one. Other tool configs are preserved ahead of the injected
/// argument because nextest gives earlier tool config arguments higher
/// priority. Insertion happens before the test-binary `--` separator.
pub(crate) fn inject(args: Vec<String>) -> Result<Vec<String>, String> {
    if has_ktstr_tool_config(&args) {
        return Ok(args);
    }
    let path = materialize()?;
    inject_with_path(args, &path)
}

/// Pure-path variant used by command-shape tests.
pub(crate) fn inject_with_path(mut args: Vec<String>, path: &Path) -> Result<Vec<String>, String> {
    if has_ktstr_tool_config(&args) {
        return Ok(args);
    }
    if !path.is_absolute() {
        return Err(format!(
            "cargo ktstr: nextest tool config path must be absolute: {}",
            path.display()
        ));
    }
    let path = path.to_str().ok_or_else(|| {
        format!(
            "cargo ktstr: nextest tool config path is not valid UTF-8: {}",
            path.display()
        )
    })?;
    let insertion = args
        .iter()
        .position(|argument| argument == "--")
        .unwrap_or(args.len());
    args.insert(insertion, format!("--tool-config-file={TOOL_NAME}:{path}"));
    Ok(args)
}

/// True when the user already supplied a ktstr-namespaced tool config.
///
/// Both clap spellings are recognized. Tokens after `--` belong to the test
/// binary and are deliberately opaque.
fn has_ktstr_tool_config(args: &[String]) -> bool {
    let mut index = 0;
    while index < args.len() {
        let argument = &args[index];
        if argument == "--" {
            return false;
        }
        if argument == "--tool-config-file" {
            if args
                .get(index + 1)
                .is_some_and(|value| tool_config_value_is_ktstr(value))
            {
                return true;
            }
            // A different tool's config does not settle the question:
            // nextest accepts this flag repeatedly, and a later occurrence
            // may already carry ktstr's namespace.
            index += 2;
            continue;
        }
        if let Some(value) = argument.strip_prefix("--tool-config-file=")
            && tool_config_value_is_ktstr(value)
        {
            return true;
        }
        index += 1;
    }
    false
}

fn tool_config_value_is_ktstr(value: &str) -> bool {
    value
        .split_once(':')
        .is_some_and(|(tool, _path)| tool == TOOL_NAME)
}

/// Materialize the embedded config at a versioned content-addressed cache path.
///
/// The fixed-seed ahash matches ktstr's fast on-disk CAS convention. The
/// schema and a length-delimited domain are included so a future semantic
/// format change cannot alias an older path even if the TOML bytes happen to
/// return to an earlier value.
fn materialize() -> Result<PathBuf, String> {
    let root = absolute_cache_root()?;
    materialize_in(&root)
}

fn materialize_in(root: &Path) -> Result<PathBuf, String> {
    std::fs::create_dir_all(root).map_err(|error| {
        format!(
            "cargo ktstr: create nextest tool-config cache {}: {error}",
            root.display()
        )
    })?;

    let target = materialized_path(root);
    if std::fs::read(&target).is_ok_and(|bytes| bytes == TOOL_CONFIG.as_bytes()) {
        return Ok(target);
    }

    let mut staging = tempfile::Builder::new()
        .prefix(".nextest-tool-staging-")
        .tempfile_in(root)
        .map_err(|error| {
            format!(
                "cargo ktstr: create nextest tool-config staging file in {}: {error}",
                root.display()
            )
        })?;
    staging
        .write_all(TOOL_CONFIG.as_bytes())
        .and_then(|()| staging.flush())
        .map_err(|error| {
            format!(
                "cargo ktstr: write nextest tool-config staging file in {}: {error}",
                root.display()
            )
        })?;
    staging.persist(&target).map_err(|error| {
        format!(
            "cargo ktstr: publish nextest tool config {}: {}",
            target.display(),
            error.error
        )
    })?;
    Ok(target)
}

fn materialized_path(root: &Path) -> PathBuf {
    let hash = config_hash(TOOL_CONFIG.as_bytes());
    root.join(format!(
        "nextest-tool-v{TOOL_CONFIG_SCHEMA}-{hash:016x}.toml"
    ))
}

fn absolute_cache_root() -> Result<PathBuf, String> {
    let root = ktstr::cache::CacheDir::default_root()
        .map_err(|error| format!("cargo ktstr: resolve cache for nextest tool config: {error:#}"))?
        .join(".nextest-tool-config");
    if root.is_absolute() {
        Ok(root)
    } else {
        std::env::current_dir()
            .map(|cwd| cwd.join(root))
            .map_err(|error| {
                format!("cargo ktstr: resolve absolute nextest tool-config cache path: {error}")
            })
    }
}

fn config_hash(bytes: &[u8]) -> u64 {
    let mut hasher = ahash::RandomState::with_seeds(0, 0, 0, 0).build_hasher();
    hash_bytes(&mut hasher, TOOL_CONFIG_DOMAIN);
    hasher.write(&TOOL_CONFIG_SCHEMA.to_le_bytes());
    hash_bytes(&mut hasher, bytes);
    hasher.finish()
}

fn hash_bytes(hasher: &mut impl Hasher, bytes: &[u8]) {
    // Fixed-width little-endian framing matches ktstr's other persisted
    // fixed-ahash identities and is stable across host word size/endian.
    hasher.write(&(bytes.len() as u64).to_le_bytes());
    hasher.write(bytes);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn strings(values: &[&str]) -> Vec<String> {
        values.iter().map(|value| (*value).to_owned()).collect()
    }

    #[test]
    fn injects_after_user_tool_configs_and_before_test_binary_separator() {
        let args = strings(&[
            "nextest",
            "run",
            "--tool-config-file=other:/tmp/other.toml",
            "-j",
            "77",
            "--",
            "--nocapture",
        ]);
        let got = inject_with_path(args, Path::new("/tmp/ktstr-tool.toml")).unwrap();
        assert_eq!(
            got,
            strings(&[
                "nextest",
                "run",
                "--tool-config-file=other:/tmp/other.toml",
                "-j",
                "77",
                "--tool-config-file=ktstr:/tmp/ktstr-tool.toml",
                "--",
                "--nocapture",
            ])
        );
    }

    #[test]
    fn user_ktstr_tool_config_suppresses_built_in_for_both_spellings() {
        for args in [
            strings(&[
                "nextest",
                "run",
                "--tool-config-file",
                "ktstr:/tmp/user.toml",
            ]),
            strings(&["nextest", "run", "--tool-config-file=ktstr:/tmp/user.toml"]),
        ] {
            let got = inject_with_path(args.clone(), Path::new("/tmp/builtin.toml")).unwrap();
            assert_eq!(got, args);
        }
    }

    #[test]
    fn separate_other_tool_config_does_not_hide_later_ktstr_config() {
        let args = strings(&[
            "nextest",
            "run",
            "--tool-config-file",
            "other:/tmp/other.toml",
            "--tool-config-file",
            "ktstr:/tmp/user.toml",
        ]);
        let got = inject_with_path(args.clone(), Path::new("/tmp/builtin.toml")).unwrap();
        assert_eq!(got, args);
    }

    #[test]
    fn explicit_test_threads_remains_a_higher_priority_cli_setting() {
        let args = strings(&["nextest", "run", "-j", "77"]);
        let got = inject_with_path(args, Path::new("/tmp/builtin.toml")).unwrap();
        assert_eq!(
            got,
            strings(&[
                "nextest",
                "run",
                "-j",
                "77",
                "--tool-config-file=ktstr:/tmp/builtin.toml",
            ]),
            "tool config injection must not consume, rewrite, or shadow an explicit CLI budget",
        );
    }

    #[test]
    fn test_binary_tool_config_looking_arg_is_opaque() {
        let args = strings(&[
            "nextest",
            "run",
            "--",
            "--tool-config-file=ktstr:/tmp/not-nextest.toml",
        ]);
        let got = inject_with_path(args, Path::new("/tmp/builtin.toml")).unwrap();
        assert_eq!(
            got,
            strings(&[
                "nextest",
                "run",
                "--tool-config-file=ktstr:/tmp/builtin.toml",
                "--",
                "--tool-config-file=ktstr:/tmp/not-nextest.toml",
            ])
        );
    }

    #[test]
    fn config_hash_is_fixed_content_address() {
        assert_eq!(config_hash(b"same"), config_hash(b"same"));
        assert_ne!(config_hash(b"same"), config_hash(b"different"));
        assert_ne!(
            config_hash(TOOL_CONFIG.as_bytes()),
            config_hash(b""),
            "embedded config contents must contribute to the cache identity"
        );
    }

    #[test]
    fn content_hash_frames_segments_without_concatenation_aliases() {
        let segmented = |segments: &[&[u8]]| {
            let mut hasher = ahash::RandomState::with_seeds(0, 0, 0, 0).build_hasher();
            for segment in segments {
                hash_bytes(&mut hasher, segment);
            }
            hasher.finish()
        };
        assert_ne!(
            segmented(&[b"ab", b"c"]),
            segmented(&[b"a", b"bc"]),
            "fixed-width length framing must distinguish equal concatenations",
        );
    }

    #[test]
    fn concurrent_materializers_replace_corruption_with_exact_bytes() {
        use std::sync::{Arc, Barrier};

        let temp = tempfile::tempdir().expect("create temp cache");
        let root = temp.path().join("nextest");
        std::fs::create_dir_all(&root).expect("create target root");
        let target = materialized_path(&root);
        std::fs::write(&target, b"corrupt partial writer").expect("seed corrupt target");

        let workers = 16;
        let barrier = Arc::new(Barrier::new(workers));
        let joins = (0..workers)
            .map(|_| {
                let barrier = Arc::clone(&barrier);
                let root = root.clone();
                std::thread::spawn(move || {
                    barrier.wait();
                    materialize_in(&root)
                })
            })
            .collect::<Vec<_>>();

        for join in joins {
            assert_eq!(
                join.join()
                    .expect("materializer thread must not panic")
                    .expect("racing materializer must succeed"),
                target,
            );
        }
        assert_eq!(
            std::fs::read(&target).expect("read published config"),
            TOOL_CONFIG.as_bytes(),
            "the final content-addressed target must never retain corruption",
        );
    }

    #[test]
    fn embedded_config_encodes_exact_generated_admission_namespaces() {
        assert!(TOOL_CONFIG.contains("test-threads = 1_000_000"));
        assert!(TOOL_CONFIG.contains("[test-groups.\"@tool:ktstr:host-tests\"]"));
        assert!(TOOL_CONFIG.contains("max-threads = \"num-cpus\""));
        assert!(TOOL_CONFIG.contains("test(/^ktstr\\//)"));
        assert!(TOOL_CONFIG.contains("test(/^gauntlet\\//)"));
        assert!(TOOL_CONFIG.contains("test(/^verifier\\//)"));
        assert!(!TOOL_CONFIG.contains("test(/^host\\//)"));
    }
}
