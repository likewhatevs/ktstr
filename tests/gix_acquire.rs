//! Deterministic tests for the build-script acquisition helper.
//!
//! Fixtures are written directly through gix's object/ref APIs. No `git` or
//! `gix` executable (including a local `git-upload-pack`) participates.

use std::io::{Read as _, Write as _};
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Barrier, Mutex};
use std::time::{Duration, Instant};

use ahash as gix_acquire_ahash;
use fs2 as gix_acquire_fs2;
use gix as gix_acquire_gix;
use jobserver as gix_acquire_jobserver;

// Each build-script consumer reaches a different subset of this shared module.
// This integration target deliberately exercises the cache, pin, recursive
// planning, and reporter seams without treating the network-only functions as
// dead-code errors under the workspace's `-D warnings`.
#[allow(dead_code)]
#[path = "../build_support/cargo_make.rs"]
mod cargo_make;
#[allow(dead_code)]
#[path = "../build_support/gix_acquire.rs"]
mod gix_acquire;

fn signature() -> gix::actor::SignatureRef<'static> {
    gix::actor::SignatureRef::from_bytes(b"ktstr test <ktstr@example.invalid> 1700000000 +0000")
        .expect("valid fixture signature")
}

fn init_repo(path: &Path) -> gix::Repository {
    let mut repo = gix::init(path).expect("initialize fixture repository");
    let _ = repo
        .committer_or_set_generic_fallback()
        .expect("configure fixture fallback identity");
    repo
}

fn copy_dir_all(source: &Path, destination: &Path) -> std::io::Result<()> {
    std::fs::create_dir_all(destination)?;
    for entry in std::fs::read_dir(source)? {
        let entry = entry?;
        let target = destination.join(entry.file_name());
        if entry.file_type()?.is_dir() {
            copy_dir_all(&entry.path(), &target)?;
        } else {
            std::fs::copy(entry.path(), target)?;
        }
    }
    Ok(())
}

fn write_sentinel_script(script: &Path, marker: &Path, stdout: Option<&str>) {
    use std::os::unix::fs::PermissionsExt as _;

    let marker = marker.to_string_lossy().replace('\'', "'\"'\"'");
    let stdout = stdout
        .map(|value| format!("printf '%s\\n' '{value}'\n"))
        .unwrap_or_default();
    std::fs::write(
        script,
        format!("#!/bin/sh\n: > '{marker}'\n{stdout}exit 0\n"),
    )
    .expect("write credential sentinel");
    let mut permissions = std::fs::metadata(script)
        .expect("stat credential sentinel")
        .permissions();
    permissions.set_mode(0o700);
    std::fs::set_permissions(script, permissions).expect("make credential sentinel executable");
}

fn commit_entries(
    repo: &gix::Repository,
    entries: Vec<gix::objs::tree::Entry>,
    message: &str,
) -> gix::ObjectId {
    let mut entries = entries;
    entries.sort_by(|left, right| left.filename.cmp(&right.filename));
    let tree = repo
        .write_object(&gix::objs::Tree { entries })
        .expect("write fixture tree")
        .detach();
    repo.commit_as(
        signature(),
        signature(),
        "HEAD",
        message,
        tree,
        std::iter::empty::<gix::ObjectId>(),
    )
    .expect("write fixture commit")
    .detach()
}

fn commit_marker(repo: &gix::Repository, marker: &[u8]) -> gix::ObjectId {
    let blob = repo
        .write_blob(marker)
        .expect("write fixture blob")
        .detach();
    commit_entries(
        repo,
        vec![gix::objs::tree::Entry {
            mode: gix::objs::tree::EntryKind::Blob.into(),
            filename: "marker".into(),
            oid: blob,
        }],
        "fixture",
    )
}

fn artifact_complete(path: &Path) -> bool {
    path.join("artifact").is_file()
}

const HELPER_MODE: &str = "KTSTR_GIX_ACQUIRE_HELPER_MODE";
const HELPER_ROOT: &str = "KTSTR_GIX_ACQUIRE_HELPER_ROOT";
const HELPER_GATE: &str = "KTSTR_GIX_ACQUIRE_HELPER_GATE";
const HELPER_COUNT: &str = "KTSTR_GIX_ACQUIRE_HELPER_COUNT";
const PROCESS_PARTS: &[&str] = &["fixture", "cross-process-builder"];

fn spawn_cache_helper(root: &Path, gate: &Path, count: &Path, mode: &str) -> std::process::Child {
    Command::new(std::env::current_exe().expect("current integration-test executable"))
        .args([
            "--exact",
            "cache_process_helper",
            "--ignored",
            "--nocapture",
            "--test-threads=1",
        ])
        .env(HELPER_MODE, mode)
        .env(HELPER_ROOT, root)
        .env(HELPER_GATE, gate)
        .env(HELPER_COUNT, count)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn cache helper process")
}

fn wait_cache_helper(mut child: std::process::Child, expected_code: i32) {
    use std::io::Read as _;

    let deadline = Instant::now() + Duration::from_secs(30);
    let (status, timed_out) = loop {
        match child.try_wait().expect("poll cache helper process") {
            Some(status) => break (status, false),
            None if Instant::now() < deadline => {
                std::thread::sleep(Duration::from_millis(10));
            }
            None => {
                let _ = child.kill();
                break (
                    child.wait().expect("reap timed-out cache helper process"),
                    true,
                );
            }
        }
    };
    let mut stdout = Vec::new();
    let mut stderr = Vec::new();
    child
        .stdout
        .take()
        .expect("cache helper stdout pipe")
        .read_to_end(&mut stdout)
        .expect("read cache helper stdout");
    child
        .stderr
        .take()
        .expect("cache helper stderr pipe")
        .read_to_end(&mut stderr)
        .expect("read cache helper stderr");
    assert!(
        !timed_out,
        "cache helper exceeded 30s\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&stdout),
        String::from_utf8_lossy(&stderr),
    );
    assert_eq!(
        status.code(),
        Some(expected_code),
        "helper status {}\nstdout:\n{}\nstderr:\n{}",
        status,
        String::from_utf8_lossy(&stdout),
        String::from_utf8_lossy(&stderr),
    );
}

#[test]
#[ignore = "invoked by the cross-process cache integration tests"]
fn cache_process_helper() {
    let mode = std::env::var(HELPER_MODE).expect("helper mode");
    let root = PathBuf::from(std::env::var_os(HELPER_ROOT).expect("helper root"));
    let gate = PathBuf::from(std::env::var_os(HELPER_GATE).expect("helper gate"));
    let count = PathBuf::from(std::env::var_os(HELPER_COUNT).expect("helper count"));
    let deadline = Instant::now() + Duration::from_secs(10);
    while !gate.exists() {
        assert!(
            Instant::now() < deadline,
            "parent never opened helper start gate"
        );
        std::thread::sleep(Duration::from_millis(10));
    }

    let result = gix_acquire::ensure_cached(
        &root,
        PROCESS_PARTS,
        "cross-process cache fixture",
        artifact_complete,
        |stage, _progress| {
            use std::io::Write;
            let mut count = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&count)
                .map_err(|err| err.to_string())?;
            writeln!(count, "{mode}:{}", std::process::id()).map_err(|err| err.to_string())?;
            count.sync_all().map_err(|err| err.to_string())?;

            match mode.as_str() {
                "waiter" | "takeover" => {
                    // Keep the elected process in its closure long enough that
                    // all peers contend on the same cross-process lock.
                    std::thread::sleep(Duration::from_millis(150));
                    std::fs::write(stage.join("artifact"), b"published")
                        .map_err(|err| err.to_string())
                }
                "crash" => {
                    std::fs::write(stage.join("partial"), b"crash-leftover")
                        .map_err(|err| err.to_string())?;
                    // Model SIGKILL/power loss: no destructors run, leaving
                    // the private stage behind while the OS releases the lock.
                    std::process::exit(97);
                }
                other => Err(format!("unknown helper mode {other}")),
            }
        },
    );
    result.expect("cache helper obtains published artifact");
}

#[test]
fn one_builder_serves_many_concurrent_processes() {
    let temp = tempfile::tempdir().expect("tempdir");
    let root = temp.path().join("cache");
    let gate = temp.path().join("start");
    let count = temp.path().join("builders");
    let children: Vec<_> = (0..8)
        .map(|_| spawn_cache_helper(&root, &gate, &count, "waiter"))
        .collect();
    std::fs::write(&gate, b"go").expect("open helper start gate");
    for child in children {
        wait_cache_helper(child, 0);
    }

    let builders = std::fs::read_to_string(&count).expect("builder record");
    assert_eq!(
        builders.lines().count(),
        1,
        "builder election must happen before expensive work; records:\n{builders}"
    );
    let entry = gix_acquire::cache_entry(&root, PROCESS_PARTS);
    assert_eq!(
        std::fs::read(entry.join("artifact")).expect("published artifact"),
        b"published"
    );
}

#[test]
fn crashed_builder_is_taken_over_and_its_stage_removed() {
    let temp = tempfile::tempdir().expect("tempdir");
    let root = temp.path().join("cache");
    let gate = temp.path().join("start");
    let count = temp.path().join("builders");

    let crashed = spawn_cache_helper(&root, &gate, &count, "crash");
    std::fs::write(&gate, b"go").expect("open helper start gate");
    wait_cache_helper(crashed, 97);

    let id = gix_acquire::content_id(PROCESS_PARTS);
    let stage_prefix = format!(".{id}.work-");
    assert!(
        std::fs::read_dir(&root)
            .expect("cache root after crash")
            .any(|entry| entry
                .expect("cache entry")
                .file_name()
                .to_string_lossy()
                .starts_with(&stage_prefix)),
        "crashed process must leave a realistic unpublished stage"
    );

    let takeover = spawn_cache_helper(&root, &gate, &count, "takeover");
    wait_cache_helper(takeover, 0);
    let entry = gix_acquire::cache_entry(&root, PROCESS_PARTS);
    assert_eq!(
        std::fs::read(entry.join("artifact")).expect("takeover artifact"),
        b"published"
    );
    assert!(
        std::fs::read_dir(&root)
            .expect("cache root after takeover")
            .all(|entry| !entry
                .expect("cache entry")
                .file_name()
                .to_string_lossy()
                .starts_with(&stage_prefix)),
        "the elected takeover must remove all crash-leftover stages"
    );
    let builders = std::fs::read_to_string(&count).expect("builder record");
    assert_eq!(builders.lines().count(), 2, "records:\n{builders}");
    assert!(
        builders
            .lines()
            .next()
            .is_some_and(|line| line.starts_with("crash:"))
    );
    assert!(
        builders
            .lines()
            .nth(1)
            .is_some_and(|line| line.starts_with("takeover:"))
    );
}

#[test]
fn source_node_keys_are_canonical_exact_and_toolchain_independent() {
    const COMMIT: &str = "0123456789abcdef0123456789abcdef01234567";
    let with_suffix = gix_acquire::SourceNode::new(
        "https://GitHub.com/anakryiko/wprof.git",
        "refs/tags/v0.4",
        COMMIT,
    )
    .expect("canonical source node");
    let without_suffix = gix_acquire::SourceNode::new(
        "https://github.com/anakryiko/wprof/",
        "refs/tags/v0.4",
        COMMIT,
    )
    .expect("equivalent GitHub source node");
    assert_eq!(
        with_suffix.id(),
        without_suffix.id(),
        "GitHub's optional .git suffix must not duplicate source objects"
    );

    let another_selector = gix_acquire::SourceNode::new(
        "https://github.com/anakryiko/wprof.git",
        "refs/heads/main",
        COMMIT,
    )
    .expect("second exact selector");
    assert_ne!(
        with_suffix.id(),
        another_selector.id(),
        "the advertised selector participates in the exact source identity"
    );

    // Compiler/target fingerprints belong to the binary cache above this
    // layer. Constructing the same source tuple for two hypothetical
    // toolchains must address the same immutable object database.
    let toolchains = ["clang-18/x86_64", "clang-20/aarch64"];
    let ids: Vec<_> = toolchains
        .iter()
        .map(|_toolchain| {
            gix_acquire::SourceNode::new(
                "https://github.com/anakryiko/wprof",
                "refs/tags/v0.4",
                COMMIT,
            )
            .expect("toolchain-independent source")
            .id()
        })
        .collect();
    assert_eq!(ids[0], ids[1]);
}

#[test]
fn breadth_first_source_graph_coalesces_duplicate_nodes() {
    const ROOT_COMMIT: &str = "1111111111111111111111111111111111111111";
    const CHILD_COMMIT: &str = "2222222222222222222222222222222222222222";
    let root = gix_acquire::SourceNode::new(
        "https://github.com/example/root.git",
        "refs/tags/v1",
        ROOT_COMMIT,
    )
    .unwrap();
    let child = gix_acquire::SourceNode::new(
        "https://github.com/example/shared.git",
        CHILD_COMMIT,
        CHILD_COMMIT,
    )
    .unwrap();
    let root_id = root.id();
    let child_id = child.id();
    let visits = Mutex::new(std::collections::HashMap::<String, usize>::new());
    let cancelled = AtomicBool::new(false);
    let occurrences =
        gix_acquire::walk_source_graph_for_test(root, 4, &cancelled, |node, _cancelled| {
            *visits.lock().unwrap().entry(node.id()).or_default() += 1;
            if node.id() == root_id {
                Ok(vec![
                    (PathBuf::from("vendor/first"), child.clone()),
                    (PathBuf::from("vendor/second"), child.clone()),
                ])
            } else {
                Ok(Vec::new())
            }
        })
        .expect("walk synthetic recursive source graph");

    let visits = visits.into_inner().unwrap();
    assert_eq!(visits.get(&root_id), Some(&1));
    assert_eq!(
        visits.get(&child_id),
        Some(&1),
        "one immutable source node must serve every committed placement"
    );
    assert_eq!(
        occurrences,
        vec![
            (root_id, PathBuf::new()),
            (child_id.clone(), PathBuf::from("vendor/first")),
            (child_id, PathBuf::from("vendor/second")),
        ]
    );
}

#[test]
fn bounded_source_workers_obey_budget_and_cancel_the_remaining_queue() {
    let active = AtomicUsize::new(0);
    let maximum = AtomicUsize::new(0);
    let first_batch = Barrier::new(3);
    let cancelled = AtomicBool::new(false);
    let values = gix_acquire::run_bounded_for_test(
        (0..9usize).collect(),
        3,
        &cancelled,
        |value, _cancelled| {
            let now = active.fetch_add(1, Ordering::AcqRel) + 1;
            maximum.fetch_max(now, Ordering::AcqRel);
            if value < 3 {
                first_batch.wait();
            }
            std::thread::sleep(Duration::from_millis(2));
            active.fetch_sub(1, Ordering::AcqRel);
            Ok(value)
        },
    )
    .expect("bounded fixture completes");
    assert_eq!(values, (0..9).collect::<Vec<_>>());
    assert_eq!(
        maximum.load(Ordering::Acquire),
        3,
        "one caller plus two permits must produce exactly three workers"
    );

    let started = Mutex::new(Vec::new());
    let failing_batch = Barrier::new(2);
    let cancelled = AtomicBool::new(false);
    let error = gix_acquire::run_bounded_for_test(
        (0..6usize).collect(),
        2,
        &cancelled,
        |value, cancelled| -> Result<usize, String> {
            started.lock().unwrap().push(value);
            failing_batch.wait();
            if value == 0 {
                return Err("injected source failure".to_string());
            }
            while !cancelled.load(Ordering::Acquire) {
                std::thread::yield_now();
            }
            Err("peer observed cancellation".to_string())
        },
    )
    .expect_err("first failed source cancels the graph");
    assert_eq!(error, "injected source failure");
    let mut started = started.into_inner().unwrap();
    started.sort_unstable();
    assert_eq!(
        started,
        vec![0, 1],
        "cancelled batches must not start queued source nodes"
    );
}

#[cfg(unix)]
#[test]
fn private_assembly_preserves_git_modes_without_metadata_or_cache_aliases() {
    use std::os::unix::fs::{MetadataExt as _, PermissionsExt as _};

    let temp = tempfile::tempdir().expect("tempdir");
    let repo = init_repo(&temp.path().join("source"));
    let regular = repo.write_blob(b"immutable source\n").unwrap().detach();
    let executable = repo.write_blob(b"#!/bin/sh\nexit 0\n").unwrap().detach();
    let link = repo.write_blob(b"regular").unwrap().detach();
    let commit = commit_entries(
        &repo,
        vec![
            gix::objs::tree::Entry {
                mode: gix::objs::tree::EntryKind::Blob.into(),
                filename: "regular".into(),
                oid: regular,
            },
            gix::objs::tree::Entry {
                mode: gix::objs::tree::EntryKind::BlobExecutable.into(),
                filename: "executable".into(),
                oid: executable,
            },
            gix::objs::tree::Entry {
                mode: gix::objs::tree::EntryKind::Link.into(),
                filename: "link".into(),
                oid: link,
            },
        ],
        "assembly fixture",
    );
    let destination = temp.path().join("assembled");
    gix_acquire::materialize_repository_commit_for_test(&repo, commit, &destination)
        .expect("assemble private tree from object database");

    assert!(!destination.join(".git").exists());
    let regular_mode = std::fs::metadata(destination.join("regular"))
        .unwrap()
        .permissions()
        .mode();
    let executable_mode = std::fs::metadata(destination.join("executable"))
        .unwrap()
        .permissions()
        .mode();
    assert_eq!(regular_mode & 0o111, 0);
    assert_ne!(
        executable_mode & 0o111,
        0,
        "Git's executable bit must survive assembly independently of umask"
    );
    assert_eq!(
        std::fs::read_link(destination.join("link")).unwrap(),
        Path::new("regular")
    );
    assert_eq!(
        std::fs::metadata(destination.join("regular"))
            .unwrap()
            .nlink(),
        1,
        "assembled files must not be hardlinked into shared state"
    );

    std::fs::write(destination.join("regular"), b"private mutation\n").unwrap();
    assert_eq!(
        repo.find_object(regular).unwrap().data,
        b"immutable source\n",
        "private build mutations must not affect cached Git objects"
    );
}

#[test]
fn annotated_tag_peel_is_checked_against_pinned_commit() {
    let temp = tempfile::tempdir().expect("tempdir");
    let repo = init_repo(temp.path());
    let commit = commit_marker(&repo, b"tag target");
    repo.tag(
        "v1",
        commit,
        gix::objs::Kind::Commit,
        Some(signature()),
        "annotated fixture tag",
        gix::refs::transaction::PreviousValue::Any,
    )
    .expect("create annotated tag");

    gix_acquire::verify_reference_commit(&repo, "refs/tags/v1", &commit.to_string())
        .expect("annotated tag peels to its pinned commit");
    let wrong = gix::ObjectId::null(gix::hash::Kind::Sha1);
    let error = gix_acquire::verify_reference_commit(&repo, "refs/tags/v1", &wrong.to_string())
        .expect_err("a moved tag must fail the immutable commit pin");
    assert!(error.contains("expected pinned commit"));
}

#[test]
fn executable_capable_transports_are_rejected_before_acquisition() {
    let temp = tempfile::tempdir().expect("tempdir");
    let reporter = gix_acquire::ProgressReporter::new("transport rejection fixture");
    for (index, url) in [
        "file:///tmp/never-open-this-repository",
        "ssh://example.invalid/repository.git",
        "git://example.invalid/repository.git",
        "hg://example.invalid/repository",
    ]
    .into_iter()
    .enumerate()
    {
        let destination = temp.path().join(format!("checkout-{index}"));
        let error = gix_acquire::checkout_exact(url, "refs/heads/main", &destination, &reporter)
            .expect_err("helper-capable transport must be rejected");
        assert!(
            error.contains("external helper")
                || error.contains("requires an HTTPS remote handled in-process"),
            "unexpected transport rejection for {url}: {error}"
        );
        assert!(
            !destination.exists(),
            "transport rejection must happen before initializing {}",
            destination.display()
        );
    }
}

#[test]
fn hermetic_open_options_disable_terminal_credentials_prompting() {
    let temp = tempfile::tempdir().expect("tempdir");
    let repo = gix::init(temp.path()).expect("initialize fixture repository");
    drop(repo);
    let mut config = std::fs::OpenOptions::new()
        .append(true)
        .open(temp.path().join(".git/config"))
        .expect("open fixture repository config");
    config
        .write_all(b"\n[credential]\n\thelper = !exit 91\n[core]\n\taskPass = /never/run/askpass\n")
        .expect("write hostile credential fixture");
    drop(config);

    let repo = gix::open_opts(temp.path(), gix_acquire::open_options())
        .expect("open with hermetic options");
    assert_eq!(
        repo.config_snapshot()
            .boolean("gitoxide.credentials.terminalPrompt"),
        Some(false),
        "public build-time acquisition must never wait for interactive credentials"
    );
    let credential_url =
        gix::Url::from_bytes(b"https://fixture.invalid/repository.git".as_slice().into())
            .expect("credential fixture URL");
    let (cascade, _, prompt) = repo
        .config_snapshot()
        .credential_helpers(credential_url)
        .expect("resolve credential policy");
    assert!(
        cascade.programs.is_empty(),
        "repository-local credential helpers must be cleared"
    );
    assert_eq!(prompt.mode, gix::prompt::Mode::Disable);
    assert!(
        prompt.askpass.is_none(),
        "hermetic public acquisition must not invoke an askpass program"
    );
    let permissions = repo.open_options().permissions;
    assert_eq!(
        permissions.env.git_prefix,
        gix::sec::Permission::Deny,
        "GIT_* credential and askpass environment must be ignored"
    );
    assert_eq!(
        permissions.env.ssh_prefix,
        gix::sec::Permission::Deny,
        "SSH_ASKPASS must be ignored"
    );
    assert!(!permissions.config.includes);
    assert!(!permissions.attributes.system);
    assert!(!permissions.attributes.git);
    assert!(!permissions.attributes.git_binary);
    assert_eq!(
        repo.refs.write_reflog,
        gix::refs::store::WriteReflog::Disable,
        "exact acquisition repositories must never require reflog identity"
    );
    let transport = repo
        .transport_options("https://fixture.invalid/repository.git".as_bytes(), None)
        .expect("resolve hermetic transport options")
        .expect("HTTPS transport options");
    let transport = transport
        .downcast_ref::<gix::protocol::transport::client::blocking_io::http::Options>()
        .expect("gix HTTP transport options");
    assert_eq!(transport.connect_timeout, Some(Duration::from_secs(20)));
    assert_eq!(transport.low_speed_limit_bytes_per_second, 1024);
    assert_eq!(transport.low_speed_time_seconds, 30);
    assert!(
        transport
            .backend
            .as_ref()
            .and_then(|backend| backend.lock().ok())
            .is_some_and(|backend| backend
                .downcast_ref::<gix::protocol::transport::client::blocking_io::http::curl::Options>(
                )
                .is_some()),
        "the gix HTTP transport must be in-process curl so low-speed limits are enforced"
    );
}

#[test]
fn public_http_authentication_cannot_execute_configured_credentials_programs() {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind authentication fixture");
    let address = listener
        .local_addr()
        .expect("authentication fixture address");
    listener
        .set_nonblocking(true)
        .expect("make authentication fixture accept bounded");
    let server_done = Arc::new(AtomicBool::new(false));
    let server_done_worker = Arc::clone(&server_done);
    let server = std::thread::spawn(move || {
        let deadline = Instant::now() + Duration::from_secs(5);
        let mut requests = Vec::new();
        while !server_done_worker.load(Ordering::Acquire) || requests.is_empty() {
            assert!(
                Instant::now() < deadline,
                "authentication fixture did not finish"
            );
            let (mut stream, _) = match listener.accept() {
                Ok(connection) => connection,
                Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                    std::thread::sleep(Duration::from_millis(5));
                    continue;
                }
                Err(error) => panic!("accept authentication request: {error}"),
            };
            stream
                .set_read_timeout(Some(Duration::from_secs(2)))
                .expect("bound authentication fixture read");
            let mut request = Vec::new();
            let mut byte = [0u8; 1];
            while !request.ends_with(b"\r\n\r\n") {
                stream
                    .read_exact(&mut byte)
                    .expect("read authentication request");
                request.push(byte[0]);
                assert!(
                    request.len() < 64 * 1024,
                    "unbounded authentication request"
                );
            }
            stream
                .write_all(
                    b"HTTP/1.1 401 Unauthorized\r\n\
                      WWW-Authenticate: Basic realm=\"ktstr-test\"\r\n\
                      Content-Length: 0\r\n\
                      Connection: close\r\n\r\n",
                )
                .expect("write authentication challenge");
            requests.push(request);
        }
        requests
    });

    let temp = tempfile::tempdir().expect("authentication tempdir");
    let helper_marker = temp.path().join("credential-helper-ran");
    let askpass_marker = temp.path().join("askpass-ran");
    let helper = temp.path().join("credential-helper");
    let askpass = temp.path().join("askpass");
    write_sentinel_script(&helper, &helper_marker, None);
    write_sentinel_script(&askpass, &askpass_marker, Some("sentinel-secret"));

    let repository = temp.path().join("repository");
    let mut repo = gix::ThreadSafeRepository::init_opts(
        &repository,
        gix::create::Kind::WithWorktree,
        gix::create::Options::default(),
        gix_acquire::open_options(),
    )
    .expect("initialize authentication repository")
    .to_thread_local();
    let overrides: Vec<gix::bstr::BString> = [
        format!("credential.helper={}", helper.display()),
        format!("core.askPass={}", askpass.display()),
        "gitoxide.credentials.terminalPrompt=true".to_string(),
        "gitoxide.http.noProxy=*".to_string(),
    ]
    .into_iter()
    .map(Into::into)
    .collect();
    let mut snapshot = repo.config_snapshot_mut();
    snapshot
        .append_config(&overrides, gix::config::Source::Api)
        .expect("append hostile credential overrides");
    snapshot
        .commit()
        .expect("commit hostile credential overrides");

    let url = format!("http://{address}/repository.git");
    let credential_url =
        gix::Url::from_bytes(url.as_bytes().into()).expect("parse authentication fixture URL");
    let (cascade, _, prompt) = repo
        .config_snapshot()
        .credential_helpers(credential_url)
        .expect("resolve hostile credential fixture");
    assert_eq!(
        cascade.programs.len(),
        1,
        "the test must configure an executable credential helper"
    );
    assert_eq!(
        prompt.askpass.as_deref(),
        Some(askpass.as_path()),
        "the test must configure an executable askpass program"
    );
    assert_ne!(
        prompt.mode,
        gix::prompt::Mode::Disable,
        "the test must override terminal prompting after the production policy"
    );

    let mut remote = repo
        .remote_at_without_url_rewrite(url.as_str())
        .expect("prepare authentication fixture remote")
        .with_fetch_tags(gix::remote::fetch::Tags::None);
    remote
        .replace_refspecs(
            ["+refs/heads/main:refs/ktstr/auth-fixture"],
            gix::remote::Direction::Fetch,
        )
        .expect("configure authentication fixture refspec");
    let mut connection = remote
        .connect(gix::remote::Direction::Fetch)
        .expect("connect authentication fixture");
    connection.set_credentials(gix_acquire::reject_credentials_for_test);
    let result = connection.prepare_fetch(
        gix::progress::Discard,
        gix::remote::ref_map::Options::default(),
    );
    server_done.store(true, Ordering::Release);
    let requests = server.join().expect("join authentication fixture");

    assert!(
        result.is_err(),
        "a public source requiring credentials must be rejected"
    );
    assert_eq!(
        requests.len(),
        1,
        "credential rejection must not retry with authentication"
    );
    assert!(
        !String::from_utf8_lossy(&requests[0])
            .to_ascii_lowercase()
            .contains("\r\nauthorization:"),
        "the sole public-source request must remain anonymous"
    );
    assert!(
        !helper_marker.exists(),
        "the public-source path executed a configured credential helper"
    );
    assert!(
        !askpass_marker.exists(),
        "the public-source path executed a configured askpass program"
    );
}

#[test]
fn stalled_smart_http_response_is_aborted_by_the_real_gix_transport() {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind stalled HTTP fixture");
    let address = listener.local_addr().expect("fixture address");
    listener
        .set_nonblocking(true)
        .expect("make fixture accept bounded");
    let server = std::thread::spawn(move || {
        let deadline = Instant::now() + Duration::from_secs(5);
        let (mut stream, _) = loop {
            match listener.accept() {
                Ok(connection) => break connection,
                Err(error)
                    if error.kind() == std::io::ErrorKind::WouldBlock
                        && Instant::now() < deadline =>
                {
                    std::thread::sleep(Duration::from_millis(10));
                }
                Err(error) => panic!("accept gix request: {error}"),
            }
        };
        stream
            .set_read_timeout(Some(Duration::from_secs(5)))
            .expect("set fixture read timeout");
        let mut request = Vec::new();
        let mut byte = [0u8; 1];
        while !request.ends_with(b"\r\n\r\n") {
            stream.read_exact(&mut byte).expect("read gix request");
            request.push(byte[0]);
            assert!(request.len() < 64 * 1024, "unbounded fixture request");
        }
        assert!(
            request.starts_with(b"GET /repository.git/info/refs?service=git-upload-pack "),
            "unexpected smart-HTTP request: {}",
            String::from_utf8_lossy(&request)
        );
        stream
            .write_all(
                b"HTTP/1.1 200 OK\r\n\
                  Content-Type: application/x-git-upload-pack-advertisement\r\n\
                  Content-Length: 1048576\r\n\
                  Connection: close\r\n\r\n",
            )
            .expect("write stalled response headers");
        // Send no advertised-ref body. The production curl backend's
        // low-speed policy must close this connection rather than wait
        // indefinitely.
        let mut drain = [0u8; 1];
        let _ = stream.read(&mut drain);
    });

    let temp = tempfile::tempdir().expect("tempdir");
    let reporter = gix_acquire::ProgressReporter::new("stalled transport fixture");
    let started = Instant::now();
    let result = gix_acquire::clone_one_with_transport_limits_for_test(
        &format!("http://{address}/repository.git"),
        "refs/heads/main",
        &temp.path().join("checkout"),
        &reporter,
        1_000,
        1024,
        1,
    );
    let elapsed = started.elapsed();
    drop(reporter);
    server.join().expect("join stalled HTTP fixture");
    assert!(result.is_err(), "stalled smart HTTP unexpectedly succeeded");
    assert!(
        elapsed < Duration::from_secs(5),
        "in-process gix transport ignored its low-speed bound for {elapsed:?}: {result:?}"
    );
}

#[test]
fn recursive_checkout_uses_committed_gitlink_not_any_child_tip() {
    let temp = tempfile::tempdir().expect("tempdir");
    let child_path = temp.path().join("child");
    let child = init_repo(&child_path);
    let pinned = commit_marker(&child, b"pinned child");
    let newer = {
        let blob = child
            .write_blob(b"newer child")
            .expect("new child blob")
            .detach();
        let tree = child
            .write_object(&gix::objs::Tree {
                entries: vec![gix::objs::tree::Entry {
                    mode: gix::objs::tree::EntryKind::Blob.into(),
                    filename: "marker".into(),
                    oid: blob,
                }],
            })
            .expect("new child tree")
            .detach();
        child
            .commit_as(
                signature(),
                signature(),
                "HEAD",
                "new child tip",
                tree,
                std::iter::once(pinned),
            )
            .expect("new child commit")
            .detach()
    };
    assert_ne!(pinned, newer);

    let parent_path = temp.path().join("parent");
    let parent = init_repo(&parent_path);
    let modules = "[submodule \"child\"]\n\tpath = child\n\turl = ../child.git\n";
    let modules_blob = parent
        .write_blob(modules.as_bytes())
        .expect("write .gitmodules")
        .detach();
    let parent_commit = commit_entries(
        &parent,
        vec![
            gix::objs::tree::Entry {
                mode: gix::objs::tree::EntryKind::Blob.into(),
                filename: ".gitmodules".into(),
                oid: modules_blob,
            },
            gix::objs::tree::Entry {
                mode: gix::objs::tree::EntryKind::Commit.into(),
                filename: "child".into(),
                oid: pinned,
            },
        ],
        "parent",
    );
    parent
        .tag(
            "v1",
            parent_commit,
            gix::objs::Kind::Commit,
            Some(signature()),
            "recursive fixture tag",
            gix::refs::transaction::PreviousValue::Any,
        )
        .expect("tag parent fixture");

    let parent_url = "https://fixture.invalid/org/parent.git";
    let child_url = "https://fixture.invalid/org/child.git";
    let plan = gix_acquire::submodule_checkouts(&parent, parent_url)
        .expect("read recursive checkout plan");
    assert_eq!(plan.len(), 1);
    assert_eq!(plan[0].path, Path::new("child"));
    assert_eq!(plan[0].url, child_url);
    assert_eq!(plan[0].commit, pinned.to_string());
    assert_ne!(plan[0].commit, newer.to_string());

    // gix's local smart transport intentionally invokes git-upload-pack, so
    // the fixture instead copies gix-created object stores and injects that
    // in-process exact materializer at the transport seam. The production
    // recursive engine, tag/commit verification, gitlink selection, checkout,
    // and nested destination handling all run unchanged.
    let output = temp.path().join("recursive-output");
    let reporter = gix_acquire::ProgressReporter::new("recursive checkout fixture");
    gix_acquire::checkout_exact_recursive_with(
        parent_url,
        "refs/tags/v1",
        &parent_commit.to_string(),
        &output,
        &reporter,
        &|url, revision, destination, progress| {
            let source = if url == parent_url {
                &parent_path
            } else if url == child_url {
                &child_path
            } else {
                return Err(format!("unknown fixture URL {url}"));
            };
            copy_dir_all(&source.join(".git"), &destination.join(".git"))
                .map_err(|err| format!("copy fixture object store: {err}"))?;
            let copied_logs = destination.join(".git/logs");
            if copied_logs.exists() {
                std::fs::remove_dir_all(&copied_logs)
                    .map_err(|err| format!("remove copied fixture reflogs: {err}"))?;
            }
            let repo = gix::open_opts(destination, gix_acquire::open_options())
                .map_err(|err| format!("open copied fixture repository: {err}"))?;
            if repo.refs.write_reflog != gix::refs::store::WriteReflog::Disable {
                return Err("fixture repository unexpectedly enables reflogs".to_string());
            }
            let commit = if revision.starts_with("refs/") {
                repo.find_reference(revision)
                    .map_err(|err| format!("find fixture ref {revision}: {err}"))?
                    .peel_to_commit()
                    .map_err(|err| format!("peel fixture ref {revision}: {err}"))?
                    .id
            } else {
                let id = gix::ObjectId::from_hex(revision.as_bytes())
                    .map_err(|err| format!("parse fixture object {revision}: {err}"))?;
                repo.find_object(id)
                    .map_err(|err| format!("find fixture object {revision}: {err}"))?
                    .peel_to_commit()
                    .map_err(|err| format!("peel fixture object {revision}: {err}"))?
                    .id
            };
            gix_acquire::materialize_commit_for_test(&repo, commit, destination, progress)
        },
    )
    .expect("materialize recursive fixture without an executable transport");
    drop(reporter);
    assert_eq!(
        std::fs::read(output.join("child/marker")).expect("checked-out child marker"),
        b"pinned child",
        "recursive checkout must materialize the parent's gitlink, not child HEAD"
    );
    for (repository, expected) in [
        (output.clone(), parent_commit),
        (output.join("child"), pinned),
    ] {
        assert!(
            !repository.join(".git/logs/HEAD").exists(),
            "exact acquisition must not create a reflog in {}",
            repository.display()
        );
        let repo = gix::open_opts(&repository, gix_acquire::open_options())
            .expect("reopen materialized exact repository");
        assert!(
            repo.head().expect("read exact HEAD").is_detached(),
            "exact acquisition HEAD must remain detached"
        );
        assert_eq!(
            repo.head_commit().expect("read exact HEAD commit").id,
            expected,
            "exact acquisition must publish the pinned commit"
        );
    }
}

#[test]
fn heartbeat_reporter_shutdown_joins_without_waiting_for_next_tick() {
    let started = Instant::now();
    let reporter = gix_acquire::ProgressReporter::new("heartbeat shutdown fixture");
    reporter.set_phase("blocked operation");
    drop(reporter);
    assert!(
        started.elapsed() < Duration::from_secs(2),
        "Condvar wake must stop and join the heartbeat thread immediately"
    );
}

#[test]
fn heartbeat_spawn_failure_keeps_acquisition_reporting_usable() {
    let reporter = gix_acquire::ProgressReporter::with_failed_spawn_for_test(
        "heartbeat spawn-failure fixture",
    );
    assert!(
        !reporter.heartbeat_thread_active_for_test(),
        "injected thread exhaustion must degrade to phase-only reporting"
    );
    reporter.set_phase("phase-only reporting remains live");
    drop(reporter);
}

#[test]
fn configured_make_retains_jobserver_descriptors_through_spawn() {
    let client = jobserver::Client::new(2).expect("create fixture jobserver");
    let mut command =
        cargo_make::CargoCoordinatedMake::with_client(Command::new("make"), Some(client));
    command
        .arg("--version")
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    let status = command
        .status()
        .expect("configured child must not fail pre-exec with EBADF");
    assert!(status.success(), "make --version must succeed");
}
