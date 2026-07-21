use super::*;

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use crate::test_support::test_helpers::{EnvVarGuard, lock_env};

fn write_fixture_tree(root: &Path, payload: &[u8]) {
    let bin = root.join("target/debug/deps");
    std::fs::create_dir_all(&bin).expect("create fixture target tree");
    std::fs::write(bin.join("harness-a1b2"), payload).expect("write fixture harness");
    std::fs::set_permissions(
        bin.join("harness-a1b2"),
        std::fs::Permissions::from_mode(0o751),
    )
    .expect("chmod fixture harness");
    std::fs::write(
        root.join("target/cargo-metadata.json"),
        b"{\"packages\":[]}",
    )
    .expect("write cargo metadata");
    std::fs::set_permissions(
        root.join("target/cargo-metadata.json"),
        std::fs::Permissions::from_mode(0o640),
    )
    .expect("chmod cargo metadata");
    std::os::unix::fs::symlink(
        "harness-a1b2",
        root.join("target/debug/deps/harness-current"),
    )
    .expect("create fixture symlink");
}

fn source_from_fixture(root: &Path) -> ArtifactTreeSource {
    let mut source = ArtifactTreeSource::new();
    source
        .insert_tree("target", root.join("target"))
        .expect("capture fixture target tree");
    source
}

#[test]
fn generated_bytes_are_staged_on_the_content_cas_filesystem() {
    let _environment = lock_env();
    let temp = tempfile::tempdir().expect("generated artifact test root");
    let cache_root = temp.path().join("cas");
    let _cache = EnvVarGuard::set(crate::KTSTR_CACHE_DIR_ENV, &cache_root);
    let mut source = ArtifactTreeSource::new();
    source
        .insert_bytes("meta/generated.json", b"{\"generated\":true}", 0o444)
        .expect("stage generated artifact bytes");

    let SourceEntry::File { pinned, mode } = source
        .entries
        .get(Path::new("meta/generated.json"))
        .expect("generated source entry")
    else {
        panic!("generated metadata must be represented by a pinned file")
    };
    assert_eq!(*mode, 0o444);
    assert_eq!(
        pinned.source().metadata().unwrap().dev(),
        std::fs::metadata(cache_root.join("objects-v2"))
            .unwrap()
            .dev(),
        "strict-FICLONE metadata staging must share the CAS filesystem",
    );
}

#[test]
fn cross_root_hit_survives_source_deletion_and_restores_modes_and_links() {
    let _environment = lock_env();
    let temp = tempfile::tempdir().expect("artifact tree test root");
    let _cache = EnvVarGuard::set(crate::KTSTR_CACHE_DIR_ENV, temp.path().join("cas"));
    let source_root = temp.path().join("checkout-a");
    let materializations = temp.path().join("materializations");
    write_fixture_tree(&source_root, b"cross-root-harness");

    let cache = ArtifactTreeCache::new(temp.path().join("records"));
    let first = cache
        .load_or_build(
            0xc205_5007,
            &materializations,
            "artifact-tree-test",
            || Ok(true),
            || false,
            || Ok(source_from_fixture(&source_root)),
        )
        .expect("publish fixture tree");
    assert!(!first.cache_hit());
    assert_eq!(
        std::fs::read(first.root().join("target/debug/deps/harness-current"))
            .expect("read cold materialization through symlink"),
        b"cross-root-harness"
    );
    std::fs::write(
        first.root().join("target/debug/deps/harness-a1b2"),
        b"private-cow-write",
    )
    .expect("write private COW materialization");
    assert_eq!(
        std::fs::read(source_root.join("target/debug/deps/harness-a1b2"))
            .expect("read untouched producer after COW write"),
        b"cross-root-harness",
        "a materialized-tree write must split extents instead of mutating its source",
    );
    drop(first);

    std::fs::remove_dir_all(&source_root).expect("delete producer checkout");
    let second = cache
        .load_or_build(
            0xc205_5007,
            &materializations,
            "artifact-tree-test",
            || Ok(true),
            || false,
            || -> Result<ArtifactTreeSource> {
                panic!("a relocated cache hit must not invoke the producer")
            },
        )
        .expect("restore fixture tree in a different root");
    assert!(second.cache_hit());
    assert!(!second.root().starts_with(&source_root));
    let harness = second.root().join("target/debug/deps/harness-a1b2");
    assert_eq!(std::fs::read(&harness).unwrap(), b"cross-root-harness");
    assert_eq!(
        std::fs::metadata(&harness).unwrap().permissions().mode() & 0o7777,
        0o751
    );
    assert_eq!(
        std::fs::metadata(second.root().join("target/cargo-metadata.json"))
            .unwrap()
            .permissions()
            .mode()
            & 0o7777,
        0o640
    );
    assert_eq!(
        std::fs::read_link(second.root().join("target/debug/deps/harness-current")).unwrap(),
        PathBuf::from("harness-a1b2")
    );
}

#[test]
fn unpublished_objects_and_crash_temp_do_not_form_a_visible_tree() {
    let _environment = lock_env();
    let temp = tempfile::tempdir().expect("artifact crash test root");
    let _cache = EnvVarGuard::set(crate::KTSTR_CACHE_DIR_ENV, temp.path().join("cas"));
    let source_root = temp.path().join("checkout");
    let record_root = temp.path().join("records");
    let materializations = temp.path().join("materializations");
    let identity = 0xc2a5_0001;
    write_fixture_tree(&source_root, b"crash-before-record");
    ensure_cache_dirs(&record_root).unwrap();

    // Model SIGKILL after every file reached the shared CAS but before the
    // atomic record rename. Dropping these leases is exactly process exit.
    drop(publish_source(identity, source_from_fixture(&source_root)).unwrap());
    let (record, _, _) = cache_paths(&record_root, identity);
    assert!(!record.exists());
    std::fs::write(
        record.parent().unwrap().join(".tmp-crashed-producer"),
        b"partial-json",
    )
    .unwrap();

    let builds = AtomicUsize::new(0);
    let tree = ArtifactTreeCache::new(&record_root)
        .load_or_build(
            identity,
            &materializations,
            "artifact-tree-crash-test",
            || Ok(true),
            || false,
            || {
                builds.fetch_add(1, Ordering::SeqCst);
                Ok(source_from_fixture(&source_root))
            },
        )
        .expect("successor publishes complete record");
    assert_eq!(builds.load(Ordering::SeqCst), 1);
    assert!(!tree.cache_hit());
    assert!(record.exists());
    assert_eq!(
        std::fs::read(tree.root().join("target/debug/deps/harness-a1b2")).unwrap(),
        b"crash-before-record"
    );
}

#[test]
fn corrupt_record_and_missing_content_object_are_reconstructible_misses() {
    let _environment = lock_env();
    let temp = tempfile::tempdir().expect("artifact corruption test root");
    let cache_root = temp.path().join("cas");
    let _cache = EnvVarGuard::set(crate::KTSTR_CACHE_DIR_ENV, &cache_root);
    let source_root = temp.path().join("checkout");
    let record_root = temp.path().join("records");
    let materializations = temp.path().join("materializations");
    let identity = 0xc022_0a7;
    write_fixture_tree(&source_root, b"reconstructible-tree");
    let cache = ArtifactTreeCache::new(&record_root);
    let builds = AtomicUsize::new(0);
    let build = || {
        builds.fetch_add(1, Ordering::SeqCst);
        Ok(source_from_fixture(&source_root))
    };

    drop(
        cache
            .load_or_build(
                identity,
                &materializations,
                "artifact-tree-corruption-test",
                || Ok(true),
                || false,
                build,
            )
            .unwrap(),
    );
    let (record_path, _, _) = cache_paths(&record_root, identity);
    std::fs::set_permissions(&record_path, std::fs::Permissions::from_mode(0o644)).unwrap();
    std::fs::write(&record_path, b"{").unwrap();
    drop(
        cache
            .load_or_build(
                identity,
                &materializations,
                "artifact-tree-corruption-test",
                || Ok(true),
                || false,
                build,
            )
            .expect("corrupt record rebuild"),
    );
    assert_eq!(builds.load(Ordering::SeqCst), 2);

    let record: ArtifactTreeRecord =
        serde_json::from_slice(&std::fs::read(&record_path).unwrap()).unwrap();
    let (content_hash, _) = record
        .entries
        .iter()
        .find_map(|entry| match entry {
            RecordEntry::File {
                content_hash, len, ..
            } => Some((*content_hash, *len)),
            _ => None,
        })
        .expect("file object in record");
    let object = cache_root
        .join("objects-v2")
        .join(format!("{content_hash:016x}.object"));
    std::fs::remove_file(&object).expect("remove content object after leases drop");
    let rebuilt = cache
        .load_or_build(
            identity,
            &materializations,
            "artifact-tree-corruption-test",
            || Ok(true),
            || false,
            build,
        )
        .expect("missing content object rebuild");
    assert_eq!(builds.load(Ordering::SeqCst), 3);
    assert!(!rebuilt.cache_hit());
    assert_eq!(
        std::fs::read(rebuilt.root().join("target/debug/deps/harness-a1b2")).unwrap(),
        b"reconstructible-tree"
    );
}

#[test]
fn rejects_tree_escape_and_non_directory_ancestors_without_poisoning_source() {
    let mut source = ArtifactTreeSource::new();
    assert!(source.insert_symlink("escape", "../outside").is_err());
    assert!(source.insert_directory("../outside", 0o755).is_err());

    let temp = tempfile::tempdir().unwrap();
    let file = temp.path().join("file");
    std::fs::write(&file, b"file").unwrap();
    source.insert_file("parent", &file).unwrap();
    assert!(source.insert_file("parent/child", &file).is_err());
    assert_eq!(source.len(), 1, "a rejected insertion must roll back");

    let mut reverse = ArtifactTreeSource::new();
    reverse.insert_directory("parent/child", 0o755).unwrap();
    assert!(
        reverse.insert_file("parent", &file).is_err(),
        "a file cannot be inserted above an existing descendant",
    );
    assert!(
        reverse.insert_symlink("parent", "target").is_err(),
        "a symlink cannot be inserted above an existing descendant",
    );
    reverse
        .insert_directory("parent", 0o755)
        .expect("a directory may be inserted after its child");
    assert_eq!(reverse.len(), 2, "failed parent insertions must not mutate");
}

#[test]
fn insertion_validation_work_is_linear_in_entry_count_at_fixed_depth() {
    const ENTRIES: usize = 10_000;

    let mut source = ArtifactTreeSource::new();
    source.insert_directory("root", 0o755).unwrap();
    for index in 0..ENTRIES {
        source
            .insert_symlink(format!("root/item-{index:05}"), "target")
            .unwrap();
    }

    assert_eq!(source.len(), ENTRIES + 1);
    assert!(
        source.insertion_validation_visits() <= 3 * ENTRIES + 2,
        "incremental validation must visit only the new path and its ancestors; visits={} entries={ENTRIES}",
        source.insertion_validation_visits(),
    );
}

#[test]
fn cow_clone_reads_the_leased_inode_after_its_path_is_replaced() {
    let temp = tempfile::tempdir().unwrap();
    let source_path = temp.path().join("source");
    let old_path = temp.path().join("old");
    let destination = temp.path().join("clone");
    std::fs::write(&source_path, b"leased-revision").unwrap();
    let source = std::fs::File::open(&source_path).unwrap();
    std::fs::rename(&source_path, &old_path).unwrap();
    std::fs::write(&source_path, b"replacement").unwrap();

    reflink_required(&source, &source_path, &destination).unwrap();
    assert_eq!(std::fs::read(destination).unwrap(), b"leased-revision");
}

#[test]
fn stale_materialization_gc_observes_cross_process_liveness_lock() {
    let temp = tempfile::tempdir().unwrap();
    let stale = temp.path().join(format!("{MATERIALIZATION_PREFIX}dead"));
    let live = temp.path().join(format!("{MATERIALIZATION_PREFIX}live"));
    std::fs::create_dir(&stale).unwrap();
    std::fs::write(stale.join(MATERIALIZATION_LIVE_LOCK), b"").unwrap();
    std::fs::create_dir(&live).unwrap();
    let live_file = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create_new(true)
        .open(live.join(MATERIALIZATION_LIVE_LOCK))
        .unwrap();
    rustix::fs::flock(&live_file, rustix::fs::FlockOperation::LockExclusive).unwrap();

    let future = SystemTime::now() + MATERIALIZATION_GC_GRACE + Duration::from_secs(1);
    gc_stale_materializations(temp.path(), future).unwrap();
    assert!(!stale.exists());
    assert!(live.exists());
    drop(live_file);
    gc_stale_materializations(
        temp.path(),
        future + MATERIALIZATION_GC_INTERVAL + Duration::from_secs(1),
    )
    .unwrap();
    assert!(!live.exists());
}

#[test]
fn materialization_gc_is_cross_process_rate_limited_and_rotates() {
    let temp = tempfile::tempdir().unwrap();
    for index in 0..=MATERIALIZATION_GC_SCAN_LIMIT {
        let path = temp
            .path()
            .join(format!("{MATERIALIZATION_PREFIX}{index:04}"));
        std::fs::create_dir(&path).unwrap();
        std::fs::write(path.join(MATERIALIZATION_LIVE_LOCK), b"").unwrap();
    }
    let first = SystemTime::now() + MATERIALIZATION_GC_GRACE + Duration::from_secs(1);
    gc_stale_materializations(temp.path(), first).unwrap();
    let remaining = std::fs::read_dir(temp.path())
        .unwrap()
        .filter_map(|entry| entry.ok())
        .filter(|entry| {
            entry
                .file_name()
                .as_bytes()
                .starts_with(MATERIALIZATION_PREFIX.as_bytes())
        })
        .count();
    assert_eq!(
        remaining, 1,
        "one candidate must remain beyond the scan cap"
    );

    gc_stale_materializations(temp.path(), first).unwrap();
    assert_eq!(
        std::fs::read_dir(temp.path())
            .unwrap()
            .filter_map(|entry| entry.ok())
            .filter(|entry| {
                entry
                    .file_name()
                    .as_bytes()
                    .starts_with(MATERIALIZATION_PREFIX.as_bytes())
            })
            .count(),
        1,
        "the persisted stamp must suppress another immediate scan",
    );

    gc_stale_materializations(
        temp.path(),
        first + MATERIALIZATION_GC_INTERVAL + Duration::from_secs(1),
    )
    .unwrap();
    assert!(
        std::fs::read_dir(temp.path())
            .unwrap()
            .filter_map(|entry| entry.ok())
            .all(|entry| !entry
                .file_name()
                .as_bytes()
                .starts_with(MATERIALIZATION_PREFIX.as_bytes())),
        "the next bounded pass must resume after the persisted cursor",
    );
}

#[test]
fn materialization_gc_skips_when_another_collector_owns_the_gate() {
    let temp = tempfile::tempdir().unwrap();
    let stale = temp.path().join(format!("{MATERIALIZATION_PREFIX}stale"));
    std::fs::create_dir(&stale).unwrap();
    std::fs::write(stale.join(MATERIALIZATION_LIVE_LOCK), b"").unwrap();
    let gate = crate::flock::try_flock(
        temp.path().join(MATERIALIZATION_GC_LOCK),
        crate::flock::FlockMode::Exclusive,
    )
    .unwrap()
    .expect("own materialization collector gate");
    let future = SystemTime::now() + MATERIALIZATION_GC_GRACE + Duration::from_secs(1);
    gc_stale_materializations(temp.path(), future).unwrap();
    assert!(stale.exists());
    drop(gate);
    gc_stale_materializations(temp.path(), future).unwrap();
    assert!(!stale.exists());
}

#[test]
fn indexed_artifact_io_preserves_result_and_error_order() {
    let values = parallel_indexed(vec![0usize, 1, 2, 3], |value| {
        std::thread::sleep(Duration::from_millis((3 - value) as u64));
        Ok(value)
    })
    .unwrap();
    assert_eq!(values, vec![0, 1, 2, 3]);

    let error = parallel_indexed(vec![0usize, 1, 2], |value| -> Result<usize> {
        std::thread::sleep(Duration::from_millis((2 - value) as u64));
        anyhow::bail!("indexed error {value}")
    })
    .unwrap_err();
    assert!(
        error.to_string().contains("indexed error 0"),
        "the lowest deterministic input error must win: {error:#}",
    );

    let available = std::thread::available_parallelism().map_or(1, usize::from);
    if available > 1 {
        let active = AtomicUsize::new(0);
        let peak = AtomicUsize::new(0);
        let expected = ARTIFACT_IO_WORKERS_MAX.min(available).min(32);
        let barrier = std::sync::Barrier::new(expected);
        parallel_indexed((0..expected * 2).collect(), |_| {
            let current = active.fetch_add(1, Ordering::SeqCst) + 1;
            peak.fetch_max(current, Ordering::SeqCst);
            barrier.wait();
            active.fetch_sub(1, Ordering::SeqCst);
            Ok(())
        })
        .unwrap();
        assert_eq!(peak.load(Ordering::SeqCst), expected);
    }
}

#[test]
fn split_validators_only_rescan_after_a_cold_producer() {
    let _environment = lock_env();
    let temp = tempfile::tempdir().unwrap();
    let _cache = EnvVarGuard::set(crate::KTSTR_CACHE_DIR_ENV, temp.path().join("cas"));
    let source_root = temp.path().join("source");
    write_fixture_tree(&source_root, b"split-validator-tree");
    let cache = ArtifactTreeCache::new(temp.path().join("records"));
    let materializations = temp.path().join("materializations");
    let built = AtomicBool::new(false);
    let cached_validations = AtomicUsize::new(0);
    let published_validations = AtomicUsize::new(0);

    let cold = cache
        .load_or_build_with_validators(
            0x5ca1_1da7,
            &materializations,
            "split-validator-test",
            || {
                cached_validations.fetch_add(1, Ordering::SeqCst);
                Ok(true)
            },
            || {
                assert!(built.load(Ordering::SeqCst));
                published_validations.fetch_add(1, Ordering::SeqCst);
                Ok(true)
            },
            || false,
            || {
                built.store(true, Ordering::SeqCst);
                Ok(source_from_fixture(&source_root))
            },
        )
        .unwrap();
    assert!(!cold.cache_hit());
    assert_eq!(cached_validations.load(Ordering::SeqCst), 0);
    assert_eq!(published_validations.load(Ordering::SeqCst), 1);
    drop(cold);

    let hit = cache
        .load_or_build_with_validators(
            0x5ca1_1da7,
            &materializations,
            "split-validator-test",
            || {
                cached_validations.fetch_add(1, Ordering::SeqCst);
                Ok(true)
            },
            || {
                published_validations.fetch_add(1, Ordering::SeqCst);
                Ok(true)
            },
            || false,
            || -> Result<ArtifactTreeSource> { panic!("cache hit invoked producer") },
        )
        .unwrap();
    assert!(hit.cache_hit());
    assert_eq!(cached_validations.load(Ordering::SeqCst), 1);
    assert_eq!(published_validations.load(Ordering::SeqCst), 1);
}

#[test]
fn stable_cargo_output_is_hashed_once_before_sealing_changes_ctime() {
    let _environment = lock_env();
    let temp = tempfile::tempdir().unwrap();
    let _cache = EnvVarGuard::set(crate::KTSTR_CACHE_DIR_ENV, temp.path().join("cas"));
    let cache = ArtifactTreeCache::new(temp.path().join("records"));
    let stable_parent = temp.path().join("stable");
    let materializations = temp.path().join("materializations");
    let identity = 0x5ea1_cab0;
    let pinned_identity = std::cell::Cell::new(None);

    let tree = cache
        .load_or_build_with_stable_cargo_output(
            identity,
            &stable_parent,
            &materializations,
            "stable-cargo-hash-test",
            || Ok(true),
            || false,
            |stable| {
                let output = stable.target_directory.join("debug/deps/harness");
                std::fs::create_dir_all(output.parent().unwrap())?;
                std::fs::write(&output, vec![0x5a; 4 << 20])?;
                let mut source = ArtifactTreeSource::new();
                source.insert_file("target/debug/deps/harness", &output)?;
                let SourceEntry::File { pinned, .. } = source
                    .entries
                    .get(Path::new("target/debug/deps/harness"))
                    .expect("captured stable Cargo output")
                else {
                    panic!("stable Cargo output was not captured as a file")
                };
                pinned_identity.set(Some(pinned.identity));
                super::super::content::reset_test_content_hash_read_count(pinned.identity);
                Ok(source)
            },
        )
        .unwrap();

    let pinned_identity = pinned_identity.get().expect("pinned output identity");
    assert_eq!(
        super::super::content::test_content_hash_read_count(pinned_identity),
        1,
        "the cold producer must hash each pinned output once before sealing changes ctime",
    );
    let stable_output = stable_parent
        .join(format!("{identity:016x}"))
        .join("target/debug/deps/harness");
    assert_eq!(
        std::fs::metadata(&stable_output)
            .unwrap()
            .permissions()
            .mode()
            & 0o222,
        0,
        "publication must still leave the stable Cargo output sealed",
    );
    assert_eq!(
        std::fs::read(tree.root().join("target/debug/deps/harness")).unwrap(),
        vec![0x5a; 4 << 20],
    );
}

#[test]
fn stable_cargo_record_recovers_from_absent_and_incomplete_output_anchor() {
    let _environment = lock_env();
    let temp = tempfile::tempdir().unwrap();
    let _cache = EnvVarGuard::set(crate::KTSTR_CACHE_DIR_ENV, temp.path().join("cas"));
    let records = temp.path().join("records");
    let cache = ArtifactTreeCache::new(&records);
    let stable_parent = temp.path().join("stable");
    let materializations = temp.path().join("materializations");
    let identity = 0x5ea1_5afe;
    let builds = AtomicUsize::new(0);
    let build = |stable: &StableCargoBuild| -> Result<ArtifactTreeSource> {
        builds.fetch_add(1, Ordering::SeqCst);
        let output = stable.target_directory.join("debug/deps/harness");
        std::fs::create_dir_all(output.parent().unwrap())?;
        std::fs::write(&output, b"recoverable-stable-output")?;
        let mut source = ArtifactTreeSource::new();
        source.insert_file("target/debug/deps/harness", output)?;
        Ok(source)
    };

    let first = cache
        .load_or_build_with_stable_cargo_output(
            identity,
            &stable_parent,
            &materializations,
            "stable-cargo-recovery-test",
            || Ok(true),
            || false,
            |stable| build(stable),
        )
        .unwrap();
    assert!(!first.cache_hit());
    drop(first);
    let (record, _, _) = cache_paths(&records, identity);
    assert!(record.is_file());
    let stable_root = stable_parent.join(format!("{identity:016x}"));

    remove_stable_tree(&stable_root).unwrap();
    let absent_recovery = cache
        .load_or_build_with_stable_cargo_output(
            identity,
            &stable_parent,
            &materializations,
            "stable-cargo-recovery-test",
            || Ok(true),
            || false,
            |stable| build(stable),
        )
        .expect("an absent stable anchor must be a reconstructible miss");
    assert!(!absent_recovery.cache_hit());
    assert_eq!(
        std::fs::read(absent_recovery.root().join("target/debug/deps/harness")).unwrap(),
        b"recoverable-stable-output",
    );
    drop(absent_recovery);

    remove_stable_tree(&stable_root).unwrap();
    std::fs::create_dir_all(stable_root.join("target")).unwrap();
    std::fs::write(stable_root.join("partial"), b"killed producer").unwrap();
    let incomplete_recovery = cache
        .load_or_build_with_stable_cargo_output(
            identity,
            &stable_parent,
            &materializations,
            "stable-cargo-recovery-test",
            || Ok(true),
            || false,
            |stable| build(stable),
        )
        .expect("an incomplete stable anchor must re-enter producer election");
    assert!(!incomplete_recovery.cache_hit());
    assert!(!stable_root.join("partial").exists());
    assert_eq!(builds.load(Ordering::SeqCst), 3);
    drop(incomplete_recovery);
    remove_stable_tree(&stable_root).unwrap();
}

#[test]
fn stable_cargo_distillation_preserves_exact_closure_without_following_outside_links() {
    let _environment = lock_env();
    let temp = tempfile::tempdir().unwrap();
    let _cache = EnvVarGuard::set(crate::KTSTR_CACHE_DIR_ENV, temp.path().join("cas"));
    let cache = ArtifactTreeCache::new(temp.path().join("records"));
    let stable_parent = temp.path().join("stable");
    let materializations = temp.path().join("materializations");
    let outside = temp.path().join("outside");
    std::fs::create_dir(&outside).unwrap();
    std::fs::write(outside.join("sentinel"), b"outside-owned").unwrap();
    let identity = 0x5ea1_d157;

    let tree = cache
        .load_or_build_with_stable_cargo_output(
            identity,
            &stable_parent,
            &materializations,
            "stable-cargo-distillation-test",
            || Ok(true),
            || false,
            |stable| {
                let harness = stable.target_directory.join("debug/deps/harness");
                let out_file = stable.root.join("build/package/out/generated");
                std::fs::create_dir_all(harness.parent().unwrap())?;
                std::fs::create_dir_all(out_file.parent().unwrap())?;
                std::fs::write(&harness, b"recorded-harness")?;
                std::fs::write(&out_file, b"recorded-out-dir")?;
                std::os::unix::fs::symlink("harness", harness.with_file_name("current"))?;

                let incremental = stable.target_directory.join("debug/incremental/junk");
                std::fs::create_dir_all(&incremental)?;
                std::fs::write(incremental.join("unrecorded"), b"discard-me")?;
                std::fs::create_dir_all(stable.root.join("build/unrelated"))?;
                std::fs::write(stable.root.join("build/unrelated/junk"), b"discard-me")?;
                std::os::unix::fs::symlink(&outside, stable.target_directory.join("outside-link"))?;

                let mut source = ArtifactTreeSource::new();
                source.insert_file("target/debug/deps/harness", &harness)?;
                source.insert_file("build/package/out/generated", &out_file)?;
                source.insert_symlink("target/debug/deps/current", "harness")?;
                source.insert_bytes("meta/runtime.json", b"{\"exact\":true}", 0o444)?;
                Ok(source)
            },
        )
        .unwrap();

    let stable_root = stable_parent.join(format!("{identity:016x}"));
    assert_eq!(
        std::fs::read(stable_root.join("target/debug/deps/harness")).unwrap(),
        b"recorded-harness",
    );
    assert_eq!(
        std::fs::read(stable_root.join("build/package/out/generated")).unwrap(),
        b"recorded-out-dir",
    );
    assert_eq!(
        std::fs::read_link(stable_root.join("target/debug/deps/current")).unwrap(),
        Path::new("harness"),
    );
    assert!(!stable_root.join("target/debug/incremental").exists());
    assert!(!stable_root.join("build/unrelated").exists());
    assert!(!stable_root.join("target/outside-link").exists());
    assert_eq!(
        std::fs::read(outside.join("sentinel")).unwrap(),
        b"outside-owned",
        "distillation must unlink only inside its no-follow-opened stable root",
    );
    assert!(
        !stable_root.join("meta/runtime.json").exists(),
        "cache metadata is part of the private execution closure, not the absolute Cargo anchor",
    );
    assert_eq!(
        std::fs::read(tree.root().join("meta/runtime.json")).unwrap(),
        b"{\"exact\":true}",
    );
    assert_eq!(
        std::fs::read(tree.root().join("target/debug/deps/harness")).unwrap(),
        b"recorded-harness",
    );
    assert_eq!(
        std::fs::read(tree.root().join("build/package/out/generated")).unwrap(),
        b"recorded-out-dir",
    );
    assert!(!tree.root().join("target/debug/incremental").exists());
    assert_eq!(
        std::fs::metadata(stable_root.join("target/debug/deps/harness"))
            .unwrap()
            .permissions()
            .mode()
            & 0o222,
        0,
    );
    remove_stable_tree(&stable_root).unwrap();
}

#[test]
fn stable_cargo_sealing_is_recursive_and_does_not_follow_symlinks() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().join("stable");
    let target = root.join("target");
    let nested = target.join("debug/deps");
    std::fs::create_dir_all(&nested).unwrap();
    let first = nested.join("first");
    let second = nested.join("second");
    std::fs::write(&first, b"first").unwrap();
    std::fs::write(&second, b"second").unwrap();
    std::fs::set_permissions(&first, std::fs::Permissions::from_mode(0o766)).unwrap();
    std::fs::set_permissions(&second, std::fs::Permissions::from_mode(0o777)).unwrap();
    for directory in [&root, &target, &nested] {
        std::fs::set_permissions(directory, std::fs::Permissions::from_mode(0o777)).unwrap();
    }
    let outside = temp.path().join("outside");
    std::fs::write(&outside, b"outside").unwrap();
    std::fs::set_permissions(&outside, std::fs::Permissions::from_mode(0o666)).unwrap();
    std::os::unix::fs::symlink(&outside, nested.join("outside-link")).unwrap();

    seal_stable_cargo_build(
        &StableCargoBuild {
            root: root.clone(),
            target_directory: target.clone(),
        },
        0x5ea1_ed,
    )
    .unwrap();

    for path in [&root, &target, &nested, &first, &second] {
        assert_eq!(
            std::fs::metadata(path).unwrap().permissions().mode() & 0o222,
            0,
            "stable Cargo node remained writable: {}",
            path.display(),
        );
    }
    assert!(
        std::fs::symlink_metadata(nested.join("outside-link"))
            .unwrap()
            .file_type()
            .is_symlink(),
    );
    assert_eq!(
        std::fs::metadata(&outside).unwrap().permissions().mode() & 0o777,
        0o666,
        "sealing must never chmod through a symlink",
    );
    assert_eq!(
        std::fs::read_to_string(root.join(STABLE_BUILD_MARKER)).unwrap(),
        "00000000005ea1ed\n",
    );
    remove_stable_tree(&root).unwrap();
}

#[test]
fn stable_tree_survives_owner_drop_and_is_immutable_on_reuse() {
    let _environment = lock_env();
    let temp = tempfile::tempdir().unwrap();
    let _cache = EnvVarGuard::set(crate::KTSTR_CACHE_DIR_ENV, temp.path().join("cas"));
    let input = temp.path().join("input");
    std::fs::write(&input, b"stable-source").unwrap();
    let records = temp.path().join("records");
    let stable = temp.path().join("stable");
    let cache = ArtifactTreeCache::new(&records);
    let first = cache
        .load_or_build_stable(
            0x57ab_1e,
            &stable,
            "stable-tree-test",
            || Ok(true),
            || false,
            || {
                let mut source = ArtifactTreeSource::new();
                source.insert_immutable_path("source/file", &input)?;
                Ok(source)
            },
        )
        .unwrap();
    let root = first.root().to_path_buf();
    assert_eq!(
        std::fs::read(root.join("source/file")).unwrap(),
        b"stable-source"
    );
    assert_eq!(
        std::fs::metadata(root.join("source/file"))
            .unwrap()
            .permissions()
            .mode()
            & 0o222,
        0
    );
    drop(first);
    assert!(root.exists());

    let second = cache
        .load_or_build_stable(
            0x57ab_1e,
            &stable,
            "stable-tree-test",
            || Ok(true),
            || false,
            || -> Result<ArtifactTreeSource> { panic!("stable hit invoked builder") },
        )
        .unwrap();
    assert!(second.cache_hit());
    assert_eq!(second.root(), root);
}

const CHILD_TEST: &str = "cache::artifact_tree::tests::artifact_tree_cache_cross_process_child";
const CHILD_ROOT: &str = "KTSTR_ARTIFACT_TREE_CHILD_ROOT";
const CHILD_INDEX: &str = "KTSTR_ARTIFACT_TREE_CHILD_INDEX";
const CHILD_READY: &str = "KTSTR_ARTIFACT_TREE_CHILD_READY";
const CHILD_START: &str = "KTSTR_ARTIFACT_TREE_CHILD_START";
const CHILD_COUNTER: &str = "KTSTR_ARTIFACT_TREE_CHILD_COUNTER";
const CHILD_RESULTS: &str = "KTSTR_ARTIFACT_TREE_CHILD_RESULTS";

#[test]
fn artifact_tree_cache_cross_process_child() {
    let Some(root) = std::env::var_os(CHILD_ROOT).map(PathBuf::from) else {
        return;
    };
    let index = std::env::var(CHILD_INDEX)
        .expect("child index")
        .parse::<usize>()
        .expect("parse child index");
    let ready = PathBuf::from(std::env::var_os(CHILD_READY).expect("child ready directory"));
    let start = PathBuf::from(std::env::var_os(CHILD_START).expect("child start lock"));
    let counter = PathBuf::from(std::env::var_os(CHILD_COUNTER).expect("child counter"));
    let results = PathBuf::from(std::env::var_os(CHILD_RESULTS).expect("child results"));
    // SAFETY: this subprocess runs one exact test with one libtest thread.
    unsafe {
        std::env::set_var(crate::KTSTR_CACHE_DIR_ENV, root.join("cache"));
    }
    let start = std::fs::File::open(start).expect("open child start barrier");
    std::fs::write(ready.join(index.to_string()), b"ready").expect("publish child ready");
    rustix::fs::flock(&start, rustix::fs::FlockOperation::LockShared)
        .expect("wait for child start");

    let source_root = root.join(format!("source-{index}"));
    write_fixture_tree(&source_root, b"one-cross-process-tree");
    let cache_root = super::super::cargo_artifact_tree_cache_root()
        .expect("resolve shared artifact-tree cache root");
    let tree = ArtifactTreeCache::new(&cache_root)
        .load_or_build(
            0xc205_e1ec7,
            &cache_root.join("materializations"),
            "artifact-tree-election-test",
            || Ok(true),
            || false,
            || {
                let mut attempts = std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(&counter)?;
                writeln!(attempts, "builder-{index}")?;
                attempts.sync_all()?;
                std::thread::sleep(Duration::from_millis(100));
                Ok(source_from_fixture(&source_root))
            },
        )
        .expect("cross-process artifact tree");
    std::fs::write(
        results.join(index.to_string()),
        std::fs::read(tree.root().join("target/debug/deps/harness-current")).unwrap(),
    )
    .expect("write child result");
}

#[test]
fn artifact_tree_cache_elects_one_cross_process_builder() {
    const CHILDREN: usize = 6;

    let temp = tempfile::tempdir().expect("artifact election test root");
    let root = temp.path().join("shared");
    let ready = temp.path().join("ready");
    let results = temp.path().join("results");
    std::fs::create_dir_all(&root).unwrap();
    std::fs::create_dir(&ready).unwrap();
    std::fs::create_dir(&results).unwrap();
    let start_path = temp.path().join("start");
    let start = std::fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(&start_path)
        .unwrap();
    rustix::fs::flock(&start, rustix::fs::FlockOperation::LockExclusive).unwrap();
    let counter = temp.path().join("builds");
    let executable = std::env::current_exe().expect("current test executable");
    let mut children = (0..CHILDREN)
        .map(|index| {
            std::process::Command::new(&executable)
                .arg("--exact")
                .arg(CHILD_TEST)
                .arg("--nocapture")
                .arg("--test-threads=1")
                .env(CHILD_ROOT, &root)
                .env(CHILD_INDEX, index.to_string())
                .env(CHILD_READY, &ready)
                .env(CHILD_START, &start_path)
                .env(CHILD_COUNTER, &counter)
                .env(CHILD_RESULTS, &results)
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::inherit())
                .spawn()
                .expect("spawn artifact cache child")
        })
        .collect::<Vec<_>>();
    let deadline = Instant::now() + Duration::from_secs(15);
    while std::fs::read_dir(&ready).unwrap().count() != CHILDREN {
        for child in &mut children {
            if let Some(status) = child.try_wait().unwrap() {
                panic!("artifact cache child exited before barrier: {status}");
            }
        }
        assert!(
            Instant::now() < deadline,
            "artifact cache children did not reach barrier"
        );
        std::thread::yield_now();
    }
    rustix::fs::flock(&start, rustix::fs::FlockOperation::Unlock).unwrap();
    for child in &mut children {
        assert!(child.wait().unwrap().success());
    }
    let attempts = std::fs::read_to_string(counter).expect("read build attempts");
    assert_eq!(attempts.lines().count(), 1);
    let outputs = std::fs::read_dir(results)
        .unwrap()
        .map(|entry| std::fs::read(entry.unwrap().path()).unwrap())
        .collect::<Vec<_>>();
    assert_eq!(outputs.len(), CHILDREN);
    assert!(
        outputs
            .iter()
            .all(|output| output == b"one-cross-process-tree")
    );
}
