//! Unit tests for the scenario::snapshot subsystem. Co-located with
//! production code via the `tests` submodule pattern — tests are
//! `super::*` consumers and use the synthetic [`FailureDumpReport`]
//! built by [`synthetic_report`] to exercise every accessor branch
//! without booting a VM.

#![cfg(test)]

use super::view::render_entry_key;
use super::*;
use crate::monitor::arena::ArenaSnapshot;
use crate::monitor::bpf_prog::ProgRuntimeStats;
use crate::monitor::btf_render::{RenderedMember, RenderedValue};
use crate::monitor::dump::{
    EventCounterSample, FailureDumpEntry, FailureDumpFdArray, FailureDumpMap,
    FailureDumpPercpuEntry, FailureDumpPercpuHashEntry, FailureDumpReport, FailureDumpRingbuf,
    FailureDumpStackTrace, PerCpuTimeStats, PerNodeNumaStats, ProbeBssCounters, SCHEMA_SINGLE,
};
use crate::monitor::task_enrichment::TaskEnrichment;
use crate::sync::MutexExt;
use std::sync::Arc;

/// Build a synthetic [`FailureDumpReport`] used by every
/// accessor unit test below.
fn synthetic_report() -> FailureDumpReport {
    let bss_value = RenderedValue::Struct {
        type_name: Some(".bss".into()),
        members: vec![
            RenderedMember {
                name: "nr_cpus_onln".into(),
                value: RenderedValue::Uint { bits: 32, value: 4 },
            },
            RenderedMember {
                name: "stall".into(),
                value: RenderedValue::Uint { bits: 8, value: 1 },
            },
            RenderedMember {
                name: "balance_factor".into(),
                value: RenderedValue::Float {
                    bits: 64,
                    value: 1.5,
                },
            },
            RenderedMember {
                name: "ctx".into(),
                value: RenderedValue::Struct {
                    type_name: Some("scx_ctx".into()),
                    members: vec![
                        RenderedMember {
                            name: "weight".into(),
                            value: RenderedValue::Uint {
                                bits: 32,
                                value: 1024,
                            },
                        },
                        RenderedMember {
                            name: "policy".into(),
                            value: RenderedValue::Enum {
                                bits: 32,
                                value: 1,
                                variant: Some("SCHED_NORMAL".into()),
                                is_signed: false,
                            },
                        },
                    ],
                },
            },
            RenderedMember {
                name: "leader".into(),
                value: RenderedValue::Ptr {
                    value: 0xffff_8000_0000_1000,
                    deref: Some(Box::new(RenderedValue::Struct {
                        type_name: Some("task_struct".into()),
                        members: vec![RenderedMember {
                            name: "pid".into(),
                            value: RenderedValue::Int {
                                bits: 32,
                                value: 1234,
                            },
                        }],
                    })),
                    deref_skipped_reason: None,
                    cast_annotation: None,
                },
            },
        ],
    };
    let bss_map = FailureDumpMap {
        name: "bpf.bss".into(),
        map_kva: 0,
        map_type: 2,
        value_size: 32,
        max_entries: 1,
        value: Some(bss_value),
        entries: Vec::new(),
        array_entries: Vec::new(),
        percpu_entries: Vec::new(),
        percpu_hash_entries: Vec::new(),
        arena: None,
        ringbuf: None,
        stack_trace: None,
        fd_array: None,
        error: None,
    };
    let hash_map = FailureDumpMap {
        name: "scx_per_task".into(),
        map_kva: 0,
        map_type: 1,
        value_size: 8,
        max_entries: 16,
        value: None,
        entries: vec![
            FailureDumpEntry {
                key: Some(RenderedValue::Uint {
                    bits: 32,
                    value: 100,
                }),
                key_hex: "64000000".into(),
                value: Some(RenderedValue::Struct {
                    type_name: Some("task_ctx".into()),
                    members: vec![
                        RenderedMember {
                            name: "tid".into(),
                            value: RenderedValue::Int {
                                bits: 32,
                                value: 100,
                            },
                        },
                        RenderedMember {
                            name: "runtime_ns".into(),
                            value: RenderedValue::Uint {
                                bits: 64,
                                value: 5_000_000,
                            },
                        },
                    ],
                }),
                value_hex: "0064000000000000".into(),
                payload: None,
            },
            FailureDumpEntry {
                key: Some(RenderedValue::Uint {
                    bits: 32,
                    value: 200,
                }),
                key_hex: "c8000000".into(),
                value: Some(RenderedValue::Struct {
                    type_name: Some("task_ctx".into()),
                    members: vec![
                        RenderedMember {
                            name: "tid".into(),
                            value: RenderedValue::Int {
                                bits: 32,
                                value: 200,
                            },
                        },
                        RenderedMember {
                            name: "runtime_ns".into(),
                            value: RenderedValue::Uint {
                                bits: 64,
                                value: 9_000_000,
                            },
                        },
                    ],
                }),
                value_hex: "00c8000000000000".into(),
                payload: None,
            },
        ],
        array_entries: Vec::new(),
        percpu_entries: Vec::new(),
        percpu_hash_entries: Vec::new(),
        arena: None,
        ringbuf: None,
        stack_trace: None,
        fd_array: None,
        error: None,
    };
    let percpu_map = FailureDumpMap {
        name: "scx_pcpu".into(),
        map_kva: 0,
        map_type: 6,
        value_size: 8,
        max_entries: 1,
        value: None,
        entries: Vec::new(),
        array_entries: Vec::new(),
        percpu_entries: vec![FailureDumpPercpuEntry {
            key: 0,
            per_cpu: vec![
                Some(RenderedValue::Uint {
                    bits: 64,
                    value: 11,
                }),
                Some(RenderedValue::Uint {
                    bits: 64,
                    value: 22,
                }),
                None,
                Some(RenderedValue::Uint {
                    bits: 64,
                    value: 44,
                }),
            ],
        }],
        percpu_hash_entries: Vec::new(),
        arena: None,
        ringbuf: None,
        stack_trace: None,
        fd_array: None,
        error: None,
    };
    FailureDumpReport {
        schema: SCHEMA_SINGLE.to_string(),
        active_map_kvas: Vec::new(),
        maps: vec![bss_map, hash_map, percpu_map],
        ..Default::default()
    }
}

#[test]
fn snapshot_var_walks_into_bss_struct() {
    let r = synthetic_report();
    let snap = Snapshot::new(&r);
    assert_eq!(snap.var("nr_cpus_onln").as_u64().unwrap(), 4);
    assert!(snap.var("stall").as_bool().unwrap());
    assert!((snap.var("balance_factor").as_f64().unwrap() - 1.5).abs() < f64::EPSILON);
}

/// `SnapshotField::as_bool` returns `*value != 0`; the false side of
/// that contract is otherwise unguarded. `stall = 1` above only catches
/// an inverted `== 0` predicate, NOT an always-true bug (e.g. returning
/// `Ok(true)` unconditionally), which would misreport a zero-valued flag
/// — the worst false positive for a stall flag. Pin the zero → false
/// case directly.
#[test]
fn snapshot_var_as_bool_false_for_zero_scalar() {
    let report = make_report_with_maps(vec![make_global_map(
        "scx_test.bss",
        vec![("balanced", uint_v(0))],
    )]);
    let snap = Snapshot::new(&report);
    assert!(
        !snap.var("balanced").as_bool().unwrap(),
        "a zero-valued scalar must read false, not be reported true",
    );
}

#[test]
fn snapshot_var_dotted_path_walks_nested_struct() {
    let r = synthetic_report();
    let snap = Snapshot::new(&r);
    assert_eq!(snap.var("ctx").get("weight").as_u64().unwrap(), 1024);
    assert_eq!(
        snap.var("ctx").get("policy").as_str().unwrap(),
        "SCHED_NORMAL"
    );
    assert_eq!(snap.var("ctx").get("policy").as_i64().unwrap(), 1);
}

#[test]
fn dotted_path_follows_ptr_deref_transparently() {
    let r = synthetic_report();
    let snap = Snapshot::new(&r);
    assert_eq!(snap.var("leader").get("pid").as_i64().unwrap(), 1234);
}

#[test]
fn missing_var_lists_available_globals() {
    let r = synthetic_report();
    let snap = Snapshot::new(&r);
    let f = snap.var("absent");
    let err = f.error().expect("missing field carries an error");
    match err {
        SnapshotError::VarNotFound {
            requested,
            available,
        } => {
            assert_eq!(requested, "absent");
            assert!(available.contains(&"nr_cpus_onln".to_string()));
            assert!(available.contains(&"ctx".to_string()));
        }
        other => panic!("unexpected error variant: {other:?}"),
    }
    assert!(f.as_u64().is_err());
    assert!(f.as_i64().is_err());
    assert!(f.as_bool().is_err());
}

/// Pin the `Snapshot::var` ambiguity-detection invariant: when
/// two global-section maps expose a top-level member with the
/// same name, var() MUST surface AmbiguousVar with both map
/// names rather than silently returning the first match. The
/// previous first-match behavior depended on `report.maps`
/// ordering which mirrors kernel IDR allocation order — a
/// non-deterministic source. Regression: removing the
/// hits.len() > 1 arm or short-circuiting on first hit would
/// surface here as an `Ok` SnapshotField::Value with no error.
#[test]
fn snapshot_var_ambiguity_lists_every_match() {
    let mut r = synthetic_report();
    // Add a second .data global-section map that ALSO exposes a
    // top-level `nr_cpus_onln` member. The synthetic report
    // already contains `bpf.bss` with `nr_cpus_onln`; with two
    // maps exposing the name, var() must error.
    let dup_value = RenderedValue::Struct {
        type_name: Some(".data".into()),
        members: vec![RenderedMember {
            name: "nr_cpus_onln".into(),
            value: RenderedValue::Uint {
                bits: 32,
                value: 99,
            },
        }],
    };
    r.maps.push(FailureDumpMap {
        name: "other.data".into(),
        map_kva: 0,
        map_type: 2,
        value_size: 32,
        max_entries: 1,
        value: Some(dup_value),
        entries: Vec::new(),
        array_entries: Vec::new(),
        percpu_entries: Vec::new(),
        percpu_hash_entries: Vec::new(),
        arena: None,
        ringbuf: None,
        stack_trace: None,
        fd_array: None,
        error: None,
    });
    let snap = Snapshot::new(&r);
    let f = snap.var("nr_cpus_onln");
    let err = f
        .error()
        .expect("duplicate global must surface AmbiguousVar");
    match err {
        SnapshotError::AmbiguousVar {
            requested,
            found_in,
        } => {
            assert_eq!(requested, "nr_cpus_onln");
            assert!(
                found_in.contains(&"bpf.bss".to_string()),
                "first map must appear in found_in: {found_in:?}",
            );
            assert!(
                found_in.contains(&"other.data".to_string()),
                "second map must appear in found_in: {found_in:?}",
            );
            assert_eq!(
                found_in.len(),
                2,
                "AmbiguousVar must list every map where the name was found, no more no less: {found_in:?}",
            );
        }
        other => panic!("expected AmbiguousVar, got: {other:?}"),
    }
    // Display must mention both map names so the test author
    // can pick the right disambiguation target.
    let rendered = err.to_string();
    assert!(rendered.contains("nr_cpus_onln"), "{rendered}");
    assert!(rendered.contains("bpf.bss"), "{rendered}");
    assert!(rendered.contains("other.data"), "{rendered}");
    // Caller can disambiguate via map() — verify both maps
    // resolve independently.
    let bss = snap
        .map("bpf.bss")
        .unwrap()
        .at(0)
        .get("nr_cpus_onln")
        .as_u64()
        .unwrap();
    let data = snap
        .map("other.data")
        .unwrap()
        .at(0)
        .get("nr_cpus_onln")
        .as_u64()
        .unwrap();
    assert_eq!(bss, 4);
    assert_eq!(data, 99);
}

#[test]
fn missing_field_in_struct_lists_available_members() {
    let r = synthetic_report();
    let snap = Snapshot::new(&r);
    let f = snap.var("ctx").get("nonexistent");
    let err = f.error().expect("missing field carries an error");
    match err {
        SnapshotError::FieldNotFound {
            component,
            available,
            ..
        } => {
            assert_eq!(component, "nonexistent");
            assert!(available.contains(&"weight".to_string()));
            assert!(available.contains(&"policy".to_string()));
        }
        other => panic!("unexpected error variant: {other:?}"),
    }
}

#[test]
fn missing_map_lists_available_maps() {
    let r = synthetic_report();
    let snap = Snapshot::new(&r);
    let err = snap.map("does_not_exist").unwrap_err();
    match err {
        SnapshotError::MapNotFound {
            requested,
            available,
        } => {
            assert_eq!(requested, "does_not_exist");
            assert!(available.contains(&"bpf.bss".to_string()));
            assert!(available.contains(&"scx_per_task".to_string()));
            assert!(available.contains(&"scx_pcpu".to_string()));
        }
        other => panic!("unexpected error variant: {other:?}"),
    }
}

#[test]
fn empty_path_component_returns_error() {
    let r = synthetic_report();
    let snap = Snapshot::new(&r);
    let f = snap.var("ctx").get("weight..value");
    match f.error().expect("missing carries error") {
        SnapshotError::EmptyPathComponent { requested } => {
            assert_eq!(requested, "weight..value");
        }
        other => panic!("unexpected error: {other:?}"),
    }
}

#[test]
fn wrong_kind_at_path_step_explains() {
    let r = synthetic_report();
    let snap = Snapshot::new(&r);
    let f = snap.var("ctx").get("weight").get("inner");
    match f.error().expect("missing carries error") {
        SnapshotError::NotAStruct { kind, .. } => {
            assert_eq!(*kind, "Uint");
        }
        other => panic!("unexpected error: {other:?}"),
    }
}

#[test]
fn map_at_returns_hash_entry() {
    let r = synthetic_report();
    let snap = Snapshot::new(&r);
    let entry = snap.map("scx_per_task").unwrap().at(0);
    assert!(entry.is_present());
    assert_eq!(entry.get("tid").as_i64().unwrap(), 100);
    assert_eq!(entry.get("runtime_ns").as_u64().unwrap(), 5_000_000);
}

#[test]
fn map_at_out_of_range_carries_index_and_len() {
    let r = synthetic_report();
    let snap = Snapshot::new(&r);
    let entry = snap.map("scx_per_task").unwrap().at(99);
    match entry {
        SnapshotEntry::Missing(SnapshotError::IndexOutOfRange { index, len, .. }) => {
            assert_eq!(index, 99);
            assert_eq!(len, 2);
        }
        other => panic!("unexpected entry: present={}", other.is_present()),
    }
}

#[test]
fn map_find_returns_first_match() {
    let r = synthetic_report();
    let snap = Snapshot::new(&r);
    let map = snap.map("scx_per_task").unwrap();
    let entry = map.find(|e| e.get("tid").as_i64().unwrap_or(-1) == 200);
    assert!(entry.is_present());
    assert_eq!(entry.get("runtime_ns").as_u64().unwrap(), 9_000_000);
}

#[test]
fn map_find_no_match_carries_op_name_len_and_keys() {
    let r = synthetic_report();
    let snap = Snapshot::new(&r);
    let map = snap.map("scx_per_task").unwrap();
    let entry = map.find(|e| e.get("tid").as_i64().unwrap_or(-1) == 999);
    match entry {
        SnapshotEntry::Missing(SnapshotError::NoMatch {
            op,
            len,
            available_keys,
            ..
        }) => {
            assert_eq!(op, "find");
            // synthetic_report's scx_per_task has 2 entries (tid=100 and
            // tid=200) — every traversal that exits via NoMatch must
            // have walked all 2.
            assert_eq!(len, 2);
            // 2 entries, cap is NO_MATCH_KEY_SAMPLE=3, so we keep both.
            assert_eq!(available_keys.len(), 2);
        }
        other => panic!("expected NoMatch, got present={}", other.is_present()),
    }
}

#[test]
fn map_max_by_no_match_reports_empty_map() {
    // Build a fixture whose scx_per_task map has zero entries so
    // max_by's NoMatch arm fires. This is the only path through
    // max_by that produces NoMatch — non-empty maps always have
    // a maximum.
    let mut r = synthetic_report();
    for m in r.maps.iter_mut() {
        if m.name == "scx_per_task" {
            m.entries.clear();
        }
    }
    let snap = Snapshot::new(&r);
    let map = snap.map("scx_per_task").unwrap();
    let entry = map.max_by(|e| e.get("runtime_ns").as_u64().unwrap_or(0));
    match entry {
        SnapshotEntry::Missing(SnapshotError::NoMatch {
            op,
            len,
            available_keys,
            ..
        }) => {
            assert_eq!(op, "max_by");
            assert_eq!(len, 0);
            assert!(available_keys.is_empty());
        }
        other => panic!("expected NoMatch, got present={}", other.is_present()),
    }
}

/// Display rendering must surface `len` and `available_keys` for
/// each of the three cases — empty map, populated map without
/// sampled keys (all keys were unrenderable), and populated map
/// with sampled keys. Without this pin, the Display impl could
/// silently drop the new fields and every structural test would
/// still pass.
#[test]
fn no_match_display_renders_three_arms() {
    let empty = SnapshotError::NoMatch {
        map: "m".to_string(),
        op: "find".to_string(),
        len: 0,
        available_keys: Vec::new(),
    };
    let rendered = format!("{empty}");
    assert!(rendered.contains("'m'"), "{rendered}");
    assert!(rendered.contains("empty"), "{rendered}");

    let unrendered = SnapshotError::NoMatch {
        map: "m".to_string(),
        op: "max_by".to_string(),
        len: 7,
        available_keys: Vec::new(),
    };
    let rendered = format!("{unrendered}");
    assert!(rendered.contains("'m'"), "{rendered}");
    assert!(rendered.contains("7"), "{rendered}");
    assert!(rendered.contains("unavailable"), "{rendered}");

    let sampled = SnapshotError::NoMatch {
        map: "m".to_string(),
        op: "find".to_string(),
        len: 9,
        available_keys: vec!["k0".to_string(), "k1".to_string(), "k2".to_string()],
    };
    let rendered = format!("{sampled}");
    assert!(rendered.contains("'m'"), "{rendered}");
    assert!(rendered.contains("9"), "{rendered}");
    assert!(rendered.contains("k0"), "{rendered}");
    assert!(rendered.contains("k2"), "{rendered}");
}

/// `render_entry_key` must truncate wide struct keys (e.g.
/// a 50-field struct) to [`NO_MATCH_KEY_CHAR_CAP`] chars with a
/// trailing `…`. Without the cap a single wide-struct key would
/// blow the failure-message size budget.
#[test]
fn render_entry_key_caps_wide_struct_keys() {
    let oversized = "x".repeat(NO_MATCH_KEY_CHAR_CAP * 4);
    let entry_fixture = FailureDumpEntry {
        key: None,
        key_hex: oversized.clone(),
        value: None,
        value_hex: String::new(),
        payload: None,
    };
    let entry = SnapshotEntry::Hash(&entry_fixture);
    let rendered = render_entry_key(&entry).expect("hash entry has a key");
    assert!(rendered.chars().count() <= NO_MATCH_KEY_CHAR_CAP);
    assert!(rendered.ends_with('…'));
    assert!(rendered.starts_with("hex:x"));
}

/// Internal-contract pin: when a HASH entry's `key` field is
/// `None` (BTF was missing at capture for the map's key type),
/// `render_entry_key` must fall back to a string that BEGINS
/// with `"hex:"` followed by the raw `key_hex` bytes. The
/// BTF-missing-hint detection on the NoMatch Display arm uses
/// `available_keys.iter().all(|k| k.starts_with("hex:"))` as
/// the discriminator — a silent change to the prefix (drop,
/// rename to `"0x"`, etc.) would break the hint without
/// surfacing in any sibling test. `key_hex` is space-separated
/// hex pairs per `monitor::dump::hex_dump`, so the fixture
/// uses that format to match what production wire data looks
/// like.
#[test]
fn render_entry_key_hash_fallback_uses_hex_prefix() {
    let entry_fixture = FailureDumpEntry {
        key: None,
        key_hex: "de ad be ef".into(),
        value: None,
        value_hex: String::new(),
        payload: None,
    };
    let entry = SnapshotEntry::Hash(&entry_fixture);
    let rendered = render_entry_key(&entry).expect("hash entry has a key");
    assert!(
        rendered.starts_with("hex:"),
        "Hash fallback must use 'hex:' prefix that NoMatch Display \
             uses as the BTF-missing-hint discriminator; got {rendered:?}",
    );
    assert!(
        rendered.contains("de ad be ef"),
        "key_hex bytes must be preserved verbatim so the operator \
             can disambiguate keys; got {rendered:?}",
    );
}

/// Sibling of `render_entry_key_hash_fallback_uses_hex_prefix`
/// for the [`SnapshotEntry::PercpuHash`] variant. Both Hash and
/// PercpuHash entries hit the same `hex:{key_hex}` fallback in
/// `render_entry_key` when their `key` field is `None`; a
/// regression that altered the fallback shape for ONLY one of
/// the two variants would silently break the BTF-missing-hint
/// gate for that variant's map type. The pair pins the
/// internal contract for both variants explicitly.
#[test]
fn render_entry_key_percpu_hash_fallback_uses_hex_prefix() {
    let entry_fixture = FailureDumpPercpuHashEntry {
        key: None,
        key_hex: "ca fe ba be".into(),
        per_cpu: vec![None],
    };
    let entry = SnapshotEntry::PercpuHash(&entry_fixture);
    let rendered = render_entry_key(&entry).expect("percpu-hash entry has a key");
    assert!(
        rendered.starts_with("hex:"),
        "PercpuHash fallback must use 'hex:' prefix; got {rendered:?}",
    );
    assert!(
        rendered.contains("ca fe ba be"),
        "key_hex bytes must be preserved verbatim; got {rendered:?}",
    );
}

/// End-to-end pin for the `available_keys.is_empty() && len > 0`
/// Display arm: a single-value ARRAY map iterates as one
/// [`SnapshotEntry::Value`], `render_entry_key` returns None for
/// Value variants, so `find()` traverses one entry, never pushes
/// a key, and constructs NoMatch with `len = 1, available_keys
/// = []`. The Display impl's middle arm fires
/// ("sample keys unavailable"). Complements
/// [`no_match_display_renders_three_arms`] which exercises the
/// same arm via direct struct-literal construction — this test
/// proves the production find/iter_entries path actually reaches
/// it.
#[test]
fn map_find_no_match_on_single_value_array_renders_unavailable_keys() {
    let array_map = FailureDumpMap {
        name: "scx_singleton".into(),
        map_kva: 0,
        map_type: 2,
        value_size: 8,
        max_entries: 1,
        value: Some(RenderedValue::Uint {
            bits: 64,
            value: 42,
        }),
        entries: Vec::new(),
        array_entries: Vec::new(),
        percpu_entries: Vec::new(),
        percpu_hash_entries: Vec::new(),
        arena: None,
        ringbuf: None,
        stack_trace: None,
        fd_array: None,
        error: None,
    };
    let r = FailureDumpReport {
        schema: SCHEMA_SINGLE.to_string(),
        active_map_kvas: Vec::new(),
        maps: vec![array_map],
        ..Default::default()
    };
    let snap = Snapshot::new(&r);
    let entry = snap.map("scx_singleton").unwrap().find(|_| false);
    let SnapshotEntry::Missing(err) = entry else {
        panic!("expected Missing, got present={}", entry.is_present());
    };
    match &err {
        SnapshotError::NoMatch {
            op,
            len,
            available_keys,
            ..
        } => {
            assert_eq!(*op, "find");
            assert_eq!(*len, 1, "single Value entry iterated");
            assert!(
                available_keys.is_empty(),
                "Value entries are unrenderable for key sampling: {available_keys:?}",
            );
        }
        other => panic!("expected NoMatch, got {other:?}"),
    }
    let rendered = format!("{err}");
    assert!(rendered.contains("'scx_singleton'"), "{rendered}");
    assert!(rendered.contains("matched none of 1"), "{rendered}");
    assert!(rendered.contains("sample keys unavailable"), "{rendered}");
}

/// `find()` must clamp `available_keys` at
/// [`NO_MATCH_KEY_SAMPLE`] regardless of how many entries the
/// underlying map carries. Pins the cap-clamp inside `find()`
/// and the FIRST-N-preservation order (keys 0/1/2, not the
/// last 3). A regression that drops the cap-gate, or that
/// flips it from `<` to `<=`, would push a 4th key into the
/// sample and trip the `available_keys.len() ==
/// NO_MATCH_KEY_SAMPLE` assertion. A regression that reverses
/// iteration order or swaps to a random sample would trip the
/// literal-vec assertion below.
#[test]
fn map_find_no_match_caps_sampled_keys_at_no_match_key_sample() {
    const N: u32 = 10;
    let entries: Vec<FailureDumpEntry> = (0..N)
        .map(|i| FailureDumpEntry {
            key: Some(RenderedValue::Uint {
                bits: 32,
                value: u64::from(i),
            }),
            key_hex: format!("{i:08x}"),
            value: Some(RenderedValue::Uint {
                bits: 32,
                value: u64::from(i * 10),
            }),
            value_hex: format!("{:08x}", i * 10),
            payload: None,
        })
        .collect();
    let hash_map = FailureDumpMap {
        name: "scx_big".into(),
        map_kva: 0,
        map_type: 1,
        value_size: 4,
        max_entries: 64,
        value: None,
        entries,
        array_entries: Vec::new(),
        percpu_entries: Vec::new(),
        percpu_hash_entries: Vec::new(),
        arena: None,
        ringbuf: None,
        stack_trace: None,
        fd_array: None,
        error: None,
    };
    let r = FailureDumpReport {
        schema: SCHEMA_SINGLE.to_string(),
        active_map_kvas: Vec::new(),
        maps: vec![hash_map],
        ..Default::default()
    };
    let snap = Snapshot::new(&r);
    let entry = snap.map("scx_big").unwrap().find(|_| false);
    match entry {
        SnapshotEntry::Missing(SnapshotError::NoMatch {
            op,
            len,
            available_keys,
            ..
        }) => {
            assert_eq!(op, "find");
            assert_eq!(
                len,
                usize::try_from(N).unwrap(),
                "all {N} entries must be traversed before NoMatch",
            );
            assert_eq!(
                available_keys.len(),
                NO_MATCH_KEY_SAMPLE,
                "cap must clamp sample at NO_MATCH_KEY_SAMPLE",
            );
            assert_eq!(
                available_keys,
                vec!["0".to_string(), "1".to_string(), "2".to_string()],
                "first-N preservation: sample must hold the FIRST 3 keys in \
                     iteration order, not last 3 / random",
            );
        }
        other => panic!("expected NoMatch, got present={}", other.is_present()),
    }
}

/// The `find()` `available_keys` sample preserves duplicates in
/// iteration order — there is no dedup at the snapshot layer.
/// `iter_entries()` walks `self.map.entries.iter()` directly for
/// the HASH branch (and `percpu_entries.iter()` /
/// `percpu_hash_entries.iter()` for the per-CPU branches); all
/// three are raw `Vec::iter()` calls. The wire format is a `Vec`,
/// not a `Map`, so duplicate keys are syntactically legal even
/// though real BPF maps cannot produce them — and the operator's
/// failure-message contract depends on `len` reflecting EVERY
/// entry seen and `available_keys` showing the sample as
/// observed. This test pins the iter-layer no-dedup invariant; a
/// future "optimization" that adds dedup to `iter_entries()`
/// would collapse the duplicate `"100"` here and fail loudly.
#[test]
fn map_find_no_match_preserves_duplicate_keys_in_sample() {
    let hash_map = FailureDumpMap {
        name: "scx_dup".into(),
        map_kva: 0,
        map_type: 1,
        value_size: 4,
        max_entries: 16,
        value: None,
        entries: vec![
            FailureDumpEntry {
                key: Some(RenderedValue::Uint {
                    bits: 32,
                    value: 100,
                }),
                key_hex: "64000000".into(),
                value: Some(RenderedValue::Uint { bits: 32, value: 1 }),
                value_hex: "01000000".into(),
                payload: None,
            },
            FailureDumpEntry {
                key: Some(RenderedValue::Uint {
                    bits: 32,
                    value: 100,
                }),
                key_hex: "64000000".into(),
                value: Some(RenderedValue::Uint { bits: 32, value: 2 }),
                value_hex: "02000000".into(),
                payload: None,
            },
            FailureDumpEntry {
                key: Some(RenderedValue::Uint {
                    bits: 32,
                    value: 200,
                }),
                key_hex: "c8000000".into(),
                value: Some(RenderedValue::Uint { bits: 32, value: 3 }),
                value_hex: "03000000".into(),
                payload: None,
            },
            FailureDumpEntry {
                key: Some(RenderedValue::Uint {
                    bits: 32,
                    value: 300,
                }),
                key_hex: "2c010000".into(),
                value: Some(RenderedValue::Uint { bits: 32, value: 4 }),
                value_hex: "04000000".into(),
                payload: None,
            },
        ],
        array_entries: Vec::new(),
        percpu_entries: Vec::new(),
        percpu_hash_entries: Vec::new(),
        arena: None,
        ringbuf: None,
        stack_trace: None,
        fd_array: None,
        error: None,
    };
    let r = FailureDumpReport {
        schema: SCHEMA_SINGLE.to_string(),
        active_map_kvas: Vec::new(),
        maps: vec![hash_map],
        ..Default::default()
    };
    let snap = Snapshot::new(&r);
    let entry = snap.map("scx_dup").unwrap().find(|_| false);
    match entry {
        SnapshotEntry::Missing(SnapshotError::NoMatch {
            op,
            len,
            available_keys,
            ..
        }) => {
            assert_eq!(op, "find");
            assert_eq!(
                len, 4,
                "duplicate-key entries each count toward len; iter_entries \
                     Hash branch iterates the Vec directly with no dedup",
            );
            assert_eq!(available_keys.len(), NO_MATCH_KEY_SAMPLE);
            assert_eq!(
                available_keys,
                vec!["100".to_string(), "100".to_string(), "200".to_string()],
                "no dedup at sample-collection: the duplicate key '100' \
                     appears twice in iteration order before the cap fires \
                     against the third unique key",
            );
            let dup_count = available_keys
                .iter()
                .filter(|k| k.as_str() == "100")
                .count();
            assert_eq!(
                dup_count, 2,
                "position-insensitive backup pin: key '100' must appear \
                     exactly twice in available_keys (saw {dup_count} in \
                     {available_keys:?})",
            );
        }
        other => panic!("expected NoMatch, got present={}", other.is_present()),
    }
}

/// `find()`'s NoMatch Display appends a BTF-missing hint when
/// every sampled key carries the `hex:` prefix from
/// [`render_entry_key`]'s fallback path. Pins both the hint
/// substring AND that the prior arm's key listing is still
/// rendered (the hint is additive, not a replacement).
#[test]
fn no_match_display_appends_btf_hint_when_all_keys_are_hex() {
    let err = SnapshotError::NoMatch {
        map: "scx_per_task".to_string(),
        op: "find".to_string(),
        len: 5,
        available_keys: vec![
            "hex:64000000".to_string(),
            "hex:c8000000".to_string(),
            "hex:2c010000".to_string(),
        ],
    };
    let rendered = format!("{err}");
    assert!(rendered.contains("'scx_per_task'"), "{rendered}");
    assert!(rendered.contains("matched none of 5"), "{rendered}");
    assert!(
        rendered.contains("hex:64000000"),
        "underlying key list still rendered: {rendered}",
    );
    assert!(rendered.contains("BTF missing"), "{rendered}");
    assert!(rendered.contains("CONFIG_DEBUG_INFO_BTF"), "{rendered}");
}

/// The BTF-missing hint MUST NOT fire when the sample contains
/// any typed (non-`hex:`-prefixed) key. A mixed sample means BTF
/// IS present and resolving most keys; the unresolved hex ones
/// are a per-entry capture race, not a uniform-config problem,
/// so the kernel-rebuild advice would mislead the operator.
#[test]
fn no_match_display_omits_btf_hint_when_some_keys_are_typed() {
    let err = SnapshotError::NoMatch {
        map: "mixed".to_string(),
        op: "find".to_string(),
        len: 5,
        available_keys: vec![
            "hex:64000000".to_string(),
            "100".to_string(),
            "hex:2c010000".to_string(),
        ],
    };
    let rendered = format!("{err}");
    assert!(rendered.contains("hex:64000000"), "{rendered}");
    assert!(rendered.contains("100"), "{rendered}");
    assert!(
        !rendered.contains("BTF missing"),
        "BTF hint must not fire on mixed-typed sample: {rendered}",
    );
    assert!(!rendered.contains("CONFIG_DEBUG_INFO_BTF"), "{rendered}");
}

/// The BTF-missing hint MUST NOT fire on the `available_keys
/// .is_empty()` arm even though `[].iter().all(_)` returns true
/// vacuously. There is no evidence of a BTF problem when no
/// keys were sampled; the hint here would be a false positive.
#[test]
fn no_match_display_omits_btf_hint_when_no_keys_sampled() {
    let err = SnapshotError::NoMatch {
        map: "m".to_string(),
        op: "find".to_string(),
        len: 5,
        available_keys: Vec::new(),
    };
    let rendered = format!("{err}");
    assert!(rendered.contains("sample keys unavailable"), "{rendered}");
    assert!(!rendered.contains("BTF missing"), "{rendered}");
    assert!(!rendered.contains("CONFIG_DEBUG_INFO_BTF"), "{rendered}");
}

/// The BTF-missing hint MUST NOT fire on the empty-map arm
/// (`len == 0`). The empty-map message is its own dedicated
/// branch; an unconditional hint append would render the
/// operator-facing message contradictory.
#[test]
fn no_match_display_omits_btf_hint_when_map_is_empty() {
    let err = SnapshotError::NoMatch {
        map: "m".to_string(),
        op: "find".to_string(),
        len: 0,
        available_keys: Vec::new(),
    };
    let rendered = format!("{err}");
    assert!(rendered.contains("map is empty"), "{rendered}");
    assert!(!rendered.contains("BTF missing"), "{rendered}");
    assert!(!rendered.contains("CONFIG_DEBUG_INFO_BTF"), "{rendered}");
}

/// Boundary case: when entry count equals [`NO_MATCH_KEY_SAMPLE`]
/// exactly, every entry's key fits in the sample. `len ==
/// available_keys.len() == NO_MATCH_KEY_SAMPLE` guards against
/// a regression that gates the push too tightly — e.g.
/// `available_keys.len() < NO_MATCH_KEY_SAMPLE - 1` would push
/// only 2 keys at N=3 and trip the
/// `available_keys.len() == NO_MATCH_KEY_SAMPLE` assertion.
/// Pairs with
/// [`map_find_no_match_caps_sampled_keys_at_no_match_key_sample`]
/// (which catches the OTHER direction — `<=` would push a 4th
/// key at N >= NO_MATCH_KEY_SAMPLE+1) so both sides of the
/// gate-condition are pinned.
#[test]
fn map_find_no_match_cap_exact_threshold() {
    let entries: Vec<FailureDumpEntry> = (0..NO_MATCH_KEY_SAMPLE as u32)
        .map(|i| FailureDumpEntry {
            key: Some(RenderedValue::Uint {
                bits: 32,
                value: u64::from(i),
            }),
            key_hex: format!("{i:08x}"),
            value: Some(RenderedValue::Uint {
                bits: 32,
                value: u64::from(i),
            }),
            value_hex: format!("{i:08x}"),
            payload: None,
        })
        .collect();
    let hash_map = FailureDumpMap {
        name: "scx_threshold".into(),
        map_kva: 0,
        map_type: 1,
        value_size: 4,
        max_entries: 16,
        value: None,
        entries,
        array_entries: Vec::new(),
        percpu_entries: Vec::new(),
        percpu_hash_entries: Vec::new(),
        arena: None,
        ringbuf: None,
        stack_trace: None,
        fd_array: None,
        error: None,
    };
    let r = FailureDumpReport {
        schema: SCHEMA_SINGLE.to_string(),
        active_map_kvas: Vec::new(),
        maps: vec![hash_map],
        ..Default::default()
    };
    let snap = Snapshot::new(&r);
    let entry = snap.map("scx_threshold").unwrap().find(|_| false);
    match entry {
        SnapshotEntry::Missing(SnapshotError::NoMatch {
            len,
            available_keys,
            ..
        }) => {
            assert_eq!(len, NO_MATCH_KEY_SAMPLE);
            assert_eq!(available_keys.len(), NO_MATCH_KEY_SAMPLE);
        }
        other => panic!("expected NoMatch, got present={}", other.is_present()),
    }
}

#[test]
fn map_filter_collects_matches() {
    let r = synthetic_report();
    let snap = Snapshot::new(&r);
    let map = snap.map("scx_per_task").unwrap();
    let matches = map.filter(|e| e.get("runtime_ns").as_u64().unwrap_or(0) > 0);
    assert_eq!(matches.len(), 2);
}

#[test]
fn map_max_by_picks_largest() {
    let r = synthetic_report();
    let snap = Snapshot::new(&r);
    let map = snap.map("scx_per_task").unwrap();
    let busiest = map.max_by(|e| e.get("runtime_ns").as_u64().unwrap_or(0));
    assert!(busiest.is_present());
    assert_eq!(busiest.get("tid").as_i64().unwrap(), 200);
}

#[test]
fn percpu_array_cpu_narrow_reads_per_cpu_slot() {
    let r = synthetic_report();
    let snap = Snapshot::new(&r);
    let entry = snap.map("scx_pcpu").unwrap().cpu(1).at(0);
    assert!(entry.is_present());
    assert_eq!(entry.get("").as_u64().unwrap(), 22);
}

#[test]
fn percpu_array_unmapped_cpu_returns_unmapped_error() {
    let r = synthetic_report();
    let snap = Snapshot::new(&r);
    let entry = snap.map("scx_pcpu").unwrap().cpu(2).at(0);
    match entry {
        SnapshotEntry::Missing(SnapshotError::PerCpuSlot { cpu, unmapped, .. }) => {
            assert_eq!(cpu, 2);
            assert!(unmapped);
        }
        _ => panic!("expected unmapped PerCpuSlot"),
    }
}

#[test]
fn percpu_array_out_of_range_cpu_returns_oor_error() {
    let r = synthetic_report();
    let snap = Snapshot::new(&r);
    let entry = snap.map("scx_pcpu").unwrap().cpu(99).at(0);
    match entry {
        SnapshotEntry::Missing(SnapshotError::PerCpuSlot {
            cpu, unmapped, len, ..
        }) => {
            assert_eq!(cpu, 99);
            assert!(!unmapped);
            assert_eq!(len, 4);
        }
        _ => panic!("expected out-of-range PerCpuSlot"),
    }
}

#[test]
fn percpu_array_get_without_narrow_explains() {
    let r = synthetic_report();
    let snap = Snapshot::new(&r);
    let entry = snap.map("scx_pcpu").unwrap().at(0);
    let f = entry.get("anything");
    match f.error().expect("missing") {
        SnapshotError::PerCpuNotNarrowed { .. } => {}
        other => panic!("unexpected error: {other:?}"),
    }
}

/// The scx_pcpu fixture has per_cpu = [Some(11), Some(22), None,
/// Some(44)] (synthetic_report, tests.rs). Per-CPU aggregators
/// skip the None slot at index 2 and sum / max / min the others.
#[test]
fn snapshot_entry_cpu_sum_u64_skips_none_slots() {
    let r = synthetic_report();
    let snap = Snapshot::new(&r);
    let entry = snap.map("scx_pcpu").unwrap().at(0);
    assert_eq!(entry.cpu_sum_u64("").unwrap(), 11 + 22 + 44);
}

#[test]
fn snapshot_entry_cpu_max_u64_returns_largest_present() {
    let r = synthetic_report();
    let snap = Snapshot::new(&r);
    let entry = snap.map("scx_pcpu").unwrap().at(0);
    assert_eq!(entry.cpu_max_u64("").unwrap(), 44);
}

#[test]
fn snapshot_entry_cpu_min_u64_returns_smallest_present() {
    let r = synthetic_report();
    let snap = Snapshot::new(&r);
    let entry = snap.map("scx_pcpu").unwrap().at(0);
    assert_eq!(entry.cpu_min_u64("").unwrap(), 11);
}

/// `cpu_max_u64` / `cpu_min_u64` on an all-None per_cpu vec
/// return NoMatch — no slot contributes a value, so neither max
/// nor min has a meaningful answer.
#[test]
fn snapshot_entry_cpu_sum_max_min_no_slots_returns_no_match() {
    let entry_struct = FailureDumpPercpuEntry {
        key: 0,
        per_cpu: vec![None, None, None],
    };
    let entry = SnapshotEntry::Percpu(&entry_struct);
    match entry.cpu_max_u64("").expect_err("empty must err") {
        SnapshotError::NoMatch { op, len, .. } => {
            assert_eq!(op, "cpu_max_u64");
            assert_eq!(len, 0);
        }
        other => panic!("unexpected: {other:?}"),
    }
    // cpu_sum_* ALSO errors on all-None: a None slot is UNREADABLE
    // (host-read failure), not a real zero, so summing to 0 would
    // silently drop the missing data — parity with max/min, NOT a
    // sum-identity 0.
    match entry.cpu_sum_u64("").expect_err("all-None sum must err") {
        SnapshotError::NoMatch { op, len, .. } => {
            assert_eq!(op, "cpu_sum_u64");
            assert_eq!(len, 0);
        }
        other => panic!("unexpected: {other:?}"),
    }
    entry
        .cpu_sum_i64("")
        .expect_err("all-None i64 sum must err");
    entry
        .cpu_sum_f64("")
        .expect_err("all-None f64 sum must err");
}

#[test]
fn snapshot_entry_cpu_each_visits_only_present_slots() {
    let r = synthetic_report();
    let snap = Snapshot::new(&r);
    let entry = snap.map("scx_pcpu").unwrap().at(0);
    let mut visited: Vec<(usize, u64)> = Vec::new();
    entry
        .cpu_each("", |cpu, v| {
            visited.push((cpu, SnapshotField::Value(v).as_u64()?));
            Ok(())
        })
        .unwrap();
    assert_eq!(visited, vec![(0, 11), (1, 22), (3, 44)]);
}

/// Non-percpu variants — Hash, Value — produce TypeMismatch.
#[test]
fn snapshot_entry_cpu_sum_on_non_percpu_variant_errors() {
    let r = synthetic_report();
    let snap = Snapshot::new(&r);
    let bss_entry = snap.map("bpf.bss").unwrap().at(0);
    // .bss is a Value entry, not Percpu.
    match bss_entry.cpu_sum_u64("").expect_err("non-percpu must err") {
        SnapshotError::TypeMismatch {
            expected, actual, ..
        } => {
            assert!(expected.contains("Percpu"));
            assert_eq!(actual, "Value");
        }
        other => panic!("unexpected: {other:?}"),
    }
}

/// cpu_min_u64 on all-None must also return NoMatch (sibling
/// of the cpu_max_u64 empty case already pinned).
#[test]
fn snapshot_entry_cpu_min_u64_no_slots_returns_no_match() {
    let entry_struct = FailureDumpPercpuEntry {
        key: 0,
        per_cpu: vec![None, None],
    };
    let entry = SnapshotEntry::Percpu(&entry_struct);
    match entry.cpu_min_u64("").expect_err("empty must err") {
        SnapshotError::NoMatch { op, len, .. } => {
            assert_eq!(op, "cpu_min_u64");
            assert_eq!(len, 0);
        }
        other => panic!("unexpected: {other:?}"),
    }
}

fn percpu_entry_with_floats(per_cpu: Vec<Option<f64>>) -> FailureDumpPercpuEntry {
    FailureDumpPercpuEntry {
        key: 0,
        per_cpu: per_cpu
            .into_iter()
            .map(|opt| opt.map(|v| RenderedValue::Float { bits: 64, value: v }))
            .collect(),
    }
}

#[test]
fn snapshot_entry_cpu_sum_f64_skips_none_and_sums() {
    let e = percpu_entry_with_floats(vec![Some(1.5), Some(2.5), None, Some(3.0)]);
    let entry = SnapshotEntry::Percpu(&e);
    let sum = entry.cpu_sum_f64("").unwrap();
    assert!((sum - 7.0).abs() < f64::EPSILON, "sum was {sum}");
}

#[test]
fn snapshot_entry_cpu_max_f64_returns_largest() {
    let e = percpu_entry_with_floats(vec![Some(1.5), Some(9.25), Some(3.0)]);
    let entry = SnapshotEntry::Percpu(&e);
    assert!((entry.cpu_max_f64("").unwrap() - 9.25).abs() < f64::EPSILON);
}

#[test]
fn snapshot_entry_cpu_min_f64_returns_smallest() {
    let e = percpu_entry_with_floats(vec![Some(1.5), Some(9.25), Some(3.0)]);
    let entry = SnapshotEntry::Percpu(&e);
    assert!((entry.cpu_min_f64("").unwrap() - 1.5).abs() < f64::EPSILON);
}

/// NaN propagates through f64 sum (per IEEE-754 +=) but is
/// filtered out by f64::max / f64::min (per Rust stdlib
/// `maximumNumber` / `minimumNumber` semantics).
#[test]
fn snapshot_entry_cpu_f64_aggregators_handle_nan_per_docs() {
    let e = percpu_entry_with_floats(vec![Some(1.0), Some(f64::NAN), Some(3.0)]);
    let entry = SnapshotEntry::Percpu(&e);
    assert!(entry.cpu_sum_f64("").unwrap().is_nan());
    // f64::max filters NaN — the result is the max of 1.0/3.0.
    assert!((entry.cpu_max_f64("").unwrap() - 3.0).abs() < f64::EPSILON);
    // f64::min filters NaN — the result is the min of 1.0/3.0.
    assert!((entry.cpu_min_f64("").unwrap() - 1.0).abs() < f64::EPSILON);
}

/// Nested path lookup descends into a Struct rendered value.
/// Each slot's value is `{count: Uint(N)}` and cpu_sum_u64
/// walks "count" to get N.
#[test]
fn snapshot_entry_cpu_sum_u64_walks_nested_path() {
    let struct_for = |n: u64| RenderedValue::Struct {
        type_name: Some("slot".into()),
        members: vec![RenderedMember {
            name: "count".into(),
            value: RenderedValue::Uint { bits: 64, value: n },
        }],
    };
    let e = FailureDumpPercpuEntry {
        key: 0,
        per_cpu: vec![
            Some(struct_for(10)),
            Some(struct_for(20)),
            None,
            Some(struct_for(30)),
        ],
    };
    let entry = SnapshotEntry::Percpu(&e);
    assert_eq!(entry.cpu_sum_u64("count").unwrap(), 60);
}

/// The PercpuHash variant takes the same per_cpu walk as
/// Percpu (both dispatch through cpu_each's `&e.per_cpu`
/// arm). Pin both variant dispatch paths so a refactor that
/// drops the PercpuHash arm surfaces as a TypeMismatch error
/// the test catches.
#[test]
fn snapshot_entry_cpu_aggregators_apply_to_percpu_hash_variant() {
    let e = FailureDumpPercpuHashEntry {
        key: Some(RenderedValue::Uint { bits: 32, value: 0 }),
        key_hex: "00000000".into(),
        per_cpu: vec![
            Some(RenderedValue::Uint { bits: 64, value: 5 }),
            Some(RenderedValue::Uint { bits: 64, value: 7 }),
            None,
        ],
    };
    let entry = SnapshotEntry::PercpuHash(&e);
    assert_eq!(entry.cpu_sum_u64("").unwrap(), 12);
    assert_eq!(entry.cpu_max_u64("").unwrap(), 7);
    assert_eq!(entry.cpu_min_u64("").unwrap(), 5);
}

/// A slot whose rendered value is a Struct (not directly an
/// integer) makes the empty-path cpu_sum_u64 fail with
/// TypeMismatch — the SnapshotField::Value(v).as_u64() chain
/// rejects Struct.
#[test]
fn snapshot_entry_cpu_sum_u64_struct_slot_errors_with_type_mismatch() {
    let s = RenderedValue::Struct {
        type_name: Some("slot".into()),
        members: vec![RenderedMember {
            name: "count".into(),
            value: RenderedValue::Uint { bits: 64, value: 7 },
        }],
    };
    let e = FailureDumpPercpuEntry {
        key: 0,
        per_cpu: vec![Some(s)],
    };
    let entry = SnapshotEntry::Percpu(&e);
    match entry.cpu_sum_u64("").expect_err("struct cannot as_u64") {
        SnapshotError::TypeMismatch { actual, .. } => {
            assert!(actual.contains("Struct"), "actual was {actual}");
        }
        other => panic!("unexpected: {other:?}"),
    }
}

#[test]
fn snapshot_bridge_capture_stores_under_name() {
    let report = synthetic_report();
    let cb: CaptureCallback = Arc::new(move |_name| Some(report.clone()));
    let bridge = SnapshotBridge::new(cb);
    assert!(bridge.is_empty());
    assert!(bridge.capture("test_name"));
    assert_eq!(bridge.len(), 1);
    let drained = bridge.drain();
    assert!(drained.contains_key("test_name"));
    assert_eq!(drained["test_name"].maps.len(), 3);
}

#[test]
fn snapshot_bridge_capture_failure_returns_false() {
    let cb: CaptureCallback = Arc::new(|_| None);
    let bridge = SnapshotBridge::new(cb);
    assert!(!bridge.capture("oops"));
    assert!(bridge.is_empty());
}

#[test]
fn snapshot_bridge_register_watch_without_callback_errors() {
    let cb: CaptureCallback = Arc::new(|_| None);
    let bridge = SnapshotBridge::new(cb);
    let err = bridge
        .register_watch("kernel.foo")
        .expect_err("no watch register installed");
    assert!(err.contains("no watch-register callback installed"));
    // Cap rollback: failed register must not consume a slot.
    assert_eq!(bridge.watch_count(), 0);
}

#[test]
fn snapshot_bridge_register_watch_enforces_max_3() {
    let cb: CaptureCallback = Arc::new(|_| None);
    let reg: WatchRegisterCallback = Arc::new(|_symbol| Ok(()));
    let bridge = SnapshotBridge::new(cb).with_watch_register(reg);
    assert!(bridge.register_watch("kernel.a").is_ok());
    assert!(bridge.register_watch("kernel.b").is_ok());
    assert!(bridge.register_watch("kernel.c").is_ok());
    assert_eq!(bridge.watch_count(), MAX_WATCH_SNAPSHOTS);
    let err = bridge
        .register_watch("kernel.d")
        .expect_err("4th watch must be rejected");
    assert!(err.contains("cap exceeded"));
    // Cap rollback: rejection does not consume a slot.
    assert_eq!(bridge.watch_count(), MAX_WATCH_SNAPSHOTS);
}

#[test]
fn snapshot_bridge_register_watch_propagates_callback_error() {
    let cb: CaptureCallback = Arc::new(|_| None);
    let reg: WatchRegisterCallback =
        Arc::new(|symbol| Err(format!("symbol '{symbol}' did not resolve")));
    let bridge = SnapshotBridge::new(cb).with_watch_register(reg);
    let err = bridge
        .register_watch("kernel.nonexistent")
        .expect_err("callback errored");
    assert!(err.contains("kernel.nonexistent"));
    // Failed register must not consume a slot.
    assert_eq!(bridge.watch_count(), 0);
}

/// Pin the WatchSlotGuard panic-safety invariant: a panic inside
/// the watch-register callback must NOT leak the reserved slot.
/// Before the guard was added, the manual fetch_sub rollback only
/// ran on the explicit `Err(reason)` arm — a panicking callback
/// left `watch_count` permanently incremented, eventually exhausting
/// the cap with no real watchpoints armed. The guard's `Drop` impl
/// runs on every exit path including unwind; success commits via
/// `mem::forget`. Regression: removing the guard or moving
/// `mem::forget` before the callback would surface here as
/// `watch_count() != 0` after the catch_unwind below.
#[test]
fn snapshot_bridge_register_watch_panic_releases_slot() {
    let cb: CaptureCallback = Arc::new(|_| None);
    let reg: WatchRegisterCallback = Arc::new(|_symbol| {
        panic!("synthetic register_watch panic — slot must still release");
    });
    let bridge = SnapshotBridge::new(cb).with_watch_register(reg);
    let bridge_clone = bridge.clone();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = bridge_clone.register_watch("kernel.panic_path");
    }));
    assert!(
        result.is_err(),
        "callback panic must propagate out of register_watch",
    );
    // Slot must be released — guard's Drop ran during unwind.
    assert_eq!(
        bridge.watch_count(),
        0,
        "WatchSlotGuard must release the reserved slot on panic; \
             a non-zero count means the slot leaked and the cap will \
             eventually exhaust with no real watchpoints armed",
    );
    // Cap must remain reachable: a fresh non-panicking callback
    // can now register all 3 user slots.
    let cb2: CaptureCallback = Arc::new(|_| None);
    let reg2: WatchRegisterCallback = Arc::new(|_| Ok(()));
    let bridge2 = SnapshotBridge::new(cb2).with_watch_register(reg2);
    for i in 0..MAX_WATCH_SNAPSHOTS {
        assert!(bridge2.register_watch(&format!("kernel.s{i}")).is_ok());
    }
    assert_eq!(bridge2.watch_count(), MAX_WATCH_SNAPSHOTS);
}

#[test]
fn snapshot_bridge_thread_local_install_and_restore() {
    assert!(with_active_bridge(|_| ()).is_none());
    let report = synthetic_report();
    let cb: CaptureCallback = Arc::new(move |_| Some(report.clone()));
    let bridge = SnapshotBridge::new(cb);
    let bridge_clone = bridge.clone();
    {
        let _g = bridge.set_thread_local();
        let captured = with_active_bridge(|b| b.capture("nested"));
        assert_eq!(captured, Some(true));
    }
    assert!(with_active_bridge(|_| ()).is_none());
    assert_eq!(bridge_clone.len(), 1);
}

#[test]
fn snapshot_bridge_is_send_sync() {
    fn assert_send_sync<T: Send + Sync>(_: &T) {}
    let cb: CaptureCallback = Arc::new(|_| None);
    let bridge = SnapshotBridge::new(cb);
    assert_send_sync(&bridge);
}

/// [`SnapshotBridge::dispatch_kernel_op`] returns `None` when no
/// kernel-op callback is installed — the executor's
/// bridge-first / wire-fallback dispatcher relies on this signal
/// to know it should try the in-guest virtio-console path
/// instead. A regression that returned `Some(default-reply)` for
/// the no-callback case would silently mask missing test setup.
#[test]
fn snapshot_bridge_dispatch_kernel_op_returns_none_without_callback() {
    let cb: CaptureCallback = Arc::new(|_| None);
    let bridge = SnapshotBridge::new(cb);
    let request = crate::vmm::wire::KernelOpRequestPayload {
        request_id: 1,
        mode: crate::vmm::wire::KernelOpMode::Hot,
        direction: crate::vmm::wire::KernelOpDirection::Write,
        tag: "scratch".into(),
        entries: vec![],
    };
    assert!(bridge.dispatch_kernel_op(&request).is_none());
    // Drain log must remain empty — no callback ran, nothing to
    // record.
    assert!(bridge.drain_kernel_ops().is_empty());
}

/// [`SnapshotBridge::dispatch_kernel_op`] invokes the installed
/// callback, returns its reply, AND records the reply in the
/// per-tag drain log. Pins the full callback→reply→log triple:
/// a regression that skipped the drain log push would break
/// test fixtures' verify-after-execute_steps pattern; a
/// regression that skipped the return would break the
/// executor's error-propagation chain.
#[test]
fn snapshot_bridge_dispatch_kernel_op_invokes_callback_and_logs_reply() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    let cb: CaptureCallback = Arc::new(|_| None);
    let invocation_count = Arc::new(AtomicUsize::new(0));
    let ic = invocation_count.clone();
    let kernel_op_cb: KernelOpCallback = Arc::new(move |req| {
        ic.fetch_add(1, Ordering::Relaxed);
        crate::vmm::wire::KernelOpReplyPayload {
            request_id: req.request_id,
            success: true,
            reason: String::new(),
            read_values: vec![crate::vmm::wire::KernelOpValue::U64(0xDEAD)],
        }
    });
    let bridge = SnapshotBridge::new(cb).with_kernel_op(kernel_op_cb);
    let request = crate::vmm::wire::KernelOpRequestPayload {
        request_id: 99,
        mode: crate::vmm::wire::KernelOpMode::Cold,
        direction: crate::vmm::wire::KernelOpDirection::Read,
        tag: "rq_clock".into(),
        entries: vec![crate::vmm::wire::KernelOpEntry {
            target: crate::vmm::wire::KernelOpTarget::PerCpuField {
                symbol: "runqueues".into(),
                field: "clock".into(),
                cpu: 0,
            },
            value: crate::vmm::wire::KernelOpValue::U64(0),
        }],
    };
    let reply = bridge
        .dispatch_kernel_op(&request)
        .expect("callback installed");
    assert!(reply.success);
    assert!(reply.reason.is_empty());
    assert_eq!(reply.request_id, 99);
    // Pin the read VALUE, not just the count: the whole point of
    // dispatch_kernel_op is to return the host-read result verbatim, so
    // a passthrough/clone bug that zeroed or variant-swapped the element
    // (while keeping len==1) must fail.
    assert_eq!(reply.read_values.len(), 1);
    assert_eq!(
        reply.read_values[0],
        crate::vmm::wire::KernelOpValue::U64(0xDEAD),
    );
    assert_eq!(invocation_count.load(Ordering::Relaxed), 1);
    // Drain log captures the (tag, reply) pair — including the read
    // values, which must equal the returned reply (the logged copy is
    // load-bearing for failure dumps).
    let log = bridge.drain_kernel_ops();
    assert_eq!(log.len(), 1);
    assert_eq!(log[0].0, "rq_clock");
    assert_eq!(log[0].1.request_id, 99);
    assert_eq!(
        log[0].1.read_values,
        vec![crate::vmm::wire::KernelOpValue::U64(0xDEAD)],
    );
    // Second drain after the first returns empty — drain is
    // destructive.
    assert!(bridge.drain_kernel_ops().is_empty());
}

/// Multiple kernel-op dispatches accumulate in the drain log in
/// insertion order. Pins the FIFO contract a multi-op step
/// (e.g. a `with_uptime` batch + a sentinel-read) relies on
/// when reconstructing the per-tag outcome sequence.
#[test]
fn snapshot_bridge_dispatch_kernel_op_drain_log_is_ordered() {
    let cb: CaptureCallback = Arc::new(|_| None);
    let kernel_op_cb: KernelOpCallback = Arc::new(|req| crate::vmm::wire::KernelOpReplyPayload {
        request_id: req.request_id,
        success: true,
        reason: String::new(),
        read_values: vec![],
    });
    let bridge = SnapshotBridge::new(cb).with_kernel_op(kernel_op_cb);
    for (i, tag) in ["first", "second", "third"].iter().enumerate() {
        let req = crate::vmm::wire::KernelOpRequestPayload {
            request_id: (i as u32) + 1,
            mode: crate::vmm::wire::KernelOpMode::Hot,
            direction: crate::vmm::wire::KernelOpDirection::Write,
            tag: (*tag).into(),
            entries: vec![],
        };
        assert!(bridge.dispatch_kernel_op(&req).is_some());
    }
    let log = bridge.drain_kernel_ops();
    assert_eq!(log.len(), 3);
    assert_eq!(log[0].0, "first");
    assert_eq!(log[1].0, "second");
    assert_eq!(log[2].0, "third");
    // Request ids match insertion order — confirming the
    // callback's per-dispatch id-preservation contract.
    assert_eq!(log[0].1.request_id, 1);
    assert_eq!(log[1].1.request_id, 2);
    assert_eq!(log[2].1.request_id, 3);
}

/// Filling [`SnapshotBridge`] beyond [`MAX_STORED_SNAPSHOTS`]
/// must FIFO-evict the oldest tag and keep the newest. Pins
/// the cap-and-evict invariant of `store_internal`'s
/// `while reports.len() > MAX_STORED_SNAPSHOTS` loop: it
/// `order.pop_front()`s the oldest insertion and removes the
/// corresponding entry from `reports`. A regression that drops
/// the sweep, replaces FIFO with LIFO, or skips the
/// `reports.remove` step would surface here as either an
/// over-cap `len()` or the wrong tag missing/present.
#[test]
fn snapshot_bridge_store_fifo_evicts_oldest() {
    let cb: CaptureCallback = Arc::new(|_| None);
    let bridge = SnapshotBridge::new(cb);
    // Insert exactly MAX_STORED_SNAPSHOTS distinct tags. The
    // store invariant at the cap is `len() == cap`; nothing
    // has been evicted yet.
    for i in 0..MAX_STORED_SNAPSHOTS {
        bridge.store(&format!("tag_{i:04}"), FailureDumpReport::default());
    }
    assert_eq!(
        bridge.len(),
        MAX_STORED_SNAPSHOTS,
        "store at cap must hold exactly {MAX_STORED_SNAPSHOTS} entries",
    );
    // Insert one more — `tag_0000` (the oldest) must be the
    // evicted FIFO front; the freshest tag must now be
    // resident.
    let overflow_tag = format!("tag_{MAX_STORED_SNAPSHOTS:04}");
    bridge.store(&overflow_tag, FailureDumpReport::default());
    assert_eq!(
        bridge.len(),
        MAX_STORED_SNAPSHOTS,
        "post-overflow len must remain at cap (one in, one out)",
    );
    let drained = bridge.drain();
    assert!(
        !drained.contains_key("tag_0000"),
        "FIFO eviction must drop the oldest tag (tag_0000)",
    );
    assert!(
        drained.contains_key(&overflow_tag),
        "newest tag ({overflow_tag}) must be resident after the overflow store",
    );
    // The other 63 originally-inserted tags (tag_0001 ..
    // tag_0063) must all survive — the FIFO is one-in-one-out,
    // not a wholesale flush.
    for i in 1..MAX_STORED_SNAPSHOTS {
        let tag = format!("tag_{i:04}");
        assert!(
            drained.contains_key(&tag),
            "tag {tag} must survive single-overflow eviction",
        );
    }
}

/// Storing the same tag twice must REPLACE the report and
/// move the tag to the BACK of the FIFO order — refreshing
/// its position so a hot-rewritten tag does not stay near
/// the eviction front. Pins the overwrite-refresh invariant in
/// `store_internal`'s insert-collision branch: it searches
/// `order` via `order.iter().position(|k| k == name)`, removes
/// that slot, then `order.push_back`s the fresh occurrence.
///
/// The proof shape: pre-fill to cap with tag_0 .. tag_{cap-1},
/// re-store tag_0 (refreshing its position to back), then
/// store one fresh overflow tag. If overwrite-refresh
/// works, the evicted tag MUST be tag_1 (now the oldest);
/// without the refresh, tag_0 would stay at front and be
/// evicted instead.
#[test]
fn snapshot_bridge_store_overwrite_refreshes_position() {
    let cb: CaptureCallback = Arc::new(|_| None);
    let bridge = SnapshotBridge::new(cb);
    for i in 0..MAX_STORED_SNAPSHOTS {
        bridge.store(&format!("tag_{i:04}"), FailureDumpReport::default());
    }
    // Refresh tag_0 by overwriting it. The doc invariant: the
    // overwrite path moves tag_0 from front to back of `order`
    // and replaces its report in `reports`. Use a non-default
    // schema to make the overwrite observable on the value
    // side too.
    let refreshed = FailureDumpReport {
        schema: "refreshed".to_string(),
        active_map_kvas: Vec::new(),
        ..Default::default()
    };
    bridge.store("tag_0000", refreshed);
    assert_eq!(
        bridge.len(),
        MAX_STORED_SNAPSHOTS,
        "overwrite must not change resident count",
    );
    // Push one fresh overflow tag. With overwrite-refresh,
    // the evicted entry is tag_0001 (now the FIFO front);
    // without it, tag_0000 would still be front and would
    // be evicted instead.
    let overflow_tag = format!("tag_{MAX_STORED_SNAPSHOTS:04}");
    bridge.store(&overflow_tag, FailureDumpReport::default());
    let drained = bridge.drain();
    assert!(
        drained.contains_key("tag_0000"),
        "tag_0000 must survive eviction — overwrite refreshed its FIFO \
             position to the back. A regression to a no-refresh overwrite \
             path would evict tag_0000 instead of tag_0001 here.",
    );
    assert_eq!(
        drained
            .get("tag_0000")
            .expect("tag_0000 resident after overwrite")
            .schema,
        "refreshed",
        "overwrite must replace the report value, not just refresh order",
    );
    assert!(
        !drained.contains_key("tag_0001"),
        "tag_0001 must be the evicted tag — refreshed tag_0000 displaced \
             tag_0001 to the FIFO front",
    );
    assert!(
        drained.contains_key(&overflow_tag),
        "newest tag ({overflow_tag}) must be resident after the overflow store",
    );
}

#[test]
fn enum_variant_round_trips() {
    let r = synthetic_report();
    let snap = Snapshot::new(&r);
    let policy = snap.var("ctx").get("policy");
    assert_eq!(policy.as_i64().unwrap(), 1);
    assert_eq!(policy.as_u64().unwrap(), 1);
    assert_eq!(policy.as_str().unwrap(), "SCHED_NORMAL");
}

#[test]
fn rendered_passthrough_returns_raw_value() {
    let r = synthetic_report();
    let snap = Snapshot::new(&r);
    let f = snap.var("ctx").get("weight");
    let rendered = f.raw().expect("weight is a Value");
    match rendered {
        RenderedValue::Uint { bits, value } => {
            assert_eq!(*bits, 32);
            assert_eq!(*value, 1024);
        }
        other => panic!("unexpected rendered shape: {other:?}"),
    }
}

#[test]
fn snapshot_error_display_includes_path_and_alternatives() {
    let r = synthetic_report();
    let snap = Snapshot::new(&r);
    let err = snap.var("ctx").get("nope").error().unwrap().to_string();
    assert!(err.contains("nope"));
    assert!(err.contains("weight"));
}

#[test]
fn var_exact_match_does_not_split_dotted_paths() {
    let r = synthetic_report();
    let snap = Snapshot::new(&r);
    // Chained `var(...).get(...)` walks the rendered struct's
    // members and yields the leaf value — the canonical way to
    // reach a sub-field.
    let chained = snap.var("ctx").get("weight");
    assert_eq!(chained.as_u64().unwrap(), 1024);
    // `Snapshot::var` does not split on `.` — a dotted
    // string is treated as one global variable name. Since
    // no top-level member named `"ctx.weight"` exists, the
    // call resolves to `Missing`.
    let dotted = snap.var("ctx.weight");
    assert!(dotted.error().is_some());
}

#[test]
fn type_mismatch_carries_actual_kind() {
    let r = synthetic_report();
    let snap = Snapshot::new(&r);
    // weight is a Uint — try to read it as bool variant
    // string. Any Value that is not Enum-with-name lands in
    // TypeMismatch.
    let result = snap.var("ctx").get("weight").as_str();
    match result {
        Err(SnapshotError::TypeMismatch {
            expected, actual, ..
        }) => {
            assert_eq!(expected, "str (enum variant name)");
            assert_eq!(actual, "Uint");
        }
        _ => panic!("expected TypeMismatch"),
    }
}

/// `SnapshotBridge::drain_ordered` returns every stored
/// `(name, report)` pair in INSERTION order — the same order the
/// internal [`SnapshotStore::order`] `VecDeque` records. This is
/// load-bearing for periodic-capture consumers: the freeze
/// coordinator's run-loop publishes `periodic_000`, `periodic_001`,
/// ... at monotonically-increasing wall-clock times, and the test
/// author needs to walk the captures in the same order to compare
/// adjacent timeline samples. `drain()` returns a `HashMap` whose
/// iteration order is non-deterministic across runs, so periodic
/// consumers MUST go through `drain_ordered` to read the timeline
/// in cadence order.
///
/// Pin the FIFO contract:
///   * insertion order survives through `store()` calls
///   * the result is keyed by `String` and carries the full
///     `FailureDumpReport` value
///   * `drain_ordered()` empties the bridge (matching `drain()`)
///     so a follow-up `len()` is 0
///   * a tag overwrite refreshes its position to the back, in
///     lock-step with the FIFO eviction invariant
#[test]
fn snapshot_bridge_drain_ordered_preserves_insertion_order() {
    let cb: CaptureCallback = Arc::new(|_| None);
    let bridge = SnapshotBridge::new(cb);
    // Insert distinct tags in a non-alphabetical order so an
    // accidental sort-by-key implementation surfaces as a test
    // failure instead of silently appearing to work.
    let inputs: &[&str] = &[
        "periodic_002",
        "periodic_000",
        "periodic_005",
        "periodic_001",
        "periodic_003",
    ];
    for (i, tag) in inputs.iter().enumerate() {
        let r = FailureDumpReport {
            schema: format!("schema_{i}"),
            active_map_kvas: Vec::new(),
            ..Default::default()
        };
        bridge.store(tag, r);
    }
    let drained: Vec<(String, FailureDumpReport)> = bridge.drain_ordered();
    assert_eq!(
        drained.len(),
        inputs.len(),
        "drain_ordered must yield every stored entry exactly once",
    );
    let drained_names: Vec<&str> = drained.iter().map(|(n, _)| n.as_str()).collect();
    assert_eq!(
        drained_names, inputs,
        "drain_ordered must yield insertion order, not sorted or hash order",
    );
    for (i, (_, report)) in drained.iter().enumerate() {
        assert_eq!(
            report.schema,
            format!("schema_{i}"),
            "drained entry {i} must carry the originally-stored report",
        );
    }
    assert_eq!(
        bridge.len(),
        0,
        "drain_ordered must empty the bridge (matching drain())",
    );
    // A subsequent drain_ordered on the empty bridge yields an
    // empty vec — guards against double-drain leaving a stray
    // entry behind in `order` after `reports` is drained.
    let second: Vec<(String, FailureDumpReport)> = bridge.drain_ordered();
    assert!(
        second.is_empty(),
        "second drain_ordered on empty bridge must be empty, got len={}",
        second.len(),
    );
}

/// Re-storing an existing tag refreshes its position to the
/// BACK of the insertion order. This is the same invariant that
/// `snapshot_bridge_store_overwrite_refreshes_position` pins for
/// the FIFO eviction path; `drain_ordered` must surface the
/// refreshed order so downstream consumers see the updated
/// cadence position. A regression that overwrote the report but
/// left the order entry in place would surface here as the
/// refreshed tag still appearing at its original index.
#[test]
fn snapshot_bridge_drain_ordered_overwrite_refreshes_position() {
    let cb: CaptureCallback = Arc::new(|_| None);
    let bridge = SnapshotBridge::new(cb);
    bridge.store("a", FailureDumpReport::default());
    bridge.store("b", FailureDumpReport::default());
    bridge.store("c", FailureDumpReport::default());
    // Overwrite "a" — its position must move from front to
    // back.
    bridge.store(
        "a",
        FailureDumpReport {
            schema: "refreshed".to_string(),
            active_map_kvas: Vec::new(),
            ..Default::default()
        },
    );
    let drained = bridge.drain_ordered();
    let names: Vec<&str> = drained.iter().map(|(n, _)| n.as_str()).collect();
    assert_eq!(
        names,
        vec!["b", "c", "a"],
        "overwrite of 'a' must move it to the back of the insertion order",
    );
    let a = drained
        .iter()
        .find(|(n, _)| n == "a")
        .expect("'a' resident after overwrite");
    assert_eq!(
        a.1.schema, "refreshed",
        "drain_ordered must surface the refreshed report value, not the prior one",
    );
}

/// `store_with_stats` bundles a stats JSON and an elapsed-ms
/// timestamp alongside the report. `drain_ordered_with_stats`
/// returns the matching `(tag, report, stats, elapsed)` tuple
/// per stored entry; non-paired entries (added via plain
/// `store`) report `None` for both parallel slots.
#[test]
fn snapshot_bridge_store_with_stats_round_trips() {
    let cb: CaptureCallback = Arc::new(|_| None);
    let bridge = SnapshotBridge::new(cb);
    let stats = serde_json::json!({"busy": 75.0});
    bridge.store_with_stats(
        "periodic_000",
        FailureDumpReport::default(),
        Some(Ok(stats.clone())),
        Some(123),
    );
    bridge.store("periodic_001", FailureDumpReport::default());
    let drained = bridge.drain_ordered_with_stats();
    assert_eq!(drained.len(), 2);
    assert_eq!(drained[0].tag, "periodic_000");
    assert_eq!(drained[0].stats, Ok(stats));
    assert_eq!(drained[0].elapsed_ms, Some(123));
    assert_eq!(drained[1].tag, "periodic_001");
    assert_eq!(
        drained[1].stats,
        Err(crate::scenario::snapshot::MissingStatsReason::NoSchedulerBinary),
        "non-stats store collapses to NoSchedulerBinary",
    );
    assert!(drained[1].elapsed_ms.is_none());
}

/// FIFO eviction at `MAX_STORED_SNAPSHOTS` sweeps the parallel
/// stats / elapsed maps in lock-step so a stranded entry can
/// never outlive its report.
#[test]
fn snapshot_bridge_store_with_stats_evicts_in_lockstep() {
    let cb: CaptureCallback = Arc::new(|_| None);
    let bridge = SnapshotBridge::new(cb);
    for i in 0..MAX_STORED_SNAPSHOTS {
        bridge.store_with_stats(
            &format!("tag_{i:04}"),
            FailureDumpReport::default(),
            Some(Ok(serde_json::json!({"i": i}))),
            Some(i as u64),
        );
    }
    let overflow_tag = format!("tag_{MAX_STORED_SNAPSHOTS:04}");
    bridge.store_with_stats(
        &overflow_tag,
        FailureDumpReport::default(),
        Some(Ok(serde_json::json!({"overflow": true}))),
        Some(9_999),
    );
    let drained = bridge.drain_ordered_with_stats();
    // tag_0000 must be evicted.
    let names: Vec<&str> = drained.iter().map(|e| e.tag.as_str()).collect();
    assert!(!names.contains(&"tag_0000"));
    // Newest must be present with its parallel data.
    let last = drained
        .iter()
        .find(|e| e.tag == overflow_tag)
        .expect("overflow tag resident after evict");
    assert_eq!(last.stats, Ok(serde_json::json!({"overflow": true})));
    assert_eq!(last.elapsed_ms, Some(9_999));
}

/// Overwriting a tag with a `None` stats slot clears the prior
/// stats — guards against a stale stats / elapsed value
/// silently surviving across an overwrite that did not bundle
/// fresh values.
#[test]
fn snapshot_bridge_store_with_stats_overwrite_clears_stale_values() {
    let cb: CaptureCallback = Arc::new(|_| None);
    let bridge = SnapshotBridge::new(cb);
    bridge.store_with_stats(
        "periodic_000",
        FailureDumpReport::default(),
        Some(Ok(serde_json::json!({"first": true}))),
        Some(100),
    );
    // Overwrite via plain `store(...)` — should clear the
    // parallel slots since neither was passed.
    bridge.store("periodic_000", FailureDumpReport::default());
    let drained = bridge.drain_ordered_with_stats();
    assert_eq!(drained.len(), 1);
    assert_eq!(
        drained[0].stats,
        Err(crate::scenario::snapshot::MissingStatsReason::NoSchedulerBinary),
        "cleared stats slot collapses to NoSchedulerBinary",
    );
    assert!(drained[0].elapsed_ms.is_none());
}

/// `store_with_stats_and_step` populates the parallel
/// `SnapshotStore.step_index` map and `drain_ordered_with_stats`
/// surfaces it via `DrainedSnapshotEntry::step_index`. Pins the
/// step-stamped path's round-trip.
#[test]
fn snapshot_bridge_store_with_stats_and_step_round_trips() {
    let cb: CaptureCallback = Arc::new(|_| None);
    let bridge = SnapshotBridge::new(cb);
    bridge.store_with_stats_and_step(
        "periodic_000",
        FailureDumpReport::default(),
        Some(Ok(serde_json::json!({"phase": "warmup"}))),
        Some(123),
        Some(456),
        7,
    );
    let drained = bridge.drain_ordered_with_stats();
    assert_eq!(drained.len(), 1);
    assert_eq!(drained[0].tag, "periodic_000");
    assert_eq!(
        drained[0].step_index,
        Some(7),
        "step-stamped path surfaces step_index Some(N) to the drain consumer",
    );
    assert_eq!(drained[0].elapsed_ms, Some(123));
    assert_eq!(
        drained[0].boundary_offset_ms,
        Some(456),
        "boundary_offset_ms round-trips through the stamped store path \
         alongside step_index/elapsed_ms",
    );
}

/// Legacy unstamped `store_with_stats` path drains with
/// `step_index = None` — fixtures and tests that do not care about
/// phase attribution still drain cleanly. Pins the
/// None-for-legacy contract (non-stamped paths surface as None
/// rather than collapse to BASELINE).
#[test]
fn snapshot_bridge_drain_ordered_with_stats_step_index_none_for_legacy_store() {
    let cb: CaptureCallback = Arc::new(|_| None);
    let bridge = SnapshotBridge::new(cb);
    bridge.store_with_stats(
        "periodic_legacy",
        FailureDumpReport::default(),
        Some(Ok(serde_json::json!({"any": 1}))),
        Some(50),
    );
    let drained = bridge.drain_ordered_with_stats();
    assert_eq!(drained.len(), 1);
    assert!(
        drained[0].step_index.is_none(),
        "unstamped store path must surface step_index=None (not BASELINE), so the drain consumer can distinguish unstamped fixture captures from explicit phase-0 stamps",
    );
}

/// FIFO eviction at `MAX_STORED_SNAPSHOTS` must sweep the
/// `step_index` map in lock-step with the parallel reports /
/// stats / elapsed maps. A stranded `step_index` entry whose
/// paired report has been evicted would silently leak through the
/// next-drain shape — the parallel-map invariant is exactly the
/// silent-data-loss class this field exists to guard against.
#[test]
fn snapshot_bridge_store_with_stats_and_step_evicts_step_index_in_lockstep() {
    let cb: CaptureCallback = Arc::new(|_| None);
    let bridge = SnapshotBridge::new(cb);
    for i in 0..MAX_STORED_SNAPSHOTS {
        bridge.store_with_stats_and_step(
            &format!("tag_{i:04}"),
            FailureDumpReport::default(),
            Some(Ok(serde_json::json!({"i": i}))),
            Some(i as u64),
            None,
            (i as u16) % 8 + 1,
        );
    }
    let overflow_tag = format!("tag_{MAX_STORED_SNAPSHOTS:04}");
    bridge.store_with_stats_and_step(
        &overflow_tag,
        FailureDumpReport::default(),
        Some(Ok(serde_json::json!({"overflow": true}))),
        Some(9_999),
        None,
        42,
    );
    let drained = bridge.drain_ordered_with_stats();
    let names: Vec<&str> = drained.iter().map(|e| e.tag.as_str()).collect();
    assert!(!names.contains(&"tag_0000"), "tag_0000 evicted");
    let last = drained
        .iter()
        .find(|e| e.tag == overflow_tag)
        .expect("overflow tag resident after evict");
    assert_eq!(
        last.step_index,
        Some(42),
        "newest survives with its step_index — eviction sweep did not strand the parallel slot",
    );
    // Every loop-stored survivor kept its own step_index ((i%8)+1) —
    // exercises the lock-step sweep across ALL survivors, not just the
    // newest (pinned above). A stranded step_index whose report was
    // evicted is invisible post-drain (drain_ordered_with_stats emits no
    // row for an evicted tag), so that path is covered by the
    // store_internal eviction sweep, verified by review.
    for e in drained.iter().filter(|e| e.tag != overflow_tag) {
        let i: u64 = e
            .tag
            .strip_prefix("tag_")
            .and_then(|s| s.parse().ok())
            .unwrap_or_else(|| panic!("unexpected survivor tag {}", e.tag));
        assert_eq!(
            e.step_index,
            Some((i as u16 % 8) + 1),
            "survivor {} must keep its own step_index ((i%8)+1), proving \
             the parallel map was not cross-wired during eviction",
            e.tag,
        );
    }
}

/// FIFO eviction at `MAX_STORED_SNAPSHOTS` must sweep the
/// `boundary_offset_ms` map in lock-step with the parallel reports /
/// stats / elapsed / step_index maps — the same parallel-map invariant
/// the step_index eviction test pins. A stranded `boundary_offset_ms`
/// whose paired report was evicted would silently mis-attribute the
/// next drain's phase remap: exactly the silent-data-loss class this
/// field exists to guard against.
#[test]
fn snapshot_bridge_store_with_stats_and_step_evicts_boundary_offset_in_lockstep() {
    let cb: CaptureCallback = Arc::new(|_| None);
    let bridge = SnapshotBridge::new(cb);
    for i in 0..MAX_STORED_SNAPSHOTS {
        bridge.store_with_stats_and_step(
            &format!("tag_{i:04}"),
            FailureDumpReport::default(),
            Some(Ok(serde_json::json!({"i": i}))),
            Some(i as u64),
            Some((i as u64) * 10),
            (i as u16) % 8 + 1,
        );
    }
    let overflow_tag = format!("tag_{MAX_STORED_SNAPSHOTS:04}");
    bridge.store_with_stats_and_step(
        &overflow_tag,
        FailureDumpReport::default(),
        Some(Ok(serde_json::json!({"overflow": true}))),
        Some(9_999),
        Some(99_990),
        42,
    );
    let drained = bridge.drain_ordered_with_stats();
    let names: Vec<&str> = drained.iter().map(|e| e.tag.as_str()).collect();
    assert!(!names.contains(&"tag_0000"), "tag_0000 evicted");
    let last = drained
        .iter()
        .find(|e| e.tag == overflow_tag)
        .expect("overflow tag resident after evict");
    assert_eq!(
        last.boundary_offset_ms,
        Some(99_990),
        "newest survives with its boundary_offset_ms — eviction sweep did not strand the parallel slot",
    );
    // Every loop-stored survivor kept its own boundary_offset (i*10) —
    // exercises the lock-step sweep across ALL survivors, not just the
    // newest (pinned above). A stranded offset whose report was evicted
    // is invisible post-drain (drain_ordered_with_stats emits no row for
    // an evicted tag), so that path is covered by the store_internal
    // eviction sweep, verified by review.
    for e in drained.iter().filter(|e| e.tag != overflow_tag) {
        let i: u64 = e
            .tag
            .strip_prefix("tag_")
            .and_then(|s| s.parse().ok())
            .unwrap_or_else(|| panic!("unexpected survivor tag {}", e.tag));
        assert_eq!(
            e.boundary_offset_ms,
            Some(i * 10),
            "survivor {} must keep its own boundary_offset (i*10), proving \
             the parallel map was not cross-wired during eviction",
            e.tag,
        );
    }
}

/// `drain()` (the non-stats variant) must clear the parallel
/// `boundary_offset_ms` map. It returns only reports, so a stranded
/// offset is invisible to its own caller — but a re-used bridge whose
/// next capture re-uses a tag via the plain `store()` path (whose
/// new-insert branch never removes a stale offset) would inherit the
/// previous run's offset and mis-attribute the phase boundary. Pin
/// that `drain()` scrubs the offset so a re-used tag starts clean.
#[test]
fn drain_clears_boundary_offset_so_reused_tag_does_not_inherit_stale_offset() {
    let cb: CaptureCallback = Arc::new(|_| None);
    let bridge = SnapshotBridge::new(cb);
    // First run: a periodic capture carrying a phase-boundary offset.
    bridge.store_with_stats_and_step(
        "periodic_000",
        FailureDumpReport::default(),
        Some(Ok(serde_json::json!({"run": 1}))),
        Some(0),
        Some(1_500),
        1,
    );
    // Non-stats drain: returns reports, drops the parallel metadata.
    let _ = bridge.drain();
    // Second run re-uses the tag via plain store() (no offset), which
    // hits store_internal's new-insert branch — the branch that never
    // removes a stale boundary_offset entry.
    bridge.store("periodic_000", FailureDumpReport::default());
    let drained = bridge.drain_ordered_with_stats();
    let e = drained
        .iter()
        .find(|e| e.tag == "periodic_000")
        .expect("re-used tag resident after second store");
    assert_eq!(
        e.boundary_offset_ms, None,
        "drain() must clear boundary_offset_ms; a re-used tag must not \
         inherit the previous run's stale phase-boundary offset",
    );
}

/// Same invariant as the `drain()` test above, for the insertion-
/// ordered `drain_ordered()` variant — it too must scrub
/// `boundary_offset_ms` so a re-used tag does not inherit a stale
/// phase-boundary offset.
#[test]
fn drain_ordered_clears_boundary_offset_so_reused_tag_does_not_inherit_stale_offset() {
    let cb: CaptureCallback = Arc::new(|_| None);
    let bridge = SnapshotBridge::new(cb);
    bridge.store_with_stats_and_step(
        "periodic_000",
        FailureDumpReport::default(),
        Some(Ok(serde_json::json!({"run": 1}))),
        Some(0),
        Some(1_500),
        1,
    );
    let _ = bridge.drain_ordered();
    bridge.store("periodic_000", FailureDumpReport::default());
    let drained = bridge.drain_ordered_with_stats();
    let e = drained
        .iter()
        .find(|e| e.tag == "periodic_000")
        .expect("re-used tag resident after second store");
    assert_eq!(
        e.boundary_offset_ms, None,
        "drain_ordered() must clear boundary_offset_ms; a re-used tag \
         must not inherit the previous run's stale phase-boundary offset",
    );
}

// ---------- stats_path JSON accessor ----------

/// `stats_path` walks a JSON object along a dotted path and
/// returns a [`JsonField`] view at the leaf.
#[test]
fn stats_path_walks_dotted_path() {
    let v = serde_json::json!({"layers": {"batch": {"util": 75.5}}});
    let f = stats_path(&v, "layers.batch.util");
    assert_eq!(f.as_f64().unwrap(), 75.5);
}

/// Empty path returns the root unchanged.
#[test]
fn stats_path_empty_returns_root() {
    let v = serde_json::json!(42);
    let f = stats_path(&v, "");
    assert_eq!(f.as_u64().unwrap(), 42);
}

/// Missing key surfaces FieldNotFound with the available keys.
#[test]
fn stats_path_missing_key_lists_alternatives() {
    let v = serde_json::json!({"busy": 50.0, "antistall": 0});
    let f = stats_path(&v, "missing");
    let err = f.error().expect("missing must error");
    match err {
        SnapshotError::FieldNotFound {
            component,
            available,
            ..
        } => {
            assert_eq!(component, "missing");
            assert!(available.contains(&"busy".to_string()));
            assert!(available.contains(&"antistall".to_string()));
        }
        other => panic!("expected FieldNotFound, got {other:?}"),
    }
}

/// Walking through a non-object cursor surfaces NotAStruct.
#[test]
fn stats_path_through_scalar_errors_not_a_struct() {
    let v = serde_json::json!({"x": 5});
    let f = stats_path(&v, "x.y");
    match f.error().expect("must error") {
        SnapshotError::NotAStruct { component, .. } => {
            assert_eq!(component, "y");
        }
        other => panic!("expected NotAStruct, got {other:?}"),
    }
}

/// Empty path component (`a..b`) reports EmptyPathComponent.
#[test]
fn stats_path_empty_component_errors() {
    let v = serde_json::json!({"a": {"b": 1}});
    let f = stats_path(&v, "a..b");
    match f.error().expect("must error") {
        SnapshotError::EmptyPathComponent { requested } => {
            assert_eq!(requested, "a..b");
        }
        other => panic!("expected EmptyPathComponent, got {other:?}"),
    }
}

/// String-encoded numeric coerces via as_u64 (scx_stats
/// stringifies large counters to avoid 53-bit float collapse).
#[test]
fn stats_path_string_to_u64_coerces() {
    let v = serde_json::json!({"counter": "12345678901234"});
    let f = stats_path(&v, "counter");
    assert_eq!(f.as_u64().unwrap(), 12_345_678_901_234);
}

#[test]
fn snapshot_error_hash_consistent_with_eq() {
    use std::collections::HashSet;
    let e1 = SnapshotError::VarNotFound {
        requested: "nr_cpus".into(),
        available: vec!["nr_iters".into()],
    };
    let e2 = SnapshotError::VarNotFound {
        requested: "nr_cpus".into(),
        available: vec!["nr_iters".into()],
    };
    let mut set: HashSet<SnapshotError> = HashSet::new();
    set.insert(e1);
    assert!(set.contains(&e2));
}

/// Round-trip every SnapshotError variant through serde_json to
/// pin that the Serialize+Deserialize derives stay byte-stable
/// across every field shape. Catches a future regression that
/// reverts a String field to `&'static str` (which would silently
/// drop the variant from the Deserialize impl).
#[test]
fn snapshot_error_serde_round_trip() {
    let cases = vec![
        SnapshotError::MapNotFound {
            requested: "nr_cpus".into(),
            available: vec!["scx_bss".into(), "scx_data".into()],
        },
        SnapshotError::VarNotFound {
            requested: "stall".into(),
            available: vec!["nr_cpus_onln".into()],
        },
        SnapshotError::AmbiguousVar {
            requested: "ctx".into(),
            found_in: vec!["scx_bss".into(), "scx_data".into()],
        },
        SnapshotError::FieldNotFound {
            requested: "scx_bss.ctx.missing".into(),
            walked: "scx_bss.ctx".into(),
            component: "missing".into(),
            available: vec!["nr_cpus".into(), "stall".into()],
        },
        SnapshotError::NotAStruct {
            requested: "scx_bss.stall.x".into(),
            walked: "scx_bss.stall".into(),
            component: "x".into(),
            kind: "Uint".to_string(),
        },
        SnapshotError::TypeMismatch {
            expected: "u64".to_string(),
            actual: "Struct".to_string(),
            requested: "scx_bss.ctx".into(),
        },
        SnapshotError::IndexOutOfRange {
            map: "scx_data".into(),
            index: 5,
            len: 2,
        },
        SnapshotError::PerCpuSlot {
            map: "scx_percpu".into(),
            cpu: 7,
            len: 4,
            unmapped: false,
        },
        SnapshotError::NoMatch {
            map: "scx_data".into(),
            op: "find".to_string(),
            len: 3,
            available_keys: vec!["k0".into(), "k1".into()],
        },
        SnapshotError::EmptyPathComponent {
            requested: "a..b".into(),
        },
        SnapshotError::PerCpuNotNarrowed {
            map: "scx_percpu".into(),
        },
        SnapshotError::NoRendered {
            map: "scx_percpu".into(),
            side: "value".to_string(),
        },
        SnapshotError::PlaceholderSample {
            tag: "primary".into(),
            reason: "vCPU rendezvous timed out".into(),
        },
        SnapshotError::MissingStats {
            tag: "scheduler".into(),
            reason: crate::scenario::snapshot::MissingStatsReason::NoSchedulerBinary,
        },
    ];
    for case in cases {
        let json = serde_json::to_string(&case).expect("serialize");
        let back: SnapshotError = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(case, back, "round-trip mismatch for {case:?}");
    }
}

/// Contract pin for the documented escape-hatch role of
/// `Snapshot::report()`. The docstring blesses
/// `snap.report().<field>` as the official bypass for
/// FailureDumpReport fields without typed accessors. A
/// regression that removed `pub fn report()` would silently
/// break the documented promise without breaking any other
/// test in the tree (per-field bypass consumers are not yet
/// landed at the doc time). This test makes the contract
/// compile-and-test enforced: the method must remain pub, must
/// pass through the underlying `FailureDumpReport` unchanged.
#[test]
fn snapshot_report_escape_hatch_passes_through_underlying_report() {
    let report = FailureDumpReport {
        schema: "escape-hatch-contract-pin".to_string(),
        active_map_kvas: Vec::new(),
        ..Default::default()
    };
    let snap = Snapshot::new(&report);
    assert_eq!(
        snap.report().schema,
        "escape-hatch-contract-pin",
        "Snapshot::report() must hand back the underlying \
             FailureDumpReport unchanged — this is the documented \
             escape-hatch contract",
    );
}

// -------------------------------------------------------------
// First-class accessor tests — pin every Snapshot / SnapshotMap
// accessor against the underlying FailureDumpReport / FailureDumpMap
// field so a rename or removal lights up as a compile error AND
// the empty-by-default contract holds for callers checking
// "did the walker run?" via the slice length.
// -------------------------------------------------------------

fn task_enrichment_fixture(pid: i32, comm: &str) -> TaskEnrichment {
    // TaskEnrichment derives Default with all zero/empty fields;
    // override only the identity fields the lookup tests pin
    // (and weight/prio/static/normal_prio for the round-trip
    // pin further below — defaults are 0, lookup tests assert
    // non-default values).
    TaskEnrichment {
        pid,
        tgid: pid,
        comm: comm.to_string(),
        weight: 100,
        prio: 120,
        static_prio: 120,
        normal_prio: 120,
        ..Default::default()
    }
}

#[test]
fn snapshot_event_counter_timeline_is_empty_by_default() {
    let r = FailureDumpReport::default();
    let snap = Snapshot::new(&r);
    assert!(snap.event_counter_timeline().is_empty());
}

#[test]
fn snapshot_event_counter_timeline_borrows_populated_vec() {
    let r = FailureDumpReport {
        event_counter_timeline: vec![
            EventCounterSample {
                elapsed_ms: 10,
                select_cpu_fallback: 7,
                ..Default::default()
            },
            EventCounterSample {
                elapsed_ms: 20,
                select_cpu_fallback: 11,
                ..Default::default()
            },
        ],
        ..Default::default()
    };
    let snap = Snapshot::new(&r);
    let timeline = snap.event_counter_timeline();
    assert_eq!(timeline.len(), 2);
    assert_eq!(timeline[0].elapsed_ms, 10);
    assert_eq!(timeline[1].select_cpu_fallback, 11);
}

#[test]
fn snapshot_rq_scx_states_borrows_populated_vec() {
    let r = FailureDumpReport {
        rq_scx_states: vec![
            crate::monitor::scx_walker::RqScxState::default(),
            crate::monitor::scx_walker::RqScxState::default(),
        ],
        ..Default::default()
    };
    let snap = Snapshot::new(&r);
    assert_eq!(snap.rq_scx_states().len(), 2);
}

#[test]
fn snapshot_dsq_states_borrows_populated_vec() {
    let r = FailureDumpReport {
        dsq_states: vec![crate::monitor::scx_walker::DsqState::default()],
        ..Default::default()
    };
    let snap = Snapshot::new(&r);
    assert_eq!(snap.dsq_states().len(), 1);
}

#[test]
fn snapshot_scx_sched_state_threads_option() {
    let r_absent = FailureDumpReport::default();
    assert!(Snapshot::new(&r_absent).scx_sched_state().is_none());

    let r_present = FailureDumpReport {
        scx_sched_state: Some(crate::monitor::scx_walker::ScxSchedState::default()),
        ..Default::default()
    };
    assert!(Snapshot::new(&r_present).scx_sched_state().is_some());
}

#[test]
fn snapshot_per_cpu_time_at_finds_by_cpu_field_not_position() {
    let r = FailureDumpReport {
        per_cpu_time: vec![
            PerCpuTimeStats {
                cpu: 2,
                cpustat_user_ns: 200,
                ..Default::default()
            },
            PerCpuTimeStats {
                cpu: 0,
                cpustat_user_ns: 0,
                ..Default::default()
            },
        ],
        ..Default::default()
    };
    let snap = Snapshot::new(&r);
    assert_eq!(snap.per_cpu_time().len(), 2);
    // Lookup by the `cpu` field, not vec position — first entry
    // has cpu=2 even though it's at index 0.
    let row = snap.per_cpu_time_at(2).expect("cpu 2 present");
    assert_eq!(row.cpustat_user_ns, 200);
    let zero = snap.per_cpu_time_at(0).expect("cpu 0 present");
    assert_eq!(zero.cpustat_user_ns, 0);
    assert!(snap.per_cpu_time_at(99).is_none());
}

#[test]
fn snapshot_per_node_numa_at_finds_by_node_field() {
    let r = FailureDumpReport {
        per_node_numa: vec![
            PerNodeNumaStats {
                node: 1,
                numa_hit: 500,
                ..Default::default()
            },
            PerNodeNumaStats {
                node: 0,
                numa_hit: 100,
                ..Default::default()
            },
        ],
        ..Default::default()
    };
    let snap = Snapshot::new(&r);
    assert_eq!(snap.per_node_numa().len(), 2);
    assert_eq!(snap.per_node_numa_at(0).unwrap().numa_hit, 100);
    assert_eq!(snap.per_node_numa_at(1).unwrap().numa_hit, 500);
    assert!(snap.per_node_numa_at(99).is_none());
}

#[test]
fn snapshot_task_enrichment_by_pid_finds_first_match() {
    let r = FailureDumpReport {
        task_enrichments: vec![
            task_enrichment_fixture(42, "alpha"),
            task_enrichment_fixture(7, "beta"),
        ],
        ..Default::default()
    };
    let snap = Snapshot::new(&r);
    assert_eq!(snap.task_enrichments().len(), 2);
    assert_eq!(snap.task_enrichment_by_pid(7).unwrap().comm, "beta");
    assert_eq!(snap.task_enrichment_by_pid(42).unwrap().comm, "alpha");
    assert!(snap.task_enrichment_by_pid(1).is_none());
}

#[test]
fn snapshot_prog_runtime_stats_by_name_finds_match() {
    let r = FailureDumpReport {
        prog_runtime_stats: vec![
            ProgRuntimeStats {
                name: "dispatch".into(),
                cnt: 100,
                nsecs: 5000,
                misses: 1,
            },
            ProgRuntimeStats {
                name: "select_cpu".into(),
                cnt: 50,
                nsecs: 2000,
                misses: 0,
            },
        ],
        ..Default::default()
    };
    let snap = Snapshot::new(&r);
    assert_eq!(snap.prog_runtime_stats().len(), 2);
    assert_eq!(
        snap.prog_runtime_stats_by_name("dispatch").unwrap().cnt,
        100
    );
    assert_eq!(
        snap.prog_runtime_stats_by_name("select_cpu").unwrap().nsecs,
        2000
    );
    assert!(snap.prog_runtime_stats_by_name("nope").is_none());
}

/// Pin the first-match-on-duplicate contract on all 4 keyed
/// lookups. Production captures dedupe before push so this case
/// shouldn't occur, but the contract is documented as
/// first-match-wins and a future "return all matches" refactor
/// without explicit migration would silently surface a
/// different row.
#[test]
fn snapshot_keyed_lookups_return_first_match_on_duplicate() {
    let r = FailureDumpReport {
        per_cpu_time: vec![
            PerCpuTimeStats {
                cpu: 0,
                cpustat_user_ns: 11,
                ..Default::default()
            },
            PerCpuTimeStats {
                cpu: 0,
                cpustat_user_ns: 22,
                ..Default::default()
            },
        ],
        per_node_numa: vec![
            PerNodeNumaStats {
                node: 0,
                numa_hit: 100,
                ..Default::default()
            },
            PerNodeNumaStats {
                node: 0,
                numa_hit: 200,
                ..Default::default()
            },
        ],
        task_enrichments: vec![
            task_enrichment_fixture(7, "first"),
            task_enrichment_fixture(7, "second"),
        ],
        prog_runtime_stats: vec![
            ProgRuntimeStats {
                name: "p".into(),
                cnt: 1,
                nsecs: 10,
                misses: 0,
            },
            ProgRuntimeStats {
                name: "p".into(),
                cnt: 2,
                nsecs: 20,
                misses: 0,
            },
        ],
        ..Default::default()
    };
    let snap = Snapshot::new(&r);
    assert_eq!(snap.per_cpu_time_at(0).unwrap().cpustat_user_ns, 11);
    assert_eq!(snap.per_node_numa_at(0).unwrap().numa_hit, 100);
    assert_eq!(snap.task_enrichment_by_pid(7).unwrap().comm, "first");
    assert_eq!(snap.prog_runtime_stats_by_name("p").unwrap().cnt, 1);
}

#[test]
fn snapshot_probe_counters_threads_option() {
    let r_absent = FailureDumpReport::default();
    assert!(Snapshot::new(&r_absent).probe_counters().is_none());

    let r_present = FailureDumpReport {
        probe_counters: Some(ProbeBssCounters::default()),
        ..Default::default()
    };
    assert!(Snapshot::new(&r_present).probe_counters().is_some());
}

fn map_with_shape(
    name: &str,
    ringbuf: Option<FailureDumpRingbuf>,
    arena: Option<ArenaSnapshot>,
    fd_array: Option<FailureDumpFdArray>,
    stack_trace: Option<FailureDumpStackTrace>,
    error: Option<String>,
) -> FailureDumpMap {
    FailureDumpMap {
        name: name.into(),
        map_kva: 0,
        arena,
        ringbuf,
        stack_trace,
        fd_array,
        error,
        ..Default::default()
    }
}

#[test]
fn snapshot_map_ringbuf_threads_option() {
    let m_absent = map_with_shape("ring", None, None, None, None, None);
    let r1 = FailureDumpReport {
        maps: vec![m_absent],
        ..Default::default()
    };
    let snap1 = Snapshot::new(&r1);
    assert!(snap1.map("ring").unwrap().ringbuf().is_none());

    let m_present = map_with_shape(
        "ring2",
        Some(FailureDumpRingbuf {
            capacity: 4096,
            consumer_pos: 100,
            producer_pos: 500,
            pending_pos: 200,
            pending_bytes: 400,
        }),
        None,
        None,
        None,
        None,
    );
    let r2 = FailureDumpReport {
        maps: vec![m_present],
        ..Default::default()
    };
    let snap2 = Snapshot::new(&r2);
    let rb = snap2.map("ring2").unwrap().ringbuf().expect("present");
    assert_eq!(rb.capacity, 4096);
    assert_eq!(rb.pending_bytes, 400);
}

#[test]
fn snapshot_map_arena_threads_option() {
    let m_absent = map_with_shape("arena_a", None, None, None, None, None);
    let r_a = FailureDumpReport {
        maps: vec![m_absent],
        ..Default::default()
    };
    assert!(
        Snapshot::new(&r_a)
            .map("arena_a")
            .unwrap()
            .arena()
            .is_none()
    );

    let m_present = map_with_shape(
        "arena_m",
        None,
        Some(ArenaSnapshot::default()),
        None,
        None,
        None,
    );
    let r = FailureDumpReport {
        maps: vec![m_present],
        ..Default::default()
    };
    let snap = Snapshot::new(&r);
    assert!(snap.map("arena_m").unwrap().arena().is_some());
}

#[test]
fn snapshot_map_fd_array_threads_option() {
    let m_absent = map_with_shape("fda_a", None, None, None, None, None);
    let r_a = FailureDumpReport {
        maps: vec![m_absent],
        ..Default::default()
    };
    assert!(
        Snapshot::new(&r_a)
            .map("fda_a")
            .unwrap()
            .fd_array()
            .is_none()
    );

    let m_present = map_with_shape(
        "fda",
        None,
        None,
        Some(FailureDumpFdArray {
            populated: 3,
            scanned: 5,
            indices: vec![0, 2, 4],
            truncated: false,
            indices_truncated: false,
            unreadable: 0,
        }),
        None,
        None,
    );
    let r = FailureDumpReport {
        maps: vec![m_present],
        ..Default::default()
    };
    let snap = Snapshot::new(&r);
    let fda = snap.map("fda").unwrap().fd_array().expect("present");
    assert_eq!(fda.populated, 3);
    assert_eq!(fda.indices, vec![0, 2, 4]);
}

#[test]
fn snapshot_map_stack_trace_threads_option() {
    let m_absent = map_with_shape("stack_a", None, None, None, None, None);
    let r_a = FailureDumpReport {
        maps: vec![m_absent],
        ..Default::default()
    };
    assert!(
        Snapshot::new(&r_a)
            .map("stack_a")
            .unwrap()
            .stack_trace()
            .is_none()
    );

    let m_present = map_with_shape(
        "stack",
        None,
        None,
        None,
        Some(FailureDumpStackTrace {
            n_buckets: 32,
            entries: Vec::new(),
            truncated: false,
            buckets_unreadable: 0,
        }),
        None,
    );
    let r = FailureDumpReport {
        maps: vec![m_present],
        ..Default::default()
    };
    let snap = Snapshot::new(&r);
    let st = snap.map("stack").unwrap().stack_trace().expect("present");
    assert_eq!(st.n_buckets, 32);
}

#[test]
fn snapshot_map_error_threads_option() {
    let m_absent = map_with_shape("err_a", None, None, None, None, None);
    let r_a = FailureDumpReport {
        maps: vec![m_absent],
        ..Default::default()
    };
    assert!(
        Snapshot::new(&r_a)
            .map("err_a")
            .unwrap()
            .map_error()
            .is_none()
    );

    let m_present = map_with_shape(
        "err_m",
        None,
        None,
        None,
        None,
        Some("BTF offset unresolved".to_string()),
    );
    let r = FailureDumpReport {
        maps: vec![m_present],
        ..Default::default()
    };
    let snap = Snapshot::new(&r);
    assert_eq!(
        snap.map("err_m").unwrap().map_error(),
        Some("BTF offset unresolved")
    );
}

// -------------------------------------------------------------
// `*_unavailable` companion accessors — pin against the
// matching `Option<String>` field on FailureDumpReport. Each
// accessor returns `None` for the default report and
// `Some(&str)` after the walker writes a reason.
// -------------------------------------------------------------

#[test]
fn snapshot_unavailable_accessors_thread_option() {
    let r_absent = FailureDumpReport::default();
    let s = Snapshot::new(&r_absent);
    assert!(s.scx_walker_unavailable().is_none());
    assert!(s.task_enrichments_unavailable().is_none());
    assert!(s.prog_runtime_stats_unavailable().is_none());
    assert!(s.per_node_numa_unavailable().is_none());
    assert!(s.sdt_alloc_unavailable().is_none());

    let r_present = FailureDumpReport {
        scx_walker_unavailable: Some("scx_root unset".into()),
        task_enrichments_unavailable: Some("no task walker available".into()),
        prog_runtime_stats_unavailable: Some("prog accessor unavailable".into()),
        per_node_numa_unavailable: Some("no NUMA walker".into()),
        sdt_alloc_unavailable: Some("sdt symbol absent".into()),
        ..Default::default()
    };
    let s = Snapshot::new(&r_present);
    assert_eq!(s.scx_walker_unavailable(), Some("scx_root unset"));
    assert_eq!(
        s.task_enrichments_unavailable(),
        Some("no task walker available")
    );
    assert_eq!(
        s.prog_runtime_stats_unavailable(),
        Some("prog accessor unavailable")
    );
    assert_eq!(s.per_node_numa_unavailable(), Some("no NUMA walker"));
    assert_eq!(s.sdt_alloc_unavailable(), Some("sdt symbol absent"));
}

// -------------------------------------------------------------
// Serde round-trip pin for the 13 new accessor target types.
// Populates a FailureDumpReport with one entry per accessor,
// serializes to JSON, deserializes back, and hits every new
// accessor to assert field-level values match. Pins the wire
// format against silent `#[serde(skip)]` regressions on any
// of the underlying structs.
// -------------------------------------------------------------

#[test]
fn snapshot_accessor_targets_round_trip_through_serde_json() {
    let original = FailureDumpReport {
        event_counter_timeline: vec![EventCounterSample {
            elapsed_ms: 42,
            select_cpu_fallback: 7,
            ..Default::default()
        }],
        rq_scx_states: vec![crate::monitor::scx_walker::RqScxState {
            cpu: 5,
            ..Default::default()
        }],
        dsq_states: vec![crate::monitor::scx_walker::DsqState {
            nr: 9,
            ..Default::default()
        }],
        scx_sched_state: Some(crate::monitor::scx_walker::ScxSchedState {
            exit_kind: 3,
            ..Default::default()
        }),
        per_cpu_time: vec![PerCpuTimeStats {
            cpu: 3,
            cpustat_user_ns: 9_999,
            ..Default::default()
        }],
        per_node_numa: vec![PerNodeNumaStats {
            node: 1,
            numa_hit: 12_345,
            ..Default::default()
        }],
        task_enrichments: vec![task_enrichment_fixture(101, "rt-test")],
        prog_runtime_stats: vec![ProgRuntimeStats {
            name: "dispatch".into(),
            cnt: 50,
            nsecs: 12_500,
            misses: 2,
        }],
        probe_counters: Some(ProbeBssCounters {
            trigger_count: 17,
            ..Default::default()
        }),
        maps: vec![FailureDumpMap {
            name: "shape_map".into(),
            map_kva: 0,
            map_type: 0,
            value_size: 0,
            max_entries: 0,
            value: None,
            entries: Vec::new(),
            array_entries: Vec::new(),
            percpu_entries: Vec::new(),
            percpu_hash_entries: Vec::new(),
            arena: Some(ArenaSnapshot {
                declared_pages: 256,
                ..Default::default()
            }),
            ringbuf: Some(FailureDumpRingbuf {
                capacity: 8192,
                consumer_pos: 50,
                producer_pos: 700,
                pending_pos: 100,
                pending_bytes: 650,
            }),
            stack_trace: Some(FailureDumpStackTrace {
                n_buckets: 16,
                entries: Vec::new(),
                truncated: true,
                buckets_unreadable: 0,
            }),
            fd_array: Some(FailureDumpFdArray {
                populated: 2,
                scanned: 4,
                indices: vec![1, 3],
                truncated: false,
                indices_truncated: false,
                unreadable: 0,
            }),
            error: Some("decode failed".to_string()),
        }],
        ..Default::default()
    };

    let json = serde_json::to_string(&original).expect("serialize");
    let round_tripped: FailureDumpReport = serde_json::from_str(&json).expect("deserialize");
    let snap = Snapshot::new(&round_tripped);

    assert_eq!(snap.event_counter_timeline().len(), 1);
    assert_eq!(snap.event_counter_timeline()[0].elapsed_ms, 42);
    assert_eq!(snap.event_counter_timeline()[0].select_cpu_fallback, 7);

    assert_eq!(snap.rq_scx_states().len(), 1);
    assert_eq!(snap.rq_scx_states()[0].cpu, 5);
    assert_eq!(snap.dsq_states().len(), 1);
    assert_eq!(snap.dsq_states()[0].nr, 9);
    let sched = snap.scx_sched_state().expect("scx_sched_state");
    assert_eq!(sched.exit_kind, 3);

    assert_eq!(snap.per_cpu_time().len(), 1);
    let cpu_row = snap.per_cpu_time_at(3).expect("cpu 3 row");
    assert_eq!(cpu_row.cpustat_user_ns, 9_999);

    assert_eq!(snap.per_node_numa().len(), 1);
    let numa_row = snap.per_node_numa_at(1).expect("node 1 row");
    assert_eq!(numa_row.numa_hit, 12_345);

    assert_eq!(snap.task_enrichments().len(), 1);
    let task = snap.task_enrichment_by_pid(101).expect("pid 101");
    assert_eq!(task.comm, "rt-test");
    // Fixture defaults pin non-pid/comm fields too — verifies
    // they survive serde without #[serde(skip)] regressions.
    assert_eq!(task.weight, 100);
    assert_eq!(task.prio, 120);

    assert_eq!(snap.prog_runtime_stats().len(), 1);
    let prog = snap
        .prog_runtime_stats_by_name("dispatch")
        .expect("dispatch prog");
    assert_eq!(prog.cnt, 50);
    assert_eq!(prog.nsecs, 12_500);
    assert_eq!(prog.misses, 2);

    let probe = snap.probe_counters().expect("probe_counters");
    assert_eq!(probe.trigger_count, 17);

    let map = snap.map("shape_map").expect("map present");
    let rb = map.ringbuf().expect("ringbuf");
    assert_eq!(rb.capacity, 8192);
    assert_eq!(rb.pending_bytes, 650);
    let arena = map.arena().expect("arena");
    assert_eq!(arena.declared_pages, 256);
    let fda = map.fd_array().expect("fd_array");
    assert_eq!(fda.populated, 2);
    assert_eq!(fda.indices, vec![1, 3]);
    let st = map.stack_trace().expect("stack_trace");
    assert_eq!(st.n_buckets, 16);
    assert!(st.truncated);
    assert_eq!(map.map_error(), Some("decode failed"));
}

// -- SnapshotBridgeEvent + drain_events -------------------------------

/// Construct a bridge whose capture callback always returns the
/// caller-supplied report. Tests that need to trigger
/// `CaptureUnavailable` build their own bridge with a None-returning
/// callback inline.
fn bridge_with_capture_returning(report: FailureDumpReport) -> SnapshotBridge {
    SnapshotBridge::new(std::sync::Arc::new(move |_name| Some(report.clone())))
}

/// `capture` whose callback returns `None` records exactly one
/// `CaptureUnavailable` event under the requested tag and returns
/// `false` (the Op::CaptureSnapshot no-op contract).
#[test]
fn snapshot_bridge_event_capture_unavailable_recorded() {
    let bridge = SnapshotBridge::new(std::sync::Arc::new(|_name| None));
    assert!(!bridge.capture("tag_x"));
    let events = bridge.drain_events();
    assert_eq!(events.len(), 1);
    match &events[0] {
        SnapshotBridgeEvent::CaptureUnavailable { tag } => {
            assert_eq!(tag, "tag_x");
        }
        other => panic!("expected CaptureUnavailable, got {other:?}"),
    }
}

/// `store` of a tag that already has a stored report records
/// exactly one `Overwrite` event with the prior report's `schema`.
#[test]
fn snapshot_bridge_event_overwrite_recorded() {
    let bridge = bridge_with_capture_returning(synthetic_report());
    bridge.store("dup_tag", synthetic_report());
    bridge.store("dup_tag", synthetic_report());
    let events = bridge.drain_events();
    assert_eq!(events.len(), 1);
    match &events[0] {
        SnapshotBridgeEvent::Overwrite { tag, prior_schema } => {
            assert_eq!(tag, "dup_tag");
            assert_eq!(prior_schema, SCHEMA_SINGLE);
        }
        other => panic!("expected Overwrite, got {other:?}"),
    }
}

/// `store` that triggers FIFO cap-eviction records one `Eviction`
/// event per evicted tag, with `cap` reflecting
/// [`MAX_STORED_SNAPSHOTS`] and `new_tag` naming the storing
/// operation that pushed the bridge over the cap.
#[test]
fn snapshot_bridge_event_eviction_recorded() {
    let bridge = bridge_with_capture_returning(synthetic_report());
    for i in 0..MAX_STORED_SNAPSHOTS {
        bridge.store(&format!("tag_{i}"), synthetic_report());
    }
    // The first MAX_STORED_SNAPSHOTS stores fit; none should
    // have evicted anything.
    assert_eq!(bridge.event_count(), 0);
    // One more store crosses the cap — exactly one Eviction.
    bridge.store("overflow_tag", synthetic_report());
    let events = bridge.drain_events();
    assert_eq!(events.len(), 1);
    match &events[0] {
        SnapshotBridgeEvent::Eviction {
            evicted_tag,
            new_tag,
            cap,
        } => {
            assert_eq!(evicted_tag, "tag_0");
            assert_eq!(new_tag, "overflow_tag");
            assert_eq!(*cap, MAX_STORED_SNAPSHOTS);
        }
        other => panic!("expected Eviction, got {other:?}"),
    }
}

/// `drain_events` empties the log — a follow-up call returns an
/// empty vec without re-yielding the previously-drained events.
#[test]
fn snapshot_bridge_drain_events_consumes_log() {
    let bridge = SnapshotBridge::new(std::sync::Arc::new(|_name| None));
    bridge.capture("a");
    bridge.capture("b");
    let first = bridge.drain_events();
    assert_eq!(first.len(), 2);
    let second = bridge.drain_events();
    assert!(second.is_empty(), "second drain must return empty vec");
    assert_eq!(bridge.event_count(), 0);
}

/// `event_count` reports queued events without consuming them —
/// a follow-up `drain_events` still yields the same events.
#[test]
fn snapshot_bridge_event_count_is_non_draining() {
    let bridge = SnapshotBridge::new(std::sync::Arc::new(|_name| None));
    bridge.capture("only");
    assert_eq!(bridge.event_count(), 1);
    // Re-checking does not drain.
    assert_eq!(bridge.event_count(), 1);
    let events = bridge.drain_events();
    assert_eq!(events.len(), 1);
}

/// A bridge whose storage and drain paths run cleanly produces
/// zero events — the structured log is silent on the happy path.
#[test]
fn snapshot_bridge_no_events_on_clean_capture_and_drain() {
    let bridge = bridge_with_capture_returning(synthetic_report());
    assert!(bridge.capture("clean_a"));
    assert!(bridge.capture("clean_b"));
    let _ = bridge.drain_ordered_with_stats();
    assert_eq!(
        bridge.event_count(),
        0,
        "clean capture-then-drain must not record any bridge events",
    );
}

/// `tracing::warn!` is preserved alongside the event push — the
/// "promote, don't replace" contract. Cannot directly observe the
/// warn without a subscriber install; this test pins that the
/// event is still recorded even after a clean drain (the warn
/// path is exercised the same way regardless of subscriber state,
/// so observing the structured event covers both axes).
#[test]
fn snapshot_bridge_event_recorded_even_after_drain() {
    let bridge = SnapshotBridge::new(std::sync::Arc::new(|_name| None));
    bridge.capture("post_drain_tag");
    let _ = bridge.drain_ordered_with_stats();
    // Events are independent of report drain — they remain
    // queued for inspection until drain_events is called.
    assert_eq!(bridge.event_count(), 1);
    let events = bridge.drain_events();
    assert!(
        matches!(
            events[0],
            SnapshotBridgeEvent::CaptureUnavailable { ref tag } if tag == "post_drain_tag",
        ),
        "post-drain event must remain in the log",
    );
}

/// `DrainOrderingInvariantViolation` fires from
/// [`SnapshotBridge::drain_ordered`] when a report exists in
/// `reports` without a matching entry in `order`. The
/// public API keeps the two collections in lock-step, so the
/// only way to exercise this path is to poke `snapshots`
/// directly from inside the test module — simulating an internal
/// refactor bug that desynchronised the two. Pins both the
/// tag field and the `drain_variant` literal so a regression
/// that drops the event push (leaving only the warn) surfaces
/// here.
#[test]
fn snapshot_bridge_event_drain_ordering_invariant_violation_drain_ordered() {
    let bridge = bridge_with_capture_returning(synthetic_report());
    {
        let mut store = bridge.snapshots.lock_unpoisoned();
        store
            .reports
            .insert("orphan_tag".to_string(), synthetic_report());
    }
    let drained = bridge.drain_ordered();
    assert_eq!(drained.len(), 1, "orphan must surface at the tail");
    assert_eq!(drained[0].0, "orphan_tag");
    let events = bridge.drain_events();
    assert_eq!(events.len(), 1);
    match &events[0] {
        SnapshotBridgeEvent::DrainOrderingInvariantViolation { tag, drain_variant } => {
            assert_eq!(tag, "orphan_tag");
            assert_eq!(*drain_variant, "drain_ordered");
        }
        other => panic!("expected DrainOrderingInvariantViolation, got {other:?}"),
    }
}

/// Parallel test for `drain_ordered_with_stats` — the second
/// production site for `DrainOrderingInvariantViolation`. The
/// `drain_variant` discriminator is the only difference; a single
/// test cannot pin both sites simultaneously because each drain
/// method consumes the desync.
#[test]
fn snapshot_bridge_event_drain_ordering_invariant_violation_drain_ordered_with_stats() {
    let bridge = bridge_with_capture_returning(synthetic_report());
    {
        let mut store = bridge.snapshots.lock_unpoisoned();
        store
            .reports
            .insert("orphan_tag".to_string(), synthetic_report());
    }
    let drained = bridge.drain_ordered_with_stats();
    assert_eq!(drained.len(), 1, "orphan must surface at the tail");
    assert_eq!(drained[0].tag, "orphan_tag");
    let events = bridge.drain_events();
    assert_eq!(events.len(), 1);
    match &events[0] {
        SnapshotBridgeEvent::DrainOrderingInvariantViolation { tag, drain_variant } => {
            assert_eq!(tag, "orphan_tag");
            assert_eq!(*drain_variant, "drain_ordered_with_stats");
        }
        other => panic!("expected DrainOrderingInvariantViolation, got {other:?}"),
    }
}

/// The cap-enforcement loop supports multi-iteration eviction
/// (e.g. if `reports.len()` ever exceeds `cap + 1`). The public
/// API never produces this state because `store_internal`
/// inserts one entry and pops one per loop iteration, but a
/// future refactor that pushed N entries in one shot would
/// rely on the loop running N - cap times. Pre-seed
/// `reports + order` in lock-step with `cap + 1` entries, then
/// trigger one store to drive the loop past the cap by 2, and
/// pin that exactly 2 Eviction events fire — one per iteration,
/// in FIFO order. A regression that moved the `events.push`
/// outside the while-loop body (single push after the loop with
/// the last evicted tag) would only fire 1 event and fail here.
#[test]
fn snapshot_bridge_event_eviction_loop_fires_once_per_iteration() {
    let bridge = bridge_with_capture_returning(synthetic_report());
    {
        let mut store = bridge.snapshots.lock_unpoisoned();
        for i in 0..=MAX_STORED_SNAPSHOTS {
            let tag = format!("seed_{i:03}");
            store.reports.insert(tag.clone(), synthetic_report());
            store.order.push_back(tag);
        }
    }
    bridge.store("trigger_tag", synthetic_report());
    let events = bridge.drain_events();
    assert_eq!(
        events.len(),
        2,
        "loop must fire exactly one Eviction per popped entry, not one batched event for the last pop"
    );
    match &events[0] {
        SnapshotBridgeEvent::Eviction {
            evicted_tag,
            new_tag,
            cap,
        } => {
            assert_eq!(evicted_tag, "seed_000", "first pop is FIFO-oldest");
            assert_eq!(new_tag, "trigger_tag");
            assert_eq!(*cap, MAX_STORED_SNAPSHOTS);
        }
        other => panic!("expected Eviction at events[0], got {other:?}"),
    }
    match &events[1] {
        SnapshotBridgeEvent::Eviction {
            evicted_tag,
            new_tag,
            cap,
        } => {
            assert_eq!(evicted_tag, "seed_001", "second pop is FIFO-next-oldest");
            assert_eq!(new_tag, "trigger_tag");
            assert_eq!(*cap, MAX_STORED_SNAPSHOTS);
        }
        other => panic!("expected Eviction at events[1], got {other:?}"),
    }
}

/// `CapInvariantViolation` fires when the cap-enforcement loop
/// in `store_internal` finds `reports.len() > cap` while `order`
/// is empty — the bulk-clear branch nukes everything to restore
/// the invariant. Unreachable through the public API (which
/// always keeps the two collections in lock-step), so this test
/// pre-seeds the desync directly: `cap + 2` orphan entries in
/// `reports` with empty `order`, then `bridge.store("trigger_tag",
/// ...)` to enter `store_internal`. `store_internal` pushes
/// trigger_tag onto `order` BEFORE the loop, so the first loop
/// iteration pops trigger_tag (Eviction event with self-eviction
/// shape — both evicted_tag and new_tag = trigger_tag); the
/// second iteration finds `order` empty while `reports.len()`
/// still over cap and fires CapInvariantViolation, bulk-clears,
/// then breaks. Test asserts both events fire in this order so a
/// regression in either site surfaces.
#[test]
fn snapshot_bridge_event_cap_invariant_violation_recorded() {
    let bridge = bridge_with_capture_returning(synthetic_report());
    {
        let mut store = bridge.snapshots.lock_unpoisoned();
        for i in 0..(MAX_STORED_SNAPSHOTS + 2) {
            store
                .reports
                .insert(format!("orphan_{i:03}"), synthetic_report());
        }
    }
    bridge.store("trigger_tag", synthetic_report());
    let events = bridge.drain_events();
    assert_eq!(
        events.len(),
        2,
        "expected one Eviction (self-evicting trigger_tag from order) + one CapInvariantViolation (bulk-clearing the remaining orphans)"
    );
    match &events[0] {
        SnapshotBridgeEvent::Eviction {
            evicted_tag,
            new_tag,
            cap,
        } => {
            assert_eq!(
                evicted_tag, "trigger_tag",
                "self-eviction: trigger_tag is the only entry in order"
            );
            assert_eq!(new_tag, "trigger_tag");
            assert_eq!(*cap, MAX_STORED_SNAPSHOTS);
        }
        other => panic!("expected Eviction at events[0], got {other:?}"),
    }
    match &events[1] {
        SnapshotBridgeEvent::CapInvariantViolation { reports_len, cap } => {
            assert_eq!(
                *reports_len,
                MAX_STORED_SNAPSHOTS + 2,
                "reports_len at bulk-clear: 66 orphans remain after trigger_tag was evicted in iteration 1"
            );
            assert_eq!(*cap, MAX_STORED_SNAPSHOTS);
        }
        other => panic!("expected CapInvariantViolation at events[1], got {other:?}"),
    }
    let store = bridge.snapshots.lock_unpoisoned();
    assert_eq!(
        store.reports.len(),
        0,
        "bulk-clear must nuke reports to restore the invariant"
    );
    assert_eq!(store.stats.len(), 0);
    assert_eq!(store.elapsed_ms.len(), 0);
}

/// `events` Vec is capped at [`MAX_STORED_EVENTS`] via FIFO
/// eviction. A scenario that triggers many events without
/// draining (e.g. a runaway capture loop) must NOT grow the
/// log unboundedly. Push `cap + 5` events, verify `event_count`
/// is exactly `cap`, then drain and verify the drained vec is
/// `cap + 1` long (cap real events + 1 synthetic
/// [`SnapshotBridgeEvent::EventLogTruncated`] at the tail) with
/// `dropped_count = 5`. Also pins FIFO eviction order: the
/// oldest events are dropped, the newest are retained.
#[test]
fn snapshot_bridge_events_capped_at_max_stored_events() {
    let bridge = SnapshotBridge::new(std::sync::Arc::new(|_name| None));
    let total = MAX_STORED_EVENTS + 5;
    for i in 0..total {
        bridge.capture(&format!("runaway_{i:05}"));
    }
    assert_eq!(
        bridge.event_count(),
        MAX_STORED_EVENTS,
        "events Vec must be capped at MAX_STORED_EVENTS (event_count excludes the synthetic truncation marker)"
    );
    let drained = bridge.drain_events();
    assert_eq!(
        drained.len(),
        MAX_STORED_EVENTS + 1,
        "drain must yield cap real events + 1 synthetic EventLogTruncated tail marker"
    );
    // The first event surviving FIFO eviction is the 6th push
    // (indices 0-4 = 5 events were dropped to make room).
    match &drained[0] {
        SnapshotBridgeEvent::CaptureUnavailable { tag } => {
            assert_eq!(
                tag, "runaway_00005",
                "oldest surviving event must be index 5 (indices 0-4 dropped by FIFO eviction)"
            );
        }
        other => panic!("expected CaptureUnavailable at events[0], got {other:?}"),
    }
    // The tail marker carries the dropped count.
    match drained.last().expect("non-empty drain") {
        SnapshotBridgeEvent::EventLogTruncated { dropped_count } => {
            assert_eq!(
                *dropped_count, 5,
                "5 events were dropped from the front before the cap held"
            );
        }
        other => panic!("expected EventLogTruncated at events[last], got {other:?}"),
    }
}

/// Pushing EXACTLY [`MAX_STORED_EVENTS`] events must NOT trigger
/// any eviction — `push_event`'s cap-enforcement predicate
/// (`events.len() >= MAX_STORED_EVENTS`) is `>=`, so the cap-th
/// push fits without dropping. Drain then
/// yields exactly `cap` events with NO `EventLogTruncated` tail
/// marker. Pins the boundary: a regression that changed `>=`
/// to `>` would silently shift the cap by one and only be
/// catchable via tests that exercise the exact-cap case (the
/// existing over-cap tests would still pass with a one-off
/// dropped_count).
#[test]
fn snapshot_bridge_events_exactly_at_cap_no_truncation_marker() {
    let bridge = SnapshotBridge::new(std::sync::Arc::new(|_name| None));
    for i in 0..MAX_STORED_EVENTS {
        bridge.capture(&format!("at_cap_{i:05}"));
    }
    assert_eq!(
        bridge.event_count(),
        MAX_STORED_EVENTS,
        "exact-cap push must NOT trigger eviction — len == cap, dropped == 0"
    );
    let drained = bridge.drain_events();
    assert_eq!(
        drained.len(),
        MAX_STORED_EVENTS,
        "drain must yield exactly cap events with NO synthetic tail marker"
    );
    assert!(
        drained
            .iter()
            .all(|e| !matches!(e, SnapshotBridgeEvent::EventLogTruncated { .. })),
        "no EventLogTruncated marker at exact-cap — events_dropped must remain 0"
    );
}

/// A second overflow batch after a clean drain must produce a
/// FRESH truncation marker with its own dropped_count — NOT 0
/// (missed reset), NOT carried-forward from the prior batch,
/// NOT accumulated. Pins the events_dropped reset path against
/// regressions where the counter persists across drains (e.g.
/// `events_dropped = events_dropped.saturating_sub(...)`
/// instead of `= 0`, or a reset path that forgets to fire on
/// the empty-events case).
#[test]
fn snapshot_bridge_events_truncation_marker_fresh_after_reset() {
    let bridge = SnapshotBridge::new(std::sync::Arc::new(|_name| None));
    // First overflow: cap+5 push → marker with dropped_count=5.
    for i in 0..(MAX_STORED_EVENTS + 5) {
        bridge.capture(&format!("first_{i:05}"));
    }
    let first = bridge.drain_events();
    match first.last().expect("non-empty first drain") {
        SnapshotBridgeEvent::EventLogTruncated { dropped_count } => {
            assert_eq!(*dropped_count, 5, "first batch dropped 5");
        }
        other => panic!("expected EventLogTruncated tail in first drain, got {other:?}"),
    }
    // Second overflow: cap+7 push → marker with dropped_count=7
    // (fresh count, NOT 5+7=12 stale-accumulated, NOT 0
    // missed-reset).
    for i in 0..(MAX_STORED_EVENTS + 7) {
        bridge.capture(&format!("second_{i:05}"));
    }
    let second = bridge.drain_events();
    match second.last().expect("non-empty second drain") {
        SnapshotBridgeEvent::EventLogTruncated { dropped_count } => {
            assert_eq!(
                *dropped_count, 7,
                "second batch dropped 7 from a clean post-reset counter"
            );
        }
        other => panic!("expected EventLogTruncated tail in second drain, got {other:?}"),
    }
}

/// `events_dropped` resets to 0 after every `drain_events` call —
/// a subsequent drain with no over-cap pushes in between must NOT
/// re-emit a stale truncation marker. Push to overflow, drain
/// (consuming the marker), push within-cap, drain again — the
/// second drain must NOT include EventLogTruncated.
#[test]
fn snapshot_bridge_events_dropped_resets_after_drain() {
    let bridge = SnapshotBridge::new(std::sync::Arc::new(|_name| None));
    for i in 0..(MAX_STORED_EVENTS + 3) {
        bridge.capture(&format!("over_{i:05}"));
    }
    let first = bridge.drain_events();
    assert!(matches!(
        first.last(),
        Some(SnapshotBridgeEvent::EventLogTruncated { dropped_count: 3 })
    ));
    // Second drain immediately — events log is empty, dropped
    // counter reset. Must be empty, NOT carry a stale truncation
    // marker.
    let second = bridge.drain_events();
    assert!(
        second.is_empty(),
        "second drain must NOT re-emit EventLogTruncated — dropped counter reset to 0 after first drain"
    );
    // Push 3 events that fit under the cap. Drain — no
    // truncation marker because nothing was dropped this batch.
    for i in 0..3 {
        bridge.capture(&format!("clean_{i:05}"));
    }
    let third = bridge.drain_events();
    assert_eq!(third.len(), 3, "3 captures, no truncation marker");
    assert!(
        third
            .iter()
            .all(|e| !matches!(e, SnapshotBridgeEvent::EventLogTruncated { .. }))
    );
}

// ---------------------------------------------------------------------------
// v2 unified-accessor API tests: Snapshot::active / vars / live_var
// + SnapshotField array accessors + JsonField rename and array accessors.
// ---------------------------------------------------------------------------

fn make_report_with_maps(maps: Vec<FailureDumpMap>) -> FailureDumpReport {
    let mut r = FailureDumpReport::placeholder("synthetic_test");
    r.is_placeholder = false;
    r.maps = maps;
    r
}

fn make_global_map(name: &str, members: Vec<(&str, RenderedValue)>) -> FailureDumpMap {
    FailureDumpMap {
        name: name.to_string(),
        map_kva: 0,
        map_type: 2,
        max_entries: 1,
        value: Some(RenderedValue::Struct {
            type_name: Some(name.to_string()),
            members: members
                .into_iter()
                .map(|(n, v)| RenderedMember {
                    name: n.to_string(),
                    value: v,
                })
                .collect(),
        }),
        ..FailureDumpMap::default()
    }
}

fn uint_v(v: u64) -> RenderedValue {
    RenderedValue::Uint { bits: 64, value: v }
}

/// A map whose contents failed to render at capture: no `value`,
/// no entries, `error` set — exactly the shape `render_map` emits
/// when the guest-memory read of a map's value region fails.
fn make_errored_map(name: &str, error: &str) -> FailureDumpMap {
    FailureDumpMap {
        name: name.to_string(),
        map_kva: 0,
        map_type: 2,
        max_entries: 1,
        value: None,
        error: Some(error.to_string()),
        ..FailureDumpMap::default()
    }
}

/// A render-failed global-section map (`.bss` naming so `var`
/// considers it) — a render failure surfaces as MapRenderIncomplete,
/// not a false VarNotFound.
#[test]
fn snapshot_var_surfaces_render_failure_not_var_not_found() {
    let report = make_report_with_maps(vec![make_errored_map(
        "scx_test.bss",
        "ARRAY value region unreadable",
    )]);
    let snap = Snapshot::new(&report);
    match snap.var("counter") {
        SnapshotField::Missing(SnapshotError::MapRenderIncomplete { map, error }) => {
            assert_eq!(map, "scx_test.bss");
            assert_eq!(error, "ARRAY value region unreadable");
        }
        other => panic!("expected MapRenderIncomplete, got {other:?}"),
    }
}

/// A successfully-rendered global map with no errored sibling still
/// reports VarNotFound for a genuinely-absent var — the render-failure
/// path must not fire when the contents are present.
#[test]
fn snapshot_var_absent_in_rendered_map_is_var_not_found() {
    let report = make_report_with_maps(vec![make_global_map(
        "scx_test.bss",
        vec![("counter", uint_v(7))],
    )]);
    let snap = Snapshot::new(&report);
    match snap.var("missing") {
        SnapshotField::Missing(SnapshotError::VarNotFound { requested, .. }) => {
            assert_eq!(requested, "missing");
        }
        other => panic!("expected VarNotFound, got {other:?}"),
    }
}

/// `at` on a render-failed map surfaces MapRenderIncomplete rather
/// than IndexOutOfRange { len: 0 } — a capture gap is not an empty map.
#[test]
fn snapshot_at_on_render_failed_map_surfaces_render_failure() {
    let report = make_report_with_maps(vec![make_errored_map("scx_test.bss", "boom")]);
    let snap = Snapshot::new(&report);
    let entry = snap.map("scx_test.bss").unwrap().at(0);
    match entry {
        SnapshotEntry::Missing(SnapshotError::MapRenderIncomplete { map, error }) => {
            assert_eq!(map, "scx_test.bss");
            assert_eq!(error, "boom");
        }
        other => panic!("expected MapRenderIncomplete, got {other:?}"),
    }
}

/// `find` / `max_by` over a render-failed map surface
/// MapRenderIncomplete rather than NoMatch { len: 0 }.
#[test]
fn snapshot_find_and_max_by_on_render_failed_map_surface_render_failure() {
    let report = make_report_with_maps(vec![make_errored_map("scx_test.bss", "boom")]);
    let snap = Snapshot::new(&report);
    let m = snap.map("scx_test.bss").unwrap();
    match m.find(|_| true) {
        SnapshotEntry::Missing(SnapshotError::MapRenderIncomplete { error, .. }) => {
            assert_eq!(error, "boom");
        }
        other => panic!("find: expected MapRenderIncomplete, got {other:?}"),
    }
    match m.max_by(|_| 0) {
        SnapshotEntry::Missing(SnapshotError::MapRenderIncomplete { error, .. }) => {
            assert_eq!(error, "boom");
        }
        other => panic!("max_by: expected MapRenderIncomplete, got {other:?}"),
    }
}

// Narrowing is NOT observable for the single-obj fast arm by
// construction: with one obj there is nothing to narrow away. Adding a
// second global-section obj does not strengthen THIS arm — two obj
// prefixes route active() to the `(multiple, _)` arm (NoActiveScheduler
// error) unless report.active_obj_name is set, which is the separate
// WALKER path exercised by snapshot_active_multi_bss_same_name_with_-
// walker_resolves. So this test correctly pins what the single-obj arm
// can verify: active() SUCCEEDS (does not error) and resolves the obj's
// var through the filtered view.
#[test]
fn snapshot_active_single_obj_returns_filtered_view() {
    let report = make_report_with_maps(vec![make_global_map(
        "scx_test.bss",
        vec![("counter", uint_v(42))],
    )]);
    let snap = Snapshot::new(&report);
    let active = snap.active().expect("single obj => active() succeeds");
    assert_eq!(active.var("counter").as_u64().ok(), Some(42));
}

/// When the active filter has a KVA whitelist whose entries
/// match NONE of the captured `<active_obj>.*` maps' `map_kva`
/// values, the filter narrows to zero — but the operator deserves
/// to see WHICH maps were rejected and WHAT KVAs the walker
/// expected. A generic `VarNotFound { available: [] }` would leave
/// the operator unable to tell "var truly absent from this
/// scheduler's bss" apart from "every captured map's KVA was
/// filtered out (stale capture / KVA aliasing / walker bug)." The
/// dedicated `ActiveFilterExcludedMaps` variant surfaces the
/// captured names + their KVAs, the walker's expected whitelist,
/// and a structural hint about why the filter excluded them.
#[test]
fn snapshot_var_kva_filter_excluded_surfaces_active_filter_excluded_maps() {
    let mut alpha_bss = make_global_map("alpha.bss", vec![("counter", uint_v(42))]);
    alpha_bss.map_kva = 0x1000;
    let mut alpha_data = make_global_map("alpha.data", vec![("flag", uint_v(1))]);
    alpha_data.map_kva = 0x2000;
    let mut report = make_report_with_maps(vec![alpha_bss, alpha_data]);
    report.active_obj_name = Some("alpha".to_string());
    report.active_map_kvas = vec![0x9999];

    let snap = Snapshot::new(&report);
    let active = snap
        .active()
        .expect("active() succeeds — active_obj_name + captured prefix match");
    let err = active
        .var("counter")
        .error()
        .cloned()
        .expect("zero-map filter yields error");
    match err {
        SnapshotError::ActiveFilterExcludedMaps {
            requested,
            active_obj,
            excluded_maps,
            whitelist_kvas,
        } => {
            assert_eq!(requested, "counter");
            assert_eq!(active_obj, "alpha");
            assert_eq!(whitelist_kvas, vec![0x9999]);
            assert_eq!(excluded_maps.len(), 2, "both alpha-prefixed maps excluded");
            assert!(
                excluded_maps
                    .iter()
                    .any(|m| m.name == "alpha.bss" && m.map_kva == 0x1000)
            );
            assert!(
                excluded_maps
                    .iter()
                    .any(|m| m.name == "alpha.data" && m.map_kva == 0x2000)
            );
        }
        other => panic!("expected ActiveFilterExcludedMaps, got {other:?}"),
    }
}

#[test]
fn snapshot_map_kva_filter_excluded_surfaces_active_filter_excluded_maps() {
    let mut alpha_bss = make_global_map("alpha.bss", vec![("counter", uint_v(42))]);
    alpha_bss.map_kva = 0x1000;
    let mut report = make_report_with_maps(vec![alpha_bss]);
    report.active_obj_name = Some("alpha".to_string());
    report.active_map_kvas = vec![0xDEAD];

    let snap = Snapshot::new(&report);
    let active = snap.active().expect("active() succeeds");
    let err = active
        .map("alpha.bss")
        .expect_err("filter excluded the only candidate");
    match err {
        SnapshotError::ActiveFilterExcludedMaps {
            requested,
            active_obj,
            excluded_maps,
            whitelist_kvas,
        } => {
            assert_eq!(requested, "alpha.bss");
            assert_eq!(active_obj, "alpha");
            assert_eq!(whitelist_kvas, vec![0xDEAD]);
            assert_eq!(excluded_maps.len(), 1);
            assert_eq!(excluded_maps[0].name, "alpha.bss");
            assert_eq!(excluded_maps[0].map_kva, 0x1000);
        }
        other => panic!("expected ActiveFilterExcludedMaps, got {other:?}"),
    }
}

#[test]
fn snapshot_live_vars_via_kva_filter_excluded_surfaces_active_filter_excluded_maps() {
    let mut alpha_bss = make_global_map("alpha.bss", vec![("a", uint_v(1)), ("b", uint_v(2))]);
    alpha_bss.map_kva = 0x1000;
    let mut report = make_report_with_maps(vec![alpha_bss]);
    report.active_obj_name = Some("alpha".to_string());
    report.active_map_kvas = vec![0xBEEF];

    let snap = Snapshot::new(&report);
    let active = snap.active().expect("active() succeeds");
    let err = active
        .live_vars_via(&["a", "b"], |_| {
            panic!("picker must not be called when filter excluded everything")
        })
        .unwrap_err();
    match err {
        SnapshotError::ActiveFilterExcludedMaps {
            requested,
            active_obj,
            excluded_maps,
            whitelist_kvas,
        } => {
            assert_eq!(requested, "[a, b]", "joined name list per live_vars_via");
            assert_eq!(active_obj, "alpha");
            assert_eq!(whitelist_kvas, vec![0xBEEF]);
            assert_eq!(excluded_maps.len(), 1);
            assert_eq!(excluded_maps[0].name, "alpha.bss");
            assert_eq!(excluded_maps[0].map_kva, 0x1000);
        }
        other => panic!("expected ActiveFilterExcludedMaps, got {other:?}"),
    }
}

#[test]
fn active_filter_excluded_maps_display_renders_kva_mismatch_hint() {
    let err = SnapshotError::ActiveFilterExcludedMaps {
        requested: "counter".to_string(),
        active_obj: "alpha".to_string(),
        excluded_maps: vec![
            crate::scenario::snapshot::ExcludedMap {
                name: "alpha.bss".to_string(),
                map_kva: 0x1000,
            },
            crate::scenario::snapshot::ExcludedMap {
                name: "alpha.data".to_string(),
                map_kva: 0x2000,
            },
        ],
        whitelist_kvas: vec![0x9999],
    };
    let msg = format!("{err}");
    assert!(msg.contains("counter"), "names the lookup: {msg}");
    assert!(msg.contains("alpha"), "names the active obj: {msg}");
    assert!(msg.contains("0x9999"), "names the whitelist: {msg}");
    assert!(
        msg.contains("alpha.bss@0x1000"),
        "names excluded map: {msg}"
    );
    assert!(
        msg.contains("alpha.data@0x2000"),
        "names excluded map: {msg}"
    );
    assert!(
        msg.contains("Op::ReplaceScheduler") || msg.contains("Op::AttachScheduler"),
        "pure-mismatch hint must name the operation a stale capture would post-date: {msg}"
    );
    assert!(
        msg.contains("Snapshot::vars") && msg.contains("Snapshot::map"),
        "Display must steer the operator at the escape hatches: {msg}"
    );
}

#[test]
fn active_filter_excluded_maps_display_renders_zero_kva_hint() {
    let err = SnapshotError::ActiveFilterExcludedMaps {
        requested: "counter".to_string(),
        active_obj: "alpha".to_string(),
        excluded_maps: vec![crate::scenario::snapshot::ExcludedMap {
            name: "alpha.bss".to_string(),
            map_kva: 0,
        }],
        whitelist_kvas: vec![0x9999],
    };
    let msg = format!("{err}");
    assert!(
        msg.contains("no recorded KVAs") || msg.contains("did not record"),
        "all-zero-kva hint must steer the operator at the pre-walker / capture-bug \
         interpretation rather than the KVA-aliasing one: {msg}"
    );
    assert!(
        msg.contains("Snapshot::vars") && msg.contains("Snapshot::map"),
        "Display must steer the operator at the escape hatches even on the zero-kva path: {msg}"
    );
}

#[test]
fn active_filter_excluded_maps_display_renders_mixed_zero_and_mismatched_hint() {
    let err = SnapshotError::ActiveFilterExcludedMaps {
        requested: "counter".to_string(),
        active_obj: "alpha".to_string(),
        excluded_maps: vec![
            crate::scenario::snapshot::ExcludedMap {
                name: "alpha.bss".to_string(),
                map_kva: 0x1000,
            },
            crate::scenario::snapshot::ExcludedMap {
                name: "alpha.data".to_string(),
                map_kva: 0,
            },
        ],
        whitelist_kvas: vec![0x9999],
    };
    let msg = format!("{err}");
    assert!(
        msg.contains("some captured maps lack KVAs") && msg.contains("post-swap"),
        "mixed-cause hint must name BOTH the missing-KVA AND the post-swap-window class \
         so the operator sees both possible causes: {msg}"
    );
}

#[test]
fn snapshot_live_var_via_kva_filter_excluded_surfaces_active_filter_excluded_maps() {
    let mut alpha_bss = make_global_map("alpha.bss", vec![("counter", uint_v(42))]);
    alpha_bss.map_kva = 0x1000;
    let mut report = make_report_with_maps(vec![alpha_bss]);
    report.active_obj_name = Some("alpha".to_string());
    report.active_map_kvas = vec![0xCAFE];

    let snap = Snapshot::new(&report);
    let active = snap.active().expect("active() succeeds");
    let err = active
        .live_var_via("counter", |_| {
            panic!("picker must not run when filter excluded everything")
        })
        .error()
        .cloned()
        .expect("live_var_via must surface error from excluded-filter helper");
    match err {
        SnapshotError::ActiveFilterExcludedMaps {
            requested,
            active_obj,
            excluded_maps,
            whitelist_kvas,
        } => {
            assert_eq!(
                requested, "counter",
                "single-name path keeps the bare name, no joined brackets"
            );
            assert_eq!(active_obj, "alpha");
            assert_eq!(whitelist_kvas, vec![0xCAFE]);
            assert_eq!(excluded_maps.len(), 1);
            assert_eq!(excluded_maps[0].name, "alpha.bss");
        }
        other => panic!("expected ActiveFilterExcludedMaps, got {other:?}"),
    }
}

#[test]
fn snapshot_live_var_kva_filter_excluded_surfaces_active_filter_excluded_maps() {
    let mut alpha_bss = make_global_map("alpha.bss", vec![("counter", uint_v(42))]);
    alpha_bss.map_kva = 0x1000;
    let mut report = make_report_with_maps(vec![alpha_bss]);
    report.active_obj_name = Some("alpha".to_string());
    report.active_map_kvas = vec![0xCAFE];

    let snap = Snapshot::new(&report);
    let err = snap
        .live_var("counter")
        .error()
        .cloned()
        .expect("live_var must surface the same excluded-filter error as live_var_via");
    assert!(
        matches!(err, SnapshotError::ActiveFilterExcludedMaps { .. }),
        "live_var convenience wrapper must reach the new variant, got {err:?}"
    );
}

#[test]
fn snapshot_var_kva_filter_admits_all_falls_through_to_var_not_found() {
    // When the KVA whitelist admits every captured obj-prefix map but the requested
    // name is not present, the lookup miss is a real absence — we must NOT falsely
    // steer the operator at the filter. Fall through to VarNotFound.
    let mut alpha_bss = make_global_map("alpha.bss", vec![("counter", uint_v(42))]);
    alpha_bss.map_kva = 0x1000;
    let mut alpha_data = make_global_map("alpha.data", vec![("flag", uint_v(1))]);
    alpha_data.map_kva = 0x2000;
    let mut report = make_report_with_maps(vec![alpha_bss, alpha_data]);
    report.active_obj_name = Some("alpha".to_string());
    report.active_map_kvas = vec![0x1000, 0x2000];

    let snap = Snapshot::new(&report);
    let active = snap.active().expect("active() succeeds");
    let err = active
        .var("does_not_exist")
        .error()
        .cloned()
        .expect("absent name still yields an error");
    assert!(
        matches!(err, SnapshotError::VarNotFound { .. }),
        "filter admits everything → lookup miss is a real typo, must surface as \
         VarNotFound (not ActiveFilterExcludedMaps): {err:?}"
    );
}

#[test]
fn snapshot_var_partial_admit_with_missing_name_falls_through_to_var_not_found() {
    // Same-binary post-swap case: one of two obj-prefix maps survives the KVA
    // filter, the other is rejected. Requested name is absent from the admitted
    // map. The variant must NOT fire — the admitted set is non-empty, so the
    // miss is a real absence, not a filter artefact.
    let mut alpha_bss = make_global_map("alpha.bss", vec![("counter", uint_v(42))]);
    alpha_bss.map_kva = 0x1000;
    let mut alpha_data = make_global_map("alpha.data", vec![("flag", uint_v(1))]);
    alpha_data.map_kva = 0x2000;
    let mut report = make_report_with_maps(vec![alpha_bss, alpha_data]);
    report.active_obj_name = Some("alpha".to_string());
    report.active_map_kvas = vec![0x1000];

    let snap = Snapshot::new(&report);
    let active = snap.active().expect("active() succeeds");
    let err = active
        .var("flag")
        .error()
        .cloned()
        .expect("absent name still yields an error");
    assert!(
        matches!(err, SnapshotError::VarNotFound { .. }),
        "filter admitted alpha.bss; lookup of 'flag' (genuinely absent from alpha.bss) \
         must surface as VarNotFound rather than falsely blaming the filter: {err:?}"
    );
}

#[test]
fn active_filter_excluded_maps_serde_round_trip() {
    let err = SnapshotError::ActiveFilterExcludedMaps {
        requested: "counter".to_string(),
        active_obj: "alpha".to_string(),
        excluded_maps: vec![crate::scenario::snapshot::ExcludedMap {
            name: "alpha.bss".to_string(),
            map_kva: 0x1000,
        }],
        whitelist_kvas: vec![0x9999, 0xCAFE],
    };
    let json = serde_json::to_string(&err).expect("serialize");
    let back: SnapshotError = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(err, back, "round-trip preserves every field");
}

#[test]
fn snapshot_active_multi_obj_returns_no_active_scheduler() {
    let report = make_report_with_maps(vec![
        make_global_map("scx_a.bss", vec![("counter", uint_v(1))]),
        make_global_map("scx_b.bss", vec![("counter", uint_v(2))]),
    ]);
    let snap = Snapshot::new(&report);
    let err = snap
        .active()
        .expect_err("multiple obj names => NoActiveScheduler");
    assert!(matches!(err, SnapshotError::NoActiveScheduler { .. }));
    let msg = err.to_string();
    assert!(
        msg.contains("vars(name)"),
        "diagnostic must steer the operator at vars(): {msg}"
    );
}

#[test]
fn snapshot_active_on_placeholder_returns_placeholder_error() {
    let report = FailureDumpReport::placeholder("freeze_timed_out");
    let snap = Snapshot::new(&report);
    assert!(matches!(
        snap.active(),
        Err(SnapshotError::PlaceholderSnapshot { .. })
    ));
}

/// Same-binary `Op::ReplaceScheduler` swap window with the walker
/// unavailable: two `bpf_bpf.bss` copies share the prefix; the
/// helper at `monitor/dump/mod.rs` returned `None` so
/// `active_obj_name` is `None` and `active_map_kvas` is empty.
/// The consumer's per-section count detects bss_count=2 for the
/// single prefix and surfaces `NoActiveScheduler` with a
/// multi-copy reason — preventing the silent `AmbiguousVar`
/// downstream that this whole batch exists to fix.
#[test]
fn snapshot_active_multi_bss_same_name_no_walker_returns_no_active_scheduler() {
    let mut bss_a = make_global_map("bpf_bpf.bss", vec![("counter", uint_v(1))]);
    bss_a.map_kva = 0x1000;
    let mut bss_b = make_global_map("bpf_bpf.bss", vec![("counter", uint_v(2))]);
    bss_b.map_kva = 0x2000;
    let report = make_report_with_maps(vec![bss_a, bss_b]);
    let snap = Snapshot::new(&report);
    let err = snap
        .active()
        .expect_err("multi-bss same-name + no walker => NoActiveScheduler");
    assert!(matches!(err, SnapshotError::NoActiveScheduler { .. }));
    let msg = err.to_string();
    assert!(
        msg.contains("bpf_bpf.bss × 2"),
        "diagnostic must name the multi-copy section + count: {msg}"
    );
    assert!(
        msg.contains("max_by_sum_u64") || msg.contains("max_by_counter_value"),
        "diagnostic must steer at the picker-based disambiguators: {msg}"
    );
}

/// Multi-`.data` (rather than `.bss`) symmetrically forces
/// `NoActiveScheduler` — pins that the consumer's per-section
/// count check covers all three section types.
#[test]
fn snapshot_active_multi_data_same_name_no_walker_returns_no_active_scheduler() {
    let bss = make_global_map("bpf_bpf.bss", vec![("counter", uint_v(0))]);
    let mut data_a = make_global_map("bpf_bpf.data", vec![("flag", uint_v(1))]);
    data_a.map_kva = 0x1000;
    let mut data_b = make_global_map("bpf_bpf.data", vec![("flag", uint_v(2))]);
    data_b.map_kva = 0x2000;
    let report = make_report_with_maps(vec![bss, data_a, data_b]);
    let snap = Snapshot::new(&report);
    let err = snap
        .active()
        .expect_err("multi-data same-name + no walker => NoActiveScheduler");
    assert!(matches!(err, SnapshotError::NoActiveScheduler { .. }));
    let msg = err.to_string();
    assert!(
        msg.contains("bpf_bpf.data × 2"),
        ".data multi-copy must be named in the diagnostic: {msg}"
    );
}

/// Multi-`.rodata` symmetrically forces `NoActiveScheduler`.
#[test]
fn snapshot_active_multi_rodata_same_name_no_walker_returns_no_active_scheduler() {
    let bss = make_global_map("bpf_bpf.bss", vec![("counter", uint_v(0))]);
    let data = make_global_map("bpf_bpf.data", vec![("flag", uint_v(0))]);
    let mut rodata_a = make_global_map("bpf_bpf.rodata", vec![("const", uint_v(7))]);
    rodata_a.map_kva = 0x1000;
    let mut rodata_b = make_global_map("bpf_bpf.rodata", vec![("const", uint_v(7))]);
    rodata_b.map_kva = 0x2000;
    let report = make_report_with_maps(vec![bss, data, rodata_a, rodata_b]);
    let snap = Snapshot::new(&report);
    let err = snap
        .active()
        .expect_err("multi-rodata same-name + no walker => NoActiveScheduler");
    assert!(matches!(err, SnapshotError::NoActiveScheduler { .. }));
    let msg = err.to_string();
    assert!(
        msg.contains("bpf_bpf.rodata × 2"),
        ".rodata multi-copy must be named in the diagnostic: {msg}"
    );
}

/// Same-binary multi-bss BUT the walker IS available
/// (`active_obj_name = Some` + `active_map_kvas` non-empty). The
/// walker-fast-path in `Snapshot::active` returns `Ok` with the
/// KVA whitelist threaded through; `var("counter")` then narrows
/// to the single live bss copy via maps_iter's KVA filter and
/// resolves cleanly. Pins that the new multi-copy detection
/// does NOT regress the walker-resolved happy path.
#[test]
fn snapshot_active_multi_bss_same_name_with_walker_resolves() {
    let mut live_bss = make_global_map("bpf_bpf.bss", vec![("counter", uint_v(99))]);
    live_bss.map_kva = 0x1000;
    let mut stale_bss = make_global_map("bpf_bpf.bss", vec![("counter", uint_v(7))]);
    stale_bss.map_kva = 0x2000;
    let mut report = make_report_with_maps(vec![live_bss, stale_bss]);
    report.active_obj_name = Some("bpf_bpf".to_string());
    report.active_map_kvas = vec![0x1000];
    let snap = Snapshot::new(&report);
    let active = snap
        .active()
        .expect("active_obj_name + non-empty active_map_kvas => Ok");
    let counter = active
        .var("counter")
        .as_u64()
        .expect("KVA filter narrows to the live bss");
    assert_eq!(
        counter, 99,
        "live bss (kva 0x1000, value 99) must be picked, not the stale one (value 7)"
    );
}

/// Same-binary multi-bss with `active_obj_name = Some` BUT
/// `active_map_kvas` empty (walker resolved the obj name via
/// Phase 1 short-circuit, didn't publish a whitelist; then later
/// a new bss copy arrived via a swap). The walker-fast-path
/// detects the multi-copy section AND empty whitelist; surfaces
/// `NoActiveScheduler` with the multi-copy reason instead of
/// silently admitting both copies through the obj-prefix-only
/// filter.
#[test]
fn snapshot_active_walker_obj_name_set_but_empty_whitelist_multi_copy_errors() {
    let mut live_bss = make_global_map("bpf_bpf.bss", vec![("counter", uint_v(99))]);
    live_bss.map_kva = 0x1000;
    let mut stale_bss = make_global_map("bpf_bpf.bss", vec![("counter", uint_v(7))]);
    stale_bss.map_kva = 0x2000;
    let mut report = make_report_with_maps(vec![live_bss, stale_bss]);
    report.active_obj_name = Some("bpf_bpf".to_string());
    // active_map_kvas intentionally empty.
    let snap = Snapshot::new(&report);
    let err = snap
        .active()
        .expect_err("active_obj_name set + empty whitelist + multi-copy => NoActiveScheduler");
    assert!(matches!(err, SnapshotError::NoActiveScheduler { .. }));
    let msg = err.to_string();
    assert!(
        msg.contains("bpf_bpf.bss × 2"),
        "diagnostic must name the multi-copy section: {msg}"
    );
}

/// Raw `var()` (no explicit `.active()`) auto-falls-back to the
/// active-scheduler resolution when the raw scan is ambiguous.
/// Two same-name `bpf_bpf.bss` copies expose `counter`; the walker
/// published `active_obj_name` + a single-entry `active_map_kvas`
/// whitelist. `var("counter")` finds 2 raw hits, takes `var()`'s
/// `active_obj.is_none() && let Ok(active) = self.active()`
/// fallback arm, and the KVA filter narrows to the live copy.
/// The only existing ambiguity test uses two DISTINCT obj prefixes
/// so active() fails and execution falls to the AmbiguousVar arm;
/// the walker-resolved fallback-SUCCESS path is exercised only here.
#[test]
fn snapshot_var_autofallback_to_active_resolves_ambiguity() {
    let mut live_bss = make_global_map("bpf_bpf.bss", vec![("counter", uint_v(99))]);
    live_bss.map_kva = 0x1000;
    let mut stale_bss = make_global_map("bpf_bpf.bss", vec![("counter", uint_v(7))]);
    stale_bss.map_kva = 0x2000;
    let mut report = make_report_with_maps(vec![live_bss, stale_bss]);
    report.active_obj_name = Some("bpf_bpf".to_string());
    report.active_map_kvas = vec![0x1000];
    let snap = Snapshot::new(&report);
    // NOT snap.active().var() — raw var() must auto-narrow internally.
    assert_eq!(
        snap.var("counter").as_u64().unwrap(),
        99,
        "raw var() auto-narrows the two same-name bpf_bpf.bss copies \
         (kva 0x1000=99 live, 0x2000=7 stale) down to the whitelisted live copy"
    );
}

/// `active()` walker fast-path with `active_obj_name = Some` AND an
/// EMPTY `active_map_kvas` whitelist AND a single global-section
/// copy returns `Ok` with an empty (`&[]`) KVA filter — the
/// post-multi-copy-guard `Ok` arm. Every other
/// `active_obj_name = Some` test carries a non-empty whitelist (the
/// `!active_map_kvas.is_empty()` arm); the two empty-whitelist
/// walker tests are both multi-copy and hit the multi-copy
/// `NoActiveScheduler` error arm. The single-copy + empty-whitelist
/// Ok arm is reached only here.
#[test]
fn snapshot_active_walker_obj_name_single_copy_empty_whitelist_resolves() {
    let mut report = make_report_with_maps(vec![make_global_map(
        "bpf_bpf.bss",
        vec![("counter", uint_v(55))],
    )]);
    report.active_obj_name = Some("bpf_bpf".to_string());
    // active_map_kvas left empty (make_report_with_maps default).
    let snap = Snapshot::new(&report);
    let active = snap
        .active()
        .expect("active_obj_name + single copy + empty whitelist => Ok");
    assert_eq!(active.var("counter").as_u64().unwrap(), 55);
    assert_eq!(active.map_count(), 1);
}

/// `map_count()` reflects the active filter: an unfiltered view
/// counts every captured map; an `active()`-filtered view counts
/// only the active obj's maps. Three maps across two obj prefixes
/// (alpha.bss, alpha.data, beta.bss); `active_obj_name = Some("alpha")`
/// narrows to the two alpha.* maps and excludes beta.bss.
#[test]
fn snapshot_map_count_reflects_active_filter_vs_unfiltered() {
    let mut report = make_report_with_maps(vec![
        make_global_map("alpha.bss", vec![("counter", uint_v(1))]),
        make_global_map("alpha.data", vec![("flag", uint_v(1))]),
        make_global_map("beta.bss", vec![("other", uint_v(1))]),
    ]);
    report.active_obj_name = Some("alpha".to_string());
    let snap = Snapshot::new(&report);
    assert_eq!(
        snap.map_count(),
        3,
        "unfiltered view counts every captured map"
    );
    let active = snap
        .active()
        .expect("active_obj_name names a captured prefix");
    assert_eq!(
        active.map_count(),
        2,
        "active() narrows to alpha.bss + alpha.data, excludes the unrelated beta.bss"
    );
}

/// MIXED render-failure case: a successfully-rendered global map
/// that does NOT carry the requested name PLUS a separate errored
/// global map. `var()` finds zero hits, then surfaces
/// `MapRenderIncomplete` from its render-gap arm (the
/// `value.is_none() && error.is_some()` map probe) rather than
/// VarNotFound — with a real map rendered, the search cannot
/// confirm absence, so the render gap must still win. The existing
/// render-failure test has ONLY the errored map present.
#[test]
fn snapshot_var_render_failure_shadows_absence_when_other_maps_render() {
    let report = make_report_with_maps(vec![
        make_global_map("scx_test.bss", vec![("present", uint_v(1))]),
        make_errored_map("scx_test.rodata", "value region unreadable"),
    ]);
    let snap = Snapshot::new(&report);
    match snap.var("absent_name") {
        SnapshotField::Missing(SnapshotError::MapRenderIncomplete { map, error }) => {
            assert_eq!(map, "scx_test.rodata");
            assert_eq!(error, "value region unreadable");
        }
        other => panic!("expected MapRenderIncomplete (render gap shadows absence), got {other:?}"),
    }
}

#[test]
fn snapshot_var_on_placeholder_returns_placeholder_error() {
    let report = FailureDumpReport::placeholder("freeze_timed_out");
    let snap = Snapshot::new(&report);
    let f = snap.var("anything");
    assert!(matches!(
        f.error(),
        Some(SnapshotError::PlaceholderSnapshot { .. })
    ));
}

#[test]
fn snapshot_map_on_placeholder_returns_placeholder_error() {
    let report = FailureDumpReport::placeholder("freeze_timed_out");
    let snap = Snapshot::new(&report);
    assert!(matches!(
        snap.map("anything"),
        Err(SnapshotError::PlaceholderSnapshot { .. })
    ));
}

#[test]
fn snapshot_vars_iterates_every_copy() {
    let report = make_report_with_maps(vec![
        make_global_map("scx_a.bss", vec![("shared", uint_v(11))]),
        make_global_map("scx_b.bss", vec![("shared", uint_v(22))]),
    ]);
    let snap = Snapshot::new(&report);
    let mut hits: Vec<(String, u64)> = snap
        .vars("shared")
        .filter_map(|(name, f)| f.as_u64().ok().map(|v| (name.to_string(), v)))
        .collect();
    hits.sort();
    assert_eq!(
        hits,
        vec![("scx_a.bss".to_string(), 11), ("scx_b.bss".to_string(), 22)],
    );
}

#[test]
fn snapshot_live_var_folds_no_active_scheduler_into_field() {
    let report = make_report_with_maps(vec![
        make_global_map("scx_a.bss", vec![("x", uint_v(1))]),
        make_global_map("scx_b.bss", vec![("x", uint_v(2))]),
    ]);
    let snap = Snapshot::new(&report);
    let f = snap.live_var("x");
    assert!(matches!(
        f.error(),
        Some(SnapshotError::NoActiveScheduler { .. })
    ));
}

#[test]
fn snapshot_field_as_u32_array_success_and_overflow() {
    let arr_ok = RenderedValue::Array {
        len: 3,
        elements: vec![uint_v(1), uint_v(2), uint_v(3)],
    };
    let field = SnapshotField::Value(&arr_ok);
    assert_eq!(field.as_u32_array().unwrap(), vec![1u32, 2, 3]);
    let arr_overflow = RenderedValue::Array {
        len: 1,
        elements: vec![uint_v(u64::from(u32::MAX) + 1)],
    };
    let field = SnapshotField::Value(&arr_overflow);
    assert!(matches!(
        field.as_u32_array(),
        Err(SnapshotError::TypeMismatch { .. })
    ));
}

#[test]
fn snapshot_field_as_u64_array_rejects_struct() {
    let s = RenderedValue::Struct {
        type_name: None,
        members: vec![],
    };
    let field = SnapshotField::Value(&s);
    assert!(matches!(
        field.as_u64_array(),
        Err(SnapshotError::TypeMismatch { .. })
    ));
}

#[test]
fn json_field_get_renamed_from_path_walks_dotted() {
    use serde_json::json;
    let v = json!({"a": {"b": {"c": 42}}});
    let field = stats_path(&v, "a");
    let inner = field.get("b.c");
    assert_eq!(inner.as_u64().unwrap(), 42);
}

#[test]
fn json_field_as_u32_array_extracts() {
    use serde_json::json;
    let v = json!({"counters": [1, 2, 3]});
    let field = stats_path(&v, "counters");
    assert_eq!(field.as_u32_array().unwrap(), vec![1u32, 2, 3]);
}

#[test]
fn json_field_as_bool_array_rejects_mixed() {
    use serde_json::json;
    let v = json!([true, 1, false]);
    let field = stats_path(&v, "");
    assert!(matches!(
        field.as_bool_array(),
        Err(SnapshotError::TypeMismatch { .. })
    ));
}

#[test]
fn snapshot_field_as_u32_array_on_percpu_key_returns_type_mismatch() {
    let field = SnapshotField::PercpuKey { key: 7 };
    let err = field
        .as_u32_array()
        .expect_err("percpu key has no array shape");
    let msg = err.to_string();
    assert!(
        msg.contains("percpu key"),
        "diagnostic must name the percpu-key actual type so the operator \
         knows the cpu-narrowing went wrong, not the array request: {msg}"
    );
}

#[test]
fn snapshot_field_as_u64_array_on_percpu_key_returns_type_mismatch() {
    let field = SnapshotField::PercpuKey { key: 42 };
    let err = field
        .as_u64_array()
        .expect_err("percpu key has no array shape");
    let msg = err.to_string();
    assert!(
        msg.contains("percpu key"),
        "diagnostic must name the percpu-key actual type: {msg}"
    );
}

#[test]
fn snapshot_field_as_i64_array_on_percpu_key_returns_type_mismatch() {
    let field = SnapshotField::PercpuKey { key: 9 };
    let err = field
        .as_i64_array()
        .expect_err("percpu key has no array shape");
    assert!(err.to_string().contains("percpu key"));
}

#[test]
fn snapshot_field_as_f64_array_on_percpu_key_returns_type_mismatch() {
    let field = SnapshotField::PercpuKey { key: 11 };
    let err = field
        .as_f64_array()
        .expect_err("percpu key has no array shape");
    assert!(err.to_string().contains("percpu key"));
}

#[test]
fn snapshot_field_as_bool_array_on_percpu_key_returns_type_mismatch() {
    let field = SnapshotField::PercpuKey { key: 5 };
    let err = field
        .as_bool_array()
        .expect_err("percpu key has no array shape");
    assert!(err.to_string().contains("percpu key"));
}

#[test]
fn snapshot_field_iter_members_walks_array_of_struct() {
    let arr = RenderedValue::Array {
        len: 3,
        elements: vec![
            RenderedValue::Struct {
                type_name: Some("entry".into()),
                members: vec![
                    RenderedMember {
                        name: "id".into(),
                        value: uint_v(10),
                    },
                    RenderedMember {
                        name: "weight".into(),
                        value: uint_v(100),
                    },
                ],
            },
            RenderedValue::Struct {
                type_name: Some("entry".into()),
                members: vec![
                    RenderedMember {
                        name: "id".into(),
                        value: uint_v(20),
                    },
                    RenderedMember {
                        name: "weight".into(),
                        value: uint_v(200),
                    },
                ],
            },
            RenderedValue::Struct {
                type_name: Some("entry".into()),
                members: vec![
                    RenderedMember {
                        name: "id".into(),
                        value: uint_v(30),
                    },
                    RenderedMember {
                        name: "weight".into(),
                        value: uint_v(300),
                    },
                ],
            },
        ],
    };
    let field = SnapshotField::Value(&arr);
    let ids: Vec<u64> = field
        .iter_members()
        .filter_map(|el| el.get("id").as_u64().ok())
        .collect();
    assert_eq!(ids, vec![10, 20, 30]);
}

#[test]
fn snapshot_field_iter_members_peels_ptr_deref() {
    let arr_inner = RenderedValue::Array {
        len: 2,
        elements: vec![uint_v(7), uint_v(11)],
    };
    let ptr_to_arr = RenderedValue::Ptr {
        value: 0xffff_8000_0000_2000,
        deref: Some(Box::new(arr_inner)),
        deref_skipped_reason: None,
        cast_annotation: None,
    };
    let field = SnapshotField::Value(&ptr_to_arr);
    let scalars: Vec<u64> = field
        .iter_members()
        .filter_map(|el| el.as_u64().ok())
        .collect();
    assert_eq!(scalars, vec![7, 11]);
}

#[test]
fn snapshot_field_iter_members_peels_truncated() {
    let arr_inner = RenderedValue::Array {
        len: 5,
        elements: vec![uint_v(1), uint_v(2), uint_v(3)],
    };
    let trunc = RenderedValue::Truncated {
        needed: 5 * 8,
        had: 3 * 8,
        partial: Box::new(arr_inner),
    };
    let field = SnapshotField::Value(&trunc);
    let scalars: Vec<u64> = field
        .iter_members()
        .filter_map(|el| el.as_u64().ok())
        .collect();
    assert_eq!(scalars, vec![1, 2, 3]);
}

#[test]
fn snapshot_field_iter_members_yields_empty_for_non_array_shapes() {
    let scalar = uint_v(42);
    assert_eq!(SnapshotField::Value(&scalar).iter_members().count(), 0);

    let s = RenderedValue::Struct {
        type_name: None,
        members: vec![RenderedMember {
            name: "x".into(),
            value: uint_v(1),
        }],
    };
    assert_eq!(SnapshotField::Value(&s).iter_members().count(), 0);

    assert_eq!(
        SnapshotField::PercpuKey { key: 3 }.iter_members().count(),
        0
    );

    let missing = SnapshotField::Missing(SnapshotError::VarNotFound {
        requested: "x".into(),
        available: vec![],
    });
    assert_eq!(missing.iter_members().count(), 0);
}

#[test]
fn json_field_iter_members_walks_array_of_object() {
    use serde_json::json;
    let v = json!({
        "entries": [
            {"id": 10, "weight": 100},
            {"id": 20, "weight": 200},
            {"id": 30, "weight": 300},
        ],
    });
    let field = stats_path(&v, "entries");
    let ids: Vec<u64> = field
        .iter_members()
        .filter_map(|el| el.get("id").as_u64().ok())
        .collect();
    assert_eq!(ids, vec![10, 20, 30]);
}

#[test]
fn json_field_iter_members_yields_empty_for_non_array() {
    use serde_json::json;
    let scalar = json!(42);
    assert_eq!(stats_path(&scalar, "").iter_members().count(), 0);

    let obj = json!({"a": 1, "b": 2});
    assert_eq!(stats_path(&obj, "").iter_members().count(), 0);

    let missing_root = json!({"a": 1});
    assert_eq!(
        stats_path(&missing_root, "missing").iter_members().count(),
        0
    );

    // JSON null is its own variant — verify iter_members doesn't
    // treat it as array.
    let null = json!(null);
    assert_eq!(stats_path(&null, "").iter_members().count(), 0);
}

#[test]
fn json_field_iter_members_walks_array_of_arrays() {
    use serde_json::json;
    let v = json!([[1, 2, 3], [4, 5, 6]]);
    // Each yielded element is itself a JSON array; iter_members on
    // each yields its scalar elements. Validates that the empty-
    // iterator fallback doesn't swallow nested arrays.
    let sums: Vec<u64> = stats_path(&v, "")
        .iter_members()
        .map(|inner| inner.iter_members().filter_map(|e| e.as_u64().ok()).sum())
        .collect();
    assert_eq!(sums, vec![6, 15]);
}

#[test]
fn snapshot_field_iter_members_on_empty_array_yields_nothing() {
    // Empty Array {len: 0, elements: vec![]} — separately from the
    // non-array-shapes case, pin that an explicitly-empty array
    // matches the empty-iterator contract (no surprise yield of a
    // placeholder element).
    let empty = RenderedValue::Array {
        len: 0,
        elements: vec![],
    };
    let field = SnapshotField::Value(&empty);
    assert_eq!(field.iter_members().count(), 0);
}

// ---------- live_var_via (Snapshot::live_var_via picker disambig) ----------

/// Build a multi-instance report: two `<obj>.bss` maps, both
/// carrying the same top-level member `tick`. Mirrors the
/// pre/post Op::ReplaceScheduler shape where the same scheduler
/// binary is loaded twice and the principled walker cannot tell
/// the bss copies apart from prefix alone — the [`Snapshot::live_var_via`]
/// picker is the disambiguator the caller supplies.
fn two_instance_report(values: (u64, u64)) -> FailureDumpReport {
    let mk_bss = |obj: &str, value: u64| FailureDumpMap {
        name: format!("{obj}.bss"),
        map_kva: 0,
        map_type: 2,
        value_size: 8,
        max_entries: 1,
        value: Some(RenderedValue::Struct {
            type_name: Some(".bss".into()),
            members: vec![RenderedMember {
                name: "tick".into(),
                value: RenderedValue::Uint { bits: 64, value },
            }],
        }),
        entries: Vec::new(),
        array_entries: Vec::new(),
        percpu_entries: Vec::new(),
        percpu_hash_entries: Vec::new(),
        arena: None,
        ringbuf: None,
        stack_trace: None,
        fd_array: None,
        error: None,
    };
    FailureDumpReport {
        schema: SCHEMA_SINGLE.to_string(),
        active_map_kvas: Vec::new(),
        maps: vec![mk_bss("alpha", values.0), mk_bss("beta", values.1)],
        ..Default::default()
    }
}

#[test]
fn live_var_via_picker_selects_named_candidate() {
    let r = two_instance_report((1, 2));
    let snap = Snapshot::new(&r);
    let f = snap.live_var_via("tick", |cands| {
        cands.iter().position(|(obj, _)| *obj == "beta.bss")
    });
    assert_eq!(
        f.as_u64().unwrap(),
        2,
        "picker chose beta.bss → must read beta's tick value",
    );
}

#[test]
fn live_var_via_picker_returns_none_surfaces_projection_failed() {
    let r = two_instance_report((1, 2));
    let snap = Snapshot::new(&r);
    let f = snap.live_var_via("tick", |_| None);
    let err = f
        .error()
        .expect("None picker must surface SnapshotField::Missing");
    match err {
        SnapshotError::ProjectionFailed { reason } => {
            assert!(
                reason.contains("live_var_via picker for 'tick' returned None"),
                "reason must name the picker source: {reason}",
            );
        }
        other => panic!("expected ProjectionFailed, got {other:?}"),
    }
}

#[test]
fn live_var_via_picker_out_of_range_surfaces_projection_failed() {
    let r = two_instance_report((1, 2));
    let snap = Snapshot::new(&r);
    let f = snap.live_var_via("tick", |_| Some(99));
    let err = f.error().expect("out-of-range index must surface error");
    match err {
        SnapshotError::ProjectionFailed { reason } => {
            assert!(
                reason.contains("index 99 out of range") && reason.contains("count = 2"),
                "reason must name the bad index AND the candidate count: {reason}",
            );
        }
        other => panic!("expected ProjectionFailed, got {other:?}"),
    }
}

#[test]
fn live_var_via_no_candidates_surfaces_var_not_found() {
    let r = two_instance_report((1, 2));
    let snap = Snapshot::new(&r);
    // Pick a name absent from every bss map — candidates is empty
    // before the picker is invoked; must surface VarNotFound with
    // the available globals list (not ProjectionFailed).
    let f = snap.live_var_via("absent_member", |_| panic!("picker must NOT be called"));
    let err = f.error().expect("missing var must carry an error");
    match err {
        SnapshotError::VarNotFound {
            requested,
            available,
        } => {
            assert_eq!(requested, "absent_member");
            assert!(available.contains(&"alpha.bss".to_string()));
            assert!(available.contains(&"beta.bss".to_string()));
        }
        other => panic!("expected VarNotFound, got {other:?}"),
    }
}

// ---------- live_vars_via (N-name co-picker) ----------

/// Build a multi-instance report carrying TWO bss copies, each
/// with two named u64 vars (same + cross). Default values follow
/// the swap pattern: alpha (the "old" inactive scheduler) carries
/// small frozen counts, beta (the "new" active scheduler) carries
/// larger accumulating counts. Pickers using `max_by_sum_u64`
/// should pick beta.
fn two_instance_two_var_report() -> FailureDumpReport {
    let mk_bss = |obj: &str, same: u64, cross: u64| FailureDumpMap {
        name: format!("{obj}.bss"),
        map_kva: 0,
        map_type: 2,
        value_size: 16,
        max_entries: 1,
        value: Some(RenderedValue::Struct {
            type_name: Some(".bss".into()),
            members: vec![
                RenderedMember {
                    name: "same".into(),
                    value: RenderedValue::Uint {
                        bits: 64,
                        value: same,
                    },
                },
                RenderedMember {
                    name: "cross".into(),
                    value: RenderedValue::Uint {
                        bits: 64,
                        value: cross,
                    },
                },
            ],
        }),
        entries: Vec::new(),
        array_entries: Vec::new(),
        percpu_entries: Vec::new(),
        percpu_hash_entries: Vec::new(),
        arena: None,
        ringbuf: None,
        stack_trace: None,
        fd_array: None,
        error: None,
    };
    FailureDumpReport {
        schema: SCHEMA_SINGLE.to_string(),
        active_map_kvas: Vec::new(),
        maps: vec![mk_bss("alpha", 10, 5), mk_bss("beta", 100, 50)],
        ..Default::default()
    }
}

#[test]
fn live_vars_via_picker_selects_named_candidate() {
    let r = two_instance_two_var_report();
    let snap = Snapshot::new(&r);
    let fields = snap
        .live_vars_via(&["same", "cross"], |rows| {
            rows.iter().position(|(obj, _)| *obj == "beta.bss")
        })
        .expect("co-pick succeeds");
    assert_eq!(fields.len(), 2);
    assert_eq!(fields[0].as_u64().unwrap(), 100, "same from beta");
    assert_eq!(fields[1].as_u64().unwrap(), 50, "cross from beta");
}

#[test]
fn live_vars_via_picker_returns_none_surfaces_projection_failed() {
    let r = two_instance_two_var_report();
    let snap = Snapshot::new(&r);
    let err = snap
        .live_vars_via(&["same", "cross"], |_| None)
        .unwrap_err();
    match err {
        SnapshotError::ProjectionFailed { reason } => {
            assert!(
                reason.contains("live_vars_via picker for [same, cross] returned None"),
                "reason must name the picker AND the input name list, got {reason}",
            );
        }
        other => panic!("expected ProjectionFailed, got {other:?}"),
    }
}

#[test]
fn live_vars_via_picker_out_of_range_surfaces_projection_failed() {
    let r = two_instance_two_var_report();
    let snap = Snapshot::new(&r);
    let err = snap
        .live_vars_via(&["same", "cross"], |_| Some(99))
        .unwrap_err();
    match err {
        SnapshotError::ProjectionFailed { reason } => {
            assert!(
                reason.contains("index 99 out of range") && reason.contains("count = 2"),
                "reason must name the bad index AND the candidate count, got {reason}",
            );
        }
        other => panic!("expected ProjectionFailed, got {other:?}"),
    }
}

#[test]
fn live_vars_via_no_candidates_surfaces_var_not_found() {
    let r = two_instance_two_var_report();
    let snap = Snapshot::new(&r);
    let err = snap
        .live_vars_via(&["absent_a", "absent_b"], |_| {
            panic!("picker must NOT be called when no candidates")
        })
        .unwrap_err();
    match err {
        SnapshotError::VarNotFound {
            requested,
            available,
        } => {
            assert!(
                requested.contains("absent_a") && requested.contains("absent_b"),
                "requested must list both missing names, got {requested}",
            );
            assert!(available.contains(&"alpha.bss".to_string()));
            assert!(available.contains(&"beta.bss".to_string()));
        }
        other => panic!("expected VarNotFound, got {other:?}"),
    }
}

#[test]
fn live_vars_via_placeholder_snapshot_surfaces_placeholder_error() {
    let r = FailureDumpReport {
        is_placeholder: true,
        ..Default::default()
    };
    let snap = Snapshot::new(&r);
    let err = snap
        .live_vars_via(&["x"], |_| {
            panic!("picker must NOT be called on placeholder")
        })
        .unwrap_err();
    match err {
        SnapshotError::PlaceholderSnapshot { tag } => {
            assert!(tag.is_none());
        }
        other => panic!("expected PlaceholderSnapshot, got {other:?}"),
    }
}

#[test]
fn live_vars_via_empty_names_slice_errors() {
    let r = two_instance_two_var_report();
    let snap = Snapshot::new(&r);
    let err = snap
        .live_vars_via(&[], |_| panic!("picker must NOT be called for empty names"))
        .unwrap_err();
    match err {
        SnapshotError::ProjectionFailed { reason } => {
            assert!(
                reason.contains("empty names slice"),
                "empty-names diagnostic must name the cause, got {reason}",
            );
        }
        other => panic!("expected ProjectionFailed, got {other:?}"),
    }
}

#[test]
fn live_vars_via_partial_coverage_map_excluded() {
    // alpha has [same, cross]; beta has [same] only (no cross).
    // Co-pick on [same, cross] must drop beta entirely.
    let mut r = FailureDumpReport::default();
    let alpha = FailureDumpMap {
        name: "alpha.bss".into(),
        map_kva: 0,
        map_type: 2,
        value_size: 16,
        max_entries: 1,
        value: Some(RenderedValue::Struct {
            type_name: Some(".bss".into()),
            members: vec![
                RenderedMember {
                    name: "same".into(),
                    value: RenderedValue::Uint {
                        bits: 64,
                        value: 10,
                    },
                },
                RenderedMember {
                    name: "cross".into(),
                    value: RenderedValue::Uint { bits: 64, value: 5 },
                },
            ],
        }),
        entries: Vec::new(),
        array_entries: Vec::new(),
        percpu_entries: Vec::new(),
        percpu_hash_entries: Vec::new(),
        arena: None,
        ringbuf: None,
        stack_trace: None,
        fd_array: None,
        error: None,
    };
    let beta = FailureDumpMap {
        name: "beta.bss".into(),
        map_kva: 0,
        map_type: 2,
        value_size: 8,
        max_entries: 1,
        value: Some(RenderedValue::Struct {
            type_name: Some(".bss".into()),
            members: vec![RenderedMember {
                name: "same".into(),
                value: RenderedValue::Uint {
                    bits: 64,
                    value: 999,
                },
            }],
        }),
        entries: Vec::new(),
        array_entries: Vec::new(),
        percpu_entries: Vec::new(),
        percpu_hash_entries: Vec::new(),
        arena: None,
        ringbuf: None,
        stack_trace: None,
        fd_array: None,
        error: None,
    };
    r.maps = vec![alpha, beta];
    r.schema = SCHEMA_SINGLE.to_string();
    let snap = Snapshot::new(&r);
    let mut seen_objs: Vec<String> = Vec::new();
    let fields = snap
        .live_vars_via(&["same", "cross"], |rows| {
            for (obj, _) in rows {
                seen_objs.push((*obj).to_string());
            }
            Some(0)
        })
        .expect("co-pick succeeds");
    assert_eq!(seen_objs, vec!["alpha.bss".to_string()]);
    assert_eq!(fields[0].as_u64().unwrap(), 10);
    assert_eq!(fields[1].as_u64().unwrap(), 5);
}

#[test]
fn live_vars_via_with_max_by_sum_u64_picks_active_instance() {
    let r = two_instance_two_var_report();
    let snap = Snapshot::new(&r);
    let fields = snap
        .live_vars_via(
            &["same", "cross"],
            crate::scenario::snapshot::pickers::max_by_sum_u64,
        )
        .expect("co-pick succeeds");
    assert_eq!(
        fields[0].as_u64().unwrap(),
        100,
        "beta has the larger sum (100+50=150 vs alpha 10+5=15), so beta's same wins",
    );
    assert_eq!(fields[1].as_u64().unwrap(), 50, "beta's cross");
}

#[test]
fn live_var_via_placeholder_snapshot_surfaces_placeholder_error() {
    // Pin sibling-accessor parity: var() and map() both short-circuit
    // on is_placeholder before any lookup and return
    // SnapshotError::PlaceholderSnapshot. live_var_via must do the same
    // — otherwise a placeholder snapshot silently surfaces as
    // VarNotFound with an empty `available` list, hiding the real
    // freeze-rendezvous-failed signal from the caller's match arm.
    let r = FailureDumpReport {
        is_placeholder: true,
        ..Default::default()
    };
    let snap = Snapshot::new(&r);
    let f = snap.live_var_via("anything", |_| {
        panic!("picker must NOT be called on placeholder")
    });
    let err = f.error().expect("placeholder snapshot must carry an error");
    match err {
        SnapshotError::PlaceholderSnapshot { tag } => {
            assert!(
                tag.is_none(),
                "placeholder-from-view path leaves tag = None; per-capture tag \
                 belongs on placeholder errors that surface through the bridge \
                 drain, not snapshot view lookups",
            );
        }
        other => panic!("expected PlaceholderSnapshot, got {other:?}"),
    }
}
