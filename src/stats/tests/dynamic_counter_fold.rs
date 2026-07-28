//! Cross-run SUM-fold of Dynamic monotonic-counter ext-metric keys (the
//! level-suffixed `lb_*`/`alb_*` schedstat deltas + watched `ScalarCounter` bpf
//! fields). These keys are built at runtime and are NOT in the static `METRICS`
//! registry, so the cross-run aggregator classifies them via
//! `GauntletRow::ext_counter_keys` and SUM-folds them (matching the registered-
//! Counter convention) instead of averaging.

use super::*;

/// Dynamic monotonic-counter ext keys (lb_*/alb_* schedstat deltas, ScalarCounter
/// bpf fields) are NOT in the static METRICS registry, so the cross-run fold can
/// only classify them via `GauntletRow::ext_counter_keys`. A tagged key SUM-folds
/// (matching registered Counters); an untagged Dynamic key (a gauge / per-CPU
/// `_avg`) keeps the legacy mean-fold. The tags carry onto the aggregated row so
/// a second-level cross-run fold keeps SUM-folding them.
#[test]
fn group_and_average_dynamic_counter_ext_keys_sum_fold() {
    let mk = |lb: f64, bpf_ctr: f64, bpf_gauge: f64| {
        let mut r = make_row("t", "4cpu-1llc-nosmt", true, 0.0);
        // Two Dynamic counter keys (unregistered) — tagged -> SUM-fold.
        r.ext_metrics.insert("lb_count_mc".into(), lb);
        r.ext_metrics.insert("scx_x_allocs".into(), bpf_ctr);
        r.ext_counter_keys.insert("lb_count_mc".into());
        r.ext_counter_keys.insert("scx_x_allocs".into());
        // A Dynamic gauge key (unregistered, untagged) -> mean-fold.
        r.ext_metrics.insert("scx_x_lat_avg".into(), bpf_gauge);
        r
    };
    let out = group_and_average_by(
        &[mk(1000.0, 50.0, 4.0), mk(1000.0, 70.0, 8.0)],
        LEGACY_PAIRING_DIMS,
    );
    assert_eq!(out.len(), 1);
    let row = &out[0].row;
    // Tagged counters SUM-fold (1000+1000, 50+70) — NOT the mean (1000, 60).
    assert_eq!(row.ext_metrics.get("lb_count_mc").copied(), Some(2000.0));
    assert_eq!(row.ext_metrics.get("scx_x_allocs").copied(), Some(120.0));
    // The untagged gauge mean-folds (4,8 -> 6), NOT summed (12).
    assert_eq!(row.ext_metrics.get("scx_x_lat_avg").copied(), Some(6.0));
    // Counter-key tags carry onto the aggregated row (second-level fold); the
    // gauge stays untagged.
    assert!(row.ext_counter_keys.contains("lb_count_mc"));
    assert!(row.ext_counter_keys.contains("scx_x_allocs"));
    assert!(!row.ext_counter_keys.contains("scx_x_lat_avg"));
}

/// Single-run cohort (n=1): a tagged Dynamic counter folds to its value (SUM and
/// mean coincide at n=1 — confirm the counter branch emits the value, does not
/// skip or error) and the tag carries onto the aggregated row.
#[test]
fn group_and_average_dynamic_counter_single_run_emits_value() {
    let mut r = make_row("t", "4cpu-1llc-nosmt", true, 0.0);
    r.ext_metrics.insert("lb_count_mc".into(), 1000.0);
    r.ext_counter_keys.insert("lb_count_mc".into());
    let out = group_and_average_by(&[r], LEGACY_PAIRING_DIMS);
    assert_eq!(out.len(), 1);
    let row = &out[0].row;
    assert_eq!(row.ext_metrics.get("lb_count_mc").copied(), Some(1000.0));
    assert!(row.ext_counter_keys.contains("lb_count_mc"));
}

/// Partial-presence: a tagged counter present in only SOME rows of a multi-run
/// group SUM-folds the PRESENT values only — NOT Σ/n_group (which would dilute
/// by the absent run) and NOT the mean-of-present. An absent run's counter delta
/// is unmeasured, not zero — matching the registered-Counter convention.
#[test]
fn group_and_average_dynamic_counter_partial_presence_sums_present_only() {
    let with = |v: f64| {
        let mut r = make_row("t", "4cpu-1llc-nosmt", true, 0.0);
        r.ext_metrics.insert("scx_x_allocs".into(), v);
        r.ext_counter_keys.insert("scx_x_allocs".into());
        r
    };
    // Two rows carry the counter (50, 70); the third omits it entirely.
    let without = make_row("t", "4cpu-1llc-nosmt", true, 0.0);
    let out = group_and_average_by(&[with(50.0), with(70.0), without], LEGACY_PAIRING_DIMS);
    assert_eq!(out.len(), 1);
    let row = &out[0].row;
    // Σ over the 2 present rows = 120 — NOT Σ/3 = 40, NOT mean-of-present = 60.
    assert_eq!(row.ext_metrics.get("scx_x_allocs").copied(), Some(120.0));
    assert!(row.ext_counter_keys.contains("scx_x_allocs"));
}

/// `GauntletRow::ext_counter_keys` serde: a populated set roundtrips; an empty
/// set is omitted (`skip_serializing_if`) and a row lacking the field
/// deserializes to an empty set (`#[serde(default)]`) — the pre-existing behavior
/// (those keys mean-fold), so a stale sidecar degrades, not hard-fails.
#[test]
fn gauntlet_row_ext_counter_keys_serde() {
    let mut r = make_row("t", "4cpu-1llc-nosmt", true, 0.0);
    r.ext_counter_keys.insert("lb_count_mc".into());
    let json = serde_json::to_string(&r).unwrap();
    assert!(
        json.contains("ext_counter_keys"),
        "populated set is serialized"
    );
    let back: GauntletRow = serde_json::from_str(&json).unwrap();
    assert!(back.ext_counter_keys.contains("lb_count_mc"));
    // Empty set: omitted on serialize, defaults to empty on deserialize.
    let empty = make_row("t", "4cpu-1llc-nosmt", true, 0.0);
    let json_empty = serde_json::to_string(&empty).unwrap();
    assert!(
        !json_empty.contains("ext_counter_keys"),
        "empty set is omitted (skip_serializing_if)"
    );
    let back_empty: GauntletRow = serde_json::from_str(&json_empty).unwrap();
    assert!(
        back_empty.ext_counter_keys.is_empty(),
        "absent field -> empty set"
    );
}
