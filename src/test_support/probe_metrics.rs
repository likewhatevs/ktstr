//! Flat-metric lookup helpers for the jemalloc-probe integration
//! tests.
//!
//! Lives in the crate (not inline in `tests/jemalloc_probe_tests.rs`)
//! so the logic is reachable from `#[cfg(test)]` unit tests. The
//! probe test binary registers `#[ktstr_test]` entries, which
//! activates the early-dispatch ctor's `--list` intercept — any
//! plain `#[test]` fn declared in that binary is hidden from
//! nextest's discovery (see the long comment at the head of
//! `tests/jemalloc_alloc_worker_exit_codes.rs`). Moving the
//! helpers here lets the ExceedsCap branch and friends be
//! pinned by lib-crate unit tests that run under `cargo nextest
//! run --lib` without the ctor path.
//!
//! All items are `pub` so the integration test file at
//! `tests/jemalloc_probe_tests.rs` can import them through the
//! `test_support` re-export surface.

use super::payload::{Metric, PayloadMetrics};

/// Outcome of scanning the flat metric list for a tid-keyed thread
/// entry. Distinguishes "tid not present" from "tid present but
/// `allocated_bytes` missing" AND from "probe emitted more than
/// [`MAX_SCAN_INDEX`] contiguous threads without the caller's
/// tid appearing in the prefix" — so a caller can issue a precise
/// diagnostic instead of a blanket "not found".
pub enum ThreadLookup {
    /// `snapshots.{snap_idx}.threads.N.Ok.tid == worker_tid` and
    /// `snapshots.{snap_idx}.threads.N.Ok.allocated_bytes` are both
    /// present. Returns the observed counter plus the companion
    /// `deallocated_bytes` (if emitted).
    Found {
        allocated_bytes: u64,
        deallocated_bytes: Option<u64>,
    },
    /// Probe emitted a `snapshots.{snap_idx}.threads.N.{Ok,Err}.tid`
    /// matching `worker_tid`, but no
    /// `snapshots.{snap_idx}.threads.N.Ok.allocated_bytes` sibling.
    /// Either the tid was on the `Err` arm (`.Err.tid`), where the
    /// probe hit an error on that thread and `error`/`error_kind`
    /// replace the counter fields, or it was on the `Ok` arm but the
    /// `allocated_bytes` sibling was absent.
    MissingAllocatedBytes,
    /// No `snapshots.{snap_idx}.threads.N.{Ok,Err}.tid == worker_tid`
    /// entry in the flat metric list. Probe did not visit the worker
    /// at all.
    TidAbsent,
    /// The flat metric list contained at least [`MAX_SCAN_INDEX`]
    /// contiguous `snapshots.{snap_idx}.threads.N.{Ok,Err}.tid`
    /// entries, none of which matched `worker_tid`, and the scan hit
    /// the cap
    /// before reaching the array terminator. The worker's tid may
    /// exist at a later index and be invisible to the scan.
    /// Distinct from `TidAbsent` — this outcome means the lookup is
    /// inconclusive, not that the probe definitively skipped the
    /// worker.
    ExceedsCap,
}

/// Safety bound on the `snapshots.*.threads.N.tid` scan in
/// [`lookup_thread`], [`snapshot_worker_allocated`], [`thread_count`],
/// and [`snapshot_count`]. Realistic probe runs see at most a few
/// dozen threads in a single-allocator worker process; hitting this
/// cap indicates either an unexpectedly wide target or a flat-metric
/// schema change that broke the terminator convention.
pub const MAX_SCAN_INDEX: usize = 1024;

/// Find a metric by exact name. Returns `None` if absent.
pub fn find_metric<'a>(metrics: &'a PayloadMetrics, key: &str) -> Option<&'a Metric> {
    metrics.metrics.iter().find(|m| m.name == key)
}

/// Does the flat metric list contain a metric with this exact name?
/// Thin wrapper around [`find_metric`] for the common existence
/// check — avoids forcing every call site to spell `.is_some()`.
pub fn has_metric(metrics: &PayloadMetrics, key: &str) -> bool {
    find_metric(metrics, key).is_some()
}

/// Fetch a metric by exact name and return its numeric value as a
/// `u64`. Returns `None` if the metric is absent. Thin wrapper
/// around [`find_metric`] + `value as u64` for the common
/// numeric-lookup shape.
///
/// # `f64` → `u64` precision
///
/// JSON numbers parse into the probe's flat-metric list as `f64`
/// (serde_json's number type). Integer values round-trip through
/// `f64` without precision loss only up to `2^53`
/// (`9_007_199_254_740_992`); above that bound, adjacent `u64`
/// values collapse onto the same `f64` and `value as u64` loses
/// the low-order bits. The probe's emitted counters
/// (`allocated_bytes`, `deallocated_bytes`, tid numbers, snapshot
/// timestamps in seconds) are in practice far below this
/// threshold on realistic workloads: a 64-bit byte counter would
/// require >8 PiB of total-allocated memory, and Linux pids are
/// capped at `2^22`. The bound is therefore a soft invariant —
/// consumers should NOT feed arbitrary externally-controlled
/// values through this helper without a prior range check.
///
/// A `debug_assert!` on the same bound catches the invariant
/// locally so a future metric that genuinely exceeds `2^53` lights
/// up in a debug build before the truncation silently corrupts a
/// downstream comparison; release builds trust the soft invariant
/// and perform the `as u64` cast unconditionally.
pub fn find_metric_u64(metrics: &PayloadMetrics, key: &str) -> Option<u64> {
    find_metric(metrics, key).map(|m| {
        debug_assert!(
            m.value.is_finite() && m.value >= 0.0 && m.value <= (1u64 << 53) as f64,
            "metric {:?} value {} outside the f64→u64 lossless range \
             [0, 2^53]; the `as u64` cast will truncate silently. \
             Either range-check externally-sourced input before \
             landing it in the flat metrics list, or consume the \
             metric via `.value` (f64) instead of this u64 helper.",
            m.name,
            m.value,
        );
        m.value as u64
    })
}

/// Walk `0..cap` applying `key_fn(i)` to form a metric name and
/// count how many consecutive indices yield a present metric.
/// Stops at the first miss — the probe's `walk_json_leaves`
/// flattening yields indices 0..N contiguously, so the first gap is
/// the array terminator. Returns the count, which may be `cap` if
/// every index below the bound is present (inconclusive — the
/// caller should treat `cap` as "saturated scan, real count may be
/// larger").
pub fn count_indexed_metrics<F>(metrics: &PayloadMetrics, cap: usize, key_fn: F) -> usize
where
    F: Fn(usize) -> String,
{
    let mut n = 0;
    for i in 0..cap {
        if find_metric(metrics, &key_fn(i)).is_some() {
            n += 1;
        } else {
            break;
        }
    }
    n
}

/// Extract the `allocated_bytes` / `deallocated_bytes` values for
/// `worker_tid` from snapshot 0 in the flat metric list produced by
/// `walk_json_leaves` over the probe's JSON output.
///
/// `ThreadResult` is externally-tagged so the probe emits
/// `{"pid":P,"snapshots":[{"timestamp_unix_sec":T,"threads":[{"Ok":{"tid":T,"allocated_bytes":A,"deallocated_bytes":D,...}}, {"Err":{"tid":T,"error":...,"error_kind":...}}, ...]}, ...]}`
/// which `walk_json_leaves` flattens per array index into contiguous
/// keys `snapshots.0.threads.0.Ok.tid`, `snapshots.0.threads.1.Err.tid`,
/// … with no gaps. Each index carries exactly one variant wrapper.
/// The scan stops at the first index where neither `.Ok.tid` nor
/// `.Err.tid` exists (the natural array terminator) and returns
/// [`ThreadLookup::TidAbsent`]. If the cap is reached without hitting
/// the terminator AND without matching `worker_tid`, returns
/// [`ThreadLookup::ExceedsCap`]. If the matching tid is on the `Err`
/// arm (no `allocated_bytes` sibling), returns
/// [`ThreadLookup::MissingAllocatedBytes`].
pub fn lookup_thread(metrics: &PayloadMetrics, worker_tid: i32) -> ThreadLookup {
    let worker_tid_f64 = worker_tid as f64;
    for i in 0..MAX_SCAN_INDEX {
        let ok_tid_key = format!("snapshots.0.threads.{i}.Ok.tid");
        let err_tid_key = format!("snapshots.0.threads.{i}.Err.tid");
        let (tid_m, is_ok) = match find_metric(metrics, &ok_tid_key) {
            Some(m) => (m, true),
            None => match find_metric(metrics, &err_tid_key) {
                Some(m) => (m, false),
                None => return ThreadLookup::TidAbsent,
            },
        };
        if tid_m.value == worker_tid_f64 {
            if !is_ok {
                return ThreadLookup::MissingAllocatedBytes;
            }
            let alloc_key = format!("snapshots.0.threads.{i}.Ok.allocated_bytes");
            let dealloc_key = format!("snapshots.0.threads.{i}.Ok.deallocated_bytes");
            let allocated_bytes = match find_metric(metrics, &alloc_key).map(|m| m.value as u64) {
                Some(v) => v,
                None => return ThreadLookup::MissingAllocatedBytes,
            };
            let deallocated_bytes = find_metric(metrics, &dealloc_key).map(|m| m.value as u64);
            return ThreadLookup::Found {
                allocated_bytes,
                deallocated_bytes,
            };
        }
    }
    // Loop ran to completion — every index 0..MAX_SCAN_INDEX had a
    // tid entry (Ok or Err), and none matched. A contiguous-array
    // terminator would have early-returned `TidAbsent`, so the cap
    // was hit with data remaining. Surface the inconclusive outcome
    // distinctly from genuine absence.
    ThreadLookup::ExceedsCap
}

/// Extract `snapshots.{snap_idx}.threads[*].allocated_bytes` for the
/// thread whose tid matches `worker_tid`. Returns [`ThreadLookup`]
/// so callers distinguish "tid absent" from "cap hit before tid
/// seen" from "allocated_bytes sibling missing" — parallel to
/// [`lookup_thread`], which covers the single-snapshot path.
pub fn snapshot_worker_allocated(
    metrics: &PayloadMetrics,
    snap_idx: usize,
    worker_tid: i32,
) -> ThreadLookup {
    let worker_tid_f64 = worker_tid as f64;
    for j in 0..MAX_SCAN_INDEX {
        let ok_tid_key = format!("snapshots.{snap_idx}.threads.{j}.Ok.tid");
        let err_tid_key = format!("snapshots.{snap_idx}.threads.{j}.Err.tid");
        let (tid_m, is_ok) = match find_metric(metrics, &ok_tid_key) {
            Some(m) => (m, true),
            None => match find_metric(metrics, &err_tid_key) {
                Some(m) => (m, false),
                None => return ThreadLookup::TidAbsent,
            },
        };
        if tid_m.value == worker_tid_f64 {
            if !is_ok {
                return ThreadLookup::MissingAllocatedBytes;
            }
            let alloc_key = format!("snapshots.{snap_idx}.threads.{j}.Ok.allocated_bytes");
            let dealloc_key = format!("snapshots.{snap_idx}.threads.{j}.Ok.deallocated_bytes");
            let allocated_bytes = match find_metric(metrics, &alloc_key).map(|m| m.value as u64) {
                Some(v) => v,
                None => return ThreadLookup::MissingAllocatedBytes,
            };
            let deallocated_bytes = find_metric(metrics, &dealloc_key).map(|m| m.value as u64);
            return ThreadLookup::Found {
                allocated_bytes,
                deallocated_bytes,
            };
        }
    }
    ThreadLookup::ExceedsCap
}

/// Count the number of `snapshots.0.threads.N.{Ok,Err}.tid` entries
/// in the flat metric list, capped at [`MAX_SCAN_INDEX`]. Each index
/// carries exactly one variant wrapper (`Ok` or `Err`); the count
/// terminates at the first index where neither exists.
pub fn thread_count(metrics: &PayloadMetrics) -> usize {
    let mut n = 0;
    for i in 0..MAX_SCAN_INDEX {
        let ok_key = format!("snapshots.0.threads.{i}.Ok.tid");
        let err_key = format!("snapshots.0.threads.{i}.Err.tid");
        if find_metric(metrics, &ok_key).is_some() || find_metric(metrics, &err_key).is_some() {
            n += 1;
        } else {
            break;
        }
    }
    n
}

/// Count the number of `snapshots.N.timestamp_unix_sec` entries in
/// the flat metric list, capped at [`MAX_SCAN_INDEX`].
pub fn snapshot_count(metrics: &PayloadMetrics) -> usize {
    count_indexed_metrics(metrics, MAX_SCAN_INDEX, |i| {
        format!("snapshots.{i}.timestamp_unix_sec")
    })
}

/// Flatten the full `(name, value)` pair list for diagnostic
/// rendering inside error messages. Returned as an owned
/// `Vec<(&str, f64)>` so call sites spell the diagnostic as a single
/// `{:?}` formatter argument instead of re-typing the
/// `.iter().map(...).collect()` chain at every site.
///
/// Intended for "probe returned nothing we expected" error paths —
/// when a lookup helper ([`lookup_thread`], [`snapshot_worker_allocated`],
/// [`find_metric_u64`]) returns a miss, dumping the observed flat metric
/// list into the failure message is usually the fastest triage step.
pub fn flat_metrics_dump(metrics: &PayloadMetrics) -> Vec<(&str, f64)> {
    metrics
        .metrics
        .iter()
        .map(|m| (m.name.as_str(), m.value))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn metric(name: &str, value: f64) -> Metric {
        use super::super::payload::{MetricSource, MetricStream, Polarity};
        Metric {
            name: name.to_owned(),
            value,
            polarity: Polarity::Unknown,
            unit: String::new(),
            source: MetricSource::Json,
            stream: MetricStream::Stdout,
        }
    }

    fn empty_payload() -> PayloadMetrics {
        PayloadMetrics {
            payload_index: 0,
            metrics: Vec::new(),
            exit_code: 0,
        }
    }

    fn push_ok_tid(metrics: &mut PayloadMetrics, idx: usize, tid: f64) {
        metrics
            .metrics
            .push(metric(&format!("snapshots.0.threads.{idx}.Ok.tid"), tid));
    }

    fn push_err_tid(metrics: &mut PayloadMetrics, idx: usize, tid: f64) {
        metrics
            .metrics
            .push(metric(&format!("snapshots.0.threads.{idx}.Err.tid"), tid));
    }

    fn push_alloc(metrics: &mut PayloadMetrics, idx: usize, alloc: f64) {
        metrics.metrics.push(metric(
            &format!("snapshots.0.threads.{idx}.Ok.allocated_bytes"),
            alloc,
        ));
    }

    /// Empty flat-metric list → no tid entries at all → terminator
    /// at index 0 → `TidAbsent`.
    #[test]
    fn lookup_thread_empty_metrics_returns_tid_absent() {
        let m = empty_payload();
        assert!(matches!(lookup_thread(&m, 42), ThreadLookup::TidAbsent));
    }

    /// Matching tid with an `allocated_bytes` sibling → `Found`
    /// carrying the observed counter.
    #[test]
    fn lookup_thread_matching_tid_returns_found() {
        let mut m = empty_payload();
        push_ok_tid(&mut m, 0, 42.0);
        push_alloc(&mut m, 0, 1_048_576.0);
        match lookup_thread(&m, 42) {
            ThreadLookup::Found {
                allocated_bytes,
                deallocated_bytes,
            } => {
                assert_eq!(allocated_bytes, 1_048_576);
                assert_eq!(deallocated_bytes, None);
            }
            _ => panic!("expected ThreadLookup::Found"),
        }
    }

    /// Matching tid but no `allocated_bytes` sibling → the probe hit
    /// an error on that thread → `MissingAllocatedBytes`.
    #[test]
    fn lookup_thread_missing_allocated_bytes_returns_missing_variant() {
        let mut m = empty_payload();
        push_ok_tid(&mut m, 0, 42.0);
        // no matching `.allocated_bytes`
        assert!(matches!(
            lookup_thread(&m, 42),
            ThreadLookup::MissingAllocatedBytes
        ));
    }

    /// Matching tid on the Err arm — `ThreadResult::Err` carries the
    /// tid but no `allocated_bytes` sibling by design. Direct
    /// detection via the `.Err.tid` path discriminator returns
    /// `MissingAllocatedBytes` so callers route the Err arm through
    /// the same diagnostic path as a malformed Ok entry.
    #[test]
    fn lookup_thread_err_arm_returns_missing_allocated_bytes() {
        let mut m = empty_payload();
        push_err_tid(&mut m, 0, 42.0);
        assert!(matches!(
            lookup_thread(&m, 42),
            ThreadLookup::MissingAllocatedBytes
        ));
    }

    /// A contiguous run of tids that does NOT include the caller's
    /// tid, but terminates BEFORE the cap → natural-terminator path
    /// → `TidAbsent` (not `ExceedsCap`).
    #[test]
    fn lookup_thread_contiguous_prefix_without_match_returns_tid_absent() {
        let mut m = empty_payload();
        for i in 0..10 {
            push_ok_tid(&mut m, i, (1000 + i) as f64);
        }
        assert!(matches!(lookup_thread(&m, 42), ThreadLookup::TidAbsent));
    }

    /// The full-cap case: fill indices `0..MAX_SCAN_INDEX`
    /// with non-matching tids, then call lookup_thread with a tid
    /// that isn't in the list. The scan runs all 1024 iterations,
    /// never hits a terminator, never matches, and therefore must
    /// return `ExceedsCap` — distinct from `TidAbsent`.
    #[test]
    fn lookup_thread_saturated_scan_without_match_returns_exceeds_cap() {
        let mut m = empty_payload();
        for i in 0..MAX_SCAN_INDEX {
            // tids chosen so none is equal to the probe tid below.
            push_ok_tid(&mut m, i, (1_000_000 + i) as f64);
        }
        let target_tid: i32 = 42;
        let outcome = lookup_thread(&m, target_tid);
        assert!(
            matches!(outcome, ThreadLookup::ExceedsCap),
            "saturated scan without match must return ExceedsCap; got other variant"
        );
    }

    /// Same invariant for `snapshot_worker_allocated` (the
    /// multi-snapshot path): fill 1024 tid entries for snapshot
    /// index 0, call with a non-matching tid, assert `ExceedsCap`.
    #[test]
    fn snapshot_worker_allocated_saturated_scan_returns_exceeds_cap() {
        let mut m = empty_payload();
        for i in 0..MAX_SCAN_INDEX {
            push_ok_tid(&mut m, i, (1_000_000 + i) as f64);
        }
        let outcome = snapshot_worker_allocated(&m, 0, 42);
        assert!(
            matches!(outcome, ThreadLookup::ExceedsCap),
            "saturated multi-snapshot scan without match must return ExceedsCap"
        );
    }

    /// `snapshot_worker_allocated` with an empty metric list must
    /// return `TidAbsent` — parallel to the single-snapshot path.
    #[test]
    fn snapshot_worker_allocated_empty_returns_tid_absent() {
        let m = empty_payload();
        assert!(matches!(
            snapshot_worker_allocated(&m, 0, 42),
            ThreadLookup::TidAbsent
        ));
    }
}
