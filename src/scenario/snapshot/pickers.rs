//! Predefined disambiguator closures for
//! [`Snapshot::live_var_via`](super::view::Snapshot::live_var_via).
//!
//! Pickers are plain functions matching the closure shape
//! `live_var_via`'s `picker` arg expects (`FnOnce(&[(&str,
//! SnapshotField)]) -> Option<usize>`). Pass by name:
//!
//! ```ignore
//! snap.live_var_via("nr_dispatched", pickers::max_by_counter_value)
//! ```
//!
//! Every picker baked into this module is a HEURISTIC for some
//! "live instance vs prior instance" question; the picker NAME
//! tells you what operation it performs (not what semantic the
//! caller can conclude). Choose deliberately — the caller is
//! responsible for picking a picker whose operation matches the
//! metric's shape (counter vs gauge vs sentinel) and the test's
//! timing relative to scheduler swaps.

use super::SnapshotField;

/// Pick the candidate whose `as_u64()` value is largest. The
/// operation is purely mechanical: project each candidate via
/// `SnapshotField::as_u64`, drop those that fail to project
/// (wrong type, missing field, etc.), and return the index of
/// the surviving candidate with the largest u64.
///
/// # Active-bss heuristic
///
/// For a cumulative non-decreasing counter where the inactive
/// instance's bss stays at BSS-zero init (e.g. the just-killed
/// scheduler's `bpf_bpf.bss` copy after `Op::ReplaceScheduler`),
/// the live instance has by definition accumulated more events,
/// so max-by-value reliably picks the live bss copy. This is the
/// load-bearing usage pattern multi-instance A/B tests reach for.
///
/// # When this picks WRONG
///
/// Treat the picker as "max by u64," not as "the live instance":
///
/// - **Gauges or stateful fields.** An inactive copy can carry a
///   sticky non-zero value from before detach (e.g. a high-water
///   mark, a last-seen-timestamp). Max-by-value would still pick
///   it after the live instance starts from zero.
/// - **Sentinel-init counters** (e.g. `u64::MAX` on init meaning
///   "unset"). Max is meaningless here — the uninitialized copy
///   wins every comparison.
/// - **Immediately-post-swap reads** before the new instance has
///   any events. The just-killed copy still has the larger
///   cumulative count for some window after the swap. For
///   `Op::ReplaceScheduler` races the new
///   `wait_for_accessor_publish_advance` gate already serializes
///   most of that window away, but a snapshot fired before the
///   first post-swap counter increment will still observe the
///   old instance's cumulative value as the maximum.
///
/// For any of these, write a picker that names the live instance
/// explicitly — by binary fingerprint, by obj-name prefix
/// equality, or by a separately captured liveness signal — and
/// pass it to `live_var_via` instead.
///
/// # Return semantic
///
/// Returns the index of the max candidate, or `None` if every
/// candidate failed to project to `u64` (in which case
/// `live_var_via` surfaces a
/// [`SnapshotError::ProjectionFailed`](super::SnapshotError::ProjectionFailed)
/// naming the picker as the source). Ties resolve to the LAST
/// tied index (deterministic; matches `Iterator::max_by_key`'s
/// "last element of equal-maximum group wins" rule). Test
/// authors who need a different tie-break write a sibling picker
/// rather than relying on a specific ordering rule that's not
/// load-bearing for the active-bss heuristic this picker targets.
pub fn max_by_counter_value(fields: &[(&str, SnapshotField<'_>)]) -> Option<usize> {
    fields
        .iter()
        .enumerate()
        .filter_map(|(i, (_, f))| Some((i, f.as_u64().ok()?)))
        .max_by_key(|(_, v)| *v)
        .map(|(i, _)| i)
}

/// Pick the candidate row whose fields sum to the largest u64 — the
/// "max-activity bss" heuristic generalized to N co-picked variables.
/// For each row, the picker `as_u64`-projects every field and
/// `saturating_add`s them; rows containing ANY field that fails to
/// project to u64 are dropped entirely (not partial-summed); among
/// the surviving rows, the one with the largest sum wins.
///
/// Pairs with [`super::view::Snapshot::live_vars_via`] for ratio /
/// fraction metrics: the row whose counters have accumulated the
/// most TOTAL events is the live scheduler instance, and selecting
/// that row guarantees all N returned fields come from the same
/// source map (no cross-bss-copy corruption).
///
/// # When this picks WRONG
///
/// Inherits every failure mode of [`max_by_counter_value`] (gauges,
/// sentinel-init counters like `u64::MAX`, immediately-post-swap
/// windows before the new instance has accumulated past the old's
/// BSS-zero point), plus one sum-specific mode:
///
/// **Any-non-u64-field row excluded.** A single rodata or
/// non-counter variable accidentally included in the name set
/// silently makes every candidate row ineligible (every row would
/// have that non-u64 field). Verify all N names project to u64
/// counters BEFORE composing the picker.
///
/// Sums use `saturating_add`: a row containing a sentinel
/// `u64::MAX` saturates to `u64::MAX` and wins all comparisons.
/// Treat that as a sentinel-mode failure (per max_by_counter_value's
/// caveat).
///
/// Ties resolve to the LAST tied row (matches
/// [`Iterator::max_by_key`] semantics — same as
/// [`max_by_counter_value`]).
pub fn max_by_sum_u64(rows: &[(&str, Vec<SnapshotField<'_>>)]) -> Option<usize> {
    rows.iter()
        .enumerate()
        .filter_map(|(i, (_, fields))| {
            let mut sum: u64 = 0;
            for f in fields {
                let v = f.as_u64().ok()?;
                sum = sum.saturating_add(v);
            }
            Some((i, sum))
        })
        .max_by_key(|(_, s)| *s)
        .map(|(i, _)| i)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::monitor::btf_render::RenderedValue;

    fn u64_field(value: u64) -> RenderedValue {
        RenderedValue::Uint { bits: 64, value }
    }

    fn struct_field() -> RenderedValue {
        RenderedValue::Struct {
            type_name: Some("opaque".into()),
            members: vec![],
        }
    }

    #[test]
    fn max_by_counter_value_picks_largest_u64() {
        let a = u64_field(10);
        let b = u64_field(100);
        let c = u64_field(5);
        let fields = vec![
            ("alpha.bss", SnapshotField::Value(&a)),
            ("beta.bss", SnapshotField::Value(&b)),
            ("gamma.bss", SnapshotField::Value(&c)),
        ];
        assert_eq!(max_by_counter_value(&fields), Some(1));
    }

    #[test]
    fn max_by_counter_value_skips_non_u64_picks_max_from_remainder() {
        let a = u64_field(10);
        let s = struct_field();
        let c = u64_field(100);
        let fields = vec![
            ("alpha.bss", SnapshotField::Value(&a)),
            ("beta.bss", SnapshotField::Value(&s)),
            ("gamma.bss", SnapshotField::Value(&c)),
        ];
        assert_eq!(
            max_by_counter_value(&fields),
            Some(2),
            "Struct candidate drops out via as_u64 error; max of \
             {{alpha=10, gamma=100}} is gamma at index 2",
        );
    }

    #[test]
    fn max_by_counter_value_returns_none_when_all_candidates_non_u64() {
        let s1 = struct_field();
        let s2 = struct_field();
        let fields = vec![
            ("alpha.bss", SnapshotField::Value(&s1)),
            ("beta.bss", SnapshotField::Value(&s2)),
        ];
        assert_eq!(
            max_by_counter_value(&fields),
            None,
            "every candidate fails as_u64 — picker returns None so \
             live_var_via surfaces ProjectionFailed naming the picker",
        );
    }

    #[test]
    fn max_by_counter_value_tie_picks_last() {
        let a = u64_field(50);
        let b = u64_field(50);
        let c = u64_field(50);
        let fields = vec![
            ("alpha.bss", SnapshotField::Value(&a)),
            ("beta.bss", SnapshotField::Value(&b)),
            ("gamma.bss", SnapshotField::Value(&c)),
        ];
        assert_eq!(
            max_by_counter_value(&fields),
            Some(2),
            "Iterator::max_by_key returns the LAST element of an \
             equal-maximum group; picker tie-break is deterministic \
             across runs but lands on the last tied index, not the \
             first",
        );
    }

    #[test]
    fn max_by_counter_value_empty_input_returns_none() {
        let fields: Vec<(&str, SnapshotField<'_>)> = vec![];
        assert_eq!(max_by_counter_value(&fields), None);
    }

    // ---------- max_by_sum_u64 ----------

    #[test]
    fn max_by_sum_u64_picks_largest_summed_row() {
        let a1 = u64_field(10);
        let a2 = u64_field(20);
        let b1 = u64_field(5);
        let b2 = u64_field(5);
        let c1 = u64_field(30);
        let c2 = u64_field(40);
        let rows = vec![
            (
                "alpha.bss",
                vec![SnapshotField::Value(&a1), SnapshotField::Value(&a2)],
            ),
            (
                "beta.bss",
                vec![SnapshotField::Value(&b1), SnapshotField::Value(&b2)],
            ),
            (
                "gamma.bss",
                vec![SnapshotField::Value(&c1), SnapshotField::Value(&c2)],
            ),
        ];
        assert_eq!(
            max_by_sum_u64(&rows),
            Some(2),
            "row 2 sum = 70, beats row 0 (30) and row 1 (10)",
        );
    }

    #[test]
    fn max_by_sum_u64_tie_picks_last() {
        let v = u64_field(10);
        let rows = vec![
            (
                "a",
                vec![SnapshotField::Value(&v), SnapshotField::Value(&v)],
            ),
            (
                "b",
                vec![SnapshotField::Value(&v), SnapshotField::Value(&v)],
            ),
            (
                "c",
                vec![SnapshotField::Value(&v), SnapshotField::Value(&v)],
            ),
        ];
        assert_eq!(
            max_by_sum_u64(&rows),
            Some(2),
            "Iterator::max_by_key returns the LAST tied element; \
             matches max_by_counter_value semantic for caller \
             consistency",
        );
    }

    #[test]
    fn max_by_sum_u64_drops_rows_with_any_non_u64_field() {
        let u = u64_field(10);
        let s = struct_field();
        let big = u64_field(1000);
        let rows = vec![
            // row 0: u64 + Struct → drop entirely (any non-u64 → ineligible)
            (
                "alpha",
                vec![SnapshotField::Value(&u), SnapshotField::Value(&s)],
            ),
            // row 1: u64 + u64 → eligible, sum = 1010
            (
                "beta",
                vec![SnapshotField::Value(&u), SnapshotField::Value(&big)],
            ),
            // row 2: Struct + u64 → drop entirely
            (
                "gamma",
                vec![SnapshotField::Value(&s), SnapshotField::Value(&big)],
            ),
        ];
        assert_eq!(
            max_by_sum_u64(&rows),
            Some(1),
            "only row 1 survives the any-non-u64-eligibility filter; \
             picker is NOT partial-summing (Struct + u64 doesn't fall \
             back to summing the u64 alone)",
        );
    }

    #[test]
    fn max_by_sum_u64_all_rows_have_non_u64_field_returns_none() {
        let u = u64_field(10);
        let s = struct_field();
        let rows = vec![
            (
                "alpha",
                vec![SnapshotField::Value(&u), SnapshotField::Value(&s)],
            ),
            (
                "beta",
                vec![SnapshotField::Value(&s), SnapshotField::Value(&u)],
            ),
        ];
        assert_eq!(
            max_by_sum_u64(&rows),
            None,
            "every row has at least one non-u64 field; downstream \
             live_vars_via surfaces ProjectionFailed",
        );
    }

    #[test]
    fn max_by_sum_u64_empty_rows_returns_none() {
        let rows: Vec<(&str, Vec<SnapshotField<'_>>)> = vec![];
        assert_eq!(max_by_sum_u64(&rows), None);
    }

    #[test]
    fn max_by_sum_u64_saturates_on_overflow() {
        let big = u64_field(u64::MAX);
        let one = u64_field(1);
        let rows = vec![
            (
                "alpha",
                vec![SnapshotField::Value(&big), SnapshotField::Value(&one)],
            ),
            (
                "beta",
                vec![SnapshotField::Value(&one), SnapshotField::Value(&one)],
            ),
        ];
        // alpha saturates to u64::MAX; beta sums to 2. Alpha wins.
        assert_eq!(max_by_sum_u64(&rows), Some(0));
    }
}
