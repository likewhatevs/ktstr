//! `#[cfg(feature = "llm")]` host-side LlmExtract: pairs raw payload
//! outputs with their PayloadMetrics by index, runs the on-host LLM
//! extraction, and validates the extracted metrics (structural sanity +
//! bound checks). Split out of eval/mod.rs to keep the module under the
//! size ceiling; gated as a whole, so the inner per-item cfg attributes
//! are redundant but harmless.

use super::*;

/// Run [`crate::test_support::model::extract_via_llm`] against every
/// `OutputFormat::LlmExtract` raw output drained from SHM, replace
/// the paired empty-metrics `PayloadMetrics` slot with the extracted
/// result, and return any failure details that should fold into the
/// test's AssertResult.
///
/// Pairing is by explicit
/// [`crate::test_support::PayloadMetrics::payload_index`] equality:
/// every guest-side payload-pipeline emission allocates one index
/// from the per-process counter (see
/// [`crate::scenario::payload_run`]) and stamps it onto BOTH the
/// `MSG_TYPE_RAW_PAYLOAD_OUTPUT` and the
/// `MSG_TYPE_PAYLOAD_METRICS` message it emits. The host walks
/// `raw_outputs`, looks up each entry's index in a
/// `HashMap<payload_index, vec position>` built once over
/// `payload_metrics`, and writes the extracted metrics into the
/// matched slot. Non-LlmExtract payloads (Json, ExitCode) also
/// emit `MSG_TYPE_PAYLOAD_METRICS` with their own per-invocation
/// index, but the host's pairing loop walks the `raw_outputs`
/// slice; non-LlmExtract entries are never inspected because they
/// have no companion raw output.
///
/// Index-based pairing replaces the prior emission-order pairing
/// which conflated a `Json` payload that legitimately produced zero
/// metrics (no numeric leaves) with an `LlmExtract` placeholder.
///
/// Failure shape:
/// - Model load fails (e.g. `KTSTR_MODEL_OFFLINE=1` with cold cache,
///   SHA mismatch on a corrupted cached GGUF): append a single
///   `LlmExtract model load failed: <reason>` detail. metrics
///   remain empty. No structural-sanity checks fire — we have
///   nothing to check against.
/// - Structural-sanity violation (duplicate metric name, non-finite
///   value, source tag drift): every violation found contributes
///   its own detail (see [`validate_llm_extraction`]). The metric
///   set is still populated on the PayloadMetrics slot so debugging
///   tools and the sidecar see what the model produced.
/// - Raw output's `payload_index` has no matching `PayloadMetrics`
///   entry (guest emitted a raw output without its companion empty-
///   metrics PM, or emission was lost to SHM ring overflow):
///   append a `LlmExtract host pairing` detail naming the orphan
///   index and skip the extraction for that raw output. The other
///   raw outputs still get extracted — dropping every extraction
///   because one orphan exists would lose information the test
///   author can still act on.
/// - Per-payload bounds violation (when the payload declared
///   `metric_bounds`, see [`crate::test_support::MetricBounds`]):
///   each violation surfaces as its own detail via
///   [`validate_metric_bounds`] — minimum metric count below the
///   declared floor, value below `value_min`, value above
///   `value_max`. The bounds pass runs AFTER the structural-sanity
///   pass and ONLY when extraction succeeded; load-failed pairs
///   skip the bounds check (the empty placeholder would otherwise
///   spuriously trip a `min_count` violation on every offline-gated
///   test).
/// - Orphan `PayloadMetrics` (a guest-side LlmExtract emission
///   produced an empty-metrics `PayloadMetrics` whose
///   `payload_index` has NO matching `RawPayloadOutput` companion):
///   the post-pairing scan flags the missing raw output. Most
///   common cause is a CRC-bad raw-output message silently dropped
///   during SHM drain — the drops counter only tracks ring-full
///   in `shm_write`, so a CRC drop does NOT inflate `shm_drops`
///   yet still loses the raw output. Pairs symmetrically with the
///   raw-output orphan-pairing detail above.
#[cfg(feature = "llm")]
pub(crate) fn host_side_llm_extract(
    payload_metrics: &mut [crate::test_support::PayloadMetrics],
    raw_outputs: &[crate::test_support::RawPayloadOutput],
) -> Vec<crate::assert::AssertDetail> {
    let mut failures = Vec::new();
    if raw_outputs.is_empty() {
        return failures;
    }
    // Build a HashMap from each PayloadMetrics' payload_index to its
    // position in the slice. Last-occurrence wins on duplicate
    // indices — but the guest's per-process counter is monotonic
    // and never reuses a value within a single VM run, so a
    // duplicate index in this map is a guest-side bug. The
    // `fetch_add(1, Relaxed)` atomic counter at
    // [`crate::scenario::payload_run::PAYLOAD_INVOCATION_COUNTER`]
    // guarantees uniqueness across threads as well — `Relaxed`
    // does not reorder the increment relative to itself, so
    // concurrent emits from N threads each receive a distinct
    // value. The "guest-side bug" framing applies to a future
    // regression that bypassed the counter, not to multi-thread
    // emit per se. The map is keyed by usize (the index) and
    // valued by usize (the slice position) so the pair-loop below
    // can rewrite the matching slot in O(1).
    let pm_index_lookup: std::collections::HashMap<usize, usize> = payload_metrics
        .iter()
        .enumerate()
        .map(|(pos, pm)| (pm.payload_index, pos))
        .collect();
    for raw in raw_outputs {
        let Some(&pm_pos) = pm_index_lookup.get(&raw.payload_index) else {
            // Orphan raw output — no PayloadMetrics carries the
            // matching index. Most likely cause is SHM ring overflow
            // dropping the empty-metrics PM, or a guest-side emit
            // path that ships RawPayloadOutput without its companion
            // PayloadMetrics. Surface as a failure detail so the
            // test fails loudly; skip extraction for this raw entry
            // and keep going on the rest.
            failures.push(crate::assert::AssertDetail::new(
                crate::assert::DetailKind::Other,
                format!(
                    "LlmExtract host pairing: raw output at payload_index={} has no \
                     matching PayloadMetrics slot — guest emission contract violated, \
                     or SHM ring dropped the empty-metrics companion message",
                    raw.payload_index,
                ),
            ));
            continue;
        };
        let hint_ref = raw.hint.as_deref();
        // Stdout-primary: try stdout first.
        let stdout_result = super::super::model::extract_via_llm(
            &raw.stdout,
            hint_ref,
            crate::test_support::MetricStream::Stdout,
        );
        let (mut metrics, load_err) = match stdout_result {
            Ok(m) => (m, None::<String>),
            Err(reason) => (Vec::new(), Some(reason)),
        };
        // Stderr fallback — only if stdout produced no metrics AND
        // the stdout call did not surface a load-failure reason
        // (the failure reason is identical across both calls; no
        // point re-invoking inference). Mirrors the legacy guest-
        // side fallback gate exactly. The Err arm here is
        // theoretically unreachable: when stdout's call returned
        // `Ok`, the model is memoized in `MODEL_CACHE` and a second
        // call cannot fail to load. Handled defensively in case a
        // future refactor changes that invariant — same surface
        // shape as a stdout-side load failure.
        if metrics.is_empty() && load_err.is_none() && !raw.stderr.is_empty() {
            match super::super::model::extract_via_llm(
                &raw.stderr,
                hint_ref,
                crate::test_support::MetricStream::Stderr,
            ) {
                Ok(m) => metrics = m,
                Err(reason) => {
                    failures.push(crate::assert::AssertDetail::new(
                        crate::assert::DetailKind::Other,
                        format!("{LLM_MODEL_LOAD_FAILED_PREFIX}{reason}"),
                    ));
                    continue;
                }
            }
        }
        if let Some(reason) = load_err {
            failures.push(crate::assert::AssertDetail::new(
                crate::assert::DetailKind::Other,
                format!("{LLM_MODEL_LOAD_FAILED_PREFIX}{reason}"),
            ));
            // Leave metrics empty in the PayloadMetrics slot. Skip
            // the structural-sanity check below — running it on an
            // empty vec would either no-op (no metrics to scan) or
            // produce a misleading detail that buries the real
            // load-failure reason.
            continue;
        }
        // Apply payload-author-declared polarity / unit hints. The
        // guest shipped these in `raw.metric_hints` because the
        // model-driven extraction runs post-VM-exit on the host —
        // the original `&'static [MetricHint]` slice cannot
        // round-trip through SHM. Mirrors the guest-side
        // `resolve_polarities` pass that runs on Json / ExitCode
        // payloads inside `payload_run::evaluate` so LlmExtract
        // metrics reach the sidecar with the same polarity / unit
        // classification a Json payload would receive.
        crate::scenario::payload_run::resolve_polarities_owned(&mut metrics, &raw.metric_hints);
        // Structural-sanity check. Every violation found surfaces
        // its own AssertDetail so a metric set that breaks multiple
        // invariants (e.g. NaN values AND a duplicate name) gives
        // the test author the full picture in one run rather than
        // forcing them to fix one defect class, re-run, fix the
        // next, re-run again.
        for reason in validate_llm_extraction(&metrics) {
            failures.push(crate::assert::AssertDetail::new(
                crate::assert::DetailKind::Other,
                reason,
            ));
        }
        // Per-payload bounds check. Workload-specific bounds
        // (minimum metric count, value magnitude) declared on the
        // payload's `metric_bounds` field run AFTER the universal
        // structural-sanity pass; they apply only to extracted
        // metrics that already passed unique-name / finite /
        // source-tag checks. A payload that didn't declare
        // `metric_bounds` (the common case) skips this pass.
        if let Some(bounds) = raw.metric_bounds.as_ref() {
            for reason in validate_metric_bounds(&metrics, bounds) {
                failures.push(crate::assert::AssertDetail::new(
                    crate::assert::DetailKind::Other,
                    reason,
                ));
            }
        }
        // Replace the empty-metrics slot with the extracted result.
        // Even if validation fails above, populate the PayloadMetrics
        // so debugging tools and the sidecar see what the model
        // emitted. The accompanying AssertDetail communicates the
        // rejection.
        payload_metrics[pm_pos].metrics = metrics;
    }

    // Post-pairing scan: flag empty-metrics PayloadMetrics whose
    // payload_index has no matching RawPayloadOutput. The most
    // likely cause is a CRC-bad RawPayloadOutput silently dropped
    // during SHM drain (the drain at run_ktstr_test_inner skips
    // CRC-bad entries without recording the loss in the
    // shm_drops counter, since that counter only tracks
    // ring-full and overflow paths in `shm_write`). Without this
    // surfacing, an LlmExtract test whose raw-output bytes
    // arrived corrupted would silently produce empty metrics and
    // fail downstream `MetricCheck::Min` / `MetricCheck::Exists` evaluations
    // with a "metric not found" message that hides the real cause.
    //
    // Ambiguity disclosure: we cannot tell from PayloadMetrics
    // alone which empty-metrics entries were intended as
    // LlmExtract placeholders versus legitimate Json-with-no-leaves
    // or ExitCode-only payloads. We only reach this scan when
    // `raw_outputs` is non-empty (the function early-returned at
    // the top of the body when it was empty), so by construction
    // the test exercises LlmExtract and a dropped raw-output is at
    // least possible. The detail's prose calls out the ambiguity
    // so an operator running a mixed-format test (LlmExtract + Json)
    // can dismiss false positives. Surfaces as a single combined
    // detail listing the suspicious indices rather than per-PM,
    // keeping the failure-rendering compact when many empty PMs
    // coexist.
    let raw_indices: std::collections::HashSet<usize> =
        raw_outputs.iter().map(|raw| raw.payload_index).collect();
    let suspicious: Vec<usize> = payload_metrics
        .iter()
        .filter(|pm| pm.metrics.is_empty() && !raw_indices.contains(&pm.payload_index))
        .map(|pm| pm.payload_index)
        .collect();
    if !suspicious.is_empty() {
        failures.push(crate::assert::AssertDetail::new(
            crate::assert::DetailKind::Other,
            format!(
                "LlmExtract host pairing: {} empty-metrics PayloadMetrics \
                 entries at payload_index={:?} have no matching RawPayloadOutput. \
                 If these were intended as LlmExtract payloads, the raw-output \
                 SHM messages may have been silently dropped during drain \
                 (CRC mismatch — the drop is invisible to the shm_drops \
                 counter, which only tracks ring-full / overflow). Re-run; \
                 transient CRC corruption is rare. False-positive case: a \
                 `Json` payload with no numeric leaves and an `ExitCode` \
                 payload both produce empty-metrics PayloadMetrics by design \
                 and would also surface here in a mixed-format test — \
                 dismiss this detail if your test mixes LlmExtract with \
                 legitimately-empty other formats.",
                suspicious.len(),
                suspicious,
            ),
        ));
    }

    failures
}

/// Structural-sanity check on a freshly-extracted
/// `OutputFormat::LlmExtract` metric set. Returns a `Vec<String>`
/// of every violation found; an empty vec means the set is
/// structurally well-formed.
///
/// Every metric is checked against ALL three invariants — a single
/// metric can contribute up to three violations (e.g. a duplicate
/// name AND a NaN value AND a non-LlmExtract source tag) so the
/// test author sees every defect class in one failure rather than
/// having to re-run after fixing each one in turn. Across the
/// whole set, every duplicate-name occurrence beyond the first
/// reports its own violation.
///
/// Universal checks only — every condition here is workload-
/// agnostic. Workload-specific assertions (latency ranges, RPS
/// ceilings, sign / magnitude bounds, minimum metric count) belong
/// in a per-payload validation API the framework does not yet
/// expose; the test author owns those.
///
/// 1. Every metric name is unique. Duplicate dotted paths imply
///    the LLM walker emitted the same key twice (malformed JSON
///    walkthrough or a walker aggregation bug) — downstream stats
///    would misattribute one value to the other regardless of which
///    workload produced the output.
/// 2. Every value is finite. NaN / ±inf in `PayloadMetrics`
///    poisons percentile comparisons downstream and never
///    represents a legitimate measurement, regardless of workload.
/// 3. Every metric carries `MetricSource::LlmExtract`. The host's
///    `extract_via_llm` walker stamps this field unconditionally,
///    so any drift here points at a bypass — the value didn't come
///    from the LLM-driven path even though it landed in a slot
///    we marked LlmExtract.
#[cfg(feature = "llm")]
pub(crate) fn validate_llm_extraction(metrics: &[crate::test_support::Metric]) -> Vec<String> {
    use std::collections::HashSet;
    // Empty-input fast-path mirrors the symmetric helper
    // [`crate::scenario::payload_run::resolve_polarities_owned`]:
    // skip the HashSet allocation and the for-loop so the no-op
    // case is structurally a no-op rather than an empty-iterator
    // walk. The capacity-zero allocation HashSet would amount to
    // is essentially free, but the early-return makes the contract
    // visible to a reader scanning the function.
    if metrics.is_empty() {
        return Vec::new();
    }
    let mut violations = Vec::new();
    let mut seen: HashSet<&str> = HashSet::with_capacity(metrics.len());
    for m in metrics {
        if !seen.insert(m.name.as_str()) {
            violations.push(format!(
                "LlmExtract emitted duplicate metric name '{}' — downstream stats would \
                 misattribute one value to the other; check the LLM walker for an \
                 aggregation bug or a malformed JSON path emitted by the model",
                m.name,
            ));
        }
        if !m.value.is_finite() {
            violations.push(format!(
                "LlmExtract metric '{}' has non-finite value {} — NaN / ±inf must not \
                 propagate into PayloadMetrics",
                m.name, m.value,
            ));
        }
        if m.source != crate::test_support::MetricSource::LlmExtract {
            violations.push(format!(
                "LlmExtract metric '{}' has source {:?}, expected MetricSource::LlmExtract — \
                 a value reached the LlmExtract slot without traversing the LLM walker",
                m.name, m.source,
            ));
        }
    }
    violations
}

/// Per-payload-bounds check applied AFTER the universal
/// structural-sanity pass in [`validate_llm_extraction`]. Returns
/// a `Vec<String>` of every violation found; an empty vec means
/// the metric set satisfies the declared bounds.
///
/// Each declared bound on [`crate::test_support::MetricBounds`] is
/// `Option`-wrapped, so a payload's bounds can scope to any subset
/// of the three checks. Disabled bounds (the `None` case) are
/// no-ops here — the function inspects each `Some(_)` branch
/// independently and emits per-violation diagnostics.
///
/// Diagnostics surface as `AssertDetail::new(DetailKind::Other, ...)`
/// at the call site in [`host_side_llm_extract`], so the per-bound
/// failure shape mirrors the universal-invariant violations: one
/// detail per violation, every detail carries enough context for
/// the operator to identify which bound fired and why.
///
/// 1. **`min_count`**: when set, an extracted set whose `.len()`
///    is below the threshold surfaces a violation naming the
///    expected minimum and the actual count. Pins the "did the
///    model produce enough metrics?" check that schbench-style
///    payloads need (an LLM regression that emits 1 metric on a
///    payload that historically produced 5+ silently degrades
///    downstream stats).
///
/// 2. **`value_min`**: when set, every metric whose value is
///    strictly below the threshold surfaces a violation naming
///    the metric, the value, and the bound. Pin the
///    non-negative-microseconds invariant for percentile
///    payloads — a negative latency reading is either a model
///    extraction error or a unit confusion, both of which the
///    bound surfaces loudly.
///
/// 3. **`value_max`**: symmetric upper-bound check. Catches
///    runaway values (a typo'd unit converter that read seconds
///    as microseconds and produced a 1e15 latency) before they
///    reach downstream stats.
///
/// Pre-1.0 design pin: callers MUST evaluate the universal
/// invariants in [`validate_llm_extraction`] FIRST. A NaN-bearing
/// metric would silently bypass the magnitude bounds here
/// because `NaN < x` and `NaN > x` both return false. The
/// universal pass rejects NaN unconditionally, so by the time
/// `validate_metric_bounds` runs the input is finite.
#[cfg(feature = "llm")]
pub(crate) fn validate_metric_bounds(
    metrics: &[crate::test_support::Metric],
    bounds: &crate::test_support::MetricBounds,
) -> Vec<String> {
    let mut violations = Vec::new();
    if let Some(min_count) = bounds.min_count
        && metrics.len() < min_count
    {
        violations.push(format!(
            "LlmExtract bounds: extracted {} metric(s), payload requires at least {} — \
             the model produced fewer metrics than the payload declared as a sanity \
             floor. Common causes: a regression in the LLM walker that drops branches \
             of the JSON tree, a payload output that's structurally different from \
             what the prompt template assumes, or a too-tight floor on `min_count`.",
            metrics.len(),
            min_count,
        ));
    }
    for m in metrics {
        if let Some(lo) = bounds.value_min
            && m.value < lo
        {
            violations.push(format!(
                "LlmExtract bounds: metric '{}' has value {} below payload's declared \
                 lower bound {} — values below the floor are either an extraction \
                 error or a unit-confusion bug. Adjust `value_min` if the floor is \
                 too tight, or fix the payload's output schema if the value should \
                 not have crossed the floor.",
                m.name, m.value, lo,
            ));
        }
        if let Some(hi) = bounds.value_max
            && m.value > hi
        {
            violations.push(format!(
                "LlmExtract bounds: metric '{}' has value {} above payload's declared \
                 upper bound {} — values above the ceiling are either an extraction \
                 error or a runaway from a typo'd unit converter. Adjust `value_max` \
                 if the ceiling is too tight, or fix the payload's output if the \
                 value should have stayed bounded.",
                m.name, m.value, hi,
            ));
        }
    }
    violations
}
