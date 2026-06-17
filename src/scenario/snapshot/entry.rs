//! [`SnapshotEntry`] — one entry's view across a `FailureDumpMap`
//! variant. Carries the rendered key (when BTF was present at
//! capture) plus accessors for typed reads, per-CPU narrowing, and
//! aggregation across per-CPU slots.

use crate::monitor::btf_render::RenderedValue;
use crate::monitor::dump::{FailureDumpEntry, FailureDumpPercpuEntry, FailureDumpPercpuHashEntry};

use super::{SnapshotError, SnapshotField, SnapshotResult, walk_dotted_path};

/// One entry's view — either a HASH (key, value) pair, a per-CPU
/// array entry, a per-CPU hash entry, a single rendered value, or
/// a missing-entry marker.
#[derive(Debug)]
#[must_use = "SnapshotEntry is a borrowed view; chain accessors"]
#[non_exhaustive]
pub enum SnapshotEntry<'a> {
    /// HASH map entry — `(key, value)` pair.
    Hash(&'a FailureDumpEntry),
    /// PERCPU_ARRAY entry — outer u32 key, inner per-CPU vec.
    Percpu(&'a FailureDumpPercpuEntry),
    /// PERCPU_HASH entry — rendered key, inner per-CPU vec.
    PercpuHash(&'a FailureDumpPercpuHashEntry),
    /// Single rendered value (ARRAY map's `value` field, or a
    /// per-CPU slot resolved via [`super::SnapshotMap::cpu`]).
    Value(&'a RenderedValue),
    /// No entry matched.
    Missing(SnapshotError),
}

impl<'a> SnapshotEntry<'a> {
    /// True when the lookup succeeded.
    pub fn is_present(&self) -> bool {
        !matches!(self, SnapshotEntry::Missing(_))
    }

    /// Walk into the entry's value side along a dotted path. Each
    /// path component names a [`RenderedValue::Struct`] member;
    /// pointer dereferences are followed transparently. Returns
    /// [`SnapshotField::Missing`] with an actionable error
    /// when the path cannot be resolved.
    pub fn get(&self, path: &str) -> SnapshotField<'a> {
        let value = match self {
            SnapshotEntry::Hash(e) => e.value.as_ref(),
            SnapshotEntry::Percpu(_) | SnapshotEntry::PercpuHash(_) => {
                let map_name = match self {
                    SnapshotEntry::Percpu(_) => "<percpu-array>".to_string(),
                    SnapshotEntry::PercpuHash(_) => "<percpu-hash>".to_string(),
                    _ => String::new(),
                };
                return SnapshotField::Missing(SnapshotError::PerCpuNotNarrowed { map: map_name });
            }
            SnapshotEntry::Value(v) => Some(*v),
            SnapshotEntry::Missing(err) => {
                return SnapshotField::Missing(err.clone());
            }
        };
        let Some(v) = value else {
            return SnapshotField::Missing(SnapshotError::NoRendered {
                map: "<entry>".to_string(),
                side: "value".to_string(),
            });
        };
        walk_dotted_path(v, path)
    }

    /// Look up the entry's KEY side along a dotted path. Mirror
    /// of [`Self::get`] but operates on the key's rendered
    /// structure (HASH / PERCPU_HASH only).
    pub fn key(&self, path: &str) -> SnapshotField<'a> {
        match self {
            SnapshotEntry::Hash(e) => match e.key.as_ref() {
                Some(v) => walk_dotted_path(v, path),
                None => SnapshotField::Missing(SnapshotError::NoRendered {
                    map: "<entry>".to_string(),
                    side: "key".to_string(),
                }),
            },
            SnapshotEntry::PercpuHash(e) => match e.key.as_ref() {
                Some(v) => walk_dotted_path(v, path),
                None => SnapshotField::Missing(SnapshotError::NoRendered {
                    map: "<entry>".to_string(),
                    side: "key".to_string(),
                }),
            },
            SnapshotEntry::Percpu(e) => {
                if path.is_empty() {
                    SnapshotField::PercpuKey { key: e.key }
                } else {
                    SnapshotField::Missing(SnapshotError::TypeMismatch {
                        expected: "Struct".to_string(),
                        actual: "Uint(percpu key)".to_string(),
                        requested: path.to_string(),
                    })
                }
            }
            SnapshotEntry::Value(_) => SnapshotField::Missing(SnapshotError::TypeMismatch {
                expected: "key".to_string(),
                actual: "single Value (no key)".to_string(),
                requested: path.to_string(),
            }),
            SnapshotEntry::Missing(err) => SnapshotField::Missing(err.clone()),
        }
    }

    // -----------------------------------------------------------------
    // Per-CPU aggregators. Apply only to `Percpu` / `PercpuHash`
    // entries; other variants return `Err(TypeMismatch)`. Inside the
    // per_cpu vec, slots whose value is `None` (CPU unmapped / out of
    // range — see `read_percpu_array_value` semantics) skip the
    // aggregation; slots whose rendered value can't decode to the
    // requested scalar return `Err(TypeMismatch)` immediately.
    //
    // `cpu_sum_*`, `cpu_max_*`, and `cpu_min_*` all return
    // `Err(NoMatch)` when no slot contributes (every slot `None`). A
    // `None` slot is UNREADABLE (host-read failure), NOT a real zero,
    // so an all-None sum of `0` would silently drop the missing data;
    // a readable map whose slots are all `0` still sums to `Ok(0)`.
    // -----------------------------------------------------------------

    /// Sum the per-CPU values at `path` as `u64`. Returns
    /// `Err(NoMatch)` when every slot is `None`: a `None` slot is
    /// UNREADABLE (host-read failure), not a real zero, so an all-None
    /// sum of `0` would silently drop the unreadable data. A readable
    /// map whose slots are all `0` still sums to `Ok(0)`. A slot whose
    /// rendered value cannot decode to `u64` propagates an Err
    /// immediately and stops the aggregation.
    pub fn cpu_sum_u64(&self, path: &str) -> SnapshotResult<u64> {
        let mut acc: u64 = 0;
        let mut contributed = false;
        self.try_for_each_cpu_value(path, |v| {
            acc = acc.saturating_add(SnapshotField::Value(v).as_u64()?);
            contributed = true;
            Ok(())
        })?;
        if contributed {
            Ok(acc)
        } else {
            Err(self.empty_aggregate_error("cpu_sum_u64"))
        }
    }

    /// Maximum of per-CPU values at `path` as `u64`. Returns
    /// `Err(NoMatch)` when every slot is `None` (no slot contributed).
    /// A slot whose rendered value cannot decode to `u64` propagates
    /// an Err immediately.
    pub fn cpu_max_u64(&self, path: &str) -> SnapshotResult<u64> {
        let mut best: Option<u64> = None;
        self.try_for_each_cpu_value(path, |v| {
            let n = SnapshotField::Value(v).as_u64()?;
            best = Some(best.map_or(n, |b| b.max(n)));
            Ok(())
        })?;
        best.ok_or_else(|| self.empty_aggregate_error("cpu_max_u64"))
    }

    /// Minimum of per-CPU values at `path` as `u64`. Returns
    /// `Err(NoMatch)` when every slot is `None`. A slot whose
    /// rendered value cannot decode to `u64` propagates an Err
    /// immediately.
    pub fn cpu_min_u64(&self, path: &str) -> SnapshotResult<u64> {
        let mut best: Option<u64> = None;
        self.try_for_each_cpu_value(path, |v| {
            let n = SnapshotField::Value(v).as_u64()?;
            best = Some(best.map_or(n, |b| b.min(n)));
            Ok(())
        })?;
        best.ok_or_else(|| self.empty_aggregate_error("cpu_min_u64"))
    }

    /// Sum the per-CPU values at `path` as `i64`. Returns
    /// `Err(NoMatch)` when every slot is `None`, matching
    /// [`Self::cpu_sum_u64`]: a `None` slot is UNREADABLE, not a real
    /// zero, so an all-None sum of `0` would be a silent drop. A
    /// readable all-zero map still sums to `Ok(0)`. The sum saturates
    /// at `i64::MIN` / `i64::MAX`. A slot whose rendered value cannot
    /// decode to `i64` propagates an Err immediately and stops the
    /// aggregation.
    pub fn cpu_sum_i64(&self, path: &str) -> SnapshotResult<i64> {
        let mut acc: i64 = 0;
        let mut contributed = false;
        self.try_for_each_cpu_value(path, |v| {
            acc = acc.saturating_add(SnapshotField::Value(v).as_i64()?);
            contributed = true;
            Ok(())
        })?;
        if contributed {
            Ok(acc)
        } else {
            Err(self.empty_aggregate_error("cpu_sum_i64"))
        }
    }

    /// Maximum of per-CPU values at `path` as `i64`. Returns
    /// `Err(NoMatch)` when every slot is `None` (no slot contributed).
    /// A slot whose rendered value cannot decode to `i64` propagates
    /// an Err immediately.
    pub fn cpu_max_i64(&self, path: &str) -> SnapshotResult<i64> {
        let mut best: Option<i64> = None;
        self.try_for_each_cpu_value(path, |v| {
            let n = SnapshotField::Value(v).as_i64()?;
            best = Some(best.map_or(n, |b| b.max(n)));
            Ok(())
        })?;
        best.ok_or_else(|| self.empty_aggregate_error("cpu_max_i64"))
    }

    /// Minimum of per-CPU values at `path` as `i64`. Returns
    /// `Err(NoMatch)` when every slot is `None`. A slot whose
    /// rendered value cannot decode to `i64` propagates an Err
    /// immediately.
    pub fn cpu_min_i64(&self, path: &str) -> SnapshotResult<i64> {
        let mut best: Option<i64> = None;
        self.try_for_each_cpu_value(path, |v| {
            let n = SnapshotField::Value(v).as_i64()?;
            best = Some(best.map_or(n, |b| b.min(n)));
            Ok(())
        })?;
        best.ok_or_else(|| self.empty_aggregate_error("cpu_min_i64"))
    }

    /// Sum the per-CPU values at `path` as `f64`. Returns
    /// `Err(NoMatch)` when every slot is `None`: a `None` slot is
    /// UNREADABLE, not a real zero, so an all-None sum of `0.0` would
    /// be a silent drop. A readable all-zero map still sums to
    /// `Ok(0.0)`. A slot whose rendered value cannot decode to `f64`
    /// propagates an Err immediately. NaN slot values propagate through
    /// `+=` per IEEE-754 — a single NaN slot makes the result NaN.
    pub fn cpu_sum_f64(&self, path: &str) -> SnapshotResult<f64> {
        let mut acc: f64 = 0.0;
        let mut contributed = false;
        self.try_for_each_cpu_value(path, |v| {
            acc += SnapshotField::Value(v).as_f64()?;
            contributed = true;
            Ok(())
        })?;
        if contributed {
            Ok(acc)
        } else {
            Err(self.empty_aggregate_error("cpu_sum_f64"))
        }
    }

    /// Maximum of per-CPU values at `path` as `f64`. Returns
    /// `Err(NoMatch)` when every slot is `None`. A slot whose
    /// rendered value cannot decode to `f64` propagates an Err
    /// immediately. NaN slot values are filtered out per
    /// `f64::max` semantics — `f64::max(NaN, x)` returns `x`, so a
    /// NaN slot never wins against a non-NaN slot. An all-NaN run
    /// is an edge case: the first NaN slot sets `best=NaN`, then
    /// subsequent `NaN.max(NaN)` returns NaN, so the final result
    /// is `Ok(NaN)` rather than NoMatch.
    pub fn cpu_max_f64(&self, path: &str) -> SnapshotResult<f64> {
        let mut best: Option<f64> = None;
        self.try_for_each_cpu_value(path, |v| {
            let n = SnapshotField::Value(v).as_f64()?;
            best = Some(best.map_or(n, |b| b.max(n)));
            Ok(())
        })?;
        best.ok_or_else(|| self.empty_aggregate_error("cpu_max_f64"))
    }

    /// Minimum of per-CPU values at `path` as `f64`. Returns
    /// `Err(NoMatch)` when every slot is `None`. A slot whose
    /// rendered value cannot decode to `f64` propagates an Err
    /// immediately. NaN slot values are filtered out per
    /// `f64::min` semantics — `f64::min(NaN, x)` returns `x`, so a
    /// NaN slot never wins against a non-NaN slot. An all-NaN run
    /// yields `Ok(NaN)` rather than NoMatch — same edge case as
    /// `cpu_max_f64`.
    pub fn cpu_min_f64(&self, path: &str) -> SnapshotResult<f64> {
        let mut best: Option<f64> = None;
        self.try_for_each_cpu_value(path, |v| {
            let n = SnapshotField::Value(v).as_f64()?;
            best = Some(best.map_or(n, |b| b.min(n)));
            Ok(())
        })?;
        best.ok_or_else(|| self.empty_aggregate_error("cpu_min_f64"))
    }

    /// Iterate non-None per-CPU rendered values at `path`. For each
    /// successful slot, invokes `f(cpu_idx, &RenderedValue)`. Slots
    /// whose value is `None` are skipped silently; the iteration
    /// stops at the first slot whose value cannot be reached via
    /// `path` (returning the path-walk error). Returns `Err` for
    /// non-percpu variants.
    pub fn cpu_each<F>(&self, path: &str, mut f: F) -> SnapshotResult<()>
    where
        F: FnMut(usize, &'a RenderedValue) -> SnapshotResult<()>,
    {
        let per_cpu: &[Option<RenderedValue>] = match self {
            SnapshotEntry::Percpu(e) => &e.per_cpu,
            SnapshotEntry::PercpuHash(e) => &e.per_cpu,
            SnapshotEntry::Hash(_) | SnapshotEntry::Value(_) => {
                return Err(SnapshotError::TypeMismatch {
                    expected: "Percpu / PercpuHash".to_string(),
                    actual: self.variant_name().to_string(),
                    requested: path.to_string(),
                });
            }
            SnapshotEntry::Missing(err) => return Err(err.clone()),
        };
        for (cpu_idx, slot) in per_cpu.iter().enumerate() {
            let Some(rendered) = slot.as_ref() else {
                continue;
            };
            let walked = walk_dotted_path(rendered, path);
            let value = match walked {
                SnapshotField::Value(v) => v,
                SnapshotField::PercpuKey { .. } => {
                    return Err(SnapshotError::TypeMismatch {
                        expected: "rendered value".to_string(),
                        actual: "PercpuKey".to_string(),
                        requested: path.to_string(),
                    });
                }
                SnapshotField::Missing(err) => return Err(err),
            };
            f(cpu_idx, value)?;
        }
        Ok(())
    }

    /// Shared walk helper for `cpu_sum_*` / `cpu_max_*` / `cpu_min_*`
    /// — invokes `f` on every non-None slot's rendered value.
    fn try_for_each_cpu_value<F>(&self, path: &str, mut f: F) -> SnapshotResult<()>
    where
        F: FnMut(&'a RenderedValue) -> SnapshotResult<()>,
    {
        self.cpu_each(path, |_, v| f(v))
    }

    /// Name for diagnostic messages. Used by the per-CPU aggregator
    /// `TypeMismatch` paths so the error names the actual variant.
    fn variant_name(&self) -> &'static str {
        match self {
            SnapshotEntry::Hash(_) => "Hash",
            SnapshotEntry::Percpu(_) => "Percpu",
            SnapshotEntry::PercpuHash(_) => "PercpuHash",
            SnapshotEntry::Value(_) => "Value",
            SnapshotEntry::Missing(_) => "Missing",
        }
    }

    /// Build the `NoMatch` error for an empty per-CPU aggregate
    /// (max / min of all-None or all-decode-fail). `op` names the
    /// caller so the error message points at the right method.
    fn empty_aggregate_error(&self, op: &str) -> SnapshotError {
        SnapshotError::NoMatch {
            map: format!("<{}>", self.variant_name()),
            op: op.to_string(),
            len: 0,
            available_keys: Vec::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// SnapshotField — terminal traversal value
