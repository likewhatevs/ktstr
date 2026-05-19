//! [`JsonField`] — terminal accessor for a typed read out of a
//! [`serde_json::Value`] (the scheduler-stats JSON the relay
//! captures), plus the [`stats_path`] entry point and the
//! [`walk_json_path`] / [`describe_json_kind`] / [`json_to_u64`] /
//! [`json_to_i64`] / [`json_to_f64`] helpers it funnels through.

use super::{SnapshotError, SnapshotResult};

/// One value's view at the leaf of a dotted-path walk over a
/// [`serde_json::Value`]. Returned by [`stats_path`] / `StatsValue::path`.
///
/// Mirrors the [`super::SnapshotField`] shape so test authors who already
/// know the BPF-snapshot accessor surface get the same `as_u64` /
/// `as_i64` / `as_f64` / `as_bool` / `as_str` terminals on the
/// scx_stats JSON projection. Errors flow through the same
/// [`SnapshotError`] variants — `FieldNotFound` carries the
/// available object keys, `NotAStruct` flags a non-object cursor,
/// `TypeMismatch` reports the actual JSON shape — so failure-path
/// rendering in temporal assertions is identical regardless of
/// which side of the
/// [`Sample`](crate::scenario::sample::Sample) bundle the lookup
/// originated on.
#[derive(Debug, Clone)]
#[must_use = "JsonField is a borrowed view; call as_u64 / as_i64 / etc. to extract"]
#[non_exhaustive]
pub enum JsonField<'a> {
    /// Resolved JSON value at the leaf of the path walk.
    Value(&'a serde_json::Value),
    /// Path could not be resolved.
    Missing(SnapshotError),
}

impl<'a> JsonField<'a> {
    /// True when the path resolved.
    pub fn is_present(&self) -> bool {
        !matches!(self, JsonField::Missing(_))
    }

    /// Underlying JSON value if present.
    pub fn raw(&self) -> Option<&'a serde_json::Value> {
        match self {
            JsonField::Value(v) => Some(*v),
            JsonField::Missing(_) => None,
        }
    }

    /// Error reference when the path could not be resolved.
    pub fn error(&self) -> Option<&SnapshotError> {
        match self {
            JsonField::Missing(err) => Some(err),
            _ => None,
        }
    }

    /// Walk further into a sub-field. Composable with the result of
    /// [`stats_path`] — `stats_path(v, "layers").path("batch.util")`
    /// is the canonical "drill into a periodic-stats object" shape.
    pub fn path(&self, path: &str) -> JsonField<'a> {
        match self {
            JsonField::Value(v) => walk_json_path(v, path),
            JsonField::Missing(err) => JsonField::Missing(err.clone()),
        }
    }

    /// Read as `u64`. Accepts JSON integers (positive only), JSON
    /// booleans (true → 1, false → 0), and JSON strings whose
    /// content parses as a u64 (scx_stats sometimes stringifies
    /// large counters to avoid 53-bit float collapse). Returns
    /// [`SnapshotError::TypeMismatch`] otherwise.
    pub fn as_u64(&self) -> SnapshotResult<u64> {
        match self {
            JsonField::Value(v) => json_to_u64(v),
            JsonField::Missing(err) => Err(err.clone()),
        }
    }

    /// Read as `i64`. Accepts JSON integers (any sign), JSON
    /// booleans (true → 1, false → 0), and JSON strings whose
    /// content parses as an i64.
    pub fn as_i64(&self) -> SnapshotResult<i64> {
        match self {
            JsonField::Value(v) => json_to_i64(v),
            JsonField::Missing(err) => Err(err.clone()),
        }
    }

    /// Read as `f64`. Accepts JSON numbers (integers and
    /// floating-point) and JSON strings whose content parses as
    /// f64.
    pub fn as_f64(&self) -> SnapshotResult<f64> {
        match self {
            JsonField::Value(v) => json_to_f64(v),
            JsonField::Missing(err) => Err(err.clone()),
        }
    }

    /// Read as `bool`. Accepts JSON booleans directly; rejects
    /// everything else. Distinct from `as_u64() != 0` so the call
    /// site reads honestly: a `bool` claim wants a JSON `true`/
    /// `false`, not a stringified `"1"` that happens to parse.
    pub fn as_bool(&self) -> SnapshotResult<bool> {
        match self {
            JsonField::Value(serde_json::Value::Bool(b)) => Ok(*b),
            JsonField::Value(other) => Err(SnapshotError::TypeMismatch {
                expected: "bool".to_string(),
                actual: describe_json_kind(other),
                requested: String::new(),
            }),
            JsonField::Missing(err) => Err(err.clone()),
        }
    }

    /// Read as `&str`. Accepts JSON strings only.
    pub fn as_str(&self) -> SnapshotResult<&'a str> {
        match self {
            JsonField::Value(serde_json::Value::String(s)) => Ok(s.as_str()),
            JsonField::Value(other) => Err(SnapshotError::TypeMismatch {
                expected: "str".to_string(),
                actual: describe_json_kind(other),
                requested: String::new(),
            }),
            JsonField::Missing(err) => Err(err.clone()),
        }
    }
}

/// Build a [`JsonField`] view rooted at `value` and walk along the
/// dotted path. An empty path returns the root unchanged so a
/// caller writing `stats_path(v, "").as_f64()` (e.g. for a
/// scalar-rooted stats response) hits the typed scalar accessor
/// directly.
///
/// Mirrors [`super::Snapshot::var`] / [`super::SnapshotEntry::get`] in error
/// shape: typos and missing keys surface as
/// [`SnapshotError::FieldNotFound`] with the available sibling
/// keys at the failing depth — the same diagnostic experience the
/// BPF-snapshot side already provides. scx_stats payloads commonly
/// nest layer / cgroup / cpu maps under top-level keys, so the
/// dotted form `"layers.batch.util"` is the canonical drill-down
/// for layered scheduler stats.
pub fn stats_path<'a>(value: &'a serde_json::Value, path: &str) -> JsonField<'a> {
    walk_json_path(value, path)
}

fn walk_json_path<'a>(root: &'a serde_json::Value, path: &str) -> JsonField<'a> {
    if path.is_empty() {
        return JsonField::Value(root);
    }
    let mut cursor: &serde_json::Value = root;
    let mut walked = String::new();
    for component in path.split('.') {
        if component.is_empty() {
            return JsonField::Missing(SnapshotError::EmptyPathComponent {
                requested: path.to_string(),
            });
        }
        match cursor {
            serde_json::Value::Object(map) => {
                let Some(next) = map.get(component) else {
                    let mut available: Vec<String> = map.keys().cloned().collect();
                    available.sort();
                    return JsonField::Missing(SnapshotError::FieldNotFound {
                        requested: path.to_string(),
                        walked: walked.clone(),
                        component: component.to_string(),
                        available,
                    });
                };
                cursor = next;
            }
            other => {
                return JsonField::Missing(SnapshotError::NotAStruct {
                    requested: path.to_string(),
                    walked: walked.clone(),
                    component: component.to_string(),
                    kind: describe_json_kind(other),
                });
            }
        }
        if !walked.is_empty() {
            walked.push('.');
        }
        walked.push_str(component);
    }
    JsonField::Value(cursor)
}

fn describe_json_kind(v: &serde_json::Value) -> String {
    match v {
        serde_json::Value::Null => "Null",
        serde_json::Value::Bool(_) => "Bool",
        serde_json::Value::Number(_) => "Number",
        serde_json::Value::String(_) => "String",
        serde_json::Value::Array(_) => "Array",
        serde_json::Value::Object(_) => "Object",
    }
    .to_string()
}

fn json_to_u64(v: &serde_json::Value) -> SnapshotResult<u64> {
    match v {
        serde_json::Value::Number(n) => {
            if let Some(u) = n.as_u64() {
                Ok(u)
            } else if let Some(i) = n.as_i64() {
                if i < 0 {
                    Err(SnapshotError::TypeMismatch {
                        expected: "u64".to_string(),
                        actual: "Int(negative)".to_string(),
                        requested: String::new(),
                    })
                } else {
                    Ok(i as u64)
                }
            } else if let Some(f) = n.as_f64() {
                if !f.is_finite() || f < 0.0 {
                    Err(SnapshotError::TypeMismatch {
                        expected: "u64".to_string(),
                        actual: "Float(non-coercible)".to_string(),
                        requested: String::new(),
                    })
                } else if f.fract() != 0.0 {
                    Err(SnapshotError::TypeMismatch {
                        expected: "integer".to_string(),
                        actual: "non-integer float".to_string(),
                        requested: String::new(),
                    })
                } else {
                    Ok(f as u64)
                }
            } else {
                Err(SnapshotError::TypeMismatch {
                    expected: "u64".to_string(),
                    actual: "Number(unrepresentable)".to_string(),
                    requested: String::new(),
                })
            }
        }
        serde_json::Value::Bool(b) => Ok(u64::from(*b)),
        serde_json::Value::String(s) => s.parse::<u64>().map_err(|_| SnapshotError::TypeMismatch {
            expected: "u64".to_string(),
            actual: "String(non-numeric)".to_string(),
            requested: String::new(),
        }),
        other => Err(SnapshotError::TypeMismatch {
            expected: "u64".to_string(),
            actual: describe_json_kind(other),
            requested: String::new(),
        }),
    }
}

fn json_to_i64(v: &serde_json::Value) -> SnapshotResult<i64> {
    match v {
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Ok(i)
            } else if let Some(u) = n.as_u64() {
                if u > i64::MAX as u64 {
                    Err(SnapshotError::TypeMismatch {
                        expected: "i64".to_string(),
                        actual: "Uint(>i64::MAX)".to_string(),
                        requested: String::new(),
                    })
                } else {
                    Ok(u as i64)
                }
            } else if let Some(f) = n.as_f64() {
                if !f.is_finite() {
                    Err(SnapshotError::TypeMismatch {
                        expected: "i64".to_string(),
                        actual: "Float(non-finite)".to_string(),
                        requested: String::new(),
                    })
                } else if f.fract() != 0.0 {
                    Err(SnapshotError::TypeMismatch {
                        expected: "integer".to_string(),
                        actual: "non-integer float".to_string(),
                        requested: String::new(),
                    })
                } else {
                    Ok(f as i64)
                }
            } else {
                Err(SnapshotError::TypeMismatch {
                    expected: "i64".to_string(),
                    actual: "Number(unrepresentable)".to_string(),
                    requested: String::new(),
                })
            }
        }
        serde_json::Value::Bool(b) => Ok(i64::from(*b)),
        serde_json::Value::String(s) => s.parse::<i64>().map_err(|_| SnapshotError::TypeMismatch {
            expected: "i64".to_string(),
            actual: "String(non-numeric)".to_string(),
            requested: String::new(),
        }),
        other => Err(SnapshotError::TypeMismatch {
            expected: "i64".to_string(),
            actual: describe_json_kind(other),
            requested: String::new(),
        }),
    }
}

fn json_to_f64(v: &serde_json::Value) -> SnapshotResult<f64> {
    match v {
        serde_json::Value::Number(n) => n.as_f64().ok_or(SnapshotError::TypeMismatch {
            expected: "f64".to_string(),
            actual: "Number(unrepresentable)".to_string(),
            requested: String::new(),
        }),
        serde_json::Value::String(s) => s.parse::<f64>().map_err(|_| SnapshotError::TypeMismatch {
            expected: "f64".to_string(),
            actual: "String(non-numeric)".to_string(),
            requested: String::new(),
        }),
        other => Err(SnapshotError::TypeMismatch {
            expected: "f64".to_string(),
            actual: describe_json_kind(other),
            requested: String::new(),
        }),
    }
}

// ---------------------------------------------------------------------------
// Dotted-path walker
