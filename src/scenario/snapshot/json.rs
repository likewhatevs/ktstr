//! [`JsonField`] — terminal accessor for a typed read out of a
//! [`serde_json::Value`] (the scheduler-stats JSON the relay
//! captures), plus the [`stats_path`] entry point and the
//! [`walk_json_path`] / [`describe_json_kind`] / [`json_to_u64`] /
//! [`json_to_i64`] / [`json_to_f64`] helpers it funnels through.

use super::{SnapshotError, SnapshotResult};

/// One value's view at the leaf of a dotted-path walk over a
/// [`serde_json::Value`]. Returned by [`stats_path`] / `StatsValue::get`.
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
    /// [`stats_path`] — `stats_path(v, "layers").get("batch.util")`
    /// is the canonical "drill into a periodic-stats object" shape.
    /// Mirrors [`super::SnapshotField::get`] so a test author moves
    /// between the BTF-rendered and JSON-rendered surfaces without
    /// re-learning the navigator method name.
    pub fn get(&self, path: &str) -> JsonField<'a> {
        match self {
            JsonField::Value(v) => walk_json_path(v, path),
            JsonField::Missing(err) => JsonField::Missing(err.clone()),
        }
    }

    /// Read as `u64`. Accepts JSON integers (positive only), JSON
    /// booleans (true → 1, false → 0), JSON strings whose
    /// content parses as a u64 (scx_stats sometimes stringifies
    /// large counters to avoid 53-bit float collapse), and JSON
    /// floats that are integral and non-negative (`5.0` → 5;
    /// fractional, negative, or non-finite floats error). Returns
    /// [`SnapshotError::TypeMismatch`] otherwise.
    pub fn as_u64(&self) -> SnapshotResult<u64> {
        match self {
            JsonField::Value(v) => json_to_u64(v),
            JsonField::Missing(err) => Err(err.clone()),
        }
    }

    /// Read as `i64`. Accepts JSON integers (any sign), JSON
    /// booleans (true → 1, false → 0), JSON strings whose
    /// content parses as an i64, and integral finite JSON floats
    /// (`9.0` → 9; fractional or non-finite floats error).
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

    /// Read as `Vec<u64>` from a [`serde_json::Value::Array`] whose
    /// every element coerces via [`Self::as_u64`]'s rules. Mirrors
    /// [`super::SnapshotField::as_u64_array`] so JSON-side stats
    /// reads use the same method name as BTF-side BPF reads.
    pub fn as_u64_array(&self) -> SnapshotResult<Vec<u64>> {
        json_to_typed_array(self, json_to_u64, "u64")
    }

    /// Read as `Vec<u32>` from a JSON array. Mirrors
    /// [`super::SnapshotField::as_u32_array`]; out-of-range values
    /// error rather than silently truncate.
    pub fn as_u32_array(&self) -> SnapshotResult<Vec<u32>> {
        json_to_typed_array(
            self,
            |v| {
                json_to_u64(v).and_then(|x| {
                    u32::try_from(x).map_err(|_| SnapshotError::TypeMismatch {
                        expected: "u32".to_string(),
                        actual: "Uint(>u32::MAX)".to_string(),
                        requested: String::new(),
                    })
                })
            },
            "u32",
        )
    }

    /// Read as `Vec<i64>` from a JSON array. Mirrors
    /// [`super::SnapshotField::as_i64_array`].
    pub fn as_i64_array(&self) -> SnapshotResult<Vec<i64>> {
        json_to_typed_array(self, json_to_i64, "i64")
    }

    /// Read as `Vec<f64>` from a JSON array. Mirrors
    /// [`super::SnapshotField::as_f64_array`].
    pub fn as_f64_array(&self) -> SnapshotResult<Vec<f64>> {
        json_to_typed_array(self, json_to_f64, "f64")
    }

    /// Read as `Vec<bool>` from a JSON array of booleans. Mirrors
    /// [`super::SnapshotField::as_bool_array`]; rejects mixed
    /// arrays (no implicit truthiness coercion — JSON-side `bool`
    /// already has a wire shape).
    pub fn as_bool_array(&self) -> SnapshotResult<Vec<bool>> {
        json_to_typed_array(
            self,
            |v| match v {
                serde_json::Value::Bool(b) => Ok(*b),
                other => Err(SnapshotError::TypeMismatch {
                    expected: "bool".to_string(),
                    actual: describe_json_kind(other),
                    requested: String::new(),
                }),
            },
            "bool",
        )
    }

    /// Iterate the elements of a JSON array as [`JsonField`]s so
    /// chained navigation composes for arrays-of-objects:
    /// `field.iter_members().filter_map(|el| el.get("name").as_u64().ok())`.
    /// Mirrors [`super::SnapshotField::iter_members`].
    ///
    /// Yields nothing for non-array values or missing fields —
    /// the empty iterator is the natural "no elements" shape when
    /// the chain just wants to fold over what's there. Callers
    /// needing to distinguish "absent" from "empty" check
    /// [`Self::is_present`] or [`Self::error`] explicitly.
    pub fn iter_members(&self) -> impl Iterator<Item = JsonField<'a>> + '_ {
        let elements: &[serde_json::Value] = match self {
            JsonField::Value(serde_json::Value::Array(a)) => a.as_slice(),
            _ => &[],
        };
        elements.iter().map(JsonField::Value)
    }
}

/// Shared typed-array coercion used by [`JsonField::as_u64_array`]
/// and siblings. Mirrors the SnapshotField helper so callers see
/// uniform diagnostics across surfaces.
fn json_to_typed_array<T, F>(
    field: &JsonField<'_>,
    coerce: F,
    type_name: &'static str,
) -> SnapshotResult<Vec<T>>
where
    F: Fn(&serde_json::Value) -> SnapshotResult<T>,
{
    let value = match field {
        JsonField::Value(v) => *v,
        JsonField::Missing(err) => return Err(err.clone()),
    };
    let elements = match value {
        serde_json::Value::Array(a) => a.as_slice(),
        other => {
            return Err(SnapshotError::TypeMismatch {
                expected: format!("[{type_name}]"),
                actual: describe_json_kind(other),
                requested: String::new(),
            });
        }
    };
    let mut out = Vec::with_capacity(elements.len());
    for element in elements {
        out.push(coerce(element)?);
    }
    Ok(out)
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

#[cfg(test)]
mod tests_coercion {
    //! Host-pure coverage for the JSON coercion helpers
    //! ([`json_to_u64`] / [`json_to_i64`] / [`json_to_f64`] /
    //! [`describe_json_kind`]) and the [`JsonField`] terminal
    //! accessors. The dotted-path walk is exercised by the snapshot
    //! integration tests; this module pins the per-branch coercion
    //! rules that decide whether a stats read SUCCEEDS or surfaces a
    //! typed [`SnapshotError`] — the silent-wrong-answer surface where
    //! a swapped arm would coerce a value the device should reject.
    use super::*;
    use serde_json::json;

    // ---- describe_json_kind: every Value discriminant ----

    #[test]
    fn describe_json_kind_names_each_value_shape() {
        assert_eq!(describe_json_kind(&json!(null)), "Null");
        assert_eq!(describe_json_kind(&json!(true)), "Bool");
        assert_eq!(describe_json_kind(&json!(1)), "Number");
        assert_eq!(describe_json_kind(&json!("s")), "String");
        assert_eq!(describe_json_kind(&json!([])), "Array");
        assert_eq!(describe_json_kind(&json!({})), "Object");
    }

    // ---- json_to_u64: each accepted/rejected branch ----

    #[test]
    fn json_to_u64_accepts_uint_bool_and_numeric_string() {
        assert_eq!(json_to_u64(&json!(42u64)).unwrap(), 42);
        assert_eq!(json_to_u64(&json!(true)).unwrap(), 1);
        assert_eq!(json_to_u64(&json!(false)).unwrap(), 0);
        // scx_stats stringifies large counters to dodge 53-bit float collapse.
        assert_eq!(
            json_to_u64(&json!("18446744073709551615")).unwrap(),
            u64::MAX
        );
    }

    #[test]
    fn json_to_u64_accepts_integral_float_rejects_fractional_and_negative() {
        // A JSON float that happens to be integral coerces.
        assert_eq!(json_to_u64(&json!(5.0)).unwrap(), 5);
        // A fractional float is not an integer.
        match json_to_u64(&json!(2.5)) {
            Err(SnapshotError::TypeMismatch {
                expected, actual, ..
            }) => {
                assert_eq!(expected, "integer");
                assert_eq!(actual, "non-integer float");
            }
            other => panic!("expected non-integer-float TypeMismatch, got {other:?}"),
        }
        // A negative float cannot be a u64.
        match json_to_u64(&json!(-2.5)) {
            Err(SnapshotError::TypeMismatch {
                expected, actual, ..
            }) => {
                assert_eq!(expected, "u64");
                assert_eq!(actual, "Float(non-coercible)");
            }
            other => panic!("expected negative-float TypeMismatch, got {other:?}"),
        }
    }

    #[test]
    fn json_to_u64_rejects_negative_int_nonnumeric_string_and_other_shapes() {
        match json_to_u64(&json!(-5)) {
            Err(SnapshotError::TypeMismatch { actual, .. }) => {
                assert_eq!(actual, "Int(negative)");
            }
            other => panic!("expected negative-int TypeMismatch, got {other:?}"),
        }
        match json_to_u64(&json!("abc")) {
            Err(SnapshotError::TypeMismatch { actual, .. }) => {
                assert_eq!(actual, "String(non-numeric)");
            }
            other => panic!("expected non-numeric-string TypeMismatch, got {other:?}"),
        }
        // null / array / object fall through to the describe_json_kind arm.
        assert!(
            json_to_u64(&json!(null))
                .unwrap_err()
                .to_string()
                .contains("Null")
        );
        assert!(matches!(
            json_to_u64(&json!({"k": 1})),
            Err(SnapshotError::TypeMismatch { .. })
        ));
    }

    // ---- json_to_i64: signed-specific branches ----

    #[test]
    fn json_to_i64_accepts_signed_bool_string_and_integral_float() {
        assert_eq!(json_to_i64(&json!(-7)).unwrap(), -7);
        assert_eq!(json_to_i64(&json!(true)).unwrap(), 1);
        assert_eq!(json_to_i64(&json!("-123")).unwrap(), -123);
        assert_eq!(json_to_i64(&json!(9.0)).unwrap(), 9);
    }

    #[test]
    fn json_to_i64_rejects_uint_over_i64_max_and_fractional_float() {
        // u64::MAX is stored as a u64 Number; it overflows i64.
        match json_to_i64(&json!(u64::MAX)) {
            Err(SnapshotError::TypeMismatch { actual, .. }) => {
                assert_eq!(actual, "Uint(>i64::MAX)");
            }
            other => panic!("expected over-i64::MAX TypeMismatch, got {other:?}"),
        }
        match json_to_i64(&json!(2.5)) {
            Err(SnapshotError::TypeMismatch {
                expected, actual, ..
            }) => {
                assert_eq!(expected, "integer");
                assert_eq!(actual, "non-integer float");
            }
            other => panic!("expected fractional-float TypeMismatch, got {other:?}"),
        }
    }

    #[test]
    fn json_to_i64_rejects_nonnumeric_string_and_other_shapes() {
        match json_to_i64(&json!("nope")) {
            Err(SnapshotError::TypeMismatch { actual, .. }) => {
                assert_eq!(actual, "String(non-numeric)");
            }
            other => panic!("expected non-numeric-string TypeMismatch, got {other:?}"),
        }
        assert!(matches!(
            json_to_i64(&json!([1, 2])),
            Err(SnapshotError::TypeMismatch { .. })
        ));
    }

    // ---- json_to_f64 ----

    #[test]
    fn json_to_f64_accepts_number_and_numeric_string_rejects_others() {
        assert_eq!(json_to_f64(&json!(1.5)).unwrap(), 1.5);
        // An integer Number coerces to f64.
        assert_eq!(json_to_f64(&json!(3)).unwrap(), 3.0);
        assert_eq!(json_to_f64(&json!("2.25")).unwrap(), 2.25);
        match json_to_f64(&json!("x")) {
            Err(SnapshotError::TypeMismatch { actual, .. }) => {
                assert_eq!(actual, "String(non-numeric)");
            }
            other => panic!("expected non-numeric-string TypeMismatch, got {other:?}"),
        }
        assert!(matches!(
            json_to_f64(&json!(null)),
            Err(SnapshotError::TypeMismatch { .. })
        ));
    }

    // ---- JsonField terminal accessors + Missing propagation ----

    #[test]
    fn json_field_is_present_raw_and_error_reflect_variant() {
        let v = json!({"a": 1});
        let present = stats_path(&v, "a");
        assert!(present.is_present());
        assert!(present.raw().is_some());
        assert!(present.error().is_none());

        let missing = stats_path(&v, "nope");
        assert!(!missing.is_present());
        assert!(missing.raw().is_none());
        assert!(missing.error().is_some());
    }

    #[test]
    fn json_field_as_bool_accepts_bool_rejects_other_and_propagates_missing() {
        let t = json!({"on": true});
        assert!(stats_path(&t, "on").as_bool().unwrap());
        // A stringified "1" is NOT a bool — honest typing.
        let n = json!({"on": "1"});
        assert!(matches!(
            stats_path(&n, "on").as_bool(),
            Err(SnapshotError::TypeMismatch { .. })
        ));
        // A missing path propagates the FieldNotFound, not a type error.
        assert!(matches!(
            stats_path(&n, "absent").as_bool(),
            Err(SnapshotError::FieldNotFound { .. })
        ));
    }

    #[test]
    fn json_field_as_str_accepts_string_rejects_other_and_propagates_missing() {
        let v = json!({"name": "batch"});
        assert_eq!(stats_path(&v, "name").as_str().unwrap(), "batch");
        let num = json!({"name": 7});
        assert!(matches!(
            stats_path(&num, "name").as_str(),
            Err(SnapshotError::TypeMismatch { .. })
        ));
        assert!(matches!(
            stats_path(&v, "absent").as_str(),
            Err(SnapshotError::FieldNotFound { .. })
        ));
    }

    #[test]
    fn json_field_scalar_accessors_propagate_missing_error() {
        let v = json!({"x": 1});
        let missing = stats_path(&v, "absent");
        // Every typed terminal returns the SAME FieldNotFound, not a coercion error.
        assert!(matches!(
            missing.as_u64(),
            Err(SnapshotError::FieldNotFound { .. })
        ));
        assert!(matches!(
            missing.as_i64(),
            Err(SnapshotError::FieldNotFound { .. })
        ));
        assert!(matches!(
            missing.as_f64(),
            Err(SnapshotError::FieldNotFound { .. })
        ));
    }

    #[test]
    fn json_field_get_on_missing_stays_missing() {
        let v = json!({"x": 1});
        // Drilling into an already-failed lookup keeps the original error.
        let chained = stats_path(&v, "absent").get("deeper");
        assert!(matches!(
            chained.error(),
            Some(SnapshotError::FieldNotFound { .. })
        ));
    }

    #[test]
    fn json_field_iter_members_yields_elements_and_empty_for_nonarray() {
        let arr = json!([10, 20, 30]);
        let got: Vec<u64> = stats_path(&arr, "")
            .iter_members()
            .map(|el| el.as_u64().unwrap())
            .collect();
        assert_eq!(got, vec![10, 20, 30]);
        // A non-array value yields nothing (the natural "no elements" shape).
        let scalar = json!(5);
        assert_eq!(stats_path(&scalar, "").iter_members().count(), 0);
        // A missing field also yields nothing.
        let obj = json!({"a": 1});
        assert_eq!(stats_path(&obj, "absent").iter_members().count(), 0);
    }

    // ---- typed-array coercion: success, out-of-range, non-array, Missing ----

    #[test]
    fn json_field_typed_arrays_extract_each_element_type() {
        let v = json!({
            "u": [1, 2, 3],
            "i": [-1, 0, 5],
            "f": [1.5, 2.0],
            "b": [true, false, true],
        });
        assert_eq!(
            stats_path(&v, "u").as_u64_array().unwrap(),
            vec![1u64, 2, 3]
        );
        assert_eq!(
            stats_path(&v, "i").as_i64_array().unwrap(),
            vec![-1i64, 0, 5]
        );
        assert_eq!(stats_path(&v, "f").as_f64_array().unwrap(), vec![1.5, 2.0]);
        assert_eq!(
            stats_path(&v, "b").as_bool_array().unwrap(),
            vec![true, false, true]
        );
    }

    #[test]
    fn json_field_as_u32_array_rejects_out_of_range_element() {
        // An element exceeding u32::MAX must error, not silently truncate.
        let v = json!({"c": [1, 4294967296u64]});
        match stats_path(&v, "c").as_u32_array() {
            Err(SnapshotError::TypeMismatch {
                expected, actual, ..
            }) => {
                assert_eq!(expected, "u32");
                assert_eq!(actual, "Uint(>u32::MAX)");
            }
            other => panic!("expected u32 out-of-range TypeMismatch, got {other:?}"),
        }
    }

    #[test]
    fn json_field_typed_array_on_nonarray_value_errors_with_element_type() {
        // The "expected" diagnostic names the element type in brackets.
        let v = json!({"scalar": 7});
        match stats_path(&v, "scalar").as_u64_array() {
            Err(SnapshotError::TypeMismatch {
                expected, actual, ..
            }) => {
                assert_eq!(expected, "[u64]");
                assert_eq!(actual, "Number");
            }
            other => panic!("expected non-array TypeMismatch, got {other:?}"),
        }
    }

    #[test]
    fn json_field_typed_array_propagates_missing_error() {
        let v = json!({"x": 1});
        assert!(matches!(
            stats_path(&v, "absent").as_u64_array(),
            Err(SnapshotError::FieldNotFound { .. })
        ));
    }
}
