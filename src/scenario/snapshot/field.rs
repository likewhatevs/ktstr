//! [`SnapshotField`] — terminal accessor for a typed read out of a
//! rendered `crate::monitor::btf_render::RenderedValue` tree, plus
//! the lazy traversal helpers ([`walk_dotted_path`],
//! [`lookup_member`], [`peel_pointer`], [`describe_kind`]) and the
//! [`render_to_u64`] / [`render_to_i64`] coercion paths the
//! accessors funnel through.

use crate::monitor::btf_render::{RenderedMember, RenderedValue};

use super::{SnapshotError, SnapshotResult};

/// One field's view at the leaf of a dotted-path walk.
///
/// Returned by [`super::Snapshot::var`], [`super::SnapshotEntry::get`], and
/// [`super::SnapshotEntry::key`]. Terminal `as_*` accessors return
/// [`SnapshotResult`] so a missing or type-mismatched field
/// surfaces as a recoverable error rather than a panic.
#[derive(Debug)]
#[must_use = "SnapshotField is a borrowed view; call as_u64 / as_i64 / etc. to extract"]
#[non_exhaustive]
pub enum SnapshotField<'a> {
    /// Resolved rendered value at the leaf of the path walk.
    Value(&'a RenderedValue),
    /// Dedicated per-CPU array key shape (u32, no struct).
    PercpuKey { key: u32 },
    /// Path could not be resolved.
    Missing(SnapshotError),
}

impl<'a> SnapshotField<'a> {
    /// Walk into a sub-field. Composable with
    /// [`super::SnapshotEntry::get`].
    pub fn get(&self, path: &str) -> SnapshotField<'a> {
        match self {
            SnapshotField::Value(v) => walk_dotted_path(v, path),
            SnapshotField::PercpuKey { .. } => {
                SnapshotField::Missing(SnapshotError::TypeMismatch {
                    expected: "Struct".to_string(),
                    actual: "Uint(percpu key)".to_string(),
                    requested: path.to_string(),
                })
            }
            SnapshotField::Missing(err) => SnapshotField::Missing(err.clone()),
        }
    }

    /// True when the field resolved successfully.
    pub fn is_present(&self) -> bool {
        !matches!(self, SnapshotField::Missing(_))
    }

    /// Read as `u64`. Accepts [`RenderedValue::Uint`],
    /// [`RenderedValue::Int`] (errors on negative),
    /// [`RenderedValue::Bool`] (0/1), [`RenderedValue::Char`]
    /// (raw byte), [`RenderedValue::Enum`] (raw enum integer),
    /// [`RenderedValue::Ptr`] (pointer value), and the
    /// percpu-array u32 key.
    pub fn as_u64(&self) -> SnapshotResult<u64> {
        match self {
            SnapshotField::Value(v) => render_to_u64(v),
            SnapshotField::PercpuKey { key } => Ok(u64::from(*key)),
            SnapshotField::Missing(err) => Err(err.clone()),
        }
    }

    /// Read as `i64`.
    pub fn as_i64(&self) -> SnapshotResult<i64> {
        match self {
            SnapshotField::Value(v) => render_to_i64(v),
            SnapshotField::PercpuKey { key } => Ok(i64::from(*key)),
            SnapshotField::Missing(err) => Err(err.clone()),
        }
    }

    /// Read as `bool`. [`RenderedValue::Bool`] direct, ints / enums
    /// non-zero is true.
    pub fn as_bool(&self) -> SnapshotResult<bool> {
        match self {
            SnapshotField::Value(v) => match v {
                RenderedValue::Bool { value } => Ok(*value),
                RenderedValue::Int { value, .. } => Ok(*value != 0),
                RenderedValue::Uint { value, .. } => Ok(*value != 0),
                RenderedValue::Char { value } => Ok(*value != 0),
                RenderedValue::Enum { value, .. } => Ok(*value != 0),
                RenderedValue::Ptr { value, .. } => Ok(*value != 0),
                other => Err(SnapshotError::TypeMismatch {
                    expected: "bool".to_string(),
                    actual: describe_kind(other),
                    requested: String::new(),
                }),
            },
            SnapshotField::PercpuKey { key } => Ok(*key != 0),
            SnapshotField::Missing(err) => Err(err.clone()),
        }
    }

    /// Read as `f64`.
    pub fn as_f64(&self) -> SnapshotResult<f64> {
        match self {
            SnapshotField::Value(v) => match v {
                RenderedValue::Float { value, .. } => Ok(*value),
                RenderedValue::Int { value, .. } => Ok(*value as f64),
                RenderedValue::Uint { value, .. } => Ok(*value as f64),
                RenderedValue::Enum { value, .. } => Ok(*value as f64),
                other => Err(SnapshotError::TypeMismatch {
                    expected: "f64".to_string(),
                    actual: describe_kind(other),
                    requested: String::new(),
                }),
            },
            SnapshotField::PercpuKey { key } => Ok(f64::from(*key)),
            SnapshotField::Missing(err) => Err(err.clone()),
        }
    }

    /// Read the variant string for an [`RenderedValue::Enum`] with
    /// a resolved variant name.
    pub fn as_str(&self) -> SnapshotResult<&'a str> {
        match self {
            SnapshotField::Value(v) => match v {
                RenderedValue::Enum {
                    variant: Some(name),
                    ..
                } => Ok(name.as_str()),
                other => Err(SnapshotError::TypeMismatch {
                    expected: "str (enum variant name)".to_string(),
                    actual: describe_kind(other),
                    requested: String::new(),
                }),
            },
            SnapshotField::PercpuKey { .. } => Err(SnapshotError::TypeMismatch {
                expected: "str".to_string(),
                actual: "Uint(percpu key)".to_string(),
                requested: String::new(),
            }),
            SnapshotField::Missing(err) => Err(err.clone()),
        }
    }

    /// Read as `Vec<u64>` from an [`RenderedValue::Array`] whose
    /// every element coerces via [`Self::as_u64`]'s rules. Errors
    /// with [`SnapshotError::TypeMismatch`] when self is not an
    /// array, or when any element fails the coercion (no partial
    /// results — the caller cannot tell which element silently
    /// dropped). Mirrors [`RenderedValue::as_u64_array`] but
    /// propagates the captured [`SnapshotError`] through the
    /// [`SnapshotField::Missing`] arm.
    pub fn as_u64_array(&self) -> SnapshotResult<Vec<u64>> {
        render_to_typed_array(self, RenderedValue::as_u64, "u64")
    }

    /// Read as `Vec<u32>` from an array. Mirrors
    /// [`RenderedValue::as_u32_array`]; out-of-range values
    /// (Uint exceeding `u32::MAX`) error rather than truncate.
    pub fn as_u32_array(&self) -> SnapshotResult<Vec<u32>> {
        render_to_typed_array(self, |v| v.as_u64().and_then(|x| u32::try_from(x).ok()), "u32")
    }

    /// Read as `Vec<i64>` from an array. Mirrors
    /// [`RenderedValue::as_i64_array`].
    pub fn as_i64_array(&self) -> SnapshotResult<Vec<i64>> {
        render_to_typed_array(self, RenderedValue::as_i64, "i64")
    }

    /// Read as `Vec<f64>` from an array. Mirrors
    /// [`RenderedValue::as_f64_array`].
    pub fn as_f64_array(&self) -> SnapshotResult<Vec<f64>> {
        render_to_typed_array(self, RenderedValue::as_f64, "f64")
    }

    /// Read as `Vec<bool>` from an array. Mirrors
    /// [`RenderedValue::as_bool_array`].
    pub fn as_bool_array(&self) -> SnapshotResult<Vec<bool>> {
        render_to_typed_array(self, RenderedValue::as_bool, "bool")
    }

    /// Drop into the raw [`RenderedValue`] for direct
    /// [`RenderedValue::member`] / [`RenderedValue::get`] /
    /// [`RenderedValue::index`] navigation. Use when the
    /// pattern-matched-into-known-shape access pattern (Option-
    /// returning terminals, no rich error context) reads more
    /// naturally than the SnapshotField's Result-propagating
    /// chain. `None` for [`SnapshotField::PercpuKey`] (no
    /// underlying tree) and [`SnapshotField::Missing`].
    pub fn raw(&self) -> Option<&'a RenderedValue> {
        match self {
            SnapshotField::Value(v) => Some(v),
            _ => None,
        }
    }

    /// Error reference when the field is missing; `None`
    /// otherwise.
    pub fn error(&self) -> Option<&SnapshotError> {
        match self {
            SnapshotField::Missing(err) => Some(err),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// JSON dotted-path accessor (mirrors SnapshotField for stats values)
// ---------------------------------------------------------------------------

/// Walk a [`RenderedValue`] tree along a dotted path. Each
/// component matches a [`RenderedMember::name`] inside a
/// [`RenderedValue::Struct`]; [`RenderedValue::Ptr`] dereferences
/// are followed transparently. An empty path returns the root.
pub(crate) fn walk_dotted_path<'a>(root: &'a RenderedValue, path: &str) -> SnapshotField<'a> {
    if path.is_empty() {
        return SnapshotField::Value(root);
    }
    let mut cursor: &RenderedValue = root;
    let mut walked = String::new();
    for component in path.split('.') {
        if component.is_empty() {
            return SnapshotField::Missing(SnapshotError::EmptyPathComponent {
                requested: path.to_string(),
            });
        }
        cursor = peel_pointer(cursor);
        let RenderedValue::Struct { members, .. } = cursor else {
            return SnapshotField::Missing(SnapshotError::NotAStruct {
                requested: path.to_string(),
                walked: walked.clone(),
                component: component.to_string(),
                kind: describe_kind(cursor),
            });
        };
        let next = members.iter().find(|m| m.name == component);
        let Some(member) = next else {
            let names: Vec<String> = members.iter().map(|m| m.name.clone()).collect();
            return SnapshotField::Missing(SnapshotError::FieldNotFound {
                requested: path.to_string(),
                walked: walked.clone(),
                component: component.to_string(),
                available: names,
            });
        };
        cursor = &member.value;
        if !walked.is_empty() {
            walked.push('.');
        }
        walked.push_str(component);
    }
    SnapshotField::Value(cursor)
}

/// Look up a single top-level member by exact name. Used by
/// [`super::Snapshot::var`].
pub(super) fn lookup_member<'a>(value: &'a RenderedValue, name: &str) -> Option<&'a RenderedValue> {
    let v = peel_pointer(value);
    let RenderedValue::Struct { members, .. } = v else {
        return None;
    };
    members
        .iter()
        .find(|m: &&RenderedMember| m.name == name)
        .map(|m| &m.value)
}

/// Peel through any [`RenderedValue::Ptr`] layers whose `deref`
/// is `Some`. Stops at the first non-pointer (or a pointer
/// without a chased deref).
fn peel_pointer(mut v: &RenderedValue) -> &RenderedValue {
    let mut steps = 0;
    while let RenderedValue::Ptr {
        deref: Some(inner), ..
    } = v
    {
        v = inner.as_ref();
        steps += 1;
        if steps > 16 {
            break;
        }
    }
    v
}

/// Human-readable variant name used in error messages.
fn describe_kind(v: &RenderedValue) -> String {
    match v {
        RenderedValue::Int { .. } => "Int",
        RenderedValue::Uint { .. } => "Uint",
        RenderedValue::Bool { .. } => "Bool",
        RenderedValue::Char { .. } => "Char",
        RenderedValue::Float { .. } => "Float",
        RenderedValue::Enum { .. } => "Enum",
        RenderedValue::Struct { .. } => "Struct",
        RenderedValue::Array { .. } => "Array",
        RenderedValue::CpuList { .. } => "CpuList",
        RenderedValue::Ptr { .. } => "Ptr",
        RenderedValue::Bytes { .. } => "Bytes",
        RenderedValue::Truncated { .. } => "Truncated",
        RenderedValue::Unsupported { .. } => "Unsupported",
    }
    .to_string()
}

/// Shared typed-array coercion used by [`SnapshotField::as_u64_array`]
/// and siblings. `coerce` is the per-element scalar extractor that
/// returns `None` when the element fails the coercion (matches the
/// [`RenderedValue`] inherent `.as_*` Option-returning shape).
/// `type_name` names the requested element type for diagnostics.
fn render_to_typed_array<T, F>(
    field: &SnapshotField<'_>,
    coerce: F,
    type_name: &'static str,
) -> SnapshotResult<Vec<T>>
where
    F: Fn(&RenderedValue) -> Option<T>,
{
    let value = match field {
        SnapshotField::Value(v) => *v,
        SnapshotField::PercpuKey { .. } => {
            return Err(SnapshotError::TypeMismatch {
                expected: format!("[{type_name}]"),
                actual: "Uint(percpu key)".to_string(),
                requested: String::new(),
            });
        }
        SnapshotField::Missing(err) => return Err(err.clone()),
    };
    let elements = match value {
        RenderedValue::Array { elements, .. } => elements.as_slice(),
        RenderedValue::Ptr {
            deref: Some(inner), ..
        } => match inner.as_ref() {
            RenderedValue::Array { elements, .. } => elements.as_slice(),
            other => {
                return Err(SnapshotError::TypeMismatch {
                    expected: format!("[{type_name}]"),
                    actual: describe_kind(other),
                    requested: String::new(),
                });
            }
        },
        RenderedValue::Truncated { partial, .. } => match partial.as_ref() {
            RenderedValue::Array { elements, .. } => elements.as_slice(),
            other => {
                return Err(SnapshotError::TypeMismatch {
                    expected: format!("[{type_name}]"),
                    actual: format!("Truncated(partial={})", describe_kind(other)),
                    requested: String::new(),
                });
            }
        },
        other => {
            return Err(SnapshotError::TypeMismatch {
                expected: format!("[{type_name}]"),
                actual: describe_kind(other),
                requested: String::new(),
            });
        }
    };
    let mut out = Vec::with_capacity(elements.len());
    for (i, element) in elements.iter().enumerate() {
        let v = coerce(element).ok_or_else(|| SnapshotError::TypeMismatch {
            expected: format!("[{type_name}]"),
            actual: format!("{}[{i}]={}", "Array", describe_kind(element)),
            requested: String::new(),
        })?;
        out.push(v);
    }
    Ok(out)
}

/// Shared u64 coercion used by [`SnapshotField::as_u64`].
fn render_to_u64(v: &RenderedValue) -> SnapshotResult<u64> {
    match v {
        RenderedValue::Uint { value, .. } => Ok(*value),
        RenderedValue::Int { value, .. } => {
            if *value < 0 {
                Err(SnapshotError::TypeMismatch {
                    expected: "u64".to_string(),
                    actual: "Int(negative)".to_string(),
                    requested: String::new(),
                })
            } else {
                Ok(*value as u64)
            }
        }
        RenderedValue::Bool { value } => Ok(u64::from(*value)),
        RenderedValue::Char { value } => Ok(u64::from(*value)),
        RenderedValue::Enum { value, .. } => {
            if *value < 0 {
                Err(SnapshotError::TypeMismatch {
                    expected: "u64".to_string(),
                    actual: "Enum(negative)".to_string(),
                    requested: String::new(),
                })
            } else {
                Ok(*value as u64)
            }
        }
        RenderedValue::Ptr { value, .. } => Ok(*value),
        other => Err(SnapshotError::TypeMismatch {
            expected: "u64".to_string(),
            actual: describe_kind(other),
            requested: String::new(),
        }),
    }
}

/// Shared i64 coercion used by [`SnapshotField::as_i64`].
fn render_to_i64(v: &RenderedValue) -> SnapshotResult<i64> {
    match v {
        RenderedValue::Int { value, .. } => Ok(*value),
        RenderedValue::Uint { value, .. } => {
            if *value > i64::MAX as u64 {
                Err(SnapshotError::TypeMismatch {
                    expected: "i64".to_string(),
                    actual: "Uint(>i64::MAX)".to_string(),
                    requested: String::new(),
                })
            } else {
                Ok(*value as i64)
            }
        }
        RenderedValue::Bool { value } => Ok(i64::from(*value)),
        RenderedValue::Char { value } => Ok(i64::from(*value)),
        RenderedValue::Enum { value, .. } => Ok(*value),
        other => Err(SnapshotError::TypeMismatch {
            expected: "i64".to_string(),
            actual: describe_kind(other),
            requested: String::new(),
        }),
    }
}
