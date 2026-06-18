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
                    actual:
                        "Uint(percpu key) — call as_u64/as_i64/as_f64/as_bool for the key value"
                            .to_string(),
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
                actual: "Uint(percpu key) — call as_u64/as_i64/as_f64/as_bool for the key value"
                    .to_string(),
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
        render_to_typed_array(
            self,
            |v| v.as_u64().and_then(|x| u32::try_from(x).ok()),
            "u32",
        )
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

    /// Iterate the elements of an array-shaped field as
    /// [`SnapshotField`]s so chained navigation composes:
    /// `field.iter_members().filter_map(|el| el.get("name").as_u64().ok())`.
    /// Bridges the gap left by the scalar `as_*_array` terminals
    /// on array-of-struct shapes: those terminals coerce each
    /// element to a scalar via the shared coercion helper and
    /// return [`SnapshotError::TypeMismatch`] on the first
    /// non-scalar element, which is exactly what an array-of-struct
    /// triggers. `iter_members` instead hands the caller each raw
    /// element so they can chain `.get(field).as_u64()` per element.
    /// Peels [`RenderedValue::Ptr`] dereferences and
    /// [`RenderedValue::Truncated`] partial-array wrappers the
    /// same way [`Self::as_u64_array`] does.
    ///
    /// Yields nothing for non-array shapes, percpu-key fields, or
    /// missing fields — the empty iterator pattern is the natural
    /// "no elements to walk" representation when the chain just
    /// wants to fold over what's there. `iter_members` itself never
    /// surfaces [`SnapshotError::TypeMismatch`]; callers needing to
    /// distinguish "absent" from "empty" check [`Self::is_present`]
    /// or [`Self::error`] explicitly.
    pub fn iter_members(&self) -> impl Iterator<Item = SnapshotField<'a>> + '_ {
        let elements = match self {
            SnapshotField::Value(v) => array_elements_of(v),
            _ => &[],
        };
        elements.iter().map(SnapshotField::Value)
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

/// Shared array-elements walker: peel [`RenderedValue::Ptr`]'s
/// deref and [`RenderedValue::Truncated`]'s partial recursively
/// to reach an [`RenderedValue::Array`], returning the elements
/// slice on success. On a non-array variant (after peeling),
/// `Err` carries the unwrapped inner value so callers that want
/// a typed mismatch diagnostic can name the actual variant via
/// [`describe_kind`].
fn array_elements_or_mismatch(v: &RenderedValue) -> Result<&[RenderedValue], &RenderedValue> {
    match v {
        RenderedValue::Array { elements, .. } => Ok(elements.as_slice()),
        RenderedValue::Ptr {
            deref: Some(inner), ..
        } => array_elements_or_mismatch(inner.as_ref()),
        RenderedValue::Truncated { partial, .. } => array_elements_or_mismatch(partial.as_ref()),
        other => Err(other),
    }
}

/// Borrow the elements slice of an [`RenderedValue::Array`],
/// peeling [`RenderedValue::Ptr`]'s deref and
/// [`RenderedValue::Truncated`]'s partial. Returns the empty
/// slice for any non-array variant so the caller's iterator
/// chain yields no elements cleanly. Thin wrapper over
/// [`array_elements_or_mismatch`] that swallows the typed
/// mismatch — appropriate for [`SnapshotField::iter_members`]
/// whose empty-iterator contract distinguishes absent vs empty
/// via [`SnapshotField::is_present`] / [`SnapshotField::error`]
/// rather than via a returned error.
fn array_elements_of(v: &RenderedValue) -> &[RenderedValue] {
    array_elements_or_mismatch(v).unwrap_or(&[])
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
                actual: "Uint(percpu key) — call as_u64/as_i64/as_f64/as_bool for the key value"
                    .to_string(),
                requested: String::new(),
            });
        }
        SnapshotField::Missing(err) => return Err(err.clone()),
    };
    let elements = array_elements_or_mismatch(value).map_err(|other| {
        // Diagnostic wrapping mirrors the operator-facing form the
        // legacy code emitted for the common one-deep cases:
        //   - top-level `Truncated{partial: NonArray}` reports
        //     `Truncated(partial=<inner-kind>)` so the operator
        //     can tell the partial wrapper hid a non-array shape
        //     (vs the top level just not being an array).
        //   - all other paths (top-level non-array, top-level Ptr,
        //     and any deeper-nested wrapper combination) report
        //     the unwrapped leaf kind directly.
        // The shared walker recurses through Ptr+Truncated, so
        // arbitrary nesting around an Array now succeeds (matches
        // RenderedValue::array_elements semantics at
        // src/monitor/btf_render/mod.rs). On failure of a nested
        // shape (e.g. Ptr→Truncated→NonArray), the diagnostic
        // collapses to the leaf kind rather than narrating the
        // wrapper stack — sufficient context for the operator
        // since the wrapper structure is renderer-internal and
        // not load-bearing for assertions.
        let actual = match value {
            RenderedValue::Truncated { .. } => {
                format!("Truncated(partial={})", describe_kind(other))
            }
            _ => describe_kind(other),
        };
        SnapshotError::TypeMismatch {
            expected: format!("[{type_name}]"),
            actual,
            requested: String::new(),
        }
    })?;
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
        RenderedValue::Enum {
            value, is_signed, ..
        } => {
            // Mirror RenderedValue::as_u64's enum dispatch so the two
            // surfaces agree on signedness handling: unsigned enums
            // reinterpret the bit pattern (the renderer stores i64,
            // so an unsigned u64 wire value with the high bit set
            // arrives here as a negative i64 — `as u64` recovers the
            // bits); signed enums reject negative variants as
            // out-of-range. Without this branch, an unsigned 64-bit
            // enum at u64::MAX (stored as i64=-1) returned
            // TypeMismatch from this path while RenderedValue::as_u64
            // returned Some(u64::MAX) — same value, two surfaces
            // disagreed.
            if *is_signed && *value < 0 {
                Err(SnapshotError::TypeMismatch {
                    expected: "u64".to_string(),
                    actual: "Enum(signed-negative)".to_string(),
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

#[cfg(test)]
mod tests_coercion {
    //! Host-pure coverage for the [`SnapshotField`] coercion paths
    //! ([`render_to_u64`] / [`render_to_i64`], the inline `as_bool` /
    //! `as_f64` / `as_str` matches, the dotted-path walk, and the
    //! typed-array terminals). These decide whether a BTF-snapshot
    //! read SUCCEEDS or surfaces a typed [`SnapshotError`]; a swapped
    //! arm silently coerces a value the accessor should reject.
    use super::*;

    fn ptr(value: u64) -> RenderedValue {
        RenderedValue::Ptr {
            value,
            deref: None,
            deref_skipped_reason: None,
            cast_annotation: None,
        }
    }

    /// REGRESSION: the scalar `SnapshotField::as_bool` and the array
    /// `as_bool_array` (which coerces each element via
    /// `RenderedValue::as_bool`) must AGREE on a pointer. Before the
    /// `Ptr` arm landed on `RenderedValue::as_bool`, the scalar
    /// accepted a pointer (non-null test) while the array path errored
    /// on the first pointer element — `field.as_bool()` succeeded and
    /// `field.as_bool_array()` failed on the same shape.
    #[test]
    fn as_bool_scalar_and_array_agree_on_pointer() {
        let non_null = ptr(0x1000);
        let null = ptr(0);
        // Scalar: pointer coerces as a non-null test.
        assert!(SnapshotField::Value(&non_null).as_bool().unwrap());
        assert!(!SnapshotField::Value(&null).as_bool().unwrap());
        // Array: the same per-element coercion, no TypeMismatch.
        let arr = RenderedValue::Array {
            len: 2,
            elements: vec![ptr(0x2000), ptr(0)],
        };
        assert_eq!(
            SnapshotField::Value(&arr).as_bool_array().unwrap(),
            vec![true, false],
        );
    }

    /// `as_u64` accepts the pointer / char / bool scalar variants
    /// (pointer as its numeric address, char as the raw byte, bool as
    /// 0/1) — the wider integer-coercion set.
    #[test]
    fn as_u64_accepts_ptr_char_bool() {
        assert_eq!(SnapshotField::Value(&ptr(0xdead)).as_u64().unwrap(), 0xdead);
        let c = RenderedValue::Char { value: 65 };
        assert_eq!(SnapshotField::Value(&c).as_u64().unwrap(), 65);
        let b = RenderedValue::Bool { value: true };
        assert_eq!(SnapshotField::Value(&b).as_u64().unwrap(), 1);
    }

    /// `render_to_u64`'s enum arm mirrors `RenderedValue::as_u64`:
    /// an unsigned 64-bit enum at `u64::MAX` (stored as `i64 = -1`)
    /// reinterprets the bit pattern, while a signed-negative enum is
    /// rejected as out-of-range.
    #[test]
    fn as_u64_enum_signedness() {
        let unsigned_max = RenderedValue::Enum {
            bits: 64,
            value: -1,
            variant: None,
            is_signed: false,
        };
        assert_eq!(
            SnapshotField::Value(&unsigned_max).as_u64().unwrap(),
            u64::MAX,
        );
        let signed_neg = RenderedValue::Enum {
            bits: 32,
            value: -5,
            variant: None,
            is_signed: true,
        };
        assert!(matches!(
            SnapshotField::Value(&signed_neg).as_u64(),
            Err(SnapshotError::TypeMismatch { .. })
        ));
    }

    /// `as_u64` rejects a negative `Int`; `as_i64` rejects a `Uint`
    /// above `i64::MAX` — the sign-loss boundaries.
    #[test]
    fn integer_sign_boundaries_error() {
        let neg = RenderedValue::Int {
            bits: 32,
            value: -1,
        };
        assert!(matches!(
            SnapshotField::Value(&neg).as_u64(),
            Err(SnapshotError::TypeMismatch { .. })
        ));
        let big = RenderedValue::Uint {
            bits: 64,
            value: u64::MAX,
        };
        assert!(matches!(
            SnapshotField::Value(&big).as_i64(),
            Err(SnapshotError::TypeMismatch { .. })
        ));
    }

    /// `as_f64` is narrower than `as_u64`: it accepts Float / Int /
    /// Uint / Enum but rejects Char, Bool, and Ptr (a float of a
    /// pointer or a char is not meaningful).
    #[test]
    fn as_f64_rejects_char_bool_ptr() {
        let f = RenderedValue::Float {
            bits: 64,
            value: 1.5,
        };
        assert_eq!(SnapshotField::Value(&f).as_f64().unwrap(), 1.5);
        for v in [
            RenderedValue::Char { value: 1 },
            RenderedValue::Bool { value: true },
            ptr(0x10),
        ] {
            assert!(
                matches!(
                    SnapshotField::Value(&v).as_f64(),
                    Err(SnapshotError::TypeMismatch { .. })
                ),
                "as_f64 must reject {v:?}",
            );
        }
    }

    /// `as_str` reads an enum's resolved variant name and rejects
    /// non-enum / nameless-enum / percpu-key shapes.
    #[test]
    fn as_str_reads_enum_variant_else_errors() {
        let named = RenderedValue::Enum {
            bits: 32,
            value: 2,
            variant: Some("SCX_OPS_ENABLED".to_string()),
            is_signed: false,
        };
        assert_eq!(
            SnapshotField::Value(&named).as_str().unwrap(),
            "SCX_OPS_ENABLED"
        );
        let nameless = RenderedValue::Enum {
            bits: 32,
            value: 2,
            variant: None,
            is_signed: false,
        };
        assert!(SnapshotField::Value(&nameless).as_str().is_err());
        assert!(SnapshotField::PercpuKey { key: 3 }.as_str().is_err());
    }

    /// The dotted-path walk peels `Ptr{deref: Some}` to the pointed-at
    /// struct, resolves a member, and surfaces structured errors for a
    /// non-struct cursor and an empty path component.
    #[test]
    fn walk_dotted_path_peels_pointer_and_reports_errors() {
        let inner = RenderedValue::Struct {
            type_name: Some("scx_bss".to_string()),
            members: vec![RenderedMember {
                name: "stall".to_string(),
                value: RenderedValue::Uint { bits: 8, value: 1 },
            }],
        };
        let through_ptr = RenderedValue::Ptr {
            value: 0xffff_0000,
            deref: Some(Box::new(inner)),
            deref_skipped_reason: None,
            cast_annotation: None,
        };
        assert_eq!(
            SnapshotField::Value(&through_ptr)
                .get("stall")
                .as_u64()
                .unwrap(),
            1,
        );
        // Walking into a scalar surfaces NotAStruct.
        let scalar = RenderedValue::Uint { bits: 8, value: 5 };
        assert!(matches!(
            SnapshotField::Value(&scalar).get("x"),
            SnapshotField::Missing(SnapshotError::NotAStruct { .. })
        ));
        // An empty component (`a..b`) surfaces EmptyPathComponent.
        let s = RenderedValue::Struct {
            type_name: None,
            members: vec![RenderedMember {
                name: "a".to_string(),
                value: RenderedValue::Uint { bits: 8, value: 0 },
            }],
        };
        assert!(matches!(
            SnapshotField::Value(&s).get("a..b"),
            SnapshotField::Missing(SnapshotError::EmptyPathComponent { .. })
        ));
    }

    /// `get` on a percpu-key field surfaces a TypeMismatch naming the
    /// percpu-key shape (no struct to walk into); `is_present`/`raw`/
    /// `error` reflect each variant.
    #[test]
    fn percpu_key_navigation_and_view_helpers() {
        let pk = SnapshotField::PercpuKey { key: 7 };
        assert!(matches!(
            pk.get("x"),
            SnapshotField::Missing(SnapshotError::TypeMismatch { .. })
        ));
        assert!(pk.is_present());
        assert!(pk.raw().is_none());
        assert!(pk.error().is_none());
        assert_eq!(pk.as_u64().unwrap(), 7);

        let missing = SnapshotField::Missing(SnapshotError::EmptyPathComponent {
            requested: "x".to_string(),
        });
        assert!(!missing.is_present());
        assert!(missing.error().is_some());
        assert!(matches!(
            missing.as_u64(),
            Err(SnapshotError::EmptyPathComponent { .. })
        ));
    }

    /// `iter_members` peels `Truncated{partial: Array}` and yields the
    /// preserved elements; a non-array (after peeling) yields nothing.
    #[test]
    fn iter_members_peels_truncated_array_else_empty() {
        let truncated = RenderedValue::Truncated {
            needed: 32,
            had: 16,
            partial: Box::new(RenderedValue::Array {
                len: 4,
                elements: vec![
                    RenderedValue::Uint { bits: 8, value: 10 },
                    RenderedValue::Uint { bits: 8, value: 20 },
                ],
            }),
        };
        let got: Vec<u64> = SnapshotField::Value(&truncated)
            .iter_members()
            .map(|el| el.as_u64().unwrap())
            .collect();
        assert_eq!(got, vec![10, 20]);
        // A scalar yields no elements.
        let scalar = RenderedValue::Uint { bits: 8, value: 1 };
        assert_eq!(SnapshotField::Value(&scalar).iter_members().count(), 0);
    }
}
