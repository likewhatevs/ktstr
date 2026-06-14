//! Helpers shared across more than one macro implementation.

/// Return the last segment's identifier of a path, or `None` for an
/// empty path. Used by helpers that match on a path's tail name
/// (`Some`, `BTreeSet`, etc.) without forcing the caller to import
/// the full path.
pub(crate) fn path_last_segment_ident(path: &syn::Path) -> Option<&syn::Ident> {
    path.segments.last().map(|s| &s.ident)
}
