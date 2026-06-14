//! `json!` macro implementation: converts JSON-like Rust token trees
//! into a compile-time string. The `#[proc_macro] json` entry point
//! lives in `lib.rs` (Rust requires proc-macro fns at the crate root);
//! this module holds the token-stream-to-string conversion it calls.

pub(crate) fn tokens_to_json(out: &mut String, tokens: proc_macro2::TokenStream) {
    for tt in tokens {
        match tt {
            proc_macro2::TokenTree::Group(g) => match g.delimiter() {
                proc_macro2::Delimiter::Brace => {
                    out.push('{');
                    emit_comma_separated(out, g.stream(), true);
                    out.push('}');
                }
                proc_macro2::Delimiter::Bracket => {
                    out.push('[');
                    emit_comma_separated(out, g.stream(), false);
                    out.push(']');
                }
                proc_macro2::Delimiter::Parenthesis => {
                    out.push('(');
                    tokens_to_json(out, g.stream());
                    out.push(')');
                }
                proc_macro2::Delimiter::None => {
                    tokens_to_json(out, g.stream());
                }
            },
            proc_macro2::TokenTree::Literal(lit) => {
                // Canonicalize to valid JSON rather than emitting the
                // verbatim Rust source. `lit.to_string()` passes through
                // type suffixes (`1u32`), digit separators (`1_000`),
                // brace unicode escapes (`"\u{41}"`), and raw strings
                // (`r"x"`) — all invalid JSON that would surface only as
                // an opaque guest-side config-parse error.
                match syn::Lit::new(lit.clone()) {
                    syn::Lit::Str(s) => json_escape_str(out, &s.value()),
                    syn::Lit::Int(i) => out.push_str(i.base10_digits()),
                    syn::Lit::Float(f) => {
                        // base10_digits() preserves a trailing dot ("1."
                        // for `1.`), which is invalid JSON (RFC 8259
                        // requires a digit after the decimal point);
                        // normalize to a trailing "0".
                        let d = f.base10_digits();
                        out.push_str(d);
                        if d.ends_with('.') {
                            out.push('0');
                        }
                    }
                    syn::Lit::Bool(b) => {
                        out.push_str(if b.value { "true" } else { "false" })
                    }
                    // Byte strings, c-strings, byte/char literals, and
                    // verbatim tokens are not JSON values; emit verbatim
                    // so a malformed result surfaces at parse time as a
                    // clear user error rather than being silently coerced.
                    _ => out.push_str(&lit.to_string()),
                }
            }
            proc_macro2::TokenTree::Ident(id) => {
                let s = id.to_string();
                match s.as_str() {
                    "true" | "false" | "null" => out.push_str(&s),
                    _ => {
                        out.push('"');
                        out.push_str(&s);
                        out.push('"');
                    }
                }
            }
            proc_macro2::TokenTree::Punct(p) => {
                let ch = p.as_char();
                if ch == '-' {
                    out.push('-');
                } else if ch == ':' {
                    out.push(':');
                } else if ch == ',' {
                    out.push(',');
                } else {
                    out.push(ch);
                }
            }
        }
    }
}

fn emit_comma_separated(out: &mut String, tokens: proc_macro2::TokenStream, _is_object: bool) {
    let items = split_on_commas(tokens);
    let mut first = true;
    for item in &items {
        if item.is_empty() {
            continue;
        }
        if !first {
            out.push(',');
        }
        first = false;
        tokens_to_json(out, item.clone());
    }
}

fn split_on_commas(tokens: proc_macro2::TokenStream) -> Vec<proc_macro2::TokenStream> {
    let mut result = Vec::new();
    let mut current = proc_macro2::TokenStream::new();
    for tt in tokens {
        match &tt {
            proc_macro2::TokenTree::Punct(p) if p.as_char() == ',' => {
                result.push(current);
                current = proc_macro2::TokenStream::new();
            }
            _ => {
                current.extend(std::iter::once(tt));
            }
        }
    }
    if !current.is_empty() {
        result.push(current);
    }
    result
}

/// Append `s` to `out` as a quoted, escaped JSON string literal.
/// Escapes `"`, `\`, the standard short escapes, and other control
/// characters as `\u00XX` per RFC 8259. Non-control, non-quote, and
/// non-backslash characters (including non-ASCII UTF-8) pass through —
/// JSON strings carry UTF-8 directly.
fn json_escape_str(out: &mut String, s: &str) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => {
                out.push_str(&format!("\\u{:04x}", c as u32));
            }
            c => out.push(c),
        }
    }
    out.push('"');
}
