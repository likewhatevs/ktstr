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
                out.push_str(&lit.to_string());
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
