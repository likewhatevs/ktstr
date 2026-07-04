//! Pre-parse argv partition so ktstr-owned flags on the
//! nextest-passthrough subcommands are position-independent.
//!
//! The test-running subcommands (`test` / `coverage` / `llvm-cov` /
//! `verifier` / `replay` / `perf-delta`) forward unrecognized trailing
//! args to an inner `cargo nextest` (or, for `perf-delta`, to both
//! sides' `cargo ktstr test`). The passthrough field is declared
//! `#[arg(last = true)]`, i.e. it receives only what follows a `--`.
//!
//! [`rewrite`] makes the ktstr flags position-independent WITHOUT
//! requiring the user to type `--`: it splits a passthrough
//! subcommand's args into ktstr-owned flags and passthrough tokens
//! BEFORE clap parses, using the subcommand's OWN clap arg specs as the
//! flag/arity oracle (so the table is never hand-maintained and cannot
//! drift from the flag definitions). The rewritten argv places every
//! ktstr flag first, then a single literal `--`, then the passthrough —
//! so clap parses the ktstr flags in any order and the passthrough
//! lands in the `last = true` field verbatim.
//!
//! The auto-derived oracle above covers a subcommand's LIVE flags. One
//! deliberately hand-maintained exception ([`rejected_flags`]): flags REMOVED
//! from a subcommand's native surface are routed into the ktstr bucket too, so
//! clap emits a clean top-level unknown-flag error instead of forwarding a
//! stale flag to the inner tool as a confusing nested error.
//!
//! A user-supplied `--` is honored as a hard passthrough boundary
//! (everything after it is passthrough, even a token spelled like a
//! ktstr flag). A token that SHARES a spelling with a ktstr flag —
//! nextest's own `--profile`, or `-E` on `replay`/`perf-delta` — is
//! treated as ktstr-owned; to forward such a token to the inner tool,
//! place it after `--`. That is the only case where `--` is still
//! required, and it is irreducible: two flags cannot share one spelling
//! without a boundary.
//!
//! Non-passthrough subcommands (`shell`, `stats`, `kernel`, ...) are
//! returned unchanged — they have no `last = true` field, so inserting
//! a `--` would make clap reject the trailing args.

use std::collections::HashMap;
use std::ffi::OsString;

use clap::ArgAction;

/// Rewrite full process `argv` (`[bin, "ktstr", <SUB>, tail..]`) so a
/// passthrough subcommand's ktstr-owned flags parse regardless of their
/// position relative to nextest passthrough args. Returns `argv`
/// unchanged when it does not target a passthrough subcommand.
///
/// `root` is the top-level `Cargo` clap command
/// (`<Cargo as clap::CommandFactory>::command()`); the ktstr-flag table
/// for the target subcommand is derived from it.
pub(crate) fn rewrite(root: &clap::Command, argv: &[OsString]) -> Vec<OsString> {
    // Need at least [bin, "ktstr", <SUB>].
    if argv.len() < 3 || argv[1].to_str() != Some("ktstr") {
        return argv.to_vec();
    }
    let sub_name = match argv[2].to_str() {
        Some(s) => s,
        None => return argv.to_vec(),
    };
    let ktstr_cmd = match root.get_subcommands().find(|c| sub_matches(c, "ktstr")) {
        Some(c) => c,
        None => return argv.to_vec(),
    };
    let sub = match ktstr_cmd
        .get_subcommands()
        .find(|c| sub_matches(c, sub_name))
    {
        Some(c) => c,
        None => return argv.to_vec(),
    };
    // Only subcommands with a `last = true` passthrough positional get
    // rewritten — those are exactly the nextest-forwarding verbs.
    if !sub.get_arguments().any(clap::Arg::is_last_set) {
        return argv.to_vec();
    }

    let (longs, shorts) = owned_flags(sub);
    // Key the denylist on the RESOLVED command's canonical name (sub was
    // matched by name-or-alias at line above), not the raw argv token, so an
    // aliased spelling still gets the rejected-flag treatment.
    let (ktstr_bucket, pass_bucket) =
        partition(&argv[3..], &longs, &shorts, rejected_flags(sub.get_name()));

    let mut out = Vec::with_capacity(argv.len() + 1);
    out.push(argv[0].clone());
    out.push(argv[1].clone());
    out.push(argv[2].clone());
    out.extend(ktstr_bucket);
    // Always emit the `--` so the `last = true` field receives the
    // passthrough (empty is fine). The ktstr flags precede it, so clap
    // parses them as native flags in any order.
    out.push(OsString::from("--"));
    out.extend(pass_bucket);
    out
}

/// True when clap command `c`'s name or any of its aliases equals
/// `name` (so `nextest` resolves to the `test` command).
fn sub_matches(c: &clap::Command, name: &str) -> bool {
    c.get_name() == name || c.get_all_aliases().any(|a| a == name)
}

/// True when the arg's action consumes a value (`Set` for a scalar
/// `Option<T>` / `T`, `Append` for a repeatable `Vec<T>`). Bool flags
/// (`SetTrue`/`SetFalse`), counters, and `--help`/`--version` do not.
fn action_takes_value(action: &ArgAction) -> bool {
    matches!(action, ArgAction::Set | ArgAction::Append)
}

/// Build the {long -> takes_value} and {short -> takes_value} tables for
/// one subcommand from its clap arg specs. The passthrough positional
/// (no long, no short) is naturally excluded. `--help`/`-h` are added
/// explicitly so a `cargo ktstr <sub> --help` is never forwarded to the
/// inner tool, even if the auto-help arg is not enumerated pre-build.
fn owned_flags(sub: &clap::Command) -> (HashMap<String, bool>, HashMap<char, bool>) {
    let mut longs: HashMap<String, bool> = HashMap::new();
    let mut shorts: HashMap<char, bool> = HashMap::new();
    for a in sub.get_arguments() {
        let takes = action_takes_value(a.get_action());
        if let Some(names) = a.get_long_and_visible_aliases() {
            for l in names {
                longs.insert(l.to_string(), takes);
            }
        }
        if let Some(chs) = a.get_short_and_visible_aliases() {
            for s in chs {
                shorts.insert(s, takes);
            }
        }
    }
    longs.entry("help".to_string()).or_insert(false);
    shorts.entry('h').or_insert(false);
    (longs, shorts)
}

/// Flags removed from a subcommand's native surface that users still reach for.
/// On a lookup miss [`partition`] routes these into the ktstr bucket (not the
/// nextest passthrough) so clap emits a clean top-level "unexpected argument"
/// error rather than forwarding them to the inner tool as a confusing nested
/// error. Names carry NO leading `--` (matching the `longs` keys). Empty for
/// every subcommand except `perf-delta`, whose per-side A/B axis
/// (`--a-scheduler` / `--b-scheduler` / `--a-topology` / `--b-topology`) was
/// removed; cross-config comparison now lives in-test in the Verdict DSL
/// (`better_across_phases`), so those flags no longer exist on the command.
fn rejected_flags(sub_name: &str) -> &'static [&'static str] {
    match sub_name {
        "perf-delta" => &["a-scheduler", "b-scheduler", "a-topology", "b-topology"],
        _ => &[],
    }
}

/// Split `tail` into (ktstr-owned tokens, passthrough tokens). See the
/// module doc for the rules; a literal `--` ends the scan (its
/// remainder is passthrough and the `--` itself is dropped, since
/// [`rewrite`] re-emits exactly one).
fn partition(
    tail: &[OsString],
    longs: &HashMap<String, bool>,
    shorts: &HashMap<char, bool>,
    rejected: &[&str],
) -> (Vec<OsString>, Vec<OsString>) {
    let mut ktstr: Vec<OsString> = Vec::new();
    let mut pass: Vec<OsString> = Vec::new();
    let mut i = 0;
    while i < tail.len() {
        let tok = &tail[i];
        match tok.to_str() {
            Some("--") => {
                pass.extend(tail[i + 1..].iter().cloned());
                break;
            }
            Some(t) if t.starts_with("--") => {
                // `--flag` or `--flag=value`.
                let name = &t[2..];
                let (flag, has_eq) = match name.split_once('=') {
                    Some((f, _)) => (f, true),
                    None => (name, false),
                };
                match longs.get(flag) {
                    Some(&takes_value) => {
                        ktstr.push(tok.clone());
                        if takes_value && !has_eq && i + 1 < tail.len() {
                            ktstr.push(tail[i + 1].clone());
                            i += 1;
                        }
                    }
                    // A flag REMOVED from this subcommand's native surface that
                    // the user still typed: route it into the ktstr bucket (not
                    // the nextest passthrough) so clap emits a clean top-level
                    // "unexpected argument" error instead of forwarding a stray
                    // --flag to the inner tool as a confusing nested error. A
                    // space-separated value (if any) stays in `pass`; clap errors
                    // on the flag first, so it never matters.
                    None if rejected.contains(&flag) => ktstr.push(tok.clone()),
                    None => pass.push(tok.clone()),
                }
            }
            Some(t) if t.starts_with('-') && t.len() > 1 => {
                // Short flag `-X` or glued `-Xvalue`. `-` alone (stdin
                // sentinel) falls through to the passthrough arm below.
                let first = t[1..].chars().next().expect("len > 1");
                match shorts.get(&first) {
                    Some(&takes_value) => {
                        ktstr.push(tok.clone());
                        // `-X value` (nothing glued) pulls the next token
                        // as the value; `-Xvalue` already carries it.
                        let glued_len = t.len() - 1 - first.len_utf8();
                        if takes_value && glued_len == 0 && i + 1 < tail.len() {
                            ktstr.push(tail[i + 1].clone());
                            i += 1;
                        }
                    }
                    None => pass.push(tok.clone()),
                }
            }
            // Bare positional, `-`, or non-UTF8 token: passthrough.
            _ => pass.push(tok.clone()),
        }
        i += 1;
    }
    (ktstr, pass)
}
