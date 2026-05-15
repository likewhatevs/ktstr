//! Pin doc-quoted runtime emit strings against their source. A doc
//! that quotes a tracing emit verbatim must drift when the emit drifts,
//! not silently mismatch — otherwise the operator follows
//! out-of-date troubleshooting advice and hits a different error than
//! the one the doc claims.

const TROUBLESHOOTING_MD: &str = include_str!("../doc/guide/src/troubleshooting.md");

const RUST_INIT_RS: &str = include_str!("../src/vmm/rust_init.rs");

/// The troubleshooting.md "send_sys_rdy timeout" section quotes the
/// WARN emitted by src/vmm/rust_init.rs when the budget runs out. The
/// format string `"ktstr-init: send_sys_rdy retry budget exhausted
/// ({} ms, {} vCPUs); see doc/guide/src/troubleshooting.md#send_sys_rdy-timeout
/// for tuning"` has four pinnable substrings: the prefix before the
/// placeholders, the ` ms,` unit + separator between the two
/// formatted values, the closing ` vCPUs);`, and the docs pointer
/// `troubleshooting.md#send_sys_rdy-timeout` that directs the
/// operator from the bare WARN to the tuning prose. All four must
/// appear on the SAME doc line so the example renders as a single
/// real WARN line — a future refactor that deleted the WARN block
/// but left the surrounding prose (which also mentions ` ms,` and
/// ` vCPUs)` in placeholder form) would still pass substring-
/// anywhere checks; the co-location pin closes that gap. The docs
/// pointer substring also fails the test if the WARN drops the
/// hint, so the operator-facing direction can't silently regress.
#[test]
fn troubleshooting_send_sys_rdy_doc_matches_emit_fmt() {
    // Source-side check: the format string literal still lives at the
    // emit site. If a refactor moves or rewords it the test trips
    // here first, before drifting against the doc check below.
    //
    // The source-side format string is split across two source lines
    // via Rust's `\<newline>` continuation. `include_str!` returns the
    // raw source bytes (preserving the line break + indent), while a
    // literal here is collapsed by the compiler before the `contains`
    // call — so we cannot put the full WARN in one `contains` literal.
    // Pin the two source-side halves separately, each as a substring
    // that appears verbatim in the raw source bytes.
    assert!(
        RUST_INIT_RS.contains("ktstr-init: send_sys_rdy retry budget exhausted ({} ms, {} vCPUs);"),
        "src/vmm/rust_init.rs must still emit the WARN prefix \
         `ktstr-init: send_sys_rdy retry budget exhausted ({{}} ms, {{}} vCPUs);` \
         that troubleshooting.md quotes; if you rewrote the WARN, \
         update doc/guide/src/troubleshooting.md and this test in the \
         same change",
    );
    assert!(
        RUST_INIT_RS
            .contains("see doc/guide/src/troubleshooting.md#send_sys_rdy-timeout for tuning"),
        "src/vmm/rust_init.rs must still emit the docs-pointer suffix \
         `see doc/guide/src/troubleshooting.md#send_sys_rdy-timeout for tuning` \
         after the budget prefix; if you rewrote the WARN, update \
         doc/guide/src/troubleshooting.md and this test in the same \
         change",
    );

    // Doc-side co-located pin: at least one line in troubleshooting.md
    // must carry all four substrings together. This is stronger than
    // independent substring-anywhere checks because ` ms,` and
    // ` vCPUs)` also appear in the placeholder-explanation prose
    // ("`(NNNNN ms, V vCPUs)`") — a regression that deleted the WARN
    // block but kept the prose would still pass `contains` on each
    // substring individually. The docs-pointer substring rules out
    // a regression that dropped the tuning hint while keeping the
    // bare-WARN body.
    let warn_line = TROUBLESHOOTING_MD.lines().find(|line| {
        line.contains("ktstr-init: send_sys_rdy retry budget exhausted")
            && line.contains(" ms,")
            && line.contains(" vCPUs);")
            && line.contains("troubleshooting.md#send_sys_rdy-timeout")
    });
    assert!(
        warn_line.is_some(),
        "troubleshooting.md must carry the full WARN-line example on a \
         single line containing the prefix `ktstr-init: send_sys_rdy \
         retry budget exhausted`, the ` ms,` unit-separator, the \
         closing ` vCPUs);`, and the docs pointer \
         `troubleshooting.md#send_sys_rdy-timeout` — together these \
         pin a real rendering of the format string at \
         src/vmm/rust_init.rs:684",
    );
}
