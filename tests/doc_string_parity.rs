//! Pin doc-quoted runtime emit strings against their source. A doc
//! that quotes a tracing emit verbatim must drift when the emit drifts,
//! not silently mismatch — otherwise the operator follows
//! out-of-date troubleshooting advice and hits a different error than
//! the one the doc claims.

const TROUBLESHOOTING_MD: &str =
    include_str!("../doc/guide/src/troubleshooting.md");

const RUST_INIT_RS: &str =
    include_str!("../src/vmm/rust_init.rs");

/// The troubleshooting.md "send_sys_rdy timeout" section quotes the
/// WARN emitted by src/vmm/rust_init.rs when the budget runs out. The
/// format string `"ktstr-init: send_sys_rdy retry budget exhausted
/// ({} ms, {} vCPUs)"` has three pinnable substrings: the prefix
/// before the placeholders, the ` ms,` unit + separator between the
/// two formatted values, and the closing ` vCPUs)`. All three must
/// appear on the SAME doc line so the example renders as a single
/// real WARN line — a future refactor that deletes the WARN block
/// but leaves the surrounding prose (which also mentions ` ms,` and
/// ` vCPUs)` in placeholder form) would still pass substring-anywhere
/// checks; the co-location pin closes that gap.
#[test]
fn troubleshooting_send_sys_rdy_doc_matches_emit_fmt() {
    // Source-side check: the format string literal still lives at the
    // emit site. If a refactor moves or rewords it the test trips
    // here first, before drifting against the doc check below.
    assert!(
        RUST_INIT_RS
            .contains("ktstr-init: send_sys_rdy retry budget exhausted ({} ms, {} vCPUs)"),
        "src/vmm/rust_init.rs must still emit the exact WARN format \
         that troubleshooting.md quotes; if you rewrote the WARN, \
         update doc/guide/src/troubleshooting.md and this test in the \
         same change",
    );

    // Doc-side co-located pin: at least one line in troubleshooting.md
    // must carry all three substrings together. This is stronger than
    // independent substring-anywhere checks because ` ms,` and
    // ` vCPUs)` also appear in the placeholder-explanation prose
    // ("`(NNNNN ms, V vCPUs)`") — a regression that deleted the WARN
    // block but kept the prose would still pass `contains` on each
    // substring individually.
    let warn_line = TROUBLESHOOTING_MD.lines().find(|line| {
        line.contains("ktstr-init: send_sys_rdy retry budget exhausted")
            && line.contains(" ms,")
            && line.contains(" vCPUs)")
    });
    assert!(
        warn_line.is_some(),
        "troubleshooting.md must carry the full WARN-line example on a \
         single line containing the prefix `ktstr-init: send_sys_rdy \
         retry budget exhausted`, the ` ms,` unit-separator, and the \
         closing ` vCPUs)` — together these pin a real rendering of \
         the format string at src/vmm/rust_init.rs:684",
    );
}
