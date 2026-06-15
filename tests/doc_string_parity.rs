//! Pin doc-quoted runtime emit strings against their source. A doc
//! that quotes a tracing emit verbatim must drift when the emit drifts,
//! not silently mismatch — otherwise the operator follows
//! out-of-date troubleshooting advice and hits a different error than
//! the one the doc claims.

const TROUBLESHOOTING_MD: &str = include_str!("../doc/guide/src/troubleshooting.md");

const TOPOLOGY_RS: &str = include_str!("../src/vmm/rust_init/topology.rs");

/// The troubleshooting.md "send_sys_rdy timeout" section documents
/// the WARN emitted by `send_sys_rdy_with_retry` when the budget
/// runs out. Pin the emit-site message, the docs anchor URL, and
/// every diagnostic field name on BOTH sides so a refactor that
/// changes either side without the other trips this test.
#[test]
fn troubleshooting_send_sys_rdy_doc_matches_emit_fmt() {
    // Source-side: the WARN must contain the message text and the
    // published docs URL.
    assert!(
        TOPOLOGY_RS.contains("send_sys_rdy failed within boot budget"),
        "src/vmm/rust_init/topology.rs must emit a WARN containing \
         `send_sys_rdy failed within boot budget`; if you rewrote \
         the WARN, update doc/guide/src/troubleshooting.md and this \
         test in the same change",
    );
    assert!(
        TOPOLOGY_RS.contains(
            "https://likewhatevs.github.io/ktstr/guide/troubleshooting.html#send_sys_rdy-timeout"
        ),
        "src/vmm/rust_init/topology.rs must emit the published docs URL \
         `https://likewhatevs.github.io/ktstr/guide/troubleshooting.html#send_sys_rdy-timeout`; \
         a repo-source path is not clickable from a VM dmesg",
    );

    // Source-side field-name pin: the WARN must emit each
    // diagnostic field that troubleshooting.md documents. Without
    // this, a refactor that drops a field from the macro but
    // leaves the doc unchanged passes the doc-side loop below.
    for field in [
        "port_exists",
        "kern_addrs_sent",
        "budget_ms",
        "elapsed_ms",
        "vcpus",
    ] {
        assert!(
            TOPOLOGY_RS.contains(field),
            "src/vmm/rust_init/topology.rs must emit the `{field}` structured \
             field in the send_sys_rdy WARN; troubleshooting.md \
             documents it",
        );
    }

    // Doc-side: the troubleshooting section must quote the WARN
    // message and document each diagnostic field the operator
    // needs to interpret the failure.
    assert!(
        TROUBLESHOOTING_MD.contains("send_sys_rdy failed within boot budget"),
        "troubleshooting.md must quote the WARN message \
         `send_sys_rdy failed within boot budget`",
    );
    assert!(
        TROUBLESHOOTING_MD.contains(
            "https://likewhatevs.github.io/ktstr/guide/troubleshooting.html#send_sys_rdy-timeout"
        ),
        "troubleshooting.md must show the published docs URL the \
         WARN emits, so an operator landing on this section can \
         confirm they're at the right page",
    );
    for field in ["port_exists", "kern_addrs_sent", "budget_ms", "elapsed_ms"] {
        assert!(
            TROUBLESHOOTING_MD.contains(field),
            "troubleshooting.md must document the `{field}` \
             diagnostic field from the send_sys_rdy WARN",
        );
    }
}
