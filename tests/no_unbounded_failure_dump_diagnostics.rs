//! Regression pin: failure-dump assertions must point at the persisted JSON
//! artifact instead of copying unbounded captured payloads onto test stdout.

use std::path::Path;

const ASSERTION_FILES: &[&str] = &["failure_dump_e2e.rs", "silent_drop_e2e.rs"];
const FORBIDDEN: &[&str] = &[
    "{value}",
    "Full JSON",
    "payload: {value}",
    "{arena_map}",
    "{arena_field}",
    "page: {page}",
    "Full vcpu_regs",
    "Full rq_scx_states",
    "Full task_enrichments",
];

#[test]
fn failure_dump_assertions_never_embed_unbounded_payloads() {
    let tests_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests");
    let mut violations = Vec::new();
    for relative in ASSERTION_FILES {
        let path = tests_dir.join(relative);
        let source = std::fs::read_to_string(&path)
            .unwrap_or_else(|error| panic!("read {}: {error}", path.display()));
        for (line_index, line) in source.lines().enumerate() {
            for forbidden in FORBIDDEN {
                if line.contains(forbidden) {
                    violations.push(format!(
                        "{relative}:{} contains {forbidden:?}: {}",
                        line_index + 1,
                        line.trim(),
                    ));
                }
            }
        }
    }
    assert!(
        violations.is_empty(),
        "failure-dump diagnostics must stay bounded and reference \
         failure_dump_artifact(result); found:\n{}",
        violations.join("\n"),
    );
}
