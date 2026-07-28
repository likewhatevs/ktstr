//! Regression pin: failure-dump assertions must point at the persisted JSON
//! artifact instead of copying unbounded captured payloads onto test stdout.

use std::path::Path;

const ASSERTION_FILES: &[&str] = &[
    "tests/cast_analysis_e2e.rs",
    "tests/failure_dump_e2e.rs",
    "tests/silent_drop_e2e.rs",
];
const PRODUCTION_FAILURE_OUTPUT_FILES: &[&str] = &["src/test_support/probe.rs"];
const FORBIDDEN: &[&str] = &[
    "{any}",
    "{report}",
    "{value}",
    "{dump}",
    "{payload}",
    "{task_storage}",
    "{entry}",
    "{parent}",
    "{bss_value}",
    "{deref_struct}",
    "{holder_outer}",
    "{arena_target}",
    "{array_map}",
    "{val}",
    "Full JSON",
    "Full dump",
    "full payload",
    "payload: {value}",
    "{arena_map}",
    "{arena_field}",
    "page: {page}",
    "Full vcpu_regs",
    "Full rq_scx_states",
    "Full task_enrichments",
];
const PRODUCTION_FORBIDDEN: &[&str] = &[
    "FailureDumpReportAny::from_json(&json)",
    "write!(buf, \"{any}\")",
    "write!(buf, \"{report}\")",
];

#[test]
fn failure_dump_assertions_never_embed_unbounded_payloads() {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut violations = Vec::new();
    for (files, forbidden_tokens) in [
        (ASSERTION_FILES, FORBIDDEN),
        (PRODUCTION_FAILURE_OUTPUT_FILES, PRODUCTION_FORBIDDEN),
    ] {
        for relative in files {
            let path = manifest_dir.join(relative);
            let source = std::fs::read_to_string(&path)
                .unwrap_or_else(|error| panic!("read {}: {error}", path.display()));
            for (line_index, line) in source.lines().enumerate() {
                for forbidden in forbidden_tokens {
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
    }
    assert!(
        violations.is_empty(),
        "failure-dump diagnostics must stay bounded and reference \
         failure_dump_artifact(result); found:\n{}",
        violations.join("\n"),
    );
}
