// An unrecognized constructor inside #[default_check(...)] must
// fail to compile — the macro prepends
// `::ktstr::test_support::MetricCheck::` so a typo like
// `nonexistent_check(...)` resolves to
// `::ktstr::test_support::MetricCheck::nonexistent_check(...)`, which
// the `MetricCheck` enum has no such constructor for, so rustc emits
// E0599 "no variant or associated item named `nonexistent_check` found
// for enum `MetricCheck`" at the generated `Payload::new` call site.
// The qualified `MetricCheck::`-prefixed form is exercised separately in
// `derive_payload_unknown_check_qualified.rs`.
use ktstr::Payload;

#[derive(Payload)]
#[payload(binary = "bad_check_bin")]
#[default_check(nonexistent_check("metric", 1.0))]
#[allow(dead_code)]
struct BadCheckPayload;

fn main() {}
