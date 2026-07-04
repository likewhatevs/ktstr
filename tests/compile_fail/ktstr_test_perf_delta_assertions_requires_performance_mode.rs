use ktstr::ktstr_test;
use ktstr::test_support::PerfDeltaAssertion;

// A declared perf_delta_assertions without performance_mode — the gate tightens
// the perf-delta noise threshold on a metric, which is only meaningful on a
// pinned run. Isolates the perf_delta_assertions-requires-performance_mode gate.
#[allow(dead_code)]
const GATE: PerfDeltaAssertion = PerfDeltaAssertion::new("worst_spread");

#[ktstr_test(perf_delta_assertions = [GATE])]
fn bad(_ctx: &ktstr::scenario::Ctx) -> anyhow::Result<ktstr::assert::AssertResult> {
    Ok(ktstr::assert::AssertResult::pass())
}

fn main() {}
