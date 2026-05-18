// Pin: `rq.nr_running` is intentionally absent from the v1
// `WriteableField` allow-list. A regression that added a
// `NrRunning` variant would re-introduce the silent-scheduler-
// state-corruption class — writing
// nr_running corrupts the runnable counter and surfaces as
// wrong load-balancing on every subsequent dispatch.
//
// The compile_fail pin is strictly stronger than a runtime test:
// it eliminates the possibility of any code path constructing
// a forbidden write at all, rather than only checking that
// apply_op rejects it at runtime.
//
// v1 allow-list:
//   - RqClock
//   - RqClockTask
//   - RqScxClock
//   - RqScxFlags
//   - Jiffies64
//
// `NrRunning` is NOT in the list and never will be in v1.

fn main() {
    let _ = ktstr::scenario::ops::WriteableField::NrRunning;
}
