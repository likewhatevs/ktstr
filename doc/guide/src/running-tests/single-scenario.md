# Single Scenario

## Running a specific test

```sh
cargo ktstr test --kernel ../linux -- -E 'test(sched_basic_proportional)'
```

The `-E 'test(NAME)'` form is a nextest filterset expression. `test(NAME)`
is a substring match against the test name; use `test(=NAME)` for an exact
match. See nextest's filterset docs for the full grammar.

Integration-feature-gated tests (e.g. the jemalloc probe e2es) need
`--features integration`, placed after the `--` separator so it is
forwarded to the underlying `cargo nextest run`:

```sh
cargo ktstr test --kernel ../linux -- --features integration -E 'test(jemalloc_probe_external_target_observes_known_allocation)'
```

## Running with a panic backtrace

```sh
RUST_BACKTRACE=1 cargo ktstr test --kernel ../linux -- -E 'test(sched_basic_proportional)'
```

`RUST_BACKTRACE=1` controls Rust panic backtraces, not VM console
verbosity. To stream the guest kernel console (dmesg, scheduler logs),
use the `--dmesg` flag on `cargo ktstr shell`.

## Investigating failures

```sh
RUST_BACKTRACE=1 cargo ktstr test --kernel ../linux -- -E 'test(cover_cgroup_cpuset_cross_llc_race)'
```

## VM topology

Each `#[ktstr_test]` declares its topology via macro attributes:

```rust,ignore
#[ktstr_test(llcs = 2, cores = 4, threads = 2)]
```

The test framework boots a VM with the specified topology
automatically.

See [Investigate a Crash](../recipes/investigate-crash.md) for
interpreting failure output and
[Troubleshooting](../troubleshooting.md) for common error messages.
