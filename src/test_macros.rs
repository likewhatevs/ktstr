//! Test-only macros shared across the crate.
//!
//! Hoisted to crate root via `#[macro_use]` on the module declaration
//! in `lib.rs`, so `skip!` and `skip_on_contention!` are reachable from
//! any `#[cfg(test)]` code without an explicit `use`.

/// Emit a canonical `ktstr: SKIP: ...` message and return from the
/// caller. Routes through [`crate::report::test_skip`] so the
/// prefix lives in one place — the alternative (15+ open-coded
/// `eprintln!` sites) drifts into inconsistent casings that break
/// every grep-based test-summary tool.
///
/// Only callable from functions returning `()` — the macro expands to
/// an early `return;` with no value. Production code that returns a
/// non-unit type (dispatcher fns returning `i32`, helpers returning
/// `Option<T>`, loop bodies that `continue`) calls
/// [`crate::report::test_skip`] directly and drives its own control
/// flow.
macro_rules! skip {
    // Zero-args arm: `skip!()` emits the banner with an empty
    // reason. `format_args!()` itself requires at least a format
    // string, so the variadic arm below cannot handle this case
    // — a dedicated rule routes it to an empty literal.
    () => {{
        $crate::report::test_skip(format_args!(""));
        return;
    }};
    ($($arg:tt)*) => {{
        $crate::report::test_skip(format_args!($($arg)*));
        return;
    }};
}

/// Evaluate a `Result`-returning builder (or any `anyhow::Result`
/// expression) and either unwrap the value or skip gracefully on a
/// skip-class host-insufficiency error. Routes the error through the
/// shared `crate::test_support::classify_host_error` — the single source
/// of truth also used by `err_to_exit_code` and the `#[ktstr_test]` macro
/// body — so this helper can never drift from them. A `HostClass::Skip`
/// (no kernel resolved, topology insufficient, or perf-mode unavailable —
/// chain-aware, so a `.context(...)`-wrapped
/// instance still skips) emits
/// the canonical SKIP banner and early-returns. Everything else panics:
/// a `HostClass::Fail` (the hard-error
/// `CpuBudgetUnsatisfiable` / `TopologyUnrepresentable`, or a
/// `ResourceContention` — TRANSIENT peer contention is a RETRYABLE
/// failure the nextest retry budget re-runs, never a silent skip) panics
/// with the classified `ktstr: FAIL: {reason}` verdict, and any
/// non-host-class error panics with the raw `{e:#}` rendering — both are
/// real failures, not skips. `no_skip` is passed
/// `false` — this helper always skips the skip-class errors and has no
/// `KTSTR_NO_SKIP_MODE` promotion (unchanged from its prior behavior).
///
/// Replaces the recurring `match ... { Ok => v, Err(e) if
/// ResourceContention => skip!(...), Err(e) => panic!(...) }`
/// boilerplate. Inherits `skip!`'s early-return behavior, so callers
/// must return `()`.
macro_rules! skip_on_contention {
    ($expr:expr) => {
        match $expr {
            Ok(v) => v,
            Err(e) => match $crate::test_support::classify_host_error(&e, false) {
                $crate::test_support::HostClass::Skip { reason } => {
                    skip!("{reason}");
                }
                // Fail (cpu-budget / topology-unrepresentable /
                // resource-contention, the last RETRYABLE via nextest) is
                // a real failure, not a skip — panic with the classified
                // verdict so the banner carries the class framing (e.g.
                // "transient resource contention", never bare holder
                // noise), mirroring the `#[ktstr_test]` codegen body's
                // Fail arm. no_skip is false above, so the skip-class
                // errors classify as Skip here, never Fail.
                $crate::test_support::HostClass::Fail { reason } => {
                    panic!("ktstr: FAIL: {reason}");
                }
                // Any non-host-class error is likewise a real failure —
                // panic with the raw rendering, exactly as the prior
                // open-coded catch-all did.
                $crate::test_support::HostClass::NotHostClass => panic!("{e:#}"),
            },
        }
    };
}

/// Skip the calling test when the current process lacks a Linux
/// capability. Uses `prctl(PR_CAPBSET_READ, cap)` to probe the
/// capability bounding set — returns 1 when the cap is present, 0
/// when absent, -1 on EINVAL (unknown cap number).
///
/// Typical use: `require_capability!(libc::CAP_SYS_RESOURCE);` at the
/// top of a test that calls `setrlimit` to raise a hard limit.
#[allow(unused_macros)]
macro_rules! require_capability {
    ($cap:expr) => {{
        let ret = unsafe { libc::prctl(libc::PR_CAPBSET_READ, $cap, 0, 0, 0) };
        if ret != 1 {
            skip!(
                "missing capability {} (prctl PR_CAPBSET_READ returned {})",
                stringify!($cap),
                ret
            );
        }
    }};
}

/// Compile-time pin that a type does NOT impl [`Default`]. Mirrors
/// `static_assertions::assert_not_impl_any!(T, Default)` without
/// taking the dep — the `AmbiguousIfImpl<_>` blanket-vs-specialized
/// impl trick produces a compile error when both impls match for the
/// target type, which only happens if `T: Default`.
///
/// Use when a type's docs forbid `Default` because the zero / unset
/// state is semantically invalid. Existing call sites: `CgroupDef`
/// (name="cg_0" footgun), `Migration` (zeroed migration = self-
/// migration is contradictory), `WorkerExitInfo` (default-pick of
/// TimedOut variant masquerades as a real outcome), and
/// `DualFailureDumpReport` (default-empty `late` report would
/// silently lie about a successful capture). When adding a new
/// call site, append it here so a future debugger of an
/// AmbiguousIfImpl compile error sees the existing pattern.
///
/// Expands to a `const _: fn() = ...` block; safe to invoke at module
/// scope inside `#[cfg(test)] mod tests` or anywhere a `const` item
/// is valid.
#[allow(unused_macros)]
macro_rules! assert_not_impl_default {
    ($t:ty) => {
        const _: fn() = || {
            trait AmbiguousIfImpl<A> {
                fn some_item() {}
            }
            impl<T: ?Sized> AmbiguousIfImpl<()> for T {}
            #[allow(dead_code)]
            struct InvalidDefault;
            impl<T: ?Sized + Default> AmbiguousIfImpl<InvalidDefault> for T {}
            let _ = <$t as AmbiguousIfImpl<_>>::some_item;
        };
    };
}

#[cfg(test)]
mod tests {
    use crate::vmm::host_topology::{
        CpuBudgetUnsatisfiable, PerfModeUnavailable, ResourceContention, TopologyInsufficient,
    };

    /// A ResourceContention — wrapped in `.context(...)` or not — is a
    /// RETRYABLE FAILURE, not a skip. Queue-based lock acquisition
    /// waits for authoritative release; `ResourceContention` still
    /// covers transient host-resource syscall failures and must panic
    /// here (nextest re-runs the test), never silently green the cell
    /// as a skip. Chain-awareness still matters: the classifier
    /// must find the typed error under the context layers and produce
    /// the transient-contention Fail wording, not fall into the
    /// NotHostClass catch-all.
    #[test]
    #[should_panic(expected = "resource contention")]
    fn skip_on_contention_fails_context_wrapped_contention() {
        fn skip_fn() {
            let err: anyhow::Error = anyhow::Error::new(ResourceContention {
                reason: "simulated contention".into(),
            })
            .context("wrapping context layer 1")
            .context("wrapping context layer 2");
            let _: () = skip_on_contention!(Err::<(), _>(err));
        }
        skip_fn();
    }

    /// Unwrapped ResourceContention: same retryable-failure routing as
    /// the context-wrapped case, and the panic message must carry the
    /// transient framing — never "cannot run" host-incapacity wording.
    #[test]
    #[should_panic(expected = "transient")]
    fn skip_on_contention_fails_direct_contention() {
        fn skip_fn() {
            let err: anyhow::Error = anyhow::Error::new(ResourceContention {
                reason: "direct contention".into(),
            });
            let _: () = skip_on_contention!(Err::<(), _>(err));
        }
        skip_fn();
    }

    /// Non-contention errors still panic (negative case).
    #[test]
    #[should_panic(expected = "unrelated error")]
    fn skip_on_contention_panics_on_non_contention_error() {
        fn skip_fn() {
            let err = anyhow::anyhow!("unrelated error");
            let _: () = skip_on_contention!(Err::<(), _>(err));
        }
        skip_fn();
    }

    /// A [`TopologyInsufficient`] (the VM cannot boot on this host — a kvm
    /// hardware cap) routes to skip, including when wrapped in
    /// `.context(...)`.
    ///
    /// `#[cfg(panic = "unwind")]`: this test uses `std::panic::catch_unwind`
    /// to assert the macro does NOT panic. Under `panic = "abort"` (the
    /// release profile's setting — see `Cargo.toml [profile.release]`)
    /// panics cannot be caught; the panic aborts the whole test binary
    /// instead of returning an `Err` from `catch_unwind`. Gating the
    /// test on the panic strategy lets `cargo ktstr test --release`
    /// skip it without false-failing the binary.
    #[test]
    #[cfg(panic = "unwind")]
    fn skip_on_contention_skips_topology_insufficient() {
        let result = std::panic::catch_unwind(|| {
            fn skip_fn() {
                let err: anyhow::Error = anyhow::Error::new(TopologyInsufficient {
                    reason: "vCPU count 600 exceeds KVM_CAP_MAX_VCPUS 512; cannot boot a VM \
                             this wide"
                        .into(),
                })
                .context("build ktstr_test VM");
                let _: () = skip_on_contention!(Err::<(), _>(err));
                unreachable!("skip_on_contention! should have early-returned");
            }
            skip_fn();
        });
        assert!(
            result.is_ok(),
            "context-wrapped TopologyInsufficient must skip, not panic",
        );
    }

    /// A [`PerfModeUnavailable`] (the host fundamentally cannot honor
    /// perf-mode — too few CPUs for an exclusive LLC + a service CPU)
    /// routes to skip, including when wrapped in `.context(...)`. Pins the
    /// `skip_on_contention!` perf-mode arm above its `Err(e) => panic!`
    /// catch-all: a future reorder that drops it below the catch-all would
    /// compile but panic real perf-incapable hosts.
    ///
    /// `#[cfg(panic = "unwind")]`: same rationale as the
    /// TopologyInsufficient skip test —
    /// `catch_unwind` is unusable under `panic = "abort"`.
    #[test]
    #[cfg(panic = "unwind")]
    fn skip_on_contention_skips_perf_mode_unavailable() {
        let result = std::panic::catch_unwind(|| {
            fn skip_fn() {
                let err: anyhow::Error = anyhow::Error::new(PerfModeUnavailable {
                    reason: "host too small for perf topology".into(),
                })
                .context("build ktstr_test VM");
                let _: () = skip_on_contention!(Err::<(), _>(err));
                unreachable!("skip_on_contention! should have early-returned");
            }
            skip_fn();
        });
        assert!(
            result.is_ok(),
            "context-wrapped PerfModeUnavailable must skip, not panic",
        );
    }

    /// Anti-fragility: a plain error whose message HAPPENS to contain
    /// "need" + "CPU" but carries no typed skip-class error must PANIC
    /// (it is a real failure), not skip. The replaced string-match
    /// (`"need"` + `"CPU"`/`"LLC"`) would have wrongly skipped this.
    #[test]
    #[should_panic(expected = "did not get the CPU")]
    fn skip_on_contention_panics_on_unrelated_need_cpu_message() {
        fn skip_fn() {
            let err =
                anyhow::anyhow!("scheduler regression: workload did not get the CPU time it needs");
            let _: () = skip_on_contention!(Err::<(), _>(err));
        }
        skip_fn();
    }

    /// A typed HARD-FAIL host error is NOT in skip_on_contention!'s skip
    /// set: classify_host_error returns HostClass::Fail for a
    /// CpuBudgetUnsatisfiable (an operator --cpu-cap the host cannot
    /// satisfy), which the macro's `_ =>` arm panics — a typed hard-fail
    /// must never be swallowed as a skip. Pins the Fail->panic boundary the
    /// classify_host_error routing depends on; the skip tests cover the
    /// Skip set (TI/perf) and the plain-NotHostClass panics, but not
    /// this typed-Fail edge.
    #[test]
    #[should_panic(expected = "exceeds the allowed cpuset")]
    fn skip_on_contention_panics_on_typed_hard_fail() {
        fn skip_fn() {
            let err: anyhow::Error = anyhow::Error::new(CpuBudgetUnsatisfiable {
                reason: "--cpu-cap = 999 exceeds the allowed cpuset".into(),
            });
            let _: () = skip_on_contention!(Err::<(), _>(err));
        }
        skip_fn();
    }

    /// The `skip!` macro must emit the canonical `ktstr: SKIP:
    /// <reason>` banner to stderr AND early-return from the calling
    /// function. Prior tests exercise `test_skip` (the lower-level
    /// emitter) and `skip_on_contention!` (the wrapper macro) but
    /// the bare `skip!` macro was left uncovered — a regression that
    /// silently broke the format_args expansion or the `return;`
    /// tail would slip through until a downstream consumer
    /// parsed the wrong line.
    ///
    /// This test uses the crate-shared stderr-capture helper and
    /// verifies BOTH invariants: the captured bytes carry the
    /// canonical banner, and a post-`skip!` line in the helper fn
    /// is never reached (pinned via a sentinel flag).
    #[test]
    fn skip_macro_emits_banner_and_early_returns() {
        use crate::test_support::test_helpers::capture_stderr;
        use std::sync::atomic::{AtomicBool, Ordering};

        let reached_tail = AtomicBool::new(false);
        let (_, bytes) = capture_stderr(|| {
            // Helper fn returning `()` so `skip!` can emit its
            // `return;` tail. The AtomicBool is set only if the
            // line AFTER `skip!` executes — a regression that
            // dropped the `return;` tail would trip it. The two
            // `#[allow(...)]` attributes are load-bearing: when
            // `skip!` correctly returns, `reached.store` is dead
            // code AND `reached` falls out of the live set —
            // which is exactly what this test is designed to
            // pin. Without the allows, compilation warns about
            // the very invariant the test verifies.
            #[allow(unused_variables, unreachable_code)]
            fn helper(reached: &AtomicBool) {
                skip!("macro-level reason with {} substitution", "format-args");
                reached.store(true, Ordering::SeqCst);
            }
            helper(&reached_tail);
        });
        let text = std::str::from_utf8(&bytes).expect("stderr is UTF-8");
        assert_eq!(
            text, "ktstr: SKIP: macro-level reason with format-args substitution\n",
            "expected canonical banner with format-args substitution",
        );
        assert!(
            !reached_tail.load(Ordering::SeqCst),
            "skip! must early-return; lines after the macro must not execute",
        );
    }

    /// `skip!` with a literal (no format args) still emits the
    /// banner. Pairs with the substitution test above to cover the
    /// no-args branch of the `format_args!($($arg)*)` expansion.
    #[test]
    fn skip_macro_literal_reason_emits_banner() {
        use crate::test_support::test_helpers::capture_stderr;
        let (_, bytes) = capture_stderr(|| {
            fn helper() {
                skip!("literal skip reason");
            }
            helper();
        });
        let text = std::str::from_utf8(&bytes).unwrap();
        assert_eq!(text, "ktstr: SKIP: literal skip reason\n");
    }

    /// `skip!()` with ZERO arguments expands to
    /// `format_args!()` — an empty reason. The banner still fires
    /// with the canonical prefix + colon + empty tail + newline.
    /// Pins the degenerate-input behavior so a regression that
    /// rejected zero-argument expansion (e.g. a macro arm
    /// requiring at least one token tree) fails here instead of at
    /// some downstream call site that happens to call `skip!()`
    /// for "I don't care why, just skip" semantics.
    #[test]
    fn skip_macro_zero_args_emits_banner_with_empty_reason() {
        use crate::test_support::test_helpers::capture_stderr;
        let (_, bytes) = capture_stderr(|| {
            fn helper() {
                skip!();
            }
            helper();
        });
        let text = std::str::from_utf8(&bytes).unwrap();
        assert_eq!(text, "ktstr: SKIP: \n");
    }

    /// Pin the contract that the `#[ktstr_test]` macro's generated
    /// expect_ok body relies on: when `run_ktstr_test` returns
    /// `Err(ResourceContention)` (possibly wrapped in `.context(...)`),
    /// the macro must PANIC with the `ktstr: FAIL:` transient-contention
    /// verdict — a retryable failure nextest re-runs — and must NOT emit
    /// a SKIP banner (a skip is a libtest pass, which nextest never
    /// retries: the silent-coverage-loss bug). The macro lives in
    /// `ktstr-macros` and expands to a `match` whose catch-all `Err(e)`
    /// arm routes through the REAL
    /// [`crate::test_support::classify_host_error`] (the shared
    /// single-source-of-truth classifier, also used by
    /// `err_to_exit_code`) and maps a [`HostClass::Fail`] to `panic!`.
    /// We can't invoke the proc-macro from a unit test, but we CAN
    /// exercise the real classifier + the same control-flow shape and
    /// assert the observable behaviour: the panic fires, its message
    /// carries the transient framing (extracted typed reason, not the
    /// `.context(...)` chain), and it never claims host incapacity.
    ///
    /// `no_skip` is passed `false` directly (rather than read from the
    /// env) so the test deterministically pins that contention is a Fail
    /// EVEN in skip-default mode — the env read is the macro's concern,
    /// not the classifier's (its env-independence is the whole
    /// testability win).
    ///
    /// `#[cfg(panic = "unwind")]`: `catch_unwind` is unusable under
    /// `panic = "abort"` (the release profile's setting).
    #[test]
    #[cfg(panic = "unwind")]
    fn ktstr_test_macro_body_fails_retryably_on_resource_contention() {
        use crate::test_support::{HostClass, classify_host_error};
        use crate::vmm::host_topology::ResourceContention;

        let result = std::panic::catch_unwind(|| {
            // Simulates the catch-all `Err(e)` arm of the body that
            // `ktstr-macros::ktstr_test` expands into for a
            // non-`expect_err` test: classify via the real shared fn,
            // map Fail -> panic (the generated body's Fail arm).
            fn helper() {
                let result: Result<(), anyhow::Error> =
                    Err(anyhow::Error::new(ResourceContention {
                        reason: "all 3 LLC slots busy".into(),
                    })
                    .context("build ktstr_test VM"));
                match result {
                    Ok(_) => {}
                    Err(e) => match classify_host_error(&e, false) {
                        HostClass::Skip { reason } => {
                            eprintln!("ktstr: SKIP: {reason}");
                        }
                        HostClass::Fail { reason } => panic!("ktstr: FAIL: {reason}"),
                        HostClass::NotHostClass => panic!("{e:#}"),
                    },
                }
            }
            helper();
        });
        let panic_payload = result.expect_err("contention must panic (retryable fail), not skip");
        let msg = panic_payload
            .downcast_ref::<String>()
            .cloned()
            .unwrap_or_default();
        assert!(
            msg.contains("ktstr: FAIL:") && msg.contains("resource contention"),
            "panic must carry the FAIL verdict with the contention reason; got: {msg:?}",
        );
        assert!(
            msg.contains("all 3 LLC slots busy"),
            "panic must surface the extracted typed reason; got: {msg:?}",
        );
        assert!(
            msg.contains("transient") && !msg.contains("cannot run"),
            "panic must frame contention as transient, never host incapacity; got: {msg:?}",
        );
    }
}
