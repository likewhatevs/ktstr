//! Link-time registry of declarative scenarios.
//!
//! [`KTSTR_TESTS`](super::KTSTR_TESTS) records how to *run* a test.
//! This parallel slice records what a test's workload *is*: every
//! [`#[ktstr_scenario]`](crate::ktstr_scenario) expands to one
//! [`ScenarioEntry`] here alongside its ordinary `KtstrTestEntry`, so
//! the scenario can be enumerated and read as data without booting a
//! guest, running the test, or knowing the test file's name.
//!
//! The two slices are keyed by the same `name`, which is how a
//! consumer joins "what the workload is" (here) to "what topology,
//! scheduler and gates it runs under" (the `KtstrTestEntry`).
//!
//! # Why a registry rather than calling the builder directly
//!
//! A scenario builder is a plain function, so any code that can name
//! it can call it. The registry exists for the code that CANNOT name
//! it: a tool that wants every scenario in a test binary has no list
//! to iterate, and hand-maintaining one drifts the moment somebody
//! adds a test. Registering at the definition site keeps the
//! enumeration exact by construction — the same reason
//! `KTSTR_TESTS` is a distributed slice rather than a `match` in a
//! dispatcher.
//!
//! # Invariant
//!
//! [`ScenarioEntry::build`] must be callable on the host, with no
//! `&Ctx` and no guest. That is enforced upstream by the
//! `#[ktstr_scenario]` macro, which rejects a builder that takes any
//! parameter. Nothing here can re-check it, so the macro is the only
//! gate — do not construct a `ScenarioEntry` by hand around a builder
//! that closes over runtime state.

use linkme::distributed_slice;

use crate::scenario::ScenarioDef;

/// One registered declarative scenario: the test name it belongs to,
/// and a host-callable builder for its [`ScenarioDef`].
#[derive(Clone, Copy, Debug)]
pub struct ScenarioEntry {
    /// The registering test's name — the same string as the paired
    /// [`KtstrTestEntry::name`](super::KtstrTestEntry::name), which is
    /// how the two registries join.
    pub name: &'static str,
    /// Build the scenario. Pure with respect to the guest: safe to
    /// call on the host, at any time, as many times as you like.
    pub build: fn() -> ScenarioDef,
}

/// Distributed slice collecting all `#[ktstr_scenario]` registrations
/// via linkme.
#[distributed_slice]
pub static KTSTR_SCENARIOS: [ScenarioEntry];

/// Look up a registered scenario by test name.
#[must_use]
pub fn find_scenario(name: &str) -> Option<&'static ScenarioEntry> {
    KTSTR_SCENARIOS.iter().find(|e| e.name == name)
}
