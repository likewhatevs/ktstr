#![cfg(feature = "wprof")]
//! End-to-end: `#[ktstr_test(wprof, wprof_args = "...")]` propagates
//! the override args to the guest kernel cmdline (verbatim) and
//! still produces a valid `.wprof.pb`.
//!
//! Distinct from `tests/wprof_artifact_e2e.rs` (which pins the
//! default-args wprof chain) — this test pins the OVERRIDE hop:
//!
//!   1. macro `#[ktstr_test(wprof_args = "<literal>")]` sets
//!      `KtstrTestEntry::wprof_args = Some("<literal>")`
//!   2. `attach_wprof_if_requested` in `src/test_support/runtime.rs`
//!      replaces `WprofConfig::args` (whose default comes from
//!      `WprofConfig::default_args()`) with the override split
//!      by whitespace
//!   3. `src/vmm/setup/mod.rs`'s cmdline builder appends
//!      `KTSTR_WPROF_ARGS=<joined>` to the guest cmdline
//!   4. Guest `/proc/cmdline` therefore exposes the verbatim
//!      `KTSTR_WPROF_ARGS=<literal>` token — readable by this
//!      test body inside the VM
//!
//! Asserting on `/proc/cmdline` (guest-side, inside the test
//! body) directly tests both the `attach_wprof_if_requested`
//! override site AND the cmdline-write site in a single read —
//! no fuzzy `.pb` size comparison, no second baseline run, no
//! race against wprof's own exit.
//!
//! ## Host-side verification via post_vm
//!
//! The `.wprof.pb` shape check runs in the HOST-side
//! [`assert_wprof_pb_landed`] callback rather than the
//! guest test body. The body runs INSIDE the guest VM and
//! cannot read the host sidecar directory (no virtio-fs mount
//! for it per `src/vmm/rust_init/mounts.rs`'s mount table); the
//! `post_vm` callback runs HOST-side after `vm.run()` returns
//! with `.wprof.pb` already on disk. The shape check is
//! inherited from the shared
//! `ktstr::test_support::wprof::assert_wprof_pb_shape` helper,
//! invoked via `VmResult::assert_wprof_pb_landed`
//! (size ≥ 4096 bytes + first byte 0x0a) so an override that
//! propagates but produces a broken trace is caught here too.
//!
//! Runs on the self-hosted CI runners (`[ktstr-x64]` /
//! `[ktstr-arm64]`). ktstr supplies the guest kernel itself via
//! its kernel-build cache.

use anyhow::Result;
use ktstr::assert::{AssertDetail, AssertResult, DetailKind};
use ktstr::ktstr_test;
use ktstr::prelude::{Scheduler, SchedulerSpec, VmResult, WorkType};
use ktstr::scenario::Ctx;
use ktstr::scenario::ops::{HoldSpec, Step, execute_steps};

const KTSTR_SCHED: Scheduler =
    Scheduler::named("ktstr_sched").binary(SchedulerSpec::Discover("scx-ktstr"));

/// Override args, listed as a Vec so the test can re-encode them
/// with the same per-arg delimiter the framework uses on the
/// kernel cmdline. `-d 100` differs from the default `-d 500`,
/// `-e tidpid` differs from the default `-e sched` (both are
/// valid `--emit-feature` values per `wprof --help`), and the
/// ringbuf sizing (`--ringbuf-size=8192 --ringbuf-cnt=2`) differs
/// from the default `--ringbuf-size=16384 --ringbuf-cnt=1`. Any one
/// of these differences would suffice; the combination makes
/// accidental substring overlap with the default-args signature
/// impossible.
///
/// The override arena (round-pow-2(8192 KiB) × 2 = 16 MiB) stays
/// under the guest's universal 256 MiB memory floor, so this test
/// needs no `memory_mib` override — the tiny default wprof floor no
/// longer pins every wprof cell at 2 GiB.
///
/// MUST match the `wprof_args = "..."` literal in the attribute
/// below. The macro accepts either a `Lit::Str` or a path to a
/// `const &'static str` for `wprof_args` (ktstr-macros
/// src/ktstr_test/mod.rs), but OVERRIDE_ARGS is a `&[&str]` slice,
/// not a `&'static str`, so it can't be passed to the attribute
/// directly — hence the duplicated joined literal.
/// Drift between the two surfaces as the cmdline-contains
/// assertion failing at runtime.
const OVERRIDE_ARGS: &[&str] = &[
    "-d",
    "100",
    "-e",
    "tidpid",
    "--ringbuf-size=8192",
    "--ringbuf-cnt=2",
];

/// Per-arg delimiter used on the kernel cmdline. The framework
/// encodes wprof args with ASCII Unit Separator (`\x1F`) via
/// `WprofConfig::args_cmdline` because kernel cmdline tokenization
/// would truncate a space-joined value at the first space — guest
/// `cmdline_val("KTSTR_WPROF_ARGS")` reads via `split_whitespace`
/// and would otherwise see only the first arg. This test
/// re-encodes the same way so its substring assertion stays in
/// lockstep with the framework's encoding.
const WPROF_ARGS_CMDLINE_DELIM: char = '\x1F';

#[ktstr_test(
    scheduler = KTSTR_SCHED,
    llcs = 1,
    cores = 2,
    threads = 1,
    duration_s = 3,
    watchdog_timeout_s = 15,
    wprof,
    wprof_args = "-d 100 -e tidpid --ringbuf-size=8192 --ringbuf-cnt=2",
    auto_repro = false,
    post_vm = VmResult::assert_wprof_pb_landed,
)]
fn wprof_args_override_propagates_to_guest_cmdline(ctx: &Ctx) -> Result<AssertResult> {
    // Minimal SpinWait workload — gives wprof's sched tracer
    // something to capture so the `.pb` shape check downstream
    // (in `assert_wprof_pb_landed`) exercises a real
    // trace, not just an empty interned-string table.
    let steps = vec![Step {
        setup: vec![
            ctx.cgroup_def("wl")
                .workers(1)
                .work_type(WorkType::SpinWait),
        ]
        .into(),
        ops: vec![],
        hold: HoldSpec::FULL,
    }];
    let mut result = execute_steps(ctx, steps)?;

    // Read the guest's /proc/cmdline. `KTSTR_WPROF_ARGS=<args>`
    // is appended by `src/vmm/setup/mod.rs` from
    // `WprofConfig::args_cmdline()`; the override flows through
    // `attach_wprof_if_requested`'s `config.args = custom_args
    // .split_whitespace()...` and the cmdline write joins with the
    // ASCII Unit Separator (\x1F) via `WprofConfig::args_cmdline()`.
    // This is the ONE check that legitimately
    // belongs in the guest body — `/proc/cmdline` is guest-local
    // procfs, readable here without crossing the host boundary.
    let cmdline = match std::fs::read_to_string("/proc/cmdline") {
        Ok(s) => s,
        Err(e) => {
            result.record_fail(AssertDetail::new(
                DetailKind::Other,
                format!(
                    "read /proc/cmdline: {e}. The guest kernel must \
                     expose /proc/cmdline (procfs auto-mounted by \
                     ktstr's init); a failure here means procfs is \
                     missing or unreadable, not that wprof_args \
                     propagation regressed."
                ),
            ));
            return Ok(result);
        }
    };

    // Positive: the override args must appear verbatim, encoded
    // with the framework's per-arg delimiter (ASCII Unit
    // Separator, U+001F). The override → cmdline round-trip
    // (`split_whitespace().collect::<Vec<_>>().join("\x1F")`) is
    // whitespace-NORMALIZING, not identity-preserving — input
    // `"  -d 100  "` would emerge as `"-d\x1F100"`. The chosen
    // OVERRIDE_ARGS const is canonically-spaced (one token per
    // slice element, no leading/trailing whitespace) so for THIS
    // input the round-trip is bit-exact and the `contains` check
    // is a substring match against the freshly-joined expected
    // value. Operators copying this pattern with multi-space or
    // padded inputs must canonicalize the input first OR build
    // the expected token from a post-`split_whitespace` re-join.
    let joined = OVERRIDE_ARGS
        .to_vec()
        .join(&WPROF_ARGS_CMDLINE_DELIM.to_string());
    let expected = format!("KTSTR_WPROF_ARGS={joined}");
    if !cmdline.contains(&expected) {
        result.record_fail(AssertDetail::new(
            DetailKind::Other,
            format!(
                "/proc/cmdline missing override token {expected:?}; \
                 got cmdline = {cmdline:?}. The override flow is \
                 #[ktstr_test(wprof_args = LIT)] → \
                 KtstrTestEntry::wprof_args → \
                 attach_wprof_if_requested replaces \
                 WprofConfig::args → setup.rs appends \
                 KTSTR_WPROF_ARGS=<args> to cmdline. A missing \
                 token means one of those hops dropped the \
                 override OR re-derived defaults."
            ),
        ));
        return Ok(result);
    }

    // Exactly-one emission: a regression where a future setup.rs
    // refactor calls the cmdline-write twice would yield a
    // cmdline like `... KTSTR_WPROF_ARGS=-d 100 ... KTSTR_WPROF_ARGS=-d 100 ...`
    // — the positive check above still passes but the wprof
    // process would see the LAST occurrence (kernel cmdline
    // parsing semantics), or the value-merger would silently
    // pick one. Pin emission count to 1.
    let emission_count = cmdline.matches("KTSTR_WPROF_ARGS=").count();
    if emission_count != 1 {
        result.record_fail(AssertDetail::new(
            DetailKind::Other,
            format!(
                "/proc/cmdline contains {emission_count} \
                 `KTSTR_WPROF_ARGS=` tokens; expected exactly 1. \
                 A duplicate-emission regression (e.g. a setup.rs \
                 refactor that wrote the wprof-args env twice) \
                 would silently leave the override intact AND \
                 produce a doubled token here. cmdline = \
                 {cmdline:?}"
            ),
        ));
        return Ok(result);
    }

    // Negative: the default-args prefix
    // `KTSTR_WPROF_ARGS=-d\x1F500` must NOT appear. Catches a
    // regression where the override is added in addition to —
    // rather than replacing — the defaults. The bare `-d\x1F500`
    // (no trailing delimiter) is chosen because the default-args
    // list could later shrink (e.g. drop ringbuf-size); the
    // `-d\x1F500` prefix is the narrowest stable signature of
    // "defaults emitted at all" under the Unit-Separator encoding.
    let default_signature = format!("KTSTR_WPROF_ARGS=-d{WPROF_ARGS_CMDLINE_DELIM}500");
    if cmdline.contains(&default_signature) {
        result.record_fail(AssertDetail::new(
            DetailKind::Other,
            format!(
                "/proc/cmdline contains default-args prefix \
                 {default_signature:?} despite wprof_args override; \
                 got cmdline = {cmdline:?}. The override must \
                 REPLACE WprofConfig::args, not stack on top of \
                 the defaults — `attach_wprof_if_requested` at \
                 src/test_support/runtime.rs assigns \
                 `config.args = custom_args.split_whitespace()...` \
                 unconditionally, so a leak here means a refactor \
                 introduced an additional cmdline-emission path \
                 that bypassed the override site."
            ),
        ));
        return Ok(result);
    }

    // .wprof.pb shape check runs in `assert_wprof_pb_landed`
    // above — host-side, with access to the sidecar file.
    Ok(result)
}
