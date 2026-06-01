//! End-to-end test for the failure-dump pipeline.
//!
//! Each test boots scx-ktstr with `--stall-after=1`, lets the BPF
//! probe latch the resulting `SCX_EXIT_ERROR_STALL` exit, and asserts
//! that the freeze coordinator's host-side dump renders a distinct
//! facet of the captured scheduler state (see the per-test bars
//! below).
//!
//! The freeze coordinator writes the JSON-pretty `FailureDumpReport`
//! to a per-test path inside the run's sidecar directory
//! (`{sidecar_dir()}/{test_name}.failure-dump.json`). The test
//! framework's primary dispatch
//! (`test_support::eval::run_ktstr_test_inner`) attaches that
//! path on every VM builder it constructs — no env var required,
//! no per-scenario setup beyond reading the path back here after
//! the run.
//!
//! User-facing test bar (per project memory): "I see variable names
//! and values in the logs when a scheduler stalls." Each test boots
//! the stall scenario, then asserts on the host-written dump from a
//! `post_vm_unconditional` callback — the guest is a separate process
//! and cannot read the host-side dump, and a callback's `Err` is a
//! hard FAIL (via `PostVmAssertionFailure`) that `expect_err` does
//! not invert. The four callbacks enforce complementary halves of
//! that bar:
//!   - [`check_bss_dump`]: BTF-rendered `.bss` field names (`stall`,
//!     `crash`), a live `BPF_MAP_TYPE_ARENA` page carrying the
//!     `KTSTR_ARENA_MAGIC` sentinel, non-zero `ktstr_alloc_count`,
//!     and populated per-vCPU register snapshots;
//!   - [`check_array_entries_dump`]: every key of a multi-entry
//!     `BPF_MAP_TYPE_ARRAY` renders (not just key 0);
//!   - [`check_capture_dump`]: the freeze coordinator's live-walker
//!     captures (per-CPU rq->scx state, DSQs, scx_sched scalar,
//!     task enrichments, NUMA stats);
//!   - [`check_probe_dump`]: the probe's per-CPU `.bss` counters
//!     (`trigger_count`, `probe_count`) surface via the host decoder.

mod common;

use anyhow::Result;
use common::failure_dump::read_failure_dump;
use ktstr::assert::AssertResult;
use ktstr::ktstr_test;
use ktstr::prelude::{SCHEMA_SINGLE, VmResult};
use ktstr::scenario::ops::{HoldSpec, Step, await_accessor_ready, execute_steps};
use ktstr::test_support::{Scheduler, SchedulerSpec};

const KTSTR_SCHED: Scheduler =
    Scheduler::named("ktstr_sched").binary(SchedulerSpec::Discover("scx-ktstr"));

fn scenario_failure_dump_renders_bss_fields(ctx: &ktstr::scenario::Ctx) -> Result<AssertResult> {
    // Adopt the freeze coordinator's accessor before the
    // --stall-after=1 stall fires, so the dump renders real
    // .bss/arena state rather than placeholder values (the accessor is
    // built async and may not be adopted by the time an early stall
    // freezes the VM). The host-side `check_bss_dump` post_vm callback
    // does the assertions — the guest is a separate process and cannot
    // read the host-written dump.
    await_accessor_ready();
    let steps = vec![Step {
        setup: vec![ctx.cgroup_def("cg_0")].into(),
        ops: vec![],
        hold: HoldSpec::FULL,
    }];
    execute_steps(ctx, steps)
}

/// Host-side post_vm assertion for `failure_dump_renders_bss_fields`.
/// Reads the freeze coordinator's dump from the HOST sidecar path and
/// verifies the scheduler `.bss` BTF render (named fields plus the live
/// `stall` flag), the per-vCPU register snapshots, and the BPF arena
/// capture (live `KTSTR_ARENA_MAGIC` pages). Runs unconditionally; its
/// Err is a hard FAIL via the framework's `PostVmAssertionFailure`
/// marker even though `expect_err` inverts the stall itself to PASS.
fn check_bss_dump(result: &VmResult) -> Result<()> {
    let value = read_failure_dump(result)?;

    // The full-dump happy path expects SCHEMA_SINGLE; a `degraded`
    // schema here means the freeze coordinator's capture-vs-degraded
    // dispatch (gate cross-reference or rendezvous-timeout path) fired
    // when it should not have.
    let schema = value
        .get("schema")
        .and_then(|s| s.as_str())
        .ok_or_else(|| {
            anyhow::anyhow!(
                "dump JSON missing top-level `schema` field — the dispatcher \
                 at `FailureDumpReportAny::from_json` requires an explicit \
                 discriminant"
            )
        })?;
    anyhow::ensure!(
        schema == SCHEMA_SINGLE,
        "happy-path dump must carry schema=SCHEMA_SINGLE ({SCHEMA_SINGLE:?}); \
         got schema={schema}"
    );

    // Top-level shape: {"maps": [...]}. `non_exhaustive` does not
    // affect serde output, so the field name is stable.
    let maps = value
        .get("maps")
        .and_then(|m| m.as_array())
        .ok_or_else(|| anyhow::anyhow!("dump JSON missing top-level `maps` array"))?;

    // Find the scx-ktstr `.bss` map. libbpf composes
    // `<obj_name>.bss` for the global-section map, and scx-ktstr's
    // BPF object is `bpf` (per `scx-ktstr/build.rs`'s
    // `enable_skel("src/bpf/main.bpf.c", "bpf")` call), so the dump
    // should carry an entry whose name ends with `.bss` and is NOT
    // one of the framework probes filtered by `KTSTR_INTERNAL_MAPS`.
    let bss_map = maps
        .iter()
        .find(|m| {
            m.get("name")
                .and_then(|n| n.as_str())
                .map(|n| {
                    n.ends_with(".bss")
                        && !n.starts_with("probe_bp.")
                        && !n.starts_with("fentry_p.")
                })
                .unwrap_or(false)
        })
        .ok_or_else(|| {
            anyhow::anyhow!(
                "dump has no scheduler `.bss` map (got {} maps): {value}",
                maps.len()
            )
        })?;

    // The rendered value is a Struct whose members enumerate the
    // BTF-resolved global names. Serde tags the variant via
    // `kind = "struct"`; members are at `.value.members[]`. Each
    // member is `{ "name": "<field>", "value": {...} }`.
    let value_field = bss_map
        .get("value")
        .ok_or_else(|| anyhow::anyhow!(".bss map has no `value` field"))?;
    let kind = value_field
        .get("kind")
        .and_then(|k| k.as_str())
        .unwrap_or("");
    if kind != "struct" {
        anyhow::bail!(
            "expected .bss value to render as a Struct (kind=\"struct\"), got kind={kind:?}: \
             {value_field}"
        );
    }
    let members = value_field
        .get("members")
        .and_then(|m| m.as_array())
        .ok_or_else(|| anyhow::anyhow!(".bss Struct has no `members` array"))?;

    // The user-facing test bar: BTF field names must appear in the
    // rendered output, NOT hex offsets. scx-ktstr's main.bpf.c
    // declares `stall`, `crash`, `degrade_rt`, `degrade_cnt`,
    // `slow_cnt` (and others) at file scope. At minimum the trigger
    // field `stall` and the headline error fields must be visible —
    // the others may shift across scx-ktstr versions, so don't
    // pin the full set.
    let names: std::collections::HashSet<&str> = members
        .iter()
        .filter_map(|m| m.get("name").and_then(|n| n.as_str()))
        .collect();
    for required in ["stall", "crash"] {
        if !names.contains(required) {
            anyhow::bail!(
                "BTF-rendered .bss missing required field `{required}` — \
                 either the field was renamed in scx-ktstr's main.bpf.c \
                 or the renderer fell through to an Unsupported branch \
                 instead of recursing into the Struct. members: {names:?}"
            );
        }
    }

    // Stall flag must be a non-zero unsigned integer — proves the
    // dump captured the LIVE state at error-exit time, not a
    // pre-init zero. scx-ktstr writes `stall = 1` from
    // `--stall-after=1` before the watchdog fires.
    let stall_value = members
        .iter()
        .find(|m| m.get("name").and_then(|n| n.as_str()) == Some("stall"))
        .and_then(|m| m.get("value"))
        .ok_or_else(|| anyhow::anyhow!("`stall` member found but has no `value`"))?;
    let stall_kind = stall_value
        .get("kind")
        .and_then(|k| k.as_str())
        .unwrap_or("");
    let stall_int = stall_value
        .get("value")
        .and_then(|v| v.as_u64())
        .ok_or_else(|| {
            anyhow::anyhow!(
                "`stall` value is not a numeric u64 (kind={stall_kind:?}): {stall_value}"
            )
        })?;
    if stall_int == 0 {
        anyhow::bail!(
            "`stall` rendered as 0 — the scheduler-side stall flag never flipped, \
             or the freeze coordinator captured pre-stall state. Full value: {stall_value}"
        );
    }

    // Per-vCPU register snapshots: the freeze coordinator
    // attaches a `vcpu_regs` array (BSP at index 0, APs at
    // 1..N). Each entry is `null` when capture failed for that
    // vCPU OR a `{instruction_pointer, stack_pointer,
    // page_table_root}` object otherwise. The test asserts the
    // array exists, has at least one populated entry, and that
    // entry's `instruction_pointer` is non-zero (a zero RIP/PC
    // would mean the snapshot was captured but holds garbage —
    // possibly indicating the vCPU thread crashed before the
    // capture or the kernel/userspace VA was uninitialized).
    //
    // `vcpu_regs` is opt-out via serde's `skip_serializing_if =
    // "Vec::is_empty"`, so its absence here would mean the
    // freeze coordinator's regs-attach path didn't fire — a
    // genuine regression on the host-side capture wiring.
    let vcpu_regs = value
        .get("vcpu_regs")
        .and_then(|v| v.as_array())
        .ok_or_else(|| {
            anyhow::anyhow!(
                "dump JSON missing top-level `vcpu_regs` array — \
                 freeze coordinator did not attach per-vCPU register \
                 snapshots after rendezvous"
            )
        })?;
    if vcpu_regs.is_empty() {
        anyhow::bail!(
            "dump JSON `vcpu_regs` array is empty — expected at least \
             one entry (BSP idx 0 plus N APs)"
        );
    }
    let populated_with_ip: Vec<&serde_json::Value> = vcpu_regs
        .iter()
        .filter(|slot| {
            slot.is_object()
                && slot
                    .get("instruction_pointer")
                    .and_then(|ip| ip.as_u64())
                    .is_some_and(|ip| ip != 0)
        })
        .collect();
    if populated_with_ip.is_empty() {
        anyhow::bail!(
            "dump JSON `vcpu_regs` has no entry with non-zero \
             instruction_pointer — every slot is null or has zero IP. \
             Capture-on-vCPU-thread path may be broken or rendezvous \
             timed out before any vCPU completed handle_freeze. \
             Full vcpu_regs: {vcpu_regs:?}"
        );
    }

    // user_page_table_root is arch-conditional:
    //   x86_64: always None → JSON key absent
    //     (skip_serializing_if = "Option::is_none").
    //   aarch64: best-effort Some(ttbr0_el1) when the TTBR0_EL1
    //     KVM_GET_ONE_REG read succeeds; otherwise still absent.
    //
    // Pin per-arch behaviour so a future field rename or a regression
    // (e.g. accidentally always populating on x86_64) is caught.
    #[cfg(target_arch = "x86_64")]
    {
        for slot in &populated_with_ip {
            assert!(
                slot.get("user_page_table_root").is_none(),
                "x86_64 vcpu_regs entry must NOT carry user_page_table_root \
                 (CR3 alone identifies the active mm); got: {slot}"
            );
        }
    }
    // (aarch64 doesn't get a hard requirement here because the
    // sysreg read can be gated by the host kernel — best-effort
    // capture per the design. The serde test inside exit_dispatch
    // already pins the JSON-key contract for both states.)

    // Arena map presence + live-data assertions.
    //
    // A BPF arena (BPF_MAP_TYPE_ARENA) is a sparse, page-granular
    // memory region the host walker translates page-by-page from the
    // guest's kernel page tables. scx-ktstr uses sdt_alloc-backed
    // arena memory: every task gets a `struct ktstr_arena_ctx`
    // allocated via `scx_task_alloc` in `ktstr_init_task` (see
    // `scx-ktstr/src/bpf/main.bpf.c`). Each allocation stamps:
    //   - magic   = KTSTR_ARENA_MAGIC (0xDEADBEEFCAFEBABE)
    //   - counter = KTSTR_TASK_COUNTER (42; not separately asserted —
    //               magic alone proves liveness)
    // and increments `ktstr_alloc_count` (u64 in .bss). After
    // `--stall-after=1` triggers the watchdog, tasks have already
    // run through `init_task`, so the dump must capture:
    //   1. a non-zero `ktstr_alloc_count` member in `.bss` — proves
    //      the alloc path executed and the counter was captured live.
    //   2. an arena map (BPF_MAP_TYPE_ARENA = 33) by map_type — the
    //      arena is declared bare-named ("arena") via the __weak
    //      SEC(".maps") declaration in lib/arena_map.h, with no
    //      libbpf <obj>.<section> prefix.
    //   3. at least one captured page in arena.pages — proves the
    //      walker translated user_addr → kern_vm and read live
    //      memory, not an empty snapshot.
    //   4. the magic constant inside at least one page's bytes — the
    //      bar test ("LIVE data, not zeros") requires content
    //      verification, not just non-empty pages.
    //
    // BPF_MAP_TYPE_ARENA is hardcoded as 33 here to match the test's
    // existing pattern of not importing crate internals (the test
    // operates on JSON shape, not Rust types).

    // Read `ktstr_alloc_count` first — it cross-validates the alloc
    // path independently of the arena walker, and the magic-scan bail
    // below uses its value to narrow down the failure mode (alloc ran
    // but capture broken, vs alloc never ran).
    let alloc_count_value = members
        .iter()
        .find(|m| m.get("name").and_then(|n| n.as_str()) == Some("ktstr_alloc_count"))
        .and_then(|m| m.get("value"))
        .ok_or_else(|| {
            anyhow::anyhow!(
                "BTF-rendered .bss missing `ktstr_alloc_count` — \
                 either the field was renamed in scx-ktstr's main.bpf.c, \
                 or the BTF Datasec walker did not surface it. members: \
                 {names:?}"
            )
        })?;
    let alloc_count_kind = alloc_count_value
        .get("kind")
        .and_then(|k| k.as_str())
        .unwrap_or("");
    let alloc_count_int = alloc_count_value
        .get("value")
        .and_then(|v| v.as_u64())
        .ok_or_else(|| {
            anyhow::anyhow!(
                "`ktstr_alloc_count` value is not a numeric u64 \
                 (kind={alloc_count_kind:?}): {alloc_count_value}"
            )
        })?;
    if alloc_count_int == 0 {
        anyhow::bail!(
            "`ktstr_alloc_count` rendered as 0 — the alloc path never \
             ran (no `__sync_fetch_and_add` in ktstr_init_task), or \
             the dump captured pre-init state. Full value: \
             {alloc_count_value}"
        );
    }

    const BPF_MAP_TYPE_ARENA: u64 = 33;
    let arena_map = maps
        .iter()
        .find(|m| {
            m.get("map_type")
                .and_then(|t| t.as_u64())
                .is_some_and(|t| t == BPF_MAP_TYPE_ARENA)
        })
        .ok_or_else(|| {
            anyhow::anyhow!(
                "dump has no BPF_MAP_TYPE_ARENA (33) map — scx-ktstr \
                 declares one via lib/arena_map.h, so either the dump \
                 path filtered it out, the map enumeration missed it, \
                 or the scheduler failed to load the arena. Got {} \
                 maps total: {value}",
                maps.len()
            )
        })?;
    // Arena map JSON shape:
    //   {"map_type": 33, "arena": {"pages": [{"user_addr": N,
    //    "bytes": [u8, u8, ...]}, ...], "declared_pages": N,
    //    "truncated"?: bool, "span_capped"?: bool}, ...}.
    // `pages` is `skip_serializing_if = "Vec::is_empty"`, so an empty
    // page set means the key is absent from JSON entirely. Both flags
    // and the inner `arena` object can also be absent depending on
    // the snapshot path.
    let arena_field = arena_map.get("arena").ok_or_else(|| {
        anyhow::anyhow!(
            "arena map present but `arena` field absent — render_map's \
             BPF_MAP_TYPE_ARENA arm did not populate ArenaSnapshot \
             (likely arena_offsets was None: kernel BTF lacks \
             struct bpf_arena, or BpfArenaOffsets::from_btf failed). \
             arena map JSON: {arena_map}"
        )
    })?;

    // ArenaSnapshot.pages uses `skip_serializing_if = "Vec::is_empty"`,
    // so an empty pages vector is absent from JSON entirely, not present
    // as `[]`. Treat both shapes (absent + present-but-empty) as the
    // same "no pages captured" failure mode — the bar is non-empty.
    let arena_pages: &[serde_json::Value] = match arena_field.get("pages") {
        Some(p) => p.as_array().map(|a| a.as_slice()).ok_or_else(|| {
            anyhow::anyhow!(
                "arena.pages is present but not an array — \
                 ArenaSnapshot serde shape changed. arena field: {arena_field}"
            )
        })?,
        None => &[],
    };
    if arena_pages.is_empty() {
        anyhow::bail!(
            "arena.pages is empty (absent or zero-length) — snapshot_arena \
             returned no pages. Either the PTE walker found no mapped \
             pgoffs (kern_vm translation failed for every page), \
             max_entries is 0, or scx_task_alloc never ran on any task \
             (alloc_count={alloc_count_int}). arena field: {arena_field}"
        );
    }

    // declared_pages sanity: must be > 0 (proves max_entries was
    // readable from `struct bpf_map` at dump time) and must be at
    // least as large as the captured page set (a captured page count
    // exceeding the declared capacity would mean the walker over-
    // walked or the snapshot accidentally accumulated stale entries).
    // Absent key falls back to 0 — the default for ArenaSnapshot.
    let declared_pages = arena_field
        .get("declared_pages")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    if declared_pages == 0 {
        anyhow::bail!(
            "arena.declared_pages is 0 (or absent) — \
             ArenaWalkPlan computed a zero-page span, meaning \
             `info.max_entries` was unreadable or zero at dump time. \
             arena field: {arena_field}"
        );
    }
    if (arena_pages.len() as u64) > declared_pages {
        anyhow::bail!(
            "arena.pages.len() ({}) exceeds declared_pages ({}) — \
             walker invariant violated; ArenaWalkPlan should never \
             emit more pages than the declared capacity. arena field: \
             {arena_field}",
            arena_pages.len(),
            declared_pages
        );
    }

    // Magic-byte scan: the LE bytes of KTSTR_ARENA_MAGIC must appear
    // in at least one captured page. Each ArenaPage.bytes serializes
    // as a JSON array of u8 (serde's default Vec<u8> serialization);
    // collect into Vec<u8> per page and use windows() to find the
    // 8-byte LE pattern. The constant is derived from the u64 source
    // value so a future change to KTSTR_ARENA_MAGIC in main.bpf.c
    // only needs the matching update here, no manual byte-reversal.
    //
    // Per-page scan (vs. cross-page concatenation): scx-ktstr's
    // sdt_alloc slot layout is 24 bytes (8-byte sdt_data header +
    // 16-byte ktstr_arena_ctx) and slots align within pages — no slot
    // crosses a page boundary, so the magic u64 is always contiguous
    // within a single captured page.
    const KTSTR_ARENA_MAGIC: u64 = 0xDEADBEEFCAFEBABE;
    const KTSTR_ARENA_MAGIC_LE: [u8; 8] = KTSTR_ARENA_MAGIC.to_le_bytes();
    let mut magic_hits = 0usize;
    let mut total_bytes = 0usize;
    for page in arena_pages {
        let bytes = page
            .get("bytes")
            .and_then(|b| b.as_array())
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "arena page missing `bytes` array — \
                     ArenaPage serde shape changed. page: {page}"
                )
            })?;
        // Each element must be a u8; collect into a flat Vec<u8>.
        let raw: Vec<u8> = bytes
            .iter()
            .map(|v| {
                v.as_u64()
                    .and_then(|n| u8::try_from(n).ok())
                    .ok_or_else(|| anyhow::anyhow!("arena page byte is not a u8 (0..=255): {v}"))
            })
            .collect::<Result<Vec<u8>>>()?;
        total_bytes += raw.len();
        if raw
            .windows(KTSTR_ARENA_MAGIC_LE.len())
            .any(|w| w == KTSTR_ARENA_MAGIC_LE)
        {
            magic_hits += 1;
        }
    }
    if magic_hits == 0 {
        anyhow::bail!(
            "no arena page contained KTSTR_ARENA_MAGIC \
             (0x{KTSTR_ARENA_MAGIC:016x}) — pages were captured but \
             contain no live-stamped data. Most diagnostic case: \
             alloc_count={alloc_count_int} (>0 means the alloc path \
             ran, so the magic stamp was lost OR the walker captured \
             the wrong pages); alloc_count=0 would mean no tasks \
             were initialized in the first place. {} pages totalling \
             {} bytes scanned.",
            arena_pages.len(),
            total_bytes
        );
    }

    // Confirming detail so the test log shows the captured values.
    eprintln!(
        "scheduler .bss render OK: stall={stall_int}, \
         ktstr_alloc_count={alloc_count_int}, members={}, vcpu_regs={} \
         ({} populated with non-zero IP), arena pages={} ({total_bytes} \
         bytes, {magic_hits} with KTSTR_ARENA_MAGIC, \
         declared_pages={declared_pages})",
        members.len(),
        vcpu_regs.len(),
        populated_with_ip.len(),
        arena_pages.len(),
    );

    Ok(())
}

#[ktstr::distributed_slice(ktstr::test_support::KTSTR_TESTS)]
#[linkme(crate = ktstr::linkme)]
static __KTSTR_ENTRY_FAILURE_DUMP_BSS: ktstr::test_support::KtstrTestEntry =
    ktstr::test_support::KtstrTestEntry {
        name: "failure_dump_renders_bss_fields",
        func: scenario_failure_dump_renders_bss_fields,
        scheduler: &KTSTR_SCHED,
        // --stall-after=1 makes the scheduler return early from
        // dispatch after 1 second of operation, triggering
        // SCX_EXIT_ERROR_STALL via the kernel watchdog.
        extra_sched_args: &["--stall-after=1"],
        // Watchdog timeout snug to the stall budget so the run
        // teardown stays under the test duration.
        watchdog_timeout: std::time::Duration::from_secs(3),
        duration: std::time::Duration::from_secs(10),
        // expect_err inverts the SCX_EXIT_ERROR_STALL (the expected
        // outcome of --stall-after=1) to PASS. The real render
        // assertions live in `check_bss_dump`, a post_vm_unconditional
        // callback whose Err is a hard FAIL via PostVmAssertionFailure —
        // so a wrong .bss/arena render fails the test even though the
        // stall itself is inverted.
        expect_err: true,
        post_vm_unconditional: Some(check_bss_dump),
        ..ktstr::test_support::KtstrTestEntry::DEFAULT
    };

/// Asserts that the host-side dump renders EVERY entry of a
/// multi-entry plain `BPF_MAP_TYPE_ARRAY` (not just key 0).
///
/// scx-ktstr declares `ktstr_array_fixture` (16 entries of
/// `struct ktstr_array_value { magic, key_echo, _pad }`) and stamps
/// each slot in `ktstr_init` at attach. After the `--stall-after=1`
/// freeze, the dump's renderer walks the ARRAY per key
/// (src/monitor/dump/render_map.rs) into `array_entries`. This is the
/// e2e for the "render all entries" change — before it, the renderer
/// surfaced only key 0 with an "only key 0 shown" error.
///
/// User-facing bar: an operator inspecting a stalled scheduler sees
/// every cell of a multi-entry array, with the right per-key values —
/// proof the per-key stride math read the correct entry, not key 0
/// repeated.
///
/// The failure dump is written by the freeze coordinator in the HOST
/// process; the guest is a separate process and cannot read it. The
/// guest body only triggers the stall that produces the dump — the
/// assertions run in the [`check_array_entries_dump`]
/// `post_vm_unconditional` callback, which reads the host sidecar path.
#[ktstr_test(
    scheduler = KTSTR_SCHED,
    extra_sched_args = ["--stall-after=1"],
    watchdog_timeout_s = 3,
    duration_s = 10,
    expect_err = true,
    post_vm_unconditional = check_array_entries_dump,
)]
fn failure_dump_renders_array_entries(ctx: &ktstr::scenario::Ctx) -> Result<AssertResult> {
    // Wait for the freeze coordinator to ADOPT its accessor before the
    // --stall-after=1 stall fires; otherwise the dump renders placeholder
    // ARRAY values (the accessor is built async and may not be adopted by
    // the time an early stall freezes the VM). This host-side gate is what
    // makes the no-exit-stall dump render real per-key values.
    await_accessor_ready();
    // Trigger the --stall-after=1 SCX_EXIT_ERROR_STALL that drives the
    // freeze coordinator to capture the dump; the host-side
    // `check_array_entries_dump` post_vm callback does the assertions
    // (the guest is a separate process and cannot read the
    // host-written dump).
    let steps = vec![Step {
        setup: vec![ctx.cgroup_def("cg_0")].into(),
        ops: vec![],
        hold: HoldSpec::FULL,
    }];
    execute_steps(ctx, steps)
}

/// Host-side post_vm assertion for `failure_dump_renders_array_entries`.
/// Reads the freeze coordinator's dump via `read_failure_dump` (the
/// HOST sidecar path) and verifies the multi-entry ARRAY fixture
/// rendered every key. Runs unconditionally (the run "fails"
/// via the expected stall, so a conditional post_vm could be
/// suppressed); the framework attaches the `PostVmAssertionFailure`
/// marker to this callback's Err so a wrong render stays a hard FAIL
/// even though expect_err inverts the stall itself to PASS.
fn check_array_entries_dump(result: &VmResult) -> Result<()> {
    let value = read_failure_dump(result)?;

    let schema = value
        .get("schema")
        .and_then(|s| s.as_str())
        .ok_or_else(|| anyhow::anyhow!("dump JSON missing top-level `schema` field"))?;
    anyhow::ensure!(
        schema == SCHEMA_SINGLE,
        "happy-path dump must carry schema=SCHEMA_SINGLE ({SCHEMA_SINGLE:?}); got {schema}"
    );

    let maps = value
        .get("maps")
        .and_then(|m| m.as_array())
        .ok_or_else(|| anyhow::anyhow!("dump JSON missing top-level `maps` array"))?;

    // Match by shape, not name: the kernel truncates the declared map
    // name to 15 chars (BPF_OBJ_NAME_LEN), so pin on map_type==2
    // (BPF_MAP_TYPE_ARRAY) AND max_entries==KTSTR_ARRAY_ENTRIES, which
    // uniquely identifies `ktstr_array_fixture` — the only plain ARRAY
    // with max_entries>1 in the repo (the .bss/.data/.rodata global
    // sections are ARRAY-typed but max_entries==1).
    const KTSTR_ARRAY_ENTRIES: u64 = 16;
    const KTSTR_ARRAY_MAGIC: u64 = 0xABCDEF0123456789;
    const BPF_MAP_TYPE_ARRAY: u64 = 2;
    let array_map = maps
        .iter()
        .find(|m| {
            m.get("map_type").and_then(|t| t.as_u64()) == Some(BPF_MAP_TYPE_ARRAY)
                && m.get("max_entries").and_then(|e| e.as_u64()) == Some(KTSTR_ARRAY_ENTRIES)
        })
        .ok_or_else(|| {
            anyhow::anyhow!(
                "dump has no multi-entry ARRAY fixture (map_type=2, max_entries=16) — \
                 the scx-ktstr ktstr_array_fixture map is missing from the IDR walk \
                 or was mis-typed by the renderer. maps={}: {value}",
                maps.len()
            )
        })?;

    let name = array_map.get("name").and_then(|n| n.as_str()).unwrap_or("");
    anyhow::ensure!(
        name.contains("array"),
        "ARRAY fixture name should contain `array` (kernel-truncated \
         ktstr_array_fixture); got {name:?}"
    );

    // The whole point of the change: a multi-entry ARRAY populates
    // `array_entries`, NOT the single-entry `value`.
    anyhow::ensure!(
        array_map.get("value").is_none_or(|v| v.is_null()),
        "multi-entry ARRAY must populate `array_entries`, not the single-entry \
         `value`: {array_map}"
    );
    // 16 < MAX_ARRAY_KEYS (4096) and every entry is mapped → no error.
    anyhow::ensure!(
        array_map.get("error").is_none_or(|e| e.is_null()),
        "ARRAY render must be error-free for 16 mapped entries: {array_map}"
    );

    let entries = array_map
        .get("array_entries")
        .and_then(|a| a.as_array())
        .ok_or_else(|| {
            anyhow::anyhow!(
                "ARRAY fixture has no `array_entries` — the prior 'only key 0' \
                 behaviour would leave this empty with `value` set: {array_map}"
            )
        })?;
    anyhow::ensure!(
        entries.len() == KTSTR_ARRAY_ENTRIES as usize,
        "expected {KTSTR_ARRAY_ENTRIES} array_entries (every key rendered), got {}: \
         {array_map}",
        entries.len()
    );

    // Each entry i: key == i, value is a struct whose `magic` member is
    // KTSTR_ARRAY_MAGIC and whose `key_echo` echoes the key. Proves the
    // per-key stride read the correct entry (not key 0 repeated, not a
    // wrong-stride overlap).
    for (i, entry) in entries.iter().enumerate() {
        let key = entry
            .get("key")
            .and_then(|k| k.as_u64())
            .ok_or_else(|| anyhow::anyhow!("array_entries[{i}] missing u32 `key`: {entry}"))?;
        anyhow::ensure!(
            key == i as u64,
            "array_entries[{i}].key == {key}, expected {i} (entries must be key-ordered)"
        );

        let val = entry.get("value").ok_or_else(|| {
            anyhow::anyhow!("array_entries[{i}] has no value (unreadable key?): {entry}")
        })?;
        anyhow::ensure!(
            val.get("kind").and_then(|k| k.as_str()) == Some("struct"),
            "array_entries[{i}].value must render as a struct: {val}"
        );
        let members = val
            .get("members")
            .and_then(|m| m.as_array())
            .ok_or_else(|| {
                anyhow::anyhow!("array_entries[{i}].value struct has no members: {val}")
            })?;
        let member_u64 = |nm: &str| -> Option<u64> {
            members
                .iter()
                .find(|m| m.get("name").and_then(|n| n.as_str()) == Some(nm))
                .and_then(|m| m.get("value"))
                .and_then(|v| v.get("value"))
                .and_then(|v| v.as_u64())
        };
        let magic = member_u64("magic");
        let key_echo = member_u64("key_echo");
        anyhow::ensure!(
            magic == Some(KTSTR_ARRAY_MAGIC),
            "array_entries[{i}].magic must be KTSTR_ARRAY_MAGIC (0x{KTSTR_ARRAY_MAGIC:x}); \
             got {magic:?}: {val}"
        );
        anyhow::ensure!(
            key_echo == Some(i as u64),
            "array_entries[{i}].key_echo must echo the key {i}; got {key_echo:?}: {val}"
        );
    }

    eprintln!(
        "multi-entry ARRAY fixture `{name}` rendered all {} keys with correct \
         magic + echoed key",
        entries.len(),
    );
    Ok(())
}

/// Asserts that the freeze coordinator's host-side capture modules
/// (`crate::vmm::capture_scx`, `crate::vmm::capture_tasks`,
/// `crate::vmm::capture_numa`) populate
/// [`crate::monitor::dump::FailureDumpReport`] with non-default
/// data when the `--stall-after=1` SCX_EXIT_ERROR_STALL path
/// triggers a freeze.
///
/// User-facing test bar (per project memory): "captures must
/// always produce data" — when scx-ktstr is loaded and tasks are
/// runnable, the dump should carry per-CPU rq->scx state, at
/// least the global DSQ, the scx_sched scalar state, and at
/// least one task enrichment record. NUMA stats either populate
/// (CONFIG_NUMA=y kernel) or carry the diagnostic reason that
/// explains why they didn't.
///
/// Distinct from `scenario_failure_dump_renders_bss_fields`:
/// that test exercises the BTF / arena render path; this one
/// exercises the live-walker captures wired into freeze_coord.
fn scenario_failure_dump_renders_capture_modules(
    ctx: &ktstr::scenario::Ctx,
) -> Result<AssertResult> {
    // Adopt the accessor before the --stall-after=1 freeze so the dump
    // renders a full report (live walker captures present); the
    // host-side `check_capture_dump` callback does the assertions.
    await_accessor_ready();
    let steps = vec![Step {
        setup: vec![ctx.cgroup_def("cg_0")].into(),
        ops: vec![],
        hold: HoldSpec::FULL,
    }];
    execute_steps(ctx, steps)
}

/// Host-side post_vm assertion for `failure_dump_renders_capture_modules`.
/// Reads the dump via `read_failure_dump` and verifies the freeze
/// coordinator's live-walker captures (per-CPU rq->scx state, DSQs, the
/// scx_sched scalar, task enrichments, NUMA stats). Runs unconditionally;
/// its Err is a hard FAIL via the framework's `PostVmAssertionFailure`
/// marker even though `expect_err` inverts the stall itself to PASS.
fn check_capture_dump(result: &VmResult) -> Result<()> {
    let value = read_failure_dump(result)?;

    // The freeze coordinator captures one `vcpu_regs` slot per booted
    // vCPU (BSP + APs), so its length is the authoritative online-CPU
    // count for cross-checking the per-CPU walker below — the post_vm
    // callback has no `Ctx::topo` to read it from. read_failure_dump has
    // already gated out the no-maps partial/placeholder dumps, so a full
    // dump here always carries vcpu_regs.
    let num_cpus = value
        .get("vcpu_regs")
        .and_then(|v| v.as_array())
        .map(|a| a.len())
        .ok_or_else(|| {
            anyhow::anyhow!(
                "dump JSON missing `vcpu_regs` — cannot determine the expected \
                 per-CPU count for the walker cross-check. Full JSON: {value}"
            )
        })?;

    // -- scx_walker capture (rq_scx_states / dsq_states / scx_sched_state) --
    //
    // The walker pushes one entry per CPU whose rq + scx_rq + task
    // sub-group offsets resolved. With CONFIG_SCHED_CLASS_EXT=y and
    // a debug-info kernel (per ktstr.kconfig) every CPU resolves, so
    // the vec length must equal num_cpus. Surface the absent /
    // partial state diagnostic when the walker fails so the failure
    // mode is identifiable from the dump alone.
    if let Some(reason) = value.get("scx_walker_unavailable").and_then(|r| r.as_str()) {
        anyhow::bail!(
            "scx_walker_unavailable={reason:?} — capture_scx::build returned \
             None or the walker reached no state. Captures must always \
             produce data when scx-ktstr is loaded. Full JSON: {value}"
        );
    }
    let rq_scx_states = value
        .get("rq_scx_states")
        .and_then(|s| s.as_array())
        .ok_or_else(|| {
            anyhow::anyhow!(
                "dump JSON missing `rq_scx_states` array — capture_scx \
                 wiring did not populate the field. Full JSON: {value}"
            )
        })?;
    if rq_scx_states.len() != num_cpus {
        anyhow::bail!(
            "rq_scx_states.len()={} != num_cpus={num_cpus} (num_cpus = dump \
             vcpu_regs.len(), one slot per booted vCPU — freeze_coord \
             collect_vcpu_regs). walk_rq_scx silently skips a CPU on \
             sub-group offset / per-CPU rq translate failure, so fewer \
             entries means a skipped CPU; more means a walker over-count. \
             Full rq_scx_states: {rq_scx_states:?}",
            rq_scx_states.len(),
        );
    }
    // At least one CPU must show evidence of scheduler activity:
    // either a non-zero `nr_running` (tasks queued on rq->scx) or a
    // non-zero `flags` (any scx_rq.flags bit set). Both being zero
    // across every CPU would mean the walker ran but read pre-init
    // state — an empty walker is no better than no walker.
    let any_active = rq_scx_states.iter().any(|s| {
        let nr = s.get("nr_running").and_then(|v| v.as_u64()).unwrap_or(0);
        let flags = s.get("flags").and_then(|v| v.as_u64()).unwrap_or(0);
        nr > 0 || flags != 0
    });
    if !any_active {
        anyhow::bail!(
            "no rq_scx_states entry has nr_running>0 OR flags!=0 — every \
             CPU's rq->scx scalar read came back zero, meaning the walker \
             ran but every per-CPU scx_rq is empty. Either no scx tasks \
             were ever runnable or the rq_pa translate produced wrong \
             addresses. Full rq_scx_states: {rq_scx_states:?}"
        );
    }

    let dsq_states = value
        .get("dsq_states")
        .and_then(|s| s.as_array())
        .ok_or_else(|| {
            anyhow::anyhow!(
                "dump JSON missing `dsq_states` array — capture_scx \
                 wiring did not populate the field. Full JSON: {value}"
            )
        })?;
    if dsq_states.is_empty() {
        anyhow::bail!(
            "dsq_states is empty — walk_dsqs reached no DSQs. The \
             global DSQ (SCX_DSQ_GLOBAL per-node) must always be \
             reachable when *scx_root is non-null. Full JSON: {value}"
        );
    }

    if value.get("scx_sched_state").is_none()
        || value.get("scx_sched_state").is_some_and(|v| v.is_null())
    {
        anyhow::bail!(
            "scx_sched_state is absent or null — read_scx_sched_state \
             returned None. *scx_root was unreadable or the BTF offsets \
             didn't resolve. Full JSON: {value}"
        );
    }

    // -- task_enrichments capture --
    //
    // The runnable_list walker pushes one entry per task on each
    // CPU's rq->scx.runnable_list. With workers_per_cgroup=2 driving
    // active workloads, at least one task should be runnable at the
    // freeze instant. An empty enrichment vec when scx-ktstr is
    // loaded means the walker missed every task — a real defect.
    if let Some(reason) = value
        .get("task_enrichments_unavailable")
        .and_then(|r| r.as_str())
    {
        anyhow::bail!(
            "task_enrichments_unavailable={reason:?} — capture_tasks::build \
             returned None or the walker yielded zero tasks. Captures \
             must always produce data when scx tasks are runnable. Full \
             JSON: {value}"
        );
    }
    let task_enrichments = value
        .get("task_enrichments")
        .and_then(|s| s.as_array())
        .ok_or_else(|| {
            anyhow::anyhow!(
                "dump JSON missing `task_enrichments` array — \
                 capture_tasks wiring did not populate the field. \
                 Full JSON: {value}"
            )
        })?;
    if task_enrichments.is_empty() {
        anyhow::bail!(
            "task_enrichments is empty — runnable_list walker found no \
             tasks. With workers_per_cgroup>0 driving load, at least \
             one task must be runnable at freeze time. Full JSON: {value}"
        );
    }
    // At least one enrichment must carry an identity that proves the
    // task_struct read produced live data: non-empty comm AND pid > 0.
    // pid==0 is the swapper / idle task — possible but not proof of
    // liveness; insist on a real userspace task slip through the
    // walker. comm is null-terminated and skipped when zero-length
    // wouldn't be skip-serialized but a zero-byte read would surface
    // as an empty string, not absent.
    let has_real_task = task_enrichments.iter().any(|t| {
        let pid = t.get("pid").and_then(|v| v.as_i64()).unwrap_or(0);
        let comm = t.get("comm").and_then(|v| v.as_str()).unwrap_or("");
        pid > 0 && !comm.is_empty()
    });
    if !has_real_task {
        anyhow::bail!(
            "no task_enrichment entry has pid>0 AND non-empty comm — \
             every task_struct read produced pid<=0 or empty comm, \
             meaning the slab translate fell back to garbage memory. \
             Full task_enrichments: {task_enrichments:?}"
        );
    }

    // -- per_node_numa capture --
    //
    // ktstr.kconfig sets CONFIG_NUMA=y, so capture_numa::build runs.
    // With nr_nodes=1 (default topology) it walks node 0 and emits
    // one PerNodeNumaStats row. If for any reason the walker bails
    // (symbol absent, BTF offsets unresolved, pgdat translate failed),
    // per_node_numa stays empty and per_node_numa_unavailable carries
    // the diagnostic. Both shapes are acceptable; what's NOT
    // acceptable is the empty vec without a diagnostic.
    let per_node_numa = value
        .get("per_node_numa")
        .and_then(|s| s.as_array())
        .map(|a| a.as_slice())
        .unwrap_or(&[]);
    let per_node_numa_unavailable = value
        .get("per_node_numa_unavailable")
        .and_then(|r| r.as_str());
    if per_node_numa.is_empty() && per_node_numa_unavailable.is_none() {
        anyhow::bail!(
            "per_node_numa is empty AND per_node_numa_unavailable is \
             absent — the dump pipeline broke its own contract that \
             one of the two must be populated. Full JSON: {value}"
        );
    }

    eprintln!(
        "capture-module data OK: rq_scx_states.len()={} (num_cpus={num_cpus}), \
         dsq_states.len()={}, scx_sched_state present, \
         task_enrichments.len()={}, per_node_numa.len()={} \
         (unavailable={:?})",
        rq_scx_states.len(),
        dsq_states.len(),
        task_enrichments.len(),
        per_node_numa.len(),
        per_node_numa_unavailable,
    );

    Ok(())
}

#[ktstr::distributed_slice(ktstr::test_support::KTSTR_TESTS)]
#[linkme(crate = ktstr::linkme)]
static __KTSTR_ENTRY_FAILURE_DUMP_CAPTURES: ktstr::test_support::KtstrTestEntry =
    ktstr::test_support::KtstrTestEntry {
        name: "failure_dump_renders_capture_modules",
        func: scenario_failure_dump_renders_capture_modules,
        scheduler: &KTSTR_SCHED,
        // --stall-after=1 makes the scheduler return early from
        // dispatch after 1 second of operation, triggering
        // SCX_EXIT_ERROR_STALL via the kernel watchdog.
        extra_sched_args: &["--stall-after=1"],
        // Watchdog timeout snug to the stall budget so the run
        // teardown stays under the test duration.
        watchdog_timeout: std::time::Duration::from_secs(3),
        duration: std::time::Duration::from_secs(10),
        // expect_err inverts the SCX_EXIT_ERROR_STALL (the expected
        // outcome of --stall-after=1) to PASS. The real capture
        // assertions live in `check_capture_dump`, a
        // post_vm_unconditional callback whose Err is a hard FAIL via
        // PostVmAssertionFailure — so a missing/empty capture fails the
        // test even though the stall itself is inverted.
        expect_err: true,
        post_vm_unconditional: Some(check_capture_dump),
        ..ktstr::test_support::KtstrTestEntry::DEFAULT
    };

/// Asserts that the failure dump's `probe_counters` field captures
/// non-zero `trigger_count` after an SCX_EXIT_ERROR_STALL fires.
///
/// User-facing test bar (per project memory): the BPF probe's
/// per-CPU diagnostic counters must surface in the failure dump
/// with values that prove each tracepoint actually fired during
/// the run. After the per-CPU conversion landed (replacing N
/// shared-global counters with a `[MAX_CPUS][KTSTR_PCPU_NR]`
/// 2D array in `.bss`), this test pins:
///   1. `probe_counters` is present and structured (not absent /
///      null in the JSON);
///   2. `probe_counters.trigger_count > 0` — the
///      `tp_btf/sched_ext_exit` handler fired at least once during
///      the stall, which proves the per-CPU sum reaches the host;
///   3. `probe_counters.probe_count > 0` — kprobes attached and
///      fired (confirms the host-side sum walks the array, since
///      a stub-empty array would produce 0 even on a working run).
///
/// Distinct from `scenario_failure_dump_renders_bss_fields` (which
/// asserts the scheduler's own `.bss` BTF render) and
/// `scenario_failure_dump_renders_capture_modules` (which asserts
/// the live walker captures): this test exercises the host-side
/// `decode_probe_counters_snapshot` reader specifically.
fn scenario_failure_dump_renders_probe_counters(
    ctx: &ktstr::scenario::Ctx,
) -> Result<AssertResult> {
    // Adopt the accessor before the --stall-after=1 freeze so the dump
    // renders a full report (probe_counters present); the host-side
    // `check_probe_dump` callback does the assertions.
    await_accessor_ready();
    let steps = vec![Step {
        setup: vec![ctx.cgroup_def("cg_0")].into(),
        ops: vec![],
        hold: HoldSpec::FULL,
    }];
    execute_steps(ctx, steps)
}

/// Host-side post_vm assertion for `failure_dump_renders_probe_counters`.
/// Reads the dump via `read_failure_dump` and verifies the probe's
/// per-CPU `.bss` counters surfaced (non-zero `trigger_count` and
/// `probe_count`). Runs unconditionally; its Err is a hard FAIL via the
/// framework's `PostVmAssertionFailure` marker even though `expect_err`
/// inverts the stall itself to PASS.
fn check_probe_dump(result: &VmResult) -> Result<()> {
    let value = read_failure_dump(result)?;

    // `probe_counters` is `skip_serializing_if = "Option::is_none"`,
    // so its absence in the JSON means the host-side decoder
    // returned None. That's a regression — when the probe has
    // attached and fired (which the stall scenario guarantees),
    // the decoder must produce a populated struct.
    let probe_counters = value.get("probe_counters").ok_or_else(|| {
        anyhow::anyhow!(
            "dump JSON missing `probe_counters` field — \
             decode_probe_counters_snapshot returned None. \
             Probe `.bss` map absent, BTF lookup failed, or the \
             `ktstr_pcpu_counters` array offset didn't resolve. \
             Full JSON: {value}"
        )
    })?;
    if probe_counters.is_null() {
        anyhow::bail!(
            "`probe_counters` is null — decoder ran but produced None; \
             same prerequisite-missing failure modes as above. \
             Full JSON: {value}"
        );
    }

    // `trigger_count` is the structural assertion — a stall
    // scenario is guaranteed to fire `tp_btf/sched_ext_exit`
    // (the SCX kernel emits SCX_EXIT_ERROR_STALL through the
    // tracepoint), so a zero value here means either (a) the
    // probe didn't attach the trigger handler, (b) the handler
    // fired but the per-CPU slot bump didn't land, or (c) the
    // host-side cross-CPU sum walked the wrong slot index.
    let trigger_count = probe_counters
        .get("trigger_count")
        .and_then(|v| v.as_u64())
        .ok_or_else(|| {
            anyhow::anyhow!(
                "`probe_counters.trigger_count` missing or non-numeric — \
                 ProbeBssCounters serde shape changed. \
                 probe_counters: {probe_counters}"
            )
        })?;
    if trigger_count == 0 {
        anyhow::bail!(
            "`probe_counters.trigger_count == 0` — `tp_btf/sched_ext_exit` \
             never fired (or the per-CPU slot didn't increment). The stall \
             scenario must produce at least one tracepoint fire. \
             probe_counters: {probe_counters}"
        );
    }

    // `probe_count` cross-validates the array walk: the kprobe
    // handler is attached to multiple kernel functions (sched
    // entry / dispatch path) and fires throughout the run, so a
    // healthy stall scenario produces hundreds-to-millions of
    // fires. A non-zero value here proves the host-side reader
    // walked the per-CPU slots (rather than reading a stub-zero
    // value from index 0 of an empty array).
    let probe_count = probe_counters
        .get("probe_count")
        .and_then(|v| v.as_u64())
        .ok_or_else(|| {
            anyhow::anyhow!(
                "`probe_counters.probe_count` missing or non-numeric — \
                 ProbeBssCounters serde shape changed. \
                 probe_counters: {probe_counters}"
            )
        })?;
    if probe_count == 0 {
        anyhow::bail!(
            "`probe_counters.probe_count == 0` — kprobe path never fired \
             across the run. Either probe attach failed, ktstr_enabled \
             never flipped to true, or the host-side sum walked the wrong \
             slot index. probe_counters: {probe_counters}"
        );
    }

    eprintln!(
        "probe_counters render OK: trigger_count={trigger_count}, \
         probe_count={probe_count} (per-CPU sum walked across CPUs in \
         `.bss` `ktstr_pcpu_counters` array)"
    );

    Ok(())
}

#[ktstr::distributed_slice(ktstr::test_support::KTSTR_TESTS)]
#[linkme(crate = ktstr::linkme)]
static __KTSTR_ENTRY_FAILURE_DUMP_PROBE_COUNTERS: ktstr::test_support::KtstrTestEntry =
    ktstr::test_support::KtstrTestEntry {
        name: "failure_dump_renders_probe_counters",
        func: scenario_failure_dump_renders_probe_counters,
        scheduler: &KTSTR_SCHED,
        // --stall-after=1 fires SCX_EXIT_ERROR_STALL on watchdog
        // timeout. The probe's tp_btf/sched_ext_exit handler
        // bumps `KTSTR_PCPU_TRIGGER_COUNT` on every fire, so a
        // single stall produces a non-zero cross-CPU sum.
        extra_sched_args: &["--stall-after=1"],
        // Watchdog timeout snug to the stall budget so the run
        // teardown stays under the test duration.
        watchdog_timeout: std::time::Duration::from_secs(3),
        duration: std::time::Duration::from_secs(10),
        // expect_err inverts the SCX_EXIT_ERROR_STALL (the expected
        // outcome of --stall-after=1) to PASS. The real counter
        // assertions live in `check_probe_dump`, a
        // post_vm_unconditional callback whose Err is a hard FAIL via
        // PostVmAssertionFailure — so a missing/zero counter fails the
        // test even though the stall itself is inverted.
        expect_err: true,
        post_vm_unconditional: Some(check_probe_dump),
        ..ktstr::test_support::KtstrTestEntry::DEFAULT
    };
