use super::*;

// ---- Datasec rendering ------------------------------------------
//
// The renderer recognises `BTF_KIND_DATASEC` (the value type
// libbpf assigns to a global-section ARRAY map like `.bss`) and
// walks its `VarSecinfo` entries to render each variable. Before
// this support landed the renderer returned `Unsupported`, so a
// failure dump's `.bss` map showed an opaque hex dump instead of
// `stall=1, crash=0, ...`.
//
// The probe BPF object built by `build.rs` contains a known
// `.bss` Datasec (declared via the `volatile u32
// ktstr_err_exit_detected = 0;` latch, the per-CPU counter
// array `ktstr_pcpu_counters`, the sticky `ktstr_last_trigger_ts`
// / `ktstr_exit_*` snapshot vars, and the `ktstr_miss_log` /
// `ktstr_miss_log_idx` log buffer in `src/bpf/probe.bpf.c`). The
// tests below load that BTF directly via `load_btf_from_path`
// (which falls back to goblin's `.BTF` ELF section parse for
// non-vmlinux files) and exercise the Datasec render path
// against it. Hard-fail on a missing probe.o because build.rs
// always produces it; a silent skip would hide the regression
// the test is designed to catch.

/// Locate the `.bss` Datasec type id in the probe BTF.
/// `resolve_ids_by_name(".bss")` returns the list of type ids
/// named `.bss`; libbpf normally emits exactly one
/// `BTF_KIND_DATASEC`, but the resolver returns `Vec<u32>`, so we
/// resolve each id via `resolve_type_by_id` and keep the Datasec
/// variant. Returns `(btf, ds_id)` or panics if the
/// build fixture is missing.
fn load_probe_btf_and_bss_id() -> (Btf, u32) {
    let probe_obj = std::path::PathBuf::from(env!("OUT_DIR")).join("probe.o");
    let btf = crate::monitor::btf_offsets::load_btf_from_path(&probe_obj).unwrap_or_else(|e| {
        panic!(
            "load_btf_from_path({}) failed: {e}. \
                 build.rs always produces probe.o; a missing or \
                 unparseable artifact means the build pipeline is \
                 broken.",
            probe_obj.display()
        )
    });
    let ids = btf
        .resolve_ids_by_name(".bss")
        .expect("probe BTF must carry a `.bss` BTF_KIND_DATASEC");
    // Pick the first id that resolves to a Datasec — there
    // should be exactly one, but we don't blow up if libbpf
    // ever emits something else under the `.bss` name.
    for &id in &ids {
        if let Ok(Type::Datasec(_)) = btf.resolve_type_by_id(id) {
            return (btf, id);
        }
    }
    panic!("probe BTF has `.bss` ids {ids:?} but none resolve to BTF_KIND_DATASEC");
}

#[test]
fn render_datasec_emits_struct_with_named_variables() {
    let (btf, bss_id) = load_probe_btf_and_bss_id();
    // Compute the section size by summing each VarSecinfo's
    // (offset + size) — libbpf-emitted Datasecs aren't laid out
    // contiguously (alignment + ordering), so the section's
    // total size is `max(offset + size)` across all entries.
    // Allocate a zeroed buffer of that size so every variable's
    // slice fits.
    let Type::Datasec(ds) = btf.resolve_type_by_id(bss_id).unwrap() else {
        panic!(".bss id did not resolve to Datasec");
    };
    let section_size = ds
        .variables
        .iter()
        .map(|v| v.offset() as usize + v.size())
        .max()
        .expect("`.bss` Datasec must have at least one variable");
    let bytes = vec![0u8; section_size];
    let rendered = render_value(&btf, bss_id, &bytes);

    // Datasec must render as a Struct (not Unsupported, not
    // Truncated — section_size matches the actual section
    // extent so no variable should overrun).
    let RenderedValue::Struct { type_name, members } = rendered else {
        panic!(
            "expected RenderedValue::Struct for Datasec, got something else \
             — Datasec dispatch in render_value_inner must be reachable"
        );
    };
    assert_eq!(
        type_name.as_deref(),
        Some(".bss"),
        "section name must surface as type_name"
    );
    // Variable names: probe.bpf.c declares
    // `ktstr_err_exit_detected` as a writable global. It MUST
    // appear in the rendered .bss members; the freeze
    // coordinator depends on this exact name being resolvable.
    let names: std::collections::HashSet<&str> = members.iter().map(|m| m.name.as_str()).collect();
    assert!(
        names.contains("ktstr_err_exit_detected"),
        "rendered .bss must contain `ktstr_err_exit_detected` \
         (the freeze latch). Found names: {names:?}"
    );
    // Diagnostic counters and snapshots are writable globals →
    // expected in .bss too. Pin every variable name a downstream
    // consumer relies on so a future addition that lands in
    // .bss without renderer coverage surfaces here. The hot
    // counters live in the `ktstr_pcpu_counters` per-CPU array
    // (host-side reader sums across CPUs); sticky snapshot vars
    // remain individual globals because they're written exactly
    // once per error-class exit.
    for required in [
        // Per-CPU diagnostic counter array — replaces the
        // previous N independent globals
        // (ktstr_trigger_count / ktstr_probe_count / etc.).
        "ktstr_pcpu_counters",
        // Sticky timestamp + scheduler-state snapshots written
        // by the tp_btf/sched_ext_exit handler at the first
        // error-class exit.
        "ktstr_last_trigger_ts",
        // SCX_EV_* counter snapshot taken by `scx_bpf_events`
        // at the first error-class exit. Surfaces the
        // system-wide event totals at fault time.
        "ktstr_exit_event_stats",
    ] {
        assert!(
            names.contains(required),
            "rendered .bss must contain `{required}` \
             diagnostic counter. Found names: {names:?}"
        );
    }
    // Each member's `value` must be a concrete renderable
    // type (Uint, Int, Array of int, etc.) — NOT Unsupported.
    // A zero byte buffer can't be Truncated for variables that
    // fit within section_size, so any Truncated result here
    // would indicate a slicing bug.
    for m in &members {
        assert!(
            !matches!(m.value, RenderedValue::Unsupported { .. }),
            "member {:?} rendered as Unsupported: {:?}",
            m.name,
            m.value
        );
        assert!(
            !matches!(m.value, RenderedValue::Truncated { .. }),
            "member {:?} rendered as Truncated despite section_size \
             buffer: {:?}",
            m.name,
            m.value
        );
    }
    // ktstr_err_exit_detected is `volatile u32 = 0;` — must
    // decode to Uint{bits:32, value:0} given a zero buffer.
    let latch = members
        .iter()
        .find(|m| m.name == "ktstr_err_exit_detected")
        .expect("latch member must be present (asserted above)");
    match &latch.value {
        RenderedValue::Uint { bits, value } => {
            assert_eq!(*bits, 32, "latch is u32 (32 bits)");
            assert_eq!(*value, 0, "latch was zeroed in the buffer");
        }
        other => panic!("expected Uint{{32,0}} for latch, got {other:?}"),
    }
}

#[test]
fn render_datasec_truncates_overrunning_variables() {
    // Feed a byte buffer that's too small to cover every
    // variable in the .bss Datasec. Variables whose
    // (offset + size) extends past the buffer must surface as
    // Truncated members, while variables that fit must render
    // normally. The Struct itself is NOT wrapped in Truncated
    // — the section name and the per-variable partial render
    // both stay intact.
    let (btf, bss_id) = load_probe_btf_and_bss_id();
    let Type::Datasec(ds) = btf.resolve_type_by_id(bss_id).unwrap() else {
        panic!(".bss id did not resolve to Datasec");
    };
    // Buffer that holds only the first variable (smallest
    // offset). Variables with higher offsets become Truncated.
    let min_var = ds
        .variables
        .iter()
        .min_by_key(|v| v.offset())
        .expect("`.bss` must have at least one variable");
    let buf_size = (min_var.offset() as usize) + min_var.size();
    let bytes = vec![0u8; buf_size];
    let rendered = render_value(&btf, bss_id, &bytes);

    let RenderedValue::Struct { type_name, members } = rendered else {
        panic!("expected RenderedValue::Struct even with short buffer");
    };
    assert_eq!(type_name.as_deref(), Some(".bss"));
    // At least one member must be Truncated (one of the
    // higher-offset variables). At least one member must be
    // non-Truncated (the variable at min_var's offset).
    let truncated_count = members
        .iter()
        .filter(|m| matches!(m.value, RenderedValue::Truncated { .. }))
        .count();
    let decoded_count = members.len() - truncated_count;
    assert!(
        decoded_count >= 1,
        "at least one member must decode (the variable at the smallest offset, \
         which fits in buf_size={buf_size})"
    );
    // If there is more than one variable in .bss (probe.bpf.c
    // declares several), the short buffer must produce at
    // least one Truncated. A single-variable .bss would have
    // truncated_count == 0, but our probe has multiple — so
    // assert > 0.
    if members.len() > 1 {
        assert!(
            truncated_count >= 1,
            "multi-variable .bss with short buffer must produce >= 1 \
             Truncated member; got 0 from {members:?}"
        );
    }
}

#[test]
fn render_datasec_empty_buffer_yields_struct_with_truncated_members() {
    // Edge case: zero-byte buffer for a non-empty Datasec.
    // Every variable must surface as Truncated rather than
    // crashing the renderer or returning the legacy
    // Unsupported.
    let (btf, bss_id) = load_probe_btf_and_bss_id();
    let rendered = render_value(&btf, bss_id, &[]);
    let RenderedValue::Struct { members, .. } = rendered else {
        panic!("expected Struct render even with empty buffer");
    };
    assert!(!members.is_empty(), "probe `.bss` Datasec is non-empty");
    for m in &members {
        // Every variable should report Truncated{ needed: var
        // size, had: 0, partial: ... }.
        assert!(
            matches!(m.value, RenderedValue::Truncated { had: 0, .. }),
            "member {:?} should be Truncated{{had:0}} for empty buffer, got {:?}",
            m.name,
            m.value
        );
    }
}

// ---- format_cpu_list -------------------------------------------
//
// Range-collapses a sorted CPU id list into a compact string.
// Pin every shape: empty, single, single-range, sparse (gaps),
// multiple ranges + singletons. Collapse rule: a run of >= 2
// consecutive ids renders as `start-end`; a singleton renders as
// just the id.

#[test]
fn format_cpu_list_empty_is_empty_string() {
    assert_eq!(format_cpu_list(&[]), "");
}

#[test]
fn format_cpu_list_single_element() {
    assert_eq!(format_cpu_list(&[5]), "5");
}

#[test]
fn format_cpu_list_contiguous_range() {
    assert_eq!(format_cpu_list(&[0, 1, 2, 3, 4]), "0-4");
}

#[test]
fn format_cpu_list_two_consecutive_collapses_to_range() {
    // The two-element edge: 0,1 must render as "0-1", not "0,1".
    // The end-of-loop flush has its own start==end branch, so a
    // pure-range input exercises the in-loop range emission.
    assert_eq!(format_cpu_list(&[0, 1]), "0-1");
}

#[test]
fn format_cpu_list_gaps_between_ranges() {
    // Mixed: range, singleton, range. Pins the comma-separator
    // and the singleton-formatting path inside the loop.
    assert_eq!(format_cpu_list(&[0, 1, 2, 5, 7, 8, 9]), "0-2,5,7-9");
}

#[test]
fn format_cpu_list_all_singletons() {
    assert_eq!(format_cpu_list(&[0, 2, 4, 6]), "0,2,4,6");
}

#[test]
fn format_cpu_list_first_range_then_singleton() {
    // Trailing singleton — covers the post-loop flush after a
    // mid-list range when the final cpu is alone.
    assert_eq!(format_cpu_list(&[0, 1, 5]), "0-1,5");
}

#[test]
fn format_cpu_list_singleton_then_trailing_range() {
    // Leading singleton followed by a closing range — the
    // post-loop flush emits the trailing range.
    assert_eq!(format_cpu_list(&[0, 3, 4, 5]), "0,3-5");
}

// ---- try_render_cpumask_bits -----------------------------------
//
// Reads u64 LE words from the byte slice and renders set bits as
// a CpuList. The function:
//   - returns None when fewer than 8 bytes are supplied
//     (insufficient for a single u64 word)
//   - returns Some(CpuList { "" }) when bytes are zeroed
//   - extracts bits across multiple words at offset = word*64+bit

#[test]
fn try_render_cpumask_bits_too_short_returns_none() {
    // Strictly less than 8 bytes can't form a u64 word.
    assert!(try_render_cpumask_bits(&[], u32::MAX).is_none());
    assert!(try_render_cpumask_bits(&[0u8; 1], u32::MAX).is_none());
    assert!(try_render_cpumask_bits(&[0u8; 7], u32::MAX).is_none());
}

#[test]
fn try_render_cpumask_bits_all_zero_yields_empty_list() {
    // 8 zeroed bytes: Some(CpuList { cpus: "" }) — the loop sees
    // word=0 and skips it; format_cpu_list of an empty Vec is "".
    let v = try_render_cpumask_bits(&[0u8; 8], u32::MAX);
    match v {
        Some(RenderedValue::CpuList { cpus }) => {
            assert_eq!(cpus, "", "all-zero bytes must produce empty cpu list");
        }
        other => panic!("expected Some(CpuList), got {other:?}"),
    }
}

#[test]
fn try_render_cpumask_bits_single_word_low_bits() {
    // Single word with bits 0,1,2 set → "0-2".
    let bits: u64 = 0b111;
    let bytes = bits.to_le_bytes();
    let v = try_render_cpumask_bits(&bytes, u32::MAX);
    match v {
        Some(RenderedValue::CpuList { cpus }) => assert_eq!(cpus, "0-2"),
        other => panic!("expected CpuList with 0-2, got {other:?}"),
    }
}

#[test]
fn try_render_cpumask_bits_single_word_high_bit() {
    // Bit 63 set → cpu 63.
    let bits: u64 = 1u64 << 63;
    let bytes = bits.to_le_bytes();
    let v = try_render_cpumask_bits(&bytes, u32::MAX);
    match v {
        Some(RenderedValue::CpuList { cpus }) => assert_eq!(cpus, "63"),
        other => panic!("expected CpuList with 63, got {other:?}"),
    }
}

#[test]
fn try_render_cpumask_bits_fully_online_above_128_cpus() {
    // Four all-ones u64 words = CPUs 0-255 fully online. The old
    // `word > 0xFFFF_FFFF && set_cpus.len() > 64` gate bailed at word 2,
    // silently dropping CPUs 128-255; the all-ones-exempting gate (mirror
    // of the kptr path) must render the full set.
    let bytes = [0xFFu8; 32];
    let v = try_render_cpumask_bits(&bytes, 256);
    match v {
        Some(RenderedValue::CpuList { cpus }) => {
            assert_eq!(
                cpus, "0-255",
                "fully-online 256-CPU mask must render all CPUs, got {cpus}"
            );
        }
        other => panic!("expected CpuList 0-255, got {other:?}"),
    }
}

#[test]
fn try_render_cpumask_bits_caps_at_nr_cpu_ids() {
    // 8 CPUs: bits 0..=7 are real, bits 8..=63 are slab padding
    // / freelist garbage. With max_cpus=8, only bits 0..=7
    // should appear in the rendered list even when the word
    // has additional bits set higher up. Pins the fix:
    // an 8-CPU guest must not render bits 64..4035 from a
    // 1024-byte cpumask slab.
    let bits: u64 = 0xFFFF_FFFF_FFFF_FFFF; // every bit set
    let bytes = bits.to_le_bytes();
    let v = try_render_cpumask_bits(&bytes, 8);
    match v {
        Some(RenderedValue::CpuList { cpus }) => {
            assert_eq!(cpus, "0-7", "max_cpus=8 must cap at cpu 7, got {cpus}");
        }
        other => panic!("expected CpuList with 0-7, got {other:?}"),
    }
}

#[test]
fn try_render_cpumask_bits_caps_across_word_boundary() {
    // Two words, all bits set. max_cpus=8 must stop walking
    // immediately in word 0 (cap is partial-word). max_cpus=
    // 64 must walk word 0 fully and stop at the start of word
    // 1 (cap is whole-word).
    let mut bytes = [0u8; 16];
    bytes[0..8].copy_from_slice(&u64::MAX.to_le_bytes());
    bytes[8..16].copy_from_slice(&u64::MAX.to_le_bytes());

    // Partial-word cap inside word 0.
    let v = try_render_cpumask_bits(&bytes, 8);
    match v {
        Some(RenderedValue::CpuList { cpus }) => assert_eq!(cpus, "0-7"),
        other => panic!("expected CpuList 0-7, got {other:?}"),
    }

    // Whole-word cap: 64 means stop at start of word 1.
    // (Word 0 contains bits 0..=63, all 64 of them set.)
    let v = try_render_cpumask_bits(&bytes, 64);
    match v {
        Some(RenderedValue::CpuList { cpus }) => assert_eq!(cpus, "0-63"),
        other => panic!("expected CpuList 0-63, got {other:?}"),
    }
}

#[test]
fn try_render_cpumask_bits_multi_word_offsets() {
    // Two words: word[0] bit 0 → cpu 0, word[1] bit 0 → cpu 64,
    // word[1] bit 1 → cpu 65. Pins the word-index to cpu-id
    // arithmetic (word*64 + bit).
    let mut bytes = [0u8; 16];
    bytes[0..8].copy_from_slice(&1u64.to_le_bytes());
    let w1: u64 = 0b11;
    bytes[8..16].copy_from_slice(&w1.to_le_bytes());
    let v = try_render_cpumask_bits(&bytes, u32::MAX);
    match v {
        Some(RenderedValue::CpuList { cpus }) => assert_eq!(cpus, "0,64-65"),
        other => panic!("expected CpuList with 0,64-65, got {other:?}"),
    }
}

#[test]
fn try_render_cpumask_bits_partial_trailing_bytes_ignored() {
    // 12 bytes (1.5 words) — only the first complete word
    // (8 bytes) parses. The trailing 4 bytes are ignored
    // because n_words = 12/8 = 1.
    let mut bytes = [0u8; 12];
    bytes[0..8].copy_from_slice(&1u64.to_le_bytes());
    // Trailing 4 bytes hold bit 0 (cpu 64 if read as a word)
    // but should NOT be parsed.
    bytes[8] = 0xff;
    let v = try_render_cpumask_bits(&bytes, u32::MAX);
    match v {
        Some(RenderedValue::CpuList { cpus }) => assert_eq!(cpus, "0"),
        other => panic!("expected CpuList with 0, got {other:?}"),
    }
}

/// Garbage cpumask data with a sane nr_cpu_ids cap produces
/// capped output rather than enumerating phantom CPUs from
/// slab-padding / freelist bytes. 16 words (128 bytes) all-FF
/// would otherwise surface 1024 bit-positions; with max_cpus=4
/// only cpus 0..=3 must appear. Pins the backstop: the
/// nr_cpu_ids cap protects the renderer when SLAB_FREELIST_
/// HARDENED XOR-encoding defeats the top-byte heuristic.
#[test]
fn try_render_cpumask_bits_garbage_capped_at_max_cpus() {
    // 128 bytes = 16 u64 words, every bit set. Without the
    // cap this would render 0-1023; with max_cpus=4 the walker
    // stops after bit 3 of word 0.
    let bytes = vec![0xFFu8; 128];
    let v = try_render_cpumask_bits(&bytes, 4);
    match v {
        Some(RenderedValue::CpuList { cpus }) => {
            assert_eq!(
                cpus, "0-3",
                "max_cpus=4 must clip 1024-bit garbage to cpus 0-3, got: {cpus}",
            );
        }
        other => panic!("expected CpuList 0-3, got {other:?}"),
    }
}

/// A multi-word inline cpumask whose trailing word carries the
/// canonical kernel-pointer top byte (`0xff..`) but is NOT all-ones
/// must trigger the per-word plausibility break (`try_render_cpumask_bits`'s
/// `word != u64::MAX && word >> 56 == 0xff`) — that word is slab /
/// pointer garbage, not online CPUs. word0 = 0b11 (cpus 0,1); word1 =
/// 0xff00_0000_0000_0000 (top byte 0xff; bits 120-127 if trusted).
/// max_cpus = 256 is well past word1, so the cap is NOT what stops the
/// walk (that path is pinned by
/// `try_render_cpumask_bits_garbage_capped_at_max_cpus`) — only the
/// 0xff break can drop word1, so the result is "0-1", never
/// "0-1,120-127". The all-ones EXEMPTION side (a fully-online >128-CPU
/// host) is pinned by `try_render_cpumask_bits_fully_online_above_128_cpus`.
#[test]
fn try_render_cpumask_bits_kptr_pattern_word_breaks_walk() {
    let mut bytes = [0u8; 16];
    bytes[0..8].copy_from_slice(&0b11u64.to_le_bytes());
    // Top byte 0xff, rest zero: a kernel-pointer-class word, distinct
    // from all-ones (a legitimately fully-online 64-CPU chunk).
    let w1: u64 = 0xff00_0000_0000_0000;
    bytes[8..16].copy_from_slice(&w1.to_le_bytes());
    let v = try_render_cpumask_bits(&bytes, 256);
    match v {
        Some(RenderedValue::CpuList { cpus }) => assert_eq!(
            cpus, "0-1",
            "0xff-top-byte word must break the walk; only cpus 0,1 from \
             word0 may surface (the cap is well past word1), got: {cpus}",
        ),
        other => panic!("expected CpuList 0-1, got {other:?}"),
    }
}

/// `max_cpus = 0` produces an empty cpu list. The whole-word
/// gate (`word_first_cpu >= max_cpus as u64`) fires immediately
/// at word 0. Defensive: a malformed reader returning
/// `nr_cpu_ids = 0` (not the trait default `u32::MAX`) would
/// otherwise expose the per-bit `cpu >= max_cpus` check to a
/// loop entry; verify both paths converge on empty output.
#[test]
fn try_render_cpumask_bits_max_cpus_zero_yields_empty_list() {
    let bits: u64 = 0xFFFF_FFFF_FFFF_FFFF;
    let bytes = bits.to_le_bytes();
    let v = try_render_cpumask_bits(&bytes, 0);
    match v {
        Some(RenderedValue::CpuList { cpus }) => {
            assert_eq!(cpus, "", "max_cpus=0 must produce empty list, got: {cpus}");
        }
        other => panic!("expected empty CpuList, got {other:?}"),
    }
}

/// Cap matches the actual mask width (max_cpus=64, all 64 bits
/// set in word 0): every bit must surface. A regression that
/// clipped one bit early (an off-by-one such as `cpu + 1 >=
/// max_cpus`, or using `max_cpus - 1`) would produce "0-62"
/// instead of "0-63". Pins the upper-edge off-by-one on the
/// correct `cpu >= max_cpus` cap.
#[test]
fn try_render_cpumask_bits_max_cpus_matches_word_width_keeps_all_bits() {
    let bits: u64 = u64::MAX;
    let bytes = bits.to_le_bytes();
    let v = try_render_cpumask_bits(&bytes, 64);
    match v {
        Some(RenderedValue::CpuList { cpus }) => {
            assert_eq!(
                cpus, "0-63",
                "max_cpus=64 must surface all 64 bits, got: {cpus}",
            );
        }
        other => panic!("expected CpuList 0-63, got {other:?}"),
    }
}

/// `MemReader` trait default `nr_cpu_ids` is `u32::MAX`.
/// Callers that don't override produce no cap, preserving the
/// pre-fix behavior (every set bit reported). A regression
/// that flipped the default to `0` would silently empty every
/// cpumask render.
#[test]
fn mem_reader_default_nr_cpu_ids_is_u32_max() {
    struct DefaultReader;
    impl MemReader for DefaultReader {
        fn read_kva(&self, _: u64, _: usize) -> Option<Vec<u8>> {
            None
        }
    }
    let r = DefaultReader;
    assert_eq!(
        r.nr_cpu_ids(),
        u32::MAX,
        "default nr_cpu_ids must be u32::MAX",
    );
}

/// A custom `MemReader` impl overrides `nr_cpu_ids`. Pin the
/// override path so a regression that ignored the override
/// (always returning the default) is caught.
#[test]
fn mem_reader_custom_nr_cpu_ids_returns_overridden_value() {
    struct CustomReader {
        cpu_count: u32,
    }
    impl MemReader for CustomReader {
        fn read_kva(&self, _: u64, _: usize) -> Option<Vec<u8>> {
            None
        }
        fn nr_cpu_ids(&self) -> u32 {
            self.cpu_count
        }
    }
    let r = CustomReader { cpu_count: 16 };
    assert_eq!(r.nr_cpu_ids(), 16);
}

/// `render_struct` consults `MemReader::nr_cpu_ids` when
/// rendering a cpumask-family struct. With max_cpus=8 supplied
/// by the reader, garbage bits beyond cpu 7 are dropped —
/// the `let max_cpus = mem.map(|m| m.nr_cpu_ids())`
/// in `render_struct` (btf_render/mod.rs) wires the reader
/// value through to `try_render_cpumask_bits`. A regression that
/// passed `u32::MAX` instead would surface phantom cpus 8..63.
#[test]
fn render_value_with_mem_caps_cpumask_at_reader_nr_cpu_ids() {
    let Some(btf) = test_btf() else {
        crate::report::test_skip("test_btf returned None");
        return;
    };
    let Ok(ids) = btf.resolve_ids_by_name("cpumask") else {
        crate::report::test_skip("BTF missing 'cpumask' struct");
        return;
    };
    let Some(&id) = ids.first() else {
        crate::report::test_skip("BTF resolved 'cpumask' to empty id list");
        return;
    };
    // Resolve the underlying struct so we can size the buffer.
    let Some(ty) = peel_modifiers(&btf, id) else {
        crate::report::test_skip("could not peel cpumask modifiers");
        return;
    };
    let size = match type_size(&btf, &ty) {
        Some(n) if n >= 8 => n,
        _ => {
            crate::report::test_skip("cpumask size unresolved or < 8");
            return;
        }
    };
    // Fill the entire cpumask buffer with all-FFs garbage.
    let bytes = vec![0xFFu8; size];
    struct EightCpuReader;
    impl MemReader for EightCpuReader {
        fn read_kva(&self, _: u64, _: usize) -> Option<Vec<u8>> {
            None
        }
        fn nr_cpu_ids(&self) -> u32 {
            8
        }
    }
    let reader = EightCpuReader;
    let v = render_value_with_mem(&btf, id, &bytes, &reader);
    match v {
        RenderedValue::CpuList { cpus } => {
            assert_eq!(
                cpus, "0-7",
                "render_struct must propagate reader.nr_cpu_ids=8 to cpu-list \
                 rendering; got: {cpus}",
            );
        }
        other => panic!("expected CpuList from cpumask render, got {other:?}"),
    }
}

/// Without a `MemReader` (the `render_value` entry point passes
/// `None`), the cpumask renderer falls back to `u32::MAX` cap
/// — every set bit surfaces. Pins the `mem.map(...).unwrap_or(u32::MAX)`
/// fallback in `render_struct` for the no-reader code path.
#[test]
fn render_value_without_mem_uses_u32_max_cap() {
    let Some(btf) = test_btf() else {
        crate::report::test_skip("test_btf returned None");
        return;
    };
    let Ok(ids) = btf.resolve_ids_by_name("cpumask") else {
        crate::report::test_skip("BTF missing 'cpumask' struct");
        return;
    };
    let Some(&id) = ids.first() else {
        crate::report::test_skip("BTF resolved 'cpumask' to empty id list");
        return;
    };
    // 8 bytes (one word) of all-FFs: every bit 0..63 set.
    // No-cap (u32::MAX) means all 64 bits surface as cpus 0-63.
    let bytes = [0xFFu8; 8];
    let v = render_value(&btf, id, &bytes);
    match v {
        RenderedValue::CpuList { cpus } => {
            assert_eq!(
                cpus, "0-63",
                "no-reader cpumask must use u32::MAX cap (all 64 bits), got: {cpus}",
            );
        }
        other => panic!("expected CpuList, got {other:?}"),
    }
}
