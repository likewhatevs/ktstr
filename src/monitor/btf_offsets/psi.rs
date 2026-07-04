//! BTF offsets + decode for the system-wide PSI-irq host memory walk.
//!
//! Captures the guest's IRQ Pressure-Stall-Information observer-free —
//! by walking the global `struct psi_group psi_system` (`kernel/sched/psi.c`,
//! external linkage → KASLR-locatable like `jiffies_64`) from host-side guest
//! memory, instead of a guest `/proc/pressure/irq` read. Two fields are read:
//!
//! - `total[NR_PSI_AGGREGATORS][NR_PSI_STATES-1]` (`u64`, cumulative stall ns):
//!   `total[PSI_AVGS][PSI_IRQ_FULL]` is the monotonic IRQ-full stall accumulator
//!   (`collect_percpu_times` folds per-CPU `times[PSI_IRQ_FULL]` into it,
//!   psi.c:362-415). `psi_show` renders it as µs via `div_u64(.., NSEC_PER_USEC)`
//!   (psi.c:1280-1281), so [`decode_total_us`] divides the raw ns by 1000.
//! - `avg[NR_PSI_STATES-1][3]` (`unsigned long`, the EWMA): `avg[PSI_IRQ_FULL][0]`
//!   is the 10s window. `calc_avgs` (psi.c:356-360) scales the percent by
//!   `FIXED_1` (`pct *= FIXED_1`) and `calc_load` is scale-preserving, so the
//!   stored value is `percent * FIXED_1`; [`decode_avg10_percent`] divides by
//!   `FIXED_1` to recover percent `[0,100]` (matches `psi_show`'s
//!   `LOAD_INT/LOAD_FRAC` = `raw/FIXED_1`, psi.c:1286-1288).
//!
//! GATING (loud-absent): `PSI_IRQ_FULL` is `#ifdef CONFIG_IRQ_TIME_ACCOUNTING`
//! in `enum psi_states` (`include/linux/psi_types.h`), so on a kernel built
//! without it the enumerator is absent from BTF → [`super::enum_value`] returns
//! `None` → [`PsiGroupOffsets::psi_irq_full_idx`] is `None` → the metrics read
//! loud-absent, the `avg_irq_util` BTF-gated pattern. ktstr.kconfig enables it,
//! so the index resolves to 6 on ktstr VMs. A config-ON kernel with
//! IRQ-time accounting disabled at runtime (`irqtime_enabled()` static key off)
//! leaves the accumulators at a real 0 → a measured `Some(0.0)`, NOT absent —
//! the `total_steal_time_ns` precedent (a present-but-accounting-off field reads
//! a constant 0).

use anyhow::Result;
use btf_rs::Btf;

use super::{enum_value, find_struct, member_byte_offset};

/// `PSI_AVGS` aggregator index into `psi_group.total`/`.avg`'s outer dimension
/// (`enum psi_aggregators`, `include/linux/psi_types.h`: `PSI_AVGS=0`,
/// `PSI_POLL=1`). The running-average aggregator — the one `/proc` reads.
pub const PSI_AVGS_AGG: usize = 0;

/// [`PsiGroupOffsets::total_irq_full_off`] reads the `PSI_AVGS` aggregator row
/// and relies on it being row 0 — so the row begins at the array base and the
/// element offset reduces to `idx * elem_size` (no aggregator-stride term).
/// Pin that here: a kernel that ever renumbered `enum psi_aggregators` so
/// `PSI_AVGS != 0` fails to compile rather than silently reading the wrong
/// aggregator's totals.
const _: () = assert!(PSI_AVGS_AGG == 0);

/// Number of EWMA windows in `psi_group.avg[state][N]` — avg10/avg60/avg300
/// (`unsigned long avg[NR_PSI_STATES-1][3]`, `psi_types.h`). Fixed at 3,
/// config-independent (unlike the state dimension, which is `#ifdef`-gated).
pub const PSI_NR_WINDOWS: usize = 3;

/// Byte width of one `psi_group.total`/`.avg` element on a 64-bit guest:
/// `total` is `u64`, `avg` is `unsigned long` (both 8B on the LP64 targets ktstr
/// supports — x86_64 / aarch64). `read_u64` is byte-correct for both. ktstr is
/// 64-bit-only (the LE/64-bit host assumption documented for the VMM walks).
pub const PSI_ELEM_SIZE: usize = 8;

/// `FIXED_1 = 1<<FSHIFT`, `FSHIFT=11` (`include/linux/sched/loadavg.h`) — the
/// fixed-point scale of `psi_group.avg[]`. The stored EWMA is `percent*FIXED_1`
/// (`calc_avgs` psi.c:356-360), so `raw/FIXED_1` recovers percent `[0,100]`.
pub const PSI_FIXED_1: f64 = 2048.0;

/// `NSEC_PER_USEC` — `psi_show` (psi.c:1280-1281) emits `total` as
/// `div_u64(total_ns, NSEC_PER_USEC)`, so the stored `total[]` ns ÷ this = µs.
pub const PSI_NSEC_PER_USEC: f64 = 1000.0;

/// Decode a raw `psi_group.avg[PSI_IRQ_FULL][0]` (fixed-point) to percent,
/// clamped to `[0,100]` — `(raw / FIXED_1).min(100.0)`. The kernel bounds the
/// steady-state value (`update_averages` caps `sample = period` before
/// `calc_avgs`, `kernel/sched/psi.c:525-576`), so `raw <= 100*FIXED_1`; the `.min(100.0)`
/// hardens against a TORN lockless host-read — `collect_percpu_times` samples
/// per-CPU buckets locklessly and a delta can transiently slip past the period
/// before the clamp settles, so a host-walk mid-update could observe a raw a hair
/// above `100*FIXED_1`. The clamped value otherwise matches `/proc/pressure/irq`'s
/// `avg10=NN.MM` (the text floors to 2 decimals).
pub fn decode_avg10_percent(raw: u64) -> f64 {
    (raw as f64 / PSI_FIXED_1).min(100.0)
}

/// Decode a raw `psi_group.total[PSI_AVGS][PSI_IRQ_FULL]` (cumulative stall ns)
/// to microseconds — `raw_ns / NSEC_PER_USEC` (the `psi_show` total= unit).
pub fn decode_total_us(raw_ns: u64) -> f64 {
    raw_ns as f64 / PSI_NSEC_PER_USEC
}

/// Byte offsets within `struct psi_group` (`include/linux/psi_types.h`) for the
/// system-wide PSI-irq host-walk, plus the config-gated `PSI_IRQ_FULL` state
/// index. The two array bases are BTF-resolved (a future leading-field addition
/// surfaces here, not as a silent miscalculation); the element size / window
/// count / `PSI_AVGS` index are layout constants (hardcoded with the cited
/// header values, like the `cpu_time` enum indices).
#[derive(Debug, Clone, Copy)]
pub struct PsiGroupOffsets {
    /// Offset of `total[NR_PSI_AGGREGATORS][NR_PSI_STATES-1]` (`u64`, ns).
    pub psi_group_total: usize,
    /// Offset of `avg[NR_PSI_STATES-1][3]` (`unsigned long`, fixed-point EWMA).
    pub psi_group_avg: usize,
    /// Value of `PSI_IRQ_FULL` from `enum psi_states`, or `None` when the
    /// enumerator is absent (kernel built without `CONFIG_IRQ_TIME_ACCOUNTING`)
    /// — the loud-absent gate. `Some(6)` on ktstr VMs.
    pub psi_irq_full_idx: Option<usize>,
}

impl PsiGroupOffsets {
    /// Resolve the `psi_group` offsets + the `PSI_IRQ_FULL` index from a
    /// pre-loaded BTF object. `Err` only when `psi_group` itself is absent (a
    /// PSI-less / stripped vmlinux); `psi_irq_full_idx` is `None` (not `Err`)
    /// when only the IRQ state is config-gated off, so the caller can still
    /// resolve the struct on a no-IRQ-accounting kernel and surface the metrics
    /// loud-absent.
    pub fn from_btf(btf: &Btf) -> Result<Self> {
        let (psi_group, _) = find_struct(btf, "psi_group")?;
        let psi_group_total = member_byte_offset(btf, &psi_group, "total")?;
        let psi_group_avg = member_byte_offset(btf, &psi_group, "avg")?;
        let psi_irq_full_idx =
            enum_value(btf, "psi_states", "PSI_IRQ_FULL").and_then(|v| usize::try_from(v).ok());
        Ok(Self {
            psi_group_total,
            psi_group_avg,
            psi_irq_full_idx,
        })
    }

    /// Byte offset of `total[PSI_AVGS][PSI_IRQ_FULL]` (cumulative IRQ stall, ns)
    /// from the `psi_group` base, or `None` when `PSI_IRQ_FULL` is config-absent.
    /// `PSI_AVGS == 0` (pinned by the module const-assert), so the row starts at
    /// the array base and the element offset is `idx * elem_size` — no
    /// aggregator-stride term (were `PSI_AVGS` nonzero this would add
    /// `PSI_AVGS * (NR_PSI_STATES-1)`, the gated inner dimension).
    pub fn total_irq_full_off(&self) -> Option<usize> {
        self.psi_irq_full_idx
            .map(|idx| self.psi_group_total + idx * PSI_ELEM_SIZE)
    }

    /// Byte offset of `avg[PSI_IRQ_FULL][0]` (the avg10 EWMA window) from the
    /// `psi_group` base, or `None` when `PSI_IRQ_FULL` is config-absent. `avg` is
    /// `[state][PSI_NR_WINDOWS]`, so row `idx` starts at `idx*PSI_NR_WINDOWS`
    /// elements in and window 0 (avg10) is the first (avg60/avg300 are +1/+2).
    pub fn avg10_irq_full_off(&self) -> Option<usize> {
        self.psi_irq_full_idx
            .map(|idx| self.psi_group_avg + idx * PSI_NR_WINDOWS * PSI_ELEM_SIZE)
    }
}

#[cfg(test)]
mod tests {
    use super::super::load_btf_from_path;
    use super::*;

    /// Resolve [`PsiGroupOffsets`] against the test vmlinux and pin the layout
    /// the host-walk depends on: `total`/`avg` at distinct offsets,
    /// `PSI_IRQ_FULL == 6` (the index under `CONFIG_IRQ_TIME_ACCOUNTING`, which
    /// the test kernel enables), and the derived element offsets.
    #[test]
    fn parse_psi_group_offsets_from_vmlinux() {
        let path = match crate::monitor::find_test_vmlinux() {
            Some(p) => p,
            None => return,
        };
        let btf = match load_btf_from_path(&path) {
            Ok(b) => b,
            Err(e) => skip!("vmlinux BTF load failed: {e}"),
        };
        let offsets = match PsiGroupOffsets::from_btf(&btf) {
            Ok(o) => o,
            Err(e) => skip!("PsiGroupOffsets::from_btf failed: {e}"),
        };
        assert_ne!(
            offsets.psi_group_total, offsets.psi_group_avg,
            "psi_group total[] and avg[] must be at distinct offsets"
        );
        // The test kernel sets CONFIG_IRQ_TIME_ACCOUNTING (ktstr.kconfig),
        // so PSI_IRQ_FULL is present at index 6 (enum psi_states: IO_SOME=0 ..
        // CPU_FULL=5, PSI_IRQ_FULL=6).
        assert_eq!(
            offsets.psi_irq_full_idx,
            Some(6),
            "PSI_IRQ_FULL must resolve to 6 under CONFIG_IRQ_TIME_ACCOUNTING"
        );
        // Derived element offsets: total[0][6] = base + 6*8; avg[6][0] = base + 6*3*8.
        assert_eq!(
            offsets.total_irq_full_off(),
            Some(offsets.psi_group_total + 6 * PSI_ELEM_SIZE),
        );
        assert_eq!(
            offsets.avg10_irq_full_off(),
            Some(offsets.psi_group_avg + 6 * PSI_NR_WINDOWS * PSI_ELEM_SIZE),
        );
    }

    /// The decode math matches the kernel's `/proc` rendering: avg10 = raw/2048
    /// (FIXED_1), total = raw_ns/1000 (NSEC_PER_USEC). Worked values from the
    /// `calc_avgs`/`psi_show` citations: raw 102400 → 50.0%, raw 204800 → 100.0%.
    #[test]
    fn psi_decode_math() {
        assert_eq!(decode_avg10_percent(2048), 1.0);
        assert_eq!(decode_avg10_percent(102_400), 50.0);
        assert_eq!(decode_avg10_percent(204_800), 100.0);
        assert_eq!(decode_avg10_percent(0), 0.0);
        // A torn lockless host-read can transiently exceed 100*FIXED_1 before
        // the kernel's update_averages clamp settles; decode clamps to 100.0.
        assert_eq!(decode_avg10_percent(204_801), 100.0);
        assert_eq!(decode_avg10_percent(1_000_000), 100.0);
        assert_eq!(decode_total_us(1_000), 1.0);
        assert_eq!(decode_total_us(2_500_000), 2500.0);
        assert_eq!(decode_total_us(0), 0.0);
    }
}
