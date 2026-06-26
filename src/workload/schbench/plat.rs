//! schbench's fio-derived log2 latency histogram + percentile extraction,
//! ported bit-faithfully from `schbench.c` (`plat_val_to_idx` :438,
//! `plat_idx_to_val` :476, `calc_percentiles` :502, `add_lat` :743,
//! `combine_stats` :730). schbench's reported numbers ARE this exact
//! bucketing + percentile math, so it is reproduced here rather than
//! approximated with ktstr's sampling reservoir or a generic percentile
//! crate — fidelity is the whole point of schbench_rs.
//!
//! Per-phase use: [`PlatStats::take`] snapshots-and-resets the histogram at
//! a phase boundary so each ktstr phase (e.g. scx vs detached-EEVDF)
//! reports isolated percentiles.

/// Histogram tuning, identical to `schbench.c:34-37`.
const PLAT_BITS: u32 = 8;
/// `1 << PLAT_BITS` = 256 buckets per log group.
const PLAT_VAL: u32 = 1 << PLAT_BITS;
/// Number of log groups (`schbench.c:36`).
const PLAT_GROUP_NR: usize = 19;
/// Total bucket count: `19 * 256 = 4864` (`schbench.c:37`).
const PLAT_NR: usize = PLAT_GROUP_NR * PLAT_VAL as usize;

/// The percentiles schbench computes (`schbench.c:132`). Latency tables
/// print {50, 90, 99, 99.9}, RPS prints {20, 50, 90}; both are subsets
/// read back via [`Percentiles::value_at`].
pub(crate) const PLIST: [f64; 5] = [20.0, 50.0, 90.0, 99.0, 99.9];

/// Map a value to its histogram bucket index. Faithful port of
/// `schbench.c` `plat_val_to_idx` (:438): values whose MSB is within
/// `PLAT_BITS` index directly; larger values discard `error_bits` low bits
/// and land in a log-group bucket, clamped to the last bucket.
fn plat_val_to_idx(val: u32) -> usize {
    // C: `msb = 31 - __builtin_clz(val)`, with `clz(0)` UB-guarded to
    // `msb = 0`. Rust `u32::leading_zeros(0) == 32` would underflow, so
    // guard `val == 0` explicitly (the documented divergence point).
    let msb = if val == 0 {
        0
    } else {
        31 - val.leading_zeros()
    };

    // MSB within PLAT_BITS: every bit is significant, index == value.
    if msb <= PLAT_BITS {
        return val as usize;
    }

    let error_bits = msb - PLAT_BITS;
    let base = (error_bits + 1) << PLAT_BITS;
    let offset = (PLAT_VAL - 1) & (val >> error_bits);
    ((base + offset) as usize).min(PLAT_NR - 1)
}

/// Map a bucket index back to its representative (mean-of-range) value.
/// Faithful port of `schbench.c` `plat_idx_to_val` (:476), computing
/// `base + (k + 0.5) * (1 << error_bits)` in `f64`. For every index that
/// reaches this formula (`idx >= 2*PLAT_VAL`, so `error_bits >= 1`),
/// `(k + 0.5) * 2^error_bits = k*2^e + 2^(e-1)` is always integral and
/// exactly f64-representable — the cast truncates NOTHING, it only mirrors
/// C's `(unsigned int)` return cast. (The only `.5`-bearing case,
/// `error_bits == 0`, maps to indices 256..511, which the identity branch
/// returns before this point.)
fn plat_idx_to_val(idx: usize) -> u32 {
    debug_assert!(idx < PLAT_NR, "bucket index {idx} out of range");

    // idx < 2*PLAT_VAL: identity range (`schbench.c:487`).
    if idx < (PLAT_VAL as usize) << 1 {
        return idx as u32;
    }

    let error_bits = (idx >> PLAT_BITS) as u32 - 1;
    let base = 1u32 << (error_bits + PLAT_BITS);
    let k = (idx % PLAT_VAL as usize) as f64;
    // The f64 product is integral here (error_bits >= 1), so this cast
    // loses no precision; it mirrors C's `(unsigned int)` return cast.
    (base as f64 + (k + 0.5) * (1u32 << error_bits) as f64) as u32
}

/// schbench's per-stat latency histogram (`schbench.c` `struct stats`
/// :112). The ~19 KiB `plat` array is HEAP-boxed (`Box<[u32; PLAT_NR]>`), not
/// inline: this type is the per-phase wire carrier and rides inside
/// `PhaseSlice` / `PhaseCgroupStats`, which serde constructs on the stack
/// during (de)serialization — an inline 19 KiB array (×2 in
/// `SchbenchPhaseStats`) overflows the bounded test/worker thread stack in
/// serde's recursive descent. Boxing keeps the struct pointer-sized; the
/// behavior is byte-identical to schbench's inline `unsigned int plat[]`.
///
/// `Serialize`/`Deserialize` + `Eq` make it the carrier. Every field is
/// integer, so the histogram re-pools across workers via [`PlatStats::combine`]
/// (bucket-count addition) host-side — percentiles are NEVER averaged, the only
/// correct cross-worker pooling. The `plat` array uses a `#[serde(with)]`
/// adapter because serde has no array impl past length 32.
#[derive(Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub(crate) struct PlatStats {
    #[serde(with = "plat_array_serde")]
    plat: Box<[u32; PLAT_NR]>,
    nr_samples: u64,
    max: u32,
    /// `0` is the unset sentinel (`schbench.c` `add_lat`/`combine_stats`),
    /// so a genuine 0-usec sample does NOT update `min`.
    min: u32,
}

/// serde adapter for [`PlatStats`]'s fixed `[u32; PLAT_NR]` histogram. serde
/// 1.x has no `Serialize`/`Deserialize` for arrays longer than 32 and this
/// crate does not depend on `serde_big_array`, so the array is carried as a
/// length-prefixed sequence. Deserialization is FAIL-LOUD on a wrong length: a
/// truncated or over-long payload is a corrupt carrier, never silently padded
/// or clipped (the no-silent-wrong-answer rule).
mod plat_array_serde {
    use super::PLAT_NR;
    use serde::Deserialize;
    use serde::de::Error as _;

    // `&Box<[u32; PLAT_NR]>` (not `&[u32; PLAT_NR]`) is mandated by serde's
    // `#[serde(with)]` contract: the field type is `Box<[u32; PLAT_NR]>`, so
    // `serialize` receives `&FieldType`. clippy's borrowed_box does not apply.
    #[allow(clippy::borrowed_box)]
    pub(super) fn serialize<S>(arr: &Box<[u32; PLAT_NR]>, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.collect_seq(arr.iter())
    }

    pub(super) fn deserialize<'de, D>(deserializer: D) -> Result<Box<[u32; PLAT_NR]>, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let v: Vec<u32> = Vec::deserialize(deserializer)?;
        let len = v.len();
        // Convert the heap Vec directly to a heap-boxed array — no intermediate
        // stack array (the whole point of boxing): `Box<[u32]>` -> `Box<[u32;
        // PLAT_NR]>` is a heap-only length check. FAIL-LOUD on a wrong length.
        let boxed: Box<[u32]> = v.into_boxed_slice();
        boxed.try_into().map_err(|_| {
            D::Error::custom(format!(
                "PlatStats histogram length {len} != PLAT_NR {PLAT_NR}"
            ))
        })
    }
}

impl Default for PlatStats {
    fn default() -> Self {
        // Heap-allocate the zeroed histogram directly (a zeroed Vec → boxed
        // slice → boxed array): no `[0; PLAT_NR]` stack array, so even
        // `PlatStats::default()` never materializes 19 KiB on the stack.
        Self {
            plat: vec![0u32; PLAT_NR]
                .into_boxed_slice()
                .try_into()
                .expect("PLAT_NR-length zeroed histogram"),
            nr_samples: 0,
            max: 0,
            min: 0,
        }
    }
}

// Manual Debug: a derived one would dump all 4864 buckets, which is useless in
// a `PhaseSlice` diagnostic. Summarize by the non-array fields instead.
impl core::fmt::Debug for PlatStats {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("PlatStats")
            .field("nr_samples", &self.nr_samples)
            .field("min", &self.min)
            .field("max", &self.max)
            .finish_non_exhaustive()
    }
}

impl PlatStats {
    /// Record one sample. Faithful port of `schbench.c` `add_lat` (:743):
    /// updates max/min (with the `min == 0` sentinel) and increments the
    /// bucket + sample count. Does NOT gate `us == 0` — the caller decides
    /// whether a zero is a real sample. schbench's latency callers GATE on
    /// `delta > 0` (a non-positive delta is dropped, `schbench.c:1035-1037,
    /// 1481-1483`); schbench_rs mirrors that in `msg_and_wait` / `rps_wait`.
    /// schbench's RPS caller instead substitutes 0 for a non-finite rate
    /// (`schbench.c:1782`); schbench_rs's RPS caller (`control_loop`) DROPS the
    /// tick via its `dt > 0` guard — unreachable on the monotonic clock and a
    /// cleaner estimator (no synthetic 0 in bucket 0).
    ///
    /// The `u32` bucket increments with `+= 1`: in a debug build this
    /// panics if a single bucket exceeds 2^32 samples; in release it wraps,
    /// matching C's `__sync_fetch_and_add` on `unsigned int`. 2^32 samples
    /// in one bucket is not a realistic phase, and the debug overflow catch
    /// is worth keeping over a silent wrap.
    pub(crate) fn add_lat(&mut self, us: u32) {
        if us > self.max {
            self.max = us;
        }
        if self.min == 0 || us < self.min {
            self.min = us;
        }
        self.plat[plat_val_to_idx(us)] += 1;
        self.nr_samples += 1;
    }

    /// Fold `other` into `self`: per-bucket add, sample-count add,
    /// max-of-max, and a min-with-sentinel.
    ///
    /// The `min` fold GUARDS against an empty operand — a DELIBERATE
    /// divergence from schbench that fixes a schbench bug. `schbench.c:738`
    /// is literally `if (d->min == 0 || s->min < d->min) d->min = s->min`.
    /// schbench's `combine_message_thread_stats` (`schbench.c:1690-1699`)
    /// folds EVERY worker UNCONDITIONALLY, and the per-interval accumulator
    /// is re-folded over all workers each interval (`schbench.c:1785-1792`),
    /// so an idle/starved worker — whose stat is empty (`min == 0`, e.g.
    /// every delta dropped by the `delta > 0` gate) — gets folded after a
    /// non-empty one, where `0 < d->min` fires and ZEROS the accumulated
    /// `min`. That is a real schbench bug: a single idle worker corrupts
    /// schbench's reported `min` to 0, and starvation is exactly what
    /// schbench exists to surface. schbench_rs's `other.min != 0` guard
    /// skips an empty operand and keeps the true smallest sample, because an
    /// idle / late-spawn / early-detach per-phase worker is expected in the
    /// per-phase model and a spurious `min == 0` would corrupt the phase
    /// result. The two agree whenever no empty operand is folded (the
    /// common, no-idle case).
    pub(crate) fn combine(&mut self, other: &PlatStats) {
        for i in 0..PLAT_NR {
            self.plat[i] += other.plat[i];
        }
        self.nr_samples += other.nr_samples;
        if other.max > self.max {
            self.max = other.max;
        }
        if other.min != 0 && (self.min == 0 || other.min < self.min) {
            self.min = other.min;
        }
    }

    /// Extract percentiles. Faithful port of `schbench.c`
    /// `calc_percentiles` (:502): cumulative bucket sum, the threshold test
    /// `sum >= plist[j]/100 * nr` (`>=`, and a `while` so several
    /// percentiles can fall in one bucket), then the cumulative counts are
    /// differenced into per-bucket counts. With zero samples every value is
    /// 0 (the `0 >= 0` test fires for each percentile).
    pub(crate) fn percentiles(&self) -> Percentiles {
        let nr = self.nr_samples;
        let len = PLIST.len();
        let mut values = [0u32; 5];
        let mut counts = [0u64; 5];

        let mut sum: u64 = 0;
        let mut j = 0usize;
        let mut is_last = false;
        let mut i = 0usize;
        while i < PLAT_NR && !is_last {
            sum += self.plat[i] as u64;
            while (sum as f64) >= PLIST[j] / 100.0 * nr as f64 {
                values[j] = plat_idx_to_val(i);
                counts[j] = sum;
                is_last = j == len - 1;
                if is_last {
                    break;
                }
                j += 1;
            }
            i += 1;
        }

        // Difference cumulative counts into per-bucket counts
        // (`schbench.c:543-546`). Cumulative ⇒ each delta is >= 0.
        let mut last = 0u64;
        for idx in 1..len {
            last += counts[idx - 1];
            counts[idx] -= last;
        }

        Percentiles {
            values,
            counts,
            min: self.min,
            max: self.max,
            nr_samples: nr,
        }
    }

    /// Snapshot the accumulated stats and reset to empty — the per-phase
    /// boundary primitive (ruling: each ktstr phase reports isolated
    /// percentiles, e.g. scx vs detached-EEVDF in one run).
    pub(crate) fn take(&mut self) -> PlatStats {
        std::mem::take(self)
    }

    /// Number of samples recorded (`schbench.c` `nr_samples`). O(1) — reads the
    /// running count without scanning the histogram, so callers can gate an
    /// empty histogram (no metric emitted) or tally dropped samples without the
    /// `percentiles()` scan.
    pub(crate) fn sample_count(&self) -> u64 {
        self.nr_samples
    }
}

/// Extracted percentile values + min/max for a [`PlatStats`].
pub(crate) struct Percentiles {
    /// Value at each [`PLIST`] percentile, same order (20, 50, 90, 99,
    /// 99.9).
    pub(crate) values: [u32; 5],
    /// Per-bucket (differenced) sample counts, parallel to `values`.
    pub(crate) counts: [u64; 5],
    pub(crate) min: u32,
    pub(crate) max: u32,
    pub(crate) nr_samples: u64,
}

/// The percentiles schbench reports, as an index enum. Reading by enum
/// (not `f64`) avoids float-equality on the read path — a percentile
/// computed arithmetically (e.g. `999.0/10.0`) need not bit-match the
/// `99.9` literal. Each variant maps to a [`PLIST`] slot, in order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Pct {
    /// `PLIST[0]` = 20.0 — schbench's RPS-distribution percentile
    /// (`PLIST_FOR_RPS`, `schbench.c:130`). Read by the per-phase RPS derivation
    /// (`run_metrics.rs` `derive_phase_metrics`, `rps_p20`); the latency
    /// metrics use P50-P999.
    P20,
    P50,
    P90,
    P99,
    P999,
}

impl Pct {
    fn index(self) -> usize {
        match self {
            Pct::P20 => 0,
            Pct::P50 => 1,
            Pct::P90 => 2,
            Pct::P99 => 3,
            Pct::P999 => 4,
        }
    }
}

impl Percentiles {
    /// The value at a given percentile. Every [`Pct`] is a valid [`PLIST`]
    /// slot, so this is total (no `Option`).
    pub(crate) fn value_at(&self, p: Pct) -> u32 {
        self.values[p.index()]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plat_val_to_idx_identity_range() {
        // MSB within PLAT_BITS (val <= 511) indexes directly.
        assert_eq!(plat_val_to_idx(0), 0);
        assert_eq!(plat_val_to_idx(1), 1);
        assert_eq!(plat_val_to_idx(255), 255);
        assert_eq!(plat_val_to_idx(256), 256); // msb == 8 == PLAT_BITS
        assert_eq!(plat_val_to_idx(511), 511); // msb == 8
    }

    #[test]
    fn plat_val_to_idx_log_group_and_clamp() {
        // msb == 9 > PLAT_BITS: error_bits=1, base=512, offset=255&(512>>1).
        // 512>>1 = 256; 255 & 256 == 0 -> idx 512.
        assert_eq!(plat_val_to_idx(512), 512);
        // 513>>1 = 256; 255 & 256 == 0 -> still 512.
        assert_eq!(plat_val_to_idx(513), 512);
        // Huge value clamps to the last bucket.
        assert_eq!(plat_val_to_idx(u32::MAX), PLAT_NR - 1);
    }

    #[test]
    fn plat_idx_round_trips_identity_range() {
        for idx in [0usize, 1, 255, 256, 511] {
            assert_eq!(plat_idx_to_val(idx), idx as u32);
        }
    }

    #[test]
    fn plat_idx_to_val_mean_of_range() {
        // Mean-of-range arithmetic — NOT truncation: for every reachable
        // idx >= 512, error_bits >= 1, so the f64 product is integral and
        // no `.5` fraction survives to be truncated.
        // idx 512: error_bits=(512>>8)-1=1, base=1<<9=512, k=0,
        // val = 512 + (0.5 * 2) = 513.
        assert_eq!(plat_idx_to_val(512), 513);
        // idx 513: k=1, val = 512 + (1.5 * 2) = 515.
        assert_eq!(plat_idx_to_val(513), 515);
    }

    #[test]
    fn empty_histogram_yields_zero_percentiles_len5() {
        let s = PlatStats::default();
        let p = s.percentiles();
        assert_eq!(p.nr_samples, 0);
        assert_eq!(p.values, [0, 0, 0, 0, 0]);
        assert_eq!(p.min, 0);
        assert_eq!(p.max, 0);
        assert_eq!(p.value_at(Pct::P99), 0);
    }

    #[test]
    fn single_sample_all_percentiles_equal_that_value() {
        let mut s = PlatStats::default();
        s.add_lat(100);
        let p = s.percentiles();
        assert_eq!(p.nr_samples, 1);
        assert_eq!(p.values, [100, 100, 100, 100, 100]);
        assert_eq!(p.min, 100);
        assert_eq!(p.max, 100);
        // Cumulative 1 at p20, differenced to [1,0,0,0,0].
        assert_eq!(p.counts, [1, 0, 0, 0, 0]);
    }

    #[test]
    fn zero_sample_records_bucket_but_not_min() {
        // add_lat(0): bucket 0 increments, but min stays at the 0 sentinel
        // and max stays 0 (faithful to schbench add_lat).
        let mut s = PlatStats::default();
        s.add_lat(0);
        let p = s.percentiles();
        assert_eq!(p.nr_samples, 1);
        assert_eq!(p.min, 0);
        assert_eq!(p.max, 0);
        assert_eq!(p.values, [0, 0, 0, 0, 0]);
    }

    #[test]
    fn percentiles_split_across_buckets() {
        // 99 samples at 10us, 1 sample at 10000us. p50/p90/p99 land in the
        // 10us bucket; p99.9 in the tail. min=10, max=10000.
        let mut s = PlatStats::default();
        for _ in 0..99 {
            s.add_lat(10);
        }
        s.add_lat(10000);
        let p = s.percentiles();
        assert_eq!(p.nr_samples, 100);
        assert_eq!(p.min, 10);
        assert_eq!(p.max, 10000);
        assert_eq!(p.value_at(Pct::P50), 10);
        assert_eq!(p.value_at(Pct::P90), 10);
        assert_eq!(p.value_at(Pct::P99), 10);
        // p99.9 of 100 samples (>= 99.9% reached at the 100th sample) lands
        // on the 10000us tail bucket.
        let p999 = p.value_at(Pct::P999);
        assert!(p999 >= 10000, "p99.9 should be the tail bucket: {p999}");
    }

    #[test]
    fn value_at_pins_each_pct_to_distinct_bucket() {
        // Each percentile lands in a DISTINCT bucket, so value_at(Pct) pins
        // the whole Pct -> PLIST-slot correspondence — including P20, which no
        // other test exercises. A PLIST reorder or a wrong Pct::index arm
        // would read the wrong slot and fail here.
        //
        // percentiles() compares the per-bucket cumulative sum against the
        // float threshold `PLIST[j]/100 * nr`. With nr=1000 the thresholds are
        // ~200/500/900/990/999 — but e.g. `99.9/100*1000` rounds to just ABOVE
        // 999.0 in f64, so a threshold sitting exactly on a bucket boundary
        // can spill into the next bucket. Sizing the blocks so the running
        // cumsum (250/550/920/995/1000) brackets each threshold strictly
        // INSIDE one bucket keeps the mapping robust to that rounding:
        //   P20  ~200 in (0,250]    -> bucket 10
        //   P50  ~500 in (250,550]  -> bucket 20
        //   P90  ~900 in (550,920]  -> bucket 30
        //   P99  ~990 in (920,995]  -> bucket 40
        //   P999 ~999 in (995,1000] -> bucket 50
        let mut s = PlatStats::default();
        for _ in 0..250 {
            s.add_lat(10); // cumsum 250
        }
        for _ in 0..300 {
            s.add_lat(20); // cumsum 550
        }
        for _ in 0..370 {
            s.add_lat(30); // cumsum 920
        }
        for _ in 0..75 {
            s.add_lat(40); // cumsum 995
        }
        for _ in 0..5 {
            s.add_lat(50); // cumsum 1000
        }
        let p = s.percentiles();
        assert_eq!(p.nr_samples, 1000);
        assert_eq!(p.value_at(Pct::P20), 10);
        assert_eq!(p.value_at(Pct::P50), 20);
        assert_eq!(p.value_at(Pct::P90), 30);
        assert_eq!(p.value_at(Pct::P99), 40);
        assert_eq!(p.value_at(Pct::P999), 50);
    }

    #[test]
    fn combine_folds_buckets_counts_and_extremes() {
        let mut a = PlatStats::default();
        a.add_lat(5);
        a.add_lat(50);
        let mut b = PlatStats::default();
        b.add_lat(7);
        b.add_lat(9000);
        a.combine(&b);
        let p = a.percentiles();
        assert_eq!(p.nr_samples, 4);
        assert_eq!(p.min, 5); // min of {5,50,7,9000}
        assert_eq!(p.max, 9000);
        // Percentile VALUES pin the per-bucket fold itself (nr/min/max above
        // come from non-bucket fields and would survive dropping the bucket
        // merge). Post-combine buckets are {5, 7, 50, 1561} over nr=4, so the
        // cumulative thresholds land: P20 (0.8) on bucket 5, P50 (2.0) on
        // bucket 7 — present ONLY because b's buckets were folded in — and
        // P90/P99/P999 (3.6/3.96/3.996) on b's 9000 sample
        // (plat_idx_to_val(1561) = 9008). Dropping the `self.plat[i] +=
        // other.plat[i]` loop leaves only a's {5, 50}: P50 would read 50, not 7.
        assert_eq!(p.value_at(Pct::P20), 5);
        assert_eq!(p.value_at(Pct::P50), 7);
        assert_eq!(p.value_at(Pct::P90), 9008);
        assert_eq!(p.value_at(Pct::P99), 9008);
        assert_eq!(p.value_at(Pct::P999), 9008);
    }

    #[test]
    fn combine_empty_does_not_zero_min() {
        // The min fold guards against an empty operand: schbench DOES fold
        // empty worker stats (combine_message_thread_stats schbench.c:
        // 1690-1699, unconditional), so its literal min fold (schbench.c:
        // 738) lets an idle worker zero schbench's own reported min.
        // schbench_rs diverges to keep the real min for the per-phase
        // consumer (which CAN have an empty per-phase worker).
        let mut a = PlatStats::default();
        a.add_lat(42);
        assert_eq!(a.percentiles().min, 42);
        let empty = PlatStats::default();
        a.combine(&empty);
        assert_eq!(
            a.percentiles().min,
            42,
            "guarded: folding an empty operand must not zero min",
        );
        assert_eq!(a.percentiles().nr_samples, 1, "empty fold adds no samples");
    }

    #[test]
    fn take_snapshots_and_resets_for_per_phase() {
        let mut s = PlatStats::default();
        s.add_lat(100);
        s.add_lat(200);
        let phase1 = s.take();
        assert_eq!(phase1.percentiles().nr_samples, 2);
        // After take, the live stats are empty (new phase starts clean).
        assert_eq!(s.percentiles().nr_samples, 0);
        s.add_lat(300);
        assert_eq!(s.percentiles().nr_samples, 1);
        assert_eq!(s.percentiles().min, 300);
    }

    #[test]
    fn platstats_serde_roundtrips_json_and_postcard() {
        // The per-phase carrier rides PlatStats over postcard (guest->host) and
        // is also JSON-serializable for the worker report pipe. Both must
        // preserve the full 4864-bucket histogram + nr_samples/max/min, since
        // the host re-derives percentiles from the deserialized buckets.
        let mut s = PlatStats::default();
        s.add_lat(5);
        s.add_lat(9000);
        s.add_lat(0); // bucket 0, min stays sentinel
        let json = serde_json::to_string(&s).expect("serialize json");
        let back: PlatStats = serde_json::from_str(&json).expect("deserialize json");
        assert_eq!(s, back, "json roundtrip preserves histogram + extremes");
        let bytes = postcard::to_stdvec(&s).expect("serialize postcard");
        let back2: PlatStats = postcard::from_bytes(&bytes).expect("deserialize postcard");
        assert_eq!(
            s, back2,
            "postcard roundtrip preserves histogram + extremes"
        );
        // The deserialized histogram yields identical percentiles (the host path).
        assert_eq!(back2.percentiles().values, s.percentiles().values);
    }

    #[test]
    fn platstats_deserialize_wrong_len_is_fail_loud() {
        // A short plat array is a corrupt carrier: the adapter rejects it rather
        // than padding/truncating (the no-silent-wrong-answer rule).
        let bad = r#"{"plat":[1,2,3],"nr_samples":0,"max":0,"min":0}"#;
        let err = serde_json::from_str::<PlatStats>(bad).expect_err("wrong-len must fail");
        assert!(
            err.to_string().contains("PLAT_NR"),
            "error names the length mismatch: {err}"
        );
    }
}
