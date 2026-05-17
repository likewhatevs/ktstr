//! Linux scheduling-class + sched-policy declarative types for the
//! workload pipeline.
//!
//! Holds [`SchedPolicy`] (the per-task `sched_setattr` shape),
//! [`SchedClass`] (the coarse class identifier consumed by
//! `WorkType::AsymmetricWaker`), and three orthogonal knobs used by
//! specific work types: [`FutexLockMode`] (PI vs plain futex for
//! `WorkType::PriorityInversion`), [`WakeMechanism`] (pipe vs futex
//! wake between stages of `WorkType::WakeChain`), and [`AluWidth`]
//! (scalar / SIMD width for `WorkType::AluHot`).
//!
//! These types are declarative — the corresponding kernel-call
//! helpers live in the [`crate::workload::worker`] submodule
//! (`set_sched_policy` in `worker/sched.rs`, `apply_sched_class`).

use std::time::Duration;

use super::humantime_serde_helper;

/// Linux scheduling policy for a worker process.
///
/// `Fifo`, `RoundRobin`, and `Deadline` all require `CAP_SYS_NICE`
/// (`user_check_sched_setscheduler` in `kernel/sched/syscalls.c`
/// routes rt_policy and dl_policy through `req_priv`). `Normal`,
/// `Batch`, and (entering) `Idle` are unprivileged transitions for
/// fair-policy tasks. Priority values for `Fifo`/`RoundRobin` are
/// clamped to 1-99.
#[derive(
    Debug, Clone, Copy, Default, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize,
)]
#[serde(rename_all = "snake_case")]
pub enum SchedPolicy {
    /// `SCHED_NORMAL` (CFS/EEVDF).
    #[default]
    Normal,
    /// `SCHED_BATCH`.
    Batch,
    /// `SCHED_IDLE`.
    Idle,
    /// `SCHED_FIFO` with the given priority (1-99).
    Fifo(u32),
    /// `SCHED_RR` with the given priority (1-99).
    RoundRobin(u32),
    /// `SCHED_DEADLINE` with explicit `runtime`, `deadline`, and
    /// `period`. Applied via `sched_setattr(2)`.
    ///
    /// Each field is a [`Duration`] — the nanosecond representation
    /// the kernel requires is materialised at the syscall site, so
    /// callers express intent in idiomatic Rust units
    /// (`Duration::from_micros(100)`, `Duration::from_millis(1)`,
    /// etc.) and don't have to thread integer-nanosecond literals
    /// through their test fixtures.
    ///
    /// Constraints (from `__checkparam_dl` in
    /// `kernel/sched/deadline.c`):
    /// - `deadline != Duration::ZERO`.
    /// - `runtime` must be at least 1024 ns (the kernel's
    ///   `DL_SCALE` floor); shorter runtimes are silently truncated
    ///   inside the kernel and break bandwidth accounting.
    /// - `runtime <= deadline`.
    /// - `period == Duration::ZERO` is legal — the kernel
    ///   substitutes `deadline` for the period when zero. When
    ///   non-zero, `deadline <= period`.
    /// - The effective period (`period` if non-zero, else
    ///   `deadline`) is checked against
    ///   `/proc/sys/kernel/sched_deadline_period_min_us` (default
    ///   100us = 100_000 ns) and
    ///   `/proc/sys/kernel/sched_deadline_period_max_us` (default
    ///   `1 << 22` us = 4_194_304_000 ns), inclusive. Both sysctls
    ///   are runtime-tunable; this crate does not pre-validate the
    ///   sysctl range and lets the kernel surface out-of-range
    ///   values as `EINVAL`.
    /// - The nanosecond count of `deadline` and `period` must each
    ///   fit in 63 bits (`< 1 << 63`, i.e. `<= i64::MAX` ns ≈ 292
    ///   years) — the kernel uses bit 63 internally. Any longer
    ///   `Duration` is rejected at the syscall site.
    ///
    /// Transitions to/from `Deadline` always require `CAP_SYS_NICE`.
    /// Tasks set to `Deadline` get exclusive bandwidth on the
    /// admission-controlled root domain; oversubscription returns
    /// `EBUSY` (see `sched_dl_overflow` in `kernel/sched/deadline.c`).
    ///
    /// `set_sched_policy` validates the structural constraints
    /// (zero-deadline, DL_SCALE floor, ordering, top-bit) before
    /// invoking `sched_setattr` so a malformed `Deadline` fails
    /// fast in user space rather than tunneling an `EINVAL`
    /// through the syscall.
    Deadline {
        /// Runtime budget per period.
        #[serde(with = "humantime_serde_helper")]
        runtime: Duration,
        /// Relative deadline from period start.
        #[serde(with = "humantime_serde_helper")]
        deadline: Duration,
        /// Period. `Duration::ZERO` means "use `deadline` as the
        /// period" per the kernel's `__checkparam_dl` substitution.
        #[serde(with = "humantime_serde_helper")]
        period: Duration,
    },
}

impl SchedPolicy {
    /// `SCHED_FIFO` with the given priority (1-99).
    pub const fn fifo(priority: u32) -> Self {
        SchedPolicy::Fifo(priority)
    }

    /// `SCHED_RR` with the given priority (1-99).
    pub const fn round_robin(priority: u32) -> Self {
        SchedPolicy::RoundRobin(priority)
    }

    /// `SCHED_DEADLINE` with the given runtime / deadline / period.
    /// See [`SchedPolicy::Deadline`] for parameter constraints.
    ///
    /// All three arguments share the same [`Duration`] type. The
    /// canonical order is `(runtime, deadline, period)` — runtime
    /// budget first, then the relative deadline, then the period.
    /// For tests that need to make the order obvious at the call
    /// site, prefer the struct-literal form
    /// `SchedPolicy::Deadline { runtime: ..., deadline: ...,
    /// period: ... }` which carries the field names through the
    /// reader's eye.
    ///
    /// ```
    /// # use std::time::Duration;
    /// # use ktstr::workload::SchedPolicy;
    /// // Convenience constructor — canonical (runtime, deadline, period) order.
    /// let p = SchedPolicy::deadline(
    ///     Duration::from_micros(500), // runtime
    ///     Duration::from_millis(1),   // deadline
    ///     Duration::from_millis(10),  // period
    /// );
    /// // Struct-literal form — names elide positional confusion.
    /// let q = SchedPolicy::Deadline {
    ///     runtime: Duration::from_micros(500),
    ///     deadline: Duration::from_millis(1),
    ///     period: Duration::from_millis(10),
    /// };
    /// assert!(matches!(p, SchedPolicy::Deadline { .. }));
    /// assert!(matches!(q, SchedPolicy::Deadline { .. }));
    /// ```
    pub const fn deadline(runtime: Duration, deadline: Duration, period: Duration) -> Self {
        SchedPolicy::Deadline {
            runtime,
            deadline,
            period,
        }
    }
}

/// Whether `WorkType::PriorityInversion` uses a PI-aware mutex
/// or a plain futex.
///
/// `Pi` exercises `FUTEX_LOCK_PI` and the rt_mutex priority-boost
/// chain (`kernel/futex/pi.c`). When the low-priority lock holder
/// is preempted by a medium-priority worker, the kernel boosts
/// the holder to the high-priority waiter's priority for the
/// duration of the hold — both unblocking `high` and pinning
/// `medium` from preempting it. `Plain` uses a non-PI futex so
/// the inversion is left unrepaired and the scheduler must
/// surface the stall.
///
/// Carried as a typed wrapper rather than a `bool` to avoid
/// positional-argument confusion at call sites and so the
/// failure-dump diagnostic names the choice explicitly
/// ("pi_mode = Pi" vs "pi_mode = Plain") instead of a bare
/// boolean.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FutexLockMode {
    /// `FUTEX_LOCK_PI` with rt_mutex PI chain.
    Pi,
    /// Plain futex (no PI boost). The default — exercises the
    /// uncorrected inversion the workload exists to surface.
    #[default]
    Plain,
}

/// Wake mechanism between stages of a `WorkType::WakeChain`.
///
/// Carried as a typed enum rather than a `bool` so call sites
/// name the choice explicitly (`Pipe` / `Futex`) instead of a
/// bare `sync: true` / `sync: false`. The serde wire format is
/// `"pipe"` / `"futex"` (snake_case).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WakeMechanism {
    /// Anon-pipe ring (`depth` pipes per chain). Wakes carry
    /// `WF_SYNC` via `wake_up_interruptible_sync_poll`, biasing
    /// scheduler placement against migration. Tests the
    /// `SCX_WAKE_SYNC` path that scx variants must respect. The
    /// default — see `WakeChain` in `WorkType` for the kernel
    /// call-chain citations.
    #[default]
    Pipe,
    /// Single shared futex word per chain. The active stage
    /// advances the word and `FUTEX_WAKE`s; the stage whose
    /// `pos` matches runs, others re-park. No `WF_SYNC`.
    Futex,
}

/// ALU/SIMD execution width for `WorkType::AluHot`.
///
/// Selects the widest data-path the worker exercises per
/// multiply chain. Today every variant executes the same scalar
/// four-stream multiply chain — the width selector is preserved
/// on the wire so a downstream classifier can distinguish runs
/// that requested SIMD from runs that requested scalar even
/// though the dispatch is uniform. Wider variants WILL drive
/// more functional-unit pressure and (for AVX-512 / AMX) draw
/// the package into a frequency-throttled mode the kernel
/// scheduler must observe once SIMD intrinsics land per-arm.
/// The serde wire form is snake_case (`"scalar"`, `"vec128"`,
/// `"vec256"`, `"vec512"`, `"amx"`, `"widest"`).
///
/// # Current behaviour
///
/// All widths run the same four-stream scalar multiply path;
/// the width selector is preserved on the wire and on
/// [`WorkerReport`](crate::workload::WorkerReport) so a
/// downstream classifier can distinguish runs that requested
/// SIMD from runs that requested scalar even though the
/// dispatch is uniform.
///
/// # Default semantics
///
/// `Scalar` is the type-level Rust default (the
/// `#[derive(Default)]` fallback that serde uses when an
/// `AluWidth` field is missing on the wire — keeps backward-
/// compat for older capture data). `Widest` is the
/// workload-level default the
/// `super::defaults::ALU_HOT_WIDTH` constant resolves at runtime
/// via `resolve_alu_width`: tests that take
/// `WorkType::from_name("AluHot")` get the host's widest
/// available data-path, not the type-level scalar fallback.
/// The asymmetry is deliberate — type-level Default favours
/// "always available everywhere"; workload-level default
/// favours "stress the host as hard as it can run."
///
/// # Resolution rules
///
/// `Widest` is a runtime-resolved sentinel: at worker entry the
/// dispatch arm probes the host CPU via
/// [`std::is_x86_feature_detected!`] (x86_64) and picks the
/// widest available variant in the order
/// `Amx > Vec512 > Vec256 > Vec128 > Scalar`. On `aarch64` only
/// `Scalar` and `Vec128` (NEON) are available; `Vec256` /
/// `Vec512` / `Amx` are absent and `Widest` resolves to NEON
/// when present, falling back to `Scalar`. A configured value
/// that the host cannot run is downgraded to the next-widest
/// available variant with a one-shot `tracing::warn!` so the
/// test still produces useful telemetry rather than
/// hard-failing — silent downgrade without the warn would
/// mask the host capability gap.
///
/// # Frequency throttle on x86_64
///
/// On Intel client / server SKUs the AVX-512 license raises the
/// per-core voltage and lowers the all-core turbo for the
/// package; running [`Vec512`](Self::Vec512) workers under one
/// scheduler while other workers run under another biases the
/// comparison because the throttle is package-wide, not
/// per-task. Tests that A/B-compare schedulers under
/// [`Vec512`](Self::Vec512) or [`Amx`](Self::Amx) need the
/// runs serialized on the same package — the framework does
/// not currently coordinate this serialization across worker
/// groups.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AluWidth {
    /// 64-bit scalar integer multiply chain. Drives the integer
    /// pipeline only; no SIMD or AVX licensing involved.
    /// Available on every supported architecture.
    #[default]
    Scalar,
    /// 128-bit vector integer multiply chain (SSE2 on x86_64,
    /// NEON on aarch64). The widest baseline both architectures
    /// support; a reasonable default when the test cares about
    /// "vectorized ALU" without architecture-specific tuning.
    Vec128,
    /// 256-bit vector integer multiply chain (AVX2 on x86_64).
    /// Not available on aarch64 — falls back to `Vec128`
    /// (NEON) at worker entry with a one-shot warn.
    Vec256,
    /// 512-bit vector integer multiply chain (AVX-512F on
    /// x86_64). Triggers the package-wide frequency throttle
    /// described above. Not available on aarch64 — falls back
    /// to `Vec128` (NEON) at worker entry.
    Vec512,
    /// AMX tile multiply chain (x86_64 server SKUs with AMX-INT8
    /// or AMX-BF16). The widest data-path on x86_64; uses XFD
    /// gating in the kernel
    /// (`arch/x86/kernel/traps.c::handle_xfd_event` raises the
    /// #NM trap, then
    /// `arch/x86/kernel/fpu/xstate.c::__xfd_enable_feature`
    /// allocates the dynamic XSAVE area) so the first AMX
    /// instruction triggers a #NM fault and the kernel allocates
    /// the dynamic XSAVE area lazily — adds a one-time per-task
    /// latency spike on first use.
    ///
    /// AMX additionally requires
    /// `prctl(ARCH_REQ_XCOMP_PERM, XFEATURE_XTILE_DATA)` per
    /// process before the first AMX instruction; the framework
    /// does NOT issue this prctl, so AMX is not yet runnable.
    /// `resolve_alu_width` therefore downgrades `AluWidth::Amx`
    /// to the host's widest stable-detectable variant; AMX is
    /// not currently runnable end-to-end on this framework.
    ///
    /// Not available on aarch64 — falls back to `Vec128`.
    Amx,
    /// Resolve to the widest variant the host supports at
    /// worker entry. See the type-level doc for the resolution
    /// order. Useful as a default when the test author wants
    /// "as much ALU pressure as the host can sustain" without
    /// hardcoding an architecture or feature level.
    Widest,
}

/// Coarse Linux scheduling class identifier.
///
/// Maps to one of the kernel's six core scheduler classes:
/// `fair_sched_class` (CFS / EEVDF — covers `SCHED_NORMAL`,
/// `SCHED_BATCH`, `SCHED_IDLE`), `rt_sched_class` (covers
/// `SCHED_FIFO` and `SCHED_RR`), `dl_sched_class` (covers
/// `SCHED_DEADLINE`), and `ext_sched_class` (covers `SCHED_EXT`
/// when sched_ext is loaded). The class is a coarser concept
/// than [`SchedPolicy`] — `Cfs` covers Normal/Batch/Idle, `Rt`
/// covers Fifo/RoundRobin — and is what
/// `WorkType::AsymmetricWaker` consumes when it wants to
/// describe a waker / wakee pair without specifying priority
/// values. When a per-worker class is applied,
/// `apply_sched_class` maps the variant to the equivalent
/// [`SchedPolicy`] (using a default priority where applicable)
/// and routes through `set_sched_policy`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SchedClass {
    /// `fair_sched_class` — `SCHED_NORMAL` (CFS / EEVDF). The
    /// default; matches a freshly-forked task before any policy
    /// override.
    #[default]
    Cfs,
    /// `fair_sched_class` — `SCHED_BATCH` (background-friendly
    /// fair task with longer wakeup latency targets).
    Batch,
    /// `fair_sched_class` — `SCHED_IDLE` (lowest fair-class
    /// weight; runs only when nothing else is runnable).
    Idle,
    /// `rt_sched_class` — `SCHED_FIFO` at default priority
    /// `RT_DEFAULT_PRIO`. Requires `CAP_SYS_NICE`. For explicit
    /// priority control use [`SchedPolicy::Fifo`] directly.
    Rt,
    /// `dl_sched_class` — `SCHED_DEADLINE`. Maps to a
    /// minimum-bandwidth deadline reservation
    /// ([`SchedClass::default_deadline_reservation`]) so
    /// `SchedClass::Deadline` is constructible without picking
    /// runtime/deadline/period. Callers needing precise
    /// reservations should use [`SchedPolicy::Deadline`]
    /// directly.
    Deadline,
    /// `ext_sched_class` — `SCHED_EXT`. Routes the worker
    /// through the loaded sched_ext BPF scheduler. Under
    /// switch-all (the default scx-ktstr regime), this is the
    /// same effective class as `Cfs` because every fair-policy
    /// task already reroutes to ext via `task_should_scx` (see
    /// kernel/sched/ext.c). `Cfs` is preserved as the explicit
    /// "I want fair semantics" knob the user expresses; `Ext`
    /// is preserved for tests that explicitly want
    /// `policy == SCHED_EXT` set on the task_struct.
    Ext,
}

/// Default `RT_DEFAULT_PRIO` for [`SchedClass::Rt`] when mapped to
/// a [`SchedPolicy`]. Picked at the middle of the 1..=99 valid range
/// so the worker neither preempts every other RT task in the system
/// nor sits at the floor; tests that need a specific RT priority
/// must construct [`SchedPolicy::Fifo`] directly.
const RT_DEFAULT_PRIO: u32 = 50;

impl SchedClass {
    /// Resolve to an equivalent [`SchedPolicy`]. `Rt` uses
    /// `RT_DEFAULT_PRIO`; `Deadline` uses the minimum-bandwidth
    /// reservation (1us runtime over 1ms period — passes
    /// `__checkparam_dl` and the default sysctl bounds).
    /// `Ext` maps to `SchedPolicy::Normal` because there is no
    /// userspace `SCHED_EXT` constant in libc; tests that want
    /// the kernel to read `policy == SCHED_EXT` (which
    /// requires sched_ext-aware userspace) cannot be expressed
    /// via this helper and must call the raw syscall path.
    pub const fn to_policy(self) -> SchedPolicy {
        match self {
            SchedClass::Cfs | SchedClass::Ext => SchedPolicy::Normal,
            SchedClass::Batch => SchedPolicy::Batch,
            SchedClass::Idle => SchedPolicy::Idle,
            SchedClass::Rt => SchedPolicy::Fifo(RT_DEFAULT_PRIO),
            SchedClass::Deadline => Self::default_deadline_reservation(),
        }
    }

    /// Minimum-bandwidth `SCHED_DEADLINE` reservation that passes
    /// `__checkparam_dl`'s `runtime >= DL_SCALE` floor and the
    /// kernel's default `sched_deadline_period_min_us` (100us).
    /// 1us runtime, 1ms deadline, 10ms period — bandwidth fraction
    /// 0.0001, well below admission-control limits.
    pub const fn default_deadline_reservation() -> SchedPolicy {
        SchedPolicy::Deadline {
            runtime: Duration::from_micros(1),
            deadline: Duration::from_millis(1),
            period: Duration::from_millis(10),
        }
    }
}
