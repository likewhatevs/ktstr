//! Progress bars for kernel fetches (tarball download + git clone).
//!
//! Two bar types, both rendered through a single
//! [`indicatif::MultiProgress`] so concurrent fetches (the parallel
//! `cargo ktstr test --kernel A --kernel B` resolve) never garble
//! each other's terminal output, and both degrading to a no-op when
//! stderr is not a TTY (CI, piped output) — the same contract as
//! [`crate::cli::Spinner`].
//!
//! - [`FetchProgress`] — the group handle. One per resolve operation,
//!   shared across the rayon workers by `&` (the inner
//!   [`indicatif::MultiProgress`] is `Send + Sync`). Hands out child
//!   bars and replaces the single-instance [`crate::cli::Spinner`] the
//!   download path previously wrapped (a `Spinner` cannot host
//!   concurrent bars). The build phase renders through this group too:
//!   `kernel_build_pipeline` takes a group and draws the configure /
//!   build / `compile_commands.json` phases as `step_bar` spinners. So
//!   no code path runs a standalone `Spinner` alongside a
//!   live `MultiProgress`, and the parallel resolve's concurrent builds
//!   share the one group rather than racing the process-global
//!   `SPINNER_ACTIVE` guard. The only remaining `Spinner` is
//!   `auto_download_kernel`'s brief version-fetch, which always finishes
//!   before any group bar is created.
//! - [`GroupBar`] — a determinate byte bar for tarball downloads:
//!   transfer rate and ETA derived from `Content-Length`. Falls back
//!   to a live byte counter (rate, no ETA) when the response carries
//!   no `Content-Length`.
//! - [`CloneProgress`] — a determinate object/file bar for git clones,
//!   driven by polling gix's prodash progress tree via
//!   `tree::Root::sorted_snapshot`. Shows a real bar + ETA whenever gix
//!   reports a bounded total — resolving deltas and checkout files
//!   always do; the receiving/read-pack phase does only when the server
//!   advertises a pack size (often unknown for shallow smart-HTTP
//!   clones) — and a spinner otherwise (negotiation, or an unadvertised
//!   pack size). Single renderer (indicatif), zero new dependencies —
//!   prodash's own line renderer is not compiled in this dependency
//!   graph, so the tree is read rather than rendered by prodash.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::JoinHandle;
use std::time::Duration;

use gix::progress::prodash::progress::{Key, Task};
use gix::progress::tree::{Item, Root};
use indicatif::{MultiProgress, ProgressBar, ProgressDrawTarget, ProgressStyle};

use crate::cli::stderr_color;

/// Frame cadence shared by the bars' steady tick and the gix poll
/// loop. The poll loop samples the prodash tree at this rate — a
/// display ticker, not an evented wait: prodash exposes no
/// "counter advanced" notification, only the pull-based
/// `sorted_snapshot`, so a fixed-cadence sample (the same approach
/// prodash's own renderer takes via `frames_per_second`) is the
/// correct shape here.
const TICK: Duration = Duration::from_millis(100);

/// Determinate byte-download template: spinner, label, bar, bytes,
/// rate, ETA. `bytes`/`total_bytes`/`bytes_per_sec` render in binary
/// units (MiB/KiB); `eta` is `Content-Length`-derived.
const DOWNLOAD_TEMPLATE: &str =
    "{spinner:.green} {msg} [{wide_bar:.cyan/blue}] {bytes}/{total_bytes} ({bytes_per_sec}, {eta})";

/// Byte-counter template for a download with no `Content-Length`: no
/// total means no percent/ETA, but a live byte count + rate still
/// shows real movement.
const DOWNLOAD_TEMPLATE_NO_TOTAL: &str = "{spinner:.green} {msg} {bytes} ({bytes_per_sec})";

/// Label-only spinner template for a non-quantifiable phase (kernel
/// configure / build / compile_commands generation) — a moving spinner
/// plus a message, with no byte/percent/ETA fields (there is nothing to
/// count). Mirrors the cyan spinner the old build `Spinner` used.
const STEP_TEMPLATE: &str = "{spinner:.cyan} {msg}";

/// Determinate clone template: spinner, label/phase, bar, count,
/// percent, ETA. Driven by gix's bounded object/file counts; the ETA
/// is derived by indicatif from the count rate.
const CLONE_TEMPLATE: &str =
    "{spinner:.green} {msg} [{wide_bar:.cyan/blue}] {pos}/{len} ({percent}%, {eta})";

/// Spinner template for the brief negotiation window before any
/// bounded gix task exists (no total ⇒ nothing to estimate).
const CLONE_TEMPLATE_SPINNER: &str = "{spinner:.green} {msg}";

/// Progress glyphs for the filled bar (filled, current, remaining).
const PROGRESS_CHARS: &str = "##-";

/// Shared owner of the concurrent fetch progress bars for one resolve
/// operation.
///
/// Wraps an [`indicatif::MultiProgress`] (constructed against a hidden
/// draw target off-TTY) and hands out child bars via `download_bar` /
/// `clone_progress`. Created once per resolve and shared across the rayon
/// `resolve_specs_parallel` workers by `&` reference — `MultiProgress`
/// is `Send + Sync` and `add` is internally serialized on an
/// `Arc<RwLock>`, so concurrent `download_bar`/`clone_progress` calls
/// are safe.
///
/// Off-TTY (`!stderr_color()`) the group draws to a hidden target:
/// every child bar is a no-op, `inc`/`finish` emit nothing, and the
/// gix poll thread is never spawned — piped/CI stderr stays
/// escape-free, matching the [`crate::cli::Spinner`] degradation
/// contract.
pub struct FetchProgress {
    /// Hidden draw target off-TTY; stderr otherwise. Always present
    /// (vs `Option`) so child-bar construction is unconditional and
    /// the hidden/visible branch lives in one place: the draw target.
    multi: MultiProgress,
}

impl FetchProgress {
    /// Construct a fetch-progress group. On a TTY it renders to
    /// stderr; otherwise (non-TTY / `NO_COLOR` / `TERM=dumb`) it uses
    /// a hidden draw target so nothing reaches piped output.
    ///
    /// The `stderr_color()` gate mirrors [`crate::cli::Spinner`];
    /// indicatif additionally auto-hides on a non-TTY, so the hidden
    /// behavior is correct even if the two ever disagreed.
    pub fn new() -> Self {
        let multi = if stderr_color() {
            MultiProgress::new()
        } else {
            MultiProgress::with_draw_target(ProgressDrawTarget::hidden())
        };
        Self { multi }
    }

    /// Whether the group draws to a hidden target (non-TTY / forced).
    /// When hidden, [`Self::clone_progress`] skips its poll thread.
    pub(crate) fn is_hidden(&self) -> bool {
        self.multi.is_hidden()
    }

    /// Add a determinate byte-download bar. `total` is the response
    /// `Content-Length`: `Some` ⇒ percent + ETA; `None` ⇒ a live byte
    /// counter (rate, no ETA). `label` names the kernel being fetched
    /// so concurrent bars are distinguishable. Off-TTY the returned
    /// bar is hidden.
    pub(crate) fn download_bar(&self, label: &str, total: Option<u64>) -> GroupBar {
        let pb = match total {
            Some(len) => ProgressBar::new(len),
            None => ProgressBar::new_spinner(),
        };
        let pb = self.multi.add(pb);
        let template = if total.is_some() {
            DOWNLOAD_TEMPLATE
        } else {
            DOWNLOAD_TEMPLATE_NO_TOTAL
        };
        pb.set_style(
            ProgressStyle::with_template(template)
                .expect("valid download template")
                .progress_chars(PROGRESS_CHARS),
        );
        pb.set_message(label.to_string());
        // Skip the steady-tick ticker thread on the hidden path — it
        // would redraw nothing yet still spawn a thread per bar (the
        // exact CI/non-TTY case this module no-ops).
        if !self.is_hidden() {
            pb.enable_steady_tick(TICK);
        }
        GroupBar { pb }
    }

    /// Add a label-only spinner bar for a non-quantifiable build phase
    /// (kernel configure / build / `compile_commands.json` generation).
    /// Unlike [`Self::download_bar`] with `total = None`, it carries no
    /// byte counter — the build phase has no bytes to count, only a
    /// moving spinner and a message. Off-TTY it is hidden and the
    /// steady-tick thread is skipped, matching [`Self::download_bar`].
    ///
    /// Routing the build phase through the group (rather than a
    /// standalone [`crate::cli::Spinner`]) is what lets concurrent
    /// builds in the parallel resolve render together without racing the
    /// process-global `SPINNER_ACTIVE` guard.
    pub(crate) fn step_bar(&self, label: &str) -> GroupBar {
        let pb = self.multi.add(ProgressBar::new_spinner());
        pb.set_style(ProgressStyle::with_template(STEP_TEMPLATE).expect("valid step template"));
        pb.set_message(label.to_string());
        if !self.is_hidden() {
            pb.enable_steady_tick(TICK);
        }
        GroupBar { pb }
    }

    /// Add a git-clone progress bar and spawn the gix-tree → indicatif
    /// poll bridge. `label` names the clone (its git ref). The bar is
    /// determinate (with ETA) whenever gix reports a bounded task and
    /// a spinner otherwise; see [`CloneProgress`].
    ///
    /// Off-TTY (`is_hidden()`), no poll thread is spawned — the
    /// returned [`CloneProgress`] still yields a valid (no-op) gix
    /// progress sink via [`CloneProgress::item`].
    pub(crate) fn clone_progress(&self, label: &str) -> CloneProgress {
        let root = Root::new();
        let bar = self.multi.add(ProgressBar::new_spinner());
        bar.set_style(
            ProgressStyle::with_template(CLONE_TEMPLATE_SPINNER)
                .expect("valid clone spinner template"),
        );
        bar.set_message(format!("cloning {label}"));

        // Hidden: no rendering, so skip both the steady-tick ticker
        // thread and the poll thread. The tree (root) still exists so
        // `item()` hands gix a valid NestedProgress sink that simply
        // goes unread.
        if self.is_hidden() {
            return CloneProgress {
                root,
                bar,
                stop: Arc::new(AtomicBool::new(false)),
                poller: None,
            };
        }

        bar.enable_steady_tick(TICK);
        let stop = Arc::new(AtomicBool::new(false));
        let poller = std::thread::spawn({
            let root = Arc::clone(&root);
            let bar = bar.clone();
            let stop = Arc::clone(&stop);
            let label = label.to_string();
            move || poll_clone_tree(root, bar, stop, label)
        });
        CloneProgress {
            root,
            bar,
            stop,
            poller: Some(poller),
        }
    }

    /// Print a status line that coordinates with the live bars.
    ///
    /// On a visible group this routes through
    /// [`indicatif::MultiProgress::println`] so the line lands above
    /// the bars without garbling them (a raw `eprintln!` while bars
    /// are drawing corrupts their cursor accounting). On a hidden
    /// group (non-TTY / CI) `MultiProgress::println` is a no-op draw
    /// that would *swallow* the line, so it falls back to `eprintln!`
    /// to preserve the status output piped/CI consumers rely on.
    ///
    /// Best-effort: a `println` error (e.g. broken pipe) is discarded
    /// — a status line must never abort a fetch.
    pub(crate) fn println(&self, line: &str) {
        if self.is_hidden() {
            eprintln!("{line}");
        } else {
            let _ = self.multi.println(line);
        }
    }

    /// Remove all bars from the group. Best-effort: a clear failure is
    /// discarded so it can never mask the resolve result the caller is
    /// returning.
    pub fn clear(&self) {
        let _ = self.multi.clear();
    }
}

impl Default for FetchProgress {
    fn default() -> Self {
        Self::new()
    }
}

/// Poll the gix prodash tree at [`TICK`] cadence and mirror the most
/// relevant task onto `bar` until `stop` is set.
///
/// Picks the deepest task that has a *bounded* counter (`done_at =
/// Some` — a known total) so the bar shows a real position/length and
/// indicatif can compute an ETA. Resolving deltas and checkout files
/// are always bounded; receiving/read-pack is bounded only when the
/// server advertises a pack size. Falls back to the deepest live task's
/// name (spinner only) when no bounded task exists yet (negotiation, or
/// an unadvertised pack size). The bar style is switched only on a
/// determinate/indeterminate transition, not every tick.
fn poll_clone_tree(root: Arc<Root>, bar: ProgressBar, stop: Arc<AtomicBool>, label: String) {
    let mut snapshot: Vec<(Key, Task)> = Vec::new();
    let mut determinate = false;
    while !stop.load(Ordering::Relaxed) {
        poll_tick(&root, &bar, &mut determinate, &label, &mut snapshot);
        std::thread::sleep(TICK);
    }
}

/// One poll iteration: snapshot the tree and mirror the most relevant
/// task onto `bar`. Extracted from [`poll_clone_tree`] so the
/// snapshot→bar mapping is unit-testable without spawning a thread or
/// running a real clone. `determinate` carries the determinate/spinner
/// style state across calls so the style is switched only on a
/// transition, not every tick. `snapshot` is a caller-owned scratch
/// buffer reused across ticks to avoid a per-tick allocation.
fn poll_tick(
    root: &Root,
    bar: &ProgressBar,
    determinate: &mut bool,
    label: &str,
    snapshot: &mut Vec<(Key, Task)>,
) {
    root.sorted_snapshot(snapshot);

    // Deepest bounded task (known total) if any; else deepest task with
    // any live counter. `sorted_snapshot` orders by Key lexicographically
    // (pre-order DFS over the tree); the clone is a single parent chain
    // at any instant (root → "clone" → gix's active child), so the last
    // matching entry is the deepest active task.
    let bounded = snapshot
        .iter()
        .rev()
        .find(|(_, t)| t.progress.as_ref().and_then(|v| v.done_at).is_some());
    let any = snapshot.iter().rev().find(|(_, t)| t.progress.is_some());

    if let Some((_, task)) = bounded {
        // Safe: `bounded` only matched tasks whose progress is Some.
        if let Some(value) = task.progress.as_ref() {
            if !*determinate {
                bar.set_style(
                    ProgressStyle::with_template(CLONE_TEMPLATE)
                        .expect("valid clone template")
                        .progress_chars(PROGRESS_CHARS),
                );
                *determinate = true;
            }
            bar.set_length(value.done_at.unwrap_or(0) as u64);
            bar.set_position(value.step.load(Ordering::SeqCst) as u64);
            bar.set_message(format!("{label}: {}", task.name));
        }
    } else {
        // No bounded task: spinner mode. This covers negotiation, an
        // unadvertised pack size, AND a no-counter task (prodash gives
        // `progress = None` when init is called with neither a max nor a
        // unit, e.g. gix's "negotiate") — in all of which `any` may be
        // None. Flip the style back on the determinate→spinner
        // transition regardless, so the bar never stays stuck showing a
        // stale determinate length; surface the deepest live task's name
        // when one exists.
        if *determinate {
            bar.set_style(
                ProgressStyle::with_template(CLONE_TEMPLATE_SPINNER)
                    .expect("valid clone spinner template"),
            );
            *determinate = false;
        }
        if let Some((_, task)) = any {
            bar.set_message(format!("{label}: {}", task.name));
        }
    }
}

/// A single bar within a [`FetchProgress`] group.
///
/// Backs two uses. A determinate (or byte-counter) download bar
/// ([`FetchProgress::download_bar`]): its inner
/// [`indicatif::ProgressBar`] is handed to
/// `crate::fetch::DownloadStream::with_progress` via [`Self::bar`] and
/// advanced with `inc(n)` beside the stream's own byte accounting — a
/// single source of truth, so `position()` always equals the streamed
/// `bytes_total`. And a label-only build-phase spinner
/// ([`FetchProgress::step_bar`]) that only ever calls [`Self::finish`]
/// (its [`Self::bar`] is unused — there is no stream to attach). Off-TTY
/// the bar is hidden and every method is a no-op draw.
pub(crate) struct GroupBar {
    pb: ProgressBar,
}

impl GroupBar {
    /// A clone of the inner bar to attach to the download stream
    /// (download-bar use only). `ProgressBar` is `Arc`-backed, so the
    /// clone drives the same bar.
    pub(crate) fn bar(&self) -> ProgressBar {
        self.pb.clone()
    }

    /// Finish and clear the bar. Idempotent.
    pub(crate) fn finish(&self) {
        self.pb.finish_and_clear();
    }
}

impl Drop for GroupBar {
    /// Clear the bar if it was not explicitly finished — covers the
    /// download fns' extraction-error exit (bar already created, the
    /// xz/gzip unpack fails via `?` before [`Self::finish`]) that
    /// bails with a live, unfinished bar. HTTP failure and HTML reject
    /// bail before the bar is constructed; a sha256 mismatch bails
    /// after `finish()` has already run, so Drop is only a redundant
    /// no-op there.
    /// Idempotent with `finish`: a second `finish_and_clear` is a
    /// no-op on an already-cleared bar.
    fn drop(&mut self) {
        self.pb.finish_and_clear();
    }
}

/// Live git-clone progress bridge.
///
/// Holds the prodash [`tree::Root`](Root), the indicatif bar, and (on
/// a TTY) the background poll thread spawned by
/// [`FetchProgress::clone_progress`]. [`Self::item`] yields a fresh gix
/// progress sink ([`tree::Item`](Item)) for each gix call; both the
/// fetch and checkout phases feed the one polled tree.
///
/// Shutdown is leak-proof: [`Self::finish`] (success path) and the
/// [`Drop`] impl (a `git_clone` that bails via `?`) both signal the
/// poll thread to stop and join it, then clear the bar — never detached,
/// never leaked. Under the release profile's `panic = "abort"`, `Drop`
/// does not run on a panic, but the whole process aborts so the poll
/// thread cannot outlive it either.
pub(crate) struct CloneProgress {
    /// The prodash progress tree gix writes into. Shared (`Arc`) with
    /// the poll thread, which snapshots it.
    root: Arc<Root>,
    /// The indicatif bar the poll thread drives. Cleared on shutdown.
    bar: ProgressBar,
    /// Set by [`Self::shutdown`] to stop the poll loop.
    stop: Arc<AtomicBool>,
    /// The poll thread handle. `None` off-TTY (no thread spawned) and
    /// after [`Self::shutdown`] takes it to join.
    poller: Option<JoinHandle<()>>,
}

impl CloneProgress {
    /// A fresh gix progress sink for one gix call. `git_clone` calls
    /// this once for the fetch phase and once for checkout; both
    /// children feed the single polled tree, so the one bar reflects
    /// whichever phase is active.
    pub(crate) fn item(&self) -> Item {
        self.root.add_child("clone")
    }

    /// Stop the poll thread, join it, and clear the bar. Consuming
    /// form for the success path; [`Drop`] performs the same work on
    /// the error/panic path.
    pub(crate) fn finish(mut self) {
        self.shutdown();
    }

    /// Idempotent teardown: signal stop, join the poll thread (if any
    /// — `Option::take` makes a second call a no-op), then clear the
    /// bar. The join happens after both gix calls have returned, so
    /// the poll thread is never holding the snapshot lock across it.
    fn shutdown(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(handle) = self.poller.take() {
            let _ = handle.join();
        }
        self.bar.finish_and_clear();
    }
}

impl Drop for CloneProgress {
    fn drop(&mut self) {
        self.shutdown();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A force-hidden group, independent of whether the test host has a
    /// TTY. Constructs `FetchProgress` directly (white-box: the tests
    /// are a child module of `progress`) so no production-only `hidden`
    /// constructor is needed.
    fn hidden_group() -> FetchProgress {
        FetchProgress {
            multi: MultiProgress::with_draw_target(ProgressDrawTarget::hidden()),
        }
    }

    /// Under nextest, stderr is captured (not a TTY), so the default
    /// group degrades to a hidden draw target — the arm CI runs.
    #[test]
    fn fetch_progress_new_is_hidden_under_non_tty() {
        assert!(
            FetchProgress::new().is_hidden(),
            "a non-TTY FetchProgress must use a hidden draw target so piped/CI stderr stays clean",
        );
    }

    /// A determinate download bar tracks `inc` exactly and exposes the
    /// configured total — the contract `DownloadStream` relies on
    /// (position advances by the bytes it streams).
    #[test]
    fn download_bar_determinate_tracks_inc() {
        let fp = hidden_group();
        let bar = fp.download_bar("6.14.2", Some(100));
        bar.pb.inc(40);
        bar.pb.inc(60);
        assert_eq!(bar.pb.position(), 100);
        assert_eq!(bar.pb.length(), Some(100));
    }

    /// With no Content-Length the bar is indeterminate (no total) but
    /// still counts bytes — a live counter rather than a percent bar.
    #[test]
    fn download_bar_indeterminate_when_no_total() {
        let fp = hidden_group();
        let bar = fp.download_bar("6.15-rc3", None);
        assert_eq!(bar.pb.length(), None);
        bar.pb.inc(10);
        assert_eq!(bar.pb.position(), 10);
    }

    /// `finish` is idempotent and the `Drop` impl also clears — neither
    /// the explicit-finish nor the drop-without-finish path panics.
    #[test]
    fn download_bar_finish_and_drop_no_panic() {
        let fp = hidden_group();
        let bar = fp.download_bar("k", Some(10));
        bar.finish();
        bar.finish();
        drop(bar);
        // A bar dropped without an explicit finish (error path) clears
        // via Drop without panic.
        let bar2 = fp.download_bar("k2", Some(10));
        bar2.pb.inc(5);
        drop(bar2);
    }

    /// Adding and driving many bars concurrently from worker threads
    /// (the rayon `resolve_specs_parallel` shape) must not panic or
    /// deadlock — `MultiProgress` serializes `add` on its `RwLock`,
    /// and `FetchProgress` is `Sync` so `&fp` shares across threads.
    #[test]
    fn fetch_progress_concurrent_bars_no_panic() {
        let fp = hidden_group();
        std::thread::scope(|s| {
            for i in 0..16u32 {
                let fp = &fp;
                s.spawn(move || {
                    let bar = fp.download_bar(&format!("k{i}"), Some(50));
                    for _ in 0..50 {
                        bar.pb.inc(1);
                    }
                    assert_eq!(bar.pb.position(), 50, "each bar accounts independently");
                    bar.finish();
                });
            }
        });
        fp.clear();
    }

    /// A hidden group skips the poll thread entirely (nothing to
    /// render), yet still yields a valid gix progress sink via
    /// `item()`, and `finish` is clean.
    #[test]
    fn clone_progress_hidden_skips_poller() {
        let fp = hidden_group();
        let cp = fp.clone_progress("for-next");
        assert!(
            cp.poller.is_none(),
            "hidden group must not spawn the poll thread",
        );
        let _item = cp.item();
        cp.finish();
    }

    /// The poll thread shuts down and joins cleanly. Constructs a
    /// `CloneProgress` over a real (unrendered) tree with the poller
    /// running, then `finish` must stop + join within one tick without
    /// hanging — pinning the leak-proof shutdown path.
    #[test]
    fn clone_progress_finish_joins_poll_thread() {
        let root = Root::new();
        let bar = ProgressBar::hidden();
        let stop = Arc::new(AtomicBool::new(false));
        let poller = std::thread::spawn({
            let root = Arc::clone(&root);
            let bar = bar.clone();
            let stop = Arc::clone(&stop);
            move || poll_clone_tree(root, bar, stop, "for-next".to_string())
        });
        let cp = CloneProgress {
            root,
            bar,
            stop,
            poller: Some(poller),
        };
        // Must not hang: sets stop, the loop exits within one TICK, join
        // returns.
        cp.finish();
    }

    /// A bounded gix task (known total) maps onto a determinate bar:
    /// length = `done_at`, position = `step`. Deterministic — drives
    /// the tree and runs one tick, no thread, no sleep.
    #[test]
    fn poll_tick_maps_bounded_task_to_determinate_bar() {
        let root = Root::new();
        let bar = ProgressBar::hidden();
        let child = root.add_child("receiving objects");
        child.init(Some(200), None);
        child.set(80);

        let mut determinate = false;
        let mut snapshot = Vec::new();
        poll_tick(&root, &bar, &mut determinate, "for-next", &mut snapshot);

        assert!(
            determinate,
            "a bounded task must switch the bar to determinate"
        );
        assert_eq!(bar.length(), Some(200));
        assert_eq!(bar.position(), 80);
    }

    /// An unbounded gix task (no total — e.g. the negotiation phase)
    /// leaves the bar in spinner mode: `determinate` stays false and no
    /// length is set.
    #[test]
    fn poll_tick_unbounded_task_stays_spinner() {
        let root = Root::new();
        let bar = ProgressBar::hidden();
        let child = root.add_child("negotiate");
        child.init(None, None);
        child.set(3);

        let mut determinate = false;
        let mut snapshot = Vec::new();
        poll_tick(&root, &bar, &mut determinate, "for-next", &mut snapshot);

        assert!(
            !determinate,
            "an unbounded task must not flip the bar to determinate",
        );
        assert_eq!(bar.length(), None, "no total ⇒ no determinate length");
    }

    /// Once a bounded task has flipped the bar to determinate, a later
    /// snapshot whose only live task is unbounded flips it back to
    /// spinner mode — the bounded→unbounded transition branch in
    /// `poll_tick` (gix can drop a bounded child before the next bounded
    /// phase appears).
    #[test]
    fn poll_tick_flips_back_to_spinner_when_unbounded() {
        let root = Root::new();
        let bar = ProgressBar::hidden();
        let mut snapshot = Vec::new();
        let mut determinate = false;

        let bounded = root.add_child("resolving deltas");
        bounded.init(Some(50), None);
        bounded.set(10);
        poll_tick(&root, &bar, &mut determinate, "for-next", &mut snapshot);
        assert!(determinate, "a bounded task must flip to determinate first");

        // Drop the bounded task (gix removes a finished child from the
        // tree) and leave only an unbounded one.
        drop(bounded);
        let unbounded = root.add_child("negotiate");
        unbounded.init(None, None);
        unbounded.set(1);
        poll_tick(&root, &bar, &mut determinate, "for-next", &mut snapshot);
        assert!(
            !determinate,
            "an unbounded-only snapshot must flip the bar back to spinner mode",
        );
    }

    /// `println` on a hidden group selects the `eprintln!` arm (gated by
    /// `is_hidden()`) rather than `multi.println` — which would swallow
    /// the line on a hidden draw target — and does not panic. Pins
    /// branch selection + no-panic; like the sibling `Spinner` tests it
    /// does not capture stderr to assert the bytes are emitted (that is
    /// std `eprintln!` behavior). The anti-swallow rationale is
    /// documented on [`FetchProgress::println`] itself.
    #[test]
    fn fetch_progress_println_hidden_no_panic() {
        let fp = hidden_group();
        // is_hidden() == true selects the eprintln! arm inside println.
        assert!(fp.is_hidden());
        fp.println("cargo ktstr: downloading linux-6.14.2 (130.0 MiB)");
    }

    /// Several real poll threads created and shut down concurrently from
    /// worker threads (the rayon multi-`git+URL` fan-out shape) must
    /// neither panic nor hang — pins the concurrent clone-shutdown path
    /// that the single-threaded `clone_progress_finish_joins_poll_thread`
    /// test does not reach.
    #[test]
    fn concurrent_clone_progress_shutdown_no_panic() {
        std::thread::scope(|s| {
            for _ in 0..8 {
                s.spawn(|| {
                    let root = Root::new();
                    let bar = ProgressBar::hidden();
                    let stop = Arc::new(AtomicBool::new(false));
                    let poller = std::thread::spawn({
                        let root = Arc::clone(&root);
                        let bar = bar.clone();
                        let stop = Arc::clone(&stop);
                        move || poll_clone_tree(root, bar, stop, "ref".to_string())
                    });
                    let cp = CloneProgress {
                        root,
                        bar,
                        stop,
                        poller: Some(poller),
                    };
                    let _ = cp.item();
                    cp.finish();
                });
            }
        });
    }

    /// A `step_bar` is a label-only spinner: no byte counter (length is
    /// `None`), and finish + drop-without-finish are both clean.
    #[test]
    fn step_bar_no_counter_no_panic() {
        let fp = hidden_group();
        let bar = fp.step_bar("Building kernel...");
        assert_eq!(
            bar.pb.length(),
            None,
            "a step bar has no quantifiable total"
        );
        bar.finish();
        // Drop-without-finish (the build error path) must also be clean.
        drop(fp.step_bar("Configuring kernel..."));
    }

    /// Many build-phase `step_bar`s driven concurrently on one group
    /// (the parallel-resolve shape where multiple workers build at once)
    /// must not panic. Direct regression pin for the old
    /// concurrent-`Spinner` `SPINNER_ACTIVE` race that this change
    /// removes by routing the build phase through the group instead of a
    /// standalone `Spinner`.
    #[test]
    fn concurrent_step_bars_no_panic() {
        let fp = hidden_group();
        std::thread::scope(|s| {
            for i in 0..16u32 {
                let fp = &fp;
                s.spawn(move || {
                    let bar = fp.step_bar(&format!("Building kernel {i}..."));
                    bar.finish();
                });
            }
        });
        fp.clear();
    }
}
