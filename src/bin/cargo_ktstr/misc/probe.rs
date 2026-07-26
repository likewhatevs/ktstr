//! Shared probe-loop scaffolding for cargo-ktstr subcommands which execute
//! selected test binaries with private discriminator flags.
//!
//! Both [`super::export::run_export`] and
//! `super::shell::resolve_shell_from_test_entry` consume the same
//! [`super::export::build_test_binaries`] output. [`probe_collect`] walks every
//! shell descriptor candidate. [`probe_first`] walks bounded export-acceptance
//! candidates until the first success, then executes only that owner through a
//! distinct long-running command. Their shared exit-code categorisation
//! (0 = accepted, 2 = "registered but rejected", other = "not registered
//! here") and miss-stderr bookkeeping live in [`process_bin_with_runner`].
//!
//! Quick shell and export-acceptance discriminators have a 60-second child
//! deadline. Scheduler manifests take a separate no-exec path which reads
//! versioned ELF records. Export first selects an accepted owner through the
//! bounded process path, then invokes only that binary's real exporter through
//! an unbounded anchored runner which tees both output streams and emits
//! elapsed heartbeats while scheduler build and executable packaging proceed.
//!
//! The export/shell exit-code contract and scheduler-manifest ELF wire format
//! are private agreements between cargo-ktstr and test-support, not
//! ktstr-library general capabilities.

use std::collections::{HashMap, HashSet};
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use super::export::build_test_binaries;

const PROBE_CHILD_TIMEOUT: Duration = Duration::from_secs(60);
const EXPORT_HEARTBEAT_INTERVAL: Duration = Duration::from_secs(10);
const SCHEDULER_STAMP_READ_MAX_PARALLELISM: usize = 16;

/// One decoded combined scheduler manifest retained with the executable which
/// emitted it. `executable` is rewritten through `provenance` when the caller
/// probes descriptor-backed warmed binaries.
#[derive(Debug)]
pub(crate) struct ProbedSchedulerManifest {
    pub executable: PathBuf,
    pub manifest: ktstr::test_support::SchedulerManifestProbe,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ProbeStreams {
    Capture,
    Tee,
}

impl ProbeStreams {
    fn tees_stdout(self) -> bool {
        matches!(self, Self::Tee)
    }

    fn tees_stderr(self) -> bool {
        matches!(self, Self::Tee)
    }
}

struct ProbeHeartbeat {
    started: Instant,
    next: Instant,
    interval: Duration,
    binary: PathBuf,
}

impl ProbeHeartbeat {
    fn new_at(binary: &Path, started: Instant, interval: Duration) -> Self {
        assert!(
            !interval.is_zero(),
            "probe heartbeat interval must be non-zero"
        );
        Self {
            started,
            next: started + interval,
            interval,
            binary: binary.to_path_buf(),
        }
    }

    fn next_tick_in_at(&self, now: Instant) -> Duration {
        self.next.saturating_duration_since(now)
    }

    fn message_at(&mut self, now: Instant) -> Option<String> {
        if now < self.next {
            return None;
        }
        while self.next <= now {
            self.next += self.interval;
        }
        Some(format!(
            "cargo ktstr: export via {} still running ({:.1}s elapsed)\n",
            self.binary.display(),
            now.saturating_duration_since(self.started).as_secs_f64(),
        ))
    }
}

struct ProbeObserver {
    deadline: Option<(Instant, String)>,
    streams: ProbeStreams,
    heartbeat: Option<ProbeHeartbeat>,
    forward_error: Option<String>,
}

impl ProbeObserver {
    fn bounded(description: &str, binary: &Path, timeout: Duration, streams: ProbeStreams) -> Self {
        Self {
            deadline: Some((
                Instant::now() + timeout,
                format!(
                    "{description} probe in {} exceeded {:.1}s; its complete descendant \
                     subtree was terminated",
                    binary.display(),
                    timeout.as_secs_f64(),
                ),
            )),
            streams,
            heartbeat: None,
            forward_error: None,
        }
    }

    fn export(binary: &Path) -> Self {
        Self::export_at(binary, Instant::now(), EXPORT_HEARTBEAT_INTERVAL)
    }

    fn export_at(binary: &Path, started: Instant, heartbeat_interval: Duration) -> Self {
        Self {
            deadline: None,
            streams: ProbeStreams::Tee,
            heartbeat: Some(ProbeHeartbeat::new_at(binary, started, heartbeat_interval)),
            forward_error: None,
        }
    }

    fn cancellation_error_at(&self, now: Instant) -> Option<std::io::Error> {
        if let Some(error) = &self.forward_error {
            return Some(std::io::Error::other(error.clone()));
        }
        self.deadline
            .as_ref()
            .and_then(|(deadline, diagnostic)| probe_cancellation_error(*deadline, now, diagnostic))
    }

    fn next_tick_in_at(&self, now: Instant) -> Duration {
        let deadline = self
            .deadline
            .as_ref()
            .map(|(deadline, _)| deadline.saturating_duration_since(now));
        let heartbeat = self
            .heartbeat
            .as_ref()
            .map(|heartbeat| heartbeat.next_tick_in_at(now));
        match (deadline, heartbeat) {
            (Some(deadline), Some(heartbeat)) => deadline.min(heartbeat),
            (Some(deadline), None) => deadline,
            (None, Some(heartbeat)) => heartbeat,
            (None, None) => Duration::from_secs(1),
        }
    }
}

fn probe_cancellation_error(
    deadline: Instant,
    now: Instant,
    diagnostic: &str,
) -> Option<std::io::Error> {
    (now >= deadline)
        .then(|| std::io::Error::new(std::io::ErrorKind::TimedOut, diagnostic.to_string()))
}

impl crate::interrupt::StdoutObserver for ProbeObserver {
    fn observe_stdout(&mut self, bytes: &[u8]) {
        if self.streams.tees_stdout() && self.forward_error.is_none() {
            let mut stdout = std::io::stdout().lock();
            if let Err(error) = stdout.write_all(bytes).and_then(|()| stdout.flush()) {
                self.forward_error = Some(format!("forward probe stdout: {error}"));
            }
        }
    }

    fn observe_stderr(&mut self, bytes: &[u8]) {
        if self.streams.tees_stderr() && self.forward_error.is_none() {
            let mut stderr = std::io::stderr().lock();
            if let Err(error) = stderr.write_all(bytes).and_then(|()| stderr.flush()) {
                self.forward_error = Some(format!("forward probe stderr: {error}"));
            }
        }
    }

    fn tick(&mut self) {
        let message = self
            .heartbeat
            .as_mut()
            .and_then(|heartbeat| heartbeat.message_at(Instant::now()));
        if let Some(message) = message
            && self.forward_error.is_none()
        {
            let mut stderr = std::io::stderr().lock();
            if let Err(error) = stderr
                .write_all(ktstr::cli::status_line(&message).as_bytes())
                .and_then(|()| stderr.flush())
            {
                self.forward_error = Some(format!("write probe heartbeat: {error}"));
            }
        }
    }

    fn next_tick_in(&self) -> Duration {
        self.next_tick_in_at(Instant::now())
    }

    fn cancellation_error(&mut self) -> Option<std::io::Error> {
        self.cancellation_error_at(Instant::now())
    }

    fn finished(&mut self, _status: &std::process::ExitStatus) {}

    fn failed(&mut self, _error: &std::io::Error) {}
}

fn run_bounded_probe_output(
    description: &str,
    binary: &Path,
    command: Command,
    streams: ProbeStreams,
) -> std::io::Result<Output> {
    let observer = ProbeObserver::bounded(description, binary, PROBE_CHILD_TIMEOUT, streams);
    crate::interrupt::run_output_observed_anchored(command, observer)
}

fn run_unbounded_export_output(binary: &Path, command: Command) -> std::io::Result<Output> {
    let observer = ProbeObserver::export(binary);
    crate::interrupt::run_output_observed_anchored(command, observer)
}

/// Execute every supplied binary once through the supplied runner. Numeric
/// non-zero exits mean that binary does not link the requested probe ctor and
/// remain ordinary misses. Signals, runner errors, and successful-output
/// decode errors are terminal and count as failed progress items.
#[cfg(test)]
fn collect_probe_outputs<T>(
    bins: &[PathBuf],
    configure_cmd: impl Fn(&Path) -> Command,
    on_success: impl Fn(&Path, &Output) -> Result<T, String>,
    mut run: impl FnMut(&Path, Command) -> std::io::Result<Output>,
    mut item_finished: impl FnMut(bool),
) -> Result<Vec<T>, String> {
    let mut collected = Vec::new();
    for binary in bins {
        let output = match run(binary, configure_cmd(binary)) {
            Ok(output) => output,
            Err(error) => {
                item_finished(false);
                return Err(format!(
                    "exec scheduler probe {}: {error}",
                    binary.display(),
                ));
            }
        };
        if output.status.success() {
            match on_success(binary, &output) {
                Ok(value) => {
                    collected.push(value);
                    item_finished(true);
                }
                Err(error) => {
                    item_finished(false);
                    return Err(error);
                }
            }
        } else if output.status.code().is_none() {
            item_finished(false);
            return Err(format!(
                "scheduler probe {} terminated by {}",
                binary.display(),
                output.status,
            ));
        } else {
            item_finished(true);
        }
    }
    Ok(collected)
}

/// Read independent ELF stamps with bounded host parallelism while retaining
/// input-ordered results and errors.
///
/// Completion reporting follows real completion order so long reads continue
/// to produce useful progress. Semantic consumption happens only after every
/// worker joins and walks the indexed result vector, making the selected error
/// deterministic even when a later binary fails first.
fn read_scheduler_stamps_parallel_with<T: Send>(
    bins: &[PathBuf],
    max_parallelism: usize,
    read: impl Fn(&Path) -> Result<Option<T>, String> + Sync,
    mut item_finished: impl FnMut(bool),
) -> Result<Vec<Option<T>>, String> {
    if bins.is_empty() {
        return Ok(Vec::new());
    }
    let workers = max_parallelism.max(1).min(bins.len());
    let next = AtomicUsize::new(0);
    let (sender, receiver) = std::sync::mpsc::channel();
    let mut results = std::iter::repeat_with(|| None)
        .take(bins.len())
        .collect::<Vec<Option<Result<Option<T>, String>>>>();

    std::thread::scope(|scope| {
        for _ in 0..workers {
            let sender = sender.clone();
            let read = &read;
            let next = &next;
            scope.spawn(move || {
                loop {
                    let index = next.fetch_add(1, Ordering::Relaxed);
                    let Some(binary) = bins.get(index) else {
                        break;
                    };
                    if sender.send((index, read(binary))).is_err() {
                        break;
                    }
                }
            });
        }
        drop(sender);
        for (index, result) in receiver {
            item_finished(result.is_ok());
            results[index] = Some(result);
        }
    });

    results
        .into_iter()
        .enumerate()
        .map(|(index, result)| {
            result.ok_or_else(|| {
                format!(
                    "scheduler stamp worker returned no result for {} (input index {index})",
                    bins[index].display(),
                )
            })?
        })
        .collect()
}

fn scheduler_stamp_read_parallelism(binary_count: usize) -> usize {
    std::thread::available_parallelism()
        .map_or(1, usize::from)
        .min(SCHEDULER_STAMP_READ_MAX_PARALLELISM)
        .min(binary_count.max(1))
}

/// Read the combined scheduler payload from every distinct selected binary.
///
/// The payload is a versioned, link-retained ELF stamp emitted by
/// `declare_scheduler!` and `#[ktstr_test]`; no candidate binary is executed,
/// so discovery is independent of the dynamic loader. Descriptor-backed
/// verifier binaries can provide a map back to Cargo's canonical emitted
/// executable so cell ownership retains stable provenance after the descriptors
/// are dropped.
pub(crate) fn probe_scheduler_manifests_from_bins(
    bins: &[PathBuf],
    provenance: Option<&HashMap<PathBuf, PathBuf>>,
    description: &str,
) -> Result<Vec<ProbedSchedulerManifest>, String> {
    if bins.is_empty() {
        return Ok(Vec::new());
    }
    let mut bins = bins.to_vec();
    bins.sort();
    bins.dedup();

    let mut progress = crate::run_cargo::ItemProgress::start(
        &format!("cargo ktstr: reading {description}"),
        bins.len(),
    );
    let result: Result<Vec<ProbedSchedulerManifest>, String> = (|| {
        let reads = read_scheduler_stamps_parallel_with(
            &bins,
            scheduler_stamp_read_parallelism(bins.len()),
            |bin| {
                let Some(manifest) =
                    ktstr::test_support::read_scheduler_manifest_and_validate_admission_stamp(bin)?
                else {
                    return Ok(None);
                };
                let executable = if let Some(provenance) = provenance {
                    provenance.get(bin).cloned().ok_or_else(|| {
                        format!(
                            "scheduler-manifest probe path {} has no warmed Cargo \
                             executable provenance",
                            bin.display(),
                        )
                    })?
                } else {
                    bin.to_path_buf()
                };
                Ok(Some(ProbedSchedulerManifest {
                    executable,
                    manifest,
                }))
            },
            |success| progress.item_finished(success),
        )?;
        let mut manifests = reads.into_iter().flatten().collect::<Vec<_>>();
        manifests.sort_by(|left, right| left.executable.cmp(&right.executable));
        let mut seen = HashSet::new();
        manifests.retain(|binary| seen.insert(binary.executable.clone()));
        Ok(manifests)
    })();
    match &result {
        Ok(_) => progress.finish_success(),
        Err(error) => progress.finish_error(error),
    }
    result
}

/// Aggregate of all-binaries-tried-and-missed. The caller renders
/// a subject-specific message via [`ProbeMiss::render`].
#[derive(Debug)]
pub(crate) struct ProbeMiss {
    pub bins_tried: usize,
    pub rejection_stderr: Option<String>,
    pub last_miss_stderr: String,
}

impl ProbeMiss {
    /// Render the miss diagnostic. `rejection_subject` slots into
    /// the "registered but {rejection_subject}" template (e.g.
    /// `"cannot be exported"`, `"cannot be used for shell mode"`);
    /// the trailing "not found in any workspace test binary" form
    /// is identical across callers.
    pub(crate) fn render(&self, test: &str, rejection_subject: &str) -> String {
        if let Some(reason) = &self.rejection_stderr {
            return format!(
                "test '{test}' is registered but {rejection_subject}:\n{}",
                reason.trim_end(),
            );
        }
        format!(
            "test '{test}' not found in any workspace test binary ({} candidates tried). \
             Last stderr from a candidate:\n{}",
            self.bins_tried,
            self.last_miss_stderr.trim_end(),
        )
    }
}

/// Helper-level error: either setup failed
/// ([`ProbeError::Setup`] — `build_test_binaries` errored or
/// returned empty, either phase hit an I/O/status error, or on_success bubbled
/// up a String) or every bounded discriminator missed
/// ([`ProbeError::Miss`]).
#[derive(Debug)]
pub(crate) enum ProbeError {
    Setup(String),
    Miss(ProbeMiss),
}

/// Per-binary categorised outcome — only the success arm carries
/// payload. exit-1 / exit-2 / other-nonzero outcomes are bucketed
/// into the caller's `&mut rejection_stderr` / `&mut
/// last_miss_stderr` accumulators by [`process_bin_with_runner`] before
/// returning [`BinOutcome::Continue`].
enum BinOutcome<T> {
    Success(T),
    Continue,
}

/// Exec one bin via `configure_cmd`, bucket its outcome:
/// exit 0 → call `on_success`, return [`BinOutcome::Success`].
/// exit 2 → set `rejection_stderr` IFF currently None (first-
/// rejection-wins), return [`BinOutcome::Continue`].
/// other non-zero → overwrite `last_miss_stderr` (last-write-
/// wins for miss diagnostics), return [`BinOutcome::Continue`].
/// I/O error spawning the child → [`ProbeError::Setup`].
fn process_bin_with_runner<T>(
    bin: &Path,
    configure_cmd: &impl Fn(&Path) -> Command,
    on_success: &impl Fn(&Path, &Output) -> Result<T, String>,
    rejection_stderr: &mut Option<String>,
    last_miss_stderr: &mut String,
    run: &mut impl FnMut(&Path, Command) -> std::io::Result<Output>,
) -> Result<BinOutcome<T>, ProbeError> {
    let out = run(bin, configure_cmd(bin))
        .map_err(|e| ProbeError::Setup(format!("exec {}: {e}", bin.display())))?;
    if out.status.success() {
        let value = on_success(bin, &out).map_err(ProbeError::Setup)?;
        return Ok(BinOutcome::Success(value));
    }
    if out.status.code().is_none() {
        return Err(ProbeError::Setup(format!(
            "probe {} terminated by {}",
            bin.display(),
            out.status,
        )));
    }
    let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
    // Exit 2 = "registered but rejected here" (host_only,
    // bpf_map_write, KernelBuiltin, etc.). ALWAYS the most
    // informative outcome — save the FIRST and keep probing;
    // other candidates might still admit the test. Exit 1 (and
    // other non-2 non-zero) = "not registered here"; overwrite
    // each pass so the operator sees the most-recent miss
    // diagnostic.
    if out.status.code() == Some(2) {
        if rejection_stderr.is_none() {
            *rejection_stderr = Some(stderr);
        }
    } else {
        *last_miss_stderr = stderr;
    }
    Ok(BinOutcome::Continue)
}

/// Empty-bins diagnostic shared by [`probe_first`] /
/// [`probe_collect`] + their `_with_bins_*` cores. The message
/// exactly matches the pre-refactor `run_export` /
/// `resolve_shell_from_test_entry` text so failure shape is
/// preserved across the refactor.
const EMPTY_BINS_SETUP_ERROR: &str = "cargo build --tests produced no executable artifacts; \
     ensure the workspace has at least one [[test]] target or \
     a [lib]/[bin] with #[cfg(test)] tests";

/// Select and execute the first export-capable workspace test binary.
///
/// `package` + `release` flow straight to
/// [`build_test_binaries`]. `configure_check_cmd` creates the cheap private
/// ownership/eligibility discriminator run under the 60-second bounded
/// anchored observer. Once one candidate accepts, `configure_execute_cmd`
/// creates the real exporter and only that selected binary runs again, through
/// the unbounded anchored observer with live stdout/stderr and heartbeats.
///
/// Errors: [`ProbeError::Setup`] for build failure / empty
/// executables / either runner's I/O failure / selected-export failure /
/// `on_success` error. [`ProbeError::Miss`] when every bounded acceptance
/// probe returned non-zero.
pub(crate) fn probe_first<T>(
    package: Option<&str>,
    release: bool,
    configure_check_cmd: impl Fn(&Path) -> Command,
    configure_execute_cmd: impl Fn(&Path) -> Command,
    on_success: impl Fn(&Path, &Output) -> Result<T, String>,
) -> Result<T, ProbeError> {
    let bins = build_test_binaries(package, release).map_err(ProbeError::Setup)?;
    probe_first_with_bins(
        &bins,
        configure_check_cmd,
        configure_execute_cmd,
        on_success,
    )
}

/// Walk every workspace test binary, call `on_success` on each
/// hit, return the accumulated Vec. Used by
/// `resolve_shell_from_test_entry` (walk-all + ambiguity-bail
/// semantics — caller checks `Vec::len()` after the walk).
///
/// Each child is bounded by [`PROBE_CHILD_TIMEOUT`]. See [`probe_first`] for
/// argument and error semantics.
pub(crate) fn probe_collect<T>(
    package: Option<&str>,
    release: bool,
    configure_cmd: impl Fn(&Path) -> Command,
    on_success: impl Fn(&Path, &Output) -> Result<T, String>,
) -> Result<Vec<T>, ProbeError> {
    let bins = build_test_binaries(package, release).map_err(ProbeError::Setup)?;
    probe_collect_with_bins(&bins, configure_cmd, on_success)
}

/// Unit-testable core of [`probe_first`]: bins pre-built by
/// caller. Separated so the loop + bookkeeping + Setup/Miss
/// dispatch can be exercised without spawning a real
/// `cargo build --tests`.
///
/// PRIVATE on purpose: scheduler-manifest callers with an authoritative
/// prebuilt set use [`probe_scheduler_manifests_from_bins`]; ordinary
/// subcommands use [`probe_first`] or [`probe_collect`] so they cannot
/// accidentally bypass the canonical build. Tests below reach this fn via
/// Rust's same-module sibling visibility.
fn probe_first_with_bins<T>(
    bins: &[PathBuf],
    configure_check_cmd: impl Fn(&Path) -> Command,
    configure_execute_cmd: impl Fn(&Path) -> Command,
    on_success: impl Fn(&Path, &Output) -> Result<T, String>,
) -> Result<T, ProbeError> {
    probe_first_with_bins_using(
        bins,
        configure_check_cmd,
        configure_execute_cmd,
        on_success,
        |binary, command| {
            run_bounded_probe_output("export acceptance", binary, command, ProbeStreams::Capture)
        },
        run_unbounded_export_output,
    )
}

fn probe_first_with_bins_using<T>(
    bins: &[PathBuf],
    configure_check_cmd: impl Fn(&Path) -> Command,
    configure_execute_cmd: impl Fn(&Path) -> Command,
    on_success: impl Fn(&Path, &Output) -> Result<T, String>,
    mut run_check: impl FnMut(&Path, Command) -> std::io::Result<Output>,
    mut run_execute: impl FnMut(&Path, Command) -> std::io::Result<Output>,
) -> Result<T, ProbeError> {
    if bins.is_empty() {
        return Err(ProbeError::Setup(EMPTY_BINS_SETUP_ERROR.to_string()));
    }
    let mut rejection_stderr: Option<String> = None;
    let mut last_miss_stderr = String::new();
    let accepted = |binary: &Path, _output: &Output| Ok::<PathBuf, String>(binary.to_path_buf());
    for bin in bins {
        match process_bin_with_runner(
            bin,
            &configure_check_cmd,
            &accepted,
            &mut rejection_stderr,
            &mut last_miss_stderr,
            &mut run_check,
        )? {
            BinOutcome::Success(selected) => {
                return execute_selected_with_runner(
                    &selected,
                    &configure_execute_cmd,
                    &on_success,
                    &mut run_execute,
                );
            }
            BinOutcome::Continue => continue,
        }
    }
    Err(ProbeError::Miss(ProbeMiss {
        bins_tried: bins.len(),
        rejection_stderr,
        last_miss_stderr,
    }))
}

fn execute_selected_with_runner<T>(
    binary: &Path,
    configure_cmd: &impl Fn(&Path) -> Command,
    on_success: &impl Fn(&Path, &Output) -> Result<T, String>,
    run: &mut impl FnMut(&Path, Command) -> std::io::Result<Output>,
) -> Result<T, ProbeError> {
    let output = run(binary, configure_cmd(binary)).map_err(|error| {
        ProbeError::Setup(format!(
            "exec selected exporter {}: {error}",
            binary.display(),
        ))
    })?;
    if !output.status.success() {
        return Err(ProbeError::Setup(format!(
            "selected exporter {} failed with {}",
            binary.display(),
            output.status,
        )));
    }
    on_success(binary, &output).map_err(ProbeError::Setup)
}

/// Unit-testable core of [`probe_collect`]: bins pre-built by
/// caller. See [`probe_first_with_bins`] for the
/// PRIVATE-on-purpose rationale.
fn probe_collect_with_bins<T>(
    bins: &[PathBuf],
    configure_cmd: impl Fn(&Path) -> Command,
    on_success: impl Fn(&Path, &Output) -> Result<T, String>,
) -> Result<Vec<T>, ProbeError> {
    probe_collect_with_bins_using(bins, configure_cmd, on_success, |binary, command| {
        run_bounded_probe_output("test registration", binary, command, ProbeStreams::Capture)
    })
}

fn probe_collect_with_bins_using<T>(
    bins: &[PathBuf],
    configure_cmd: impl Fn(&Path) -> Command,
    on_success: impl Fn(&Path, &Output) -> Result<T, String>,
    mut run: impl FnMut(&Path, Command) -> std::io::Result<Output>,
) -> Result<Vec<T>, ProbeError> {
    if bins.is_empty() {
        return Err(ProbeError::Setup(EMPTY_BINS_SETUP_ERROR.to_string()));
    }
    let mut collected: Vec<T> = Vec::new();
    let mut rejection_stderr: Option<String> = None;
    let mut last_miss_stderr = String::new();
    for bin in bins {
        match process_bin_with_runner(
            bin,
            &configure_cmd,
            &on_success,
            &mut rejection_stderr,
            &mut last_miss_stderr,
            &mut run,
        )? {
            BinOutcome::Success(value) => collected.push(value),
            BinOutcome::Continue => continue,
        }
    }
    if !collected.is_empty() {
        return Ok(collected);
    }
    Err(ProbeError::Miss(ProbeMiss {
        bins_tried: bins.len(),
        rejection_stderr,
        last_miss_stderr,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fake_bin(idx: usize) -> PathBuf {
        PathBuf::from(format!("/fake/bin{idx}"))
    }

    fn probe_output(code: i32, stdout: &[u8], stderr: &[u8]) -> Output {
        use std::os::unix::process::ExitStatusExt as _;

        Output {
            status: std::process::ExitStatus::from_raw(code << 8),
            stdout: stdout.to_vec(),
            stderr: stderr.to_vec(),
        }
    }

    fn fake_bin_index(binary: &Path) -> usize {
        binary
            .file_name()
            .and_then(|name| name.to_str())
            .and_then(|name| name.strip_prefix("bin"))
            .and_then(|index| index.parse().ok())
            .expect("fake binary carries its numeric index")
    }

    #[test]
    fn parallel_stamp_reads_are_bounded_and_preserve_input_order() {
        let bins = (0..8).map(fake_bin).collect::<Vec<_>>();
        let gate = std::sync::Barrier::new(3);
        let active = AtomicUsize::new(0);
        let maximum = AtomicUsize::new(0);
        let mut completed = Vec::new();
        let results = read_scheduler_stamps_parallel_with(
            &bins,
            3,
            |binary| {
                let index = fake_bin_index(binary);
                let current_active = active.fetch_add(1, Ordering::SeqCst) + 1;
                maximum.fetch_max(current_active, Ordering::SeqCst);
                if index < 3 {
                    gate.wait();
                }
                std::thread::yield_now();
                active.fetch_sub(1, Ordering::SeqCst);
                Ok((index != 5).then_some(index))
            },
            |success| completed.push(success),
        )
        .expect("parallel reads succeed");

        assert_eq!(maximum.load(Ordering::SeqCst), 3);
        assert_eq!(completed, vec![true; bins.len()]);
        assert_eq!(
            results,
            vec![
                Some(0),
                Some(1),
                Some(2),
                Some(3),
                Some(4),
                None,
                Some(6),
                Some(7),
            ],
            "worker completion order must not reorder the semantic results",
        );
    }

    #[test]
    fn parallel_stamp_reads_report_the_first_input_error_after_all_workers_finish() {
        let bins = (0..4).map(fake_bin).collect::<Vec<_>>();
        let gate = std::sync::Barrier::new(4);
        let mut completed = Vec::new();
        let error = read_scheduler_stamps_parallel_with(
            &bins,
            4,
            |binary| {
                let index = fake_bin_index(binary);
                gate.wait();
                match index {
                    1 => {
                        std::thread::sleep(Duration::from_millis(10));
                        Err("earlier input failed later".to_string())
                    }
                    2 => Err("later input failed first".to_string()),
                    _ => Ok(Some(index)),
                }
            },
            |success| completed.push(success),
        )
        .expect_err("two reads fail");

        assert_eq!(error, "earlier input failed later");
        assert_eq!(completed.len(), bins.len());
        assert_eq!(completed.iter().filter(|success| !**success).count(), 2);
    }

    #[test]
    fn probe_deadline_boundary_and_diagnostic_are_deterministic() {
        let deadline = Instant::now();
        assert!(
            probe_cancellation_error(
                deadline,
                deadline - Duration::from_nanos(1),
                "probe deadline",
            )
            .is_none(),
        );
        let error = probe_cancellation_error(deadline, deadline, "probe deadline")
            .expect("the exact deadline cancels");
        assert_eq!(error.kind(), std::io::ErrorKind::TimedOut);
        assert_eq!(error.to_string(), "probe deadline");
    }

    #[test]
    fn accepted_export_survives_beyond_probe_deadline() {
        let started = Instant::now();
        let observer = ProbeObserver::export_at(
            Path::new("/fake/exporter"),
            started,
            Duration::from_secs(10),
        );
        assert!(
            observer
                .cancellation_error_at(started + PROBE_CHILD_TIMEOUT + Duration::from_secs(1),)
                .is_none(),
            "the selected long exporter has no acceptance-probe deadline",
        );
    }

    #[test]
    fn export_observer_tees_both_streams_and_advances_heartbeat() {
        let started = Instant::now();
        let mut observer = ProbeObserver::export_at(
            Path::new("/fake/exporter"),
            started,
            Duration::from_secs(10),
        );
        assert_eq!(observer.streams, ProbeStreams::Tee);
        assert!(observer.streams.tees_stdout());
        assert!(observer.streams.tees_stderr());
        assert_eq!(observer.next_tick_in_at(started), Duration::from_secs(10),);

        let heartbeat = observer
            .heartbeat
            .as_mut()
            .expect("export observer carries heartbeat state");
        assert!(
            heartbeat
                .message_at(started + Duration::from_secs(9))
                .is_none(),
        );
        assert_eq!(
            heartbeat.message_at(started + Duration::from_secs(10)),
            Some(
                "cargo ktstr: export via /fake/exporter still running \
                 (10.0s elapsed)\n"
                    .to_string(),
            ),
        );
        assert!(
            heartbeat
                .message_at(started + Duration::from_secs(10))
                .is_none(),
            "one deadline produces one heartbeat",
        );
        assert_eq!(
            heartbeat.message_at(started + Duration::from_secs(25)),
            Some(
                "cargo ktstr: export via /fake/exporter still running \
                 (25.0s elapsed)\n"
                    .to_string(),
            ),
            "a delayed tick reports real elapsed time and advances past missed slots",
        );
        assert_eq!(
            heartbeat.next_tick_in_at(started + Duration::from_secs(25)),
            Duration::from_secs(5),
        );
    }

    #[test]
    fn bounded_probe_core_visits_each_binary_once_in_order() {
        let bins = vec![fake_bin(0), fake_bin(1), fake_bin(2)];
        let visited = std::cell::RefCell::new(Vec::new());
        let outcomes = std::cell::RefCell::new(Vec::new());
        let values = collect_probe_outputs(
            &bins,
            |binary| Command::new(binary),
            |binary, output| {
                assert_eq!(output.stdout, b"hit");
                Ok(binary.to_path_buf())
            },
            |binary, _command| {
                visited.borrow_mut().push(binary.to_path_buf());
                let code = if binary.ends_with("bin0") { 1 } else { 0 };
                Ok(probe_output(code, b"hit", b"miss"))
            },
            |success| outcomes.borrow_mut().push(success),
        )
        .expect("deterministic probe succeeds");
        assert_eq!(*visited.borrow(), bins);
        assert_eq!(*outcomes.borrow(), [true, true, true]);
        assert_eq!(values, bins[1..]);
    }

    #[test]
    fn bounded_probe_core_stops_on_runner_error_with_binary_context() {
        let bins = vec![fake_bin(0), fake_bin(1), fake_bin(2)];
        let visited = std::cell::RefCell::new(Vec::new());
        let outcomes = std::cell::RefCell::new(Vec::new());
        let error = collect_probe_outputs(
            &bins,
            |binary| Command::new(binary),
            |_binary, _output| Ok(()),
            |binary, _command| {
                visited.borrow_mut().push(binary.to_path_buf());
                if binary.ends_with("bin1") {
                    Err(std::io::Error::new(
                        std::io::ErrorKind::TimedOut,
                        "complete subtree terminated",
                    ))
                } else {
                    Ok(probe_output(1, b"", b"miss"))
                }
            },
            |success| outcomes.borrow_mut().push(success),
        )
        .expect_err("runner failure is terminal");
        assert_eq!(*visited.borrow(), bins[..2]);
        assert_eq!(*outcomes.borrow(), [true, false]);
        assert_eq!(
            error,
            "exec scheduler probe /fake/bin1: complete subtree terminated",
        );
    }

    #[test]
    fn bounded_probe_core_rejects_signal_exit_with_binary_context() {
        use std::os::unix::process::ExitStatusExt as _;

        let bins = vec![PathBuf::from("/fake/signaled")];
        let outcomes = std::cell::RefCell::new(Vec::new());
        let signal_status = std::process::ExitStatus::from_raw(libc::SIGKILL);
        let expected = format!("scheduler probe /fake/signaled terminated by {signal_status}");
        let error = collect_probe_outputs(
            &bins,
            |binary| Command::new(binary),
            |_binary, _output| Ok(()),
            |_binary, _command| {
                Ok(Output {
                    status: std::process::ExitStatus::from_raw(libc::SIGKILL),
                    stdout: Vec::new(),
                    stderr: Vec::new(),
                })
            },
            |success| outcomes.borrow_mut().push(success),
        )
        .expect_err("signal termination is not an ordinary unlinked-ctor miss");
        assert_eq!(error, expected);
        assert_eq!(*outcomes.borrow(), [false]);
    }

    #[test]
    fn bounded_probe_core_counts_decode_failure_as_failed_item() {
        let bins = vec![PathBuf::from("/fake/malformed")];
        let outcomes = std::cell::RefCell::new(Vec::new());
        let error = collect_probe_outputs(
            &bins,
            |binary| Command::new(binary),
            |_binary, _output| Err::<(), _>("malformed scheduler manifest".to_string()),
            |_binary, _command| Ok(probe_output(0, b"not-json", b"")),
            |success| outcomes.borrow_mut().push(success),
        )
        .expect_err("decode failure is terminal");
        assert_eq!(error, "malformed scheduler manifest");
        assert_eq!(*outcomes.borrow(), [false]);
    }

    /// Walking all bins, every Ok(T) is appended in iteration
    /// order. Pins the Vec append contract (caller relies on it
    /// for the shell-mode ambiguity-bail path which checks
    /// `matches.len() > 1`).
    #[test]
    fn probe_collect_with_bins_appends_in_order() {
        let bins = vec![fake_bin(0), fake_bin(1)];
        let configure_cmd = |_bin: &Path| Command::new("true");
        let on_success =
            |bin: &Path, _out: &Output| -> Result<PathBuf, String> { Ok(bin.to_path_buf()) };
        let result =
            probe_collect_with_bins_using(&bins, configure_cmd, on_success, |_bin, _command| {
                Ok(probe_output(0, b"", b""))
            })
            .expect("two successes should collect");
        assert_eq!(result, vec![fake_bin(0), fake_bin(1)]);
    }

    #[test]
    fn export_selects_first_acceptance_and_executes_only_it_once() {
        let bins = vec![fake_bin(0), fake_bin(1), fake_bin(2)];
        let checked = std::cell::RefCell::new(Vec::new());
        let executed = std::cell::RefCell::new(Vec::new());
        let selected = probe_first_with_bins_using(
            &bins,
            |binary| Command::new(binary),
            |binary| Command::new(binary),
            |binary, _output| Ok(binary.to_path_buf()),
            |binary, _command| {
                checked.borrow_mut().push(binary.to_path_buf());
                let (status, stderr) = if binary.ends_with("bin0") {
                    (2, &b"registered but ineligible"[..])
                } else {
                    (0, &b""[..])
                };
                Ok(probe_output(status, b"", stderr))
            },
            |binary, _command| {
                executed.borrow_mut().push(binary.to_path_buf());
                Ok(probe_output(0, b"exported", b"wrote runfile"))
            },
        )
        .expect("the second candidate accepts and exports");
        assert_eq!(selected, fake_bin(1));
        assert_eq!(*checked.borrow(), bins[..2]);
        assert_eq!(*executed.borrow(), [fake_bin(1)]);
    }

    #[test]
    fn hung_acceptance_miss_is_terminal_and_never_executes() {
        let bins = vec![fake_bin(0), fake_bin(1), fake_bin(2)];
        let checked = std::cell::RefCell::new(Vec::new());
        let execute_count = std::cell::Cell::new(0usize);
        let result = probe_first_with_bins_using(
            &bins,
            |binary| Command::new(binary),
            |binary| Command::new(binary),
            |_binary, _output| Ok(()),
            |binary, _command| {
                checked.borrow_mut().push(binary.to_path_buf());
                if binary.ends_with("bin0") {
                    Ok(probe_output(1, b"", b"not registered"))
                } else {
                    Err(std::io::Error::new(
                        std::io::ErrorKind::TimedOut,
                        "export acceptance probe exceeded 60.0s; complete subtree terminated",
                    ))
                }
            },
            |_binary, _command| {
                execute_count.set(execute_count.get() + 1);
                Ok(probe_output(0, b"", b""))
            },
        );
        match result {
            Err(ProbeError::Setup(message)) => assert_eq!(
                message,
                "exec /fake/bin1: export acceptance probe exceeded 60.0s; \
                 complete subtree terminated",
            ),
            _ => panic!("a timed-out bounded acceptance probe must be terminal"),
        }
        assert_eq!(*checked.borrow(), bins[..2]);
        assert_eq!(execute_count.get(), 0);
    }

    /// Empty bin list surfaces [`ProbeError::Setup`] with the
    /// shared empty-bins diagnostic. Pins the "no executable
    /// artifacts" wording that the operator may grep for.
    #[test]
    fn probe_first_with_bins_empty_returns_setup_error() {
        let bins: Vec<PathBuf> = vec![];
        let on_success = |_bin: &Path, _out: &Output| -> Result<(), String> { Ok(()) };
        match probe_first_with_bins_using(
            &bins,
            |_bin| Command::new("true"),
            |_bin| Command::new("true"),
            on_success,
            |_bin, _command| unreachable!("an empty probe cannot execute"),
            |_bin, _command| unreachable!("an empty exporter cannot execute"),
        ) {
            Err(ProbeError::Setup(msg)) => assert!(
                msg.contains("no executable artifacts"),
                "expected empty-bins diagnostic, got {msg:?}",
            ),
            _ => panic!("expected Setup error"),
        }
    }

    /// `probe_collect_with_bins` empty-bins error mirrors
    /// `probe_first_with_bins` — both helpers share
    /// [`EMPTY_BINS_SETUP_ERROR`].
    #[test]
    fn probe_collect_with_bins_empty_returns_setup_error() {
        let bins: Vec<PathBuf> = vec![];
        let configure_cmd = |_bin: &Path| Command::new("true");
        let on_success = |_bin: &Path, _out: &Output| -> Result<(), String> { Ok(()) };
        match probe_collect_with_bins_using(&bins, configure_cmd, on_success, |_bin, _command| {
            unreachable!("an empty probe cannot execute")
        }) {
            Err(ProbeError::Setup(msg)) => assert!(msg.contains("no executable artifacts")),
            _ => panic!("expected Setup error"),
        }
    }

    /// Exit-2 rejection from the FIRST exit-2 bin wins; later
    /// exit-2 bins do NOT overwrite. Exit-1 (and other
    /// non-2-non-zero) miss diagnostics overwrite per iteration
    /// (last-write-wins). Pins both contracts on a single
    /// fixture so a regression in either flips the assertion
    /// loudly.
    #[test]
    fn probe_collect_with_bins_exit_2_first_wins_exit_1_overwrites() {
        let bins = vec![fake_bin(0), fake_bin(1), fake_bin(2), fake_bin(3)];
        let configure_cmd = |bin: &Path| Command::new(bin);
        let run = |bin: &Path, _command: Command| {
            let suffix = bin.file_name().unwrap().to_str().unwrap();
            let (code, stderr) = match suffix {
                "bin0" => (2, "REJECTED_A"),
                "bin1" => (2, "REJECTED_B"),
                "bin2" => (1, "MISS_C"),
                "bin3" => (1, "MISS_D"),
                _ => unreachable!(),
            };
            Ok(probe_output(code, b"", stderr.as_bytes()))
        };
        let on_success = |_bin: &Path, _out: &Output| -> Result<(), String> { Ok(()) };
        match probe_collect_with_bins_using(&bins, configure_cmd, on_success, run) {
            Err(ProbeError::Miss(miss)) => {
                assert_eq!(miss.bins_tried, 4);
                assert_eq!(
                    miss.rejection_stderr.as_deref(),
                    Some("REJECTED_A"),
                    "first rejection must win — REJECTED_B should be ignored",
                );
                assert_eq!(
                    miss.last_miss_stderr, "MISS_D",
                    "last miss must overwrite — MISS_C should be replaced",
                );
            }
            _ => panic!("expected Miss with rejection + last-miss"),
        }
    }

    /// All-miss with no exit-2 produces ProbeMiss with the
    /// correct bin count and `rejection_stderr = None`. Pins
    /// the "no rejection_stderr → operator sees the last-miss
    /// diagnostic" contract.
    #[test]
    fn probe_collect_with_bins_all_miss_no_rejection_returns_probe_miss() {
        let bins = vec![fake_bin(0), fake_bin(1), fake_bin(2)];
        let configure_cmd = |_bin: &Path| Command::new("false");
        let on_success = |_bin: &Path, _out: &Output| -> Result<(), String> { Ok(()) };
        match probe_collect_with_bins_using(&bins, configure_cmd, on_success, |_bin, _command| {
            Ok(probe_output(1, b"", b"miss"))
        }) {
            Err(ProbeError::Miss(miss)) => {
                assert_eq!(miss.bins_tried, 3);
                assert!(
                    miss.rejection_stderr.is_none(),
                    "no exit-2 → rejection_stderr must stay None",
                );
            }
            _ => panic!("expected Miss"),
        }
    }
}
