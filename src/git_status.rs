//! Shared policy for in-process gix status walks.
//!
//! A status walk is normally nested inside a larger cargo/nextest operation.
//! Letting every process independently size its gix worker pool from the full
//! host CPU count turns useful outer parallelism into thousands of short-lived
//! inner workers on large CI hosts. Keep each walk serial and let callers'
//! existing process-level parallelism provide concurrency.

/// Apply ktstr's bounded worker policy to an index/worktree status walk.
///
/// This only controls worker parallelism. Callers retain ownership of semantic
/// choices such as untracked-file enumeration, submodule handling, and rewrite
/// tracking.
#[doc(hidden)]
pub fn configure_index_worktree_parallelism(options: &mut gix::status::index_worktree::Options) {
    options.thread_limit = Some(1);
}

/// Consume a producer-backed iterator to EOF while remembering whether it
/// yielded an item.
///
/// gix may run a status producer on a helper thread even with a one-worker
/// status policy. Reaching EOF joins that producer before the caller mutates
/// process-global state or starts another operation.
#[doc(hidden)]
pub fn consume_has_any(iter: impl Iterator) -> bool {
    iter.fold(false, |_, _| true)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn index_worktree_policy_is_single_threaded() {
        let mut options = gix::status::index_worktree::Options::default();
        configure_index_worktree_parallelism(&mut options);
        assert_eq!(options.thread_limit, Some(1));
    }

    #[test]
    fn boolean_consumer_reaches_eof() {
        let yielded = std::cell::Cell::new(0);
        let iter = (0..4).inspect(|_| yielded.set(yielded.get() + 1));
        assert!(consume_has_any(iter));
        assert_eq!(yielded.get(), 4);
        assert!(!consume_has_any(std::iter::empty::<()>()));
    }
}
