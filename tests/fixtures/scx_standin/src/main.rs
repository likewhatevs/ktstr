//! Userspace loader for the minimum-weight stand-in sched_ext scheduler.
//!
//! Opens and loads the BPF skeleton, attaches the `sched_ext_ops` struct_ops
//! (which registers + enables the scheduler — `/sys/kernel/sched_ext/root/ops`
//! goes non-empty and `state` reads `enabled`, the cell's passing-attach
//! signal), then idles until the cell's teardown signal and exits 0. No
//! scx_utils: the load/attach/shutdown is driven directly through libbpf-rs.

use std::mem::MaybeUninit;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use anyhow::Result;
use libbpf_rs::skel::{OpenSkel, SkelBuilder};

mod bpf_skel {
    include!(concat!(env!("OUT_DIR"), "/standin.skel.rs"));
}

use bpf_skel::StandinSkelBuilder;

static SHUTDOWN: AtomicBool = AtomicBool::new(false);

extern "C" fn request_shutdown(_signal: libc::c_int) {
    SHUTDOWN.store(true, Ordering::Relaxed);
}

fn main() -> Result<()> {
    // The cell tears the scheduler down with SIGTERM/SIGINT; handle both so the
    // struct_ops link drops cleanly (detaching sched_ext) and the process exits
    // 0 rather than dying by signal.
    // SAFETY: `request_shutdown` is async-signal-safe (a single atomic store).
    unsafe {
        libc::signal(
            libc::SIGINT,
            request_shutdown as *const () as libc::sighandler_t,
        );
        libc::signal(
            libc::SIGTERM,
            request_shutdown as *const () as libc::sighandler_t,
        );
    }

    let mut open_object = MaybeUninit::uninit();
    let open_skel = StandinSkelBuilder::default().open(&mut open_object)?;
    let mut skel = open_skel.load()?;
    let _link = skel.maps.standin_ops.attach_struct_ops()?;

    while !SHUTDOWN.load(Ordering::Relaxed) {
        std::thread::sleep(Duration::from_millis(100));
    }
    Ok(())
}
