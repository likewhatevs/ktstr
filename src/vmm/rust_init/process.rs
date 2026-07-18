//! Scheduler-process lifecycle: reboot, sched-pid/monitor state, SIGCHLD handling, kill-with-grace.
//!
//! Split from rust_init.rs; the shared consts/statics/imports live in the
//! parent module (`super`), reached via the glob below.
use super::*;

const PIDFD_RIGHTS_PAYLOAD: u8 = 0x4b;
const PIDFD_RIGHTS_ACK: u8 = 0xb7;
const PIDFD_RIGHTS_SPACE: usize =
    unsafe { libc::CMSG_SPACE(std::mem::size_of::<libc::c_int>() as libc::c_uint) as usize };

/// Ancillary-data storage aligned for `cmsghdr`.
#[repr(C)]
union PidfdRightsControl {
    _header: std::mem::ManuallyDrop<libc::cmsghdr>,
    bytes: [u8; PIDFD_RIGHTS_SPACE],
}

/// Spawn a child and acquire its pidfd before `exec`, without ever reopening a
/// numeric pid in the parent.
///
/// Stable Rust does not expose `CLONE_PIDFD` through `Command`. Instead, a
/// CLOEXEC `SOCK_SEQPACKET` socket is inherited across `fork`; the child's
/// `pre_exec` hook opens a pidfd for `getpid()` while that task is necessarily
/// alive and passes it to a parent receiver thread with `SCM_RIGHTS`. The child
/// does not return from `pre_exec` until that receiver has transferred the
/// validated descriptor to this thread and acknowledged it. A receive,
/// validation, or receiver-thread failure therefore closes the socket and
/// fails the pre-exec hook: `Command::spawn` never returns a live exec'd child
/// unless its exact pidfd is already owned by the parent. The queued descriptor
/// pins that exact task even if the exec'd program exits and is auto-reaped
/// before `Command::spawn` returns.
///
/// The command is one-shot after this call: `pre_exec` hooks cannot be removed
/// from `std::process::Command`, and this hook captures per-spawn socket fds.
pub(crate) fn spawn_with_pidfd(command: &mut Command) -> std::io::Result<(Child, OwnedFd)> {
    spawn_with_pidfd_inner(command, false, false, false, |_| {})
}

fn spawn_with_pidfd_inner(
    command: &mut Command,
    force_child_handshake_failure: bool,
    force_invalid_child_payload: bool,
    force_parent_receive_failure: bool,
    after_spawn: impl FnOnce(&mut Child),
) -> std::io::Result<(Child, OwnedFd)> {
    let mut sockets = [-1; 2];
    // SAFETY: `sockets` is a two-element output array. SOCK_CLOEXEC prevents
    // either endpoint from leaking past the child exec or a later unrelated
    // exec in the parent.
    if unsafe {
        libc::socketpair(
            libc::AF_UNIX,
            libc::SOCK_SEQPACKET | libc::SOCK_CLOEXEC,
            0,
            sockets.as_mut_ptr(),
        )
    } != 0
    {
        return Err(std::io::Error::last_os_error());
    }
    // SAFETY: socketpair returned two fresh descriptors owned by this scope.
    let parent_socket = unsafe { OwnedFd::from_raw_fd(sockets[0]) };
    let child_socket = unsafe { OwnedFd::from_raw_fd(sockets[1]) };
    let parent_raw = parent_socket.as_raw_fd();
    let child_raw = child_socket.as_raw_fd();

    // SAFETY: the hook performs only libc syscalls and stack writes. It neither
    // allocates nor touches locks, which is the required post-fork discipline
    // in a potentially multi-threaded process.
    unsafe {
        command.pre_exec(move || {
            let _ = libc::close(parent_raw);
            if force_child_handshake_failure {
                let _ = libc::close(child_raw);
                return Err(std::io::Error::from_raw_os_error(libc::EIO));
            }

            let pidfd = libc::syscall(libc::SYS_pidfd_open, libc::getpid(), 0u32) as libc::c_int;
            if pidfd < 0 {
                let error = std::io::Error::last_os_error();
                let _ = libc::close(child_raw);
                return Err(error);
            }

            let mut payload = if force_invalid_child_payload {
                PIDFD_RIGHTS_PAYLOAD ^ 0xff
            } else {
                PIDFD_RIGHTS_PAYLOAD
            };
            let mut iov = libc::iovec {
                iov_base: (&mut payload as *mut u8).cast::<libc::c_void>(),
                iov_len: 1,
            };
            let mut control = PidfdRightsControl {
                bytes: [0; PIDFD_RIGHTS_SPACE],
            };
            let mut message: libc::msghdr = std::mem::zeroed();
            message.msg_iov = &mut iov;
            message.msg_iovlen = 1;
            message.msg_control = control.bytes.as_mut_ptr().cast::<libc::c_void>();
            message.msg_controllen = PIDFD_RIGHTS_SPACE;

            let cmsg = libc::CMSG_FIRSTHDR(&message);
            if cmsg.is_null() {
                let _ = libc::close(pidfd);
                let _ = libc::close(child_raw);
                return Err(std::io::Error::from_raw_os_error(libc::EINVAL));
            }
            (*cmsg).cmsg_level = libc::SOL_SOCKET;
            (*cmsg).cmsg_type = libc::SCM_RIGHTS;
            (*cmsg).cmsg_len =
                libc::CMSG_LEN(std::mem::size_of::<libc::c_int>() as libc::c_uint) as usize;
            std::ptr::write(libc::CMSG_DATA(cmsg).cast::<libc::c_int>(), pidfd);

            let sent = libc::sendmsg(child_raw, &message, libc::MSG_NOSIGNAL);
            let send_error = if sent == 1 {
                None
            } else if sent < 0 {
                Some(std::io::Error::last_os_error())
            } else {
                Some(std::io::Error::from_raw_os_error(libc::EIO))
            };
            let _ = libc::close(pidfd);
            if let Some(error) = send_error {
                let _ = libc::close(child_raw);
                return Err(error);
            }

            let mut ack = 0u8;
            let received = loop {
                let rc = libc::recv(
                    child_raw,
                    (&mut ack as *mut u8).cast::<libc::c_void>(),
                    1,
                    0,
                );
                if rc >= 0 {
                    break rc;
                }
                let error = std::io::Error::last_os_error();
                if error.kind() != std::io::ErrorKind::Interrupted {
                    let _ = libc::close(child_raw);
                    return Err(error);
                }
            };
            let _ = libc::close(child_raw);
            if received != 1 || ack != PIDFD_RIGHTS_ACK {
                Err(std::io::Error::from_raw_os_error(libc::EPROTO))
            } else {
                Ok(())
            }
        });
    }

    // The channel is bounded so the receiver can transfer ownership before it
    // sends the ACK without waiting for this thread, which is blocked inside
    // Command::spawn until the child leaves pre_exec.
    let (pidfd_tx, pidfd_rx) = std::sync::mpsc::sync_channel(1);
    let receiver = std::thread::Builder::new()
        .name("ktstr-pidfd-recv".into())
        .spawn(move || -> std::io::Result<()> {
            if force_parent_receive_failure {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::ConnectionAborted,
                    "injected parent pidfd receive failure",
                ));
            }
            let pidfd = recv_spawn_pidfd(&parent_socket).map_err(|(error, _pidfd)| error)?;
            pidfd_tx.send(pidfd).map_err(|_| {
                std::io::Error::new(
                    std::io::ErrorKind::BrokenPipe,
                    "spawn caller dropped the received pidfd channel",
                )
            })?;
            send_spawn_pidfd_ack(&parent_socket)
        })?;

    let spawn_result = command.spawn();
    // The child closes this endpoint in pre_exec. Dropping the parent's copy
    // after Command::spawn returns also guarantees a receiver blocked on a
    // child-side setup failure observes EOF before we join it.
    drop(child_socket);
    let receiver_result = match receiver.join() {
        Ok(result) => result,
        Err(_) => Err(std::io::Error::other(
            "pidfd receiver thread panicked before completing the handshake",
        )),
    };

    let mut child = match spawn_result {
        Ok(child) => child,
        Err(error) => return Err(error),
    };

    // A successful exec proves the child received PIDFD_RIGHTS_ACK. The
    // receiver sends that ACK only after this bounded channel owns the exact
    // descriptor, so disconnection here is a broken internal invariant rather
    // than a recoverable no-handle Child state. The PID-1 guest uses
    // panic=abort, which also prevents a process from escaping ownership if
    // memory unsafety ever violated that invariant.
    let pidfd = pidfd_rx.recv().expect(
        "successful pidfd-gated exec must have transferred its exact descriptor before ACK",
    );
    if let Err(error) = receiver_result {
        let cleanup = terminate_scheduler_via_pidfd(&mut child, &pidfd);
        return match cleanup {
            Ok(()) => Err(error),
            Err(cleanup_error) => Err(std::io::Error::new(
                error.kind(),
                format!(
                    "{error}; exact pidfd cleanup after receiver failure failed: {cleanup_error}"
                ),
            )),
        };
    }
    after_spawn(&mut child);
    Ok((child, pidfd))
}

fn send_spawn_pidfd_ack(socket: &OwnedFd) -> std::io::Result<()> {
    let ack = PIDFD_RIGHTS_ACK;
    loop {
        // SAFETY: `ack` is one readable byte and `socket` is the live parent
        // endpoint of the private SOCK_SEQPACKET handshake.
        let sent = unsafe {
            libc::send(
                socket.as_raw_fd(),
                (&ack as *const u8).cast::<libc::c_void>(),
                1,
                libc::MSG_NOSIGNAL,
            )
        };
        if sent == 1 {
            return Ok(());
        }
        if sent >= 0 {
            return Err(std::io::Error::from_raw_os_error(libc::EIO));
        }
        let error = std::io::Error::last_os_error();
        if error.kind() != std::io::ErrorKind::Interrupted {
            return Err(error);
        }
    }
}

type SpawnPidfdReceiveError = (std::io::Error, Option<OwnedFd>);

fn invalid_pidfd_envelope(message: &'static str, pidfd: Option<OwnedFd>) -> SpawnPidfdReceiveError {
    (
        std::io::Error::new(std::io::ErrorKind::InvalidData, message),
        pidfd,
    )
}

fn recv_spawn_pidfd(socket: &OwnedFd) -> Result<OwnedFd, SpawnPidfdReceiveError> {
    let mut payload = 0u8;
    let mut iov = libc::iovec {
        iov_base: (&mut payload as *mut u8).cast::<libc::c_void>(),
        iov_len: 1,
    };
    let mut control = PidfdRightsControl {
        bytes: [0; PIDFD_RIGHTS_SPACE],
    };
    let mut message: libc::msghdr = unsafe { std::mem::zeroed() };
    message.msg_iov = &mut iov;
    message.msg_iovlen = 1;
    message.msg_control = unsafe { control.bytes.as_mut_ptr().cast::<libc::c_void>() };
    message.msg_controllen = PIDFD_RIGHTS_SPACE;

    let received = loop {
        // recvmsg mutates these fields even on some interrupted paths. Restore
        // the full input contract before every attempt so an EINTR cannot turn
        // the retry into a zero-sized ancillary receive or preserve stale
        // truncation flags.
        payload = 0;
        message.msg_controllen = PIDFD_RIGHTS_SPACE;
        message.msg_flags = 0;
        unsafe {
            control.bytes.fill(0);
        }
        // SAFETY: all message buffers are live and writable for the syscall.
        // MSG_CMSG_CLOEXEC makes the received descriptor CLOEXEC atomically.
        let rc = unsafe { libc::recvmsg(socket.as_raw_fd(), &mut message, libc::MSG_CMSG_CLOEXEC) };
        if rc >= 0 {
            break rc;
        }
        let error = std::io::Error::last_os_error();
        if error.kind() != std::io::ErrorKind::Interrupted {
            return Err((error, None));
        }
    };

    // Extract and own an installed SCM_RIGHTS descriptor before validating the
    // surrounding envelope. This is load-bearing: malformed payload/flags must
    // keep every installed descriptor owned while the receiver rejects the
    // handshake, and every invalid return must close it through OwnedFd.
    let cmsg = unsafe { libc::CMSG_FIRSTHDR(&message) };
    let cmsg_data_base = unsafe { libc::CMSG_LEN(0) as usize };
    let one_fd_len =
        unsafe { libc::CMSG_LEN(std::mem::size_of::<libc::c_int>() as libc::c_uint) as usize };
    let first_cmsg_is_one_right = !cmsg.is_null()
        && unsafe { (*cmsg).cmsg_level } == libc::SOL_SOCKET
        && unsafe { (*cmsg).cmsg_type } == libc::SCM_RIGHTS
        && unsafe { (*cmsg).cmsg_len } == one_fd_len;
    let trailing = if cmsg.is_null() {
        std::ptr::null_mut()
    } else {
        // SAFETY: cmsg came from CMSG_FIRSTHDR for this message.
        unsafe { libc::CMSG_NXTHDR(&message, cmsg) }
    };

    let mut pidfd = None;
    let mut received_rights = 0usize;
    let mut current = cmsg;
    while !current.is_null() {
        let header = unsafe { &*current };
        if header.cmsg_level == libc::SOL_SOCKET
            && header.cmsg_type == libc::SCM_RIGHTS
            && header.cmsg_len >= cmsg_data_base
            && header.cmsg_len <= message.msg_controllen
        {
            let data_bytes = header.cmsg_len - cmsg_data_base;
            if data_bytes % std::mem::size_of::<libc::c_int>() == 0 {
                let count = data_bytes / std::mem::size_of::<libc::c_int>();
                for index in 0..count {
                    // SAFETY: the kernel reported `count` complete c_ints in
                    // this SCM_RIGHTS record. Take ownership of every one;
                    // extras are dropped immediately so malformed envelopes
                    // cannot leak descriptors.
                    let raw = unsafe {
                        std::ptr::read(libc::CMSG_DATA(current).cast::<libc::c_int>().add(index))
                    };
                    if raw >= 0 {
                        received_rights += 1;
                        let owned = unsafe { OwnedFd::from_raw_fd(raw) };
                        if pidfd.is_none() {
                            pidfd = Some(owned);
                        }
                    }
                }
            }
        }
        // SAFETY: `current` is an ancillary header within `message`.
        current = unsafe { libc::CMSG_NXTHDR(&message, current) };
    }
    let cmsg_is_one_right = first_cmsg_is_one_right && trailing.is_null() && received_rights == 1;

    if received != 1 || payload != PIDFD_RIGHTS_PAYLOAD {
        return Err(invalid_pidfd_envelope(
            "child pidfd handshake returned an invalid payload",
            pidfd,
        ));
    }
    if message.msg_flags & (libc::MSG_CTRUNC | libc::MSG_TRUNC) != 0 {
        return Err(invalid_pidfd_envelope(
            "child pidfd handshake ancillary data was truncated",
            pidfd,
        ));
    }
    if !cmsg_is_one_right || pidfd.is_none() {
        return Err(invalid_pidfd_envelope(
            "child pidfd handshake did not contain exactly one SCM_RIGHTS descriptor",
            pidfd,
        ));
    }
    Ok(pidfd
        .take()
        .expect("validated child pidfd handshake owns one descriptor"))
}

#[cfg(test)]
pub(crate) fn spawn_with_pidfd_after_spawn_for_test(
    command: &mut Command,
    after_spawn: impl FnOnce(&mut Child),
) -> std::io::Result<(Child, OwnedFd)> {
    spawn_with_pidfd_inner(command, false, false, false, after_spawn)
}

#[cfg(test)]
pub(crate) fn spawn_with_pidfd_handshake_failure_for_test(
    command: &mut Command,
) -> std::io::Result<(Child, OwnedFd)> {
    spawn_with_pidfd_inner(command, true, false, false, |_| {})
}

#[cfg(test)]
pub(crate) fn spawn_with_pidfd_invalid_payload_for_test(
    command: &mut Command,
) -> std::io::Result<(Child, OwnedFd)> {
    spawn_with_pidfd_inner(command, false, true, false, |_| {})
}

#[cfg(test)]
pub(crate) fn spawn_with_pidfd_parent_receive_failure_for_test(
    command: &mut Command,
) -> std::io::Result<(Child, OwnedFd)> {
    spawn_with_pidfd_inner(command, false, false, true, |_| {})
}

#[cfg(test)]
mod spawn_pidfd_tests {
    use super::*;

    const TWO_RIGHTS_SPACE: usize = unsafe {
        libc::CMSG_SPACE((2 * std::mem::size_of::<libc::c_int>()) as libc::c_uint) as usize
    };

    #[repr(C)]
    union TwoRightsControl {
        _header: std::mem::ManuallyDrop<libc::cmsghdr>,
        bytes: [u8; TWO_RIGHTS_SPACE],
    }

    fn open_fd_count() -> usize {
        fs::read_dir("/proc/self/fd")
            .expect("enumerate process fds")
            .count()
    }

    fn send_test_rights(socket: &OwnedFd, fds: &[&OwnedFd], payload: u8) {
        assert!(!fds.is_empty() && fds.len() <= 2);
        let mut payload = payload;
        let mut iov = libc::iovec {
            iov_base: (&mut payload as *mut u8).cast::<libc::c_void>(),
            iov_len: 1,
        };
        let mut control = TwoRightsControl {
            bytes: [0; TWO_RIGHTS_SPACE],
        };
        let mut message: libc::msghdr = unsafe { std::mem::zeroed() };
        message.msg_iov = &mut iov;
        message.msg_iovlen = 1;
        message.msg_control = unsafe { control.bytes.as_mut_ptr().cast::<libc::c_void>() };
        message.msg_controllen = unsafe {
            libc::CMSG_SPACE((fds.len() * std::mem::size_of::<libc::c_int>()) as libc::c_uint)
                as usize
        };
        let cmsg = unsafe { libc::CMSG_FIRSTHDR(&message) };
        assert!(!cmsg.is_null());
        unsafe {
            (*cmsg).cmsg_level = libc::SOL_SOCKET;
            (*cmsg).cmsg_type = libc::SCM_RIGHTS;
            (*cmsg).cmsg_len =
                libc::CMSG_LEN((fds.len() * std::mem::size_of::<libc::c_int>()) as libc::c_uint)
                    as usize;
            for (index, fd) in fds.iter().enumerate() {
                std::ptr::write(
                    libc::CMSG_DATA(cmsg).cast::<libc::c_int>().add(index),
                    fd.as_raw_fd(),
                );
            }
        }
        let sent = unsafe { libc::sendmsg(socket.as_raw_fd(), &message, libc::MSG_NOSIGNAL) };
        assert_eq!(
            sent,
            1,
            "send malformed pidfd envelope: {}",
            std::io::Error::last_os_error()
        );
    }

    fn socket_pair() -> (OwnedFd, OwnedFd) {
        let mut sockets = [-1; 2];
        let rc = unsafe {
            libc::socketpair(
                libc::AF_UNIX,
                libc::SOCK_SEQPACKET | libc::SOCK_CLOEXEC,
                0,
                sockets.as_mut_ptr(),
            )
        };
        assert_eq!(
            rc,
            0,
            "create test socketpair: {}",
            std::io::Error::last_os_error()
        );
        unsafe {
            (
                OwnedFd::from_raw_fd(sockets[0]),
                OwnedFd::from_raw_fd(sockets[1]),
            )
        }
    }

    fn wait_for_pidfd_exit(pidfd: &OwnedFd) {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
        loop {
            if !pidfd_is_alive(pidfd).expect("poll child pidfd") {
                return;
            }
            assert!(
                std::time::Instant::now() < deadline,
                "fast child did not reach pidfd exit readiness"
            );
            std::thread::yield_now();
        }
    }

    #[test]
    fn fast_exit_still_returns_the_pre_exec_process_identity() {
        let mut command = Command::new("/bin/sh");
        command.args(["-c", "exit 23"]);
        let (mut child, pidfd) = spawn_with_pidfd_after_spawn_for_test(&mut command, |_| {
            std::thread::sleep(std::time::Duration::from_millis(50));
        })
        .expect("receive pidfd after child fast-exit");
        wait_for_pidfd_exit(&pidfd);
        let _ = child.wait();
    }

    #[test]
    fn child_handshake_failure_fails_spawn_without_a_numeric_cleanup() {
        let mut command = Command::new("/bin/true");
        let error = match spawn_with_pidfd_handshake_failure_for_test(&mut command) {
            Err(error) => error,
            Ok(_) => panic!("pre-exec handshake failure must abort spawn"),
        };
        assert_eq!(error.raw_os_error(), Some(libc::EIO));
    }

    #[test]
    fn parent_receive_failure_prevents_exec_without_a_numeric_cleanup() {
        let dir = tempfile::tempdir().expect("create pidfd handshake tempdir");
        let marker = dir.path().join("exec-ran");
        let mut command = Command::new("/bin/sh");
        command
            .args(["-c", "printf executed > \"$1\"", "sh"])
            .arg(&marker);

        let error = match spawn_with_pidfd_parent_receive_failure_for_test(&mut command) {
            Err(error) => error,
            Ok(_) => panic!("parent receive failure must fail the gated spawn"),
        };
        assert!(
            matches!(
                error.raw_os_error(),
                Some(code)
                    if code == libc::EPIPE
                        || code == libc::ECONNRESET
                        || code == libc::EPROTO
            ),
            "unexpected child-side handshake failure: {error}"
        );
        assert!(
            !marker.exists(),
            "child reached exec without its exact pidfd owned by the parent"
        );
    }

    #[test]
    fn invalid_ancillary_envelope_prevents_exec_without_a_numeric_cleanup() {
        let dir = tempfile::tempdir().expect("create pidfd handshake tempdir");
        let marker = dir.path().join("exec-ran");
        let mut command = Command::new("/bin/sh");
        command
            .args(["-c", "printf executed > \"$1\"", "sh"])
            .arg(&marker);

        let error = match spawn_with_pidfd_invalid_payload_for_test(&mut command) {
            Err(error) => error,
            Ok(_) => panic!("invalid ancillary envelope must fail the gated spawn"),
        };
        assert_eq!(
            error.raw_os_error(),
            Some(libc::EPROTO),
            "child must reject the missing parent ACK: {error}"
        );
        assert!(
            !marker.exists(),
            "child reached exec after the parent rejected its pidfd envelope"
        );
    }

    #[test]
    fn malformed_envelopes_close_received_fds_and_leave_bystander_alive() {
        // Detach a short-lived helper thread's fd table so parallel tests
        // opening unrelated descriptors cannot perturb the exact before/after
        // count. CLONE_FILES unshare is unprivileged and dies with this thread.
        std::thread::spawn(|| {
            assert_eq!(
                unsafe { libc::unshare(libc::CLONE_FILES) },
                0,
                "unshare fd table for deterministic leak test: {}",
                std::io::Error::last_os_error()
            );
            let mut command = Command::new("/bin/sleep");
            command.arg("30");
            let (mut bystander, bystander_pidfd) =
                spawn_with_pidfd(&mut command).expect("spawn exact bystander");
            let baseline = open_fd_count();

            for _ in 0..32 {
                let (receiver, sender) = socket_pair();
                send_test_rights(&sender, &[&bystander_pidfd], PIDFD_RIGHTS_PAYLOAD ^ 0xff);
                let (error, received) =
                    recv_spawn_pidfd(&receiver).expect_err("reject malformed payload");
                assert_eq!(error.kind(), std::io::ErrorKind::InvalidData);
                assert!(
                    received.is_some(),
                    "retain the installed exact descriptor through validation failure"
                );
                drop(received);
                drop(sender);
                drop(receiver);

                let (receiver, sender) = socket_pair();
                send_test_rights(
                    &sender,
                    &[&bystander_pidfd, &bystander_pidfd],
                    PIDFD_RIGHTS_PAYLOAD,
                );
                let (error, received) =
                    recv_spawn_pidfd(&receiver).expect_err("reject multiple SCM_RIGHTS fds");
                assert_eq!(error.kind(), std::io::ErrorKind::InvalidData);
                assert!(
                    received.is_some(),
                    "retain one exact descriptor while closing malformed extras"
                );
                drop(received);
                drop(sender);
                drop(receiver);
            }

            assert_eq!(
                open_fd_count(),
                baseline,
                "malformed SCM_RIGHTS envelopes leaked descriptors"
            );
            assert!(
                pidfd_is_alive(&bystander_pidfd).expect("poll bystander pidfd"),
                "envelope rejection must never signal the unrelated pinned process"
            );
            terminate_scheduler_via_pidfd(&mut bystander, &bystander_pidfd)
                .expect("clean exact bystander");
        })
        .join()
        .expect("join isolated malformed-envelope test");
    }

    #[test]
    fn exact_pidfd_wait_returns_at_its_bound_without_numeric_cleanup() {
        let mut command = Command::new("/bin/sleep");
        command.arg("30");
        let (mut child, pidfd) =
            spawn_with_pidfd(&mut command).expect("spawn exact timeout stand-in");
        let timeout = std::time::Duration::from_millis(20);
        let started = std::time::Instant::now();
        let error = wait_scheduler_pidfd_exit_bounded(&mut child, &pidfd, timeout)
            .expect_err("live unsignaled process must hit the finite bound");
        assert!(error.contains("timed out"));
        assert!(
            started.elapsed() < std::time::Duration::from_secs(1),
            "finite pidfd wait unexpectedly blocked: {:?}",
            started.elapsed()
        );
        assert!(
            pidfd_is_alive(&pidfd).expect("poll exact timeout stand-in"),
            "timeout path must not synthesize a numeric-pid kill"
        );
        terminate_scheduler_via_pidfd(&mut child, &pidfd).expect("clean exact timeout stand-in");
    }
}

/// Reboot immediately. Used for fatal init errors and normal shutdown.
pub(crate) fn force_reboot() -> ! {
    let _ = reboot(RebootMode::RB_AUTOBOOT);
    // The kernel is rebooting — no event will ever fire. Park the
    // thread forever; this is cheaper than a sleep loop because
    // `park` blocks in the kernel without a wake-up timer attached.
    // No `unpark` call exists in this path; the process dies when
    // the reboot syscall completes.
    loop {
        std::thread::park();
    }
}

/// Every resource which identifies or observes the live scheduler process.
///
/// Keeping the child, its original pidfd, declared scheduler identity, log,
/// attach generation, and monitor in one record prevents a lifecycle Op from
/// publishing a new pid while Phase 6 still owns the boot child, or from
/// dropping the only reap handle after a successful replacement.
pub(crate) struct CurrentSchedulerProcess {
    pub(crate) generation: u64,
    pub(crate) child: Child,
    pub(crate) pidfd: OwnedFd,
    pub(crate) log_path: String,
    pub(crate) scheduler: Option<&'static crate::test_support::SchedulerSpec>,
    pub(crate) monitor: Option<SchedExitStop>,
    /// A bounded terminal wait already expired (or failed). Drop may issue one
    /// final exact signal, but must not spend a second teardown budget waiting
    /// for a process the imminent VM reboot will reap.
    pub(crate) drop_reap_exhausted: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PidfdSignalOutcome {
    Delivered,
    AlreadyExited,
}

impl CurrentSchedulerProcess {
    pub(crate) fn pid(&self) -> libc::pid_t {
        self.child.id() as libc::pid_t
    }

    /// Stop and join the monitor before an intentional process transition.
    /// The returned bit preserves an already-observed exit for verifier
    /// cleanup.
    pub(crate) fn stop_monitor(&mut self) -> bool {
        self.monitor
            .take()
            .is_some_and(SchedExitStop::stop_and_join)
    }

    /// Signal this exact process identity through its retained pidfd. The
    /// typed outcome distinguishes kernel-accepted delivery from ESRCH, which
    /// lets verifier cleanup reject an already-dead process.
    pub(crate) fn send_signal(&self, signal: libc::c_int) -> Result<PidfdSignalOutcome, String> {
        pidfd_send_signal(&self.pidfd, signal)
    }

    /// Revalidate this exact process without consulting a numeric pid or
    /// procfs.
    pub(crate) fn is_alive(&self) -> Result<bool, String> {
        pidfd_is_alive(&self.pidfd)
    }

    /// Kill and reap this exact owner. The pidfd pins the target across the
    /// bounded readiness wait; `Child::wait` only reaps the already-pinned
    /// child after the pidfd reports terminal readiness.
    pub(crate) fn terminate_exact(&mut self) -> Result<(), String> {
        let result = terminate_scheduler_via_pidfd(&mut self.child, &self.pidfd);
        self.drop_reap_exhausted = result.is_err();
        result
    }

    /// Wait for terminal readiness on the original pidfd, then consume the
    /// Child wait status exactly once. A timeout leaves both handles owned for
    /// a later SIGKILL/reap.
    pub(crate) fn reap_bounded_status(
        &mut self,
        timeout: std::time::Duration,
    ) -> Option<std::process::ExitStatus> {
        let status = self.reap_bounded_status_inner(timeout);
        self.drop_reap_exhausted = status.is_none();
        status
    }

    fn reap_bounded_status_inner(
        &mut self,
        timeout: std::time::Duration,
    ) -> Option<std::process::ExitStatus> {
        if let Ok(Some(status)) = self.child.try_wait() {
            return Some(status);
        }
        let deadline = std::time::Instant::now() + timeout;
        loop {
            let now = std::time::Instant::now();
            if now >= deadline {
                return self.child.try_wait().ok().flatten();
            }
            let remaining = deadline.saturating_duration_since(now);
            let timeout_ms =
                remaining.as_millis().clamp(1, libc::c_int::MAX as u128) as libc::c_int;
            let mut pfd = libc::pollfd {
                fd: self.pidfd.as_raw_fd(),
                events: libc::POLLIN,
                revents: 0,
            };
            // SAFETY: one retained pidfd and a finite millisecond timeout.
            let rc = unsafe { libc::poll(&mut pfd, 1, timeout_ms) };
            if rc < 0 {
                let error = std::io::Error::last_os_error();
                if error.kind() == std::io::ErrorKind::Interrupted {
                    continue;
                }
                tracing::warn!(
                    pid = self.pid(),
                    error = %error,
                    "scheduler pidfd terminal wait failed"
                );
                return self.child.try_wait().ok().flatten();
            }
            if rc == 0 {
                continue;
            }
            if pfd.revents & (libc::POLLIN | libc::POLLHUP) != 0 {
                return self.child.wait().ok();
            }
            if pfd.revents & (libc::POLLERR | libc::POLLNVAL) != 0 {
                tracing::warn!(
                    pid = self.pid(),
                    revents = pfd.revents,
                    "scheduler pidfd returned invalid terminal wait events"
                );
                return self.child.try_wait().ok().flatten();
            }
        }
    }
}

impl Drop for CurrentSchedulerProcess {
    fn drop(&mut self) {
        let _ = self.stop_monitor();
        if self.drop_reap_exhausted {
            // Phase 6 already consumed its finite reap allowance. Preserve the
            // exact-target invariant with a final pidfd signal, but leave the
            // uninterruptible straggler for the immediately following reboot.
            let _ = pidfd_send_signal(&self.pidfd, libc::SIGKILL);
            let _ = self.child.try_wait();
            return;
        }
        if let Err(error) = self.terminate_exact() {
            tracing::error!(
                pid = self.pid(),
                generation = self.generation,
                error = %error,
                "failed to exact-kill scheduler owner from Drop"
            );
        }
    }
}

static CURRENT_SCHEDULER_PROCESS: std::sync::Mutex<Option<CurrentSchedulerProcess>> =
    std::sync::Mutex::new(None);

/// Guard which serializes every detach/attach/replace transition. Callers keep
/// this guard across old-owner teardown, replacement spawn, and new-owner
/// commit, so no observer can compose fields from different generations.
pub(crate) struct SchedulerProcessOwnerGuard<'a> {
    guard: std::sync::MutexGuard<'a, Option<CurrentSchedulerProcess>>,
}

impl SchedulerProcessOwnerGuard<'_> {
    pub(crate) fn current(&self) -> Option<&CurrentSchedulerProcess> {
        self.guard.as_ref()
    }

    pub(crate) fn current_mut(&mut self) -> Option<&mut CurrentSchedulerProcess> {
        self.guard.as_mut()
    }

    pub(crate) fn take(&mut self) -> Option<CurrentSchedulerProcess> {
        self.guard.take()
    }

    pub(crate) fn install(
        &mut self,
        process: CurrentSchedulerProcess,
    ) -> Result<(), CurrentSchedulerProcess> {
        if self.guard.is_some() {
            return Err(process);
        }
        *self.guard = Some(process);
        Ok(())
    }
}

pub(crate) fn lock_scheduler_process_owner() -> SchedulerProcessOwnerGuard<'static> {
    SchedulerProcessOwnerGuard {
        guard: CURRENT_SCHEDULER_PROCESS
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner),
    }
}

#[cfg(test)]
pub(crate) fn scheduler_process_owner_for_test(
    slot: &std::sync::Mutex<Option<CurrentSchedulerProcess>>,
) -> SchedulerProcessOwnerGuard<'_> {
    SchedulerProcessOwnerGuard {
        guard: slot
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner),
    }
}

/// Remove the one live owner for terminal Phase 6 cleanup. No later scheduler
/// transition is legal after this edge, so returning the owned record is both
/// exact and sufficient.
pub(crate) fn take_current_scheduler_process() -> Option<CurrentSchedulerProcess> {
    lock_scheduler_process_owner().take()
}

/// Immutable boot scheduler declaration. Restart always returns to this spec,
/// independent of whichever staged scheduler a prior Replace installed.
static BOOT_SCHEDULER: OnceLock<Option<&'static crate::test_support::SchedulerSpec>> =
    OnceLock::new();

pub(crate) fn install_boot_scheduler(
    scheduler: Option<&'static crate::test_support::SchedulerSpec>,
) {
    let _ = BOOT_SCHEDULER.set(scheduler);
}

pub(crate) fn boot_scheduler() -> Option<&'static crate::test_support::SchedulerSpec> {
    BOOT_SCHEDULER.get().copied().flatten()
}

/// Boot-captured context used for every pending scheduler-exit monitor.
struct SchedExitMonitorBootCtx {
    suppress_sched_log: Arc<AtomicBool>,
    probe_output_done: Option<Arc<crate::sync::Latch>>,
}

static SCHED_EXIT_MONITOR_BOOT_CTX: OnceLock<SchedExitMonitorBootCtx> = OnceLock::new();

pub(crate) fn install_sched_exit_monitor_context(
    suppress_sched_log: Arc<AtomicBool>,
    probe_output_done: Option<Arc<crate::sync::Latch>>,
) {
    let _ = SCHED_EXIT_MONITOR_BOOT_CTX.set(SchedExitMonitorBootCtx {
        suppress_sched_log,
        probe_output_done,
    });
}

/// Build a pending monitor against a duplicate of the exact spawn pidfd.
/// Publication and arming are deliberately separate in
/// [`commit_spawned_scheduler`] so an early exit cannot race ahead of the
/// owner record.
pub(crate) fn start_pending_sched_exit_monitor_with_log(
    pid: u32,
    pidfd: OwnedFd,
    log_path: Option<&str>,
) -> std::io::Result<SchedExitStop> {
    let ctx = SCHED_EXIT_MONITOR_BOOT_CTX.get().ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "scheduler-exit monitor boot context is not installed",
        )
    })?;
    start_pending_sched_exit_monitor(
        pid,
        pidfd,
        log_path,
        ctx.suppress_sched_log.clone(),
        ctx.probe_output_done.clone(),
    )
}

/// Finish a provisional spawn and atomically install the one live scheduler
/// owner. The host's Finished ACK precedes owner publication. The monitor gate
/// then re-polls its exact pidfd and keeps publication Pending while emitting
/// Settled plus the caller's success frames; only after those FIFO frames are
/// queued does it arm unexpected-exit publication.
pub(crate) fn commit_spawned_scheduler(
    owner: &mut SchedulerProcessOwnerGuard<'_>,
    spawned: &mut SpawnedScheduler,
    scheduler: Option<&'static crate::test_support::SchedulerSpec>,
    publish_success: impl FnOnce() -> Result<(), String>,
) -> Result<(), String> {
    if let Some(current) = owner.current() {
        let error = format!(
            "cannot commit scheduler while pid {} generation {} is still owned",
            current.pid(),
            current.generation
        );
        let cleanup = spawned.terminate_after_monitor_failure();
        return Err(format!("{error}; provisional cleanup={cleanup:?}"));
    }

    let pid = match spawned.child_id() {
        Ok(pid) => pid,
        Err(error) => {
            let cleanup = spawned.terminate_after_monitor_failure();
            return Err(format!(
                "scheduler child handoff failed: {error}; provisional cleanup={cleanup:?}"
            ));
        }
    };
    let monitor_pidfd = match spawned.clone_pidfd() {
        Ok(pidfd) => pidfd,
        Err(error) => {
            let cleanup = spawned.terminate_after_monitor_failure();
            return Err(format!(
                "scheduler pidfd handoff failed: {error}; provisional cleanup={cleanup:?}"
            ));
        }
    };
    let pending = match start_pending_sched_exit_monitor_with_log(
        pid,
        monitor_pidfd,
        Some(&spawned.log_path),
    ) {
        Ok(pending) => pending,
        Err(error) => {
            let cleanup = spawned.terminate_after_monitor_failure();
            return Err(format!(
                "scheduler exit-monitor installation failed: {error}; \
                 provisional cleanup={cleanup:?}"
            ));
        }
    };

    if let Err(error) = spawned.confirm_alive_after_monitor_install() {
        let monitor_observed_exit = pending.stop_and_join();
        let cleanup = spawned.terminate_after_monitor_failure();
        return Err(format!(
            "scheduler exited during exit-monitor handoff: {error}; \
             monitor_observed_exit={monitor_observed_exit}; provisional cleanup={cleanup:?}"
        ));
    }

    let finished = spawned.await_attach_finished_ack();
    if let Err(error) = finished {
        let finished_acked = spawned.finished_ack_consumed();
        let monitor_observed_exit = pending.stop_and_join();
        let process_cleanup = spawned.terminate_provisional_process();
        let attach_close = spawned.close_failed_attach_attempt();
        return Err(format!(
            "scheduler attach Finished boundary failed: {error}; \
             finished_acked={finished_acked}; monitor_observed_exit={monitor_observed_exit}; \
             process_cleanup={process_cleanup:?}; attach_close={attach_close:?}"
        ));
    }

    let process = match spawned.take_current_process(scheduler, pending) {
        Ok(process) => process,
        Err(error) => {
            let cleanup = spawned.terminate_after_monitor_failure();
            return Err(format!(
                "construct scheduler owner after Finished ACK: {error}; cleanup={cleanup:?}"
            ));
        }
    };
    if let Err(mut process) = owner.install(process) {
        let existing = owner
            .current()
            .map(|current| format!("pid {} generation {}", current.pid(), current.generation))
            .unwrap_or_else(|| "unknown owner".to_string());
        let _ = process.stop_monitor();
        let cleanup = process.terminate_exact();
        let attach_close = spawned.close_failed_attach_attempt();
        return Err(format!(
            "scheduler owner became occupied by {existing}; cleanup={cleanup:?}; \
             attach_close={attach_close:?}"
        ));
    }

    let monitor_commit = owner
        .current()
        .and_then(|process| process.monitor.as_ref())
        .expect("installed scheduler owner always carries a pending monitor")
        .commit_with(|| {
            spawned.settle_attach_attempt()?;
            publish_success()
        });
    if let Err(error) = monitor_commit {
        let mut failed = owner
            .take()
            .expect("failed pending monitor belongs to installed owner");
        let monitor_observed_exit = failed.stop_monitor();
        let cleanup = failed.terminate_exact();
        let attach_close = spawned.close_failed_attach_attempt();
        return Err(format!(
            "scheduler owner success publication failed under monitor gate: {error}; \
             monitor_observed_exit={monitor_observed_exit}; cleanup={cleanup:?}; \
             attach_close={attach_close:?}"
        ));
    }
    Ok(())
}

/// Read both scheduler pid and identity from the one ownership record.
pub(crate) fn sched_pid() -> Option<libc::pid_t> {
    #[cfg(test)]
    {
        let injected = TEST_SCHED_PID.load(Ordering::Acquire);
        if injected != 0 {
            return Some(injected);
        }
    }
    CURRENT_SCHEDULER_PROCESS
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .as_ref()
        .map(CurrentSchedulerProcess::pid)
}

/// Host-only unit tests which exercise probe folding without a guest process
/// can inject a synthetic liveness pid. Production has no split pid channel.
#[cfg(test)]
static TEST_SCHED_PID: std::sync::atomic::AtomicI32 = std::sync::atomic::AtomicI32::new(0);

#[cfg(test)]
pub(crate) fn set_sched_pid(pid: libc::pid_t) {
    TEST_SCHED_PID.store(pid, Ordering::Release);
}

pub fn current_scheduler() -> Option<&'static crate::test_support::SchedulerSpec> {
    CURRENT_SCHEDULER_PROCESS
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .as_ref()
        .and_then(|process| process.scheduler)
}

pub(crate) fn current_scheduler_alive() -> Result<bool, String> {
    Ok(current_scheduler_liveness()?.unwrap_or(false))
}

/// Read exact scheduler presence and liveness under one owner lock. `None`
/// distinguishes an intentional no-scheduler/Detach state from a terminal
/// pidfd, so scenario liveness checks never need to reopen a numeric pid.
pub(crate) fn current_scheduler_liveness() -> Result<Option<bool>, String> {
    let owner = lock_scheduler_process_owner();
    match owner.current() {
        Some(process) => process.is_alive().map(Some),
        None => Ok(None),
    }
}

/// Clone the exact current scheduler identity for an evented observer without
/// reopening its numeric pid after releasing the owner lock.
pub(crate) fn clone_current_scheduler_pidfd() -> Result<Option<(libc::pid_t, OwnedFd)>, String> {
    let owner = lock_scheduler_process_owner();
    owner
        .current()
        .map(|process| {
            process
                .pidfd
                .try_clone()
                .map(|pidfd| (process.pid(), pidfd))
                .map_err(|error| format!("clone current scheduler pidfd: {error}"))
        })
        .transpose()
}

pub(crate) fn pidfd_send_signal(
    pidfd: &OwnedFd,
    signal: libc::c_int,
) -> Result<PidfdSignalOutcome, String> {
    // SAFETY: `pidfd` pins one process identity; a null siginfo and zero flags
    // are the documented pidfd_send_signal(2) form.
    let rc = unsafe {
        libc::syscall(
            libc::SYS_pidfd_send_signal,
            pidfd.as_raw_fd(),
            signal,
            std::ptr::null::<libc::siginfo_t>(),
            0u32,
        )
    };
    if rc == 0 {
        return Ok(PidfdSignalOutcome::Delivered);
    }
    let error = std::io::Error::last_os_error();
    if error.raw_os_error() == Some(libc::ESRCH) {
        Ok(PidfdSignalOutcome::AlreadyExited)
    } else {
        Err(format!("pidfd_send_signal({signal}): {error}"))
    }
}

pub(crate) fn pidfd_is_alive(pidfd: &OwnedFd) -> Result<bool, String> {
    let mut pfd = libc::pollfd {
        fd: pidfd.as_raw_fd(),
        events: libc::POLLIN,
        revents: 0,
    };
    // SAFETY: `pfd` is one initialized pollfd and the zero timeout makes this
    // an authoritative nonblocking liveness snapshot.
    let rc = unsafe { libc::poll(&mut pfd, 1, 0) };
    if rc < 0 {
        return Err(format!(
            "scheduler pidfd liveness poll: {}",
            std::io::Error::last_os_error()
        ));
    }
    if pfd.revents & (libc::POLLERR | libc::POLLNVAL) != 0 {
        return Err(format!(
            "scheduler pidfd returned invalid events {:#x}",
            pfd.revents
        ));
    }
    Ok(pfd.revents & (libc::POLLIN | libc::POLLHUP) == 0)
}

pub(crate) fn terminate_scheduler_via_pidfd(
    child: &mut Child,
    pidfd: &OwnedFd,
) -> Result<(), String> {
    let _delivered = pidfd_send_signal(pidfd, libc::SIGKILL)?;
    wait_scheduler_pidfd_exit_bounded(child, pidfd, SCHED_REAP_TIMEOUT)
}

fn wait_scheduler_pidfd_exit_bounded(
    child: &mut Child,
    pidfd: &OwnedFd,
    timeout: std::time::Duration,
) -> Result<(), String> {
    if child.try_wait().ok().flatten().is_some() {
        return Ok(());
    }
    let deadline = std::time::Instant::now() + timeout;
    loop {
        let now = std::time::Instant::now();
        if now >= deadline {
            if !pidfd_is_alive(pidfd)? {
                let _ = child.wait();
                return Ok(());
            }
            return Err(format!(
                "timed out after {timeout:?} waiting for signaled scheduler pidfd"
            ));
        }
        let remaining = deadline.saturating_duration_since(now);
        let timeout_ms = remaining
            .as_millis()
            .saturating_add(u128::from(
                !remaining.subsec_nanos().is_multiple_of(1_000_000),
            ))
            .clamp(1, libc::c_int::MAX as u128) as libc::c_int;
        let mut pfd = libc::pollfd {
            fd: pidfd.as_raw_fd(),
            events: libc::POLLIN,
            revents: 0,
        };
        // SAFETY: one retained pidfd; EINTR is retried and the finite timeout
        // keeps every cleanup site within the guest teardown allowance.
        let rc = unsafe { libc::poll(&mut pfd, 1, timeout_ms) };
        if rc < 0 {
            let error = std::io::Error::last_os_error();
            if error.kind() == std::io::ErrorKind::Interrupted {
                continue;
            }
            return Err(format!("wait for SIGKILLed scheduler pidfd: {error}"));
        }
        if rc == 0 {
            continue;
        }
        if pfd.revents & (libc::POLLERR | libc::POLLNVAL) != 0 {
            return Err(format!(
                "SIGKILLed scheduler pidfd returned unexpected poll events {:#x}",
                pfd.revents
            ));
        }
        if pfd.revents & (libc::POLLIN | libc::POLLHUP) != 0 {
            let _ = child.wait();
            return Ok(());
        }
    }
}

/// RAII guard that flips SIGCHLD to a target disposition on
/// construction and restores the previous handler on drop. Used by
/// [`with_sigchld_default`] so a panic inside the closure cannot
/// leak `SIG_DFL` into the rest of the guest's lifetime — Drop
/// runs even on unwind.
///
/// `libc::signal` returns the previous handler on every call, so
/// the snapshot we capture in `install` is the authoritative value
/// to restore in `Drop`. Re-installing the snapshot makes the
/// guard idempotent across nested calls (an outer guard's restore
/// observes the inner guard's restore as a no-op rebind to the
/// same handler).
struct SigchldDispositionGuard {
    prev: libc::sighandler_t,
}

impl SigchldDispositionGuard {
    /// Install `handler` as the SIGCHLD disposition and capture
    /// the previous handler for restoration on drop.
    ///
    /// SAFETY: signal disposition is a process-wide property. PID
    /// 1 owns the disposition for the whole guest, so no other
    /// thread can race the signal install. `libc::signal` is
    /// async-signal-safe per POSIX.1-2008 TC2.
    ///
    /// # Panics
    ///
    /// Panics if `libc::signal` returns `SIG_ERR` — the libc
    /// failure indicator (`!0 as sighandler_t`) for an invalid
    /// signal number or other install failure. Without the check,
    /// `SIG_ERR` would be captured into `prev` as if it were a
    /// valid handler, and Drop would then attempt to install
    /// `SIG_ERR` (which the kernel rejects with `EINVAL`,
    /// surfacing as a separate `SIG_ERR` return that the no-check
    /// Drop also drops on the floor — silently leaking the
    /// install error). For SIGCHLD the failure path is
    /// implausible in practice (the signal number is valid and
    /// `SIG_DFL`/`SIG_IGN` are always-installable handlers), but
    /// the library invariant is general — `signal(2)` returning
    /// `SIG_ERR` is a programming error, not a runtime condition,
    /// so panicking is the right discipline.
    fn install(handler: libc::sighandler_t) -> Self {
        let prev = unsafe { libc::signal(libc::SIGCHLD, handler) };
        assert_ne!(
            prev,
            libc::SIG_ERR,
            "failed to install SIGCHLD handler — libc::signal returned SIG_ERR; \
             check signum / handler validity",
        );
        Self { prev }
    }
}

impl Drop for SigchldDispositionGuard {
    fn drop(&mut self) {
        // SAFETY: `self.prev` was returned by an earlier
        // `libc::signal` call on the same signal number, so
        // re-installing it is the documented restore pattern. The
        // `Drop` runs on both the normal-return and panic-unwind
        // paths, so a panic inside the protected closure cannot
        // leak the temporary disposition into the rest of the
        // process.
        unsafe {
            libc::signal(libc::SIGCHLD, self.prev);
        }
    }
}

/// Run `f` with SIGCHLD temporarily restored to `SIG_DFL` so the
/// kernel does not auto-reap any child spawned inside the closure.
/// `Command::status()` calls `waitpid(2)`, which returns `ECHILD`
/// when SIGCHLD is `SIG_IGN` (the default installed by
/// [`ktstr_guest_init`] for zombie prevention) — losing the real
/// exit status. Restoring `SIG_DFL` for the closure's lifetime
/// re-enables `waitpid` reaping; the post-closure restore puts
/// the previous disposition back so subsequent guest children
/// continue to be auto-reaped without leaking zombies.
///
/// Mirrors the inline save/restore pattern formerly open-coded at
/// the [`ktstr_guest_init`] shell `--exec` site (now also routed
/// through this helper). Both call sites share the same
/// SIGCHLD-vs-`waitpid` hazard; centralising the helper prevents
/// drift between the two implementations.
///
/// Restore is panic-safe via [`SigchldDispositionGuard`]: a panic
/// in `f` runs the guard's `Drop`, which re-installs the previous
/// SIGCHLD handler before unwinding past the helper boundary.
/// Without the guard, a panicking child-spawn site would leak
/// `SIG_DFL` into the rest of the guest, breaking PID 1's zombie
/// reaping for every subsequent fork.
///
/// The closure must reap every child it spawns before returning.
/// Leaving an unreaped child at the boundary where `SIG_IGN` is
/// restored would orphan the zombie until the next reaper cycle.
/// `Command::status()` waits synchronously, so the typical caller
/// satisfies this invariant by construction.
pub(crate) fn with_sigchld_default<F, R>(f: F) -> R
where
    F: FnOnce() -> R,
{
    let _guard = SigchldDispositionGuard::install(libc::SIG_DFL);
    f()
}

/// Whether `/proc/{pid}` exists. Used as a `waitpid`-free liveness
/// probe: under SIGCHLD `SIG_IGN` the kernel auto-reaps children, so
/// `waitpid` returns `ECHILD` even when the child exited cleanly.
/// `/proc/{pid}` removal is signal-disposition-independent — the
/// directory disappears the moment the kernel finishes
/// `release_task` for the pid (see kernel/exit.c
/// `release_task` → `proc_flush_pid`), regardless of whether
/// `waitpid` ever ran.
///
/// Returns `true` when `/proc/{pid}` exists (process alive or
/// pre-reap), `false` when it does not (process exited and the
/// kernel has dropped the procfs entry).
/// SIGCHLD = SIG_IGN-safe liveness probe via procfs. The guest init
/// installs `SIGCHLD = SIG_IGN` process-wide (see
/// [`with_sigchld_default`] doc) so the kernel auto-reaps children
/// without explicit `waitpid`. Under that disposition `waitpid`
/// returns `ECHILD` even on a clean exit, so a `Command::status` /
/// `Child::wait` is the wrong tool for "is this pid still running".
///
/// `/proc/{pid}` removal is signal-disposition-independent: the
/// directory disappears the moment the kernel finishes `release_task`
/// for the pid (see kernel/exit.c `release_task` →
/// `proc_flush_pid`), regardless of how SIGCHLD is handled. Polling
/// `/proc/{pid}` therefore observes the real exit on every code path
/// where SIGCHLD might be ignored. Returns `true` when `/proc/{pid}`
/// exists (process alive or pre-reap), `false` when it does not
/// (process exited and the kernel has dropped the procfs entry).
#[cfg(test)]
pub(crate) fn proc_pid_alive(pid: u32) -> bool {
    Path::new(&format!("/proc/{pid}")).exists()
}

/// Outcome reported by a successful [`kill_scheduler_process`] call.
/// Three variants because the operator-visible signal (caller-side
/// logging, sidecar event) differs by how the child responded:
/// already-gone callers know there was nothing to do; sigterm-graceful
/// exit is the scx-convention happy path; sigkill-escalation is the
/// notable case (the scheduler binary either ignored SIGTERM or its
/// userspace signal handler ran too slow against the grace window).
//
// This legacy generic numeric-pid helper is test-only. Production scheduler
// transitions retain the original pidfd inside CurrentSchedulerProcess and
// signal that exact owner around the sched_ext disabled-state barrier.
#[cfg(test)]
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum KillSchedulerOutcome {
    /// `pid` was not alive when the call started — `/proc/{pid}`
    /// already absent. Treated as success because lifecycle ops
    /// (Op::DetachScheduler) are idempotent: detaching when nothing
    /// is running is a no-op, not an error.
    AlreadyExited,
    /// SIGTERM landed and the scheduler exited cleanly within the
    /// grace window. The scx convention (per scx_simple.c
    /// `sigint_handler` at L37-39 of the upstream
    /// tools/sched_ext/scx_simple.c) is to catch SIGTERM, drop the
    /// BPF skeleton, run scx_disable_workfn via the destructor path,
    /// and exit. This is the operator-visible happy path.
    ExitedAfterSigterm,
    /// SIGTERM did not produce an exit within the grace window;
    /// SIGKILL was sent and the process reaped. The scheduler
    /// either failed to install its SIGTERM handler, was stuck in
    /// uninterruptible kernel state, or its handler took longer
    /// than the grace allowed. Operators may want to inspect the
    /// scheduler binary's signal-handler implementation when this
    /// fires.
    EscalatedToSigkill,
}

/// Failure modes for [`kill_scheduler_process`]. Both indicate the
/// caller-supplied invariant (a kill-able pid) was violated or the
/// kernel refused to honor a SIGKILL — neither is recoverable at the
/// call site, but both carry distinct operator diagnostics.
#[cfg(test)]
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum KillSchedulerError {
    /// `pid` was not a positive pid_t value. POSIX `kill(2)` reserves
    /// 0 (the caller's process group) and negative values (signal a
    /// process group) for special semantics — the scheduler-lifecycle
    /// call site only ever wants to signal a specific known pid, so a
    /// non-positive value is a programming error in the caller.
    InvalidPid,
    /// SIGKILL was sent but `/proc/{pid}` was still present after the
    /// post-SIGKILL grace window. POSIX guarantees SIGKILL cannot be
    /// caught or ignored, so this indicates either kernel-side stall
    /// (rare uninterruptible D-state) or a process that re-spawned a
    /// new pid before procfs cleaned up — neither plausible in the
    /// scheduler-binary case but reported distinctly so the caller
    /// can surface a "scheduler refused to die" diagnostic rather
    /// than silently believing the detach succeeded.
    StillAliveAfterSigkill,
}

/// Send SIGTERM to `pid`, wait up to `sigterm_grace` for the process
/// to exit (observed via `/proc/{pid}` removal), then escalate to
/// SIGKILL if the polite shutdown did not land. Returns the variant
/// that describes how the kill resolved.
///
/// # Why procfs polling instead of `waitpid`
///
/// The guest init installs SIGCHLD = SIG_IGN globally so PID 1 does
/// not have to reap every zombie (see [`with_sigchld_default`] and
/// the doc on [`proc_pid_alive`]). Under that disposition the kernel
/// auto-reaps children before `waitpid` runs, so `waitpid` returns
/// `ECHILD` even on a clean exit. `/proc/{pid}` removal is
/// signal-disposition-independent: the directory disappears the
/// moment the kernel runs `release_task` for the pid, regardless of
/// how SIGCHLD is handled. Polling `/proc/{pid}` therefore observes
/// the real exit on every code path where SIGCHLD might be ignored.
///
/// # Why SIGTERM first, SIGKILL fallback
///
/// scx schedulers (per the upstream
/// `tools/sched_ext/scx_simple.c:71-72` convention) install one
/// shared signal handler for SIGINT + SIGTERM: setting an exit-
/// request flag that the scheduler's main loop polls, then dropping
/// the BPF skeleton which triggers the kernel's `scx_disable_workfn`
/// path. SIGTERM is the safe shutdown signal — every well-behaved
/// scx scheduler honors it. SIGKILL bypasses the userspace handler
/// (final-log-flush, graceful destructor) but the kernel still
/// observes the BPF program refcount drop and runs the disable path,
/// so the kernel-side scheduler state cleans up regardless. SIGKILL
/// after a bounded SIGTERM grace is the strict-correctness fallback
/// for a scheduler binary that has no SIGTERM handler installed or
/// took longer than `sigterm_grace` to exit.
///
/// # Pid lifecycle semantic
///
/// This generic helper does not inspect or mutate the live scheduler owner;
/// callers may exercise it against any spawned child pid.
///
/// # Wait mechanism
///
/// Absence is detected via the kernel-evented pidfd path in
/// [`poll_proc_pid_absent`] (`pidfd_wait_exit`), not a sleep-poll
/// loop; the `interval` argument passed there is currently unused.
/// The post-SIGKILL grace is the module-level [`POST_SIGKILL_GRACE`]
/// const (see that const's doc for the 200ms-vs-magic-number
/// rationale).
#[cfg(test)]
pub(crate) fn kill_scheduler_process(
    pid: libc::pid_t,
    sigterm_grace: std::time::Duration,
) -> Result<KillSchedulerOutcome, KillSchedulerError> {
    if pid <= 0 {
        return Err(KillSchedulerError::InvalidPid);
    }
    let pid_u32 = pid as u32;

    // Already-absent short-circuit: lifecycle ops are idempotent, so a
    // detach against a non-running scheduler is a no-op success.
    if !proc_pid_alive(pid_u32) {
        return Ok(KillSchedulerOutcome::AlreadyExited);
    }

    // SAFETY: libc::kill is async-signal-safe per POSIX and the
    // pid was validated above. EPERM (signal denied) or ESRCH
    // (process exited between the alive check and the kill) are
    // both observable via the subsequent procfs poll — EPERM means
    // the process keeps running and we'll escalate to SIGKILL;
    // ESRCH means the process is already gone and the poll will
    // immediately observe procfs absence.
    let _ = unsafe { libc::kill(pid, libc::SIGTERM) };

    let interval = std::time::Duration::from_millis(50);
    if poll_proc_pid_absent(pid_u32, interval, sigterm_grace) {
        return Ok(KillSchedulerOutcome::ExitedAfterSigterm);
    }

    // SIGTERM grace elapsed — escalate. SAFETY identical to the
    // SIGTERM call above; SIGKILL cannot be caught or ignored per
    // POSIX so the kernel will run the exit path even if the
    // scheduler binary was actively ignoring SIGTERM.
    let _ = unsafe { libc::kill(pid, libc::SIGKILL) };

    if poll_proc_pid_absent(pid_u32, interval, POST_SIGKILL_GRACE) {
        Ok(KillSchedulerOutcome::EscalatedToSigkill)
    } else {
        Err(KillSchedulerError::StillAliveAfterSigkill)
    }
}

/// Post-SIGKILL grace inside [`kill_scheduler_process`]. SIGKILL
/// triggers the kernel's `exit_notify` → `release_task` cascade
/// (kernel/exit.c) which removes `/proc/{pid}`; the wait here covers
/// both the routine reap path (sub-100ms for a simple userspace
/// process) AND the scheduler-lifecycle Op kill path where an scx
/// scheduler's exit blocks on `scx_disable_workfn`
/// (`kernel/sched/ext.c:6101`) tearing down BPF programs from a
/// workqueue. BPF tear-down dominates the SIGKILL→/proc removal
/// latency for scx_* binaries and routinely exceeds 1s on
/// loaded kernels; 2s leaves comfortable headroom while keeping
/// the unit-test fast for the simple-process case (the test
/// closure exits immediately on SIGKILL so the post-SIGKILL poll
/// returns in <50ms).
///
/// A `StillAliveAfterSigkill` firing AFTER this budget indicates a
/// structurally wrong target — D-state hang, kernel UB, BPF cleanup
/// deadlock — and operators should treat the variant as a debug
/// signal, not a transient retry case. Carried as a module-level
/// const so the value is greppable + paired with a single doc
/// explaining the choice rather than left as a magic number at the
/// call site.
#[cfg(test)]
const POST_SIGKILL_GRACE: std::time::Duration = std::time::Duration::from_secs(2);

/// Wait (kernel-evented via `pidfd_wait_exit`) for `/proc/{pid}`
/// absence up to `timeout`. Returns `true` if the pid's procfs
/// entry disappears within the budget, `false` otherwise. The
/// `interval` parameter is currently unused (the evented path
/// carries no sleep cadence).
///
/// Single source of truth for "wait until the kernel runs
/// release_task for this pid": [`kill_scheduler_process`] uses it to
/// observe SIGTERM / SIGKILL aftermath, and [`poll_startup`]'s
/// pidfd-unavailable fallback uses it to observe early-death during
/// scheduler launch. Both call sites need the same SIG_IGN-safe
/// latency profile, so folding the loop here keeps a future EINTR
/// or signal-pause refinement applied uniformly.
#[cfg(test)]
pub(crate) fn poll_proc_pid_absent(
    pid: u32,
    _interval: std::time::Duration,
    timeout: std::time::Duration,
) -> bool {
    // Evented via `pidfd_wait_exit` in the shared
    // `freeze_coord::evented_wait` module. The kernel fires POLLIN
    // on the pidfd when the task enters EXIT_ZOMBIE
    // (do_notify_pidfd from exit_notify in kernel/exit.c). The
    // closure passes `proc_pid_alive` as the source of truth so
    // races between SIGTERM/SIGKILL and `pidfd_open` resolve to
    // the /proc-observable answer.
    let start = std::time::Instant::now();
    let deadline = start + timeout;
    let exited = crate::vmm::freeze_coord::evented_wait::pidfd_wait_exit(pid, deadline, || {
        proc_pid_alive(pid)
    });
    if !exited {
        // Log on timeout so the caller chain — which may swallow
        // the bool into a non-error path — leaves a visible
        // breadcrumb in /tmp/ktstr*.log per the "log on timeout
        // when no error surfaces" rule.
        tracing::warn!(
            pid,
            elapsed_s = start.elapsed().as_secs_f64(),
            timeout_s = timeout.as_secs_f64(),
            "poll_proc_pid_absent: timeout — pid still alive after deadline; \
             pidfd POLLIN never fired and /proc entry persists. Common causes: \
             scheduler not honoring SIGTERM (check its signal handler), scheduler \
             stuck in D-state on a kernel mutex, or the caller's grace window is \
             too tight for the scheduler's exit path (post-libbpf-detach can take \
             seconds on cold caches)"
        );
    }
    exited
}
