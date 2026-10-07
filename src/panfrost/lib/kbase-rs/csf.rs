// Copyright © 2026 Collabora, Ltd.
// Copyright © 2026 Pix Philosophy (HK) Limited
// SPDX-License-Identifier: MIT

use std::ffi::{c_int, c_void};
use std::io;
use std::mem::size_of;
use std::os::fd::{AsRawFd, FromRawFd, IntoRawFd, OwnedFd};
use std::ptr;
use std::sync::atomic::Ordering::Relaxed;

use crate::dev::{Dev, Kcpu, Pending};
use crate::pan_kmod::KmodDev;
use crate::uapi::*;
use crate::{LogErr, mmap, set_errno, slice};

#[repr(C)]
pub struct UserIo {
    map: *mut c_void,
    doorbell: *mut u8,
    input: *mut u8,
    output: *mut u8,
}

fn poll_in(fd: c_int, timeout_ns: i64) -> io::Result<i16> {
    let mut pfd = libc::pollfd {
        fd,
        events: libc::POLLIN,
        revents: 0,
    };
    let ts = libc::timespec {
        tv_sec: (timeout_ns / 1_000_000_000) as _,
        tv_nsec: (timeout_ns % 1_000_000_000) as _,
    };
    match unsafe { libc::ppoll(&mut pfd, 1, &ts, ptr::null()) } {
        -1 => Err(io::Error::last_os_error()),
        _ => Ok(pfd.revents),
    }
}

fn poll_fence(fence: &OwnedFd, timeout_ns: i64) -> io::Result<bool> {
    loop {
        match poll_in(fence.as_raw_fd(), timeout_ns) {
            Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
            Err(e) => return Err(e),
            Ok(0) => return Ok(false),
            Ok(ev)
                if ev & (libc::POLLIN | libc::POLLERR | libc::POLLHUP) != 0 =>
            {
                return Ok(true);
            }
            Ok(ev) => {
                return Err(io::Error::other(format!("poll events {ev:#x}")));
            }
        }
    }
}

impl Dev {
    fn kcpu_enqueue(&self, id: u8, cmd: KcpuCommand) -> io::Result<c_int> {
        let req = KcpuEnqueue {
            addr: ptr::from_ref(&cmd) as u64,
            nr_commands: 1,
            id,
            ..Default::default()
        };
        self.ioctl(KBASE_IOCTL_KCPU_QUEUE_ENQUEUE, req)
    }

    fn kcpu_fence(
        &self,
        id: u8,
        objs: &[CqsWaitOperation],
    ) -> io::Result<OwnedFd> {
        if !objs.is_empty() {
            self.kcpu_enqueue(id, KcpuCommand::cqs_wait(objs))?;
        }
        let mut fence = Fence {
            fd: -1,
            stream_fd: -1,
        };
        self.kcpu_enqueue(id, KcpuCommand::fence_signal(&mut fence))?;
        if fence.fd < 0 {
            return Err(io::Error::other("no sync fence"));
        }
        Ok(unsafe { OwnedFd::from_raw_fd(fence.fd) })
    }

    fn kcpu_disable(&self, kcpu: &mut Kcpu) -> c_int {
        kcpu.pending = None;
        if let Some(id) = kcpu.id.take() {
            let _ = self.ioctl(KBASE_IOCTL_KCPU_QUEUE_DELETE, id as u64);
        }
        loge!("kcpu queue disabled due to error");
        -1
    }

    fn notification(&self, ev: &CsfNotification) -> c_int {
        if ev.ty != BASE_CSF_NOTIFICATION_GPU_QUEUE_GROUP_ERROR {
            match ev.ty {
                BASE_CSF_NOTIFICATION_EVENT => {
                    logd!("received CSF event notification")
                }
                BASE_CSF_NOTIFICATION_CPU_QUEUE_DUMP => {
                    logw!("received CSF CPU queue dump notification")
                }
                ty => logw!("received unknown CSF notification type {ty}"),
            }
            return 0;
        }

        self.csf_error.store(true, Relaxed);
        let group = ev.group_handle;
        let queue_error = |kind: &str| {
            loge!(
                "CSF group {group} CSI {} {kind}: status {:#010x} \
                 (exception {:#04x}), sideband {:#018x}",
                ev.csi_index,
                ev.status,
                ev.status & 0xff,
                ev.sideband
            );
            if ev.has_extra != 0 {
                loge!(
                    "CSF group {group} CSI {} fault trace: id0 {:#010x}, \
                     id1 {:#010x}, task {:#010x}",
                    ev.csi_index,
                    ev.trace_id0,
                    ev.trace_id1,
                    ev.trace_task
                );
            }
        };
        match ev.error_type {
            BASE_GPU_QUEUE_GROUP_ERROR_FATAL => loge!(
                "CSF group {group} fatal error: status {:#010x} \
                 (exception {:#04x}), sideband {:#018x}",
                ev.status,
                ev.status & 0xff,
                ev.sideband
            ),
            BASE_GPU_QUEUE_GROUP_QUEUE_ERROR_FATAL => {
                queue_error("fatal error")
            }
            BASE_GPU_QUEUE_GROUP_ERROR_TIMEOUT => {
                loge!("CSF group {group} progress timeout notification")
            }
            BASE_GPU_QUEUE_GROUP_ERROR_TILER_HEAP_OOM => {
                loge!("CSF group {group} tiler heap OOM notification")
            }
            BASE_GPU_QUEUE_GROUP_QUEUE_ERROR_FAULT => {
                queue_error("recoverable fault")
            }
            ty => loge!("CSF group {group} unknown error type {ty}"),
        }
        ev.error_type as c_int + 1
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn kbase_kmod_csf_group_create(
    dev: *mut KmodDev,
    cs_queue_count: u32,
    tiler_oom_handler: bool,
    max_cores: u32,
    dvs_buf: u64,
    compute_priority: bool,
    group_handle: *mut u32,
) -> c_int {
    let d = unsafe { Dev::get(dev) };
    let cores = if max_cores == 0 {
        64
    } else {
        max_cores.min(64) as u8
    };
    let csi_handlers = if tiler_oom_handler {
        BASE_CSF_TILER_OOM_EXCEPTION_FLAG
    } else {
        0
    };
    let base = GroupCreate {
        tiler_mask: 1,
        fragment_mask: !0,
        compute_mask: !0,
        cs_min: cs_queue_count as u8,
        tiler_max: 1,
        fragment_max: cores,
        compute_max: cores,
        ..Default::default()
    };

    let mut attempts = Vec::new();
    if d.version_at_least(1, 25) {
        let req = if d.version_at_least(1, 36) {
            KBASE_IOCTL_CS_QUEUE_GROUP_CREATE
        } else {
            KBASE_IOCTL_CS_QUEUE_GROUP_CREATE_1_19
        };
        for pri in (0..=compute_priority as u8).rev() {
            let create = GroupCreate {
                csi_handlers,
                cs_fault_report_enable: 1,
                dvs_buf,
                comp_pri_threshold: 4 * pri,
                comp_pri_ratio: 3 * pri,
                ..base
            };
            attempts.push((req, create));
        }
    }
    if dvs_buf == 0 && d.version_at_least(1, 18) {
        let create = GroupCreate {
            csi_handlers,
            ..base
        };
        attempts.push((KBASE_IOCTL_CS_QUEUE_GROUP_CREATE_1_18, create));
    }
    if dvs_buf == 0 && !tiler_oom_handler {
        attempts.push((KBASE_IOCTL_CS_QUEUE_GROUP_CREATE_1_6, base));
    }

    for (req, create) in attempts {
        match d.inout::<_, u8>(req, create) {
            Ok(handle) => {
                logd!(
                    "created CSF group {handle}, DVS buffer {dvs_buf:#x}, \
                     compute priority {}/{}",
                    create.comp_pri_threshold,
                    create.comp_pri_ratio
                );
                unsafe { group_handle.write(handle.into()) };
                return 0;
            }
            Err(e) => logw!("CS_QUEUE_GROUP_CREATE ({req:#x}) failed: {e}"),
        }
    }
    if dvs_buf == 0 && tiler_oom_handler {
        loge!(
            "incremental rendering requires CS_QUEUE_GROUP_CREATE 1.18 or newer"
        );
    }
    -1
}

#[unsafe(no_mangle)]
pub extern "C" fn kbase_kmod_csf_group_destroy(
    dev: *mut KmodDev,
    group_handle: u32,
) {
    unsafe { Dev::get(dev) }
        .ioctl(
            KBASE_IOCTL_CS_QUEUE_GROUP_TERMINATE,
            group_handle as u8 as u64,
        )
        .log_err("KBASE_IOCTL_CS_QUEUE_GROUP_TERMINATE");
}

#[unsafe(no_mangle)]
pub extern "C" fn kbase_kmod_csf_queue_bind(
    dev: *mut KmodDev,
    group_handle: u32,
    csi_index: u32,
    ringbuf_va: u64,
    ringbuf_size: u32,
    io: *mut UserIo,
) -> bool {
    let d = unsafe { Dev::get(dev) };
    let reg = CsQueueRegister {
        buffer_gpu_addr: ringbuf_va,
        buffer_size: ringbuf_size,
        ..Default::default()
    };
    if d.ioctl(KBASE_IOCTL_CS_QUEUE_REGISTER, reg)
        .log_err("KBASE_IOCTL_CS_QUEUE_REGISTER")
        .is_none()
    {
        return false;
    }

    let bind = CsQueueBind {
        buffer_gpu_addr: ringbuf_va,
        group_handle: group_handle as u8,
        csi_index: csi_index as u8,
        ..Default::default()
    };
    let len = BASEP_QUEUE_NR_MMAP_USER_PAGES * 4096;
    let rw = libc::PROT_READ | libc::PROT_WRITE;
    let Some(map) = d
        .inout(KBASE_IOCTL_CS_QUEUE_BIND, bind)
        .log_err("KBASE_IOCTL_CS_QUEUE_BIND")
        .and_then(|handle| {
            mmap(d.fd(), len, rw, handle).log_err("mmap of CS USER_IO pages")
        })
    else {
        let _ = d.ioctl(KBASE_IOCTL_CS_QUEUE_TERMINATE, ringbuf_va);
        return false;
    };

    let slot = if d.version_at_least(1, 35) {
        16 * csi_index as usize
    } else {
        0
    };
    let pages = map.cast::<u8>();
    unsafe {
        io.write(UserIo {
            map,
            doorbell: pages,
            input: pages.add(4096 + slot),
            output: pages.add(2 * 4096 + slot),
        })
    };
    true
}

#[unsafe(no_mangle)]
pub extern "C" fn kbase_kmod_csf_queue_term(
    dev: *mut KmodDev,
    ringbuf_va: u64,
    io: *mut UserIo,
) {
    unsafe {
        let map = (*io).map;
        if !map.is_null() {
            libc::munmap(map, BASEP_QUEUE_NR_MMAP_USER_PAGES * 4096);
        }
        io.write_bytes(0, 1);
        Dev::get(dev)
            .ioctl(KBASE_IOCTL_CS_QUEUE_TERMINATE, ringbuf_va)
            .log_err("KBASE_IOCTL_CS_QUEUE_TERMINATE");
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn kbase_kmod_csf_queue_kick(
    dev: *mut KmodDev,
    ringbuf_va: u64,
) -> c_int {
    unsafe { Dev::get(dev) }
        .ioctl(KBASE_IOCTL_CS_QUEUE_KICK, ringbuf_va)
        .log_err("KBASE_IOCTL_CS_QUEUE_KICK")
        .map_or(-1, |_| 0)
}

#[unsafe(no_mangle)]
pub extern "C" fn kbase_kmod_csf_wait_event(
    dev: *mut KmodDev,
    timeout_ns: i64,
) -> c_int {
    let d = unsafe { Dev::get(dev) };
    match poll_in(d.fd(), timeout_ns) {
        Ok(ev) if ev & libc::POLLIN != 0 => {}
        Err(e) if e.kind() != io::ErrorKind::Interrupted => return -1,
        _ => return 0,
    }

    let mut ev = CsfNotification::default();
    let size = size_of::<CsfNotification>();
    loop {
        match unsafe { libc::read(d.fd(), (&raw mut ev).cast(), size) } {
            n if n == size as isize => return d.notification(&ev),
            -1 if io::Error::last_os_error().kind()
                == io::ErrorKind::Interrupted => {}
            -1 => return -1,
            _ => {
                set_errno(libc::EIO);
                return -1;
            }
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn kbase_kmod_csf_wait_cqs64(
    dev: *mut KmodDev,
    addr: u64,
    target_minus_one: u64,
    timeout_ns: i64,
) -> c_int {
    let d = unsafe { Dev::get(dev) };
    if !d.csf || addr & 15 != 0 {
        return -1;
    }
    let mut kcpu = d.kcpu.lock().unwrap();
    let Some(id) = kcpu.id else {
        return -1;
    };

    if let Some(pending) = &kcpu.pending {
        let same = pending.addr == addr && pending.target == target_minus_one;
        match poll_fence(&pending.fence, if same { timeout_ns } else { 0 }) {
            Ok(true) => {
                kcpu.pending = None;
                if same {
                    return 1;
                }
            }
            Ok(false) => return if same { 0 } else { -1 },
            Err(e) => {
                logw!("KCPU sync fence poll failed: {e}");
                return d.kcpu_disable(&mut kcpu);
            }
        }
    }

    let wait = [CqsWaitOperation::gt_u64(addr, target_minus_one)];
    let fence = match d.kcpu_fence(id, &wait) {
        Ok(fence) => fence,
        Err(e) => {
            logd!("KCPU CQS wait unavailable: {e}");
            return d.kcpu_disable(&mut kcpu);
        }
    };
    match poll_fence(&fence, timeout_ns) {
        Ok(true) => 1,
        Ok(false) => {
            kcpu.pending = Some(Pending {
                addr,
                target: target_minus_one,
                fence,
            });
            0
        }
        Err(e) => {
            loge!("KCPU sync fence poll failed: {e}");
            d.kcpu_disable(&mut kcpu)
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn kbase_kmod_csf_has_kcpu(dev: *const KmodDev) -> bool {
    unsafe { Dev::get(dev) }.export_queue.is_some()
}

#[unsafe(no_mangle)]
pub extern "C" fn kbase_kmod_csf_cqs_fence_create(
    dev: *mut KmodDev,
    addrs: *const u64,
    targets: *const u64,
    count: u32,
) -> c_int {
    let d = unsafe { Dev::get(dev) };
    let count = count as usize;
    let Some(id) = d.export_queue.filter(|_| count <= KBASE_KCPU_CQS_MAX_OBJS)
    else {
        return -1;
    };
    let cells =
        unsafe { slice(addrs, count).iter().zip(slice(targets, count)) };
    if cells
        .clone()
        .any(|(&addr, &target)| addr & 15 != 0 || target == 0)
    {
        return -1;
    }
    let objs: Vec<_> = cells
        .map(|(&addr, &target)| CqsWaitOperation::gt_u64(addr, target - 1))
        .collect();

    let _serialize = d.kcpu.lock().unwrap();
    match d.kcpu_fence(id, &objs) {
        Ok(fence) => fence.into_raw_fd(),
        Err(e) => {
            logw!("KCPU export fence enqueue failed: {e}");
            -1
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn kbase_kmod_csf_has_error(dev: *const KmodDev) -> bool {
    unsafe { Dev::get(dev) }.csf_error.load(Relaxed)
}

#[unsafe(no_mangle)]
pub extern "C" fn kbase_kmod_csf_tiler_heap_create(
    dev: *mut KmodDev,
    chunk_size: u32,
    initial_chunks: u32,
    max_chunks: u32,
    target_in_flight: u32,
    heap_ctx_va: *mut u64,
    first_chunk_va: *mut u64,
) -> c_int {
    let d = unsafe { Dev::get(dev) };
    let req = TilerHeapInit {
        chunk_size,
        initial_chunks,
        max_chunks,
        target_in_flight: target_in_flight.min(u16::MAX.into()) as u16,
        group_id: d.mem_group,
        padding: 0,
    };
    match d
        .inout::<_, [u64; 2]>(KBASE_IOCTL_CS_TILER_HEAP_INIT, req)
        .log_err("KBASE_IOCTL_CS_TILER_HEAP_INIT")
    {
        Some([heap, chunk]) => {
            unsafe {
                heap_ctx_va.write(heap);
                first_chunk_va.write(chunk);
            }
            0
        }
        None => -1,
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn kbase_kmod_csf_tiler_heap_destroy(
    dev: *mut KmodDev,
    heap_ctx_va: u64,
) {
    unsafe { Dev::get(dev) }
        .ioctl(KBASE_IOCTL_CS_TILER_HEAP_TERM, heap_ctx_va)
        .log_err("KBASE_IOCTL_CS_TILER_HEAP_TERM");
}
