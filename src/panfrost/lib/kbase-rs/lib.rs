// Copyright © 2026 Collabora, Ltd.
// Copyright © 2026 Pix Philosophy (HK) Limited
// SPDX-License-Identifier: MIT

use std::ffi::{CString, c_int, c_void};
use std::{fmt, io, ptr};

macro_rules! loge {
    ($($arg:tt)*) => {
        $crate::log($crate::pan_kmod::MESA_LOG_ERROR, format_args!($($arg)*))
    };
}

macro_rules! logw {
    ($($arg:tt)*) => {
        $crate::log($crate::pan_kmod::MESA_LOG_WARN, format_args!($($arg)*))
    };
}

macro_rules! logd {
    ($($arg:tt)*) => {
        if cfg!(debug_assertions) {
            $crate::log($crate::pan_kmod::MESA_LOG_DEBUG, format_args!($($arg)*))
        }
    };
}

mod bo;
mod csf;
mod dev;
mod pan_kmod;
mod uapi;

use pan_kmod::{KmodOps, mesa_log};

fn log(level: u32, args: fmt::Arguments) {
    let msg = CString::new(format!("kbase: {args}")).unwrap_or_default();
    unsafe { mesa_log(level, c"MESA".as_ptr(), c"%s".as_ptr(), msg.as_ptr()) };
}

trait LogErr<T> {
    fn log_err(self, what: &str) -> Option<T>;
}

impl<T> LogErr<T> for io::Result<T> {
    fn log_err(self, what: &str) -> Option<T> {
        self.map_err(|e| loge!("{what} failed: {e}")).ok()
    }
}

fn raw_ioctl<T>(fd: c_int, req: u32, arg: &mut T) -> io::Result<c_int> {
    loop {
        match unsafe { libc::ioctl(fd, req as _, ptr::from_mut(arg)) } {
            -1 => {
                let err = io::Error::last_os_error();
                if err.kind() != io::ErrorKind::Interrupted {
                    return Err(err);
                }
            }
            ret => return Ok(ret),
        }
    }
}

fn ioctl<T>(fd: c_int, req: u32, mut arg: T) -> io::Result<c_int> {
    raw_ioctl(fd, req, &mut arg)
}

fn inout<I: Copy, O: Copy>(fd: c_int, req: u32, i: I) -> io::Result<O> {
    let mut arg = uapi::InOut { i };
    raw_ioctl(fd, req, &mut arg)?;
    Ok(unsafe { arg.o })
}

fn mmap(
    fd: c_int,
    len: usize,
    prot: c_int,
    offset: u64,
) -> io::Result<*mut c_void> {
    let ptr = unsafe {
        libc::mmap(
            ptr::null_mut(),
            len,
            prot,
            libc::MAP_SHARED,
            fd,
            offset as _,
        )
    };
    if ptr == libc::MAP_FAILED {
        Err(io::Error::last_os_error())
    } else {
        Ok(ptr)
    }
}

fn set_errno(err: c_int) {
    #[cfg(target_os = "android")]
    unsafe {
        *libc::__errno() = err
    };
    #[cfg(not(target_os = "android"))]
    unsafe {
        *libc::__errno_location() = err
    };
}

unsafe fn slice<'a, T>(data: *const T, len: usize) -> &'a [T] {
    if len == 0 {
        &[]
    } else {
        unsafe { std::slice::from_raw_parts(data, len) }
    }
}

#[unsafe(no_mangle)]
#[allow(non_upper_case_globals)]
pub static kbase_kmod_ops: KmodOps = KmodOps {
    dev_create: Some(dev::dev_create),
    dev_destroy: Some(dev::dev_destroy),
    dev_query_user_va_range: Some(dev::dev_query_user_va_range),
    bo_alloc: Some(bo::bo_alloc),
    bo_free: Some(bo::bo_free),
    bo_import: None,
    bo_import_fd: Some(bo::bo_import_fd),
    bo_export_fd: Some(bo::bo_export_fd),
    bo_export: None,
    bo_get_mmap_offset: None,
    bo_mmap: Some(bo::bo_mmap),
    bo_munmap: Some(bo::bo_munmap),
    flush_bo_map_syncs: Some(bo::flush_bo_map_syncs),
    bo_wait: Some(bo::bo_wait),
    bo_make_evictable: None,
    bo_make_unevictable: None,
    vm_create: Some(bo::vm_create),
    vm_destroy: Some(bo::vm_destroy),
    vm_bind: Some(bo::vm_bind),
    vm_query_state: None,
    query_timestamp: Some(dev::query_timestamp),
    bo_set_label: None,
    perf: [None; 5],
};
