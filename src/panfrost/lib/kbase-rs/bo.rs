// Copyright © 2026 Collabora, Ltd.
// Copyright © 2026 Pix Philosophy (HK) Limited
// SPDX-License-Identifier: MIT

use std::env;
use std::ffi::{c_int, c_void};
use std::mem::size_of;
use std::os::fd::{AsRawFd, BorrowedFd, FromRawFd, IntoRawFd, OwnedFd};
use std::ptr;
use std::sync::LazyLock;
use std::sync::atomic::Ordering::Relaxed;

use crate::dev::Dev;
use crate::pan_kmod::*;
use crate::uapi::*;
use crate::{LogErr, inout, mmap, set_errno, slice};

#[repr(C)]
pub struct Bo {
    base: KmodBo,
    gpu_va: u64,
    map: *mut c_void,
    cpu: *mut c_void,
    dmabuf: Option<OwnedFd>,
    same_va: bool,
}

impl Bo {
    fn new(
        dev: &Dev,
        size: u64,
        map: *mut c_void,
        va: u64,
        same_va: bool,
    ) -> Bo {
        Bo {
            base: KmodBo::new(ptr::from_ref(&dev.base).cast_mut(), size),
            gpu_va: if same_va { map as u64 } else { va },
            map,
            cpu: map,
            dmabuf: None,
            same_va,
        }
    }

    unsafe fn get<'a>(bo: *const KmodBo) -> &'a Bo {
        unsafe { &*bo.cast() }
    }

    fn into_kmod(mut self, vm: *mut KmodVm, flags: u32) -> *mut KmodBo {
        let dev = unsafe { Dev::get(self.base.dev) };
        self.base.handle = dev.next_handle.fetch_add(1, Relaxed);
        self.base.flags = flags;
        self.base.exclusive_vm = vm;
        unsafe { &*dev.base.allocator }.alloc(self).cast()
    }
}

impl Drop for Bo {
    fn drop(&mut self) {
        let size = self.base.size as usize;
        unsafe {
            if !self.cpu.is_null() && self.cpu != self.map {
                libc::munmap(self.cpu, size);
            }
            libc::munmap(self.map, size);
        }
        let dev = unsafe { Dev::get(self.base.dev) };
        if self.same_va {
            dev.free_deferred_phys();
        } else {
            dev.ioctl(KBASE_IOCTL_MEM_FREE, self.gpu_va)
                .log_err("KBASE_IOCTL_MEM_FREE");
        }
    }
}

fn mem_flags(flags: u32) -> u64 {
    static COHERENT_LOCAL: LazyLock<bool> = LazyLock::new(|| {
        env::var("PANVK_KBASE_COHERENT_LOCAL").as_deref() != Ok("0")
    });
    const FLAGS: [(u32, u64); 5] = [
        (PAN_KMOD_BO_FLAG_GPU_UNCACHED, BASE_MEM_UNCACHED_GPU),
        (PAN_KMOD_BO_FLAG_WB_MMAP, BASE_MEM_CACHED_CPU),
        (PAN_KMOD_BO_FLAG_IO_COHERENT, BASE_MEM_COHERENT_SYSTEM),
        (PAN_KMOD_BO_FLAG_ALLOC_ON_FAULT, BASE_MEM_GROW_ON_GPF),
        (PAN_KMOD_BO_FLAG_CSF_EVENT, BASE_MEM_CSF_EVENT),
    ];

    let mut mem =
        BASE_MEM_PROT_CPU_RD | BASE_MEM_PROT_CPU_WR | BASE_MEM_PROT_GPU_RD;
    mem |= if flags & PAN_KMOD_BO_FLAG_EXECUTABLE != 0 {
        BASE_MEM_PROT_GPU_EX
    } else {
        BASE_MEM_PROT_GPU_WR | BASE_MEM_SAME_VA
    };
    let private = PAN_KMOD_BO_FLAG_CSF_EVENT | PAN_KMOD_BO_FLAG_GPU_PRIVATE;
    if *COHERENT_LOCAL && flags & private == 0 {
        mem |= BASE_MEM_COHERENT_LOCAL;
    }
    FLAGS
        .iter()
        .filter(|(kmod, _)| flags & kmod != 0)
        .fold(mem, |mem, (_, base)| mem | base)
}

fn import(
    dev: &Dev,
    dmabuf: OwnedFd,
    flags: u32,
    external: bool,
) -> *mut KmodBo {
    let mut mem = BASE_MEM_PROT_CPU_RD
        | BASE_MEM_PROT_CPU_WR
        | BASE_MEM_PROT_GPU_RD
        | BASE_MEM_PROT_GPU_WR
        | BASE_MEM_IMPORT_SHARED
        | BASE_MEM_COHERENT_LOCAL;
    if !external {
        mem |= BASE_MEM_COHERENT_SYSTEM;
    }
    if flags & PAN_KMOD_BO_FLAG_GPU_UNCACHED != 0 {
        mem |= BASE_MEM_UNCACHED_GPU;
    }
    if flags & PAN_KMOD_BO_FLAG_WB_MMAP != 0 {
        mem |= BASE_MEM_CACHED_CPU;
    }

    let fd = dmabuf.as_raw_fd();
    let req = MemImport {
        flags: mem,
        phandle: ptr::from_ref(&fd) as u64,
        ty: BASE_MEM_IMPORT_TYPE_UMM,
        ..Default::default()
    };
    let Some(out) = dev
        .inout_compat::<_, MemOut>(
            &dev.mem_import_legacy,
            KBASE_IOCTL_MEM_IMPORT,
            KBASE_IOCTL_MEM_IMPORT_LEGACY,
            req,
        )
        .log_err("KBASE_IOCTL_MEM_IMPORT")
    else {
        return ptr::null_mut();
    };

    let size = out.va_pages * 4096;
    let rw = libc::PROT_READ | libc::PROT_WRITE;
    let map = match mmap(dev.fd(), size as usize, rw, out.gpu_va) {
        Ok(map) => map,
        Err(e) => {
            loge!("mmap of imported dma-buf failed: {e}");
            let _ = dev.ioctl(KBASE_IOCTL_MEM_FREE, out.gpu_va);
            return ptr::null_mut();
        }
    };
    let same_va = out.flags & (BASE_MEM_SAME_VA | BASE_MEM_NEED_MMAP) != 0;
    let mut bo = Bo::new(dev, size, map, out.gpu_va, same_va);
    bo.dmabuf = Some(dmabuf);
    bo.cpu = ptr::null_mut();
    if flags & PAN_KMOD_BO_FLAG_NO_MMAP == 0 {
        match mmap(fd, size as usize, rw, 0) {
            Ok(cpu) => bo.cpu = cpu,
            Err(e) => {
                loge!("CPU mmap of imported dma-buf failed: {e}");
                return ptr::null_mut();
            }
        }
    }

    logd!("imported dma-buf {fd}, size {size}");
    let imported = if external {
        PAN_KMOD_BO_FLAG_IMPORTED
    } else {
        0
    };
    bo.into_kmod(ptr::null_mut(), flags | imported)
}

fn alloc_dmabuf(
    dev: &Dev,
    heap: &OwnedFd,
    size: u64,
    flags: u32,
) -> *mut KmodBo {
    let req = DmaHeapAlloc {
        len: size.next_multiple_of(4096),
        fd_flags: (libc::O_RDWR | libc::O_CLOEXEC) as u32,
        ..Default::default()
    };
    match inout::<_, DmaHeapAlloc>(heap.as_raw_fd(), DMA_HEAP_IOCTL_ALLOC, req)
        .log_err("DMA_HEAP_IOCTL_ALLOC")
    {
        Some(out) => {
            let dmabuf = unsafe { OwnedFd::from_raw_fd(out.fd as c_int) };
            import(dev, dmabuf, flags, false)
        }
        None => ptr::null_mut(),
    }
}

pub extern "C" fn bo_alloc(
    dev: *mut KmodDev,
    vm: *mut KmodVm,
    size: u64,
    flags: u32,
) -> *mut KmodBo {
    let d = unsafe { Dev::get(dev) };
    let gpu_only = PAN_KMOD_BO_FLAG_EXECUTABLE
        | PAN_KMOD_BO_FLAG_ALLOC_ON_FAULT
        | PAN_KMOD_BO_FLAG_CSF_EVENT;
    if let Some(heap) = &d.dma_heap {
        if vm.is_null() && flags & gpu_only == 0 {
            return alloc_dmabuf(d, heap, size, flags);
        }
    }

    let pages = size.div_ceil(4096);
    let grow = flags & PAN_KMOD_BO_FLAG_ALLOC_ON_FAULT != 0;
    let req = MemAlloc {
        va_pages: pages,
        commit_pages: if grow { 0 } else { pages },
        extension: if grow { (2 << 20) / 4096 } else { 0 },
        flags: mem_flags(flags)
            | (d.mem_group as u64) << BASEP_MEM_GROUP_ID_SHIFT,
        ..Default::default()
    };
    let out: Option<MemOut> = if d.csf && d.version_at_least(1, 9) {
        d.inout_compat(
            &d.mem_alloc_ex_1_9,
            KBASE_IOCTL_MEM_ALLOC_EX,
            KBASE_IOCTL_MEM_ALLOC_EX_1_9,
            req,
        )
        .log_err("KBASE_IOCTL_MEM_ALLOC_EX")
    } else {
        d.inout(KBASE_IOCTL_MEM_ALLOC, req)
            .log_err("KBASE_IOCTL_MEM_ALLOC")
    };
    let Some(out) = out else {
        return ptr::null_mut();
    };

    let size = pages * 4096;
    let rw = libc::PROT_READ | libc::PROT_WRITE;
    match mmap(d.fd(), size as usize, rw, out.gpu_va) {
        Ok(map) => {
            let same_va = out.flags & BASE_MEM_SAME_VA != 0;
            Bo::new(d, size, map, out.gpu_va, same_va).into_kmod(vm, flags)
        }
        Err(e) => {
            loge!(
                "mmap of BO (cookie/VA {:#x}, size {size}) failed: {e}",
                out.gpu_va
            );
            let _ = d.ioctl(KBASE_IOCTL_MEM_FREE, out.gpu_va);
            ptr::null_mut()
        }
    }
}

pub extern "C" fn bo_free(bo: *mut KmodBo) {
    unsafe {
        let dev = (*bo).dev;
        if (*bo).has_pending_deferred_syncs {
            pan_kmod_flush_bo_map_syncs(dev);
        }
        (*(*dev).allocator).dealloc(bo.cast::<Bo>());
    }
}

pub extern "C" fn bo_import_fd(
    dev: *mut KmodDev,
    fd: c_int,
    _size: u64,
) -> *mut KmodBo {
    let d = unsafe { Dev::get(dev) };
    match unsafe { BorrowedFd::borrow_raw(fd) }.try_clone_to_owned() {
        Ok(dmabuf) => import(d, dmabuf, 0, true),
        Err(e) => {
            loge!("failed to duplicate dma-buf: {e}");
            ptr::null_mut()
        }
    }
}

pub extern "C" fn bo_export_fd(bo: *mut KmodBo) -> c_int {
    match &unsafe { Bo::get(bo) }.dmabuf {
        Some(dmabuf) => dmabuf.try_clone().map_or(-1, IntoRawFd::into_raw_fd),
        None => {
            set_errno(libc::ENOSYS);
            -1
        }
    }
}

pub extern "C" fn bo_mmap(
    bo: *mut KmodBo,
    _prot: c_int,
    _flags: c_int,
    host_addr: *mut c_void,
) -> *mut c_void {
    let bo = unsafe { Bo::get(bo) };
    if !host_addr.is_null() && host_addr != bo.cpu {
        loge!(
            "mapping a BO at a caller-chosen address is not supported \
             (SAME_VA)"
        );
        set_errno(libc::ENOTSUP);
        libc::MAP_FAILED
    } else if bo.cpu.is_null() {
        set_errno(libc::EINVAL);
        libc::MAP_FAILED
    } else {
        bo.cpu
    }
}

pub extern "C" fn bo_munmap(
    _bo: *mut KmodBo,
    _addr: *mut c_void,
    _size: usize,
) -> c_int {
    0
}

pub extern "C" fn bo_wait(
    _bo: *mut KmodBo,
    _timeout_ns: i64,
    _read_only: bool,
) -> bool {
    true
}

pub extern "C" fn flush_bo_map_syncs(dev: *mut KmodDev) -> c_int {
    let d = unsafe { Dev::get(dev) };
    let pending = &d.base.pending_bo_syncs.array;
    let count = pending.size as usize / size_of::<DeferredBoSync>();
    for sync in unsafe { slice(pending.data.cast::<DeferredBoSync>(), count) } {
        let bo = unsafe { Bo::get(sync.bo) };
        let req = MemSync {
            handle: bo.gpu_va,
            user_addr: bo.cpu as u64 + sync.start,
            size: sync.size,
            ty: if sync.ty == PAN_KMOD_BO_SYNC_CPU_CACHE_FLUSH {
                BASE_SYNCSET_OP_MSYNC
            } else {
                BASE_SYNCSET_OP_CSYNC
            },
            ..Default::default()
        };
        if d.ioctl(KBASE_IOCTL_MEM_SYNC, req)
            .log_err("KBASE_IOCTL_MEM_SYNC")
            .is_none()
        {
            return -1;
        }
    }
    0
}

pub extern "C" fn vm_create(
    dev: *mut KmodDev,
    flags: u32,
    _va_start: u64,
    _va_range: u64,
) -> *mut KmodVm {
    unsafe { &*(*dev).allocator }.alloc(KmodVm::new(dev, flags))
}

pub extern "C" fn vm_destroy(vm: *mut KmodVm) {
    unsafe {
        pan_kmod_bo_put((*vm).sparse_dummy_bo);
        let allocator = (*(*vm).dev).allocator;
        (*allocator).dealloc(vm);
    }
}

pub extern "C" fn vm_bind(
    _vm: *mut KmodVm,
    mode: u32,
    ops: *mut VmOp,
    count: u32,
) -> c_int {
    if mode == PAN_KMOD_VM_OP_MODE_ASYNC {
        loge!("PAN_KMOD_VM_OP_MODE_ASYNC not supported");
        return -1;
    }
    for i in 0..count as usize {
        let op = unsafe { &mut *ops.add(i) };
        if op.ty != PAN_KMOD_VM_OP_TYPE_MAP {
            continue;
        }
        let va = unsafe { Bo::get(op.bo) }.gpu_va + op.bo_offset as u64;
        if op.va_start == PAN_KMOD_VM_MAP_AUTO_VA {
            op.va_start = va;
        } else if op.va_start != va {
            loge!(
                "cannot map BO at explicit VA {:#x} (kbase-assigned VA is \
                 {va:#x})",
                op.va_start
            );
            return -1;
        }
    }
    0
}

#[unsafe(no_mangle)]
pub extern "C" fn kbase_kmod_alias_create(
    dev: *mut KmodDev,
    bo_va: u64,
    size: u64,
    nents: u32,
) -> u64 {
    let d = unsafe { Dev::get(dev) };
    debug_assert!((1..=4).contains(&nents));
    debug_assert!(size % 4096 == 0 && bo_va % 4096 == 0);

    let info = [AliasingInfo {
        handle: bo_va,
        offset: 0,
        length: size / 4096,
    }; 4];
    let req = MemAlias {
        flags: BASE_MEM_PROT_GPU_RD
            | BASE_MEM_PROT_GPU_WR
            | BASE_MEM_PROT_CPU_RD,
        stride: size / 4096,
        nents: nents.into(),
        aliasing_info: info.as_ptr() as u64,
    };
    let total = size * nents as u64;
    for _ in 0..16 {
        let Some(out) = d
            .inout::<_, MemOut>(KBASE_IOCTL_MEM_ALIAS, req)
            .log_err("KBASE_IOCTL_MEM_ALIAS")
        else {
            return 0;
        };
        let va = match mmap(d.fd(), total as usize, libc::PROT_READ, out.gpu_va)
        {
            Ok(ptr) => ptr as u64,
            Err(e) => {
                loge!("mmap of alias region failed: {e}");
                let _ = d.ioctl(KBASE_IOCTL_MEM_FREE, out.gpu_va);
                return 0;
            }
        };
        if va >> 32 == (va + total - 1) >> 32 {
            return va;
        }
        kbase_kmod_alias_destroy(dev, va, size, nents);
    }
    loge!("could not get an alias mapping that doesn't cross a 4G boundary");
    0
}

#[unsafe(no_mangle)]
pub extern "C" fn kbase_kmod_alias_destroy(
    dev: *mut KmodDev,
    va: u64,
    size: u64,
    nents: u32,
) {
    unsafe { libc::munmap(va as *mut c_void, (size * nents as u64) as usize) };
    unsafe { Dev::get(dev) }.free_deferred_phys();
}
