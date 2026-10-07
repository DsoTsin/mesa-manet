// Copyright © 2026 Collabora, Ltd.
// Copyright © 2026 Pix Philosophy (HK) Limited
// SPDX-License-Identifier: MIT

#![allow(dead_code)]

use std::ffi::{c_char, c_int, c_void};
use std::mem::size_of;
use std::ptr;

pub const MESA_LOG_ERROR: u32 = 0;
pub const MESA_LOG_WARN: u32 = 1;
pub const MESA_LOG_DEBUG: u32 = 3;

pub const PAN_KMOD_BO_FLAG_EXECUTABLE: u32 = 1 << 0;
pub const PAN_KMOD_BO_FLAG_ALLOC_ON_FAULT: u32 = 1 << 1;
pub const PAN_KMOD_BO_FLAG_NO_MMAP: u32 = 1 << 2;
pub const PAN_KMOD_BO_FLAG_IMPORTED: u32 = 1 << 4;
pub const PAN_KMOD_BO_FLAG_GPU_UNCACHED: u32 = 1 << 5;
pub const PAN_KMOD_BO_FLAG_WB_MMAP: u32 = 1 << 6;
pub const PAN_KMOD_BO_FLAG_IO_COHERENT: u32 = 1 << 7;
pub const PAN_KMOD_BO_FLAG_CSF_EVENT: u32 = 1 << 8;
pub const PAN_KMOD_BO_FLAG_GPU_PRIVATE: u32 = 1 << 9;

pub const PAN_KMOD_GROUP_ALLOW_PRIORITY_LOW: u32 = 1 << 0;
pub const PAN_KMOD_GROUP_ALLOW_PRIORITY_MEDIUM: u32 = 1 << 1;

pub const PAN_KMOD_DEV_FLAG_OWNS_FD: u32 = 1 << 0;
pub const PAN_KMOD_DEV_FLAG_MMAP_SYNC_THROUGH_KERNEL: u32 = 1 << 1;

pub const PAN_KMOD_VM_OP_MODE_ASYNC: u32 = 1;
pub const PAN_KMOD_VM_OP_TYPE_MAP: u32 = 0;
pub const PAN_KMOD_VM_MAP_AUTO_VA: u64 = !0;
pub const PAN_KMOD_BO_SYNC_CPU_CACHE_FLUSH: u32 = 0;
pub const PAN_PGSIZE_4K: u64 = 0x1000;

#[repr(C)]
pub struct Driver {
    pub major: u32,
    pub minor: u32,
}

#[repr(C)]
pub struct DevProps {
    pub gpu_id: u64,
    pub gpu_variant: u32,
    pub shader_present: u64,
    pub tiler_features: u32,
    pub mem_features: u32,
    pub mmu_features: u32,
    pub texture_features: [u32; 4],
    pub l2_features: u32,
    pub max_threads_per_core: u32,
    pub max_tasks_per_core: u8,
    pub max_threads_per_wg: u32,
    pub num_threads_active_granularity: u32,
    pub num_registers_per_core: u32,
    pub max_tls_instance_per_core: u32,
    pub afbc_features: u32,
    pub gpu_can_query_timestamp: bool,
    pub timestamp_device_coherent: bool,
    pub timestamp_frequency: u64,
    pub timestamp_cycles_to_ns_factor: f64,
    pub allowed_group_priorities_mask: u32,
    pub supported_bo_flags: u32,
    pub supported_vm_op_flags: u32,
    pub is_io_coherent: bool,
    pub pgsize_bitmap: u64,
}

#[repr(C)]
pub struct Allocator {
    pub zalloc: extern "C" fn(*const Allocator, usize, bool) -> *mut c_void,
    pub free: extern "C" fn(*const Allocator, *mut c_void),
    pub priv_: *mut c_void,
}

#[repr(C)]
pub struct SparseArray {
    pub elem_size: usize,
    pub node_size_log2: u32,
    pub root: usize,
}

#[repr(C)]
pub struct DynArray {
    pub mem_ctx: *mut c_void,
    pub data: *mut c_void,
    pub size: u32,
    pub capacity: u32,
}

#[repr(C)]
pub struct HandleToBo {
    pub array: SparseArray,
    pub lock: u32,
}

#[repr(C)]
pub struct PendingBoSyncs {
    pub user_cache_ops_pending: bool,
    pub array: DynArray,
    pub lock: u32,
}

#[repr(C)]
pub struct KmodDev {
    pub fd: c_int,
    pub flags: u32,
    pub driver: Driver,
    pub props: DevProps,
    pub ops: *const KmodOps,
    pub handle_to_bo: HandleToBo,
    pub pending_bo_syncs: PendingBoSyncs,
    pub allocator: *const Allocator,
    pub user_priv: *mut c_void,
}

#[repr(C)]
pub struct KmodBo {
    pub refcnt: i32,
    pub size: u64,
    pub handle: u32,
    pub flags: u32,
    pub has_pending_deferred_syncs: bool,
    pub exclusive_vm: *mut KmodVm,
    pub dev: *mut KmodDev,
    pub user_priv: *mut c_void,
}

#[repr(C)]
pub struct KmodVm {
    pub pgsize_bitmap: u64,
    pub flags: u32,
    pub handle: u32,
    pub dev: *mut KmodDev,
    pub sparse_dummy_bo: *mut KmodBo,
    pub sparse_dummy_lock: u32,
}

#[repr(C)]
pub struct SyncOps {
    pub count: u32,
    pub array: *const c_void,
}

#[repr(C)]
pub struct VmOp {
    pub ty: u32,
    pub va_start: u64,
    pub va_size: u64,
    pub bo: *mut KmodBo,
    pub bo_offset: i64,
    pub signal: SyncOps,
    pub wait: SyncOps,
    pub flags: u32,
}

#[repr(C)]
pub struct DeferredBoSync {
    pub bo: *mut KmodBo,
    pub start: u64,
    pub size: u64,
    pub ty: u32,
}

#[repr(C)]
pub struct VaRange {
    pub start: u64,
    pub size: u64,
}

#[repr(C)]
#[derive(Default)]
pub struct CsifInfo {
    pub csg_slot_count: u32,
    pub cs_slot_count: u32,
    pub cs_reg_count: u32,
    pub scoreboard_slot_count: u32,
    pub unpreserved_cs_reg_count: u32,
    pub pad: u32,
}

type Unused = Option<extern "C" fn()>;

#[repr(C)]
pub struct KmodOps {
    pub dev_create: Option<
        extern "C" fn(
            c_int,
            u32,
            *const Driver,
            *const Allocator,
        ) -> *mut KmodDev,
    >,
    pub dev_destroy: Option<extern "C" fn(*mut KmodDev)>,
    pub dev_query_user_va_range:
        Option<extern "C" fn(*const KmodDev) -> VaRange>,
    pub bo_alloc: Option<
        extern "C" fn(*mut KmodDev, *mut KmodVm, u64, u32) -> *mut KmodBo,
    >,
    pub bo_free: Option<extern "C" fn(*mut KmodBo)>,
    pub bo_import: Unused,
    pub bo_import_fd:
        Option<extern "C" fn(*mut KmodDev, c_int, u64) -> *mut KmodBo>,
    pub bo_export_fd: Option<extern "C" fn(*mut KmodBo) -> c_int>,
    pub bo_export: Unused,
    pub bo_get_mmap_offset: Unused,
    pub bo_mmap: Option<
        extern "C" fn(*mut KmodBo, c_int, c_int, *mut c_void) -> *mut c_void,
    >,
    pub bo_munmap:
        Option<extern "C" fn(*mut KmodBo, *mut c_void, usize) -> c_int>,
    pub flush_bo_map_syncs: Option<extern "C" fn(*mut KmodDev) -> c_int>,
    pub bo_wait: Option<extern "C" fn(*mut KmodBo, i64, bool) -> bool>,
    pub bo_make_evictable: Unused,
    pub bo_make_unevictable: Unused,
    pub vm_create:
        Option<extern "C" fn(*mut KmodDev, u32, u64, u64) -> *mut KmodVm>,
    pub vm_destroy: Option<extern "C" fn(*mut KmodVm)>,
    pub vm_bind:
        Option<extern "C" fn(*mut KmodVm, u32, *mut VmOp, u32) -> c_int>,
    pub vm_query_state: Unused,
    pub query_timestamp: Option<extern "C" fn(*const KmodDev) -> u64>,
    pub bo_set_label: Unused,
    pub perf: [Unused; 5],
}

const _: () = assert!(
    size_of::<KmodDev>() == 240
        && size_of::<KmodBo>() == 56
        && size_of::<KmodVm>() == 40
        && size_of::<VmOp>() == 80
        && size_of::<DeferredBoSync>() == 32
        && size_of::<KmodOps>() == 27 * size_of::<usize>()
);

unsafe extern "C" {
    pub fn mesa_log(level: u32, tag: *const c_char, format: *const c_char, ...);
    pub fn util_sparse_array_finish(arr: *mut SparseArray);
    pub fn pan_kmod_flush_bo_map_syncs(dev: *mut KmodDev);
    pub fn pan_kmod_bo_put(bo: *mut KmodBo);
}

impl Allocator {
    pub fn alloc<T>(&self, value: T) -> *mut T {
        let ptr = (self.zalloc)(self, size_of::<T>(), false).cast::<T>();
        if !ptr.is_null() {
            unsafe { ptr.write(value) };
        }
        ptr
    }

    pub unsafe fn dealloc<T>(&self, ptr: *mut T) {
        unsafe { ptr.drop_in_place() };
        (self.free)(self, ptr.cast());
    }
}

impl KmodDev {
    pub fn new(
        fd: c_int,
        flags: u32,
        driver: Driver,
        props: DevProps,
        ops: *const KmodOps,
        allocator: *const Allocator,
    ) -> Self {
        KmodDev {
            fd,
            flags,
            driver,
            props,
            ops,
            handle_to_bo: HandleToBo {
                array: SparseArray {
                    elem_size: size_of::<*mut KmodBo>(),
                    node_size_log2: 512u32.ilog2(),
                    root: 0,
                },
                lock: 0,
            },
            pending_bo_syncs: PendingBoSyncs {
                user_cache_ops_pending: false,
                array: DynArray {
                    mem_ctx: ptr::null_mut(),
                    data: ptr::null_mut(),
                    size: 0,
                    capacity: 0,
                },
                lock: 0,
            },
            allocator,
            user_priv: ptr::null_mut(),
        }
    }

    pub unsafe fn cleanup(&mut self) {
        unsafe {
            if self.flags & PAN_KMOD_DEV_FLAG_OWNS_FD != 0 {
                libc::close(self.fd);
            }
            libc::free(self.pending_bo_syncs.array.data);
            util_sparse_array_finish(&mut self.handle_to_bo.array);
        }
    }
}

impl KmodBo {
    pub fn new(dev: *mut KmodDev, size: u64) -> Self {
        KmodBo {
            refcnt: 1,
            size,
            handle: 0,
            flags: 0,
            has_pending_deferred_syncs: false,
            exclusive_vm: ptr::null_mut(),
            dev,
            user_priv: ptr::null_mut(),
        }
    }
}

impl KmodVm {
    pub fn new(dev: *mut KmodDev, flags: u32) -> Self {
        KmodVm {
            pgsize_bitmap: unsafe { (*dev).props.pgsize_bitmap },
            flags,
            handle: 0,
            dev,
            sparse_dummy_bo: ptr::null_mut(),
            sparse_dummy_lock: 0,
        }
    }
}
