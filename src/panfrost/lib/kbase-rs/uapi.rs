// Copyright © 2026 Collabora, Ltd.
// Copyright © 2026 Pix Philosophy (HK) Limited
// SPDX-License-Identifier: MIT

#![allow(dead_code)]

use std::mem::size_of;
use std::ptr;

const W: u32 = 1;
const R: u32 = 2;
const RW: u32 = 3;
const KBASE: u8 = 0x80;

const fn ioc(dir: u32, ty: u8, nr: u8, size: usize) -> u32 {
    dir << 30 | (size as u32) << 16 | (ty as u32) << 8 | nr as u32
}

const fn kbase<T>(dir: u32, nr: u8) -> u32 {
    ioc(dir, KBASE, nr, size_of::<T>())
}

pub const KBASE_IOCTL_VERSION_CHECK_JM: u32 = kbase::<VersionCheck>(RW, 0);
pub const KBASE_IOCTL_SET_FLAGS: u32 = kbase::<u32>(W, 1);
pub const KBASE_IOCTL_GET_GPUPROPS: u32 = kbase::<GetGpuprops>(W, 3);
pub const KBASE_IOCTL_MEM_ALLOC: u32 = ioc(RW, KBASE, 5, 32);
pub const KBASE_IOCTL_MEM_FREE: u32 = kbase::<u64>(W, 7);
pub const KBASE_IOCTL_MEM_JIT_INIT: u32 = kbase::<MemJitInit>(W, 14);
pub const KBASE_IOCTL_MEM_SYNC: u32 = kbase::<MemSync>(W, 15);
pub const KBASE_IOCTL_MEM_ALIAS: u32 = kbase::<MemAlias>(RW, 21);
pub const KBASE_IOCTL_MEM_IMPORT: u32 = kbase::<MemImport>(RW, 22);
pub const KBASE_IOCTL_MEM_IMPORT_LEGACY: u32 = ioc(RW, KBASE, 22, 24);
pub const KBASE_IOCTL_CS_QUEUE_REGISTER: u32 = kbase::<CsQueueRegister>(W, 36);
pub const KBASE_IOCTL_CS_QUEUE_KICK: u32 = kbase::<u64>(W, 37);
pub const KBASE_IOCTL_MEM_EXEC_INIT: u32 = kbase::<u64>(W, 38);
pub const KBASE_IOCTL_CS_QUEUE_BIND: u32 = kbase::<CsQueueBind>(RW, 39);
pub const KBASE_IOCTL_CS_QUEUE_TERMINATE: u32 = kbase::<u64>(W, 41);
pub const KBASE_IOCTL_CS_QUEUE_GROUP_CREATE_1_6: u32 = ioc(RW, KBASE, 42, 32);
pub const KBASE_IOCTL_CS_QUEUE_GROUP_TERMINATE: u32 = kbase::<u64>(W, 43);
pub const KBASE_IOCTL_KCPU_QUEUE_CREATE: u32 = kbase::<u64>(R, 45);
pub const KBASE_IOCTL_KCPU_QUEUE_DELETE: u32 = kbase::<u64>(W, 46);
pub const KBASE_IOCTL_KCPU_QUEUE_ENQUEUE: u32 = kbase::<KcpuEnqueue>(W, 47);
pub const KBASE_IOCTL_CS_TILER_HEAP_INIT: u32 = kbase::<TilerHeapInit>(RW, 48);
pub const KBASE_IOCTL_CS_TILER_HEAP_TERM: u32 = kbase::<u64>(W, 49);
pub const KBASE_IOCTL_GET_CPU_GPU_TIMEINFO: u32 = kbase::<TimeInfo>(RW, 50);
pub const KBASE_IOCTL_CS_GET_GLB_IFACE: u32 = kbase::<GlbIface>(RW, 51);
pub const KBASE_IOCTL_VERSION_CHECK_CSF: u32 = kbase::<VersionCheck>(RW, 52);
pub const KBASE_IOCTL_CS_QUEUE_GROUP_CREATE_1_18: u32 = ioc(RW, KBASE, 58, 40);
pub const KBASE_IOCTL_CS_QUEUE_GROUP_CREATE_1_19: u32 = ioc(RW, KBASE, 58, 112);
pub const KBASE_IOCTL_MEM_ALLOC_EX: u32 = kbase::<MemAlloc>(RW, 59);
pub const KBASE_IOCTL_MEM_ALLOC_EX_1_9: u32 = ioc(RW, KBASE, 59, 64);
pub const KBASE_IOCTL_CS_QUEUE_GROUP_CREATE: u32 = kbase::<GroupCreate>(RW, 63);
pub const KBASE_IOCTL_DEFER_FREE_PHYS: u32 = kbase::<u32>(W, 81);
pub const DMA_HEAP_IOCTL_ALLOC: u32 =
    ioc(RW, b'H', 0, size_of::<DmaHeapAlloc>());

pub const BASE_MEM_PROT_CPU_RD: u64 = 1 << 0;
pub const BASE_MEM_PROT_CPU_WR: u64 = 1 << 1;
pub const BASE_MEM_PROT_GPU_RD: u64 = 1 << 2;
pub const BASE_MEM_PROT_GPU_WR: u64 = 1 << 3;
pub const BASE_MEM_PROT_GPU_EX: u64 = 1 << 4;
pub const BASE_MEM_GROW_ON_GPF: u64 = 1 << 9;
pub const BASE_MEM_COHERENT_SYSTEM: u64 = 1 << 10;
pub const BASE_MEM_COHERENT_LOCAL: u64 = 1 << 11;
pub const BASE_MEM_CACHED_CPU: u64 = 1 << 12;
pub const BASE_MEM_SAME_VA: u64 = 1 << 13;
pub const BASE_MEM_NEED_MMAP: u64 = 1 << 14;
pub const BASE_MEM_IMPORT_SHARED: u64 = 1 << 18;
pub const BASE_MEM_CSF_EVENT: u64 = 1 << 19;
pub const BASE_MEM_UNCACHED_GPU: u64 = 1 << 21;
pub const BASEP_MEM_GROUP_ID_SHIFT: u32 = 22;

pub const BASE_MEM_MAP_TRACKING_HANDLE: u64 = 3 << 12;
pub const BASEP_MEM_CSF_USER_REG_PAGE_HANDLE: u64 = 47 << 12;
pub const BASEP_QUEUE_NR_MMAP_USER_PAGES: usize = 3;
pub const BASE_MEM_IMPORT_TYPE_UMM: u32 = 2;
pub const BASE_SYNCSET_OP_MSYNC: u8 = 1;
pub const BASE_SYNCSET_OP_CSYNC: u8 = 2;
pub const BASE_CSF_TILER_OOM_EXCEPTION_FLAG: u8 = 1 << 0;
pub const BASE_TIMEINFO_TIMESTAMP_FLAG: u32 = 1 << 1;

pub const BASE_KCPU_COMMAND_TYPE_FENCE_SIGNAL: u8 = 0;
pub const BASE_KCPU_COMMAND_TYPE_CQS_WAIT_OPERATION: u8 = 4;
pub const BASEP_CQS_WAIT_OPERATION_GT: u8 = 1;
pub const BASEP_CQS_DATA_TYPE_U64: u8 = 1;
pub const KBASE_KCPU_CQS_MAX_OBJS: usize = 32;

pub const BASE_CSF_NOTIFICATION_EVENT: u8 = 0;
pub const BASE_CSF_NOTIFICATION_GPU_QUEUE_GROUP_ERROR: u8 = 1;
pub const BASE_CSF_NOTIFICATION_CPU_QUEUE_DUMP: u8 = 2;

pub const BASE_GPU_QUEUE_GROUP_ERROR_FATAL: u8 = 0;
pub const BASE_GPU_QUEUE_GROUP_QUEUE_ERROR_FATAL: u8 = 1;
pub const BASE_GPU_QUEUE_GROUP_ERROR_TIMEOUT: u8 = 2;
pub const BASE_GPU_QUEUE_GROUP_ERROR_TILER_HEAP_OOM: u8 = 3;
pub const BASE_GPU_QUEUE_GROUP_QUEUE_ERROR_FAULT: u8 = 4;

pub mod gpuprop {
    pub const PRODUCT_ID: u32 = 1;
    pub const VERSION_STATUS: u32 = 2;
    pub const MINOR_REVISION: u32 = 3;
    pub const MAJOR_REVISION: u32 = 4;
    pub const MAX_REGISTERS: u32 = 21;
    pub const MAX_TASK_QUEUE: u32 = 22;
    pub const RAW_SHADER_PRESENT: u32 = 25;
    pub const RAW_CORE_FEATURES: u32 = 30;
    pub const RAW_MEM_FEATURES: u32 = 31;
    pub const RAW_MMU_FEATURES: u32 = 32;
    pub const RAW_TILER_FEATURES: u32 = 51;
    pub const RAW_TEXTURE_FEATURES_0: u32 = 52;
    pub const RAW_TEXTURE_FEATURES_1: u32 = 53;
    pub const RAW_TEXTURE_FEATURES_2: u32 = 54;
    pub const RAW_GPU_ID: u32 = 55;
    pub const RAW_THREAD_MAX_THREADS: u32 = 56;
    pub const RAW_THREAD_MAX_WORKGROUP_SIZE: u32 = 57;
    pub const RAW_THREAD_FEATURES: u32 = 59;
    pub const RAW_COHERENCY_MODE: u32 = 60;
    pub const RAW_TEXTURE_FEATURES_3: u32 = 81;
    pub const TLS_ALLOC: u32 = 84;
    pub const THREAD_NUM_ACTIVE_GRANULARITY: u32 = 88;
}

#[repr(C)]
#[derive(Clone, Copy)]
pub union InOut<I: Copy, O: Copy> {
    pub i: I,
    pub o: O,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct VersionCheck {
    pub major: u16,
    pub minor: u16,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct GetGpuprops {
    pub buffer: u64,
    pub size: u32,
    pub flags: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct MemAlloc {
    pub va_pages: u64,
    pub commit_pages: u64,
    pub extension: u64,
    pub flags: u64,
    pub fixed_address: u64,
    pub extra: [u64; 5],
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct MemOut {
    pub flags: u64,
    pub gpu_va: u64,
    pub va_pages: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct MemJitInit {
    pub va_pages: u64,
    pub max_allocations: u8,
    pub trim_level: u8,
    pub group_id: u8,
    pub padding: [u8; 5],
    pub phys_pages: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct MemSync {
    pub handle: u64,
    pub user_addr: u64,
    pub size: u64,
    pub ty: u8,
    pub padding: [u8; 7],
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct AliasingInfo {
    pub handle: u64,
    pub offset: u64,
    pub length: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct MemAlias {
    pub flags: u64,
    pub stride: u64,
    pub nents: u64,
    pub aliasing_info: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct MemImport {
    pub flags: u64,
    pub phandle: u64,
    pub ty: u32,
    pub padding: u32,
    pub extra: [u64; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct CsQueueRegister {
    pub buffer_gpu_addr: u64,
    pub buffer_size: u32,
    pub priority: u8,
    pub padding: [u8; 3],
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct CsQueueBind {
    pub buffer_gpu_addr: u64,
    pub group_handle: u8,
    pub csi_index: u8,
    pub padding: [u8; 6],
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct GroupCreate {
    pub tiler_mask: u64,
    pub fragment_mask: u64,
    pub compute_mask: u64,
    pub cs_min: u8,
    pub priority: u8,
    pub tiler_max: u8,
    pub fragment_max: u8,
    pub compute_max: u8,
    pub csi_handlers: u8,
    pub neural_max: u8,
    pub cs_fault_report_enable: u8,
    pub dvs_buf: u64,
    pub neural_mask: u64,
    pub comp_pri_threshold: u8,
    pub comp_pri_ratio: u8,
    pub padding: [u8; 6],
    pub reserved: [u64; 8],
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct GroupCreateOutput {
    pub handle: u8,
    pub padding: [u8; 3],
    pub uid: u32,
}

const _: () = {
    assert!(std::mem::size_of::<GroupCreateOutput>() == 8);
    assert!(std::mem::offset_of!(GroupCreateOutput, uid) == 4);
};

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct KcpuCommand {
    pub ty: u8,
    pub padding: [u8; 7],
    pub info: [u64; 2],
}

const _: () = {
    assert!(std::mem::size_of::<KcpuCommand>() == 24);
    assert!(std::mem::offset_of!(KcpuCommand, info) == 8);
};

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct CqsWaitOperation {
    pub addr: u64,
    pub val: u64,
    pub operation: u8,
    pub data_type: u8,
    pub padding: [u8; 6],
}

#[repr(C)]
pub struct Fence {
    pub fd: i32,
    pub stream_fd: i32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct KcpuEnqueue {
    pub addr: u64,
    pub nr_commands: u32,
    pub id: u8,
    pub padding: [u8; 3],
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct TilerHeapInit {
    pub chunk_size: u32,
    pub initial_chunks: u32,
    pub max_chunks: u32,
    pub target_in_flight: u16,
    pub group_id: u8,
    pub padding: u8,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct GlbIface {
    pub max_group_num: u32,
    pub max_total_stream_num: u32,
    pub groups_ptr: u64,
    pub streams_ptr: u64,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct GlbIfaceOut {
    pub glb_version: u32,
    pub features: u32,
    pub group_num: u32,
    pub prfcnt_size: u32,
    pub total_stream_num: u32,
    pub instr_features: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct StreamControl {
    pub features: u32,
    pub padding: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct GroupControl {
    pub features: u32,
    pub stream_num: u32,
    pub suspend_size: u32,
    pub padding: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct TimeInfo {
    pub sec: u64,
    pub nsec: u32,
    pub padding: u32,
    pub timestamp: u64,
    pub cycle_counter: u64,
}

#[repr(C)]
#[derive(Default)]
pub struct CsfNotification {
    pub ty: u8,
    pub padding0: [u8; 7],
    pub group_handle: u8,
    pub padding1: [u8; 7],
    pub error_type: u8,
    pub padding2: [u8; 7],
    pub sideband: u64,
    pub status: u32,
    pub csi_index: u8,
    pub padding3: [u8; 6],
    pub has_extra: u8,
    pub trace_id0: u32,
    pub trace_id1: u32,
    pub trace_task: u32,
    pub padding4: [u8; 8],
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct DmaHeapAlloc {
    pub len: u64,
    pub fd: u32,
    pub fd_flags: u32,
    pub heap_flags: u64,
}

const _: () = assert!(
    size_of::<MemAlloc>() == 80
        && size_of::<MemImport>() == 40
        && size_of::<GroupCreate>() == 120
        && size_of::<KcpuCommand>() == 24
        && size_of::<CqsWaitOperation>() == 24
        && size_of::<CsfNotification>() == 64
);

impl KcpuCommand {
    pub fn cqs_wait(objs: &[CqsWaitOperation]) -> Self {
        KcpuCommand {
            ty: BASE_KCPU_COMMAND_TYPE_CQS_WAIT_OPERATION,
            info: [objs.as_ptr() as u64, objs.len() as u64],
            ..Default::default()
        }
    }

    pub fn fence_signal(fence: &mut Fence) -> Self {
        KcpuCommand {
            ty: BASE_KCPU_COMMAND_TYPE_FENCE_SIGNAL,
            info: [ptr::from_mut(fence) as u64, 0],
            ..Default::default()
        }
    }
}

impl CqsWaitOperation {
    pub fn gt_u64(addr: u64, val: u64) -> Self {
        CqsWaitOperation {
            addr,
            val,
            operation: BASEP_CQS_WAIT_OPERATION_GT,
            data_type: BASEP_CQS_DATA_TYPE_U64,
            ..Default::default()
        }
    }
}
