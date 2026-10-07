// Copyright © 2026 Collabora, Ltd.
// Copyright © 2026 Pix Philosophy (HK) Limited
// SPDX-License-Identifier: MIT

use std::collections::HashMap;
use std::env;
use std::ffi::{c_int, c_void};
use std::fs::File;
use std::io;
use std::os::fd::OwnedFd;
use std::ptr;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering::Relaxed};

use crate::pan_kmod::*;
use crate::uapi::*;
use crate::{LogErr, inout, ioctl, kbase_kmod_ops, mmap};

pub struct Map(pub *mut c_void, usize);

impl Map {
    fn new(
        fd: c_int,
        len: usize,
        prot: c_int,
        offset: u64,
    ) -> io::Result<Self> {
        mmap(fd, len, prot, offset).map(|ptr| Map(ptr, len))
    }
}

impl Drop for Map {
    fn drop(&mut self) {
        unsafe { libc::munmap(self.0, self.1) };
    }
}

pub struct Pending {
    pub addr: u64,
    pub target: u64,
    pub fence: OwnedFd,
}

pub struct Kcpu {
    pub id: Option<u8>,
    pub pending: Option<Pending>,
}

#[repr(C)]
pub struct Dev {
    pub base: KmodDev,
    pub csf: bool,
    pub csif: CsifInfo,
    pub mem_group: u8,
    pub defer_free_phys: bool,
    pub mem_alloc_ex_1_9: AtomicBool,
    pub mem_import_legacy: AtomicBool,
    pub next_handle: AtomicU32,
    pub csf_error: AtomicBool,
    pub kcpu: Mutex<Kcpu>,
    pub export_queue: Option<u8>,
    pub dma_heap: Option<OwnedFd>,
    pub user_reg: Option<Map>,
    _tracking: Map,
}

impl Dev {
    pub unsafe fn get<'a>(dev: *const KmodDev) -> &'a Dev {
        unsafe { &*dev.cast() }
    }

    pub fn fd(&self) -> c_int {
        self.base.fd
    }

    pub fn ioctl<T>(&self, req: u32, arg: T) -> io::Result<c_int> {
        ioctl(self.fd(), req, arg)
    }

    pub fn inout<I: Copy, O: Copy>(&self, req: u32, i: I) -> io::Result<O> {
        inout(self.fd(), req, i)
    }

    pub fn inout_compat<I: Copy, O: Copy>(
        &self,
        legacy: &AtomicBool,
        req: u32,
        legacy_req: u32,
        i: I,
    ) -> io::Result<O> {
        if !legacy.load(Relaxed) {
            match self.inout(req, i) {
                Err(e) if e.raw_os_error() == Some(libc::ENOTTY) => {
                    legacy.store(true, Relaxed)
                }
                ret => return ret,
            }
        }
        self.inout(legacy_req, i)
    }

    pub fn version_at_least(&self, major: u32, minor: u32) -> bool {
        (self.base.driver.major, self.base.driver.minor) >= (major, minor)
    }

    pub fn free_deferred_phys(&self) {
        if self.defer_free_phys {
            self.ioctl(KBASE_IOCTL_DEFER_FREE_PHYS, 0u32)
                .log_err("KBASE_IOCTL_DEFER_FREE_PHYS");
        }
    }

    fn new(
        fd: c_int,
        mut flags: u32,
        allocator: *const Allocator,
    ) -> Option<Dev> {
        let (csf, ver) = version_check(fd)?;
        ioctl(fd, KBASE_IOCTL_SET_FLAGS, 0u32)
            .log_err("KBASE_IOCTL_SET_FLAGS")?;
        let tracking =
            Map::new(fd, 4096, libc::PROT_NONE, BASE_MEM_MAP_TRACKING_HANDLE)
                .log_err("mmap(BASE_MEM_MAP_TRACKING_HANDLE)")?;

        let props = dev_props(&gpuprops(fd)?);
        if props.gpu_id == 0 {
            loge!("failed to determine GPU ID from properties");
            return None;
        }
        let mem_group = mem_group(csf, props.gpu_id)?;

        let jit = MemJitInit {
            va_pages: 1 << 25,
            max_allocations: 255,
            group_id: mem_group,
            phys_pages: 1 << 25,
            ..Default::default()
        };
        if let Err(e) = ioctl(fd, KBASE_IOCTL_MEM_JIT_INIT, jit) {
            logw!(
                "KBASE_IOCTL_MEM_JIT_INIT failed: {e} \
                 (tiler heap creation will not work)"
            );
        }
        if let Err(e) = ioctl(fd, KBASE_IOCTL_MEM_EXEC_INIT, 0x100000u64) {
            logw!(
                "KBASE_IOCTL_MEM_EXEC_INIT failed: {e} \
                 (executable BO allocation will not work)"
            );
        }

        let csif = if csf {
            csif_info(fd)?
        } else {
            CsifInfo::default()
        };
        let user_reg = if csf {
            let page = BASEP_MEM_CSF_USER_REG_PAGE_HANDLE;
            Map::new(fd, 4096, libc::PROT_READ, page)
                .map_err(|e| {
                    logw!("mmap of the CSF USER register page failed: {e}")
                })
                .ok()
        } else {
            None
        };

        let heap = env::var("PANVK_KBASE_DMA_HEAP")
            .ok()
            .filter(|path| !path.is_empty())
            .unwrap_or_else(|| "/dev/dma_heap/system".into());
        let dma_heap = File::open(&heap)
            .map_err(|e| logd!("dma-heap unavailable at {heap}: {e}"))
            .ok();

        if env_is_zero("PANVK_KBASE_USER_CACHE_SYNC") {
            flags |= PAN_KMOD_DEV_FLAG_MMAP_SYNC_THROUGH_KERNEL;
        }

        let kcpu = if csf && !env_is_zero("PANVK_KBASE_KCPU_SYNC") {
            kcpu_queue_create(fd)
        } else {
            None
        };

        let driver = Driver {
            major: ver.major.into(),
            minor: ver.minor.into(),
        };

        Some(Dev {
            base: KmodDev::new(
                fd,
                flags,
                driver,
                props,
                &kbase_kmod_ops,
                allocator,
            ),
            csf,
            csif,
            mem_group,
            defer_free_phys: ioctl(fd, KBASE_IOCTL_DEFER_FREE_PHYS, 0u32)
                .is_ok(),
            mem_alloc_ex_1_9: AtomicBool::new(false),
            mem_import_legacy: AtomicBool::new(false),
            next_handle: AtomicU32::new(2),
            csf_error: AtomicBool::new(false),
            kcpu: Mutex::new(Kcpu {
                id: kcpu,
                pending: None,
            }),
            export_queue: kcpu.and_then(|_| kcpu_queue_create(fd)),
            dma_heap: dma_heap.map(OwnedFd::from),
            user_reg,
            _tracking: tracking,
        })
    }
}

impl Drop for Dev {
    fn drop(&mut self) {
        let kcpu = self.kcpu.get_mut().unwrap();
        kcpu.pending = None;
        for id in kcpu.id.into_iter().chain(self.export_queue) {
            let _ =
                ioctl(self.base.fd, KBASE_IOCTL_KCPU_QUEUE_DELETE, id as u64);
        }
    }
}

fn version_check(fd: c_int) -> Option<(bool, VersionCheck)> {
    let zero = VersionCheck::default();
    if let Ok(ver) =
        inout::<_, VersionCheck>(fd, KBASE_IOCTL_VERSION_CHECK_CSF, zero)
    {
        logd!("CSF driver, uAPI version {}.{}", ver.major, ver.minor);
        return Some((true, ver));
    }
    match inout::<_, VersionCheck>(fd, KBASE_IOCTL_VERSION_CHECK_JM, zero) {
        Ok(ver) if ver.major >= 11 => {
            logd!("JM driver, uAPI version {}.{}", ver.major, ver.minor);
            Some((false, ver))
        }
        Ok(ver) => {
            loge!(
                "legacy JM driver version {}.{} not supported (need >= 11.0)",
                ver.major,
                ver.minor
            );
            None
        }
        Err(e) => {
            logd!("version handshake rejected: {e}");
            None
        }
    }
}

fn gpuprops(fd: c_int) -> Option<HashMap<u32, u64>> {
    let size = ioctl(fd, KBASE_IOCTL_GET_GPUPROPS, GetGpuprops::default())
        .log_err("KBASE_IOCTL_GET_GPUPROPS")?;
    let mut buf = vec![0u8; size as usize];
    let req = GetGpuprops {
        buffer: buf.as_mut_ptr() as u64,
        size: size as u32,
        flags: 0,
    };
    ioctl(fd, KBASE_IOCTL_GET_GPUPROPS, req)
        .log_err("KBASE_IOCTL_GET_GPUPROPS")?;

    let mut props = HashMap::new();
    let mut rest = buf.as_slice();
    while let Some((hdr, tail)) = rest.split_first_chunk::<4>() {
        let hdr = u32::from_le_bytes(*hdr);
        let Some((val, tail)) = tail.split_at_checked(1 << (hdr & 3)) else {
            break;
        };
        let mut bytes = [0; 8];
        bytes[..val.len()].copy_from_slice(val);
        props.entry(hdr >> 2).or_insert(u64::from_le_bytes(bytes));
        rest = tail;
    }
    Some(props)
}

fn dev_props(props: &HashMap<u32, u64>) -> DevProps {
    let get = |key| props.get(&key).copied();
    let prop = |key| get(key).unwrap_or(0) as u32;

    let gpu_id = match get(gpuprop::RAW_GPU_ID).unwrap_or(0) {
        0 => {
            (prop(gpuprop::PRODUCT_ID) as u64) << 16
                | ((prop(gpuprop::MAJOR_REVISION) & 0xf) as u64) << 12
                | ((prop(gpuprop::MINOR_REVISION) & 0xff) as u64) << 4
                | (prop(gpuprop::VERSION_STATUS) & 0xf) as u64
        }
        id if id >> 28 & 0xf == 0xf => id,
        id => id & 0xffff_ffff,
    };

    let thread_features = prop(gpuprop::RAW_THREAD_FEATURES);
    let max_threads = prop(gpuprop::RAW_THREAD_MAX_THREADS);
    let registers = get(gpuprop::MAX_REGISTERS)
        .map_or(thread_features & 0xffff, |v| v as u32);
    let tasks = get(gpuprop::MAX_TASK_QUEUE)
        .map_or(thread_features >> 24, |v| v as u32);
    let tls = prop(gpuprop::TLS_ALLOC);
    let freq = cntfrq();

    DevProps {
        gpu_id,
        gpu_variant: prop(gpuprop::RAW_CORE_FEATURES) & 0xff,
        shader_present: get(gpuprop::RAW_SHADER_PRESENT).unwrap_or(0),
        tiler_features: prop(gpuprop::RAW_TILER_FEATURES),
        mem_features: prop(gpuprop::RAW_MEM_FEATURES),
        mmu_features: prop(gpuprop::RAW_MMU_FEATURES),
        texture_features: [
            gpuprop::RAW_TEXTURE_FEATURES_0,
            gpuprop::RAW_TEXTURE_FEATURES_1,
            gpuprop::RAW_TEXTURE_FEATURES_2,
            gpuprop::RAW_TEXTURE_FEATURES_3,
        ]
        .map(prop),
        l2_features: 0,
        max_threads_per_core: max_threads,
        max_tasks_per_core: tasks.max(1) as u8,
        max_threads_per_wg: prop(gpuprop::RAW_THREAD_MAX_WORKGROUP_SIZE),
        num_threads_active_granularity: prop(
            gpuprop::THREAD_NUM_ACTIVE_GRANULARITY,
        ),
        num_registers_per_core: match (registers, pan_arch(gpu_id)) {
            (0, 4 | 5) => max_threads * 4,
            (0, 6) => max_threads * 64,
            (0, _) => max_threads * 32,
            (registers, _) => registers,
        },
        max_tls_instance_per_core: if tls != 0 { tls } else { max_threads },
        afbc_features: 0,
        gpu_can_query_timestamp: gpu_id != 0,
        timestamp_device_coherent: prop(gpuprop::RAW_COHERENCY_MODE) != 0,
        timestamp_frequency: freq,
        timestamp_cycles_to_ns_factor: if freq != 0 {
            1e9 / freq as f64
        } else {
            0.0
        },
        allowed_group_priorities_mask: PAN_KMOD_GROUP_ALLOW_PRIORITY_MEDIUM
            | PAN_KMOD_GROUP_ALLOW_PRIORITY_LOW,
        supported_bo_flags: PAN_KMOD_BO_FLAG_EXECUTABLE
            | PAN_KMOD_BO_FLAG_ALLOC_ON_FAULT
            | PAN_KMOD_BO_FLAG_NO_MMAP
            | PAN_KMOD_BO_FLAG_WB_MMAP
            | PAN_KMOD_BO_FLAG_GPU_UNCACHED
            | PAN_KMOD_BO_FLAG_CSF_EVENT
            | PAN_KMOD_BO_FLAG_GPU_PRIVATE,
        supported_vm_op_flags: 0,
        is_io_coherent: false,
        pgsize_bitmap: PAN_PGSIZE_4K,
    }
}

fn pan_arch(gpu_id: u64) -> u64 {
    match gpu_id >> 16 & 0xffff {
        0x600 | 0x620 | 0x720 => 4,
        0x750 | 0x820 | 0x830 | 0x860 | 0x880 => 5,
        _ if gpu_id >> 28 & 0xf == 0xf => gpu_id >> 56,
        _ => gpu_id >> 28 & 0xf,
    }
}

#[cfg(target_arch = "aarch64")]
fn cntfrq() -> u64 {
    let freq: u64;
    unsafe { std::arch::asm!("mrs {}, cntfrq_el0", out(reg) freq) };
    freq
}

#[cfg(not(target_arch = "aarch64"))]
fn cntfrq() -> u64 {
    0
}

fn mem_group(csf: bool, gpu_id: u64) -> Option<u8> {
    let Ok(value) = env::var("PANVK_KBASE_MEMORY_GROUP") else {
        let mt6995 = csf && pan_arch(gpu_id) == 15 && is_mt6995();
        return Some(if mt6995 { 6 } else { 0 });
    };
    let group = value.parse().ok().filter(|&group: &u8| group <= 15);
    if group.is_none() {
        loge!("invalid PANVK_KBASE_MEMORY_GROUP: {value}");
    }
    group
}

#[cfg(target_os = "android")]
fn is_mt6995() -> bool {
    let mut value = [0u8; libc::PROP_VALUE_MAX as usize];
    let name = c"ro.board.platform".as_ptr();
    let len =
        unsafe { libc::__system_property_get(name, value.as_mut_ptr().cast()) };
    len > 0 && value.starts_with(b"mt6995\0")
}

#[cfg(not(target_os = "android"))]
fn is_mt6995() -> bool {
    false
}

fn env_is_zero(name: &str) -> bool {
    env::var(name).is_ok_and(|value| value == "0")
}

fn csif_info(fd: c_int) -> Option<CsifInfo> {
    let probe: GlbIfaceOut =
        inout(fd, KBASE_IOCTL_CS_GET_GLB_IFACE, GlbIface::default())
            .log_err("KBASE_IOCTL_CS_GET_GLB_IFACE")?;
    if probe.group_num == 0 || probe.total_stream_num == 0 {
        loge!("GLB interface reports no queue groups/streams");
        return None;
    }

    let mut groups = vec![GroupControl::default(); probe.group_num as usize];
    let mut streams =
        vec![StreamControl::default(); probe.total_stream_num as usize];
    let req = GlbIface {
        max_group_num: probe.group_num,
        max_total_stream_num: probe.total_stream_num,
        groups_ptr: groups.as_mut_ptr() as u64,
        streams_ptr: streams.as_mut_ptr() as u64,
    };
    let glb: GlbIfaceOut = inout(fd, KBASE_IOCTL_CS_GET_GLB_IFACE, req)
        .log_err("KBASE_IOCTL_CS_GET_GLB_IFACE")?;

    let group = groups[0];
    let features = streams[0].features;
    let scoreboards = features >> 8 & 0xff;
    logd!(
        "GLB iface: version {:#x}, features {:#x}, {} groups, {} streams \
         (stream 0: features {features:#x}, group 0: {} streams, \
         suspend size {})",
        glb.glb_version,
        glb.features,
        probe.group_num,
        probe.total_stream_num,
        group.stream_num,
        group.suspend_size
    );

    Some(CsifInfo {
        csg_slot_count: probe.group_num,
        cs_slot_count: group.stream_num,
        cs_reg_count: (features & 0xff) + 1,
        scoreboard_slot_count: if (1..=16).contains(&scoreboards) {
            scoreboards
        } else {
            8
        },
        unpreserved_cs_reg_count: 4,
        pad: 0,
    })
}

fn kcpu_queue_create(fd: c_int) -> Option<u8> {
    match inout::<u64, u64>(fd, KBASE_IOCTL_KCPU_QUEUE_CREATE, 0) {
        Ok(id) => {
            logd!("created KCPU queue {}", id as u8);
            Some(id as u8)
        }
        Err(e) => {
            logd!("KCPU queue unavailable: {e}");
            None
        }
    }
}

pub extern "C" fn dev_create(
    fd: c_int,
    flags: u32,
    _driver: *const Driver,
    allocator: *const Allocator,
) -> *mut KmodDev {
    match Dev::new(fd, flags, allocator) {
        Some(dev) => unsafe { &*allocator }.alloc(dev).cast(),
        None => ptr::null_mut(),
    }
}

pub extern "C" fn dev_destroy(dev: *mut KmodDev) {
    unsafe {
        let mut base = ptr::read(dev);
        (*base.allocator).dealloc(dev.cast::<Dev>());
        base.cleanup();
    }
}

pub extern "C" fn dev_query_user_va_range(dev: *const KmodDev) -> VaRange {
    const RESERVED: u64 = 0x2000000;
    let va_bits = unsafe { (*dev).props.mmu_features } & 0xff;
    let end = if va_bits <= 32 {
        1 << 32
    } else {
        1 << (va_bits - 1)
    };
    VaRange {
        start: RESERVED,
        size: end - RESERVED,
    }
}

pub extern "C" fn query_timestamp(dev: *const KmodDev) -> u64 {
    let req = [BASE_TIMEINFO_TIMESTAMP_FLAG, 0, 0, 0, 0, 0, 0, 0];
    unsafe { Dev::get(dev) }
        .inout::<_, TimeInfo>(KBASE_IOCTL_GET_CPU_GPU_TIMEINFO, req)
        .map_or(0, |info| info.timestamp)
}

#[unsafe(no_mangle)]
pub extern "C" fn kbase_kmod_get_csif_props(
    dev: *const KmodDev,
) -> *const CsifInfo {
    &unsafe { Dev::get(dev) }.csif
}

#[unsafe(no_mangle)]
pub extern "C" fn kbase_kmod_get_flush_id(dev: *const KmodDev) -> u32 {
    let user_reg = unsafe { Dev::get(dev) }.user_reg.as_ref();
    user_reg.map_or(0, |page| unsafe { page.0.cast::<u32>().read_volatile() })
}

#[unsafe(no_mangle)]
pub extern "C" fn kbase_kmod_supports_dmabuf(dev: *const KmodDev) -> bool {
    unsafe { Dev::get(dev) }.dma_heap.is_some()
}
