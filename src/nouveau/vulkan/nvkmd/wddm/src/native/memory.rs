use super::{device::DeviceInner, sync::PagingValue, Device, FencePoint};
use crate::contract::{ByteRange, MemoryRequest};
use crate::error::{Error, ErrorKind, Result};
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::{
    atomic::{AtomicU64, AtomicU8, Ordering},
    Arc, OnceLock, Weak,
};
use windows::Wdk::Graphics::Direct3D::*;
use windows::Win32::System::Memory::{
    VirtualAlloc, VirtualFree, MEM_COMMIT, MEM_RELEASE, MEM_RESERVE,
    PAGE_READWRITE,
};

struct AllocationDump {
    path: PathBuf,
    size: u64,
}

impl AllocationDump {
    fn begin(size: u64, private: &[u8]) -> Option<Self> {
        if size != 65536 || private.len() != 586 {
            return None;
        }
        static DIRECTORY: OnceLock<Option<PathBuf>> = OnceLock::new();
        static SEQUENCE: AtomicU64 = AtomicU64::new(0);
        let directory = DIRECTORY
            .get_or_init(|| {
                std::env::var_os("NVK_WDDM_MEMORY_DUMP").map(PathBuf::from)
            })
            .as_ref()?;
        std::fs::create_dir_all(directory).ok()?;
        let sequence = SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let dump = Self {
            path: directory
                .join(format!("local-{}-{sequence}", std::process::id())),
            size,
        };
        std::fs::write(dump.path.with_extension("request.bin"), private)
            .ok()?;
        Some(dump)
    }

    fn finish(&self, private: &[u8], status: i32) {
        let _ = std::fs::write(self.path.with_extension("reply.bin"), private);
        let word = |offset: usize| {
            u32::from_le_bytes(private[offset..offset + 4].try_into().unwrap())
        };
        let metadata = format!(
            "{{\"repository_policy_marker\":\"biblioklept\",\"ntstatus\":{status},\"ntstatus_hex\":\"0x{:08x}\",\"size\":{},\"kind\":{},\"read_heaps\":{},\"write_heaps\":{},\"preferred_heap\":{}}}",
            status as u32, self.size, word(430), word(114), word(118), word(122),
        );
        let _ = std::fs::write(self.path.with_extension("json"), metadata);
    }
}

struct HostPages {
    address: usize,
    size: usize,
}

impl HostPages {
    fn new(size: u64) -> Result<Self> {
        let size: usize = size.try_into().map_err(|_| {
            Error::new(ErrorKind::HostMemory, "host allocation size")
        })?;
        let address = unsafe {
            VirtualAlloc(None, size, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE)
        } as usize;
        if address == 0 {
            return Err(Error::new(ErrorKind::HostMemory, "VirtualAlloc"));
        }
        Ok(Self { address, size })
    }
}

impl Drop for HostPages {
    fn drop(&mut self) {
        unsafe {
            let _ = VirtualFree(self.address as *mut _, 0, MEM_RELEASE);
        }
    }
}

pub(crate) struct MemoryInner {
    pub device: Arc<DeviceInner>,
    pub handle: u32,
    resource: u32,
    pub size: u64,
    host: Option<HostPages>,
    residency: AtomicU8,
}

impl Drop for MemoryInner {
    fn drop(&mut self) {
        let request = D3DKMT_DESTROYALLOCATION2 {
            hDevice: self.device.handle,
            hResource: self.resource,
            phAllocationList: if self.resource == 0 {
                &self.handle
            } else {
                std::ptr::null()
            },
            AllocationCount: u32::from(self.resource == 0),
            ..Default::default()
        };
        let status = unsafe { D3DKMTDestroyAllocation2(&request) }.0;
        if status < 0 && !self.device.lost.load(Ordering::Acquire) {
            if let Some(host) = self.host.take() {
                std::mem::forget(host);
            }
        }
    }
}

#[derive(Clone)]
pub struct Memory {
    pub(crate) inner: Arc<MemoryInner>,
}

impl Memory {
    fn nvidia_buffer_private(
        device: &Device,
        size: u64,
        local: bool,
    ) -> Result<crate::abi::PrivateData> {
        ByteRange::new(0, size, size, 4096)?;
        let identity = device.adapter().identity();
        if identity.device_id != 0x2c58 || size > u64::from(u32::MAX) {
            return Err(Error::new(
                ErrorKind::Unsupported,
                "NVIDIA buffer profile",
            ));
        }
        let unique_id = device.nvidia_unique_id()?;
        let mut private = crate::abi::PrivateData::zeroed(586)?;
        crate::nvidia::PrivateHeader {
            version: 0x10004,
            size: 586,
            caller: crate::nvidia::PrivateHeader::NVGL,
        }
        .write(&mut private)?;
        private.write_u64(
            16,
            if local {
                0x0004000002200800
            } else {
                0x0004000403000000
            },
        )?;
        let heap = if local { 1 } else { 64 };
        for (offset, value) in [
            (48, size as u32),
            (52, 1),
            (106, 1),
            (110, 1),
            (114, heap),
            (118, heap),
            (122, heap),
            (190, if local { 0x02008000 } else { 0x02000008 }),
            (194, size as u32),
            (198, 1),
            (238, 1),
            (306, 1),
            (310, 1),
            (314, 1),
        ] {
            private.write_u32(offset, value)?;
        }
        for (offset, value) in [
            (162, if local { 0 } else { u64::MAX }),
            (206, size),
            (246, u64::MAX),
            (258, unique_id),
            (278, size),
        ] {
            private.write_u64(offset, value)?;
        }
        if local {
            private.as_bytes_mut()[146..148]
                .copy_from_slice(&1u16.to_le_bytes());
        }
        Ok(private)
    }

    pub fn nvidia_host(device: &Device, size: u64) -> Result<(Self, u32)> {
        let mut private = Self::nvidia_buffer_private(device, size, false)?;
        let memory = Self::presentation_host(device, size, &mut private)?;
        let kind = u32::from_le_bytes(
            private.as_bytes()[430..434].try_into().map_err(|_| {
                Error::new(ErrorKind::Invalid, "NVIDIA buffer kind")
            })?,
        );
        if kind != 6 {
            return Err(Error::new(
                ErrorKind::Unsupported,
                "NVIDIA generic-memory allocation reply",
            ));
        }
        Ok((memory, kind))
    }

    pub fn nvidia_local(device: &Device, size: u64) -> Result<(Self, u32)> {
        let mut private = Self::nvidia_buffer_private(device, size, true)?;
        let memory =
            Self::private_allocation(device, size, &mut private, None)?;
        let word = |offset: usize| {
            u32::from_le_bytes(
                private.as_bytes()[offset..offset + 4].try_into().unwrap(),
            )
        };
        let allocated_size = u64::from_le_bytes(
            private.as_bytes()[206..214].try_into().unwrap(),
        );
        let kind = word(430);
        if kind != 6
            || word(114) != 1
            || word(118) != 1
            || word(122) != 1
            || allocated_size < size
        {
            return Err(Error::new(
                ErrorKind::Unsupported,
                "NVIDIA local-memory allocation reply",
            ));
        }
        Ok((memory, kind))
    }

    pub(crate) fn presentation_host(
        device: &Device,
        size: u64,
        private: &mut crate::abi::PrivateData,
    ) -> Result<Self> {
        device.0.ready()?;
        ByteRange::new(0, size, size, 4096)?;
        let host = HostPages::new(size)?;
        Self::private_allocation(device, size, private, Some(host))
    }

    fn private_allocation(
        device: &Device,
        size: u64,
        private: &mut crate::abi::PrivateData,
        host: Option<HostPages>,
    ) -> Result<Self> {
        device.0.ready()?;
        ByteRange::new(0, size, size, 4096)?;
        let mut allocation = D3DDDI_ALLOCATIONINFO2 {
            pPrivateDriverData: private.as_bytes_mut().as_mut_ptr().cast(),
            PrivateDriverDataSize: private.len_u32(),
            VidPnSourceId: u32::from_le_bytes(
                private.as_bytes()[72..76].try_into().map_err(|_| {
                    Error::new(ErrorKind::Invalid, "presentation source ID")
                })?,
            ),
            ..Default::default()
        };
        allocation.Anonymous1.pSystemMem = host
            .as_ref()
            .map_or(std::ptr::null(), |pages| pages.address as *const _);
        let mut request = D3DKMT_CREATEALLOCATION {
            hDevice: device.0.handle,
            NumAllocations: 1,
            ..Default::default()
        };
        request.Anonymous2.pAllocationInfo2 = &mut allocation;
        request.Flags._bitfield = 1;
        let dump = if host.is_none() {
            AllocationDump::begin(size, private.as_bytes())
        } else {
            None
        };
        let status = unsafe { D3DKMTCreateAllocation2(&mut request) }.0;
        if let Some(dump) = dump {
            dump.finish(private.as_bytes(), status);
        }
        device
            .0
            .check("D3DKMTCreateAllocation2 private memory", status)?;
        Ok(Self {
            inner: Arc::new(MemoryInner {
                device: device.0.clone(),
                handle: allocation.hAllocation,
                resource: request.hResource,
                size,
                host,
                residency: AtomicU8::new(0),
            }),
        })
    }

    pub fn host(device: &Device, size: u64) -> Result<Self> {
        device.0.ready()?;
        ByteRange::new(0, size, size, 4096)?;
        let host = HostPages::new(size)?;
        let mut allocation = D3DDDI_ALLOCATIONINFO2::default();
        allocation.Anonymous1.pSystemMem = host.address as *const _;
        let mut standard = D3DKMT_CREATESTANDARDALLOCATION {
            Type: D3DKMT_STANDARDALLOCATIONTYPE_EXISTINGHEAP,
            ..Default::default()
        };
        standard.Anonymous.ExistingHeapData.Size = host.size;
        let mut request = D3DKMT_CREATEALLOCATION {
            hDevice: device.0.handle,
            NumAllocations: 1,
            ..Default::default()
        };
        request.Anonymous1.pStandardAllocation = &mut standard;
        request.Anonymous2.pAllocationInfo2 = &mut allocation;
        request.Flags._bitfield = 0x0001_0863;
        device.0.check(
            "D3DKMTCreateAllocation2 standard heap",
            unsafe { D3DKMTCreateAllocation2(&mut request) }.0,
        )?;
        Ok(Self {
            inner: Arc::new(MemoryInner {
                device: device.0.clone(),
                handle: allocation.hAllocation,
                resource: request.hResource,
                size,
                host: Some(host),
                residency: AtomicU8::new(0),
            }),
        })
    }

    pub fn driver(device: &Device, request: MemoryRequest) -> Result<Self> {
        device.0.ready()?;
        let request = request.validate()?;
        let abi = device.0.abi.as_ref().ok_or_else(|| {
            Error::new(ErrorKind::Unsupported, "unverified allocation ABI")
        })?;
        if (request.pte_kind != 0 || request.tile_mode != 0)
            && !abi.capabilities().tiled_memory
        {
            return Err(Error::new(ErrorKind::Unsupported, "tiled allocation"));
        }
        let mut payload = abi.allocation(request)?;
        let mut allocation = D3DDDI_ALLOCATIONINFO2 {
            PrivateDriverDataSize: payload.allocation.len_u32(),
            pPrivateDriverData: payload
                .allocation
                .as_bytes_mut()
                .as_mut_ptr()
                .cast(),
            ..Default::default()
        };
        let mut native = D3DKMT_CREATEALLOCATION {
            hDevice: device.0.handle,
            NumAllocations: 1,
            PrivateDriverDataSize: payload.resource.len_u32(),
            ..Default::default()
        };
        native.Anonymous1.pPrivateDriverData =
            payload.resource.as_bytes().as_ptr().cast();
        native.Anonymous2.pAllocationInfo2 = &mut allocation;
        native.Flags._bitfield = payload.create_flags;
        device.0.check(
            "D3DKMTCreateAllocation2 NVIDIA",
            unsafe { D3DKMTCreateAllocation2(&mut native) }.0,
        )?;
        Ok(Self {
            inner: Arc::new(MemoryInner {
                device: device.0.clone(),
                handle: allocation.hAllocation,
                resource: native.hResource,
                size: request.size,
                host: None,
                residency: AtomicU8::new(0),
            }),
        })
    }

    pub fn size(&self) -> u64 {
        self.inner.size
    }

    pub unsafe fn map(&self) -> Result<*mut u8> {
        self.inner.device.ready()?;
        self.inner
            .host
            .as_ref()
            .map(|host| host.address as *mut u8)
            .ok_or_else(|| {
                Error::new(ErrorKind::Unsupported, "host mapping required")
            })
    }

    pub fn make_resident(&self) -> Result<FencePoint> {
        self.inner.device.ready()?;
        match self.inner.residency.compare_exchange(
            0,
            1,
            Ordering::AcqRel,
            Ordering::Acquire,
        ) {
            Ok(_) => {}
            Err(2) => {
                return Ok(FencePoint::paging(
                    self.inner.device.clone(),
                    self.inner.device.paging_value(),
                ))
            }
            Err(_) => {
                return Err(Error::new(ErrorKind::Busy, "residency transition"))
            }
        }
        let mut request = D3DDDI_MAKERESIDENT {
            hPagingQueue: self.inner.device.paging_queue,
            NumAllocations: 1,
            AllocationList: &self.inner.handle,
            ..Default::default()
        };
        let status = unsafe { D3DKMTMakeResident(&mut request) }.0;
        if let Err(error) =
            self.inner.device.check("D3DKMTMakeResident", status)
        {
            self.inner.residency.store(0, Ordering::Release);
            return Err(error);
        }
        PagingValue::from_status(status, &request.PagingFenceValue)
            .publish(&self.inner.device.last_paging);
        self.inner.residency.store(2, Ordering::Release);
        Ok(FencePoint::paging(
            self.inner.device.clone(),
            self.inner.device.paging_value(),
        ))
    }

    pub fn evict(&mut self) -> Result<u64> {
        self.inner.device.ready()?;
        if Arc::strong_count(&self.inner) != 1 {
            return Err(Error::new(ErrorKind::Busy, "allocation is retained"));
        }
        if self.inner.residency.load(Ordering::Acquire) == 0 {
            return Ok(0);
        }
        self.make_resident()?.wait(u64::MAX)?;
        let mut request = D3DKMT_EVICT {
            hDevice: self.inner.device.handle,
            NumAllocations: 1,
            AllocationList: &self.inner.handle,
            ..Default::default()
        };
        self.inner
            .device
            .check("D3DKMTEvict", unsafe { D3DKMTEvict(&mut request) }.0)?;
        self.inner.residency.store(0, Ordering::Release);
        Ok(request.NumBytesToTrim)
    }

    pub fn write(&mut self, offset: u64, bytes: &[u8]) -> Result<()> {
        if Arc::strong_count(&self.inner) != 1 {
            return Err(Error::new(
                ErrorKind::Busy,
                "host write with outstanding references",
            ));
        }
        let range =
            ByteRange::new(offset, bytes.len() as u64, self.inner.size, 1)?;
        let host = self.inner.host.as_ref().ok_or_else(|| {
            Error::new(ErrorKind::Unsupported, "standard host mapping required")
        })?;
        unsafe {
            std::ptr::copy_nonoverlapping(
                bytes.as_ptr(),
                (host.address + range.offset as usize) as *mut u8,
                bytes.len(),
            );
        }
        Ok(())
    }

    pub fn read(&mut self, offset: u64, bytes: &mut [u8]) -> Result<()> {
        if Arc::strong_count(&self.inner) != 1 {
            return Err(Error::new(
                ErrorKind::Busy,
                "host read with outstanding references",
            ));
        }
        let range =
            ByteRange::new(offset, bytes.len() as u64, self.inner.size, 1)?;
        let host = self.inner.host.as_ref().ok_or_else(|| {
            Error::new(ErrorKind::Unsupported, "standard host mapping required")
        })?;
        unsafe {
            std::ptr::copy_nonoverlapping(
                (host.address + range.offset as usize) as *const u8,
                bytes.as_mut_ptr(),
                bytes.len(),
            );
        }
        Ok(())
    }
}

struct VaInner {
    device: Arc<DeviceInner>,
    reserved_address: u64,
    reserved_size: u64,
    address: u64,
    size: u64,
}

impl Drop for VaInner {
    fn drop(&mut self) {
        unsafe {
            let _ =
                D3DKMTFreeGpuVirtualAddress(&D3DKMT_FREEGPUVIRTUALADDRESS {
                    hAdapter: self.device.adapter.handle(),
                    BaseAddress: self.reserved_address,
                    Size: self.reserved_size,
                });
        }
    }
}

pub struct VirtualAddress {
    inner: Arc<VaInner>,
    bindings: BTreeMap<u64, Weak<BindingInner>>,
}

impl VirtualAddress {
    pub fn reserve(
        device: &Device,
        size: u64,
        alignment: u64,
        fixed: Option<u64>,
    ) -> Result<Self> {
        device.0.ready()?;
        ByteRange::new(0, size, size, 4096)?;
        if !alignment.is_power_of_two()
            || fixed.is_some_and(|address| {
                address == 0 || address % alignment.max(65536) != 0
            })
        {
            return Err(Error::new(ErrorKind::Invalid, "GPUVA alignment"));
        }
        let alignment = alignment.max(65536);
        let rounded_size =
            size.checked_add(65535).map(|n| n & !65535).ok_or_else(|| {
                Error::new(ErrorKind::Invalid, "GPUVA rounded size overflow")
            })?;
        let reserved_size = if fixed.is_some() {
            rounded_size
        } else {
            rounded_size.checked_add(alignment - 65536).ok_or_else(|| {
                Error::new(ErrorKind::Invalid, "GPUVA size overflow")
            })?
        };
        let mut request = D3DDDI_RESERVEGPUVIRTUALADDRESS {
            BaseAddress: fixed.unwrap_or(0),
            Size: reserved_size,
            ..Default::default()
        };
        request.Anonymous1.hAdapter = device.0.adapter.handle();
        device.0.check(
            "D3DKMTReserveGpuVirtualAddress",
            unsafe { D3DKMTReserveGpuVirtualAddress(&mut request) }.0,
        )?;
        let base = request.VirtualAddress;
        let address = base
            .checked_add(alignment - 1)
            .map(|x| x & !(alignment - 1));
        let inner = Arc::new(VaInner {
            device: device.0.clone(),
            reserved_address: base,
            reserved_size,
            address: address.unwrap_or(0),
            size,
        });
        if base == 0
            || address
                .filter(|a| {
                    a.checked_add(size).is_some_and(|end| {
                        end <= base.saturating_add(reserved_size)
                    })
                })
                .is_none()
            || fixed.is_some_and(|a| a != base)
        {
            return Err(Error::new(
                ErrorKind::Invalid,
                "GPUVA reservation result",
            ));
        }
        Ok(Self {
            inner,
            bindings: BTreeMap::new(),
        })
    }

    pub fn address(&self) -> u64 {
        self.inner.address
    }
    pub fn size(&self) -> u64 {
        self.inner.size
    }

    pub fn bind(
        &mut self,
        memory: &Memory,
        va_offset: u64,
        memory_offset: u64,
        size: u64,
    ) -> Result<MemoryBinding> {
        self.bind_access(memory, va_offset, memory_offset, size, false)
    }

    pub fn bind_executable(
        &mut self,
        memory: &Memory,
        va_offset: u64,
        memory_offset: u64,
        size: u64,
    ) -> Result<MemoryBinding> {
        self.bind_access(memory, va_offset, memory_offset, size, true)
    }

    fn bind_access(
        &mut self,
        memory: &Memory,
        va_offset: u64,
        memory_offset: u64,
        size: u64,
        execute: bool,
    ) -> Result<MemoryBinding> {
        self.inner.device.ready()?;
        if !Arc::ptr_eq(&self.inner.device, &memory.inner.device) {
            return Err(Error::new(
                ErrorKind::Invalid,
                "cross-device GPUVA binding",
            ));
        }
        let range = ByteRange::new(va_offset, size, self.inner.size, 4096)?;
        ByteRange::new(memory_offset, size, memory.size(), 4096)?;
        self.bindings
            .retain(|_, binding| binding.strong_count() != 0);
        for (&start, binding) in &self.bindings {
            if let Some(binding) = binding.upgrade() {
                if range.offset < start + binding.size && range.end() > start {
                    return Err(Error::new(
                        ErrorKind::Busy,
                        "overlapping live GPUVA binding",
                    ));
                }
            }
        }
        let address = self.inner.address + va_offset;
        let mut request = D3DDDI_MAPGPUVIRTUALADDRESS {
            hPagingQueue: self.inner.device.paging_queue,
            hAllocation: memory.inner.handle,
            BaseAddress: address,
            OffsetInPages: memory_offset / 4096,
            SizeInPages: size / 4096,
            ..Default::default()
        };
        request.Protection.Anonymous.Value = if execute { 3 } else { 1 };
        let status = unsafe { D3DKMTMapGpuVirtualAddress(&mut request) }.0;
        self.inner
            .device
            .check("D3DKMTMapGpuVirtualAddress", status)?;
        PagingValue::from_status(status, &request.PagingFenceValue)
            .publish(&self.inner.device.last_paging);
        let inner = Arc::new(BindingInner {
            va: self.inner.clone(),
            memory: memory.clone(),
            address,
            size,
        });
        if request.VirtualAddress != address {
            return Err(Error::new(ErrorKind::Invalid, "GPUVA binding result"));
        }
        self.bindings.insert(va_offset, Arc::downgrade(&inner));
        Ok(MemoryBinding(inner))
    }
}

struct BindingInner {
    va: Arc<VaInner>,
    memory: Memory,
    address: u64,
    size: u64,
}

impl Drop for BindingInner {
    fn drop(&mut self) {
        if self.va.device.lost.load(Ordering::Acquire) {
            return;
        }
        let mut request = D3DDDI_MAPGPUVIRTUALADDRESS {
            hPagingQueue: self.va.device.paging_queue,
            BaseAddress: self.address,
            SizeInPages: self.size / 4096,
            ..Default::default()
        };
        request.Protection.Anonymous.Value = 8;
        let status = unsafe { D3DKMTMapGpuVirtualAddress(&mut request) }.0;
        let result = self
            .va
            .device
            .check("D3DKMTMapGpuVirtualAddress unbind", status)
            .and_then(|_| {
                PagingValue::from_status(status, &request.PagingFenceValue)
                    .publish(&self.va.device.last_paging);
                FencePoint::paging(
                    self.va.device.clone(),
                    self.va.device.paging_value(),
                )
                .wait(5_000_000_000)
            });
        if result.is_err() {
            std::mem::forget(self.memory.clone());
            std::mem::forget(self.va.clone());
        }
    }
}

#[derive(Clone)]
pub struct MemoryBinding(Arc<BindingInner>);

impl MemoryBinding {
    pub fn address(&self) -> u64 {
        self.0.address
    }
    pub fn size(&self) -> u64 {
        self.0.size
    }
    pub fn paging_point(&self) -> FencePoint {
        FencePoint::paging(
            self.0.va.device.clone(),
            self.0.va.device.paging_value(),
        )
    }
    pub fn memory(&self) -> &Memory {
        &self.0.memory
    }
    pub(crate) fn device(&self) -> &Arc<DeviceInner> {
        &self.0.va.device
    }
    pub(crate) fn contains(&self, address: u64, size: u64) -> bool {
        address >= self.address()
            && address
                .checked_add(size)
                .is_some_and(|end| end <= self.address() + self.size())
    }
}
