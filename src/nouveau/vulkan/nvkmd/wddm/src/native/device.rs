use super::channel::HardwareContext;
use super::sync::PagingValue;
use super::{Adapter, FencePoint};
use crate::abi::NvidiaAbi;
use crate::error::{Error, ErrorKind, Result};
use std::sync::{
    atomic::{AtomicBool, AtomicU64, Ordering},
    Arc, Mutex, OnceLock, Weak,
};
use std::time::Instant;
use windows::Wdk::Graphics::Direct3D::*;

pub(crate) struct DeviceInner {
    pub adapter: Adapter,
    pub handle: u32,
    pub paging_queue: u32,
    pub paging_sync: u32,
    pub paging_cpu: usize,
    pub lost: AtomicBool,
    pub last_paging: AtomicU64,
    pub abi: Option<Arc<dyn NvidiaAbi>>,
    pub(super) primary_hardware_context: Mutex<Weak<HardwareContext>>,
    nvidia_unique: OnceLock<u64>,
    state_clock: Instant,
    state_checked_ns: AtomicU64,
}

impl DeviceInner {
    pub(crate) fn paging_value(&self) -> PagingValue {
        PagingValue::decode(self.last_paging.load(Ordering::Acquire))
    }

    pub(crate) fn check_execution_state(&self) -> Result<u32> {
        let mut request = D3DKMT_GETDEVICESTATE {
            hDevice: self.handle,
            StateType: D3DKMT_DEVICESTATE_EXECUTION,
            ..Default::default()
        };
        let status = unsafe { D3DKMTGetDeviceState(&mut request) }.0;
        let state = unsafe { request.Anonymous.ExecutionState }.0;
        if status < 0 || state != D3DKMT_DEVICEEXECUTION_ACTIVE.0 {
            self.lost.store(true, Ordering::Release);
            if std::env::var_os("NVK_WDDM_DEBUG").is_some() {
                eprintln!(
                    "NVK WDDM execution state={} NTSTATUS=0x{:08x}",
                    state, status as u32
                );
                if state == D3DKMT_DEVICEEXECUTION_ERROR_DMAPAGEFAULT.0 {
                    let mut fault = D3DKMT_GETDEVICESTATE {
                        hDevice: self.handle,
                        StateType: D3DKMT_DEVICESTATE_PAGE_FAULT,
                        ..Default::default()
                    };
                    let fault_status =
                        unsafe { D3DKMTGetDeviceState(&mut fault) }.0;
                    if fault_status >= 0 {
                        let fault = unsafe { fault.Anonymous.PageFaultState };
                        eprintln!(
                            "NVK WDDM page fault address=0x{:x} primitive={} flags=0x{:x}",
                            fault.FaultedVirtualAddress,
                            fault.FaultedPrimitiveAPISequenceNumber,
                            fault.PageFaultFlags.0,
                        );
                    }
                }
            }
            return Err(Error {
                kind: ErrorKind::DeviceLost,
                operation: "D3DKMTGetDeviceState execution",
                status,
            });
        }
        Ok(state as u32)
    }

    pub fn check(&self, operation: &'static str, status: i32) -> Result<()> {
        match Error::check(operation, status) {
            Err(error) => {
                if error.kind == ErrorKind::DeviceLost {
                    self.lost.store(true, Ordering::Release);
                }
                Err(error)
            }
            Ok(()) => Ok(()),
        }
    }

    pub fn ready(&self) -> Result<()> {
        if self.lost.load(Ordering::Acquire) {
            return Err(Error::new(ErrorKind::DeviceLost, "WDDM device"));
        }
        let now = self
            .state_clock
            .elapsed()
            .as_nanos()
            .min(u128::from(u64::MAX)) as u64;
        let previous = self.state_checked_ns.load(Ordering::Relaxed);
        if (previous == 0 || now.saturating_sub(previous) >= 1_000_000)
            && self
                .state_checked_ns
                .compare_exchange(
                    previous,
                    now.max(1),
                    Ordering::AcqRel,
                    Ordering::Relaxed,
                )
                .is_ok()
        {
            self.check_execution_state()?;
        }
        Ok(())
    }
}

impl Drop for DeviceInner {
    fn drop(&mut self) {
        unsafe {
            let _ = D3DKMTDestroyPagingQueue(&mut D3DDDI_DESTROYPAGINGQUEUE {
                hPagingQueue: self.paging_queue,
            });
            let _ = D3DKMTDestroyDevice(&D3DKMT_DESTROYDEVICE {
                hDevice: self.handle,
            });
        }
    }
}

#[derive(Clone)]
pub struct Device(pub(crate) Arc<DeviceInner>);

impl Device {
    pub fn execution_state(&self) -> Result<u32> {
        self.0.ready()?;
        self.0.check_execution_state()
    }

    pub(crate) fn nvidia_unique_id(&self) -> Result<u64> {
        self.0.ready()?;
        if let Some(value) = self.0.nvidia_unique.get() {
            return Ok(*value);
        }
        let identity = self.adapter().identity();
        if identity.vendor_id != 0x10de
            || identity.driver_version
                != crate::nvidia::Nvgl61692Layout::DRIVER_VERSION
        {
            return Err(Error::new(
                ErrorKind::Unsupported,
                "NVIDIA 616.92 device query",
            ));
        }
        let mut data = crate::abi::PrivateData::zeroed(56)?;
        crate::nvidia::PrivateHeader {
            version: 0x10002,
            size: 56,
            caller: 0x4e562a2a,
        }
        .write(&mut data)?;
        data.write_u32(16, 0x01000057)?;
        let mut request = D3DKMT_ESCAPE {
            hAdapter: self.adapter().handle(),
            hDevice: self.0.handle,
            Type: D3DKMT_ESCAPE_DRIVERPRIVATE,
            pPrivateDriverData: data.as_bytes_mut().as_mut_ptr().cast(),
            PrivateDriverDataSize: data.len_u32(),
            ..Default::default()
        };
        request.Flags.Anonymous.Value = 8;
        self.0.check(
            "D3DKMTEscape device unique ID",
            unsafe { D3DKMTEscape(&request) }.0,
        )?;
        let value =
            u64::from_le_bytes(data.as_bytes()[48..56].try_into().map_err(
                |_| Error::new(ErrorKind::Invalid, "device unique ID reply"),
            )?);
        let _ = self.0.nvidia_unique.set(value);
        Ok(value)
    }

    pub(crate) fn new(
        adapter: Adapter,
        abi: Option<Arc<dyn NvidiaAbi>>,
    ) -> Result<Self> {
        let mut create = D3DKMT_CREATEDEVICE::default();
        create.Anonymous.hAdapter = adapter.handle();
        Error::check(
            "D3DKMTCreateDevice",
            unsafe { D3DKMTCreateDevice(&mut create) }.0,
        )?;
        let mut paging = D3DKMT_CREATEPAGINGQUEUE {
            hDevice: create.hDevice,
            Priority: D3DDDI_PAGINGQUEUE_PRIORITY_NORMAL,
            ..Default::default()
        };
        let status = unsafe { D3DKMTCreatePagingQueue(&mut paging) }.0;
        if let Err(error) = Error::check("D3DKMTCreatePagingQueue", status) {
            unsafe {
                let _ = D3DKMTDestroyDevice(&D3DKMT_DESTROYDEVICE {
                    hDevice: create.hDevice,
                });
            }
            return Err(error);
        }
        let device = Self(Arc::new(DeviceInner {
            adapter,
            handle: create.hDevice,
            paging_queue: paging.hPagingQueue,
            paging_sync: paging.hSyncObject,
            paging_cpu: paging.FenceValueCPUVirtualAddress as usize,
            lost: AtomicBool::new(false),
            last_paging: AtomicU64::new(0),
            abi,
            primary_hardware_context: Mutex::new(Weak::new()),
            nvidia_unique: OnceLock::new(),
            state_clock: Instant::now(),
            state_checked_ns: AtomicU64::new(0),
        }));
        if device.0.paging_cpu == 0
            || device.0.paging_cpu % 8 != 0
            || device.0.paging_sync == 0
        {
            return Err(Error::new(ErrorKind::Invalid, "paging fence mapping"));
        }
        Ok(device)
    }

    pub fn adapter(&self) -> &Adapter {
        &self.0.adapter
    }
    pub fn is_lost(&self) -> bool {
        self.0.lost.load(Ordering::Acquire)
    }
    pub fn paging_point(&self, value: u64) -> FencePoint {
        FencePoint::paging(self.0.clone(), PagingValue::new(Some(value as u32)))
    }
    pub fn last_paging_point(&self) -> FencePoint {
        FencePoint::paging(self.0.clone(), self.0.paging_value())
    }
    pub fn wait_paging(&self, timeout_ns: u64) -> Result<()> {
        self.last_paging_point().wait(timeout_ns)
    }

    pub fn mark_lost(&self) {
        self.0.lost.store(true, Ordering::Release);
    }

    pub(crate) fn same_device(&self, other: &Arc<DeviceInner>) -> bool {
        Arc::ptr_eq(&self.0, other)
    }
}
