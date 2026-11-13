use super::Device;
use crate::abi::NvidiaAbi;
use crate::contract::{
    AdapterId, Capabilities, DriverIdentity, KernelBackend, KernelDriver,
};
use crate::error::{Error, ErrorKind, Result};
use std::sync::Arc;
use windows::Wdk::Graphics::Direct3D::*;
use windows::Win32::Foundation::LUID;

struct AdapterHandle(u32);

impl Drop for AdapterHandle {
    fn drop(&mut self) {
        let args = D3DKMT_CLOSEADAPTER { hAdapter: self.0 };
        unsafe {
            let _ = D3DKMTCloseAdapter(&args);
        }
    }
}

#[derive(Clone)]
pub struct Adapter {
    handle: Arc<AdapterHandle>,
    identity: DriverIdentity,
    pub adapter_type: u32,
    pub scheduling_caps: u32,
}

impl Adapter {
    pub fn gpu_timestamp_ns(&self) -> Result<u64> {
        let mut calibration = D3DKMT_QUERYCLOCKCALIBRATION {
            hAdapter: self.handle(),
            ..Default::default()
        };
        Error::check(
            "D3DKMTQueryClockCalibration",
            unsafe { D3DKMTQueryClockCalibration(&mut calibration) }.0,
        )?;
        let frequency = calibration.ClockData.GpuFrequency;
        let counter = calibration.ClockData.GpuClockCounter;
        if frequency == 1_000_000_000 {
            return Ok(counter);
        }
        if frequency == 0 {
            return Err(Error::new(
                ErrorKind::Unsupported,
                "GPU timestamp frequency",
            ));
        }
        (u128::from(counter) * 1_000_000_000 / u128::from(frequency))
            .try_into()
            .map_err(|_| Error::new(ErrorKind::Invalid, "GPU timestamp range"))
    }

    pub fn name(&self) -> Result<String> {
        let mut info = D3DKMT_ADAPTERREGISTRYINFO::default();
        Self::query(self.handle(), KMTQAITYPE_ADAPTERREGISTRYINFO, &mut info)?;
        let end = info
            .AdapterString
            .iter()
            .position(|v| *v == 0)
            .unwrap_or(info.AdapterString.len());
        String::from_utf16(&info.AdapterString[..end]).map_err(|_| {
            Error::new(ErrorKind::Invalid, "adapter name encoding")
        })
    }

    pub fn segment_sizes(&self) -> Result<(u64, u64, u64)> {
        let mut info = D3DKMT_SEGMENTSIZEINFO::default();
        Self::query(self.handle(), KMTQAITYPE_GETSEGMENTSIZE, &mut info)?;
        Ok((
            info.DedicatedVideoMemorySize,
            info.DedicatedSystemMemorySize,
            info.SharedSystemMemorySize,
        ))
    }

    pub fn enumerate() -> Result<Vec<Self>> {
        let mut count = D3DKMT_ENUMADAPTERS2::default();
        let status = unsafe { D3DKMTEnumAdapters2(&mut count) }.0;
        if status < 0 && status as u32 != 0xC0000023 {
            return Err(Error::from_status(
                "D3DKMTEnumAdapters2 count",
                status,
            ));
        }
        for _ in 0..4 {
            if count.NumAdapters == 0 {
                return Ok(Vec::new());
            }
            if count.NumAdapters > 256 {
                return Err(Error::new(ErrorKind::Invalid, "adapter count"));
            }
            let mut info =
                vec![D3DKMT_ADAPTERINFO::default(); count.NumAdapters as usize];
            let capacity = info.len();
            count.pAdapters = info.as_mut_ptr();
            let status = unsafe { D3DKMTEnumAdapters2(&mut count) }.0;
            if status as u32 == 0xC0000023 {
                continue;
            }
            Error::check("D3DKMTEnumAdapters2", status)?;
            if count.NumAdapters as usize > capacity {
                return Err(Error::new(
                    ErrorKind::Invalid,
                    "adapter count changed",
                ));
            }
            info.truncate(count.NumAdapters as usize);
            let owned: Vec<_> = info
                .into_iter()
                .map(|entry| (AdapterHandle(entry.hAdapter), entry.AdapterLuid))
                .collect();
            let mut adapters = Vec::new();
            for (handle, luid) in owned {
                match Self::from_owned(handle, luid) {
                    Ok(adapter) => adapters.push(adapter),
                    Err(error)
                        if error.kind == ErrorKind::Unsupported
                            || error.kind == ErrorKind::Invalid => {}
                    Err(error) => return Err(error),
                }
            }
            return Ok(adapters);
        }
        Err(Error::new(ErrorKind::Busy, "adapter topology unstable"))
    }

    pub fn open(id: AdapterId) -> Result<Self> {
        let luid = LUID {
            LowPart: id.low,
            HighPart: id.high,
        };
        let mut args = D3DKMT_OPENADAPTERFROMLUID {
            AdapterLuid: luid,
            ..Default::default()
        };
        Error::check(
            "D3DKMTOpenAdapterFromLuid",
            unsafe { D3DKMTOpenAdapterFromLuid(&mut args) }.0,
        )?;
        Self::from_owned(AdapterHandle(args.hAdapter), luid)
    }

    fn query<T>(
        handle: u32,
        kind: KMTQUERYADAPTERINFOTYPE,
        value: &mut T,
    ) -> Result<()> {
        let mut args = D3DKMT_QUERYADAPTERINFO {
            hAdapter: handle,
            Type: kind,
            pPrivateDriverData: std::ptr::from_mut(value).cast(),
            PrivateDriverDataSize: std::mem::size_of::<T>()
                .try_into()
                .map_err(|_| {
                    Error::new(ErrorKind::Invalid, "adapter query size")
                })?,
        };
        Error::check(
            "D3DKMTQueryAdapterInfo",
            unsafe { D3DKMTQueryAdapterInfo(&mut args) }.0,
        )
    }

    fn from_owned(handle: AdapterHandle, luid: LUID) -> Result<Self> {
        let mut ids = D3DKMT_QUERY_DEVICE_IDS::default();
        Self::query(handle.0, KMTQAITYPE_PHYSICALADAPTERDEVICEIDS, &mut ids)?;
        let mut version = D3DKMT_UMD_DRIVER_VERSION::default();
        Self::query(handle.0, KMTQAITYPE_UMD_DRIVER_VERSION, &mut version)?;
        let mut model = D3DKMT_DRIVERVERSION::default();
        Self::query(handle.0, KMTQAITYPE_DRIVERVERSION, &mut model)?;
        let mut adapter_type = D3DKMT_ADAPTERTYPE::default();
        Self::query(handle.0, KMTQAITYPE_ADAPTERTYPE, &mut adapter_type)?;
        let mut scheduling = D3DKMT_WDDM_2_7_CAPS::default();
        let scheduling_caps = match Self::query(
            handle.0,
            KMTQAITYPE_WDDM_2_7_CAPS,
            &mut scheduling,
        ) {
            Ok(()) => unsafe { scheduling.Anonymous.Value },
            Err(error)
                if matches!(
                    error.kind,
                    ErrorKind::Invalid | ErrorKind::Unsupported
                ) =>
            {
                0
            }
            Err(error) => return Err(error),
        };
        Ok(Self {
            handle: Arc::new(handle),
            identity: DriverIdentity {
                adapter: AdapterId {
                    low: luid.LowPart,
                    high: luid.HighPart,
                },
                vendor_id: ids.DeviceIds.VendorID,
                device_id: ids.DeviceIds.DeviceID,
                driver_version: version.DriverVersion as u64,
                wddm_version: model.0 as u32,
            },
            adapter_type: unsafe { adapter_type.Anonymous.Value },
            scheduling_caps,
        })
    }

    pub fn identity(&self) -> &DriverIdentity {
        &self.identity
    }
    pub fn query_nvidia_61692(&self) -> Result<crate::nvidia::Nvgl61692Query> {
        use crate::nvidia::{Nvgl61692Layout, Nvgl61692Query};
        if self.identity.vendor_id != 0x10de
            || self.identity.driver_version != Nvgl61692Layout::DRIVER_VERSION
        {
            return Err(Error::new(
                ErrorKind::Unsupported,
                "NVIDIA adapter query profile",
            ));
        }
        let mut data = Nvgl61692Layout::query_request()?;
        unsafe {
            self.query_private(&mut data)?;
        }
        Nvgl61692Query::parse(data)
    }
    pub unsafe fn query_private(
        &self,
        data: &mut crate::abi::PrivateData,
    ) -> Result<()> {
        if data.is_empty() {
            return Err(Error::new(
                ErrorKind::Invalid,
                "private adapter query length",
            ));
        }
        let mut request = D3DKMT_QUERYADAPTERINFO {
            hAdapter: self.handle(),
            Type: KMTQAITYPE_UMDRIVERPRIVATE,
            pPrivateDriverData: data.as_bytes_mut().as_mut_ptr().cast(),
            PrivateDriverDataSize: data.len_u32(),
        };
        Error::check(
            "D3DKMTQueryAdapterInfo private",
            unsafe { D3DKMTQueryAdapterInfo(&mut request) }.0,
        )
    }
    pub(crate) fn handle(&self) -> u32 {
        self.handle.0
    }
    pub fn create_device(&self) -> Result<Device> {
        Device::new(self.clone(), None)
    }
}

pub struct WddmDriver {
    adapter: Adapter,
    abi: Option<Arc<dyn NvidiaAbi>>,
}

impl WddmDriver {
    pub fn new(adapter: Adapter) -> Result<Self> {
        if adapter.identity.vendor_id != 0x10de {
            return Err(Error::new(ErrorKind::Unsupported, "NVIDIA adapter"));
        }
        Ok(Self { adapter, abi: None })
    }

    pub fn with_abi(mut self, abi: Arc<dyn NvidiaAbi>) -> Result<Self> {
        abi.key().validate(self.adapter.identity())?;
        self.abi = Some(abi);
        Ok(self)
    }
}

impl KernelDriver for WddmDriver {
    type Device = Device;
    fn backend(&self) -> KernelBackend {
        KernelBackend::NvidiaWddm
    }
    fn identity(&self) -> &DriverIdentity {
        self.adapter.identity()
    }
    fn capabilities(&self) -> Capabilities {
        self.abi
            .as_ref()
            .map_or_else(Capabilities::default, |abi| abi.capabilities())
    }
    fn create_device(&self) -> Result<Device> {
        Device::new(self.adapter.clone(), self.abi.clone())
    }
}
