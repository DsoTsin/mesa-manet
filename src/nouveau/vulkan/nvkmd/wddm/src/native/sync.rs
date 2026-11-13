use super::{device::DeviceInner, Device};
use crate::error::{Error, ErrorKind, Result};
use std::sync::{
    atomic::{AtomicU32, AtomicU64, Ordering},
    Arc,
};
use windows::Wdk::Graphics::Direct3D::*;
use windows::Win32::Foundation::{
    CloseHandle, HANDLE, STATUS_PENDING, WAIT_OBJECT_0, WAIT_TIMEOUT,
};
use windows::Win32::System::Threading::{
    CreateEventW, WaitForSingleObject, INFINITE,
};

pub(crate) struct FenceInner {
    pub device: Arc<DeviceInner>,
    pub handle: u32,
    pub cpu: usize,
    pub gpu: u64,
}

impl Drop for FenceInner {
    fn drop(&mut self) {
        unsafe {
            let _ = D3DKMTDestroySynchronizationObject(
                &D3DKMT_DESTROYSYNCHRONIZATIONOBJECT {
                    hSyncObject: self.handle,
                },
            );
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PagingValue(Option<u32>);

impl PagingValue {
    pub(crate) fn new(value: Option<u32>) -> Self {
        Self(value)
    }

    pub(crate) fn from_status(status: i32, output: &u64) -> Self {
        Self(if status == STATUS_PENDING.0 {
            Some(*output as u32)
        } else {
            None
        })
    }

    pub(crate) fn decode(encoded: u64) -> Self {
        Self(if encoded & (1 << 32) != 0 {
            Some(encoded as u32)
        } else {
            None
        })
    }

    pub(crate) fn encode(self) -> u64 {
        self.0.map_or(0, |value| (1 << 32) | u64::from(value))
    }

    pub(crate) fn value(self) -> Option<u32> {
        self.0
    }

    pub(crate) fn publish(self, latest: &AtomicU64) {
        if let Some(value) = self.0 {
            let _ = latest.fetch_update(
                Ordering::AcqRel,
                Ordering::Acquire,
                |encoded| {
                    let current = Self::decode(encoded);
                    if current.0.is_none_or(|old| {
                        let distance = value.wrapping_sub(old);
                        distance != 0 && distance < 0x8000_0000
                    }) {
                        Some(self.encode())
                    } else {
                        None
                    }
                },
            );
        }
    }

    fn reached_by(self, current: u32) -> bool {
        self.0
            .is_none_or(|value| current.wrapping_sub(value) < 0x8000_0000)
    }
}

enum FenceOwner {
    Paging {
        device: Arc<DeviceInner>,
        required: bool,
    },
    Sync(Arc<FenceInner>),
}

#[derive(Clone)]
pub struct FencePoint {
    owner: Arc<FenceOwner>,
    pub value: u64,
}

struct Event(HANDLE);
impl Drop for Event {
    fn drop(&mut self) {
        unsafe {
            let _ = CloseHandle(self.0);
        }
    }
}

impl FencePoint {
    pub(crate) fn paging(device: Arc<DeviceInner>, value: PagingValue) -> Self {
        let value = value.value();
        Self {
            owner: Arc::new(FenceOwner::Paging {
                device,
                required: value.is_some(),
            }),
            value: u64::from(value.unwrap_or(0)),
        }
    }

    pub(crate) fn paging_value(&self) -> Option<u32> {
        match self.owner.as_ref() {
            FenceOwner::Paging { required, .. } => {
                required.then_some(self.value as u32)
            }
            FenceOwner::Sync(_) => None,
        }
    }

    pub(crate) fn wait_required(&self) -> bool {
        match self.owner.as_ref() {
            FenceOwner::Paging { required, .. } => *required,
            FenceOwner::Sync(_) => self.value != 0,
        }
    }

    pub(crate) fn device(&self) -> &Arc<DeviceInner> {
        match self.owner.as_ref() {
            FenceOwner::Paging { device, .. } => device,
            FenceOwner::Sync(fence) => &fence.device,
        }
    }

    pub(crate) fn handle(&self) -> u32 {
        match self.owner.as_ref() {
            FenceOwner::Paging { device, .. } => device.paging_sync,
            FenceOwner::Sync(fence) => fence.handle,
        }
    }

    fn cpu(&self) -> usize {
        match self.owner.as_ref() {
            FenceOwner::Paging { device, .. } => device.paging_cpu,
            FenceOwner::Sync(fence) => fence.cpu,
        }
    }

    pub fn completed(&self) -> Result<bool> {
        self.device().ready()?;
        if !self.wait_required() {
            return Ok(true);
        }
        if let FenceOwner::Paging { .. } = self.owner.as_ref() {
            let current = unsafe {
                (&*(self.cpu() as *const AtomicU32)).load(Ordering::Acquire)
            };
            return Ok(
                PagingValue::new(self.paging_value()).reached_by(current)
            );
        }
        let current = unsafe {
            (&*(self.cpu() as *const AtomicU64)).load(Ordering::Acquire)
        };
        if current == u64::MAX {
            self.device().check_execution_state()?;
        }
        Ok(current >= self.value)
    }

    pub fn wait(&self, timeout_ns: u64) -> Result<()> {
        if self.completed()? {
            return Ok(());
        }
        if timeout_ns == 0 {
            return Err(Error::new(ErrorKind::Timeout, "fence wait"));
        }
        let event =
            Event(unsafe { CreateEventW(None, true, false, None) }.map_err(
                |_| Error::new(ErrorKind::HostMemory, "CreateEventW"),
            )?);
        let handle = self.handle();
        let request = D3DKMT_WAITFORSYNCHRONIZATIONOBJECTFROMCPU {
            hDevice: self.device().handle,
            ObjectCount: 1,
            ObjectHandleArray: &handle,
            FenceValueArray: &self.value,
            hAsyncEvent: event.0,
            ..Default::default()
        };
        self.device().check(
            "D3DKMTWaitForSynchronizationObjectFromCpu",
            unsafe { D3DKMTWaitForSynchronizationObjectFromCpu(&request) }.0,
        )?;
        let milliseconds = if timeout_ns == u64::MAX {
            INFINITE
        } else {
            timeout_ns.div_ceil(1_000_000).min(u64::from(INFINITE - 1)) as u32
        };
        match unsafe { WaitForSingleObject(event.0, milliseconds) } {
            WAIT_OBJECT_0 if self.completed()? => Ok(()),
            WAIT_TIMEOUT => Err(Error::new(ErrorKind::Timeout, "fence wait")),
            _ => Err(Error::new(ErrorKind::System, "fence event")),
        }
    }
}

pub struct Fence {
    pub(crate) inner: Arc<FenceInner>,
    next: u64,
}

impl Fence {
    pub fn new(device: &Device, initial: u64) -> Result<Self> {
        device.0.ready()?;
        let mut request = D3DKMT_CREATESYNCHRONIZATIONOBJECT2 {
            hDevice: device.0.handle,
            ..Default::default()
        };
        request.Info.Type = D3DDDI_MONITORED_FENCE;
        request.Info.Anonymous.MonitoredFence =
            D3DDDI_SYNCHRONIZATIONOBJECTINFO2_0_4 {
                InitialFenceValue: initial,
                ..Default::default()
            };
        device.0.check(
            "D3DKMTCreateSynchronizationObject2",
            unsafe { D3DKMTCreateSynchronizationObject2(&mut request) }.0,
        )?;
        let mapped = unsafe { request.Info.Anonymous.MonitoredFence };
        let inner = Arc::new(FenceInner {
            device: device.0.clone(),
            handle: request.hSyncObject,
            cpu: mapped.FenceValueCPUVirtualAddress as usize,
            gpu: mapped.FenceValueGPUVirtualAddress,
        });
        if inner.cpu == 0 || inner.cpu % 8 != 0 {
            return Err(Error::new(
                ErrorKind::Invalid,
                "monitored fence mapping",
            ));
        }
        Ok(Self {
            inner,
            next: initial,
        })
    }

    pub fn reserve(&mut self) -> Result<FencePoint> {
        self.inner.device.ready()?;
        self.next = self
            .next
            .checked_add(1)
            .ok_or_else(|| Error::new(ErrorKind::Invalid, "fence overflow"))?;
        Ok(self.point(self.next))
    }

    pub fn point(&self, value: u64) -> FencePoint {
        FencePoint {
            owner: Arc::new(FenceOwner::Sync(self.inner.clone())),
            value,
        }
    }

    pub fn value(&self) -> Result<u64> {
        self.inner.device.ready()?;
        let value = unsafe {
            (&*(self.inner.cpu as *const AtomicU64)).load(Ordering::Acquire)
        };
        if value == u64::MAX {
            self.inner.device.check_execution_state()?;
        }
        Ok(value)
    }

    pub fn signal_cpu_value(&mut self, value: u64) -> Result<()> {
        self.inner.device.ready()?;
        if value <= self.value()? {
            return Err(Error::new(
                ErrorKind::Invalid,
                "non-increasing CPU fence signal",
            ));
        }
        let mut request = D3DKMT_SIGNALSYNCHRONIZATIONOBJECTFROMCPU {
            hDevice: self.inner.device.handle,
            ObjectCount: 1,
            ObjectHandleArray: &self.inner.handle,
            FenceValueArray: &value,
            ..Default::default()
        };
        request.Flags.Anonymous.Value = 4;
        self.inner.device.check(
            "D3DKMTSignalSynchronizationObjectFromCpu",
            unsafe { D3DKMTSignalSynchronizationObjectFromCpu(&request) }.0,
        )?;
        self.next = self.next.max(value);
        Ok(())
    }

    pub fn signal_cpu(&mut self) -> Result<FencePoint> {
        let point = self.reserve()?;
        self.signal_cpu_value(point.value)?;
        Ok(point)
    }

    pub fn gpu_address(&self) -> u64 {
        self.inner.gpu
    }
}

#[cfg(test)]
mod tests {
    use super::PagingValue;
    use std::sync::atomic::{AtomicU64, Ordering};
    use windows::Win32::Foundation::STATUS_PENDING;

    #[test]
    fn paging_validity_preserves_pending_zero() {
        let absent = PagingValue::new(None);
        let zero = PagingValue::new(Some(0));
        assert_eq!(absent.encode(), 0);
        assert_eq!(zero.encode(), 1 << 32);
        assert_ne!(absent, zero);
        for value in [0, 1, 0x7fff_ffff, 0x8000_0000, u32::MAX] {
            let point = PagingValue::new(Some(value));
            assert_eq!(PagingValue::decode(point.encode()), point);
        }
        assert_eq!(PagingValue::decode(0), absent);
        assert!(absent.reached_by(u32::MAX));
        assert!(!zero.reached_by(u32::MAX));
        assert!(zero.reached_by(0));
    }

    #[test]
    fn paging_outputs_are_consumed_only_for_pending_status() {
        let latest = AtomicU64::new(0);
        for output in [0, 1, u64::from(u32::MAX), u64::MAX] {
            let value = PagingValue::from_status(0, &output);
            assert_eq!(value.value(), None);
            value.publish(&latest);
            assert_eq!(latest.load(Ordering::Acquire), 0);
        }
        let value = PagingValue::from_status(STATUS_PENDING.0, &0);
        assert_eq!(value.value(), Some(0));
        value.publish(&latest);
        assert_eq!(PagingValue::decode(latest.load(Ordering::Acquire)), value);
        PagingValue::from_status(0, &u64::MAX).publish(&latest);
        assert_eq!(PagingValue::decode(latest.load(Ordering::Acquire)), value);
    }

    #[test]
    fn paging_publication_handles_wrap_and_older_responses() {
        let latest = AtomicU64::new(0);
        for value in [0xffff_fff0, 0xffff_ffff, 0, 4, 2, 0xffff_fffe, 4] {
            PagingValue::new(Some(value)).publish(&latest);
        }
        assert_eq!(
            PagingValue::decode(latest.load(Ordering::Acquire)).value(),
            Some(4)
        );
        PagingValue::new(None).publish(&latest);
        assert_eq!(
            PagingValue::decode(latest.load(Ordering::Acquire)).value(),
            Some(4)
        );
        PagingValue::new(Some(0x8000_0004)).publish(&latest);
        assert_eq!(
            PagingValue::decode(latest.load(Ordering::Acquire)).value(),
            Some(4)
        );
    }

    #[test]
    fn paging_wrap_properties() {
        let mut state = 0x5d17_0a68_e70a_d17bu64;
        for _ in 0..32_768 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let start = state as u32;
            let distance = ((state >> 32) as u32 & 0xffff) + 1;
            let end = start.wrapping_add(distance);
            let target = PagingValue::new(Some(end));
            assert!(!target.reached_by(start));
            assert!(!target.reached_by(end.wrapping_sub(1)));
            assert!(target.reached_by(end));
            assert!(target.reached_by(end.wrapping_add(distance)));
            let latest = AtomicU64::new(PagingValue::new(Some(start)).encode());
            target.publish(&latest);
            PagingValue::new(Some(start)).publish(&latest);
            assert_eq!(
                PagingValue::decode(latest.load(Ordering::Acquire)),
                target
            );
        }
    }

    #[test]
    fn paging_concurrent_publication_retains_latest() {
        let start = 0xffff_ff00u32;
        let latest = AtomicU64::new(PagingValue::new(Some(start)).encode());
        std::thread::scope(|scope| {
            for worker in 0..8 {
                let latest = &latest;
                scope.spawn(move || {
                    for index in (1..=4096).rev() {
                        let value = start.wrapping_add(index * 8 + worker);
                        PagingValue::new(Some(value)).publish(latest);
                    }
                });
            }
        });
        assert_eq!(
            PagingValue::decode(latest.load(Ordering::Acquire)).value(),
            Some(start.wrapping_add(4096 * 8 + 7))
        );
    }
}
