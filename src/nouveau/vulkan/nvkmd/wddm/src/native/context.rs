use super::channel::Channel;
use super::{Device, Fence, FencePoint, MemoryBinding};
use crate::abi::{NvidiaAbi, PrivateData};
use crate::contract::{Push, Scheduling};
use crate::error::{Error, ErrorKind, Result};
use crate::submission::QueueTransport;
use std::collections::VecDeque;
use std::sync::Arc;

struct SubmissionStorage {
    point: FencePoint,
    _bindings: Vec<MemoryBinding>,
}

pub struct Context {
    device: Device,
    abi: Arc<dyn NvidiaAbi>,
    channel: Channel,
    private: PrivateData,
    completion: Fence,
    poisoned: bool,
    submission_storage: VecDeque<SubmissionStorage>,
    last_paging_wait: Option<u32>,
}

impl Context {
    pub fn reflected_sked_address(&self) -> Result<u64> {
        self.channel.reflected_sked_address()
    }

    pub fn query_reflected_sked_address(&self) -> Result<u64> {
        self.channel.query_reflected_sked_address()
    }

    pub fn progress_values(&self) -> Result<(u64, u64)> {
        self.channel.progress_values()?.ok_or_else(|| {
            Error::new(
                ErrorKind::Unsupported,
                "virtual context progress mapping",
            )
        })
    }

    pub fn present(
        &mut self,
        frame: super::wsi::DwmFrame<'_>,
        image: &super::wsi::PresentImage,
        timeout_ns: u64,
    ) -> Result<i32> {
        use windows::Wdk::Graphics::Direct3D::*;
        use windows::Win32::Foundation::{HWND, RECT};
        self.device.0.ready()?;
        if self.poisoned {
            return Err(Error::new(
                ErrorKind::System,
                "context requires teardown",
            ));
        }
        if !self.device.same_device(&image.memory().inner.device)
            || frame.snapshot.width != image.width()
            || frame.snapshot.height != image.height()
        {
            return Err(Error::new(
                ErrorKind::Invalid,
                "presentation image device or extent",
            ));
        }
        if frame.current_snapshot()? != frame.snapshot {
            return Err(Error::new(
                ErrorKind::OutOfDate,
                "surface changed after acquisition",
            ));
        }
        if frame.physical_surface != 0 {
            return Err(Error::new(
                ErrorKind::Unsupported,
                "DWM shared destination import",
            ));
        }
        let mut private = PrivateData::zeroed(199)?;
        crate::nvidia::PrivateHeader {
            version: 0x10004,
            size: 199,
            caller: crate::nvidia::PrivateHeader::NVGL,
        }
        .write(&mut private)?;
        private.write_u32(16, u32::MAX)?;
        let rect = RECT {
            left: 0,
            top: 0,
            right: image.width() as i32,
            bottom: image.height() as i32,
        };
        let mut request = D3DKMT_PRESENT {
            hWindow: HWND(frame.hwnd() as *mut _),
            VidPnSourceId: frame.snapshot.source_id,
            hSource: image.memory().inner.handle,
            SrcRect: rect,
            SubRectCnt: 1,
            pSrcSubRects: &rect,
            PrivateDriverDataSize: private.len_u32(),
            pPrivateDriverData: private.as_bytes_mut().as_mut_ptr().cast(),
            bOptimizeForComposition: true,
            ..Default::default()
        };
        request.Flags.Anonymous.Value = 1 | 0x80 | 0x10000;
        request.PresentHistoryToken.Model = D3DKMT_PRESENT_MODEL(3);
        request.PresentHistoryToken.Token.Blt =
            D3DKMT_BLTMODEL_PRESENTHISTORYTOKEN {
                hPhysicalSurface: frame.physical_surface,
                EventId: frame.update_id,
                ..Default::default()
            };
        let point = self.completion.reserve()?;
        let mut address = super::VirtualAddress::reserve(
            &self.device,
            image.memory().size(),
            65536,
            None,
        )?;
        let binding =
            address.bind(image.memory(), 0, 0, image.memory().size())?;
        let operation = (|| {
            image.memory().make_resident()?;
            self.synchronize_paging()?;
            let status = self.channel.present(&mut request)?;
            self.channel.signal(&point, false)?;
            point.wait(timeout_ns)?;
            Ok(status)
        })();
        if operation.is_err() {
            self.poisoned = true;
            if !self.device.is_lost() {
                std::mem::forget(binding);
                std::mem::forget(address);
            }
        }
        operation
    }

    pub fn new(
        device: &Device,
        engines: u32,
        scheduling: Scheduling,
    ) -> Result<Self> {
        device.0.ready()?;
        if engines == 0 {
            return Err(Error::new(ErrorKind::Invalid, "empty engine mask"));
        }
        let device_abi = device.0.abi.as_ref().ok_or_else(|| {
            Error::new(ErrorKind::Unsupported, "unverified context ABI")
        })?;
        let abi = device_abi
            .context_profile(engines)?
            .unwrap_or_else(|| device_abi.clone());
        let mut payload = abi.context(engines, scheduling)?;
        let completion = Fence::new(device, 0)?;
        let compute_queue = engines & 16 != 0 && engines & 4 == 0;
        let channel = Channel::new(
            device,
            scheduling,
            &mut payload,
            abi.clone(),
            compute_queue,
        )?;
        Ok(Self {
            device: device.clone(),
            abi,
            channel,
            private: payload.private,
            completion,
            poisoned: false,
            submission_storage: VecDeque::new(),
            last_paging_wait: None,
        })
    }

    fn bind_host_data(&self, bytes: &[u8]) -> Result<MemoryBinding> {
        let size = u64::try_from(bytes.len())
            .ok()
            .and_then(|n| n.checked_add(4095))
            .map(|n| n & !4095)
            .filter(|n| *n != 0)
            .ok_or_else(|| {
                Error::new(ErrorKind::Invalid, "submission storage size")
            })?;
        let (mut memory, _) = super::Memory::nvidia_host(&self.device, size)?;
        memory.write(0, bytes)?;
        let mut va =
            super::VirtualAddress::reserve(&self.device, size, 65536, None)?;
        va.bind(&memory, 0, 0, size)
    }

    fn enqueue_wait(&self, point: &FencePoint) -> Result<()> {
        self.channel.wait(point)
    }

    fn synchronize_paging(&mut self) -> Result<()> {
        let point = self.device.last_paging_point();
        let value = point.paging_value();
        if value != self.last_paging_wait {
            self.enqueue_wait(&point)?;
            self.last_paging_wait = value;
        }
        Ok(())
    }

    fn enqueue_signal(&self, point: &FencePoint) -> Result<()> {
        self.channel.signal(point, false)
    }

    pub fn queue(self) -> crate::submission::Queue<Self> {
        crate::submission::Queue::new(self)
    }
}

impl QueueTransport for Context {
    type Resource = MemoryBinding;
    type Point = FencePoint;

    fn validate_resource(&self, resource: &MemoryBinding) -> Result<()> {
        if !self.device.same_device(resource.device()) {
            Err(Error::new(
                ErrorKind::Invalid,
                "cross-device submission resource",
            ))
        } else {
            Ok(())
        }
    }

    fn submit(
        &mut self,
        pushes: &[Push],
        resources: &[MemoryBinding],
        waits: &[FencePoint],
        signals: &[FencePoint],
    ) -> Result<FencePoint> {
        self.device.0.ready()?;
        if self.poisoned {
            return Err(Error::new(
                ErrorKind::System,
                "context requires teardown",
            ));
        }
        for resource in resources {
            self.validate_resource(resource)?;
        }
        for wait in waits.iter().chain(signals.iter()) {
            if !self.device.same_device(wait.device()) {
                return Err(Error::new(
                    ErrorKind::Invalid,
                    "cross-device semaphore",
                ));
            }
        }
        for push in pushes {
            push.validate()?;
            if !resources
                .iter()
                .any(|r| r.contains(push.address, push.size.into()))
            {
                return Err(Error::new(
                    ErrorKind::Invalid,
                    "unretained command memory",
                ));
            }
        }
        while let Some(storage) = self.submission_storage.front() {
            if !storage.point.completed()? {
                break;
            }
            self.submission_storage.pop_front();
        }
        let mut prepared = pushes.to_vec();
        let mut internal = Vec::new();
        if !pushes.is_empty() {
            if let Some((address, value)) = self.channel.progress_target()? {
                let bytes = self.abi.progress_release(address, value)?;
                if bytes.is_empty() || bytes.len() > 4096 {
                    return Err(Error::new(
                        ErrorKind::Invalid,
                        "progress command length",
                    ));
                }
                let binding = self.bind_host_data(&bytes)?;
                prepared.push(Push {
                    address: binding.address(),
                    size: bytes.len() as u32,
                    incomplete: false,
                    no_prefetch: false,
                });
                internal.push(binding);
            }
        }
        let mut payload = if prepared.is_empty() {
            None
        } else {
            let abi = &self.abi;
            let payload = if self.abi.requires_gpuva_gpfifo()
                || prepared.iter().any(|push| push.no_prefetch)
            {
                let entries = crate::nvidia::GpuFifoEntries::new(&prepared)?;
                let binding = self.bind_host_data(entries.as_bytes())?;
                let payload = abi.submit_gpuva(
                    self.private.as_bytes(),
                    &entries,
                    binding.address(),
                )?;
                internal.push(binding);
                payload
            } else {
                abi.submit(self.private.as_bytes(), &prepared)?
            };
            if payload.command_length == 0
                || !resources.iter().chain(internal.iter()).any(|r| {
                    r.contains(
                        payload.command_address,
                        payload.command_length.into(),
                    )
                })
            {
                return Err(Error::new(
                    ErrorKind::Invalid,
                    "ABI submit command range",
                ));
            }
            Some(payload)
        };
        let point = self.completion.reserve()?;
        if !internal.is_empty() {
            self.submission_storage.push_back(SubmissionStorage {
                point: point.clone(),
                _bindings: internal.clone(),
            });
        }
        let operation = (|| {
            for resource in resources.iter().chain(internal.iter()) {
                resource.memory().make_resident()?;
            }
            self.synchronize_paging()?;
            for wait in waits {
                self.enqueue_wait(wait)?;
            }
            if let Some(payload) = &mut payload {
                self.channel.submit(payload)?;
            }
            for signal in signals {
                self.channel.signal(signal, true)?;
            }
            self.enqueue_signal(&point)?;
            Ok(point)
        })();
        if operation.is_err() {
            self.poisoned = true;
        }
        operation
    }

    fn completed(&self, point: &FencePoint) -> Result<bool> {
        if !self.device.same_device(point.device()) {
            return Err(Error::new(ErrorKind::Invalid, "completion device"));
        }
        point.completed()
    }

    fn wait(&self, point: &FencePoint, timeout_ns: u64) -> Result<()> {
        point.wait(timeout_ns)
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        if let Some(last) = self.submission_storage.back() {
            if last.point.wait(5_000_000_000).is_err() && !self.device.is_lost()
            {
                for storage in self.submission_storage.drain(..) {
                    std::mem::forget(storage);
                }
            }
        }
    }
}
