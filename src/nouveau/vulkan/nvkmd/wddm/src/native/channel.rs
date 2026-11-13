use super::{Device, FencePoint};
use crate::abi::{ContextPayload, NvidiaAbi, PrivateData, SubmitPayload};
use crate::contract::Scheduling;
use crate::error::{Error, ErrorKind, Result};
use std::sync::{
    atomic::{AtomicBool, AtomicU64, Ordering},
    Arc,
};
use windows::Wdk::Graphics::Direct3D::*;

pub(super) struct HardwareContext {
    _device: Device,
    _abi: Arc<dyn NvidiaAbi>,
    handle: u32,
    private: PrivateData,
    node: u32,
    engine_affinity: u32,
    graphics_claimed: AtomicBool,
    free_subcontexts: AtomicU64,
}

impl HardwareContext {
    fn new(
        device: &Device,
        payload: &mut ContextPayload,
        abi: Arc<dyn NvidiaAbi>,
    ) -> Result<Arc<Self>> {
        let mut private = PrivateData::from_bytes(payload.private.as_bytes())?;
        let mut request = D3DKMT_CREATECONTEXTVIRTUAL {
            hDevice: device.0.handle,
            NodeOrdinal: payload.node,
            EngineAffinity: payload.engine_affinity,
            PrivateDriverDataSize: private.len_u32(),
            pPrivateDriverData: private.as_bytes_mut().as_mut_ptr().cast(),
            ClientHint: D3DKMT_CLIENTHINT_VULKAN,
            ..Default::default()
        };
        request.Flags.Anonymous.Value = payload.flags | 0x10;
        device.0.check(
            "D3DKMTCreateContextVirtual hardware queue",
            unsafe { D3DKMTCreateContextVirtual(&mut request) }.0,
        )?;
        payload
            .private
            .as_bytes_mut()
            .copy_from_slice(private.as_bytes());
        Ok(Arc::new(Self {
            _device: device.clone(),
            _abi: abi,
            handle: request.hContext,
            private,
            node: payload.node,
            engine_affinity: payload.engine_affinity,
            graphics_claimed: AtomicBool::new(false),
            free_subcontexts: AtomicU64::new(!1),
        }))
    }

    fn acquire(
        device: &Device,
        payload: &mut ContextPayload,
        abi: Arc<dyn NvidiaAbi>,
        compute: bool,
        shared: bool,
    ) -> Result<(Arc<Self>, u8)> {
        let owner = if shared {
            let mut primary =
                device.0.primary_hardware_context.lock().map_err(|_| {
                    Error::new(
                        ErrorKind::System,
                        "primary hardware context lock",
                    )
                })?;
            match primary.upgrade() {
                Some(owner) => {
                    if owner.node != payload.node
                        || owner.engine_affinity != payload.engine_affinity
                    {
                        if compute {
                            return Err(Error::new(
                                ErrorKind::Unsupported,
                                "primary hardware context engine",
                            ));
                        }
                        Self::new(device, payload, abi.clone())?
                    } else {
                        owner
                    }
                }
                None => {
                    let owner = Self::new(device, payload, abi.clone())?;
                    *primary = Arc::downgrade(&owner);
                    owner
                }
            }
        } else {
            Self::new(device, payload, abi.clone())?
        };
        if compute && shared {
            let mut available = owner.free_subcontexts.load(Ordering::Acquire);
            loop {
                if available == 0 {
                    return Err(Error::new(
                        ErrorKind::Busy,
                        "hardware compute subcontexts exhausted",
                    ));
                }
                let subcontext = available.trailing_zeros() as u8;
                let remaining = available & !(1u64 << subcontext);
                match owner.free_subcontexts.compare_exchange_weak(
                    available,
                    remaining,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                ) {
                    Ok(_) => return Ok((owner, subcontext)),
                    Err(current) => available = current,
                }
            }
        }
        if owner
            .graphics_claimed
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
        {
            return Ok((owner, 0));
        }
        let owner = Self::new(device, payload, abi)?;
        owner.graphics_claimed.store(true, Ordering::Release);
        Ok((owner, 0))
    }

    fn release(&self, subcontext: u8) {
        if subcontext == 0 {
            self.graphics_claimed.store(false, Ordering::Release);
        } else {
            self.free_subcontexts
                .fetch_or(1u64 << subcontext, Ordering::Release);
        }
    }
}

impl Drop for HardwareContext {
    fn drop(&mut self) {
        unsafe {
            let _ = D3DKMTDestroyContext(&D3DKMT_DESTROYCONTEXT {
                hContext: self.handle,
            });
        }
    }
}

enum ChannelHandle {
    Virtual(u32),
    Hardware {
        context: u32,
        owner: Arc<HardwareContext>,
        subcontext_id: u8,
        queue: u32,
        progress: u64,
        progress_handle: u32,
        progress_cpu: usize,
        progress_gpu: u64,
    },
}

pub(crate) struct Channel {
    device: Device,
    handle: ChannelHandle,
    reflected_sked_address: u64,
}

impl Channel {
    fn cached_reflected_sked_address(
        device: &Device,
        data: &PrivateData,
        offset: usize,
    ) -> u64 {
        let identity = device.adapter().identity();
        if identity.vendor_id == 0x10de
            && identity.driver_version
                == crate::nvidia::Nvgl61692Layout::DRIVER_VERSION
        {
            data.read_u64(offset).unwrap_or(0)
        } else {
            0
        }
    }

    pub(crate) fn reflected_sked_address(&self) -> Result<u64> {
        self.device.0.ready()?;
        if self.reflected_sked_address != 0 {
            return Ok(self.reflected_sked_address);
        }
        self.query_reflected_sked_address()
    }

    pub(crate) fn query_reflected_sked_address(&self) -> Result<u64> {
        self.device.0.ready()?;
        let identity = self.device.adapter().identity();
        if identity.vendor_id != 0x10de
            || identity.driver_version
                != crate::nvidia::Nvgl61692Layout::DRIVER_VERSION
        {
            return Err(Error::new(
                ErrorKind::Unsupported,
                "NVIDIA 616.92 reflected SKED query",
            ));
        }
        let context = match self.handle {
            ChannelHandle::Virtual(context)
            | ChannelHandle::Hardware { context, .. } => context,
        };
        let mut data = crate::nvidia::Nvgl61692Layout::channel_info_request()?;
        let mut request = D3DKMT_ESCAPE {
            hAdapter: self.device.adapter().handle(),
            hDevice: self.device.0.handle,
            hContext: context,
            Type: D3DKMT_ESCAPE_DRIVERPRIVATE,
            pPrivateDriverData: data.as_bytes_mut().as_mut_ptr().cast(),
            PrivateDriverDataSize: data.len_u32(),
            ..Default::default()
        };
        request.Flags.Anonymous.Value = 8;
        let status = unsafe { D3DKMTEscape(&request) }.0;
        if status < 0 {
            let error =
                Error::from_status("reflected SKED channel info", status);
            return Err(Error {
                kind: if error.kind == ErrorKind::DeviceLost {
                    ErrorKind::DeviceLost
                } else {
                    ErrorKind::Unsupported
                },
                ..error
            });
        }
        let address = data.read_u64(80)?;
        if address == 0 {
            return Err(Error::new(
                ErrorKind::Unsupported,
                "zero reflected SKED channel address",
            ));
        }
        Ok(address)
    }

    pub(crate) fn progress_target(&self) -> Result<Option<(u64, u64)>> {
        match self.handle {
            ChannelHandle::Virtual(_) => Ok(None),
            ChannelHandle::Hardware {
                progress,
                progress_handle,
                progress_gpu,
                ..
            } => {
                if progress_handle == 0 || progress_gpu == 0 {
                    return Err(Error::new(
                        ErrorKind::Invalid,
                        "hardware progress mapping",
                    ));
                }
                let value = progress.checked_add(1).ok_or_else(|| {
                    Error::new(ErrorKind::Invalid, "hardware progress overflow")
                })?;
                Ok(Some((progress_gpu, value)))
            }
        }
    }

    pub(crate) fn progress_values(&self) -> Result<Option<(u64, u64)>> {
        match self.handle {
            ChannelHandle::Virtual(_) => Ok(None),
            ChannelHandle::Hardware {
                progress,
                progress_cpu,
                ..
            } => {
                self.device.0.ready()?;
                let completed = unsafe {
                    (&*(progress_cpu as *const std::sync::atomic::AtomicU64))
                        .load(std::sync::atomic::Ordering::Acquire)
                };
                if completed == u64::MAX {
                    self.device.0.check_execution_state()?;
                }
                Ok(Some((progress, completed)))
            }
        }
    }
    pub(crate) fn present(
        &mut self,
        request: &mut D3DKMT_PRESENT,
    ) -> Result<i32> {
        let status = match self.handle {
            ChannelHandle::Virtual(context) => {
                request.Anonymous1.hContext = context;
                unsafe { D3DKMTPresent(request) }.0
            }
            ChannelHandle::Hardware { context, queue, .. } => {
                request.Anonymous1.hContext = context;
                let mut queue_handle = queue;
                let mut submit = D3DKMT_SUBMITPRESENTTOHWQUEUE {
                    hHwQueues: &mut queue_handle,
                    PrivatePresentData: *request,
                };
                let status =
                    unsafe { D3DKMTSubmitPresentToHwQueue(&mut submit) }.0;
                *request = submit.PrivatePresentData;
                status
            }
        };
        self.device.0.check("WDDM redirected present", status)?;
        Ok(status)
    }

    pub fn new(
        device: &Device,
        scheduling: Scheduling,
        payload: &mut ContextPayload,
        abi: Arc<dyn NvidiaAbi>,
        compute: bool,
    ) -> Result<Self> {
        match scheduling {
            Scheduling::VirtualContext => {
                let mut request = D3DKMT_CREATECONTEXTVIRTUAL {
                    hDevice: device.0.handle,
                    NodeOrdinal: payload.node,
                    EngineAffinity: payload.engine_affinity,
                    PrivateDriverDataSize: payload.private.len_u32(),
                    pPrivateDriverData: payload
                        .private
                        .as_bytes_mut()
                        .as_mut_ptr()
                        .cast(),
                    ClientHint: D3DKMT_CLIENTHINT_VULKAN,
                    ..Default::default()
                };
                request.Flags.Anonymous.Value = payload.flags;
                device.0.check(
                    "D3DKMTCreateContextVirtual",
                    unsafe {
                        D3DKMTCreateContextVirtual(std::ptr::addr_of_mut!(
                            request
                        ))
                    }
                    .0,
                )?;
                Ok(Self {
                    device: device.clone(),
                    handle: ChannelHandle::Virtual(request.hContext),
                    reflected_sked_address: Self::cached_reflected_sked_address(
                        device,
                        &payload.private,
                        crate::nvidia::Nvgl61692Layout::CONTEXT_REFLECTED_SKED_OFFSET,
                    ),
                })
            }
            Scheduling::HardwareQueue => {
                if !abi.capabilities().hardware_queue {
                    return Err(Error::new(
                        ErrorKind::Unsupported,
                        "hardware queue capability",
                    ));
                }
                let metadata =
                    abi.hardware_queue(payload.private.as_bytes(), 0)?;
                let shared =
                    matches!(metadata.read_u32(16)? & 0x300, 0x100 | 0x200);
                let (owner, subcontext_id) = HardwareContext::acquire(
                    device,
                    payload,
                    abi.clone(),
                    compute,
                    shared,
                )?;
                let context = owner.handle;
                let mut channel = Self {
                    device: device.clone(),
                    reflected_sked_address: 0,
                    handle: ChannelHandle::Hardware {
                        context,
                        owner: owner.clone(),
                        subcontext_id,
                        queue: 0,
                        progress: 0,
                        progress_handle: 0,
                        progress_cpu: 0,
                        progress_gpu: 0,
                    },
                };
                payload.private =
                    PrivateData::from_bytes(owner.private.as_bytes())?;
                let mut private = abi
                    .hardware_queue(owner.private.as_bytes(), subcontext_id)?;
                let queue_flags = private.read_u32(16)?;
                let mut queue = D3DKMT_CREATEHWQUEUE {
                    hHwContext: context,
                    PrivateDriverDataSize: private.len_u32(),
                    pPrivateDriverData: private
                        .as_bytes_mut()
                        .as_mut_ptr()
                        .cast(),
                    ..Default::default()
                };
                device.0.check(
                    "D3DKMTCreateHwQueue",
                    unsafe { D3DKMTCreateHwQueue(&mut queue) }.0,
                )?;
                channel.handle = ChannelHandle::Hardware {
                    context,
                    owner,
                    subcontext_id,
                    queue: queue.hHwQueue,
                    progress: 0,
                    progress_handle: queue.hHwQueueProgressFence,
                    progress_cpu: queue.HwQueueProgressFenceCPUVirtualAddress
                        as usize,
                    progress_gpu: queue.HwQueueProgressFenceGPUVirtualAddress,
                };
                channel.reflected_sked_address = Self::cached_reflected_sked_address(
                    device,
                    &private,
                    crate::nvidia::Nvgl61692Layout::HARDWARE_QUEUE_REFLECTED_SKED_OFFSET,
                );
                if queue.hHwQueueProgressFence == 0
                    || queue.HwQueueProgressFenceCPUVirtualAddress.is_null()
                    || (queue.HwQueueProgressFenceCPUVirtualAddress as usize)
                        % 8
                        != 0
                    || queue.HwQueueProgressFenceGPUVirtualAddress == 0
                {
                    return Err(Error::new(
                        ErrorKind::Invalid,
                        "hardware progress fence outputs",
                    ));
                }
                if std::env::var_os("NVK_WDDM_DEBUG").is_some() {
                    eprintln!(
                        "NVK WDDM hardware queue context=0x{:x} queue=0x{:x} compute={} subcontext={} flags=0x{:x}",
                        context,
                        queue.hHwQueue,
                        compute,
                        subcontext_id,
                        queue_flags,
                    );
                }
                Ok(channel)
            }
        }
    }

    pub fn wait(&self, point: &FencePoint) -> Result<()> {
        if !self.device.same_device(point.device()) {
            return Err(Error::new(ErrorKind::Invalid, "cross-device fence"));
        }
        if !point.wait_required() {
            return Ok(());
        }
        let handle = point.handle();
        match self.handle {
            ChannelHandle::Virtual(context) => {
                let mut request = D3DKMT_WAITFORSYNCHRONIZATIONOBJECTFROMGPU {
                    hContext: context,
                    ObjectCount: 1,
                    ObjectHandleArray: &handle,
                    ..Default::default()
                };
                request.Anonymous.MonitoredFenceValueArray = &point.value;
                self.device.0.check(
                    "D3DKMTWaitForSynchronizationObjectFromGpu",
                    unsafe {
                        D3DKMTWaitForSynchronizationObjectFromGpu(&request)
                    }
                    .0,
                )
            }
            ChannelHandle::Hardware { queue, .. } => {
                let request = D3DKMT_SUBMITWAITFORSYNCOBJECTSTOHWQUEUE {
                    hHwQueue: queue,
                    ObjectCount: 1,
                    ObjectHandleArray: &handle,
                    FenceValueArray: &point.value,
                };
                self.device.0.check(
                    "D3DKMTSubmitWaitForSyncObjectsToHwQueue",
                    unsafe {
                        D3DKMTSubmitWaitForSyncObjectsToHwQueue(&request)
                    }
                    .0,
                )
            }
        }
    }

    pub fn signal(&self, point: &FencePoint, allow_rewind: bool) -> Result<()> {
        if !self.device.same_device(point.device()) {
            return Err(Error::new(
                ErrorKind::Invalid,
                "cross-device GPU signal",
            ));
        }
        let handle = point.handle();
        match self.handle {
            ChannelHandle::Virtual(context) => {
                let mut request = D3DKMT_SIGNALSYNCHRONIZATIONOBJECTFROMGPU2 {
                    ObjectCount: 1,
                    ObjectHandleArray: &handle,
                    BroadcastContextCount: 1,
                    BroadcastContextArray: &context,
                    ..Default::default()
                };
                request.Flags.Anonymous.Value =
                    if allow_rewind { 4 } else { 0 };
                request.Anonymous.MonitoredFenceValueArray = &point.value;
                self.device.0.check(
                    "D3DKMTSignalSynchronizationObjectFromGpu2",
                    unsafe {
                        D3DKMTSignalSynchronizationObjectFromGpu2(&request)
                    }
                    .0,
                )
            }
            ChannelHandle::Hardware { queue, .. } => {
                let mut request = D3DKMT_SUBMITSIGNALSYNCOBJECTSTOHWQUEUE {
                    BroadcastHwQueueCount: 1,
                    BroadcastHwQueueArray: &queue,
                    ObjectCount: 1,
                    ObjectHandleArray: &handle,
                    FenceValueArray: &point.value,
                    ..Default::default()
                };
                request.Flags.Anonymous.Value =
                    if allow_rewind { 4 } else { 0 };
                self.device.0.check(
                    "D3DKMTSubmitSignalSyncObjectsToHwQueue",
                    unsafe { D3DKMTSubmitSignalSyncObjectsToHwQueue(&request) }
                        .0,
                )
            }
        }
    }

    pub fn submit(&mut self, payload: &mut SubmitPayload) -> Result<()> {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            std::arch::x86_64::_mm_sfence();
        }
        #[cfg(target_arch = "x86")]
        unsafe {
            std::arch::x86::_mm_sfence();
        }
        match &mut self.handle {
            ChannelHandle::Virtual(context) => {
                let mut request = D3DKMT_SUBMITCOMMAND {
                    Commands: payload.command_address,
                    CommandLength: payload.command_length,
                    BroadcastContextCount: 1,
                    PrivateDriverDataSize: payload.private.len_u32(),
                    pPrivateDriverData: payload
                        .private
                        .as_bytes_mut()
                        .as_mut_ptr()
                        .cast(),
                    ..Default::default()
                };
                request.BroadcastContext[0] = *context;
                self.device.0.check(
                    "D3DKMTSubmitCommand",
                    unsafe { D3DKMTSubmitCommand(&request) }.0,
                )
            }
            ChannelHandle::Hardware {
                queue, progress, ..
            } => {
                *progress = progress.checked_add(1).ok_or_else(|| {
                    Error::new(
                        ErrorKind::Invalid,
                        "hardware queue progress overflow",
                    )
                })?;
                let request = D3DKMT_SUBMITCOMMANDTOHWQUEUE {
                    hHwQueue: *queue,
                    HwQueueProgressFenceId: *progress,
                    CommandBuffer: payload.command_address,
                    CommandLength: payload.command_length,
                    PrivateDriverDataSize: payload.private.len_u32(),
                    pPrivateDriverData: payload
                        .private
                        .as_bytes_mut()
                        .as_mut_ptr()
                        .cast(),
                    ..Default::default()
                };
                self.device.0.check(
                    "D3DKMTSubmitCommandToHwQueue",
                    unsafe { D3DKMTSubmitCommandToHwQueue(&request) }.0,
                )
            }
        }
    }
}

impl Drop for Channel {
    fn drop(&mut self) {
        unsafe {
            match &self.handle {
                ChannelHandle::Virtual(context) => {
                    let _ = D3DKMTDestroyContext(&D3DKMT_DESTROYCONTEXT {
                        hContext: *context,
                    });
                }
                ChannelHandle::Hardware {
                    owner,
                    subcontext_id,
                    queue,
                    ..
                } => {
                    if *queue != 0 {
                        let _ = D3DKMTDestroyHwQueue(&D3DKMT_DESTROYHWQUEUE {
                            hHwQueue: *queue,
                        });
                    }
                    owner.release(*subcontext_id);
                }
            }
        }
    }
}
