use super::Adapter;
use crate::abi::{
    AbiKey, AllocationPayload, ContextPayload, NvidiaAbi, PrivateData,
    SubmitPayload,
};
use crate::contract::{Capabilities, MemoryRequest, Push, Scheduling};
use crate::error::{Error, ErrorKind, Result};
use crate::nvidia::{NodeInfo, Nvgl61692Layout, Nvgl61692Query, PrivateRecord};
use std::sync::Arc;
use windows::Win32::Foundation::{CloseHandle, HANDLE};
use windows::Win32::System::Threading::{CreateEventW, GetCurrentProcessId};

struct ResetEvents {
    engine: usize,
    adapter: usize,
}

impl ResetEvents {
    fn new() -> Result<Self> {
        let engine =
            unsafe { CreateEventW(None, true, false, None) }.map_err(|_| {
                Error::new(ErrorKind::HostMemory, "engine reset event")
            })?;
        let mut events = Self {
            engine: engine.0 as usize,
            adapter: 0,
        };
        let adapter = unsafe { CreateEventW(None, true, false, None) }
            .map_err(|_| {
                Error::new(ErrorKind::HostMemory, "adapter reset event")
            })?;
        events.adapter = adapter.0 as usize;
        Ok(events)
    }
}

impl Drop for ResetEvents {
    fn drop(&mut self) {
        unsafe {
            let _ = CloseHandle(HANDLE(self.engine as *mut _));
            if self.adapter != 0 {
                let _ = CloseHandle(HANDLE(self.adapter as *mut _));
            }
        }
    }
}

pub struct Nvgl61692Abi {
    key: AbiKey,
    node: NodeInfo,
    context_id: u32,
    events: ResetEvents,
}

impl Nvgl61692Abi {
    pub fn new(
        adapter: &Adapter,
        query: &Nvgl61692Query,
        node_ordinal: u32,
        context_id: u32,
    ) -> Result<Self> {
        let identity = adapter.identity();
        if identity.vendor_id != 0x10de
            || identity.driver_version != Nvgl61692Layout::DRIVER_VERSION
        {
            return Err(Error::new(
                ErrorKind::Unsupported,
                "NVIDIA 616.92 context profile",
            ));
        }
        let node = *query
            .nodes()
            .iter()
            .find(|n| n.ordinal == node_ordinal)
            .ok_or_else(|| {
                Error::new(ErrorKind::Invalid, "NVIDIA node ordinal")
            })?;
        if !node.os_managed_gpuva
            || !matches!(
                node.node_type,
                1 | 11 | 12 | 13 | 14 | 15 | 16 | 22 | 23 | 24
            )
        {
            return Err(Error::new(
                ErrorKind::Unsupported,
                "NVIDIA node class",
            ));
        }
        Ok(Self {
            key: AbiKey {
                vendor_id: identity.vendor_id,
                device_id: identity.device_id,
                driver_version: identity.driver_version,
                wddm_min: identity.wddm_version,
                wddm_max: identity.wddm_version,
            },
            node,
            context_id,
            events: ResetEvents::new()?,
        })
    }

    pub fn node(&self) -> NodeInfo {
        self.node
    }

    fn poll_info(&self, data: &mut PrivateData, offset: usize) -> Result<()> {
        data.write_u32(offset, 256)?;
        data.write_u32(offset + 3108, unsafe { GetCurrentProcessId() })?;
        let executable = std::env::current_exe()
            .map_err(|_| Error::new(ErrorKind::System, "process executable"))?;
        let filename = executable
            .file_name()
            .ok_or_else(|| {
                Error::new(ErrorKind::Invalid, "executable filename")
            })?
            .to_string_lossy();
        let directory = std::env::current_dir().map_err(|_| {
            Error::new(ErrorKind::System, "process working directory")
        })?;
        self.write_utf8(data, offset + 12, 256, &filename)?;
        self.write_utf16(data, offset + 268, 256, &filename)?;
        self.write_utf8(
            data,
            offset + 780,
            520,
            &executable.to_string_lossy(),
        )?;
        self.write_utf8(
            data,
            offset + 1300,
            520,
            &directory.to_string_lossy(),
        )?;
        Ok(())
    }

    fn write_utf8(
        &self,
        data: &mut PrivateData,
        offset: usize,
        capacity: usize,
        text: &str,
    ) -> Result<()> {
        let mut length = text.len().min(capacity - 1);
        while !text.is_char_boundary(length) {
            length -= 1;
        }
        let target = data
            .as_bytes_mut()
            .get_mut(offset..offset + capacity)
            .ok_or_else(|| {
                Error::new(ErrorKind::Invalid, "poll string bounds")
            })?;
        target.fill(0);
        target[..length].copy_from_slice(&text.as_bytes()[..length]);
        Ok(())
    }

    fn write_utf16(
        &self,
        data: &mut PrivateData,
        offset: usize,
        capacity: usize,
        text: &str,
    ) -> Result<()> {
        let target = data
            .as_bytes_mut()
            .get_mut(offset..offset + capacity * 2)
            .ok_or_else(|| {
                Error::new(ErrorKind::Invalid, "poll UTF16 bounds")
            })?;
        target.fill(0);
        for (out, unit) in target
            .chunks_exact_mut(2)
            .take(capacity - 1)
            .zip(text.encode_utf16())
        {
            out.copy_from_slice(&unit.to_le_bytes());
        }
        Ok(())
    }
}

unsafe impl NvidiaAbi for Nvgl61692Abi {
    fn key(&self) -> AbiKey {
        self.key
    }
    fn capabilities(&self) -> Capabilities {
        Capabilities {
            native_submit: self.node.hardware_scheduling,
            hardware_queue: self.node.hardware_scheduling,
            ..Default::default()
        }
    }
    fn allocation(&self, _request: MemoryRequest) -> Result<AllocationPayload> {
        Err(Error::new(
            ErrorKind::Unsupported,
            "NVIDIA allocation attribute encoding",
        ))
    }
    fn progress_release(&self, address: u64, value: u64) -> Result<Vec<u8>> {
        if address == 0 || address % 8 != 0 || value == 0 {
            return Err(Error::new(
                ErrorKind::Invalid,
                "hardware progress target",
            ));
        }
        let mut push = crate::command::PushBuffer::new();
        match self.node.node_type {
            11..=16 => {
                push.method(
                    4,
                    0x240,
                    &[
                        (address >> 32) as u32,
                        address as u32,
                        value as u32,
                        (value >> 32) as u32,
                    ],
                )?;
                push.method(4, 0x300, &[0x0800004c])?;
            }
            24 => {
                push.method(1, 0x110, &[0])?;
                push.method(
                    1,
                    0x158,
                    &[
                        value as u32,
                        (value >> 32) as u32,
                        address as u32,
                        (address >> 32) as u32,
                        0x1030,
                    ],
                )?;
                push.method(1, 0x168, &[0x2003])?;
            }
            1 | 22 | 23 => {
                push.method(1, 0x110, &[0])?;
                push.method(0, 0x1144, &[0])?;
                push.method(
                    0,
                    0x158,
                    &[
                        value as u32,
                        (value >> 32) as u32,
                        address as u32,
                        (address >> 32) as u32,
                        0x0800403c,
                    ],
                )?;
                push.method(0, 0x168, &[0x1000003f])?;
            }
            _ => {
                return Err(Error::new(
                    ErrorKind::Unsupported,
                    "progress engine",
                ))
            }
        }
        Ok(push.bytes())
    }
    fn context(
        &self,
        engines: u32,
        scheduling: Scheduling,
    ) -> Result<ContextPayload> {
        let copy = matches!(self.node.node_type, 11..=16);
        let gr_engines = 4 | 16 | u32::from(self.node.gpfifo_blit);
        if (copy && engines != 1)
            || (!copy && engines & !gr_engines != 0)
            || self.node.hardware_scheduling
                != (scheduling == Scheduling::HardwareQueue)
        {
            return Err(Error::new(
                ErrorKind::Unsupported,
                "NVIDIA node scheduling or engine mask",
            ));
        }
        let record = if self.node.hardware_scheduling {
            PrivateRecord::HardwareContext
        } else {
            PrivateRecord::VirtualContext
        };
        let header = Nvgl61692Layout::expected(record);
        let mut data = PrivateData::zeroed(header.size as usize)?;
        header.write(&mut data)?;
        data.write_u32(16, 3)?;
        if self.node.hardware_scheduling {
            data.write_u32(20, self.context_id)?;
            data.write_u32(24, 1)?;
            self.poll_info(&mut data, 28)?;
            data.write_u64(3144, self.events.engine as u64)?;
            data.write_u64(3152, self.events.adapter as u64)?;
            data.write_u64(3160, u64::MAX)?;
        } else {
            data.write_u32(20, 0x0001_400c)?;
            data.write_u32(28, 2048)?;
            data.write_u32(32, self.context_id)?;
            data.write_u32(36, 12288)?;
            self.poll_info(&mut data, Nvgl61692Layout::POLL_INFO_OFFSET)?;
            data.write_u64(3521, self.events.engine as u64)?;
            data.write_u64(3529, self.events.adapter as u64)?;
            data.write_u64(3537, u64::MAX)?;
        }
        Ok(ContextPayload {
            private: data,
            node: self.node.ordinal,
            engine_affinity: 1,
            flags: if self.node.hardware_scheduling {
                0x10
            } else {
                0
            },
        })
    }
    fn context_profile(
        &self,
        engines: u32,
    ) -> Result<Option<Arc<dyn NvidiaAbi>>> {
        if self.node.hardware_scheduling
            && matches!(self.node.node_type, 1 | 22 | 23)
            && engines & 16 != 0
            && engines & 4 == 0
        {
            return Ok(Some(Arc::new(Self {
                key: self.key,
                node: NodeInfo {
                    node_type: 24,
                    ..self.node
                },
                context_id: self.context_id,
                events: ResetEvents::new()?,
            })));
        }
        Ok(None)
    }

    fn hardware_queue(
        &self,
        _context: &[u8],
        subcontext_id: u8,
    ) -> Result<PrivateData> {
        let header = Nvgl61692Layout::expected(PrivateRecord::HardwareQueue);
        let mut data = PrivateData::zeroed(header.size as usize)?;
        header.write(&mut data)?;
        let flags = match self.node.node_type {
            1 | 22 | 23 => 0x100,
            24 => {
                0x200
                    | u32::from(subcontext_id)
                    | if subcontext_id != 0 { 0x400 } else { 0 }
            }
            _ => 0,
        };
        data.write_u32(16, flags)?;
        Ok(data)
    }
    fn submit(
        &self,
        _context: &[u8],
        pushes: &[Push],
    ) -> Result<SubmitPayload> {
        if !self.node.hardware_scheduling {
            return Err(Error::new(
                ErrorKind::Unsupported,
                "ordinary context submit metadata encoding",
            ));
        }
        crate::nvidia::InlineSubmit::encode(pushes)
    }
    fn requires_gpuva_gpfifo(&self) -> bool {
        self.node.hardware_scheduling && self.node.node_type == 24
    }
    fn submit_gpuva(
        &self,
        _context: &[u8],
        entries: &crate::nvidia::GpuFifoEntries,
        address: u64,
    ) -> Result<SubmitPayload> {
        if !self.node.supports_gpuva_gpfifo() {
            return Err(Error::new(
                ErrorKind::Unsupported,
                "NVIDIA GPUVA GPFIFO node",
            ));
        }
        entries.submit(address)
    }
}
