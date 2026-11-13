use crate::contract::{
    Capabilities, DriverIdentity, MemoryRequest, Push, Scheduling,
};
use crate::error::{Error, ErrorKind, Result};
use std::sync::Arc;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AbiKey {
    pub vendor_id: u32,
    pub device_id: u32,
    pub driver_version: u64,
    pub wddm_min: u32,
    pub wddm_max: u32,
}

impl AbiKey {
    pub fn validate(&self, identity: &DriverIdentity) -> Result<()> {
        if self.vendor_id != 0x10de
            || self.vendor_id != identity.vendor_id
            || self.device_id != identity.device_id
            || self.driver_version == 0
            || self.driver_version != identity.driver_version
            || self.wddm_min > self.wddm_max
            || !(self.wddm_min..=self.wddm_max).contains(&identity.wddm_version)
        {
            return Err(Error::new(
                ErrorKind::Unsupported,
                "NVIDIA private ABI identity",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct PrivateData {
    words: Vec<u64>,
    length: usize,
}

impl PrivateData {
    pub fn zeroed(length: usize) -> Result<Self> {
        if length > 1024 * 1024 {
            return Err(Error::new(ErrorKind::Invalid, "private payload size"));
        }
        let count = length.div_ceil(8);
        let mut words = Vec::new();
        words.try_reserve_exact(count).map_err(|_| {
            Error::new(ErrorKind::HostMemory, "private payload")
        })?;
        words.resize(count, 0);
        Ok(Self { words, length })
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let mut data = Self::zeroed(bytes.len())?;
        data.as_bytes_mut().copy_from_slice(bytes);
        Ok(data)
    }

    pub fn as_bytes(&self) -> &[u8] {
        unsafe {
            std::slice::from_raw_parts(self.words.as_ptr().cast(), self.length)
        }
    }

    pub fn as_bytes_mut(&mut self) -> &mut [u8] {
        unsafe {
            std::slice::from_raw_parts_mut(
                self.words.as_mut_ptr().cast(),
                self.length,
            )
        }
    }

    pub fn len_u32(&self) -> u32 {
        self.length as u32
    }
    pub fn is_empty(&self) -> bool {
        self.length == 0
    }

    pub fn write_u32(&mut self, offset: usize, value: u32) -> Result<()> {
        let end = offset.checked_add(4).ok_or_else(|| {
            Error::new(ErrorKind::Invalid, "ABI field offset")
        })?;
        self.as_bytes_mut()
            .get_mut(offset..end)
            .ok_or_else(|| Error::new(ErrorKind::Invalid, "ABI field bounds"))?
            .copy_from_slice(&value.to_le_bytes());
        Ok(())
    }

    pub fn write_u64(&mut self, offset: usize, value: u64) -> Result<()> {
        let end = offset.checked_add(8).ok_or_else(|| {
            Error::new(ErrorKind::Invalid, "ABI field offset")
        })?;
        self.as_bytes_mut()
            .get_mut(offset..end)
            .ok_or_else(|| Error::new(ErrorKind::Invalid, "ABI field bounds"))?
            .copy_from_slice(&value.to_le_bytes());
        Ok(())
    }

    pub fn read_u32(&self, offset: usize) -> Result<u32> {
        let end = offset.checked_add(4).ok_or_else(|| {
            Error::new(ErrorKind::Invalid, "ABI field offset")
        })?;
        let bytes = self.as_bytes().get(offset..end).ok_or_else(|| {
            Error::new(ErrorKind::Invalid, "ABI field bounds")
        })?;
        Ok(u32::from_le_bytes(bytes.try_into().map_err(|_| {
            Error::new(ErrorKind::Invalid, "ABI u32 field")
        })?))
    }

    pub fn read_u64(&self, offset: usize) -> Result<u64> {
        let end = offset.checked_add(8).ok_or_else(|| {
            Error::new(ErrorKind::Invalid, "ABI field offset")
        })?;
        let bytes = self.as_bytes().get(offset..end).ok_or_else(|| {
            Error::new(ErrorKind::Invalid, "ABI field bounds")
        })?;
        Ok(u64::from_le_bytes(bytes.try_into().map_err(|_| {
            Error::new(ErrorKind::Invalid, "ABI u64 field")
        })?))
    }
}

pub struct AllocationPayload {
    pub resource: PrivateData,
    pub allocation: PrivateData,
    pub create_flags: u32,
}

pub struct ContextPayload {
    pub private: PrivateData,
    pub node: u32,
    pub engine_affinity: u32,
    pub flags: u32,
}

pub struct SubmitPayload {
    pub private: PrivateData,
    pub command_address: u64,
    pub command_length: u32,
}

pub unsafe trait NvidiaAbi: Send + Sync {
    fn key(&self) -> AbiKey;
    fn capabilities(&self) -> Capabilities;
    fn allocation(&self, request: MemoryRequest) -> Result<AllocationPayload>;
    fn context(
        &self,
        engines: u32,
        scheduling: Scheduling,
    ) -> Result<ContextPayload>;
    fn context_profile(
        &self,
        _engines: u32,
    ) -> Result<Option<Arc<dyn NvidiaAbi>>> {
        Ok(None)
    }
    fn hardware_queue(
        &self,
        _context: &[u8],
        _subcontext_id: u8,
    ) -> Result<PrivateData> {
        Err(Error::new(
            ErrorKind::Unsupported,
            "hardware queue private ABI",
        ))
    }
    fn progress_release(&self, _address: u64, _value: u64) -> Result<Vec<u8>> {
        Err(Error::new(
            ErrorKind::Unsupported,
            "hardware queue progress release",
        ))
    }
    fn submit(&self, context: &[u8], pushes: &[Push]) -> Result<SubmitPayload>;
    fn requires_gpuva_gpfifo(&self) -> bool {
        false
    }
    fn submit_gpuva(
        &self,
        _context: &[u8],
        _entries: &crate::nvidia::GpuFifoEntries,
        _address: u64,
    ) -> Result<SubmitPayload> {
        Err(Error::new(
            ErrorKind::Unsupported,
            "GPUVA GPFIFO private ABI",
        ))
    }
}
