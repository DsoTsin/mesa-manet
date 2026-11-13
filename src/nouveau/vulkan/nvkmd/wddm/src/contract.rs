use crate::error::{Error, ErrorKind, Result};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[repr(C)]
pub struct AdapterId {
    pub low: u32,
    pub high: i32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KernelBackend {
    NouveauDrm,
    NvidiaWddm,
    NvidiaRm,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Scheduling {
    VirtualContext,
    HardwareQueue,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DriverIdentity {
    pub adapter: AdapterId,
    pub vendor_id: u32,
    pub device_id: u32,
    pub driver_version: u64,
    pub wddm_version: u32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Capabilities {
    pub native_submit: bool,
    pub tiled_memory: bool,
    pub sparse_bind: bool,
    pub hardware_queue: bool,
    pub external_memory: bool,
    pub compression: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ByteRange {
    pub offset: u64,
    pub size: u64,
}

impl ByteRange {
    pub fn new(
        offset: u64,
        size: u64,
        bound: u64,
        alignment: u64,
    ) -> Result<Self> {
        if size == 0
            || !alignment.is_power_of_two()
            || offset % alignment != 0
            || size % alignment != 0
            || offset
                .checked_add(size)
                .filter(|end| *end <= bound)
                .is_none()
        {
            return Err(Error::new(ErrorKind::Invalid, "byte range"));
        }
        Ok(Self { offset, size })
    }

    pub fn end(self) -> u64 {
        self.offset + self.size
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MemoryRequest {
    pub size: u64,
    pub alignment: u64,
    pub host_visible: bool,
    pub local: bool,
    pub pte_kind: u8,
    pub tile_mode: u16,
}

impl MemoryRequest {
    pub fn validate(self) -> Result<Self> {
        ByteRange::new(0, self.size, self.size, self.alignment)?;
        if self.alignment < 4096 {
            return Err(Error::new(ErrorKind::Invalid, "allocation alignment"));
        }
        Ok(self)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(C)]
pub struct Push {
    pub address: u64,
    pub size: u32,
    pub incomplete: bool,
    pub no_prefetch: bool,
}

impl Push {
    pub fn validate(self) -> Result<Self> {
        if self.address == 0
            || self.address % 4 != 0
            || self.size == 0
            || self.size % 4 != 0
            || self.address.checked_add(self.size.into()).is_none()
        {
            return Err(Error::new(ErrorKind::Invalid, "push range"));
        }
        Ok(self)
    }
}

pub trait KernelDriver {
    type Device;
    fn backend(&self) -> KernelBackend;
    fn identity(&self) -> &DriverIdentity;
    fn capabilities(&self) -> Capabilities;
    fn create_device(&self) -> Result<Self::Device>;
}
