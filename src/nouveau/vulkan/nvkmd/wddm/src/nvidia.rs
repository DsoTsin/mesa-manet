use crate::abi::PrivateData;
use crate::error::{Error, ErrorKind, Result};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PrivateHeader {
    pub version: u32,
    pub size: u32,
    pub caller: u32,
}

impl PrivateHeader {
    pub const TAG: u32 = 0x4e56_4441;
    pub const NVGL: u32 = 0x4e56_474c;

    pub fn parse(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < 16 {
            return Err(Error::new(ErrorKind::Invalid, "NVIDIA header size"));
        }
        let mut fields = [0u32; 4];
        for (field, chunk) in fields.iter_mut().zip(bytes[..16].chunks_exact(4))
        {
            *field = u32::from_le_bytes(chunk.try_into().map_err(|_| {
                Error::new(ErrorKind::Invalid, "NVIDIA header word")
            })?);
        }
        if fields[0] != Self::TAG
            || fields[2] < 16
            || fields[2] as usize > bytes.len()
        {
            return Err(Error::new(
                ErrorKind::Invalid,
                "NVIDIA header bounds or tag",
            ));
        }
        Ok(Self {
            version: fields[1],
            size: fields[2],
            caller: fields[3],
        })
    }

    pub fn write(self, data: &mut PrivateData) -> Result<()> {
        if self.size < 16 || self.size != data.len_u32() {
            return Err(Error::new(
                ErrorKind::Invalid,
                "NVIDIA payload length",
            ));
        }
        data.write_u32(0, Self::TAG)?;
        data.write_u32(4, self.version)?;
        data.write_u32(8, self.size)?;
        data.write_u32(12, self.caller)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PrivateRecord {
    Allocation,
    VirtualContext,
    HardwareContext,
    HardwareQueue,
}

pub struct Nvgl61692Layout;

pub struct InlineSubmit;

pub struct GpuFifoEntries {
    data: PrivateData,
    count: u32,
}

impl GpuFifoEntries {
    pub fn new(pushes: &[crate::contract::Push]) -> Result<Self> {
        if pushes.is_empty()
            || pushes.last().is_some_and(|push| push.incomplete)
        {
            return Err(Error::new(ErrorKind::Invalid, "GPUVA submit batch"));
        }
        let count = pushes
            .len()
            .try_into()
            .map_err(|_| Error::new(ErrorKind::Invalid, "GPUVA push count"))?;
        let size = pushes.len().checked_mul(8).ok_or_else(|| {
            Error::new(ErrorKind::Invalid, "GPUVA GPFIFO size")
        })?;
        let mut data = PrivateData::zeroed(size)?;
        for (index, push) in pushes.iter().enumerate() {
            push.validate()?;
            if push.address + u64::from(push.size) > 1u64 << 40
                || push.size / 4 >= 1 << 21
            {
                return Err(Error::new(
                    ErrorKind::Unsupported,
                    "GPUVA GPFIFO address or length",
                ));
            }
            let entry = push.address
                | (1u64 << 41)
                | (u64::from(push.size / 4) << 42)
                | (u64::from(push.no_prefetch) << 63);
            data.write_u64(index * 8, entry)?;
        }
        Ok(Self { data, count })
    }

    pub fn as_bytes(&self) -> &[u8] {
        self.data.as_bytes()
    }

    pub fn submit(&self, address: u64) -> Result<crate::abi::SubmitPayload> {
        if address == 0
            || address % 8 != 0
            || address
                .checked_add(u64::from(self.data.len_u32()))
                .is_none()
        {
            return Err(Error::new(ErrorKind::Invalid, "GPUVA GPFIFO range"));
        }
        let mut private = PrivateData::zeroed(280)?;
        private.write_u32(0, PrivateHeader::TAG)?;
        private.write_u32(4, 0x10000)?;
        private.write_u32(12, 280)?;
        private.write_u32(20, self.count)?;
        private.write_u64(28, address)?;
        Ok(crate::abi::SubmitPayload {
            private,
            command_address: address,
            command_length: 1,
        })
    }
}

impl InlineSubmit {
    pub fn encode(
        pushes: &[crate::contract::Push],
    ) -> Result<crate::abi::SubmitPayload> {
        if pushes.is_empty()
            || pushes.last().is_some_and(|push| push.incomplete)
        {
            return Err(Error::new(ErrorKind::Invalid, "inline submit batch"));
        }
        let size = pushes
            .len()
            .checked_mul(8)
            .and_then(|n| n.checked_add(280))
            .ok_or_else(|| {
                Error::new(ErrorKind::Invalid, "inline submit size")
            })?;
        let mut private = PrivateData::zeroed(size)?;
        private.write_u32(0, PrivateHeader::TAG)?;
        private.write_u32(4, 0x10000)?;
        private.write_u32(12, 280)?;
        private.write_u32(
            16,
            pushes.len().try_into().map_err(|_| {
                Error::new(ErrorKind::Invalid, "inline push count")
            })?,
        )?;
        private.write_u32(20, pushes.len() as u32)?;
        for (index, push) in pushes.iter().enumerate() {
            push.validate()?;
            if push.address >= 1u64 << 40
                || push.size / 4 >= 1 << 21
                || push.no_prefetch
            {
                return Err(Error::new(
                    ErrorKind::Unsupported,
                    "WDDM inline GPFIFO address, length or flags",
                ));
            }
            let entry =
                push.address | (1u64 << 40) | (u64::from(push.size / 4) << 42);
            private.write_u64(280 + index * 8, entry)?;
        }
        Ok(crate::abi::SubmitPayload {
            private,
            command_address: pushes[0].address,
            command_length: 1,
        })
    }
}

impl Nvgl61692Layout {
    pub const DRIVER_VERSION: u64 = 0x0020_0000_0010_069c;
    pub const POLL_INFO_OFFSET: usize = 296;
    pub const CHANNEL_INFO_OFFSET: usize = 3440;
    pub const CONTEXT_REFLECTED_SKED_OFFSET: usize = 3472;
    pub const HARDWARE_QUEUE_REFLECTED_SKED_OFFSET: usize = 92;
    pub const NODE_STRIDE: usize = 40;
    pub const QUERY_SIZE: usize = 50616;
    pub const QUERY_NODE_OFFSET: usize = 27367;
    pub const QUERY_NODE_COUNT_OFFSET: usize = 28607;

    pub fn channel_info_request() -> Result<PrivateData> {
        let mut data = PrivateData::zeroed(125)?;
        PrivateHeader {
            version: 0x10002,
            size: 125,
            caller: 0x4e562a2a,
        }
        .write(&mut data)?;
        data.write_u32(16, 0x02000002)?;
        Ok(data)
    }

    pub fn query_request() -> Result<PrivateData> {
        let mut data = PrivateData::zeroed(Self::QUERY_SIZE)?;
        PrivateHeader {
            version: 0x10004,
            size: Self::QUERY_SIZE as u32,
            caller: 0x4e56_2a2a,
        }
        .write(&mut data)?;
        data.write_u32(16, 1)?;
        Ok(data)
    }

    pub fn expected(record: PrivateRecord) -> PrivateHeader {
        let (version, size) = match record {
            PrivateRecord::Allocation => (0x0001_0004, 586),
            PrivateRecord::VirtualContext => (0x0001_0004, 3573),
            PrivateRecord::HardwareContext => (0x0001_0000, 3204),
            PrivateRecord::HardwareQueue => (0x0001_0000, 180),
        };
        PrivateHeader {
            version,
            size,
            caller: PrivateHeader::NVGL,
        }
    }

    pub fn validate(record: PrivateRecord, bytes: &[u8]) -> Result<()> {
        if PrivateHeader::parse(bytes)? != Self::expected(record) {
            return Err(Error::new(
                ErrorKind::Unsupported,
                "NVIDIA 616.92 record layout",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NodeInfo {
    pub ordinal: u32,
    pub node_type: u32,
    pub os_managed_gpuva: bool,
    pub hardware_scheduling: bool,
    pub gpfifo_blit: bool,
}

impl NodeInfo {
    pub fn supports_gpuva_gpfifo(self) -> bool {
        self.os_managed_gpuva
            && self.hardware_scheduling
            && self.gpfifo_blit
            && matches!(self.node_type, 1 | 22 | 24)
    }
}

pub struct Nvgl61692Query {
    payload: PrivateData,
    nodes: Vec<NodeInfo>,
}

impl Nvgl61692Query {
    pub fn parse(payload: PrivateData) -> Result<Self> {
        let header = PrivateHeader::parse(payload.as_bytes())?;
        if header.version >> 16 != 1
            || header.size != Nvgl61692Layout::QUERY_SIZE as u32
            || header.caller == 0
        {
            return Err(Error::new(
                ErrorKind::Unsupported,
                "NVIDIA 616.92 adapter query header",
            ));
        }
        let word = |offset: usize| -> Result<u32> {
            let bytes = payload
                .as_bytes()
                .get(
                    offset..offset.checked_add(4).ok_or_else(|| {
                        Error::new(ErrorKind::Invalid, "QAI offset overflow")
                    })?,
                )
                .ok_or_else(|| {
                    Error::new(ErrorKind::Invalid, "QAI field bounds")
                })?;
            Ok(u32::from_le_bytes(
                bytes
                    .try_into()
                    .map_err(|_| Error::new(ErrorKind::Invalid, "QAI word"))?,
            ))
        };
        let count = word(Nvgl61692Layout::QUERY_NODE_COUNT_OFFSET)?;
        if word(28)? == u32::MAX {
            return Err(Error::new(
                ErrorKind::Unsupported,
                "QAI device instance",
            ));
        }
        if count == 0 || count > 31 {
            return Err(Error::new(ErrorKind::Unsupported, "QAI node count"));
        }
        let mut nodes = Vec::with_capacity(count as usize);
        for ordinal in 0..count {
            let offset = Nvgl61692Layout::QUERY_NODE_OFFSET
                + ordinal as usize * Nvgl61692Layout::NODE_STRIDE;
            let flags =
                [word(offset + 4)?, word(offset + 8)?, word(offset + 12)?];
            if flags.iter().any(|v| *v > 1) {
                return Err(Error::new(
                    ErrorKind::Unsupported,
                    "QAI node boolean layout",
                ));
            }
            nodes.push(NodeInfo {
                ordinal,
                node_type: word(offset)?,
                os_managed_gpuva: flags[0] != 0,
                hardware_scheduling: flags[1] != 0,
                gpfifo_blit: flags[2] != 0,
            });
        }
        Ok(Self { payload, nodes })
    }

    pub fn nodes(&self) -> &[NodeInfo] {
        &self.nodes
    }
    pub fn payload(&self) -> &[u8] {
        self.payload.as_bytes()
    }
}
