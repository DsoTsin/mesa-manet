use super::Device;
use crate::error::{Error, ErrorKind, Result};

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct DeviceInfo {
    pub valid_fields: u64,
    pub vram_size: u64,
    pub bar_size: u64,
    pub vendor_id: u32,
    pub device_id: u32,
    pub chipset: u32,
    pub sm: u32,
    pub gpc_count: u32,
    pub tpc_count: u32,
    pub mp_per_tpc: u32,
    pub max_warps_per_mp: u32,
    pub max_blocks_per_mp: u32,
    pub cls_copy: u32,
    pub cls_eng2d: u32,
    pub cls_eng3d: u32,
    pub cls_m2mf: u32,
    pub cls_compute: u32,
    pub cls_gpfifo: u32,
    pub device_name: [u8; 64],
    pub chipset_name: [u8; 16],
}

impl DeviceInfo {
    pub fn query(device: &Device) -> Result<Self> {
        device.0.ready()?;
        let adapter = device.adapter();
        let query = adapter.query_nvidia_61692()?;
        let bytes = query.payload();
        let word = |offset: usize| -> Result<u32> {
            bytes
                .get(offset..offset + 4)
                .and_then(|v| v.try_into().ok())
                .map(u32::from_le_bytes)
                .ok_or_else(|| {
                    Error::new(ErrorKind::Invalid, "device info field")
                })
        };
        let architecture = word(236)?;
        let implementation = word(240)?;
        if architecture != 0x1b0 || implementation != 3 {
            return Err(Error::new(
                ErrorKind::Unsupported,
                "NVIDIA topology profile",
            ));
        }
        let gr = |index: u32| -> Result<u32> {
            for entry in 0..63 {
                let offset = 491 + entry * 8;
                if word(offset)? == index {
                    return word(offset + 4);
                }
            }
            Err(Error::new(ErrorKind::Unsupported, "missing NVIDIA GR info"))
        };
        let count = word(45400)? as usize;
        if count == 0 || count > 512 || 45404 + count * 4 > bytes.len() {
            return Err(Error::new(ErrorKind::Invalid, "NVIDIA class count"));
        }
        let class = |value: u32| -> Result<u32> {
            for index in 0..count {
                if word(45404 + index * 4)? == value {
                    return Ok(value);
                }
            }
            Err(Error::new(
                ErrorKind::Unsupported,
                "missing NVIDIA hardware class",
            ))
        };
        let gpc_count = gr(7)?;
        let tpc_count = gr(9)?;
        let mp_per_tpc = gr(32)?;
        let max_warps_per_mp = gr(13)?;
        if gpc_count == 0
            || gpc_count > 32
            || tpc_count == 0
            || tpc_count > 1024
            || mp_per_tpc != 2
            || gr(8)? != tpc_count * mp_per_tpc
            || max_warps_per_mp != 48
            || gr(14)? != 32
            || gr(12)? != 0xa04
        {
            return Err(Error::new(
                ErrorKind::Unsupported,
                "NVIDIA topology limits",
            ));
        }
        let mut device_name = [0u8; 64];
        let name = adapter.name()?;
        let name = name.as_bytes();
        let length = name.len().min(device_name.len() - 1);
        device_name[..length].copy_from_slice(&name[..length]);
        let mut chipset_name = [0u8; 16];
        chipset_name[..5].copy_from_slice(b"GB203");
        Ok(Self {
            valid_fields: 15,
            vram_size: adapter.segment_sizes()?.0,
            bar_size: 0,
            vendor_id: adapter.identity().vendor_id,
            device_id: adapter.identity().device_id,
            chipset: architecture | implementation,
            sm: 120,
            gpc_count,
            tpc_count,
            mp_per_tpc,
            max_warps_per_mp,
            max_blocks_per_mp: 24,
            cls_copy: class(0xcab5)?,
            cls_eng2d: class(0x902d)?,
            cls_eng3d: class(0xce97)?,
            cls_m2mf: 0,
            cls_compute: class(0xcec0)?,
            cls_gpfifo: class(0xca6f)?,
            device_name,
            chipset_name,
        })
    }
}
