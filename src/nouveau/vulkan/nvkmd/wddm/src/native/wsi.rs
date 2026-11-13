use super::{Device, Memory};
use crate::abi::PrivateData;
use crate::contract::AdapterId;
use crate::error::{Error, ErrorKind, Result};
use crate::nvidia::PrivateHeader;
use windows::core::{w, PCSTR};
use windows::Wdk::Graphics::Direct3D::{
    D3DKMTCloseAdapter, D3DKMTOpenAdapterFromGdiDisplayName,
    D3DKMT_CLOSEADAPTER, D3DKMT_OPENADAPTERFROMGDIDISPLAYNAME,
};
use windows::Win32::Foundation::{FreeLibrary, HANDLE, HMODULE};
use windows::Win32::Foundation::{HWND, RECT};
use windows::Win32::Graphics::Gdi::{
    GetMonitorInfoW, MonitorFromWindow, MONITORINFOEXW,
    MONITOR_DEFAULTTONEAREST,
};
use windows::Win32::System::LibraryLoader::{
    GetProcAddress, LoadLibraryExW, LOAD_LIBRARY_SEARCH_SYSTEM32,
};
use windows::Win32::UI::WindowsAndMessaging::{GetClientRect, IsWindow};

type AcquireSharedSurface = unsafe extern "system" fn(
    HWND,
    u64,
    usize,
    u32,
    *mut u32,
    *mut HANDLE,
    *mut u64,
) -> i32;

pub struct DwmBridge {
    module: HMODULE,
    acquire: AcquireSharedSurface,
}

pub struct DwmFrame<'a> {
    surface: &'a Surface,
    pub physical_surface: u64,
    pub update_id: u64,
    pub format: u32,
    pub status: i32,
    pub snapshot: SurfaceSnapshot,
}

impl DwmBridge {
    pub fn new() -> Result<Self> {
        let module = unsafe {
            LoadLibraryExW(w!("dwmapi.dll"), None, LOAD_LIBRARY_SEARCH_SYSTEM32)
        }
        .map_err(|_| {
            Error::new(ErrorKind::Unsupported, "DWM shared surface library")
        })?;
        let address =
            unsafe { GetProcAddress(module, PCSTR(100usize as *const u8)) };
        let Some(address) = address else {
            unsafe {
                let _ = FreeLibrary(module);
            }
            return Err(Error::new(
                ErrorKind::Unsupported,
                "DWM shared surface entrypoint",
            ));
        };
        Ok(Self {
            module,
            acquire: unsafe {
                std::mem::transmute::<
                    unsafe extern "system" fn() -> isize,
                    AcquireSharedSurface,
                >(address)
            },
        })
    }

    pub fn acquire<'a>(
        &self,
        surface: &'a Surface,
        device: &Device,
    ) -> Result<DwmFrame<'a>> {
        device.0.ready()?;
        let snapshot = surface.snapshot()?;
        if snapshot.width == 0 || snapshot.height == 0 {
            return Err(Error::new(ErrorKind::Busy, "zero-sized surface"));
        }
        let luid = device.adapter().identity().adapter;
        let render_luid = u64::from(luid.low) | ((luid.high as u64) << 32);
        let mut format = 87;
        let mut physical = HANDLE::default();
        let mut update_id = 0;
        let status = unsafe {
            (self.acquire)(
                HWND(surface.hwnd as *mut _),
                render_luid,
                0,
                0x10,
                &mut format,
                &mut physical,
                &mut update_id,
            )
        };
        Error::check("DwmpDxGetWindowSharedSurface", status)?;
        if status == 0x00263008 {
            physical = HANDLE::default();
        }
        Ok(DwmFrame {
            surface,
            physical_surface: physical.0 as usize as u64,
            update_id,
            format,
            status,
            snapshot,
        })
    }
}

impl Drop for DwmBridge {
    fn drop(&mut self) {
        unsafe {
            let _ = FreeLibrary(self.module);
        }
    }
}

impl DwmFrame<'_> {
    pub fn hwnd(&self) -> usize {
        self.surface.hwnd
    }
    pub(crate) fn current_snapshot(&self) -> Result<SurfaceSnapshot> {
        self.surface.snapshot()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(C)]
pub struct SurfaceSnapshot {
    pub display_adapter: AdapterId,
    pub source_id: u32,
    pub width: u32,
    pub height: u32,
}

pub struct Surface {
    hwnd: usize,
}

pub struct PresentImage {
    memory: Memory,
    width: u32,
    height: u32,
    pitch: u32,
    page_kind: u32,
}

pub struct Presenter {
    device: Device,
    surface: Surface,
    bridge: DwmBridge,
    context: super::Context,
}

impl Presenter {
    pub unsafe fn new(device: &Device, hwnd: usize) -> Result<Self> {
        let surface = unsafe { Surface::from_borrowed_hwnd(hwnd) }?;
        surface.snapshot()?;
        Ok(Self {
            device: device.clone(),
            surface,
            bridge: DwmBridge::new()?,
            context: super::Context::new(
                device,
                4 | 16,
                crate::contract::Scheduling::HardwareQueue,
            )?,
        })
    }

    pub fn create_image(
        &self,
        width: u32,
        height: u32,
        format: u32,
    ) -> Result<PresentImage> {
        if format != 37 {
            return Err(Error::new(
                ErrorKind::Unsupported,
                "native presentation format",
            ));
        }
        let snapshot = self.surface.snapshot()?;
        PresentImage::host_rgba8(
            &self.device,
            width,
            height,
            snapshot.source_id,
        )
    }

    pub fn present(
        &mut self,
        image: &PresentImage,
        timeout_ns: u64,
    ) -> Result<i32> {
        let frame = self.bridge.acquire(&self.surface, &self.device)?;
        self.context.present(frame, image, timeout_ns)
    }
}

impl PresentImage {
    pub fn host_rgba8(
        device: &Device,
        width: u32,
        height: u32,
        source_id: u32,
    ) -> Result<Self> {
        if width == 0 || height == 0 || width > 16384 || height > 16384 {
            return Err(Error::new(ErrorKind::Invalid, "presentation extent"));
        }
        let unique_id = device.nvidia_unique_id()?;
        let pitch = (u64::from(width) * 4 + 255) & !255;
        let size = (pitch * u64::from(height) + 65535) & !65535;
        let mut private = PrivateData::zeroed(586)?;
        PrivateHeader {
            version: 0x10004,
            size: 586,
            caller: PrivateHeader::NVGL,
        }
        .write(&mut private)?;
        private.write_u64(16, (1u64 << 50) | 0x03000008)?;
        for (offset, value) in [
            (32, 1),
            (40, 32),
            (48, width),
            (52, height),
            (56, 1),
            (60, 1),
            (72, source_id),
            (106, height),
            (110, 4),
            (114, 64),
            (118, 64),
            (122, 64),
            (190, 0x02000000),
            (194, width),
            (198, height),
            (222, 4),
            (238, 1),
            (286, 4),
            (306, 1),
            (310, 1),
            (314, 1),
        ] {
            private.write_u32(offset, value)?;
        }
        private.write_u64(206, size)?;
        private.write_u64(246, u64::MAX)?;
        private.write_u64(258, unique_id)?;
        private.write_u64(278, size)?;
        let memory = Memory::presentation_host(device, size, &mut private)?;
        let word = |offset: usize| {
            u32::from_le_bytes(
                private.as_bytes()[offset..offset + 4].try_into().unwrap(),
            )
        };
        let returned_pitch = word(270);
        let os_pitch = word(202);
        let page_kind = word(430);
        let returned_size = u64::from_le_bytes(
            private.as_bytes()[206..214].try_into().unwrap(),
        );
        if returned_pitch != os_pitch
            || returned_pitch < width * 4
            || u64::from(returned_pitch) * u64::from(height) > size
            || returned_size > size
        {
            return Err(Error::new(
                ErrorKind::Unsupported,
                "presentation linear layout reply",
            ));
        }
        Ok(Self {
            memory,
            width,
            height,
            pitch: returned_pitch,
            page_kind,
        })
    }

    pub fn memory(&self) -> &Memory {
        &self.memory
    }
    pub fn memory_mut(&mut self) -> &mut Memory {
        &mut self.memory
    }
    pub fn width(&self) -> u32 {
        self.width
    }
    pub fn height(&self) -> u32 {
        self.height
    }
    pub fn pitch(&self) -> u32 {
        self.pitch
    }
    pub fn page_kind(&self) -> u32 {
        self.page_kind
    }
}

impl Surface {
    pub unsafe fn from_borrowed_hwnd(hwnd: usize) -> Result<Self> {
        if hwnd == 0
            || !unsafe { IsWindow(Some(HWND(hwnd as *mut _))) }.as_bool()
        {
            return Err(Error::new(ErrorKind::Invalid, "Win32 surface HWND"));
        }
        Ok(Self { hwnd })
    }

    pub fn snapshot(&self) -> Result<SurfaceSnapshot> {
        let hwnd = HWND(self.hwnd as *mut _);
        if !unsafe { IsWindow(Some(hwnd)) }.as_bool() {
            return Err(Error::new(
                ErrorKind::SurfaceLost,
                "Win32 surface destroyed",
            ));
        }
        let mut extent = RECT::default();
        unsafe { GetClientRect(hwnd, &mut extent) }
            .map_err(|_| Error::new(ErrorKind::System, "GetClientRect"))?;
        let width = extent
            .right
            .checked_sub(extent.left)
            .and_then(|v| u32::try_from(v).ok())
            .ok_or_else(|| Error::new(ErrorKind::Invalid, "surface width"))?;
        let height = extent
            .bottom
            .checked_sub(extent.top)
            .and_then(|v| u32::try_from(v).ok())
            .ok_or_else(|| Error::new(ErrorKind::Invalid, "surface height"))?;
        let monitor =
            unsafe { MonitorFromWindow(hwnd, MONITOR_DEFAULTTONEAREST) };
        let mut info = MONITORINFOEXW::default();
        info.monitorInfo.cbSize = std::mem::size_of::<MONITORINFOEXW>() as u32;
        if !unsafe { GetMonitorInfoW(monitor, &mut info.monitorInfo) }.as_bool()
        {
            return Err(Error::new(ErrorKind::System, "GetMonitorInfoW"));
        }
        let mut open = D3DKMT_OPENADAPTERFROMGDIDISPLAYNAME {
            DeviceName: info.szDevice,
            ..Default::default()
        };
        let status =
            unsafe { D3DKMTOpenAdapterFromGdiDisplayName(&mut open) }.0;
        if matches!(status as u32, 0xc000000d | 0xc00000bb) {
            return Err(Error {
                kind: ErrorKind::Unsupported,
                operation: "monitor has no native WDDM display adapter",
                status,
            });
        }
        Error::check("D3DKMTOpenAdapterFromGdiDisplayName", status)?;
        let snapshot = SurfaceSnapshot {
            display_adapter: AdapterId {
                low: open.AdapterLuid.LowPart,
                high: open.AdapterLuid.HighPart,
            },
            source_id: open.VidPnSourceId,
            width,
            height,
        };
        Error::check(
            "D3DKMTCloseAdapter display",
            unsafe {
                D3DKMTCloseAdapter(&D3DKMT_CLOSEADAPTER {
                    hAdapter: open.hAdapter,
                })
            }
            .0,
        )?;
        Ok(snapshot)
    }

    pub fn matches_render_adapter(
        &self,
        render_adapter: AdapterId,
    ) -> Result<bool> {
        Ok(self.snapshot()?.display_adapter == render_adapter)
    }
}
