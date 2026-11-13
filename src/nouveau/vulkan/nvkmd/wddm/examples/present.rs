#[cfg(windows)]
unsafe extern "system" fn wndproc(
    hwnd: windows::Win32::Foundation::HWND,
    message: u32,
    wparam: windows::Win32::Foundation::WPARAM,
    lparam: windows::Win32::Foundation::LPARAM,
) -> windows::Win32::Foundation::LRESULT {
    unsafe {
        windows::Win32::UI::WindowsAndMessaging::DefWindowProcW(
            hwnd, message, wparam, lparam,
        )
    }
}

#[cfg(windows)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use nvkmd_wddm::contract::KernelDriver;
    use nvkmd_wddm::contract::Scheduling;
    use nvkmd_wddm::native::wsi::{DwmBridge, PresentImage, Surface};
    use nvkmd_wddm::native::{
        Adapter, Context, Nvgl61692Abi, VirtualAddress, WddmDriver,
    };
    use std::sync::Arc;
    use windows::core::w;
    use windows::Win32::Foundation::HINSTANCE;
    use windows::Win32::Graphics::Gdi::{GetDC, GetPixel, ReleaseDC};
    use windows::Win32::System::LibraryLoader::GetModuleHandleW;
    use windows::Win32::UI::WindowsAndMessaging::*;
    let adapter = Adapter::enumerate()?
        .into_iter()
        .find(|a| a.identity().vendor_id == 0x10de)
        .ok_or("NVIDIA adapter missing")?;
    let query = adapter.query_nvidia_61692()?;
    let abi = Arc::new(Nvgl61692Abi::new(&adapter, &query, 0, 0)?);
    let device = WddmDriver::new(adapter)?.with_abi(abi)?.create_device()?;
    let instance = HINSTANCE(
        unsafe { GetModuleHandleW(None) }
            .map_err(|e| format!("GetModuleHandleW: {e}"))?
            .0,
    );
    let class = WNDCLASSW {
        lpfnWndProc: Some(wndproc),
        hInstance: instance,
        lpszClassName: w!("NvkWddmValidation"),
        ..Default::default()
    };
    if unsafe { RegisterClassW(&class) } == 0 {
        return Err("RegisterClassW failed".into());
    }
    let hwnd = unsafe {
        CreateWindowExW(
            WINDOW_EX_STYLE::default(),
            w!("NvkWddmValidation"),
            w!("NVK WDDM native WSI validation"),
            WS_OVERLAPPEDWINDOW | WS_VISIBLE,
            100,
            100,
            400,
            320,
            None,
            None,
            Some(instance),
            None,
        )
    }
    .map_err(|e| format!("CreateWindowExW: {e}"))?;
    let mut message = MSG::default();
    while unsafe { PeekMessageW(&mut message, Some(hwnd), 0, 0, PM_REMOVE) }
        .as_bool()
    {
        unsafe {
            let _ = TranslateMessage(&message);
            DispatchMessageW(&message);
        }
    }
    let surface = unsafe { Surface::from_borrowed_hwnd(hwnd.0 as usize) }?;
    let bridge = DwmBridge::new()?;
    let frame = bridge.acquire(&surface, &device)?;
    println!("DWM acquire status=0x{:08x} physical=0x{:x} update={} format={} extent={}x{} display_luid={:08x}:{:08x} source_id={} same_adapter={}", frame.status, frame.physical_surface, frame.update_id, frame.format, frame.snapshot.width, frame.snapshot.height, frame.snapshot.display_adapter.high, frame.snapshot.display_adapter.low, frame.snapshot.source_id, surface.matches_render_adapter(device.adapter().identity().adapter)?);
    let mut image = PresentImage::host_rgba8(
        &device,
        frame.snapshot.width,
        frame.snapshot.height,
        frame.snapshot.source_id,
    )?;
    let mut pixels = vec![0u8; image.memory().size() as usize];
    for y in 0..image.height() {
        for x in 0..image.width() {
            let offset = (y * image.pitch() + x * 4) as usize;
            let color = if ((x / 32) ^ (y / 32)) & 1 == 0 {
                [224, 64, 32, 255]
            } else {
                [32, 160, 224, 255]
            };
            pixels[offset..offset + 4].copy_from_slice(&color);
        }
    }
    image.memory_mut().write(0, &pixels)?;
    let mut render_address =
        VirtualAddress::reserve(&device, image.memory().size(), 65536, None)?;
    let render_binding = render_address.bind_executable(
        image.memory(),
        0,
        0,
        image.memory().size(),
    )?;
    render_binding.paging_point().wait(5_000_000_000)?;
    image.memory().make_resident()?.wait(5_000_000_000)?;
    println!("presentation allocation=resident width={} height={} pitch={} page_kind={}", image.width(), image.height(), image.pitch(), image.page_kind());
    let mut context = Context::new(&device, 4 | 16, Scheduling::HardwareQueue)?;
    let status = context.present(frame, &image, 5_000_000_000)?;
    println!(
        "presentation submitted status=0x{:08x} queue_completion=true",
        status
    );
    let until = std::time::Instant::now() + std::time::Duration::from_secs(2);
    let mut observed = false;
    while std::time::Instant::now() < until {
        while unsafe { PeekMessageW(&mut message, Some(hwnd), 0, 0, PM_REMOVE) }
            .as_bool()
        {
            unsafe {
                let _ = TranslateMessage(&message);
                DispatchMessageW(&message);
            }
        }
        let dc = unsafe { GetDC(Some(hwnd)) };
        let first = unsafe { GetPixel(dc, 16, 16) }.0;
        let second = unsafe { GetPixel(dc, 48, 16) }.0;
        unsafe {
            ReleaseDC(Some(hwnd), dc);
        }
        if first == 0x002040e0 && second == 0x00e0a020 {
            if !observed {
                println!("presentation window_pixels=matched first=0x{:08x} second=0x{:08x} native_composition=true", first, second);
            }
            observed = true;
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    unsafe { DestroyWindow(hwnd) }
        .map_err(|e| format!("DestroyWindow: {e}"))?;
    if !observed {
        return Err("present accepted but window pixels not observed".into());
    }
    Ok(())
}

#[cfg(not(windows))]
fn main() {}
