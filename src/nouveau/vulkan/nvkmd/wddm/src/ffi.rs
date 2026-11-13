use super::native::{
    Adapter, Device, Fence, FencePoint, Memory, MemoryBinding, VirtualAddress,
};
use crate::contract::AdapterId;
use crate::contract::{KernelDriver, Push, Scheduling};
use crate::error::{Error, ErrorKind, Result};
use crate::native::{Context, Nvgl61692Abi, WddmDriver};
use crate::submission::Queue;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::Arc;

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct AdapterInfo {
    pub luid: AdapterId,
    pub vendor_id: u32,
    pub device_id: u32,
    pub driver_version: u64,
    pub wddm_version: u32,
    pub scheduling_caps: u32,
}

struct FfiCall;

#[no_mangle]
pub unsafe extern "C" fn nvkmd_wddm_nvidia_local_memory_create(
    device: *const Device,
    size: u64,
    memory: *mut *mut Memory,
    page_kind: *mut u32,
) -> i32 {
    FfiCall::result(|| unsafe {
        let out = FfiCall::object_mut(memory)?;
        let page_kind = FfiCall::object_mut(page_kind)?;
        let (memory, kind) =
            Memory::nvidia_local(FfiCall::object(device)?, size)?;
        *page_kind = kind;
        *out = Box::into_raw(Box::new(memory));
        Ok(())
    })
}

#[no_mangle]
pub unsafe extern "C" fn nvkmd_wddm_device_timestamp(
    device: *const Device,
    out: *mut u64,
) -> i32 {
    FfiCall::result(|| unsafe {
        *FfiCall::object_mut(out)? =
            FfiCall::object(device)?.adapter().gpu_timestamp_ns()?;
        Ok(())
    })
}

#[no_mangle]
pub unsafe extern "C" fn nvkmd_wddm_nvidia_host_memory_create(
    device: *const Device,
    size: u64,
    memory: *mut *mut Memory,
    page_kind: *mut u32,
) -> i32 {
    FfiCall::result(|| unsafe {
        let out = FfiCall::object_mut(memory)?;
        let page_kind = FfiCall::object_mut(page_kind)?;
        let (memory, kind) =
            Memory::nvidia_host(FfiCall::object(device)?, size)?;
        *page_kind = kind;
        *out = Box::into_raw(Box::new(memory));
        Ok(())
    })
}

#[no_mangle]
pub unsafe extern "C" fn nvkmd_wddm_presenter_create(
    device: *const Device,
    hwnd: usize,
    out: *mut *mut crate::native::wsi::Presenter,
) -> i32 {
    FfiCall::result(|| unsafe {
        FfiCall::output(
            out,
            crate::native::wsi::Presenter::new(FfiCall::object(device)?, hwnd)?,
        )
    })
}

#[no_mangle]
pub unsafe extern "C" fn nvkmd_wddm_presenter_destroy(
    presenter: *mut crate::native::wsi::Presenter,
) {
    if !presenter.is_null() {
        let _ = catch_unwind(AssertUnwindSafe(|| unsafe {
            drop(Box::from_raw(presenter));
        }));
    }
}

#[no_mangle]
pub unsafe extern "C" fn nvkmd_wddm_present_image_create(
    presenter: *const crate::native::wsi::Presenter,
    width: u32,
    height: u32,
    format: u32,
    out: *mut *mut crate::native::wsi::PresentImage,
) -> i32 {
    FfiCall::result(|| unsafe {
        FfiCall::output(
            out,
            FfiCall::object(presenter)?.create_image(width, height, format)?,
        )
    })
}

#[no_mangle]
pub unsafe extern "C" fn nvkmd_wddm_present_image_memory(
    image: *const crate::native::wsi::PresentImage,
    memory: *mut *mut Memory,
    size: *mut u64,
    pitch: *mut u32,
) -> i32 {
    FfiCall::result(|| unsafe {
        let memory = FfiCall::object_mut(memory)?;
        let size = FfiCall::object_mut(size)?;
        let pitch = FfiCall::object_mut(pitch)?;
        let image = FfiCall::object(image)?;
        *size = image.memory().size();
        *pitch = image.pitch();
        *memory = Box::into_raw(Box::new(image.memory().clone()));
        Ok(())
    })
}

#[no_mangle]
pub unsafe extern "C" fn nvkmd_wddm_present_image_destroy(
    image: *mut crate::native::wsi::PresentImage,
) {
    if !image.is_null() {
        let _ = catch_unwind(AssertUnwindSafe(|| unsafe {
            drop(Box::from_raw(image));
        }));
    }
}

#[no_mangle]
pub unsafe extern "C" fn nvkmd_wddm_presenter_present(
    presenter: *mut crate::native::wsi::Presenter,
    image: *const crate::native::wsi::PresentImage,
    timeout_ns: u64,
) -> i32 {
    FfiCall::result(|| {
        unsafe { FfiCall::object_mut(presenter) }?
            .present(unsafe { FfiCall::object(image) }?, timeout_ns)?;
        Ok(())
    })
}

#[no_mangle]
pub unsafe extern "C" fn nvkmd_wddm_device_info(
    device: *const Device,
    out: *mut crate::native::DeviceInfo,
) -> i32 {
    FfiCall::result(|| {
        *unsafe { FfiCall::object_mut(out) }? =
            crate::native::DeviceInfo::query(unsafe {
                FfiCall::object(device)
            }?)?;
        Ok(())
    })
}

pub type NativeQueue = Queue<Context>;

#[repr(C)]
pub struct NativePush {
    pub address: u64,
    pub size: u32,
    pub incomplete: u8,
    pub no_prefetch: u8,
    pub reserved: u16,
}

#[no_mangle]
pub unsafe extern "C" fn nvkmd_wddm_memory_map(
    memory: *const Memory,
    out: *mut *mut std::ffi::c_void,
) -> i32 {
    FfiCall::result(|| {
        *unsafe { FfiCall::object_mut(out) }? =
            unsafe { FfiCall::object(memory)?.map() }?.cast();
        Ok(())
    })
}

#[no_mangle]
pub unsafe extern "C" fn nvkmd_wddm_va_address(
    va: *const VirtualAddress,
    out: *mut u64,
) -> i32 {
    FfiCall::result(|| {
        *unsafe { FfiCall::object_mut(out) }? =
            unsafe { FfiCall::object(va) }?.address();
        Ok(())
    })
}

#[no_mangle]
pub unsafe extern "C" fn nvkmd_wddm_va_bind_executable(
    va: *mut VirtualAddress,
    memory: *const Memory,
    va_offset: u64,
    memory_offset: u64,
    size: u64,
    out: *mut *mut MemoryBinding,
) -> i32 {
    FfiCall::result(|| unsafe {
        FfiCall::output(
            out,
            FfiCall::object_mut(va)?.bind_executable(
                FfiCall::object(memory)?,
                va_offset,
                memory_offset,
                size,
            )?,
        )
    })
}

#[no_mangle]
pub unsafe extern "C" fn nvkmd_wddm_device_check_state(
    device: *const Device,
) -> i32 {
    FfiCall::result(|| {
        unsafe { FfiCall::object(device) }?.execution_state()?;
        Ok(())
    })
}

impl FfiCall {
    fn result(body: impl FnOnce() -> Result<()>) -> i32 {
        match catch_unwind(AssertUnwindSafe(body)) {
            Ok(Ok(())) => 0,
            Ok(Err(error)) => {
                if std::env::var_os("NVK_WDDM_DEBUG").is_some() {
                    eprintln!("NVK WDDM: {error}");
                }
                error.vk_result()
            }
            Err(_) => -13,
        }
    }

    unsafe fn object_mut<'a, T>(ptr: *mut T) -> Result<&'a mut T> {
        unsafe { ptr.as_mut() }
            .ok_or_else(|| Error::new(ErrorKind::Invalid, "null FFI object"))
    }

    unsafe fn object<'a, T>(ptr: *const T) -> Result<&'a T> {
        unsafe { ptr.as_ref() }
            .ok_or_else(|| Error::new(ErrorKind::Invalid, "null FFI object"))
    }

    unsafe fn output<T>(out: *mut *mut T, object: T) -> Result<()> {
        let out = unsafe { out.as_mut() }
            .ok_or_else(|| Error::new(ErrorKind::Invalid, "null FFI output"))?;
        *out = Box::into_raw(Box::new(object));
        Ok(())
    }
}

#[no_mangle]
pub unsafe extern "C" fn nvkmd_wddm_enumerate(
    count: *mut u32,
    adapters: *mut AdapterInfo,
) -> i32 {
    FfiCall::result(|| {
        let count = unsafe { FfiCall::object_mut(count) }?;
        let list = Adapter::enumerate()?;
        let total = u32::try_from(list.len())
            .map_err(|_| Error::new(ErrorKind::Invalid, "adapter count"))?;
        if adapters.is_null() {
            *count = total;
            return Ok(());
        }
        let write = (*count).min(total);
        for (i, adapter) in list.iter().take(write as usize).enumerate() {
            let id = adapter.identity();
            unsafe {
                adapters.add(i).write(AdapterInfo {
                    luid: id.adapter,
                    vendor_id: id.vendor_id,
                    device_id: id.device_id,
                    driver_version: id.driver_version,
                    wddm_version: id.wddm_version,
                    scheduling_caps: adapter.scheduling_caps,
                });
            }
        }
        *count = write;
        if write != total {
            Err(Error::new(ErrorKind::Busy, "adapter enumeration capacity"))
        } else {
            Ok(())
        }
    })
}

#[no_mangle]
pub unsafe extern "C" fn nvkmd_wddm_device_create(
    luid: AdapterId,
    out: *mut *mut Device,
) -> i32 {
    FfiCall::result(|| unsafe {
        FfiCall::output(out, Adapter::open(luid)?.create_device()?)
    })
}

#[no_mangle]
pub unsafe extern "C" fn nvkmd_wddm_device_destroy(device: *mut Device) {
    if !device.is_null() {
        let _ = catch_unwind(AssertUnwindSafe(|| unsafe {
            drop(Box::from_raw(device));
        }));
    }
}

#[no_mangle]
pub unsafe extern "C" fn nvkmd_wddm_host_memory_create(
    device: *mut Device,
    size: u64,
    out: *mut *mut Memory,
) -> i32 {
    FfiCall::result(|| unsafe {
        FfiCall::output(out, Memory::host(FfiCall::object(device)?, size)?)
    })
}

#[no_mangle]
pub unsafe extern "C" fn nvkmd_wddm_memory_destroy(memory: *mut Memory) {
    if !memory.is_null() {
        let _ = catch_unwind(AssertUnwindSafe(|| unsafe {
            drop(Box::from_raw(memory));
        }));
    }
}

#[no_mangle]
pub unsafe extern "C" fn nvkmd_wddm_va_reserve(
    device: *mut Device,
    size: u64,
    alignment: u64,
    fixed: u64,
    out: *mut *mut VirtualAddress,
) -> i32 {
    FfiCall::result(|| unsafe {
        FfiCall::output(
            out,
            VirtualAddress::reserve(
                FfiCall::object(device)?,
                size,
                alignment,
                (fixed != 0).then_some(fixed),
            )?,
        )
    })
}

#[no_mangle]
pub unsafe extern "C" fn nvkmd_wddm_va_destroy(va: *mut VirtualAddress) {
    if !va.is_null() {
        let _ = catch_unwind(AssertUnwindSafe(|| unsafe {
            drop(Box::from_raw(va));
        }));
    }
}

#[no_mangle]
pub unsafe extern "C" fn nvkmd_wddm_fence_create(
    device: *const Device,
    initial: u64,
    out: *mut *mut Fence,
) -> i32 {
    FfiCall::result(|| unsafe {
        FfiCall::output(out, Fence::new(FfiCall::object(device)?, initial)?)
    })
}

#[no_mangle]
pub unsafe extern "C" fn nvkmd_wddm_fence_point(
    fence: *const Fence,
    value: u64,
    out: *mut *mut FencePoint,
) -> i32 {
    FfiCall::result(|| unsafe {
        FfiCall::output(out, FfiCall::object(fence)?.point(value))
    })
}

#[no_mangle]
pub unsafe extern "C" fn nvkmd_wddm_fence_get_value(
    fence: *const Fence,
    out: *mut u64,
) -> i32 {
    FfiCall::result(|| {
        *unsafe { FfiCall::object_mut(out) }? =
            unsafe { FfiCall::object(fence) }?.value()?;
        Ok(())
    })
}

#[no_mangle]
pub unsafe extern "C" fn nvkmd_wddm_fence_signal_cpu_value(
    fence: *mut Fence,
    value: u64,
) -> i32 {
    FfiCall::result(|| {
        unsafe { FfiCall::object_mut(fence) }?.signal_cpu_value(value)
    })
}

#[no_mangle]
pub unsafe extern "C" fn nvkmd_wddm_fence_destroy(fence: *mut Fence) {
    if !fence.is_null() {
        let _ = catch_unwind(AssertUnwindSafe(|| unsafe {
            drop(Box::from_raw(fence));
        }));
    }
}

#[no_mangle]
pub unsafe extern "C" fn nvkmd_wddm_memory_make_resident(
    memory: *const Memory,
    out: *mut *mut FencePoint,
) -> i32 {
    FfiCall::result(|| unsafe {
        FfiCall::output(out, FfiCall::object(memory)?.make_resident()?)
    })
}

#[no_mangle]
pub unsafe extern "C" fn nvkmd_wddm_memory_evict(memory: *mut Memory) -> i32 {
    FfiCall::result(|| {
        unsafe { FfiCall::object_mut(memory) }?.evict()?;
        Ok(())
    })
}

#[no_mangle]
pub unsafe extern "C" fn nvkmd_wddm_memory_write(
    memory: *mut Memory,
    offset: u64,
    bytes: *const u8,
    size: usize,
) -> i32 {
    FfiCall::result(|| {
        let memory = unsafe { FfiCall::object_mut(memory) }?;
        if size == 0 {
            return Ok(());
        }
        if bytes.is_null() || size > isize::MAX as usize {
            return Err(Error::new(ErrorKind::Invalid, "host input buffer"));
        }
        memory.write(offset, unsafe { std::slice::from_raw_parts(bytes, size) })
    })
}

#[no_mangle]
pub unsafe extern "C" fn nvkmd_wddm_memory_read(
    memory: *mut Memory,
    offset: u64,
    bytes: *mut u8,
    size: usize,
) -> i32 {
    FfiCall::result(|| {
        let memory = unsafe { FfiCall::object_mut(memory) }?;
        if size == 0 {
            return Ok(());
        }
        if bytes.is_null() || size > isize::MAX as usize {
            return Err(Error::new(ErrorKind::Invalid, "host output buffer"));
        }
        memory.read(offset, unsafe {
            std::slice::from_raw_parts_mut(bytes, size)
        })
    })
}

#[no_mangle]
pub unsafe extern "C" fn nvkmd_wddm_va_bind(
    va: *mut VirtualAddress,
    memory: *const Memory,
    va_offset: u64,
    memory_offset: u64,
    size: u64,
    out: *mut *mut MemoryBinding,
) -> i32 {
    FfiCall::result(|| unsafe {
        FfiCall::output(
            out,
            FfiCall::object_mut(va)?.bind(
                FfiCall::object(memory)?,
                va_offset,
                memory_offset,
                size,
            )?,
        )
    })
}

#[no_mangle]
pub unsafe extern "C" fn nvkmd_wddm_binding_address(
    binding: *const MemoryBinding,
    out: *mut u64,
) -> i32 {
    FfiCall::result(|| {
        *unsafe { FfiCall::object_mut(out) }? =
            unsafe { FfiCall::object(binding) }?.address();
        Ok(())
    })
}

#[no_mangle]
pub unsafe extern "C" fn nvkmd_wddm_binding_destroy(
    binding: *mut MemoryBinding,
) {
    if !binding.is_null() {
        let _ = catch_unwind(AssertUnwindSafe(|| unsafe {
            drop(Box::from_raw(binding));
        }));
    }
}

#[no_mangle]
pub unsafe extern "C" fn nvkmd_wddm_fence_signal_cpu(
    fence: *mut Fence,
    out: *mut *mut FencePoint,
) -> i32 {
    FfiCall::result(|| unsafe {
        FfiCall::output(out, FfiCall::object_mut(fence)?.signal_cpu()?)
    })
}

#[no_mangle]
pub unsafe extern "C" fn nvkmd_wddm_point_wait(
    point: *const FencePoint,
    timeout_ns: u64,
) -> i32 {
    FfiCall::result(|| unsafe { FfiCall::object(point) }?.wait(timeout_ns))
}

#[no_mangle]
pub unsafe extern "C" fn nvkmd_wddm_point_destroy(point: *mut FencePoint) {
    if !point.is_null() {
        let _ = catch_unwind(AssertUnwindSafe(|| unsafe {
            drop(Box::from_raw(point));
        }));
    }
}

#[no_mangle]
pub unsafe extern "C" fn nvkmd_wddm_nvidia_device_create(
    luid: AdapterId,
    node: u32,
    context_id: u32,
    out: *mut *mut Device,
) -> i32 {
    FfiCall::result(|| {
        let adapter = Adapter::open(luid)?;
        let query = adapter.query_nvidia_61692()?;
        let abi =
            Arc::new(Nvgl61692Abi::new(&adapter, &query, node, context_id)?);
        let driver = WddmDriver::new(adapter)?.with_abi(abi)?;
        unsafe { FfiCall::output(out, driver.create_device()?) }
    })
}

#[no_mangle]
pub unsafe extern "C" fn nvkmd_wddm_context_create(
    device: *const Device,
    engines: u32,
    hardware_queue: u32,
    out: *mut *mut NativeQueue,
) -> i32 {
    FfiCall::result(|| {
        let scheduling = match hardware_queue {
            0 => Scheduling::VirtualContext,
            1 => Scheduling::HardwareQueue,
            _ => {
                return Err(Error::new(
                    ErrorKind::Invalid,
                    "scheduling selector",
                ))
            }
        };
        unsafe {
            FfiCall::output(
                out,
                Context::new(FfiCall::object(device)?, engines, scheduling)?
                    .queue(),
            )
        }
    })
}

#[no_mangle]
pub unsafe extern "C" fn nvkmd_wddm_context_destroy(context: *mut NativeQueue) {
    if !context.is_null() {
        let _ = catch_unwind(AssertUnwindSafe(|| unsafe {
            drop(Box::from_raw(context));
        }));
    }
}

#[no_mangle]
pub unsafe extern "C" fn nvkmd_wddm_context_retain(
    context: *mut NativeQueue,
    binding: *const MemoryBinding,
) -> i32 {
    FfiCall::result(|| {
        unsafe { FfiCall::object_mut(context) }?
            .retain(unsafe { FfiCall::object(binding) }?.clone())
    })
}

#[no_mangle]
pub unsafe extern "C" fn nvkmd_wddm_context_reflected_sked_address(
    context: *const NativeQueue,
    out: *mut u64,
) -> i32 {
    FfiCall::result(|| unsafe {
        let out = FfiCall::object_mut(out)?;
        *out = 0;
        *out = FfiCall::object(context)?
            .transport()
            .reflected_sked_address()?;
        Ok(())
    })
}

#[no_mangle]
pub unsafe extern "C" fn nvkmd_wddm_context_push(
    context: *mut NativeQueue,
    push: NativePush,
) -> i32 {
    FfiCall::result(|| {
        if push.incomplete > 1 || push.no_prefetch > 1 || push.reserved != 0 {
            return Err(Error::new(ErrorKind::Invalid, "push flags"));
        }
        unsafe { FfiCall::object_mut(context) }?.push(Push {
            address: push.address,
            size: push.size,
            incomplete: push.incomplete != 0,
            no_prefetch: push.no_prefetch != 0,
        })
    })
}

#[no_mangle]
pub unsafe extern "C" fn nvkmd_wddm_context_wait(
    context: *mut NativeQueue,
    point: *const FencePoint,
) -> i32 {
    FfiCall::result(|| {
        unsafe { FfiCall::object_mut(context) }?
            .wait_on(unsafe { FfiCall::object(point) }?.clone())
    })
}

#[no_mangle]
pub unsafe extern "C" fn nvkmd_wddm_context_signal(
    context: *mut NativeQueue,
    point: *const FencePoint,
) -> i32 {
    FfiCall::result(|| {
        unsafe { FfiCall::object_mut(context) }?
            .signal(unsafe { FfiCall::object(point) }?.clone())
    })
}

#[no_mangle]
pub unsafe extern "C" fn nvkmd_wddm_context_flush(
    context: *mut NativeQueue,
    out: *mut *mut FencePoint,
) -> i32 {
    FfiCall::result(|| unsafe {
        FfiCall::output(out, FfiCall::object_mut(context)?.flush()?)
    })
}

#[no_mangle]
pub unsafe extern "C" fn nvkmd_wddm_context_collect(
    context: *mut NativeQueue,
) -> i32 {
    FfiCall::result(|| {
        unsafe { FfiCall::object_mut(context) }?.collect()?;
        Ok(())
    })
}
