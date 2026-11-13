#[cfg(windows)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use nvkmd_wddm::native::{Adapter, DeviceInfo};
    let adapter = Adapter::enumerate()?
        .into_iter()
        .find(|a| a.identity().vendor_id == 0x10de)
        .ok_or("NVIDIA adapter missing")?;
    let device = adapter.create_device()?;
    let info = DeviceInfo::query(&device)?;
    println!("device_info sizeof={} valid_fields=0x{:x} pci={:04x}:{:04x} chipset=0x{:x} sm={} gpc={} tpc={} mp_per_tpc={} max_warps={} vram={} classes={:x}/{:x}/{:x}/{:x}", std::mem::size_of::<DeviceInfo>(), info.valid_fields, info.vendor_id, info.device_id, info.chipset, info.sm, info.gpc_count, info.tpc_count, info.mp_per_tpc, info.max_warps_per_mp, info.vram_size, info.cls_copy, info.cls_eng3d, info.cls_compute, info.cls_gpfifo);
    println!(
        "device_name={}",
        std::str::from_utf8(&info.device_name)?.trim_end_matches('\0')
    );
    println!("execution_state={}", device.execution_state()?);
    Ok(())
}

#[cfg(not(windows))]
fn main() {}
