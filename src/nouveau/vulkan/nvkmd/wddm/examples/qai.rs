#[cfg(windows)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use nvkmd_wddm::native::Adapter;
    use nvkmd_wddm::nvidia::Nvgl61692Layout;
    let output = std::env::args().nth(1);
    let adapter = Adapter::enumerate()?
        .into_iter()
        .find(|a| a.identity().vendor_id == 0x10de)
        .ok_or("NVIDIA adapter missing")?;
    if adapter.identity().driver_version != Nvgl61692Layout::DRIVER_VERSION {
        return Err("driver profile mismatch".into());
    }
    let mut data = Nvgl61692Layout::query_request()?;
    unsafe {
        adapter.query_private(&mut data)?;
    }
    for offset in [0usize, 4, 8, 12, 16, 27367, 27371, 27375, 27379, 28607] {
        let value =
            u32::from_le_bytes(data.as_bytes()[offset..offset + 4].try_into()?);
        println!("offset={offset} value=0x{value:08x} decimal={value}");
    }
    if let Some(path) = output {
        std::fs::write(path, data.as_bytes())?;
    }
    Ok(())
}

#[cfg(not(windows))]
fn main() {}
