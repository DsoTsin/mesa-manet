#[cfg(windows)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use nvkmd_wddm::contract::{KernelDriver, Scheduling};
    use nvkmd_wddm::native::{Adapter, Context, Nvgl61692Abi, WddmDriver};
    use std::sync::Arc;
    let adapter = Adapter::enumerate()?
        .into_iter()
        .find(|a| a.identity().vendor_id == 0x10de)
        .ok_or("NVIDIA adapter missing")?;
    let query = adapter.query_nvidia_61692()?;
    for (index, node) in query
        .nodes()
        .iter()
        .filter(|n| n.node_type == 11 || n.node_type == 12)
        .enumerate()
    {
        let abi = Arc::new(Nvgl61692Abi::new(
            &adapter,
            &query,
            node.ordinal,
            index as u32,
        )?);
        let driver = WddmDriver::new(adapter.clone())?.with_abi(abi)?;
        let device = driver.create_device()?;
        let scheduling = if node.hardware_scheduling {
            Scheduling::HardwareQueue
        } else {
            Scheduling::VirtualContext
        };
        match Context::new(&device, 1, scheduling) {
            Ok(context) => {
                println!("node={} type={} scheduling={:?} context=created gpu_commands_executed=false", node.ordinal, node.node_type, scheduling);
                drop(context);
                println!("node={} context=destroyed", node.ordinal);
            }
            Err(error) => {
                println!(
                    "node={} type={} scheduling={:?} context=failed error={}",
                    node.ordinal, node.node_type, scheduling, error
                );
                return Err(error.into());
            }
        }
    }
    Ok(())
}

#[cfg(not(windows))]
fn main() {}
