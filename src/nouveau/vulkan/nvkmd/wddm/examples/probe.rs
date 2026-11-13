#[cfg(windows)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use nvkmd_wddm::native::{Adapter, Fence, Memory, VirtualAddress};
    let memory_test = std::env::args().any(|arg| arg == "--memory");
    let private_query = std::env::args().any(|arg| arg == "--private-query");
    for adapter in Adapter::enumerate()? {
        let id = *adapter.identity();
        println!("adapter luid={:08x}:{:08x} pci={:04x}:{:04x} driver={:016x} wddm={} scheduling_caps={:08x}", id.adapter.high, id.adapter.low, id.vendor_id, id.device_id, id.driver_version, id.wddm_version, adapter.scheduling_caps);
        if id.vendor_id != 0x10de {
            continue;
        }
        if private_query {
            let info = adapter.query_nvidia_61692()?;
            for node in info.nodes() {
                println!("nvidia_qai node={} type={} os_gpuva={} hardware_scheduling={} gpfifo_blit={}", node.ordinal, node.node_type, node.os_managed_gpuva, node.hardware_scheduling, node.gpfifo_blit);
            }
            println!(
                "private_query_bytes={} header_and_node_bounds=validated",
                info.payload().len()
            );
        }
        let device = adapter.create_device()?;
        let mut fence = Fence::new(&device, 0)?;
        let point = fence.signal_cpu()?;
        point.wait(1_000_000_000)?;
        println!("device=created paging_queue=created monitored_fence=cpu_signaled fence_value={}", point.value);
        if memory_test {
            let mut memory = Memory::host(&device, 65536)?;
            let input: Vec<u8> = (0..65536)
                .map(|i| ((i * 37 + 19) ^ (i >> 7)) as u8)
                .collect();
            memory.write(0, &input)?;
            let mut va = VirtualAddress::reserve(&device, 65536, 65536, None)?;
            let binding = va.bind(&memory, 0, 0, 65536)?;
            binding.paging_point().wait(5_000_000_000)?;
            memory.make_resident()?.wait(5_000_000_000)?;
            println!("standard_host_heap=created gpuva={:016x} paging=completed residency=completed", binding.address());
            drop(binding);
            drop(va);
            memory.evict()?;
            let mut output = vec![0u8; input.len()];
            memory.read(0, &mut output)?;
            if output != input {
                return Err("host allocation contents changed".into());
            }
            println!("host_roundtrip_bytes={} matched=true gpu_commands_executed=false", output.len());
        }
    }
    Ok(())
}

#[cfg(not(windows))]
fn main() {}
