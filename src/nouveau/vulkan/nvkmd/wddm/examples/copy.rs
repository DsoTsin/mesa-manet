#[cfg(windows)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use nvkmd_wddm::command::PushBuffer;
    use nvkmd_wddm::contract::{KernelDriver, Push, Scheduling};
    use nvkmd_wddm::native::{
        Adapter, Context, Memory, Nvgl61692Abi, VirtualAddress, WddmDriver,
    };
    use std::sync::Arc;
    let adapter = Adapter::enumerate()?
        .into_iter()
        .find(|a| a.identity().vendor_id == 0x10de)
        .ok_or("NVIDIA adapter missing")?;
    let query = adapter.query_nvidia_61692()?;
    let gr = std::env::args().any(|arg| arg == "--gr");
    let node = *query
        .nodes()
        .iter()
        .find(|n| {
            n.node_type == if gr { 1 } else { 12 } && n.hardware_scheduling
        })
        .ok_or("hardware COPY node missing")?;
    let abi = Arc::new(Nvgl61692Abi::new(&adapter, &query, node.ordinal, 0)?);
    let device = WddmDriver::new(adapter)?.with_abi(abi)?.create_device()?;
    let context = Context::new(&device, 1, Scheduling::HardwareQueue)?;
    let mut queue = context.queue();
    let private = std::env::args().any(|arg| arg == "--nvidia");
    let small = std::env::args().any(|arg| arg == "--small");
    let allocation_size = if small { 4096 } else { 65536 };
    let allocate = || {
        if private {
            let (memory, kind) = Memory::nvidia_host(&device, allocation_size)?;
            println!(
                "copy allocation=nvidia_generic_host bytes={} page_kind={}",
                memory.size(),
                kind
            );
            Ok(memory)
        } else {
            Memory::host(&device, allocation_size)
        }
    };
    let mut source = allocate()?;
    let mut destination = allocate()?;
    let mut commands = allocate()?;
    let mut source_va =
        VirtualAddress::reserve(&device, allocation_size, 65536, None)?;
    let mut destination_va =
        VirtualAddress::reserve(&device, allocation_size, 65536, None)?;
    let mut command_va =
        VirtualAddress::reserve(&device, allocation_size, 65536, None)?;
    let input: Vec<u8> = (0..allocation_size as u32)
        .map(|i| ((i * 37 + 19) ^ (i >> 7)) as u8)
        .collect();
    let initial = vec![0xa5u8; allocation_size as usize];
    source.write(0, &input)?;
    destination.write(0, &initial)?;
    let source_offset = 256usize;
    let destination_offset = 512usize;
    let size = if small { 2036usize } else { 12340usize };
    let mut pushbuffer = PushBuffer::new();
    pushbuffer.copy_linear(
        source_va.address() + source_offset as u64,
        destination_va.address() + destination_offset as u64,
        size as u32,
    )?;
    let bytes = pushbuffer.bytes();
    commands.write(0, &bytes)?;
    let source_binding = source_va.bind(&source, 0, 0, allocation_size)?;
    let destination_binding =
        destination_va.bind(&destination, 0, 0, allocation_size)?;
    let command_binding = command_va.bind(&commands, 0, 0, allocation_size)?;
    queue.retain(source_binding.clone())?;
    queue.retain(destination_binding.clone())?;
    queue.retain(command_binding.clone())?;
    queue.push(Push {
        address: command_binding.address(),
        size: bytes.len() as u32,
        incomplete: false,
        no_prefetch: false,
    })?;
    println!("copy node={} source_va={:x} destination_va={:x} command_va={:x} pb_bytes={}", node.ordinal, source_binding.address(), destination_binding.address(), command_binding.address(), bytes.len());
    let completion = queue.flush()?;
    println!("copy submit=accepted fence={}", completion.value);
    completion.wait(5_000_000_000)?;
    queue.collect()?;
    println!(
        "copy queue_progress={:?} execution_state={}",
        queue.transport().progress_values()?,
        device.execution_state()?
    );
    drop(queue);
    drop(source_binding);
    drop(destination_binding);
    drop(command_binding);
    drop(source_va);
    drop(destination_va);
    drop(command_va);
    let mut output = vec![0u8; allocation_size as usize];
    destination.read(0, &mut output)?;
    let mut expected = initial;
    expected[destination_offset..destination_offset + size]
        .copy_from_slice(&input[source_offset..source_offset + size]);
    let mismatches =
        output.iter().zip(&expected).filter(|(a, b)| a != b).count();
    println!("copy fence=completed checked_bytes={} copied_bytes={} mismatches={} guard_checked=true gpu_commands_executed=true", output.len(), size, mismatches);
    if mismatches != 0 {
        return Err("GPU COPY result mismatch".into());
    }
    Ok(())
}

#[cfg(not(windows))]
fn main() {}
