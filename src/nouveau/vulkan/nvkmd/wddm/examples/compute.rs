#[cfg(windows)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use nvkmd_wddm::command::PushBuffer;
    use nvkmd_wddm::contract::{KernelDriver, Push, Scheduling};
    use nvkmd_wddm::native::{
        Adapter, Context, Memory, Nvgl61692Abi, VirtualAddress, WddmDriver,
    };
    use std::path::PathBuf;
    use std::sync::Arc;
    let fixtures = PathBuf::from(
        std::env::args()
            .nth(1)
            .ok_or("compute fixture directory required")?,
    );
    let code = std::fs::read(fixtures.join("compute.bin"))?;
    let qmd = std::fs::read(fixtures.join("compute.qmd.bin"))?;
    if code.is_empty() || code.len() % 16 != 0 || qmd.len() != 384 {
        return Err("invalid SM120 fixture size".into());
    }
    let qmd_version = u32::from_le_bytes(qmd[56..60].try_into()?);
    if (qmd_version >> 20) & 15 != 5 {
        return Err("QMD v5.0 required".into());
    }
    let adapter = Adapter::enumerate()?
        .into_iter()
        .find(|a| a.identity().vendor_id == 0x10de)
        .ok_or("NVIDIA adapter missing")?;
    let query = adapter.query_nvidia_61692()?;
    let node = *query
        .nodes()
        .iter()
        .find(|n| n.node_type == 1 && n.hardware_scheduling)
        .ok_or("GR hardware node missing")?;
    let abi = Arc::new(Nvgl61692Abi::new(&adapter, &query, node.ordinal, 0)?);
    let device = WddmDriver::new(adapter)?.with_abi(abi)?.create_device()?;
    let context = Context::new(&device, 4 | 16, Scheduling::HardwareQueue)?;
    let mut queue = context.queue();
    let private = std::env::args().any(|arg| arg == "--nvidia");
    let allocate = || {
        if private {
            let (memory, kind) = Memory::nvidia_host(&device, 65536)?;
            println!(
                "compute allocation=nvidia_generic_host bytes={} page_kind={}",
                memory.size(),
                kind
            );
            Ok(memory)
        } else {
            Memory::host(&device, 65536)
        }
    };
    let mut output = allocate()?;
    let mut program = allocate()?;
    let mut qmd_mem = allocate()?;
    let mut command_mem = allocate()?;
    let initial = vec![0xa5u8; 65536];
    output.write(0, &initial)?;
    program.write(0, &code)?;
    qmd_mem.write(0, &qmd)?;
    let mut out_va =
        VirtualAddress::reserve(&device, 65536, 65536, Some(0x10000000))?;
    let mut code_va =
        VirtualAddress::reserve(&device, 65536, 65536, Some(0x10020000))?;
    let mut qmd_va =
        VirtualAddress::reserve(&device, 65536, 65536, Some(0x10040000))?;
    let mut command_va =
        VirtualAddress::reserve(&device, 65536, 65536, Some(0x10060000))?;
    let mut pb = PushBuffer::new();
    pb.compute_launch(qmd_va.address())?;
    let bytes = pb.bytes();
    command_mem.write(0, &bytes)?;
    let out_binding = out_va.bind(&output, 0, 0, 65536)?;
    let code_binding = code_va.bind_executable(&program, 0, 0, 65536)?;
    let qmd_binding = qmd_va.bind(&qmd_mem, 0, 0, 65536)?;
    let command_binding = command_va.bind(&command_mem, 0, 0, 65536)?;
    for binding in [&out_binding, &code_binding, &qmd_binding, &command_binding]
    {
        queue.retain(binding.clone())?;
    }
    queue.push(Push {
        address: command_binding.address(),
        size: bytes.len() as u32,
        incomplete: false,
        no_prefetch: false,
    })?;
    println!("compute node={} code_bytes={} qmd_bytes={} block=1,1,1 grid=1,1,1 destination=0x10000000 program=0x10020000", node.ordinal, code.len(), qmd.len());
    let completion = queue.flush()?;
    println!("compute submit=accepted fence={}", completion.value);
    completion.wait(5_000_000_000)?;
    queue.collect()?;
    println!(
        "compute queue_progress={:?} execution_state={}",
        queue.transport().progress_values()?,
        device.execution_state()?
    );
    std::thread::sleep(std::time::Duration::from_millis(2500));
    println!(
        "compute post_tdr_window queue_progress={:?} execution_state={}",
        queue.transport().progress_values()?,
        device.execution_state()?
    );
    drop(queue);
    drop(out_binding);
    drop(code_binding);
    drop(qmd_binding);
    drop(command_binding);
    drop(out_va);
    drop(code_va);
    drop(qmd_va);
    drop(command_va);
    let mut result = vec![0u8; 65536];
    output.read(0, &mut result)?;
    let value = u32::from_le_bytes(result[..4].try_into()?);
    let guards = result[4..].iter().filter(|b| **b != 0xa5).count();
    println!("compute fence=completed actual=0x{:08x} expected=0x4e564b31 guard_mismatches={} gpu_shader_executed={}", value, guards, value == 0x4e564b31 && guards == 0);
    if value != 0x4e564b31 || guards != 0 {
        return Err("NAK compute result mismatch".into());
    }
    Ok(())
}

#[cfg(not(windows))]
fn main() {}
