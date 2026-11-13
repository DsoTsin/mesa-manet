#[cfg(windows)]
#[allow(dead_code)]
#[path = "../validation/graphics_fixture/mod.rs"]
mod graphics_fixture;

#[cfg(windows)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use graphics_fixture::{ColorFormat, Program, Triangle};
    use nvkmd_wddm::contract::{KernelDriver, Push, Scheduling};
    use nvkmd_wddm::native::{
        Adapter, Context, Memory, Nvgl61692Abi, VirtualAddress, WddmDriver,
    };
    use std::path::PathBuf;
    use std::sync::Arc;

    let args: Vec<String> = std::env::args().skip(1).collect();
    let marker_only = args.iter().any(|arg| arg == "--marker-only");
    let prefix = args
        .iter()
        .find_map(|arg| arg.strip_prefix("--prefix="))
        .map(str::parse::<usize>)
        .transpose()?;
    let diagnostic = marker_only || prefix.is_some();
    let fixtures = args
        .iter()
        .find(|arg| !arg.starts_with("--"))
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../../../../../build/nvkmd-wddm-nak-offline-highva")
        });
    let vertex = std::fs::read(fixtures.join("vertex.program.bin"))?;
    let fragment = std::fs::read(fixtures.join("fragment.program.bin"))?;
    if vertex.len() != 304 || fragment.len() != 272 {
        return Err("SM120 fixture requires vertex SPH128+ISA176 and fragment SPH128+ISA144".into());
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
    let private = args.iter().any(|arg| arg == "--nvidia");
    let allocate = || {
        if private {
            let (memory, kind) = Memory::nvidia_host(&device, 65536)?;
            println!(
                "graphics allocation=nvidia_generic_host bytes={} page_kind={}",
                memory.size(),
                kind
            );
            Ok(memory)
        } else {
            Memory::host(&device, 65536)
        }
    };
    let mut target = allocate()?;
    let mut vertex_memory = allocate()?;
    let mut fragment_memory = allocate()?;
    let mut command_memory = allocate()?;
    let mut counters_memory = allocate()?;
    let initial = vec![0xa5u8; 65536];
    target.write(0, &initial)?;
    vertex_memory.write(0, &vertex)?;
    fragment_memory.write(0, &fragment)?;
    counters_memory.write(0, &initial)?;
    let mut target_va = VirtualAddress::reserve(&device, 65536, 65536, None)?;
    let mut vertex_va = VirtualAddress::reserve(&device, 65536, 65536, None)?;
    let mut fragment_va = VirtualAddress::reserve(&device, 65536, 65536, None)?;
    let mut command_va = VirtualAddress::reserve(&device, 65536, 65536, None)?;
    let mut counters_va = VirtualAddress::reserve(&device, 65536, 65536, None)?;
    let triangle = Triangle {
        target_gpuva: target_va.address(),
        width: 61,
        height: 37,
        pitch: 256,
        format: ColorFormat::Rgba8,
        vertex: Program {
            header_gpuva: vertex_va.address(),
            code_size: (vertex.len() - 128) as u32,
            registers: 24,
        },
        fragment: Program {
            header_gpuva: fragment_va.address(),
            code_size: (fragment.len() - 128) as u32,
            registers: 24,
        },
        warps_per_tpc: 96,
    };
    let words = if marker_only {
        graphics_fixture::marker_push(
            counters_va.address() + 0x100,
            0x4752_4150,
        )?
    } else if let Some(count) = prefix {
        let full = graphics_fixture::triangle_push(triangle)?;
        let mut end = 0;
        for _ in 0..count {
            let header =
                *full.get(end).ok_or("prefix exceeds pushbuffer packets")?;
            end += 1 + ((header >> 16) & 0x1fff) as usize;
        }
        let mut words = full[..end].to_vec();
        words.extend(graphics_fixture::marker_push(
            counters_va.address() + 0x100,
            0x4752_4150,
        )?);
        println!("graphics diagnostic_prefix_packets={}", count);
        words
    } else {
        graphics_fixture::triangle_push_with_counters(
            triangle,
            Some(counters_va.address()),
        )?
    };
    let bytes: Vec<u8> =
        words.iter().flat_map(|word| word.to_le_bytes()).collect();
    if bytes.len() > 65536 {
        return Err("graphics command buffer exceeds allocation".into());
    }
    command_memory.write(0, &bytes)?;
    let target_binding = target_va.bind(&target, 0, 0, 65536)?;
    let vertex_binding =
        vertex_va.bind_executable(&vertex_memory, 0, 0, 65536)?;
    let fragment_binding =
        fragment_va.bind_executable(&fragment_memory, 0, 0, 65536)?;
    let command_binding = command_va.bind(&command_memory, 0, 0, 65536)?;
    let counters_binding = counters_va.bind(&counters_memory, 0, 0, 65536)?;
    for binding in [
        &target_binding,
        &vertex_binding,
        &fragment_binding,
        &command_binding,
        &counters_binding,
    ] {
        queue.retain(binding.clone())?;
    }
    queue.push(Push {
        address: command_binding.address(),
        size: bytes.len() as u32,
        incomplete: false,
        no_prefetch: false,
    })?;
    println!(
        "graphics mode={} node={} class=0xce97 topology=triangle_list vertices=3 instances=1 format=RGBA8 width={} height={} pitch={} pb_bytes={}",
        if marker_only { "marker-only" } else if diagnostic { "prefix" } else { "triangle" },
        node.ordinal, triangle.width, triangle.height, triangle.pitch, bytes.len()
    );
    println!(
        "graphics target_va=0x{:x} vertex_va=0x{:x} fragment_va=0x{:x} command_va=0x{:x} vertex_bytes={} fragment_bytes={} registers=24 slm=0",
        target_binding.address(), vertex_binding.address(), fragment_binding.address(),
        command_binding.address(), vertex.len(), fragment.len()
    );
    let completion = queue.flush()?;
    println!(
        "graphics submit=accepted fence={} timeout_ns=5000000000",
        completion.value
    );
    completion.wait(5_000_000_000)?;
    queue.collect()?;
    println!(
        "graphics queue_progress={:?} execution_state={}",
        queue.transport().progress_values()?,
        device.execution_state()?
    );
    drop(queue);
    drop(target_binding);
    drop(vertex_binding);
    drop(fragment_binding);
    drop(command_binding);
    drop(counters_binding);
    drop(target_va);
    drop(vertex_va);
    drop(fragment_va);
    drop(command_va);
    drop(counters_va);
    let mut output = vec![0u8; 65536];
    target.read(0, &mut output)?;
    let mut counter_bytes = vec![0u8; 65536];
    counters_memory.read(0, &mut counter_bytes)?;
    let counts: Vec<u64> = (0..6)
        .map(|i| {
            u64::from_le_bytes(
                counter_bytes[i * 16..i * 16 + 8].try_into().unwrap(),
            )
        })
        .collect();
    let marker = u32::from_le_bytes(counter_bytes[0x100..0x104].try_into()?);
    println!("graphics counters da_vertices={} da_primitives={} vs_invocations={} clipper_invocations={} clipper_primitives={} ps_invocations={} marker=0x{:08x}",
             counts[0], counts[1], counts[2], counts[3], counts[4], counts[5], marker);
    let phases: Vec<u32> = (0..6)
        .map(|i| {
            u32::from_le_bytes(
                counter_bytes[0x200 + i * 16..0x204 + i * 16]
                    .try_into()
                    .unwrap(),
            )
        })
        .collect();
    println!("graphics phases={:08x?}", phases);
    let result = graphics_fixture::check_pixels(&output, triangle, 0xa5)?;
    let first = u32::from_le_bytes(output[..4].try_into()?);
    let expected =
        u32::from_le_bytes(graphics_fixture::expected_pixel(triangle.format));
    let (expected_low, expected_high) =
        graphics_fixture::expected_pixel_bounds(triangle.format);
    let changed_bytes =
        output.iter().zip(&initial).filter(|(a, b)| a != b).count();
    let passed = if diagnostic {
        marker == 0x4752_4150 && changed_bytes == 0
    } else {
        result.pixels != 0
            && result.pixel_mismatches == 0
            && result.padding_mismatches == 0
            && marker == 0x4752_4150
            && counts[0] == 3
            && counts[1] == 1
            && counts[2] >= 3
            && counts[5] >= result.pixels as u64
    };
    println!(
        "graphics fence=completed checked_pixels={} pixel_mismatches={} checked_guard_bytes={} guard_mismatches={} changed_bytes={} first_pixel=0x{:08x} nearest_reference=0x{:08x} gpu_triangle_executed={}",
        result.pixels, result.pixel_mismatches, result.padding_bytes,
        result.padding_mismatches, changed_bytes, first, expected, passed && !diagnostic
    );
    println!("graphics permitted_unorm_low={:?} permitted_unorm_high={:?} nearest_reference_mismatches={}",
             expected_low, expected_high, result.nearest_reference_mismatches);
    if !passed {
        println!("graphics first_64_bytes={:02x?}", &output[..64]);
        return Err("native GPU triangle pixel or guard mismatch".into());
    }
    Ok(())
}

#[cfg(not(windows))]
fn main() {}
