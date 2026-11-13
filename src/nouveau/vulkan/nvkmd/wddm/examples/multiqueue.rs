#[cfg(windows)]
mod validation {
    use nvkmd_wddm::command::PushBuffer;
    use nvkmd_wddm::contract::{KernelDriver, Push, Scheduling};
    use nvkmd_wddm::error::ErrorKind;
    use nvkmd_wddm::native::{
        Adapter, Context, Device, Fence, Memory, MemoryBinding, Nvgl61692Abi,
        VirtualAddress, WddmDriver,
    };
    use nvkmd_wddm::submission::Queue;
    use std::sync::Arc;
    use std::time::Duration;

    type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
    const ALLOCATION_SIZE: usize = 4096;
    const SOURCE_OFFSET: usize = 256;
    const DESTINATION_OFFSET: usize = 512;
    const COPY_SIZE: usize = 3072;
    const TIMEOUT_NS: u64 = 5_000_000_000;

    struct Allocation {
        memory: Memory,
        binding: MemoryBinding,
        _address: VirtualAddress,
    }

    impl Allocation {
        fn new(device: &Device, label: &str, bytes: &[u8]) -> Result<Self> {
            if bytes.len() != ALLOCATION_SIZE {
                return Err("allocation initialization length".into());
            }
            let (mut memory, kind) =
                Memory::nvidia_host(device, ALLOCATION_SIZE as u64)?;
            if kind != 6 || memory.size() != ALLOCATION_SIZE as u64 {
                return Err(
                    "expected 4096-byte NVIDIA generic page kind 6".into()
                );
            }
            memory.write(0, bytes)?;
            let mut address = VirtualAddress::reserve(
                device,
                ALLOCATION_SIZE as u64,
                65536,
                None,
            )?;
            let binding =
                address.bind(&memory, 0, 0, ALLOCATION_SIZE as u64)?;
            println!(
                "multiqueue allocation={} bytes={} page_kind={} gpu_va=0x{:x}",
                label,
                memory.size(),
                kind,
                binding.address()
            );
            Ok(Self {
                memory,
                binding,
                _address: address,
            })
        }

        unsafe fn snapshot_quiescent(&self) -> Result<Vec<u8>> {
            let mut bytes = vec![0; ALLOCATION_SIZE];
            let address = unsafe { self.memory.map()? };
            for (index, byte) in bytes.iter_mut().enumerate() {
                *byte = unsafe { address.add(index).read_volatile() };
            }
            Ok(bytes)
        }
    }

    struct CopyFixture {
        label: &'static str,
        source: Allocation,
        destination: Allocation,
        commands: Allocation,
        input: Vec<u8>,
        initial: Vec<u8>,
        command_bytes: usize,
    }

    impl CopyFixture {
        fn new(device: &Device, label: &'static str, seed: u8) -> Result<Self> {
            let input: Vec<u8> = (0..ALLOCATION_SIZE as u32)
                .map(|index| {
                    ((index * 37 + u32::from(seed)) ^ (index >> 7)) as u8
                })
                .collect();
            let initial = vec![seed; ALLOCATION_SIZE];
            let source =
                Allocation::new(device, &format!("{}_source", label), &input)?;
            let destination = Allocation::new(
                device,
                &format!("{}_destination", label),
                &initial,
            )?;
            let mut push = PushBuffer::new();
            push.copy_linear(
                source.binding.address() + SOURCE_OFFSET as u64,
                destination.binding.address() + DESTINATION_OFFSET as u64,
                COPY_SIZE as u32,
            )?;
            let bytes = push.bytes();
            if bytes.len() > ALLOCATION_SIZE {
                return Err("copy pushbuffer exceeds command allocation".into());
            }
            let mut command_image = vec![0; ALLOCATION_SIZE];
            command_image[..bytes.len()].copy_from_slice(&bytes);
            let commands = Allocation::new(
                device,
                &format!("{}_commands", label),
                &command_image,
            )?;
            Ok(Self {
                label,
                source,
                destination,
                commands,
                input,
                initial,
                command_bytes: bytes.len(),
            })
        }

        fn enqueue(&self, queue: &mut Queue<Context>) -> Result<()> {
            for allocation in [&self.source, &self.destination, &self.commands]
            {
                queue.retain(allocation.binding.clone())?;
            }
            queue.push(Push {
                address: self.commands.binding.address(),
                size: self.command_bytes as u32,
                incomplete: false,
                no_prefetch: false,
            })?;
            Ok(())
        }

        unsafe fn verify_quiescent(&self, copied: bool) -> Result<()> {
            let output = unsafe { self.destination.snapshot_quiescent()? };
            let mut expected = self.initial.clone();
            if copied {
                expected[DESTINATION_OFFSET..DESTINATION_OFFSET + COPY_SIZE]
                    .copy_from_slice(
                        &self.input[SOURCE_OFFSET..SOURCE_OFFSET + COPY_SIZE],
                    );
            }
            let mismatches = output
                .iter()
                .zip(&expected)
                .filter(|(actual, expected)| actual != expected)
                .count();
            let guards = output
                .iter()
                .zip(&self.initial)
                .enumerate()
                .filter(|(index, (actual, expected))| {
                    !(DESTINATION_OFFSET..DESTINATION_OFFSET + COPY_SIZE)
                        .contains(index)
                        && actual != expected
                })
                .count();
            let source_mismatches =
                unsafe { self.source.snapshot_quiescent()? }
                    .iter()
                    .zip(&self.input)
                    .filter(|(actual, expected)| actual != expected)
                    .count();
            println!(
                "multiqueue queue={} copy_expected={} checked_bytes={} copied_bytes={} mismatches={} guard_bytes={} guard_mismatches={} source_mismatches={}",
                self.label,
                copied,
                ALLOCATION_SIZE,
                if copied { COPY_SIZE } else { 0 },
                mismatches,
                ALLOCATION_SIZE - COPY_SIZE,
                guards,
                source_mismatches
            );
            if mismatches != 0 || guards != 0 || source_mismatches != 0 {
                return Err("copy data, guard, or source mismatch".into());
            }
            Ok(())
        }
    }

    pub fn run() -> Result<()> {
        let adapter = Adapter::enumerate()?
            .into_iter()
            .find(|adapter| adapter.identity().vendor_id == 0x10de)
            .ok_or("NVIDIA adapter missing")?;
        let query = adapter.query_nvidia_61692()?;
        let node = *query
            .nodes()
            .iter()
            .find(|node| {
                node.node_type == 1
                    && node.hardware_scheduling
                    && node.gpfifo_blit
            })
            .ok_or("GR hardware queue node with GPFIFO COPY missing")?;
        let abi =
            Arc::new(Nvgl61692Abi::new(&adapter, &query, node.ordinal, 0)?);
        let device =
            WddmDriver::new(adapter)?.with_abi(abi)?.create_device()?;
        let a = CopyFixture::new(&device, "A", 0xa5)?;
        let b = CopyFixture::new(&device, "B", 0x5a)?;
        let mut gate = Fence::new(&device, 0)?;
        let mut timeline = Fence::new(&device, 0)?;
        let mut queue_a =
            Context::new(&device, 1, Scheduling::HardwareQueue)?.queue();
        let mut queue_b =
            Context::new(&device, 1, Scheduling::HardwareQueue)?.queue();
        if queue_a.transport().progress_values()? != (0, 0)
            || queue_b.transport().progress_values()? != (0, 0)
        {
            return Err("new hardware queue progress is not zero".into());
        }
        println!(
            "multiqueue node={} contexts=2 hardware_queues=2 same_device=true allocation_bytes={} timeout_ns={}",
            node.ordinal, ALLOCATION_SIZE, TIMEOUT_NS
        );
        let result = (|| -> Result<()> {
            queue_a.wait_on(gate.point(1))?;
            a.enqueue(&mut queue_a)?;
            queue_a.signal(timeline.point(10))?;
            let completion_a = queue_a.flush()?;
            queue_b.wait_on(timeline.point(5))?;
            b.enqueue(&mut queue_b)?;
            let completion_b = queue_b.flush()?;
            println!(
                "multiqueue both_batches_submitted=true A_wait_gate=1 A_signal_timeline=10 B_wait_timeline=5 A_completion={} B_completion={}",
                completion_a.value, completion_b.value
            );
            timeline.signal_cpu_value(5)?;
            completion_b.wait(TIMEOUT_NS)?;
            let first_value = timeline.value()?;
            if first_value != 5 {
                return Err(
                    "timeline changed from 5 while gate remained closed".into(),
                );
            }
            match completion_a.wait(0) {
                Err(error) if error.kind == ErrorKind::Timeout => {}
                Err(error) => return Err(error.into()),
                Ok(()) => {
                    return Err(
                        "queue A completed while its gate remained closed"
                            .into(),
                    )
                }
            }
            println!(
                "multiqueue first_phase=true timeline_value={} B_completed=true A_poll=Timeout A_progress={:?} B_progress={:?}",
                first_value,
                queue_a.transport().progress_values()?,
                queue_b.transport().progress_values()?
            );
            if gate.value()? != 0 {
                return Err("gate changed before blocked-queue snapshot".into());
            }
            unsafe {
                a.verify_quiescent(false)?;
                b.verify_quiescent(true)?;
            }
            gate.signal_cpu_value(1)?;
            completion_a.wait(TIMEOUT_NS)?;
            let final_value = timeline.value()?;
            if final_value != 10 {
                return Err("queue A did not signal timeline value 10".into());
            }
            unsafe {
                a.verify_quiescent(true)?;
                b.verify_quiescent(true)?;
            }
            queue_a.wait_idle(TIMEOUT_NS)?;
            queue_b.wait_idle(TIMEOUT_NS)?;
            println!("multiqueue second_phase=true timeline_value={} A_completed=true B_completed=true hold_ms=2500", final_value);
            std::thread::sleep(Duration::from_millis(2500));
            let state = device.execution_state()?;
            completion_a.wait(0)?;
            completion_b.wait(0)?;
            let progress_a = queue_a.transport().progress_values()?;
            let progress_b = queue_b.transport().progress_values()?;
            if progress_a != (1, 1) || progress_b != (1, 1) {
                return Err("hardware queue progress did not complete exactly one batch".into());
            }
            if timeline.value()? != 10 || gate.value()? != 1 {
                return Err(
                    "fence values changed during the post-completion hold"
                        .into(),
                );
            }
            println!(
                "multiqueue_passed=true execution_state={} active=true A_progress={:?} B_progress={:?} timeline_value=10 gate_value=1 post_completion_hold_ms=2500 guards_checked=true",
                state, progress_a, progress_b
            );
            Ok(())
        })();
        if result.is_err() {
            eprintln!(
                "multiqueue failure_cleanup_releasing_blocked_waits=true"
            );
            let _ = timeline.signal_cpu_value(10);
            let _ = gate.signal_cpu_value(1);
            let _ = queue_a.wait_idle(TIMEOUT_NS);
            let _ = queue_b.wait_idle(TIMEOUT_NS);
        }
        result
    }
}

#[cfg(windows)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    validation::run()
}

#[cfg(not(windows))]
fn main() {}
